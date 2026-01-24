import model_utils
import torch
import typing
import utils
import transformers
import tqdm, math
import quant_utils
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform
from deepseek_moe_16b_chat.modeling_deepseek import DeepseekMLP,DeepseekMoE
from mixtral_model.modeling_mixtral import MixtralBlockSparseTop2MLP, MixtralSparseMoeBlock

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    

def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        # W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        # W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q.to(W.weight.device)).to( dtype=dtype)

    
def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    # Skip for layers without self_attn (e.g., Qwen3-Next with mixed architecture)
    if not hasattr(layer, 'self_attn'):
        return

    # Skip for DeepSeek V2 MLA which has different projection structure
    if model_utils._is_deepseek_v2_type(model_type) and not hasattr(layer.self_attn, 'q_proj'):
        return

    # Skip if standard projection layers don't exist
    if not (hasattr(layer.self_attn, 'q_proj') and hasattr(layer.self_attn, 'k_proj') and hasattr(layer.self_attn, 'v_proj')):
        return

    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        # W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        # W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q.to(W.weight.device)).to(dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    # Skip for layers without self_attn (e.g., Qwen3-Next with mixed architecture)
    if not hasattr(layer, 'self_attn'):
        return

    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    elif model_utils._is_qwen_moe_type(model_type):
        W = layer.self_attn.o_proj
    elif model_utils._is_deepseek_type(model_type):
        W = layer.self_attn.o_proj
    elif model_type == model_utils.MIXTRAL_MODEL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    # W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    # W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T.to(W.weight.device), W_).to(dtype=dtype)
    if W.bias is not None:
        # b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        # W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T.to(W.bias.device), b).to(dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_utils._is_qwen_moe_type(model_type):
        mlp_inputs = [layer.mlp.gate]
        # Check if shared_expert exists (Qwen1.5/Qwen2)
        if hasattr(layer.mlp, 'shared_expert_gate') and layer.mlp.shared_expert_gate is not None:
            mlp_inputs.append(layer.mlp.shared_expert_gate)
        if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
            mlp_inputs.extend([layer.mlp.shared_expert.gate_proj, layer.mlp.shared_expert.up_proj])
        # Dynamically iterate over all experts
        num_experts = len(layer.mlp.experts)
        for i in range(num_experts):
            mlp_inputs.append(layer.mlp.experts[i].up_proj)
            mlp_inputs.append(layer.mlp.experts[i].gate_proj)
    elif model_utils._is_deepseek_type(model_type):
        # Check if it's a standard MLP or MoE layer
        # DeepSeek V1 uses DeepseekMLP/DeepseekMoE, V2 uses DeepseekV2MLP/DeepseekV2MoE
        mlp_class_name = layer.mlp.__class__.__name__
        if 'MoE' in mlp_class_name:
            # MoE layer (V1 or V2)
            mlp_inputs = [layer.mlp.gate]
            # Add shared experts if they exist
            if hasattr(layer.mlp, 'shared_experts'):
                if hasattr(layer.mlp.shared_experts, 'gate_proj'):
                    mlp_inputs.extend([layer.mlp.shared_experts.gate_proj, layer.mlp.shared_experts.up_proj])
                else:
                    # V2 shared_experts is a single MLP module
                    mlp_inputs.extend([layer.mlp.shared_experts.gate_proj, layer.mlp.shared_experts.up_proj])
            # Dynamically iterate over all experts
            if hasattr(layer.mlp, 'experts'):
                for expert in layer.mlp.experts:
                    mlp_inputs.append(expert.up_proj)
                    mlp_inputs.append(expert.gate_proj)
        else:
            # Standard MLP layer
            mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.MIXTRAL_MODEL:
        mlp_inputs = [layer.block_sparse_moe.gate]
        for i in range(8):
            mlp_inputs.append(layer.block_sparse_moe.experts[i].w1)
            mlp_inputs.append(layer.block_sparse_moe.experts[i].w3)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        # W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        # W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q.to(W.weight.device)).to(dtype=dtype)
    
def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_utils._is_qwen_moe_type(model_type):
        mlp_outputs = []
        # Check if shared_expert exists (Qwen1.5/Qwen2)
        if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None and hasattr(layer.mlp.shared_expert, 'down_proj'):
            mlp_outputs.append(layer.mlp.shared_expert.down_proj)
        # Dynamically iterate over all experts
        num_experts = len(layer.mlp.experts)
        for i in range(num_experts):
            mlp_outputs.append(layer.mlp.experts[i].down_proj)
    elif model_utils._is_deepseek_type(model_type):
        # Check if it's a standard MLP or MoE layer
        mlp_class_name = layer.mlp.__class__.__name__
        if 'MoE' in mlp_class_name:
            # MoE layer (V1 or V2)
            mlp_outputs = []
            # Add shared experts if they exist
            if hasattr(layer.mlp, 'shared_experts') and hasattr(layer.mlp.shared_experts, 'down_proj'):
                mlp_outputs.append(layer.mlp.shared_experts.down_proj)
            # Dynamically iterate over all experts
            if hasattr(layer.mlp, 'experts'):
                for expert in layer.mlp.experts:
                    mlp_outputs.append(expert.down_proj)
        else:
            # Standard MLP layer
            mlp_outputs = [layer.mlp.down_proj]
    elif model_type == model_utils.MIXTRAL_MODEL:
        mlp_outputs = []
        for i in range(8):
            mlp_outputs.append(layer.block_sparse_moe.experts[i].w2)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_outputs:
        dtype = W.weight.data.dtype
        # W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        # W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        W_ = W.weight.data.to(dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T.to(W.weight.device), W_).to(dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            # b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            # W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
            b = W.bias.data.to(dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T.to(W.bias.device), b).to(dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 

def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
    
    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    # W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    # W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    W_ = W.weight.data.to(dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q.to(W.weight.device)).to(dtype=dtype)

def rotate_ov_proj(layer, model_type, head_num, head_dim):
    # Skip for layers without self_attn (e.g., Qwen3-Next with mixed architecture)
    if not hasattr(layer, 'self_attn'):
        return

    # Skip for DeepSeek V2 MLA which has different projection structure
    if model_utils._is_deepseek_v2_type(model_type) and not hasattr(layer.self_attn, 'v_proj'):
        return

    # Skip if v_proj doesn't exist
    if not hasattr(layer.self_attn, 'v_proj'):
        return

    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    elif model_utils._is_qwen_moe_type(model_type):
        o_proj = layer.self_attn.o_proj
    elif model_utils._is_deepseek_type(model_type):
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.MIXTRAL_MODEL:
        o_proj = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')


@torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    if args.do_smooth:
        acts_scales = torch.load(args.smooth_scales)
        acts_shifts = torch.load(args.smooth_shifts)
        dtype = torch.float16
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type)
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)

@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k



def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)

def fuse_layer_norms(model):
    print("正在使用fuse_layer_norms")
    model_type = model_utils.model_type_extractor(model)
    
    # 1. Fuse the final layer norm into the LM Head
    # 将最后的 LayerNorm 融合到 LM Head 中
    norm = model_utils.get_pre_head_layernorm(model, model_type)
    lm_head = model_utils.get_lm_head(model, model_type)
    
    if hasattr(norm, 'weight'):
        with torch.no_grad():
            scale = norm.weight.data.double()
            # W_new = W_old * scale (broadcasting over input dimension)
            lm_head.weight.data = (lm_head.weight.data.double() * scale.unsqueeze(0)).to(lm_head.weight.dtype)
            
            # Reset norm weights to 1 and bias to 0
            norm.weight.data.fill_(1.0)
            if hasattr(norm, 'bias') and norm.bias is not None:
                norm.bias.data.fill_(0.0)

    # 2. Fuse Layer Norms inside the transformer layers
    # 将层内的 LayerNorm 融合到 Attention 和 MLP 的输入投影层中
    layers = model_utils.get_transformer_layers(model, model_type)
    
    for layer in tqdm.tqdm(layers, unit="layer", desc="Fusing Layer Norms"):
        # Identify norms based on model type
        if model_type in [model_utils.LLAMA_MODEL, model_utils.MIXTRAL_MODEL] or model_utils._is_qwen_moe_type(model_type) or model_utils._is_deepseek_type(model_type):
            input_norm = layer.input_layernorm
            post_norm = layer.post_attention_layernorm
        elif model_type == model_utils.OPT_MODEL:
            input_norm = layer.self_attn_layer_norm
            post_norm = layer.final_layer_norm
        else:
            continue
            
        # --- Fuse input_norm into Attention Inputs ---
        if hasattr(input_norm, 'weight'):
            with torch.no_grad():
                scale = input_norm.weight.data.double()

                # Check if layer has self_attn attribute (skip for Qwen3-Next and other special architectures)
                if not hasattr(layer, 'self_attn'):
                    # Layer doesn't have self_attn (e.g., Qwen3-Next with mixed linear attention)
                    # Skip attention fusion for this layer
                    pass
                else:
                    # Check if this is DeepSeek V2 with MLA architecture
                    is_deepseek_v2_mla = (model_utils._is_deepseek_v2_type(model_type) and
                                           not hasattr(layer.self_attn, 'q_proj'))

                    if is_deepseek_v2_mla:
                        # DeepSeek V2 uses MLA with different projection layers
                        # Skip attention fusion for now as it has different structure
                        # (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj instead of q/k/v_proj)
                        pass
                    elif hasattr(layer.self_attn, 'q_proj') and hasattr(layer.self_attn, 'k_proj') and hasattr(layer.self_attn, 'v_proj'):
                        # Standard attention with q_proj, k_proj, v_proj
                        attn_inputs = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]

                        for mod in attn_inputs:
                            mod.weight.data = (mod.weight.data.double() * scale.unsqueeze(0)).to(mod.weight.dtype)
                            # Handle bias if present (RMSNorm typically has no bias, LayerNorm does)
                            if hasattr(input_norm, 'bias') and input_norm.bias is not None and mod.bias is not None:
                                mod.bias.data = (mod.bias.data.double() + torch.matmul(mod.weight.data.double(), input_norm.bias.data.double())).to(mod.bias.dtype)

                        input_norm.weight.data.fill_(1.0)
                        if hasattr(input_norm, 'bias') and input_norm.bias is not None:
                            input_norm.bias.data.fill_(0.0)

        # --- Fuse post_norm into MLP Inputs ---
        if hasattr(post_norm, 'weight'):
             with torch.no_grad():
                scale = post_norm.weight.data.double()
                
                # Identify MLP input layers based on model type
                mlp_inputs = []
                if model_type == model_utils.LLAMA_MODEL:
                    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
                elif model_type == model_utils.OPT_MODEL:
                    mlp_inputs = [layer.fc1]
                elif model_utils._is_qwen_moe_type(model_type):
                    mlp_inputs = [layer.mlp.gate]
                    # Check if shared_expert exists (Qwen1.5/Qwen2)
                    if hasattr(layer.mlp, 'shared_expert_gate') and layer.mlp.shared_expert_gate is not None:
                        mlp_inputs.append(layer.mlp.shared_expert_gate)
                    if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
                        mlp_inputs.extend([layer.mlp.shared_expert.gate_proj, layer.mlp.shared_expert.up_proj])
                    # Dynamically iterate experts if present
                    if hasattr(layer.mlp, 'experts'):
                         for expert in layer.mlp.experts:
                             mlp_inputs.append(expert.up_proj)
                             mlp_inputs.append(expert.gate_proj)
                elif model_utils._is_deepseek_type(model_type):
                    # Check if it's a standard MLP or MoE layer
                    mlp_class_name = layer.mlp.__class__.__name__
                    if 'MoE' in mlp_class_name:
                        # MoE layer (V1 or V2)
                        mlp_inputs = [layer.mlp.gate]
                        # Add shared experts if they exist
                        if hasattr(layer.mlp, 'shared_experts'):
                            if hasattr(layer.mlp.shared_experts, 'gate_proj'):
                                mlp_inputs.extend([layer.mlp.shared_experts.gate_proj, layer.mlp.shared_experts.up_proj])
                            else:
                                # V2 shared_experts is a single MLP module
                                mlp_inputs.extend([layer.mlp.shared_experts.gate_proj, layer.mlp.shared_experts.up_proj])
                        # Dynamically iterate over all experts
                        if hasattr(layer.mlp, 'experts'):
                            for expert in layer.mlp.experts:
                                mlp_inputs.append(expert.up_proj)
                                mlp_inputs.append(expert.gate_proj)
                    else:
                        # Standard MLP layer
                        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
                elif model_type == model_utils.MIXTRAL_MODEL:
                    mlp_inputs = [layer.block_sparse_moe.gate]
                    if hasattr(layer.block_sparse_moe, 'experts'):
                        for expert in layer.block_sparse_moe.experts:
                            mlp_inputs.append(expert.w1) # w1 is gate_proj equivalent
                            mlp_inputs.append(expert.w3) # w3 is up_proj equivalent

                # Apply fusion to all MLP input layers
                for mod in mlp_inputs:
                    mod.weight.data = (mod.weight.data.double() * scale.unsqueeze(0)).to(mod.weight.dtype)
                    if hasattr(post_norm, 'bias') and post_norm.bias is not None and mod.bias is not None:
                         mod.bias.data = (mod.bias.data.double() + torch.matmul(mod.weight.data.double(), post_norm.bias.data.double())).to(mod.bias.dtype)
                
                post_norm.weight.data.fill_(1.0)
                if hasattr(post_norm, 'bias') and post_norm.bias is not None:
                    post_norm.bias.data.fill_(0.0)
