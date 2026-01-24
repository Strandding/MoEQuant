import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import logging
import torch.nn.functional as F
import math

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTQ:
    
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        # Check for NaN in input activations
        if torch.any(torch.isnan(inp)):
            logging.warning(f'NaN detected in input activations for add_batch! Replacing with zeros.')
            inp = torch.nan_to_num(inp, nan=0.0)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def add_batch_score(self, routing_scores,selected_experts,expert_num, expert_sum,inp, out):
        expert_mask = torch.nn.functional.one_hot(selected_experts[self.layer.cur_sample], num_classes=expert_sum).permute(2, 1, 0)
        idx, top_x = torch.where(expert_mask[expert_num])

        # Check if this expert was selected in the current sample
        # If not selected, skip Hessian update for this batch
        if idx.numel() == 0:
            # Expert not selected in this sample, skip this batch
            return

        s = routing_scores[self.layer.cur_sample][top_x, idx, None].to(inp.device)
        s1 = torch.sqrt(s)
        inp = inp*s1
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # tmp = (s**2).sum() #/ inp.shape[1]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def add_batch_shared_score(self, routing_shared_scores, inp, out):
        s = torch.sqrt(routing_shared_scores[self.layer.cur_sample])
        # s = routing_shared_scores[self.layer.cur_sample]
        inp = inp*s.to(inp.device)
        # inp = inp * routing_scores[self.nsamples][top_x, idx, None].unsqueeze(0).to(inp.device)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # tmp = (s**2).sum() #/ inp.shape[1]
        # tmp = torch.sum(torch.pow(s,2))/inp.shape[1]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        # Early NaN detection: check original weights before quantization
        if torch.any(torch.isnan(W)):
            logging.error(f'NaN detected in ORIGINAL weights before quantization!')
            logging.error(f'Weight shape: {W.shape}, NaN count: {torch.isnan(W).sum()} / {W.numel()}')
            logging.error(f'Weight stats (ignoring NaN): min={W[~torch.isnan(W)].min() if (~torch.isnan(W)).any() else "all NaN"}, '
                         f'max={W[~torch.isnan(W)].max() if (~torch.isnan(W)).any() else "all NaN"}')
            raise ValueError('NaN in original weights - likely caused by rotation/fusion or previous layer propagation')

        # Check Hessian for NaN
        if torch.any(torch.isnan(self.H)):
            logging.error(f'NaN detected in Hessian matrix!')
            logging.error(f'Hessian shape: {self.H.shape}, NaN count: {torch.isnan(self.H).sum()} / {self.H.numel()}')
            raise ValueError('NaN in Hessian matrix - check calibration data or input activations')

        # Check if Hessian is all zeros (expert never used)
        if torch.allclose(self.H, torch.zeros_like(self.H)):
            logging.warning(f'Hessian matrix is all zeros - this expert was never selected during calibration!')
            logging.warning(f'Skipping GPTQ quantization for this layer, using simple rounding instead.')
            # Use simple rounding quantization as fallback
            if not self.quantizer.ready():
                self.quantizer.find_params(W)
            scale = self.quantizer.scale
            zero = self.quantizer.zero
            q = torch.clamp(torch.round(W / scale) + zero, 0, self.quantizer.maxq)
            self.layer.weight.data = (scale * (q - zero)).reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            return

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        # Add numerical stability: try Cholesky decomposition with error handling
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
        except RuntimeError as e:
            logging.error(f'Cholesky decomposition failed: {e}')
            logging.error(f'Hessian matrix stats - min: {H.min()}, max: {H.max()}, mean: {H.mean()}')
            # Try with increased damping
            logging.warning('Retrying with increased damping factor (10x)')
            H[diag, diag] += damp * 9  # Total 10x damping
            try:
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
            except RuntimeError as e2:
                logging.error(f'Cholesky decomposition failed again: {e2}')
                raise ValueError('Cannot perform Cholesky decomposition on Hessian matrix')

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Add numerical stability: clamp d to avoid division by zero or very small numbers
                # This is important for rotated models where the Hessian might be ill-conditioned
                if torch.abs(d) < 1e-6:
                    logging.warning(f'Small diagonal element detected: d={d}, clamping to 1e-6')
                    d = torch.clamp(d, min=1e-6)

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.error('=' * 60)
            logging.error('NaN DETECTED IN QUANTIZED WEIGHTS')
            logging.error('=' * 60)
            import pprint
            # Use 'zero' instead of 'zero_point' to match WeightQuantizer attributes
            logging.error('Quantizer configuration:')
            pprint.pprint({'bits': self.quantizer.bits,
                          'scale_shape': self.quantizer.scale.shape,
                          'scale_min': self.quantizer.scale.min().item() if self.quantizer.scale.numel() > 0 else None,
                          'scale_max': self.quantizer.scale.max().item() if self.quantizer.scale.numel() > 0 else None,
                          'zero_shape': self.quantizer.zero.shape})
            logging.error(f'Original weight stats - min: {W.min()}, max: {W.max()}, mean: {W.mean()}')
            logging.error(f'Quantized weight stats - min: {Q.min()}, max: {Q.max()}, mean: {Q.mean()}')
            logging.error(f'NaN count: {torch.isnan(self.layer.weight.data).sum()} / {self.layer.weight.data.numel()}')
            logging.error('=' * 60)
            raise ValueError('NaN in weights - check logs for details')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)

def build_prompt(text):
    return text

def get_num_experts(config):
    """
    Dynamically get the number of experts from model config.
    Supports Qwen, DeepSeek, Mixtral, and their variants.
    """
    # Qwen models (Qwen1.5, Qwen2, Qwen3, Qwen3-Next)
    if hasattr(config, 'num_experts') and config.num_experts is not None:
        return config.num_experts
    # DeepSeek V2
    elif hasattr(config, 'n_routed_experts') and config.n_routed_experts is not None:
        return config.n_routed_experts
    # Mixtral and other models
    elif hasattr(config, 'num_local_experts') and config.num_local_experts is not None:
        return config.num_local_experts
    else:
        # Fallback: try to infer from model architecture name
        arch_name = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else ''
        if 'Qwen' in arch_name or 'qwen' in arch_name:
            return 60  # Default for Qwen1.5
        elif 'Deepseek' in arch_name or 'deepseek' in arch_name:
            return 64  # Default for DeepSeek
        elif 'Mixtral' in arch_name or 'mixtral' in arch_name:
            return 8   # Default for Mixtral
        else:
            raise ValueError(f"Cannot determine number of experts from config: {config}")

@torch.no_grad()
def gptq_fwrd(model, tokenizer, dataloader, dev, args, bit_mask):
    '''
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Get number of experts dynamically
    num_experts = get_num_experts(model.config)
    logging.info(f'---> Detected {num_experts} experts in the model')

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = model.lm_head.weight.dtype
    # dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        # (args.nsamples, 512, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            # Capture position_embeddings for Qwen3-Next
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
        def __getattr__(self, name):
            # Forward attribute access to the wrapped module
            # This allows accessing attributes like 'layer_type' for Qwen3-Next
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if args.EBSS_calib or getattr(args, 'llmqat_calib', False):
                texts = batch["text"]
                queries = [build_prompt(query) for query in texts]
                tokenizer.pad_token = tokenizer.eos_token# for mixtral
                inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=model.seqlen,padding=True).to(dev)
                # Pad both input_ids and attention_mask to model.seqlen
                pad_length = model.seqlen - inputs['input_ids'].shape[1]
                inputs['input_ids'] = F.pad(inputs['input_ids'], (0, pad_length))
                inputs['attention_mask'] = F.pad(inputs['attention_mask'], (0, pad_length), value=0)
                # inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=512,padding=True).to(dev)
                # inputs['input_ids'] =  F.pad(inputs['input_ids'],(0,512-inputs['input_ids'].shape[1]))
                model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            else:
                # model(batch.to(dev))
                model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache.get('position_embeddings')  # For Qwen3-Next

    quantizers = {}
    # sequential = [
    #             ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
    #             ['self_attn.o_proj.module'],
    #             ['mlp.up_proj.module', 'mlp.gate_proj.module'],
    #             ['mlp.down_proj.module']
    #         ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i]#.to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        if 'deepseek' in args.model.lower() and i>=1:
            full['mlp.gate'] = model.model.layers[i].mlp.gate
        sequential = list(full.keys())
        # adjust sequential
        if 'qwen' in args.model.lower():
            # Adaptive sequential ordering based on model architecture
            # Step 1: Separate into attention, expert, shared_expert, and down_proj
            attn_layers = []
            shared_expert_layers = []
            expert_layers = []
            down_proj_layers = []

            for element in sequential:
                if 'self_attn' in element:
                    attn_layers.append(element)
                elif 'down_proj' in element:
                    down_proj_layers.append(element)
                elif 'shared_expert' in element:
                    shared_expert_layers.append(element)
                elif 'mlp.experts' in element or 'mlp.gate' in element:
                    expert_layers.append(element)
                else:
                    # Other layers (shouldn't happen, but just in case)
                    expert_layers.append(element)

            # Step 2: Construct sequential order
            # Priority: attention > shared_expert > experts > down_proj
            sequential = attn_layers + shared_expert_layers + expert_layers + down_proj_layers

            # Note: For Qwen1.5/Qwen2 with shared_expert, this prioritizes shared expert
            # For Qwen3 without shared_expert, this maintains natural expert order
            # For Qwen3-Next with 512 experts + shared_expert, shared expert comes first
        elif 'deepseek' in args.model.lower():
            # DeepSeek V2 has routed experts + shared experts
            # Step 1: Separate into attention, shared_expert, expert, and down_proj
            attn_layers = []
            shared_expert_layers = []
            expert_layers = []
            down_proj_layers = []
            gate_layer = None

            for element in sequential:
                if 'self_attn' in element:
                    attn_layers.append(element)
                elif 'down_proj' in element:
                    down_proj_layers.append(element)
                elif 'shared_expert' in element:
                    shared_expert_layers.append(element)
                elif 'mlp.gate' in element:
                    gate_layer = element
                elif 'mlp.experts' in element:
                    expert_layers.append(element)
                else:
                    expert_layers.append(element)

            # Step 2: Construct sequential order
            # For layer 0: simple order without gate
            # For layer 1+: attention > gate > shared_expert > experts > down_proj
            if i == 0:
                sequential = attn_layers + shared_expert_layers + expert_layers + down_proj_layers
            else:
                sequential = attn_layers + ([gate_layer] if gate_layer else []) + shared_expert_layers + expert_layers + down_proj_layers
        elif 'mixtral' in args.model.lower():
            # Mixtral: 8 experts, w1/w3 (gate_proj/up_proj equivalent) first, then w2 (down_proj)
            attn_layers = []
            expert_layers = []
            down_proj_layers = []

            for element in sequential:
                if 'self_attn' in element:
                    attn_layers.append(element)
                elif "w2" in element:  # down_proj in Mixtral
                    down_proj_layers.append(element)
                else:
                    expert_layers.append(element)

            sequential = attn_layers + expert_layers + down_proj_layers

        # Wrap each layer name in a list for consistent processing
        for k in range(len(sequential)):
            sequential[k]=[sequential[k]]
        routing_scores = []
        selected_experts = []
        routing_scores_shared = []
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                # -----------------------------------------------------------
                # [新增] 强制过滤：只量化 Expert 内部的权重
                # 如果名字里不包含 'experts' (普通专家) 且不包含 'shared_expert' (共享专家)
                # 则跳过，不进行量化。
                # -----------------------------------------------------------
                if 'experts' not in name:
                    # print(f'(Skipping {name})', end='  ', flush=True)
                    continue

                # [新增] 额外过滤：排除 Router/Gate 路由权重
                # 通常路由层的名字以 .gate 结尾 (如 block_sparse_moe.gate 或 mlp.gate)
                # 而专家内部的 gate_proj 通常包含在 experts...gate_proj 中，不会被此条件误杀
                if name.endswith('.gate') or name.endswith('shared_expert_gate'):
                    # print(f'(Skipping Router {name})', end='  ', flush=True)
                    continue

                # [新增] 专家编号过滤：只量化0号专家
                # 解析专家编号，格式如 mlp.experts.0.gate_proj 或 block_sparse_moe.experts.0.w1
                if 'mlp.experts.' in name or 'block_sparse_moe.experts.' in name:
                    try:
                        # 提取专家编号
                        parts = name.split('.')
                        expert_idx = -1
                        for idx, part in enumerate(parts):
                            if part == 'experts' and idx + 1 < len(parts):
                                expert_idx = int(parts[idx + 1])
                                break

                        # # 只量化0号专家，跳过其他专家
                        # if expert_idx not in [0]:
                        #     # print(f'(Skipping expert {expert_idx}: {name})', end='  ', flush=True)
                        #     continue
                    except (ValueError, IndexError):
                        # 如果无法解析专家编号，保险起见跳过
                        print(f'(Warning: Cannot parse expert index from {name}, skipping)', end='  ', flush=True)
                        continue
                # -----------------------------------------------------------

                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch_deepseek(name):
                def tmp(_, inp, out):
                    if name =='mlp.gate':
                        gptq[name].add_batch(inp[0], out)
                    else:
                        gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            def add_batch_qwen(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            def add_batch_score(name,routing_scores,selected_experts,expert_sum):
                def tmp(_, inp, out):
                    expert_num = int(name.split('.')[2])
                    gptq[name].add_batch_score(routing_scores,selected_experts,expert_num, expert_sum, inp[0].data, out.data)
                return tmp

            def add_batch_shared_score(name,routing_scores_shared):
                def tmp(_, inp, out):
                    gptq[name].add_batch_shared_score(routing_scores_shared,inp[0].data, out.data)
                return tmp
            handles = []
            if 'deepseek' in args.model.lower():
                for name in subset:
                    if name not in gptq: continue
                    if 'mlp.experts' in name and i>=1:
                        handles.append(subset[name].register_forward_hook(add_batch_score(name,routing_scores,selected_experts,num_experts)))
                    else:
                        handles.append(subset[name].register_forward_hook(add_batch_deepseek(name)))
            elif 'mixtral' in args.model.lower():
                for name in subset:
                    if name not in gptq: continue
                    if 'block_sparse_moe.experts' in name and i>=1:
                        handles.append(subset[name].register_forward_hook(add_batch_score(name,routing_scores,selected_experts,num_experts)))
                    else:
                        handles.append(subset[name].register_forward_hook(add_batch_qwen(name)))
            else:
                for name in subset:
                    if name not in gptq: continue
                    if 'mlp.experts' in name and i>=1:
                        handles.append(subset[name].register_forward_hook(add_batch_score(name,routing_scores,selected_experts,num_experts)))
                    elif 'mlp.shared_expert.' in name and i>=1:
                        handles.append(subset[name].register_forward_hook(add_batch_shared_score(name,routing_scores_shared)))
                    else:
                        handles.append(subset[name].register_forward_hook(add_batch_qwen(name)))
            # all sample
            # layer.mlp.static_observer=True
            for jjj in range(args.nsamples):
                # Set cur_sample for all linear layers in subset (needed for EBSS score tracking)
                for name in subset:
                    if name in gptq:
                        subset[name].cur_sample = jjj

                # Call layer forward with appropriate parameters depending on model type
                if 'deepseek' in args.model.lower():
                    # DeepSeek models don't accept cur_sample parameter in their forward function
                    outs[jjj] = layer(inps[jjj].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                elif 'qwen3-next' in args.model.lower() or 'qwen3next' in args.model.lower():
                    # Qwen3-Next requires position_embeddings parameter
                    if position_embeddings is not None:
                        outs[jjj] = layer(inps[jjj].unsqueeze(0), attention_mask=attention_mask,
                                         position_ids=position_ids, position_embeddings=position_embeddings,
                                         cur_sample=jjj)[0]
                    else:
                        # Fallback: try without position_embeddings
                        outs[jjj] = layer(inps[jjj].unsqueeze(0), attention_mask=attention_mask,
                                         position_ids=position_ids, cur_sample=jjj)[0]
                else:
                    # Other Qwen models and models that accept cur_sample
                    outs[jjj] = layer(inps[jjj].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,cur_sample=jjj)[0]
            for h in handles:
                h.remove()
            for name in subset:
                # Part 1: 执行量化 (只针对在 gptq 列表里的层，即 Expert)
                if name in gptq:
                    layer_w_groupsize = args.w_groupsize
                    gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                    )
                    quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                    gptq[name].free()

                # Part 2: Gate 路由逻辑 (必须执行！无论是否量化)
                # 这部分代码负责填充 routing_scores，绝不能跳过
                if name=='mlp.gate' or name == 'block_sparse_moe.gate':
                    # Use model-configured top-k to keep routing stats consistent with forward.
                    moe_top_k = getattr(getattr(layer, "mlp", None), "top_k", None)
                    if moe_top_k is None:
                        moe_top_k = getattr(model.config, "num_experts_per_tok", 4)
                    moe_top_k = int(moe_top_k)
                    def save_gate_res_qwen(module, inp, out):
                        routing_score = F.softmax(out, dim=1, dtype=torch.float)
                        k = min(moe_top_k, routing_score.shape[-1])
                        routing_score, selected_expert = torch.topk(routing_score, k, dim=-1)
                        routing_scores.append(routing_score.tolist())
                        selected_experts.append(selected_expert.tolist())
                    def save_gate_res_deepseek(module, inp, out):
                        routing_scores.append(out[1].tolist())
                        selected_experts.append(out[0].tolist())
                    def save_gate_res_mixtral(module, inp, out):
                        routing_score = F.softmax(out, dim=1, dtype=torch.float)
                        routing_score, selected_expert = torch.topk(routing_score, 2, dim=-1)
                        routing_score /= routing_score.sum(dim=-1, keepdim=True)
                        routing_scores.append(routing_score.tolist())
                        selected_experts.append(selected_expert.tolist())
                    
                    handles = []
                    if 'qwen' in args.model.lower():
                        handles.append(layer.mlp.gate.register_forward_hook(save_gate_res_qwen))
                    elif 'deepseek' in args.model.lower():
                        handles.append(layer.mlp.gate.register_forward_hook(save_gate_res_deepseek))
                    elif 'mixtral' in args.model.lower():
                        handles.append(layer.block_sparse_moe.gate.register_forward_hook(save_gate_res_mixtral))
                    
                    # 重新跑一遍前向传播来获取路由信息
                    for j in range(args.nsamples):
                        if 'deepseek' in args.model.lower():
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        elif 'qwen3-next' in args.model.lower() or 'qwen3next' in args.model.lower():
                            if position_embeddings is not None:
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                               position_ids=position_ids, position_embeddings=position_embeddings)[0]
                            else:
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    
                    routing_scores = torch.tensor(routing_scores)
                    selected_experts = torch.tensor(selected_experts)
                    for h in handles:
                        h.remove()

                # Part 3: Shared Gate 逻辑 (同理，不能跳过)
                elif name=='mlp.shared_expert_gate':
                    def save_sharedgate_res(module, inp, out):
                        routing_score_shared = F.sigmoid(out)
                        routing_scores_shared.append(routing_score_shared.tolist())

                    handles = []
                    handles.append(layer.mlp.shared_expert_gate.register_forward_hook(save_sharedgate_res))
                    for j in range(args.nsamples):
                        if 'deepseek' in args.model.lower():
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        elif 'qwen3-next' in args.model.lower() or 'qwen3next' in args.model.lower():
                            if position_embeddings is not None:
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                               position_ids=position_ids, position_embeddings=position_embeddings)[0]
                            else:
                                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                    routing_scores_shared = torch.tensor(routing_scores_shared)
                    for h in handles:
                        h.remove()
        # layer.mlp.static_observer=False

        for j in range(args.nsamples):
            if 'deepseek' in args.model.lower():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            elif 'qwen3-next' in args.model.lower() or 'qwen3next' in args.model.lower():
                if position_embeddings is not None:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                   position_ids=position_ids, position_embeddings=position_embeddings)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            # Check for NaN in layer output - this would propagate to next layer
            if torch.any(torch.isnan(outs[j])):
                logging.error(f'NaN detected in layer {i} output for sample {j}!')
                logging.error(f'Output shape: {outs[j].shape}, NaN count: {torch.isnan(outs[j]).sum()}')
                # Replace NaN with zeros to prevent propagation (temporary fix)
                logging.warning(f'Replacing NaN with zeros in layer {i} output to prevent propagation')
                outs[j] = torch.nan_to_num(outs[j], nan=0.0)

        # layers[i] = layer.cpu()
        layers[i] = layer
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers



       
@torch.no_grad()
def rtn_fwrd(model, dev, args, bit_mask):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            # if 'mlp.experts' in name and bit_mask is not None:
            #         idx = int(name.split('.')[2])
            #         if bit_mask[i][idx]==0:
            #             continue
                        # layer_weight_bits =4
            #         # elif bit_mask[i][idx] ==2:
            #         #     layer_weight_bits = 2
            # else:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                layer.self_attn.q_proj.weight.dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers
