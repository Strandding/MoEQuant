import math
import time
import tqdm
import torch
import torch.nn as nn
import utils
import quant_utils
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def get_num_experts(config, model=None):
    if hasattr(config, 'num_experts') and config.num_experts is not None:
        return config.num_experts
    if hasattr(config, 'n_routed_experts') and config.n_routed_experts is not None:
        return config.n_routed_experts
    if hasattr(config, 'num_local_experts') and config.num_local_experts is not None:
        return config.num_local_experts
    if model is not None:
        for layer in getattr(model.model, 'layers', []):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                return len(layer.mlp.experts)
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
                return len(layer.block_sparse_moe.experts)
    return None

def get_top_k(config, default=2):
    if hasattr(config, 'num_experts_per_tok') and config.num_experts_per_tok is not None:
        return config.num_experts_per_tok
    return default

def _unpack_layer_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@torch.no_grad()
def bit_calib(model, dataloader, dev, args):

    logging.info('-----bit calculate-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    num_experts = get_num_experts(model.config, model)
    top_k = get_top_k(model.config, default=2)

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = model.lm_head.weight.dtype
    # dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    bit_mask = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    if num_experts is None:
        raise ValueError('Cannot determine number of experts for bit_mask.')
    bit_mask = torch.zeros(len(layers), num_experts)
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        sequential = list(full.keys())
        for k in range(len(sequential)):
            sequential[k]=[sequential[k]]


        gate_res = []
        def get_layer_output(name):
            def tmp(_, inp, out):
                gate_res.append(out)
            return tmp

        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                handles = []
                if 'mlp.gate' in name:
                    handles.append(subset[names[0]].register_forward_hook(get_layer_output(name)))
                    for j in range(args.nsamples):
                        outs[j] = _unpack_layer_output(layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids))
                for h in handles:
                    h.remove()
        top_cnt = torch.zeros(num_experts)
        score_cnt = torch.zeros(num_experts)
        for idx, gate in enumerate(gate_res):
            if isinstance(gate, (tuple, list)):
                if len(gate) < 2:
                    continue
                selected_experts = gate[0]
                routing_weights = gate[1]
            else:
                routing_weights = torch.nn.functional.softmax(gate, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            for m in range(selected_experts.shape[0]):
                for k in range(selected_experts.shape[1]):
                    expert_id = selected_experts[m][k]
                    top_cnt[expert_id] += 1
                    score_cnt[expert_id] += routing_weights[m][k]
        ave_score = score_cnt / torch.clamp(top_cnt, min=1)
        first_row_indices = torch.argsort(ave_score, descending=True)
        threshold = ave_score[first_row_indices[10]]
        bit_mask[i][ave_score>threshold]=1
        
        for j in range(args.nsamples):
            outs[j] = _unpack_layer_output(layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids))

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----bit accu Done-----\n')
    return bit_mask
