import os,sys
import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import gptq_utils_moe
# import gptq_utils_hessian_group
import eval_utils
import hadamard_utils
import bit_mask 
from quant_layers.quant_layer import QuantDecoderLayer,QuantRMSNorm,QuantLinear,QuantEmbedding,Quantizer
from evaluation.evaluate_mmlu import eval_mmlu
from evaluation.evaluate_humaneval import eval_humaneval
from evaluation.evaluate_gsm8k import eval_gsm8k
from transformers import AutoModelForCausalLM
import torch.nn as nn
import json
import torch.nn.functional as F
from evaluation import eval_lm
from tqdm import tqdm
from datasets import load_dataset
import random
def build_prompt(text):
    return text 

def eval_ppl_c4(model,tokenizer,seqlen=2048,limit=-1):
    c4_testdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(c4_testdata) - 1)
            tmp = tokenizer(c4_testdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    c4_testloader = torch.hstack(valenc)
    c4_ppl = eval_ppl_sliding(model, c4_testloader, window_len=512, stride=512, limit=limit, data_name="c4")
    return c4_ppl


def get_input_device(model):
    """
    For HF models dispatched with device_map='auto' (multi-GPU),
    input_ids should be placed on the same device as token embeddings.

    Tries common embedding module paths for your target models:
      Mixtral / Qwen / DeepSeek (remote code)
    Falls back to first parameter device.
    """
    candidates = [
        "model.embed_tokens",
        "model.model.embed_tokens",
        "model.model.model.embed_tokens",
        "transformer.wte",
        "model.transformer.wte",
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and hasattr(obj, "weight"):
            return obj.weight.device

    return next(model.parameters()).device



@torch.no_grad()
def _forward_logits(model, batch):
    """
    For your target models only:
      - Qwen3-30B-A3B-Base (qwen3moe)
      - Qwen1.5-MoE-A2.7B (qwen2moe)
      - DeepSeek-V2-Lite (deepseekv2)
      - Qwen3-Next-80B-A3B-Instruct (qwen3next)
      - Mixtral-8x7B-v0.1 (mixtral)

    Robustly extract logits from common HF / trust_remote_code outputs:
      - ModelOutput with .logits
      - dict with ['logits']
      - tuple/list where logits is the first element
    """
    # breakpoint()
    outputs = model(batch)

    # dict-style
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        # 极少数 remote_code 会用别的 key；这里宁可显式报错
        raise KeyError(f"Model output dict has no 'logits' key. Keys={list(outputs.keys())}")

    # HF ModelOutput-style
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits

    # tuple/list-style (legacy)
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        return outputs[0]

    raise TypeError(f"Unsupported model output type: {type(outputs)}")



@torch.no_grad()
def eval_ppl_sliding(model, test_loader, window_len=512, stride=512, limit=-1, data_name="wiki"):
    """
    HF-style strided sliding-window perplexity (multi-GPU safe):
      - window_len = max_length
      - stride = stride
      - only newly introduced tokens contribute to NLL
      - DOES NOT move the entire encoding to GPU (avoids OOM)
      - moves each window slice to embedding device (device_map='auto' safe)

    Args:
      test_loader:
        - if data_name=='wiki': dict with 'input_ids' (as in your data_utils loader)
        - else: LongTensor [1, N] or [N]
    """
    model.eval()

    # 1) Get encodings on CPU (do NOT .to(cuda) the whole thing)
    enc = test_loader["input_ids"] if data_name == "wiki" else test_loader
    if enc.dim() == 1:
        enc = enc.unsqueeze(0)  # [1, N]
    # Ensure on CPU for safety; slicing stays cheap
    if enc.is_cuda:
        enc = enc.cpu()

    # 2) Determine where inputs should live (embedding device)
    input_device = get_input_device(model)

    seq_len_total = enc.size(1)
    stride = max(1, min(stride, window_len))

    # 3) Build window end locations (end-based sliding)
    end_locs = list(range(window_len, seq_len_total + 1, stride))
    if len(end_locs) == 0:
        end_locs = [seq_len_total]
    elif end_locs[-1] != seq_len_total:
        end_locs.append(seq_len_total)

    if limit is not None and limit >= 0:
        end_locs = end_locs[: limit + 1]

    # 4) Accumulate on CPU scalars (avoid binding to a specific GPU)
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    with tqdm(range(len(end_locs))) as pbar:
        pbar.set_description_str("evaling ppl (sliding-window)")
        for k in pbar:
            end_loc = end_locs[k]
            begin_loc = max(0, end_loc - window_len)

            # window slice -> move to embedding device only
            input_ids = enc[:, begin_loc:end_loc].to(input_device, non_blocking=True)

            # tokens newly introduced since last step
            trg_len = end_loc - prev_end
            prev_end = end_loc

            target_ids = input_ids.clone()
            if trg_len < target_ids.size(1):
                target_ids[:, :-trg_len] = -100

            # Forward + shift
            logits = _forward_logits(model, input_ids)  # [1, L, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            # valid token count
            num_loss_tokens = int((shift_labels != -100).sum().item())
            if num_loss_tokens > 0:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_nll += float(loss.item())
                total_tokens += num_loss_tokens

            if total_tokens > 0:
                tmp_ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item())
                pbar.set_postfix_str(f"--{tmp_ppl:4.4}")

    if total_tokens == 0:
        return float("inf")
    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())



def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
        
    transformers.set_seed(args.seed)
    if 'qwen' in args.model.lower():
        model = model_utils.get_model(args.model, args.hf_token, args)
        test_loader = data_utils.get_loaders(
                args.eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=2048,
                hf_token=args.hf_token,
                eval_mode=True
            )
    elif 'deepseek' in args.model.lower():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map='auto',
            trust_remote_code=True
        )
        model.seqlen = 2048
        test_loader = data_utils.get_loaders(
                args.eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=2048,
                hf_token=args.hf_token,
                eval_mode=True
            )
    elif 'mixtral' in args.model.lower():
        from mixtral_model.modeling_mixtral import MixtralForCausalLM
        model = MixtralForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,
                                                    attn_implementation = "eager",device_map='auto',trust_remote_code=True)
        model.seqlen = 2048
        test_loader = data_utils.get_loaders(
                args.eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=2048,
                hf_token=args.hf_token,
                eval_mode=True
            )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)      
        utils.cleanup_memory(verbos=True)
        qlayers = model.model.layers
        if args.online_hadamard:
            for name in qlayers:
                if 'down_proj' in name:
                    if 'mlp.experts' in name:
                        had_K, K = hadamard_utils.get_hadK(model.config.moe_intermediate_size)
                    elif 'mlp.shared_expert' in name:
                        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if 'o_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
    # else:
    #     rotation_utils.fuse_layer_norms(model)
    #     quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
                
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            #assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict_w4 = torch.load(args.load_qmodel_path,map_location='cpu')
            model.load_state_dict(save_dict_w4["model"],strict=False)
            
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            # assert "llama" in args.model, "Only llama is supported for GPTQ!"
            if args.EBSS_calib:
                dataset = []
                cnt = 0
                with open(args.calib_path, encoding='utf-8') as file:
                    for line in file:
                        dataset.append(json.loads(line))
                        cnt = cnt +1
                        if cnt==args.nsamples:
                            break
                trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True) 
            else:
                model_name = args.model.lower()
                if 'deepseek' in model_name or 'mixtral' in model_name or 'qwen' in model_name:
                    trainloader = data_utils.get_loaders(
                        args.cal_dataset, nsamples=args.nsamples,
                        seed=args.seed, model=args.model,
                        seqlen=2048, eval_mode=False
                    )
            # load bit settings
            bit_settings = None
            if args.AGQ_GPTQ:
                quantizers = gptq_utils_moe.gptq_fwrd(model, tokenizer, trainloader, utils.DEV, args, bit_settings)
            else:
                quantizers = gptq_utils.gptq_fwrd(model, tokenizer, trainloader, utils.DEV, args, bit_settings)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            bit_settings=None
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args, bit_settings)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    if args.quant_test:  
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)  
        # ppl
        ppl = eval_ppl_sliding(model, test_loader, window_len=512, stride=512, limit=-1, data_name="wiki")
        print("wikitext2 ppl is:", ppl)
        ppl_c4 = eval_ppl_c4(model,tokenizer,seqlen=2048, limit=-1)
        print ('c4 ppl is: ', ppl_c4)

        print("\nRunning lm_eval Tasks...")
        from lm_eval.tasks.hendrycks_test import SUBJECTS
        import numpy as np

        # 1. 构建任务列表
        # MMLU 由 57 个子任务组成
        mmlu_tasks = [f"hendrycksTest-{sub}" for sub in SUBJECTS]
        
        # 其他指定的任务
        other_tasks = [
            "boolq",
            "arc_easy",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "piqa",
            "openbookqa",
            "mathqa"
        ]
        
        # 合并所有任务
        all_task_names = mmlu_tasks + other_tasks
        # all_task_names = other_tasks
        tasks_str = ",".join(all_task_names)

        print(f"Evaluating on {len(all_task_names)} tasks via lm_eval...")
        
        # 2. 运行评测
        results = eval_lm(model, tokenizer, tasks_str, num_fewshot=0, batch_size=32)
        
        # 3. 打印详细结果
        # print("Detailed Results:")
        # print(json.dumps(results, indent=2))

        # 4. 汇总并打印分数
        mmlu_accs = []
        print("\n--- Evaluation Summary ---")
        
        # 先打印非 MMLU 任务
        for task in other_tasks:
            if task in results['results']:
                metrics = results['results'][task]
                # 优先取 acc_norm, 没有则取 acc
                val = metrics.get('acc_norm', metrics.get('acc', 0))
                print(f"{task}: {val:.4f}")

        # 计算并打印 MMLU 平均分
        for task in mmlu_tasks:
            if task in results['results']:
                metrics = results['results'][task]
                val = metrics.get('acc_norm', metrics.get('acc', 0))
                mmlu_accs.append(val)
        
        if mmlu_accs:
            print(f"MMLU (Average): {np.mean(mmlu_accs):.4f}")

if __name__ == '__main__':
    main()
