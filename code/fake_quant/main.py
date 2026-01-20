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
import logging 
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
    # Detect model device for multi-GPU compatibility
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        model_device = model.model.embed_tokens.weight.device
    else:
        model_device = next(model.parameters()).device

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
    c4_ppl = eval_ppl_(model, c4_testloader, seqlen, limit,"c4")
    print(f'c4 ppl : {c4_ppl}')


# @torch.no_grad()
# def eval_ppl_(model,test_loader,seqlen=2048,limit=-1,data_name='wiki'):
#     nlls = []
#     if data_name=='wiki':
#         test_loader = test_loader['input_ids']
#     # test_loader = test_loader['input_ids']
#     nsamples = test_loader.numel() // seqlen
#     # nsamples = 1
#     # for i in tqdm(range(nsamples)):
#     with tqdm(range(nsamples)) as pbar:
#         pbar.set_description_str("evaling ppl")
#         for i in pbar:
#             batch = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)].to('cuda')
#             net_name = model.name.lower() if hasattr(model,"name") else type(model).__name__.lower()
#             if "opt" in net_name:
#                 outputs = model.model.model.decoder(batch)
#                 hidden_states = outputs[0]
#                 logits = model.model.lm_head(hidden_states)
#             elif "llama" in net_name or "mixtral" in net_name or "qwen" in net_name or 'deepseek' in net_name:
#                 outputs = model(batch)
#                 logits = outputs['logits'];outputs = None
#             elif "falcon" in net_name:
#                 outputs = model.model.transformer(batch)
#                 hidden_states = outputs[0]
#                 logits = model.model.lm_head(hidden_states)
#             elif "glm" in net_name:
#                 outputs = model(batch)
#                 logits = outputs['logits'];outputs = None
#             shift_logits = logits[:, :-1, :]
#             shift_labels = test_loader[:, (i * seqlen) : ((i + 1) * seqlen)][
#                 :, 1:
#             ].to(logits.device)
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(
#                 shift_logits.view(-1, shift_logits.size(-1)),
#                 shift_labels.view(-1),
#             )
#             neg_log_likelihood = loss.float() * seqlen
#             nlls.append(neg_log_likelihood)
#             tmp_ppl =  torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen)).item()
#             pbar.set_postfix_str(f"--{tmp_ppl:4.4}")
#             if i == limit:
#                 break
#     ppl = torch.exp(torch.stack(nlls).sum() / ((i+1) * seqlen))
#     return ppl.item()
@torch.no_grad()
def eval_ppl_(model, test_loader, seqlen=2048, limit=-1, data_name='wiki'):
    """
    使用 Hugging Face 标准滑动窗口 (Sliding Window) 计算 PPL。
    兼容 input_ids 为 Tensor 或 Dict 的情况。
    """
    # ------------------------------------------------------------------
    # 0. 检测模型所在设备 (支持多GPU分布式模型)
    # ------------------------------------------------------------------
    # 获取模型第一个参数所在的设备
    # 对于使用 device_map='auto' 的模型，不同层可能在不同设备上
    # 我们需要将输入发送到模型入口层所在的设备
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # 获取embedding层的设备
        model_device = model.model.embed_tokens.weight.device
    else:
        # 回退方案：获取第一个参数的设备
        model_device = next(model.parameters()).device

    print(f"Model device detected: {model_device}")

    # ------------------------------------------------------------------
    # 1. 更加稳健的数据提取逻辑 (仿照你之前的逻辑但更强壮)
    # ------------------------------------------------------------------
    input_ids = test_loader

    # 如果是 WikiText2 (通常是 BatchEncoding/Dict)，提取 input_ids
    if isinstance(test_loader, dict):
        if 'input_ids' in test_loader:
            input_ids = test_loader['input_ids']
        else:
            print("Warning: test_loader is a dict but has no 'input_ids'.")
    elif hasattr(test_loader, 'input_ids'): # 处理可能的 HF 对象
        input_ids = test_loader.input_ids

    # 此时 input_ids 应该是一个 Tensor，确保其形状为 [1, seq_len] 并移动到 GPU
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0) # 确保是 [1, seq_len]

    # 放到 CUDA (你源代码这里是在循环里放的，滑窗为了效率建议直接放，或者分块放)
    # 考虑到显存，我们在循环里切片后再 to(model_device)，这里先保持在 CPU 也可以
    # 但为了 input_ids.size(1) 获取方便，先不动的引用

    # ------------------------------------------------------------------
    # 2. 滑动窗口设置
    # ------------------------------------------------------------------
    max_length = seqlen
    stride = 512  # 标准滑动窗口步长
    total_len = input_ids.size(1)

    nlls = []
    print(f"Evaluating PPL with Sliding Window (stride={stride}, total_len={total_len})...")

    # ------------------------------------------------------------------
    # 3. 循环计算
    # ------------------------------------------------------------------
    # range(0, total_len, stride) 对应 HuggingFace 的标准做法
    for i in tqdm(range(0, total_len, stride), desc="PPL (Sliding)"):
        # 计算窗口的起止点
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, total_len)
        trg_len = end_loc - i # 本次窗口中真正要预测的有效长度

        # 如果有效长度 <= 0，通常是到了末尾，退出
        if trg_len <= 0:
            break

        # 取出当前窗口的 input_ids
        input_batch = input_ids[:, begin_loc:end_loc].to(model_device) # 使用检测到的模型设备
        
        # 构造标签 (Labels)
        # 逻辑：input_batch 包含了 [Context + Target]
        # 我们把 Context 部分的 label 设为 -100，这样计算 Loss 时就会忽略它们
        target_batch = input_batch.clone()
        target_batch[:, :-trg_len] = -100 

        # --------------------------------------------------------------
        # 4. 模型前向传播 (自动计算 Loss)
        # --------------------------------------------------------------
        # 大多数 HF 模型 (Llama, Mixtral, Qwen, DeepSeek) 都支持 labels 参数
        # 并在内部自动处理 shift logits，返回的 outputs.loss 就是我们要的
        try:
            outputs = model(input_batch, labels=target_batch)
            
            # 兼容性处理：有些模型输出可能是 tuple 或者没有 .loss
            if hasattr(outputs, 'loss'):
                neg_log_likelihood = outputs.loss
            else:
                # 如果模型没返回 loss，说明不支持自动计算，回退到手动计算
                # 这种情况很少见，除非是自定义模型结构
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # 手动 Shift
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_batch[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                neg_log_likelihood = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )

        except Exception as e:
            print(f"\nError at step {i}: {e}")
            break

        # 累加 Loss
        # 注意：HF 返回的 loss 通常是 mean reductions，我们需要还原回 sum 形式以便滑动平均？
        # 其实 HF 的 PPL 脚本通常直接存 loss (Per token NLL) 
        # 但标准的 PPL = exp( sum(loss) / N )
        # 由于我们每次 batch 的 trg_len 可能不同 (虽然大部分是 512)，
        # 严谨做法是：nll * trg_len 存下来，最后除以总长度。
        
        nlls.append(neg_log_likelihood * trg_len)

        # 检查 limit
        if limit != -1 and len(nlls) >= limit:
            break
            
        # 清理显存
        del input_batch, target_batch, outputs
        
    # ------------------------------------------------------------------
    # 5. 最终计算
    # ------------------------------------------------------------------
    if not nlls:
        return float('nan')
        
    # 总 Loss 求和 / 总有效 Token 数
    total_nll = torch.stack(nlls).sum()
    total_tokens = end_loc # 粗略估计，或者准确累加 trg_len
    # 这里的 total_tokens 应该等于 sum(trg_len for all steps)
    # 也就是 total_len (或者截断后的长度)
    
    # 修正：上面循环里 end_loc 最终会停在 total_len
    # 所以直接用 total_len 作为分母是比较准确的 (忽略了 limit 的情况)
    # 如果用了 limit，需要累加实际跑过的 trg_len
    
    ppl = torch.exp(total_nll / total_len)
    
    return ppl.item()



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
        # Use AutoModelForCausalLM to support both DeepSeek V1 and V2
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map='cuda',
            trust_remote_code=True
        )
        model.seqlen = 2048
        logging.info('---> Loaded DeepSeek Model (class: {})'.format(model.__class__.__name__))
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
            
            # [修改开始] 手动加载权重并显示进度条 --------------------------------------
            print("Loading checkpoint to CPU (this may take a while)...")
            save_dict_w4 = torch.load(args.load_qmodel_path, map_location='cpu')
            
            print("Applying state dict to model...")
            model_state_dict = model.state_dict()
            loaded_state_dict = save_dict_w4["model"]
            
            # 使用 tqdm 显示逐层加载进度
            from tqdm import tqdm 
            for key in tqdm(loaded_state_dict, desc="Loading weights"):
                if key in model_state_dict:
                    # 简单的形状检查，避免报错
                    if model_state_dict[key].shape == loaded_state_dict[key].shape:
                        model_state_dict[key].copy_(loaded_state_dict[key])
                    else:
                        print(f"Warning: Skipping {key} due to shape mismatch: {model_state_dict[key].shape} vs {loaded_state_dict[key].shape}")
                else:
                    # 模拟 strict=False 的行为
                    pass
            
            
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
                if 'deepseek' in args.model or 'mixtral' in args.model.lower() or 'qwen' in args.model.lower():
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

    # if args.quant_test:  
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)  
    #     # ppl
    #     ppl = eval_ppl_(model,test_loader,seqlen=2048,limit=-1,data_name='wiki')
    #     print ('wikitext2 ppl is:', ppl)
    #     ppl_c4 = eval_ppl_c4(model,tokenizer,seqlen=2048, limit=-1)
    #     print ('c4 ppl is: ', ppl_c4)
    #     # gsm8k
    #     acc = eval_gsm8k(model,tokenizer,1)
    #     print("gsm8k acc:%.5f\n"%acc)
        
    #     # mmlu 
    #     acc = eval_mmlu(model,tokenizer)
    #     # humaneval
    #     if not os.path.exists(args.human_res):
    #         os.makedirs(args.human_res)
    #     acc = eval_humaneval(model,tokenizer,args.human_res)
    #     print ('just get preds, you should run the human eval: ')
    #     print ('python human_eval/evaluate_functional_correctness.py HumanEval_res.json --problem_file=data/HumanEval.json')
    #     # # boolq 
    #     acc = eval_lm(model,tokenizer, 'boolq', 0, 2)
    #     print("boolq acc:",acc)
    #     # hellaswag
    #     acc = eval_lm(model,tokenizer, 'hellaswag', 0, 2)
    #     print("hellaswag acc:",acc)
    #     # openbookqa
    #     acc = eval_lm(model,tokenizer, 'openbookqa', 0, 2)
    #     print("openbookqa acc:",acc)
    #     # mathqa
    #     acc = eval_lm(model,tokenizer, 'mathqa', 0, 2)
    #     print("mathqa acc:",acc)
    if args.quant_test:  
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)  
        
        # ===================================================================
        # Part 1: 保留原有的 PPL 计算 (WikiText2 & C4)
        # ===================================================================
        print("Running PPL Evaluation...")
        # WikiText2 PPL
        ppl = eval_ppl_(model, test_loader, seqlen=2048, limit=-1, data_name='wiki')
        print ('wikitext2 ppl is:', ppl)
        
        # C4 PPL
        ppl_c4 = eval_ppl_c4(model, tokenizer, seqlen=2048, limit=-1)
        print ('c4 ppl is: ', ppl_c4)

        # ===================================================================
        # Part 2: 新增 - 使用 lm_eval 统一评测指定任务
        # ===================================================================
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
        # all_task_names = mmlu_tasks
        tasks_str = ",".join(all_task_names)

        print(f"Evaluating on {len(all_task_names)} tasks via lm_eval...")
        
        # 2. 运行评测
        # num_fewshot=0 (Zero-shot), batch_size 根据显存调整 (如 1, 2, 4)
        results = eval_lm(model, tokenizer, tasks_str, num_fewshot=0, batch_size=8)
        
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
