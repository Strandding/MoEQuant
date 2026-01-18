import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F

# ===========================
# 1. 专家激活监听器 (Hook)
# ===========================
class ExpertTracker:
    def __init__(self, model):
        self.activations = {}
        self.hooks = []
        self.layer_names = []
        
        # 自动识别 MoE 层并注册 Hook
        # 针对 Mixtral, Qwen1.5-MoE, DeepSeek-MoE 的常见命名
        for name, module in model.named_modules():
            # Mixtral / Qwen 的 Gate 层通常包含 'gate' 或 'router'
            # 且必须在 block_sparse_moe 结构下
            if ('gate' in name or 'router' in name) and isinstance(module, torch.nn.Linear):
                self.layer_names.append(name)
                # 注册前向传播 Hook
                self.hooks.append(module.register_forward_hook(self.get_hook(name)))
                
        print(f"Detected {len(self.layer_names)} MoE gating layers.")

    def get_hook(self, name):
        def hook(module, input, output):
            # output 通常是 router_logits [batch, seq_len, num_experts]
            # 我们需要 detach 并转到 CPU 避免显存爆炸
            self.activations[name] = output.detach()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

# ===========================
# 2. 核心构建逻辑
# ===========================
def build_ebss_dataset(
    model_path, 
    save_path, 
    source_dataset="allenai/c4", 
    subset="en", 
    num_experts=8, 
    top_k=2, 
    target_samples_per_expert=128, # 每个专家至少需要多少个样本支撑
    max_total_samples=2000,        # 最大总样本数，防止无限循环
    seq_len=2048
):
    print(f"Loading model from {model_path}...")
    # 加载模型 (FP16 即可，不需要量化)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    # 挂载监听器
    tracker = ExpertTracker(model)
    
    # 初始化专家计数器: [Layers, Experts]
    # 我们假设每一层专家数相同，可以通过 config 读取，这里为简化通用性手动指定或动态获取
    num_layers = len(tracker.layer_names)
    expert_counts = torch.zeros((num_layers, num_experts), dtype=torch.int32)
    
    print("Loading candidate dataset (streaming)...")
    try:
        # 使用 streaming 模式避免下载整个 C4
        dataset = load_dataset(source_dataset, subset, split="train", streaming=True)
    except:
        print("Failed to load C4, falling back to wikitext...")
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", streaming=True)

    selected_data = []
    total_processed = 0
    
    # 迭代数据流
    pbar = tqdm(total=max_total_samples)
    iterator = iter(dataset)
    
    while len(selected_data) < max_total_samples:
        try:
            sample = next(iterator)
        except StopIteration:
            break
            
        text = sample['text']
        if len(text) < 100: # 跳过过短的文本
            continue
            
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
        if inputs.input_ids.shape[1] < 128: # 跳过过短的序列
            continue
            
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 前向传播 (No Grad)
        tracker.clear()
        with torch.no_grad():
            _ = model(**inputs)
            
        # --- EBSS 核心筛选逻辑 ---
        is_valuable = False
        
        # 检查每一层的 Router 输出
        for layer_idx, layer_name in enumerate(tracker.layer_names):
            logits = tracker.activations[layer_name] # [1, seq, num_experts]
            
            # 计算 Top-K 选中的专家
            # 注意：不同模型 routing 机制不同，大多数是取 Top-K logits
            # 这里模拟标准的 Top-K Routing
            topk_weights, topk_indices = torch.topk(logits, top_k, dim=-1)
            
            # topk_indices: [1, seq, k]
            # 展平并去重，看这个样本激活了哪些专家
            active_experts_in_sample = torch.unique(topk_indices)
            
            # 检查这些专家是否“饥饿” (计数 < target)
            for expert_idx in active_experts_in_sample:
                expert_idx = expert_idx.item()
                if expert_idx < num_experts: # 安全检查
                    if expert_counts[layer_idx, expert_idx] < target_samples_per_expert:
                        is_valuable = True
                        # 只有当决定保留样本时，才真正更新计数？
                        # 或者为了贪婪搜索，只要有一个专家饥饿就保留，并更新所有被激活专家的计数
        
        # 如果该样本能填补任意专家的空缺，则保留
        if is_valuable:
            selected_data.append({"text": text})
            pbar.update(1)
            
            # 更新计数器 (模拟选中后的状态)
            for layer_idx, layer_name in enumerate(tracker.layer_names):
                logits = tracker.activations[layer_name]
                _, topk_indices = torch.topk(logits, top_k, dim=-1)
                active_experts = torch.unique(topk_indices)
                for exp_id in active_experts:
                    if exp_id < num_experts:
                        expert_counts[layer_idx, exp_id.item()] += 1
                        
            # 打印当前最不平衡的层的统计信息 (可选)
            if len(selected_data) % 10 == 0:
                min_coverage = expert_counts.min().item()
                pbar.set_description(f"Min Coverage: {min_coverage}")
                
            # 如果所有专家都满足了要求 (全覆盖)，可以提前停止 (可选)
            # if expert_counts.min() >= target_samples_per_expert:
            #     print("Achieved balanced coverage for all experts!")
            #     break

        total_processed += 1
        if total_processed > 20000: # 防止遍历太久找不到合适的
            print("Scanned too many samples, stopping early.")
            break
            
    pbar.close()
    tracker.remove_hooks()
    
    # 保存为 JSONL 格式 (与仓库格式一致)
    print(f"Saving {len(selected_data)} samples to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        for entry in selected_data:
            f.write(json.dumps(entry) + '\n')
            
    # 打印最终统计
    print("\nFinal Expert Coverage (First Layer):")
    print(expert_counts[0])

# ===========================
# 3. 运行入口
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the base model (e.g., Mixtral-8x7B)")
    parser.add_argument("--output", type=str, default="./EBSS_data/EBSS_new.jsonl", help="Output .jsonl file path")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts per layer")
    parser.add_argument("--top_k", type=int, default=2, help="Top-K routing parameter")
    parser.add_argument("--limit", type=int, default=128, help="Target samples per expert")
    
    args = parser.parse_args()
    
    build_ebss_dataset(
        model_path=args.model,
        save_path=args.output,
        num_experts=args.num_experts,
        top_k=args.top_k,
        target_samples_per_expert=args.limit
    )
