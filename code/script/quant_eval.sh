#!/bin/bash

# 1. 声明一个关联数组
declare -A model_map

# 2. 定义映射关系
model_map["Qwen1.5-MoE-A2.7B"]="qwen15"
model_map["Qwen3-30B-A3B-Base"]="qwen3moe"
model_map["Qwen3-Next-80B-A3B-Instruct"]="qwen3next"
model_map["Mixtral-8x7B-v0.1"]="mixtral"
model_map["DeepSeek-V2-Lite"]="dsv2"

# ================= 配置区域 =================
W_BITS=2
N_SAMPLES=128
gsize=128
model_name='Mixtral-8x7B-v0.1'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

GPUS=(2 3)
GPU_STR=$(IFS=,; echo "${GPUS[*]}")
export CUDA_VISIBLE_DEVICES=$GPU_STR
# ===========================================

# --- 新增：w_asym 逻辑判断 ---
W_ASYM_SUFFIX=""
EXTRA_ARGS=""

if [ "$W_BITS" -eq 1 ]; then
    W_ASYM_SUFFIX="_wasym"
    EXTRA_ARGS="--w_asym"
    echo "检测到 W_BITS=1，已开启非对称量化 (--w_asym)"
fi
# ---------------------------

# 3. 获取对应的缩写
short_name=${model_map[$model_name]}
if [ -z "$short_name" ]; then
    short_name="unknown"
    echo "警告: 未找到模型 $model_name 的缩写映射"
fi

# 路径配置（在文件名中加入了 ${W_ASYM_SUFFIX}）
MODEL_PATH="/dataset/common/pretrain_model/${model_name}"
SAVE_PATH="/dataset/common/quant_model/moequant_${short_name}_w${W_BITS}${W_ASYM_SUFFIX}_selfebss_nsample_${N_SAMPLES}_groupsize${gsize}_rotate.pth"
RES_DIR="./res_moe_${short_name}_base_w${W_BITS}_selfebss_nsample_${N_SAMPLES}"
LOG_FILE="./output_log/${short_name}_w${W_BITS}${W_ASYM_SUFFIX}_nsample_${N_SAMPLES}_groupsize${gsize}_${TIMESTAMP}_rotate.txt"
CALIB_DATA="./EBSS_data/ebss_${short_name}_selfbuild_merged.jsonl"

echo "模型: $model_name (缩写: $short_name)"
echo "正在启动量化任务: W_BITS=$W_BITS, N_SAMPLES=$N_SAMPLES, ASYM=$([[ -n $EXTRA_ARGS ]] && echo "True" || echo "False")"

# --- 将运行信息写入日志 ---
{
    echo "==================== 任务启动信息 ===================="
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "模型名称: $model_name"
    echo "权重位数 (W_BITS): $W_BITS"
    echo "非对称量化 (W_ASYM): $([[ -n $EXTRA_ARGS ]] && echo "True" || echo "False")"
    echo "保存路径: $SAVE_PATH"
    echo "完整执行命令:"
    echo "python fake_quant/main.py --model $MODEL_PATH --w_bits $W_BITS $EXTRA_ARGS ..."
    echo "======================================================"
    echo -e "\n\n"
} > "$LOG_FILE" 2>&1

# 第一个任务：量化与保存
# 注意：在末尾添加了 $EXTRA_ARGS
python fake_quant/main.py \
    --model "$MODEL_PATH" \
    --fp32_had \
    --a_bits 16 \
    --w_bits "$W_BITS" \
    --v_bits 16 \
    --k_bits 16 \
    --bsz 1 \
    --w_groupsize $gsize \
    --a_groupsize $gsize \
    --w_clip \
    --save_qmodel_path "$SAVE_PATH" \
    --quant_test \
    --nsamples "$N_SAMPLES" \
    --human_res "$RES_DIR" \
    --EBSS_calib \
    --rotate \
    --calib_path "$CALIB_DATA" \
    --AGQ_GPTQ \
    --gate_up_group_size 8 \
    --down_group_size 8 \
    $EXTRA_ARGS \
    >> "$LOG_FILE" 2>&1

# 检查结果
if [ $? -eq 0 ]; then
    echo "第一阶段完成，正在执行 run.py..."
    python run.py ${GPUS[0]} ${GPUS[1]}
else
    echo "错误：量化脚本执行失败，请检查日志: $LOG_FILE"
    exit 1
fi