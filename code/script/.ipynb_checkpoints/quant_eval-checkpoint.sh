#!/bin/bash

# 1. 声明一个关联数组（必须先声明）
declare -A model_map

# 2. 定义映射关系 [长名字]=缩写
model_map["Qwen1.5-MoE-A2.7B"]="qwen15"
model_map["Qwen3-30B-A3B-Base"]="qwen3moe"
model_map["Qwen3-Next-80B-A3B-Instruct"]="qwen3next"
model_map["Mixtral-8x7B-v0.1"]="mixtral"
model_map["DeepSeek-V2-Lite"]="dsv2"

# ================= 配置区域 =================
W_BITS=3
N_SAMPLES=32
gsize=128 # 不分组则-1
# 你在这里修改长名字
model_name='Qwen1.5-MoE-A2.7B' 
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 定义使用的 GPU 编号（用空格隔开）
GPUS=(3) 
# 将数组转换为逗号分隔的字符串，用于 CUDA_VISIBLE_DEVICES (即 "0,1")
GPU_STR=$(IFS=,; echo "${GPUS[*]}")
# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_STR
# ===========================================

# 3. 获取对应的缩写
# 如果找不到映射，默认使用 "unknown"
short_name=${model_map[$model_name]}
if [ -z "$short_name" ]; then
    short_name="unknown"
    echo "警告: 未找到模型 $model_name 的缩写映射"
fi


# 路径配置（现在全部使用变量，更加自动化）
MODEL_PATH="/dataset/common/pretrain_model/${model_name}"
SAVE_PATH="/dataset/common/quant_model/moequant_${short_name}_w${W_BITS}_selfebss_nsample_${N_SAMPLES}_groupsize${gsize}.pth"
RES_DIR="./res_moe_${short_name}_base_w${W_BITS}_selfebss_nsample_${N_SAMPLES}"
LOG_FILE="./output_log/${short_name}_w${W_BITS}_nsample_${N_SAMPLES}_groupsize${gsize}_${TIMESTAMP}.txt"
CALIB_DATA="./EBSS_data/ebss_${short_name}_selfbuild_merged.jsonl"

echo "模型: $model_name (缩写: $short_name)"
echo "正在启动量化任务: W_BITS=$W_BITS, N_SAMPLES=$N_SAMPLES, GROUP_SIZE=$gsize"

# --- 新增：将运行信息写入日志文件开头 ---
{
    echo "==================== 任务启动信息 ===================="
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "当前目录: $(pwd)"
    echo "模型名称: $model_name"
    echo "权重位数 (W_BITS): $W_BITS"
    echo "样本数量 (N_SAMPLES): $N_SAMPLES"
    echo "显卡设置: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "保存路径: $SAVE_PATH"
    echo "校准数据: $CALIB_DATA"
    echo "完整执行命令:"
    echo "python fake_quant/main.py --model $MODEL_PATH --w_bits $W_BITS --nsamples $N_SAMPLES --save_qmodel_path $SAVE_PATH --EBSS_calib --AGQ_GPTQ"
    echo "======================================================"
    echo -e "\n\n"
} > "$LOG_FILE" 2>&1  # 注意这里用 > 会先清空并写入上述信息

# 第一个任务：量化与保存
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
    --calib_path "$CALIB_DATA" \
    --AGQ_GPTQ \
    >> "$LOG_FILE" 2>&1

# 检查结果
if [ $? -eq 0 ]; then
    echo "第一阶段完成，正在执行 run.py..."
    python run.py ${GPUS[0]}
else
    echo "错误：量化脚本执行失败，请检查日志: $LOG_FILE"
    exit 1
fi