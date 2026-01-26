#!/bin/bash

# 1. 声明一个关联数组
declare -A model_map

# 2. 定义映射关系
model_map["Qwen1.5-MoE-A2.7B"]="qwen15"
model_map["Qwen3-30B-A3B-Base"]="qwen3moe"
model_map["Qwen3-Next-80B-A3B-Instruct"]="qwen3next"
model_map["Mixtral-8x7B-v0.1"]="mixtral"
model_map["DeepSeek-V2-Lite"]="dsv2"

# 设置可见的 GPU
export CUDA_VISIBLE_DEVICES=0,1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 定义模型路径和结果输出路径（方便后续修改）
MODEL_PATH="/dataset/common/pretrain_model/Qwen1.5-MoE-A2.7B"
QMODEL_PATH="/dataset/common/quant_model/moequant_qwen15_w2_selfebss_nsample_128_groupsize128_rotate.pth"
RES_DIR="./res_moe_qwen15_base_w2_selfebss_nsample_128"
LOG_FILE="./output_log/eval_qwen15_w2_nsample_128_group128_${TIMESTAMP}.txt"

echo "开始执行量化测试..."

# 执行第一个 Python 任务：量化测试
# 使用 2>&1 将错误输出和标准输出都重定向到文件
python fake_quant/main.py \
    --model "$MODEL_PATH" \
    --load_qmodel_path "$QMODEL_PATH" \
    --quant_test \
    --human_res "$RES_DIR" \
    > "$LOG_FILE" 2>&1

# 检查上一个命令的退出状态码
if [ $? -eq 0 ]; then
    echo "量化测试完成，开始运行后续任务..."
    # 执行第二个 Python 任务
    #python run.py 7
else
    echo "警告：第一个任务失败，请检查日志 $LOG_FILE"
    exit 1
fi