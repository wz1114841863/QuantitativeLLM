#!/bin/bash

# 1. 定义要测试的模型
MODELS=(
    "facebook/opt-125m"
    "facebook/opt-1.3b"
    "facebook/opt-6.7b"
    "huggyllama/llama-7b"
    "huggyllama/llama-13b")

# 2. 定义要测试的 Group Sizes
COMPRESSION_GROUPS=(128 256 512 1024)

# 调试:打印数组内容
echo "COMPRESSION_GROUPS array contains: ${COMPRESSION_GROUPS[@]}"

# 3. 循环运行
for model in "${MODELS[@]}"; do
    for gs in "${COMPRESSION_GROUPS[@]}"; do
        echo "Running experiment for $model with GS=$gs..."

        python ./srcs/12_v4_compress_model.py \
            --model_id "$model" \
            --group_size "$gs" \
            --output_base "./paper_experiments" \
            --save_distribution

        echo "Done with $model GS=$gs"
    done
done
