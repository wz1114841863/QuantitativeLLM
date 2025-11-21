#!/bin/bash

# 1. 定义要测试的模型
MODELS=(
    "facebook/opt-125m"
    "facebook/opt-1.3b"

    "huggyllama/llama-7b"
    "huggyllama/llama-13b")

# 2. 定义要测试的 Group Sizes
GROUPS=(128 256 512 1024)

# 3. 循环运行
for model in "${MODELS[@]}"; do
    for gs in "${GROUPS[@]}"; do
        echo "Running experiment for $model with GS=$gs..."

        python 11_v3_compress_model_param.py \
            --model_id "$model" \
            --group_size "$gs" \
            --output_base "./paper_experiments" \
            --save_distribution

        echo "Done with $model GS=$gs"
    done
done
