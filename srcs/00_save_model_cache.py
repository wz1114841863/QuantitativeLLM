import os
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

"""
文件说明:
    提前下载一次大模型, 将其自动缓存在本地.
    先下载几个不同的大模型用于分析, 最后再按系列下载不同尺寸的模型.
"""
MODELS = [
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-6.7b",
    "huggyllama/llama-7b",
    "huggyllama/llama-13b",
]


def touch_model(model_name):
    print(f"[TOUCH] {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        config=config,
        device_map="auto",
    )
    print(f"[OK] Cached: {model_name}")


if __name__ == "__main__":
    for m in MODELS:
        touch_model(m)
