import sys
import torch
import numpy as np

from transformers import AutoModelForCausalLM
from srcs.quantizer.real_quantize import real_quantize_tensor

"""
count_quantized_bits.py
独立脚本:仅计算 4-bit 分组量化后权重的总 bit/byte 数
用法:
  python count_quantized_bits.py facebook/opt-125m 512
"""


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python count_quantized_bits.py <model_id> [group_size=512]")
        sys.exit(1)

    model_id = sys.argv[1]
    group_size = int(sys.argv[2]) if len(sys.argv) == 3 else 512

    print(f"Loading model '{model_id}' ...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            # 量化
            q, zp, scale = real_quantize_tensor(
                w, zero_point=True, group_size=group_size, return_scale=128
            )
            total_elements += q.numel()

    total_bits = total_elements * 4  # 4-bit 量化
    total_bytes = total_bits / 8
    print("\n===  Quantized Weights Size  ===")
    print(f"Elements:     {total_elements:,}")
    print(f"Total bits:   {total_bits:,}")
    print(f"Total bytes:  {total_bytes:,.0f}  B")
    print(f"              {total_bytes/1024:,.2f}  KB")
    print(f"              {total_bytes/1024/1024:,.2f}  MB")


if __name__ == "__main__":
    main()
