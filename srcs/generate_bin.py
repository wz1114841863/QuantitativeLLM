import os
import torch
import numpy as np
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from srcs.save_weights.save_layer_werights import load_saved_layer
from srcs.quantizer.real_quantize import real_quantize_tensor
from srcs.difference.differential_encoding import diff_encode_int4, diff_to_drle
from srcs.utils.run_lengths_calculate import compute_run_lengths
from srcs.utils.reorder import reorder_tile

"""
    文件说明:
        生成指定层的二进制文件, 包含量化权重, 差分编码, 游程编码等
        跟专用硬件输入绑定
"""


def generate_binary_file(layer_path, index, out_dir):
    """Generate a binary file for the specified layer index."""
    os.makedirs(out_dir, exist_ok=True)
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"Loaded layer {index} from {layer_path}: {info['layer_name']}")
    quantized, scale = real_quantize_tensor(
        weight, zero_point=False, group_size=128, rerurn_scale=True
    )
    reordered, reverse_indices = reorder_tile(quantized, tile_size=128)
    diff = diff_encode_int4(reordered, tile=128)
    codes_bytes, long_runs = diff_to_drle(diff)
    first = diff[::128].to(torch.uint8) & 0xF

    # TODO: 存在计算问题, 需要修复
    stats = {
        "orig_MB": weight.nbytes / 1024**2,
        "int4_MB": quantized.numel() * 0.5 / 1024**2,  # PyTorch Tensor
        "drle_MB": codes_bytes.size * 0.5 / 1024**2,  # NumPy array
        "index_MB": len(reverse_indices) / 1024**2,  # Python list
        "total_MB": (codes_bytes.size * 0.5 + len(reverse_indices)) / 1024**2,
        "long_run%": len(long_runs) / codes_bytes.size * 100,  # NumPy array
        "zero_run%": np.sum(codes_bytes >> 2 == 0) / codes_bytes.size * 100,
    }
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    output_dir = "./output_bin/"
    layer_path = "output_weights/facebook_opt-125m_layers/"
    index = 0
    generate_binary_file(layer_path, index, output_dir)
