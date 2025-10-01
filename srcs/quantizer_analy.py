from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import pandas as pd

from srcs.quantizer.real_quantize import *
from srcs.quantizer.pre_quant import get_named_linears
from srcs.utils.save_layer_werights import load_saved_layer
from srcs.difference.differential_encoding import (
    diff_encode_int4,
    diff_encode_uint4,
    diff_decode_int4,
    stat_diff,
)
from srcs.utils.run_lengths_calculate import compute_run_lengths
from srcs.utils.utils import (
    release_memory,
    save_quantized_weigths,
    save_log,
    save_json_file,
)

"""
    文件说明:
        对比不同的量化方法下, 差分编码前后的游程统计指标
"""


def analyze_layer_quant(
    layer_weight: torch.Tensor,
    method: str,
    zero_point: bool,
    group_size: int | None,
    tile: int = 256,
):
    """
    对单层权重做指定量化 + 差分编码,返回统一格式的指标字典.
    不打印,只计算.
    """
    quantized = real_quantize_tensor(
        layer_weight, zero_point=zero_point, group_size=group_size
    )

    # 原始量化值
    runs, len_counter = compute_run_lengths(quantized)
    zero_runs = [l for v, l in runs if v == 0]
    zero_ratio_orig = sum(zero_runs) / layer_weight.numel()

    # 差分编码
    diff_fn = diff_encode_uint4 if zero_point else diff_encode_int4
    diff_encoded = diff_fn(quantized, tile=tile)

    cov2, cov3, same, long4 = stat_diff(diff_encoded, tile=tile)
    runs_diff, _ = compute_run_lengths(diff_encoded)
    zero_runs_diff = [l for v, l in runs_diff if v == 0]
    zero_ratio_diff = sum(zero_runs_diff) / layer_weight.numel()

    return {
        "method": method,
        "tile": tile,
        "orig_runs": len(runs),
        "orig_zero_ratio": zero_ratio_orig,
        "diff_runs": len(runs_diff),
        "diff_zero_ratio": zero_ratio_diff,
        "cov2": cov2,
        "cov3": cov3,
        "same": same,
        "long4": long4,
    }


def scan_model_layers(
    layer_path: str,
    start_idx: int = 0,
    end_idx: int | None = None,
    method_filter: str = "group_zero_point",
    tile: int = 256,
):
    """
    顺序加载模型指定区间层,统一用method_filter量化并返回 DataFrame.
    """
    # 解析策略
    cfg_map = {
        "real_symm": ("real_symm", False, None),
        "real_zero_point": ("real_zero_point", True, None),
        "group_symm": ("group_symm", False, 128),
        "group_zero_point": ("group_zero_point", True, 128),
    }
    if method_filter not in cfg_map:
        raise ValueError(f"Unknown method_filter: {method_filter}")
    method, zero_point, gs = cfg_map[method_filter]

    records: List[Dict] = []
    if end_idx is None:
        end_idx = start_idx + 1

    for idx in range(start_idx, end_idx):
        try:
            weight, bias, info = load_saved_layer(layer_path, idx)
        except FileNotFoundError:
            break
        rec = analyze_layer_quant(weight, method, zero_point, gs, tile=tile)
        rec["layer_idx"] = idx
        rec["layer_name"] = info["layer_name"]
        records.append(rec)

    return pd.DataFrame(records).set_index("layer_idx")


if __name__ == "__main__":
    group_size = 128
    strategies = [
        ("real_symm", False, None),
        ("real_zero_point", True, None),
        ("group_symm", False, group_size),
        ("group_zero_point", True, group_size),
    ]
    output_path = "output_weights/facebook_opt-125m_layers"
    df = scan_model_layers(
        output_path,
        start_idx=0,
        end_idx=2,
        method_filter="group_zero_point",
        tile=256,
    )

    print(df[["layer_name", "cov2", "same", "long4"]].describe())
