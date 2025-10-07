import torch
from collections import Counter

from srcs.quantizer.real_quantize import real_quantize_tensor
from srcs.difference.differential_encoding import (
    diff_encode_int4,
    diff_encode_uint4,
    stat_diff,
)


def compute_run_lengths(quantized_weights):
    """Compute run-length encoding for a 1D tensor."""
    runs = []  # List of (value, run_length)
    len_counter = Counter()  # Count of run lengths

    if quantized_weights.numel() == 0:
        return runs, len_counter

    if len(quantized_weights.shape) != 1:
        quantized_weights = quantized_weights.flatten()

    current_val = quantized_weights[0]
    count = 1
    for val in quantized_weights[1:]:
        if val == current_val:
            count += 1
        else:
            len_counter[int(count)] += 1
            runs.append((int(current_val), count))
            current_val = val
            count = 1
    runs.append((current_val.item(), count))
    len_counter[int(count)] += 1

    return runs, len_counter


def run_length_analyze(
    layer_weight: torch.Tensor,
    method: str,
    zero_point: bool,
    group_size: int | None,
    tile: int = 256,
):
    """
    对单层权重做指定量化 + 差分编码,返回统一格式的指标字典.
    不打印, 只计算.
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
