import numpy as np
import torch

from srcs.utils.save_layer_werights import load_saved_layer
from srcs.quantizer.real_quantize import *
from srcs.difference.differential_encoding import (
    diff_encode_uint4,
    stat_diff,
)
from srcs.utils.run_lengths_calculate import compute_run_lengths
from srcs.utils.reorder import reorder_tile, reorder_tile_v2

"""
文件说明:
    采用重排, 统计重排后的信息, 对重排索引采用差分看看是否可压缩
"""


def diff_encode_indices(reverse_indices, tile_size=128):
    """
    reverse_indices: list[np.ndarray] 每块长度=tile_size
    返回差分后的 1-D np.ndarray(int16/int8 按需)+ 编码元信息
    """
    # 1. 拼成 [G*tile_size]
    idx_all = np.concatenate(reverse_indices)  # dtype 可能是 uint8/uint16
    # 2. tile 内差分:首元素原值,后续差分
    diff = np.zeros_like(idx_all, dtype=np.int16)  # 先放宽
    for g in range(0, len(idx_all), tile_size):
        tile = idx_all[g : g + tile_size]
        diff[g] = tile[0]  # 首值保留
        diff[g + 1 : g + tile_size] = np.diff(tile)  # 其余差分
    # 3. 看能否压到 int8 / uint8
    if diff.min() >= -128 and diff.max() <= 127:
        diff8 = diff.astype(np.int8)
        return diff8, {"dtype": "int8", "tile_size": tile_size}
    elif diff.min() >= 0 and diff.max() <= 255:
        diff8 = diff.astype(np.uint8)
        return diff8, {"dtype": "uint8", "tile_size": tile_size}
    else:
        return diff.astype(np.uint16), {"dtype": "uint16", "tile_size": tile_size}


def stat_diff_idx(diff_indices, tile=128):
    """
    完全对标你现有的 stat_diff,只是输入换成"索引差分"
    返回:cov2, cov3, same, long4
    含义不变:
        same 高 → 大量 0 差分 → RLE/稀疏编码受益
        cov2 高 → |diff|≤1 多 → 变长编码受益
        long4 高 → 长重复差分 → 游程编码受益
    """
    # 统一转成 tensor,方便用 GPU/CPU 同一套逻辑
    if isinstance(diff_indices, np.ndarray):
        diff_indices = torch.from_numpy(diff_indices)
    # 按 tile 切
    diff_indices = diff_indices.view(-1, tile)
    # 符号覆盖率
    cov2 = (diff_indices.abs() <= 1).float().mean().item()
    cov3 = (diff_indices.abs() <= 3).float().mean().item()
    same = (diff_indices == 0).float().mean().item()
    # 长游程 ≥3
    long4 = 0.0
    for row in diff_indices:
        _, runlen = torch.unique_consecutive(row, return_counts=True)
        long4 += (runlen >= 3).sum().item()
    long4 /= diff_indices.numel()
    return cov2, cov3, same, long4


def reoder_analy(layer_path, index=0):
    """对权重进行重排, 统计重排后分布, 对重排索引进行差分, 统计压缩"""
    tile = 128
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized = real_quantize_tensor(weight, zero_point=True, group_size=128)

    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    cov2, cov3, same, long4 = stat_diff(diff_encoded, tile=tile)
    runs_diff, _ = compute_run_lengths(diff_encoded)
    zero_runs_diff = [l for v, l in runs_diff if v == 0]
    zero_ratio_diff = sum(zero_runs_diff) / weight.numel()

    print("[差分编码后]")
    print(f"  Runs: {len(runs_diff):>6} | ZeroRatio: {zero_ratio_diff:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )

    w_reorder, perm = reorder_tile_v2(quantized, tile_size=128)
    diff_reorder = diff_encode_uint4(w_reorder, tile=tile)
    cov2, cov3, same, long4 = stat_diff(diff_reorder, tile=tile)
    runs_diff, _ = compute_run_lengths(diff_reorder)
    zero_runs_diff = [l for v, l in runs_diff if v == 0]
    zero_ratio_diff = sum(zero_runs_diff) / weight.numel()

    print("[重排后]")
    print(f"  Runs: {len(runs_diff):>6} | ZeroRatio: {zero_ratio_diff:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )

    # 对indices进行分析
    diff_idx, meta = diff_encode_indices(perm, tile_size=128)
    # 2. 统计差分特性
    cov2, cov3, same, long4 = stat_diff_idx(diff_idx, tile=128)
    print("[索引差分后]")
    print(
        f"  dtype: {meta['dtype']} | Same: {same:.4f} | Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Long4: {long4:.4f}"
    )


if __name__ == "__main__":
    # layer_path = "output_weights/facebook_opt-1.3b_layers"
    layer_path = "output_weights/facebook_opt-125m_layers/"
    for i in range(0, 2):
        reoder_analy(layer_path, index=i)
