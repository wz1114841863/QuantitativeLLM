import torch
import numpy as np


def diff_encode_int4(W, tile=128, clamp=False):
    """Differential encoding for INT4 weights."""
    W = W.view(-1, tile)
    W_diff = torch.zeros_like(W)
    W_diff[:, 0] = W[:, 0]
    W_diff[:, 1:] = W[:, 1:] - W[:, :-1]
    # 差分会带来更大的值域[-15, 15], 需要合理编码, clamp操作会丢信息
    if clamp:  # 困惑度直接跑飞
        W_diff[:, 1:] = torch.round(W_diff[:, 1:]).clamp(-8, 7)
    return W_diff.view(-1)


def diff_encode_uint4(tensor_uint4, tile=128, clamp=False):
    """Differential encoding for zero-point weights."""
    tensor = tensor_uint4.view(-1, tile)
    diff = torch.zeros_like(tensor, dtype=torch.int8)  # 用int8装结果
    diff[:, 0] = tensor[:, 0].to(torch.int8)  # 基值
    diff[:, 1:] = tensor[:, 1:].to(torch.int8) - tensor[:, :-1].to(torch.int8)
    # 差分值域可能在 [-15, 15],需要合理编码
    if clamp:  # 困惑度直接NAN, 所有的负数等全部被截断为0了
        diff[:, 1:] = torch.round(diff[:, 1:]).clamp(0, 15)
    return diff.view(-1)


def diff_decode_int4(W_diff, tile=128):
    """Decode differential encoded INT4 weights."""
    W_diff = W_diff.view(-1, tile)
    W = torch.zeros_like(W_diff)
    W[:, 0] = W_diff[:, 0]
    for i in range(1, W.shape[1]):
        W[:, i] = W[:, i - 1] + W_diff[:, i]
    return W.view(-1)


def diff_decode_uint4(diff_uint4, tile=128):
    """Decode differential encoded UINT4 weights."""
    diff = diff_uint4.view(-1, tile)  # [n_tile, tile]
    rec = torch.zeros_like(diff, dtype=torch.float32)
    rec[:, 0] = diff[:, 0]
    for i in range(1, diff.shape[1]):
        rec[:, i] = rec[:, i - 1] + diff[:, i]
    return rec.view(-1)


def stat_diff(W_diff, tile=128):
    """
    same 高 → 零值多 → RLE / 稀疏编码受益
    cov2 高 → 差分小幅抖动 → 变长编码(Huffman/Golomb)受益
    long4 高 → 长重复片段多 → 游程编码受益
    """
    W_diff = W_diff.view(-1, tile)
    # 符号覆盖率
    cov2 = (W_diff.abs() <= 1).float().mean().item()  # |diff| ≤ 1
    cov3 = (W_diff.abs() <= 3).float().mean().item()  # |diff| ≤ 3
    same = (W_diff == 0).float().mean().item()  # diff == 0
    # 游程统计, 总"长游程 >= 3"个数 / 总元素数
    long4 = 0.0
    for row in W_diff:
        _, runlen = torch.unique_consecutive(row, return_counts=True)
        long4 += (runlen >= 3).sum().item()
    long4 /= W_diff.numel()
    return cov2, cov3, same, long4


def stat_diff_zp_centered(W_diff, zp, tile=128):
    """
    W_diff : 任意形状 Tensor,量化前的权重
    zp     : 形状 [G,1] 或能被 reshape 成 [G,1],每个元素是该 group 的唯一 zero-point
    tile   : 128
    return : cov2, cov3, same, long4
             全部以"W_diff - zp"作为误差序列再统计
    """
    W_diff = W_diff.view(-1, tile)  # [G, 128]
    zp = zp.view(-1, 1)  # [G, 1]  保证广播
    centered = W_diff - zp  # [G, 128]  以 zp 为零点

    # 符号覆盖率
    cov2 = (centered.abs() <= 1).float().mean().item()
    cov3 = (centered.abs() <= 3).float().mean().item()
    same = (centered == 0).float().mean().item()

    # 游程统计
    long4 = 0.0
    for row in centered:
        _, runlen = torch.unique_consecutive(row, return_counts=True)
        long4 += (runlen >= 3).sum().item()
    long4 /= centered.numel()

    return cov2, cov3, same, long4


def stat_diff_without_first(W_diff, tile=128):
    """
    W_diff: 已经做过 group-diff 的张量,shape 任意,会被 view(-1, tile)
    返回:跳过每块第 0 个样本后的覆盖率
    """
    W_diff = W_diff.view(-1, tile)  # [N, tile]
    # 去掉每块第 0 个元素
    delta_only = W_diff[:, 1:].contiguous()  # [N, tile-1]
    total = delta_only.numel()

    cov2 = (delta_only.abs() <= 1).float().sum() / total
    cov3 = (delta_only.abs() <= 3).float().sum() / total
    same = (delta_only == 0).float().sum() / total

    # 游程 ≥3 占比
    long4 = 0.0
    for row in delta_only:
        _, runlen = torch.unique_consecutive(row, return_counts=True)
        long4 += (runlen >= 3).sum().item()
    long4 /= total
    return cov2, cov3, same, long4



