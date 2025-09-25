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
    diff = torch.zeros_like(tensor)
    diff[:, 0] = tensor[:, 0]
    diff[:, 1:] = tensor[:, 1:] - tensor[:, :-1]
    # 差分值域可能在 [-15, 15],需要合理编码
    if clamp:  # 困惑度直接NAN, 所有的-1等全部被截断为0了
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
    cov2 高 → 差分小幅抖动 → **变长编码(Huffman/Golomb)**受益
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


def diff_to_drle(diff_int8):
    """Convert differential encoded tensor to DRLE format."""
    diff = diff_int8.flatten().numpy().astype(np.int8)
    code_list = []  # 每个元素是5-bit整数
    long_runs = []

    def sym_map(d):
        if d == 0:
            return 0b00
        if d == 1:
            return 0b01
        if d == -1:
            return 0b10
        return 0b11  # ESC

    # 游程编码
    i, n = 0, diff.size
    while i < n:
        d = diff[i]
        if d != 0:
            # 非零只发一次
            code_list.append(sym_map(d) | (0b000 << 2))  # run=0
            i += 1
        else:
            # 零值游程
            run = 0
            while i < n and diff[i] == 0 and run < 7:
                run += 1
                i += 1
            # 输出 (0, run)
            code_list.append(0b00 | (run << 2))
            # 剩余零继续
            if i < n and diff[i] == 0:
                long_runs.append(int(diff[i:].size))  # 记录总长,调试用
                # 这里继续发 ESC,7 直到完
                remaining = diff[i:].size
                while remaining > 0:
                    seg = min(remaining, 7)
                    code_list.append(0b11 | (seg << 2))  # ESC + seg
                    remaining -= seg
                break

    # 5-bit 打包 → 1 byte / 包(简单做法)
    code_bytes = np.array(code_list, dtype=np.uint8)
    return code_bytes, long_runs
