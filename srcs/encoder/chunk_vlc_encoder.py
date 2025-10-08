import torch


def chunk_vlc_len(quantized: torch.Tensor, tile: int = 128) -> float:
    """
    Chunk-VLC: |w|≤1 → 2 bit, |w|≤3 → 3 bit, 其余 → 4 bit
    返回平均码长 [bit/weight]
    """
    q = quantized.view(-1, tile)  # [N, 128]
    anchor = q.median(dim=1, keepdim=True)[0].round().clamp(0, 15)  # [N, 1]
    delta = q - anchor  # 以中位数为锚

    # 各区间占比
    mask1 = delta.abs() <= 1  # |Δ|≤1
    mask3 = delta.abs() <= 3  # |Δ|≤3
    cov1 = mask1.float().mean().item()
    cov3 = mask3.float().mean().item()
    cov3_only = (mask3 & ~mask1).float().mean().item()  # 1<|Δ|≤3
    cov4 = 1.0 - cov3  # |Δ|>3

    # 码长加权
    avg_bit = 2.0 * cov1 + 3.0 * cov3_only + 4.0 * cov4
    print(f"  Cov1: {cov1:.4f} | Cov3_only: {cov3_only:.4f} | Cov4: {cov4:.4f}")
    return avg_bit


def chunk_vlc_len_tmp(quantized, tile=128):
    """
    Chunk-VLC: |w|≤1 → 2 bit, |w|≤3 → 3 bit, 其余 → 4 bit
    返回平均码长 [bit/weight]
    """
    q = quantized.view(-1, tile)  # [N,128]
    anchor = q.median(dim=1, keepdim=True)[0].round().clamp(0, 15)  # [N,1] 中位数锚
    delta = q - anchor  # [-15,15] 以锚为中心
    cov1 = (delta.abs() <= 1).float().mean().item()  # |Δ|≤1
    cov3 = (delta.abs() <= 3).float().mean().item()  # |Δ|≤3

    # 码本位宽
    bit_1 = 2  # |w|≤1
    bit_3 = 3  # |w|≤3
    bit_4 = 4  # 其余

    # 平均码长
    print(f"  Cov1: {cov1:.4f} | Cov3: {cov3:.4f}")
    avg_bit = bit_1 * cov1 + bit_3 * (cov3 - cov1) + bit_4 * (1 - cov3)
    return avg_bit
