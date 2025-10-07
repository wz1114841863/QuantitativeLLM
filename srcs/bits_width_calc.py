import numpy as np
import matplotlib.pyplot as plt
import torch
import huffman


from collections import Counter
from pathlib import Path
from datetime import datetime

from srcs.quantizer.real_quantize import *
from srcs.quantizer.pre_quant import get_named_linears
from srcs.utils.save_layer_werights import load_saved_layer
from srcs.difference.differential_encoding import (
    diff_encode_int4,
    diff_encode_uint4,
    diff_decode_int4,
    stat_diff,
    stat_diff_without_first,
)
from srcs.utils.run_lengths_calculate import compute_run_lengths
from srcs.utils.utils import (
    release_memory,
    save_quantized_weigths,
    save_log,
    save_json_file,
)
from srcs.utils.reorder import reorder_tile
from srcs.encoder.range_encoder import RangeCoder4Bit, RangeCoder31

"""
文件说明:
    计算采用不同的编码后, 权重的平均位宽
"""


def load_layer_diff_weights(layer_path, index):
    """Load the weights of a specific layer from saved files."""
    group_size = 128
    tile = group_size
    zero_point = True
    strategies = [
        ("real_symm", False, None),
        ("real_zero_point", True, None),
        ("group_symm", False, group_size),
        ("group_zero_point", True, group_size),
    ]

    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized = real_quantize_tensor(
        weight, zero_point=zero_point, group_size=group_size
    )


def calc_layer_weights_width(layer_path, index):
    """Load the weights of a specific layer from saved files."""
    group_size = 128
    tile = group_size
    zero_point = True

    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    orig_elem = weight.numel()
    orig_bits = 32  # 假设原始 fp32
    orig_bytes = orig_elem * 4  # 字节

    # ---------- 量化 ----------
    quantized = real_quantize_tensor(
        weight, zero_point=zero_point, group_size=group_size
    )
    q_elem = quantized.numel()
    q_bits = 4  # INT4
    q_bytes_bit = q_elem * 4  # 位字节 = 元素数 * 4bit

    print(f"\nLayer: {name}")
    print("[量化前]")
    print(
        f"  Elements: {orig_elem:>10} | Bit-width: {orig_bits} bit | Storage: {orig_bytes:>10.2f} B"
    )
    print("[量化后 INT4]")
    print(
        f"  Elements: {q_elem:>10} | Bit-width: {q_bits} bit | Storage: {q_bytes_bit/8:>10.2f} B | Compression: {(1-q_bits/orig_bits)*100:6.2f}%"
    )

    # ---------- chunk_vlc 平均位宽 ----------
    rle_bit_per_weight = chunk_vlc_len(quantized, tile=tile)  # 你的函数
    rle_bytes_bit = q_elem * rle_bit_per_weight  # 总 bit
    print(f"[chunk_vlc后]")
    print(
        f"  Avg bit/weight: {rle_bit_per_weight:8.3f} | Storage: {rle_bytes_bit/8:>10.2f} B | vs INT4: {(1-rle_bit_per_weight/4)*100:6.2f}% vs FP32: {(1-rle_bit_per_weight/32)*100:6.2f}%"
    )


def zp_rle_len(quantized, tile=128):
    """返回 zp-anchor RLE 的平均码长 [bit/weight]"""
    q = quantized.view(-1, tile)  # [N,128]
    zp = q.median(dim=1, keepdim=True)[0].round().clamp(0, 15)  # [N,1]  per-group zp
    print(f"  Zero-point (median) range: {zp.min().item()} ~ {zp.max().item()}")
    mask = q == zp  # [N,128]  bool
    run_len = mask.sum(dim=1)  # 每块 run 长度
    n_zp = run_len.sum().item()  # 总 zp 样本数
    n_other = q.numel() - n_zp  # 非 zp 样本数
    n_run_block = (mask != 0).any(dim=1).sum().item()  # 有 run 的块数

    # 码流 bit 数
    bits_zp_run = n_run_block * 1  # 每块 1 bit run-flag
    bits_zp_len = (run_len > 1).sum().item() * 3  # run≥2 时 3 bit len-1
    bits_other = n_other * 5  # 非 zp: 1+4 bit
    total_bits = bits_zp_run + bits_zp_len + bits_other

    return total_bits / q.numel()  # bit / weight


def zp_rle_len_encode_zp(quantized, tile=128):
    """
    编码阶段重新选 zp(中位数/mode),
    并把 4 bit zp 值写进码流 → 返回总位宽 [bit/weight]
    """
    q = quantized.view(-1, tile)  # [N,128]
    # ① 重新选 zp(中位数)
    new_zp = q.median(dim=1, keepdim=True)[0].round().clamp(0, 15)  # [N,1]
    mask = q == new_zp  # [N,128]  bool
    run_len = mask.sum(dim=1)  # 每块 run 长度
    n_zp = run_len.sum().item()  # 总 zp 样本数
    n_other = q.numel() - n_zp  # 非 zp 样本数
    n_run_block = (run_len > 0).sum().item()  # 有 run 的块数

    # ② 编码位宽
    bits_zp_run_flag = n_run_block * 1  # 每块 1 bit:是否有 zp-run
    bits_zp_len = (run_len > 1).sum().item() * 3  # run≥2 时 3 bit len-1
    bits_other = n_other * 5  # 非 zp: 1+4 bit
    bits_zp_value = new_zp.numel() * 4  # 每块 4 bit 新 zp 值

    # ③ 总位宽
    total_bits = bits_zp_run_flag + bits_zp_len + bits_other + bits_zp_value
    return total_bits / q.numel()  # bit / weight


def chunk_vlc_len(quantized, tile=128):
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


def quick_profit(layer_path, index):
    group_size = 128
    tile = 32
    # print(f"tile: {tile}, group_size: {group_size}")
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    syms = diff_encoded.int().tolist()

    # syms = quantized.view(-1).int().tolist()

    freq = Counter(syms)
    huff_book = huffman.codebook(freq.items())
    bpw = sum(len(huff_book[s]) * c for s, c in freq.items()) / len(syms)
    sram_save_mb = 125e6 * (4 - bpw) / 8 / 1e6  # MB
    print(f"码长 {bpw:.2f} b → SRAM 省 {sram_save_mb:.1f} MB")


# ERROR:
def range_encode(layer_path, index):
    """Range 编码"""
    group_size = 128
    tile = 128
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")

    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )
    delta = diff_encode_uint4(quantized, tile=tile)
    sym = delta + 16
    syms = sym.int().tolist()

    freq = Counter(syms)
    coder = RangeCoder31(freq)
    byte_stream = coder.encode(syms)
    bpw = len(byte_stream) * 8 / len(syms)  # 每权重要多少 bit

    sram_save_mb = 125e6 * (4 - bpw) / 8 / 1e6
    print(f"码长 {bpw:.2f} b → SRAM 省 {sram_save_mb:.1f} MB")


def rle_encode(layer_path, index):
    """RLE 编码"""
    group_size = 128
    tile = 128
    # print(f"tile: {tile}, group_size: {group_size}")
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_encoded = diff_encode_uint4(quantized, tile=tile, clamp=True)
    syms = diff_encoded.int().tolist()

    # syms = quantized.view(-1).int().tolist()

    out = []
    i, n = 0, len(syms)
    while i < n:
        val = syms[i]
        cnt = 1
        while i + cnt < n and syms[i + cnt] == val and cnt < 15:  # 4-bit count
            cnt += 1
        out.append((val, cnt))
        i += cnt
    return syms, out  # list[(value, count)]


def rle_bpw(layer_path, index):
    syms, rle = rle_encode(layer_path, index)
    total_bits = sum(4 + 4 for (v, c) in rle)  # 4b value + 4b count
    print(f"RLE runs: {len(rle)}, total_bits / len(syms): {total_bits / len(syms)}")
    # return total_bits / len(syms)


def diff_to_drle_bits(layer_path, index):
    """RLE 编码"""
    group_size = 128
    tile = 128
    # print(f"tile: {tile}, group_size: {group_size}")
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_int8 = diff_encode_uint4(quantized, tile=tile, clamp=True)
    diff = diff_int8.flatten().numpy().astype(np.int8)
    total_bits = 0

    i, n = 0, len(diff_int8)
    while i < n:
        d = int(diff[i])
        if d == 0:  # 0000 + 3b run
            run = min(7, (diff[i : i + 7] == 0).sum())
            total_bits += 4 + 3  # 7 b
            i += run
        elif d in {1, -1, 2}:  # 2 b 短码
            total_bits += 2
            i += 1
        else:  # 其余值 6 b
            total_bits += 5  # 2 b 前缀 + 4 b 值
            i += 1
    print(f"DRLE total_bits / len(syms): {total_bits / len(diff_int8)}")
    return total_bits / len(diff_int8)


# ERROR: golomb_rice_bits只适合非负整数
def golomb_rice_bits(layer_path, index, k=2):
    """差分 int8 → Golomb-Rice 总 bit 数"""
    group_size = 128
    tile = 128
    # print(f"tile: {tile}, group_size: {group_size}")
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_int8 = diff_encode_uint4(quantized, tile=tile)
    diff = diff_int8.int().numpy().astype(np.int8)
    total_bits = 0
    for d in diff:
        d = int(d)
        if (d > 15) or (d < -16):
            print(f"Warning: diff value {d} out of range [-16, 15], clipping.")
        d = np.clip(d, -16, 15)
        q = d >> k
        r = d & ((1 << k) - 1)  # ② 商和余都来自截断后的值
        total_bits += (q + 1) + k
    print(f"Golomb-Rice k={k} total_bits / len(syms): {total_bits / len(diff_int8)}")
    return total_bits, total_bits / len(diff)


def fixed_width(layer_path, index):
    group_size = 128
    tile = 128
    # print(f"tile: {tile}, group_size: {group_size}")
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    syms = diff_encoded.int()

    total = syms.numel()
    le1 = (syms.abs() <= 1).sum().item()  # |Δw|≤1
    le3 = (syms.abs() <= 3).sum().item()  # |Δw|≤3
    ratio_le1 = le1 / total
    ratio_le3 = le3 / total

    header_bpw = 2 / 128  # 每 128 权重 2 bit 头
    exp_bpw = header_bpw + (ratio_le3 * 3 + (1 - ratio_le3) * 5)  # 3 or 5 bit

    print(f"Δw |≤1 占比 : {ratio_le1:5.1%} |≤3 占比 : {ratio_le3:5.1%}")
    print(f"定长3/5+flag 期望码长 : {exp_bpw:.2f} bit")


def fixed_width_group(layer_path, index):
    group_size = 128
    tile = 128
    # print(f"tile: {tile}, group_size: {group_size}")
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    delta = diff_encoded.int()

    n_tot = delta.numel()
    n_blk = n_tot // group_size
    delta = delta[: n_blk * group_size]  # 扔掉尾部不足
    blocks = delta.view(n_blk, group_size)  # [n_blk, 128]

    # 4. 组级窄/快判定
    narrow_mask = (blocks.abs() <= 3).all(dim=1)  # 能否 3-bit
    fast_mask = (blocks.abs() <= 1).all(dim=1)  # 能否 fast-path

    ratio_narrow = narrow_mask.float().mean().item()
    ratio_fast = fast_mask.float().mean().item()

    # 5. 真实期望码长(定长 3/5 + 2 bit header /128)
    header_bpw = 2 / group_size
    exp_bpw = header_bpw + ratio_narrow * 3 + (1 - ratio_narrow) * 5

    # 6. 输出
    print(f"Δw |≤1 整组占比 : {ratio_fast:5.1%}  |≤3 整组占比 : {ratio_narrow:5.1%}")
    print(f"真实定长3/5+flag 期望码长 : {exp_bpw:.3f} bit")


def fixed_width_S1_3(layer_path, index):
    group_size = 128
    tile = 128
    # print(f"tile: {tile}, group_size: {group_size}")
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, scale = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    delta = diff_encoded.int().view(-1, tile)

    n_tile = delta.shape[0]  # 多少 tile 行
    delta = delta[:n_tile]  # 保持 [n_tile, 128]
    exc_per_tile = (delta.abs() > 3).sum(dim=1)  # 按 tile 行统计
    max_exc = exc_per_tile.max().item()
    avg_exc = exc_per_tile.float().mean().item()
    pct_over8 = (exc_per_tile > 8).float().mean().item()

    print(f"tile行 全局≤3 : {(delta.abs() <= 3).sum().item() / delta.numel():5.1%}")
    print(
        f"tile行 逐块异常 | max={max_exc:2d}  avg={avg_exc:4.2f}  over-8={pct_over8:5.1%}"
    )


if __name__ == "__main__":
    layer_path = "output_weights/facebook_opt-125m_layers/"
    # layer_path = "output_weights/EleutherAI_gpt-neo-2.7B_layers/"

    for index in range(10, 20):
        # results = quick_profit(layer_path, index)
        # results = rle_bpw(layer_path, index)
        # results = diff_to_drle_bits(layer_path, index)
        # results = golomb_rice_bits(layer_path, index, k=2)
        results = range_encode(layer_path, index)
        # results = fixed_width(layer_path, index)
        # results = fixed_width_group(layer_path, index)
        # results = fixed_width_S1_3(layer_path, index)
