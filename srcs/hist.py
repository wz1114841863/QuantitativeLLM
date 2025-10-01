import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

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

"""
    文件说明:
        统计游程长度的分布, 画出直方图
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

    runs, len_counter = compute_run_lengths(quantized)
    cov2, cov3, same, long4 = stat_diff(quantized, tile=tile)
    runs_diff, _ = compute_run_lengths(quantized)
    zero_runs = [l for v, l in runs if v == 0]
    zero_ratio_orig = sum(zero_runs) / weight.numel()
    print("[原始量化值]")
    print(f"  Runs: {len(runs):>6} | ZeroRatio: {zero_ratio_orig:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )

    diff_encoded = diff_encode_uint4(quantized, tile=tile)
    cov2, cov3, same, long4 = stat_diff(diff_encoded, tile=tile)
    runs_diff, _ = compute_run_lengths(diff_encoded)
    zero_runs_diff = [l for v, l in runs_diff if v == 0]
    zero_ratio_diff = sum(zero_runs_diff) / weight.numel()
    quant_flat = quantized.flatten().cpu().numpy()
    diff_flat = diff_encoded.flatten().cpu().int()
    print("[差分编码后]")
    print(f"  Runs: {len(runs_diff):>6} | ZeroRatio: {zero_ratio_diff:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )

    return {
        "name": name,
        "quant": quant_flat,
        "delta": diff_flat,
        "cov2": cov2,
        "cov3": cov3,
        "same": same,
        "long4": long4,
        "runs": runs_diff,
    }


def load_layer_symm(layer_path, index):
    """Load the weights of a specific layer from saved files."""
    group_size = 128
    tile = group_size
    zero_point = False
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

    runs, len_counter = compute_run_lengths(quantized)
    cov2, cov3, same, long4 = stat_diff(quantized, tile=tile)
    runs_diff, _ = compute_run_lengths(quantized)
    zero_runs = [l for v, l in runs if v == 0]
    zero_ratio_orig = sum(zero_runs) / weight.numel()
    print("[原始量化值]")
    print(f"  Runs: {len(runs):>6} | ZeroRatio: {zero_ratio_orig:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )

    diff_encoded = diff_encode_int4(quantized, tile=tile)
    cov2, cov3, same, long4 = stat_diff(diff_encoded, tile=tile)
    runs_diff, _ = compute_run_lengths(diff_encoded)
    zero_runs_diff = [l for v, l in runs_diff if v == 0]
    zero_ratio_diff = sum(zero_runs_diff) / weight.numel()
    quant_flat = quantized.flatten().cpu().numpy()
    diff_flat = diff_encoded.flatten().cpu().int()
    print("[差分编码后]")
    print(f"  Runs: {len(runs_diff):>6} | ZeroRatio: {zero_ratio_diff:.4f}")
    print(
        f"  Cov2: {cov2:.4f} | Cov3: {cov3:.4f} | Same: {same:.4f} | Long4: {long4:.4f}"
    )

    return {
        "name": name,
        "quant": quant_flat,
        "delta": diff_flat,
        "cov2": cov2,
        "cov3": cov3,
        "same": same,
        "long4": long4,
        "runs": runs_diff,
    }


def plot_diff_histogram_split(stat_dict, bins=33, save_dir="plt_figures"):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    name = stat_dict["name"]
    quant = stat_dict["quant"]  # 原始 INT4
    delta = stat_dict["delta"]  # 差分后

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)  # 同高
    bin_range = (-16, 16)

    # 左:原始 INT4
    ax1.hist(
        quant,
        bins=bins,
        range=bin_range,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
    )
    ax1.set_title(f"Original INT4  -  {name}")
    ax1.set_xlabel("Quantized Value")
    ax1.set_ylabel("Count")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 右:差分后
    ax2.hist(
        delta,
        bins=bins,
        range=bin_range,
        color="orangered",
        edgecolor="black",
        linewidth=0.3,
    )
    ax2.set_title(f"After Diff  -  {name}")
    ax2.set_xlabel("Delta Value")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # 零线高亮
    for ax in (ax1, ax2):
        ax.axvline(0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    time_stamp = datetime.now().strftime("%m%d_%H%M%S")
    file_name = f"{time_stamp}_{name.replace('/', '_')}.png"
    save_path = save_dir / file_name
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Split histogram saved → {save_path}")


def plot_two_diff_histograms(stat_dict_A, stat_dict_B, bins=33, save_dir="plt_figures"):
    """
    同时画两个层的 Original vs Diff 对比
    A 在左列,B 在右列,上下分栏:
        上:Original INT4
        下:After Diff
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 提取数据
    name_A, quant_A, delta_A = (
        stat_dict_A["name"],
        stat_dict_A["quant"],
        stat_dict_A["delta"],
    )
    name_B, quant_B, delta_B = (
        stat_dict_B["name"],
        stat_dict_B["quant"],
        stat_dict_B["delta"],
    )
    bin_range = (-16, 16)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
    # 行 0:Original   行 1:Diff
    # 列 0:A          列 1:B

    # ---------- Original ----------
    axes[0, 0].hist(
        quant_A,
        bins=bins,
        range=bin_range,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0, 0].set_title(f"Original INT4 - {name_A}")
    axes[0, 0].set_xlabel("Quantized Value")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0, 0].axvline(0, color="black", linestyle="--", linewidth=1)

    axes[0, 1].hist(
        quant_B,
        bins=bins,
        range=bin_range,
        color="steelblue",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[0, 1].set_title(f"Original INT4 - {name_B}")
    axes[0, 1].set_xlabel("Quantized Value")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0, 1].axvline(0, color="black", linestyle="--", linewidth=1)

    # ---------- After Diff ----------
    axes[1, 0].hist(
        delta_A,
        bins=bins,
        range=bin_range,
        color="orangered",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1, 0].set_title(f"After Diff - {name_A}")
    axes[1, 0].set_xlabel("Delta Value")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(0, color="black", linestyle="--", linewidth=1)

    axes[1, 1].hist(
        delta_B,
        bins=bins,
        range=bin_range,
        color="orangered",
        edgecolor="black",
        linewidth=0.3,
    )
    axes[1, 1].set_title(f"After Diff - {name_B}")
    axes[1, 1].set_xlabel("Delta Value")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    time_stamp = datetime.now().strftime("%m%d_%H%M%S")
    safe_name = f"{name_A.replace('/', '_')}_vs_{name_B.replace('/', '_')}"
    file_name = f"{time_stamp}_{safe_name}.png"
    save_path = Path(save_dir) / file_name
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Two-layer split histogram saved → {save_path}")


def layer_dist_csv(layer_path, index, tile=128, csv_file="layer_dist.csv"):
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    q = real_quantize_tensor(weight, zero_point=True, group_size=tile).view(-1)  # INT4

    # 每 bucket 占比
    cnt = torch.bincount(q, minlength=16)  # [16]
    ratio = cnt.float() / q.numel()  # [16] 占比

    # 指标
    zp = q.view(-1, tile).min(dim=1)[0]  # 当前 zp(min)
    med = q.view(-1, tile).median(dim=1)[0]  # 中位数
    mode = q.view(-1, tile).mode(dim=1)[0]  # 众数
    cov2 = (q.abs() <= 1).float().mean().item()  # |w|≤1
    cov3 = (q.abs() <= 3).float().mean().item()  # |w|≤3
    zero = (q == 0).float().mean().item()  # zero-ratio
    long4 = stat_diff_without_first(q, tile)[3]  # Long4≥3

    # 写csv
    row = (
        [name]
        + [f"{v:.2f}" for v in ratio.cpu().tolist()]
        + [
            f"{zp.median().item():.2f}",
            f"{med.median().item():.2f}",
            f"{mode.median().item():.2f}",
            f"{zero:.2f}",
            f"{cov2:.2f}",
            f"{cov3:.2f}",
            f"{long4:.2f}",
        ]
    )
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def plot_weights(layer_path=None, start_index=0, end_index=1, dir_path="plt_figures"):
    """画出指定层的对比分布直方图"""
    for index in range(start_index, end_index):
        results = load_layer_diff_weights(layer_path, index)
        results_symm = load_layer_symm(layer_path, index)

        plot_diff_histogram_split(results, bins=33, save_dir="diff_hist")
        plot_diff_histogram_split(results_symm, bins=33, save_dir="symm_hist")

        plot_two_diff_histograms(results, results_symm, bins=33, save_dir=dir_path)


def generate_layer_dist(layer_path=None, file_path="layer_dist.csv"):
    """统计所有层的分布, 保存到 CSV"""
    header = (
        ["layer"]
        + [f"bucket_{i}" for i in range(16)]
        + [
            "zp_median",
            "med_median",
            "mode_median",
            "zero_ratio",
            "cov2",
            "cov3",
            "long4",
        ]
    )

    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    for index in range(0, 10):
        layer_dist_csv(layer_path, index, tile=128, csv_file="layer_dist.csv")


def load_csv():
    """加载 CSV, 筛选适合不同编码的层"""
    df = pd.read_csv("layer_dist.csv")

    print(df[["layer", "zero_ratio", "cov2", "med_median"]].head())

    good_zp = df[df["zero_ratio"] > 0.35]
    print("适合 zp-RLE 的层:", good_zp["layer"].tolist())

    # 筛选"适合 Chunk-VLC"的层
    good_vlc = df[df["cov2"] > 0.5]
    print("适合 Chunk-VLC 的层:", good_vlc["layer"].tolist())


if __name__ == "__main__":
    layer_path = "output_weights/facebook_opt-125m_layers/"
    # layer_path = "output_weights/EleutherAI_gpt-neo-2.7B_layers/"

    for index in range(1, 5):
        results = load_layer_diff_weights(layer_path, index)
        results_symm = load_layer_symm(layer_path, index)

        plot_diff_histogram_split(results, bins=33, save_dir="diff_hist")
        plot_diff_histogram_split(results_symm, bins=33, save_dir="symm_hist")

        plot_two_diff_histograms(results, results_symm, bins=33, save_dir="plt_figures")
