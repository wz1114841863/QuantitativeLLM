import torch
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

import torch, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from typing import List


def save_delta_dist(delta: np.ndarray, save_path: Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.histplot(
        delta,
        bins=np.arange(delta.min() - 0.5, delta.max() + 1.5, 1),
        stat="density",
        color="firebrick",
        alpha=0.8,
    )
    plt.title("Δ = code - zero_point  Distribution")
    plt.xlabel("Δ")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    # print(f"[delta fig] {save_path}")


def save_group_delta_dist(
    delta_g: np.ndarray, zp_val: int, group_id: int, save_dir: Path
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.histplot(
        delta_g,
        bins=np.arange(delta_g.min() - 0.5, delta_g.max() + 1.5, 1),
        stat="count",
        color="orange",
        alpha=0.8,
    )
    plt.title(f"Group {group_id:04d}  Δ (zp={zp_val})")
    plt.xlabel("Δ")
    plt.ylabel("Count")
    plt.tight_layout()
    fig_path = save_dir / f"group_{group_id:04d}_delta.png"
    plt.savefig(fig_path, dpi=250)
    plt.close()
    # print(f"[delta group fig] {fig_path}")


def save_codes_ratio_figure(codes: np.ndarray, save_path: Path):
    """绘制 16 个码字占比条形图"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    uniq, cnt = np.unique(codes, return_counts=True)
    ratio = cnt / cnt.sum()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=uniq.astype(int), y=ratio, color="steelblue")
    plt.title(f"Code Ratio  (total={len(codes)})")
    plt.xlabel("Quantized Code")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[fig] {save_path}")


def save_weight_dist_figure(weight: np.ndarray, save_path: Path, subsample=300_000):
    """量化后权重分布直方图"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    w = weight.flatten()
    if w.size > subsample:
        rng = np.random.default_rng(42)
        w = rng.choice(w, size=subsample, replace=False)
    plt.figure(figsize=(6, 4))
    sns.histplot(w, bins=np.arange(-0.5, 16, 1), stat="density", color="skyblue")
    plt.title("Quantized Weight Distribution")
    plt.xlabel("Code (0~15)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    # print(f"[fig] {save_path}")s


def save_group_code_dist(codes: np.ndarray, group_id: int, save_dir: Path):
    """
    只画一个组的 codes 直方图
    codes : 一维 uint8 数组(0~15)
    group_id : 组序号,仅用于文件名
    save_dir : 文件夹路径(已存在)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.histplot(
        codes,
        bins=np.arange(-0.5, 16, 1),  # 16 个 bin 对齐 0~15
        stat="count",
        color="steelblue",
        alpha=0.8,
    )
    plt.title(f"Group {group_id:04d}  Code Distribution")
    plt.xlabel("Quantized Code")
    plt.ylabel("Count")
    plt.tight_layout()

    fig_path = save_dir / f"group_{group_id:04d}_codes.png"
    plt.savefig(fig_path, dpi=250)
    plt.close()
    # print(f"[fig] {fig_path}")


@torch.no_grad()
def check_quant_fn(fn, tensor, *, group_size: int, zero_point: bool = False, **kw):
    """
    tensor: FP16 权重 [K, C]
    group_size: 任意正整数(可变)
    zero_point: 是否 zero-point
    返回 -> (RMSE, clip_rate, codes)
    """
    device = tensor.device
    K, C = tensor.shape
    numel = K * C
    num_groups = (numel + group_size - 1) // group_size  # 真实组数

    # 调用真实量化
    out = fn(tensor, zero_point=zero_point, group_size=group_size, return_scale=True)

    if zero_point:  # -> q, zp, scale
        q, zp, scale = out
        q_flat = q.view(-1)  # [numel]
        zp_flat = zp.view(-1)  # [num_groups]
        sc_flat = scale.view(-1)  # [num_groups]
        gids = torch.arange(numel, device=device) // group_size
        deq = (q_flat.float() - zp_flat[gids]) * sc_flat[gids]
    else:  # -> q, scale
        q, scale = out
        q_flat = q.view(-1)
        sc_flat = scale.view(-1)
        gids = torch.arange(numel, device=device) // group_size
        deq = q_flat.float() * sc_flat[gids]

    deq = deq.view(K, C)

    rmse = torch.sqrt(torch.mean((deq - tensor) ** 2)).item()
    clip = ((q == q.min()) | (q == q.max())).float().mean().item()
    codes = q.unique().numel()
    return rmse, clip, codes
