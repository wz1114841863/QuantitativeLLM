import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def save_group_rmse_fig(
    weight_fp: np.ndarray,
    weight_q: np.ndarray,
    group_size: int = 128,
    max_groups: int = 20_000,
    save_path: str = "group_rmse.png",
):
    """只算 RMSE 并保存直方图"""
    K, C = weight_fp.shape
    NUM_GROUP = C // group_size
    fp_grp = weight_fp.reshape(K, NUM_GROUP, group_size)
    q_grp = weight_q.reshape(K, NUM_GROUP, group_size)
    # 哪些组误差爆炸
    # 误差分布是否长尾
    noise = q_grp - fp_grp

    # 子采样
    all_idx = [(k, g) for k in range(K) for g in range(NUM_GROUP)]
    if len(all_idx) > max_groups:
        rng = np.random.default_rng(42)
        all_idx = rng.choice(all_idx, size=max_groups, replace=False)

    rmse_list = [np.sqrt(np.mean(noise[k, g] ** 2)) for k, g in all_idx]
    rmse_arr = np.array(rmse_list)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(rmse_arr, bins=100, kde=True, ax=ax)
    ax.set_xlabel("Group RMSE")
    ax.set_title(f"Group-wise Quantization Error  (group_size={group_size})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved  {save_path}")


def save_channel_drift_fig(
    weight_fp: np.ndarray,
    weight_q: np.ndarray,
    top_k: int = 32,
    save_path: str = "channel_drift.png",
):
    """只画方差漂移条形图"""
    fp_var = np.var(weight_fp, axis=1)
    q_var = np.var(weight_q, axis=1)
    # 哪些输出通道(token 维度)被量化破坏最严重
    drift = np.abs(q_var - fp_var) / (fp_var + 1e-8)

    top_idx = np.argsort(drift)[-top_k:]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(top_k), drift[top_idx])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"ch{idx}" for idx in top_idx])
    ax.set_xlabel("Relative variance drift")
    ax.set_title(f"Top-{top_k} most distorted output channels")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved  {save_path}")
