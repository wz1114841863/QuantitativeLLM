import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_basic_stats(weight):
    """Compute basic statistics of a layer."""
    print("Min:", np.min(weight))
    print("Max:", np.max(weight))
    print("Mean:", np.mean(weight))
    print("Std:", np.std(weight))
    print("Median:", np.median(weight))
    print("1st/99th percentile:", np.percentile(weight, [1, 99]))


def plot_weight_distribution(
    weight, save_path="weight_distribution.png", subsample=500_000
):
    """weight: np.ndarray, 任意 shape"""
    weight = weight.flatten()
    if weight.size > subsample:
        rng = np.random.default_rng(42)
        weight = rng.choice(weight, size=subsample, replace=False)
    plt.figure(figsize=(8, 5))
    sns.histplot(weight, bins=200, kde=True, color="skyblue", stat="density")
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")


def plot_weight_heatmap(weight, path="weight_heatmap.png", max_size=400):
    """
    weight: 2-D np.ndarray (任何 shape)
    max_size: 最大边长,默认 400 px
    """
    h, w = weight.shape
    # 统一隔行/列采样
    if h > max_size:
        row_idx = np.linspace(0, h - 1, max_size, dtype=int)
    else:
        row_idx = slice(None)
    if w > max_size:
        col_idx = np.linspace(0, w - 1, max_size, dtype=int)
    else:
        col_idx = slice(None)

    sub = weight[row_idx, :][:, col_idx]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        sub,
        cmap="coolwarm",
        center=0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"shrink": 0.6},
    )
    plt.title(f"Weight Heatmap (sub-sampled {sub.shape[0]}×{sub.shape[1]})")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {path}  (original {h}×{w})")


def mark_outliers(weight, path="outlier_dots.png", thresh=3.0):
    """
    只画 |w| > mean+thresh * std 的点, 其余留空
    """
    mean, std = weight.mean(), weight.std()
    mask = np.abs(weight) > mean + thresh * std
    y, x = np.where(mask)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, c="red", alpha=0.5)
    plt.xlim(0, weight.shape[1])
    plt.ylim(0, weight.shape[0])
    plt.gca().invert_yaxis()
    plt.title("Outlier Weight Positions")
    plt.xlabel("Input dim")
    plt.ylabel("Output dim")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
