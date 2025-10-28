from pathlib import Path
from srcs.utils.save_layer_werights import build_index_map, load_selected_layer
from srcs.analysis.basic_analy import (
    get_basic_stats,
    plot_weight_distribution,
    plot_weight_heatmap,
    mark_outliers,
)

"""
文件说明:
    对原始模型的权重分布进行分析和绘制.
    观察权重的分布
"""


def plot_dist_by_index(layers_dir: str, idx: int, subsample=500_000):
    """
    仅传目录+索引即可绘图
    """
    layer_name = build_index_map(layers_dir)[idx]
    weight, _, info = load_selected_layer(
        layers_dir, layer_name=layer_name, return_tensor=False
    )
    layers_path = Path(layers_dir)
    model_name = layers_path.name.replace("_layers", "")
    safe_layer_name = layer_name.replace(".", "_").replace("/", "_")
    save_path = Path(f"plt_figures/origin_weight/{model_name}/{safe_layer_name}.png")
    plot_weight_distribution(weight, save_path, subsample)


if __name__ == "__main__":
    model_path = "extract_weights/facebook_opt-1.3b_layers"
    for i in range(0, 3):
        plot_dist_by_index(model_path, i)
