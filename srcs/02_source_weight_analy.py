import json
import pandas as pd
import numpy as np
from pathlib import Path
from srcs.utils.save_layer_werights import build_index_map, load_selected_layer
from srcs.analysis.basic_analy import (
    get_basic_stats,
    plot_weight_distribution,
)

"""
文件说明:
    对原始模型的权重分布进行分析和绘制.
    观察权重的分布
"""


def analyze_one_layer(
    weight,
    model_name,
    layer_idx,
    layer_name,
    out_root=Path("plt_figures/origin_weight"),
):
    safe_name = layer_name.replace(".", "_").replace("/", "_")
    out_dir = out_root / model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = get_basic_stats(weight)
    stats.update(layer_idx=layer_idx, layer_name=layer_name)

    png_path = out_dir / f"{layer_idx:03d}_{safe_name}.png"
    plot_weight_distribution(weight, png_path)

    json_path = out_dir / f"{layer_idx:03d}_{safe_name}_stats.json"
    json_path.write_text(json.dumps(stats, indent=2))
    return stats


def analyze_model(
    layers_dir: str,
    model_name: str = None,
    idx_list=None,
    out_root: Path = Path("plt_figures/origin_weight"),
):
    """
    layers_dir : 模型保存的 xxx_layers 目录
    model_name : 如果 None 则自动从目录名推断
    idx_list   : 想分析的层索引,默认全部分析
    """
    layers_dir = Path(layers_dir)
    model_name = model_name or layers_dir.name.replace("_layers", "")
    idx_list = idx_list or list(range(len(build_index_map(layers_dir))))

    all_stats = []
    for idx in idx_list:
        layer_name = build_index_map(layers_dir)[idx]
        weight, _, _ = load_selected_layer(
            str(layers_dir), layer_name=layer_name, return_tensor=False
        )
        st = analyze_one_layer(weight, model_name, idx, layer_name, out_root)
        all_stats.append(st)

    csv_path = out_root / model_name.replace("/", "_") / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_stats).to_csv(csv_path, index=False)
    print(f"[csv] {csv_path}")


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
    # model_path = "extract_weights/facebook_opt-1.3b_layers"
    # for i in range(10, 20):
    #     plot_dist_by_index(model_path, i)

    layer_path = "extract_weights/facebook_opt-1.3b_layers"
    analyze_model(
        layers_dir=layer_path,
        idx_list=[0, 4, 8, 12, 16, 20, 24],
    )
