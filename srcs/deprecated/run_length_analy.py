from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

from srcs.utils.save_layer_werights import load_saved_layer
from srcs.encoder.run_lengths_calculate import run_length_analyze

"""
    文件说明:
        对比不同的量化方法下, 差分编码后的游程统计指标
"""


def scan_model_layers(
    layer_path: str,
    start_idx: int = 0,
    end_idx: int | None = None,
    method_filter: str = "group_zero_point",
    tile: int = 256,
):
    """
    顺序加载模型指定区间层,统一用method_filter量化并返回 DataFrame.
    """
    # 解析策略
    cfg_map = {
        "real_symm": ("real_symm", False, None),
        "real_zero_point": ("real_zero_point", True, None),
        "group_symm": ("group_symm", False, 128),
        "group_zero_point": ("group_zero_point", True, 128),
    }
    if method_filter not in cfg_map:
        raise ValueError(f"Unknown method_filter: {method_filter}")
    method, zero_point, gs = cfg_map[method_filter]

    records: List[Dict] = []
    if end_idx is None:
        end_idx = start_idx + 1

    for idx in range(start_idx, end_idx):
        try:
            weight, bias, info = load_saved_layer(layer_path, idx)
        except FileNotFoundError:
            break
        rec = run_length_analyze(weight, method, zero_point, gs, tile=tile)
        rec["layer_idx"] = idx
        rec["layer_name"] = info["layer_name"]
        records.append(rec)

    return pd.DataFrame(records).set_index("layer_idx")


if __name__ == "__main__":
    group_size = 128
    methods = [
        "real_symm",
        "real_zero_point",
        "group_symm",
        "group_zero_point",
    ]
    output_path = "output_weights/facebook_opt-125m_layers"
    df = scan_model_layers(
        output_path,
        start_idx=0,
        end_idx=2,
        method_filter="group_zero_point",
        tile=128,
    )

    print(df[["layer_name", "cov2", "same", "long4"]].describe())
