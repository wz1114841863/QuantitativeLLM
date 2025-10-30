import os
import time
import torch
import json
import pandas as pd
import numpy as np

from typing import Optional
from pathlib import Path
from tqdm import tqdm

from srcs.quantizer.real_quantize import (
    real_quantize_tensor,
)

from srcs.analysis.real_quantize_analy import (
    check_quant_fn,
    save_codes_ratio_figure,
    save_weight_dist_figure,
    save_group_code_dist,
    save_delta_dist,
    save_group_delta_dist,
)
from srcs.utils.save_layer_werights import (
    build_index_map,
    load_selected_layer,
    load_saved_layer,
)

"""
文件说明:
    对量化后的模型权重进行分析, 同时验证量化的正确性
"""


def analyze_one_layer(
    layer_path,
    layer_idx,
    layer_name,
    w_bit=4,
    zero_point=True,
    group_size=128,
    N=4,
    out_root=Path("quant_report"),
    device="cpu",
):
    """分析单层并写图 + JSON"""
    weight, _, info = load_selected_layer(
        layer_path, layer_name=layer_name, return_tensor=True
    )
    weight = weight.to(device)

    # group_size 校验
    total_elem = weight.numel()
    if total_elem % group_size != 0:
        raise ValueError(
            f"Layer {layer_name} of shape {weight.shape} with {total_elem} elements "
            f"cannot be evenly divided into groups of size {group_size}."
        )

    q, zp, scale = real_quantize_tensor(
        weight, zero_point=zero_point, group_size=group_size, return_scale=True
    )
    # print("q unique:", np.unique(q))
    # print("zp unique:", np.unique(zp))

    rmse, clip, codes = check_quant_fn(
        lambda t, **kw: real_quantize_tensor(t, **kw),
        weight,
        group_size=group_size,
        zero_point=True,
    )

    tag = f"{info['layer_name'].replace('/', '_').replace('.', '_')}"
    out_dir = out_root / f"layer_{layer_idx:03d}_{tag}_GS{group_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save_codes_ratio_figure(q.cpu().numpy(), out_dir / "code_ratio.png")
    save_weight_dist_figure(q.cpu().numpy(), out_dir / "weight_dist.png")

    num_groups = total_elem // group_size
    q_group = q.view(num_groups, group_size)
    zp_1d = zp.view(-1)
    delta_group = q_group.int() - zp_1d.view(-1, 1).int()
    # delta_group = q_group - zp_1d.view(-1, 1)
    delta_np = delta_group.cpu().numpy().flatten()
    save_delta_dist(delta_np, out_dir / "delta_weights_dist.png")

    seed = 42
    rng = np.random.default_rng(seed)
    sample_gid = rng.choice(num_groups, size=min(N, num_groups), replace=False)
    for gid in sample_gid:
        codes_g = q_group[gid].cpu().numpy()
        save_group_code_dist(codes_g, gid, out_dir / "group_codes")

        delta_g = delta_group[gid].cpu().numpy()
        save_group_delta_dist(
            delta_g, int(zp_1d[gid].item()), gid, out_dir / "delta_group_codes"
        )

    summary = dict(
        layer_name=info["layer_name"],
        shape=list(weight.shape),
        w_bit=w_bit,
        zero_point=zero_point,
        group_size=group_size,
        RMSE=float(rmse),
        clip_rate=float(clip),
        codes=int(codes),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # print(f"[json] {out_dir}/summary.json")
    return summary


def analyze_model(
    model_layers_dir,
    layer_indices=None,
    w_bit=4,
    zero_point=True,
    group_size=128,
    out_root="quant_report",
    device="cpu",
):
    idx2name = build_index_map(model_layers_dir)
    layer_indices = layer_indices or range(len(idx2name))
    dir_name = Path(model_layers_dir).name
    model_name = dir_name.replace("_layers", "")
    out_root = Path(out_root) / model_name.replace("/", "_")
    all_summaries = []
    for idx in tqdm(layer_indices, desc="Layer"):
        summ = analyze_one_layer(
            model_layers_dir,
            idx,
            idx2name[idx],
            w_bit=w_bit,
            zero_point=zero_point,
            group_size=group_size,
            out_root=out_root,
            device=device,
        )
        all_summaries.append(summ)

    # csv_file = (
    #     out_root / f"all_L{min(layer_indices)}-{max(layer_indices)}_GS{group_size}.csv"
    # )
    # pd.DataFrame(all_summaries).to_csv(csv_file, index=False)
    # print(f"[csv] {csv_file}")


if __name__ == "__main__":
    GROUP_SIZES = [
        128,
        256,
        512,
        1024,
    ]
    # model_path = "extract_weights/facebook_opt-125m_layers"
    model_path = "extract_weights/facebook_opt-1.3b_layers"
    analyze_model(
        model_layers_dir=model_path,
        layer_indices=range(10, 30),
        w_bit=4,
        zero_point=True,
        group_size=512,
    )
