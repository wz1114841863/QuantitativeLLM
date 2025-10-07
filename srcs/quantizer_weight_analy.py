import os
import torch

from typing import Optional
from tqdm import tqdm

from srcs.utils.save_layer_werights import load_saved_layer
from srcs.quantizer.pseudo_quantize import (
    pseudo_quantize_tensor,
)
from srcs.quantizer.real_quantize import real_quantize_tensor
from srcs.analysis.basic_analy import (
    get_basic_stats,
    plot_weight_distribution,
    plot_weight_heatmap,
    mark_outliers,
)
from srcs.analysis.pseudo_quantize_analy import (
    save_group_rmse_fig,
    save_channel_drift_fig,
)
from srcs.analysis.real_quantize_analy import (
    check_quant_fn,
)

"""
文件说明:
    对量化后的模型权重进行分析, 同时验证量化的正确性
    对量化+差分的模型权重进行分析, 对比差分带来的分布变化
"""


def pseudo_quantize_analy(
    model_path: str,
    layer_indices: range = range(3),
    w_bit: int = 4,
    zero_point: bool = False,
    group_size: Optional[int] = 128,
    out_dir: str = "tmp",
    model_tag: Optional[str] = None,
    plot_noise: bool = True,
    plot_outlier: bool = True,
    plot_rmse: bool = False,
    plot_drift: bool = False,
):
    """逐层加载保存的权重, 进行伪量化, 并分析量化后的权重分布"""
    os.makedirs(out_dir, exist_ok=True)
    tag = model_tag or os.path.basename(os.path.normpath(model_path))

    for i in tqdm(layer_indices, desc="Analysing layers"):
        weight, bias, info = load_saved_layer(model_path, layer_index=i)

        quant_tensor = pseudo_quantize_tensor(
            weight, wq_bits=w_bit, zero_point=zero_point, group_size=group_size
        )

        weight_np = weight.detach().cpu().numpy()
        quant_np = quant_tensor.detach().cpu().numpy()

        get_basic_stats(quant_np)
        plot_weight_distribution(
            quant_np,
            path=os.path.join(out_dir, f"{tag}_L{i}_W{w_bit}bit_dist.png"),
        )
        plot_weight_heatmap(
            quant_np,
            path=os.path.join(out_dir, f"{tag}_L{i}_W{w_bit}bit_heatmap.png"),
        )

        if plot_noise:
            # 红色 → 量化向上偏移,蓝色 → 向下偏移.
            # 若某一行/列颜色一致 → 该组 scale 过大/过小,后续可减小group_size或单独放大qmax
            plot_weight_heatmap(
                quant_np - weight_np,
                path=os.path.join(out_dir, f"{tag}_L{i}_W{w_bit}bit_noise.png"),
            )
        if plot_outlier:
            # 画出离群点位置
            mark_outliers(
                quant_np,
                path=os.path.join(out_dir, f"{tag}_L{i}_W{w_bit}bit_outliers.png"),
            )
        if plot_rmse:
            # 画出各组 RMSE 分布
            save_group_rmse_fig(
                weight_fp=weight_np,
                weight_q=quant_np,
                group_size=group_size or weight_np.shape[1],
                save_path=os.path.join(out_dir, f"{tag}_L{i}_W{w_bit}bit_grp_rmse.png"),
            )
        if plot_drift:
            # 画出各通道方差漂移
            save_channel_drift_fig(
                weight_fp=weight_np,
                weight_q=quant_np,
                save_path=os.path.join(
                    out_dir, f"{tag}_L{i}_W{w_bit}bit_chnl_drift.png"
                ),
            )


def real_quant_analy(
    model_path: str,
    layer_indices: range = range(3),
    w_bit: int = 4,
    zero_point: bool = False,
    group_size: Optional[int] = None,
    save_dir: str = "tmp",
    tag: str = "layer",
    plot_code_hist: bool = True,
    plot_scale_heat: bool = True,
    plot_clip_scatter: bool = True,
):
    """
    对 *真实量化* 后的权重做全套体检
    weight: FP16 权重矩阵  shape [out_ch, in_ch]
    """
    # os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(layer_indices, desc="Analysing layers"):
        weight, bias, info = load_saved_layer(model_path, layer_index=i)
        # device = weight.device

        out = real_quantize_tensor(
            weight,
            zero_point=zero_point,
            group_size=group_size,
            return_scale=True,
        )

        # 拿到 group-wise 参数
        if zero_point:
            q, zp, scale = out
            # scale/zp 先 reshape 成 [out_ch, num_groups_per_row]
            in_ch = weight.shape[1]
            groups_per_row = in_ch // group_size
            scale_2d = scale.view(-1, groups_per_row).repeat_interleave(
                group_size, dim=1
            )
            zp_2d = zp.view(-1, groups_per_row).repeat_interleave(group_size, dim=1)
            deq = (q.to(weight.dtype) - zp_2d) * scale_2d
        else:
            q, scale = out
            in_ch = weight.shape[1]
            groups_per_row = in_ch // group_size
            scale_2d = scale.view(-1, groups_per_row).repeat_interleave(
                group_size, dim=1
            )
            deq = q.to(weight.dtype) * scale_2d

        rmse, clip_rate, codes = check_quant_fn(
            lambda t, **kw: real_quantize_tensor(t, **kw, return_scale=True),
            weight,
            zero_point=zero_point,
            group_size=group_size,
        )
        print(f"[{tag}]  RMSE={rmse:.6f}  clip={clip_rate*100:.2f}%  codes={codes}")


if __name__ == "__main__":
    model_path = "output_weights/facebook_opt-125m_layers"
    # pseudo_quantize_analy(
    #     model_path=model_path,
    #     layer_indices=range(3),
    #     w_bit=4,
    #     zero_point=False,
    #     group_size=None,
    #     out_dir="tmp/quant_analysis",
    #     model_tag="opt125m",
    #     # plot_noise=True,
    #     # plot_outlier=True,
    #     plot_rmse=True,
    #     plot_drift=True,
    # )
    real_quant_analy(
        model_path=model_path,
        layer_indices=range(3),
        w_bit=4,
        zero_point=True,
        group_size=128,
    )
