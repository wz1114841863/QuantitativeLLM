from srcs.utils.save_layer_werights import save_all_linear_layers, load_saved_layer
from srcs.analysis.basic_analy import (
    get_basic_stats,
    plot_weight_distribution,
    plot_weight_heatmap,
    mark_outliers,
)

"""
文件说明:
    对原始模型的权重分布进行分析和绘制
TODO:
    通过幅度决定量化尺度
    行/列的稀疏模式决定剪枝策略
    奇异值谱分析该层有多少有效秩
    通道间相关性
"""


if __name__ == "__main__":
    model_path = "output_weights/facebook_opt-1.3b_layers"
    for i in range(0, 3):
        weight, bias, info = load_saved_layer(model_path, layer_index=i)
        weight = weight.detach().cpu().numpy()
        get_basic_stats(weight)
        # plot_weight_distribution(weight, path=f"tmp/weight_distribution_layer_{i}.png")
        # plot_weight_heatmap(weight, path=f"tmp/opt_1_3b_weight_heatmap_layer_{i}.png")
        mark_outliers(weight, path=f"tmp/opt_1_3b_outliers_layer_{i}.png")
