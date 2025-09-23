import os
import json
import numpy as np
import torch
from collections import Counter

from quantizer.pre_quant import get_named_linears


def save_all_linear_layers(model, model_name, save_dir):
    """将模型中的所有线性层分别保存为单独的文件"""
    os.makedirs(save_dir, exist_ok=True)
    modules = get_named_linears(model)

    print(f"开始保存 {model_name} 的线性层权重...")
    print(f"共发现 {len(modules)} 个线性层")

    # 保存总体信息
    model_info = {
        "model_name": model_name,
        "total_layers": len(modules),
        "layer_names": list(modules.keys()),
        "save_timestamp": str(np.datetime64("now")),
    }

    layers_dir = os.path.join(save_dir, f"{model_name.replace('/', '_')}_layers")
    os.makedirs(layers_dir, exist_ok=True)

    layers_info = {}
    for i, (name, module) in enumerate(modules.items()):
        safe_name = name.replace(".", "_").replace("/", "_")

        weight = module.weight.data.cpu().numpy()
        weight_path = os.path.join(layers_dir, f"layer_{i:03d}_{safe_name}_weights.npy")
        np.save(weight_path, weight)

        # 保存偏置(如果存在)
        bias_path = None
        if module.bias is not None:
            bias = module.bias.data.cpu().numpy()
            bias_path = os.path.join(layers_dir, f"layer_{i:03d}_{safe_name}_bias.npy")
            np.save(bias_path, bias)

        # 保存层信息
        layer_info = {
            "layer_index": i,
            "layer_name": name,
            "safe_name": safe_name,
            "weight_shape": list(module.weight.shape),
            "weight_dtype": str(module.weight.dtype),
            "has_bias": module.bias is not None,
            "weight_path": weight_path,
            "bias_path": bias_path,
            "weight_stats": {
                "min": float(weight.min()),
                "max": float(weight.max()),
                "mean": float(weight.mean()),
                "std": float(weight.std()),
                "numel": int(weight.size),
            },
        }

        layers_info[name] = layer_info

        # 保存单个层的详细信息
        layer_info_path = os.path.join(
            layers_dir, f"layer_{i:03d}_{safe_name}_info.json"
        )
        with open(layer_info_path, "w") as f:
            json.dump(layer_info, f, indent=2)

        print(f"保存层 {i:03d}: {name}")
        print(
            f"  权重形状: {module.weight.shape}, 数值范围: [{weight.min():.6f}, {weight.max():.6f}]"
        )

    # 保存总体信息文件
    model_info["layers_info"] = layers_info
    model_info_path = os.path.join(
        save_dir, f"{model_name.replace('/', '_')}_model_info.json"
    )
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\n所有层已保存到目录: {layers_dir}")
    print(f"模型信息已保存: {model_info_path}")

    return model_info_path, layers_dir


def load_saved_layer(layers_dir, layer_index=None, layer_name=None, return_tensor=True):
    """
    加载保存的层数据

    Args:
        layers_dir: 层数据目录
        layer_index: 层索引(可选)
        layer_name: 层名称(可选)
    """
    if layer_index is not None:
        # 通过索引加载
        info_files = [f for f in os.listdir(layers_dir) if f.endswith("_info.json")]
        target_file = [
            f for f in info_files if f.startswith(f"layer_{layer_index:03d}")
        ]
        if not target_file:
            raise FileNotFoundError(f"未找到索引为 {layer_index} 的层")
        info_path = os.path.join(layers_dir, target_file[0])
    elif layer_name is not None:
        # 通过名称加载
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        info_files = [f for f in os.listdir(layers_dir) if f.endswith("_info.json")]
        target_file = [f for f in info_files if safe_name in f]
        if not target_file:
            raise FileNotFoundError(f"未找到名称为 {layer_name} 的层")
        info_path = os.path.join(layers_dir, target_file[0])
    else:
        raise ValueError("必须指定 layer_index 或 layer_name")

    # 加载层信息
    with open(info_path, "r") as f:
        layer_info = json.load(f)

    # 加载权重
    weight = np.load(layer_info["weight_path"])
    if return_tensor:
        weight = torch.from_numpy(weight)
    else:
        weight = weight

    # 加载偏置(如果存在)
    bias = None
    if layer_info["has_bias"] and os.path.exists(layer_info["bias_path"]):
        bias_np = np.load(layer_info["bias_path"])
        if return_tensor:
            bias = torch.from_numpy(bias_np)
        else:
            bias = bias_np

    return weight, bias, layer_info


def analyze_saved_layers(model_info_path):
    """
    分析保存的所有层
    """
    with open(model_info_path, "r") as f:
        model_info = json.load(f)

    print(f"模型: {model_info['model_name']}")
    print(f"总层数: {model_info['total_layers']}")
    print("\n各层统计信息:")
    print("-" * 80)

    for i, (name, info) in enumerate(model_info["layers_info"].items()):
        stats = info["weight_stats"]
        print(f"{i:03d}. {name}")
        print(
            f"   形状: {info['weight_shape']} | 数值范围: [{stats['min']:.4f}, {stats['max']:.4f}]"
        )
        print(f"   均值: {stats['mean']:.6f} | 标准差: {stats['std']:.6f}")
        print()


if __name__ == "__main__":
    pass
    # weight, bias, info = load_saved_layer("./saved_layers/model_name_layers", layer_index=0)
    # weight, bias, info = load_saved_layer("./saved_layers/model_name_layers", layer_name="model.layers.0.mlp.down_proj")
