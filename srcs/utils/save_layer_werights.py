import os
import json
import numpy as np
import torch
import re

import glob
from collections import Counter, OrderedDict
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
        weight = weight.astype(np.float16, copy=False)
        weight_path = os.path.join(layers_dir, f"layer_{i:03d}_{safe_name}_weights.npy")
        np.save(weight_path, weight)

        # 保存偏置(如果存在)
        bias_path = None
        if module.bias is not None:
            bias = module.bias.data.cpu().numpy()
            bias = bias.astype(np.float16, copy=False)
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
    print(f"加载层 {layer_info['layer_index']}: {layer_info['layer_name']}")
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
    加载保存的整体的json文件, 分析保存的所有层
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


def parse_block_id(name: str) -> int:
    """
    从层名里提取 block 序号.
    支持 'model.layers.12.' / 'transformer.h.12.' / '...block_12...' 等常见写法
    返回 -1 表示未命中任何规则(这类层会被丢弃)
    """
    # huggingface系列: model.layers.12.
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    # OPT/GPT-NeoX: transformer.h.12.
    m = re.search(r"\.h\.(\d+)\.", name)
    if m:
        return int(m.group(1))
    # 其他带 block_数字 的写法
    m = re.search(r"block_?(\d+)", name)
    if m:
        return int(m.group(1))
    return -1


def save_selected_linears(
    model,
    model_name,
    save_dir,
    block_filter=None,
    layer_filter=None,
    return_info=True,
):
    """
    只保存指定 Block 内(或名字含指定字符串)的线性层
    block_filter : 如 [0,5,10]
    layer_filter : 如 [".layer", ".fc"]
    """
    os.makedirs(save_dir, exist_ok=True)
    modules = get_named_linears(model)

    if block_filter is not None:
        block_filter = set(block_filter)
        modules = {
            n: m for n, m in modules.items() if parse_block_id(n) in block_filter
        }
    elif layer_filter is not None:
        modules = {
            n: m for n, m in modules.items() if any(k in n for k in layer_filter)
        }
    if not modules:
        print("没有层满足过滤条件,退出.")
        return None, None

    layers_dir = os.path.join(save_dir, f"{model_name.replace('/', '_')}_layers")
    os.makedirs(layers_dir, exist_ok=True)

    layers_info = {}
    for idx, (name, module) in enumerate(modules.items()):
        safe_name = name.replace(".", "_").replace("/", "_")
        weight_path = os.path.join(layers_dir, f"{safe_name}_weight.npy")
        weight = module.weight.data.cpu().numpy()
        weight = weight.astype(np.float16, copy=False)
        np.save(weight_path, weight)

        info = {
            "layer_name": name,
            "weight_shape": list(module.weight.shape),
            "weight_path": weight_path,
        }

        if module.bias is not None:
            bias_path = weight_path.replace("_weight.npy", "_bias.npy")
            bias = module.bias.data.cpu().numpy()
            bias = bias.astype(np.float16, copy=False)
            np.save(bias_path, bias)
            info["bias_path"] = bias_path
            info["has_bias"] = True
        else:
            info["has_bias"] = False

        layers_info[name] = info

    index_file = os.path.join(layers_dir, "selected_layers.json")
    with open(index_file, "w") as f:
        json.dump(layers_info, f, indent=2)

    print(f"已保存 {len(modules)} 个线性层至 {layers_dir}")
    return layers_dir, layers_info


def load_selected_layer(layers_dir, layer_name=None, return_tensor=True):
    # 1. 优先用新索引
    index_file = os.path.join(layers_dir, "selected_layers.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            table = json.load(f)
        if layer_name not in table:
            raise KeyError(f"层 {layer_name} 不在已保存列表中")
        info = table[layer_name]
    else:
        # 2. 老格式:通过层名反查 info 文件
        #    直接用 layers_info 总表,而不用 glob
        model_info_path = os.path.join(
            layers_dir,
            "..",
            f"{os.path.basename(layers_dir).replace('_layers', '')}_model_info.json",
        )
        if not os.path.exists(model_info_path):
            raise FileNotFoundError("找不到模型总 info 文件")
        with open(model_info_path) as f:
            model_info = json.load(f)
        layers_info = model_info["layers_info"]
        if layer_name not in layers_info:
            raise KeyError(f"层 {layer_name} 不在保存列表中")
        info = layers_info[layer_name]

    w = np.load(info["weight_path"])
    b = None
    if info.get("has_bias", False):
        b = np.load(info["bias_path"])
    if return_tensor:
        w = torch.from_numpy(w)
        if b is not None:
            b = torch.from_numpy(b)
    return w, b, info


def load_all_saved_linears(layers_dir, return_tensor=True):
    """
    把 layers_dir 里所有出现过的线性层全部读出来,返回:
    {
      layer_name_0: (weight, bias, info),
      layer_name_1: (weight, bias, info),
      ...
    }
    """
    index_file = os.path.join(layers_dir, "selected_layers.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            table = json.load(f)
    else:
        info_files = glob.glob(os.path.join(layers_dir, "*_info.json"))
        table = {}
        for path in info_files:
            with open(path) as f:
                one = json.load(f)
                table[one["layer_name"]] = one

    result = OrderedDict()
    for name, info in table.items():
        w = np.load(info["weight_path"])
        b = None
        if info.get("has_bias", False):
            b = np.load(info["bias_path"])
        if return_tensor:
            w = torch.from_numpy(w)
            if b is not None:
                b = torch.from_numpy(b)
        result[name] = (w, b, info)
    return result


def build_index_map(layers_dir):
    new_index = os.path.join(layers_dir, "selected_layers.json")
    if os.path.isfile(new_index):
        table = json.load(open(new_index))
        # 按保存顺序(字典序)返回层名列表
        return list(table.keys())

    # 老格式:遍历 layer_000_xxx_info.json
    info_files = sorted(glob.glob(os.path.join(layers_dir, "*_info.json")))
    if not info_files:
        raise FileNotFoundError(
            "目录下找不到 selected_layers.json 也找不到 *_info.json,无法建立索引"
        )
    info_files = sorted(
        info_files, key=lambda x: int(os.path.basename(x).split("_")[1])
    )
    return [json.load(open(f))["layer_name"] for f in info_files]


def load_layer_by_index(layers_dir, idx, return_tensor=True):
    """
    用于加载使用 save_all_linear_layers 保存的层
    用整数索引加载权重,和原始 load_saved_layer 的 layer_index 同理
    """
    idx2name = build_index_map(layers_dir)
    if idx >= len(idx2name) or idx < 0:
        raise IndexError(f"idx {idx} 超出范围,当前共 {len(idx2name)} 层")
    layer_name = idx2name[idx]

    return load_selected_layer(
        layers_dir, layer_name=layer_name, return_tensor=return_tensor
    )


if __name__ == "__main__":
    weight, bias, info = load_saved_layer(
        "./extract_weights/facebook_opt-125m_layers", layer_index=0
    )
