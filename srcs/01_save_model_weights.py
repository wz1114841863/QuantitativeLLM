import torch
import glob
import json
from transformers import AutoModelForCausalLM, AutoConfig
from srcs.utils.save_layer_werights import (
    save_all_linear_layers,
    save_selected_linears,
    load_layer_by_index,
)


"""
文件说明:
    由于显存有限, 为便于分析, 保存指定模型的线性层权重到对应目录.
    小模型保存所有层, 大模型可选择性保存部分层.
"""


def save_model_weights(model_name, out_dir, block_filter=None, layer_filter=None):
    """
    保存模型线性层权重.
    如果给出 block_filter 或 layer_filter,则只保存满足条件的层;
    否则保存全部线性层.
    """
    print(f"Loading model {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        config=config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    if block_filter is not None or layer_filter is not None:
        print("进入选择性保存模式...")
        layers_dir, layers_info = save_selected_linears(
            model=model,
            model_name=model_name,
            save_dir=out_dir,
            block_filter=block_filter,
            layer_filter=layer_filter,
            return_info=True,
        )
        # 选择性保存时不生成整体模型信息文件
        model_info_path = None
    else:
        print("进入全量保存模式...")
        model_info_path, layers_dir = save_all_linear_layers(model, model_name, out_dir)

    print(f"权重已保存到目录: {layers_dir}")
    if model_info_path:
        print(f"模型信息文件: {model_info_path}")
    return model_info_path, layers_dir


def print_saved_layers_info(model_path, start=0, end=5):
    """
    打印使用save_selected_linears保存层的信息
    """
    info_files = glob.glob(f"{model_path}/*_info.json")
    for f in info_files[start:end]:
        print(json.load(open(f))["layer_name"])


if __name__ == "__main__":
    ALL_LAYERS_MODEL = [
        "facebook/opt-125m",
        "facebook/opt-1.3b",
    ]

    SELECT_LAYERS_MODEL = [
        "EleutherAI/gpt-neo-2.7B",
    ]

    # model_name = "facebook/opt-125m"
    # model_name = "EleutherAI/gpt-neo-2.7B"
    model_name = "facebook/opt-1.3b"

    out_dir = "./extract_weights"
    # 保存模型权重
    save_model_weights(model_name, out_dir)

    # 测试加载
    if 0:
        # model_path = "extract_weights/EleutherAI_gpt-neo-2.7B_layers"
        model_path = "extract_weights/facebook_opt-1.3b_layers"
        for i in range(10, 11):
            weight, bias, info = load_layer_by_index(model_path, i)
            print(
                info["layer_name"],
                weight.shape,
                bias.shape if bias is not None else None,
            )
