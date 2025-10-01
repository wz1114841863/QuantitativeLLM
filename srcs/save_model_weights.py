import torch
from transformers import AutoModelForCausalLM, AutoConfig
from srcs.utils.save_layer_werights import save_all_linear_layers, load_saved_layer


"""
    文件说明:
        为便于分析, 保存模型的线性层权重到指定目录
"""


def save_model_weights(model_name, out_dir):
    """Save all linear layer weights of a model to separate files."""
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

    model_info_path, layers_dir = save_all_linear_layers(model, model_name, out_dir)
    print(f"Model weights saved in directory: {layers_dir}")
    print(f"Model info saved at: {model_info_path}")
    return model_info_path, layers_dir


if __name__ == "__main__":
    # model_name = "facebook/opt-125m"
    # model_name = "EleutherAI/gpt-neo-2.7B"
    model_name = "facebook/opt-1.3b"
    out_dir = "./output_weights/"
    # save_model_weights(model_name, out_dir)

    # 测试加载
    model_path = "output_weights/facebook_opt-1.3b_layers"
    for i in range(0, 3):
        weight, bias, info = load_saved_layer(model_path, layer_index=i)
        print(
            f"Layer {i}: weight shape {weight.shape}, bias shape {bias.shape if bias is not None else None}"
        )
