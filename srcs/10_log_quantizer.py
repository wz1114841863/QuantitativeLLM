import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from srcs.quantizer.log_quantizer import LogQuantizer
from quantizer.pre_quant import get_blocks, get_named_linears
from utils.perplexity import calc_perplexity_wikitext

"""
文件说明:
    使用对数量化方法对模型进行量化, 并计算困惑度
    如果困惑度在可接收范围内, 统计其量化后的权重分布
"""


@torch.no_grad()
def pseudo_log_quantize_model_weight(
    model,
    quantizer,
    w_bit=4,
    group_size=128,
):
    """Pseudo log quantization to 4 bits."""
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight log quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            # m.weight.data 是一个 2D 张量, 形状为 [out_features, in_features]
            # 对每一行 (out_feature) 单独进行分组量化
            for j in range(m.weight.data.size(0)):
                row_data = m.weight.data[j]  # [in_features]
                for k in range(0, row_data.size(0), group_size):

                    start_idx = k
                    end_idx = min(start_idx + group_size, row_data.size(0))

                    weight_group_torch = row_data[start_idx:end_idx]
                    weight_group_np = (
                        weight_group_torch.cpu().numpy().astype(np.float16)
                    )

                    index_stream, best_cb_id, mse = quantizer.quantize_group(
                        weight_group_np
                    )

                    dequantized_weights_np = quantizer.dequantize_group(
                        index_stream, best_cb_id
                    )

                    dequantized_weights_torch = torch.from_numpy(
                        dequantized_weights_np
                    ).to(weight_group_torch.device, dtype=weight_group_torch.dtype)

                    row_data[start_idx:end_idx] = dequantized_weights_torch
    return model


def calc_pseudo_perplexity(model_name, quantizer, group_size):
    """Test the perplexity of an OPT model."""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )

    print("testing original model perplexity...")
    orig_perplexity = calc_perplexity_wikitext(model, tokenizer)
    # orig_perplexity = 27.655853271484375
    # print(f"Original Perplexity: {orig_perplexity:.4f}")

    print("quantizing model weights to 4-bit...")
    pseudo_quantize_model = pseudo_log_quantize_model_weight(
        model, quantizer, w_bit=4, group_size=group_size
    )

    print("testing quantized model perplexity...")
    quantized_perplexity = calc_perplexity_wikitext(pseudo_quantize_model, tokenizer)


def main():
    # model_name = "facebook/opt-125m"
    model_name = "facebook/opt-1.3b"
    group_size = 512
    quantizer = LogQuantizer(num_codebooks=4)
    calc_pseudo_perplexity(model_name, quantizer, group_size)


if __name__ == "__main__":
    main()
