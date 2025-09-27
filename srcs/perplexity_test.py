import torch

from transformers import OPTConfig, OPTModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from srcs.quantizer.pseudo_quantize import (
    pseudo_quantize_model_weight,
    pseudo_quantize_diff_weight,
)
from utils.perplexity import calc_perplexity_wikitext

"""
    文件说明:
        1. 测试模型的伪量化权重的困惑度
        2. 测试模型的伪差分量化权重的困惑度
"""


def test_pseudo_perplexity(model_name):
    """Test the perplexity of an OPT model."""
    print(f"Loading model {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )

    # print("testing original model perplexity..."): 27.65
    # orig_perplexity = calc_perplexity_wikitext(model, tokenizer)

    print("quantizing model weights to 4-bit...")
    pseudo_quantize_model = pseudo_quantize_model_weight(
        model, w_bit=4, zero_point=True, group_size=128
    )

    print("testing quantized model perplexity...")
    quantized_perplexity = calc_perplexity_wikitext(pseudo_quantize_model, tokenizer)


def test_pseudo_diff_perplexity(model_name):
    """Test the perplexity of an OPT model with pseudo diff quantization."""
    print(f"Loading model {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )

    # print("testing original model perplexity...")
    # orig_perplexity = calc_perplexity_wikitext(model, tokenizer)

    print("quantizing model weights to 4-bit with pseudo diff quantization...")
    pseudo_quantize_model = pseudo_quantize_diff_weight(
        model, w_bit=4, zero_point=True, group_size=128, tile=128, clamp_diff=True
    )

    print("testing quantized model perplexity...")
    quantized_perplexity = calc_perplexity_wikitext(model, tokenizer)


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    out_dir = "./output/"
    # test_pseudo_perplexity(model_name)
    test_pseudo_diff_perplexity(model_name)
