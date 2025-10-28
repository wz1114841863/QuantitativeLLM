import torch

from transformers import OPTConfig, OPTModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from srcs.quantizer.pseudo_quantize import pseudo_quantize_model_weight
from utils.perplexity import calc_perplexity_wikitext

"""
    文件说明:
        测试模型进行权重伪量化前后的困惑度变化
            包括最基础的权重伪量化,  与现有AWQ等算法结合的伪量化

        对于熵编码前后的困惑度变化, 由于熵编码并不改变权重值, 困惑度不会变化
            直接比对权重 和 熵编码/解码后的权重的是否相等即可
"""


def test_pseudo_perplexity(model_name):
    """Test the perplexity of an OPT model."""
    print(f"Loading model {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )

    print("testing original model perplexity...")
    orig_perplexity = calc_perplexity_wikitext(model, tokenizer)

    print("quantizing model weights to 4-bit...")
    pseudo_quantize_model = pseudo_quantize_model_weight(
        model, w_bit=4, zero_point=True, group_size=128
    )

    print("testing quantized model perplexity...")
    quantized_perplexity = calc_perplexity_wikitext(pseudo_quantize_model, tokenizer)


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    test_pseudo_perplexity(model_name)
