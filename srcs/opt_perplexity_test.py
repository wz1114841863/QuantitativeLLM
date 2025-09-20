import torch

from transformers import OPTConfig, OPTModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from quantizer.pseudo_quantize_model_weight import (
    pseudo_symm_quantize_to_4bit,
)
from utils.perplexity import calc_perplexity_wikitext


def test_opt_perplexity(model_name, out_dir="./output/"):
    """Test the perplexity of an OPT model."""
    print(f"Loading model {model_name}...")
    # 目前还是针对OPT模型, 后续需要验证其他模型
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("testing original model perplexity...")
    orig_perplexity = calc_perplexity_wikitext(model, tokenizer)

    print("quantizing model weights to 4-bit...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            pseudo_quantized_weight, scale = pseudo_symm_quantize_to_4bit(weight)
            module.weight.data = pseudo_quantized_weight

    print("testing quantized model perplexity...")
    quantized_perplexity = calc_perplexity_wikitext(model, tokenizer)


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    out_dir = "./output/"
    test_opt_perplexity(model_name, out_dir)
