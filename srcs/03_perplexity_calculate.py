import torch
import json
import time

from pathlib import Path
from transformers import OPTConfig, OPTModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from srcs.quantizer.pseudo_quantize import pseudo_quantize_model_weight
from utils.perplexity import calc_perplexity_wikitext
from utils.logger import dual_log

"""
    文件说明:
        测试模型进行权重伪量化前后的困惑度变化
            包括最基础的权重伪量化,  与现有AWQ等算法结合的伪量化

        对于熵编码前后的困惑度变化, 由于熵编码并不改变权重值, 困惑度不会变化
            直接比对权重 和 熵编码/解码后的权重的是否相等即可
"""


def calc_pseudo_perplexity(model_name):
    """Test the perplexity of an OPT model."""
    print(f"Loading model {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )

    print("testing original model perplexity...")
    # orig_perplexity = calc_perplexity_wikitext(model, tokenizer)
    orig_perplexity = 14.6238
    print(f"Original Perplexity: {orig_perplexity:.4f}")

    print("quantizing model weights to 4-bit...")
    pseudo_quantize_model = pseudo_quantize_model_weight(
        model, w_bit=4, zero_point=True, group_size=128
    )

    print("testing quantized model perplexity...")
    quantized_perplexity = calc_perplexity_wikitext(pseudo_quantize_model, tokenizer)
    print(f"Pseudo-Quantized Perplexity: {quantized_perplexity:.4f}")


def test_pseudo_perplexity(
    model_name,
    w_bit=4,
    group_size=128,
    log_root=Path("perplexity_logs"),
):
    """返回 dict 并写日志/JSON"""
    log_root.mkdir(exist_ok=True)
    ts = time.strftime("%m%d-%H%M%S")
    safe_name = model_name.replace("/", "_")
    log_file = log_root / f"{safe_name}_pseudo_{w_bit}bit_gs{group_size}_{ts}.log"
    json_file = log_root / f"{safe_name}_pseudo_{w_bit}bit_gs{group_size}_{ts}.json"

    with dual_log(log_file):
        print(f"Loading model {model_name} ...")
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto"
        )

        print("Testing original perplexity ...")
        orig_ppl = calc_perplexity_wikitext(model, tokenizer)
        print(f"Original PPL = {orig_ppl:.4f}")

        print(f"Quantizing weights → {w_bit}-bit zero-point group_size={group_size}")
        q_model = pseudo_quantize_model_weight(
            model, w_bit=w_bit, zero_point=True, group_size=group_size
        )

        print("Testing quantized perplexity ...")
        q_ppl = calc_perplexity_wikitext(q_model, tokenizer)
        print(f"Quantized PPL = {q_ppl:.4f}")
        print(f"ΔPPL = {q_ppl - orig_ppl:.4f}")

    summary = {
        "model_name": model_name,
        "quant_method": "pseudo_zero_point",
        "w_bit": w_bit,
        "group_size": group_size,
        "original_ppl": round(float(orig_ppl), 4),
        "quantized_ppl": round(float(q_ppl), 4),
        "delta_ppl": round(float(q_ppl - orig_ppl), 4),
        "log_file": str(log_file),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_file.write_text(json.dumps(summary, indent=2))
    print(f"JSON summary → {json_file}")
    return summary


if __name__ == "__main__":
    # model_name = "facebook/opt-125m"
    model_name = "facebook/opt-1.3b"
    test_pseudo_perplexity(model_name)
