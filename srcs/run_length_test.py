import torch
import os
import gc

from collections import defaultdict
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from srcs.quantizer.real_quantize import *
from srcs.utils.run_lengths_calculate import compute_run_lengths
from srcs.utils.utils import (
    save_quantized_weigths,
    save_log,
    save_json_file,
)

"""
    文件说明:
        分析模型的线性层权重的量化和游程统计
        Error: 由于WSL2的显存限制, 目前只能跑小模型
"""


def release_memory():
    """aggressive but safe for both CUDA & CPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 如果用了 mps(Mac)
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()


def analyze_model(model_name, out_dir, skip_if_exist=True):
    """Analyze a model: quantize weights, compute run-length stats, and save results."""

    os.makedirs(out_dir, exist_ok=True)
    base_dir = os.path.join(out_dir, model_name.replace("/", "_"))

    group_size = 128
    strategies = [
        ("real_symm", False, None),
        # ("real_zero_point", True, None),
        # ("group_symm", False, group_size),
        # ("group_zero_point", True, group_size),
    ]
    print(f"Loading model {model_name}...")
    # 目前还是针对OPT模型, 后续需要验证其他模型
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model.eval()

    for subdir, zero_point, gs in strategies:
        print(
            f"Processing strategy: {subdir} | zero_point: {zero_point} | group_size: {gs}"
        )
        run_dir = os.path.join(base_dir, subdir)
        os.makedirs(run_dir, exist_ok=True)

        weight_path = os.path.join(run_dir, "quantized_weights.bin")
        log_path = os.path.join(run_dir, "quantization.log")
        global_rl_path = os.path.join(run_dir, "global_run_length_stats.json")
        layer_rl_path = os.path.join(run_dir, "layer_run_length_stats.json")
        zero_ratio_path = os.path.join(run_dir, "layer_zero_ratio.json")

        if skip_if_exist and os.path.exists(weight_path) and os.path.exists(log_path):
            print(f"Skipping {subdir} for {model_name}, already processed.")
            continue

        log_lines = [f"Model: {model_name} | Strategy: {subdir}"]
        # quantized_dict = {}
        global_rl_counter = Counter()
        layer_rl_dict = {}
        zero_ratio_dict = {}
        for name, module in model.named_modules():
            print(f"Processing layer: {name}")
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data.flatten()
                quantized = real_quantize_tensor(
                    weight, zero_point=zero_point, group_size=gs
                )
                # quantized_dict[name] = quantized.cpu()

                runs, len_counter = compute_run_lengths(quantized)
                global_rl_counter.update(len_counter)
                layer_rl_dict[name] = dict(len_counter)

                # 零值比例
                zero_runs = [l for v, l in runs if v == 0]
                zero_elem = sum(zero_runs)
                zero_ratio = zero_elem / weight.numel()
                zero_ratio_dict[name] = zero_ratio

                line = (
                    f"Layer: {name} | Original: {weight.numel():>8} | "
                    f"Runs: {len(runs):>6} | ZeroRatio: {zero_ratio:.4f}"
                )
                print(line)
                log_lines.append(line)

                # 释放内存
                del weight, quantized, runs, len_counter, zero_runs
                release_memory()

                print(f"Finished layer: {name}\n")
            print(f"2 Finished module: {name}\n")
        print(f"Finished strategy: {subdir} for model: {model_name}")
        # 写入文件
        # save_quantized_weigths(quantized_dict, weight_path)
        # save_json_file(dict(global_rl_counter), global_rl_path)
        # save_json_file(layer_rl_dict, layer_rl_path)
        # save_json_file(zero_ratio_dict, zero_ratio_path)
        # with open(log_path, "w") as f:
        #     f.write("\n".join(log_lines))

        print(
            f"Strategy {subdir} for {model_name} completed. Results saved to {run_dir}"
        )

        del quantized_dict
        del obal_rl_counter, layer_rl_dict, zero_ratio_dict, log_lines
        release_memory()


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    out_dir = "./output/"
    analyze_model(model_name, out_dir)
