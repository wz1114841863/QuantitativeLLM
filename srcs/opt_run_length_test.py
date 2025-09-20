import torch
import os

from collections import defaultdict
from collections import Counter
from transformers import OPTModel, OPTConfig

from srcs.quantizer.pseudo_quantize_model_weight import (
    pseudo_quantize_to_4bit,
    pseudo_group_quantize_to_4bit,
)
from srcs.utils.run_lengths_calculate import compute_run_lengths
from srcs.utils.utils import (
    save_quantized_weigths,
    save_log,
    save_json_file,
)


def analyze_model(model_name, out_dir, skip_if_exist=False):
    """Analyze a model: quantize weights, compute run-length stats, and save results."""

    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    weight_path = os.path.join(model_dir, "quantized_weights.bin")
    log_path = os.path.join(model_dir, "quantization.log")
    layer_rl_path = os.path.join(model_dir, "layer_run_length_stats.json")
    global_rl_path = os.path.join(model_dir, "global_run_length_stats.json")
    # stats_path = os.path.join(model_dir, "run_stats")
    # plot_dir = os.path.join(model_dir, "run_length_distribution")
    # os.makedirs(plot_dir, exist_ok=True)

    if skip_if_exist and os.path.exists(weight_path) and os.path.exists(log_path):
        print(f"Skipping {model_name}, already processed.")
        return

    log_lines = [f"Model: {model_name}"]
    print(f"Loading model {model_name}...")
    # 目前还是针对OPT模型, 后续需要验证其他模型
    config = OPTConfig.from_pretrained(model_name)
    model = OPTModel.from_pretrained(model_name, config=config)

    quantized_dict = {}
    # all_runs = []
    global_rl_counter = Counter()  # 整个模型游程长度汇总
    layer_rl_dict = {}  # 逐层游程长度汇总
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data.flatten()
            quantized, scale, zero_point = pseudo_group_quantize_to_4bit(weight)
            quantized_dict[name] = quantized.cpu()
            runs, len_counter = compute_run_lengths(quantized)
            global_rl_counter.update(len_counter)
            layer_rl_dict[name] = dict(len_counter)
            # all_runs.extend(runs)
            lines = (
                f"Layer: {name} | Original: {weight.numel():>8} | Runs: {len(runs):>6} | "
                f"RunLenHist: {dict(len_counter)}"
            )
            print(lines)
            log_lines.append(lines)

    # save_quantized_weigths(quantized_dict, weight_path)
    save_log(log_lines, log_path)

    # save_json_file(global_rl_counter, global_rl_path)
    # save_json_file(layer_rl_dict, layer_rl_path)

    print(
        f"Analysis for {model_name} completed. "
        f"Quantized weights & run-length stats saved in {model_dir}."
    )
    # 统计每个数值的run长度分布
    # run_dist = defaultdict(list)
    # for val, length in all_runs:
    #     run_dist[val].append(length)
    # save_stats(run_dist, stats_path)


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    out_dir = "./output/"
    analyze_model(model_name, out_dir)
