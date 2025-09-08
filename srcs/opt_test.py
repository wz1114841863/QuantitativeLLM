import torch
import os
import json
import pickle
import matplotlib.pyplot as plt

from collections import defaultdict
from transformers import OPTModel, OPTConfig


def quantize_to_4bit(tensor):
    """Zero-point quantization to 4 bits.
    e.g.:
        tmp = torch.randint(0, 256, (1000,))
        q, s, z = quantize_to_4bit(tmp.float())
        print(q, s, z)
    """
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 15
    zero_point = torch.round(-min_val / scale).clamp(0, 15)
    quantized = torch.round(tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
    return quantized, scale.item(), zero_point.item()


def compute_run_lengths(quantized_weights):
    """Compute run-length encoding for a 1D tensor.
    e.g.:
        tmp = torch.randint(0, 256, (1000,))
        runs = compute_run_lengths(tmp)
        print(runs)
    Returns a list of (value, run_length) tuples.
    """
    if quantized_weights.numel() == 0:
        return []
    if len(quantized_weights.shape) != 1:
        quantized_weights = quantized_weights.flatten()
    runs = []
    current_val = quantized_weights[0]
    count = 1
    for val in quantized_weights[1:]:
        if val == current_val:
            count += 1
        else:
            runs.append((int(current_val), count))
            current_val = val
            count = 1
    runs.append((current_val.item(), count))
    return runs


def save_quantized_weigths(quantized_weights, path):
    """Save quantized weights to a binary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(quantized_weights, path)


def load_quantized_weights(path):
    """Load quantized weights from a binary file."""
    return torch.load(path)


def save_log(records, log_path):
    """Save log records to a text file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf8") as f:
        f.write("\n".join(records))


def save_stats(run_dist, path):
    """Save run-length statistics to a text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # json用于方便查看
    with open(path + ".json", "w", encoding="utf8") as f:
        json.dump({k: v for k, v in run_dist.items()}, f, indent=2)
    # pkl用于后续分析
    with open(path + ".pkl", "wb") as f:
        pickle.dump(run_dist, f)


def load_stats(path):
    """Load run-length statistics from a text file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_model(model_name, out_dir, skip_if_exist=True):
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    weight_path = os.path.join(model_dir, "quantized_weights.bin")
    log_path = os.path.join(model_dir, "quantization.log")
    stats_path = os.path.join(model_dir, "run_stats")
    plot_dir = os.path.join(model_dir, "run_length_distribution")
    os.makedirs(plot_dir, exist_ok=True)

    if (
        skip_if_exist
        and os.path.exists(weight_path)
        and os.path.exists(log_path)
        and os.path.exists(stats_path + ".pkl")
    ):
        print(f"Skipping {model_name}, already processed.")
        return

    log_lines = [f"Model: {model_name}"]
    print(f"Loading model {model_name}...")
    # 目前还是针对OPT模型, 后续需要验证其他模型
    config = OPTConfig.from_pretrained(model_name)
    model = OPTModel.from_pretrained(model_name, config=config)

    quantized_dict = {}
    all_runs = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Flatten the weight matrix and store it
            weight = module.weight.data.flatten()
            quantized, scale, zero_point = quantize_to_4bit(weight)
            quantized_dict[name] = quantized.cpu()
            runs = compute_run_lengths(quantized)
            all_runs.extend(runs)
            lines = (
                f"Layer: {name} | Original: {weight.numel():>8} | Runs: {len(runs):>6}"
            )
            print(lines)
            log_lines.append(lines)

    save_quantized_weigths(quantized_dict, weight_path)
    save_log(log_lines, log_path)

    # 统计每个数值的run长度分布
    run_dist = defaultdict(list)
    for val, length in all_runs:
        run_dist[val].append(length)
    save_stats(run_dist, stats_path)

    # 绘制每个数值的run长度分布图
    for i in range(16):
        if i in run_dist:
            plt.figure()
            plt.hist(run_dist[i], bins=50, alpha=0.7)
            plt.title(f"Run length distribution for quantized value {i}")
            plt.xlabel("Run length")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(plot_dir, f"run_length_value_{i}.png"))
            plt.close()
    print(f"Analysis for {model_name} completed. Results saved in {model_dir}.")


if __name__ == "__main__":
    model_name = "facebook/opt-125m"
    out_dir = "./output/"
    analyze_model(model_name, out_dir)
