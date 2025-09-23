import os
import gc
import json
import torch
import pickle


def release_memory():
    """aggressive but safe for both CUDA & CPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 如果用了 mps(Mac)
    if hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()


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


def save_layer_run_stats(layer_stats, out_dir):
    """Save per-layer run-length statistics to a text file."""
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    with open(
        os.path.join(out_dir, "layer_run_length_stats.json"), "w", encoding="utf8"
    ) as f:
        json.dump(layer_stats, f, indent=2)


def save_json_file(data, path):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
