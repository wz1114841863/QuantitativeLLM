import os

# 限制每个进程只使用 1 个线程,防止线程爆炸导致卡死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import sys
import argparse
import multiprocessing
from math import ceil, log2
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from srcs.quantizer.real_quantize import real_quantize_tensor
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- 辅助函数 ---
def clear_npz_files(output_dir):
    """清除 .npz 文件"""
    npz_files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
    if not npz_files:
        return
    print(f"Cleaning up {len(npz_files)} .npz files...")
    for filename in npz_files:
        try:
            os.remove(os.path.join(output_dir, filename))
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel LLM Compression")
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m")
    parser.add_argument("--group_size", type=int, default=512)
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--fallback_threshold", type=float, default=0.95)
    parser.add_argument("--output_base", type=str, default="./experiments")
    parser.add_argument("--save_distribution", action="store_true", default=True)
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers (0 = auto)"
    )
    return parser.parse_args()


def map_signed_to_unsigned(deltas):
    unsigned = np.zeros_like(deltas)
    unsigned[deltas >= 0] = 2 * deltas[deltas >= 0]
    unsigned[deltas < 0] = -2 * deltas[deltas < 0] - 1
    return unsigned


def golomb_rice_encoder(value, k):
    val = value.item()
    q = val >> k
    r = val & ((1 << k) - 1)
    return "1" * q + "0" + format(r, f"0{k}b")


def find_optimal_k_and_encode(grp_values, k_options):
    best_k = -1
    best_bitstream = ""
    min_bits = float("inf")

    for k in k_options:
        current_bitstream_list = []
        for v in grp_values:
            bits = golomb_rice_encoder(v, k)
            current_bitstream_list.append(bits)

        current_bitstream = "".join(current_bitstream_list)
        current_total_bits = len(current_bitstream)

        if current_total_bits < min_bits:
            min_bits = current_total_bits
            best_k = k
            best_bitstream = current_bitstream

    return best_bitstream, best_k


# --- 单层处理任务函数 ---
def compress_single_layer(
    layer_name,
    original_weights_numpy,
    group_size,
    k_options,
    n_bits,
    fallback_threshold,
    output_dir,
    dist_dir,
    is_sample_target,
):
    try:
        # 在子进程内重新转为 Tensor
        original_weights = torch.from_numpy(original_weights_numpy)

        quantized_weights, zero_points, scales = real_quantize_tensor(
            original_weights,
            zero_point=True,
            group_size=group_size,
            return_scale=128,
        )
        quantized_weights = quantized_weights.flatten().numpy()
        scales = scales.numpy()
        zero_points = zero_points.numpy()

        num_groups = len(zero_points)
        num_k_choices = len(k_options)
        flag_bits = ceil(log2(num_k_choices + 1))

        k_to_flag = {k: format(i, f"0{flag_bits}b") for i, k in enumerate(k_options)}
        fallback_flag = format(num_k_choices, f"0{flag_bits}b")

        layer_bitstream_chunks = []
        metadata_list = []
        current_byte_offset = 0
        original_group_size_bits = group_size * n_bits

        layer_mapped_deltas_sample = [] if is_sample_target else None

        for g in range(num_groups):
            metadata_list.append((current_byte_offset, zero_points[g]))

            start = g * group_size
            end = min(start + group_size, len(quantized_weights))
            group_data = quantized_weights[start:end]

            deltas = group_data.astype(np.int16) - zero_points[g]
            mapped_deltas = map_signed_to_unsigned(deltas)

            if is_sample_target:
                layer_mapped_deltas_sample.extend(mapped_deltas)

            group_bitstream, best_k = find_optimal_k_and_encode(
                mapped_deltas, k_options
            )
            compressed_size = len(group_bitstream) + flag_bits

            use_fallback = compressed_size >= (
                original_group_size_bits * fallback_threshold
            )

            if use_fallback:
                raw_bits = "".join([format(val, "04b") for val in group_data])
                final_group_stream = fallback_flag + raw_bits
            else:
                final_group_stream = k_to_flag[best_k] + group_bitstream

            num_padding_bits = (8 - len(final_group_stream) % 8) % 8
            padded_group_stream = final_group_stream + "0" * num_padding_bits

            layer_bitstream_chunks.append(padded_group_stream)
            current_byte_offset += len(padded_group_stream) // 8

        if is_sample_target and layer_mapped_deltas_sample:
            sample_path = os.path.join(dist_dir, f"{layer_name}_mapped_deltas.npy")
            np.save(sample_path, np.array(layer_mapped_deltas_sample, dtype=np.uint16))

        layer_bitstream = "".join(layer_bitstream_chunks)
        total_valid_bits = len(layer_bitstream)

        byte_array = bytearray()
        for i in range(0, total_valid_bits, 8):
            byte_chunk = layer_bitstream[i : i + 8]
            byte_array.append(int(byte_chunk, 2))
        packed_bytes = bytes(byte_array)

        metadata_dtype = np.dtype([("offset", np.uint32), ("zp", np.uint8)])
        metadata_array = np.array(metadata_list, dtype=metadata_dtype)

        output_path = os.path.join(output_dir, f"{layer_name}.npz")
        np.savez(
            output_path,
            compressed_weights=np.frombuffer(packed_bytes, dtype=np.uint8),
            metadata=metadata_array,
            scales=scales,
            original_quantized=quantized_weights,
            group_size=np.array([group_size], dtype=np.uint16),
            total_valid_bits=np.array([total_valid_bits], dtype=np.uint64),
        )

        stats = {
            "original_bits": len(quantized_weights) * n_bits,
            "compressed_weights_bits": len(packed_bytes) * 8,
            "metadata_bits": metadata_array.nbytes * 8 + scales.nbytes * 8,
        }
        return stats

    except Exception as e:
        return {"error": str(e), "layer": layer_name}


class Logger(object):
    def __init__(self, filename="summary.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    args = parse_args()

    MODEL_ID = args.model_id
    GROUP_SIZE = args.group_size
    N_BITS = args.n_bits
    K_OPTIONS = [1, 2]
    FALLBACK_THRESHOLD = args.fallback_threshold

    safe_model_name = MODEL_ID.replace("/", "_")
    OUTPUT_DIR = os.path.join(args.output_base, f"{safe_model_name}_gs{GROUP_SIZE}")
    DIST_DIR = os.path.join(OUTPUT_DIR, "distribution_samples")
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "all_summary_log.txt")

    if os.path.exists(LOG_FILE_PATH):
        print(f"Experiment completed. Check log: {LOG_FILE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.save_distribution:
        os.makedirs(DIST_DIR, exist_ok=True)

    sys.stdout = Logger(LOG_FILE_PATH)

    # 稍微减少 Worker 数量,留一点资源给系统
    max_workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 2)

    print(f"--- Parallel Compression Start ---")
    print(f"Model: {MODEL_ID} | Workers: {max_workers}")
    print(f"Group Size: {GROUP_SIZE} | Fallback: {FALLBACK_THRESHOLD}")

    print("Loading model metadata...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    target_layers = {
        n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    layer_names = list(target_layers.keys())
    print(f"Found {len(target_layers)} layers.")

    indices_to_sample = (
        [0, len(layer_names) // 2, len(layer_names) - 2]
        if len(layer_names) > 2
        else [0]
    )
    layers_to_sample = set([layer_names[i] for i in indices_to_sample])
    if args.save_distribution:
        print(f"Sampling layers: {list(layers_to_sample)}")

    g_original_bits = 0
    g_comp_bits = 0
    g_meta_bits = 0

    futures = []

    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        print("Submitting tasks...")
        for name, layer in target_layers.items():
            weights_np = layer.weight.data.float().numpy()
            is_sample = (name in layers_to_sample) and args.save_distribution

            future = executor.submit(
                compress_single_layer,
                name,
                weights_np,
                GROUP_SIZE,
                K_OPTIONS,
                N_BITS,
                FALLBACK_THRESHOLD,
                OUTPUT_DIR,
                DIST_DIR,
                is_sample,
            )
            futures.append(future)

            # 释放主进程内存
            layer.weight.data = torch.tensor([])

        print(
            "Processing layers in parallel (Please wait for first progress bar update)..."
        )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Compressing"
        ):
            stats = future.result()
            if stats and "error" not in stats:
                g_original_bits += stats["original_bits"]
                g_comp_bits += stats["compressed_weights_bits"]
                g_meta_bits += stats["metadata_bits"]
            elif stats and "error" in stats:
                print(f"\n[ERROR] Layer: {stats.get('layer')}")
                print(f"Message: {stats['error']}")

    to_mb = lambda b: b / 8 / 1024 / 1024
    total_final_mb = to_mb(g_comp_bits + g_meta_bits)
    orig_mb = to_mb(g_original_bits)
    ratio = orig_mb / total_final_mb if total_final_mb > 0 else 0

    total_weights = g_original_bits / N_BITS
    avg_bits = (g_comp_bits + g_meta_bits) / total_weights if total_weights > 0 else 0

    print(f"\nSummary (Parallel):")
    print("=" * 60)
    print(f"Original Size (4-bit raw): {orig_mb:.2f} MB")
    print(f"Compressed Weights:        {to_mb(g_comp_bits):.2f} MB")
    print(f"Metadata:                  {to_mb(g_meta_bits):.2f} MB")
    print("-" * 40)
    print(f"Total Size:                {total_final_mb:.2f} MB")
    print(f"Compression Ratio:         {ratio:.2f}x")
    print(f"Avg Bits/Weight:           {avg_bits:.2f} bits")
    print("=" * 60)

    print("Cleaning up .npz files...")
    clear_npz_files(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    # [关键修改 3]确保 Windows/Linux 安全启动
    multiprocessing.freeze_support()
    main()
