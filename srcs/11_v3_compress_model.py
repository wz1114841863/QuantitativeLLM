import torch
import numpy as np
import os
import shutil
import sys
import argparse
from math import ceil, log2
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from srcs.quantizer.real_quantize import real_quantize_tensor


def clear_npz_files(output_dir):
    """
    清除所有生成的npz文件以节省空间,但保留报告文件和分布采样数据
    """
    npz_files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]

    if not npz_files:
        print("No .npz files found to clear.")
        return

    print(f"Found {len(npz_files)} .npz files to clear...")

    cleared_count = 0
    for filename in npz_files:
        file_path = os.path.join(output_dir, filename)
        try:
            os.remove(file_path)
            cleared_count += 1
        except Exception as e:
            print(f"Error removing {filename}: {e}")

    print(f"Successfully cleared {cleared_count} .npz files from {output_dir}")

    # 检查是否还有残留文件
    remaining_npz = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
    if remaining_npz:
        print(f"Warning: {len(remaining_npz)} .npz files remain in directory")
    else:
        print("All .npz files have been successfully cleared")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress LLM weights with Grouped Quantization & GR Encoding"
    )

    parser.add_argument(
        "--model_id", type=str, default="facebook/opt-125m", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--group_size", type=int, default=512, help="Group size for quantization"
    )
    parser.add_argument(
        "--n_bits", type=int, default=4, help="Target quantization bits"
    )
    parser.add_argument(
        "--fallback_threshold", type=float, default=0.95, help="Fallback threshold"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="./experiments",
        help="Base directory for experiment results",
    )
    parser.add_argument(
        "--save_distribution",
        action="store_true",
        default=True,
        help="Whether to save raw delta distribution for plotting",
    )
    return parser.parse_args()


args = parse_args()


# 全局配置
MODEL_ID = args.model_id
GROUP_SIZE = args.group_size
N_BITS = args.n_bits
K_OPTIONS = [1, 2]
FALLBACK_SAVING_THRESHOLD = args.fallback_threshold

# 路径配置
safe_model_name = MODEL_ID.replace("/", "_")
OUTPUT_DIR = os.path.join(args.output_base, f"{safe_model_name}_gs{GROUP_SIZE}")
DIST_DIR = os.path.join(OUTPUT_DIR, "distribution_samples")
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "all_summary_log.txt")


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


def summarize_compression_results(output_dir):
    """
    生成最终统计报告 (单位统一为 MB)
    """
    total_original_bits = 0
    total_compressed_weights_bits = 0
    total_metadata_bits = 0

    npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])
    if not npz_files:
        print("No .npz files found.")
        return

    for filename in npz_files:
        path = os.path.join(output_dir, filename)
        try:
            data = np.load(path)
            # Original Size: 仅计算 4-bit 权重的净载荷
            total_original_bits += data["original_quantized"].size * N_BITS

            # Compressed Size: 压缩后的权重流
            total_compressed_weights_bits += data["compressed_weights"].nbytes * 8

            # Metadata: 索引表 + ZP + Scales
            total_metadata_bits += data["metadata"].nbytes * 8
            total_metadata_bits += data["scales"].nbytes * 8
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    # --- 单位转换 (Bits -> Bytes -> MB) ---
    to_mb = lambda bits: bits / 8 / 1024 / 1024

    original_size_mb = to_mb(total_original_bits)
    comp_weights_mb = to_mb(total_compressed_weights_bits)
    metadata_mb = to_mb(total_metadata_bits)

    total_final_mb = comp_weights_mb + metadata_mb

    # 压缩比 = 原始4bit权重 / (压缩权重 + 所有元数据)
    compression_ratio = original_size_mb / total_final_mb if total_final_mb > 0 else 0

    total_weights_count = total_original_bits / N_BITS
    # Bits Per Weight = 总比特数 / 权重个数
    avg_bits = (
        (total_final_mb * 8 * 1024 * 1024) / total_weights_count
        if total_weights_count > 0
        else 0
    )

    print(f"\ngs={GROUP_SIZE} Summary (Automated Report):")
    print("=" * 80 + "\n--- Overall Model Compression Summary ---\n" + "=" * 80)
    print(f"Model: {MODEL_ID}")
    print(f"Group Size: {GROUP_SIZE}")
    print("-" * 40)
    print(f"Original Raw Size (4-bit Weights Only):   {original_size_mb:.2f} MB")
    print("-" * 40)
    print("Final Compressed Size Breakdown:")
    print(f"  - Compressed Weights Stream:            {comp_weights_mb:.2f} MB")
    print(f"  - All Metadata (ZP, Indices, Scales):   {metadata_mb:.2f} MB")
    print("-" * 40)
    print(f"Total Final Size on Disk:                 {total_final_mb:.2f} MB")
    print("-" * 40)
    print("--- Final Performance Metrics ---")
    print(f"Overall Compression Ratio:                {compression_ratio:.2f}x")
    print(f"  (Calculated as: Original_Raw_4bit / Total_Final_Size)")
    print(f"Final Average Bits Per Weight:            {avg_bits:.2f} bits")
    print("=" * 80 + "\n")


def main():
    if os.path.exists(LOG_FILE_PATH):
        print(f"Experiment already completed for {OUTPUT_DIR}. Skipping...")
        print(f"Check log at: {LOG_FILE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.save_distribution:
        os.makedirs(DIST_DIR, exist_ok=True)

    sys.stdout = Logger(LOG_FILE_PATH)

    print(f"--- Starting Compression Experiment ---")
    print(f"Model: {MODEL_ID}")
    print(f"Group Size: {GROUP_SIZE}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # 4. 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="float16")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    target_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    layer_names = list(target_layers.keys())
    print(f"Found {len(target_layers)} linear layers to compress.")

    # --- 采样层选择 ---
    indices_to_sample = [
        0,
        len(layer_names) // 2,
        len(layer_names) - 2 if len(layer_names) > 1 else 0,
    ]
    layers_to_sample = set([layer_names[i] for i in indices_to_sample])

    if args.save_distribution:
        print(f"Distribution sampling enabled. Target layers: {list(layers_to_sample)}")

    for name, layer in tqdm(target_layers.items(), desc="Compressing Layers"):
        original_weights = layer.weight.data.clone()

        quantized_weights, zero_points, scales = real_quantize_tensor(
            original_weights,
            zero_point=True,
            group_size=GROUP_SIZE,
            return_scale=128,
        )
        quantized_weights = quantized_weights.flatten().numpy()
        scales = scales.numpy()
        zero_points = zero_points.numpy()

        num_groups = len(zero_points)

        num_k_choices = len(K_OPTIONS)
        flag_bits = ceil(log2(num_k_choices + 1))
        k_to_flag = {k: format(i, f"0{flag_bits}b") for i, k in enumerate(K_OPTIONS)}
        fallback_flag = format(num_k_choices, f"0{flag_bits}b")

        layer_bitstream_chunks = []
        metadata_list = []
        current_byte_offset = 0
        original_group_size_bits = GROUP_SIZE * N_BITS

        # 采样容器 (针对特定层收集整个层的权重)
        is_sample_target = (name in layers_to_sample) and args.save_distribution
        layer_mapped_deltas_sample = []

        for g in range(num_groups):
            metadata_list.append((current_byte_offset, zero_points[g]))

            start = g * GROUP_SIZE
            end = min(start + GROUP_SIZE, len(quantized_weights))
            group_data = quantized_weights[start:end]

            deltas = group_data.astype(np.int16) - zero_points[g]
            mapped_deltas = map_signed_to_unsigned(deltas)

            # 收集整层的 mapped deltas
            if is_sample_target:
                layer_mapped_deltas_sample.extend(mapped_deltas)

            group_bitstream, best_k = find_optimal_k_and_encode(
                mapped_deltas, K_OPTIONS
            )
            compressed_size = len(group_bitstream) + flag_bits

            use_fallback = compressed_size >= (
                original_group_size_bits * FALLBACK_SAVING_THRESHOLD
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

        if is_sample_target:
            sample_filename = f"{name}_mapped_deltas.npy"
            sample_path = os.path.join(DIST_DIR, sample_filename)
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

        output_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
        np.savez(
            output_path,
            compressed_weights=np.frombuffer(packed_bytes, dtype=np.uint8),
            metadata=metadata_array,
            scales=scales,
            original_quantized=quantized_weights,
            group_size=np.array([GROUP_SIZE], dtype=np.uint16),
            total_valid_bits=np.array([total_valid_bits], dtype=np.uint64),
        )

    print(f"\nCompression complete. Artifacts saved to '{OUTPUT_DIR}'.")
    summarize_compression_results(OUTPUT_DIR)

    # 添加清除npz文件的调用
    print("\n" + "=" * 50)
    print("Cleaning up .npz files to save space...")
    clear_npz_files(OUTPUT_DIR)
    print("Cleanup completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
