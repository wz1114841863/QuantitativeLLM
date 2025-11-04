import torch
import numpy as np
import os
from math import ceil, log2
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from srcs.quantizer.real_quantize import real_quantize_tensor

"""
文件说明:
    完整的模型压缩流程实现, 包括量化和Golomb-Rice编码
"""

MODEL_ID = "facebook/opt-125m"
# MODEL_ID = "facebook/opt-350m"
# MODEL_ID = "facebook/opt-1.3b"
GROUP_SIZE = 512
N_BITS = 4
K_OPTIONS = [1, 2, 3]
# OUTPUT_DIR = f"./compressed_opt-125m_gs{GROUP_SIZE}"
OUTPUT_DIR = f"./compressed_{MODEL_ID.replace('/', '_')}_gs{GROUP_SIZE}"


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
        current_bitstream = "".join([golomb_rice_encoder(v, k) for v in grp_values])
        if len(current_bitstream) < min_bits:
            min_bits = len(current_bitstream)
            best_k = k
            best_bitstream = current_bitstream
    return best_bitstream, best_k


def summarize_compression_results(output_dir):
    """Analyzes all compressed .npz files and prints a final summary report."""
    total_original_bits = 0
    total_compressed_weights_bits = 0
    total_index_table_bits = 0
    total_metadata_bits = 0  # For scales and zero_points
    npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])
    if not npz_files:
        return
    for filename in npz_files:
        path = os.path.join(output_dir, filename)
        data = np.load(path)
        total_original_bits += data["original_quantized"].size * 4
        total_compressed_weights_bits += data["compressed_weights"].nbytes * 8
        total_index_table_bits += data["index_table"].nbytes * 8
        total_metadata_bits += data["scales"].nbytes * 8
        total_metadata_bits += data["zero_points"].nbytes * 8
    total_final_bits = (
        total_compressed_weights_bits + total_index_table_bits + total_metadata_bits
    )
    compression_ratio = (
        total_original_bits / total_final_bits if total_final_bits > 0 else 0
    )
    total_weights = total_original_bits / N_BITS
    avg_bits = total_final_bits / total_weights if total_weights > 0 else 0
    print(f"gs={GROUP_SIZE} Summary:")
    print("\n" + "=" * 80 + "\n--- Overall Model Compression Summary ---\n" + "=" * 80)
    print(
        f"Total Original Size (4-bit weights): {total_original_bits / 8 / 1024:.2f} KB"
    )
    print("-" * 40 + "\nFinal Compressed Size Breakdown:")
    print(f"  - Compressed Weights: {total_compressed_weights_bits / 8 / 1024:.2f} KB")
    print(f"  - Index Tables:       {total_index_table_bits / 8 / 1024:.2f} KB")
    print(
        f"  - Other Metadata:     {total_metadata_bits / 8 / 1024:.2f} KB (Scales & ZPs)"
    )
    print(f"Total Final Size on Disk: {total_final_bits / 8 / 1024:.2f} KB")
    print("-" * 40 + "\n--- Final Performance Metrics ---")
    print(f"Overall Compression Ratio: {compression_ratio:.2f}x")
    print(
        f"Final Average Bits Per Weight: {avg_bits:.2f} bits (including all overhead)"
    )


def main():
    print(f"--- Loading model '{MODEL_ID}' ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    print(f"Found {len(target_layers)} linear layers to compress.")

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

        layer_bitstream = ""
        index_table = []
        current_bit_offset = 0
        original_group_size_bits = GROUP_SIZE * N_BITS

        for g in range(num_groups):
            index_table.append(current_bit_offset)

            start = g * GROUP_SIZE
            end = min(start + GROUP_SIZE, len(quantized_weights))
            group_data = quantized_weights[start:end]
            deltas = group_data.astype(np.int16) - zero_points[g]
            mapped_deltas = map_signed_to_unsigned(deltas)

            group_bitstream, best_k = find_optimal_k_and_encode(
                mapped_deltas, K_OPTIONS
            )
            compressed_size = len(group_bitstream) + flag_bits

            if compressed_size >= original_group_size_bits:
                raw_bits = "".join([format(val, "04b") for val in group_data])
                layer_bitstream += fallback_flag + raw_bits
                current_bit_offset += flag_bits + len(raw_bits)
            else:
                layer_bitstream += k_to_flag[best_k] + group_bitstream
                current_bit_offset += compressed_size

        num_padding_bits = (8 - len(layer_bitstream) % 8) % 8
        padded_bitstream = layer_bitstream + "0" * num_padding_bits

        # Second, convert the padded string to bytes chunk by chunk
        byte_array = bytearray()
        for i in range(0, len(padded_bitstream), 8):
            byte_chunk = padded_bitstream[i : i + 8]
            byte_array.append(int(byte_chunk, 2))
        packed_bytes = bytes(byte_array)  # Final bytes object

        output_path = os.path.join(OUTPUT_DIR, f"{name}.npz")
        np.savez(
            output_path,
            compressed_weights=np.frombuffer(packed_bytes, dtype=np.uint8),
            index_table=np.array(index_table, dtype=np.uint32),
            scales=scales,
            zero_points=np.array(zero_points, dtype=np.uint8),
            original_quantized=quantized_weights,
            group_size=np.array([GROUP_SIZE], dtype=np.uint16),
            total_valid_bits=np.array(
                [len(layer_bitstream)], dtype=np.uint64
            ),  # IMPORTANT: Still use the UNPADDED length here
        )

    print(f"\nCompression complete. All artifacts saved to '{OUTPUT_DIR}'.")
    summarize_compression_results(OUTPUT_DIR)


if __name__ == "__main__":
    main()
