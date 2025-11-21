import torch
import numpy as np
import os
from math import ceil, log2
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from srcs.quantizer.real_quantize import real_quantize_tensor

# MODEL_ID = "facebook/opt-125m"
# MODEL_ID = "facebook/opt-350m"
MODEL_ID = "facebook/opt-1.3b"
GROUP_SIZE = 512
N_BITS = 4
K_OPTIONS = [1, 2]

# 引入 Fallback 阈值
# 只有当 GR 编码节省超过 5% (1.0 - 0.95) 时才使用它
FALLBACK_SAVING_THRESHOLD = 0.95


OUTPUT_DIR = f"./compressed_{MODEL_ID.replace('/', '_')}_gs{GROUP_SIZE}_V2_analysis"


def map_signed_to_unsigned(deltas):
    unsigned = np.zeros_like(deltas)
    unsigned[deltas >= 0] = 2 * deltas[deltas >= 0]
    unsigned[deltas < 0] = -2 * deltas[deltas < 0] - 1
    return unsigned


def golomb_rice_encoder(value, k):
    """
    只返回比特流.
    """
    val = value.item()
    q = val >> k
    r = val & ((1 << k) - 1)

    return "1" * q + "0" + format(r, f"0{k}b")


def find_optimal_k_and_encode(grp_values, k_options):
    """根据 min_bits 选择 best_k."""
    best_k = -1
    best_bitstream = ""
    min_bits = float("inf")

    for k in k_options:
        current_bitstream_list = []
        current_total_bits = 0

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
    """更新以读取新的元数据"""
    total_original_bits = 0
    total_compressed_weights_bits = 0
    total_metadata_bits = 0  # 现在包含 ZP 和索引

    npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])
    if not npz_files:
        return

    for filename in npz_files:
        path = os.path.join(output_dir, filename)
        data = np.load(path)

        total_original_bits += data["original_quantized"].size * 4
        total_compressed_weights_bits += data["compressed_weights"].nbytes * 8

        # 读取合并后的元数据
        total_metadata_bits += data["metadata"].nbytes * 8
        total_metadata_bits += data["scales"].nbytes * 8  # Scales 仍然是独立的

    total_final_bits = total_compressed_weights_bits + total_metadata_bits
    compression_ratio = (
        total_original_bits / total_final_bits if total_final_bits > 0 else 0
    )
    total_weights = total_original_bits / N_BITS
    avg_bits = total_final_bits / total_weights if total_weights > 0 else 0

    print(f"gs={GROUP_SIZE} Summary (V2 Analysis):")
    print("\n" + "=" * 80 + "\n--- Overall Model Compression Summary ---\n" + "=" * 80)
    print(f"Original Total Size: {total_original_bits / 8 / 1024 / 1024:.2f} MB")
    print(f"  - Metadata(ZPs, Indices): {total_metadata_bits / 8 / 1024 / 1024:.2f} MB")
    print("-" * 40 + "\nFinal Compressed Size Breakdown:")
    print(f"  - Compressed Weights: {total_compressed_weights_bits / 8 / 1024:.2f} KB")
    print(f"  - Metadata (ZPs, Indices): {total_metadata_bits / 8 / 1024:.2f} KB")
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
            return_scale=True,
        )
        quantized_weights = quantized_weights.flatten().numpy()
        scales = scales.numpy()
        zero_points = zero_points.numpy()  # 这是 [num_groups]

        num_groups = len(zero_points)

        # 标志位现在只有 3 种状态
        num_k_choices = len(K_OPTIONS)  # 2
        flag_bits = ceil(log2(num_k_choices + 1))  # ceil(log2(3)) = 2 bits

        # 00: k=1, 01: k=2, 10: Fallback
        k_to_flag = {k: format(i, f"0{flag_bits}b") for i, k in enumerate(K_OPTIONS)}
        fallback_flag = format(num_k_choices, f"0{flag_bits}b")  # "10"

        layer_bitstream_chunks = []  # 存储每个组的 (padded) 字符串

        # 元数据将合并
        metadata_list = []  # 存储 (Byte_Offset, Zero_Point)
        current_byte_offset = 0

        original_group_size_bits = GROUP_SIZE * N_BITS

        for g in range(num_groups):
            # 记录这组的元数据
            # 地址是 *这组开始* 的字节偏移
            # ZP 是这组的零点
            metadata_list.append((current_byte_offset, zero_points[g]))

            start = g * GROUP_SIZE
            end = min(start + GROUP_SIZE, len(quantized_weights))
            group_data = quantized_weights[start:end]
            deltas = group_data.astype(np.int16) - zero_points[g]
            mapped_deltas = map_signed_to_unsigned(deltas)

            # 只看 min_bits
            group_bitstream, best_k = find_optimal_k_and_encode(
                mapped_deltas, K_OPTIONS
            )
            compressed_size = len(group_bitstream) + flag_bits

            # 使用新的阈值来决定是否 Fallback
            use_fallback = compressed_size >= (
                original_group_size_bits * FALLBACK_SAVING_THRESHOLD
            )

            if use_fallback:
                raw_bits = "".join([format(val, "04b") for val in group_data])
                final_group_stream = fallback_flag + raw_bits
            else:
                final_group_stream = k_to_flag[best_k] + group_bitstream

            # 每组进行 8-bit 对齐填充
            num_padding_bits = (8 - len(final_group_stream) % 8) % 8
            padded_group_stream = final_group_stream + "0" * num_padding_bits

            layer_bitstream_chunks.append(padded_group_stream)

            # 更新下一个组的字节偏移
            current_byte_offset += len(padded_group_stream) // 8

        # 1. 构建最终比特流
        layer_bitstream = "".join(layer_bitstream_chunks)
        total_valid_bits = len(layer_bitstream)  # 这是总的字节对齐后的长度

        # 2. 转换为 Bytes
        byte_array = bytearray()
        for i in range(0, total_valid_bits, 8):
            byte_chunk = layer_bitstream[i : i + 8]
            byte_array.append(int(byte_chunk, 2))
        packed_bytes = bytes(byte_array)

        # 3. 创建元数据结构体数组
        metadata_dtype = np.dtype(
            [("offset", np.uint32), ("zp", np.uint8)]  # 字节偏移量  # 零点
        )
        metadata_array = np.array(metadata_list, dtype=metadata_dtype)

        output_path = os.path.join(OUTPUT_DIR, f"{name}.npz")

        # 4. 保存格式
        np.savez(
            output_path,
            compressed_weights=np.frombuffer(packed_bytes, dtype=np.uint8),
            metadata=metadata_array,  # 保存合并后的元数据
            scales=scales,
            original_quantized=quantized_weights,  # 仅用于分析
            group_size=np.array([GROUP_SIZE], dtype=np.uint16),
            total_valid_bits=np.array([total_valid_bits], dtype=np.uint64),
        )

    print(f"\nCompression complete. All artifacts saved to '{OUTPUT_DIR}'.")
    summarize_compression_results(OUTPUT_DIR)


if __name__ == "__main__":
    main()
