import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from scipy.stats import entropy
from math import ceil, log2
from srcs.quantizer.real_quantize import *
from srcs.utils.save_layer_werights import load_saved_layer


"""
文件说明:
    统计分组量化 + 零点量化后, 每组权重的分布情况.
    中心点数值, 分布概率.
    采用不同编码方式后的位宽.
"""


def get_hist_per_group(layer_path, index):
    """统计每组分布情况
    结论: 每组分布的众数不统一, 分布范围广

    """
    zp = True
    gs = 128
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, return_scale = real_quantize_tensor(
        weight, zero_point=zp, group_size=gs, return_scale=True
    )

    group_size = gs
    q = quantized.flatten().cpu().numpy()  # 已经是 uint8
    n_total = q.size
    n_group = (n_total + group_size - 1) // group_size

    print("group  mode  dist=0  dist=1  dist=2  dist=3  dist=4  dist≥5")

    for g in range(n_group):
        start = g * group_size
        end = min(start + group_size, n_total)
        grp = q[start:end]  # 本组码值

        # 1. 众数
        values, counts = np.unique(grp, return_counts=True)
        mode_code = values[counts.argmax()]

        # 2. 距离分布
        dist = np.abs(grp - mode_code)
        bins = np.array([0, 1, 2, 3, 4, 5, 16])
        hist, _ = np.histogram(dist, bins=bins)  # len=6
        pct = hist / grp.size * 100

        # 3. 打印
        print(
            f"{g:5d}  {mode_code:4d}  "
            f"{pct[0]:6.1f}  {pct[1]:6.1f}  {pct[2]:6.1f}  "
            f"{pct[3]:6.1f}  {pct[4]:6.1f}  {pct[5]:6.1f}"
        )


def get_hist_per_layer(layer_path, index):
    """每层量化权重的分布情况"""
    zp = True
    gs = 128
    weight, bias, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\nLayer: {name} | Original elems: {weight.numel():>8}")
    quantized, zero_point, return_scale = real_quantize_tensor(
        weight, zero_point=zp, group_size=gs, return_scale=True
    )

    # 1. 拉直并转到 CPU
    q = quantized.flatten().cpu().numpy().astype(np.uint8)

    # 2. 一次性直方图
    counts = np.bincount(q, minlength=16)  # 长度固定 16
    pct = counts / counts.sum() * 100

    # 3. 打印
    print("code  count   percent")
    for code, (c, p) in enumerate(zip(counts, pct)):
        print(f"{code:4d}  {c:8d}  {p:7.3f}%")


def analyze_compression_potential(layer_path, index, group_size=128, n_bits=4):
    """
    对给定的网络层进行更全面的分组压缩潜力分析.

    分析包括:
    1.  每组的众数/零点/熵/唯一值数量.
    2.  数据围绕"众数"和"零点"的距离分布.
    3.  整个层的汇总统计:
        - 平均理论压缩率 (基于熵).
        - 众数和零点的整体分布情况.
        - 众数与零点的一致性.
        - 所有组的平均距离分布.
    """
    # 1. 数据加载与量化
    weight, _, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(
        f"\n{'='*80}\nAnalyzing Layer: {name}\nOriginal elems: {weight.numel():>8}, Group Size: {group_size}\n{'='*80}"
    )

    quantized, zero_points, _ = real_quantize_tensor(
        weight, zero_point=True, group_size=group_size, return_scale=True
    )

    q_flat = quantized.flatten().cpu().numpy()
    zps = zero_points.cpu().numpy()

    n_total = q_flat.size
    n_group = zps.size

    # 2. 逐组分析,收集数据
    group_stats = []
    max_val = 2**n_bits

    for g in range(n_group):
        start = g * group_size
        end = min(start + group_size, n_total)
        grp = q_flat[start:end]
        zp_val = zps[g]

        # 计算众数
        values, counts = np.unique(grp, return_counts=True)
        mode_val = values[counts.argmax()]

        # 计算熵 (理论最小比特数)
        probs = counts / grp.size
        # entropy_val = -np.sum(probs * np.log2(probs)) # 手动计算
        entropy_val = entropy(probs, base=2)

        # 计算距离分布
        dist_from_mode = np.abs(grp - mode_val)
        dist_from_zp = np.abs(grp - zp_val)

        # 使用np.bincount进行高效统计
        bins = np.arange(-0.5, max_val, 1)

        # 使用 np.histogram 计算直方图
        dist_mode_hist_counts, _ = np.histogram(dist_from_mode, bins=bins)
        dist_zp_hist_counts, _ = np.histogram(dist_from_zp, bins=bins)

        # 归一化为百分比
        dist_mode_hist = dist_mode_hist_counts / grp.size * 100
        dist_zp_hist = dist_zp_hist_counts / grp.size * 100

        group_stats.append(
            {
                "group_id": g,
                "mode": mode_val,
                "zero_point": zp_val,
                "entropy": entropy_val,
                "unique_values": len(values),
                "dist_from_mode_hist": dist_mode_hist,
                "dist_from_zp_hist": dist_zp_hist,
            }
        )

    # 3. 汇总分析
    df = pd.DataFrame(group_stats)

    # 计算平均理论比特数和压缩率
    avg_entropy = df["entropy"].mean()
    original_bits = n_bits
    compression_ratio = original_bits / avg_entropy

    # 统计众数和零点的分布
    mode_counts = df["mode"].value_counts().sort_index()
    zp_counts = df["zero_point"].value_counts().sort_index()

    # 众数与零点的一致性
    mode_zp_agreement = (df["mode"] == df["zero_point"]).mean() * 100

    # 平均距离分布
    avg_dist_mode_hist = np.mean(np.stack(df["dist_from_mode_hist"]), axis=0)
    avg_dist_zp_hist = np.mean(np.stack(df["dist_from_zp_hist"]), axis=0)

    # 4. 打印报告
    print("--- 1. Overall Compression Potential ---")
    print(f"Original Bits Per Weight: {original_bits:.2f}")
    print(f"Average Theoretical Bits Per Weight (Entropy): {avg_entropy:.2f}")
    print(f"Theoretical Compression Ratio: {compression_ratio:.2f}x")
    print("-" * 40)

    print("\n--- 2. Center Point Analysis ---")
    print(f"Agreement between Mode and Zero-Point: {mode_zp_agreement:.2f}%")

    # 创建一个DataFrame来并排显示分布
    dist_df = (
        pd.DataFrame({"Mode Freq.": mode_counts, "ZP Freq.": zp_counts})
        .fillna(0)
        .astype(int)
    )
    print("Distribution of Center Points (Mode vs. Zero-Point):")
    print(dist_df)
    print("-" * 40)

    print("\n--- 3. Average Group Distribution Shape ---")
    print("Distance | % of values (centered on MODE) | % of values (centered on ZP)")
    for i in range(8):  # 只显示前8个距离
        print(
            f"  {i:^6} | {avg_dist_mode_hist[i]:^30.2f} | {avg_dist_zp_hist[i]:^29.2f}"
        )

    dist_ge_8_mode = avg_dist_mode_hist[8:].sum()
    dist_ge_8_zp = avg_dist_zp_hist[8:].sum()
    print(f"  {'>=8':^6} | {dist_ge_8_mode:^30.2f} | {dist_ge_8_zp:^29.2f}")
    print("-" * 40)

    # 可以选择性地打印一些表现极端(最好/最差压缩潜力)的组
    print("\n--- 4. Example Groups ---")
    print("Groups with lowest entropy (highest compression potential):")
    print(
        df.nsmallest(3, "entropy")[
            ["group_id", "mode", "zero_point", "entropy", "unique_values"]
        ]
    )
    print("\nGroups with highest entropy (lowest compression potential):")
    print(
        df.nlargest(3, "entropy")[
            ["group_id", "mode", "zero_point", "entropy", "unique_values"]
        ]
    )

    return df  # 返回DataFrame以供后续深入分析


def map_signed_to_unsigned(deltas):
    """Maps signed deltas to unsigned integers for encoding."""
    # Mapping: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, ...
    unsigned = np.zeros_like(deltas)
    unsigned[deltas >= 0] = 2 * deltas[deltas >= 0]
    unsigned[deltas < 0] = -2 * deltas[deltas < 0] - 1
    return unsigned


def create_universal_delta_codebook(num_primary=4):
    """
    Creates a single, universal P/S codebook for mapped deltas.
    The primary symbols are always the smallest unsigned values,
    representing the most common deltas (0, -1, 1, -2).
    """
    if num_primary != 4:
        raise NotImplementedError(
            "This implementation is hardcoded for 4 primary symbols."
        )

    # Mapped values: 0 (delta 0), 1 (delta -1), 2 (delta 1), 3 (delta -2)
    primary_symbols = [0, 1, 2, 3]

    codebook = {
        primary_symbols[0]: "0",
        primary_symbols[1]: "10",
        primary_symbols[2]: "110",
        primary_symbols[3]: "1110",
    }
    escape_code = "1111"

    return codebook, escape_code


def encode_group(grp_values, codebook, escape_code, original_n_bits):
    """Calculates the bits to encode a group of (already mapped) values."""
    total_bits = 0
    for value in grp_values:
        if value in codebook:
            total_bits += len(codebook[value])
        else:
            # For deltas, the original number of bits is still relevant for the raw value
            total_bits += len(escape_code) + original_n_bits
    return total_bits


def simulate_delta_encoding(layer_path, index):
    """
    Simulates the DELTA encoding scheme and reports the final compression results.
    """
    # --- 1. Configuration ---
    GROUP_SIZE = 128
    N_BITS = 4
    NUM_PRIMARY_SYMBOLS = 4
    # With delta coding, we only need two states: compress or fallback.
    FLAG_BITS = 1

    # --- 2. Load Data and Create the Universal Codebook ---
    weight, _, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\n{'='*80}\nSimulating DELTA CODING for Layer: {name}\n{'='*80}")

    quantized, zero_points, _ = real_quantize_tensor(
        weight, zero_point=True, group_size=GROUP_SIZE, return_scale=True
    )

    q_flat = quantized.flatten().cpu().numpy()
    zps = zero_points.cpu().numpy()

    n_total = q_flat.size
    n_group = zps.size

    # Create the single, fixed codebook that will be "hardwired"
    universal_codebook, escape_code = create_universal_delta_codebook(
        NUM_PRIMARY_SYMBOLS
    )

    # --- 3. Main Simulation Loop ---
    total_original_bits = n_total * N_BITS
    total_compressed_bits = 0
    fallback_groups_count = 0
    original_group_size_bits = GROUP_SIZE * N_BITS

    for g in range(n_group):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, n_total)
        grp = q_flat[start:end]
        zp_val = zps[g]

        # NEW: Transform data to deltas and map to unsigned
        deltas = grp.astype(np.int16) - zp_val
        mapped_deltas = map_signed_to_unsigned(deltas)

        # Encode the mapped deltas using the universal codebook
        compressed_payload_bits = encode_group(
            mapped_deltas, universal_codebook, escape_code, N_BITS
        )
        size_with_overhead = compressed_payload_bits + FLAG_BITS

        # Apply Fallback Mechanism
        final_group_bits = min(size_with_overhead, original_group_size_bits)
        if final_group_bits == original_group_size_bits:
            fallback_groups_count += 1

        total_compressed_bits += final_group_bits

    # --- 4. Report Final Metrics ---
    compression_ratio = total_original_bits / total_compressed_bits
    avg_bits_per_weight = total_compressed_bits / n_total

    print("--- Simulation Parameters ---")
    print(f"Encoding Scheme: DELTA CODING")
    print(f"Group Size: {GROUP_SIZE}")
    print(f"Primary Symbols in Universal Codebook: {NUM_PRIMARY_SYMBOLS}")
    print(f"Flag Bits per Group: {FLAG_BITS}")
    print("-" * 40)
    print("--- Compression Results ---")
    print(f"Total Original Bits: {total_original_bits:,}")
    print(f"Total Compressed Bits: {total_compressed_bits:,.0f}")
    print(
        f"Groups Using Fallback: {fallback_groups_count} / {n_group} ({fallback_groups_count/n_group:.2%})"
    )
    print("\n--- Final Performance Metrics ---")
    print(f"  📊 Compression Ratio: {compression_ratio:.2f}x")
    print(f"  📉 Average Bits Per Weight: {avg_bits_per_weight:.2f} bits")


def map_signed_to_unsigned(deltas):
    unsigned = np.zeros_like(deltas)
    unsigned[deltas >= 0] = 2 * deltas[deltas >= 0]
    unsigned[deltas < 0] = -2 * deltas[deltas < 0] - 1
    return unsigned


def map_unsigned_to_signed(unsigned_deltas):
    """The inverse mapping function."""
    signed = np.zeros_like(unsigned_deltas, dtype=np.int16)
    is_even = unsigned_deltas % 2 == 0
    signed[is_even] = unsigned_deltas[is_even] // 2
    signed[~is_even] = -(unsigned_deltas[~is_even] + 1) // 2
    return signed


def analyze_delta_distribution(layer_path, index):
    """
    Aggregates all deltas from a layer and plots their distribution.
    """
    GROUP_SIZE = 128

    weight, _, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\n{'='*80}\nAnalyzing DELTA DISTRIBUTION for Layer: {name}\n{'='*80}")

    quantized, zero_points, _ = real_quantize_tensor(
        weight, zero_point=True, group_size=GROUP_SIZE, return_scale=True
    )

    q_flat = quantized.flatten().cpu().numpy()
    zps = zero_points.cpu().numpy()
    n_group = zps.size

    all_mapped_deltas = []

    for g in range(n_group):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, q_flat.size)
        grp = q_flat[start:end]
        zp_val = zps[g]

        deltas = grp.astype(np.int16) - zp_val
        mapped_deltas = map_signed_to_unsigned(deltas)
        all_mapped_deltas.extend(mapped_deltas)

    # --- Plotting the Histogram ---
    all_mapped_deltas = np.array(all_mapped_deltas)

    # Get frequency counts
    values, counts = np.unique(all_mapped_deltas, return_counts=True)
    percentages = 100 * counts / len(all_mapped_deltas)

    print("Mapped Delta | Original Delta | Percentage")
    print("-------------------------------------------")
    # Mapped values: 0 (delta 0), 1 (delta -1), 2 (delta 1), 3 (delta -2), 4 (delta 2)...
    original_deltas_str = ["0", "-1", "1", "-2", "2", "-3", "3"]
    for i in range(min(len(values), 7)):
        val = values[i]
        pct = percentages[i]
        orig_delta = original_deltas_str[i] if i < len(original_deltas_str) else "..."
        print(f"    {val:<8} |      {orig_delta:<8} | {pct:.2f}%")

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(values, percentages, width=0.8)
    plt.title(f"Distribution of Mapped Deltas for Layer: {name}")
    plt.xlabel("Mapped Delta Value (0->0, 1->-1, 2->1, ...)")
    plt.ylabel("Percentage of Occurrences (%)")
    plt.xticks(np.arange(0, 20, 1))
    plt.grid(axis="y", linestyle="--")

    # Save the plot to a file
    filename = f"delta_distribution_{name}.png"
    plt.savefig(filename)
    print(f"\nDistribution plot saved to {filename}")


def golomb_rice_encode_bits(value, k):
    q = value >> k
    return (q + 1) + k


def golomb_rice_encoder(value, k):
    """Encodes a single value and returns the bitstring."""
    q = value >> k
    r = value & ((1 << k) - 1)
    # Unary for q, binary for r
    return "1" * q + "0" + format(r, f"0{k}b")


def find_optimal_k_and_encode(grp_values, k_options):
    """Finds the best k and returns the full compressed bitstream for the group."""
    best_k = -1
    best_bitstream = ""
    min_bits = float("inf")

    for k in k_options:
        # Generate the full bitstream for the current k
        current_bitstream = "".join([golomb_rice_encoder(v, k) for v in grp_values])
        current_bits = len(current_bitstream)

        # If this k is better, save its results
        if current_bits < min_bits:
            min_bits = current_bits
            best_k = k
            best_bitstream = current_bitstream

    # Return the actual bitstream string, not its length
    return best_bitstream, best_k


def simulate_golomb_rice_encoding(layer_path, index):
    # --- 1. Configuration ---
    GROUP_SIZE = 128
    N_BITS = 4
    K_OPTIONS = [1, 2, 3]

    # --- 2. Calculate Metadata Overhead ---
    num_k_choices = len(K_OPTIONS)
    FLAG_BITS = ceil(log2(num_k_choices + 1))

    # --- 3. Load Data ---
    weight, _, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(
        f"\n{'='*80}\nSimulating GOLOMB-RICE CODING for Layer: {name} (Corrected)\n{'='*80}"
    )

    quantized, zero_points, _ = real_quantize_tensor(
        weight, zero_point=True, group_size=GROUP_SIZE, return_scale=True
    )

    q_flat = quantized.flatten().cpu().numpy()
    zps = zero_points.cpu().numpy()
    n_total = q_flat.size
    n_group = zps.size

    # --- 4. Main Simulation Loop ---
    total_original_bits = n_total * N_BITS
    total_compressed_bits = 0
    fallback_groups_count = 0
    k_usage_counts = {k: 0 for k in K_OPTIONS}
    original_group_size_bits = GROUP_SIZE * N_BITS

    for g in range(n_group):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, n_total)
        grp = q_flat[start:end]
        zp_val = zps[g]

        deltas = grp.astype(np.int16) - zp_val
        mapped_deltas = map_signed_to_unsigned(deltas)

        compressed_payload_bits, best_k = find_optimal_k_and_encode(
            mapped_deltas, K_OPTIONS
        )
        size_with_overhead = len(compressed_payload_bits) + FLAG_BITS

        final_group_bits = min(size_with_overhead, original_group_size_bits)
        if final_group_bits == original_group_size_bits:
            fallback_groups_count += 1
        else:
            k_usage_counts[best_k] += 1

        total_compressed_bits += final_group_bits

    # --- 5. Report Final Metrics ---
    compression_ratio = total_original_bits / total_compressed_bits
    avg_bits_per_weight = total_compressed_bits / n_total

    print("--- Simulation Parameters ---")
    print(f"Encoding Scheme: Adaptive Golomb-Rice")
    print(f"Group Size: {GROUP_SIZE}")
    print(f"Tested k options: {K_OPTIONS}")
    print(f"Flag Bits per Group: {FLAG_BITS} (to signal k or fallback)")
    print("-" * 40)
    print("--- Compression Results ---")
    print(f"Total Original Bits: {total_original_bits:,}")
    print(f"Total Compressed Bits: {total_compressed_bits:,.0f}")
    print(
        f"Groups Using Fallback: {fallback_groups_count} / {n_group} ({fallback_groups_count/n_group:.2%})"
    )
    print("Usage of k values (for compressed groups):")
    for k, count in k_usage_counts.items():
        print(f"  k={k}: {count} groups")
    print("\n--- Final Performance Metrics ---")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Average Bits Per Weight: {avg_bits_per_weight:.2f} bits")


def golomb_rice_decoder(bitstream, k, num_values_to_decode):
    """Decodes a bitstream using a given k."""
    decoded_values = []
    idx = 0
    for _ in range(num_values_to_decode):
        # 1. Decode Quotient (q)
        q = 0
        while bitstream[idx] == "1":
            q += 1
            idx += 1
        idx += 1  # Skip the '0' terminator

        # 2. Decode Remainder (r)
        remainder_bits = bitstream[idx : idx + k]
        r = int(remainder_bits, 2)
        idx += k

        # 3. Reconstruct Value
        value = (q << k) + r
        decoded_values.append(value)

    return np.array(decoded_values, dtype=np.uint16)


def verify_lossless_compression(layer_path, index):
    """
    Performs end-to-end encoding and decoding to verify a lossless process.
    """
    # --- 1. Configuration ---
    GROUP_SIZE = 128
    K_OPTIONS = [1, 2, 3]

    # --- 2. Load and Quantize Data ---
    weight, _, info = load_saved_layer(layer_path, index)
    name = info["layer_name"]
    print(f"\n{'='*80}\nVerifying Lossless Compression for Layer: {name}\n{'='*80}")

    quantized, zero_points, _ = real_quantize_tensor(
        weight, zero_point=True, group_size=GROUP_SIZE, return_scale=True
    )

    original_q_flat = quantized.flatten().cpu().numpy()
    zps = zero_points.cpu().numpy()
    n_group = zps.size
    reconstructed_q_flat = np.zeros_like(original_q_flat)

    # --- 3. Loop Through Each Group for End-to-End Process ---
    for g in range(n_group):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, original_q_flat.size)
        original_group = original_q_flat[start:end]
        zp_val = zps[g]

        # --- ENCODING STEPS ---
        # a. Create deltas and map them
        deltas = original_group.astype(np.int16) - zp_val
        mapped_deltas = map_signed_to_unsigned(deltas)
        # b. Find best k and generate the bitstream
        compressed_bitstream, best_k = find_optimal_k_and_encode(
            mapped_deltas, K_OPTIONS
        )

        # --- DECODING STEPS ---
        # a. Decode the bitstream to get mapped deltas
        decoded_mapped_deltas = golomb_rice_decoder(
            compressed_bitstream, best_k, len(original_group)
        )
        # b. Inverse map to get signed deltas
        decoded_signed_deltas = map_unsigned_to_signed(decoded_mapped_deltas)
        # c. Add back the zero-point to reconstruct the group
        reconstructed_group = decoded_signed_deltas + zp_val

        # Store the reconstructed group
        reconstructed_q_flat[start:end] = reconstructed_group

    # --- 4. Final Verification ---
    print("Comparing original quantized weights with the reconstructed weights...")

    # Use np.array_equal for a definitive, element-wise check
    are_equal = np.array_equal(original_q_flat, reconstructed_q_flat)

    if are_equal:
        print("\nVerification Successful!")
        print(
            "The reconstructed weights are identical to the original quantized weights."
        )
    else:
        print("\nVerification Failed!")
        print("The reconstructed weights DO NOT match the original quantized weights.")
        # Optional: Print out the first differing elements for debugging
        diff_indices = np.where(original_q_flat != reconstructed_q_flat)[0]
        first_diff = diff_indices[0]
        print(f"First mismatch found at index {first_diff}:")
        print(f"  Original value: {original_q_flat[first_diff]}")
        print(f"  Reconstructed value: {reconstructed_q_flat[first_diff]}")


if __name__ == "__main__":
    layer_path = "output_weights/facebook_opt-125m_layers/"
    # layer_path = "output_weights/EleutherAI_gpt-neo-2.7B_layers/"
    # layer_path = "output_weights/facebook_opt-1.3b_layers/"

    for index in range(0, 10):
        # results = get_hist_per_layer(layer_path, index)
        # results = analyze_compression_potential(layer_path, index)
        # simulate_delta_encoding(layer_path, index)
        # analyze_delta_distribution(layer_path, index)
        simulate_golomb_rice_encoding(layer_path, index)
        # verify_lossless_compression(layer_path, index)
