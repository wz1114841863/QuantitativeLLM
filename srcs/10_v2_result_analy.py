import numpy as np
import os
import sys
from math import ceil, log2
from tqdm import tqdm
from collections import Counter

K_OPTIONS = [1, 2]
# (Q7) 定义硬件 "快车道" 的阈值
FAST_PATH_MAX_BITS = 6


class BitstreamReader:
    """
    BitstreamReader 现在从字节数组 *和* 字节偏移量 开始工作
    """

    def __init__(self, byte_array, start_byte_offset):
        self.byte_array = byte_array
        self.start_byte = start_byte_offset
        self.bits_read_total = 0
        self.byte_idx = self.start_byte
        self.bit_idx = 0

    def read(self, n_bits):
        result = 0
        for _ in range(n_bits):
            if self.byte_idx >= len(self.byte_array):
                raise EOFError(
                    f"Attempted to read past the end of the *entire* byte array."
                )

            byte = self.byte_array[self.byte_idx]
            bit = (byte >> (7 - self.bit_idx)) & 1
            result = (result << 1) | bit
            self.bit_idx += 1
            if self.bit_idx == 8:
                self.bit_idx = 0
                self.byte_idx += 1

        self.bits_read_total += n_bits
        return result

    def read_raw(self, n_bits):
        num_vals = n_bits // 4
        for _ in range(num_vals):
            self.read(4)


def golomb_rice_decoder(reader, k, num_values_to_decode):
    """
    解码器返回每个权重的比特数
    """
    decoded_values = []
    per_weight_bit_counts = []

    for i in range(num_values_to_decode):
        bits_for_this_weight = 0
        q = 0
        while True:
            try:
                bit = reader.read(1)
                bits_for_this_weight += 1
                if bit == 0:
                    break
                q += 1
            except EOFError:
                raise EOFError(
                    f"Stream ended mid-group at weight {i+1}/{num_values_to_decode}"
                )

        r = reader.read(k)
        bits_for_this_weight += k

        value = (q << k) + r
        decoded_values.append(value)
        per_weight_bit_counts.append(bits_for_this_weight)

    return np.array(decoded_values, dtype=np.uint16), per_weight_bit_counts


def analyze_layer_final(layer_file_path):
    """
    分析 .npz 文件,并收集 *所有* 统计数据
    (Q0, Q1/Q2, Q5/Q6, Q7)
    """
    try:
        data = np.load(layer_file_path)
    except Exception as e:
        print(f"Error loading {layer_file_path}: {e}")
        return None

    # (Q0) 压缩性能统计
    original_size_bits = data["original_quantized"].size * 4
    compressed_size_bits = data["compressed_weights"].nbytes * 8
    metadata_size_bits = data["metadata"].nbytes * 8 + data["scales"].nbytes * 8

    # 数据加载
    packed_bytes = data["compressed_weights"]
    metadata = data["metadata"]
    original_quantized_weights = data["original_quantized"]
    GROUP_SIZE = data["group_size"].item()

    num_groups = len(metadata)

    # 解码常量
    num_k_choices = len(K_OPTIONS)
    flag_bits = ceil(log2(num_k_choices + 1))
    flag_to_k = {i: k for i, k in enumerate(K_OPTIONS)}
    fallback_flag_val = num_k_choices

    # --- 统计收集器 ---
    gr_groups = 0
    fallback_groups = 0
    per_k_stall_counts = {k: 0 for k in K_OPTIONS}
    total_weights_per_k = {k: 0 for k in K_OPTIONS}
    per_k_weight_bit_counts = {k: Counter() for k in K_OPTIONS}

    for g in range(num_groups):
        start_idx = g * GROUP_SIZE
        end_idx = min(start_idx + GROUP_SIZE, len(original_quantized_weights))
        current_group_size = end_idx - start_idx

        if current_group_size == 0:
            continue

        group_meta = metadata[g]
        start_byte = group_meta["offset"]

        reader = BitstreamReader(packed_bytes, start_byte)

        try:
            flag_val = reader.read(flag_bits)
        except EOFError:
            break

        if flag_val == fallback_flag_val:
            fallback_groups += 1
            try:
                reader.read_raw(current_group_size * 4)
            except EOFError:
                break

        else:
            gr_groups += 1
            k = flag_to_k[flag_val]
            try:
                decoded_mapped_deltas, per_weight_bit_counts_list = golomb_rice_decoder(
                    reader, k, current_group_size
                )

                total_weights_per_k[k] += current_group_size

                for bits in per_weight_bit_counts_list:
                    # (Q7)
                    if bits > FAST_PATH_MAX_BITS:
                        per_k_stall_counts[k] += 1

                    # (Q5/Q6)
                    per_k_weight_bit_counts[k].update([bits])

            except EOFError:
                break
            except Exception as e:
                print(f"\nError in {os.path.basename(layer_file_path)}, group {g}: {e}")
                break

    return (
        # (Q0)
        original_size_bits,
        compressed_size_bits,
        metadata_size_bits,
        # (Q1/Q2)
        gr_groups,
        fallback_groups,
        # (Q7)
        per_k_stall_counts,
        total_weights_per_k,
        # (Q5/Q6)
        per_k_weight_bit_counts,
    )


def print_and_save_summary_report(
    output_dir,
    # (Q0)
    g_total_original_bits,
    g_total_compressed_bits,
    g_total_metadata_bits,
    # (Q1/Q2)
    total_gr,
    total_fb,
    # (Q7)
    all_k_stall_counts,
    all_total_weights_k,
    # (Q5/Q6)
    all_k_bit_counts,
):
    """
    生成一个包含所有分析 (Q0, Q1/Q2, Q5/Q6, Q7) 的综合报告,
    并将其打印到屏幕和文件.
    """

    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append(f"--- Overall Model Analysis Report ---")
    report_lines.append(f"--- Analysis Target: {output_dir}")
    report_lines.append("=" * 60)

    # --- Q0: 压缩性能分析 ---
    report_lines.append(f"\n[Q0: Compression Performance ( Algorithm)]")

    g_total_final_bits = g_total_compressed_bits + g_total_metadata_bits
    g_total_weights = g_total_original_bits / 4.0  # N_BITS

    if g_total_final_bits > 0:
        compression_ratio = g_total_original_bits / g_total_final_bits
        avg_bits_per_weight = g_total_final_bits / g_total_weights
    else:
        compression_ratio = 0
        avg_bits_per_weight = 0

    report_lines.append(
        f"Total Original Size (4-bit weights): {g_total_original_bits/8/1024:,.2f} KB"
    )
    report_lines.append("-" * 40)
    report_lines.append(
        f"  - Compressed Weights Size:         {g_total_compressed_bits/8/1024:,.2f} KB"
    )
    report_lines.append(
        f"  - Metadata Size (ZPs, Indices, Scales): {g_total_metadata_bits/8/1024:,.2f} KB"
    )
    report_lines.append(
        f"Total Final Compressed Size:         {g_total_final_bits/8/1024:,.2f} KB"
    )
    report_lines.append("-" * 40)
    report_lines.append(
        f"Overall Compression Ratio:           {compression_ratio:.2f}x"
    )
    report_lines.append(
        f"Final Average Bits Per Weight:       {avg_bits_per_weight:.2f} bits"
    )

    # --- Q1 & Q2: Fallback 分析 ---
    total_groups = total_gr + total_fb
    if total_groups == 0:
        report_lines.append("\nNo groups were analyzed.")
        final_report_str = "\n".join(report_lines)
        print(final_report_str)
        return

    report_lines.append(f"\n[Q1 & Q2: Group Encoding Type Breakdown]")
    report_lines.append(f"Total Groups Analyzed: {total_groups}")
    report_lines.append(
        f"  - GR-Encoded Groups:   {total_gr} ({total_gr / total_groups:.2%})"
    )
    report_lines.append(
        f"  - Fallback Groups:     {total_fb} ({total_fb / total_groups:.2%})"
    )
    report_lines.append(
        f"  (Note: Fallback % is higher due to  'FALLBACK_SAVING_THRESHOLD=0.95')"
    )

    # --- Q5 & Q6: 每比特分布 ---
    report_lines.append(
        f"\n[Q5 & Q6: Per-Weight Bit Count Distribution (Broken down by k)]"
    )
    total_weights_all_k_Q5 = 0
    all_k_merged_counts = Counter()

    for k_val, counts in all_k_bit_counts.items():
        total_weights_for_k = sum(counts.values())
        total_weights_all_k_Q5 += total_weights_for_k
        all_k_merged_counts.update(counts)

        report_lines.append(
            f"\n--- Distribution for k={k_val} (Found {total_weights_for_k} weights) ---"
        )
        if total_weights_for_k == 0:
            report_lines.append("  No weights found for this k.")
            continue

        for bits, count in sorted(counts.items()):
            percentage = count / total_weights_for_k
            report_lines.append(
                f"  - {bits:<3} bits : {count:>12} weights ({percentage:7.2%})"
            )

    report_lines.append(f"\n--- Summary [Q5] (All k values merged) ---")
    if total_weights_all_k_Q5 > 0:
        report_lines.append(
            f"Analyzed {total_weights_all_k_Q5} individual weights from all GR groups:"
        )
        for bits, count in sorted(all_k_merged_counts.items()):
            percentage = count / total_weights_all_k_Q5
            report_lines.append(
                f"  - {bits:<3} bits : {count:>12} weights ({percentage:7.2%})"
            )
    else:
        report_lines.append("No GR-encoded weights found for Q5/Q6 analysis.")

    # --- Q7: 硬件停顿分析 ---
    report_lines.append(f"\n[Q7: Hardware Stall Analysis (Optimization 4)]")
    report_lines.append(
        f"(Simulating V9 Hardware 'Fast Path' with MAX_BITS = {FAST_PATH_MAX_BITS})"
    )

    total_stalls = 0
    total_weights_Q7 = 0

    for k_val, counts in all_k_stall_counts.items():
        total_k_weights = all_total_weights_k[k_val]
        total_stalls += counts
        total_weights_Q7 += total_k_weights

        if total_k_weights == 0:
            report_lines.append(f"\n--- Analysis for k={k_val} (Found 0 weights) ---")
            continue

        stall_percentage = counts / total_k_weights

        report_lines.append(
            f"\n--- Analysis for k={k_val} (Found {total_k_weights} weights) ---"
        )
        report_lines.append(
            f"  - 'Slow Path' Hits (> {FAST_PATH_MAX_BITS} bits): {counts} weights"
        )
        report_lines.append(f"  - Stall Rate (for k={k_val}): {stall_percentage: .4%}")

    if total_weights_Q7 > 0:
        overall_stall_rate = total_stalls / total_weights_Q7
        report_lines.append(f"\n--- Overall Stall Summary ---")
        report_lines.append(f"Total 'Slow Path' Hits (All k's): {total_stalls}")
        report_lines.append(f"Total GR-Encoded Weights Analyzed:  {total_weights_Q7}")
        report_lines.append(
            f"Overall Hardware Stall Rate:        {overall_stall_rate: .4%}"
        )
    else:
        report_lines.append("No GR-encoded weights found to analyze for stalls.")

    report_lines.append("\n" + "=" * 60)
    report_lines.append("Overall comprehensive analysis complete.")

    # --- 最终输出 ---
    final_report_str = "\n".join(report_lines)

    # 1. 打印到屏幕
    print(final_report_str)

    # 2. 保存到文件
    report_path = os.path.join(output_dir, "analysis_report_V9.3_final.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report_str)
        print(f"\nAnalysis report saved to: {report_path}")
    except Exception as e:
        print(f"\nError saving report to {report_path}: {e}")


if __name__ == "__main__":
    MODEL_ID_STR = "facebook_opt-125m"
    GROUP_SIZE_CONST = 512

    output_dir = f"./compressed_{MODEL_ID_STR}_gs{GROUP_SIZE_CONST}_V2_analysis"
    # 允许通过命令行参数覆盖
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' not found.")
        print("Please run 'compress_model_V9_analysis.py' first.")
        sys.exit(1)

    files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".npz")
    ]
    files.sort()

    # --- 全局累加器 ---
    g_total_original_bits = 0
    g_total_compressed_bits = 0
    g_total_metadata_bits = 0

    g_total_gr_groups = 0
    g_total_fallback_groups = 0

    g_total_per_k_stall_counts = {k: 0 for k in K_OPTIONS}
    g_total_weights_per_k = {k: 0 for k in K_OPTIONS}

    g_total_per_k_weight_bit_counts = {k: Counter() for k in K_OPTIONS}
    # ---

    print(
        f"Found {len(files)} .npz files. Starting comprehensive analysis on '{output_dir}'..."
    )

    for file_path in tqdm(files, desc="Analyzing Layers"):
        stats = analyze_layer_final(file_path)

        if stats:
            # 解包所有返回的统计数据
            (
                original_bits,
                comp_bits,
                meta_bits,
                gr_g,
                fb_g,
                k_stall_counts,
                k_total_weights,
                k_bit_counts,
            ) = stats

            # 累加 (Q0)
            g_total_original_bits += original_bits
            g_total_compressed_bits += comp_bits
            g_total_metadata_bits += meta_bits

            # 累加 (Q1/Q2)
            g_total_gr_groups += gr_g
            g_total_fallback_groups += fb_g

            # 累加 (Q7) & (Q5/Q6)
            for k_val in K_OPTIONS:
                g_total_per_k_stall_counts[k_val] += k_stall_counts[k_val]
                g_total_weights_per_k[k_val] += k_total_weights[k_val]
                g_total_per_k_weight_bit_counts[k_val].update(k_bit_counts[k_val])

    # 打印最终的综合报告
    print_and_save_summary_report(
        output_dir,
        g_total_original_bits,
        g_total_compressed_bits,
        g_total_metadata_bits,
        g_total_gr_groups,
        g_total_fallback_groups,
        g_total_per_k_stall_counts,
        g_total_weights_per_k,
        g_total_per_k_weight_bit_counts,
    )
