import numpy as np
import os
import sys
from math import ceil, log2
from tqdm import tqdm
from collections import Counter

K_OPTIONS = [1, 2, 3]
GROUP_SIZE_CONST = 512


class BitstreamReader:
    """Helper class to read bits from a byte array."""

    def __init__(self, byte_array, total_valid_bits):
        self.byte_array = byte_array
        self.total_valid_bits = total_valid_bits
        self.bits_read = 0
        self.byte_idx = 0
        self.bit_idx = 0

    def read(self, n_bits):
        """Reads n_bits from the bitstream."""
        if self.bits_read + n_bits > self.total_valid_bits:
            raise EOFError(
                f"Attempted to read past the end of the valid bitstream. Requested {n_bits}, remaining {self.total_valid_bits - self.bits_read}"
            )
        result = 0
        for _ in range(n_bits):
            byte = self.byte_array[self.byte_idx]
            bit = (byte >> (7 - self.bit_idx)) & 1
            result = (result << 1) | bit
            self.bit_idx += 1
            if self.bit_idx == 8:
                self.bit_idx = 0
                self.byte_idx += 1
        self.bits_read += n_bits
        return result

    def read_raw(self, n_bits):
        """Reads raw 4-bit values."""
        # This just advances the reader, we don't need the values for analysis
        num_vals = n_bits // 4
        for _ in range(num_vals):
            self.read(4)


def map_unsigned_to_signed(unsigned_deltas):
    """Reverses the signed-to-unsigned mapping."""
    signed = np.zeros_like(unsigned_deltas, dtype=np.int16)
    is_even = unsigned_deltas % 2 == 0
    signed[is_even] = unsigned_deltas[is_even] // 2
    signed[~is_even] = -(unsigned_deltas[~is_even] + 1) // 2
    return signed


def golomb_rice_decoder(reader, k, num_values_to_decode):
    """Decodes values using Golomb-Rice."""
    decoded_values = []
    for i in range(num_values_to_decode):
        q = 0
        while True:
            if reader.bits_read >= reader.total_valid_bits:
                raise EOFError(f"Stream ended while decoding quotient.")
            bit = reader.read(1)
            if bit == 0:
                break
            q += 1

        if reader.bits_read + k > reader.total_valid_bits:
            raise EOFError(f"Not enough bits for remainder.")

        r = reader.read(k)
        value = (q << k) + r
        decoded_values.append(value)
    return np.array(decoded_values, dtype=np.uint16)


def analyze_layer(layer_file_path, is_print=False):
    """
    Analyzes a single compressed layer .npz file and prints statistics.
    """
    print(f"\n--- Analyzing layer file: {os.path.basename(layer_file_path)} ---")

    try:
        data = np.load(layer_file_path)
    except FileNotFoundError:
        print(f"Error: File not found: {layer_file_path}")
        return
    except Exception as e:
        print(f"Error loading {layer_file_path}: {e}")
        return

    # Load data from .npz
    packed_bytes = data["compressed_weights"]
    zero_points = data["zero_points"]
    original_quantized_weights = data["original_quantized"]
    GROUP_SIZE = data["group_size"].item()
    total_valid_bits = data["total_valid_bits"].item()

    print(f"Detected Group Size: {GROUP_SIZE}, Total Valid Bits: {total_valid_bits}")

    reader = BitstreamReader(packed_bytes, total_valid_bits)
    num_groups = len(zero_points)

    # Setup constants for decoding
    num_k_choices = len(K_OPTIONS)
    flag_bits = ceil(log2(num_k_choices + 1))
    flag_to_k = {i: k for i, k in enumerate(K_OPTIONS)}
    fallback_flag_val = num_k_choices

    # --- Statistics Collectors ---
    gr_groups = 0
    fallback_groups = 0
    gr_mapped_delta_ranges = []  # Stores (min, max) for each GR group
    gr_avg_bits_per_weight = []  # Stores (bits / group_size) for each GR group
    # ---

    for g in tqdm(range(num_groups), desc="Analyzing Groups"):
        start_idx = g * GROUP_SIZE
        end_idx = min(start_idx + GROUP_SIZE, len(original_quantized_weights))
        current_group_size = end_idx - start_idx

        if current_group_size == 0:
            continue

        bits_before_group = reader.bits_read

        try:
            flag_val = reader.read(flag_bits)
        except EOFError:
            print(
                f"Warning: Reached end of bitstream early at group {g}. Stopping analysis."
            )
            break

        if flag_val == fallback_flag_val:
            # This is a Fallback group
            fallback_groups += 1
            # We must still read the data to advance the bitstream reader
            try:
                reader.read_raw(current_group_size * 4)
            except EOFError:
                print(
                    f"Warning: Reached end of bitstream early while reading fallback group {g}."
                )
                break

        else:
            # This is a GR-Encoded group
            gr_groups += 1
            k = flag_to_k[flag_val]

            try:
                # Decode the deltas to get their range
                decoded_mapped_deltas = golomb_rice_decoder(
                    reader, k, current_group_size
                )

                # Q3: Store the range of mapped deltas
                gr_mapped_delta_ranges.append(
                    (decoded_mapped_deltas.min(), decoded_mapped_deltas.max())
                )

                # Q4: Calculate bits per weight
                bits_after_group = reader.bits_read
                bits_for_this_group_data = (
                    bits_after_group - bits_before_group
                ) - flag_bits
                avg_bits = bits_for_this_group_data / current_group_size
                gr_avg_bits_per_weight.append(avg_bits)

            except EOFError as e:
                print(f"\nError decoding GR group {g} (k={k}): {e}")
                # Can't continue as the bitstream is now misaligned
                break
            except Exception as e:
                print(f"\nUnexpected error in group {g}: {e}")
                break

    # --- Report Statistics ---
    if is_print:
        print("\n" + "=" * 50)
        print(f"--- Analysis Report for: {os.path.basename(layer_file_path)} ---")
        print("=" * 50)

        total_groups = gr_groups + fallback_groups
        if total_groups == 0:
            print("No groups were analyzed.")
            return

        print(f"\n[Q1 & Q2: Group Encoding Type Breakdown]")
        print(f"Total Groups Analyzed: {total_groups}")
        print(f"  - GR-Encoded Groups:   {gr_groups} ({gr_groups / total_groups:.2%})")
        print(
            f"  - Fallback Groups:     {fallback_groups} ({fallback_groups / total_groups:.2%})"
        )

        print("\n[Q3 & Q4: Statistics for GR-Encoded Groups]")
        if gr_groups > 0:
            # Q3: Range of mapped deltas
            overall_min_delta = min(r[0] for r in gr_mapped_delta_ranges)
            overall_max_delta = max(r[1] for r in gr_mapped_delta_ranges)
            print(
                f"Overall Range of mapped (unsigned) deltas: [{overall_min_delta}, {overall_max_delta}]"
            )

            # Q4: Distribution of average bits per weight
            bins = [0, 1, 2, 3, 4, np.inf]
            hist, bin_edges = np.histogram(gr_avg_bits_per_weight, bins=bins)

            print("\nDistribution of Average Bits per Weight (for GR data stream):")
            bin_labels = ["0-1 bits", "1-2 bits", "2-3 bits", "3-4 bits", "4+ bits"]

            for i in range(len(hist)):
                percentage = hist[i] / gr_groups
                print(
                    f"  - {bin_labels[i]:<10}: {hist[i]:>7} groups ({percentage:7.2%})"
                )
        else:
            print("No groups were GR-encoded, so no statistics are available.")

        print("\n" + "=" * 50)
        print("Analysis complete.")

    return gr_groups, fallback_groups, gr_mapped_delta_ranges, gr_avg_bits_per_weight


def print_summary_report(total_gr, total_fb, all_ranges, all_bits):
    """
    Prints the aggregated analysis report for the entire model.
    """
    print("\n" + "=" * 60)
    print(f"--- Overall Model Analysis Report ---")
    print("=" * 60)

    total_groups = total_gr + total_fb
    if total_groups == 0:
        print("No groups were analyzed.")
        return

    print(f"\n[Q1 & Q2: Group Encoding Type Breakdown (All Layers)]")
    print(f"Total Groups Analyzed: {total_groups}")
    print(f"  - GR-Encoded Groups:   {total_gr} ({total_gr / total_groups:.2%})")
    print(f"  - Fallback Groups:     {total_fb} ({total_fb / total_groups:.2%})")

    print("\n[Q3 & Q4: Statistics for ALL GR-Encoded Groups]")
    if total_gr > 0:
        # Q3: Range of mapped deltas
        overall_min_delta = min(r[0] for r in all_ranges)
        overall_max_delta = max(r[1] for r in all_ranges)
        print(
            f"Overall Range of mapped (unsigned) deltas: [{overall_min_delta}, {overall_max_delta}]"
        )

        # Q4: Distribution of average bits per weight
        bins = [0, 1, 2, 3, 4, np.inf]
        hist, bin_edges = np.histogram(all_bits, bins=bins)

        print("\nDistribution of Average Bits per Weight (for GR data stream):")
        bin_labels = ["0-1 bits", "1-2 bits", "2-3 bits", "3-4 bits", "4+ bits"]

        for i in range(len(hist)):
            percentage = hist[i] / total_gr
            print(f"  - {bin_labels[i]:<10}: {hist[i]:>9} groups ({percentage:7.2%})")
    else:
        print("No groups were GR-encoded, so no statistics are available.")

    print("\n" + "=" * 60)
    print("Overall analysis complete.")


def analyze_layer_per_bit(layer_file_path):
    """
    Analyzes a single compressed layer .npz file and RETURNS statistics.
    """
    try:
        data = np.load(layer_file_path)
    except Exception as e:
        print(f"Error loading {layer_file_path}: {e}")
        return None

    packed_bytes = data["compressed_weights"]
    zero_points = data["zero_points"]
    original_quantized_weights = data["original_quantized"]
    GROUP_SIZE = data["group_size"].item()
    total_valid_bits = data["total_valid_bits"].item()

    reader = BitstreamReader(packed_bytes, total_valid_bits)
    num_groups = len(zero_points)

    num_k_choices = len(K_OPTIONS)
    flag_bits = ceil(log2(num_k_choices + 1))
    flag_to_k = {i: k for i, k in enumerate(K_OPTIONS)}
    fallback_flag_val = num_k_choices

    # --- Statistics Collectors ---
    gr_groups = 0
    fallback_groups = 0
    gr_mapped_delta_ranges = []
    gr_avg_bits_per_weight = []
    # NEW: Dictionary of Counters, one for each k
    per_k_weight_bit_counts = {k: Counter() for k in K_OPTIONS}
    # ---

    for g in range(num_groups):
        start_idx = g * GROUP_SIZE
        end_idx = min(start_idx + GROUP_SIZE, len(original_quantized_weights))
        current_group_size = end_idx - start_idx

        if current_group_size == 0:
            continue

        bits_before_group = reader.bits_read

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
                decoded_mapped_deltas = golomb_rice_decoder(
                    reader, k, current_group_size
                )

                # Q3 stats
                gr_mapped_delta_ranges.append(
                    (decoded_mapped_deltas.min(), decoded_mapped_deltas.max())
                )

                # Q4 stats
                bits_after_group = reader.bits_read
                bits_for_this_group_data = (
                    bits_after_group - bits_before_group
                ) - flag_bits
                avg_bits = bits_for_this_group_data / current_group_size
                gr_avg_bits_per_weight.append(avg_bits)

                # NEW: Q6 stats - Calculate bits and store in the correct k counter
                for val in decoded_mapped_deltas.tolist():
                    q = val >> k
                    bits = q + 1 + k
                    per_k_weight_bit_counts[k].update([bits])

            except EOFError:
                break
            except Exception as e:
                print(f"\nError in {os.path.basename(layer_file_path)}, group {g}: {e}")
                break

    return (
        gr_groups,
        fallback_groups,
        gr_mapped_delta_ranges,
        gr_avg_bits_per_weight,
        per_k_weight_bit_counts,
    )


def print_summary_report_per_bit(
    total_gr, total_fb, all_ranges, all_bits, all_k_weight_counts
):
    """
    Prints the aggregated analysis report for the entire model.
    """
    print("\n" + "=" * 60)
    print(f"--- Overall Model Analysis Report ---")
    print("=" * 60)

    total_groups = total_gr + total_fb
    if total_groups == 0:
        print("No groups were analyzed.")
        return

    # --- Q1 & Q2 ---
    print(f"\n[Q1 & Q2: Group Encoding Type Breakdown (All Layers)]")
    print(f"Total Groups Analyzed: {total_groups}")
    print(f"  - GR-Encoded Groups:   {total_gr} ({total_gr / total_groups:.2%})")
    print(f"  - Fallback Groups:     {total_fb} ({total_fb / total_groups:.2%})")

    # --- Q3 & Q4 ---
    print("\n[Q3 & Q4: Statistics for ALL GR-Encoded Groups]")
    if total_gr > 0:
        # Q3: Range
        overall_min_delta = min(r[0] for r in all_ranges) if all_ranges else 0
        overall_max_delta = max(r[1] for r in all_ranges) if all_ranges else 0
        print(
            f"Overall Range of mapped (unsigned) deltas: [{overall_min_delta}, {overall_max_delta}]"
        )

        # Q4: Avg Bits
        bins = [0, 1, 2, 3, 4, np.inf]
        hist, bin_edges = np.histogram(all_bits, bins=bins)

        print("\nDistribution of Average Bits per Weight (per Group):")
        bin_labels = ["0-1 bits", "1-2 bits", "2-3 bits", "3-4 bits", "4+ bits"]

        for i in range(len(hist)):
            percentage = hist[i] / total_gr
            print(f"  - {bin_labels[i]:<10}: {hist[i]:>9} groups ({percentage:7.2%})")
    else:
        print("No groups were GR-encoded.")

    # --- NEW: Q6 (Q5 is now a part of Q6) ---
    print("\n[Q5 & Q6: Per-Weight Bit Count Distribution (Broken down by k)]")

    total_weights_all_k = 0
    all_k_merged_counts = Counter()

    for k_val, counts in all_k_weight_counts.items():
        total_weights_for_k = sum(counts.values())
        total_weights_all_k += total_weights_for_k
        all_k_merged_counts.update(counts)

        print(
            f"\n--- Distribution for k={k_val} (Found {total_weights_for_k} weights) ---"
        )
        if total_weights_for_k == 0:
            print("  No weights found for this k.")
            continue

        for bits, count in sorted(counts.items()):
            percentage = count / total_weights_for_k
            print(f"  - {bits:<3} bits : {count:>12} weights ({percentage:7.2%})")

    print(f"\n--- Summary [Q5] (All k values merged) ---")
    print(f"Analyzed {total_weights_all_k} individual weights from all GR groups:")
    for bits, count in sorted(all_k_merged_counts.items()):
        percentage = count / total_weights_all_k
        print(f"  - {bits:<3} bits : {count:>12} weights ({percentage:7.2%})")

    print("\n" + "=" * 60)
    print("Overall analysis complete.")


if __name__ == "__main__":
    # This must match the MODEL_ID and GROUP_SIZE from compress_model.py
    # MODEL_ID = "facebook/opt-1.3b"
    MODEL_ID_STR = "facebook_opt-125m"

    output_dir = f"./compressed_{MODEL_ID_STR}_gs{GROUP_SIZE_CONST}"

    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' not found.")
        print("Please run 'compress_model.py' first to generate the .npz files.")
        sys.exit(1)

    files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".npz")
    ]

    # if files:
    #     # Analyze the largest .npz file as requested ("an example layer")
    #     largest_file = max(files, key=os.path.getsize)
    #     analyze_layer(largest_file)
    # else:
    #     print(f"No .npz files found in '{output_dir}'.")

    # Sort files for consistent processing order
    files.sort()

    # --- Global Statistics Accumulators ---
    total_gr_groups = 0
    total_fallback_groups = 0
    all_mapped_delta_ranges = []
    all_avg_bits_per_weight = []
    # total_per_weight_counts = Counter()
    total_per_k_weight_counts = {k: Counter() for k in K_OPTIONS}
    # ---

    print(f"Found {len(files)} .npz files. Starting analysis...")

    for file_path in tqdm(files, desc="Analyzing Layers"):
        stats = analyze_layer_per_bit(file_path)

        if stats:
            gr_g, fb_g, ranges, bits, k_weight_counts = stats

            total_gr_groups += gr_g
            total_fallback_groups += fb_g
            all_mapped_delta_ranges.extend(ranges)
            all_avg_bits_per_weight.extend(bits)
            # total_per_weight_counts.update(weight_counts)
            for k_val in K_OPTIONS:
                total_per_k_weight_counts[k_val].update(k_weight_counts[k_val])

    # After processing all files, print the final summary
    print_summary_report_per_bit(
        total_gr_groups,
        total_fallback_groups,
        all_mapped_delta_ranges,
        all_avg_bits_per_weight,
        total_per_k_weight_counts,
    )
