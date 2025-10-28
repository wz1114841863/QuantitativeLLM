import numpy as np
import os
from math import ceil, log2
from tqdm import tqdm

"""
文件说明:
    用于验证压缩后的模型权重的正确性
    解码, 并与原始量化权重对比
"""

K_OPTIONS = [1, 2, 3]


class BitstreamReader:
    def __init__(self, byte_array, total_valid_bits):
        self.byte_array = byte_array
        self.total_valid_bits = total_valid_bits
        self.bits_read = 0
        self.byte_idx = 0
        self.bit_idx = 0

    def read(self, n_bits):
        if self.bits_read + n_bits > self.total_valid_bits:
            raise EOFError("Attempted to read past the end of the valid bitstream.")
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
        return np.array([self.read(4) for _ in range(n_bits // 4)])


def map_unsigned_to_signed(unsigned_deltas):
    signed = np.zeros_like(unsigned_deltas, dtype=np.int16)
    is_even = unsigned_deltas % 2 == 0
    signed[is_even] = unsigned_deltas[is_even] // 2
    signed[~is_even] = -(unsigned_deltas[~is_even] + 1) // 2
    return signed


def golomb_rice_decoder(reader, k, num_values_to_decode):
    """
    Decodes a series of values using Golomb-Rice coding with enhanced boundary checks.
    """
    decoded_values = []
    for i in range(num_values_to_decode):
        # --- Decode Quotient (q) ---
        q = 0
        while True:
            # 关键修正 1: 在读取商的每一位前,都检查是否已到达流的末尾
            if reader.bits_read >= reader.total_valid_bits:
                raise EOFError(
                    f"Stream ended unexpectedly while decoding quotient for value {i+1}/{num_values_to_decode}."
                )

            bit = reader.read(1)
            if bit == 0:
                break  # Found the '0' terminator
            q += 1

        # --- Decode Remainder (r) ---
        # 关键修正 2: 在读取余数前,检查是否有足够的位
        if reader.bits_read + k > reader.total_valid_bits:
            raise EOFError(
                f"Not enough bits for remainder for value {i+1}/{num_values_to_decode}. "
                f"Need {k}, have {reader.total_valid_bits - reader.bits_read}."
            )

        r = reader.read(k)
        value = (q << k) + r
        decoded_values.append(value)

    return np.array(decoded_values, dtype=np.uint16)


def verify_layer(layer_file_path):
    print(f"\n--- Verifying layer file: {os.path.basename(layer_file_path)} ---")
    data = np.load(layer_file_path)
    packed_bytes, zero_points, original_quantized_weights = (
        data["compressed_weights"],
        data["zero_points"],
        data["original_quantized"],
    )
    GROUP_SIZE, total_valid_bits = (
        data["group_size"].item(),
        data["total_valid_bits"].item(),
    )
    print(f"Detected Group Size: {GROUP_SIZE}, Total Valid Bits: {total_valid_bits}")

    reader = BitstreamReader(packed_bytes, total_valid_bits)
    num_groups = len(zero_points)
    reconstructed_weights = np.zeros_like(original_quantized_weights)

    num_k_choices = len(K_OPTIONS)
    flag_bits = ceil(log2(num_k_choices + 1))
    flag_to_k = {i: k for i, k in enumerate(K_OPTIONS)}
    fallback_flag_val = num_k_choices

    for g in tqdm(range(num_groups), desc="Decoding Groups"):
        flag_val = reader.read(flag_bits)
        start_idx = g * GROUP_SIZE
        end_idx = min(start_idx + GROUP_SIZE, len(original_quantized_weights))
        current_group_size = end_idx - start_idx

        if flag_val == fallback_flag_val:
            reconstructed_group = reader.read_raw(current_group_size * 4)
        else:
            k = flag_to_k[flag_val]
            decoded_mapped_deltas = golomb_rice_decoder(reader, k, current_group_size)
            decoded_signed_deltas = map_unsigned_to_signed(decoded_mapped_deltas)
            reconstructed_group = decoded_signed_deltas + zero_points[g]

        reconstructed_weights[start_idx:end_idx] = reconstructed_group

    if np.array_equal(original_quantized_weights, reconstructed_weights):
        print("Verification Successful! Reconstructed weights are identical.")
    else:
        print("Verification FAILED! Data mismatch occurred.")


if __name__ == "__main__":
    GROUP_SIZE = 512
    # output_dir = f"./compressed_opt-125m_gs{GROUP_SIZE}"
    output_dir = f"./compressed_facebook_opt-350m_gs{GROUP_SIZE}"
    if not os.path.exists(output_dir):
        print(
            f"Error: Directory '{output_dir}' not found. Please run 'compress_model.py' first."
        )
    else:
        files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".npz")
        ]
        if files:
            largest_file = max(files, key=os.path.getsize)
            verify_layer(largest_file)
        else:
            print("No .npz files found in the directory.")
