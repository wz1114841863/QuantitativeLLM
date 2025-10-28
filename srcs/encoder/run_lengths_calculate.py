import torch
from collections import Counter


def compute_run_lengths(quantized_weights):
    """Compute run-length encoding for a 1D tensor."""
    runs = []  # List of (value, run_length)
    len_counter = Counter()  # Count of run lengths

    if quantized_weights.numel() == 0:
        return runs, len_counter

    if len(quantized_weights.shape) != 1:
        quantized_weights = quantized_weights.flatten()

    current_val = quantized_weights[0]
    count = 1
    for val in quantized_weights[1:]:
        if val == current_val:
            count += 1
        else:
            len_counter[int(count)] += 1
            runs.append((int(current_val), count))
            current_val = val
            count = 1
    runs.append((current_val.item(), count))
    len_counter[int(count)] += 1

    return runs, len_counter
