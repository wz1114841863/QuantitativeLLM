import torch
from typing import Optional
from quantizer.pre_quant import get_blocks, get_named_linears


def real_quantize_tensor(
    tensor,
    zero_point: bool = False,
    group_size: Optional[int] = None,
    return_scale=False,
):
    if zero_point:
        if group_size is None:
            quantized, zero_point, scale = real_zero_point_quantize_to_4bit(tensor)
        else:
            quantized, zero_point, scale = group_zero_point_quantize_to_4bit(
                tensor, group_size
            )
        return quantized if not return_scale else (quantized, zero_point, scale)

    else:
        if group_size is None:
            quantized, scale = real_symm_quantize_to_4bit(tensor)
        else:
            quantized, scale = group_symm_quantize_to_4bit(tensor, group_size)

        return quantized if not return_scale else (quantized, scale)


@torch.no_grad()
def real_symm_quantize_to_4bit(tensor):
    """Symmetric INT quantization to 4 bits."""
    max_abs_val = torch.max(torch.abs(tensor))
    EPS = 1e-8
    scale = max_abs_val / 7.0 if max_abs_val.item() > EPS else 1.0

    quantized_int = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)
    return quantized_int, scale


@torch.no_grad()
def group_symm_quantize_to_4bit(tensor, group_size=128):
    """Symmetric grouped INT quantization to 4 bits."""
    original_shape = tensor.shape
    flattened = tensor.flatten()
    num_groups = (flattened.numel() + group_size - 1) // group_size

    quantized_data = torch.zeros_like(flattened, dtype=torch.int8)
    scales = torch.zeros(num_groups, device=tensor.device, dtype=tensor.dtype)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, flattened.numel())
        group_data = flattened[start_idx:end_idx]

        max_abs_val = torch.max(torch.abs(group_data))
        scale = max_abs_val / 7.0 if max_abs_val != 0 else 1.0

        quantized_group = torch.round(group_data / scale).clamp(-8, 7).to(torch.int8)
        quantized_data[start_idx:end_idx] = quantized_group
        scales[i] = scale

    quantized_tensor = quantized_data.reshape(original_shape)
    return quantized_tensor, scales


@torch.no_grad()
def real_zero_point_quantize_to_4bit(tensor):
    """Zero-point quantization to 4 bits."""
    if torch.all(tensor == 0):
        scale = 1.0
        zero_point = 0
        quantized = torch.zeros_like(tensor, dtype=torch.uint8)
        return quantized, scale, zero_point
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 15
    if scale == 0:
        scale = 1.0
    zero_point = torch.round(-min_val / scale).clamp(0, 15)
    quantized = torch.round(tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
    return quantized, zero_point, scale


@torch.no_grad()
def group_zero_point_quantize_to_4bit(tensor, group_size=128):
    """Zero point, Group quantization to 4 bits."""
    original_shape = tensor.shape
    device = tensor.device
    flattened = tensor.flatten()
    num_groups = (flattened.numel() + group_size - 1) // group_size

    quantized = torch.zeros_like(flattened, dtype=torch.uint8)
    scales = torch.zeros(num_groups, device=device)
    zero_points = torch.zeros(num_groups, dtype=torch.uint8)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, flattened.numel())
        group_tensor = flattened[start_idx:end_idx]

        min_val = group_tensor.min()
        max_val = group_tensor.max()

        if max_val - min_val < 1e-6:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / 15
            zero_point = torch.round(-min_val / scale).clamp(0, 15)

        group_quantized = (
            torch.round(group_tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
        )

        quantized[start_idx:end_idx] = group_quantized
        scales[i] = scale
        zero_points[i] = zero_point

    return quantized.reshape(original_shape), zero_points, scales
