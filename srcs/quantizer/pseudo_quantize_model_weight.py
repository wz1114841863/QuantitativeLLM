import torch


def pseudo_quantize_to_4bit(tensor):
    """Zero-point quantization to 4 bits.
    e.g.:
        tmp = torch.randint(0, 256, (1000,))
        q, s, z = quantize_to_4bit(tmp.float())
        print(q, s, z)
    """
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 15
    zero_point = torch.round(-min_val / scale).clamp(0, 15)
    quantized = torch.round(tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
    return quantized, scale.item(), zero_point.item()


def pseudo_group_quantize_to_4bit(tensor, group_size=128):
    """
    伪量化, 分组量化到4位
    Args:
        tensor: 输入张量
        group_size: 分组大小
    Returns:
        quantized: 量化后的张量
        scales: 每组对应的scale值
        zero_points: 每组对应的zero_point值
    """
    original_shape = tensor.shape
    flattened = tensor.flatten()
    num_groups = (flattened.numel() + group_size - 1) // group_size  # 向上取整

    quantized = torch.zeros_like(flattened, dtype=torch.uint8)
    scales = torch.zeros(num_groups)
    zero_points = torch.zeros(num_groups, dtype=torch.uint8)

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, flattened.numel())
        group_tensor = flattened[start_idx:end_idx]

        # 对每个组进行量化
        min_val = group_tensor.min()
        max_val = group_tensor.max()

        # 处理全零或常数值的情况
        if max_val - min_val < 1e-6:
            scale = 1.0  # 避免除以零
            zero_point = 0
        else:
            scale = (max_val - min_val) / 15
            zero_point = torch.round(-min_val / scale).clamp(0, 15)

        # 量化
        group_quantized = (
            torch.round(group_tensor / scale + zero_point).clamp(0, 15).to(torch.uint8)
        )

        # 存储结果
        quantized[start_idx:end_idx] = group_quantized
        scales[i] = scale
        zero_points[i] = zero_point

    return quantized.reshape(original_shape), scales, zero_points
