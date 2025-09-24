import torch
import numpy as np


def select_min_dtype(indices):
    """根据索引值范围选择最小位宽的数据类型"""
    max_val = indices.max()

    if max_val <= 127:  # 7-bit, 用int8存储
        return indices.astype(np.int8)
    elif max_val <= 255:  # 8-bit
        return indices.astype(np.uint8)
    elif max_val <= 32767:  # 15-bit, 用int16存储
        return indices.astype(np.int16)
    elif max_val <= 65535:  # 16-bit
        return indices.astype(np.uint16)
    else:
        return indices.astype(np.int32)


def reorder_tile(w, tile_size=128):
    """Reorder the tensor in tiles of specified size."""
    assert w.numel() % tile_size == 0, "Weight size must be divisible by tile size."
    original_shape = w.shape
    original_type = w.dtype
    flat = w.flatten()
    num_groups = flat.numel() // tile_size
    groups = flat.reshape(num_groups, tile_size)

    reordered_groups = []
    reverse_indices = []

    for i in range(num_groups):
        group = groups[i]

        # 使用numpy进行高效排序(torch的argsort在某些版本有bug)
        group_np = group.numpy()
        idx = np.argsort(np.abs(group_np))  # 按绝对值排序

        # 重排权重
        w_reord = group_np[idx]
        reordered_groups.append(w_reord)

        # 生成反向索引并选择最小数据类型
        rev_idx = np.argsort(idx)
        reverse_indices.append(select_min_dtype(rev_idx))

    # 合并结果
    w_reordered = torch.tensor(np.concatenate(reordered_groups), dtype=original_type)
    w_reordered = w_reordered.reshape(original_shape)

    return w_reordered, reverse_indices
