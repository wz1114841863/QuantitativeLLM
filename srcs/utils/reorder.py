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
    assert w.numel() % tile_size == 0, "Weight size must be divisible by tile size."
    assert tile_size <= 256, "tile_size > 256 needs wider index type"
    original_shape = w.shape
    flat = w.flatten()
    num_groups = flat.numel() // tile_size
    groups = flat.reshape(num_groups, tile_size)

    reordered_groups = []
    reverse_indices = []

    for group in groups:
        group_cpu = group.cpu().numpy()
        order = np.argsort(np.abs(group_cpu), kind="stable")
        reord = group_cpu[order]
        reordered_groups.append(reord.astype(group_cpu.dtype))  # 保持宽度

        rev = np.argsort(order, kind="stable")
        reverse_indices.append(select_min_dtype(rev))

    concat = np.concatenate(reordered_groups)
    # 零拷贝回到原设备
    w_reordered = torch.from_numpy(concat).to(device=w.device, dtype=w.dtype)
    return w_reordered.view(original_shape), reverse_indices


def reorder_tile_v2(w, tile_size=128):
    original_shape = w.shape
    flat = w.flatten()
    n = flat.numel()
    groups = flat.reshape(-1, tile_size)
    perm_list, reord_list = [], []
    for g in groups:
        tile_np = g.cpu().numpy()
        perm = np.argsort(np.abs(tile_np))  # 这才是"真"排序索引
        reord = tile_np[perm]
        perm_list.append(perm.astype(np.uint8))  # tile_size<=256
        reord_list.append(reord)
    # 拼回
    reord_all = torch.from_numpy(np.concatenate(reord_list)).to(w.device, w.dtype)
    perm_all = np.concatenate(perm_list)  # 1-D
    return reord_all.view(original_shape), np.split(
        perm_all, len(perm_all) // tile_size
    )
