import torch


@torch.no_grad()
def check_quant_fn(fn, tensor, **kw):
    """返回 误差/码字/clip 率"""
    device = tensor.device
    tensor = tensor.to(device)
    out = fn(tensor, **kw)

    in_ch = tensor.shape[1]
    group_size = kw.get("group_size", 128)
    groups_per_row = in_ch // group_size

    if isinstance(out, tuple) and len(out) == 3:  # zero-point
        q, zp, scale = out
        scale_2d = scale.view(-1, groups_per_row).repeat_interleave(group_size, dim=1)
        zp_2d = zp.view(-1, groups_per_row).repeat_interleave(group_size, dim=1)
        deq = (q.to(tensor.dtype) - zp_2d) * scale_2d
    else:  # symm
        q, scale = out
        scale_2d = scale.view(-1, groups_per_row).repeat_interleave(group_size, dim=1)
        deq = q.to(tensor.dtype) * scale_2d

    rmse = torch.sqrt(torch.mean((deq - tensor) ** 2)).item()
    clip = ((q == q.min()) | (q == q.max())).float().mean().item()
    codes = q.unique().numel()
    return rmse, clip, codes
