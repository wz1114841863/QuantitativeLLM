import torch

from tqdm import tqdm
from typing import Optional
from quantizer.pre_quant import get_blocks, get_named_linears


def pseudo_quantize_model_weight(
    model, w_bit: int = 4, zero_point: bool = False, group_size=None
):
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, w_bit, zero_point, group_size
            )
    return model


@torch.no_grad()
def pseudo_quantize_tensor(
    weight, wq_bits: int = 4, zero_point: bool = False, group_size: Optional[int] = None
):
    if zero_point:
        return pseudo_zero_point_quant(weight, wq_bits, group_size)
    else:
        return pseudo_symm_quant(weight, wq_bits, group_size)


@torch.no_grad()
def pseudo_symm_quant(w_fp16, wq_bits: int = 4, group_size: Optional[int] = None):
    """Symmetric INT quantization, pseudo-quantization."""
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16
    else:
        K, C = w_fp16.size()
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)

    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = 2 ** (wq_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-3, max=1e4)
    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp), min=qmin, max=qmax)

    w_fp16_new = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def pseudo_zero_point_quant(w_fp16, wq_bits: int = 4, group_size: Optional[int] = None):
    """Zero-point quantization, pseudo-quantization."""
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16
    else:
        K, C = w_fp16.size()
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)

    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = 0
    qmax = 2**wq_bits - 1
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-3, max=1e4)
    zero_point = torch.clamp(torch.round(-rmin / scale_fp), min=qmin, max=qmax)
    q_tensor = torch.clamp(
        torch.round(w_fp16_new / scale_fp + zero_point), min=qmin, max=qmax
    )

    w_fp16_new = (q_tensor - zero_point) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)
