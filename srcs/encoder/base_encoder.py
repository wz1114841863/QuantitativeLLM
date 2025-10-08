from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Dict, Type

from srcs.quantizer.real_quantize import real_quantize_tensor
from srcs.encoder.chunk_vlc_encoder import *


@dataclass
class EncodedWeight:
    """
    任何编码器返回的容器.
    如果你需要更复杂的信息(如码表/scale/zero-point 等),
    可以继承此类进行扩展.
    """

    bits: float  # 总 bit 数

    def num_bits(self) -> float:
        return self.bits

    def num_bytes(self) -> float:
        return self.bits / 8


class BaseEncoder(ABC):
    """所有编码器的抽象基类"""

    @abstractmethod
    def encode(
        self, weight: Tensor, group_size: int, zero_point: bool
    ) -> EncodedWeight:
        raise NotImplementedError


class Int4Encoder(BaseEncoder):
    """最简单的 INT4 均匀量化，不做任何熵编码"""

    def encode(
        self, weight: Tensor, group_size: int, zero_point: bool
    ) -> EncodedWeight:
        q = real_quantize_tensor(weight, zero_point=zero_point, group_size=group_size)
        # INT4 每个元素 4 bit
        return EncodedWeight(bits=float(q.numel() * 4))


class ChunkVlcEncoder(BaseEncoder):
    """chunk_vlc_len"""

    def __init__(self, tile: int = 128):
        self.tile = tile

    def encode(
        self, weight: Tensor, group_size: int, zero_point: bool
    ) -> EncodedWeight:
        q = real_quantize_tensor(weight, zero_point=zero_point, group_size=group_size)
        avg_bits = chunk_vlc_len(q, tile=self.tile)  # 每个 weight 平均占多少 bit
        total_bits = float(q.numel() * avg_bits)
        return EncodedWeight(bits=total_bits)
