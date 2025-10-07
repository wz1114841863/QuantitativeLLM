import numpy as np
from collections import Counter
from typing import List, Dict, Tuple


# ERROR: 解码逻辑错误
class RangeCoder4Bit:
    def __init__(self, freq):
        if isinstance(freq, Counter):
            freq = [freq[i] for i in range(16)]
        total = sum(freq)
        self.cum = [0] * 17
        for i in range(16):
            self.cum[i + 1] = self.cum[i] + freq[i]
        self.total = self.cum[16]
        self.bits_out = []
        self.low = 0
        self.range = 0xFFFFFFFF

    # ---------- 编码 ----------
    def encode(self, syms):
        for s in syms:
            self._encode_symbol(s)
        self._flush()
        return self._bits_to_bytes()

    def _encode_symbol(self, s):
        l, h = self.cum[s], self.cum[s + 1]
        self.range //= self.total
        self.low += l * self.range
        self.range *= h - l
        while (self.low ^ (self.low + self.range)) >> 24 == 0:
            self.bits_out.append((self.low >> 24) & 1)
            self.range = (self.range << 8) - 1
            self.low = (self.low << 8) & 0xFFFFFFFF

    def _flush(self):
        # 再输出 5 位(32 位寄存器保证足够)而不是 4 位
        for i in range(5):
            self.bits_out.append((self.low >> 27) & 1)  # 27 而不是 24
            self.low = (self.low << 1) & 0xFFFFFFFF

    def _bits_to_bytes(self):
        pad = -len(self.bits_out) % 8
        self.bits_out += [0] * pad
        return bytearray(
            int("".join(map(str, self.bits_out[i : i + 8])), 2)
            for i in range(0, len(self.bits_out), 8)
        )

    # ---------- 解码 ----------
    def decode(self, byte_stream, n):
        bits = []
        for b in byte_stream:
            bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
        self.bits_in = bits
        self.idx = 0
        self.low = 0
        self.range = 0xFFFFFFFF
        value = 0
        for i in range(32):
            value = (value << 1) | (bits[i] if i < len(bits) else 0)
        self.value = value
        out = []
        for _ in range(n):
            out.append(self._decode_symbol())
        return np.array(out, dtype=np.uint8)

    def _decode_symbol(self):
        self.range //= self.total
        count = min((self.value - self.low) // self.range, self.total - 1)  # 关键修复
        s = 0
        while self.cum[s + 1] <= count:
            s += 1
        l, h = self.cum[s], self.cum[s + 1]
        self.low += l * self.range
        self.range *= h - l
        while (self.low ^ (self.low + self.range)) >> 24 == 0:
            self.low = (self.low << 8) & 0xFFFFFFFF
            self.range = (self.range << 8) - 1
            if self.idx + 8 < len(self.bits_in):
                self.value = (self.value << 8) | self.bits_in[self.idx + 8]
            self.idx += 8
        return s


class RangeCoder31:
    """31 符号 Range 编码器 / 解码器 [-16..15] → [0..31]"""

    def __init__(self, freq: Dict[int, int]):
        self.sym_cnt = np.array([freq.get(i, 1) for i in range(32)], dtype=np.int64)
        self.total = self.sym_cnt.sum()
        self.cum_freq = np.cumsum(np.insert(self.sym_cnt, 0, 0))  # [0, f0, f0+f1, ...]

    # ---------- 编码 ----------
    def encode(self, symbols: List[int]) -> bytes:
        low = 0
        range_ = 1 << 31  # 31-bit 精度
        byte_out = bytearray()

        for s in symbols:
            sym_low = self.cum_freq[s]
            sym_high = self.cum_freq[s + 1]
            low += sym_low * range_ // self.total
            range_ = sym_high * range_ // self.total - sym_low * range_ // self.total

            # 字节输出(归一化)
            while range_ <= 1 << 23:
                byte_out.append((low >> 23) & 0xFF)
                low = (low << 8) & ((1 << 31) - 1)
                range_ <<= 8

        # flush
        while low != 0:
            byte_out.append((low >> 23) & 0xFF)
            low = (low << 8) & ((1 << 31) - 1)
        return bytes(byte_out)

    # ---------- 解码 ----------
    def decode(self, byte_stream: bytes, length: int) -> List[int]:
        low = 0
        range_ = 1 << 31
        code = 0
        for b in byte_stream[:4]:  # 先填 4 B
            code = (code << 8) | b
        idx = 0
        out = []
        for _ in range(length):
            # 找到当前码所在符号
            cum = ((code - low) * self.total + range_ - 1) // range_
            s = np.searchsorted(self.cum_freq, cum, side="right") - 1
            out.append(s)

            # 更新区间
            sym_low = self.cum_freq[s]
            sym_high = self.cum_freq[s + 1]
            low += sym_low * range_ // self.total
            range_ = sym_high * range_ // self.total - sym_low * range_ // self.total

            # 字节输入归一化
            while range_ <= 1 << 23:
                if idx < len(byte_stream):
                    code = ((code << 8) | byte_stream[idx]) & ((1 << 31) - 1)
                    idx += 1
                else:
                    code = (code << 8) & ((1 << 31) - 1)
                low = (low << 8) & ((1 << 31) - 1)
                range_ <<= 8
        return out


if __name__ == "__main__":
    w = np.random.normal(7.5, 2, 100000).clip(0, 15).round().astype(int).tolist()
    freq = Counter(w)
    coder = RangeCoder4Bit(freq)
    byte_stream = coder.encode(w)
    bpw = len(byte_stream) * 8 / len(w)
    print("bpw = %.3f" % bpw)

    recovered = coder.decode(byte_stream, len(w))
    assert np.array_equal(recovered, np.array(w, dtype=np.uint8))
    print("round-trip ok!")
