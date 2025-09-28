import numpy as np
from collections import Counter

#ERROR: 解码逻辑错误
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
