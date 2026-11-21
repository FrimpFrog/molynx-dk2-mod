"""Decode Dungeon Keeper 2 texture cache images (EngineTextures.dir / EngineTextures.dat).

LICENCE: GPL-3.0. This is a Python port of the texture decoder recovered by the OpenKeeper
project (github.com/tonihele/OpenKeeper, tools/convert/textures/), so it carries that licence.
Do not redistribute it, or anything built on it, except under GPL-3.0 terms.

Format. EngineTextures.dir is a "TCHC" index: u32 size, u32 version, u32 count, then per entry
a null-terminated name and a u32 offset into the .dat. Names end in MM0..MM3 (mipmap level).
At the offset in the .dat: u32 width, u32 height, u32 size (bytes from here, so data is size-8
long), u16 width, u16 height, u32 flags (alpha present when flags >> 7 != 0), then
the pixel stream as little-endian u32 words.

Codec. Each 8x8 block holds up to four planes (Y/Cb/Cr style, then alpha), each coded as a
DC delta plus run-length coded DCT coefficients from a bit stream, inverse-DCT'd in fixed
point, and converted to RGB with a YCbCr-style matrix.
"""
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.dk2texture_tables import DC_CONTROL, JUMP, MAGIC_INPUT

M32 = 0xFFFFFFFF


def _w(x: int) -> int:
    """Wrap to a signed 32-bit integer, as Java ints do."""
    return ((x + 0x80000000) & M32) - 0x80000000


def _f32(x: float) -> np.float32:
    return np.float32(x)


MAGIC_OUTPUT = [((v & 0x1FFFF) << 3) + ((v & 0xFFFE0000) >> 3) for v in MAGIC_INPUT]
NORM = 0x5A82799A
C_03C = float(_f32(5.4119611e-1))
C_040 = float(_f32(1.306563))
C_044 = float(_f32(3.8268343e-1))
F000, F004 = _f32(1.048576e6), _f32(8.388608e6)
F008, F00C, F010 = _f32(1.169), _f32(-8.1300002e-1), _f32(-3.91e-1)
F014, F018 = _f32(1.602), _f32(2.0250001)
MAGIC_DOUBLE = 6.75539944108852e15


def _round_half_away(x: float) -> int:
    return int(x + (0.5 if x > 0 else -0.5))


def _idct_pass1(src, so, dst, do):
    """One 8-point inverse transform: reads src[so..so+7], writes dst[do + 9*k]."""
    if (src[so + 1] | src[so + 2] | src[so + 3] | src[so + 4] | src[so + 6] | src[so + 7]) == 0:
        a = src[so]
        for k in range(8):
            dst[do + 9 * k] = a
        return
    b = _w(src[so + 5] - src[so + 3])
    c = _w(src[so + 1] - src[so + 7])
    i = _w(src[so + 3] + src[so + 5])
    a = _w(src[so + 7] + src[so + 1])
    xf, xg = float(b), float(c)
    p = _w(i + a)
    a = _w(a - i)
    rxs = xg + xf
    rxf = xf * C_03C + C_044 * rxs
    rxg = xg * C_040 - C_044 * rxs
    ra, rb = _round_half_away(rxf), _round_half_away(rxg)
    d = _w((a * NORM) >> 32)
    b = src[so + 6]
    d = _w(d + d)
    a = src[so + 2]
    c = ra
    i = rb
    c = _w(c + d)
    d = _w(d + i)
    i = _w(i + p)
    sc, sd, si = c & M32, d & M32, i & M32
    c = src[so]
    d = src[so + 4]
    s = _w(b + a)
    a = _w(a - b)
    b = _w(d + c)
    c = _w(c - d)
    d = _w((a * NORM) >> 32)
    d = _w(d + d)
    dst[do + 18] = _w((c - d) + sc)
    dst[do + 45] = _w((c - d) - sc)
    dst[do + 27] = _w((b - (s + d)) + ra)
    dst[do + 36] = _w((b - (s + d)) - ra)
    dst[do + 0] = _w((s + d) + b + si)
    dst[do + 9] = _w(sd + d + c)
    dst[do + 54] = _w(d + c - sd)
    dst[do + 63] = _w((s + d) + b - si)


def _idct_pass2(src, so, dst, do):
    """Second inverse transform: reads src[so..so+7], writes dst[do..do+7]."""
    b = _w(src[so + 5] - src[so + 3])
    c = _w(src[so + 1] - src[so + 7])
    i = _w(src[so + 3] + src[so + 5])
    a = _w(src[so + 7] + src[so + 1])
    xf, xg = float(b), float(c)
    p = _w(i + a)
    a = _w(a - i)
    rxs = xg + xf
    rxf = xf * C_03C + C_044 * rxs
    rxg = xg * C_040 - C_044 * rxs
    ra, rb = _round_half_away(rxf), _round_half_away(rxg)
    d = _w((a * NORM) >> 32)
    b = src[so + 6]
    d = _w(d + d)
    a = src[so + 2]
    c = ra
    i = rb
    c = _w(c + d)
    d = _w(d + i)
    i = _w(i + p)
    sc, sd, si = c & M32, d & M32, i & M32
    c = src[so]
    d = src[so + 4]
    s = _w(b + a)
    a = _w(a - b)
    b = _w(d + c)
    c = _w(c - d)
    d = _w((a * NORM) >> 32)
    d = _w(d + d)
    p = _w(sc)
    s = _w(s + d)
    a = _w(d + c)
    c = _w(c - d)
    d = _w(s + b)
    b = _w(b - s)
    s = _w(c + p)
    c = _w(c - p)
    p = ra
    dst[do + 2] = s
    s = _w(sd)
    dst[do + 5] = c
    c = _w(b + p)
    b = _w(b - p)
    p = _w(si)
    dst[do + 3] = c
    dst[do + 4] = b
    b = _w(s + a)
    a = _w(a - s)
    c = _w(d + p)
    d = _w(d - p)
    dst[do + 0] = c
    dst[do + 1] = b
    dst[do + 6] = a
    dst[do + 7] = d


class _Decoder:
    def __init__(self, words):
        self.bs = list(words) + [0, 0, 0]
        self.index = 0
        self.dc = {0: 0, 1: 0, 2: 0, 3: 0}  # running DC value per plane
        self.c2 = [0] * 256
        self.c3 = [0] * 288
        self.c4 = [0] * 512

    def read(self, pos: int, bits: int) -> int:
        word, sh = pos >> 5, pos & 31
        w1 = (self.bs[word] << sh) & M32
        if sh and word + 1 < len(self.bs):
            w1 |= (self.bs[word + 1] >> (32 - sh)) & M32
        return w1 >> (32 - bits)

    def prepare(self, value: int, pos: int) -> int:
        """Fill c2 with dequantised coefficients for one plane; return the new bit position."""
        magic_index = 0x3F
        control_word = 0
        xindex = index = 0
        done = False
        self.c2[0] = _w(value * MAGIC_OUTPUT[0])
        for k in range(1, 256):
            self.c2[k] = 0
        limit = len(self.bs) * 32 + 64
        for _ in range(100000):
            if pos > limit:
                raise ValueError("corrupt texture stream (ran past the end)")
            if not done:
                xindex = index = self.read(pos, 17)
            if index >= 0x8000 or done:
                if not done:
                    index >>= 13
                    control_word = DC_CONTROL[60 + index]
                done = False
                if (control_word & 0xFF00) == 0x4100:
                    return pos + (control_word >> 16)
                if (control_word & 0xFF00) > 0x4100:
                    pos += control_word >> 16
                    unk14 = self.read(pos, 14)
                    pos += 14
                    magic_index -= (unk14 & 0xFF00) >> 8
                    unk14 &= 0xFF
                    if unk14 != 0:
                        if unk14 != 0x80:
                            if unk14 > 0x80:
                                unk14 -= 0x100
                            magic_index -= 1
                        else:
                            unk14 = self.read(pos, 8)
                            pos += 8
                            unk14 -= 0x100
                    else:
                        unk14 = self.read(pos, 8)
                        pos += 8
                    control_word = unk14
                else:
                    rem = control_word >> 16
                    xoramt = 0
                    magic_index -= (control_word & 0xFF00) >> 8
                    if xindex & (1 << (16 - rem)):
                        xoramt = -1
                    control_word &= 0xFF
                    control_word ^= xoramt
                    pos += 1
                    control_word -= xoramt
                    pos += rem
                out_index = DC_CONTROL[magic_index + 1]
                coeff = ((control_word + 0x8000) & 0xFFFF) - 0x8000  # (short) cast
                self.c2[out_index] = _w(coeff * MAGIC_OUTPUT[out_index])
            elif index >= 0x800:
                index >>= 9
                control_word = DC_CONTROL[72 + index]
                done = True
            elif index >= 0x400:
                index >>= 7
                control_word = DC_CONTROL[128 + index]
                done = True
            elif index >= 0x200:
                index >>= 5
                control_word = DC_CONTROL[128 + index]
                done = True
            elif index >= 0x100:
                index >>= 4
                control_word = DC_CONTROL[144 + index]
                done = True
            elif index >= 0x80:
                index >>= 3
                control_word = DC_CONTROL[160 + index]
                done = True
            elif index >= 0x40:
                index >>= 2
                control_word = DC_CONTROL[176 + index]
                done = True
            elif index >= 0x20:
                index >>= 1
                control_word = DC_CONTROL[192 + index]
                done = True
        raise ValueError("corrupt texture stream (no end marker)")

    def plane(self, channel: int, offset: int) -> None:
        pos = self.index
        value = 0
        jt = JUMP[self.read(pos, 8)]
        pos += jt & 0xF
        jt >>= 4
        if jt:
            value = self.read(pos, jt)
            if (value & (1 << (jt - 1))) == 0:
                value -= (1 << jt) - 1
            pos += jt
        self.dc[channel] += value
        dc = self.dc[channel]
        if self.read(pos, 2) == 2:
            pos += 2
            fill = _w(_w(dc) << 16)
            for j in range(8):
                for i in range(8):
                    self.c4[j * 64 + i + offset] = fill
            self.index = pos
        else:
            self.index = self.prepare(_w(dc), pos)
            for i in range(8):
                _idct_pass1(self.c2, i * 8, self.c3, i)
            for i in range(8):
                _idct_pass2(self.c3, i * 9, self.c4, i * 64 + offset)

    def block(self, alpha: bool) -> np.ndarray:
        """Decode one 8x8 block; returns an (8, 8, 4) uint8 RGBA array."""
        self.plane(0, 0)
        self.plane(1, 9)
        self.plane(2, 18)
        if alpha:
            self.plane(3, 27)
        c4 = np.array(self.c4, dtype=np.int64).reshape(8, 64)
        r = c4[:, 0:8].astype(np.float32)
        b = c4[:, 9:17].astype(np.float32)
        g = c4[:, 18:26].astype(np.float32)
        a = c4[:, 27:35]

        def to_int(d32):
            d = d32.astype(np.float64) + MAGIC_DOUBLE
            x = (d + 0.5).astype(np.int64) & M32
            return x.astype(np.uint32).view(np.int32) >> 16

        ir = to_int(F014 * (g - F004) + F008 * (r - F000))
        ig = to_int(F018 * (b - F004) + F008 * (r - F000))
        ib = to_int(F010 * (b - F004) + F00C * (g - F004) + F008 * (r - F000))
        out = np.empty((8, 8, 4), dtype=np.uint8)
        out[..., 0] = np.clip(ir, 0, 255)
        out[..., 1] = np.clip(ib, 0, 255)
        out[..., 2] = np.clip(ig, 0, 255)
        out[..., 3] = np.clip(a >> 16, 0, 255) if alpha else 255
        return out


def decode_texture(words, width: int, height: int, alpha: bool) -> Image.Image:
    dec = _Decoder(words)
    img = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            img[y:y + 8, x:x + 8] = dec.block(alpha)
    return Image.fromarray(img, "RGBA") if alpha else Image.fromarray(img[..., :3], "RGB")


class TextureCache:
    """Reader for EngineTextures.dir/.dat; get(name) returns a PIL image."""

    def __init__(self, dir_path: Path, dat_path: Path):
        d = Path(dir_path).read_bytes()
        if d[:4] != b"TCHC":
            raise ValueError("not a texture cache index (expected TCHC)")
        (count,) = struct.unpack_from("<I", d, 12)
        pos, self._offsets = 16, {}
        for _ in range(count):
            end = d.index(b"\0", pos)
            (off,) = struct.unpack_from("<I", d, end + 1)
            self._offsets[d[pos:end].decode("latin-1")] = off
            pos = end + 5
        self.names = list(self._offsets)
        self._dat = Path(dat_path).read_bytes()

    def _header(self, name: str):
        if name not in self._offsets:
            raise KeyError(name)
        off = self._offsets[name]
        w, h, size, _sw, _sh, flags = struct.unpack_from("<IIIHHI", self._dat, off)
        return off, w, h, size, (flags >> 7) != 0

    def size_of(self, name: str) -> tuple[int, int]:
        _, w, h, _, _ = self._header(name)
        return (w, h)

    def get(self, name: str) -> Image.Image:
        off, w, h, size, alpha = self._header(name)
        count = (size - 8) // 4
        words = np.frombuffer(self._dat, "<u4", count, off + 20).tolist()
        return decode_texture(words, w, h, alpha)
