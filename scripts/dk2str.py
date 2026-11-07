"""Dungeon Keeper 2 text files (Data\\Text\\Default\\*.str): read, and append new strings.

Layout (all little-endian):
  "BFST", u32 (1091 in Text.str, kept as is), u32 count, u32 offset[count] (from the start of the
  offset table), then the entries.
An entry is a list of chunks, each starting with a u32 whose low byte is the chunk type and whose
upper three bytes are a number:
  type 1  text: the number of bytes that follow (one byte per character, padded with zeros to a
          multiple of four); byte b is character b - 1 of the code page (MBToUni.dat: "BFMU",
          u16 0, u16 count, then count u16 code points).
  type 2  a value the game fills in, numbered (e.g. 26 and 27 in a creature's description).
  type 0  the end of the entry.
Existing entries are kept byte for byte; `append` adds new ones at the end, so their IDs are the
next numbers after the file's count.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"BFST"
CODEPAGE_MAGIC = b"BFMU"
_PLACEHOLDER = re.compile(r"\{(\d+)\}")


def read_codepage(path: Path) -> list[str]:
    data = Path(path).read_bytes()
    if data[:4] != CODEPAGE_MAGIC:
        raise ValueError(f"{path} is not a DK2 code page (no BFMU)")
    (n,) = struct.unpack_from("<H", data, 6)
    return [chr(c) for c in struct.unpack_from(f"<{n}H", data, 8)]


@dataclass
class StrFile:
    header: int
    entries: list[bytes]        # each entry's bytes, as stored
    codepage: list[str]

    @classmethod
    def read(cls, path: Path, codepage_path: Path | None = None) -> "StrFile":
        path = Path(path)
        data = path.read_bytes()
        if data[:4] != MAGIC:
            raise ValueError(f"{path} is not a DK2 text file (no BFST)")
        header, count = struct.unpack_from("<II", data, 4)
        offsets = [o + 12 for o in struct.unpack_from(f"<{count}I", data, 12)] + [len(data)]
        if offsets[0] != 12 + 4 * count or offsets != sorted(offsets):
            raise ValueError(f"{path}: the offset table does not describe the entries")
        cp = read_codepage(codepage_path or path.with_name("MBToUni.dat"))
        return cls(header, [data[a:b] for a, b in zip(offsets, offsets[1:])], cp)

    def text(self, index: int) -> str:
        """Entry `index` as text, with the game's values shown as {n}."""
        b, out, i = self.entries[index], [], 0
        while i + 4 <= len(b):
            kind = b[i]
            n = b[i + 1] | b[i + 2] << 8 | b[i + 3] << 16
            i += 4
            if kind == 0:
                break
            if kind == 1:
                out.append("".join(self.codepage[x - 1] if 0 < x <= len(self.codepage) else "?"
                                   for x in b[i:i + n]))
                i += n + (-n) % 4
            elif kind == 2:
                out.append("{%d}" % n)
            else:
                raise ValueError(f"entry {index}: unknown chunk type {kind}")
        return "".join(out)

    def encode(self, text: str) -> bytes:
        """`text` as an entry ({n}: the game's value n)."""
        index = {c: k + 1 for k, c in enumerate(self.codepage)}
        out = bytearray()
        pos = 0
        for m in [*_PLACEHOLDER.finditer(text), None]:
            piece = text[pos:m.start() if m else len(text)]
            if piece:
                missing = sorted({c for c in piece if c not in index})
                if missing:
                    raise ValueError(f"characters not in the game's code page: {missing}")
                raw = bytes(index[c] for c in piece)
                out += struct.pack("<I", 1 | len(raw) << 8) + raw + bytes((-len(raw)) % 4)
            if m:
                out += struct.pack("<I", 2 | int(m.group(1)) << 8)
                pos = m.end()
        return bytes(out + bytes(4))

    def append(self, texts: list[str]) -> list[int]:
        """Add `texts` at the end; returns their IDs."""
        first = len(self.entries)
        self.entries += [self.encode(t) for t in texts]
        return list(range(first, len(self.entries)))

    def to_bytes(self) -> bytes:
        count = len(self.entries)
        offsets, at = [], 4 * count
        for e in self.entries:
            offsets.append(at)
            at += len(e)
        return (MAGIC + struct.pack("<II", self.header, count) + struct.pack(f"<{count}I", *offsets)
                + b"".join(self.entries))
