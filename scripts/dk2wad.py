"""Read Dungeon Keeper 2 WAD archives (DWFB, version 2) and extract their files.

Layout and the decompression routine follow the community 010 Editor template and script
in github.com/ufdada/dk2-tools (Formats/Archives/wad_struct.bt, Scripts/wad_extract.1sc).

Header (88 bytes): "DWFB", u32 version, 64 unused bytes, u32 file count, u32 names offset,
u32 names size, u32 unknown. Then one 40-byte entry per file:
  u32 unk, u32 name offset, u32 name size, u32 file offset, u32 stored size,
  u32 type (0 raw, 4 compressed), u32 original size, 3 x u32 unknown.
"""
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

HEADER_SIZE = 88
ENTRY_SIZE = 40
TYPE_RAW, TYPE_COMPRESSED = 0, 4


@dataclass(frozen=True)
class WadEntry:
    name: str
    offset: int
    packed_size: int
    type: int
    size: int  # original size; only meaningful when compressed

    @property
    def compressed(self) -> bool:
        return self.type == TYPE_COMPRESSED


class Wad:
    def __init__(self, path: Path, version: int, entries: list[WadEntry]):
        self.path, self.version, self.entries = Path(path), version, entries

    def extract(self, entry: WadEntry) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(entry.offset)
            raw = f.read(entry.packed_size)
        if entry.compressed:
            return decompress(raw, entry.size)
        return raw


def read_wad(path: Path) -> Wad:
    data = Path(path).read_bytes()
    if data[:4] != b"DWFB":
        raise ValueError(f"{Path(path).name}: not a WAD file (expected DWFB, got {data[:4]!r})")
    version = struct.unpack_from("<I", data, 4)[0]
    if version != 2:
        raise ValueError(f"{Path(path).name}: unsupported WAD version {version}")
    count = struct.unpack_from("<I", data, 4 + 4 + 64)[0]
    entries = []
    for i in range(count):
        base = HEADER_SIZE + i * ENTRY_SIZE
        (_unk, name_off, name_size, file_off, packed, typ, size, *_rest) = struct.unpack_from(
            "<9I", data, base)
        name = data[name_off:name_off + name_size].split(b"\0", 1)[0].decode("latin-1")
        entries.append(WadEntry(name, file_off, packed, typ, size))
    return Wad(path, version, entries)


def decompress(src: bytes, size: int) -> bytes:
    """RefPack-style decompression used by DK2 WADs; raises ValueError on size mismatch."""
    i = 0
    if src[i] & 1:
        i += 3
    i += 1
    i += 1  # second header byte
    declared = (src[i] << 16) | (src[i + 1] << 8) | src[i + 2]
    if declared != size:
        raise ValueError(f"size mismatch: stream declares {declared}, expected {size}")
    i += 3
    out = bytearray()
    finished = False
    n = len(src)
    while i < n:
        flag = src[i]
        i += 1
        if flag & 0x80 == 0:
            tmp = src[i]
            i += 1
            lit = flag & 3
            out += src[i:i + lit]
            i += lit
            k = len(out) - ((flag & 0x60) << 3) - tmp - 1
            count = ((flag >> 2) & 7) + 3
        elif flag & 0x40 == 0:
            tmp, tmp2 = src[i], src[i + 1]
            i += 2
            lit = tmp >> 6
            out += src[i:i + lit]
            i += lit
            k = len(out) - ((tmp & 0x3F) << 8) - tmp2 - 1
            count = (flag & 0x3F) + 4
        elif flag & 0x20 == 0:
            b1, b2, b3 = src[i], src[i + 1], src[i + 2]
            i += 3
            lit = flag & 3
            out += src[i:i + lit]
            i += lit
            k = len(out) - ((flag & 0x10) << 12) - (b1 << 8) - b2 - 1
            count = b3 + ((flag & 0x0C) << 6) + 5
        else:
            lit = (flag & 0x1F) * 4 + 4
            if lit > 0x70:
                finished = True
                lit = flag & 3
            out += src[i:i + lit]
            i += lit
            if finished:
                break
            continue
        if k < 0:
            raise ValueError("corrupt stream: back-reference before start of output")
        for _ in range(count):  # byte-wise so overlapping copies repeat correctly
            out.append(out[k])
            k += 1
    if not finished:
        raise ValueError("stream ended without a terminator")
    if len(out) != size:
        raise ValueError(f"size mismatch: produced {len(out)}, expected {size}")
    return bytes(out)


def extract_all(wad_path: Path, out_dir: Path) -> int:
    """Extract every entry under out_dir, keeping sub-folders from backslash names."""
    wad = read_wad(wad_path)
    out_dir = Path(out_dir).resolve()
    for entry in wad.entries:
        parts = [p for p in entry.name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
        target = out_dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(wad.extract(entry))
    return len(wad.entries)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python3 -m scripts.dk2wad <file.WAD> <output dir>")
    print(f"extracted {extract_all(Path(sys.argv[1]), Path(sys.argv[2]))} files")
