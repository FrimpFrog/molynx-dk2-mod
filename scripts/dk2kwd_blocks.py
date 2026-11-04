"""Read and write the block container shared by Dungeon Keeper 2 *.kwd and *Globals.kld files.

A file is a plain sequence of blocks (signatures: 0x6E Terrain, 0x78 Rooms, 0x82 Traps, 0x8C Doors,
0x96 KeeperSpells, 0xA0 CreatureSpells, 0xAA Creatures, 0xF0 Objects, 0xFA EffectElements,
0x104 Shots, 0x10E Effects, ...). Every block starts with a 56-byte header (all little-endian):

  0  u32 signature          4  u32 4                 8  u32 file size (header + content)
  12 u32 signature + 1      16 u32 28                20 u32 item count
  24 u32 1                  28 10 bytes created      38 10 bytes modified
  48 u32 signature + 2      52 u32 content size (file size - 56)

followed by the content, normally `item count` records of equal size. Headers are kept as the
original bytes; the timestamps in particular are never interpreted.

LICENCE: GPL-3.0.
"""
import struct
from dataclasses import dataclass

HEADER_SIZE = 56
_SIZE_AT, _COUNT_AT, _CONTENT_SIZE_AT = 8, 20, 52


@dataclass(frozen=True)
class Block:
    signature: int
    header: bytes
    content: bytes

    @property
    def item_count(self) -> int:
        (count,) = struct.unpack_from("<I", self.header, _COUNT_AT)
        return count


def read_blocks(data: bytes) -> list[Block]:
    """Split a file into its blocks; raises ValueError if a block does not fit the data."""
    blocks: list[Block] = []
    pos = 0
    while pos < len(data):
        if len(data) - pos < HEADER_SIZE:
            raise ValueError(f"truncated block header at byte {pos}")
        header = data[pos:pos + HEADER_SIZE]
        (signature,) = struct.unpack_from("<I", header, 0)
        (size,) = struct.unpack_from("<I", header, _SIZE_AT)
        if size < HEADER_SIZE:
            raise ValueError(f"block 0x{signature:X} at byte {pos} has file size {size} < {HEADER_SIZE}")
        if pos + size > len(data):
            raise ValueError(f"truncated block 0x{signature:X} at byte {pos}: "
                             f"file size {size}, {len(data) - pos} bytes left")
        blocks.append(Block(signature, header, data[pos + HEADER_SIZE:pos + size]))
        pos += size
    return blocks


def write_blocks(blocks: list[Block]) -> bytes:
    return b"".join(b.header + b.content for b in blocks)


def rebuilt_block(like: Block, records: list[bytes]) -> Block:
    """A block with `records` as its content; item count, file size and content size are updated
    in the header and everything else is copied from `like`."""
    content = b"".join(records)
    header = bytearray(like.header)
    struct.pack_into("<I", header, _SIZE_AT, HEADER_SIZE + len(content))
    struct.pack_into("<I", header, _COUNT_AT, len(records))
    struct.pack_into("<I", header, _CONTENT_SIZE_AT, len(content))
    return Block(like.signature, bytes(header), content)
