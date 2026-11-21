import struct

import pytest

from scripts.dk2kwd_blocks import Block, read_blocks, rebuilt_block, write_blocks
from scripts.dk2paths import EDITOR_DIR, MAPS_DIR, have

needs_maps = pytest.mark.skipif(not have(MAPS_DIR), reason="DK2 editor maps not present")
needs_editor = pytest.mark.skipif(not have(EDITOR_DIR), reason="DK2 editor data not present")

CREATURES = 0xAA
HEADER_SIZE = 56
# Files whose headers break some invariants of test_header_invariants: {file name: (invariants, reason)}.
# They still frame correctly (file size is right), so the roundtrip tests cover them normally.
HEADER_EXCEPTIONS: dict[str, tuple[set[str], str]] = {
    "AutoSave.kwd": (
        {"one", "check_two", "content_size"},
        "map-info file (one 0xDC block, as in every <map>.kwd), not an item table: the header words at "
        "offsets 20 and 24 are not an item count and 1, the second check word is signature + 3, and "
        "the content size (0x6453) covers only the fixed part, 628 bytes short of the block, which "
        "ends with an inner 0xDE section.",
    ),
}


def _fields(block: Block) -> dict[str, int]:
    """Decode the header fields the invariants talk about (the header itself stays opaque bytes)."""
    sig, four, size, check_one, twenty_eight, count, one = struct.unpack_from("<7I", block.header, 0)
    check_two, content_size = struct.unpack_from("<2I", block.header, 48)
    return dict(sig=sig, four=four, size=size, check_one=check_one, twenty_eight=twenty_eight,
                count=count, one=one, check_two=check_two, content_size=content_size)


def _synthetic_block(signature: int, count: int, record: bytes) -> bytes:
    content = record * count
    header = (struct.pack("<7I", signature, 4, HEADER_SIZE + len(content), signature + 1, 28, count, 1)
              + bytes(range(10)) + bytes(range(10, 20))
              + struct.pack("<2I", signature + 2, len(content)))
    assert len(header) == HEADER_SIZE
    return header + content


def _broken_invariants(block: Block) -> set[str]:
    f = _fields(block)
    checks = {
        "header_size": len(block.header) == HEADER_SIZE,
        "signature": f["sig"] == block.signature,
        "four": f["four"] == 4,
        "twenty_eight": f["twenty_eight"] == 28,
        "one": f["one"] == 1,
        "check_one": f["check_one"] == block.signature + 1,
        "check_two": f["check_two"] == block.signature + 2,
        "file_size": f["size"] == HEADER_SIZE + len(block.content),
        "content_size": f["content_size"] == len(block.content),
    }
    return {name for name, ok in checks.items() if not ok}


def _roundtrip_failures(paths) -> list[str]:
    return [p.name for p in paths if write_blocks(read_blocks(p.read_bytes())) != p.read_bytes()]


@needs_maps
def test_every_globals_kld_roundtrips():
    files = sorted(MAPS_DIR.glob("*Globals.kld"))
    assert len(files) == 241
    assert _roundtrip_failures(files) == []


@needs_editor
def test_editor_kwd_files_roundtrip():
    files = sorted(EDITOR_DIR.glob("*.kwd"))
    assert len(files) == 13
    assert _roundtrip_failures(files) == []


@needs_maps
@needs_editor
def test_header_invariants():
    broken: dict[str, set[str]] = {}
    for path in [*sorted(MAPS_DIR.glob("*Globals.kld")), *sorted(EDITOR_DIR.glob("*.kwd"))]:
        for block in read_blocks(path.read_bytes()):
            if bad := _broken_invariants(block):
                broken.setdefault(path.name, set()).update(bad)
    # Exactly the recorded exceptions fail, and each fails exactly the invariants recorded for it.
    assert broken == {name: waived for name, (waived, _reason) in HEADER_EXCEPTIONS.items()}


@needs_maps
def test_rebuilt_block_counts():
    data = (MAPS_DIR / "01coopGlobals.kld").read_bytes()
    blocks = read_blocks(data)
    old = next(b for b in blocks if b.signature == CREATURES)
    assert old.item_count == 46
    size = len(old.content) // old.item_count
    assert size == 5537 and size * old.item_count == len(old.content)
    records = [old.content[i * size:(i + 1) * size] for i in range(old.item_count)]
    new = rebuilt_block(old, records + [records[-1]])
    assert new.item_count == 47
    assert new.header[:4] == old.header[:4]
    assert new.header[28:48] == old.header[28:48]  # created / modified stamps stay verbatim
    again = read_blocks(write_blocks([new if b is old else b for b in blocks]))
    assert [b.signature for b in again] == [b.signature for b in blocks]
    assert again[blocks.index(old)].item_count == 47
    assert again[blocks.index(old)].content == b"".join(records + [records[-1]])


def test_rebuilt_block_updates_only_count_and_sizes():
    like = read_blocks(_synthetic_block(CREATURES, 2, b"abcd"))[0]
    new = rebuilt_block(like, [b"abcd", b"efgh", b"ijkl"])
    assert new.signature == CREATURES and new.content == b"abcdefghijkl"
    f = _fields(new)
    assert (f["count"], f["size"], f["content_size"]) == (3, HEADER_SIZE + 12, 12)
    assert _broken_invariants(new) == set()
    # Everything but file size (8:12), item count (20:24) and content size (52:56) is copied verbatim.
    for lo, hi in [(0, 8), (12, 20), (24, 52)]:
        assert new.header[lo:hi] == like.header[lo:hi]


def test_read_blocks_splits_a_sequence():
    data = _synthetic_block(0x6E, 1, b"xx") + _synthetic_block(0x78, 3, b"yyy")
    blocks = read_blocks(data)
    assert [(b.signature, b.item_count, len(b.content)) for b in blocks] == [(0x6E, 1, 2), (0x78, 3, 9)]
    assert write_blocks(blocks) == data


@pytest.mark.parametrize("cut", [10, HEADER_SIZE, HEADER_SIZE + 5])
def test_read_blocks_rejects_truncated_data(cut):
    data = _synthetic_block(0x6E, 2, b"abcd")
    with pytest.raises(ValueError, match="truncated|size"):
        read_blocks(data[:cut])
