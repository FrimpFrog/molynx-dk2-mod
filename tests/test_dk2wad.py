from pathlib import Path

import pytest

from scripts.dk2wad import decompress, read_wad

MESHES = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/"
              "scratchpad/dk2game/Ai/Meshes.WAD")
needs_game = pytest.mark.skipif(not MESHES.exists(), reason="DK2 Meshes.WAD not present")


def test_decompress_literals_only():
    # header 10 FB, size 7; literal block of 4, then final literal run of 3
    src = bytes([0x10, 0xFB, 0, 0, 7, 0xE0]) + b"ABCD" + bytes([0xFF]) + b"EFG"
    assert decompress(src, 7) == b"ABCDEFG"


def test_decompress_back_reference_overlaps():
    # 2 literals "AB", then copy 4 bytes from 2 back (overlapping), then end
    src = bytes([0x10, 0xFB, 0, 0, 6, 0x06, 0x01]) + b"AB" + bytes([0xFC])
    assert decompress(src, 6) == b"ABABAB"


def test_decompress_rejects_wrong_declared_size():
    src = bytes([0x10, 0xFB, 0, 0, 7, 0xE0]) + b"ABCD" + bytes([0xFF]) + b"EFG"
    with pytest.raises(ValueError, match="size"):
        decompress(src, 9)


def test_read_wad_rejects_non_wad(tmp_path):
    bad = tmp_path / "bad.wad"
    bad.write_bytes(b"NOPE" + bytes(200))
    with pytest.raises(ValueError, match="DWFB"):
        read_wad(bad)


@needs_game
def test_meshes_wad_parses():
    wad = read_wad(MESHES)
    assert wad.version == 2
    assert len(wad.entries) > 100
    assert any(e.name.lower().endswith(".kmf") for e in wad.entries)


@needs_game
def test_every_meshes_entry_extracts_to_declared_size():
    wad = read_wad(MESHES)
    for entry in wad.entries:
        data = wad.extract(entry)
        expected = entry.size if entry.compressed else entry.packed_size
        assert len(data) == expected, entry.name


@needs_game
def test_kmf_entries_start_with_kmsh_signature():
    wad = read_wad(MESHES)
    kmf = [e for e in wad.entries if e.name.lower().endswith(".kmf")]
    assert kmf
    for entry in kmf:
        assert wad.extract(entry)[:4] == b"KMSH", entry.name
