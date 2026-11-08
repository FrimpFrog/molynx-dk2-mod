"""Tests of Molynx's sound bank files (molynx_soundbank), on small made-up banks."""
import struct
import wave

import numpy as np
import pytest

from scripts.molynx_soundbank import (FORMAT_MP2_22K, HEADER, molynx_bank_map, molynx_sdt, read_sdt,
                                      sample_name, write_sdt)
from scripts.molynx_sounds import ROLES


def _bank(names):
    head = bytes(16) + bytes(8) + struct.pack("<I", len(names)) + bytes.fromhex("ffffffff18f366005b2173") * len(names)
    return head + b"".join(struct.pack("<I", len(n) + 1) + n.encode() + b"\0" for n in names)


def test_bank_map_renames_only_his_own_bank():
    out = molynx_bank_map(_bank(["Global\\Bile", "Global\\MiscCreatures"]))
    assert out.endswith(b"\x0e\x00\x00\x00Global\\Molynx\x00\x15\x00\x00\x00Global\\MiscCreatures\x00")
    assert out[:0x1c + 22] == _bank(["Global\\Bile", "Global\\MiscCreatures"])[:0x1c + 22]


def test_bank_map_refuses_a_bank_without_the_bile_demon():
    with pytest.raises(ValueError):
        molynx_bank_map(_bank(["Global\\Troll", "Global\\MiscCreatures"]))


def _entry(name, fmt=FORMAT_MP2_22K, data=b"x" * 10):
    return HEADER.pack(40, len(data), name.encode().ljust(16, b"\0"), fmt, 0, 99, 0), data


def test_sdt_round_trip():
    entries = [_entry("a.mp2"), _entry("Blank", 0x105622, b""), _entry("b.mp2")]
    assert read_sdt(write_sdt(entries)) == entries


def test_every_sample_is_replaced_in_order_and_blanks_kept(tmp_path):
    entries, k = [], 0
    for i, bile in enumerate(ROLES):
        entries.append(_entry((bile + ".mp2")[:15]))
        if i % 10 == 3:
            entries.append(_entry("Blank", 0x105622, b""))
    for role in set(ROLES.values()):
        with wave.open(str(tmp_path / f"{role}.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050)
            w.writeframes((np.sin(np.arange(2205) / 9.0) * 8000).astype("<i2").tobytes())
    out = read_sdt(molynx_sdt(write_sdt(entries), tmp_path))
    assert [sample_name(h) for h, _ in out] == [sample_name(h) for h, _ in entries]
    for (h, d), (h0, d0) in zip(out, entries):
        if sample_name(h0) == "Blank":
            assert (h, d) == (h0, d0)
            continue
        size_h, size, _name, fmt, _z1, pcm, _z2 = HEADER.unpack(h)
        assert (size_h, size, fmt, pcm) == (40, len(d), FORMAT_MP2_22K, 2205 * 2)
        assert d[:2] == b"\xff\xf5"
