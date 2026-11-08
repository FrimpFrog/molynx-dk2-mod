"""Tests of the release folder's map choice and Variables change (build_molynx_release)."""
import struct

import pytest

from scripts.build_molynx_release import (_entries, map_kind, molynx_variables)
from scripts.dk2kld_splice import LEVEL_FLAGS_AT

HEADER = 56


def _kwd(flags: int) -> bytes:
    data = bytearray(LEVEL_FLAGS_AT + 2)
    struct.pack_into("<H", data, LEVEL_FLAGS_AT, flags)
    return bytes(data)


def _variables(entries, claimed=None) -> bytes:
    content = b"".join(struct.pack("<iiii", *e) for e in entries)
    count = len(entries) if claimed is None else claimed
    header = bytearray(HEADER)
    struct.pack_into("<9I", header, 0, 0xE6, 4, HEADER + count * 16, 0xE7, 28, count, 1, 0, 0)
    struct.pack_into("<I", header, 52, count * 16)
    return bytes(header) + content


@pytest.mark.parametrize("name,flags,kind", [
    ("Siege", 1 << 10, "skirmish"),          # Quuz's copy: multiplayer flag only
    ("Furnace", (1 << 10) | (1 << 11), "skirmish"),
    ("DK2Map2", 1 << 11, "skirmish"),
    ("mpd4", 0x84, "mpd"),                   # My Pet Dungeon by name: the game's flag is on mpd7 only
    ("MPD6", 0, "mpd"),
    ("Level4", 1 << 10, None),               # the campaign, whatever its flags
    ("level11A", 1 << 11, None),
    ("Secret3", 1 << 11, None),
    ("Whynots Alien", 0, None),              # a single-player custom level
])
def test_map_kind(name, flags, kind):
    assert map_kind(name, [_kwd(flags)]) == kind


def test_map_kind_takes_any_copy():
    assert map_kind("Warrens", [_kwd(0x404), _kwd(0xC04)]) == "skirmish"


def test_pool_and_availability_follow_the_bile_demon_per_player():
    src = [(3, 0, 0, 0),
           (1, 1, 50, 3), (1, 3, 20, 3), (1, 4, 50, 3),
           (1, 1, 30, 4), (1, 4, 30, 4),                       # player 4: no Bile Demon
           (2, (3 << 16) | 2, 3, 3), (2, (3 << 16) | 2, 4, 3),
           (2, (4 << 16) | 1, 5, 3),                           # a room: not a creature list
           (66, 0, 0, 0)]
    _block, out = _entries(molynx_variables(_variables(src)))
    assert out == [(3, 0, 0, 0),
                   (1, 1, 50, 3), (1, 3, 20, 3), (1, 4, 50, 3), (1, 47, 20, 3),
                   (1, 1, 30, 4), (1, 4, 30, 4), (1, 47, 30, 4),
                   (2, (3 << 16) | 2, 3, 3), (2, (3 << 16) | 2, 4, 3), (2, (3 << 16) | 2, 47, 3),
                   (2, (4 << 16) | 1, 5, 3),
                   (66, 0, 0, 0)]


def test_a_map_without_any_creature_pool_is_unchanged():
    data = _variables([(3, 0, 0, 0), (2, (3 << 16) | 1, 5, 3)])
    assert molynx_variables(data) == data


def test_refuses_a_map_that_already_has_molynx():
    with pytest.raises(ValueError):
        molynx_variables(_variables([(1, 47, 10, 3)]))


def test_a_header_claiming_one_entry_too_many_is_rebuilt_true():
    data = _variables([(1, 3, 10, 3)], claimed=2)      # Quuz's Spiral: 310 claimed, 309 there
    out = molynx_variables(data)
    (size,) = struct.unpack_from("<I", out, 8)
    (count,) = struct.unpack_from("<I", out, 20)
    assert (size, count) == (len(out), 2)
    assert _entries(out)[1] == [(1, 3, 10, 3), (1, 47, 10, 3)]
