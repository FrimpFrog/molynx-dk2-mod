import struct

import pytest

from scripts.dk2kld_splice import (ThingRef, find_creature_things, first_player_bile_thing,
                                   map_source_dir, molynx_for_map, pick_test_map, read_things,
                                   retarget_creature_thing, write_things)
from scripts.dk2kwd_blocks import HEADER_SIZE, read_blocks
from scripts.dk2paths import MAPS_DIR, QUUZ_MAPS_DIR, have

needs_maps = pytest.mark.skipif(not (have(MAPS_DIR) and have(QUUZ_MAPS_DIR)),
                                reason="DK2 editor maps or Quuz maps not present")

THINGS = 0xBE
CREATURE_THING = 200
SUFFIXES = (".kwd", "Map.kld", "Things.kld", "Players.kld", "Triggers.kld", "Variables.kld",
            "Globals.kld")

# The chosen test map, pinned from a hex dump of QUUZ_MAPS_DIR/SiegeThings.kld (1264 bytes, one
# 0xBE block of 32 things; the 15 creature things are the first 15, 36 bytes each from byte 56).
TEST_MAP = "Siege"
SIEGE_CREATURES = [(1, 3)] * 4 + [(1, 4)] * 4 + [(15, 4), (16, 4), (25, 4), (25, 4), (3, 3),
                                                  (22, 3), (22, 3)]
SIEGE_BILE_INDEX, SIEGE_BILE_OFFSET = 12, 488


def _things_files() -> list:
    return [p for d in (MAPS_DIR, QUUZ_MAPS_DIR) if have(d) for p in sorted(d.glob("*Things.kld"))]


def _siege_things() -> bytes:
    return (QUUZ_MAPS_DIR / "SiegeThings.kld").read_bytes()


def _changed(a: bytes, b: bytes) -> list[int]:
    assert len(a) == len(b)
    return [i for i in range(len(a)) if a[i] != b[i]]


# ---- parse / re-serialise -----------------------------------------------------------------------

@needs_maps
def test_every_things_file_is_found():
    files = _things_files()
    assert len([p for p in files if p.parent == MAPS_DIR]) == 283
    assert len([p for p in files if p.parent == QUUZ_MAPS_DIR]) == 62


@needs_maps
@pytest.mark.parametrize("path", _things_files(), ids=lambda p: f"{p.parent.parent.parent.name}/{p.name}")
def test_parse_and_reserialise_gives_identical_bytes(path):
    data = path.read_bytes()
    header, things = read_things(data)
    assert len(things) == header.item_count
    out = write_things(header, things)
    assert out == data
    assert out is not data


@needs_maps
def test_write_things_rebuilds_the_header():
    data = _siege_things()
    header, things = read_things(data)
    out = write_things(header, things[:-1])
    (block,) = read_blocks(out)
    assert block.item_count == len(things) - 1 == 31
    assert len(out) == len(data) - len(things[-1].raw)
    assert struct.unpack_from("<I", out, 8)[0] == len(out)
    assert struct.unpack_from("<I", out, 52)[0] == len(out) - HEADER_SIZE


def test_read_things_rejects_a_record_that_overruns_its_block():
    header = struct.pack("<7I", THINGS, 4, HEADER_SIZE + 8, THINGS + 1, 28, 1, 1) + bytes(20) \
        + struct.pack("<2I", THINGS + 2, 8)
    with pytest.raises(ValueError):
        read_things(header + struct.pack("<II", CREATURE_THING, 28))


def test_read_things_rejects_a_file_that_is_not_one_things_block():
    header = struct.pack("<7I", 0xAA, 4, HEADER_SIZE, 0xAB, 28, 0, 1) + bytes(20) \
        + struct.pack("<2I", 0xAC, 0)
    with pytest.raises(ValueError):
        read_things(header)


# ---- creature things ----------------------------------------------------------------------------

@needs_maps
@pytest.mark.parametrize("path", _things_files(), ids=lambda p: f"{p.parent.parent.parent.name}/{p.name}")
def test_creature_things_are_sane_in_every_file(path):
    data = path.read_bytes()
    refs = find_creature_things(data)
    (block,) = read_blocks(data)
    # Independent scan: every record is u32 type, u32 data size, then the data.
    pos, expected = HEADER_SIZE, []
    while pos < len(data):
        kind, size = struct.unpack_from("<II", data, pos)
        if kind == CREATURE_THING:
            expected.append(pos)
        pos += 8 + size
    assert [r.offset for r in refs] == expected
    for r in refs:
        assert struct.unpack_from("<II", data, r.offset) == (CREATURE_THING, 28)
        assert 1 <= r.creature_id <= 46
        assert 3 <= r.player <= 7  # keeper slots only; 1 (Good) and 2 (Neutral) never own one
        assert (r.creature_id, r.player) == (data[r.offset + 34], data[r.offset + 35])


@needs_maps
def test_test_map_creature_things_are_pinned():
    refs = find_creature_things(_siege_things())
    assert [(r.creature_id, r.player) for r in refs] == SIEGE_CREATURES
    assert [r.offset for r in refs] == [56 + 36 * i for i in range(15)]
    assert sum(r.creature_id == 3 for r in refs) == 1
    assert refs[SIEGE_BILE_INDEX] == ThingRef(SIEGE_BILE_OFFSET, 3, 3)


@needs_maps
def test_first_player_bile_thing():
    assert first_player_bile_thing(_siege_things()) == SIEGE_BILE_INDEX


@needs_maps
@pytest.mark.parametrize("owner", [1, 2, 0, 8])
def test_first_player_bile_thing_ignores_non_keeper_owners(owner):
    data = bytearray(_siege_things())
    data[SIEGE_BILE_OFFSET + 35] = owner
    with pytest.raises(ValueError):
        first_player_bile_thing(bytes(data))


# ---- retarget -----------------------------------------------------------------------------------

@needs_maps
def test_retarget_changes_exactly_the_creature_id_byte():
    data = _siege_things()
    before = bytes(data)
    out = retarget_creature_thing(data, SIEGE_BILE_INDEX, 47)
    assert data == before
    assert _changed(data, out) == [SIEGE_BILE_OFFSET + 34]
    assert out[SIEGE_BILE_OFFSET + 34] == 47


@needs_maps
def test_retargeted_file_lists_the_record_with_creature_id_47():
    data = _siege_things()
    refs = find_creature_things(data)
    out_refs = find_creature_things(retarget_creature_thing(data, SIEGE_BILE_INDEX, 47))
    assert out_refs[SIEGE_BILE_INDEX] == ThingRef(SIEGE_BILE_OFFSET, 47, 3)
    assert out_refs[:SIEGE_BILE_INDEX] + out_refs[SIEGE_BILE_INDEX + 1:] \
        == refs[:SIEGE_BILE_INDEX] + refs[SIEGE_BILE_INDEX + 1:]


@needs_maps
@pytest.mark.parametrize("index", [0, 14])
def test_retarget_any_creature_thing(index):
    data = _siege_things()
    out = retarget_creature_thing(data, index, 47)
    assert _changed(data, out) == [56 + 36 * index + 34]


@needs_maps
@pytest.mark.parametrize("index", [-1, 15, 100])
def test_retarget_rejects_a_bad_index(index):
    with pytest.raises(ValueError):
        retarget_creature_thing(_siege_things(), index, 47)


@needs_maps
@pytest.mark.parametrize("creature_id", [0, 256, -1])
def test_retarget_rejects_an_id_that_does_not_fit(creature_id):
    with pytest.raises(ValueError):
        retarget_creature_thing(_siege_things(), SIEGE_BILE_INDEX, creature_id)


# ---- the test map -------------------------------------------------------------------------------

@needs_maps
def test_pick_test_map_is_pinned():
    assert pick_test_map() == TEST_MAP


@needs_maps
def test_test_map_files_all_exist_in_its_source_dir():
    name = pick_test_map()
    folder = map_source_dir(name)
    assert folder == QUUZ_MAPS_DIR
    for suffix in SUFFIXES:
        assert (folder / (name + suffix)).is_file()
    # its Globals takes Molynx: one "Bile Demon" with creatureId 3
    molynx_for_map((folder / (name + "Globals.kld")).read_bytes(), 1)


@needs_maps
def test_map_source_dir():
    assert map_source_dir("Siege") == QUUZ_MAPS_DIR
    assert map_source_dir("13coop") == MAPS_DIR
    with pytest.raises(FileNotFoundError):
        map_source_dir("No Such Map")


def _criteria(folder, name):
    """(size, Globals shape, [(creature id, owner)]) recomputed without the module under test."""
    paths = [folder / (name + s) for s in SUFFIXES]
    (creatures,) = [b for b in read_blocks(paths[6].read_bytes()) if b.signature == 0xAA]
    shape = (creatures.item_count, len(creatures.content) // creatures.item_count)
    data = paths[2].read_bytes()
    pos, things = HEADER_SIZE, []
    while pos < len(data):
        kind, size = struct.unpack_from("<II", data, pos)
        if kind == CREATURE_THING:
            things.append((data[pos + 34], data[pos + 35]))
        pos += 8 + size
    return sum(p.stat().st_size for p in paths), shape, things


@needs_maps
def test_pick_test_map_recomputed_independently():
    def complete(folder):
        return {p.stem for p in folder.glob("*.kwd")
                if all((folder / (p.stem + s)).is_file() for s in SUFFIXES)}

    def kwd_names(folder):
        return {p.stem for p in folder.glob("*.kwd")}

    quuz, editor = complete(QUUZ_MAPS_DIR), complete(MAPS_DIR)
    rows = []
    for name in quuz | editor:
        folder = QUUZ_MAPS_DIR if name in quuz else MAPS_DIR
        size, shape, things = _criteria(folder, name)
        if shape == (46, 5537) and any(c == 3 and 3 <= o <= 7 for c, o in things):
            both = name in kwd_names(QUUZ_MAPS_DIR) and name in kwd_names(MAPS_DIR)
            rows.append((size, name, both))
    assert len(rows) == 41
    in_both = [r for r in rows if r[2]]
    pool = in_both or rows
    assert min(pool)[1] == pick_test_map()
    assert min(rows)[1] == pick_test_map()  # also the smallest of all candidates
