"""Tests for scripts/dk2_add_creatures.py (creature types 48, 49, 50... added to a resource folder).

LICENCE: GPL-3.0.
"""
import json
import struct

import pytest

from scripts.build_molynx_mod import build_mod
from scripts.dk2_add_creatures import (ELITE_PLACEHOLDERS, FLAG_AVAILABLE_VIA_PORTAL, FLAG_IS_UNIQUE, FLOOR,
                                       NewCreature, _map_tiles, _names, add_creatures, check_added, load_specs,
                                       main)
from scripts.dk2kld_splice import read_things
from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_creatures import read_creatures
from scripts.dk2paths import EDITOR_DIR, GAME_DIR, MAPS_DIR, QUUZ_MAPS_DIR, TEXT_DIR, have

needs_data = pytest.mark.skipif(
    not (have(MAPS_DIR) and have(QUUZ_MAPS_DIR) and have(GAME_DIR / "Meshes.WAD") and have(EDITOR_DIR)
         and have(TEXT_DIR)),
    reason="game data not present")
MAPS = "Data/editor/maps"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_mod(1, tmp_path_factory.mktemp("b47"), "Molynx_slot_only")


@pytest.fixture(scope="module")
def added(built, tmp_path_factory):
    return add_creatures(built, tmp_path_factory.mktemp("b50") / "Molynx_slot_only")


def _table(path):
    (block,) = [b for b in read_blocks(path.read_bytes()) if b.signature == 0xAA]
    return read_creatures(block)


def _creatures(path):
    _b, things = read_things(path.read_bytes())
    return [(t.data[26], t.data[27], struct.unpack_from("<ii", t.data), t.data[14]) for t in things if t.type == 200]


def _entries(path):
    (block,) = read_blocks(path.read_bytes())
    return [struct.unpack_from("<iiii", block.content, i) for i in range(0, len(block.content), 16)]


@needs_data
def test_the_placeholders_pass_the_independent_check(built, added):
    assert check_added(built, added) == []


@needs_data
def test_every_table_has_types_1_to_50_and_the_elite_copies(added):
    names = _names(added)
    tables = [added / "Data/editor/Creatures.kwd", *sorted((added / MAPS).glob("*Globals.kld"))]
    assert len(tables) == 5
    for path in tables:
        recs = _table(path)
        assert [c.get("creatureId") for c in recs] == list(range(1, 51)), path.name
        assert recs[46].get("name") == "Molynx"
        for c, (want, donor) in zip(recs[47:], (("Elite Troll", 35), ("Elite Dark Angel", 44),
                                                 ("Elite Salamander", 39))):
            assert c.get("name") == want and names.text(c.get("nameStringId")) == want
            assert c.get("orderInEditor") == c.get("creatureId")
            assert c.get("flags") & FLAG_AVAILABLE_VIA_PORTAL and not c.get("flags") & FLAG_IS_UNIQUE
            assert recs[donor - 1].get("flags") & FLAG_IS_UNIQUE          # the donor is a unique one
            assert c.get("hp") == recs[donor - 1].get("hp")              # the elite stats


@needs_data
def test_the_placeholders_stand_next_to_molynx_on_free_floor(added):
    for things in sorted((added / MAPS).glob("*Things.kld")):
        cs = _creatures(things)
        (molynx,) = [c for c in cs if c[0] == 47]
        new = [c for c in cs if c[0] >= 48]
        assert sorted(c[0] for c in new) == [48, 49, 50], things.name
        width, height, tiles = _map_tiles((things.parent / things.name.replace("Things", "Map")).read_bytes())
        for cid, owner, (x, y), level in new:
            assert owner == molynx[1] and level == molynx[3]
            assert max(abs(x - molynx[2][0]), abs(y - molynx[2][1])) <= 2
            assert tiles[y * width + x][0] in FLOOR
        assert len({c[2] for c in cs}) == len(cs)          # no two creatures on one tile


@needs_data
def test_the_portal_maps_put_48_to_50_in_keeper_1s_pool_like_molynx(added):
    for stem in ("Molynx Test Portal", "Dead Walking"):
        ents = _entries(added / MAPS / f"{stem}Variables.kld")
        for cid in (48, 49, 50):
            assert (1, cid, 50, 3) in ents and (2, 3 << 16 | 2, cid, 3) in ents, (stem, cid)
    ents = _entries(added / MAPS / "Molynx TestVariables.kld")       # Molynx is not in this pool
    assert not [e for e in ents if e[0] == 1 and e[1] >= 48]


@needs_data
def test_level_files_and_assets_are_unchanged(built, added):
    for path in built.rglob("*"):
        rel = path.relative_to(built)
        if path.is_file() and (rel.suffix in (".kmf", ".png", ".kwd") and "Creatures" not in rel.name):
            assert (added / rel).read_bytes() == path.read_bytes(), rel


@needs_data
def test_ids_must_follow_the_table_without_a_gap(built, tmp_path):
    with pytest.raises(ValueError, match="new ids must be"):
        add_creatures(built, tmp_path / "gap", [NewCreature(49, 35)])
    with pytest.raises(ValueError, match="no creature 60"):
        add_creatures(built, tmp_path / "missing", [NewCreature(48, 60)])
    with pytest.raises(FileExistsError):
        add_creatures(built, built)


def test_specs_are_checked_before_anything_is_read(tmp_path):
    for bad in ([], [NewCreature(46, 3)], [NewCreature(48, 3), NewCreature(48, 4)],
                [NewCreature(48, 3, level=11)], [NewCreature(48, 3, name="x" * 32)]):
        with pytest.raises(ValueError):
            add_creatures(tmp_path / "none", tmp_path / "out", bad)


def test_spec_files_load(tmp_path):
    path = tmp_path / "specs.json"
    path.write_text(json.dumps([{"creature_id": 48, "clone_of": 35, "name": "Elite Troll", "follow": 47}]))
    assert load_specs(path) == [ELITE_PLACEHOLDERS[0]]


@needs_data
def test_a_custom_spec_without_follow_adds_only_the_record(built, tmp_path):
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps([{"creature_id": 48, "clone_of": 3, "portal": None, "unique": None}]))
    out = tmp_path / "one" / "Molynx"
    assert main([str(built), str(out), "--spec", str(spec)]) == 0
    recs = _table(out / "Data/editor/Creatures.kwd")
    assert recs[47].get("flags") == recs[2].get("flags") and recs[47].get("name") == recs[2].get("name")
    for things in (out / MAPS).glob("*Things.kld"):
        assert things.read_bytes() == (built / MAPS / things.name).read_bytes()


@needs_data
def test_the_check_finds_a_wrong_record_thing_and_pool_entry(built, tmp_path):
    from scripts.dk2kwd_blocks import write_blocks
    from scripts.dk2kwd_creatures import creatures_block
    out = add_creatures(built, tmp_path / "x" / "Molynx_slot_only")
    path = out / "Data/editor/Creatures.kwd"
    blocks = read_blocks(path.read_bytes())
    k = next(i for i, b in enumerate(blocks) if b.signature == 0xAA)
    recs = read_creatures(blocks[k])
    recs[48].set("hp", recs[48].get("hp") + 1)
    blocks[k] = creatures_block(recs, blocks[k])
    path.write_bytes(write_blocks(blocks))
    things = out / MAPS / "Molynx TestThings.kld"
    things.write_bytes((built / MAPS / "Molynx TestThings.kld").read_bytes())
    problems = check_added(built, out)
    assert any("Creatures.kwd: record 49 differs" in p for p in problems)
    assert any("Molynx TestThings.kld: 0 things added, want 3" in p for p in problems)
