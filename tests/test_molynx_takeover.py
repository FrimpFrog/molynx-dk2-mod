"""Tests for scripts/molynx_takeover.py (Molynx moved from creature 47 into the unused slot 46).

LICENCE: GPL-3.0.
"""
import struct

import pytest

from scripts.build_molynx_mod import build_mod
from scripts.dk2kld_splice import find_creature_things
from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_creatures import read_creatures
from scripts.dk2paths import EDITOR_DIR, GAME_DIR, MAPS_DIR, QUUZ_MAPS_DIR, have
from scripts.molynx_takeover import _level_ids, check_takeover, takeover

needs_data = pytest.mark.skipif(
    not (have(MAPS_DIR) and have(QUUZ_MAPS_DIR) and have(GAME_DIR / "Meshes.WAD") and have(EDITOR_DIR)),
    reason="game data not present")
MAPS = "Data/editor/maps"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_mod(1, tmp_path_factory.mktemp("b47"), "Molynx_slot_only")


@pytest.fixture(scope="module")
def moved(built, tmp_path_factory):
    return takeover(built, tmp_path_factory.mktemp("b46") / "Molynx_slot_only")


@needs_data
def test_takeover_passes_its_own_check(built, moved):
    assert check_takeover(built, moved) == []


@needs_data
def test_every_creature_table_has_46_records_and_molynx_in_slot_46(moved):
    tables = [moved / "Data/editor/Creatures.kwd", *sorted((moved / MAPS).glob("*Globals.kld"))]
    assert len(tables) == 5
    for path in tables:
        (block,) = [b for b in read_blocks(path.read_bytes()) if b.signature == 0xAA]
        recs = read_creatures(block)
        assert [c.get("creatureId") for c in recs] == list(range(1, 47)), path.name
        assert recs[45].get("name") == "Molynx"
        assert recs[45].get("orderInEditor") == 46


@needs_data
def test_no_creature_47_is_left_and_molynx_is_keeper_1s_creature_46(moved):
    for things in (moved / MAPS).glob("*Things.kld"):
        refs = find_creature_things(things.read_bytes())
        assert all(r.creature_id != 47 for r in refs)
        assert [(r.creature_id, r.player) for r in refs if r.creature_id == 46] == [(46, 3)], things.name


@needs_data
def test_the_portal_map_puts_creature_46_in_keeper_1s_pool(moved):
    (block,) = read_blocks((moved / MAPS / "Molynx Test PortalVariables.kld").read_bytes())
    ents = [struct.unpack_from("<iiii", block.content, i) for i in range(0, len(block.content), 16)]
    assert (1, 46, 50, 3) in ents
    assert not any(e[0] == 1 and e[1] == 47 for e in ents)


@needs_data
def test_level_files_list_creature_46(moved):
    for kwd in (moved / MAPS).glob("*.kwd"):
        data = kwd.read_bytes()
        assert _level_ids(data)[-1] == 46
        assert struct.unpack_from("<I", data, 8)[0] == len(data)


@needs_data
def test_check_finds_a_record_47_left_in_place(built, moved, tmp_path):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(moved, bad)
    (bad / MAPS / "SiegeThings.kld").write_bytes((built / MAPS / "SiegeThings.kld").read_bytes())
    assert any("SiegeThings.kld" in p for p in check_takeover(built, bad))


@needs_data
def test_takeover_refuses_an_existing_folder(built, moved):
    with pytest.raises(FileExistsError):
        takeover(built, moved)
