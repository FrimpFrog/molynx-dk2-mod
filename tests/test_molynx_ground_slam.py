"""Ground Slam (molynx_ground_slam): the records it reuses, what it writes, and its icon. The
expectations are written out here from the design; the files are the game's own (skips without)."""
import pytest

import scripts.molynx_ground_slam as gs
from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2kwd_tables import by_id, read_records
from scripts.dk2paths import EDITOR_DIR, GAME_DIR, MAPS_DIR, QUUZ_MAPS_DIR, have

needs_data = pytest.mark.skipif(not (have(EDITOR_DIR) and have(MAPS_DIR)), reason="DK2 data not present")
SIEGE = QUUZ_MAPS_DIR / "SiegeGlobals.kld"
CHANGED = {0xA0: {23}, 0x104: {51}, 0x10E: {36, 40, 56, 127}, 0xFA: {77, 490}}


def _tables(data: bytes) -> dict:
    return {b.signature: by_id(read_records(b)) for b in read_blocks(data) if b.signature in CHANGED}


def _sources():
    out = [MAPS_DIR / "Dead WalkingGlobals.kld"]
    if have(SIEGE):
        out.append(SIEGE)
    return out


@needs_data
@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_only_the_reused_records_change(path):
    src = path.read_bytes()
    out = gs.apply_ground_slam(src)
    a, b = read_blocks(src), read_blocks(out)
    assert len(src) == len(out) and [x.signature for x in a] == [x.signature for x in b]
    for x, y in zip(a, b):
        assert x.header == y.header
        if x.signature not in CHANGED:
            assert x.content == y.content
            continue
        old, new = by_id(read_records(x)), by_id(read_records(y))
        assert {i for i in old if old[i].raw != new[i].raw} == CHANGED[x.signature]


@needs_data
def test_the_spell_and_its_shot():
    t = _tables(gs.apply_ground_slam((MAPS_DIR / "Dead WalkingGlobals.kld").read_bytes()))
    spell, gas = t[0xA0][23], t[0xA0][5]
    assert spell.name == "Ground Slam" and spell.get("creatureSpellId") == 23
    assert spell.get("shotTypeId") == 51 and spell.get("alternativeShotId") == 0
    assert spell.get("flags") == gas.get("flags") == 0x109 and spell.get("range") == gas.get("range") == 4096
    assert spell.get("shotData1") == 240 and spell.get("shotData2") == 2 and spell.get("rechargeTime") == 32768
    assert spell.get("soundEvent") == 0x2EC
    assert spell.text("guiIcon.name") == "GUI\\C-Spells\\c-ground_slam"
    assert spell.get("guiIcon.type") == gas.get("guiIcon.type") == 1
    shot, grenade = t[0x104][51], t[0x104][19]
    same = {n for n in grenade.fields() if grenade.get(n) == shot.get(n)}
    assert set(grenade.fields()) - same == {"name", "shotId", "speed", "data1", "health",
                                            "creationEffectId", "attackCategory", "soundCategory"}
    assert shot.get("threat") == grenade.get("threat")   # creatures notice it like the explosion
    assert shot.get("processType") == 0 and shot.get("damageType") == 2 and shot.get("speed") == 0
    assert shot.get("collideType") == 2 and shot.get("creationEffectId") == 36
    assert shot.text("soundCategory") == "KEEPER_SPELL_TREMOR" and shot.get("attackCategory") == 13
    assert t[0x104][36].name == "Chicken Arrow"     # the archers' alternative shot is left alone


@needs_data
def test_the_burst_is_a_chain_of_records_that_exist():
    t = _tables(gs.apply_ground_slam((MAPS_DIR / "Dead WalkingGlobals.kld").read_bytes()))
    effects, elements = t[0x10E], t[0xFA]
    root = effects[36]
    assert root.get("generationType") == 18 and root.get("flags") & 0x40 and root.get("elementsPerTurn") == 4
    kids = [root.get(f"generateIds[{k}]") for k in range(4)]
    assert kids == [40, 127, 56, 0]
    assert effects[40].get("flags") & 0x40 and effects[40].get("generateIds[0]") == 305
    for i, parts in ((127, [6, 490, 373]), (56, [536, 77])):
        assert effects[i].get("flags") & 0x80
        assert [effects[i].get(f"generateIds[{k}]") for k in range(len(parts))] == parts
        assert all(p in elements for p in parts)
    assert effects[56].get("light.radius") == 6144
    assert elements[77].get("colour") == bytes((0xB4, 0x32, 0x0A))
    assert elements[490].get("colour") == bytes((0x8C, 0x28, 0x14))
    assert elements[77].get("art.type") == elements[536].get("art.type")
    for i in (36, 40, 56, 127):
        assert effects[i].get("effectId") == i and effects[i].name.startswith("Ground Slam")


@needs_data
def test_nothing_else_names_the_reused_records():
    refs = gs.references(read_blocks((MAPS_DIR / "Dead WalkingGlobals.kld").read_bytes()))
    assert 51 not in refs["shot"]
    assert all(i not in refs["effect"] for i in (36, 40, 56, 127))
    assert all(i not in refs["element"] for i in (77, 490))
    assert "creature spell 12 alternative" in refs["shot"][36]     # why shot 36 is not used


@needs_data
def test_applying_twice_or_to_a_changed_file_is_refused():
    data = (MAPS_DIR / "Dead WalkingGlobals.kld").read_bytes()
    with pytest.raises(ValueError, match="Chicken Arrow"):
        gs.apply_ground_slam(gs.apply_ground_slam(data))
    blocks = read_blocks(data)
    k = next(i for i, b in enumerate(blocks) if b.signature == 0x104)
    records = read_records(blocks[k])
    records[0].set("deathShotId", 51)             # something now names shot 51
    blocks[k] = rebuilt_block(blocks[k], [bytes(r.raw) for r in records])
    with pytest.raises(ValueError, match="not free"):
        gs.apply_ground_slam(write_blocks(blocks))


@needs_data
def test_editor_files_get_the_same_records():
    files = gs.editor_files(EDITOR_DIR)
    assert sorted(files) == sorted(gs.EDITOR_FILES)
    globals_ = _tables(gs.apply_ground_slam((MAPS_DIR / "Dead WalkingGlobals.kld").read_bytes()))
    for name, data in files.items():
        (block,) = read_blocks(data)
        (src,) = read_blocks((EDITOR_DIR / name).read_bytes())
        assert block.header == src.header
        mine = by_id(read_records(block))
        for i in CHANGED[block.signature]:
            # the same values as in a Globals file (the records' unused padding may differ)
            for field in ("name", "generateIds[0]", "shotTypeId", "shotId", "colour", "data1"):
                try:
                    assert mine[i].get(field) == globals_[block.signature][i].get(field), (name, i, field)
                except KeyError:
                    pass


@pytest.mark.skipif(not have(GAME_DIR / "EngineTextures.dir"), reason="texture cache not present")
def test_icon_is_a_64_pixel_rgb_picture_in_the_frame_of_the_game_icons():
    from scripts.molynx_assets import texture_cache
    import numpy as np
    cache = texture_cache()
    icon = gs.make_icon(cache)
    assert icon.size == (64, 64) and icon.mode == "RGB"
    gas = np.asarray(cache.get(next(n for n in cache.names if n.lower() == "gui\\c-spells\\c-gas_cloudmm0"))
                     .convert("RGB"), float)
    mine = np.asarray(icon, float)
    border = np.ones((64, 64), bool)
    border[6:58, 6:58] = False
    assert np.abs(mine[border] - gas[border]).mean() < 12    # the same riveted frame
    assert (mine[8:56, 8:56].max(axis=2) > 180).mean() > 0.08  # a light glyph in the field
