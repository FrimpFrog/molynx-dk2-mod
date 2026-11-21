"""Checks of the cast install (the Orc 49, the Classic Reaper 50, the Dragon 51; 52 and 53 reserved):
records, Variables, placements by the heart, the checker, and the Reaper's files. Most tests read the
staged MoreCreatures folder and the built files; they skip when those are not on this machine."""
import json
import os
import shutil
import struct
from pathlib import Path

import pytest

UP = Path("/mnt/user-data/uploads/Dungeon Keeper 2")
BASE = UP / "flame/resources/MoreCreatures"
FURNACE = (UP / "flame/resources/Quuz_LevelFixes/Data/editor/maps"
           if (UP / "flame/resources/Quuz_LevelFixes/Data/editor/maps/FurnaceThings.kld").exists()
           else UP / "Data/editor/maps")                # the copy the game loads (cast_install.loaded_copy)
CODEPAGE = UP / "Data/Text/Default/MBToUni.dat"
MESHES = Path("/home/claude/dk2data/Meshes")
FILES = Path(os.environ.get("CAST_FILES", "/home/claude/cast/files"))
# the latest round (the Reaper made a regular creature) and the folder it goes into: the staged folder with
# round 1, Sejanus and the Gargoyle; the Gargoyle's round (its lists) and the folder that went into
INSTALL = Path(os.environ.get("CAST_INSTALL", "/home/claude/cast/install_reaper_regular"))
INSTALLED = Path(os.environ.get("CAST_BASE", "/home/claude/cast/base_r4/MoreCreatures"))
GARGOYLE_INSTALL = Path("/home/claude/cast/install_gargoyle")
NECRO_INFO = Path("/home/claude/necro/out/v5/files.json")

need_base = pytest.mark.skipif(not (BASE / "Data/editor/Creatures.kwd").exists(), reason="no staged MoreCreatures")
need_files = pytest.mark.skipif(not (FILES / "dragon").exists(), reason="cast files not built")
need_install = pytest.mark.skipif(not (INSTALL / "MoreCreatures").exists() or not INSTALLED.exists(),
                                  reason="cast not installed")


def _table(path: Path):
    from scripts.dk2kld_splice import _creatures_index, _records
    from scripts.dk2kwd_blocks import read_blocks
    b = read_blocks(path.read_bytes())
    return _records(b[_creatures_index(b)])


def _infos():
    from scripts.cast_install import files_info
    from scripts.cast_records import CAST, all_files
    folders = {49: FILES / "orc", 50: FILES / "hreaper", 51: FILES / "dragon", 52: FILES / "sejanus",
               53: FILES / "gargoyle"}
    return {m.creature_id: files_info(folders[m.creature_id], all_files(m)) for m in CAST}


# ------------------------------------------------------------------------------ the cast list

def test_ids_names_and_reserved():
    from scripts.cast_records import BY_ID, CAST, RESERVED
    assert [m.creature_id for m in CAST] == [49, 50, 51, 52, 53]
    assert RESERVED == {} and not set(RESERVED) & set(BY_ID)
    for m in CAST:
        assert 0 < len(m.name) <= 31 and m.name.isascii()
        assert set(m.texts) == {"name", "general", "strength", "weakness"}
        assert "{26}" in m.texts["general"] and "{27}" in m.texts["general"]


def test_file_lists():
    from scripts.cast_records import DRAGON, GARGOYLE, HREAPER, ORC, SEJANUS, all_files, slot_files
    assert len(all_files(ORC)) == 36                    # 28 Troll slots + 8 Start / End files
    assert len(all_files(HREAPER)) == 15                # Horny's 11 clips + pose + 3 Start / End
    assert len(all_files(DRAGON)) == 40
    assert len(all_files(SEJANUS)) == 45                # 41 clips + 6 poses, less Dejected / DejectedPose
    assert len(all_files(GARGOYLE)) == 45
    assert slot_files(SEJANUS)["animRunResource"] == ["Sejanus-Run", "Sejanus-RunStart", "Sejanus-RunEnd"]
    assert slot_files(ORC)["animWalkResource"] == ["Orc-walk", "Orc-walkStart", "Orc-walkEnd"]
    assert slot_files(HREAPER)["animMelee1Resource"] == ["HReaper-Melee3", "HReaper-Melee3Start", "HReaper-Melee3End"]
    assert "animDejectedPoseResource" not in slot_files(DRAGON)     # no game creature fills it


def test_reaper_file_plan_is_hornys_record():
    from scripts.dk2kwd_creatures import ANIM_SLOTS, ART_FLAG_HAS_END_ANIMATION as E, ART_FLAG_HAS_START_ANIMATION as S
    from scripts.hreaper_files import HORNY_SLOTS, file_plan, reaper_stem
    assert reaper_stem("REA3_Walk") == "HReaper-Walk" and reaper_stem("Rea3-poseframe") == "HReaper-PoseFrame"
    names = [n for _s, n, _src in file_plan()]
    assert len(names) == len(set(names)) == 15
    kwd = UP / "Data/editor/Creatures.kwd"
    if not kwd.exists():
        pytest.skip("no Creatures.kwd")
    horny = next(c for c in _table(kwd) if c.get("creatureId") == 27)
    slots = {s: (horny.resource_name(s), bool(horny.art_tail(s).flags & S), bool(horny.art_tail(s).flags & E))
             for s in ANIM_SLOTS[1:] if horny.resource_name(s)}
    assert slots == HORNY_SLOTS


# ------------------------------------------------------------------------------ records

@need_base
@need_files
def test_records_from_donors():
    from scripts.cast_records import CAST, DRAGON, HREAPER, ORC, build
    from scripts.dk2kwd_creatures import ANIM_SLOTS, ART_FLAG_HAS_START_ANIMATION as S
    table = _table(BASE / "Data/editor/Creatures.kwd")
    before = [bytes(c.raw) for c in table]
    infos = _infos()
    ids = {"name": 1, "general": 2, "strength": 3, "weakness": 4}
    recs = {m.creature_id: build(m, table, 48 + i + 1, infos[m.creature_id], ids) for i, m in enumerate(CAST)}
    assert [bytes(c.raw) for c in table] == before                    # donors untouched
    by = {c.get("creatureId"): c for c in table}
    orc, reaper, dragon = recs[49], recs[50], recs[51]
    assert orc.get("meleeDamage") > by[10].get("meleeDamage") and orc.get("hp") > by[10].get("hp")
    assert [orc.get(f"happyJobs[{i}].jobType") for i in range(3)] == [by[10].get(f"happyJobs[{i}].jobType") for i in range(3)]
    # the Reaper: below Horny, a regular creature (no IS_HORNY; Hand, slap, Portal), Horny's slot tails
    # where Horny has a clip, the shared tooltip
    from scripts.cast_records import HREAPER_FILL
    for f in ("hp", "meleeDamage", "speed", "scoreRating"):
        assert reaper.get(f) < by[27].get(f)
    assert by[27].get("flags") & 0x4000 and not reaper.get("flags") & 0x4000
    assert reaper.get("flags") & 0x6 == 0x6 and reaper.get("tooltipStringId") == 2408
    assert not reaper.get("flags") & 0x400                              # never through the Portal
    assert not any(reaper.get(f"attractions[{i}].present") for i in range(2))
    for s in ANIM_SLOTS[1:]:
        assert orc.art_tail(s) == by[10].art_tail(s)
        if s not in HREAPER_FILL:
            assert reaper.art_tail(s) == by[27].art_tail(s) and reaper.resource_name(s).startswith("HReaper-") == bool(by[27].resource_name(s))
        else:
            assert not by[27].resource_name(s) and reaper.resource_name(s).startswith("HReaper-")
    # the Dragon: the Salamander's fire, timed to his clip; Lair and Hatchery
    assert [dragon.get(f"spells[{i}].creatureSpellId") for i in range(3)] == [39, 26, 0]
    assert dragon.get("spells[0].shotDelay") == 2540 < infos[51]["Dragon-Magic"]["frames"] / 30 * 4096
    assert [(dragon.get(f"attractions[{i}].roomId"), dragon.get(f"attractions[{i}].roomSize")) for i in range(2)] == [(2, 25), (4, 25)]
    w = dragon.art_tail("animWalkResource")
    assert w.flags & S and w.start_af == infos[51]["Dragon-WalkStart"]["frames"]
    assert w.frames == infos[51]["Dragon-Walk"]["frames"] - 1                   # it loops
    # every slot names one of his files (or is empty)
    for m in CAST:
        mine = {n for n in __import__("scripts.cast_records", fromlist=["all_files"]).all_files(m)}
        for s in ANIM_SLOTS[1:]:
            n = recs[m.creature_id].resource_name(s)
            assert not n or n in mine, (m.name, s, n)
    assert ORC.icon == "orc" and HREAPER.icon == "hreaper" and DRAGON.icon == "dragon"


@need_base
@need_files
def test_cast_table_completes_the_necromancer_and_is_idempotent():
    """A table without the Necromancer (the staged copies of 8 Whynots_ maps were, on 12 Nov) gets
    him first, then the cast; a second run adds nothing."""
    from scripts.cast_install import cast_table
    from scripts.dk2kld_splice import _creatures_index, _records
    from scripts.dk2kwd_blocks import read_blocks, write_blocks
    from scripts.dk2kwd_creatures import creatures_block
    g = BASE / "Data/editor/maps/Whynots_Q_Globals.kld"
    blocks = read_blocks(g.read_bytes())
    at = _creatures_index(blocks)
    recs = _records(blocks[at])
    without = [c for c in recs if c.get("creatureId") < 48]       # the staged folder may already hold the cast
    blocks[at] = creatures_block(without, blocks[at])
    table = write_blocks(blocks)
    texts = {k: {"name": 9, "general": 9, "strength": 9, "weakness": 9} for k in (49, 50, 51, 52, 53)}
    necro = (json.loads(NECRO_INFO.read_text()), {"name": 3090, "general": 3091, "strength": 3092, "weakness": 3093,
                                                 "tooltip": 2408})
    data, added = cast_table(table, _infos(), texts, necro)
    assert added == [48, 49, 50, 51, 52, 53]
    b = read_blocks(data)
    out = {c.get("creatureId"): c for c in _records(b[_creatures_index(b)])}
    assert sorted(out) == list(range(1, 54))
    if any(c.get("creatureId") == 48 for c in recs):                # rebuilt exactly as installed
        assert bytes(out[48].raw) == bytes(next(c for c in recs if c.get("creatureId") == 48).raw)
    again, none = cast_table(data, _infos(), texts, necro)
    assert again == data and none == []


@pytest.mark.skipif(not (UP / "flame/config.toml").exists(), reason="no flame config")
def test_furnace_loads_from_the_first_folder_after_morecreatures():
    from scripts.cast_install import alt_resources, loaded_copy
    order = alt_resources(UP)
    assert order[0] == "MoreCreatures"                  # so its FurnaceThings wins over every other copy
    things, _ = loaded_copy(UP, "Data/editor/maps/FurnaceThings.kld")
    fmap, _ = loaded_copy(UP, "Data/editor/maps/FurnaceMap.kld")
    assert things.parent == fmap.parent
    if (UP / "flame/resources/Quuz_LevelFixes/Data/editor/maps/FurnaceThings.kld").exists():
        assert "Quuz_LevelFixes" in str(things)


# ------------------------------------------------------------------------------ variables

def test_keeper_variables_adds_once_for_keeper1_only():
    from scripts.build_molynx_release import ENTRY, _entries
    from scripts.cast_install import keeper_variables
    from scripts.dk2kwd_blocks import Block, HEADER_SIZE, write_blocks
    ents = [(1, 10, 50, 3), (1, 3, 40, 3), (1, 10, 50, 4), (2, (3 << 16) | 2, 10, 3), (2, (4 << 16) | 2, 10, 3)]
    content = b"".join(ENTRY.pack(*e) for e in ents)
    header = bytearray(HEADER_SIZE)
    struct.pack_into("<IIIII", header, 0, 0xE6, 4, HEADER_SIZE + len(content), 0xE7, 28)
    struct.pack_into("<I", header, 20, len(ents))
    struct.pack_into("<I", header, HEADER_SIZE - 4, len(content))
    data = write_blocks([Block(0xE6, bytes(header), content)])
    out = keeper_variables(data, [49, 51], count_from={49: 10, 51: 3})
    new = _entries(out)[1]
    assert (1, 49, 50, 3) in new and (1, 51, 40, 3) in new
    assert (2, (3 << 16) | 2, 49, 3) in new and (2, (3 << 16) | 2, 51, 3) in new
    assert not [e for e in new if e[0] == 1 and e[3] == 4 and e[1] in (49, 51)]
    assert keeper_variables(out, [49, 51]) == out


# ------------------------------------------------------------------------------ placement

@need_base
def test_tiles_by_the_heart():
    from scripts.cast_install import keeper_heart, read_map_tiles, tile_ok
    from scripts.dk2kld_splice import read_things
    maps = BASE / "Data/editor/maps"
    m = read_map_tiles((maps / "Dead WalkingMap.kld").read_bytes())
    _b, things = read_things((maps / "Dead WalkingThings.kld").read_bytes())
    heart = keeper_heart(m, things)
    assert heart == (2, 35, 6, 39)
    assert tile_ok(m, things, heart, (8, 40), "claimed") is None             # free before and after the cast
    assert "taken" in tile_ok(m, things, heart, (7, 37), "claimed")          # Molynx stands there
    assert tile_ok(m, things, heart, (4, 37), "claimed")                      # in the heart
    assert tile_ok(m, things, heart, (12, 37), "claimed")                     # too far
    assert tile_ok(m, things, heart, (0, 37), "claimed")                      # edge of the map
    if (FURNACE / "FurnaceMap.kld").exists():
        fm = read_map_tiles((FURNACE / "FurnaceMap.kld").read_bytes())
        _b, ft = read_things((FURNACE / "FurnaceThings.kld").read_bytes())
        fh = keeper_heart(fm, ft)
        assert fh == (6, 52, 10, 56)
        assert tile_ok(fm, ft, fh, (7, 52), "heart ring") is None
        assert tile_ok(fm, ft, fh, (8, 54), "heart ring")                     # the heart itself
        assert tile_ok(fm, ft, fh, (5, 54), "heart ring")                     # his Reinforced Wall
        assert tile_ok(fm, ft, fh, (7, 56), "heart ring")                     # an Imp's tile
        assert tile_ok(fm, ft, fh, (54, 6), "heart ring")                     # Keeper 2's heart


def test_placements_and_reserved_tiles_do_not_clash():
    from scripts.cast_install import PLACEMENTS, RESERVED_TILES
    for name in PLACEMENTS:
        tiles = list(PLACEMENTS[name].values()) + list(RESERVED_TILES[name].values())
        assert len(tiles) == len(set(tiles))
        assert not RESERVED_TILES[name]                  # every reserved tile is taken now
        assert {49, 50, 51, 52, 53} <= set(PLACEMENTS[name])


# ------------------------------------------------------------------------------ the install and the checker

@need_install
@need_base
def test_install_checks_out_and_the_checker_catches_faults(tmp_path):
    from scripts.cast_check import check
    from scripts.dk2texture import TextureCache
    from scripts.dk2kld_splice import Thing, read_things, write_things
    from scripts.dk2kwd_blocks import read_blocks, write_blocks
    from scripts.dk2kwd_creatures import creatures_block
    cache = TextureCache(UP / "Data/EngineTextures.dir", UP / "Data/EngineTextures.dat")
    out = INSTALL / "MoreCreatures"
    assert check(out, INSTALLED, FURNACE, MESHES, CODEPAGE, cache) == []
    # the state before this round (his Horny flags) is caught
    empty = tmp_path / "empty"
    empty.mkdir()
    assert any("IS_HORNY" in p for p in check(empty, INSTALLED, FURNACE, MESHES, CODEPAGE, cache))
    bad = tmp_path / "MoreCreatures"
    shutil.copytree(out, bad)
    ed = bad / "Data/editor/Creatures.kwd"
    blocks = read_blocks(ed.read_bytes())
    i = next(i for i, b in enumerate(blocks) if b.signature == 0xAA)
    recs = _records_of(blocks[i])
    r = next(c for c in recs if c.get("creatureId") == 50)
    r.set("flags", r.get("flags") | 0x4000)
    t = r.art_tail("animHappyResource")
    r.set_art_tail("animHappyResource", t._replace(frames=t.frames + 3))
    blocks[i] = creatures_block(recs, blocks[i])
    ed.write_bytes(write_blocks(blocks))
    things = bad / "Data/editor/maps/FurnaceThings.kld"
    block, ts = read_things((INSTALLED / "Data/editor/maps/FurnaceThings.kld").read_bytes())
    j = next(j for j, t in enumerate(ts) if t.type == 200 and t.data[26] == 53)
    d = bytearray(ts[j].data)
    struct.pack_into("<ii", d, 0, 5, 54)                                       # onto his wall
    ts[j] = Thing(200, bytes(d))
    things.write_bytes(write_things(block, ts))
    (bad / "Data/Meshes").mkdir(parents=True, exist_ok=True)
    problems = check(bad, INSTALLED, FURNACE, MESHES, CODEPAGE, cache)
    assert any("record 50 has IS_HORNY" in p for p in problems)
    assert any("animHappyResource: tail" in p for p in problems)
    assert any("creature 53 at (5, 54)" in p for p in problems)


def _records_of(block):
    from scripts.dk2kwd_creatures import read_creatures
    return read_creatures(block)


@need_install
def test_install_lists():
    lines = (INSTALL / "install_list.txt").read_text().splitlines()
    rep = json.loads((INSTALL / "report.json").read_text())
    assert len(lines) == len(rep["files"]) == len(rep["new"]) + len(rep["replaces"])
    assert all("flame\\resources\\MoreCreatures\\" in line for line in lines)
    # this round rebuilds record 50 in every table and his two texts; no new file, no mesh, no map
    assert rep["new"] == [] and not [f for f in rep["files"] if f.endswith((".kmf", ".png"))]
    assert all(v == [50] for v in rep["replaced_records"].values())
    tables = [f for f in rep["files"] if f.endswith(("Globals.kld", "Creatures.kwd"))]
    assert len(tables) == len(rep["replaced_records"]) == len(list((INSTALLED / "Data/editor/maps").glob("*Globals.kld"))) + 1
    # off Keeper 1's pool and availability on the two maps (the only Variables that had him)
    assert sorted(rep["variables"]) == ["Dead WalkingVariables.kld", "FurnaceVariables.kld"]
    assert len(rep["files"]) == len(tables) + 1 + 2
    assert sorted(rep["texts"]) == ["3100", "3101"]
    backup = (INSTALL / "backup_list.txt").read_text().splitlines()
    assert len(backup) == len(rep["replaces"])


@pytest.mark.skipif(not (GARGOYLE_INSTALL / "report.json").exists(), reason="no Gargoyle install")
def test_gargoyle_install_lists():
    rep = json.loads((GARGOYLE_INSTALL / "report.json").read_text())
    assert all(set(v) == {"53"} for v in rep["placed"].values())        # that round added the Gargoyle only


@need_install
@need_base
def test_reaper_regular_changes_only_his_record():
    from scripts.cast_records import HREAPER_FILL, HREAPER_FLAGS
    for f in sorted((INSTALL / "MoreCreatures").rglob("*.k[wl]d")):
        if f.name.endswith("Variables.kld"):
            continue
        rel = f.relative_to(INSTALL / "MoreCreatures")
        a, b = _table(INSTALLED / rel), _table(f)
        assert [c.get("creatureId") for c in a] == [c.get("creatureId") for c in b]
        for x, y in zip(a, b):
            if x.get("creatureId") != 50:
                assert x.raw == y.raw, (rel, x.get("creatureId"))            # Horny (27) among them
                continue
            assert y.get("flags") == HREAPER_FLAGS and not y.get("flags") & 0x4000
            assert y.get("orderInEditor") == x.get("orderInEditor")
            assert all(y.get(k) == x.get(k) for k in ("nameStringId", "generalDescriptionStringId",
                                                      "strengthStringId", "weaknessStringId", "tooltipStringId"))
            assert not any(y.get(f"attractions[{i}].present") for i in range(2)) and not y.get("flags") & 0x400
            assert y.get("pay") == 1000 and y.get("timeSleep") == x.get("timeSleep") == 0
            assert y.get("hungerRate") == x.get("hungerRate")              # Horny's: never hungry
            assert y.get("happyJobs[0].jobType") == 4                        # trains
            for slot in HREAPER_FILL:
                assert y.resource_name(slot).startswith("HReaper-") and not x.resource_name(slot)
    from scripts.build_molynx_release import _entries
    for name in ("Dead Walking", "Furnace"):
        a = _entries((INSTALLED / f"Data/editor/maps/{name}Variables.kld").read_bytes())[1]
        b = _entries((INSTALL / f"MoreCreatures/Data/editor/maps/{name}Variables.kld").read_bytes())[1]
        assert [e for e in a if e not in b] == [(1, 50, 50, 3), (2, (3 << 16) | 2, 50, 3)] and all(e in a for e in b)


# ------------------------------------------------------------------------------ files

@need_files
@pytest.mark.parametrize("folder", ["orc", "hreaper", "dragon", "sejanus", "gargoyle"])
def test_cast_files_parse_and_write_back(folder):
    from scripts.dk2kmf import parse_kmf
    from scripts.dk2kmf_write import anim_invariants, write_kmf
    from PIL import Image
    for p in sorted((FILES / folder).glob("*.kmf")):
        b = p.read_bytes()
        k = parse_kmf(b)
        assert write_kmf(k) == b, p.name
        x = k.anim or k.mesh
        assert all(len(s.vertices) <= 255 and len(s.triangles[0]) <= 255 for s in x.sprites), p.name
        if k.anim:
            assert anim_invariants(k.anim) == [], p.name
    for p in (FILES / folder).glob("*.png"):
        assert max(Image.open(p).size) <= 128, p.name        # the engine's texture ceiling


@need_files
def test_reaper_files_follow_hornys_frames():
    from scripts.dk2kmf import parse_kmf
    from scripts.hreaper_files import file_plan
    for _slot, name, src in file_plan():
        mine = parse_kmf((FILES / "hreaper" / f"{name}.kmf").read_bytes())
        low = {p.name.lower(): p for p in MESHES.glob("REA3*")} | {p.name.lower(): p for p in MESHES.glob("Rea3*")}
        theirs = parse_kmf(low[f"{src}.kmf".lower()].read_bytes())
        assert (mine.kind, bool(mine.anim and mine.anim.wrap)) == (theirs.kind, bool(theirs.anim and theirs.anim.wrap))
        if mine.anim:
            assert mine.anim.frame_count == theirs.anim.frame_count
        assert not [m for m in mine.materials if m.textures[0].lower().startswith("reaperface")], name


@pytest.mark.skipif(not Path("/home/claude/cast/panel/orc.png").exists(), reason="no panel pictures")
def test_panel_pictures():
    from PIL import Image
    for icon in ("orc", "hreaper", "dragon", "sejanus", "gargoyle"):
        assert Image.open(f"/home/claude/cast/panel/{icon}.png").size == (64, 64)
        assert Image.open(f"/home/claude/cast/panel/{icon}-bg.png").size == (64, 128)


@need_base
@need_files
def test_sejanus_record():
    from scripts.cast_records import SEJANUS, build
    from scripts.dk2kwd_creatures import ANIM_SLOTS, ART_FLAG_HAS_END_ANIMATION as E, ART_FLAG_HAS_START_ANIMATION as S
    table = _table(BASE / "Data/editor/Creatures.kwd")
    info = _infos()[52]
    n = build(SEJANUS, table, 52, info, {"name": 1, "general": 2, "strength": 3, "weakness": 4})
    by = {c.get("creatureId"): c for c in table}
    # speeds matched to his clips (0.654 and 2.049 units a second), the run above every game creature's
    assert (n.get("speed"), n.get("runSpeed")) == (2679, 8393)
    assert n.get("runSpeed") > max(c.get("runSpeed") for i, c in by.items() if i <= 46)
    # mid-tier: between the Dark Elf and the Vampire
    assert by[6].get("hp") < n.get("hp") < by[8].get("hp")
    assert by[6].get("meleeDamage") < n.get("meleeDamage") < by[8].get("meleeDamage")
    assert n.get("lairObjectId") == 18                          # the bed in his Sleep clip
    assert (n.get("spells[0].creatureSpellId"), n.get("spells[0].levelAvailable")) == (13, 4)
    assert 0 < n.get("spells[0].shotDelay") < info["Sejanus-Magic"]["frames"] / 30 * 4096
    for slot in ANIM_SLOTS[1:]:
        name, t = n.resource_name(slot), n.art_tail(slot)
        if not name:
            continue
        assert name in info, slot
        if t.type == 6:
            assert t.frames == info[name]["frames"] - (1 if info[name]["wrap"] else 0), slot
            if t.flags & S:
                assert t.start_af == info[name + "Start"]["frames"], slot
            if t.flags & E:
                assert t.end_af == info[name + "End"]["frames"], slot


@need_base
@need_files
def test_gargoyle_record():
    from scripts.cast_records import GARGOYLE, build
    from scripts.dk2kwd_creatures import ANIM_SLOTS, ART_FLAG_HAS_END_ANIMATION as E, ART_FLAG_HAS_START_ANIMATION as S
    table = _table(BASE / "Data/editor/Creatures.kwd")
    info = _infos()[53]
    n = build(GARGOYLE, table, 53, info, {"name": 1, "general": 2, "strength": 3, "weakness": 4})
    gob = next(c for c in table if c.get("creatureId") == 7)
    assert (n.get("speed"), n.get("runSpeed")) == (1323, 3002)       # his clips' 0.323 / 0.733 units/s
    assert n.get("hp") > gob.get("hp") and n.get("pay") < gob.get("pay")
    assert n.get("lairObjectId") == 18                                # the Bile Demon's nest (as the Dragon)
    assert all(n.get(f"spells[{i}].creatureSpellId") == 0 for i in range(3))
    for slot in ANIM_SLOTS[1:]:
        name, t = n.resource_name(slot), n.art_tail(slot)
        if not name:
            continue
        assert name in info, slot
        if t.type == 6:
            assert t.frames == info[name]["frames"] - (1 if info[name]["wrap"] else 0), slot
            if t.flags & S:
                assert t.start_af == info[name + "Start"]["frames"], slot
            if t.flags & E:
                assert t.end_af == info[name + "End"]["frames"], slot


@pytest.mark.skipif(not (FILES / "orc/Orc-prayStart.kmf").exists(), reason="cast files not built")
def test_orc_portrait_pose_moves_only_his_arms():
    from scripts.cast_portraits import SHOTS, posed_frame
    from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
    import numpy as np
    shot = SHOTS["orc"]
    anim = parse_kmf((FILES / "orc" / f"{shot.clip}.kmf").read_bytes()).anim
    plain, posed = evaluate_anim_frame(anim, shot.frame), posed_frame(anim, shot.frame, shot.pose)
    assert all(np.array_equal(a, b) for a, b in zip(plain, posed_frame(anim, shot.frame, ())))
    rest = evaluate_anim_frame(anim, 0)
    parts = {e[1] for e in shot.pose if e[0] == "part"}           # the belt knob, moved on purpose
    for k, (a, b, r) in enumerate(zip(plain, posed, rest)):
        if k in parts:
            continue
        still = np.linalg.norm(a - r, axis=1) <= 0.04          # his body and head: as the game draws them
        assert np.allclose(a[still], b[still], atol=1e-6)
    assert max(np.abs(a - b).max() for a, b in zip(plain, posed)) > 0.05   # the arms did turn


SKIN_INSTALL = Path("/home/claude/cast/install_sejanus_skin")


@pytest.mark.skipif(not (GARGOYLE_INSTALL / "MoreCreatures").exists() or not Path("/home/claude/gargoyle/build").exists(),
                    reason="no Gargoyle install or build")
def test_gargoyle_install_has_the_build_files():
    from scripts.cast_records import GARGOYLE, all_files
    for n in all_files(GARGOYLE):
        latest = Path("/home/claude/cast/install_gargoyle_nest/MoreCreatures/Data/Meshes") / f"{n}.kmf"
        if not latest.exists():                                          # the nest round re-ships his Sleep
            latest = GARGOYLE_INSTALL / f"MoreCreatures/Data/Meshes/{n}.kmf"
        assert latest.read_bytes() == \
            Path(f"/home/claude/gargoyle/build/{n}.kmf").read_bytes(), n        # Pray polish (6a720d5) included


@pytest.mark.skipif(not (SKIN_INSTALL / "report.json").exists(), reason="no Sejanus skin install")
def test_sejanus_skin_install():
    from PIL import Image
    rep = json.loads((SKIN_INSTALL / "report.json").read_text())
    assert rep["new"] == [] and len(rep["files"]) == len(rep["replaces"]) == 4
    for n in ("Body", "Head", "Limbs"):
        rel = f"Data/EngineTextures/Sejanus_{n}.png"
        p = SKIN_INSTALL / "MoreCreatures" / rel
        assert p.read_bytes() == Path(f"/home/claude/sejanus/recolour_B/Sejanus_{n}.png").read_bytes()
        old = Image.open(INSTALLED / rel)
        assert (Image.open(p).size, Image.open(p).mode) == (old.size, old.mode) == ((128, 128), "RGB")
    bg = Image.open(SKIN_INSTALL / "MoreCreatures/Data/EngineTextures/GUI/Creatures/Evil/sejanus-bg.png")
    assert bg.size == (64, 128)


@pytest.mark.skipif(not (Path("/home/claude/cast/install_portraits/report.json")).exists(), reason="no portraits install")
def test_portraits_install_is_soles_picks():
    import hashlib
    E = Path("/home/claude/cast/install_portraits/MoreCreatures/Data/EngineTextures/GUI/Creatures/Evil")
    assert sorted(p.name for p in E.iterdir()) == ["dragon-bg.png", "hreaper-bg.png", "hreaper.png", "orc-bg.png"]
    for name, picked in (("hreaper.png", "panel_v2"), ("hreaper-bg.png", "panel_v3"), ("dragon-bg.png", "panel_v3")):
        assert (E / name).read_bytes() == Path(f"/home/claude/cast/{picked}/{name}").read_bytes(), name


MPD1_INSTALL = Path("/home/claude/cast/install_mpd1")
MPD1_SRC = UP / "flame/resources/Quuz_LevelFixes/Data/editor/maps"
BASE_R6 = Path("/home/claude/cast/base_r6/MoreCreatures")


@pytest.mark.skipif(not (MPD1_INSTALL / "report.json").exists() or not (MPD1_SRC / "mpd1Things.kld").exists()
                    or not BASE_R6.exists(), reason="no mpd1 install or its staged files")
def test_mpd1_install(tmp_path):
    from scripts.build_molynx_release import _entries
    from scripts.cast_check import check
    from scripts.cast_install import KEEP_CLEAR, MPD1_PLACEMENTS, loaded_copy
    from scripts.dk2kld_splice import Thing, read_things, write_things
    from scripts.dk2texture import TextureCache
    assert loaded_copy(UP, "Data/editor/maps/mpd1Things.kld")[0].parent == MPD1_SRC
    out = MPD1_INSTALL / "MoreCreatures"
    rep = json.loads((MPD1_INSTALL / "report.json").read_text())
    assert rep["new"] == ["Data/editor/maps/mpd1Things.kld"] and rep["replaces"] == ["Data/editor/maps/mpd1Variables.kld"]
    cache = TextureCache(UP / "Data/EngineTextures.dir", UP / "Data/EngineTextures.dat")
    args = (FURNACE, MESHES, CODEPAGE, cache)
    assert check(out, BASE_R6, *args, map_dirs={"mpd1": MPD1_SRC}) == []
    # every thing of the copy the game loads is still there; the cast at level 5, Keeper 1's, no trigger
    _b, before = read_things((MPD1_SRC / "mpd1Things.kld").read_bytes())
    block, after = read_things((out / "Data/editor/maps/mpd1Things.kld").read_bytes())
    assert all(t in after for t in before) and len(after) == len(before) + 7
    mine = {t.data[26]: t for t in after if t.type == 200 and t.data[26] >= 47}
    assert {k: struct.unpack_from("<ii", t.data)[:2] for k, t in mine.items()} == MPD1_PLACEMENTS
    assert all(t.data[14] == 5 and t.data[27] == 3 and struct.unpack_from("<H", t.data, 24)[0] == 0 for t in mine.values())
    assert not set(MPD1_PLACEMENTS.values()) & KEEP_CLEAR["mpd1"]
    # Variables: 49, 51..53 added for Keeper 1, the Reaper (placed only) on no pool, nothing else changed
    a = _entries((BASE_R6 / "Data/editor/maps/mpd1Variables.kld").read_bytes())[1]
    b = _entries((out / "Data/editor/maps/mpd1Variables.kld").read_bytes())[1]
    assert all(e in b for e in a)
    assert sorted({e[1] if e[0] == 1 else e[2] for e in b if e not in a}) == [49, 51, 52, 53]
    assert not [e for e in b if (e[0] == 1 and e[1] == 50) or (e[0] == 2 and e[2] == 50)]
    # the checker catches a creature put in the corridor
    bad = tmp_path / "MoreCreatures"
    shutil.copytree(out, bad)
    i = next(i for i, t in enumerate(after) if t.type == 200 and t.data[26] == 53)
    d = bytearray(after[i].data)
    struct.pack_into("<ii", d, 0, 30, 35)
    after[i] = Thing(200, bytes(d))
    (bad / "Data/editor/maps/mpd1Things.kld").write_bytes(write_things(block, after))
    assert any("creature 53 at (30, 35)" in p for p in check(bad, BASE_R6, *args, map_dirs={"mpd1": MPD1_SRC}))


NEST_INSTALL = Path("/home/claude/cast/install_gargoyle_nest")
BASE_R7 = Path("/home/claude/cast/base_r7/MoreCreatures")


@pytest.mark.skipif(not (NEST_INSTALL / "report.json").exists() or not BASE_R7.exists(), reason="no nest install")
def test_gargoyle_nest_install():
    """Record 53's lairObjectId 22 -> 18 in every table, nothing else; his Sleep carries the nest."""
    from scripts.cast_check import check
    from scripts.dk2kmf import parse_kmf
    from scripts.dk2texture import TextureCache
    out = NEST_INSTALL / "MoreCreatures"
    rep = json.loads((NEST_INSTALL / "report.json").read_text())
    assert rep["new"] == [] and [f for f in rep["files"] if f.endswith(".kmf")] == ["Data/Meshes/Gargoyle-Sleep.kmf"]
    tables = sorted(out.rglob("*.k[wl]d"))
    assert len(tables) == len(list((BASE_R7 / "Data/editor/maps").glob("*Globals.kld"))) + 1
    for f in tables:
        a, b = _table(BASE_R7 / f.relative_to(out)), _table(f)
        for x, y in zip(a, b):
            if x.get("creatureId") != 53:
                assert x.raw == y.raw
                continue
            changed = [i for i, (p, q) in enumerate(zip(x.raw, y.raw)) if p != q]      # one byte: that field
            assert (x.get("lairObjectId"), y.get("lairObjectId")) == (22, 18) and len(changed) == 1
    k = parse_kmf((out / "Data/Meshes/Gargoyle-Sleep.kmf").read_bytes())
    tex = {k.materials[s.material].textures[0] for s in k.anim.sprites}
    assert "Creature Bed_Bile Demon1" in tex and "Creature Bed_Goblin IN" not in tex
    cache = TextureCache(UP / "Data/EngineTextures.dir", UP / "Data/EngineTextures.dat")
    assert check(out, BASE_R7, FURNACE, MESHES, CODEPAGE, cache, map_dirs={"mpd1": MPD1_SRC}) == []
