"""The Molynx mod: the builder (`build_mod`, stages 1 and 2), the independent audit (`audit`) and
the test maps.

The builds read the real DK2 maps, so every test that needs them skips when they are absent. The
stage-2 builds take Molynx's files from a fresh export (molynx_assets.export_molynx_assets) made
once per module.
"""
import re
import shutil
import struct
import tomllib
from pathlib import Path

import pytest

from PIL import Image

from molynx_files import ANIM_INFO, FILE_WRAPS
import scripts.audit_molynx_mod as audit_module
import scripts.build_molynx_mod as builder
import scripts.molynx_assets as molynx_assets
from scripts.audit_molynx_mod import audit
from scripts.build_molynx_mod import build_mod
from scripts.dk2kld_splice import (MAP_SUFFIXES, level_flags, map_source_dir, pick_skirmish_test_map,
                                   pick_test_map)
from scripts.dk2kwd_blocks import HEADER_SIZE, Block, read_blocks, write_blocks
from scripts.dk2kmf import parse_kmf
from scripts.dk2kmf_write import write_kmf
from scripts.dk2kwd_creatures import ANIM_SLOTS, FIELDS, Creature, creatures_block, read_creatures
from scripts.dk2paths import BEHEMOTH_DIR, EDITOR_DIR, GAME_DIR, MAPS_DIR, MESHES_DIR, QUUZ_MAPS_DIR, have

needs_data = pytest.mark.skipif(
    not (have(MAPS_DIR) and have(QUUZ_MAPS_DIR) and have(GAME_DIR / "Meshes.WAD")),
    reason="DK2 editor maps, Quuz maps or Meshes.WAD not present")
needs_assets = pytest.mark.skipif(
    not (have(MESHES_DIR) and have(EDITOR_DIR) and have(GAME_DIR / "EngineTextures.dir")
         and have(BEHEMOTH_DIR / "behemoth.xml")),
    reason="the data Molynx's files are exported from is not present")

CREATURES = 0xAA
SKIRMISH_FLAG = 1 << 11
MULTIPLAYER_FLAG = 1 << 10
FOLDER = "Molynx_slot_only"
SECOND = "Dead Walking"  # the skirmish map whose first keeper Bile Demon is Keeper 1's
MAPS = ["Siege", SECOND]
TEXT_IDS = {"name": 3001, "tooltip": 3002, "general": 3003, "strength": 3004, "weakness": 3005}
# The files a stage-2 folder delivers (literal; the record's file plan for the test maps' Bile Demons):
# 18 clips (staggering only for Siege's type NONE dejectedPose slot), the variant walk_once, the 5
# poses (stunnedPose and elec take static poses), and the 2 textures.
STAGE2_MESHES = sorted(f"Molynx-{stem}.kmf" for stem in (
    "walk", "walk_once", "DraggedPose", "getting_hit", "StunnedPose", "ElecPose", "attack", "die",
    "DiePose", "happy", "angry", "staggering", "being_picked_up", "sleeping", "eating", "stand",
    "being_slapped", "manufacturing", "praying", "knockback", "waking_up", "celebrating",
    "fidgeting", "PoseFrame"))
STAGE2_TEXTURES = ["Molynx_Body.png", "Molynx_Mace.png"]
UNUSED_CLIPS = ["running", "victory_celebration", "training", "fighting_another_creature",
                "patrolling", "guarding", "interacting_with_objects"]
TAIL_AT = {"flags": (0, 4), "frames": (4, 4), "fps": (8, 4), "startDist": (12, 2),
           "endDist": (14, 2), "type": (16, 1), "startAf": (17, 1), "endAf": (18, 1),
           "sometimesOne": (19, 1)}  # the 20 bytes after a slot's name: (offset, size)
# slot-table.md (plan Task 8), copied here independently of the record module and the audit, plus
# the rulings (entrance -> walk_once; the static mesh slots stunnedPose and elec take the static
# poses StunnedPose and ElecPose); other slots: stand.
SLOT_TABLE = """walk->walk, run->running, draggedPose->DraggedPose, recoilHff->getting_hit,
recoilHfb->being_slapped, melee1/melee2/special->attack, die->die, diePose->DiePose, happy->happy,
angry->angry, swing->being_picked_up, sleep->sleeping, eat/eatCorpse/drink->eating,
building->manufacturing, pray->praying, fallback->knockback, getUp->waking_up, dance->celebrating,
drunk/drunk2/stunnedPose/dejectedPose->staggering, idle1->stand, idle2/drunkIdle->fidgeting,
poseFrame->PoseFrame, walk2->patrolling, entrance/walkback->walk,
research/loot->interacting_with_objects, magic/torture/elec/electrocute->getting_hit"""
VARIANTS = {"entrance": "walk_once", "stunnedPose": "StunnedPose", "elec": "ElecPose"}


def expected_slot_files() -> dict[str, str]:
    """Record field -> stem, from SLOT_TABLE and VARIANTS."""
    out = {}
    for item in " ".join(SLOT_TABLE.split()).split(", "):
        shorts, stem = item.split("->")
        for short in shorts.split("/"):
            field = short if short == "drunkIdle" else f"anim{short[0].upper()}{short[1:]}Resource"
            out[field] = VARIANTS.get(short, stem)
    return out


# ---- helpers ------------------------------------------------------------------------------------

def maps_dir(mod: Path) -> Path:
    return mod / "Data" / "editor" / "maps"


def fail_lines(report, rule: str | None = None, map_name: str | None = None) -> list[str]:
    prefix = "FAIL" + (f" ({rule})" if rule else "")
    return [line for line in report.lines
            if line.startswith(prefix) and (map_name is None or f" {map_name}:" in line)]


def pass_lines(report, rule: str, map_name: str) -> list[str]:
    return [line for line in report.lines if line.startswith(f"PASS ({rule}) {map_name}:")]


def edit_globals(mod: Path, map_name: str, edit) -> None:
    """Rewrite `<map>Globals.kld` of the mod folder with `edit(list[Block]) -> list[Block]`."""
    path = maps_dir(mod) / f"{map_name}Globals.kld"
    path.write_bytes(write_blocks(edit(read_blocks(path.read_bytes()))))


def edit_creatures(mod: Path, map_name: str, edit) -> None:
    """Rewrite the Creatures block of a mod map with `edit(list[Creature]) -> list[Creature]`."""
    def on_blocks(blocks: list[Block]) -> list[Block]:
        (at,) = [i for i, b in enumerate(blocks) if b.signature == CREATURES]
        blocks[at] = creatures_block(edit(read_creatures(blocks[at])), blocks[at])
        return blocks
    edit_globals(mod, map_name, on_blocks)


def flip_byte(path: Path, index: int) -> None:
    data = bytearray(path.read_bytes())
    data[index] ^= 0xFF
    path.write_bytes(bytes(data))


def record_47(mod: Path, map_name: str) -> Creature:
    blocks = read_blocks((maps_dir(mod) / f"{map_name}Globals.kld").read_bytes())
    (block,) = [b for b in blocks if b.signature == CREATURES]
    return read_creatures(block)[-1]


def source_bile(map_name: str) -> Creature:
    folder = map_source_dir(map_name)
    blocks = read_blocks((folder / f"{map_name}Globals.kld").read_bytes())
    (block,) = [b for b in blocks if b.signature == CREATURES]
    (bile,) = [c for c in read_creatures(block) if c.get("creatureId") == 3]
    return bile


def creature_things(things: bytes) -> list[tuple[int, int, int]]:
    """(record offset, creatureId, owner) of every type-200 creature thing, by a plain walk of the file."""
    pos, found = HEADER_SIZE, []
    while pos < len(things):
        kind, size = struct.unpack_from("<II", things, pos)
        if kind == 200:
            found.append((pos, things[pos + 34], things[pos + 35]))
        pos += 8 + size
    return found


def with_creature_id(things: bytes, index: int, creature_id: int) -> bytes:
    """`things` with the creature id byte of its `index`-th creature thing set to `creature_id`."""
    data = bytearray(things)
    data[creature_things(things)[index][0] + 34] = creature_id
    return bytes(data)


def differing_fields(a: bytes, b: bytes) -> set[str]:
    return {name for name, f in FIELDS.items() if f.end <= len(a) and a[f.offset:f.end] != b[f.offset:f.end]}


def add_placeholder_meshes(mod: Path, names: set[str]) -> None:
    meshes = mod / "Data" / "Meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    for name in names:
        (meshes / f"{name}.kmf").write_bytes(b"placeholder " + name.encode())


def renamed_slots(mod: Path, map_name: str) -> set[str]:
    """The animation names record 47 uses that differ from the Bile Demon's (what stage 2 delivers)."""
    molynx, bile = record_47(mod, map_name), source_bile(map_name)
    return {molynx.resource_name(s) for s in ANIM_SLOTS
            if s != "unk_1" and molynx.resource_name(s) != bile.resource_name(s)}


@pytest.fixture(scope="module")
def stage1_dir(tmp_path_factory):
    return build_mod(1, tmp_path_factory.mktemp("stage1"), FOLDER)


@pytest.fixture
def mod(stage1_dir, tmp_path):
    """A private, writable copy of the stage-1 folder that a test may damage."""
    target = tmp_path / FOLDER
    shutil.copytree(stage1_dir, target)
    return target


@pytest.fixture(scope="module")
def assets_dir(tmp_path_factory):
    """A fresh export of Molynx's files (models, variants, poses, textures; no portraits)."""
    out = tmp_path_factory.mktemp("assets") / "molynx_assets"
    molynx_assets.export_molynx_assets(out, portrait_src=None)
    return out


@pytest.fixture
def assets_copy(assets_dir, tmp_path):
    """A private, writable copy of the export that a test may damage."""
    target = tmp_path / "assets_copy"
    shutil.copytree(assets_dir, target)
    return target


@pytest.fixture(scope="module")
def stage2_dir(tmp_path_factory, assets_dir):
    return build_mod(2, tmp_path_factory.mktemp("stage2"), "Molynx_stage2", maps=["Siege"],
                     assets_dir=assets_dir)


@pytest.fixture(scope="module")
def stage2_full(tmp_path_factory, assets_dir):
    """The stage-2 folder as it is handed over: name Molynx, both test maps."""
    return build_mod(2, tmp_path_factory.mktemp("stage2_full"), assets_dir=assets_dir)


@pytest.fixture
def mod2(stage2_dir, tmp_path):
    target = tmp_path / "Molynx_stage2"
    shutil.copytree(stage2_dir, target)
    return target


# ---- the test maps ------------------------------------------------------------------------------

@needs_data
def test_test_maps_are_pinned():
    assert pick_test_map() == "Siege"
    assert pick_skirmish_test_map() == SECOND
    assert builder.test_maps() == MAPS


@needs_data
def test_the_users_siege_copy_has_no_skirmish_flag_but_the_second_map_does():
    # Why a second test map exists: the Quuz Siege.kwd (the copy the game loads) is a multiplayer
    # map but not a skirmish map, so it may not show in the Skirmish list; the second map does.
    siege = (map_source_dir("Siege") / "Siege.kwd").read_bytes()
    second = (map_source_dir(SECOND) / f"{SECOND}.kwd").read_bytes()
    assert level_flags(siege) == 0x404
    assert level_flags(siege) & MULTIPLAYER_FLAG and not level_flags(siege) & SKIRMISH_FLAG
    assert level_flags(second) == 0xC04
    assert level_flags(second) & SKIRMISH_FLAG


@needs_data
def test_level_flags_are_the_u16_at_byte_25540():
    data = (map_source_dir("Siege") / "Siege.kwd").read_bytes()
    assert level_flags(data) == struct.unpack_from("<H", data, 56 + 25484)[0]
    with pytest.raises(ValueError):
        level_flags(data[:1000])


def _first_keeper_bile_owner(things: bytes) -> int | None:
    """Owner of the first type-200 creature thing with creatureId 3 owned by a keeper (3..7), or None;
    a plain walk of the Things records, independent of the module under test."""
    pos = HEADER_SIZE
    while pos < len(things):
        kind, size = struct.unpack_from("<II", things, pos)
        if kind == 200 and things[pos + 34] == 3 and 3 <= things[pos + 35] <= 7:
            return things[pos + 35]
        pos += 8 + size
    return None


def _candidates_recomputed():
    """(not in both, total bytes, name, level flags, owner of the first keeper Bile Demon) of the 46x5537
    maps that have a keeper's Bile Demon, recomputed here from the files."""
    def complete(folder):
        return {p.stem for p in folder.glob("*.kwd")
                if all((folder / (p.stem + s)).is_file() for s in MAP_SUFFIXES)}

    quuz, editor = complete(QUUZ_MAPS_DIR), complete(MAPS_DIR)
    rows = []
    for name in quuz | editor:
        folder = QUUZ_MAPS_DIR if name in quuz else MAPS_DIR
        paths = [folder / (name + s) for s in MAP_SUFFIXES]
        (creatures,) = [b for b in read_blocks(paths[6].read_bytes()) if b.signature == CREATURES]
        if (creatures.item_count, len(creatures.content) // creatures.item_count) != (46, 5537):
            continue
        owner = _first_keeper_bile_owner(paths[2].read_bytes())
        if owner is None:
            continue
        both = all((d / (name + ".kwd")).is_file() for d in (QUUZ_MAPS_DIR, MAPS_DIR))
        flags = struct.unpack_from("<H", paths[0].read_bytes(), 56 + 25484)[0]
        rows.append((not both, sum(p.stat().st_size for p in paths), name, flags, owner))
    return rows


@needs_data
def test_skirmish_pick_recomputed_independently():
    rows = _candidates_recomputed()
    assert len(rows) == 41
    others = [r for r in rows if r[2] != pick_test_map() and r[3] & SKIRMISH_FLAG]
    assert len(others) == 30
    # the smallest skirmish map alone would be 13coop, but its first keeper Bile Demon is Keeper 4's
    assert min(others)[2] == "13coop" and min(others)[4] == 6
    # the second map must also put Molynx in the dungeon of Keeper 1 (player 3), the human's
    keeper_1 = [r for r in others if r[4] == 3]
    assert [r[2] for r in keeper_1] == [SECOND], "Dead Walking is the only such skirmish map"
    assert min(keeper_1)[2] == pick_skirmish_test_map() == SECOND
    assert min(keeper_1)[1] == 745_684
    assert (map_source_dir(SECOND) / f"{SECOND}.kwd").parent == MAPS_DIR


@needs_data
def test_the_first_test_map_is_not_changed_by_the_owner_criterion():
    rows = _candidates_recomputed()
    assert min(rows)[2] == pick_test_map() == "Siege"
    assert [r[4] for r in rows if r[2] == "Siege"] == [3]


# ---- build_mod: refusals ------------------------------------------------------------------------

@pytest.mark.parametrize("stage", [0, 4, -1, 2.5, "1", None])
def test_build_refuses_a_bad_stage(tmp_path, stage):
    with pytest.raises(ValueError, match="stage"):
        build_mod(stage, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("stage", [2, 3])
def test_build_stage_2_and_3_need_an_assets_dir(tmp_path, stage):
    for extra in ({}, {"anim_info": ANIM_INFO, "file_wraps": FILE_WRAPS}):
        with pytest.raises(ValueError, match="assets_dir"):
            build_mod(stage, tmp_path, maps=["Siege"], text_ids=TEXT_IDS, **extra)
    assert list(tmp_path.iterdir()) == []


def test_build_stage_1_takes_no_assets_dir(tmp_path):
    with pytest.raises(ValueError, match="assets_dir"):
        build_mod(1, tmp_path, maps=["Siege"], assets_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_build_stage_3_needs_text_ids(tmp_path):
    with pytest.raises(ValueError, match="text_ids"):
        build_mod(3, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["", "a.b", "a;b", "a/b", "a\\b", "a b", "..", "Molynx*"])
def test_build_refuses_a_name_flame_cannot_list(tmp_path, name):
    # alt-resources is split at ';', and the launcher skips folders with a '.' in the name
    with pytest.raises(ValueError, match="name"):
        build_mod(1, tmp_path, name)
    assert list(tmp_path.iterdir()) == []


def test_build_refuses_an_existing_folder_and_leaves_it_alone(tmp_path):
    existing = tmp_path / "Molynx"
    existing.mkdir()
    (existing / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        build_mod(1, tmp_path, maps=["Siege"])
    assert [p.name for p in existing.iterdir()] == ["keep.txt"]
    assert (existing / "keep.txt").read_text() == "mine"


@needs_data
def test_build_refuses_a_missing_map_without_leaving_a_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_mod(1, tmp_path, maps=["Siege", "No Such Map"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("maps", [[], ["Siege", "Siege"], ["../Siege"], ["a\\b"], [""]])
def test_build_refuses_an_empty_or_repeated_map_list(tmp_path, maps):
    with pytest.raises(ValueError, match="maps"):
        build_mod(1, tmp_path, maps=maps)
    assert list(tmp_path.iterdir()) == []


# ---- build_mod: stage 1 -------------------------------------------------------------------------

@needs_data
def test_stage1_folder_lists_exactly_the_expected_files(stage1_dir):
    assert stage1_dir.name == FOLDER
    found = sorted(p.relative_to(stage1_dir).as_posix() for p in stage1_dir.rglob("*") if p.is_file())
    expected = sorted(["resource.toml", *(f"Data/editor/maps/{name}{suffix}"
                                          for name in MAPS for suffix in MAP_SUFFIXES)])
    assert found == expected
    assert len(found) == 1 + 2 * 7
    assert sorted(p.relative_to(stage1_dir).as_posix() for p in stage1_dir.rglob("*") if p.is_dir()) \
        == ["Data", "Data/editor", "Data/editor/maps"]


@needs_data
def test_stage1_changes_only_globals_and_things_of_each_map(stage1_dir):
    for name in MAPS:
        source = map_source_dir(name)
        for suffix in MAP_SUFFIXES:
            same = (maps_dir(stage1_dir) / (name + suffix)).read_bytes() == (source / (name + suffix)).read_bytes()
            assert same == (suffix not in ("Globals.kld", "Things.kld")), (name, suffix)


@needs_data
def test_stage1_record_47_is_the_bile_demon_with_three_fields_changed(stage1_dir):
    for name in MAPS:
        molynx, bile = record_47(stage1_dir, name), source_bile(name)
        assert len(molynx.raw) == len(bile.raw) == 5537
        assert differing_fields(bytes(molynx.raw), bytes(bile.raw)) == {"name", "creatureId", "orderInEditor"}
        assert molynx.get("name") == "Molynx" and molynx.get("creatureId") == 47
        assert molynx.get("orderInEditor") == 47


@needs_data
def test_molynx_stands_in_keeper_1s_dungeon_on_every_test_map(stage1_dir):
    for name in MAPS:
        things = (maps_dir(stage1_dir) / f"{name}Things.kld").read_bytes()
        pos, owners = HEADER_SIZE, []
        while pos < len(things):
            kind, size = struct.unpack_from("<II", things, pos)
            if kind == 200 and things[pos + 34] == 47:
                owners.append(things[pos + 35])
            pos += 8 + size
        assert owners == [3], name  # exactly one Molynx thing, owned by player 3 = "Keeper 1"


@needs_data
def test_resource_toml_has_the_format_of_the_real_mods(stage1_dir):
    raw = (stage1_dir / "resource.toml").read_bytes()
    assert b"\r\n" in raw and not re.search(rb"(?<!\r)\n", raw), "the real mods use CRLF line ends"
    data = tomllib.loads(raw.decode("ascii"))
    assert list(data) == ["resource"]
    assert list(data["resource"]) == ["name", "version", "authors", "description"]
    assert data["resource"]["name"] == FOLDER
    assert all(isinstance(v, str) for v in data["resource"].values())
    real = QUUZ_MAPS_DIR.parents[2] / "resource.toml"
    if real.is_file():
        assert list(tomllib.loads(real.read_text())["resource"])[:4] == list(data["resource"])


@needs_data
def test_building_twice_gives_identical_bytes(stage1_dir, tmp_path):
    again = build_mod(1, tmp_path, FOLDER)
    files = sorted(p.relative_to(stage1_dir) for p in stage1_dir.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    assert all((stage1_dir / f).read_bytes() == (again / f).read_bytes() for f in files)


@needs_data
def test_build_main_builds_the_slot_only_folder_by_default(tmp_path, capsys):
    assert builder.main(["--out", str(tmp_path / "bundle")]) == 0
    assert (tmp_path / "bundle" / FOLDER / "resource.toml").is_file()
    assert capsys.readouterr().out.strip().endswith(FOLDER)


@needs_data
def test_default_name_is_molynx_and_default_maps_are_the_test_maps(tmp_path):
    out = build_mod(1, tmp_path)
    assert out == tmp_path / "Molynx"
    names = {p.name[:-len("Globals.kld")] for p in maps_dir(out).glob("*Globals.kld")}
    assert names == set(MAPS)


# ---- audit: a good stage-1 folder ---------------------------------------------------------------

@needs_data
def test_stage1_audit_is_ok_and_reports_every_rule_for_every_map(stage1_dir):
    report = audit(stage1_dir)
    assert report.ok, "\n".join(report.lines)
    assert not fail_lines(report)
    assert all(line.startswith("PASS") for line in report.lines)
    for name in MAPS:
        for rule in "abcdef":
            assert len(pass_lines(report, rule, name)) == 1, (rule, name, report.lines)
    for rule in "gh":
        assert [line for line in report.lines if line.startswith(f"PASS ({rule}) folder:")]


@needs_data
def test_audit_infers_stage_1_from_the_folder(stage1_dir):
    report = audit(stage1_dir)
    assert any("stage 1" in line for line in report.lines if line.startswith("PASS (b)"))
    assert not any("stage 2" in line or "stage 3" in line for line in report.lines)


@needs_data
def test_audit_does_not_grade_the_builder_with_itself():
    source = Path(audit_module.__file__).read_text()
    for forbidden in ("build_molynx", "molynx_record", "build_mod", "splice_creatures",
                      "molynx_for_map", "retarget_creature_thing", "file_plan", "molynx_assets",
                      "slot_assignment", "SLOT_CLIPS", "art_tail", "anim_info"):
        assert forbidden not in source, forbidden


def test_audit_of_a_missing_folder_fails(tmp_path):
    report = audit(tmp_path / "nothing")
    assert not report.ok
    assert fail_lines(report, "g")


def test_audit_main_returns_the_verdict_as_an_exit_code(tmp_path, capsys):
    assert audit_module.main([str(tmp_path / "nothing")]) == 1
    assert "FAIL" in capsys.readouterr().out


@needs_data
def test_audit_main_prints_every_line_of_a_good_folder(stage1_dir, capsys):
    assert audit_module.main([str(stage1_dir)]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and out.count("PASS") >= 2 * 6 + 2


# ---- audit: damaged stage-1 folders -------------------------------------------------------------

@needs_data
def test_a_corrupted_byte_in_a_copied_block_names_the_block(mod):
    def corrupt_terrain(blocks):
        (at,) = [i for i, b in enumerate(blocks) if b.signature == 0x6E]
        content = bytearray(blocks[at].content)
        content[len(content) // 2] ^= 0x01
        blocks[at] = Block(0x6E, blocks[at].header, bytes(content))
        return blocks

    edit_globals(mod, "Siege", corrupt_terrain)
    report = audit(mod)
    assert not report.ok
    (line,) = fail_lines(report, "a")
    assert "Siege" in line and "Terrain" in line and "0x6E" in line
    assert not fail_lines(report, "a", SECOND)
    assert not fail_lines(report, "b") and not fail_lines(report, "c")


@needs_data
@pytest.mark.parametrize("signature,label", [(0x78, "Rooms"), (0x8C, "Doors"), (0xF0, "Objects"),
                                             (0x10E, "Effects"), (0xFA, "EffectElements")])
def test_a_corrupted_header_of_any_other_block_names_that_block(mod, signature, label):
    def corrupt_header(blocks):
        (at,) = [i for i, b in enumerate(blocks) if b.signature == signature]
        header = bytearray(blocks[at].header)
        header[30] ^= 0x01  # inside the 'created' timestamp
        blocks[at] = Block(signature, bytes(header), blocks[at].content)
        return blocks

    edit_globals(mod, SECOND, corrupt_header)
    report = audit(mod)
    (line,) = fail_lines(report, "a")
    assert SECOND in line and label in line


@needs_data
def test_a_dropped_block_fails_rule_a(mod):
    edit_globals(mod, "Siege", lambda blocks: [b for b in blocks if b.signature != 0x96])
    report = audit(mod)
    assert fail_lines(report, "a", "Siege")


@needs_data
def test_a_changed_byte_in_a_copied_map_file_names_the_file(mod):
    flip_byte(maps_dir(mod) / "SiegeMap.kld", 1000)
    report = audit(mod)
    assert not report.ok
    (line,) = fail_lines(report, "e")
    assert "Siege" in line and "SiegeMap.kld" in line
    assert not fail_lines(report, "e", SECOND)


@needs_data
@pytest.mark.parametrize("suffix", [".kwd", "Players.kld", "Triggers.kld", "Variables.kld"])
def test_every_other_copied_file_is_compared_with_its_source(mod, suffix):
    flip_byte(maps_dir(mod) / f"{SECOND}{suffix}", 60)
    report = audit(mod)
    (line,) = fail_lines(report, "e")
    assert f"{SECOND}{suffix}" in line


@needs_data
def test_a_things_file_with_two_bytes_changed_fails(mod):
    path = maps_dir(mod) / "SiegeThings.kld"
    flip_byte(path, len(path.read_bytes()) - 1)  # one more byte besides the retargeted creature id
    report = audit(mod)
    (line,) = fail_lines(report, "d")
    assert "Siege" in line and "2 bytes" in line


@needs_data
def test_a_things_file_that_is_the_untouched_source_fails(mod):
    shutil.copyfile(map_source_dir("Siege") / "SiegeThings.kld", maps_dir(mod) / "SiegeThings.kld")
    report = audit(mod)
    (line,) = fail_lines(report, "d")
    assert "Siege" in line and "0 bytes" in line


@needs_data
def test_a_things_file_retargeted_to_another_id_fails(mod):
    path = maps_dir(mod) / f"{SECOND}Things.kld"
    data = bytearray(path.read_bytes())
    (at,) = [i for i, (a, b) in enumerate(zip(data, (map_source_dir(SECOND) / f"{SECOND}Things.kld").read_bytes()))
             if a != b]
    data[at] = 46
    path.write_bytes(bytes(data))
    report = audit(mod)
    (line,) = fail_lines(report, "d")
    assert SECOND in line and "47" in line


@needs_data
def test_a_changed_byte_that_is_not_a_creature_id_fails(mod):
    path = maps_dir(mod) / "SiegeThings.kld"
    source = (map_source_dir("Siege") / "SiegeThings.kld").read_bytes()
    data = bytearray(source)
    data[HEADER_SIZE + 8] ^= 0x01  # an x coordinate of the first thing, not an id byte
    path.write_bytes(bytes(data))
    (line,) = fail_lines(audit(mod), "d")
    assert "Siege" in line


@needs_data
def test_a_changed_creature_id_on_a_creature_that_is_not_a_bile_demon_fails(mod):
    """Retargeting must hit a Bile Demon: the same byte of another creature thing is wrong."""
    source = (map_source_dir("Siege") / "SiegeThings.kld").read_bytes()
    pos, found = HEADER_SIZE, None
    while pos < len(source):
        kind, size = struct.unpack_from("<II", source, pos)
        if kind == 200 and source[pos + 34] != 3:
            found = pos + 34
            break
        pos += 8 + size
    assert found is not None
    data = bytearray(source)
    data[found] = 47
    (maps_dir(mod) / "SiegeThings.kld").write_bytes(bytes(data))
    (line,) = fail_lines(audit(mod), "d")
    assert "Siege" in line


@needs_data
def test_dead_walking_has_two_keeper_bile_demons_and_the_first_is_keeper_1s():
    things = (map_source_dir(SECOND) / f"{SECOND}Things.kld").read_bytes()
    found = creature_things(things)
    assert [(i, owner) for i, (_pos, cid, owner) in enumerate(found) if cid == 3] == [(24, 3), (25, 4)]


@needs_data
def test_rule_d_names_the_retargeted_bile_demon_and_its_owner(stage1_dir):
    report = audit(stage1_dir)
    (siege,) = pass_lines(report, "d", "Siege")
    (second,) = pass_lines(report, "d", SECOND)
    assert "#12" in siege and "first keeper-owned Bile Demon" in siege and "Keeper 1 (player 3)" in siege
    assert "#24" in second and "first keeper-owned Bile Demon" in second and "Keeper 1 (player 3)" in second


@needs_data
def test_retargeting_the_second_keeper_bile_demon_fails_naming_both(mod):
    """Dead Walking: thing #24 (owner 3) is the one to retarget; thing #25 (owner 4) is also a keeper's Bile Demon."""
    source = (map_source_dir(SECOND) / f"{SECOND}Things.kld").read_bytes()
    (maps_dir(mod) / f"{SECOND}Things.kld").write_bytes(with_creature_id(source, 25, 47))
    report = audit(mod)
    (line,) = fail_lines(report)
    assert line.startswith(f"FAIL (d) {SECOND}:")
    assert "#25" in line and "player 4" in line and "#24" in line and "first" in line


@needs_data
def test_a_first_keeper_bile_demon_that_is_not_keeper_1s_fails(tmp_path):
    """13coop: the first keeper Bile Demon is Keeper 4's (player 6); the human's side is player 3."""
    out = build_mod(1, tmp_path, "Molynx_13coop", maps=["13coop"])
    (line,) = fail_lines(audit(out))
    assert line.startswith("FAIL (d) 13coop:") and "player 6" in line and "Keeper 1" in line


@needs_data
def test_a_retargeted_bile_demon_that_is_not_a_keepers_fails(mod, tmp_path, monkeypatch):
    """A neutral (player 2) Bile Demon thing, made in a private copy of the source and of the output."""
    real = map_source_dir("Siege")
    src = tmp_path / "source_copy"
    src.mkdir()
    for suffix in MAP_SUFFIXES:
        shutil.copyfile(real / f"Siege{suffix}", src / f"Siege{suffix}")
    things = (src / "SiegeThings.kld").read_bytes()
    ((pos, _cid, _owner),) = [t for t in creature_things(things) if t[1] == 3]
    neutral = bytearray(things)
    neutral[pos + 35] = 2
    (src / "SiegeThings.kld").write_bytes(bytes(neutral))
    neutral[pos + 34] = 47
    (maps_dir(mod) / "SiegeThings.kld").write_bytes(bytes(neutral))
    monkeypatch.setattr(audit_module, "map_source_dir",
                        lambda name: src if name == "Siege" else map_source_dir(name))
    (line,) = fail_lines(audit(mod))
    assert line.startswith("FAIL (d) Siege:") and "player 2" in line and "not a keeper" in line


@needs_data
def test_record_47_with_an_extra_changed_field_fails_naming_it(mod):
    def bump_hp(creatures):
        creatures[-1].set("hp", creatures[-1].get("hp") + 1)
        return creatures

    edit_creatures(mod, "Siege", bump_hp)
    report = audit(mod)
    (line,) = fail_lines(report, "b")
    assert "Siege" in line and "hp" in line
    assert not fail_lines(report, "c")


@needs_data
@pytest.mark.parametrize("field,value,needle", [("name", "Molynx2", "name"),
                                                ("creatureId", 46, "creatureId"),
                                                ("orderInEditor", 46, "orderInEditor"),
                                                ("orderInEditor", 48, "orderInEditor")])
def test_record_47_with_a_wrong_stage_1_value_fails(mod, field, value, needle):
    def wrong(creatures):
        creatures[-1].set(field, value)
        return creatures

    edit_creatures(mod, SECOND, wrong)
    (line,) = fail_lines(audit(mod), "b")
    assert SECOND in line and needle in line


@needs_data
def test_a_name_padded_with_junk_after_the_nul_fails(mod):
    def junk(creatures):
        creatures[-1].raw[31] = 0x41  # last byte of the 32-byte name field
        return creatures

    edit_creatures(mod, "Siege", junk)
    (line,) = fail_lines(audit(mod), "b")
    assert "name" in line


@needs_data
def test_record_47_duplicated_fails(mod):
    edit_creatures(mod, "Siege", lambda creatures: [*creatures, Creature(creatures[-1].raw)])
    report = audit(mod)
    (line,) = fail_lines(report, "b")
    assert "Siege" in line and "48" in line
    assert not fail_lines(report, "c")  # the rebuilt header is consistent; only the count is wrong


@needs_data
def test_record_47_missing_fails(mod):
    edit_creatures(mod, "Siege", lambda creatures: creatures[:-1])
    (line,) = fail_lines(audit(mod), "b")
    assert "46" in line


@needs_data
def test_one_of_the_first_46_records_changed_names_the_record(mod):
    def change(creatures):
        creatures[10].set("hp", creatures[10].get("hp") + 1)
        return creatures

    edit_creatures(mod, "Siege", change)
    (line,) = fail_lines(audit(mod), "b")
    assert "record 11" in line


@needs_data
def test_record_47_that_kept_the_bile_demons_creature_id_fails(mod):
    def rename(creatures):
        creatures[-1].set("creatureId", 3)  # two records with id 3, none with 47
        return creatures

    edit_creatures(mod, "Siege", rename)
    (line,) = fail_lines(audit(mod), "b")
    assert "Siege" in line and "creatureId" in line


@needs_data
@pytest.mark.parametrize("offset,what", [(4, "4"), (12, "check"), (16, "28"), (24, "1"), (48, "check"),
                                         (52, "content size")])
def test_a_broken_header_invariant_in_the_creatures_block_fails_rule_c(mod, offset, what):
    def break_header(blocks):
        (at,) = [i for i, b in enumerate(blocks) if b.signature == CREATURES]
        header = bytearray(blocks[at].header)
        header[offset] ^= 0x01
        blocks[at] = Block(CREATURES, bytes(header), blocks[at].content)
        return blocks

    edit_globals(mod, "Siege", break_header)
    report = audit(mod)
    assert not report.ok
    (line,) = fail_lines(report, "c")
    assert "Siege" in line and what in line


@needs_data
def test_a_changed_timestamp_in_the_creatures_header_fails_rule_c(mod):
    """The check words and sizes are valid, but the header is no longer the source's apart from the counts."""
    def touch_timestamp(blocks):
        (at,) = [i for i, b in enumerate(blocks) if b.signature == CREATURES]
        header = bytearray(blocks[at].header)
        header[40] ^= 0x01
        blocks[at] = Block(CREATURES, bytes(header), blocks[at].content)
        return blocks

    edit_globals(mod, SECOND, touch_timestamp)
    (line,) = fail_lines(audit(mod), "c")
    assert SECOND in line


@needs_data
@pytest.mark.parametrize("extra", [bytes(10), bytes(100)], ids=["short-tail", "zero-block"])
def test_trailing_bytes_after_the_last_globals_block_fail_rules_a_b_c(mod, extra):
    path = maps_dir(mod) / "SiegeGlobals.kld"
    path.write_bytes(path.read_bytes() + extra)
    report = audit(mod)
    for rule in "abc":
        (line,) = fail_lines(report, rule, "Siege")
        assert "does not parse" in line
    assert not fail_lines(report, map_name=SECOND)


@needs_data
def test_an_extra_whole_block_after_the_last_globals_block_fails_rule_a(mod):
    path = maps_dir(mod) / "SiegeGlobals.kld"
    data = path.read_bytes()
    last = read_blocks(data)[-1]
    path.write_bytes(data + last.header + last.content)  # a valid block, one too many
    (line,) = fail_lines(audit(mod), "a", "Siege")
    assert "blocks are" in line


@needs_data
def test_the_globals_of_one_map_in_the_slot_of_the_other_fail(mod):
    shutil.copyfile(maps_dir(mod) / f"{SECOND}Globals.kld", maps_dir(mod) / "SiegeGlobals.kld")
    report = audit(mod)
    (line,) = fail_lines(report, "a", "Siege")
    assert "Terrain" in line
    assert not fail_lines(report, map_name=SECOND)


@needs_data
def test_an_extra_file_fails_rule_g_naming_it(mod):
    (mod / "readme.txt").write_text("hello")
    report = audit(mod)
    (line,) = fail_lines(report, "g")
    assert "readme.txt" in line


@needs_data
def test_an_extra_mesh_file_in_a_stage_1_folder_fails_rule_g(mod):
    add_placeholder_meshes(mod, {"Molynx-walk"})
    (line,) = fail_lines(audit(mod), "g")
    assert "Molynx-walk.kmf" in line


@needs_data
def test_an_extra_map_file_fails_rule_g(mod):
    (maps_dir(mod) / "SiegeExtra.kld").write_bytes(b"x")
    (line,) = fail_lines(audit(mod), "g")
    assert "SiegeExtra.kld" in line


@needs_data
def test_a_missing_map_file_fails_rule_g(mod):
    (maps_dir(mod) / "SiegePlayers.kld").unlink()
    report = audit(mod)
    assert any("SiegePlayers.kld" in line for line in fail_lines(report, "g"))
    assert not report.ok


@needs_data
def test_a_missing_or_malformed_resource_toml_fails_rule_h(mod):
    (mod / "resource.toml").write_bytes(b"[resource\nname = ")
    (line,) = fail_lines(audit(mod), "h")
    assert "resource.toml" in line
    (mod / "resource.toml").unlink()
    assert fail_lines(audit(mod), "h")


@needs_data
def test_a_resource_toml_name_that_is_not_the_folder_name_fails_rule_h(mod):
    toml = (mod / "resource.toml").read_bytes()
    (mod / "resource.toml").write_bytes(toml.replace(f'name = "{FOLDER}"'.encode(), b'name = "Other"'))
    (line,) = fail_lines(audit(mod), "h")
    assert '"Other"' in line and f'"{FOLDER}"' in line


# ---- build_mod: stage 2 (his own files) ---------------------------------------------------------

def files_under(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def tail_range(slot: str, sub: str) -> str:
    """'record 47 bytes a-b' of one sub-field of the 20 bytes after a slot's name (written out here)."""
    at, size = TAIL_AT[sub]
    start = FIELDS[slot].offset + 64 + at
    return f"record 47 bytes {start}-{start + size - 1}"


def set_tail(mod: Path, map_name: str, slot: str, sub: str, value: int) -> None:
    at, size = TAIL_AT[sub]

    def edit(creatures):
        start = FIELDS[slot].offset + 64 + at
        creatures[-1].raw[start:start + size] = value.to_bytes(size, "little")
        return creatures
    edit_creatures(mod, map_name, edit)


def get_tail(record: Creature, slot: str, sub: str) -> int:
    at, size = TAIL_AT[sub]
    start = FIELDS[slot].offset + 64 + at
    return int.from_bytes(record.raw[start:start + size], "little")


def rewrite_kmf(path: Path, change) -> None:
    model = parse_kmf(path.read_bytes())
    change(model)
    path.write_bytes(write_kmf(model))


def meshes(mod: Path) -> Path:
    return mod / "Data" / "Meshes"


def textures(mod: Path) -> Path:
    return mod / "Data" / "EngineTextures"


@needs_data
@needs_assets
def test_stage2_folder_holds_exactly_the_files_the_record_names(stage2_full):
    assert stage2_full.name == "Molynx"
    found = sorted(p.relative_to(stage2_full).as_posix() for p in stage2_full.rglob("*") if p.is_file())
    expected = sorted(["resource.toml",
                       *(f"Data/editor/maps/{name}{suffix}" for name in MAPS for suffix in MAP_SUFFIXES),
                       *(f"Data/Meshes/{f}" for f in STAGE2_MESHES),
                       *(f"Data/EngineTextures/{f}" for f in STAGE2_TEXTURES)])
    assert found == expected
    assert len(STAGE2_MESHES) == 24
    for clip in UNUSED_CLIPS:
        assert not (meshes(stage2_full) / f"Molynx-{clip}.kmf").exists(), clip


@needs_data
@needs_assets
def test_stage2_names_every_delivered_mesh_in_record_47_of_each_map(stage2_full):
    named = {name: {n + ".kmf" for n in renamed_slots(stage2_full, name)} for name in MAPS}
    assert named["Siege"] == set(STAGE2_MESHES)
    assert named[SECOND] == set(STAGE2_MESHES) - {"Molynx-staggering.kmf"}  # no dejectedPose there


@needs_data
@needs_assets
def test_stage2_delivers_the_union_of_the_maps_file_plans(stage2_full):
    from scripts.molynx_assets import molynx_files
    from scripts.molynx_record import stems_used
    plans = [molynx_files(source_bile(name)) for name in MAPS]
    stems = set().union(*(stems_used(source_bile(n), p) for n, p in zip(MAPS, plans)))
    assert {f"Molynx-{stem}.kmf" for stem in stems} == set(files_under(meshes(stage2_full)))


@needs_data
@needs_assets
def test_stage2_files_are_byte_copies_of_the_export(stage2_full, assets_dir):
    for f in STAGE2_MESHES:
        assert (meshes(stage2_full) / f).read_bytes() == (assets_dir / f).read_bytes(), f
    for f in STAGE2_TEXTURES:
        assert (textures(stage2_full) / f).read_bytes() == (assets_dir / f).read_bytes(), f


@needs_data
@needs_assets
def test_stage2_record_frames_come_from_the_files(stage2_dir):
    """A few literal values: walk loops (47 frames -> 46), sleeping does not (136); stunnedPose and
    elec name static poses and keep the Bile Demon's mesh tails (type 5, scale 4096)."""
    molynx = record_47(stage2_dir, "Siege")
    assert get_tail(molynx, "animWalkResource", "frames") == 46
    assert get_tail(molynx, "animSleepResource", "frames") == 136
    assert get_tail(molynx, "animSleepResource", "fps") == 30
    bile = source_bile("Siege")
    for slot in ("animElecResource", "animStunnedPoseResource"):
        start = FIELDS[slot].offset
        assert molynx.raw[start + 64:start + 84] == bile.raw[start + 64:start + 84], slot
        assert get_tail(molynx, slot, "type") == 5 and get_tail(molynx, slot, "frames") == 4096
    assert molynx.resource_name("animElecResource") == "Molynx-ElecPose"
    assert molynx.resource_name("animStunnedPoseResource") == "Molynx-StunnedPose"
    assert get_tail(molynx, "animPoseFrameResource", "type") == 5
    assert get_tail(molynx, "animWalkResource", "flags") & 0x18 == 0


@needs_data
@needs_assets
def test_stage2_building_twice_gives_identical_bytes(stage2_dir, assets_dir, tmp_path):
    again = build_mod(2, tmp_path, "Molynx_stage2", maps=["Siege"], assets_dir=assets_dir)
    files = sorted(p.relative_to(stage2_dir) for p in stage2_dir.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    assert all((stage2_dir / f).read_bytes() == (again / f).read_bytes() for f in files)


@needs_data
@needs_assets
def test_stage2_agreeing_anim_info_and_file_wraps_are_accepted(assets_dir, tmp_path, stage2_dir):
    out = build_mod(2, tmp_path, "Molynx_stage2", maps=["Siege"], assets_dir=assets_dir,
                    anim_info=ANIM_INFO, file_wraps=FILE_WRAPS)
    assert (maps_dir(out) / "SiegeGlobals.kld").read_bytes() == \
        (maps_dir(stage2_dir) / "SiegeGlobals.kld").read_bytes()


def _refused(tmp_path, assets, match, **extra):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=match):
        build_mod(2, out, maps=["Siege"], assets_dir=assets, **extra)
    assert not out.exists()


@needs_data
@needs_assets
def test_stage2_refuses_a_missing_file_and_writes_nothing(assets_copy, tmp_path):
    (assets_copy / "Molynx-walk_once.kmf").unlink()
    _refused(tmp_path, assets_copy, "Molynx-walk_once.kmf")


@needs_data
@needs_assets
def test_stage2_refuses_a_file_named_in_another_case(assets_copy, tmp_path):
    (assets_copy / "Molynx-DraggedPose.kmf").rename(assets_copy / "Molynx-draggedpose.kmf")
    _refused(tmp_path, assets_copy, "other case")


@needs_data
@needs_assets
def test_stage2_refuses_a_file_whose_inside_name_differs(assets_copy, tmp_path):
    rewrite_kmf(assets_copy / "Molynx-stand.kmf", lambda m: setattr(m.anim, "name", "Molynx-Stand"))
    _refused(tmp_path, assets_copy, "Molynx-Stand")


@needs_data
@needs_assets
def test_stage2_refuses_a_file_that_loops_against_the_file_plan(assets_copy, tmp_path):
    rewrite_kmf(assets_copy / "Molynx-praying.kmf", lambda m: setattr(m.anim, "wrap", False))
    _refused(tmp_path, assets_copy, "file plan")


@needs_data
@needs_assets
def test_stage2_refuses_a_pose_that_is_an_animation(assets_copy, tmp_path):
    shutil.copyfile(assets_copy / "Molynx-stand.kmf", assets_copy / "Molynx-DiePose.kmf")
    _refused(tmp_path, assets_copy, "Molynx-DiePose")


@needs_data
@needs_assets
@pytest.mark.parametrize("damage", ["trailing", "truncated"])
def test_stage2_refuses_a_file_that_does_not_parse_to_its_end(assets_copy, tmp_path, damage):
    path = assets_copy / "Molynx-angry.kmf"
    data = path.read_bytes()
    path.write_bytes(data + b"\0" if damage == "trailing" else data[:len(data) // 2])
    _refused(tmp_path, assets_copy, "Molynx-angry.kmf")


@needs_data
@needs_assets
def test_stage2_refuses_a_missing_texture(assets_copy, tmp_path):
    (assets_copy / "Molynx_Mace.png").unlink()
    _refused(tmp_path, assets_copy, "Molynx_Mace.png")


@needs_data
@needs_assets
@pytest.mark.parametrize("size", [(256, 256), (100, 128)])
def test_stage2_refuses_a_texture_the_game_cannot_take(assets_copy, tmp_path, size):
    Image.new("RGB", size).save(assets_copy / "Molynx_Body.png")
    _refused(tmp_path, assets_copy, "Molynx_Body.png")


@needs_data
@needs_assets
def test_stage2_refuses_anim_info_that_disagrees_with_the_files(assets_dir, tmp_path):
    wrong = {**ANIM_INFO, "walk": {"frames": 48, "fps": 30, "wrap": True}}
    _refused(tmp_path, assets_dir, "anim_info.*walk", anim_info=wrong)
    _refused(tmp_path, assets_dir, "anim_info", anim_info=[("walk", 47)])


@needs_data
@needs_assets
def test_stage2_refuses_file_wraps_that_disagree_with_the_bile_demons_files(assets_dir, tmp_path):
    _refused(tmp_path, assets_dir, "file_wraps", file_wraps={**FILE_WRAPS, "bile_demon-walk": False})


def test_stage2_refuses_an_assets_dir_that_is_not_a_folder(tmp_path):
    _refused(tmp_path, tmp_path / "nothing", "assets_dir")


@needs_data
@needs_assets
def test_build_main_builds_stage_2_from_an_export(assets_dir, tmp_path, capsys):
    out = tmp_path / "bundle"
    assert builder.main(["--stage", "2", "--assets", str(assets_dir), "--out", str(out),
                         "--maps", "Siege"]) == 0
    assert capsys.readouterr().out.strip().endswith("Molynx")
    assert files_under(meshes(out / "Molynx")) == STAGE2_MESHES
    assert audit(out / "Molynx").ok


@pytest.mark.parametrize("argv,needle", [(["--stage", "2"], "--assets"),
                                         (["--stage", "1", "--assets", "x"], "--assets")])
def test_build_main_refuses_assets_at_the_wrong_stage(tmp_path, capsys, argv, needle):
    with pytest.raises(SystemExit) as exit_info:
        builder.main([*argv, "--out", str(tmp_path / "bundle")])
    assert exit_info.value.code == 2
    assert needle in capsys.readouterr().err
    assert not (tmp_path / "bundle").exists()


def test_build_main_help_names_the_stages_and_the_assets_option(capsys):
    with pytest.raises(SystemExit):
        builder.main(["--help"])
    out = capsys.readouterr().out
    assert "--assets" in out and "--stage" in out and "Stage 2" in out


# ---- audit: stage 2 -----------------------------------------------------------------------------

@needs_data
@needs_assets
def test_stage2_audit_of_the_handed_over_folder_passes_every_rule(stage2_full):
    report = audit(stage2_full)
    assert report.ok, "\n".join(report.lines)
    for name in MAPS:
        for rule in "abcdef":
            assert len(pass_lines(report, rule, name)) == 1, (rule, name)
        (line,) = pass_lines(report, "b", name)
        assert "stage 2" in line and "24 rewritten" in line
    (g,) = [line for line in report.lines if line.startswith("PASS (g) folder:")]
    assert "24 meshes" in g and "2 textures" in g


@needs_data
@needs_assets
def test_stage2_audit_passes_with_every_renamed_animation_present(mod2):
    names = renamed_slots(mod2, "Siege")
    assert "Molynx-walk" in names and len(names) == 24
    report = audit(mod2)
    assert report.ok, "\n".join(report.lines)
    assert any("stage 2" in line for line in report.lines if line.startswith("PASS (b)"))
    (line,) = pass_lines(report, "f", "Siege")
    assert f"{len(names)} Molynx meshes" in line and "Molynx_Body, Molynx_Mace" in line


@needs_data
@needs_assets
def test_stage2_audit_lists_the_sleeping_wrap_exception_by_name(mod2):
    (line,) = pass_lines(audit(mod2), "f", "Siege")
    assert "1 recorded exception(s)" in line
    assert "animSleepResource Molynx-sleeping does not loop" in line
    assert "BileDemon-bed(sleeping)" in line
    assert set(audit_module.WRAP_EXCEPTIONS) == {"animSleepResource"}


@needs_data
@needs_assets
def test_stage2_audit_fails_naming_every_missing_mesh_when_none_is_delivered(mod2):
    shutil.rmtree(meshes(mod2))
    shutil.rmtree(textures(mod2))
    report = audit(mod2)
    assert not report.ok
    (line,) = fail_lines(report)
    assert line.startswith("FAIL (f) Siege:")
    for name in ("Molynx-walk", "Molynx-DraggedPose", "Molynx-stand"):
        assert f"Data/Meshes/{name}.kmf is missing" in line
    (b,) = pass_lines(report, "b", "Siege")
    assert "cannot be checked" in b


@needs_data
@needs_assets
def test_stage2_audit_fails_when_one_mesh_is_missing(mod2):
    (meshes(mod2) / "Molynx-walk.kmf").unlink()
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (f) Siege:")
    assert "Molynx-walk.kmf is missing" in line and "Molynx-attack" not in line


@needs_data
@needs_assets
def test_stage2_mesh_names_must_match_the_case_exactly(mod2):
    (meshes(mod2) / "Molynx-DraggedPose.kmf").rename(meshes(mod2) / "Molynx-draggedpose.kmf")
    report = audit(mod2)
    (f,) = fail_lines(report, "f")
    assert "Molynx-DraggedPose.kmf is missing" in f and "other case" in f
    (g,) = fail_lines(report, "g")
    assert "Molynx-draggedpose.kmf" in g
    assert len(fail_lines(report)) == 2


@needs_data
@needs_assets
def test_stage2_an_empty_mesh_file_fails(mod2):
    (meshes(mod2) / "Molynx-walk.kmf").write_bytes(b"")
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (f) Siege:") and "Molynx-walk.kmf is empty" in line


@needs_data
@needs_assets
@pytest.mark.parametrize("damage", ["trailing", "truncated"])
def test_stage2_a_mesh_that_does_not_parse_to_its_end_fails(mod2, damage):
    """A copied file that no longer reads cleanly. A changed byte inside the vertex data is NOT
    found: the audit has no copy of the export to compare with (its expectations are the game's
    files and the rules, not the builder's inputs); the builder's byte copy is tested above."""
    path = meshes(mod2) / "Molynx-angry.kmf"
    data = path.read_bytes()
    path.write_bytes(data + b"\0\0" if damage == "trailing" else data[:-40])
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (f) Siege:") and "Molynx-angry.kmf" in line


@needs_data
@needs_assets
def test_stage2_a_slot_left_as_the_bile_demons_is_named_with_its_bytes(mod2):
    """The whole 84-byte slot restored: rule (b) names the slot and its bytes, nothing else fails."""
    bile = source_bile("Siege")
    start = FIELDS["animWalkResource"].offset

    def restore(creatures):
        creatures[-1].raw[start:start + 84] = bile.raw[start:start + 84]
        return creatures

    edit_creatures(mod2, "Siege", restore)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert f"animWalkResource is left as the Bile Demon's (record 47 bytes {start}-{start + 83})" in line
    assert line.count("animWalkResource") == 1, "named once, not again as a wrong name or tail"


@needs_data
@needs_assets
def test_stage2_a_slot_not_named_molynx_fails(mod2):
    """animWalkResource renamed Other-walk, with a matching, valid Other-walk.kmf delivered."""
    shutil.copyfile(meshes(mod2) / "Molynx-walk.kmf", meshes(mod2) / "Other-walk.kmf")
    rewrite_kmf(meshes(mod2) / "Other-walk.kmf", lambda m: setattr(m.anim, "name", "Other-walk"))
    start = FIELDS["animWalkResource"].offset

    def rename(creatures):
        creatures[-1].set_resource_name("animWalkResource", "Other-walk")
        return creatures

    edit_creatures(mod2, "Siege", rename)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert f"animWalkResource name (record 47 bytes {start}-{start + 63})" in line and "Molynx-" in line


@needs_data
@needs_assets
def test_stage2_a_changed_tail_of_a_slot_fails(mod2):
    """The 20 bytes after a renamed slot's name follow the stage-2 rule; one changed byte there (in
    the frame count) is named with the slot and its byte range."""
    def tweak(creatures):
        creatures[-1].raw[FIELDS["animWalkResource"].offset + 70] ^= 0x01
        return creatures

    edit_creatures(mod2, "Siege", tweak)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert f"animWalkResource frames ({tail_range('animWalkResource', 'frames')}) is " in line
    assert "expected 46" in line


@needs_data
@needs_assets
@pytest.mark.parametrize("slot,frames", [("animWalkResource", 47), ("animIdle1Resource", 60),
                                         ("animSleepResource", 135)])
def test_stage2_a_frame_count_off_by_one_fails(mod2, slot, frames):
    set_tail(mod2, "Siege", slot, "frames", frames)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert f"{slot} frames ({tail_range(slot, 'frames')}) is {frames}," in line


@needs_data
@needs_assets
def test_stage2_start_and_end_flag_bits_left_set_fail(mod2):
    set_tail(mod2, "Siege", "animWalkResource", "flags", 0x219)  # the Bile Demon's own Siege flags
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert f"animWalkResource flags ({tail_range('animWalkResource', 'flags')}) is 0x219, " \
           "expected 0x201" in line
    assert "0x8 and 0x10" in line


@needs_data
@needs_assets
@pytest.mark.parametrize("slot", ["animStunnedPoseResource", "animElecResource"])
def test_stage2_a_pose_slot_turned_into_type_6_fails(mod2, slot):
    set_tail(mod2, "Siege", slot, "type", 6)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert f"{slot} type ({tail_range(slot, 'type')}) is 6, expected 5" in line


@needs_data
@needs_assets
def test_stage2_a_wrong_fps_fails(mod2):
    set_tail(mod2, "Siege", "animSleepResource", "fps", 15)  # the Bile Demon's sleep speed
    (line,) = fail_lines(audit(mod2))
    assert f"animSleepResource fps ({tail_range('animSleepResource', 'fps')}) is 15, expected 30" in line


@needs_data
@needs_assets
def test_stage2_a_mesh_whose_inside_name_differs_fails(mod2):
    rewrite_kmf(meshes(mod2) / "Molynx-stand.kmf", lambda m: setattr(m.anim, "name", "Molynx-Stand"))
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (f) Siege:")
    assert "Molynx-stand.kmf calls itself 'Molynx-Stand' inside" in line


@needs_data
@needs_assets
def test_stage2_a_triangle_index_out_of_range_fails(mod2):
    def damage(model):
        sprite = model.mesh.sprites[-1]
        a, b, _c = sprite.triangles[0][0]
        sprite.triangles[0][0] = (a, b, len(sprite.vertices))
    rewrite_kmf(meshes(mod2) / "Molynx-DiePose.kmf", damage)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (f) Siege:") and "Molynx-DiePose.kmf" in line
    assert "triangle index outside" in line


@needs_data
@needs_assets
def test_stage2_a_missing_texture_fails(mod2):
    (textures(mod2) / "Molynx_Mace.png").unlink()
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (f) Siege:") and "Data/EngineTextures/Molynx_Mace.png is missing" in line


@needs_data
@needs_assets
def test_stage2_a_texture_named_in_another_case_fails(mod2):
    (textures(mod2) / "Molynx_Mace.png").rename(textures(mod2) / "molynx_mace.png")
    report = audit(mod2)
    (f,) = fail_lines(report, "f")
    assert "Molynx_Mace.png is missing" in f and "other case" in f
    (g,) = fail_lines(report, "g")
    assert "molynx_mace.png" in g


@needs_data
@needs_assets
@pytest.mark.parametrize("size", [(256, 256), (128, 100)])
def test_stage2_a_texture_of_the_wrong_size_fails(mod2, size):
    Image.new("RGB", size).save(textures(mod2) / "Molynx_Body.png")
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (f) Siege:") and f"Molynx_Body.png is {size[0]}x{size[1]}" in line


@needs_data
@needs_assets
def test_stage2_a_texture_that_is_not_a_png_fails(mod2):
    (textures(mod2) / "Molynx_Body.png").write_bytes(b"not a picture")
    (line,) = fail_lines(audit(mod2))
    assert "Molynx_Body.png is not a PNG file" in line


@needs_data
@needs_assets
def test_stage2_an_extra_unreferenced_mesh_fails_rule_g(mod2, assets_dir):
    shutil.copyfile(assets_dir / "Molynx-running.kmf", meshes(mod2) / "Molynx-running.kmf")
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (g) folder:") and "Data/Meshes/Molynx-running.kmf" in line


@needs_data
@needs_assets
def test_stage2_an_extra_unreferenced_texture_fails_rule_g(mod2):
    Image.new("RGB", (64, 64)).save(textures(mod2) / "Molynx_Spare.png")
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (g) folder:") and "Molynx_Spare.png" in line


@needs_data
@needs_assets
def test_stage2_a_subfolder_of_engine_textures_fails_rule_g(mod2):
    (textures(mod2) / "GUI").mkdir()
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (g) folder:") and "Data/EngineTextures/GUI" in line


@needs_data
@needs_assets
def test_stage2_a_flipped_loop_flag_fails_rule_f_and_the_frame_count(mod2):
    """praying must loop like Bile_Demon-Pray. Opened, the file no longer matches the Bile Demon's
    loop flag (rule f), and its frame count no longer gives the record's (121, not 121 - 1: rule b)."""
    rewrite_kmf(meshes(mod2) / "Molynx-praying.kmf", lambda m: setattr(m.anim, "wrap", False))
    report = audit(mod2)
    (f,) = fail_lines(report, "f")
    assert "animPrayResource: Molynx-praying does not loop, but the Bile Demon's Bile_Demon-Pray loops" in f
    (b,) = fail_lines(report, "b")
    assert f"animPrayResource frames ({tail_range('animPrayResource', 'frames')}) is 120, expected 121" in b
    assert len(fail_lines(report)) == 2


@needs_data
@needs_assets
def test_stage2_a_looped_sleeping_file_is_no_exception_but_breaks_its_frame_count(mod2):
    """The exception covers only a non-looping sleeping file; looped, it matches the Bile Demon's
    flag (no exception listed) and the record's 136 is then one too many (rule b)."""
    rewrite_kmf(meshes(mod2) / "Molynx-sleeping.kmf", lambda m: setattr(m.anim, "wrap", True))
    report = audit(mod2)
    (b,) = fail_lines(report)
    assert b.startswith("FAIL (b) Siege:") and "animSleepResource frames" in b and "expected 135" in b
    (f,) = pass_lines(report, "f", "Siege")
    assert "exception" not in f


@needs_data
@needs_assets
def test_stage2_an_animation_in_a_static_pose_slot_fails(mod2):
    """elec replaces the Bile Demon's static Bil-Elec: its file must be a static pose, not an
    animation (here getting_hit, delivered under the pose's name)."""
    shutil.copyfile(meshes(mod2) / "Molynx-getting_hit.kmf", meshes(mod2) / "Molynx-ElecPose.kmf")
    rewrite_kmf(meshes(mod2) / "Molynx-ElecPose.kmf", lambda m: setattr(m.anim, "name", "Molynx-ElecPose"))
    report = audit(mod2)
    (f,) = fail_lines(report, "f")
    assert "animElecResource: Molynx-ElecPose is an animation" in f and "Bil-Elec" in f
    (b,) = fail_lines(report, "b")
    assert "animElecResource names Molynx-ElecPose, an animation" in b
    assert len(fail_lines(report)) == 2


@needs_data
@needs_assets
def test_stage2_empty_bile_slots_stay_empty_and_other_fields_are_unchanged(mod2):
    molynx, bile = record_47(mod2, "Siege"), source_bile("Siege")
    assert molynx.resource_name("animRunResource") == "" == bile.resource_name("animRunResource")
    assert molynx.resource_name("unk_1") == bile.resource_name("unk_1")
    assert all(molynx.resource_name(s).startswith("Molynx-")
               for s in ANIM_SLOTS if s != "unk_1" and bile.resource_name(s))


# ---- audit: stage 3 is fail-closed until its icons and texts can be checked (Task 10) -----------

@needs_data
@needs_assets
def test_stage3_fields_are_recognised_but_icons_and_textures_are_not_yet_checked(tmp_path, assets_dir):
    out = build_mod(3, tmp_path, "Molynx_full", maps=["Siege"], text_ids=TEXT_IDS, assets_dir=assets_dir)
    report = audit(out)
    assert any("stage 3" in line for line in report.lines if line.startswith("PASS (b)"))
    assert not report.ok
    (line,) = fail_lines(report)
    assert line.startswith("FAIL (f)") and "icon" in line and "not checked" in line
    assert "stage 3 is not audited yet" in line


@needs_data
@needs_assets
def test_stage3_text_ids_are_part_of_the_record(tmp_path, assets_dir):
    out = build_mod(3, tmp_path, "Molynx_full", maps=[SECOND], text_ids=TEXT_IDS, assets_dir=assets_dir)
    molynx = record_47(out, SECOND)
    assert molynx.get("nameStringId") == 3001 and molynx.get("weaknessStringId") == 3005
    assert molynx.resource_name("icon1Resource") == r"gui\creatures\evil\molynx"
    assert molynx.get("hp") > source_bile(SECOND).get("hp")


# ---- fix round 1: which file each slot names, every tail sub-field, file_wraps, the bundle -------

@needs_data
@needs_assets
def test_stage2_every_slot_names_the_file_of_the_slot_table_on_both_maps(stage2_full):
    table = expected_slot_files()
    assert set(table) <= set(ANIM_SLOTS)
    for name in MAPS:
        molynx, bile = record_47(stage2_full, name), source_bile(name)
        for slot in ANIM_SLOTS:
            if slot == "unk_1":
                continue
            want = f"Molynx-{table.get(slot, 'stand')}" if bile.resource_name(slot) else ""
            assert molynx.resource_name(slot) == want, (name, slot)


@needs_data
@needs_assets
def test_the_audits_slot_table_is_the_design_table():
    assert audit_module.SLOT_FILES == expected_slot_files()
    assert audit_module.DEFAULT_FILE == "stand"


def _swap(mod: Path, a: str, b: str, size: int) -> None:
    """Swap the first `size` bytes (64: the names, 84: the whole resources) of two slots."""
    oa, ob = FIELDS[a].offset, FIELDS[b].offset

    def edit(creatures):
        raw = creatures[-1].raw
        raw[oa:oa + size], raw[ob:ob + size] = bytes(raw[ob:ob + size]), bytes(raw[oa:oa + size])
        return creatures
    edit_creatures(mod, "Siege", edit)


@needs_data
@needs_assets
def test_stage2_swapped_names_of_two_slots_fail_naming_both(mod2):
    """happy and angry both end (like the Bile Demon's files), so only the table can tell them apart."""
    _swap(mod2, "animHappyResource", "animAngryResource", 64)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    happy, angry = FIELDS["animHappyResource"].offset, FIELDS["animAngryResource"].offset
    assert (f"animHappyResource name (record 47 bytes {happy}-{happy + 63}) is 'Molynx-angry', "
            "expected 'Molynx-happy'") in line
    assert (f"animAngryResource name (record 47 bytes {angry}-{angry + 63}) is 'Molynx-happy', "
            "expected 'Molynx-angry'") in line


@needs_data
@needs_assets
def test_stage2_swapped_whole_resources_of_two_slots_fail(mod2):
    _swap(mod2, "animHappyResource", "animAngryResource", 84)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert "animHappyResource name" in line and "expected 'Molynx-happy'" in line
    assert "animAngryResource name" in line and "expected 'Molynx-angry'" in line
    assert f"animHappyResource frames ({tail_range('animHappyResource', 'frames')}) is 23, expected 59" in line


@needs_data
@needs_assets
@pytest.mark.parametrize("slot,sub,value,expected", [
    ("animWalkResource", "startDist", 2159, 2158),      # kept from the Bile Demon's walk
    ("animWalkResource", "endDist", 0, 1622),
    ("animWalkResource", "startAf", 40, 0),             # no Start file
    ("animWalkResource", "endAf", 40, 0),               # no End file
    ("drunkIdle", "sometimesOne", 0, 1),                # kept
    ("animIdle1Resource", "sometimesOne", 1, 0),
    ("animStunnedPoseResource", "startDist", 0, 1128),  # a pose slot keeps the Bile Demon's tail
    ("animStunnedPoseResource", "endDist", 0, 158),
    ("animElecResource", "endDist", 0, 200),
    ("animElecResource", "sometimesOne", 1, 0),
    ("animElecResource", "frames", 17, 4096),
])
def test_stage2_every_tail_sub_field_is_checked(mod2, slot, sub, value, expected):
    set_tail(mod2, "Siege", slot, sub, value)
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (b) Siege:")
    assert f"{slot} {sub} ({tail_range(slot, sub)}) is {value}, expected {expected}" in line
    assert line.count(f"{slot} ") == 1


@needs_data
@needs_assets
@pytest.mark.parametrize("bad", [[("bile_demon-walk", True)], "bile_demon-walk"])
def test_stage2_file_wraps_that_is_not_a_mapping_is_a_value_error(assets_dir, tmp_path, bad):
    _refused(tmp_path, assets_dir, "file_wraps", file_wraps=bad)


@needs_data
def test_resource_toml_descriptions_say_which_folder_to_use(stage1_dir):
    one = tomllib.loads((stage1_dir / "resource.toml").read_text())["resource"]["description"]
    assert one.startswith("Molynx test A fallback (slot only). Use this only if the folder Molynx fails.")
    assert "Enable only one Molynx folder at a time." in one


@needs_data
@needs_assets
def test_stage2_resource_toml_description_says_use_this_first(stage2_full):
    two = tomllib.loads((stage2_full / "resource.toml").read_text())["resource"]["description"]
    assert two.startswith("Molynx test A (his own look). Use this first.")
    assert "Enable only one Molynx folder at a time." in two


def _synthetic_mods(root: Path) -> Path:
    for name in ("Molynx", "Molynx_slot_only"):
        (root / name / "Data" / "editor" / "maps").mkdir(parents=True)
        (root / name / "resource.toml").write_bytes(b"[resource]\r\n")
        (root / name / "Data" / "editor" / "maps" / "Siege.kwd").write_bytes(name.encode())
    (root / "Molynx" / "Data" / "Meshes").mkdir()
    (root / "Molynx" / "Data" / "Meshes" / "Molynx-walk.kmf").write_bytes(b"walk")
    return root


def test_bundle_is_sorted_with_fixed_times_and_modes(tmp_path):
    import zipfile
    from scripts.make_molynx_bundle import make_bundle
    mods = _synthetic_mods(tmp_path / "mods")
    guide = tmp_path / "guide.md"
    guide.write_text("# guide\n")
    out = tmp_path / "a.zip"
    names = make_bundle(mods, guide, out)
    expected = ["Molynx/Data/Meshes/Molynx-walk.kmf", "Molynx/Data/editor/maps/Siege.kwd",
                "Molynx/resource.toml", "Molynx_slot_only/Data/editor/maps/Siege.kwd",
                "Molynx_slot_only/resource.toml", "Molynx_test_guide.md"]
    assert names == expected
    with zipfile.ZipFile(out) as z:
        infos = z.infolist()
        assert [i.filename for i in infos] == expected
        assert all(i.date_time == (2026, 11, 5, 0, 0, 0) for i in infos)
        assert all(i.external_attr >> 16 == 0o644 for i in infos)
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in infos)
        assert z.read("Molynx_test_guide.md") == b"# guide\n"
        assert z.read("Molynx_slot_only/Data/editor/maps/Siege.kwd") == b"Molynx_slot_only"
    again = tmp_path / "b.zip"
    make_bundle(mods, guide, again)
    assert out.read_bytes() == again.read_bytes()
    with pytest.raises(FileExistsError):
        make_bundle(mods, guide, out)


@pytest.mark.parametrize("stray", ["Molynx/readme.txt", "Molynx/Data/.DS_Store",
                                   "Molynx_slot_only/Data/editor/maps/Siege.kwd~",
                                   "Molynx/Other/x.kmf", "Molynx_slot_only/notes.md"])
def test_bundle_refuses_stray_files_and_writes_nothing(tmp_path, stray):
    from scripts.make_molynx_bundle import make_bundle
    mods = _synthetic_mods(tmp_path / "mods")
    (mods / stray).parent.mkdir(parents=True, exist_ok=True)
    (mods / stray).write_bytes(b"x")
    guide = tmp_path / "guide.md"
    guide.write_text("g")
    with pytest.raises(ValueError):
        make_bundle(mods, guide, tmp_path / "a.zip")
    assert not (tmp_path / "a.zip").exists()


def test_bundle_refuses_a_missing_folder(tmp_path):
    from scripts.make_molynx_bundle import make_bundle
    mods = _synthetic_mods(tmp_path / "mods")
    shutil.rmtree(mods / "Molynx_slot_only")
    guide = tmp_path / "guide.md"
    guide.write_text("g")
    with pytest.raises(ValueError, match="Molynx_slot_only"):
        make_bundle(mods, guide, tmp_path / "a.zip")
    assert not (tmp_path / "a.zip").exists()
