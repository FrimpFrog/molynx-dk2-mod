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
MAPS = ["Siege", SECOND]  # the Test A maps (built with maps=MAPS: both copied and retargeted)
# The folders as handed over (Test A, map 2): Siege as in Test A, and the flat test map under three names;
# stem -> (Molynx in the portal pool, the name the level file shows).
FLAT = {"Molynx Test": (False, "Molynx Test"), "Molynx Test Portal": (True, "Molynx Test Portal"),
        SECOND: (True, "Molynx Test")}   # Dead Walking: the portal variant (review round 2)
SHIPPED_MAPS = sorted(["Siege", *FLAT])
PANEL_FILES = ["molynx-bg.png", "molynx.png"]
TEXT_IDS = {"name": 3001, "tooltip": 3002, "general": 3003, "strength": 3004, "weakness": 3005}
# The files a stage-2 folder delivers (literal; the record's file plan for the test maps' Bile Demons):
# 20 clips (staggering only for Siege's type NONE dejectedPose slot), the variant walk_once, the 5
# poses (stunnedPose and elec take static poses), and the 4 textures.
STAGE2_MESHES = sorted(f"Molynx-{stem}.kmf" for stem in (
    "walk", "walk_once", "DraggedPose", "getting_hit", "StunnedPose", "ElecPose", "attack", "die",
    "DiePose", "happy", "angry", "staggering", "being_picked_up", "sleeping", "eating", "stand",
    "being_slapped", "manufacturing", "praying", "falling", "waking_up", "celebrating",
    "fidgeting", "PoseFrame", "torture", "drinking", "ground_slam"))
STAGE2_TEXTURES = ["Molynx_Body.png", "Molynx_Head.png", "Molynx_Mace.png", "Molynx_Neck.png"]
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
angry->angry, swing->being_picked_up, sleep->sleeping, eat/eatCorpse->eating, drink->drinking,
building->manufacturing, pray->praying, fallback->falling, getUp->waking_up, dance->celebrating,
drunk/drunk2/stunnedPose/dejectedPose->staggering, idle1->stand, idle2/drunkIdle->fidgeting,
poseFrame->PoseFrame, walk2->patrolling, entrance/walkback->walk,
research/loot->interacting_with_objects, elec/electrocute->getting_hit, magic->ground_slam,
torture->torture"""
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


EDITOR_LABEL = "editor Creatures.kwd"


def fail_lines(report, rule: str | None = None, map_name: str | None = None) -> list[str]:
    """FAIL lines (of `rule`, of `map_name`). Without `map_name` the editor Creatures.kwd's lines
    are left out: its record 47 names the same Molynx files as the maps' records, so a damaged
    asset fails rule (f) there too (test_a_damaged_mesh_fails_the_editor_table_too checks that)."""
    prefix = "FAIL" + (f" ({rule})" if rule else "")
    return [line for line in report.lines
            if line.startswith(prefix) and (f" {map_name}:" in line if map_name is not None
                                            else f" {EDITOR_LABEL}:" not in line)]


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
    """Test A at stage 1: Siege and Dead Walking, both copied and retargeted (no flat map)."""
    return build_mod(1, tmp_path_factory.mktemp("stage1"), FOLDER, maps=MAPS)


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
    molynx_assets.export_molynx_assets(out, portrait_src=molynx_assets.RENDERED)
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
    """The stage-2 folder as it is handed over: name Molynx, Siege and the three flat maps."""
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


def test_build_stage_3_needs_assets_too(tmp_path):
    with pytest.raises(ValueError, match="assets_dir"):
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
    expected = sorted(["resource.toml", "Data/editor/Creatures.kwd",
                       *(f"Data/editor/maps/{name}{suffix}" for name in MAPS for suffix in MAP_SUFFIXES)])
    assert found == expected
    assert len(found) == 2 + 2 * 7
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
    again = build_mod(1, tmp_path, FOLDER, maps=MAPS)
    files = sorted(p.relative_to(stage1_dir) for p in stage1_dir.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    assert all((stage1_dir / f).read_bytes() == (again / f).read_bytes() for f in files)


@needs_data
def test_build_main_builds_the_slot_only_folder_by_default(tmp_path, capsys):
    assert builder.main(["--out", str(tmp_path / "bundle")]) == 0
    assert (tmp_path / "bundle" / FOLDER / "resource.toml").is_file()
    assert capsys.readouterr().out.strip().endswith(FOLDER)


@needs_data
def test_default_name_is_molynx_and_default_maps_are_siege_and_the_flat_maps(tmp_path):
    out = build_mod(1, tmp_path)
    assert out == tmp_path / "Molynx"
    names = {p.name[:-len("Globals.kld")] for p in maps_dir(out).glob("*Globals.kld")}
    assert names == set(SHIPPED_MAPS)
    assert builder.FLAT_MAPS == {stem: in_pool for stem, (in_pool, _shown) in FLAT.items()}


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
                      "slot_assignment", "SLOT_CLIPS", "art_tail", "anim_info",
                      "molynx_ground_slam", "apply_ground_slam", "ground_slam_records"):
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
    expected = sorted(["resource.toml", "Data/editor/Creatures.kwd",
                       *(f"Data/editor/maps/{name}{suffix}" for name in SHIPPED_MAPS
                         for suffix in MAP_SUFFIXES),
                       *(f"Data/Meshes/{f}" for f in STAGE2_MESHES),
                       *(f"Data/EngineTextures/{f}" for f in STAGE2_TEXTURES),
                       *(f"Data/EngineTextures/GUI/Creatures/Evil/{f}" for f in PANEL_FILES)])
    assert found == expected
    assert len(STAGE2_MESHES) == 27
    for clip in UNUSED_CLIPS:
        assert not (meshes(stage2_full) / f"Molynx-{clip}.kmf").exists(), clip


@needs_data
@needs_assets
def test_stage2_names_every_delivered_mesh_in_record_47_of_each_map(stage2_full):
    named = {name: {n + ".kmf" for n in renamed_slots(stage2_full, name)} for name in MAPS}
    assert named["Siege"] == set(STAGE2_MESHES)
    # Dead Walking is the flat map here, whose creature table is Siege's (its own pins its Imps)
    assert named[SECOND] == set(STAGE2_MESHES)


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
    """A few literal values: walk loops (47 frames -> 46), so does sleeping (55 -> 54); stunnedPose and
    elec name static poses and keep the Bile Demon's mesh tails (type 5, scale 4096)."""
    molynx = record_47(stage2_dir, "Siege")
    assert get_tail(molynx, "animWalkResource", "frames") == 46
    assert get_tail(molynx, "animSleepResource", "frames") == 54
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
    for name in SHIPPED_MAPS:
        for rule in ("abcdef" if name == "Siege" else "abcf"):
            assert len(pass_lines(report, rule, name)) == 1, (rule, name)
        (line,) = pass_lines(report, "b", name)
        assert "stage 2" in line and "24 rewritten" in line
    (g,) = [line for line in report.lines if line.startswith("PASS (g) folder:")]
    assert "27 meshes" in g and "4 textures" in g


@needs_data
@needs_assets
def test_stage2_audit_passes_with_every_renamed_animation_present(mod2):
    names = renamed_slots(mod2, "Siege")
    assert "Molynx-walk" in names and len(names) == 27
    report = audit(mod2)
    assert report.ok, "\n".join(report.lines)
    assert any("stage 2" in line for line in report.lines if line.startswith("PASS (b)"))
    (line,) = pass_lines(report, "f", "Siege")
    assert f"{len(names)} Molynx meshes" in line
    assert "Molynx_Body, Molynx_Head, Molynx_Mace, Molynx_Neck" in line


@needs_data
@needs_assets
def test_stage2_audit_lists_no_wrap_exception(mod2):
    """Since the review round sleeping loops like the Bile Demon's bed file: no slot is excepted."""
    (line,) = pass_lines(audit(mod2), "f", "Siege")
    assert "exception" not in line
    assert audit_module.WRAP_EXCEPTIONS == {}


@needs_data
@needs_assets
def test_stage2_audit_fails_naming_every_missing_mesh_when_none_is_delivered(mod2):
    shutil.rmtree(meshes(mod2))
    shutil.rmtree(textures(mod2))
    report = audit(mod2)
    assert not report.ok
    # rule (f) for the meshes, and rule (g): the panel pictures went with the textures
    (line,) = [x for x in fail_lines(report) if x.startswith("FAIL (f) Siege:")]
    (folder,) = [x for x in fail_lines(report) if x.startswith("FAIL (g) folder:")]
    assert len(fail_lines(report)) == 2 and "GUI/Creatures/Evil must hold" in folder
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
    (textures(mod2) / "Extra").mkdir()
    (line,) = fail_lines(audit(mod2))
    assert line.startswith("FAIL (g) folder:") and "Data/EngineTextures/Extra" in line


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
def test_stage2_an_unlooped_sleeping_file_fails_its_loop_flag(mod2):
    """No exception covers sleeping any more: a sleeping file that does not loop fails rule (f),
    since the Bile Demon's bed(sleeping) file loops."""
    rewrite_kmf(meshes(mod2) / "Molynx-sleeping.kmf", lambda m: setattr(m.anim, "wrap", False))
    report = audit(mod2)
    f = [line for line in fail_lines(report) if line.startswith("FAIL (f) Siege:")]
    assert len(f) == 1
    assert "animSleepResource: Molynx-sleeping does not loop, but the Bile Demon's" in f[0]


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
def test_stage3_ships_his_texts_and_passes_the_audit(tmp_path, assets_dir):
    """Stage 3 without text_ids: his texts are added after the game's own in Text.str, the record
    names them, and the audit reads them back."""
    from scripts.dk2paths import TEXT_DIR
    from scripts.dk2str import StrFile
    from scripts.molynx_record import TEXTS
    out = build_mod(3, tmp_path, "Molynx_full", maps=["Siege"], assets_dir=assets_dir)
    report = audit(out)
    assert report.ok, fail_lines(report)
    assert any("stage 3" in line for line in report.lines if line.startswith("PASS (b)"))
    text = StrFile.read(out / "Data" / "Text" / "Default" / "Text.str", TEXT_DIR / "MBToUni.dat")
    game = StrFile.read(TEXT_DIR / "Text.str")
    molynx = record_47(out, "Siege")
    fields = {"name": "nameStringId", "general": "generalDescriptionStringId",
              "strength": "strengthStringId", "weakness": "weaknessStringId"}
    for key, field in fields.items():
        assert molynx.get(field) >= len(game.entries)
        assert text.text(molynx.get(field)) == TEXTS[key]
    assert molynx.get("tooltipStringId") == 2408


@needs_data
@needs_assets
def test_stage3_with_text_ids_but_no_text_file_fails_the_audit(tmp_path, assets_dir):
    out = build_mod(3, tmp_path, "Molynx_full", maps=["Siege"], text_ids=TEXT_IDS, assets_dir=assets_dir)
    report = audit(out)
    assert not report.ok
    (line,) = fail_lines(report)
    assert line.startswith("FAIL (b)") and "Text.str is missing" in line


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
    for name in MAPS:   # Dead Walking is the flat map here: its creature table is Siege's
        molynx, bile = record_47(stage2_full, name), source_bile("Siege")
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


# ---- the flat test map in the folders (builder) -------------------------------------------------
# Offsets used below (written out here, from the Task 1 reconnaissance): Map.kld header 36 bytes,
# then 69 x 69 tiles of 4 bytes (terrain, player, flag, unknown); Things.kld header 56 bytes (count
# u32 at 20, file size at 8, content size at 52), records u32 type, u32 size, data (a creature: x, y
# i32 at +0, creatureId at +26, owner at +27); Variables entries of 16 bytes after a 56-byte header;
# the level file's name wchar[64] at 56, description wchar[1024] at 184, path records (72 bytes:
# u32 type, u32 0, char[64]) from byte 56 + 25683 + 8.

MAP_HEADER, MAP_W = 36, 69
PATHS_AT = 56 + 25683 + 8
DESCRIPTION = "Flat test map: Keeper 2 is sealed in, no enemies, no script."


@pytest.fixture(scope="module")
def shipped1(tmp_path_factory):
    """The stage-1 folder as it is handed over: Siege (Test A) and the three flat maps."""
    return build_mod(1, tmp_path_factory.mktemp("shipped1"), FOLDER)


@pytest.fixture
def flat_mod(shipped1, tmp_path):
    target = tmp_path / FOLDER
    shutil.copytree(shipped1, target)
    return target


def map_file(mod: Path, stem: str, suffix: str) -> bytes:
    return (maps_dir(mod) / f"{stem}{suffix}").read_bytes()


def set_tile(mod: Path, stem: str, x: int, y: int, tile: bytes) -> None:
    path = maps_dir(mod) / f"{stem}Map.kld"
    data = bytearray(path.read_bytes())
    at = MAP_HEADER + 4 * (y * MAP_W + x)
    data[at:at + 4] = tile
    path.write_bytes(bytes(data))


def level_strings(kwd: bytes) -> tuple[str, str, list[str]]:
    name = kwd[56:56 + 128].decode("utf-16-le").split("\0")[0]
    desc = kwd[184:184 + 2048].decode("utf-16-le").split("\0")[0]
    count = struct.unpack_from("<H", kwd, 20)[0]
    paths = [kwd[PATHS_AT + 72 * i + 8:PATHS_AT + 72 * i + 72].split(b"\0")[0].decode("latin-1")
             for i in range(count)]
    return name, desc, paths


def variable_entries(data: bytes) -> list[bytes]:
    return [data[i:i + 16] for i in range(56, len(data), 16)]


def creature_records(things: bytes) -> list[bytes]:
    pos, out = HEADER_SIZE, []
    while pos < len(things):
        kind, size = struct.unpack_from("<II", things, pos)
        if kind == 200:
            out.append(things[pos:pos + 8 + size])
        pos += 8 + size
    return out


def add_thing_record(things: bytes, record: bytes) -> bytes:
    out = bytearray(things + record)
    struct.pack_into("<I", out, 8, len(out))
    struct.pack_into("<I", out, 20, struct.unpack_from("<I", out, 20)[0] + 1)
    struct.pack_into("<I", out, 52, len(out) - HEADER_SIZE)
    return bytes(out)


@needs_data
def test_shipped_folder_lists_exactly_the_expected_files(shipped1):
    found = sorted(p.relative_to(shipped1).as_posix() for p in shipped1.rglob("*") if p.is_file())
    assert found == sorted(["resource.toml", "Data/editor/Creatures.kwd",
                            *(f"Data/editor/maps/{name}{suffix}"
                              for name in SHIPPED_MAPS for suffix in MAP_SUFFIXES)])
    assert len(found) == 2 + 4 * 7


@needs_data
def test_flat_maps_are_the_generators_files(shipped1):
    from scripts.dk2_testmap import testmap_files
    for stem, (in_pool, shown) in FLAT.items():
        files = testmap_files(stem, molynx_in_pool=in_pool, display_name=shown)
        assert sorted(files) == sorted(stem + s for s in MAP_SUFFIXES if s != "Globals.kld")
        for name, data in files.items():
            assert (maps_dir(shipped1) / name).read_bytes() == data, name


@needs_data
def test_flat_maps_globals_are_siege_s_with_record_47(shipped1, stage1_dir):
    """Siege's Globals spliced with the folder's record 47 (the bytes of the copied Siege): not
    Dead Walking's, whose creature table makes its Imps impossible to pick up."""
    test_a = map_file(stage1_dir, "Siege", "Globals.kld")
    for stem in FLAT:
        data = map_file(shipped1, stem, "Globals.kld")
        assert data == test_a, stem
        (block,) = [b for b in read_blocks(data) if b.signature == CREATURES]
        assert block.item_count == 47
        assert read_creatures(block)[-1].get("creatureId") == 47


@needs_data
@needs_assets
def test_stage2_flat_maps_carry_the_stage2_record_and_need_no_other_files(stage2_full):
    first = map_file(stage2_full, "Molynx Test", "Globals.kld")
    for stem in FLAT:
        assert map_file(stage2_full, stem, "Globals.kld") == first, stem
        assert {n + ".kmf" for n in renamed_slots_from(stage2_full, stem, "Siege")} \
            == set(STAGE2_MESHES)
    assert files_under(meshes(stage2_full)) == STAGE2_MESHES
    assert files_under(textures(stage2_full)) == sorted(STAGE2_TEXTURES + ["GUI"])  # GUI: the panel pictures


def renamed_slots_from(mod: Path, stem: str, base: str) -> set[str]:
    molynx, bile = record_47(mod, stem), source_bile(base)
    return {molynx.resource_name(s) for s in ANIM_SLOTS
            if s != "unk_1" and molynx.resource_name(s) != bile.resource_name(s)}


@needs_data
def test_the_portal_variant_differs_only_in_the_variables_and_the_level_file(shipped1):
    plain, portal = "Molynx Test", "Molynx Test Portal"
    for suffix in ("Map.kld", "Things.kld", "Players.kld", "Triggers.kld", "Globals.kld"):
        assert map_file(shipped1, plain, suffix) == map_file(shipped1, portal, suffix), suffix
    a, b = (variable_entries(map_file(shipped1, s, "Variables.kld")) for s in (plain, portal))
    added = [e for e in b if e not in a]
    assert [struct.unpack("<iiii", e) for e in added] == [(1, 47, 50, 3), (2, 2 | 3 << 16, 47, 3)]
    assert [e for e in b if e not in added] == a


@needs_data
def test_dead_walking_is_the_flat_map_under_the_shipped_name(shipped1):
    for suffix in MAP_SUFFIXES:
        if suffix != ".kwd":
            assert map_file(shipped1, SECOND, suffix) == map_file(shipped1, "Molynx Test Portal", suffix), suffix
    name, desc, paths = level_strings(map_file(shipped1, SECOND, ".kwd"))
    assert (name, desc) == ("Molynx Test", DESCRIPTION)
    source = (map_source_dir(SECOND) / f"{SECOND}.kwd").read_bytes()
    assert paths == level_strings(source)[2]
    assert level_strings(map_file(shipped1, "Molynx Test Portal", ".kwd"))[0] == "Molynx Test Portal"


@needs_data
def test_every_level_file_path_names_a_file_of_the_folder(shipped1):
    present = {p.name for p in maps_dir(shipped1).iterdir()}
    for stem in FLAT:
        _name, _desc, paths = level_strings(map_file(shipped1, stem, ".kwd"))
        named = [p.rsplit("\\", 1)[1] for p in paths if p.startswith("Data\\Editor\\Maps\\")]
        assert sorted(named) == sorted(stem + s for s in MAP_SUFFIXES if s != ".kwd"), stem
        assert set(named) <= present


@needs_data
def test_shipped_building_twice_gives_identical_bytes(shipped1, tmp_path):
    again = build_mod(1, tmp_path, FOLDER)
    files = sorted(p.relative_to(shipped1) for p in shipped1.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    assert all((shipped1 / f).read_bytes() == (again / f).read_bytes() for f in files)


@needs_data
def test_a_map_both_copied_and_flat_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Dead Walking"):
        build_mod(1, tmp_path, maps=["Siege", SECOND], flat_maps={SECOND: False})
    with pytest.raises(ValueError, match="maps"):
        build_mod(1, tmp_path, maps=[], flat_maps={})
    assert list(tmp_path.iterdir()) == []


@needs_data
def test_explicit_maps_and_flat_maps_build_the_handed_over_layout(tmp_path):
    out = build_mod(1, tmp_path, maps=["Siege"], flat_maps={"Molynx Test": False, "Molynx Test Portal": True,
                                                            SECOND: True})
    report = audit(out)
    assert report.ok, "\n".join(report.lines)
    alone = build_mod(1, tmp_path / "alone", maps=[], flat_maps={"Molynx Test": False,
                                                                 "Molynx Test Portal": True, SECOND: True})
    (line,) = fail_lines(audit(alone))
    assert line.startswith("FAIL (g) folder:") and "map Siege is missing" in line


@needs_data
def test_shipped_resource_toml_names_the_flat_maps(shipped1):
    one = tomllib.loads((shipped1 / "resource.toml").read_text())["resource"]
    assert one["name"] == FOLDER
    assert "Molynx Test" in one["description"] and "Molynx Test Portal" in one["description"]
    assert "Enable only one Molynx folder at a time." in one["description"]


# ---- the audit of the flat maps -----------------------------------------------------------------

FLAT_RULES = [f"m{i}" for i in range(1, 13)]


@needs_data
def test_shipped_audit_is_ok_with_every_flat_map_rule(shipped1):
    report = audit(shipped1)
    assert report.ok, "\n".join(report.lines)
    for stem in FLAT:
        for rule in ["a", "b", "c", "f", *FLAT_RULES]:
            assert len(pass_lines(report, rule, stem)) == 1, (rule, stem, report.lines)
        (na,) = [line for line in report.lines if line.startswith(f"N/A (d, e) {stem}:")]
        assert "flat test map" in na
    (dw,) = [line for line in report.lines if line.startswith(f"N/A (d, e) {SECOND}:")]
    assert "no longer the shipped map" in dw
    for rule in "abcdef":
        assert len(pass_lines(report, rule, "Siege")) == 1
    assert [line for line in report.lines if line.startswith("PASS (m13) folder:")]
    (m5,) = pass_lines(report, "m5", "Molynx Test")
    assert "6 tiles thick" in m5
    (m10,) = pass_lines(report, "m10", "Molynx Test")
    assert "its 6 map paths name its own files, all in the folder; the 7th is the game's " \
           "GlobalVariables.kwd" in m10


@needs_data
@needs_assets
def test_stage2_shipped_audit_passes_the_flat_map_rules(stage2_full):
    report = audit(stage2_full)
    assert report.ok
    for stem in FLAT:
        for rule in FLAT_RULES:
            assert len(pass_lines(report, rule, stem)) == 1, (rule, stem)


@needs_data
def test_the_audit_does_not_use_the_map_generator():
    source = Path(audit_module.__file__).read_text()
    for forbidden in ("dk2_testmap", "testmap_files", "validate_testmap", "preview_png"):
        assert forbidden not in source, forbidden


def test_importing_the_audit_does_not_load_the_map_generator():
    """A fresh interpreter: importing the audit (and everything it imports) leaves dk2_testmap unloaded."""
    import os
    import subprocess
    import sys
    root = Path(audit_module.__file__).resolve().parent.parent
    code = ("import sys, scripts.audit_molynx_mod; "
            "assert 'scripts.audit_molynx_mod' in sys.modules; "
            "assert 'scripts.dk2_testmap' not in sys.modules, 'dk2_testmap was imported'; print('ok')")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(root)!r}); {code}"],
                          cwd=root, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


@needs_data
def test_a_flipped_path_record_type_fails_m10(flat_mod):
    path = maps_dir(flat_mod) / "Molynx Test.kwd"
    flip_byte(path, PATHS_AT + 72 * 1)  # the u32 type of the second path record (180, Players)
    (line,) = fail_lines(audit(flat_mod), "m10", "Molynx Test")
    assert "path record" in line


@needs_data
@pytest.mark.parametrize("at,what", [(56 + 2 * len("Molynx Test") + 4, "name"),
                                     (184 + 2 * 70, "description"),
                                     (PATHS_AT + 72 * 2 + 8 + 60, "path")])
def test_a_byte_after_the_nul_of_a_text_field_fails_m10(flat_mod, at, what):
    path = maps_dir(flat_mod) / f"{SECOND}.kwd"
    data = bytearray(path.read_bytes())
    assert data[at] == 0
    data[at] = 0x41
    path.write_bytes(bytes(data))
    (line,) = fail_lines(audit(flat_mod), "m10", SECOND)
    assert "after its NUL" in line and what in line


@needs_data
def test_a_flat_map_folder_without_siege_fails_rule_g(flat_mod):
    for suffix in MAP_SUFFIXES:
        (maps_dir(flat_mod) / f"Siege{suffix}").unlink()
    (line,) = fail_lines(audit(flat_mod), "g")
    assert "Siege" in line


@needs_data
def test_a_hole_in_keeper_2s_wall_fails(flat_mod):
    set_tile(flat_mod, "Molynx Test", 54, 58, bytes.fromhex("03020100"))  # Dirt beside the chamber
    report = audit(flat_mod)
    assert not report.ok
    assert fail_lines(report, "m5", "Molynx Test")
    assert fail_lines(report, "m13")  # Dead Walking is no longer the same map


@needs_data
def test_a_corridor_that_thins_keeper_2s_wall_to_nothing_fails(flat_mod):
    for x in range(36, 55):
        set_tile(flat_mod, "Molynx Test Portal", x, 55, bytes.fromhex("03020100"))
    (line,) = fail_lines(audit(flat_mod), "m5", "Molynx Test Portal")
    assert "Impenetrable" in line


@needs_data
def test_a_breach_into_keeper_1s_area_fails(flat_mod):
    for x in range(34, 55):
        set_tile(flat_mod, SECOND, x, 52, bytes.fromhex("03020100"))
    for y in range(52, 56):  # down to (54, 55), beside the chamber's corner (55, 55)
        set_tile(flat_mod, SECOND, 54, y, bytes.fromhex("03020100"))
    (line,) = fail_lines(audit(flat_mod), "m5", SECOND)
    assert "reaches" in line


@needs_data
def test_a_second_molynx_thing_fails(flat_mod):
    path = maps_dir(flat_mod) / "Molynx TestThings.kld"
    things = path.read_bytes()
    (molynx,) = [r for r in creature_records(things) if r[8 + 26] == 47]
    extra = bytearray(molynx)
    struct.pack_into("<i", extra, 8, 9)  # x 9, on the claimed floor
    path.write_bytes(add_thing_record(things, bytes(extra)))
    report = audit(flat_mod)
    assert fail_lines(report, "m12", "Molynx Test")
    assert fail_lines(report, "m6", "Molynx Test")


@needs_data
def test_a_changed_border_byte_fails(flat_mod):
    path = maps_dir(flat_mod) / "Molynx TestMap.kld"
    flip_byte(path, MAP_HEADER + 4 * 5 + 1)  # the player byte of tile (5, 0)
    assert fail_lines(audit(flat_mod), "m1", "Molynx Test")


@needs_data
def test_portal_variables_equal_to_the_plain_ones_fail(flat_mod):
    shutil.copyfile(maps_dir(flat_mod) / "Molynx TestVariables.kld",
                    maps_dir(flat_mod) / "Molynx Test PortalVariables.kld")
    report = audit(flat_mod)
    assert fail_lines(report, "m8", "Molynx Test Portal")
    (line,) = fail_lines(report, "m13")
    assert "Molynx Test Portal" in line


@needs_data
def test_a_missing_flat_map_file_fails(flat_mod):
    (maps_dir(flat_mod) / "Molynx Test PortalTriggers.kld").unlink()
    report = audit(flat_mod)
    assert any("Molynx Test PortalTriggers.kld" in line for line in fail_lines(report, "g"))
    assert fail_lines(report, "m9", "Molynx Test Portal")


@needs_data
def test_a_missing_flat_map_fails(flat_mod):
    for suffix in MAP_SUFFIXES:
        (maps_dir(flat_mod) / f"Molynx Test Portal{suffix}").unlink()
    (line,) = fail_lines(audit(flat_mod), "g")
    assert "Molynx Test Portal" in line


@needs_data
def test_an_unexpected_extra_file_fails(flat_mod):
    (maps_dir(flat_mod) / "Molynx Test 2.kwd").write_bytes(map_file(flat_mod, "Molynx Test", ".kwd"))
    assert fail_lines(audit(flat_mod), "g")


@needs_data
def test_a_changed_globals_record_of_a_flat_map_fails(flat_mod):
    def change(creatures):
        creatures[10].set("hp", creatures[10].get("hp") + 1)
        return creatures

    edit_creatures(flat_mod, "Molynx Test", change)
    (line,) = fail_lines(audit(flat_mod), "b", "Molynx Test")
    assert "record 11" in line


@needs_data
def test_a_moved_heart_thing_fails(flat_mod):
    path = maps_dir(flat_mod) / "Molynx Test PortalThings.kld"
    data = bytearray(path.read_bytes())
    pos = HEADER_SIZE
    while pos < len(data):
        kind, size = struct.unpack_from("<II", data, pos)
        if kind == 204 and data[pos + 8 + 19] == 4:
            struct.pack_into("<i", data, pos + 8, 57)
        pos += 8 + size
    path.write_bytes(bytes(data))
    assert fail_lines(audit(flat_mod), "m2", "Molynx Test Portal")


@needs_data
def test_a_disabled_room_fails(flat_mod):
    path = maps_dir(flat_mod) / "Molynx TestVariables.kld"
    data = bytearray(path.read_bytes())
    for at in range(56, len(data), 16):
        if struct.unpack_from("<iHHii", data, at)[:4] == (2, 1, 3, 14):
            struct.pack_into("<i", data, at + 12, 1)
    path.write_bytes(bytes(data))
    (line,) = fail_lines(audit(flat_mod), "m8", "Molynx Test")
    assert "room 14" in line


@needs_data
def test_a_changed_description_fails(flat_mod):
    path = maps_dir(flat_mod) / f"{SECOND}.kwd"
    data = bytearray(path.read_bytes())
    data[184:186] = "X".encode("utf-16-le")
    path.write_bytes(bytes(data))
    (line,) = fail_lines(audit(flat_mod), "m10", SECOND)
    assert "description" in line


@needs_data
def test_a_triggers_file_with_a_script_fails(flat_mod):
    shutil.copyfile(map_source_dir(SECOND) / f"{SECOND}Triggers.kld",
                    maps_dir(flat_mod) / "Molynx TestTriggers.kld")
    assert fail_lines(audit(flat_mod), "m9", "Molynx Test")


@needs_data
def test_a_richer_keeper_2_fails(flat_mod):
    path = maps_dir(flat_mod) / "Molynx Test PortalPlayers.kld"
    data = bytearray(path.read_bytes())
    data[56 + 3 * 205] ^= 0x01  # Keeper 2's starting gold
    path.write_bytes(bytes(data))
    assert fail_lines(audit(flat_mod), "m7", "Molynx Test Portal")


@needs_data
@needs_assets
@pytest.mark.parametrize("field,value", [("scale", 4096), ("scale", 5735), ("height", 2662),
                                         ("eyeHeight", 1515)])
def test_stage2_record_must_have_the_horned_reapers_size(mod2, field, value):
    def wrong(creatures):
        creatures[-1].set(field, value)
        return creatures

    edit_creatures(mod2, "Siege", wrong)
    lines = fail_lines(audit(mod2), "b")
    assert lines and all("Siege" in line for line in lines) and any(field in line for line in lines)


# ---- the base creature table (Data/editor/Creatures.kwd, rule i) --------------------------------

@needs_data
def test_editor_creatures_is_the_editors_file_with_molynx_as_record_47(stage1_dir):
    """The game loads the editor's Creatures.kwd as its base creature table and applies a map's
    Globals as an update by creatureId (the Test B crash: no creature 47 to update)."""
    source = (EDITOR_DIR / "Creatures.kwd").read_bytes()
    (src_block,) = read_blocks(source)
    (block,) = read_blocks((stage1_dir / "Data" / "editor" / "Creatures.kwd").read_bytes())
    old, new = read_creatures(src_block), read_creatures(block)
    assert len(old) == 46 and len(new) == 47
    assert [c.raw for c in new[:46]] == [c.raw for c in old]
    molynx, bile = new[46], next(c for c in old if c.get("creatureId") == 3)
    assert (molynx.get("creatureId"), molynx.get("name")) == (47, "Molynx")
    assert len(molynx.raw) == len(bile.raw) == 5449
    assert molynx.get("orderInEditor") == max(c.get("orderInEditor") for c in old) + 1


@needs_data
def test_build_can_leave_the_editor_creatures_out(tmp_path):
    out = build_mod(1, tmp_path, FOLDER, maps=["Siege"], editor_creatures=False)
    assert not (out / "Data" / "editor" / "Creatures.kwd").exists()


@needs_data
def test_audit_fails_rule_i_without_the_editor_creatures(mod):
    (mod / "Data" / "editor" / "Creatures.kwd").unlink()
    report = audit(mod)
    assert not report.ok
    assert any(line.startswith("FAIL (i)") and "Creatures.kwd is missing" in line for line in report.lines)


@needs_data
def test_audit_fails_when_the_editor_creatures_lacks_record_47(mod):
    (mod / "Data" / "editor" / "Creatures.kwd").write_bytes((EDITOR_DIR / "Creatures.kwd").read_bytes())
    report = audit(mod)
    assert not report.ok
    assert any(line.startswith("FAIL (b) editor Creatures.kwd") for line in report.lines)


@needs_data
def test_stage2_editor_creatures_names_the_same_molynx_files(stage2_full):
    report = audit(stage2_full)
    assert report.ok, "\n".join(l for l in report.lines if l.startswith("FAIL"))
    assert any(line.startswith("PASS (f) editor Creatures.kwd") and "Molynx meshes" in line
               for line in report.lines)


@needs_data
@needs_assets
def test_a_damaged_mesh_fails_the_editor_table_too(mod2):
    (meshes(mod2) / "Molynx-walk.kmf").write_bytes(b"")
    report = audit(mod2)
    assert fail_lines(report, "f", EDITOR_LABEL)
    assert fail_lines(report, "f", "Siege")


@needs_data
def test_a_level_file_with_the_bases_script_root_fails_m10(flat_mod):
    """Root trigger 1 with the empty Triggers file hung the game as the map started (Test B)."""
    path = maps_dir(flat_mod) / "Molynx Test.kwd"
    data = bytearray(path.read_bytes())
    struct.pack_into("<H", data, audit_module.LEVEL_SCRIPT_ROOT, 1)
    path.write_bytes(bytes(data))
    assert any("root trigger" in line for line in fail_lines(audit(flat_mod), "m10", "Molynx Test"))


CAN_BE_PICKED_UP = 0x2   # CreatureFlag bit (OpenKeeper's Creature.CreatureFlag)


@needs_data
def test_flat_maps_imps_can_be_picked_up(shipped1):
    """Second review round: on the flat maps the Imps could only be slapped. Dead Walking's own
    creature table clears the Imp's can-be-picked-up flag; the flat maps now use Siege's table,
    whose Imp is the editor's."""
    def imp(data):
        (block,) = [b for b in read_blocks(data) if b.signature == CREATURES]
        (rec,) = [c for c in read_creatures(block) if c.get("creatureId") == 1]
        return rec
    assert not imp((map_source_dir(SECOND) / f"{SECOND}Globals.kld").read_bytes()).get("flags") \
        & CAN_BE_PICKED_UP
    for stem in FLAT:
        assert imp(map_file(shipped1, stem, "Globals.kld")).get("flags") & CAN_BE_PICKED_UP, stem


@needs_data
@needs_assets
def test_stage2_ships_his_creature_panel_pictures(stage2_full):
    """Second review round: his own panel pictures, where Flame finds the textures the record's
    icon1 / portrait resources name (GUI\\Creatures\\Evil\\molynx and molynx-bg)."""
    from PIL import Image
    folder = stage2_full / "Data" / "EngineTextures" / "GUI" / "Creatures" / "Evil"
    assert {p.name: Image.open(p).size for p in folder.iterdir()} == {
        "molynx.png": (64, 64), "molynx-bg.png": (64, 128)}
    for name in MAPS:
        molynx = record_47(stage2_full, name)
        assert molynx.resource_name("icon1Resource") == r"gui\creatures\evil\molynx"
        assert molynx.resource_name("portraitResource") == r"gui\creatures\evil\molynx-bg"


# ---- Ground Slam (stage 3): every Globals, the editor's four tables, the icon; rule (j) ----------

SLAM_FILES = ["Data/EngineTextures/GUI/C-Spells/c-ground_slam.png", "Data/editor/CreatureSpells.kwd",
              "Data/editor/EffectElements.kwd", "Data/editor/Effects.kwd", "Data/editor/Shots.kwd"]


@pytest.fixture(scope="module")
def stage3_dir(tmp_path_factory, assets_dir):
    return build_mod(3, tmp_path_factory.mktemp("stage3"), "Molynx_full", maps=["Siege"],
                     flat_maps={}, assets_dir=assets_dir)


@pytest.fixture
def mod3(stage3_dir, tmp_path):
    target = tmp_path / "Molynx_full"
    shutil.copytree(stage3_dir, target)
    return target


@needs_data
@needs_assets
def test_stage3_ships_ground_slam_and_passes_rule_j(stage3_dir):
    files = sorted(p.relative_to(stage3_dir).as_posix() for p in stage3_dir.rglob("*") if p.is_file())
    assert set(SLAM_FILES) <= set(files)
    report = audit(stage3_dir)
    assert report.ok, fail_lines(report)
    (line,) = [l for l in report.lines if l.startswith("PASS (j)")]
    assert "Ground Slam in the Globals of 1 map(s)" in line
    assert any("Ground Slam's 8 records" in l for l in report.lines if l.startswith("PASS (a) Siege"))
    with Image.open(stage3_dir / SLAM_FILES[0]) as icon:
        assert icon.size == (64, 64)
    molynx = record_47(stage3_dir, "Siege")
    assert (molynx.get("spells[0].creatureSpellId"), molynx.get("spells[1].creatureSpellId")) == (23, 15)
    assert molynx.resource_name("animMagicResource") == "Molynx-ground_slam"


@needs_data
@needs_assets
def test_stage2_has_no_ground_slam(stage2_dir):
    assert not any((stage2_dir / f).exists() for f in SLAM_FILES)
    report = audit(stage2_dir)
    (line,) = [l for l in report.lines if l.startswith("PASS (j)")]
    assert "no Ground Slam at stage 2" in line


@needs_data
@needs_assets
def test_a_changed_ground_slam_field_fails_rule_a(mod3):
    from scripts.dk2kwd_tables import read_records

    def edit(blocks):
        out = []
        for b in blocks:
            if b.signature == 0x104:
                records = read_records(b)
                (shot,) = [r for r in records if r.id == 51]
                shot.set("data1", 999)
                b = _rebuilt(b, records)
            out.append(b)
        return out
    edit_globals(mod3, "Siege", edit)
    report = audit(mod3)
    lines = fail_lines(report, "a", "Siege")
    assert lines and "Shots 51 is not Ground Slam's: fields ['data1']" in lines[0]


def _rebuilt(block, records):
    from scripts.dk2kwd_blocks import rebuilt_block
    return rebuilt_block(block, [bytes(r.raw) for r in records])


@needs_data
@needs_assets
def test_ground_slam_in_only_some_tables_fails_rule_a(mod3):
    source = read_blocks((map_source_dir("Siege") / "SiegeGlobals.kld").read_bytes())
    effects = next(b for b in source if b.signature == 0x10E)
    edit_globals(mod3, "Siege", lambda blocks: [effects if b.signature == 0x10E else b for b in blocks])
    lines = fail_lines(audit(mod3), "a", "Siege")
    assert lines and "only some of the tables" in lines[0]


@needs_data
@needs_assets
@pytest.mark.parametrize("missing", SLAM_FILES)
def test_a_missing_ground_slam_file_fails_rule_j(mod3, missing):
    (mod3 / missing).unlink()
    report = audit(mod3)
    assert not report.ok
    lines = fail_lines(report, "j")
    assert lines and "missing" in lines[0]


@needs_data
@needs_assets
def test_an_editor_table_without_ground_slam_fails_rule_j(mod3):
    shutil.copyfile(EDITOR_DIR / "Effects.kwd", mod3 / "Data/editor/Effects.kwd")
    lines = fail_lines(audit(mod3), "j")
    assert lines and "Effects.kwd is the editor's own, without Ground Slam" in lines[0]


@needs_data
@needs_assets
def test_a_wrong_sized_icon_fails_rule_j(mod3):
    Image.new("RGB", (32, 32)).save(mod3 / SLAM_FILES[0])
    lines = fail_lines(audit(mod3), "j")
    assert lines and "is not a 64 x 64 PNG" in lines[0]
