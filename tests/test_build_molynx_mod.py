"""Stage-1 Molynx mod: the builder (`build_mod`), the independent audit (`audit`) and the test maps.

The builds read the real DK2 maps, so every test that needs them skips when they are absent.
"""
import re
import shutil
import struct
import tomllib
from pathlib import Path

import pytest

import scripts.audit_molynx_mod as audit_module
import scripts.build_molynx_mod as builder
from scripts.audit_molynx_mod import audit
from scripts.build_molynx_mod import build_mod
from scripts.dk2kld_splice import (MAP_SUFFIXES, level_flags, map_source_dir, pick_skirmish_test_map,
                                   pick_test_map)
from scripts.dk2kwd_blocks import HEADER_SIZE, Block, read_blocks, write_blocks
from scripts.dk2kwd_creatures import ANIM_SLOTS, FIELDS, Creature, creatures_block, read_creatures
from scripts.dk2paths import GAME_DIR, MAPS_DIR, QUUZ_MAPS_DIR, have

needs_data = pytest.mark.skipif(
    not (have(MAPS_DIR) and have(QUUZ_MAPS_DIR) and have(GAME_DIR / "Meshes.WAD")),
    reason="DK2 editor maps, Quuz maps or Meshes.WAD not present")

CREATURES = 0xAA
SKIRMISH_FLAG = 1 << 11
MULTIPLAYER_FLAG = 1 << 10
FOLDER = "Molynx_slot_only"
MAPS = ["Siege", "13coop"]
TEXT_IDS = {"name": 3001, "tooltip": 3002, "general": 3003, "strength": 3004, "weakness": 3005}


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
def stage2_dir(tmp_path_factory):
    return build_mod(2, tmp_path_factory.mktemp("stage2"), "Molynx_stage2", maps=["Siege"])


@pytest.fixture
def mod2(stage2_dir, tmp_path):
    target = tmp_path / "Molynx_stage2"
    shutil.copytree(stage2_dir, target)
    return target


# ---- the test maps ------------------------------------------------------------------------------

@needs_data
def test_test_maps_are_pinned():
    assert pick_test_map() == "Siege"
    assert pick_skirmish_test_map() == "13coop"
    assert builder.test_maps() == MAPS


@needs_data
def test_the_users_siege_copy_has_no_skirmish_flag_but_the_second_map_does():
    # Why a second test map exists: the Quuz Siege.kwd (the copy the game loads) is a multiplayer
    # map but not a skirmish map, so it may not show in the Skirmish list; 13coop does.
    siege = (map_source_dir("Siege") / "Siege.kwd").read_bytes()
    second = (map_source_dir("13coop") / "13coop.kwd").read_bytes()
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


@needs_data
def test_skirmish_pick_recomputed_independently():
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
        data, pos, bile = paths[2].read_bytes(), HEADER_SIZE, False
        while pos < len(data):
            kind, size = struct.unpack_from("<II", data, pos)
            bile = bile or (kind == 200 and data[pos + 34] == 3 and 3 <= data[pos + 35] <= 7)
            pos += 8 + size
        if not bile:
            continue
        both = all((d / (name + ".kwd")).is_file() for d in (QUUZ_MAPS_DIR, MAPS_DIR))
        flags = struct.unpack_from("<H", paths[0].read_bytes(), 56 + 25484)[0]
        rows.append((not both, sum(p.stat().st_size for p in paths), name, flags))
    assert len(rows) == 41
    others = [r for r in rows if r[2] != pick_test_map() and r[3] & SKIRMISH_FLAG]
    assert min(others)[2] == pick_skirmish_test_map() == "13coop"
    assert min(others)[1] == 734_662


# ---- build_mod: refusals ------------------------------------------------------------------------

@pytest.mark.parametrize("stage", [0, 4, -1, 2.5, "1", None])
def test_build_refuses_a_bad_stage(tmp_path, stage):
    with pytest.raises(ValueError, match="stage"):
        build_mod(stage, tmp_path)
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
        diff = [i for i in range(len(bile.raw)) if molynx.raw[i] != bile.raw[i]]
        assert diff, "Molynx must differ from the Bile Demon"
        assert molynx.get("name") == "Molynx" and molynx.get("creatureId") == 47
        assert molynx.get("orderInEditor") == 47


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
                      "molynx_for_map", "retarget_creature_thing"):
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
    assert not fail_lines(report, "a", "13coop")
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

    edit_globals(mod, "13coop", corrupt_header)
    report = audit(mod)
    (line,) = fail_lines(report, "a")
    assert "13coop" in line and label in line


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
    assert not fail_lines(report, "e", "13coop")


@needs_data
@pytest.mark.parametrize("suffix", [".kwd", "Players.kld", "Triggers.kld", "Variables.kld"])
def test_every_other_copied_file_is_compared_with_its_source(mod, suffix):
    flip_byte(maps_dir(mod) / f"13coop{suffix}", 60)
    report = audit(mod)
    (line,) = fail_lines(report, "e")
    assert f"13coop{suffix}" in line


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
    path = maps_dir(mod) / "13coopThings.kld"
    data = bytearray(path.read_bytes())
    (at,) = [i for i, (a, b) in enumerate(zip(data, (map_source_dir("13coop") / "13coopThings.kld").read_bytes()))
             if a != b]
    data[at] = 46
    path.write_bytes(bytes(data))
    report = audit(mod)
    (line,) = fail_lines(report, "d")
    assert "13coop" in line and "47" in line


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

    edit_creatures(mod, "13coop", wrong)
    (line,) = fail_lines(audit(mod), "b")
    assert "13coop" in line and needle in line


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

    edit_globals(mod, "13coop", touch_timestamp)
    (line,) = fail_lines(audit(mod), "c")
    assert "13coop" in line


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


# ---- audit: stage 2 (animation and mesh names) --------------------------------------------------

@needs_data
def test_stage2_audit_passes_with_every_renamed_animation_present(mod2):
    names = renamed_slots(mod2, "Siege")
    assert "Molynx-walk" in names and len(names) > 10
    add_placeholder_meshes(mod2, names)
    report = audit(mod2)
    assert report.ok, "\n".join(report.lines)
    assert any("stage 2" in line for line in report.lines if line.startswith("PASS (b)"))
    (line,) = pass_lines(report, "f", "Siege")
    assert f"{len(names)} Molynx meshes" in line


@needs_data
def test_stage2_audit_fails_naming_every_missing_mesh_when_none_is_delivered(mod2):
    report = audit(mod2)
    assert not report.ok
    (line,) = fail_lines(report, "f")
    for name in ("Molynx-walk", "Molynx-DraggedPose", "Molynx-stand"):
        assert f"{name}.kmf" in line
    assert not fail_lines(report, "a") and not fail_lines(report, "b") and not fail_lines(report, "d")


@needs_data
def test_stage2_audit_fails_when_one_mesh_is_missing(mod2):
    add_placeholder_meshes(mod2, renamed_slots(mod2, "Siege"))
    (mod2 / "Data" / "Meshes" / "Molynx-walk.kmf").unlink()
    (line,) = fail_lines(audit(mod2), "f")
    assert "Molynx-walk.kmf" in line and "Molynx-attack" not in line


@needs_data
def test_stage2_mesh_names_must_match_the_case_exactly(mod2):
    names = renamed_slots(mod2, "Siege")
    add_placeholder_meshes(mod2, names - {"Molynx-DraggedPose"})
    add_placeholder_meshes(mod2, {"Molynx-draggedpose"})
    (line,) = fail_lines(audit(mod2), "f")
    assert "Molynx-DraggedPose.kmf" in line


@needs_data
def test_stage2_an_empty_mesh_file_fails(mod2):
    add_placeholder_meshes(mod2, renamed_slots(mod2, "Siege"))
    (mod2 / "Data" / "Meshes" / "Molynx-walk.kmf").write_bytes(b"")
    (line,) = fail_lines(audit(mod2), "f")
    assert "Molynx-walk.kmf" in line and "empty" in line


@needs_data
def test_stage2_a_slot_left_as_the_bile_demons_matches_no_stage(mod2):
    add_placeholder_meshes(mod2, renamed_slots(mod2, "Siege"))
    bile = source_bile("Siege")

    def restore(creatures):
        creatures[-1].set_resource_name("animWalkResource", bile.resource_name("animWalkResource"))
        return creatures

    edit_creatures(mod2, "Siege", restore)
    (line,) = fail_lines(audit(mod2), "b")
    assert "animWalkResource" in line


@needs_data
def test_stage2_a_slot_not_named_molynx_fails(mod2):
    add_placeholder_meshes(mod2, renamed_slots(mod2, "Siege") | {"Other-walk"})

    def rename(creatures):
        creatures[-1].set_resource_name("animWalkResource", "Other-walk")
        return creatures

    edit_creatures(mod2, "Siege", rename)
    (line,) = fail_lines(audit(mod2), "b")
    assert "animWalkResource" in line and "Molynx-" in line


@needs_data
def test_stage2_a_changed_tail_of_a_slot_fails(mod2):
    """Only the 64-byte name of a slot may change; its other 20 bytes stay the Bile Demon's."""
    add_placeholder_meshes(mod2, renamed_slots(mod2, "Siege"))

    def tweak(creatures):
        creatures[-1].raw[FIELDS["animWalkResource"].offset + 70] ^= 0x01
        return creatures

    edit_creatures(mod2, "Siege", tweak)
    (line,) = fail_lines(audit(mod2), "b")
    assert "animWalkResource" in line


@needs_data
def test_stage2_empty_bile_slots_stay_empty_and_other_fields_are_unchanged(mod2):
    molynx, bile = record_47(mod2, "Siege"), source_bile("Siege")
    assert molynx.resource_name("animRunResource") == "" == bile.resource_name("animRunResource")
    assert molynx.resource_name("unk_1") == bile.resource_name("unk_1")
    assert all(molynx.resource_name(s).startswith("Molynx-")
               for s in ANIM_SLOTS if s != "unk_1" and bile.resource_name(s))


# ---- audit: stage 3 is fail-closed until its icons and textures can be checked -------------------

@needs_data
def test_stage3_fields_are_recognised_but_icons_and_textures_are_not_yet_checked(tmp_path):
    out = build_mod(3, tmp_path, "Molynx_full", maps=["Siege"], text_ids=TEXT_IDS)
    add_placeholder_meshes(out, renamed_slots(out, "Siege"))
    report = audit(out)
    assert any("stage 3" in line for line in report.lines if line.startswith("PASS (b)"))
    assert not report.ok
    (line,) = fail_lines(report)
    assert line.startswith("FAIL (f)") and "icon" in line and "not checked" in line


@needs_data
def test_stage3_text_ids_are_part_of_the_record(tmp_path):
    out = build_mod(3, tmp_path, "Molynx_full", maps=["13coop"], text_ids=TEXT_IDS)
    molynx = record_47(out, "13coop")
    assert molynx.get("nameStringId") == 3001 and molynx.get("weaknessStringId") == 3005
    assert molynx.resource_name("icon1Resource") == r"gui\creatures\evil\molynx"
    assert molynx.get("hp") > source_bile("13coop").get("hp")
