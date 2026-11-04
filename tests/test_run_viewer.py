import json
from pathlib import Path

import pytest

from scripts.run_viewer import find_game_files, prepare

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
GAME = S / "dk2game/Ai"


def test_find_game_files_is_case_insensitive(tmp_path):
    for n in ("MESHES.wad", "enginetextures.DIR", "EngineTextures.dat"):
        (tmp_path / n).write_bytes(b"x")
    wad, d, dat = find_game_files(tmp_path)
    assert (wad.name, d.name, dat.name) == ("MESHES.wad", "enginetextures.DIR", "EngineTextures.dat")


def test_find_game_files_names_what_is_missing(tmp_path):
    (tmp_path / "Meshes.WAD").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="EngineTextures"):
        find_game_files(tmp_path)


def test_find_game_files_looks_one_level_down(tmp_path):
    sub = tmp_path / "Data"
    sub.mkdir()
    for n in ("Meshes.WAD", "EngineTextures.dir", "EngineTextures.dat"):
        (sub / n).write_bytes(b"x")
    assert find_game_files(tmp_path)[0].parent == sub


@pytest.mark.skipif(not GAME.exists(), reason="DK2 files not present")
def test_prepare_extracts_and_builds_then_reuses(tmp_path):
    site = prepare(GAME, tmp_path / "work", ["chicken"])
    idx = json.loads((site / "data/index.json").read_text())
    assert [c["id"] for c in idx["creatures"]] == ["chicken", "golem", "battlemage", "behemoth"]   # the bundled Golem, Battlemage and Behemoth ride along; the Nautiloth was removed
    assert (site / "index.html").exists()
    stamp = (site / "data/index.json").stat().st_mtime_ns
    assert prepare(GAME, tmp_path / "work", ["chicken"]) == site
    assert (site / "data/index.json").stat().st_mtime_ns == stamp      # second run does not rebuild


def test_files_may_be_in_different_folders(tmp_path):
    root = tmp_path / "Dungeon Keeper 2"
    (root / "Data").mkdir(parents=True)
    (root / "Data/Meshes.WAD").write_bytes(b"x")
    (root / "EngineTextures.dir").write_bytes(b"x")
    (root / "EngineTextures.dat").write_bytes(b"x")
    wad, d, dat = find_game_files(root / "Data")      # user pasted the Data folder
    assert wad.parent.name == "Data" and d.parent == root and dat.parent == root
    assert find_game_files(root)[0] == wad            # user pasted the game folder


def test_missing_cache_message_says_where_to_get_it(tmp_path):
    (tmp_path / "Meshes.WAD").write_bytes(b"x")
    with pytest.raises(FileNotFoundError) as e:
        find_game_files(tmp_path)
    assert "EngineTextures.dir" in str(e.value) and "copy" in str(e.value).lower()


@pytest.mark.skipif(not GAME.exists(), reason="DK2 files not present")
def test_prepare_without_custom_creatures_and_rebuild_on_change(tmp_path):
    empty = tmp_path / "none"; empty.mkdir()
    site = prepare(GAME, tmp_path / "work", ["chicken"], custom_dir=empty)
    assert [c["id"] for c in json.loads((site / "data/index.json").read_text())["creatures"]] == ["chicken", "golem", "battlemage", "behemoth"]   # no custom creatures; the Golem, Battlemage and Behemoth are bundled
    (empty / "x.txt").write_text("changed")          # any change in the custom folder forces a rebuild
    stamp = (site / "data/index.json").stat().st_mtime_ns
    prepare(GAME, tmp_path / "work", ["chicken"], custom_dir=empty)
    assert (site / "data/index.json").stat().st_mtime_ns != stamp


@pytest.mark.skipif(not GAME.exists(), reason="DK2 files not present")
def test_prepared_viewer_has_the_golem_with_animations_and_sounds_and_no_nautiloth(tmp_path):
    site = prepare(GAME, tmp_path / "work", ["chicken"])
    idx = json.loads((site / "data/index.json").read_text())
    ids = [c["id"] for c in idx["creatures"]]
    assert "golem" in ids and "nautiloth" not in ids
    golem = next(c for c in idx["creatures"] if c["id"] == "golem")
    labels = [a["label"] for a in golem["animations"]]
    assert {"stand", "walk", "attack", "die", "eating", "knockback", "fighting_another_creature"} <= set(labels)
    assert "casting_spell" not in labels and "researching" not in labels      # a brute, like the Bile Demon
    assert golem["sounds"] == "golem.sounds.json" and (site / "data/golem.sounds.json").exists()


@pytest.mark.skipif(not GAME.exists(), reason="DK2 files not present")
def test_prepared_viewer_has_the_battlemage_with_casting_research_casino_and_sounds(tmp_path):
    site = prepare(GAME, tmp_path / "work", ["chicken"])
    idx = json.loads((site / "data/index.json").read_text())
    bm = next(c for c in idx["creatures"] if c["id"] == "battlemage")
    labels = [a["label"] for a in bm["animations"]]
    assert {"stand", "walk", "attack", "die", "casting_spell", "researching", "praying", "knockback"} <= set(labels)
    assert bm["sounds"] == "battlemage.sounds.json" and (site / "data/battlemage.sounds.json").exists()


@pytest.mark.skipif(not GAME.exists(), reason="DK2 files not present")
def test_prepared_viewer_has_the_behemoth_with_dk2_clips_no_invented_ones_and_sounds(tmp_path):
    site = prepare(GAME, tmp_path / "work", ["chicken"])
    idx = json.loads((site / "data/index.json").read_text())
    bh = next(c for c in idx["creatures"] if c["id"] == "behemoth")
    names = {a["label"] for a in bh["animations"]}
    assert {"stand", "walk", "attack", "die", "running", "sleeping", "praying", "knockback", "manufacturing"} <= names
    assert not names & {"casting_spell", "researching", "gambling", "blocking"}
    assert bh["sounds"] == "behemoth.sounds.json" and (site / "data/behemoth.sounds.json").exists()
