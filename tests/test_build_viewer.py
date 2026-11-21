import json
from pathlib import Path

import pytest

from scripts.viewer_creatures import CREATURES, find_animations

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
MESHES = S / "extracted/Meshes"
needs_game = pytest.mark.skipif(not MESHES.exists(), reason="DK2 files not present")


@needs_game
def test_every_creature_in_table_has_pose_and_at_least_one_animation():
    anims, _ = find_animations(MESHES)
    stems = {p.stem.lower() for p in MESHES.glob("*.kmf")}
    for cid, d in CREATURES.items():
        assert anims[cid], cid
        assert d.pose == "" or d.pose.lower() in stems, (cid, d.pose)


@needs_game
def test_unattributed_list_contains_no_creature_prefix_in_table():
    _, unattributed = find_animations(MESHES)
    prefixes = tuple(p for d in CREATURES.values() for p in d.prefixes)
    assert not [u for u in unattributed if u.lower().startswith(prefixes)]


@needs_game
def test_build_writes_a_complete_site(tmp_path):
    from scripts.build_viewer import build
    from scripts.viewer_textures import texture_file
    from scripts.dk2texture import TextureCache
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    summary = build(tmp_path, MESHES, cache, ["goblin", "chicken"])
    idx = json.loads((tmp_path / "data/index.json").read_text())
    assert [c["id"] for c in idx["creatures"]] == ["goblin", "chicken"]
    for name in ("index.html", "app.js", "dk2v.js", "vendor/three.module.js", "vendor/OrbitControls.js"):
        assert (tmp_path / name).exists(), name
    for c in idx["creatures"]:
        assert (tmp_path / "data" / c["file"]).exists()
        for t in c["textures"]:
            f = texture_file(t)
            assert (tmp_path / "textures/original" / f).exists() and (tmp_path / "textures/hd" / f).exists()
    assert summary["total_bytes"] > 0


@needs_game
def test_text_data_build_replaces_binary_files(tmp_path):
    from scripts.build_viewer import build
    from scripts.dk2texture import TextureCache
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    build(tmp_path, MESHES, cache, ["chicken"], text_data=True)
    idx = json.loads((tmp_path / "data/index.json").read_text())
    c = idx["creatures"][0]
    assert c["file"] == "chicken.dk2v.txt" and c["encoding"] == "gzip-base64"
    assert (tmp_path / "data/chicken.dk2v.txt").exists()
    assert not list((tmp_path / "data").glob("*.dk2v"))


@needs_game
def test_build_includes_custom_creature_with_its_own_textures(tmp_path):
    from PIL import Image
    from scripts.build_viewer import build
    from scripts.dk2texture import TextureCache
    from scripts.viewer_textures import texture_file
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    custom = Path(__file__).resolve().parent.parent / "creatures_archive"      # the Nautiloth spec, kept as the custom-creature example
    summary = build(tmp_path / "site", MESHES, cache, ["chicken"], custom_dir=custom)
    idx = json.loads((tmp_path / "site/data/index.json").read_text())
    ids = [c["id"] for c in idx["creatures"]]
    assert ids == ["chicken", "nautiloth"]
    naut = idx["creatures"][1]
    assert naut["label"] == "Nautiloth" and not naut["missing_textures"]
    assert len(naut["animations"]) >= 30
    for t in naut["textures"]:
        o = tmp_path / "site/textures/original" / texture_file(t)
        h = tmp_path / "site/textures/hd" / texture_file(t)
        if t.startswith("custom/"):
            assert Image.open(h).size == (4 * Image.open(o).size[0], 4 * Image.open(o).size[1])
        assert o.exists() and h.exists()
    guides = list((custom / "nautiloth/generated").glob("*_uv.png"))
    assert len(guides) == 6 and "nautiloth" in summary["custom"]
