import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
from scripts.kmf_textured import _yup
from scripts.viewer_creatures import CREATURES, find_animations
from scripts.viewer_export import export_creature, read_dk2v, write_index

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
MESHES = S / "extracted/Meshes"
needs_game = pytest.mark.skipif(not MESHES.exists(), reason="DK2 files not present")
ONLY = {k: CREATURES[k] for k in ("goblin", "bile_demon")}


class FakeCache:
    def __init__(self, names):
        self.names = names


def _all_names():
    from scripts.dk2texture import TextureCache
    return TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")


@pytest.fixture(scope="module")
def cache():
    return _all_names()


@pytest.fixture(scope="module")
def goblin(cache, tmp_path_factory):
    out = tmp_path_factory.mktemp("dk2v")
    path = export_creature("goblin", MESHES, cache, out, ONLY)
    return path, read_dk2v(path)


def _expected(file_stem, frame):
    anim = parse_kmf((MESHES / f"{file_stem}.kmf").read_bytes())
    sprites = anim.anim.sprites
    pts = evaluate_anim_frame(anim.anim, frame)
    return [_yup(p) for s, p in zip(sprites, pts) if s.triangles[0]]


def _check(model, label, stem, frames):
    span = max(np.array(model["meta"]["bbox"]["max"]) - np.array(model["meta"]["bbox"]["min"]))
    tol = span / 65535 * 1.5
    anim = model["animations"][label]
    for f in frames:
        got, want = anim["frames"][f], _expected(stem, f)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g.shape == w.shape
            assert np.abs(g - w).max() <= tol, (label, f)


@needs_game
def test_roundtrip_matches_evaluate_anim_frame(goblin):
    n = goblin[1]["animations"]["walk"]["meta"]["frame_count"]
    _check(goblin[1], "walk", "Goblin-Walk", [0, 10, n - 1])


@needs_game
def test_frames_past_128(goblin):
    n = goblin[1]["animations"]["dance"]["meta"]["frame_count"]
    assert n > 150
    _check(goblin[1], "dance", "GOB-dance", [127, 128, 150, n - 1])


@needs_game
def test_animation_with_extra_parts_is_self_contained(cache, tmp_path):
    path = export_creature("bile_demon", MESHES, cache, tmp_path, ONLY)
    model = read_dk2v(path)
    drink = parse_kmf((MESHES / "Bile_Demon-Drink.kmf").read_bytes())
    want = sum(1 for s in drink.anim.sprites if s.triangles[0])
    assert want > 3
    assert len(model["animations"]["drink"]["parts"]) == want
    assert len(model["animations"]["walk"]["parts"]) == 3


@needs_game
def test_missing_texture_is_reported_not_fatal(cache, tmp_path):
    names = [n for n in cache.names if n != "GoblinFrontMM0"]
    path = export_creature("goblin", MESHES, FakeCache(names), tmp_path, ONLY)
    model = read_dk2v(path)
    assert "GoblinFront" in model["meta"]["missing_textures"]
    assert any(p["texture"] == "" for p in model["animations"]["walk"]["parts"])


@needs_game
def test_parts_without_lod0_triangles_are_skipped(goblin):
    for a in goblin[1]["animations"].values():
        assert all(p["index_count"] > 0 for p in a["parts"])


@needs_game
def test_index_lists_creatures_and_animations(cache, tmp_path):
    p = export_creature("goblin", MESHES, cache, tmp_path, ONLY)
    idx = json.loads(write_index(tmp_path, [p]).read_text())
    g = idx["creatures"][0]
    assert g["id"] == "goblin" and g["file"] == "goblin.dk2v" and g["label"] == "Goblin"
    assert {"walk", "dance"} <= {a["label"] for a in g["animations"]}
    assert "GoblinFrontMM0" in g["textures"]
