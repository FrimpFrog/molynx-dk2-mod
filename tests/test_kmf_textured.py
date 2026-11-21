from pathlib import Path

import numpy as np
import pytest

from scripts.dk2kmf import parse_kmf
from scripts.dk2texture import TextureCache
from scripts.kmf_textured import material_texture_name, render_textured, write_textured_obj

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
MESHES = S / "extracted/Meshes"
needs_game = pytest.mark.skipif(not MESHES.exists(), reason="DK2 files not present")


@pytest.fixture(scope="module")
def cache():
    return TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")


def _model(name):
    return parse_kmf((MESHES / name).read_bytes())


@needs_game
@pytest.mark.parametrize("name", ["Rea3-poseframe.kmf", "Gob-PoseFrame.kmf", "BD-PoseFrame.kmf"])
def test_every_material_resolves_to_a_texture(cache, name):
    model = _model(name)
    for sprite in model.mesh.sprites:
        if not sprite.triangles[0]:
            continue
        tex = material_texture_name(cache, model.materials[sprite.material])
        assert tex in cache.names, (name, model.materials[sprite.material].name)


@needs_game
def test_render_covers_part_of_the_image_and_is_coloured(cache):
    img = render_textured(_model("Rea3-poseframe.kmf"), cache, azimuth=180, size=256)
    arr = np.asarray(img.convert("RGB"), dtype=float)
    bg = arr[0, 0]
    covered = (np.abs(arr - bg).sum(axis=2) > 12).mean()
    assert 0.1 < covered < 0.8
    assert arr[..., 0].mean() > arr[..., 2].mean()  # reaper is red, not blue


@needs_game
def test_textured_obj_writes_mtl_and_png(cache, tmp_path):
    model = _model("Rea3-poseframe.kmf")
    obj = write_textured_obj(model, cache, tmp_path, "reaper")
    text = obj.read_text()
    assert "mtllib reaper.mtl" in text
    parts = [s for s in model.mesh.sprites if s.triangles[0]]
    assert text.count("usemtl ") == len(parts)
    mtl = (tmp_path / "reaper.mtl").read_text()
    pngs = [line.split()[1] for line in mtl.splitlines() if line.startswith("map_Kd")]
    assert pngs and all((tmp_path / p).exists() for p in pngs)
