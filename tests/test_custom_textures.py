import numpy as np
import pytest
from PIL import Image

from scripts import custom_textures as ct


class FakeCache:
    def get(self, name):
        rng = np.random.default_rng(0)
        return Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))


def _built():
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]], np.float32)
    idx = np.array([0, 1, 2, 1, 3, 2], np.uint16)
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    return {"geometry": {"custom/t/a": {"uv": uv, "indices": idx, "points": pts}},
            "sources": {"custom/t/a": ("base", "Src")}}


SPEC = {"id": "t", "textures": {"a": {"recolour": "body", "overlays": [
    {"pattern": "plates", "region": {"y": [0.5, 2]}, "strength": 1.0}], "eyes": [[10, 10]]}}}


def test_uv_mask_covers_triangle_area_only():
    uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    m = ct.uv_mask((16, 16), uv, np.array([0, 1, 2]))
    assert m[1, 1] == 1 and m[14, 14] == 0 and 0.4 < m.mean() < 0.6


def test_generated_texture_keeps_source_size_and_guide_matches():
    out = ct.generate_textures(SPEC, _built(), FakeCache())
    img, guide = out["custom/t/a"]
    assert img.size == (32, 32) and guide.size == (128, 128)


def test_overlay_changes_only_selected_region():
    plain = {**SPEC, "textures": {"a": {"recolour": "body"}}}
    a = np.asarray(ct.generate_textures(plain, _built(), FakeCache())["custom/t/a"][0], int)
    b = np.asarray(ct.generate_textures({**SPEC, "textures": {"a": SPEC["textures"]["a"] | {"eyes": []}}},
                                        _built(), FakeCache())["custom/t/a"][0], int)
    diff = np.abs(a - b).sum(-1) > 0
    assert diff.any()
    g = _built()["geometry"]["custom/t/a"]
    sel = ct.tri_in_region(g["points"], g["indices"], {"y": [0.5, 2]})
    assert sel.tolist() == [False, True]
    near = ct._blur(ct.uv_mask((32, 32), g["uv"], g["indices"], sel), 2.0) > 0.001
    assert not (diff & ~near).any()          # nothing changed outside the selected triangle's UV area
    assert diff.sum() < 32 * 32 * 0.7


def test_hand_painted_texture_replaces_generated_and_size_is_checked(tmp_path):
    (tmp_path / "textures").mkdir()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(tmp_path / "textures/a.png")
    out = ct.generate_textures(SPEC, _built(), FakeCache(), custom_dir=tmp_path)
    assert out["custom/t/a"][0].getpixel((5, 5)) == (1, 2, 3)
    Image.new("RGB", (16, 16)).save(tmp_path / "textures/a.png")
    with pytest.raises(ValueError):
        ct.generate_textures(SPEC, _built(), FakeCache(), custom_dir=tmp_path)
