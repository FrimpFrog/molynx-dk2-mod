from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.viewer_export import export_creature
from scripts.viewer_textures import build_hd_pack, export_textures, texture_file, texture_names
from scripts.viewer_creatures import CREATURES

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
MESHES = S / "extracted/Meshes"
needs_game = pytest.mark.skipif(not MESHES.exists(), reason="DK2 files not present")


@pytest.fixture(scope="module")
def cache():
    from scripts.dk2texture import TextureCache
    return TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")


@needs_game
def test_original_pngs_written_for_every_needed_texture(cache, tmp_path):
    dk = export_creature("goblin", MESHES, cache, tmp_path / "d", {"goblin": CREATURES["goblin"]})
    names = texture_names([dk])
    assert {"GoblinFrontMM0", "GoblinBackMM0"} <= names
    written = export_textures(cache, names, tmp_path / "original")
    assert {p.name for p in written} == {texture_file(n) for n in names}
    assert all(Image.open(p).size == cache.size_of(n) for p, n in zip(sorted(written), sorted(names)))


def _make(dirpath: Path, name="a.png", size=(8, 4), mode="RGB"):
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (size[1], size[0], 4 if mode == "RGBA" else 3), dtype=np.uint8)
    Image.fromarray(arr, mode).save(dirpath / name)


def test_hd_is_exactly_4x_with_same_names(tmp_path):
    _make(tmp_path / "o", "a.png"); _make(tmp_path / "o", "b.png", (16, 8))
    out = build_hd_pack(tmp_path / "o", tmp_path / "hd")
    assert sorted(p.name for p in out) == ["a.png", "b.png"]
    assert Image.open(tmp_path / "hd/a.png").size == (32, 16)
    assert Image.open(tmp_path / "hd/b.png").size == (64, 32)


def test_hd_differs_from_plain_resize(tmp_path):
    _make(tmp_path / "o")
    build_hd_pack(tmp_path / "o", tmp_path / "hd")
    plain = np.asarray(Image.open(tmp_path / "o/a.png").resize((32, 16), Image.LANCZOS), float)
    hd = np.asarray(Image.open(tmp_path / "hd/a.png").convert("RGB"), float)
    assert np.abs(plain - hd).mean() > 0


def test_alpha_is_preserved(tmp_path):
    _make(tmp_path / "o", "t.png", mode="RGBA")
    build_hd_pack(tmp_path / "o", tmp_path / "hd")
    assert Image.open(tmp_path / "hd/t.png").mode == "RGBA"


def test_hd_pack_is_idempotent(tmp_path):
    _make(tmp_path / "o")
    build_hd_pack(tmp_path / "o", tmp_path / "hd")
    first = (tmp_path / "hd/a.png").read_bytes()
    build_hd_pack(tmp_path / "o", tmp_path / "hd")
    assert (tmp_path / "hd/a.png").read_bytes() == first


def test_texture_file_sanitises_gui_names():
    assert texture_file("GUI\\Creatures\\Evil\\reaperMM0") == "GUI__Creatures__Evil__reaperMM0.png"


def test_texture_file_is_url_safe():
    import re
    for name in ["#ADD#King-CleaveMM0", "&mask&wing1MM0", "Creature Bed_Goblin INMM0", "GUI\\a b\\c#d"]:
        f = texture_file(name)
        assert re.fullmatch(r"[A-Za-z0-9._-]+", f), f
    assert texture_file("GoblinFrontMM0") == "GoblinFrontMM0.png"   # ordinary names unchanged


def test_distinct_names_never_share_a_file(tmp_path):
    import pytest
    from scripts.viewer_textures import export_textures

    class Fake:
        def get(self, n):
            return Image.new("RGB", (2, 2))

    with pytest.raises(ValueError):
        export_textures(Fake(), {"a b", "a_b"}, tmp_path)
