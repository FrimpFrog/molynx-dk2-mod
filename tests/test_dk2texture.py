import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.dk2texture import TextureCache

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
DIR, DAT = S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat"
PNGS = S / "extracted/EngineTextures"
needs_game = pytest.mark.skipif(not DIR.exists(), reason="DK2 texture cache not present")


@pytest.fixture(scope="module")
def cache():
    return TextureCache(DIR, DAT)


def _mean_abs_diff(decoded: Image.Image, reference: Image.Image) -> float:
    a = np.asarray(decoded.convert("RGB"), dtype=float)
    b = np.asarray(reference.convert("RGB"), dtype=float)
    assert a.shape == b.shape
    return float(np.abs(a - b).mean())


@needs_game
def test_cache_lists_every_texture(cache):
    assert len(cache.names) == 5767
    assert "ReaperLegsMM0" in cache.names


@needs_game
def test_unknown_texture_raises(cache):
    with pytest.raises(KeyError):
        cache.get("NoSuchTextureMM0")


@needs_game
def test_decoded_portrait_matches_reference_png(cache):
    img = cache.get("GUI\\Creatures\\Evil\\reaperMM0")
    assert img.size == (64, 64)
    # codec is lossy, so allow a small difference from the PNG stored in the WAD
    assert _mean_abs_diff(img, Image.open(PNGS / "reaper.png")) < 10


@needs_game
def test_decoded_tall_portrait_matches_reference_png(cache):
    img = cache.get("GUI\\Creatures\\Evil\\reaper-bgMM0")
    assert img.size == (64, 128)
    assert _mean_abs_diff(img, Image.open(PNGS / "reaper-bg.png")) < 10


@needs_game
def test_creature_skins_decode_with_expected_sizes(cache):
    expected = {"ReaperLegsMM0": (128, 128), "ReaperBackMM0": (128, 128),
                "ReaperTorsoMM0": (128, 128), "ReaperFaceMM0": (64, 64),
                "ReaperSchytheMM0": (128, 64)}
    for name, size in expected.items():
        img = cache.get(name)
        assert img.size == size, name
        assert np.asarray(img).std() > 10, f"{name} looks blank"


@needs_game
def test_random_sample_of_all_sizes_decodes(cache):
    rng = random.Random(7)
    by_size = {}
    for name in cache.names:
        by_size.setdefault(cache.size_of(name), []).append(name)
    for size, names in by_size.items():
        for name in rng.sample(names, min(2, len(names))):
            img = cache.get(name)
            assert img.size == size, name
