from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from scripts.check_viewer_frames import run  # noqa: E402

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
SITE = S / "viewer/full"


@pytest.mark.skipif(not (SITE / "data/index.json").exists(), reason="viewer site not built")
def test_viewer_matches_reference_renderer(tmp_path):
    from scripts.dk2texture import TextureCache
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    for cid, label, iou, mirrored in run(SITE, S / "extracted/Meshes", cache, tmp_path):
        assert iou >= 0.75, (cid, label, iou)             # thin horns/scythe cost the Reaper a few points
        assert iou > mirrored + 0.2, (cid, label, iou, mirrored)   # orientation is right, not mirrored
