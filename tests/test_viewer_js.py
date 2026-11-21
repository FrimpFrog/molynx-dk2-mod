import json
import subprocess
from pathlib import Path

import pytest

from scripts.viewer_creatures import CREATURES
from scripts.viewer_export import export_creature, read_dk2v

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
MESHES = S / "extracted/Meshes"


@pytest.mark.skipif(not MESHES.exists(), reason="DK2 files not present")
def test_js_parser_matches_python_reader(tmp_path):
    from scripts.dk2texture import TextureCache
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    path = export_creature("goblin", MESHES, cache, tmp_path, {"goblin": CREATURES["goblin"]})
    walk = read_dk2v(path)["animations"]["walk"]
    want = {"part_count": len(walk["parts"]), "vertex_counts": [p["vertex_count"] for p in walk["parts"]],
            "frame": 10, "first_positions": walk["frames"][10][0][:6].reshape(-1).tolist()}
    (tmp_path / "want.json").write_text(json.dumps(want))
    r = subprocess.run(["node", "tests/viewer/test_dk2v.mjs", str(path), str(tmp_path / "want.json")],
                       capture_output=True, text=True, cwd="/home/claude")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not MESHES.exists(), reason="DK2 files not present")
def test_js_decodes_gzip_base64_text_files(tmp_path):
    from scripts.dk2texture import TextureCache
    from scripts.viewer_export import pack_text
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    path = export_creature("goblin", MESHES, cache, tmp_path, {"goblin": CREATURES["goblin"]})
    walk = read_dk2v(path)["animations"]["walk"]
    want = {"part_count": len(walk["parts"]), "frame": 10,
            "first_positions": walk["frames"][10][0][:6].reshape(-1).tolist()}
    (tmp_path / "want.json").write_text(json.dumps(want))
    txt = pack_text(path)
    assert txt.name == "goblin.dk2v.txt" and txt.stat().st_size < path.stat().st_size * 1.34
    r = subprocess.run(["node", "tests/viewer/test_decode.mjs", str(txt), str(tmp_path / "want.json")],
                       capture_output=True, text=True, cwd="/home/claude")
    assert r.returncode == 0, r.stdout + r.stderr
