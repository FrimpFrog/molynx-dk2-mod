"""Build the viewer site with the red-moss Golem, its animations and its MegaGlest sounds. LICENCE: GPL-3.0.

Usage: python3 -m scripts.build_golem_site <site dir> <Meshes dir> <EngineTextures.dir> <EngineTextures.dat> <golem unit dir> <MegaGlest sounds dir> [--text]
"""
import sys
from pathlib import Path

from scripts.build_viewer import build
from scripts.dk2texture import TextureCache
from scripts.glest_export import bake_inferno
from scripts.golem_anims import build_all

if __name__ == "__main__":
    site, meshes, d, dat, unit, sounds, *rest = sys.argv[1:]
    cache = TextureCache(Path(d), Path(dat))
    lib = build_all(Path(meshes), cache, Path(unit))
    out = build(Path(site), Path(meshes), cache, None, text_data="--text" in rest,
                glest_units=[("golem", "Golem (MegaGlest, red moss)", unit,
                              {"baker": bake_inferno, "extra_anims": lib, "sound_dir": sounds})])
    print(f"{len(out['creatures'])} creatures, {out['total_bytes'] / 1e6:.1f} MB")
