"""Build the viewer site with every DK2 creature plus the bundled MegaGlest units (Golem, Battlemage). LICENCE: GPL-3.0.

Usage: python3 -m scripts.build_full_site <site dir> <Meshes dir> <EngineTextures.dir> <EngineTextures.dat> [--text]
"""
import sys
from pathlib import Path

from scripts.build_viewer import build
from scripts.dk2texture import TextureCache
from scripts.run_viewer import battlemage_units, behemoth_units, golem_units

if __name__ == "__main__":
    site, meshes, d, dat, *rest = sys.argv[1:]
    cache = TextureCache(Path(d), Path(dat))
    units = golem_units(Path(meshes), cache) + battlemage_units(Path(meshes), cache) + behemoth_units(Path(meshes), cache)
    out = build(Path(site), Path(meshes), cache, None, text_data="--text" in rest, glest_units=units,
                custom_dir=Path("/nonexistent"))
    print(f"{len(out['creatures'])} creatures, {out['total_bytes'] / 1e6:.1f} MB")
