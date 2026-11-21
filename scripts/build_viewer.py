"""Build the complete web-viewer site folder: page, three.js, .dk2v data, original and HD textures.

LICENCE: GPL-3.0 (uses the OpenKeeper-derived texture decoder).
Usage: python3 -m scripts.build_viewer <site dir> <Meshes dir> <EngineTextures.dir> <EngineTextures.dat> [--text] [--custom=<folder>] [--glest=id:Label:<unit folder>] [creature ids...]
"""
import json
import shutil
import sys
from pathlib import Path

from scripts.viewer_creatures import CREATURES
from scripts.viewer_export import export_creature, pack_text, write_index
from scripts.viewer_textures import build_hd_pack, export_textures, texture_file, texture_names

VIEWER_SRC = Path(__file__).resolve().parent.parent / "viewer"
PAGE_FILES = ("index.html", "app.js", "dk2v.js")


def build(site_dir: Path, meshes_dir: Path, cache, creature_ids: list[str] | None = None,
          text_data: bool = False, custom_dir: Path | None = None,
          glest_units: list | None = None) -> dict:
    """text_data=True stores each creature as gzip+base64 .dk2v.txt (for hosts that serve only text types).
    custom_dir: folder of custom creature specs (*.json), added after the game's creatures.
    glest_units: [(id, label, unit folder)] MegaGlest units to add (see scripts/glest_export.py)."""
    site_dir = Path(site_dir)
    ids = creature_ids or list(CREATURES)
    site_dir.mkdir(parents=True, exist_ok=True)
    for name in PAGE_FILES:
        shutil.copy2(VIEWER_SRC / name, site_dir / name)
    shutil.copytree(VIEWER_SRC / "vendor", site_dir / "vendor", dirs_exist_ok=True)
    data = site_dir / "data"
    paths = [export_creature(cid, meshes_dir, cache, data) for cid in ids]
    custom_images, custom_ids = {}, []
    if custom_dir is not None and Path(custom_dir).exists():
        from scripts.custom_export import custom_specs, export_custom
        for spec in custom_specs(custom_dir):
            p, imgs, _ = export_custom(spec, meshes_dir, cache, data)
            paths.append(p); custom_images.update(imgs); custom_ids.append(p.stem)
    for uid, label, folder, *opt in glest_units or []:
        from scripts.glest_export import export_glest_unit
        p, imgs = export_glest_unit(Path(folder), uid, label, data, **(opt[0] if opt else {}))
        paths.append(p); custom_images.update(imgs)
    write_index(data, paths)
    needed_textures = texture_names(paths)
    sizes = {p.name: p.stat().st_size for p in paths}
    if text_data:
        index = json.loads((data / "index.json").read_text())
        for c in index["creatures"]:
            pack_text(data / c["file"])
            (data / c["file"]).unlink()
            c["file"] += ".txt"
            c["encoding"] = "gzip-base64"
        (data / "index.json").write_text(json.dumps(index, indent=1))
    export_textures(cache, needed_textures - set(custom_images), site_dir / "textures/original")
    for name, img in custom_images.items():
        img.save(site_dir / "textures/original" / texture_file(name))
    build_hd_pack(site_dir / "textures/original", site_dir / "textures/hd")
    total = sum(f.stat().st_size for f in site_dir.rglob("*") if f.is_file())
    return {"creatures": ids, "custom": custom_ids, "dk2v_sizes": sizes, "total_bytes": total}


if __name__ == "__main__":
    from scripts.dk2texture import TextureCache
    site, meshes, d, dat, *rest = sys.argv[1:]
    text = "--text" in rest
    custom = next((Path(r.split("=", 1)[1]) for r in rest if r.startswith("--custom=")), None)
    glest = [tuple(r.split("=", 1)[1].split(":", 2)) for r in rest if r.startswith("--glest=")]   # id:Label:folder
    ids = [r for r in rest if r != "--text" and not r.startswith(("--custom=", "--glest="))]
    out = build(Path(site), Path(meshes), TextureCache(Path(d), Path(dat)), ids or None, text_data=text,
                custom_dir=custom, glest_units=glest or None)
    print(f"{len(out['creatures'])} creatures, {out['total_bytes'] / 1e6:.1f} MB total")
    for n, b in sorted(out["dk2v_sizes"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {n}: {b / 1e6:.1f} MB")
