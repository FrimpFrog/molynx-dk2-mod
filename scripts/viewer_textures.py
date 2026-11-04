"""Texture folders for the web viewer: original PNGs and the 4x "Enhanced (non-AI)" HD pack.

LICENCE: GPL-3.0 (uses the OpenKeeper-derived texture decoder through TextureCache).
Drop AI-upscaled PNGs with the same file names into the hd folder to replace the pack.
"""
import re
from pathlib import Path

from PIL import Image, ImageFilter

from scripts.viewer_export import read_header


def texture_file(name: str) -> str:
    """URL-safe file name for a cache texture name: path separators become '__', anything else
    outside [A-Za-z0-9._-] becomes '_'. viewer/dk2v.js textureFile() must stay identical."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name.replace("\\", "__").replace("/", "__")) + ".png"


def texture_names(dk2v_paths: list[Path]) -> set[str]:
    return {pm["texture"] for p in dk2v_paths for a in read_header(p)[0]["animations"].values()
            for pm in a["parts"] if pm["texture"]}


def export_textures(cache, names: set[str], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    seen: dict[str, str] = {}
    for n in sorted(names):
        f = texture_file(n)
        if seen.setdefault(f, n) != n:
            raise ValueError(f"texture names {seen[f]!r} and {n!r} would share the file {f}")
        p = out_dir / f
        cache.get(n).save(p)
        written.append(p)
    return written


def build_hd_pack(original_dir: Path, hd_dir: Path, scale: int = 4) -> list[Path]:
    hd_dir = Path(hd_dir); hd_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sorted(Path(original_dir).glob("*.png")):
        img = Image.open(src)
        size = (img.width * scale, img.height * scale)
        rgb = img.convert("RGB").resize(size, Image.LANCZOS)
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))
        if "A" in img.getbands():
            rgb.putalpha(img.getchannel("A").resize(size, Image.LANCZOS))
        dst = hd_dir / src.name
        rgb.save(dst)
        written.append(dst)
    return written
