"""GIFs of the Necromancer's files, rendered from the written KMF files themselves (so what is shown
is what the game will load): one GIF per animation (3/4 front view, 15 fps), a slow turntable for
each static pose and for the model, and a contact sheet.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.dk2kmf import parse_kmf
from scripts.kmf_textured import fixed_view, render_textured
from scripts.necro_build import TextureOverlay

SIZE = 360
AZ = 150


def _label(im: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle([0, SIZE - 22, SIZE, SIZE], fill=(20, 20, 24))
    d.text((8, SIZE - 18), text, fill=(220, 220, 230))
    return im


def gif_of(path: Path, cache, out: Path, step: int = 2, title: str = "") -> Image.Image:
    k = parse_kmf(path.read_bytes())
    if k.anim is None:
        frames = []
        for i in range(24):
            az = (AZ + i * 15) % 360
            view = fixed_view([(k, None)], az)
            frames.append(_label(render_textured(k, cache, azimuth=az, size=SIZE, view=view), title))
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
        return frames[0]
    n = k.anim.frame_count
    view = fixed_view([(k, f) for f in range(0, n, 3)], AZ)
    idx = list(range(0, n, step))
    if k.anim.wrap and idx[-1] == n - 1:
        idx = idx[:-1]                       # the last frame of a loop repeats the first
    frames = [_label(render_textured(k, cache, azimuth=AZ, size=SIZE, frame=f, view=view), title) for f in idx]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * step / 30), loop=0, optimize=True)
    return frames[len(frames) // 2]


def turntable(path: Path, cache, out: Path, title: str) -> None:
    k = parse_kmf(path.read_bytes())
    frames = []
    for i in range(36):
        az = (180 + i * 10) % 360
        view = fixed_view([(k, 0)], 180)
        frames.append(_label(render_textured(k, cache, azimuth=az, size=480 if False else SIZE, frame=0, view=view), title))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)


def main(files_dir: Path, out_dir: Path, cache, names: list[str] | None = None) -> None:
    files_dir, out_dir = Path(files_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {p.stem: Image.open(p) for p in files_dir.glob("Necro_*.png")}
    tc = TextureOverlay(cache, extra)
    stills = []
    for p in sorted(files_dir.glob("Necro-*.kmf")):
        if names and p.stem not in names:
            continue
        title = p.stem.replace("Necro-", "")
        print(title, flush=True)
        stills.append(gif_of(p, tc, out_dir / f"{p.stem}.gif", title=title))
    if not names:
        turntable(files_dir / "Necro-Idle1.kmf", tc, out_dir / "Necromancer_turntable.gif", "Necromancer")
        cols = 8
        rows = (len(stills) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 180, rows * 180), (38, 38, 42))
        for i, im in enumerate(stills):
            sheet.paste(im.resize((180, 180), Image.LANCZOS), ((i % cols) * 180, (i // cols) * 180))
        sheet.save(out_dir / "Necromancer_all_animations.png")


if __name__ == "__main__":
    from scripts.dk2texture import TextureCache
    data = Path(sys.argv[3])
    main(Path(sys.argv[1]), Path(sys.argv[2]), TextureCache(data / "EngineTextures.dir", data / "EngineTextures.dat"),
         sys.argv[4].split(",") if len(sys.argv) > 4 else None)
