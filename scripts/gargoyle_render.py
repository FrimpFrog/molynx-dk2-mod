"""Review renders of the Gargoyle from his exported files (what the game will load), with the Dragon's
software renderer (scripts.dragon_render via scripts.kmf_render.render: orthographic, z-buffered, back
faces culled the way the game culls them, so a wrongly wound triangle shows as a hole; a floor disk at
the files' z = 0).

  python3 -m scripts.gargoyle_render <export dir> <out dir> [<the game's Meshes dir>]
writes
  Gargoyle-turntable.gif   the PoseFrame turning round (and tilting down to DK2's high camera)
  Gargoyle-views.png       front, 3/4, side, back and the high camera
  Gargoyle-size.png        beside the game's own Imp and Goblin, at the same scale
  Gargoyle-idle.gif        the standing idle

LICENCE: GPL-3.0.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.dk2kmf import parse_kmf
from scripts.kmf_render import Textures, render

BG = (26, 25, 30)
INK = (235, 220, 140)
CENTRE = (0, .25, 0)
SPAN = .62
GAME_TEXTURES = "/mnt/user-data/uploads/Dungeon Keeper 2/Data/EngineTextures"


# glowing eyes and the mouth: few pixels, so an adaptive palette would drop them (the eyes went white)
KEEP = [(255, 240, 160), (255, 205, 90), (255, 170, 36), (230, 130, 25), (200, 96, 12), (140, 60, 10),
        (90, 30, 22), (60, 24, 19), (120, 40, 30)]


def save_gif(frames, out, duration):
    """All frames on one palette: the frames' own colours plus KEEP."""
    w, h = frames[0].size
    sheet = Image.new("RGB", (w * min(len(frames), 8), h + 40))
    for i, f in enumerate(frames[::max(1, len(frames) // 8)][:8]):
        sheet.paste(f.convert("RGB"), (i * w, 0))
    band = sheet.width // len(KEEP)
    for i, c in enumerate(KEEP):
        sheet.paste(c, (i * band, h, (i + 1) * band, h + 40))
    pal = sheet.quantize(255, method=Image.Quantize.MEDIANCUT)
    q = [f.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=duration, loop=0)


def textures(export_dir):
    from scripts.dk2texture import TextureCache
    cache = TextureCache(Path(GAME_TEXTURES + ".dir"), Path(GAME_TEXTURES + ".dat"))
    skins = {p.stem: Image.open(p) for p in Path(export_dir).glob("Gargoyle_*.png")}
    return Textures(skins, cache)


def _load(d, stem):
    return parse_kmf((Path(d) / f"{stem}.kmf").read_bytes())


def wings_open_kmf(meshes_dir):
    """The crouch with the wings spread (the bind pose) through the same export plan, in memory only: a
    preview of the wings opening for hops and attacks (stage 2 builds the clips that use it)."""
    from scripts.dragon_export import pose_kmf
    from scripts.gargoyle_build import build
    from scripts.gargoyle_model import gargoyle_skinner
    from scripts.gargoyle_pose import GargoylePose
    built = build(meshes_dir)
    frame = built.plan.frame(gargoyle_skinner(built.parts)(*GargoylePose().world()))
    return pose_kmf(built.plan, "WingsOpen", frame)


def views(export_dir, out_png, tex, S=380, meshes_dir=None):
    pose = _load(export_dir, "Gargoyle-PoseFrame")
    tiles = [(0, 12, "front"), (35, 15, "3/4 front"), (90, 10, "side"), (180, 12, "back"),
             (215, 25, "3/4 back"), (30, 50, "DK2's high camera")]
    extra = wings_open_kmf(meshes_dir) if meshes_dir else None
    out = Image.new("RGB", (S * (len(tiles) + (extra is not None)), S + 24), BG)
    for i, (az, el, label) in enumerate(tiles):
        high = el > 40
        out.paste(render(pose, tex, None, az, el, (S, S), centre=(0, .12, 0) if high else CENTRE,
                         span=SPAN * 1.15 if high else SPAN), (i * S, 24))
        ImageDraw.Draw(out).text((i * S + 8, 6), label, fill=INK)
    if extra is not None:
        i = len(tiles)
        out.paste(render(extra, tex, None, 160, 25, (S, S), centre=(0, .2, 0), span=SPAN * 1.45), (i * S, 24))
        ImageDraw.Draw(out).text((i * S + 8, 6), "wings open (preview, from behind)", fill=INK)
    out.save(out_png)


def turntable(export_dir, out_gif, tex, S=420):
    pose = _load(export_dir, "Gargoyle-PoseFrame")
    frames = []
    for i in range(36):
        az = i * 10
        el = 12 + 33 * (0.5 - 0.5 * np.cos(2 * np.pi * i / 36))       # down to the game's camera and back
        im = render(pose, tex, None, az, el, (S, S), centre=CENTRE, span=SPAN)
        ImageDraw.Draw(im).text((8, 6), "Gargoyle-PoseFrame", fill=INK)
        frames.append(im)
    save_gif(frames, out_gif, 100)


def idle(export_dir, out_gif, tex, S=380):
    k = _load(export_dir, "Gargoyle-Idle1")
    n = k.anim.frame_count
    frames = []
    for f in range(0, n - 1, 2):
        im = render(k, tex, f, 35, 20, (S, S), centre=CENTRE, span=SPAN)
        ImageDraw.Draw(im).text((8, 6), f"Gargoyle-Idle1  {f}/{n - 1}", fill=INK)
        frames.append(im)
    save_gif(frames, out_gif, 66)


def size(export_dir, meshes_dir, out_png, tex, S=420):
    """The Imp, the Gargoyle and the Goblin side by side (3/4 front and side), one scale for all."""
    who = [(Path(meshes_dir), "Imp-PoseFrame", "DK2 Imp"), (Path(export_dir), "Gargoyle-PoseFrame", "Gargoyle"),
           (Path(meshes_dir), "Gob-PoseFrame", "DK2 Goblin")]
    span, centre = 0.8, (0, .33, 0)
    out = Image.new("RGB", (S * 3, S * 2 + 24), BG)
    d = ImageDraw.Draw(out)
    for i, (folder, stem, label) in enumerate(who):
        k = _load(folder, stem)
        from scripts.dk2kmf import mesh_points
        h = np.concatenate(list(mesh_points(k)))[:, 2]
        h = h.max() - h.min()
        for r, (az, el) in enumerate(((30, 12), (90, 6))):
            out.paste(render(k, tex, None, az, el, (S, S), centre=centre, span=span), (i * S, 24 + r * S))
        d.text((i * S + 8, 6), f"{label}  ({h:.2f} units tall)", fill=INK)
    # a scale bar every 0.1 KMF units at the left of each row
    px = S / span
    for r in range(2):
        floor = 24 + r * S + S / 2 + centre[1] * px
        for k in range(8):
            y = floor - k * 0.1 * px
            d.line([(2, y), (10 if k % 5 else 18, y)], fill=INK)
    out.save(out_png)


def face(export_dir, old_dir, out_png, S=330):
    """The face close up (front, 3/4, side, DK2's high camera), the old one above the new one."""
    tiles = [(0, 8, (0, .325, 0), .27), (32, 12, (0, .325, 0), .27), (85, 8, (0, .325, 0), .27),
             (25, 50, (0, .2, 0), .5)]
    rows = [(old_dir, "before"), (export_dir, "after: angrier, scarier")] if old_dir else [(export_dir, "")]
    out = Image.new("RGB", (S * len(tiles), (S + 22) * len(rows)), BG)
    d = ImageDraw.Draw(out)
    for r, (folder, label) in enumerate(rows):
        tex = textures(folder)
        pose = _load(folder, "Gargoyle-PoseFrame")
        for i, (az, el, c, span) in enumerate(tiles):
            out.paste(render(pose, tex, None, az, el, (S, S), centre=c, span=span), (i * S, r * (S + 22) + 22))
        d.text((8, r * (S + 22) + 5), label, fill=INK)
    out.save(out_png)


CLIP_AZ = {"Walk": 70, "Run": 70, "Melee": -40, "RunStart": 40, "RunEnd": 40, "Walkback": 70,
           "WalkbackStart": 40, "WalkbackEnd": 40, "Drink": -40, "Melee2": -40, "Fallback": 60, "GetUp": 60}


def clip_gif(export_dir, stem, out_gif, tex, S=400, step=1):
    """One clip from its game file at the game's 30 fps, from a 3/4 camera (more from the side for the
    gaits) framed once round the whole clip (so nothing jitters)."""
    from scripts.dragon_render import view
    from scripts.dk2kmf import evaluate_anim_frame
    k = _load(export_dir, f"Gargoyle-{stem}")
    n = k.anim.frame_count
    az, el = CLIP_AZ.get(stem, 35), 20
    R = view(az, el)
    pts = np.concatenate([np.concatenate(evaluate_anim_frame(k.anim, f)) for f in range(0, n, 2)])
    q = np.stack([-pts[:, 0], -pts[:, 2], pts[:, 1]], 1) @ R.T
    q = np.vstack([q, np.array([[0, 0, 0]]) @ R.T])
    lo, hi = q.min(0), q.max(0)
    span = float(max(hi[:2] - lo[:2])) * 1.3
    c = (lo + hi) / 2
    last = n - 1 if k.anim.wrap else n                  # a loop's last frame repeats its first
    frames = []
    for f in range(0, last, step):
        im = render(k, tex, f, az, el, (S, S), centre=(c[0], c[1], 0), span=span)   # (view space)
        ImageDraw.Draw(im).text((8, 6), f"Gargoyle-{stem}  {f}/{n - 1}", fill=INK)
        frames.append(im)
    save_gif(frames, out_gif, int(round(1000 * step / 30)))
    return frames


def main(export_dir, out_dir, meshes_dir):
    tex = textures(export_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    views(export_dir, out_dir / "Gargoyle-views.png", tex, meshes_dir=meshes_dir)
    size(export_dir, meshes_dir, out_dir / "Gargoyle-size.png", tex)
    turntable(export_dir, out_dir / "Gargoyle-turntable.gif", tex)
    gifs = out_dir / "gifs"
    gifs.mkdir(exist_ok=True)
    for p in sorted(Path(export_dir).glob("Gargoyle-*.kmf")):
        stem = p.stem.replace("Gargoyle-", "")
        if parse_kmf(p.read_bytes()).kind == "anim":
            clip_gif(export_dir, stem, gifs / f"Gargoyle-{stem}.gif", tex)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "/home/claude/dk2data/Meshes")
