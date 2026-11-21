"""Review renders of Krefft from his exported files (what the game will load), with the Dragon's software
renderer (scripts.kmf_render.render: orthographic, z-buffered, back faces culled the way the game culls
them, so a wrongly wound triangle shows as a hole; a floor disk at the files' z = 0).

  python3 -m scripts.krefft_render <export dir> <out dir> [<the game's Meshes dir>] [--draft]
writes
  krefft_turntable(_draft).gif   the PoseFrame turning round, dipping to DK2's high camera and back
  krefft_views.png               front, 3/4, side, back, 3/4 back and the high camera
  krefft_head.png                the head close up, 3/4 from each side (the stripe, the eyes, the gold
                                 tooth, the chain)
  krefft_shell_top.png           the shell from straight above
  krefft_size.png                beside the game's own Salamander and Bile Demon, one scale for all
  krefft_skins.png               the three skins, 4x, beside the Salamander's and the Bile Demon's

LICENCE: GPL-3.0.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.dk2kmf import mesh_points, parse_kmf, evaluate_anim_frame
from scripts.kmf_render import Textures, render

BG = (26, 25, 30)
INK = (235, 220, 140)
CENTRE = (0, .13, 0)
SPAN = 1.05
GAME_TEXTURES = "/mnt/user-data/uploads/Dungeon Keeper 2/Data/EngineTextures"
POSE = "Krefft-PoseFrame"
# the gold and the eyes: few pixels, so an adaptive palette would drop them
KEEP = [(255, 236, 140), (214, 164, 48), (120, 76, 14), (178, 192, 168), (214, 200, 140), (112, 168, 36),
        (168, 214, 70), (170, 140, 132)]


def save_gif(frames, out, duration):
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
    skins = {p.stem: Image.open(p) for p in Path(export_dir).glob("Krefft_*.png")}
    return Textures(skins, cache)


def _load(d, stem):
    return parse_kmf((Path(d) / f"{stem}.kmf").read_bytes())


def turntable(export_dir, out_gif, tex, S=440, label="Krefft-PoseFrame"):
    pose = _load(export_dir, POSE)
    frames = []
    for i in range(36):
        az = i * 10
        el = 14 + 31 * (0.5 - 0.5 * np.cos(2 * np.pi * i / 36))
        im = render(pose, tex, None, az, el, (S, S), centre=CENTRE, span=SPAN)
        ImageDraw.Draw(im).text((8, 6), label, fill=INK)
        frames.append(im)
    save_gif(frames, out_gif, 100)


def views(export_dir, out_png, tex, S=360):
    pose = _load(export_dir, POSE)
    tiles = [(0, 12, "front"), (35, 15, "3/4 front"), (90, 8, "side"), (180, 12, "back"),
             (215, 25, "3/4 back"), (30, 50, "DK2's high camera")]
    out = Image.new("RGB", (S * 3, (S + 24) * 2), BG)
    for i, (az, el, lab) in enumerate(tiles):
        x, y = (i % 3) * S, (i // 3) * (S + 24)
        out.paste(render(pose, tex, None, az, el, (S, S), centre=CENTRE, span=SPAN), (x, y + 24))
        ImageDraw.Draw(out).text((x + 8, y + 6), lab, fill=INK)
    out.save(out_png)


def head(export_dir, out_png, tex, S=420):
    """3/4 close-ups of the head from his left (gold tooth side) and his right (the scarred eye)."""
    pose = _load(export_dir, POSE)
    k = np.concatenate(list(mesh_points(pose)))
    P = np.stack([-k[:, 0], -k[:, 2], k[:, 1]], 1)                  # the renderer's space
    sel = P[P[:, 2] > P[:, 2].max() - .12]                         # the head (in front of the chain)
    hc = np.array([0, (sel[:, 1].min() + sel[:, 1].max()) / 2, P[:, 2].max() - .085])
    tiles = [(-40, 14, "3/4 from his left: the gold tooth, the stripe"),
             (40, 14, "3/4 from his right: the scarred eye"),
             (0, 8, "front: the smug mouth"), (-90, 4, "his left side")]
    G = 16                                                     # a gap between the tiles
    out = Image.new("RGB", ((S + G) * len(tiles), S + 24), (60, 58, 66))
    for i, (az, el, lab) in enumerate(tiles):
        from scripts.dragon_render import view
        out.paste(render(pose, tex, None, az, el, (S, S), centre=tuple(view(az, el) @ hc), span=.3, floor=False),
                  (i * (S + G), 24))
        ImageDraw.Draw(out).text((i * (S + G) + 8, 6), lab, fill=INK)
    out.save(out_png)


def shell_top(export_dir, out_png, tex, S=520):
    pose = _load(export_dir, POSE)
    im = render(pose, tex, None, 180, 89.9, (S, S), centre=(0, -.02, 0), span=1.0)
    ImageDraw.Draw(im).text((8, 6), "shell from above (head up the image)", fill=INK)
    im.save(out_png)


def size(export_dir, meshes_dir, out_png, tex, S=420):
    who = [(Path(meshes_dir), "Sal-PoseFrame", "DK2 Salamander"), (Path(export_dir), POSE, "Krefft"),
           (Path(meshes_dir), "BileDemon-Idle1", "DK2 Bile Demon")]
    span, centre = 1.15, (0, .3, 0)
    out = Image.new("RGB", (S * 3, S * 2 + 24), BG)
    d = ImageDraw.Draw(out)
    for i, (folder, stem, lab) in enumerate(who):
        k = _load(folder, stem)
        pts = np.concatenate(list(mesh_points(k))) if k.kind == "mesh" else np.concatenate(evaluate_anim_frame(k.anim, 0))
        ext = pts.max(0) - pts.min(0)
        for r, (az, el) in enumerate(((30, 14), (90, 6))):
            out.paste(render(k, tex, None if k.kind == "mesh" else 0, az, el, (S, S), centre=centre, span=span),
                      (i * S, 24 + r * S))
        d.text((i * S + 8, 6), f"{lab}: {ext[2]:.2f} tall, {ext[1]:.2f} long, {ext[0]:.2f} wide", fill=INK)
    px = S / span
    for r in range(2):
        floor = 24 + r * S + S / 2 + centre[1] * px
        for k in range(7):
            y = floor - k * 0.1 * px
            d.line([(2, y), (10 if k % 5 else 18, y)], fill=INK)
    out.save(out_png)


def skins(export_dir, out_png):
    data = Path("/home/claude/video2/cgi/web/data")
    ims = [(Image.open(Path(export_dir) / f"{n}.png"), n) for n in ("Krefft_Body", "Krefft_Head", "Krefft_Shell")]
    ims += [(Image.open(data / "tex_Salamander.png"), "vanilla Salamander"),
            (Image.open(data / "tex_BDemonFrnt.png"), "vanilla Bile Demon")]
    out = Image.new("RGB", (512 * 3, 512 * 2 + 48), BG)
    d = ImageDraw.Draw(out)
    for i, (im, lab) in enumerate(ims):
        x, y = (i % 3) * 512, (i // 3) * (512 + 24)
        out.paste(im.convert("RGB").resize((512, 512), Image.NEAREST), (x, y + 24))
        a = np.asarray(im.convert("L"), float)
        d.text((x + 8, y + 6), f"{lab}  (detail {np.abs(np.diff(a, axis=1)).mean():.1f})", fill=INK)
    out.save(out_png)


def compare(dirs, out_png, S=360):
    """Rows of the same angles (3/4 front, side, DK2's high camera, head close) for each (dir, label)."""
    tiles = [(35, 15), (90, 8), (30, 50)]
    out = Image.new("RGB", (S * len(tiles), (S + 24) * len(dirs)), BG)
    d = ImageDraw.Draw(out)
    for r, (folder, lab, stem) in enumerate(dirs):
        tex = textures(folder)
        k = _load(folder, stem)
        pts = np.concatenate(list(mesh_points(k)))
        h = pts[:, 2].max() - pts[:, 2].min()
        cy = h / 2 if h > .45 else CENTRE[1]
        span = max(SPAN, h * 1.35)
        for i, (az, el) in enumerate(tiles):
            c = (0, cy * (0.6 if el > 40 else 1.0), 0)
            out.paste(render(k, tex, None, az, el, (S, S), centre=c, span=span), (i * S, r * (S + 24) + 24))
        d.text((8, r * (S + 24) + 6), lab, fill=INK)
    out.save(out_png)


def main(export_dir, out_dir, meshes_dir, draft=False, suffix=""):
    tex = textures(export_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if draft:
        turntable(export_dir, out_dir / "krefft_turntable_draft.gif", tex, label="Krefft (draft)")
        return
    views(export_dir, out_dir / f"krefft_views{suffix}.png", tex)
    head(export_dir, out_dir / f"krefft_head{suffix}.png", tex)
    shell_top(export_dir, out_dir / f"krefft_shell_top{suffix}.png", tex)
    size(export_dir, meshes_dir, out_dir / f"krefft_size{suffix}.png", tex)
    skins(export_dir, out_dir / f"krefft_skins{suffix}.png")
    turntable(export_dir, out_dir / f"krefft_turntable{suffix}.gif", tex)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    suffix = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--suffix=")), "")
    main(args[0], args[1], args[2] if len(args) > 2 else "/home/claude/dk2data/Meshes", "--draft" in sys.argv, suffix)
