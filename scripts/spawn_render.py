"""Review renders of the Demon Spawn from its exported files (the Dragon's software renderer,
scripts.dragon_render.Canvas: orthographic, z-buffered, back faces culled the way the game culls them, so a
wrongly wound triangle shows as a hole). A floor square is drawn at the files' z = 0, so anything sunk into
the floor is cut off by it and anything floating shows a gap.

  python3 -m scripts.spawn_render <export dir> <out dir>
writes DemonSpawn_preview.png (a 5-angle turntable of stand frame 0 over the DK1 sprite's five Stand views)
and DemonSpawn_animations.png (key frames of the main clips). Needs KEEPERFX_SRC and the DK1 DATA folder
(DK1_DATA, default /home/claude/dk1data/DATA) for the sprites, and the game's texture cache for the props.

LICENCE: GPL-3.0.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf
from scripts.dragon_render import Canvas, view

BG = (26, 25, 30)
FLOOR = (52, 50, 58)


def kmf_parts(model, frame=None):
    """-> [(points (model space: Y up, x to the creature's left), triangles, uv, texture name)]."""
    if model.kind == "mesh":
        sprites, raw = model.mesh.sprites, mesh_points(model)
    else:
        sprites, raw = model.anim.sprites, evaluate_anim_frame(model.anim, frame or 0)
    out = []
    for sp, p in zip(sprites, raw):
        if not sp.triangles[0]:
            continue
        P = np.stack([-p[:, 0], -p[:, 2], p[:, 1]], 1)          # undo the export's x mirror
        uv = np.array([(v.uv[0] / 32768.0, v.uv[1] / 32768.0) for v in sp.vertices])
        out.append((P, np.array(sp.triangles[0]), uv, model.materials[sp.material].textures[0]))
    return out


class Textures:
    """Skin PNGs by name, then the game's cache (props)."""

    def __init__(self, skins: dict, cache=None):
        self.skins = {k: np.asarray(v.convert("RGB"), np.float32) for k, v in skins.items()}
        self.cache = cache
        self.memo = {}

    def __call__(self, name):
        if name in self.skins:
            return self.skins[name]
        if name not in self.memo:
            img = None
            if self.cache is not None:
                low = {n.lower(): n for n in self.cache.names}
                key = low.get(f"{name}MM0".lower())
                if key:
                    img = np.asarray(self.cache.get(key).convert("RGB"), np.float32)
            self.memo[name] = img if img is not None else np.full((4, 4, 3), 160, np.float32)
        return self.memo[name]


def render(model, tex, frame=None, az=45, el=18, size=(320, 280), centre=(0, .12, 0), span=0.8, floor=True,
           label=None):
    c = Canvas(size, centre=centre, span=span, bg=BG)
    R = view(az, el)
    if floor:                     # a floor disk at z = 0
        n, r = 32, 0.36
        P = np.array([(0, 0, 0)] + [(r * np.cos(a), 0, r * np.sin(a))
                                    for a in np.linspace(0, 2 * np.pi, n, endpoint=False)], float)
        T = np.array([(0, 1 + (k + 1) % n, 1 + k) for k in range(n)])
        c.draw(P, T, R, flat_colours=[FLOOR] * n, ambient=1.0, cull=False)
    for P, T, uv, name in kmf_parts(model, frame):
        c.draw(P, T, R, uv=uv, tex=tex(name))
    img = c.image()
    if label:
        ImageDraw.Draw(img).text((6, 4), label, fill=(235, 220, 140))
    return img


def _load(d, stem):
    return parse_kmf((Path(d) / f"DemonSpawn-{stem}.kmf").read_bytes())


def preview(export_dir, out_png, tex):
    """The turntable (front, front-diagonal, side, back-diagonal, back: the DK1 sprite views' order)
    over the DK1 Stand views."""
    from scripts.dk1sprites import Sprites
    stand = _load(export_dir, "stand")
    W, H = 300, 260
    tiles = [render(stand, tex, 0, az, 15, (W, H), centre=(0, .13, 0), span=.78) for az in (0, 45, 90, 135, 180)]
    sp = Sprites(Path(os.environ.get("DK1_DATA", "/home/claude/dk1data/DATA")))
    dk = sp.views(666, 0)
    out = Image.new("RGB", (W * 5, H * 2 + 40), BG)
    d = ImageDraw.Draw(out)
    for i, t in enumerate(tiles):
        out.paste(t, (i * W, 20))
        im = dk[i].resize((dk[i].width * 5, dk[i].height * 5), Image.NEAREST)
        out.paste(im, (i * W + (W - im.width) // 2, H + 40 + (H - im.height) // 2), im)
    d.text((6, 4), "Demon Spawn (DK2 files: DemonSpawn-stand frame 0)", fill=(235, 220, 140))
    d.text((6, H + 24), "DK1 Demon Spawn, Stand (666): front, front-diagonal, side, back-diagonal, back",
           fill=(235, 220, 140))
    out.save(out_png)


ANIM_TILES = [  # (clip, frame or seconds as float, azimuth, label)
    ("walk", 8, 60, "walk"), ("running", 5, 60, "running"), ("attack", 0.42, 70, "attack: rears"),
    ("attack", 0.58, 70, "attack: pounce bite"), ("breathe_fire", 0.7, 50, "breathe_fire (spit)"),
    ("celebrating", 0.7, 40, "celebrating"), ("dance", 0.5, 30, "dance"),
    ("eating", 0.58, 70, "eating: tongue"), ("eating", 0.8, 70, "eating: reel in"),
    ("sleeping", 0, 40, "sleeping (lair bed)"), ("being_picked_up", 0.3, 60, "being_picked_up"),
    ("getting_hit", 0.1, 50, "getting_hit"), ("praying", 1.6, 70, "praying (bow)"),
    ("fidgeting", 3.6, 40, "fidgeting: roar"), ("die", -1, 50, "die (last)"), ("torture", 0, 30, "torture"),
]


def animations(export_dir, out_png, tex, tiles=ANIM_TILES, cols=4):
    W, H = 330, 280
    out = Image.new("RGB", (W * cols, H * ((len(tiles) + cols - 1) // cols)), BG)
    for k, (clip, f, az, label) in enumerate(tiles):
        m = _load(export_dir, clip)
        n = m.anim.frame_count
        if isinstance(f, float):
            f = int(round(f * 30))
        f = n - 1 if f < 0 else min(f, n - 1)
        hand = clip == "being_picked_up"
        centre = (0, -.18, 0) if hand else ((0, .14, 0) if clip in ("sleeping", "torture") else (0, .14, .05))
        img = render(m, tex, f, az, 20, (W, H), centre=centre, span=.9, floor=not hand and clip != "torture",
                     label=f"{label}  [{f}/{n - 1}]")
        out.paste(img, ((k % cols) * W, (k // cols) * H))
    out.save(out_png)


def main(export_dir, out_dir):
    from scripts.dk2texture import TextureCache
    from scripts.dk2paths import GAME_DIR          # .../dk2game/Ai
    cache = TextureCache(Path(GAME_DIR) / "EngineTextures.dir", Path(GAME_DIR) / "EngineTextures.dat")
    skins = {p.stem: Image.open(p) for p in Path(export_dir).glob("DemonSpawn_*.png")}
    tex = Textures(skins, cache)
    out_dir = Path(out_dir)
    preview(export_dir, out_dir / "DemonSpawn_preview.png", tex)
    animations(export_dir, out_dir / "DemonSpawn_animations.png", tex)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
