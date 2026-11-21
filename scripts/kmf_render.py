"""The review renderer for exported KMF files (the Dragon's software renderer, scripts.dragon_render.Canvas:
orthographic, z-buffered, back faces culled the way the game culls them, so a wrongly wound triangle shows
as a hole). A floor disk is drawn at the files' z = 0, so anything sunk into the floor is cut off by it and
anything floating shows a gap.

LICENCE: GPL-3.0.
"""
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
