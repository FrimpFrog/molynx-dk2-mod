"""Previews for the Classic Reaper's grit test: a small software renderer for his KMF files with
game-like light (a key light from above, warm, a dim ambient and a cool rim, bilinear texture
filtering like the game's), the camera tilted down a little like the Keeper's view.

Skins come from a folder of PNGs (HReaper_*.png), so the old and the new skins render the same way.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf

KEY = np.array([-0.35, 0.85, -0.40])          # from above, a little from the front-left
KEY = KEY / np.linalg.norm(KEY)
KEY_COL = np.array([1.00, 0.92, 0.80])
AMB_COL = np.array([0.36, 0.33, 0.34])
RIM = np.array([0.40, 0.30, 0.85])
RIM = RIM / np.linalg.norm(RIM)
RIM_COL = np.array([0.18, 0.20, 0.28])


def load(path: Path):
    return parse_kmf(Path(path).read_bytes())


def skins(folder: Path) -> dict:
    return {p.stem: np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
            for p in Path(folder).glob("HReaper_*.png")}


def _parts(k, frame):
    if k.kind == "mesh":
        return k.mesh.sprites, mesh_points(k)
    return k.anim.sprites, evaluate_anim_frame(k.anim, frame or 0)


def _yup(p):
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def _rot(az, el):
    a, e = np.radians(az), np.radians(el)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ ry


def view_of(k, frames, az, el):
    r = _rot(az, el)
    pts = []
    for f in frames:
        _, raw = _parts(k, f)
        pts += [_yup(p) @ r.T for p in raw]
    p = np.concatenate(pts)
    lo, hi = p.min(0), p.max(0)
    return (lo + hi) / 2, float(max(hi[:2] - lo[:2])) * 1.12


def _bilinear(tex, u, t):
    h, w = tex.shape[:2]
    x = u * w - 0.5
    y = t * h - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0m, x1m = x0 % w, (x0 + 1) % w
    y0m, y1m = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return ((tex[y0m, x0m] * (1 - fx) + tex[y0m, x1m] * fx) * (1 - fy) +
            (tex[y1m, x0m] * (1 - fx) + tex[y1m, x1m] * fx) * fy)


def background(size):
    v, u = np.mgrid[0:size, 0:size] / (size - 1)
    g = 0.10 + 0.06 * np.exp(-((u - 0.5) ** 2 + (v - 0.45) ** 2) / 0.12)
    return np.stack([g * 1.05, g * 0.95, g * 0.90], -1)


def render(k, tex: dict, az=150.0, el=-18.0, size=400, frame=None, view=None, ss=2) -> Image.Image:
    sprites, raw = _parts(k, frame)
    r = _rot(az, el)
    pts = [_yup(p) @ r.T for p in raw]
    if view is None:
        p = np.concatenate(pts)
        lo, hi = p.min(0), p.max(0)
        view = ((lo + hi) / 2, float(max(hi[:2] - lo[:2])) * 1.12)
    centre, span = view
    big = size * ss
    col = np.asarray(Image.fromarray((background(size) * 255).astype(np.uint8)).resize((big, big)), np.float32) / 255
    zb = np.full((big, big), np.inf, np.float32)
    for sp, p3 in zip(sprites, pts):
        if not sp.triangles[0]:
            continue
        name = k.materials[sp.material].textures[0]
        if name not in tex:
            continue
        T = tex[name]
        v = p3 - centre
        sx = big / 2 + v[:, 0] / span * big
        sy = big / 2 - v[:, 1] / span * big
        uv = np.array([(x.uv[0] / 32768.0, x.uv[1] / 32768.0) for x in sp.vertices])
        nrm = _yup(np.array([x.normal for x in sp.vertices], float)) @ r.T
        for a, b, c in sp.triangles[0]:
            x0, x1 = int(max(min(sx[[a, b, c]]) - 1, 0)), int(min(max(sx[[a, b, c]]) + 2, big))
            y0, y1 = int(max(min(sy[[a, b, c]]) - 1, 0)), int(min(max(sy[[a, b, c]]) + 2, big))
            if x1 <= x0 or y1 <= y0:
                continue
            den = (sy[b] - sy[c]) * (sx[a] - sx[c]) + (sx[c] - sx[b]) * (sy[a] - sy[c])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            w0 = ((sy[b] - sy[c]) * (gx - sx[c]) + (sx[c] - sx[b]) * (gy - sy[c])) / den
            w1 = ((sy[c] - sy[a]) * (gx - sx[c]) + (sx[a] - sx[c]) * (gy - sy[c])) / den
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not ins.any():
                continue
            z = w0 * v[a, 2] + w1 * v[b, 2] + w2 * v[c, 2]
            win = ins & (z < zb[y0:y1, x0:x1])
            if not win.any():
                continue
            u = (w0 * uv[a, 0] + w1 * uv[b, 0] + w2 * uv[c, 0]) % 1.0
            t = (w0 * uv[a, 1] + w1 * uv[b, 1] + w2 * uv[c, 1]) % 1.0
            texel = _bilinear(T, u, t)
            n = w0[..., None] * nrm[a] + w1[..., None] * nrm[b] + w2[..., None] * nrm[c]
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
            # normals of the camera-facing side (the KMF normals point either way on thin parts)
            n = np.where((n[..., 2:3] > 0), -n, n)
            key = np.clip(n @ KEY, 0, 1)[..., None]
            rim = np.clip(n @ RIM, 0, 1)[..., None] ** 2
            light = AMB_COL + KEY_COL * key * 0.72 + RIM_COL * rim
            reg = col[y0:y1, x0:x1]
            reg[win] = (texel * light)[win]
            zb[y0:y1, x0:x1][win] = z[win]
    img = Image.fromarray((np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8))
    return img.resize((size, size), Image.LANCZOS) if ss > 1 else img
