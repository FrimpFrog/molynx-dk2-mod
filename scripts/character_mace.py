"""The Troll's spiked mace for the custom character: a faceted iron ball with spikes on a wooden handle, built to replace the Behemoth's club
in the same place and at the same size. LICENCE: GPL-3.0.

Textures are cut from the Dungeon Keeper 2 Troll's own weapon texture (iron head, wooden handle). The shape follows the original Dungeon
Keeper troll's mace: an icosahedron ball (20 flat faces) with a spike on each corner, kept to the low polygon count of the game.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.character_parts import Mesh, basis

ART = Path(__file__).resolve().parent.parent / "data" / "character_art"
# material cells of the mace texture (128x128): wood, dark iron, rusty iron, pale spike steel
CELLS = {"wood": (0, 0), "iron": (1, 0), "rust": (0, 1), "spike": (1, 1)}


def mace_texture() -> Image.Image:
    metal = Image.open(ART / "troll_metal.png").convert("RGB")
    wood = Image.open(ART / "troll_wood.png").convert("RGB")
    out = Image.new("RGB", (128, 128))
    m = np.asarray(metal.resize((64, 64), Image.LANCZOS), float)
    cells = {
        "wood": np.clip(np.asarray(wood.resize((64, 64), Image.LANCZOS), float).mean(-1, keepdims=True) * np.array([1.35, 0.92, 0.58]), 0, 255),
        "iron": np.clip(m * 0.62 + np.array([4, 4, 8]), 0, 255),
        "rust": np.clip(m.mean(-1, keepdims=True) * np.array([0.95, 0.88, 0.82]), 0, 255),
        "spike": np.clip(m * 0.85 + 6, 0, 255),
    }
    for k, (cx, cy) in CELLS.items():
        out.paste(Image.fromarray(cells[k].astype(np.uint8)), (cx * 64, cy * 64))
    return out


def _uv(cell, u, v):
    cx, cy = CELLS[cell]
    return ((cx * 64 + 4 + np.clip(u, 0, 1) * 56) / 128.0, (cy * 64 + 4 + np.clip(v, 0, 1) * 56) / 128.0)


def _icosahedron():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v, f


def build_mace(club_rest: np.ndarray) -> Mesh:
    """Mace in the rest pose of the club (142 vertices): same handle end, same direction, head where the club's head is."""
    c = club_rest
    cm = c.mean(0)
    ax = np.linalg.svd(c - cm, full_matrices=False)[2][0]
    t = (c - cm) @ ax
    end_lo, end_hi = c[t < t.min() + 0.15].mean(0), c[t > t.max() - 0.15].mean(0)
    # the thick end is the head; the handle end is the thin one
    rad = np.linalg.norm((c - cm) - np.outer(t, ax), axis=1)
    head_side = -1 if rad[t < 0].mean() > rad[t > 0].mean() else 1
    head_end, handle_end = (end_lo, end_hi) if head_side < 0 else (end_hi, end_lo)
    head_c = c[(t * head_side) > (t * head_side).max() - 1.0].mean(0)
    w = head_c - handle_end
    length = np.linalg.norm(w)
    w /= length
    u, v, _ = basis(w)
    m = Mesh()
    # handle: a six-sided wooden shaft from the handle end to the ball, a little thicker at the grip end
    sides = 6
    angs = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ball_r, spike_len = 0.30, 0.22
    shaft = length - 0.05
    for a, b in zip(angs, np.roll(angs, -1)):
        p = lambda d, r, ang: handle_end + w * d + r * (np.cos(ang) * u + np.sin(ang) * v)
        u0, u1 = a / (2 * np.pi), b / (2 * np.pi) if b > a else 1.0
        m.quad(p(0, 0.075, a), p(0, 0.075, b), p(shaft, 0.062, b), p(shaft, 0.062, a), _uv("wood", u0, 0), _uv("wood", u1, 0), _uv("wood", u1, 1), _uv("wood", u0, 1))
    cap = handle_end - w * 0.0
    for a, b in zip(angs, np.roll(angs, -1)):
        m.tri(handle_end + 0.075 * (np.cos(b) * u + np.sin(b) * v), handle_end + 0.075 * (np.cos(a) * u + np.sin(a) * v), cap, _uv("wood", 0, 0), _uv("wood", 1, 0), _uv("wood", .5, 1))
    # the ball: an icosahedron, faces alternating dark and rusty iron; a four-sided spike on every corner
    iv, ifaces = _icosahedron()
    ball_c = head_c
    local = lambda p: ball_c + p[0] * u * ball_r + p[1] * v * ball_r + p[2] * w * ball_r
    for k, f in enumerate(ifaces):
        cell = "iron" if k % 2 == 0 else "rust"
        m.tri(local(iv[f[0]]), local(iv[f[1]]), local(iv[f[2]]), _uv(cell, 0, 0), _uv(cell, 1, 0), _uv(cell, .5, 1))
    for corner in iv:
        n = corner / np.linalg.norm(corner)
        dirn = n[0] * u + n[1] * v + n[2] * w
        su, sv, _ = basis(dirn)
        base = ball_c + dirn * ball_r * 0.97
        tip = ball_c + dirn * (ball_r + spike_len)
        ring = [base + 0.085 * (np.cos(a) * su + np.sin(a) * sv) for a in np.linspace(0, 2 * np.pi, 4, endpoint=False)]
        for i in range(4):
            m.tri(ring[i], ring[(i + 1) % 4], tip, _uv("spike", 0, 0), _uv("spike", 1, 0), _uv("spike", .5, 1))
    return m
