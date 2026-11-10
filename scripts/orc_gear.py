"""The Orc's own pieces on top of the Troll: his shaggy straw mane and his round wooden buckler, and the
128 x 128 skin they share (Orc_Gear), painted in the DK2 manner (shaded, textured, not flat colour).

From the DK1 Orc sprites (CREATURE.JTY, animations 182-200): a big bushy mane of straw-coloured hair
sitting on top of the head with a spiky crest at the front and falling down the back of the head to the
shoulders; a round shield with a gold rim, a dark wooden face and a central boss, carried on the arm
away from the mallet.

Geometry is in KMF space of the Troll's reference frame (Troll-Idle1 frame 0): x across, y forward,
z down (height = -z).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from scripts.orc_build import Accessory

GEAR = "Orc_Gear"
SIZE = 128
# texture regions (u0, v0, u1, v1)
SHIELD_FRONT = (0.0, 0.0, 0.5, 0.5)
SHIELD_BACK = (0.5, 0.0, 1.0, 0.5)
HAIR = (0.0, 0.5, 1.0, 1.0)


# ------------------------------------------------------------------------------------- geometry


def _icosphere(level=1):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f)


def _outward(p, tris, centre):
    """Wind each triangle so its normal points away from centre, in the game's convention
    (outward = -(b - a) x (c - a) in KMF axes)."""
    out = []
    for a, b, c in tris:
        n = -np.cross(p[b] - p[a], p[c] - p[a])
        if np.dot(n, (p[a] + p[b] + p[c]) / 3 - centre) < 0:
            b, c = c, b
        out.append((a, b, c))
    return np.array(out)


def mane(head: np.ndarray, seed: int = 7) -> Accessory:
    """A shaggy closed bush over the top and back of the head, clear of the face, with spiky tufts and
    a crest at the front. head: the head's reference points (KMF)."""
    rng = np.random.default_rng(seed)
    lo, hi = head.min(0), head.max(0)
    cx = (lo[0] + hi[0]) / 2
    top = lo[2]                                   # z of the crown (most negative)
    back = lo[1]
    centre = np.array([cx, back + 0.030, top + 0.052])
    radii = np.array([(hi[0] - lo[0]) / 2 + 0.014, 0.064, 0.078])
    v, f = _icosphere(1)
    p = centre + v * radii
    hairline = lo[1] + 0.62 * (hi[1] - lo[1])      # nothing forward of this but the crest ...
    brow = top + 0.026                             # ... and in front of the ears nothing below the brow
    for i in range(len(p)):
        up = (centre[2] - p[i, 2]) / radii[2]     # +1 at the top of the bush, -1 at its bottom
        lim = hairline + (0.02 if up > 0.55 else 0.0)
        if p[i, 1] > lim:
            p[i, 1] = lim
        fwd = (p[i, 1] - (back + 0.012)) / 0.035  # 0 at the back of the skull, 1 by the ears
        if fwd > 0 and p[i, 2] > brow:
            p[i, 2] = brow + (p[i, 2] - brow) * max(0.0, 1 - fwd) * 0.3
        # spikes: tufts pushed out from the centre, longest on the top and back
        d = p[i] - centre
        d /= np.linalg.norm(d)
        k = 0.008 + 0.020 * rng.random() * (0.6 + 0.4 * max(up, 0.0)) * (1.2 if p[i, 1] < centre[1] else 1.0)
        if i % 3 == 0:
            k *= 1.7
        p[i] += d * k
    # the crest: four tall tufts along the top, from the hairline back
    crest = []
    for j, y in enumerate(np.linspace(hairline + 0.01, back + 0.03, 4)):
        base = np.array([cx + rng.uniform(-0.01, 0.01), y, top - 0.012])
        tip = base + np.array([rng.uniform(-0.008, 0.008), 0.018 - 0.012 * j, -0.05 + 0.006 * j])
        w = 0.016
        crest.append((base + [-w, -w * 0.6, 0], base + [w, -w * 0.6, 0], base + [0, w, 0], tip))
    pts = [p]
    tris = [f]
    n = len(p)
    for b0, b1, b2, tip in crest:
        pts.append(np.array([b0, b1, b2, tip]))
        tris.append(np.array([(n, n + 1, n + 3), (n + 1, n + 2, n + 3), (n + 2, n, n + 3), (n, n + 2, n + 1)]))
        n += 4
    P = np.concatenate(pts)
    T = np.concatenate(tris)
    T = _outward(P, T, centre)
    # UVs: around the vertical axis for u, height for v, inside the hair region
    az = (np.arctan2(P[:, 0] - cx, P[:, 1] - centre[1]) / (2 * np.pi)) % 1.0
    ht = np.clip((P[:, 2] - (top - 0.07)) / (radii[2] * 2 + 0.07), 0, 1)
    uv = np.stack([HAIR[0] + (HAIR[2] - HAIR[0]) * az, HAIR[1] + (HAIR[3] - HAIR[1]) * ht], 1)
    return Accessory("Mane", P, T, uv, GEAR, "head")


def buckler(hand: np.ndarray, radius: float = 0.082) -> Accessory:
    """A round shield strapped to the free forearm, facing outwards and a little forwards, its centre
    just outside the hand. hand: the free forearm's reference points (KMF)."""
    c0 = hand.mean(0)
    side = np.sign(c0[0]) or -1.0
    n = np.array([side, 0.75, 0.0])
    n /= np.linalg.norm(n)
    centre = c0 + n * 0.024 + np.array([0, 0, -0.012])
    a = np.cross(n, [0, 0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    seg = 12
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = lambda r, off: centre + n * off + np.outer(np.cos(ang), a) * r + np.outer(np.sin(ang), b) * r
    rim_out = ring(radius, 0.000)          # 0..11
    rim_in = ring(radius * 0.84, 0.008)    # 12..23
    boss = centre + n * 0.020              # 24
    back = ring(radius * 0.96, -0.010)     # 25..36
    back_c = centre - n * 0.012            # 37
    P = np.vstack([rim_out, rim_in, boss, back, back_c])
    T = []
    for i in range(seg):
        j = (i + 1) % seg
        T += [(i, j, seg + j), (i, seg + j, seg + i)]          # bevelled rim
        T += [(seg + i, seg + j, 24)]                           # the face, rising to the boss
        T += [(i, 25 + j, j), (i, 25 + i, 25 + j)]              # the edge
        T += [(25 + i, 37, 25 + j)]                             # the back
    T = _outward(P, np.array(T), centre)
    # UVs: the face and rim planar in the front circle, the back planar in the back circle
    def planar(q, box, r):
        u = (q - centre) @ a / (2 * r) + 0.5
        v = (q - centre) @ b / (2 * r) + 0.5
        return np.stack([box[0] + (box[2] - box[0]) * u, box[1] + (box[3] - box[1]) * v], 1)
    uv = np.vstack([planar(P[:25], SHIELD_FRONT, radius * 1.02), planar(P[25:], SHIELD_BACK, radius * 1.02)])
    return Accessory("Buckler", P, T, uv, GEAR, "hand")


# ------------------------------------------------------------------------------------- the skin


def _noise(shape, scale, rng):
    small = rng.random((max(2, shape[0] // scale), max(2, shape[1] // scale)))
    im = Image.fromarray((small * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255


def _capsule(px, py, a, b, rad):
    """Distance field of a capsule from a to b, minus its radius (negative inside)."""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy)) - rad


def _shield_face(h: int, rng, k: int = 4) -> np.ndarray:
    n = h * k
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    px = (xx + 0.5) / n * 2 - 1                  # -1..1 across, +x right
    py = (yy + 0.5) / n * 2 - 1                  # -1..1 down, +y down
    r = np.hypot(px, py)

    # base: dark olive-brown, mottled at two scales, lit from the upper left
    m1 = _noise((n, n), 10 * k, rng)
    m2 = _noise((n, n), 3 * k, rng)
    base = np.array([0.20, 0.17, 0.08]) * (0.55 + 0.45 * m1[..., None]) * (0.8 + 0.3 * m2[..., None])
    base *= (1.05 - 0.25 * ((px - py) * 0.5 + 0.5))[..., None] * 1.1
    base += np.array([0.05, 0.05, 0.0]) * np.clip(m1 - 0.6, 0, 1)[..., None] * 2   # olive blotches

    # the emblem as a height field: two crossed bones (a ridge, knobbed ends) under a domed skull
    hf = np.zeros((n, n), np.float32)
    for (x0, y0), (x1, y1) in (((0.50, -0.50), (-0.50, 0.50)), ((-0.50, -0.50), (0.50, 0.50))):
        d = _capsule(px, py, (x0, y0), (x1, y1), 0.065)
        dx, dy = (x1 - x0), (y1 - y0)
        ln = np.hypot(dx, dy)
        nx, ny = -dy / ln, dx / ln               # across the bone
        for ex, ey, sx, sy in ((x0, y0, dx, dy), (x1, y1, -dx, -dy)):
            for side in (-1, 1):                 # the two knobs at each end
                cx = ex + nx * side * 0.075 + sx / ln * 0.0
                cy = ey + ny * side * 0.075 + sy / ln * 0.0
                d = np.minimum(d, np.hypot(px - cx, py - cy) - 0.088)
        bone = np.clip(-d / 0.05, 0, 1)
        hf = np.maximum(hf, bone ** 0.6 * 0.55)
    # the skull, front on, as a silhouette: a round cranium, the temples pinching in, the cheekbones
    # flaring, then the jaw narrowing to a flat chin (no eye sockets, nose or mouth)
    def width(y):                                   # half-width of the skull at height y (down = +)
        cran = np.where(np.abs(y + 0.12) <= 0.33, 0.37 * np.sqrt(np.clip(1 - ((y + 0.12) / 0.33) ** 2, 0, 1)), 0)
        cheek = np.where((y > 0.02) & (y < 0.22), 0.30 - 0.6 * np.abs(y - 0.12) ** 1.6, 0)
        jaw = np.where((y >= 0.16) & (y <= 0.40), 0.22 - 0.16 * ((y - 0.16) / 0.24) ** 2, 0)
        return np.maximum(np.maximum(cran, cheek), jaw)
    w = width(py)
    inside = np.clip((w - np.abs(px)) / 0.05, 0, 1) * (w > 0)
    skull = inside
    dome = np.sqrt(np.clip(1 - (px / 0.4) ** 2 - ((py + 0.05) / 0.48) ** 2, 0, 1))
    hf = hf * (1 - skull) + skull * (0.55 + 0.45 * dome)
    emblem = (hf > 0.02).astype(np.float32)

    # lighting of the relief: light from the upper left
    gy, gx = np.gradient(hf * n * 0.02)
    lit = np.clip(0.6 - (gx * -0.7 + gy * -0.7) * 3.0, 0, 1.5)
    bronze = np.array([0.66, 0.44, 0.20])
    hi = np.array([1.0, 0.82, 0.48])
    metal = bronze * (0.35 + 0.75 * lit[..., None]) + (hi - bronze) * np.clip(lit - 0.95, 0, 1)[..., None] * 1.6
    metal *= (0.85 + 0.25 * m2[..., None])           # hammered, uneven
    # a soft shadow of the emblem on the face, cast down-right
    sh = np.roll(np.roll(emblem, 2 * k, 0), 2 * k, 1)
    base *= (1 - 0.45 * sh)[..., None]
    out = base * (1 - emblem[..., None]) + metal * emblem[..., None]

    # the rim: a thin bronze band with a dark inner lip and its own sheen
    ang = np.arctan2(py, px)
    sheen = 0.55 + 0.45 * np.cos(ang + 2.35)
    rim = r > 0.86
    out[rim] = (bronze * (0.5 + 0.7 * sheen[..., None]) * (0.85 + 0.25 * m2[..., None]))[rim]
    out[(r > 0.83) & (r <= 0.86)] *= 0.35
    out[r > 0.985] *= 0.55
    out = np.clip(out, 0, 1)
    im = Image.fromarray((out * 255 + 0.5).astype(np.uint8)).resize((h, h), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255


def gear_texture(seed: int = 3) -> Image.Image:
    rng = np.random.default_rng(seed)
    S = SIZE
    img = np.zeros((S, S, 3), np.float32)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)

    # --- shield face (top-left 64 x 64), after the DK1 sprite: a dark mottled olive-brown face, a thin
    # bronze rim and an embossed bronze skull (the head only, no eyes or mouth) on two crossed bones.
    # Painted at 4x and scaled down.
    h = S // 2
    cy = h / 2
    face = _shield_face(h, rng)
    img[:h, :h] = face

    # --- shield back (top-right): plain planks and a leather strap
    back = np.array([0.36, 0.27, 0.15]) * (0.7 + 0.3 * _noise((h, h), 3, rng)[..., None])
    back[(xx[:h, :h] % 9) < 1] *= 0.5
    strap = np.abs(yy[:h, :h] - cy) < 5
    back[strap] = np.array([0.24, 0.13, 0.07]) * (0.8 + 0.2 * _noise((h, h), 2, rng)[strap][..., None])
    img[:h, h:] = back

    # --- hair (bottom half): straw strands running down, darker at the roots and in clumps
    H = S - h
    strands = np.zeros((H, S), np.float32)
    for _ in range(420):
        x0 = rng.uniform(0, S)
        y0 = rng.uniform(-10, H)
        ln = rng.uniform(8, 26)
        dx = rng.uniform(-0.25, 0.25)
        b = rng.uniform(0.5, 1.0)
        for t in np.linspace(0, ln, int(ln * 2)):
            xi, yi = int(x0 + dx * t) % S, int(y0 + t)
            if 0 <= yi < H:
                strands[yi, xi] = max(strands[yi, xi], b * (1 - t / ln * 0.5))
    strands = np.asarray(Image.fromarray((strands * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)), np.float32) / 255
    clump = _noise((H, S), 10, rng)
    base = np.array([0.62, 0.48, 0.24])
    hi = np.array([0.95, 0.82, 0.52])
    hair = base * (0.50 + 0.40 * clump[..., None]) + (hi - base) * strands[..., None] * 0.9
    hair *= (0.75 + 0.25 * (yy[h:, :] - h)[..., None] / H)
    img[h:, :] = hair
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
