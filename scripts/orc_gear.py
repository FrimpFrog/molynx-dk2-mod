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
# The top half is laid out so every piece gets about the texel density of the Troll's own skins (about
# 290 texels per model unit, as DK2's creature skins: Dwarf 245-275, Knight 230, Bile Demon 175):
# the 0.167-unit shield gets 48 x 48 texels a side, the ball 16 x 16, the shield's edge a 32 x 8 band.
SHIELD_FRONT = (0.0, 0.0, 0.375, 0.375)
SHIELD_BACK = (0.375, 0.0, 0.75, 0.375)
RIM_EDGE = (0.75, 0.125, 1.0, 0.1875)
HAIR = (0.0, 0.5, 1.0, 1.0)
BALL = (0.75, 0.0, 0.875, 0.125)


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
    """A full mane brushed back: a big cap over the top of the head from the hairline, loosely tufted,
    swept a little backwards and down behind the skull; clear of the face.
    head: the head's reference points (KMF)."""
    rng = np.random.default_rng(seed)
    lo, hi = head.min(0), head.max(0)
    cx = (lo[0] + hi[0]) / 2
    top = lo[2]                                   # z of the crown (most negative)
    back = lo[1]
    centre = np.array([cx, back + 0.028, top + 0.044])
    radii = np.array([(hi[0] - lo[0]) / 2 + 0.010, 0.076, 0.060])
    v, f = _icosphere(1)
    p = centre + v * radii
    # the face: the forehead's top edge is at about 0.66 of the head's depth, the eyes just below the
    # crown at the front (Troll: crown 0.498, eyes about 0.475); the hair may come forward over the top
    # of the forehead but never down over the brow, and behind the eyes it may come down the sides
    forehead = lo[1] + 0.70 * (hi[1] - lo[1])
    ears = lo[1] + 0.40 * (hi[1] - lo[1])
    brow = top + 0.010
    for i in range(len(p)):
        # loose tufts, then the backward sweep: behind the crown the hair is pulled back and a little
        # down and narrows towards the tail; the lower half hugs the skull
        d = p[i] - centre
        d /= np.linalg.norm(d)
        p[i] += d * (0.004 + 0.010 * rng.random()) * (1.5 if i % 3 == 0 else 1.0)
        e = max(0.0, (centre[1] - p[i, 1]) / radii[1])
        p[i, 1] -= 0.018 * e * e
        p[i, 2] += 0.012 * e * e
        p[i, 0] = cx + (p[i, 0] - cx) * (1 - 0.18 * e)
        below = max(0.0, (p[i, 2] - centre[2]) / radii[2])
        p[i, 0] = cx + (p[i, 0] - cx) * (1 - 0.12 * below)
        # last, keep the face clear: nothing forward of the forehead's edge, and forward of the ears
        # nothing below the brow (fading in from the ears to the forehead)
        if p[i, 1] > forehead:
            p[i, 1] = forehead
        fwd = np.clip((p[i, 1] - ears) / (forehead - ears), 0, 1)
        lowest = brow + (1 - fwd) * 0.08            # the allowed bottom rises towards the face
        if p[i, 2] > lowest:
            p[i, 2] = lowest
    pts = [p]
    tris = [f]
    P = np.concatenate(pts)
    T = np.concatenate(tris)
    T = _outward(P, T, centre)
    # UVs: around the vertical axis for u, height for v, inside the hair region
    az = (np.arctan2(P[:, 0] - cx, centre[1] - P[:, 1]) / (2 * np.pi) + 0.5) % 1.0   # seam at the front
    ht = np.clip((P[:, 2] - (top - 0.07)) / (radii[2] * 2 + 0.11), 0, 1)
    uv = np.stack([HAIR[0] + (HAIR[2] - HAIR[0]) * az, HAIR[1] + (HAIR[3] - HAIR[1]) * ht], 1)
    return Accessory("Mane", P, T, uv, GEAR, "head")


def ball(at: np.ndarray, radius: float = 0.017) -> Accessory:
    """The little brown ball DK1 gave its Orc and Troll between the legs, hanging just below and in
    front of the crotch."""
    v, f = _icosphere(0)
    centre = at + np.array([0.0, 0.004, radius * 0.6])
    P = centre + v * np.array([radius, radius * 0.9, radius * 1.1])
    T = _outward(P, f, centre)
    u = BALL[0] + (BALL[2] - BALL[0]) * (0.5 + 0.45 * v[:, 0])
    w = BALL[1] + (BALL[3] - BALL[1]) * (0.5 + 0.45 * v[:, 2])
    return Accessory("Ball", P, T, np.stack([u, w], 1), GEAR, "hips")


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
    P = np.vstack([rim_out, rim_in, boss, back, back_c, rim_out, back])   # 38..49, 50..61: the edge's own
    E0, E1 = 38, 50
    T = []
    for i in range(seg):
        j = (i + 1) % seg
        T += [(i, j, seg + j), (i, seg + j, seg + i)]          # bevelled rim
        T += [(seg + i, seg + j, 24)]                           # the face, rising to the boss
        T += [(E0 + i, E1 + j, E0 + j), (E0 + i, E1 + i, E1 + j)]   # the edge (its own band of texture)
        T += [(25 + i, 37, 25 + j)]                             # the back
    T = _outward(P, np.array(T), centre)
    # UVs: the face and rim planar in the front square, the back planar in the back square, the edge a
    # band of bronze (u around the shield, v across the edge)
    def planar(q, box, r):
        u = (q - centre) @ a / (2 * r) + 0.5
        v = (q - centre) @ b / (2 * r) + 0.5
        return np.stack([box[0] + (box[2] - box[0]) * u, box[1] + (box[3] - box[1]) * v], 1)
    k = (np.arange(seg) + 0.5) / seg
    band = lambda v: np.stack([RIM_EDGE[0] + (RIM_EDGE[2] - RIM_EDGE[0]) * k,
                               np.full(seg, RIM_EDGE[1] + (RIM_EDGE[3] - RIM_EDGE[1]) * v)], 1)
    uv = np.vstack([planar(P[:25], SHIELD_FRONT, radius * 1.02), planar(P[25:38], SHIELD_BACK, radius * 1.02),
                    band(0.15), band(0.85)])
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

    # --- shield face (48 x 48), after the DK1 sprite: a dark mottled olive-brown face, a thin bronze
    # rim and an embossed bronze skull (the head only, no eyes or mouth) on two crossed bones. Painted
    # at 4x and scaled down.
    h = S // 2
    q = int(round(SHIELD_FRONT[2] * S))                       # 48
    img[:q, :q] = _shield_face(q, rng)
    # --- shield back (48 x 48): plain planks and a leather strap across
    x0 = int(round(SHIELD_BACK[0] * S))
    back = np.array([0.36, 0.27, 0.15]) * (0.7 + 0.3 * _noise((q, q), 3, rng)[..., None])
    back[(xx[:q, :q] % 7) < 1] *= 0.5
    strap = np.abs(yy[:q, :q] - q / 2) < 4
    back[strap] = np.array([0.24, 0.13, 0.07]) * (0.8 + 0.2 * _noise((q, q), 2, rng)[strap][..., None])
    img[:q, x0:x0 + q] = back
    # --- the ball (16 x 16): a shaded brown, lit from the upper left
    bx0, by0 = int(BALL[0] * S), int(BALL[1] * S)
    py, px = np.mgrid[0:16, 0:16]
    rr = np.hypot(px - 6.0, py - 6.0) / 10
    img[by0:by0 + 16, bx0:bx0 + 16] = np.array([0.50, 0.32, 0.16]) * (1.05 - 0.5 * rr)[..., None] \
        * (0.9 + 0.2 * _noise((16, 16), 3, rng)[..., None])
    # --- the shield's edge (32 x 8): bronze, lighter along the middle
    ex0, ey0 = int(RIM_EDGE[0] * S), int(RIM_EDGE[1] * S)
    ev = np.abs(np.arange(8) - 3.5)[:, None] / 4
    img[ey0:ey0 + 8, ex0:ex0 + 32] = np.array([0.66, 0.44, 0.20]) * (1.0 - 0.45 * ev)[..., None] \
        * (0.85 + 0.25 * _noise((8, 32), 2, rng)[..., None])

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
