"""The Dragon's three 128 x 128 skins (Dragon_Body, Dragon_Head, Dragon_Wing), painted by baking.

Each part's UV islands are placed in an atlas (ATLAS: texture, island -> pixel rectangle, and whether the
island's v runs across the texture). Every texel is then traced back to the surface it covers (its
triangle in UV space gives the bind-pose position, normal and the island's own u, v), and coloured by a
painter for that island: olive-green scales, the tan ribbed belly plates of the DK1 dragon from the tail's
underside up to the throat, ivory claws, a red mouth with a row of teeth, glowing yellow eyes, membranes
with dark finger veins. Like the game's own skins (painted renders), the bake carries soft light from
above and darkening in the folds; the engine's own lighting goes on top.

The bake runs at 4x and is reduced (Lanczos) to 128, with every island's edge bled outward so filtering
and mip-maps never pick up the background.

LICENCE: GPL-3.0.
"""
import numpy as np
from PIL import Image

SIZE = 128
SS = 4

# texture -> island -> (x0, y0, x1, y1) in pixels of the 128 texture, transpose (v runs along x)
ATLAS = {
    "Dragon_Body": {
        "trunk": ((0, 0, 128, 76), True),
        "leg": ((0, 78, 82, 128), True),
        "toe": ((84, 78, 128, 128), False),
    },
    "Dragon_Head": {
        "skull": ((0, 0, 128, 62), True),
        "jaw": ((0, 64, 100, 96), True),
        "spike": ((100, 64, 128, 96), False),
        "horn": ((0, 98, 30, 128), False),
        "brow": ((32, 98, 52, 128), False),
        "fang": ((54, 98, 72, 128), False),
        "nostril": ((74, 98, 100, 128), False),
    },
    "Dragon_Wing": {
        "membrane": ((0, 0, 128, 128), False),        # v2: the whole skin (no arm tube) ...
        "toe": ((0, 0, 26, 22), False),               # ... but the corner the fan leaves empty
    },
}

# where the tail ends on the trunk's length (fraction), and how much of the trunk island it gets
TAIL_SHARE = 0.30


def trunk_v(v, split):
    """Arc-length fraction -> island v: the tail (0..split) gets TAIL_SHARE, the body the rest."""
    v = np.asarray(v, float)
    return np.where(v < split, v / split * TAIL_SHARE,
                    TAIL_SHARE + (v - split) / (1 - split) * (1 - TAIL_SHARE))


def trunk_split():
    from scripts.dragon_model import PELVIS_RING, TRUNK_PATH, TRUNK_TIP, TRUNK_TOP
    pts = np.vstack([TRUNK_TIP, TRUNK_PATH, TRUNK_TOP])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return seg[1 + PELVIS_RING] / seg[-1], seg[-1]


def island_local(part):
    """Each vertex's island-local (u, v) (the trunk's v remapped, see trunk_v)."""
    uv = np.array(part.uv, float)
    if part.name == "trunk":
        split, _ = trunk_split()
        uv[:, 1] = trunk_v(uv[:, 1], split)
    if part.name == "wings":
        from scripts.dragon_model import WING_UV_EXTENT
        isl = np.array(part.island)
        m = isl == "membrane"
        ext = WING_UV_EXTENT
        uv[m] = uv[m] / np.maximum(ext, 1e-9) * 1.0     # fill the rectangle (aspect set by ATLAS)
    return uv


def atlas_uv(part, atlas=None, local=None):
    """Texture-space (0..1) UVs of a part's vertices (`atlas`, `local`: another creature's ATLAS and
    island_local; default the dragon's)."""
    atlas = ATLAS if atlas is None else atlas
    loc = (island_local if local is None else local)(part)
    out = np.zeros_like(loc)
    for i, (isl, (u, v)) in enumerate(zip(part.island, loc)):
        (x0, y0, x1, y1), tr = atlas[part.texture][isl]
        a, b = (v, u) if tr else (u, v)
        # half a texel inside the rectangle
        out[i] = ((x0 + 0.5 + a * (x1 - x0 - 1)) / SIZE, (y0 + 0.5 + b * (y1 - y0 - 1)) / SIZE)
    return out


# ------------------------------------------------------------------------------------------- noise

def _hash2(ix, iy, seed=0):
    h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def value_noise(x, y, seed=0):
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _hash2(ix, iy, seed)
    b = _hash2(ix + 1, iy, seed)
    c = _hash2(ix, iy + 1, seed)
    d = _hash2(ix + 1, iy + 1, seed)
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy


def fbm(x, y, seed=0, octaves=3):
    out, amp, f, tot = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        out = out + amp * value_noise(x * f, y * f, seed + o)
        tot += amp
        amp *= 0.5
        f *= 2.0
    return out / tot


def scales(x, y, seed=0):
    """Overlapping scales on a staggered grid: (edge darkness 0..1, per-scale random 0..1).
    x around, y along (scale units)."""
    row = np.floor(y)
    xs = x + 0.5 * (row % 2)
    cx = np.floor(xs)
    fx = xs - cx - 0.5
    fy = y - row
    # a scale is a rounded shield: brighter at its top, dark rim at its lower edge
    d = np.sqrt((fx * 1.6) ** 2 + (np.clip(fy - 0.15, 0, None) * 1.25) ** 2)
    rim = np.clip((d - 0.55) / 0.3, 0, 1)
    rnd = _hash2(cx.astype(np.int64), row.astype(np.int64), seed)
    return rim, rnd, fy


# ------------------------------------------------------------------------------------------- palette
# v2 (Soles' community feedback, 15 Nov: "needs a texture rehaul to look more worn and gritty"): the same
# olive-green / tan family, painted the way the game's own skins are (the Salamander's, the Bile Demon's):
# chunky scales of 4-5 texels with near-black crevices between them, each lit from above, chipped and
# scuffed ones among them, soot round the mouth and nostrils, grime on the belly, feet and claws, ridged
# and scuffed horns, wings of veined, mottled, scarred leather with notches in the hem.

C = {k: np.array(v, float) for k, v in {
    "scale_hi": (210, 194, 88), "scale": (134, 126, 46), "scale_lo": (86, 88, 34), "scale_dark": (48, 46, 18),
    "crevice": (30, 25, 10), "back": (84, 76, 28), "worn": (170, 162, 118), "pit": (60, 44, 22),
    "belly_hi": (226, 178, 108), "belly": (176, 118, 60), "belly_lo": (128, 80, 38), "belly_groove": (54, 30, 12),
    "claw": (160, 136, 90), "claw_tip": (206, 190, 148), "claw_base": (70, 54, 30),
    "mouth": (102, 26, 18), "mouth_hi": (150, 54, 36), "tooth": (226, 214, 170), "tooth_stain": (150, 128, 72),
    "eye": (230, 60, 10), "eye_core": (255, 214, 70), "eye_rim": (40, 26, 6), "pupil": (20, 10, 0),
    "membrane": (140, 120, 56), "membrane_hi": (186, 164, 88), "membrane_lo": (90, 74, 34), "vein": (52, 30, 14),
    "scar": (118, 92, 58), "hole": (16, 12, 8),
    "horn": (206, 188, 136), "horn_base": (84, 72, 34), "spike": (104, 92, 38), "spike_tip": (214, 194, 142),
    "soot": (22, 18, 16), "grime": (58, 44, 24), "mud": (58, 42, 24),
}.items()}


def mix(a, b, t):
    t = np.asarray(t, float)[..., None]
    return a * (1 - t) + b * t


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------- v2 helpers

def worley(x, y, seed=0, jitter=0.75):
    """Scales on a staggered, jittered grid (unit cells): distance to the nearest and second nearest
    centre (F1, F2), the nearest cell's random 0..1 and the offset (dx, dy) from its centre."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    f1 = np.full(x.shape, 9.0)
    f2 = np.full(x.shape, 9.0)
    rid = np.zeros(x.shape)
    ox = np.zeros(x.shape)
    oy = np.zeros(x.shape)
    iy0 = np.floor(y).astype(np.int64)
    for dy in (-1, 0, 1):
        cy = iy0 + dy
        shift = 0.5 * (cy % 2)
        ix0 = np.floor(x + shift).astype(np.int64)
        for dx in (-1, 0, 1):
            cx = ix0 + dx
            jx = (_hash2(cx, cy, seed) - 0.5) * jitter
            jy = (_hash2(cx, cy, seed + 7) - 0.5) * jitter
            px = cx + 0.5 + jx - shift
            py = cy + 0.5 + jy
            ddx, ddy = x - px, y - py
            d = np.sqrt(ddx * ddx + ddy * ddy)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            rid = np.where(closer, _hash2(cx, cy, seed + 13), rid)
            ox = np.where(closer, ddx, ox)
            oy = np.where(closer, ddy, oy)
            f1 = np.where(closer, d, f1)
    return f1, f2, rid, ox, oy


def scale_field(x, y, seed, lx, ly, crev=0.16, bump=1.0):
    """Painted scales at texel coordinates x, y (cell units): -> (crevice 0..1, scale random 0..1,
    painted light factor ~1 (lit from (lx, ly), the surface's "up" in these coordinates), dome 0..1).
    Every scale is drawn a little differently: its own size, its own lean (so no gradient is copied from
    cell to cell), sunk into a crevice where it meets its neighbours. v2b: the light is the dome's own
    smooth slope toward the light (one soft lit side, one shaded side per scale) instead of a finite
    difference over the cell field, which left stepped bands inside the cells at close range."""
    f1, f2, r, ox, oy = worley(x, y, seed)
    edge = f2 - f1
    r2 = _hash2(np.floor(r * 9973).astype(np.int64), np.int64(5), seed)
    R = 0.5 + 0.22 * r2
    q = f1 / R
    dome = np.clip(1 - q ** 2, 0, 1)
    a = (r - 0.5) * 1.6                                  # each scale leans its own way
    llx, lly = lx * np.cos(a) - ly * np.sin(a), lx * np.sin(a) + ly * np.cos(a)
    face = np.clip(-(ox * llx + oy * lly) / R, -1.2, 1.2)                  # +1 on the side facing the light
    crevice = 1 - smooth(0.0, crev, edge)
    lit = 1 + 0.3 * bump * (2.0 * face * (0.35 + 0.65 * smooth(0.0, 0.45, edge)) - 0.3 * smooth(0.5, 1.1, q))
    return crevice, r, lit, dome


def blotch(x, y, seed, freq=1.0):
    """Soft painted mottling (0..1) at a few texels' size: never finer than the skin can hold."""
    return fbm(x * freq, y * freq, seed, octaves=2)


def gouges(x, y, seed, n=4, length=7.0, width=0.7, box=(128.0, 128.0)):
    """A few short, tapered gouges (texel coordinates): -> (dark groove 0..1, light lip 0..1)."""
    groove = np.zeros(np.shape(x))
    lip = np.zeros(np.shape(x))
    for k in range(n):
        cx = _hash2(np.int64(k), np.int64(1), seed) * box[0]
        cy = _hash2(np.int64(k), np.int64(2), seed) * box[1]
        a = _hash2(np.int64(k), np.int64(3), seed) * np.pi
        L = length * (0.6 + 0.8 * _hash2(np.int64(k), np.int64(4), seed))
        dx, dy = np.cos(a), np.sin(a)
        px, py = x - cx, y - cy
        along = px * dx + py * dy
        across = -px * dy + py * dx
        taper = np.clip(1 - np.abs(along) / (L / 2), 0, 1)              # thick in the middle, pointed ends
        w = width * (0.35 + 0.65 * taper)
        groove = np.maximum(groove, (taper > 0) * smooth(w, w * 0.3, np.abs(across)))
        lip = np.maximum(lip, (taper > 0) * smooth(w * 2.2, w * 1.2, np.abs(across - w * 1.4)) * 0.7)
    return groove, lip


def _light(nrm, pos=None):
    """Painted light from above (the game's skins are lit renders): stronger than v1's."""
    up = np.clip(nrm[:, 1], -1, 1)
    front = np.clip(nrm[:, 2], -1, 1)
    return 0.8 + 0.34 * up + 0.06 * front


def _texel(t, island_w_around, island_w_along):
    """Island-local u (around) and v (along) in texels."""
    return t["u"] * island_w_around, t["v"] * island_w_along


def _grime(col, amount):
    return mix(col, col * 0.5 + C["grime"] * 0.5, np.clip(amount, 0, 1))


def _soot_near(pos, centres, radius):
    """0..1: how close (bind position) to any of `centres` (model space), soft over `radius`."""
    out = np.zeros(len(pos))
    for c in centres:
        d = np.linalg.norm(pos - np.asarray(c, float), axis=1)
        out = np.maximum(out, np.exp(-(d / radius) ** 2))
    return out


# ------------------------------------------------------------------------------------------- painters
# each gets a dict of per-texel arrays: u, v (island-local), pos (bind, (n,3)), nrm, and returns RGB

def _scaled_hide(x, y, seed, lx, ly, size, worn=0.12, crev=0.22, dark=0.55):
    """Olive scales, painted: each a soft dome lit on its upper side, per-scale colour, dark (not black)
    crevices between them, chipped (pale, worn edge) and pitted (dark, a scale gone) ones among them."""
    cr, r, lit, dome = scale_field(x / size[0], y / size[1], seed, lx, ly, crev=crev, bump=0.8)
    col = mix(C["scale_lo"], C["scale"], 0.4 + 0.6 * r)
    hi = 0.6 + 0.8 * _hash2(np.floor(r * 7919).astype(np.int64), np.int64(3), seed)   # each its own shine
    col = mix(col, C["scale_hi"], np.clip(0.55 * hi * dome * (lit - 0.92) * 3.0, 0, 0.75))
    col = col * np.clip(0.84 + 0.16 * lit, 0.68, 1.18)[..., None]
    chip = (r > 1 - worn)
    pit = (r < worn * 0.45)
    col = np.where(chip[..., None], mix(col, C["worn"], 0.6 * smooth(0.2, 0.7, 1 - dome)), col)
    col = np.where(pit[..., None], mix(col, C["pit"], 0.7 * smooth(0.0, 0.6, dome)), col)
    col = mix(col, C["crevice"], dark * cr)
    return col, cr, r, dome


def _zoned_hide(x, y, seed, lx, ly, big, small, zone, **kw):
    """Two scale sizes (whole fields, so no cell is ever stretched), blended across `zone` (0 big .. 1 small)."""
    a = _scaled_hide(x, y, seed, lx, ly, big, **kw)
    b = _scaled_hide(x, y, seed + 101, lx, ly, small, **kw)
    z = np.clip(zone, 0, 1)
    pick = z > 0.5 + 0.35 * (value_noise(x / 3.0, y / 3.0, seed + 7) - 0.5)       # ragged border, whole scales
    col = np.where(pick[..., None], b[0], a[0])
    return col, np.where(pick, b[1], a[1]), np.where(pick, b[2], a[2]), np.where(pick, b[3], a[3])


def tail_tip_v():
    """Island-local v of the trunk's first ring: the tail's tip cone (6 triangles fanned from TRUNK_TIP)
    lies between v = 0 (its tip) and this."""
    from scripts.dragon_model import TRUNK_PATH, TRUNK_TIP
    split, total = trunk_split()
    return float(trunk_v(np.linalg.norm(TRUNK_PATH[0] - TRUNK_TIP) / total, split))


# v2d: scales on the tail's tip cone come this many times shorter along it than on the rest of the tail
TIP_ALONG = 3.0
TIP_DARK = 0.3            # how much darker the very tip is (fading in over the cone)


def _tail_tip(u, v):
    """v2d (Soles: "the minor missing texture on the end of his tail"): the tip cone's six triangles all
    meet at one texel (u 0.5, v 0), so in texel space each is a long sheared sliver and the skin painted
    across them came out as dark streaks converging on the tip, with the belly edge running diagonally
    over it. Painting the cone in its own polar coordinates (around: where the line from the tip through
    the texel meets the first ring; along: v) gives every one of its faces the same scales as the ring it
    grows from, continuous over its edges and into the tail; nothing in the UVs or the mesh changes.
    -> (u, v) to paint with, and 0..1 how near the tip."""
    v0 = tail_tip_v()
    cone = v < v0
    ue = np.where(cone, np.clip(0.5 + (u - 0.5) * v0 / np.maximum(v, 1e-4), 0.0, 1.0), u)
    ve = np.where(cone, v0 - (v0 - v) * TIP_ALONG, v)
    return ue, ve, np.where(cone, 1 - v / v0, 0.0)


def paint_trunk(t):
    u, v, nrm, pos = t["u"], t["v"], t["nrm"], t["pos"]
    u, v, tip = _tail_tip(u, v)
    t = dict(t, u=u, v=v)
    x, y = _texel(t, 76.0, 128.0)                           # around (dorsal 0 .. ventral 76), along
    tail = smooth(TAIL_SHARE + 0.04, TAIL_SHARE - 0.04, v)
    lx, ly = -0.55 * (1 - tail) - 1.0 * tail, 0.85 * (1 - tail) + 0.1 * tail
    ln = np.hypot(lx, ly)
    lx, ly = lx / ln, ly / ln
    # scales: big plates down the back and shoulders, smaller toward the belly and up the neck
    zone = np.maximum(smooth(0.36, 0.5, u), smooth(0.82, 0.92, v))
    green, cr, r, dome = _zoned_hide(x, y, 3, lx, ly, (7.2, 6.8 + 2.0 * tail), (4.8, 4.6 + 1.2 * tail), zone,
                                     worn=0.14, dark=0.5, crev=0.2)
    green = mix(green, C["back"] * 0.9, 0.45 * smooth(0.25, 0.0, u))           # darker along the spine
    green *= (0.82 + 0.32 * blotch(x / 16, y / 16, 4))[..., None]                 # large painted forms
    # belly plates across the ventral faces (one every 5.6 texels along the trunk): each lit along its
    # upper edge, shadowed at its lower, a dark gap under it, a few chips knocked out of the edges
    k = y / (5.6 + 1.6 * tail) + 0.08 * (fbm(x / 20, y / 20, 6) - 0.5)
    f = k - np.floor(k)
    plate = np.floor(k).astype(np.int64)
    pid = _hash2(plate, np.zeros_like(plate), 8)
    band = mix(C["belly_lo"], C["belly"], 0.35 + 0.65 * smooth(0.0, 0.55, f))
    band = mix(band, C["belly_hi"], 0.45 * smooth(0.5, 0.85, f) * smooth(1.0, 0.88, f))
    band = mix(band, C["belly_groove"], 0.34 * smooth(0.88, 0.99, f) + 0.25 * smooth(0.08, 0.0, f))
    band *= (0.9 + 0.16 * pid)[..., None] * (0.88 + 0.22 * blotch(x / 10, y / 12, 9))[..., None]
    cx = (np.floor(x / 9) + 0.5 + 0.6 * (_hash2(plate, np.floor(x / 9).astype(np.int64), 10) - 0.5)) * 9
    chip = (_hash2(plate, np.floor(x / 9).astype(np.int64), 11) > 0.72) & (np.hypot(x - cx, (f - 0.95) * 6) < 1.6)
    band = np.where(chip[..., None], mix(band, C["belly_lo"] * 0.8, 0.6), band)
    g, lip = gouges(x, y, 12, n=5, length=8.0, box=(76.0, 128.0))
    band = mix(mix(band, C["belly_hi"] * 1.05, 0.5 * lip), C["belly_groove"], 0.75 * g)
    # where plates meet scales: a ragged line following the scales, with a dark crevice
    edge_u = 0.6 + 0.035 * (r - 0.5)
    edge = smooth(edge_u - 0.025, edge_u + 0.025, u)
    col = mix(green, band, edge)
    col = mix(col, C["crevice"], 0.55 * np.exp(-((u - edge_u + 0.01) / 0.018) ** 2))
    # grime: the lower belly, the groin, the tail's underside and its tip on the floor
    low = smooth(0.45, 0.12, pos[:, 1]) * (0.45 + 0.55 * smooth(0.4, 0.8, u))
    low = low * (0.6 + 0.6 * blotch(x / 7, y / 7, 13))
    col = _grime(col, np.clip(1.0 * low, 0, 0.85))
    col = col * (1 - TIP_DARK * smooth(0.0, 1.0, tip))[..., None]           # the tip a little darker
    return col * _light(nrm)[..., None]


def paint_leg(t):
    u, v, nrm, pos = t["u"], t["v"], t["nrm"], t["pos"]
    x, y = _texel(t, 50.0, 82.0)                            # around (front 0 .. back 50), along (hip -> foot)
    # big scales on the thigh, smaller at the knee and down the shin and foot
    zone = np.maximum(np.exp(-((v - 0.45) / 0.1) ** 2), smooth(0.65, 0.8, v))
    col, cr, r, dome = _zoned_hide(x, y, 7, 0.0, -1.0, (6.4, 6.0), (4.4, 4.2), zone, worn=0.16, dark=0.48, crev=0.2)
    # big scutes down the front of the shin and the foot: each lit on its top edge, a dark gap under it,
    # chipped here and there
    k = y / 5.2
    f = k - np.floor(k)
    sid = _hash2(np.floor(k).astype(np.int64), np.ones_like(k, np.int64), 13)
    scute = mix(C["belly_lo"] * 0.8, C["belly"] * 0.92, smooth(0.95, 0.35, f)) * (0.85 + 0.22 * sid)[..., None]
    scute = mix(scute, C["belly_hi"] * 0.95, 0.6 * smooth(0.25, 0.08, f) * smooth(0.0, 0.08, f))
    scute = mix(scute, C["belly_groove"], 0.75 * smooth(0.88, 0.99, f))
    nick = (_hash2(np.floor(k).astype(np.int64), np.floor(x / 4).astype(np.int64), 19) > 0.8) & (f < 0.3)
    scute = np.where(nick[..., None], mix(scute, C["belly_groove"], 0.55), scute)
    front = smooth(0.28, 0.12, u) * smooth(0.56, 0.62, v)                  # shin and foot only
    col = np.where((front > 0.5)[..., None], scute, col)
    col = mix(col, C["scale_dark"], 0.4 * smooth(0.6, 1.0, u) * smooth(0.5, 0.0, v))   # inner thigh
    col *= (0.72 + 0.3 * blotch(x / 14, y / 14, 14))[..., None]
    # scuffed knees (pale, worn) and mud caked up the feet
    knee = np.exp(-((v - 0.48) / 0.06) ** 2) * smooth(0.3, 0.05, u)
    col = mix(col, C["worn"], 0.3 * knee * smooth(0.45, 0.75, blotch(x / 3, y / 3, 15)))
    mud = smooth(0.2, 0.04, pos[:, 1]) * (0.6 + 0.6 * blotch(x / 5, y / 5, 16))
    col = mix(col, C["mud"] * (0.7 + 0.5 * blotch(x / 3, y / 3, 17))[..., None], np.clip(0.85 * mud, 0, 0.9))
    return col * _light(nrm)[..., None]


def paint_claw(t):
    """Claws: dirty at the root, horn-coloured, split along their length, the tips worn pale with nicks."""
    along, u = t["along"], t["u"]
    col = _ridged(along, u, C["claw_base"], C["claw"], seed=18, pits=False)
    # v2b: dirtier. Caked mud up the lower half, darker roots, grime run into the grooves toward the tip;
    # only the very point is worn pale
    dirt = smooth(0.92, 0.25, along + 0.12 * (blotch(u * 6, along * 6, 21) - 0.5)) * (0.75 + 0.45 * blotch(u * 8, along * 10, 19))
    col = mix(col, C["mud"] * (0.6 + 0.4 * blotch(u * 14, along * 14, 20))[..., None], np.clip(dirt, 0, 0.95))
    col = mix(col, C["soot"] * 1.4, 0.7 * smooth(0.5, 0.0, along))                  # dark roots
    tide = 0.5 + 0.06 * (blotch(u * 10, along * 3, 22) - 0.5)                      # a hard-ish tide line of mud
    col = mix(col, C["mud"] * 0.75, 0.85 * smooth(tide + 0.03, tide - 0.03, along))
    streak = np.exp(-((((u * 3.1 + 0.35 * along) % 1) - 0.5) / 0.06) ** 2) * smooth(0.2, 0.85, along) * smooth(1.0, 0.8, along)
    col = mix(col, C["grime"], 0.5 * streak)
    g = np.abs(((u * 2.3 + 0.8 * along) % 1) - 0.5) < 0.04                       # a scratch or two
    col = np.where((g & (along > 0.45))[..., None], col * 0.62, col)
    return col * _light(t["nrm"])[..., None]


EYE_V = 0.31        # the eyes: on the brow ring (ring 1), on the upper side of the head
NOSTRIL_V = 0.95


def paint_skull(t):
    from scripts.dragon_model import NOSTRILS
    u, v, nrm, pos = t["u"], t["v"], t["nrm"], t["pos"]
    x, y = _texel(t, 62.0, 128.0)                           # around (top 0 .. palate 62), along (back -> snout)
    # big plates on the crown, small scales on the snout
    col, cr, r, dome = _zoned_hide(x, y, 11, -1.0, 0.15, (6.6, 6.6), (4.2, 4.2), smooth(0.45, 0.65, v), worn=0.14,
                                   dark=0.62, crev=0.2)
    col = mix(col, C["back"] * 0.8, 0.4 * smooth(0.2, 0.0, u))                   # dark top of the head
    col = mix(col, C["belly_hi"] * 0.8, 0.3 * smooth(0.42, 0.62, u) * smooth(0.0, 0.2, v))   # yellow cheeks
    col *= (0.8 + 0.34 * blotch(x / 12, y / 14, 12))[..., None]
    # deep wrinkles across the snout, deepening toward the nose, each lit on its upper lip
    k = y / 5.5
    f = k - np.floor(k)
    zone = smooth(0.45, 0.75, v) * smooth(0.62, 0.25, u)
    col = mix(col, C["crevice"], 0.4 * zone * np.exp(-((f - 0.5) / 0.13) ** 2))
    col = mix(col, C["scale_hi"], 0.15 * zone * np.exp(-((f - 0.25) / 0.1) ** 2))
    # the mouth: the palate (u > 2/3) a mid red, ridged, darker toward the throat
    mouth = smooth(0.66, 0.70, u) * smooth(0.9, 0.84, v)
    kp = y / 4.5
    ridge = np.abs(kp - np.floor(kp) - 0.5) * 2
    palate = mix(C["mouth"], C["mouth_hi"], 0.35 + 0.35 * (1 - ridge) + 0.3 * smooth(0.3, 0.8, v))
    palate = palate * (0.85 + 0.25 * blotch(x / 3, y / 3, 34))[..., None] * (0.6 + 0.4 * smooth(0.0, 0.4, v))[..., None]
    palate = mix(palate, C["soot"] * 1.5, 0.75 * smooth(0.74, 0.68, u))          # scorched lip edge
    palate = mix(palate, C["soot"] * 2.2, 0.6 * smooth(0.62, 0.86, v))            # v2b: sooty toward the front
    col = mix(col, palate, mouth)
    # a row of big, stained, jagged teeth along the lip line, some broken
    s = v * 0.47
    k = s / 0.024
    tid = np.floor(k).astype(np.int64)
    tooth_w = np.abs((k - np.floor(k)) - 0.5) * 2
    tl = 0.13 * (0.7 + 0.3 * _hash2(tid, np.zeros_like(tid), 31))
    tl = np.where(_hash2(tid, np.zeros_like(tid), 32) > 0.82, tl * 0.45, tl)
    tooth = (u > 0.69 - tl) & (u < 0.70) & ((0.70 - u) / tl > tooth_w * 0.95) & (v > 0.2) & (v < 0.95)
    tcol = mix(C["tooth"], C["tooth_stain"], 0.25 + 0.5 * smooth(0.5, 1.0, 1 - (0.70 - u) / np.maximum(tl, 1e-6)))
    tcol = tcol * (0.78 + 0.22 * (1 - tooth_w))[..., None]
    col = np.where(tooth[..., None], tcol, col)
    col = mix(col, C["mouth"] * 0.4, 0.7 * np.exp(-((u - 0.575) / 0.014) ** 2) * (v > 0.2))   # gum line
    # soot: a fire-breather's lips blackened toward the front, and a scorched ring round each nostril
    lipline = np.exp(-((u - 0.63) / 0.09) ** 2) * smooth(0.2, 0.75, v)
    nos = [b * (s_, 1, 1) for b, _ in NOSTRILS for s_ in (1, -1)]
    nose = _soot_near(pos, nos, 0.065)
    tipsoot = smooth(0.82, 0.97, v) * 0.45                                     # the muzzle tip (v2b: lighter,
    soot = np.clip(1.0 * lipline + 0.7 * nose + tipsoot, 0, 1) * (0.6 + 0.55 * blotch(x / 4, y / 5, 33))  # nostrils read)
    col = mix(col, C["soot"], np.clip(soot, 0, 0.92) * (1 - 0.7 * tooth))
    # a heavy dark brow, low over the front of the eye
    v_eye = (v - EYE_V)
    brow = np.exp(-(((u - (0.225 + 0.35 * v_eye)) / 0.045) ** 2) - ((v_eye / 0.095) ** 2))
    col = mix(col, C["crevice"] * 1.2, 0.85 * brow)
    # the eye: a slanted slit of red-orange glow, its front corner lower, in a dark socket (as approved)
    du = (u - 0.305 - 0.5 * v_eye) / 0.044
    dv = v_eye / 0.066
    e = du ** 2 + dv ** 2
    eye = e < 1
    socket = (e >= 1) & (e < 2.4)
    col = mix(col, C["eye_rim"], np.where(socket, 0.85 * smooth(2.4, 1.0, e), 0.0))
    glow = mix(C["eye_core"], C["eye"], smooth(0.05, 0.9, e))
    glow = np.where((np.abs(dv) < 0.13)[..., None] & eye[..., None] & (np.abs(du) < 0.75)[..., None],
                    C["pupil"], glow)
    col = np.where(eye[..., None], glow / np.maximum(_light(nrm), 0.5)[..., None], col)   # eyes don't shade
    return col * _light(nrm)[..., None]


def paint_jaw(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    # u: 0 .. 0.3 the tongue (the jaw's top face, left lip to right lip), 0.5 the lips, 1 the keel
    x, y = _texel(t, 32.0, 100.0)
    side, cr, r, dome = _scaled_hide(x, y, 17, 0.0, 1.0, (4.4, 5.0), worn=0.12, dark=0.5)
    side *= (0.84 + 0.26 * blotch(x / 6, y / 6, 20))[..., None]
    tc, tcr, tr, tdome = scale_field(x / 4.0, y / 4.4, 24, 0.0, 1.0, crev=0.2, bump=0.6)
    throat = mix(C["belly_lo"], C["belly_hi"] * 0.95, 0.3 + 0.5 * tr) * np.clip(0.85 + 0.2 * tdome, 0.7, 1.1)[..., None]
    throat = mix(throat, C["belly_groove"], 0.45 * tc)
    throat = _grime(throat, 0.3 * blotch(x / 5, y / 5, 21))
    col = mix(side, throat, smooth(0.8, 0.9, u))
    # lower teeth just under the lip, stained
    tw = np.abs((v * 0.4 / 0.024 - np.floor(v * 0.4 / 0.024)) - 0.5) * 2
    tooth = (u > 0.5) & (u < 0.58) & ((0.58 - u) / 0.08 > tw * 0.95) & (v > 0.22) & (v < 0.9)
    col = np.where(tooth[..., None], mix(C["tooth"], C["tooth_stain"], 0.4) * (0.8 + 0.2 * (1 - tw))[..., None],
                   col)
    soot = np.exp(-((u - 0.54) / 0.09) ** 2) * smooth(0.3, 0.9, v) * (0.5 + 0.7 * blotch(x / 4, y / 4, 22))
    col = mix(col, C["soot"], np.clip(soot, 0, 0.85) * (1 - 0.6 * tooth))
    # the tongue face: a mid-red tongue between two rows of teeth (seen when the jaw drops), darker deep
    # in the throat
    tu = np.clip(u / 0.3, 0, 1)
    tongue = mix(C["mouth"], C["mouth_hi"], 0.45 + 0.4 * blotch(x / 3, y / 4, 23) - 0.3 * np.abs(tu - 0.5))
    tongue = tongue * (0.55 + 0.45 * smooth(0.0, 0.5, v))[..., None]
    tongue = mix(tongue, C["soot"] * 2.2, 0.5 * smooth(0.55, 0.8, v))             # v2b: sooty toward the front
    rows = np.minimum(tu, 1 - tu)
    tk = v * 0.4 / 0.022
    tw2 = np.abs((tk - np.floor(tk)) - 0.5) * 2
    lower = (rows < 0.2) & ((0.2 - rows) / 0.2 > tw2 * 0.9) & (v > 0.2) & (v < 0.92)
    tongue = np.where(lower[..., None], mix(C["tooth"], C["tooth_stain"], 0.35) * (0.8 + 0.2 * (1 - tw2))[..., None],
                      tongue)
    # the very front of the tongue face shows between the lips at the chin: scaled lip, not red
    tongue = mix(tongue, side * 0.8, smooth(0.72, 0.82, v))
    col = np.where((u < 0.32)[..., None], tongue, col)
    return col * _light(nrm)[..., None]


def _ridged(along, u, base, tip, seed=40, pits=True):
    """Horn and spike: a dark, dirty base, long grooves up the shaft, a worn pale tip with nicks."""
    col = mix(base, tip, smooth(0.0, 0.7, along))
    groove = np.abs(((u * 4 + 0.6 * along) % 1) - 0.5)
    col = mix(col, C["crevice"] * 1.4, 0.55 * smooth(0.06, 0.0, 0.5 - groove) * smooth(0.95, 0.6, along))
    col = col * (0.82 + 0.3 * blotch(u * 12, along * 18, seed + 3))[..., None]
    # chips and nicks near the tip: small dark pits, each with a pale scuffed edge above it
    f1, f2, rr, ox, oy = worley(u * 7, along * 9, seed + 5)
    pit = (rr > 0.6) & (f1 < 0.28) & (along > 0.45) & pits
    col = np.where(pit[..., None], col * 0.45, col)
    col = mix(col, C["claw_tip"], 0.55 * smooth(0.78, 1.0, along) * (~pit))                 # scuffed tip
    col = _grime(col, 0.75 * smooth(0.3, 0.0, along))
    return col


def paint_horn(t):
    col = _ridged(t["along"], t["u"], C["horn_base"], C["horn"], seed=41)
    return col * _light(t["nrm"])[..., None]


def paint_spike(t):
    col = _ridged(t["along"], t["u"], C["spike"], C["spike_tip"] * 1.05, seed=42)
    return col * (0.92 + 0.12 * np.sin(t["u"] * 9))[..., None]


def paint_brow(t):
    col = mix(C["crevice"] * 1.3, C["back"], t["along"])
    col = col * (0.85 + 0.3 * blotch(t["u"] * 20, t["along"] * 20, 43))[..., None]
    col = mix(col, C["worn"] * 0.8, 0.35 * smooth(0.75, 1.0, t["along"]))
    return col * _light(t["nrm"])[..., None]


def paint_nostril(t):
    """The raised nostril tubes (Soles' favourite): scorched brown-black from the soot of the fire, a
    thin worn lip round the hole so they still read, and the black hole."""
    along = t["along"]
    col = mix(C["soot"] * 1.6, C["scale_lo"] * 0.9, smooth(0.0, 0.35, along))
    col = mix(col, C["soot"] * 2.2, smooth(0.35, 0.55, along))
    # v2b: a worn, lit lip round the top of each tube so the square nostrils read in the front view
    col = mix(col, C["worn"] * 0.9, 0.7 * smooth(0.4, 0.55, along) * smooth(0.62, 0.55, along) * smooth(-0.2, 0.5, t["nrm"][:, 1]))
    u, v = t["u"], t["v"]
    e = np.minimum(np.minimum(u - 0.25, 0.75 - u) / 0.25, np.minimum(v - 0.75, 1.0 - v) / 0.125)
    cap = mix(C["scale_lo"] * 0.55, C["soot"], smooth(0.08, 0.22, e))                  # scorched rim
    cap = mix(cap, C["hole"], smooth(0.22, 0.34, e))
    cap = mix(cap, np.array([170.0, 60.0, 14.0]), 0.55 * smooth(0.6, 1.0, e))         # an ember deep inside
    col = np.where((along > 0.9)[..., None], cap, col)
    return col * _light(t["nrm"])[..., None]


def paint_fang(t):
    col = mix(C["tooth_stain"] * 0.8, C["tooth"], smooth(0.0, 0.6, t["along"]))
    return mix(col, C["claw_tip"], 0.4 * smooth(0.8, 1.0, t["along"]))


def _wing_up():
    """World up (+y) in the membrane's island coordinates (the spread wing's plane, as _wing_uv lays it
    out, each axis scaled by its extent): the painted light comes from there."""
    from scripts.dragon_model import WING_POINTS, WING_UV_EXTENT
    P = np.array(list(WING_POINTS.values()), float)
    _, _, vt = np.linalg.svd(P - P.mean(0))
    up = np.array([vt[0][1], vt[1][1]]) / np.maximum(WING_UV_EXTENT, 1e-9)
    return up / np.linalg.norm(up)


def _stroke(p, a, b, width, up):
    """A painted line from a to b (island coordinates): -> (dark core 0..1, lit lip 0..1 on its upper
    side, along 0..1). Tapered at both ends like a brush stroke."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L = np.linalg.norm(ab)
    t = ab / L
    n = np.array([-t[1], t[0]])
    if n @ up < 0:
        n = -n
    rel = p - a
    al = rel @ t / L
    ac = rel @ n
    taper = np.clip(np.sin(np.pi * np.clip(al, 0, 1)), 0, 1) ** 0.6 * ((al > 0) & (al < 1))
    w = width * (0.3 + 0.7 * taper)
    core = np.exp(-(ac / np.maximum(w, 1e-6)) ** 2) * taper
    lip = np.exp(-((ac - 1.7 * w) / np.maximum(w, 1e-6)) ** 2) * taper
    return core, lip, al


def paint_membrane(t):
    """v2b: stretched leather, not stone. A smooth warm membrane lit from above (each panel billowing
    between its fingers: lit under the upper finger, shaded toward the lower), fine wrinkles fanning
    from the wrist and off the fingers, dark veins running beside the finger bones and branching into
    the panels, a darker trailing edge, a few deliberate notches and a slit, healed scar patches. The
    finger bones and the arm stay scaly."""
    from scripts.dragon_model import WING_UV, WING_UV_EXTENT
    u, v = t["u"], t["v"]
    ext = np.maximum(WING_UV_EXTENT, 1e-9)
    key = {k: np.array(q) / ext for k, q in WING_UV.items()}
    p = np.stack([u, v], -1)
    x, y = u * 128.0, v * 128.0                              # texels
    up = _wing_up()
    W = key["W"]

    def seg(a, b):
        a, b = key[a], key[b]
        ab = b - a
        h = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        return np.linalg.norm(p - (a + h[..., None] * ab), axis=-1), h

    # the base: one smooth warm gradient lit from above, a little lighter where the membrane is thinnest
    # (the middle of each panel, away from bones and hem), two or three broad soft stains
    hs = {k: (q - W) @ up for k, q in key.items()}
    g = smooth(min(hs.values()), max(hs.values()), (p - W) @ up)
    col = mix(C["membrane_lo"], C["membrane"], 0.3 + 0.7 * g)
    col = mix(col, C["membrane_hi"], 0.35 * smooth(0.45, 1.0, g))
    col = col * (0.94 + 0.1 * fbm(x / 26, y / 26, 50, octaves=2))[..., None]
    # each panel billows between its fingers: the part under the upper finger faces the light, the part
    # over the lower finger faces down
    a_p = np.arctan2(v - W[1], u - W[0])
    order = ["F1", "F2", "F3", "B", "S"]
    angs = [np.arctan2(key[k][1] - W[1], key[k][0] - W[0]) for k in order]
    angs = np.unwrap(angs)
    a_p = np.where(a_p < angs[0] - np.pi, a_p + 2 * np.pi, a_p)
    a_p = np.where(a_p > angs[0] + np.pi, a_p - 2 * np.pi, a_p)
    billow = np.ones(u.shape)
    for i in range(len(order) - 1):
        a0, a1 = angs[i], angs[i + 1]
        lo_, hi_ = min(a0, a1), max(a0, a1)
        inside = (a_p >= lo_) & (a_p <= hi_)
        s_ = np.clip((a_p - a0) / (a1 - a0), 0, 1)                   # 0 at the upper finger
        upper_first = hs[order[i]] >= hs[order[i + 1]]
        s_ = s_ if upper_first else 1 - s_
        b_ = 1 + 0.12 * np.sin(np.pi * s_ ** 0.7) - 0.08 * s_ * np.sin(np.pi * s_) ** 2   # 1 at both fingers: no seam
        billow = np.where(inside, b_, billow)
    col = col * billow[..., None]
    # fine wrinkles: fanning out from the wrist (where the membrane gathers) and short tension lines off
    # each finger toward the hem; each its own length, angle and strength, lit on its upper lip
    rng = np.random.default_rng(71)
    dark = np.zeros(u.shape)
    lite = np.zeros(u.shape)
    span = max(np.linalg.norm(key[k] - W) for k in ("F1", "F2", "F3", "B"))
    for i in range(len(order) - 2):
        a0, a1 = angs[i], angs[i + 1]
        n_w = 5 if i < 2 else 6
        for j in range(n_w):
            aa = a0 + (a1 - a0) * (j + 0.5 + 0.35 * (rng.random() - 0.5)) / n_w
            r0 = span * (0.04 + 0.05 * rng.random())
            r1 = span * (0.22 + 0.3 * rng.random())
            bend = 0.12 * (rng.random() - 0.5)
            rm = (r0 + r1) / 2
            A = W + r0 * np.array([np.cos(aa), np.sin(aa)])
            M_ = W + rm * np.array([np.cos(aa + bend), np.sin(aa + bend)])
            B_ = W + r1 * np.array([np.cos(aa + 0.4 * bend), np.sin(aa + 0.4 * bend)])
            wd = 0.0035 + 0.002 * rng.random()
            k_ = 0.5 + 0.5 * rng.random()
            for q0, q1 in ((A, M_ + (M_ - A) * 0.15), (M_ - (B_ - M_) * 0.15, B_)):
                c_, l_, _ = _stroke(p, q0, q1, wd, up)
                dark = np.maximum(dark, k_ * c_)
                lite = np.maximum(lite, k_ * l_)
    for fa, fb, sd in (("W", "F1", 1), ("W", "F2", 1), ("W", "F2", -1), ("W", "F3", 1), ("W", "F3", -1)):
        A0, B0 = key[fa], key[fb]
        tdir = (B0 - A0) / np.linalg.norm(B0 - A0)
        for j in range(4):
            at = 0.2 + 0.18 * j + 0.06 * rng.random()
            o = A0 + (B0 - A0) * at
            rot = sd * (0.45 + 0.25 * rng.random())
            d2 = np.array([tdir[0] * np.cos(rot) - tdir[1] * np.sin(rot), tdir[0] * np.sin(rot) + tdir[1] * np.cos(rot)])
            L = span * (0.06 + 0.06 * rng.random())
            c_, l_, _ = _stroke(p, o + d2 * 0.012, o + d2 * L, 0.003 + 0.0015 * rng.random(), up)
            k_ = 0.45 + 0.4 * rng.random()
            dark = np.maximum(dark, k_ * c_)
            lite = np.maximum(lite, k_ * l_)
    col = mix(col, C["membrane_lo"] * 0.62, 0.55 * dark)
    col = mix(col, C["membrane_hi"] * 1.08, 0.35 * lite * (1 - dark))
    # broad soft tension folds from the wrist out into the middle of each half panel (wide enough to survive
    # at play zoom; kept off the triangle diagonals so they never double a geometric fold)
    for fk, ck, at in (("F1", "C12", .45), ("F2", "C12", .5), ("F2", "C23", .5), ("F3", "C23", .45), ("F3", "C3B", .5)):
        A0, B0 = W, key[fk] + (key[ck] - key[fk]) * at
        c_, l_, _ = _stroke(p, A0 + (B0 - A0) * 0.08, A0 + (B0 - A0) * 0.92, 0.022, up)
        col = col * (1 - 0.13 * c_)[..., None]
        col = mix(col, C["membrane_hi"], 0.12 * l_ * (1 - c_))
    # veins: one running beside each finger bone (on its lower side), thick at the wrist, thinning toward
    # the tip, with a few branches curving off into the panel; two more down from the arm into the inner
    # membrane. Dark, soft, raised (a faint lit edge above)
    vd = np.zeros(u.shape)
    vl = np.zeros(u.shape)
    for (fa, fb), sd in zip((("W", "F1"), ("W", "F2"), ("W", "F3")), (72, 73, 74)):
        A0, B0 = key[fa], key[fb]
        L = np.linalg.norm(B0 - A0)
        tdir = (B0 - A0) / L
        n = np.array([-tdir[1], tdir[0]])
        if n @ up > 0:
            n = -n                                             # the lower side of the finger
        pts = [A0 + tdir * L * f + n * (0.046 - 0.024 * f + 0.005 * np.sin(9 * f + sd)) for f in np.linspace(0.1, 0.86, 7)]
        for q0, q1 in zip(pts[:-1], pts[1:]):
            f = (q0 - A0) @ tdir / L
            c_, l_, _ = _stroke(p, q0 - (q1 - q0) * 0.25, q1 + (q1 - q0) * 0.25, 0.0095 - 0.004 * f, up)
            vd = np.maximum(vd, c_)
            vl = np.maximum(vl, l_)
        r = np.random.default_rng(sd)
        for j in range(3):
            f = 0.25 + 0.22 * j + 0.05 * r.random()
            o = A0 + tdir * L * f + n * (0.046 - 0.024 * f)
            d1 = _unit2(tdir * 0.55 + n * 0.8)
            d2 = _unit2(tdir * 0.9 + n * 0.45)
            l1 = 0.05 + 0.04 * r.random()
            m_ = o + d1 * l1 * 0.5
            for q0, q1, w_ in ((o, m_, .0062), (m_, m_ + d2 * l1 * 0.6, .0048)):
                c_, l_, _ = _stroke(p, q0 - (q1 - q0) * .2, q1, w_, up)
                vd = np.maximum(vd, 0.85 * c_)
                vl = np.maximum(vl, 0.85 * l_)
    for at, lean, ln in ((0.35, 0.15, 0.16), (0.7, 0.35, 0.13)):
        o = key["S"] + (key["W"] - key["S"]) * at
        d1 = _unit2(-up + _unit2(key["W"] - key["S"]) * lean)
        q1 = o + d1 * ln
        c_, l_, _ = _stroke(p, o, q1, 0.007, up)
        vd = np.maximum(vd, 0.8 * c_)
        vl = np.maximum(vl, 0.8 * l_)
    col = mix(col, C["vein"], 0.85 * vd)
    col = mix(col, C["membrane_hi"] * 1.08, 0.45 * vl * (1 - vd))
    # the trailing edge: darker, thicker leather toward the hem, a thin dark rolled rim
    hem = np.full(u.shape, 9.0)
    for a, b in (("F1", "C12"), ("C12", "F2"), ("F2", "C23"), ("C23", "F3"), ("F3", "C3B"), ("C3B", "B")):
        d, _ = seg(a, b)
        hem = np.minimum(hem, d)
    hem = hem + 0.004 * (fbm(x / 4, y / 4, 59) - 0.5)
    col = col * (1 - 0.32 * smooth(0.075, 0.0, hem))[..., None]
    col = mix(col, C["vein"] * 0.9, 0.6 * smooth(0.012, 0.004, hem))
    # healed scar patches: paler, puckered, ragged-edged leather lying along the panel, a raised lit rim
    for (fa, fb, at, back), (la, lc), sd in ((("F1", "F2", .55, .38), (.07, .022), 66),
                                             (("F3", "B", .4, .42), (.06, .019), 67)):
        c0 = key[fa] + (key[fb] - key[fa]) * at
        c0 = c0 + (W - c0) * back
        ax = _unit2(c0 - W)
        rel = p - c0
        ra, rc = rel @ ax, rel @ np.array([-ax[1], ax[0]])
        warp = 0.35 * (fbm(x / 2.2, y / 2.2, sd, octaves=2) - 0.5)
        dd = np.sqrt((ra / la) ** 2 + (rc / lc) ** 2) + warp
        inside = smooth(1.0, 0.75, dd)
        pucker = 0.5 + 0.5 * np.sin(ra / la * 9 + 2.5 * warp)
        col = mix(col, C["scar"] * (0.8 + 0.25 * pucker)[..., None], 0.6 * inside)
        rim = smooth(0.18, 0.0, np.abs(dd - 1.0))
        col = mix(col, C["membrane_hi"] * 1.05, 0.4 * rim * ((rel @ up) > 0))       # raised rim, lit side only
    A = key["F2"] + (key["F3"] - key["F2"]) * 0.3
    A = A + (W - A) * 0.45
    B_ = A + _unit2(key["C23"] - W) * 0.12
    c_, l_, _ = _stroke(p, A, B_, 0.007, up)
    col = mix(col, C["scar"], 0.55 * c_)
    col = mix(col, C["membrane_hi"], 0.3 * l_ * (1 - c_))
    # a few deliberate notches bitten out of the trailing edge (a dark bite with a pale torn lip) and one
    # slit torn through near the hem
    for (a, b), at, size in ((("C12", "F2"), 0.42, .05), (("F3", "C3B"), 0.55, .045), (("C3B", "B"), 0.4, .04)):
        A0, B0 = key[a], key[b]
        c0 = A0 + (B0 - A0) * at
        tdir = (B0 - A0) / np.linalg.norm(B0 - A0)
        n = np.array([-tdir[1], tdir[0]])
        if np.dot(W - c0, n) < 0:
            n = -n
        rel = p - c0
        din = rel @ n + 0.006 * (fbm(x / 1.5, y / 1.5, 60) - 0.5)
        dt = np.abs(rel @ tdir)
        wid = size * 0.38 * np.clip(1 - din / size, 0, 1)
        inside = (din < size) & (din > -0.03)
        fr = 0.004 * (fbm(x / 1.2, y / 1.2, 61) - 0.5)
        col = mix(col, C["membrane_hi"] * 1.05, 0.55 * inside * smooth(wid + 0.011, wid, dt + fr))
        bite = (inside & (dt + fr < wid)).astype(float)
        col = mix(col, C["vein"] * 0.75 * (0.85 + 0.3 * smooth(0, size, din))[..., None], 0.9 * bite)
    A0, B0 = key["C12"], key["F1"]
    c0 = A0 + (B0 - A0) * 0.45 + (W - A0) * 0.1
    dvec = _unit2(W - c0)
    rel = p - c0
    al = rel @ dvec
    ac = rel @ np.array([-dvec[1], dvec[0]])
    tap = np.clip(1 - np.abs(al - 0.025) / 0.025, 0, 1)
    col = mix(col, C["membrane_hi"], 0.55 * (tap > 0) * smooth(0.011 * tap + 0.006, 0.005 * tap + 0.002, np.abs(ac)))
    col = mix(col, C["vein"] * 0.7, 0.9 * ((tap > 0) & (np.abs(ac) < 0.005 * tap + 0.001)).astype(float))
    # the finger bones and the arm: scaly ridges, lit on top, a dark seam where the membrane meets them
    lu = np.array([up[0], up[1]])
    for a, b, width in (("W", "F1", .02), ("W", "F2", .018), ("W", "F3", .017), ("S", "E", .03), ("E", "W", .026)):
        d, h = seg(a, b)
        A0, B0 = key[a], key[b]
        tdir = (B0 - A0) / np.linalg.norm(B0 - A0)
        n = np.array([-tdir[1], tdir[0]])
        if n @ up < 0:
            n = -n
        side = (p - A0) @ n                                     # + above the bone's axis
        w_ = width * (1 - 0.35 * h * (a == "W"))                  # fingers taper toward the tip
        ridge = smooth(w_, w_ * 0.55, d)
        sc, cr, rr, dome = _scaled_hide(x, y, 81, lu[0], -lu[1], (2.6, 2.6), worn=0.1, crev=0.25, dark=0.4)
        bone = sc * (1.0 + 0.32 * np.clip(side / np.maximum(w_, 1e-6), -1, 1))[..., None]
        col = mix(col, bone, ridge)
        col = mix(col, C["crevice"] * 1.2, 0.55 * smooth(w_ * 1.5, w_, d) * (1 - ridge))
    return col


def _unit2(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


PAINTERS = {
    ("Dragon_Body", "trunk"): paint_trunk, ("Dragon_Body", "leg"): paint_leg,
    ("Dragon_Body", "toe"): paint_claw,
    ("Dragon_Head", "skull"): paint_skull, ("Dragon_Head", "jaw"): paint_jaw,
    ("Dragon_Head", "spike"): paint_spike, ("Dragon_Head", "horn"): paint_horn,
    ("Dragon_Head", "brow"): paint_brow, ("Dragon_Head", "fang"): paint_fang,
    ("Dragon_Head", "nostril"): paint_nostril,
    ("Dragon_Wing", "membrane"): paint_membrane,
    ("Dragon_Wing", "toe"): paint_claw,
}


# ------------------------------------------------------------------------------------------- levels
# The game's skins span their whole value range (Salamander, Bile Demon, Dark Angel: crevices at about
# 20-30 luminance, painted highlights at 145-160); a final levels pass per skin, over the texels the mesh
# uses, puts the dragon's in the same range without changing its hues.
LEVELS = {"Dragon_Body": (32, 144), "Dragon_Head": (24, 152), "Dragon_Wing": (30, 132)}   # v2b: wings under the body
# v2d: the body skin's input range, pinned to what the approved v2c bake measured (its 5th / 95th luminance
# percentiles), so that repainting the tail's tip leaves every other texel of the skin exactly as it was
LEVELS_IN = {"Dragon_Body": (34.10599899291992, 117.61910209655761)}


def uv_mask(parts, texture, size=SIZE):
    from PIL import ImageDraw
    im = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(im)
    for p in parts:
        if p.texture != texture:
            continue
        uv = atlas_uv(p) * size
        for a, b, c in p.tris:
            dr.polygon([tuple(uv[a]), tuple(uv[b]), tuple(uv[c])], fill=255)
    return np.asarray(im) > 0


# v2b: chroma lift per skin (the dragon was the least saturated creature beside the Salamander, Bile Demon
# and Troll); stronger in the lit tops than in the crevices, so the highlights warm up and the darks stay
# dark: overall saturation then sits with the Bile Demon's and the Troll's, still olive, not cartoony
SATURATION = {"Dragon_Body": 1.24, "Dragon_Head": 1.2, "Dragon_Wing": 1.08}


def levels(img, mask, lo_out, hi_out, p_lo=5, p_hi=95, saturation=1.0, pin=None):
    """`pin`: (lo, hi) input range to use instead of the masked luminance percentiles."""
    a = np.asarray(img, np.float32)
    w = np.array([0.299, 0.587, 0.114], np.float32)
    lum = a @ w
    lo, hi = np.percentile(lum[mask], [p_lo, p_hi]) if pin is None else pin
    k = (hi_out - lo_out) / max(hi - lo, 1e-3)
    scale = (lo_out + (lum - lo) * k) / np.maximum(lum, 1e-3)
    out = np.clip(a * np.clip(scale, 0, 3)[..., None], 0, 255)
    if saturation != 1.0:
        l2 = out @ w
        s_ = 1 + (saturation - 1) * smooth(18.0, 80.0, l2)
        out = np.clip(l2[..., None] + (out - l2[..., None]) * s_[..., None], 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def bake_skins(parts, occlusion=None):
    """The Dragon's three skins: bake, then the levels pass."""
    tex = bake(parts, occlusion=occlusion)
    return {name: levels(img, uv_mask(parts, name), *LEVELS[name], saturation=SATURATION.get(name, 1.0),
                         pin=LEVELS_IN.get(name)) for name, img in tex.items()}


# ------------------------------------------------------------------------------------------- bake

def _vertex_normals(P, T):
    n = np.zeros_like(P)
    f = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(n, T[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _radius(part, islands=("trunk", "leg")):
    """Per vertex: distance from its loft ring's centre (for scale sizes); 0.05 elsewhere."""
    P = np.array(part.points)
    out = np.full(len(P), 0.05)
    groups = {}
    for i, tg in enumerate(part.tag):
        if "ring" in tg and part.island[i] in islands:
            groups.setdefault((part.island[i], tg["ring"], tg.get("side", 0)), []).append(i)
    for idx in groups.values():
        c = P[idx].mean(0)
        r = np.linalg.norm(P[idx] - c, axis=1).mean()
        out[idx] = r
    return out


def vertex_occlusion(parts, posed=None, rays=48, reach=0.18, skip=("wings",)):
    """Per part, per vertex: the fraction of `rays` hemisphere rays (round its normal) that leave
    without hitting the body within `reach` (1 = open). The wings neither cast nor receive (they move)."""
    pts = posed or [np.array(p.points) for p in parts]
    casters = [(P, np.array(p.tris)) for p, P in zip(parts, pts) if p.name not in skip]
    V0 = np.concatenate([P[T[:, 0]] for P, T in casters])
    E1 = np.concatenate([P[T[:, 1]] - P[T[:, 0]] for P, T in casters])
    E2 = np.concatenate([P[T[:, 2]] - P[T[:, 0]] for P, T in casters])
    rng = np.random.default_rng(5)
    dirs = rng.normal(size=(rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = {}
    for p, P in zip(parts, pts):
        T = np.array(p.tris)
        N = _vertex_normals(P, T)
        if p.name in skip:
            out[p.name] = np.ones(len(P))
            continue
        res = np.ones(len(P))
        for i in range(len(P)):
            d = dirs * np.sign(dirs @ N[i])[:, None]
            o = P[i] + N[i] * 0.004
            pv = np.cross(d[:, None, :], E2[None])                 # (R, T, 3)
            det = np.einsum("tk,rtk->rt", E1, pv)
            ok = np.abs(det) > 1e-9
            inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
            tv = o - V0
            uu = np.einsum("tk,rtk->rt", tv, pv) * inv
            qv = np.cross(tv, E1)
            vv = np.einsum("rk,tk->rt", d, qv) * inv
            tt = np.einsum("tk,tk->t", E2, qv)[None] * inv
            hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 1e-4) & (tt < reach)
            cosw = np.abs(d @ N[i])
            res[i] = 1 - (hit.any(1) * cosw).sum() / cosw.sum()
        out[p.name] = res
    return out


def bake(parts, size=SIZE, ss=SS, occlusion=None, atlas=None, painters=None, local=None,
         ring_islands=("trunk", "leg")):
    """-> {texture name: PIL RGB image (size x size)}. `occlusion`: vertex_occlusion(parts).
    `atlas`, `painters`, `local` (island_local), `ring_islands` (lofts whose ring radius sizes the
    scales): another creature's; default the dragon's."""
    atlas = ATLAS if atlas is None else atlas
    painters = PAINTERS if painters is None else painters
    local = island_local if local is None else local
    big = size * ss
    layers = {tex: (np.zeros((big, big, 3), np.float32), np.zeros((big, big), bool)) for tex in atlas}
    for part in parts:
        P = np.array(part.points)
        T = np.array(part.tris)
        N = _vertex_normals(P, T)
        loc = local(part)
        tex_uv = atlas_uv(part, atlas, local) * big
        rad = _radius(part, ring_islands)
        along = np.array([tg.get("along", 0.0) for tg in part.tag])
        ao = occlusion.get(part.name) if occlusion else None
        col, mask = layers[part.texture]
        for isl in sorted(set(part.island)):
            painter = painters[(part.texture, isl)]
            tris = [t for t in T if part.island[t[0]] == isl]
            samples = {k: [] for k in ("u", "v", "pos", "nrm", "radius", "along", "ao")}
            where = []
            for a, b, c in tris:
                x, y = tex_uv[[a, b, c], 0], tex_uv[[a, b, c], 1]
                area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
                if abs(area) < 1e-9:
                    continue
                x0, x1 = int(max(np.floor(x.min()) - 1, 0)), int(min(np.ceil(x.max()) + 1, big))
                y0, y1 = int(max(np.floor(y.min()) - 1, 0)), int(min(np.ceil(y.max()) + 1, big))
                gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
                w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
                w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
                w2 = 1 - w0 - w1
                ins = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
                if not ins.any():
                    continue
                W = np.stack([w0[ins], w1[ins], w2[ins]], 1)
                samples["u"].append(W @ loc[[a, b, c], 0])
                samples["v"].append(W @ loc[[a, b, c], 1])
                samples["pos"].append(W @ P[[a, b, c]])
                n = W @ N[[a, b, c]]
                samples["nrm"].append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9))
                samples["radius"].append(W @ rad[[a, b, c]])
                samples["along"].append(W @ along[[a, b, c]])
                samples["ao"].append(W @ (ao[[a, b, c]] if ao is not None else np.ones(3)))
                where.append((gy[ins].astype(int), gx[ins].astype(int)))
            if not where:
                continue
            t = {k: np.concatenate(v) for k, v in samples.items()}
            rgb = painter(t)
            if "ao" in t:
                rgb = rgb * (0.45 + 0.55 * t["ao"])[..., None]
            ys = np.concatenate([w[0] for w in where])
            xs = np.concatenate([w[1] for w in where])
            col[ys, xs] = rgb
            mask[ys, xs] = True
    out = {}
    for tex, (col, mask) in layers.items():
        col = _bleed(col, mask, 24)
        img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
        out[tex] = img
    return out


def _bleed(col, mask, steps):
    """Grow coloured texels into the empty ones (average of coloured neighbours), `steps` times."""
    col = col.copy()
    m = mask.copy()
    for _ in range(steps):
        acc = np.zeros_like(col)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            sm = np.roll(np.roll(m, dy, 0), dx, 1)
            sc = np.roll(np.roll(col, dy, 0), dx, 1)
            acc += sc * sm[..., None]
            cnt += sm
        grow = (~m) & (cnt > 0)
        if not grow.any():
            break
        col[grow] = acc[grow] / cnt[grow][:, None]
        m = m | grow
    col[~m] = col[m].mean(0) if m.any() else 0
    return col
