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
    },
    "Dragon_Wing": {
        "membrane": ((0, 0, 128, 100), False),
        "arm": ((0, 102, 96, 128), True),
        "toe": ((98, 102, 128, 128), False),
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
    from scripts.dragon_model import TRUNK_PATH, TRUNK_TIP, TRUNK_TOP
    pts = np.vstack([TRUNK_TIP, TRUNK_PATH, TRUNK_TOP])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return seg[1 + 4] / seg[-1], seg[-1]          # ring 4 = the pelvis


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


def atlas_uv(part):
    """Texture-space (0..1) UVs of a part's vertices."""
    loc = island_local(part)
    out = np.zeros_like(loc)
    for i, (isl, (u, v)) in enumerate(zip(part.island, loc)):
        (x0, y0, x1, y1), tr = ATLAS[part.texture][isl]
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

C = {k: np.array(v, float) for k, v in {
    "scale_hi": (162, 150, 72), "scale": (114, 106, 42), "scale_dark": (50, 44, 15),
    "back": (82, 74, 28),
    "belly_hi": (222, 170, 102), "belly": (182, 122, 64), "belly_groove": (86, 48, 20),
    "claw": (226, 212, 168), "claw_tip": (250, 244, 220), "claw_base": (120, 104, 60),
    "mouth": (128, 24, 18), "mouth_hi": (186, 62, 44), "tooth": (244, 236, 204),
    "eye": (230, 60, 10), "eye_core": (255, 214, 70), "eye_rim": (40, 26, 6), "pupil": (20, 10, 0),
    "membrane": (108, 92, 44), "membrane_hi": (146, 126, 62), "vein": (56, 44, 18),
    "horn": (214, 196, 144), "horn_base": (96, 84, 40),
    "spike": (110, 98, 40), "spike_tip": (226, 204, 150),
}.items()}


def mix(a, b, t):
    t = np.asarray(t, float)[..., None]
    return a * (1 - t) + b * t


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------- painters
# each gets a dict of per-texel arrays: u, v (island-local), pos (bind, (n,3)), nrm, and returns RGB

def _light(nrm, pos=None):
    up = np.clip(nrm[:, 1], -1, 1)
    front = np.clip(nrm[:, 2], -1, 1)
    return 0.80 + 0.28 * up + 0.06 * front


def paint_trunk(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    split, length = trunk_split()
    # arc length in model units along the trunk
    s = np.where(v < TAIL_SHARE, v / TAIL_SHARE * split, split + (v - TAIL_SHARE) / (1 - TAIL_SHARE) * (1 - split)) * length
    r = t["radius"]
    around = u * np.pi * r                                  # model units from the dorsal line
    rim, rnd, fy = scales(around / 0.017, s / 0.015, 1)
    green = mix(C["scale"], C["scale_hi"], 0.3 + 0.35 * rnd * (1 - fy) + 0.25 * fbm(around * 9, s * 9, 2))
    green = mix(green, C["scale_dark"], 0.38 * rim)
    green = mix(green, C["back"], 0.55 * smooth(0.3, 0.0, u))          # darker along the spine
    green *= (0.86 + 0.24 * fbm(around * 14, s * 14, 3))[..., None]
    # belly plates on the ventral faces: one plate every PLATE units (smaller on the tail and neck)
    # plates every ~0.022 (narrower on the tail tip and up the neck); the phase is the integral of 1/width
    grid = np.linspace(0, length, 400)
    width = np.interp(grid, [0, split * length * 0.5, split * length, (split + 0.15) * length, length],
                      [0.016, 0.02, 0.026, 0.028, 0.02])
    phase = np.r_[0, np.cumsum(np.diff(grid) / width[1:])]
    k = np.interp(s, grid, phase)
    f = k - np.floor(k)
    band = mix(C["belly_hi"], C["belly"], smooth(0.1, 0.85, f))
    band = mix(band, C["belly_groove"], smooth(0.78, 0.97, f))
    band = mix(band, C["belly_groove"], 0.5 * smooth(0.06, 0.0, f))
    band *= (0.92 + 0.16 * fbm(around * 30, s * 6, 5))[..., None]
    edge = smooth(0.58, 0.68, u)
    col = mix(green, band, edge)
    # a dark seam where the plates meet the scales
    col = mix(col, C["scale_dark"], 0.55 * np.exp(-((u - 0.6) / 0.025) ** 2))
    return col * _light(nrm)[..., None]


def paint_leg(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    r = t["radius"]
    around = u * np.pi * r
    s = v * 0.45
    rim, rnd, fy = scales(around / 0.016, s / 0.014, 7)
    col = mix(C["scale"], C["scale_hi"], 0.25 + 0.35 * rnd * (1 - fy) + 0.25 * fbm(around * 9, s * 9, 6))
    col = mix(col, C["scale_dark"], 0.38 * rim)
    # scutes (bands) on the front of the shin and foot
    k = s / 0.022
    f = k - np.floor(k)
    scute = mix(C["belly"] * 0.85, C["belly_groove"], smooth(0.7, 0.95, f))
    front = smooth(0.32, 0.12, u) * smooth(0.55, 0.7, v)
    col = mix(col, scute, 0.75 * front)
    # darker inside of the thigh
    col = mix(col, C["scale_dark"], 0.35 * smooth(0.6, 1.0, u) * smooth(0.5, 0.0, v))
    col *= (0.9 + 0.2 * fbm(around * 20, s * 20, 9))[..., None]
    return col * _light(nrm)[..., None]


def paint_claw(t):
    along = t["along"]
    col = mix(C["scale_dark"], C["claw_base"], smooth(0.0, 0.25, along))
    col = mix(col, C["claw"], smooth(0.25, 0.55, along))
    col = mix(col, C["claw_tip"], smooth(0.75, 1.0, along))
    col = mix(col, col * 0.7, 0.5 * smooth(0.5, 1.0, t["u"]))
    return col * _light(t["nrm"])[..., None]


EYE_V = 0.36        # the eyes: on the brow ring (ring 1), on the upper side of the head
NOSTRIL_V = 0.93


def paint_skull(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    length = 0.47
    s = v * length
    around = u * 0.24
    rim, rnd, fy = scales(around / 0.012, s / 0.012, 11)
    col = mix(C["scale"], C["scale_hi"], 0.3 + 0.35 * rnd + 0.2 * fbm(u * 8, v * 12, 12))
    col = mix(col, C["scale_dark"], 0.32 * rim)
    col = mix(col, C["back"] * 0.85, 0.45 * smooth(0.22, 0.0, u))         # dark top of the head
    # lighter, yellower cheeks and lips
    col = mix(col, C["belly_hi"] * 0.85, 0.35 * smooth(0.4, 0.62, u) * smooth(0.0, 0.2, v))
    col *= (0.9 + 0.18 * fbm(u * 12, v * 30, 13))[..., None]
    # wrinkles across the snout, deepening toward the nose
    k = s / 0.026
    f = k - np.floor(k)
    wr = smooth(0.5, 0.75, v) * smooth(0.62, 0.3, u) * np.exp(-((f - 0.5) / 0.14) ** 2)
    col = mix(col, C["scale_dark"] * 0.7, 0.55 * wr)
    # mouth: the palate (u > 2/3) is dark red, with a row of big jagged teeth along the lip line
    mouth = smooth(0.66, 0.70, u)
    col = mix(col, mix(C["mouth"] * 0.8, C["mouth_hi"], smooth(1.0, 0.75, u)), mouth)
    k = (s / 0.024)
    tooth_w = np.abs((k - np.floor(k)) - 0.5) * 2               # 0 at a tooth's centre
    tl = 0.12 * (0.75 + 0.25 * _hash2(np.floor(k).astype(np.int64), np.zeros_like(k, np.int64), 31))
    tooth = (u > 0.69 - tl) & (u < 0.70) & ((0.70 - u) / tl > tooth_w * 0.95) & (v > 0.22) & (v < 0.95)
    col = np.where(tooth[..., None], C["tooth"] * (0.8 + 0.2 * (1 - tooth_w))[..., None], col)
    col = mix(col, C["mouth"] * 0.45, 0.7 * np.exp(-((u - 0.575) / 0.012) ** 2) * (v > 0.2))   # gum line
    # a heavy dark brow, low over the front of the eye
    v_eye = (v - EYE_V)
    brow = np.exp(-(((u - (0.225 + 0.35 * v_eye)) / 0.045) ** 2) - ((v_eye / 0.12) ** 2))
    col = mix(col, C["scale_dark"] * 0.55, 0.85 * brow)
    # the eye: a slanted slit of red-orange glow, its front corner lower, in a dark socket
    du = (u - 0.305 - 0.4 * v_eye) / 0.042
    dv = v_eye / 0.085
    e = du ** 2 + dv ** 2
    eye = e < 1
    socket = (e >= 1) & (e < 2.4)
    col = mix(col, C["eye_rim"], np.where(socket, 0.85 * smooth(2.4, 1.0, e), 0.0))
    glow = mix(C["eye_core"], C["eye"], smooth(0.05, 0.9, e))
    glow = np.where((np.abs(dv) < 0.13)[..., None] & eye[..., None] & (np.abs(du) < 0.75)[..., None],
                    C["pupil"], glow)
    col = np.where(eye[..., None], glow / np.maximum(_light(nrm), 0.5)[..., None], col)   # eyes don't shade
    # nostrils: flared, near the snout tip
    n = ((u - 0.2) / 0.05) ** 2 + ((v - NOSTRIL_V) / 0.025) ** 2
    col = mix(col, C["scale_dark"] * 0.3, smooth(1.6, 0.8, n))
    return col * _light(nrm)[..., None]


def paint_jaw(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    # u: 0 tongue (inside), 0.5 lips, 1 keel (under the chin)
    inside = smooth(0.46, 0.40, u)
    s = v * 0.4
    k = s / 0.02
    f = k - np.floor(k)
    throat = mix(C["belly_hi"] * 0.95, C["belly"], smooth(0.2, 0.9, f))
    throat = mix(throat, C["belly_groove"], smooth(0.85, 0.98, f) * 0.7)
    rim, rnd, fy = scales(u / 0.1, s / 0.018, 17)
    side = mix(C["scale"], C["scale_hi"], 0.3 + 0.4 * rnd)
    side = mix(side, C["scale_dark"], 0.45 * rim)
    col = mix(side, throat, smooth(0.62, 0.78, u))
    tw = np.abs((s / 0.024 - np.floor(s / 0.024)) - 0.5) * 2
    tooth = (u > 0.36) & (u < 0.5) & ((u - 0.36) / 0.14 > tw * 0.95) & (v > 0.22) & (v < 0.95)
    col = np.where(tooth[..., None], C["tooth"], col)
    col = mix(col, mix(C["mouth"], C["mouth_hi"], smooth(0.0, 0.3, u)), inside)
    return col * _light(nrm)[..., None]


def paint_horn(t):
    along = t["along"]
    col = mix(C["horn_base"], C["horn"], smooth(0.05, 0.6, along))
    k = along * 7
    col = col * (0.85 + 0.15 * (k - np.floor(k)))[..., None]      # ridges
    col = mix(col, C["claw_tip"], smooth(0.85, 1.0, along))
    return col * _light(t["nrm"])[..., None]


def paint_spike(t):
    along = t["along"]
    col = mix(C["spike"], C["spike_tip"], smooth(0.2, 0.95, along))
    col = col * (0.9 + 0.1 * np.sin(t["u"] * 20))[..., None]
    return col


def paint_brow(t):
    col = mix(C["scale_dark"], C["back"], t["along"])
    return col * _light(t["nrm"])[..., None]


def paint_fang(t):
    return mix(C["claw"], C["claw_tip"], t["along"])


def paint_membrane(t):
    from scripts.dragon_model import WING_UV, WING_UV_EXTENT
    u, v = t["u"], t["v"]
    ext = np.maximum(WING_UV_EXTENT, 1e-9)
    key = {k: np.array(q) / ext for k, q in WING_UV.items()}
    p = np.stack([u, v], -1)

    def seg(a, b):
        a, b = key[a], key[b]
        ab = b - a
        h = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        return np.linalg.norm(p - (a + h[..., None] * ab), axis=-1), h

    col = mix(C["membrane"], C["membrane_hi"], 0.5 + 0.7 * (fbm(u * 6, v * 6, 21) - 0.5))
    # darker toward the arm (the leading edge), lighter and thinner toward the trailing hem
    dl, _ = seg("S", "E")
    dw, _ = seg("E", "W")
    col *= (0.78 + 0.3 * smooth(0.0, 0.35, np.minimum(dl, dw)))[..., None]
    # wrinkles running out from the wrist
    w = key["W"]
    ang = np.arctan2(v - w[1], u - w[0])
    col *= (0.94 + 0.06 * np.sin(ang * 24 + fbm(u * 8, v * 8, 23) * 6))[..., None]
    for a, b, width in (("W", "M1", .022), ("M1", "F1", .016), ("W", "M2", .02), ("M2", "F2", .014),
                        ("W", "M3", .02), ("M3", "F3", .014), ("S", "E", .03), ("E", "W", .026)):
        d, h = seg(a, b)
        col = mix(col, C["vein"], 0.9 * smooth(width, width * 0.35, d))
    for a, b in (("F1", "C12"), ("C12", "F2"), ("F2", "C23"), ("C23", "F3"), ("F3", "C3B"), ("C3B", "B")):
        d, _ = seg(a, b)
        col = mix(col, C["vein"] * 0.9, 0.7 * smooth(0.045, 0.0, d))   # darker hem
    return col


def paint_arm(t):
    u, v = t["u"], t["v"]
    rim, rnd, fy = scales(u / 0.25, v / 0.06, 29)
    col = mix(C["scale"], C["scale_hi"], 0.3 + 0.4 * rnd)
    col = mix(col, C["scale_dark"], 0.5 * rim)
    return col * _light(t["nrm"])[..., None]


PAINTERS = {
    ("Dragon_Body", "trunk"): paint_trunk, ("Dragon_Body", "leg"): paint_leg,
    ("Dragon_Body", "toe"): paint_claw,
    ("Dragon_Head", "skull"): paint_skull, ("Dragon_Head", "jaw"): paint_jaw,
    ("Dragon_Head", "spike"): paint_spike, ("Dragon_Head", "horn"): paint_horn,
    ("Dragon_Head", "brow"): paint_brow, ("Dragon_Head", "fang"): paint_fang,
    ("Dragon_Wing", "membrane"): paint_membrane, ("Dragon_Wing", "arm"): paint_arm,
    ("Dragon_Wing", "toe"): paint_claw,
}


# ------------------------------------------------------------------------------------------- bake

def _vertex_normals(P, T):
    n = np.zeros_like(P)
    f = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(n, T[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _radius(part):
    """Per vertex: distance from its loft ring's centre (for scale sizes); 0.05 elsewhere."""
    P = np.array(part.points)
    out = np.full(len(P), 0.05)
    groups = {}
    for i, tg in enumerate(part.tag):
        if "ring" in tg and part.island[i] in ("trunk", "leg"):
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


def bake(parts, size=SIZE, ss=SS, occlusion=None):
    """-> {texture name: PIL RGB image (size x size)}. `occlusion`: vertex_occlusion(parts)."""
    big = size * ss
    layers = {tex: (np.zeros((big, big, 3), np.float32), np.zeros((big, big), bool)) for tex in ATLAS}
    for part in parts:
        P = np.array(part.points)
        T = np.array(part.tris)
        N = _vertex_normals(P, T)
        loc = island_local(part)
        tex_uv = atlas_uv(part) * big
        rad = _radius(part)
        along = np.array([tg.get("along", 0.0) for tg in part.tag])
        ao = occlusion.get(part.name) if occlusion else None
        col, mask = layers[part.texture]
        for isl in sorted(set(part.island)):
            painter = PAINTERS[(part.texture, isl)]
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
