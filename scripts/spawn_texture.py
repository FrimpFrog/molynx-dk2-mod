"""The Demon Spawn's two 128 x 128 skins (DemonSpawn_Body, DemonSpawn_Head), painted by baking with the
Dragon's baker (scripts.dragon_texture.bake, on this ATLAS and these painters).

Painted like the game's own skins (shaded renders, not flat colour): olive-green overlapping scales with a
light top edge and a dark rim, mottled blotches and dark grime low on the flanks and legs, a darker ridge
along the spine, ochre belly plates from the throat down the underside of the tail, ivory claws and
spikes, painted highlights from above (a soft specular sheen on the hump, the head and the plates) and
darkening in the folds (vertex occlusion). The DK1 spawn's glowing parts do not take the light: small
orange-red eyes in dark sockets under a heavy brow, and the gaping maw, whose palate, tongue bed and throat
glow from yellow-white at the back to red at the lips, edged by a row of pale teeth.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_texture import SIZE, _hash2, bake as _bake, fbm, mix, scales, smooth, vertex_occlusion, \
    atlas_uv as _atlas_uv

BODY, HEAD = "DemonSpawn_Body", "DemonSpawn_Head"

# texture -> island -> (x0, y0, x1, y1) in pixels of the 128 texture, transpose (v runs along x)
ATLAS = {
    BODY: {
        "trunk": ((0, 0, 128, 84), True),
        "leg": ((0, 86, 96, 128), True),
        "claw": ((98, 86, 128, 128), False),
    },
    HEAD: {
        "skull": ((0, 0, 128, 70), True),
        "jaw": ((0, 72, 90, 108), True),
        "spike": ((92, 72, 128, 108), False),
        "fang": ((0, 110, 30, 128), False),
        "tongue": ((32, 110, 90, 128), True),
        "throat": ((92, 110, 128, 128), False),
    },
}


def island_local(part):
    return np.clip(np.array(part.uv, float), 0, 1)


def atlas_uv(part):
    return _atlas_uv(part, ATLAS, island_local)


# ------------------------------------------------------------------------------------------- palette

C = {k: np.array(v, float) for k, v in {
    "scale_hi": (184, 170, 78), "scale": (118, 110, 38), "scale_dark": (50, 46, 14), "back": (84, 78, 26),
    "moss": (96, 102, 34), "grime": (58, 46, 22),
    "belly_hi": (204, 178, 100), "belly": (156, 128, 62), "belly_groove": (72, 56, 22),
    "claw": (222, 206, 160), "claw_tip": (248, 240, 216), "claw_base": (100, 88, 46),
    "spike": (204, 184, 124), "spike_tip": (255, 248, 222),
    "glow_core": (255, 236, 150), "glow": (255, 150, 40), "maw": (226, 70, 22), "maw_edge": (120, 22, 10),
    "tongue": (176, 36, 34), "tongue_hi": (236, 112, 96), "tooth": (240, 232, 200),
    "eye": (255, 84, 16), "eye_core": (255, 222, 96), "eye_rim": (26, 18, 4),
    "sheen": (238, 236, 170),
}.items()}

HALF = np.array([.25, 1.0, .45]) / np.linalg.norm([.25, 1.0, .45])


def lit(col, nrm, sheen=0.35, power=10):
    """The painted light: soft from above, a little from the front, and a specular sheen."""
    up = np.clip(nrm[:, 1], -1, 1)
    front = np.clip(nrm[:, 2], -1, 1)
    k = 0.66 + 0.42 * up + 0.08 * front
    spec = np.clip(nrm @ HALF, 0, 1) ** power * sheen
    return col * k[..., None] + (C["sheen"] - col * 0.5) * spec[..., None]


def glow_comp(t, keep=0.35):
    """Undo most of the bake's occlusion darkening (0.45 + 0.55 ao) on glowing texels."""
    f = 0.45 + 0.55 * t["ao"]
    return (1 / f) ** (1 - keep)


def _skin(around, along, seed, size=0.016):
    """Olive scales with mottling: -> RGB."""
    # warp the scale grid a little so it doesn't read as a quilt
    wx = around + size * 0.35 * (fbm(around * 11, along * 11, seed + 9) - 0.5)
    wy = along + size * 0.35 * (fbm(around * 11 + 5, along * 11, seed + 10) - 0.5)
    rim, rnd, fy = scales(wx / size, wy / (size * .9), seed)
    broad = fbm(around * 4, along * 4, seed + 1)                   # broad painted light and shade
    col = mix(C["scale"], C["scale_hi"], 0.15 + 0.25 * rnd * (1 - fy) + 0.45 * broad)
    col = mix(col, C["moss"], 0.35 * smooth(0.45, 0.75, fbm(around * 5, along * 5, seed + 2)))
    col = mix(col, C["scale_dark"], 0.3 * rim)
    col = mix(col, C["scale_hi"] * 1.05, 0.18 * (1 - fy) * (1 - rim))    # each scale's lit top edge
    blot = smooth(0.58, 0.8, fbm(around * 7 + 3, along * 7, seed + 3))
    col = mix(col, C["scale_dark"] * 1.15, 0.38 * blot)
    col *= (0.88 + 0.2 * fbm(around * 18, along * 18, seed + 4))[..., None]
    return col


# ------------------------------------------------------------------------------------------- painters

def _trunk_length():
    from scripts.spawn_model import TRUNK_PATH, TRUNK_TIP
    pts = np.vstack([TRUNK_TIP, TRUNK_PATH])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return seg[-1], seg[1 + 3] / seg[-1]         # total, the pelvis ring's fraction


def paint_trunk(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    length, pelvis = _trunk_length()
    s = v * length
    around = u * np.pi * t["radius"]
    col = _skin(around, s, 1, size=0.034)
    # the spine: a darker ridge with lighter knobbly scutes between the spikes
    ridge = smooth(0.16, 0.0, u)
    col = mix(col, C["back"], 0.55 * ridge)
    k = s / 0.03
    knob = np.exp(-(((k - np.floor(k)) - 0.5) / 0.22) ** 2) * smooth(0.08, 0.0, u)
    col = mix(col, C["scale_hi"] * 0.9, 0.5 * knob)
    # belly plates on the ventral faces (narrower down the tail)
    grid = np.linspace(0, length, 400)
    width = np.interp(grid, [0, pelvis * length, length], [0.014, 0.024, 0.026])
    phase = np.r_[0, np.cumsum(np.diff(grid) / width[1:])]
    kk = np.interp(s, grid, phase)
    f = kk - np.floor(kk)
    band = mix(C["belly_hi"], C["belly"], smooth(0.1, 0.85, f))
    band = mix(band, C["belly_groove"], smooth(0.78, 0.97, f))
    band *= (0.88 + 0.2 * fbm(around * 30, s * 6, 5))[..., None]
    edge = smooth(0.66, 0.76, u)
    col = mix(col, band, edge)
    col = mix(col, C["scale_dark"], 0.6 * np.exp(-((u - 0.67) / 0.03) ** 2))       # the seam
    # grime low on the flanks and the belly, worse on the tail where it drags on the floor
    dirt = smooth(0.45, 0.85, u) * (0.45 + 0.55 * smooth(0.5, 0.15, v))
    dirt *= smooth(0.35, 0.7, fbm(around * 12, s * 12, 7))
    col = mix(col, C["grime"], 0.55 * dirt)
    return lit(col, nrm, sheen=0.4)


def paint_leg(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    around = u * np.pi * t["radius"]
    s = v * 0.32
    col = _skin(around, s, 7, size=0.024)
    # knuckle scutes down the front of the forearm / shin
    k = s / 0.02
    f = k - np.floor(k)
    scute = mix(C["belly"] * 0.9, C["belly_groove"], smooth(0.7, 0.95, f))
    front = smooth(0.35, 0.12, np.abs(u - 0.5) * 2) * smooth(0.45, 0.6, v)
    col = mix(col, scute, 0.6 * front)
    # dirty feet, darker inner side
    col = mix(col, C["grime"], 0.65 * smooth(0.6, 1.0, v) * smooth(0.3, 0.6, fbm(around * 20, s * 20, 8)))
    col = mix(col, C["scale_dark"], 0.3 * smooth(0.6, 1.0, u))
    return lit(col, nrm, sheen=0.3)


def paint_claw(t):
    along = t["along"]
    col = mix(C["grime"], C["claw_base"], smooth(0.0, 0.2, along))
    col = mix(col, C["claw"], smooth(0.2, 0.55, along))
    col = mix(col, C["claw_tip"], smooth(0.8, 1.0, along))
    k = along * 6
    col *= (0.88 + 0.12 * (k - np.floor(k)))[..., None]          # growth rings
    return lit(col, t["nrm"], sheen=0.5, power=6)


def _skull_v():
    from scripts.spawn_model import HEAD_RINGS, SKULL_BACK, SNOUT_TIP
    length = np.linalg.norm(SNOUT_TIP - SKULL_BACK)
    return [np.linalg.norm(np.asarray(r[0]) - SKULL_BACK) / length for r in HEAD_RINGS], length


def paint_skull(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    (v0, v1, v2), length = _skull_v()
    s = v * length
    around = u * 0.3
    col = _skin(around, s, 11, size=0.022)
    col = mix(col, C["back"] * 0.9, 0.45 * smooth(0.2, 0.0, u))             # the dark crown
    # lighter cheeks toward the lips
    col = mix(col, C["belly_hi"] * 0.8, 0.3 * smooth(0.45, 0.64, u) * smooth(0.95, 0.6, u))
    # wrinkles across the snout
    k = s / 0.022
    f = k - np.floor(k)
    wr = smooth(v1, v2, v) * smooth(0.5, 0.2, u) * np.exp(-((f - 0.5) / 0.15) ** 2)
    col = mix(col, C["scale_dark"] * 0.7, 0.5 * wr)
    # nostrils: two dark slits on top near the tip
    dn = ((u - 0.12) / 0.045) ** 2 + ((v - (v2 + 0.6 * (1 - v2))) / 0.05) ** 2
    col = mix(col, C["eye_rim"], 0.9 * smooth(1.0, 0.4, dn))
    # the lip: a dark line, then the row of teeth, then the glowing palate
    col = mix(col, C["scale_dark"] * 0.6, 0.8 * np.exp(-((u - 0.64) / 0.016) ** 2))
    k = s / 0.02
    tw = np.abs((k - np.floor(k)) - 0.5) * 2
    tl = 0.075 * (0.7 + 0.3 * _hash2(np.floor(k).astype(np.int64), np.zeros_like(k, np.int64), 31))
    tooth = (u > 0.665) & (u < 0.665 + tl) & ((0.665 + tl - u) / tl > tw * 0.95) & (v > 0.25)
    inside = smooth(0.66, 0.70, u)
    glow = mix(C["maw"], C["glow"], smooth(0.75, 1.0, u) * smooth(0.95, 0.3, v))
    glow = mix(glow, C["glow_core"], 0.6 * smooth(0.85, 1.0, u) * smooth(0.5, 0.1, v))
    glow = mix(glow, C["maw_edge"], smooth(0.75, 0.67, u))
    col = lit(col, nrm, sheen=0.45)
    col = mix(col, glow * glow_comp(t)[..., None], inside)
    col = np.where(tooth[..., None], C["tooth"] * (0.75 + 0.25 * (1 - tw))[..., None], col)
    # a heavy brow over a small slanted eye on the top of the head (DK1: orange-red, glowing)
    ve = v - v1
    brow = np.exp(-(((u - (0.17 + 0.25 * ve)) / 0.05) ** 2) - (ve / 0.12) ** 2)
    col = mix(col, C["scale_dark"] * 0.5, 0.7 * brow)
    du = (u - 0.24 - 0.35 * ve) / 0.05
    dv = ve / 0.085
    e = du ** 2 + dv ** 2
    col = mix(col, C["eye_rim"], 0.9 * smooth(2.6, 1.0, e))
    eye = mix(C["eye_core"], C["eye"], smooth(0.0, 0.8, e))
    eye = mix(eye, C["maw_edge"], smooth(0.7, 1.0, e))
    col = np.where((e < 1)[..., None], eye * glow_comp(t, 0.1)[..., None], col)
    return col


def paint_jaw(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    # u: 0 tongue bed (inside), 0.5 lips, 1 keel (under the chin)
    s = v * 0.24
    rim, rnd, fy = scales(u / 0.09, s / 0.016, 17)
    side = _skin(u * 0.15, s, 17, size=0.02)
    k = s / 0.018
    f = k - np.floor(k)
    throat = mix(C["belly_hi"], C["belly"], smooth(0.2, 0.9, f))
    throat = mix(throat, C["belly_groove"], 0.7 * smooth(0.85, 0.98, f))
    col = mix(side, throat, smooth(0.66, 0.8, u))
    col = lit(col, nrm, sheen=0.3)
    inside = smooth(0.46, 0.40, u)
    glow = mix(C["glow"], C["maw"], smooth(0.1, 0.4, u))
    glow = mix(glow, C["glow_core"], 0.5 * smooth(0.25, 0.0, u) * smooth(0.8, 0.2, v))
    glow = mix(glow, C["maw_edge"], smooth(0.36, 0.44, u))
    col = mix(col, glow * glow_comp(t)[..., None], inside)
    tw = np.abs((s / 0.02 - np.floor(s / 0.02)) - 0.5) * 2
    tooth = (u > 0.36) & (u < 0.46) & ((u - 0.36) / 0.1 > tw * 0.95) & (v > 0.2)
    return np.where(tooth[..., None], C["tooth"], col)


def paint_throat(t):
    u, v = t["u"], t["v"]
    r = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2
    return mix(C["glow_core"], C["glow"], smooth(0.0, 0.9, r)) * glow_comp(t, 0.2)[..., None]


def paint_tongue(t):
    u, v = t["u"], t["v"]
    col = mix(C["tongue_hi"], C["tongue"], smooth(0.0, 0.6, u))
    col = mix(col, C["maw_edge"], 0.6 * np.exp(-(u / 0.08) ** 2))        # the groove down the middle
    col = mix(col, C["maw_edge"] * 0.8, smooth(0.6, 1.0, u))             # the underside
    col *= (0.9 + 0.15 * fbm(u * 10, v * 20, 41))[..., None]
    return lit(col, t["nrm"], sheen=0.6, power=14)


def paint_spike(t):
    along = t["along"]
    col = mix(C["back"], C["spike"], smooth(0.0, 0.1, along))
    col = mix(col, C["spike_tip"], smooth(0.25, 0.9, along))
    col *= (0.88 + 0.12 * np.sin(t["u"] * 26 + along * 4))[..., None]       # streaks
    return col          # double-sided fins: painted, not lit


def paint_fang(t):
    along = t["along"]
    col = mix(C["claw"], C["claw_tip"], along)
    return col


PAINTERS = {
    (BODY, "trunk"): paint_trunk, (BODY, "leg"): paint_leg, (BODY, "claw"): paint_claw,
    (HEAD, "skull"): paint_skull, (HEAD, "jaw"): paint_jaw, (HEAD, "spike"): paint_spike,
    (HEAD, "fang"): paint_fang, (HEAD, "tongue"): paint_tongue, (HEAD, "throat"): paint_throat,
}


def bake(parts, occlusion=None, size=SIZE):
    """-> {texture name: 128 x 128 RGB image}."""
    if occlusion is None:
        occlusion = vertex_occlusion(parts, skip=("spikes", "tongue"), reach=0.1)
    return _bake(parts, size=size, occlusion=occlusion, atlas=ATLAS, painters=PAINTERS, local=island_local,
                 ring_islands=("trunk", "leg"))
