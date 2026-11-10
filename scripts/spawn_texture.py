"""The Demon Spawn's two 128 x 128 skins (DemonSpawn_Body, DemonSpawn_Head), painted by baking with the
Dragon's baker (scripts.dragon_texture.bake, on this ATLAS and these painters).

Painted like the game's own skins (shaded renders, not flat colour), as a real lizard's hide: an irregular
pebbled skin of rounded scales of mixed sizes (cellular noise) with dark cracks between them, each pebble
lit on its upper side and shadowed below; bigger raised tubercles and warts scattered over the back and the
flanks; rough dark patches, mottling and grime low on the flanks, the legs and the tail, with plenty of
contrast. Olive green above, ochre belly plates from the throat down the underside of the tail, a darker
ridge along the spine. Horn-yellow claws and tan spikes; big ivory fangs. The glowing parts do not take the
light: the bulging orange-red eyes (a yellow core round a black pupil) and the gaping maw, glowing from
yellow-white at the back to red at the lips.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_texture import SIZE, _hash2, atlas_uv as _atlas_uv, bake as _bake, fbm, mix, smooth, \
    vertex_occlusion

BODY, HEAD = "DemonSpawn_Body", "DemonSpawn_Head"

# texture -> island -> (x0, y0, x1, y1) in pixels of the 128 texture, transpose (v runs along x)
ATLAS = {
    BODY: {
        "trunk": ((0, 0, 128, 76), True),
        "leg": ((0, 78, 70, 128), True),
        "arm": ((72, 78, 112, 128), True),
        "claw": ((114, 78, 128, 128), False),
    },
    HEAD: {
        "skull": ((0, 0, 128, 64), True),
        "jaw": ((0, 66, 84, 100), True),
        "spike": ((86, 66, 106, 100), False),
        "eye": ((108, 66, 128, 100), False),
        "fang": ((0, 102, 30, 128), False),
        "tongue": ((32, 102, 90, 128), True),
        "throat": ((92, 102, 128, 128), False),
    },
}


def island_local(part):
    return np.clip(np.array(part.uv, float), 0, 1)


def atlas_uv(part):
    return _atlas_uv(part, ATLAS, island_local)


# ------------------------------------------------------------------------------------------- palette

C = {k: np.array(v, float) for k, v in {
    "scale_hi": (176, 166, 70), "scale": (112, 108, 36), "scale_dark": (44, 42, 12), "crack": (22, 20, 6),
    "back": (78, 74, 22), "moss": (92, 100, 30), "grime": (54, 42, 20), "wart": (150, 138, 62),
    "belly_hi": (210, 172, 92), "belly": (164, 124, 56), "belly_groove": (64, 44, 16),
    "claw": (206, 162, 64), "claw_tip": (238, 206, 120), "claw_base": (96, 70, 26),
    "spike": (190, 152, 80), "spike_tip": (236, 204, 132),
    "glow_core": (255, 236, 150), "glow": (255, 150, 40), "maw": (226, 70, 22), "maw_edge": (120, 22, 10),
    "tongue": (176, 36, 34), "tongue_hi": (236, 112, 96),
    "tooth": (240, 228, 186), "tooth_tip": (255, 250, 230), "tooth_base": (170, 140, 90),
    "eye": (250, 70, 14), "eye_ring": (255, 200, 60), "eye_rim": (26, 18, 4), "pupil": (14, 6, 0),
    "sheen": (238, 236, 170),
}.items()}

HALF = np.array([.25, 1.0, .45]) / np.linalg.norm([.25, 1.0, .45])


def lit(col, nrm, sheen=0.3, power=10):
    """The painted light: from above, a little from the front, and a specular sheen."""
    up = np.clip(nrm[:, 1], -1, 1)
    front = np.clip(nrm[:, 2], -1, 1)
    k = 0.64 + 0.44 * up + 0.08 * front
    spec = np.clip(nrm @ HALF, 0, 1) ** power * sheen
    return col * k[..., None] + (C["sheen"] - col * 0.5) * spec[..., None]


def glow_comp(t, keep=0.35):
    """Undo most of the bake's occlusion darkening (0.45 + 0.55 ao) on glowing texels."""
    return (1 / (0.45 + 0.55 * t["ao"])) ** (1 - keep)


# ------------------------------------------------------------------------------------------- the hide

def cells(x, y, seed=0, jitter=0.85):
    """Cellular noise on a unit grid: -> (F1, F2, the nearest cell's random 0..1, unit vector from its
    centre to the point (dx, dy))."""
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    best = np.full(x.shape, 9.0)
    second = np.full(x.shape, 9.0)
    rnd = np.zeros(x.shape)
    dx = np.zeros(x.shape)
    dy = np.zeros(x.shape)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            cx, cy = ix + ox, iy + oy
            px = cx + 0.5 + jitter * (_hash2(cx, cy, seed) - 0.5)
            py = cy + 0.5 + jitter * (_hash2(cx, cy, seed + 101) - 0.5)
            ddx, ddy = x - px, y - py
            d = np.sqrt(ddx * ddx + ddy * ddy)
            closer = d < best
            second = np.where(closer, best, np.minimum(second, d))
            best = np.where(closer, d, best)
            rnd = np.where(closer, _hash2(cx, cy, seed + 7), rnd)
            dx = np.where(closer, ddx, dx)
            dy = np.where(closer, ddy, dy)
    n = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-9)
    return best, second, rnd, dx / n, dy / n


def _pebbles(a, b, size, seed, light=(-.45, -.9)):
    """Raised pebbles of a lizard's hide: -> (shade -1..1, crack 0..1, per-pebble random). `a` across,
    `b` along (model units); light: the direction (in a, b) the pebbles are lit from."""
    f1, f2, rnd, ux, uy = cells(a / size, b / size, seed)
    crack = smooth(0.16, 0.0, f2 - f1)                          # dark cracks between the pebbles
    dome = smooth(0.75, 0.05, f1)                               # each pebble a low dome
    lx, ly = light
    slope = (ux * lx + uy * ly) * smooth(0.05, 0.6, f1)         # lit on one side, shadowed on the other
    return (0.55 * slope + 0.35 * dome - 0.2), crack, rnd


def hide(a, b, seed, size=0.016, warts=0.0, rough=0.5):
    """The lizard hide: pebbles of two sizes, warts, mottling and rough patches. -> RGB."""
    shade, crack, rnd = _pebbles(a, b, size, seed)
    shade2, crack2, rnd2 = _pebbles(a, b, size * 2.3, seed + 17)          # larger tubercles among them
    big = smooth(0.62, 0.8, fbm(a * 8, b * 8, seed + 3))                  # where the big ones are
    broad = fbm(a * 3.5, b * 3.5, seed + 1)
    col = mix(C["scale"], C["scale_hi"], 0.15 + 0.4 * broad + 0.25 * rnd)
    col = mix(col, C["moss"], 0.4 * smooth(0.45, 0.75, fbm(a * 5, b * 5, seed + 2)))
    s = shade * (1 - big) + shade2 * big
    cr = crack * (1 - big) + crack2 * big
    col = col * (1 + 0.55 * s)[..., None]
    col = mix(col, C["crack"], 0.8 * cr)
    if np.any(warts):                                                       # scattered warts
        f1, _, wr, ux, uy = cells(a / (size * 4.5), b / (size * 4.5), seed + 31, jitter=0.6)
        on = (wr > 1 - warts) * smooth(0.42, 0.18, f1)
        wcol = mix(C["wart"], C["scale_hi"] * 1.1, smooth(0.0, 0.6, -(ux * -.45 + uy * -.9)))
        wcol = wcol * (1 + 0.5 * (-(ux * .45 + uy * .9)) * smooth(0.05, 0.4, f1))[..., None]
        col = mix(col, wcol, on)
        col = mix(col, C["crack"], 0.7 * (wr > 1 - warts) * np.exp(-((f1 - 0.44) / 0.05) ** 2))
    patch = smooth(0.55, 0.78, fbm(a * 7 + 3, b * 7, seed + 4)) * rough   # rough dark patches
    col = mix(col, C["scale_dark"], 0.55 * patch)
    col *= (0.82 + 0.3 * fbm(a * 22, b * 22, seed + 5))[..., None]
    return col


# ------------------------------------------------------------------------------------------- painters

def _trunk_length():
    from scripts.spawn_model import PELVIS_RING, TRUNK_PATH, TRUNK_TIP
    pts = np.vstack([TRUNK_TIP, TRUNK_PATH])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return seg[-1], seg[1 + PELVIS_RING] / seg[-1]         # total, the pelvis ring's fraction


def paint_trunk(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    length, pelvis = _trunk_length()
    s = v * length
    around = u * np.pi * t["radius"]
    col = hide(around, s, 1, size=0.017, warts=0.3 * smooth(0.6, 0.2, u) + 0.12)
    ridge = smooth(0.18, 0.0, u)                                             # the spine
    col = mix(col, C["back"] * (0.8 + 0.4 * fbm(s * 40, u, 9))[..., None], 0.5 * ridge)
    # belly plates on the ventral faces (narrower down the tail), cracked and dirty
    grid = np.linspace(0, length, 400)
    width = np.interp(grid, [0, pelvis * length, length], [0.014, 0.024, 0.026])
    phase = np.r_[0, np.cumsum(np.diff(grid) / width[1:])]
    kk = np.interp(s, grid, phase)
    f = kk - np.floor(kk)
    band = mix(C["belly_hi"], C["belly"], smooth(0.1, 0.85, f))
    band = mix(band, C["belly_groove"], smooth(0.78, 0.97, f))
    band *= (0.8 + 0.3 * fbm(around * 30, s * 8, 5))[..., None]
    band = mix(band, C["grime"], 0.4 * smooth(0.5, 0.8, fbm(around * 12, s * 12, 6)))
    edge = smooth(0.66, 0.75, u)
    col = mix(col, band, edge)
    col = mix(col, C["crack"], 0.6 * np.exp(-((u - 0.665) / 0.03) ** 2))       # the seam
    dirt = smooth(0.45, 0.8, u) * (0.5 + 0.5 * smooth(0.55, 0.1, v))           # low flanks, the tail
    dirt *= smooth(0.35, 0.7, fbm(around * 12, s * 12, 7))
    col = mix(col, C["grime"], 0.6 * dirt)
    return lit(col, nrm, sheen=0.3)


def _limb(t, seed, length, size):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    around = u * np.pi * t["radius"]
    s = v * length
    col = hide(around, s, seed, size=size, warts=0.08)
    # knobbly scutes down the front of the limb toward the foot
    k = s / 0.02
    f = k - np.floor(k)
    scute = mix(C["belly"] * 0.85, C["belly_groove"], smooth(0.65, 0.95, f))
    front = smooth(0.35, 0.1, u) * smooth(0.45, 0.65, v)
    col = mix(col, scute * (0.8 + 0.3 * fbm(around * 30, s * 30, seed))[..., None], 0.55 * front)
    col = mix(col, C["grime"], 0.7 * smooth(0.55, 1.0, v) * smooth(0.3, 0.6, fbm(around * 20, s * 20, seed + 1)))
    col = mix(col, C["scale_dark"], 0.35 * smooth(0.6, 1.0, u))                 # the inner side
    return lit(col, nrm, sheen=0.25)


def paint_leg(t):
    return _limb(t, 7, 0.36, 0.014)


def paint_arm(t):
    return _limb(t, 13, 0.26, 0.012)


def paint_claw(t):
    """Horn-yellow claws: dark at the root, ochre, a pale yellow tip, with growth rings."""
    along = t["along"]
    col = mix(C["grime"], C["claw_base"], smooth(0.0, 0.18, along))
    col = mix(col, C["claw"], smooth(0.18, 0.5, along))
    col = mix(col, C["claw_tip"], smooth(0.75, 1.0, along))
    k = along * 6
    col *= (0.85 + 0.15 * (k - np.floor(k)))[..., None]
    up = np.clip(t["nrm"][:, 1], -1, 1)                  # claws point down: a flatter light keeps them yellow
    return col * (0.9 + 0.2 * up)[..., None]


def _skull_v():
    from scripts.spawn_model import HEAD_RINGS, SKULL_BACK, SNOUT_TIP
    length = np.linalg.norm(SNOUT_TIP - SKULL_BACK)
    return [np.linalg.norm(np.asarray(r[0]) - SKULL_BACK) / length for r in HEAD_RINGS], length


def paint_skull(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    (v0, v1, v2), length = _skull_v()
    s = v * length
    around = u * 0.3
    col = hide(around, s, 11, size=0.012, warts=0.25 * smooth(0.45, 0.1, u))
    col = mix(col, C["back"] * 0.85, 0.4 * smooth(0.2, 0.0, u))                # the dark crown
    col = mix(col, C["belly_hi"] * 0.75, 0.3 * smooth(0.45, 0.64, u) * smooth(0.95, 0.6, u))   # cheeks
    k = s / 0.02
    f = k - np.floor(k)
    wr = smooth(v1, v2, v) * smooth(0.5, 0.2, u) * np.exp(-((f - 0.5) / 0.15) ** 2)
    col = mix(col, C["crack"], 0.5 * wr)                                         # snout wrinkles
    dn = ((u - 0.12) / 0.045) ** 2 + ((v - (v2 + 0.6 * (1 - v2))) / 0.05) ** 2
    col = mix(col, C["pupil"], 0.9 * smooth(1.0, 0.4, dn))                       # nostrils
    # a dark sunken ring round each eye dome (the domes are geometry: paint_eye)
    ve = v - 0.5
    ring = np.exp(-(((u - 0.2) / 0.09) ** 2) - (ve / 0.13) ** 2)
    col = mix(col, C["scale_dark"] * .6, 0.5 * ring)
    # the lip: a dark line, then the glowing palate
    col = mix(col, C["crack"], 0.8 * np.exp(-((u - 0.64) / 0.016) ** 2))
    inside = smooth(0.66, 0.70, u)
    glow = mix(C["maw"], C["glow"], smooth(0.75, 1.0, u) * smooth(0.95, 0.3, v))
    glow = mix(glow, C["glow_core"], 0.6 * smooth(0.85, 1.0, u) * smooth(0.5, 0.1, v))
    glow = mix(glow, C["maw_edge"], smooth(0.75, 0.67, u))
    col = lit(col, nrm, sheen=0.4)
    return mix(col, glow * glow_comp(t)[..., None], inside)


def paint_eye(t):
    """A bulging, glowing eye: a dark lid at its root, orange-red, a yellow ring round a black pupil."""
    along = t["along"]
    col = mix(C["eye_rim"], C["eye"], smooth(0.05, 0.4, along))
    col = mix(col, C["eye_ring"], smooth(0.5, 0.72, along))
    col = mix(col, C["pupil"], smooth(0.8, 0.9, along))
    col = col * glow_comp(t, 0.1)[..., None]
    lid = smooth(0.25, 0.0, along)                                               # the lid takes the light
    return mix(col, lit(C["scale_dark"] * 1.4, t["nrm"]), lid)


def paint_jaw(t):
    u, v, nrm = t["u"], t["v"], t["nrm"]
    # u: 0 tongue bed (inside), 0.5 lips, 1 keel (under the chin)
    s = v * 0.24
    side = hide(u * 0.15, s, 17, size=0.011, warts=0.1)
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
    return mix(col, glow * glow_comp(t)[..., None], inside)


def paint_throat(t):
    u, v = t["u"], t["v"]
    r = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2
    return mix(C["glow_core"], C["glow"], smooth(0.0, 0.9, r)) * glow_comp(t, 0.2)[..., None]


def paint_tongue(t):
    u, v = t["u"], t["v"]
    col = mix(C["tongue_hi"], C["tongue"], smooth(0.0, 0.6, u))
    col = mix(col, C["maw_edge"], 0.6 * np.exp(-(u / 0.08) ** 2))
    col = mix(col, C["maw_edge"] * 0.8, smooth(0.6, 1.0, u))
    col *= (0.9 + 0.15 * fbm(u * 10, v * 20, 41))[..., None]
    return lit(col, t["nrm"], sheen=0.6, power=14)


def paint_spike(t):
    along = t["along"]
    col = mix(C["back"], C["spike"], smooth(0.0, 0.12, along))
    col = mix(col, C["spike_tip"], smooth(0.3, 0.95, along))
    col *= (0.86 + 0.14 * np.sin(t["u"] * 26 + along * 4))[..., None]       # streaks
    return col          # double-sided fins: painted, not lit


def paint_fang(t):
    along = t["along"]
    col = mix(C["tooth_base"], C["tooth"], smooth(0.0, 0.35, along))
    col = mix(col, C["tooth_tip"], smooth(0.7, 1.0, along))
    return lit(col, t["nrm"], sheen=0.5, power=8) * 0.6 + col * 0.45


PAINTERS = {
    (BODY, "trunk"): paint_trunk, (BODY, "leg"): paint_leg, (BODY, "arm"): paint_arm, (BODY, "claw"): paint_claw,
    (HEAD, "skull"): paint_skull, (HEAD, "jaw"): paint_jaw, (HEAD, "spike"): paint_spike,
    (HEAD, "eye"): paint_eye, (HEAD, "fang"): paint_fang, (HEAD, "tongue"): paint_tongue,
    (HEAD, "throat"): paint_throat,
}


def bake(parts, occlusion=None, size=SIZE):
    """-> {texture name: 128 x 128 RGB image}."""
    if occlusion is None:
        occlusion = vertex_occlusion(parts, skip=("spikes", "tongue"), reach=0.1)
    return _bake(parts, size=size, occlusion=occlusion, atlas=ATLAS, painters=PAINTERS, local=island_local,
                 ring_islands=("trunk", "leg", "arm"))
