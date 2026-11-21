"""The Gargoyle's three 128 x 128 skins (Gargoyle_Body, Gargoyle_Head, Gargoyle_Wing), painted by baking
with the Dragon's baker (scripts.dragon_texture.bake, on this ATLAS and these painters).

Painted like the game's own skins (shaded renders, not flat colour), as weathered statue stone: the warm
grey of DK2's own gargoyle (GargoyleMM0 on the Dungeon Heart's arches; its palette measured: shadow
46,38,28 - mid 117,109,101 - light 184,179,175), mottled by weather, streaked darker where the rain ran
down, pitted with small dark holes with a lit lower rim (as the game's gargoyle is), split by fine dark
cracks with a lit edge, chipped (paler fresh stone with a shadowed upper edge) and with a little
green-brown moss in the crevices, the folds and the cracks. The painted light comes from above, with a
sheen on the upper surfaces; the bake's occlusion darkens the folds.

The anatomy is painted into the stone, from the bind pose's 3D positions (so it lands where it is meant to
whatever the UVs): a spine ridge of knobbly vertebrae and ribs on the hunched back and flanks, a lean,
ribbed chest and belly, kneecaps and shin tendons, scaly talons, knuckles; on the head a heavy brow over
deep sockets with slightly glowing amber eyes, frown furrows up the forehead, a pug nose with flared
nostrils, snarl wrinkles across the muzzle, a dark lip line and the dark red-black mouth with a row of
small teeth; ringed horns; ears with dark creased hollows; finger bones radiating down the stone wings
(lit on top, shadowed below) with thinner membrane between, darker scalloped hems.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_texture import SIZE, atlas_uv as _atlas_uv, bake as _bake, mix, smooth, vertex_occlusion

BODY, HEAD, WING = "Gargoyle_Body", "Gargoyle_Head", "Gargoyle_Wing"

# texture -> island -> (x0, y0, x1, y1) in pixels of the 128 texture, transpose (v runs along x)
ATLAS = {
    BODY: {
        "trunk": ((0, 0, 128, 54), True),
        "leg": ((0, 56, 128, 86), True),
        "arm": ((0, 88, 80, 108), True),
        "claw": ((82, 88, 128, 108), False),
        "tail": ((0, 110, 64, 128), True),
        "spade": ((66, 110, 84, 128), False),
    },
    HEAD: {
        "skull": ((0, 0, 128, 100), True),
        "jaw": ((0, 102, 40, 128), True),
        "brow": ((42, 102, 50, 128), True),
        "horn": ((52, 102, 76, 128), True),
        "ear": ((78, 102, 104, 128), False),
        "fang": ((106, 102, 117, 128), False),
        "throat": ((119, 102, 128, 128), False),
    },
    WING: {
        "membrane": ((0, 0, 128, 104), False),
        "warm": ((0, 106, 128, 128), True),
    },
}


def island_local(part):
    uv = np.clip(np.array(part.uv, float), 0, 1)
    if part.name == "wings":
        from scripts.gargoyle_model import WING_UV_EXTENT
        m = np.array(part.island) == "membrane"
        uv[m] = uv[m] / np.maximum(WING_UV_EXTENT, 1e-9)
    return uv


def atlas_uv(part):
    return _atlas_uv(part, ATLAS, island_local)


# ------------------------------------------------------------------------------------------- palette

C = {k: np.array(v, float) for k, v in {
    "shadow": (62, 54, 45), "dark": (104, 96, 85), "stone": (146, 138, 127), "light": (184, 177, 166),
    "hi": (222, 217, 208), "warm": (160, 142, 118), "cool": (136, 137, 136),
    "crack": (40, 34, 28), "pit": (48, 41, 34), "chip": (220, 216, 206),
    "moss": (92, 106, 46), "moss_hi": (134, 142, 66), "moss_brown": (108, 92, 52), "grime": (88, 78, 62),
    "eye_core": (255, 236, 120), "eye": (255, 136, 14), "eye_edge": (196, 60, 6), "socket": (26, 19, 15),
    "mouth": (84, 12, 12), "mouth_dark": (16, 5, 5), "tongue": (74, 14, 13), "tooth": (220, 214, 200),
    "sheen": (232, 228, 220),
}.items()}


# ------------------------------------------------------------------------------------------- 3D noise

def _hash3(ix, iy, iz, seed=0):
    h = (ix * 374761393 + iy * 668265263 + iz * 1440662683 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def vnoise3(p, seed=0):
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * \
                    (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out


def fbm3(p, seed=0, octaves=3):
    out, amp, f, tot = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        out = out + amp * vnoise3(p * f, seed + o * 17)
        tot += amp
        amp *= 0.5
        f *= 2.0
    return out / tot


def worley3(p, seed=0, jitter=0.85):
    """-> F1, F2, the nearest cell's random 0..1, unit vector from its centre to the point."""
    i = np.floor(p).astype(np.int64)
    n = len(p)
    best, second = np.full(n, 9.0), np.full(n, 9.0)
    rnd = np.zeros(n)
    vec = np.zeros((n, 3))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = i + np.array([dx, dy, dz])
                cen = c + 0.5 + jitter * (np.stack([_hash3(c[:, 0], c[:, 1], c[:, 2], seed + k)
                                                    for k in (1, 2, 3)], 1) - 0.5)
                d = p - cen
                dist = np.linalg.norm(d, axis=1)
                closer = dist < best
                second = np.where(closer, best, np.minimum(second, dist))
                best = np.where(closer, dist, best)
                rnd = np.where(closer, _hash3(c[:, 0], c[:, 1], c[:, 2], seed + 7), rnd)
                vec = np.where(closer[:, None], d, vec)
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    return best, second, rnd, vec


# ------------------------------------------------------------------------------------------- stone

LIGHT = np.array([-.25, 1.0, .35]) / np.linalg.norm([-.25, 1.0, .35])
HALF = (LIGHT + np.array([0, 0, 1.0])) / np.linalg.norm(LIGHT + np.array([0, 0, 1.0]))


def sym(pos):
    """Bind positions mirrored to the left (the UVs are shared left and right)."""
    q = np.array(pos, float)
    q[:, 0] = np.abs(q[:, 0])
    return q


def lit(col, nrm, sheen=0.22, power=8, ambient=0.74):
    """The painted light: from above (a little from the front), a sheen on the upper surfaces."""
    nrm = np.array(nrm, float)
    nrm[:, 0] = np.abs(nrm[:, 0])            # mirrored halves share texels: light them alike
    k = ambient + 0.42 * (nrm @ LIGHT)
    spec = np.clip(nrm @ HALF, 0, 1) ** power * sheen
    return col * np.clip(k, 0.56, 1.2)[..., None] + (C["sheen"] - col * 0.6) * spec[..., None]


def emboss(p, freq, seed, octaves=2, h=0.25):
    """Noise read as a carved relief lit from above: + where the surface rises toward the light."""
    up = np.array([-.25, 1.0, .2]) * h / freq
    return (fbm3((p + up) * freq, seed, octaves) - fbm3((p - up) * freq, seed, octaves)) / 0.35


def stone(t, seed, scale=1.0, pits=1.0, cracks=1.0, moss=1.0, chips=1.0, streaks=1.0):
    """Weathered statue stone at the texels' bind positions. -> (RGB unlit, crevice 0..1 for the moss)."""
    p = sym(t["pos"]) / scale
    ao = t["ao"]
    n = len(p)
    big = fbm3(p * 9, seed)                                           # weather mottling
    col = mix(C["dark"], C["light"], 0.15 + 0.85 * big)
    col = mix(col, C["warm"], 0.45 * smooth(0.45, 0.75, fbm3(p * 5 + 3, seed + 1)))
    col = mix(col, C["cool"], 0.3 * smooth(0.5, 0.8, fbm3(p * 6 + 7, seed + 2)))
    grain = fbm3(p * 70, seed + 3, 2)                                 # the grain of the stone
    col *= (0.86 + 0.26 * grain)[..., None]
    relief = 0.55 * emboss(p, 14, seed + 20, 3) + 0.35 * emboss(p, 45, seed + 21, 2)   # chiselled lumps
    col *= np.clip(1 + 0.5 * relief, 0.55, 1.45)[..., None]
    if streaks:                                                       # rain streaks running down
        st = fbm3(p * np.array([45, 4, 45]), seed + 4, 2)
        col = mix(col, C["grime"], streaks * 0.4 * smooth(0.55, 0.8, st))
    crevice = np.zeros(n)
    if cracks:                                                        # fine cracks, lit on their lower lip
        f1, f2, _, _ = worley3(p * 22, seed + 5)
        edge = f2 - f1
        where = smooth(0.5, 0.65, fbm3(p * 8, seed + 6))
        line = smooth(0.07, 0.0, edge) * where * cracks
        col = mix(col, C["crack"], 0.85 * line)
        below = smooth(0.16, 0.07, edge) * (1 - smooth(0.07, 0.0, edge)) * where * cracks
        col = mix(col, C["hi"], 0.25 * below)
        crevice = np.maximum(crevice, line)
    if pits:                                                          # small pits with a lit lower rim
        f1, _, r, v = worley3(p * 48, seed + 8, 0.7)
        on = r > 1 - 0.32 * pits
        hole = on * smooth(0.22, 0.1, f1)
        rim = on * smooth(0.36, 0.24, f1) * (1 - smooth(0.22, 0.12, f1)) * np.clip(-v[:, 1] + 0.2, 0, 1)
        col = mix(col, C["pit"], 0.8 * hole)
        col = mix(col, C["hi"], 0.45 * rim)
    if chips:                                                         # chipped: paler fresh stone
        f1, _, r, v = worley3(p * 13, seed + 9, 0.8)
        on = r > 1 - 0.12 * chips
        inside = on * smooth(0.36, 0.28, f1)
        col = mix(col, C["chip"] * (0.9 + 0.15 * grain)[..., None], 0.75 * inside)
        edge = on * smooth(0.26, 0.36, f1) * smooth(0.46, 0.36, f1) * np.clip(v[:, 1] + 0.3, 0, 1)
        col = mix(col, C["crack"], 0.7 * edge)
    crevice = np.maximum(crevice, smooth(0.75, 0.35, ao))
    if moss:                                                          # moss in the crevices
        m = smooth(0.35, 0.7, fbm3(p * 16, seed + 10)) * crevice * moss
        m = np.maximum(m, moss * 0.8 * smooth(0.72, 0.85, fbm3(p * 11, seed + 11)) * smooth(0.6, 0.3, ao))
        mc = mix(C["moss"], C["moss_hi"], fbm3(p * 60, seed + 12, 2))
        mc = mix(mc, C["moss_brown"], 0.5 * smooth(0.4, 0.7, fbm3(p * 20, seed + 13)))
        col = mix(col, mc, np.clip(m, 0, 0.85))
    return col, crevice


def glow_comp(t, keep=0.3):
    """Undo most of the bake's occlusion darkening (0.45 + 0.55 ao) on glowing texels."""
    return (1 / (0.45 + 0.55 * t["ao"])) ** (1 - keep)


def _bands(x, period, width=0.25):
    """Soft dark grooves every `period` along x: 0 between, 1 in a groove."""
    f = x / period - np.floor(x / period)
    return np.exp(-((f - 0.5) / width) ** 2 * 4)


# ------------------------------------------------------------------------------------------- painters

def paint_trunk(t):
    p = sym(t["pos"])
    col, _ = stone(t, 1)
    u = t["u"]
    # the spine: a dark furrow with a knobbly ridge of vertebrae, lit on top
    s_len = p[:, 1] * 0.8 + p[:, 2] * 0.6
    knob = np.sin(s_len / 0.028 * 2 * np.pi) * 0.5 + 0.5
    spine = smooth(0.14, 0.0, u)
    col = mix(col, C["light"], 0.4 * spine * knob)
    col = mix(col, C["dark"], 0.45 * spine * (1 - knob))
    # ribs on the back and flanks of the hunched chest
    chest = smooth(0.30, 0.37, p[:, 1]) * smooth(0.48, 0.42, p[:, 1])
    rib = _bands(p[:, 1] - 0.4 * p[:, 2], 0.03, 0.18)
    side = smooth(0.1, 0.3, u) * smooth(0.85, 0.6, u)
    col = mix(col, C["shadow"], 0.38 * chest * side * rib)
    col = mix(col, C["hi"], 0.18 * chest * side * (1 - rib))
    # the belly: lean muscle segments down the front
    belly = smooth(0.6, 0.8, u)
    seg = _bands(p[:, 1] + 0.01 * fbm3(p * 30, 33), 0.05, 0.14)
    col = mix(col, C["shadow"], 0.28 * belly * seg * smooth(0.22, 0.27, p[:, 1]) * smooth(0.42, 0.36, p[:, 1]))
    col = mix(col, C["dark"], 0.5 * smooth(0.9, 1.0, u) * smooth(0.36, 0.3, p[:, 1]))     # the midline
    # the pectorals: lit above, shadowed below
    pec = smooth(0.55, 0.75, u) * np.exp(-((p[:, 1] - 0.43) / 0.03) ** 2)
    col = mix(col, C["hi"], 0.3 * pec)
    col = mix(col, C["shadow"], 0.45 * smooth(0.55, 0.75, u) * np.exp(-((p[:, 1] - 0.395) / 0.012) ** 2))
    return lit(col, t["nrm"])


def paint_tail(t):
    col, _ = stone(t, 2, pits=0.6)
    seg = _bands(t["v"], 0.12, 0.15)                                  # knobbly segments
    col = mix(col, C["shadow"], 0.4 * seg)
    col = mix(col, C["dark"], 0.3 * smooth(0.5, 1.0, t["u"]))
    return lit(col, t["nrm"])


def paint_leg(t):
    p = sym(t["pos"])
    col, _ = stone(t, 3)
    u, v = t["u"], t["v"]
    # the kneecap: a lit knob on the front of the knee, shadowed round
    knee = np.linalg.norm((p - np.array([.13, .25, .13])) / np.array([1, 1, 1.0]), axis=1)
    col = mix(col, C["hi"], 0.5 * smooth(0.035, 0.0, knee))
    col = mix(col, C["shadow"], 0.4 * smooth(0.03, 0.045, knee) * smooth(0.06, 0.045, knee))
    # thigh muscle: a groove along its side
    thigh = smooth(0.4, 0.3, v)
    col = mix(col, C["shadow"], 0.4 * thigh * np.exp(-((u - 0.5) / 0.08) ** 2))
    # the shin's tendon and the foot's scutes
    shin = smooth(0.45, 0.55, v)
    col = mix(col, C["light"], 0.3 * shin * smooth(0.25, 0.0, u))
    col = mix(col, C["shadow"], 0.45 * smooth(0.8, 0.85, v) * _bands(v, 0.05, 0.15))
    col = mix(col, C["dark"], 0.35 * smooth(0.6, 1.0, u))                         # the inner side
    return lit(col, t["nrm"])


def paint_arm(t):
    p = sym(t["pos"])
    col, _ = stone(t, 4, chips=0.6)
    u, v = t["u"], t["v"]
    elbow = np.linalg.norm(p - np.array([.165, .33, .015]), axis=1)
    col = mix(col, C["hi"], 0.4 * smooth(0.03, 0.0, elbow))
    col = mix(col, C["shadow"], 0.4 * np.exp(-((u - 0.5) / 0.1) ** 2) * smooth(0.1, 0.3, v) * smooth(0.5, 0.4, v))
    # the hand: knuckles
    hand = smooth(0.82, 0.9, v)
    col = mix(col, C["shadow"], 0.5 * hand * _bands(u, 0.33, 0.2))
    col = mix(col, C["light"], 0.3 * hand * (1 - _bands(u, 0.33, 0.2)))
    col = mix(col, C["dark"], 0.3 * smooth(0.6, 1.0, u))
    return lit(col, t["nrm"])


def paint_claw(t):
    along = t["along"]
    col, _ = stone(t, 5, pits=0, chips=0, moss=0, streaks=0, cracks=0.4)
    col = mix(col * 0.75, col, smooth(0.0, 0.3, along))
    col = mix(col, C["shadow"], 0.5 * smooth(0.7, 0.95, along))
    col = mix(col, C["chip"], 0.6 * smooth(0.9, 1.0, along))
    col *= (0.85 + 0.15 * _bands(along, 0.25, 0.3))[..., None]
    return lit(col, t["nrm"], sheen=0.35)


# the head's features (left side; the right mirrors), bind positions. Soles, 13 Nov: "make the
# gargoyle's face angrier and scarier": narrow, slanted, hooded eyes burning under a heavy V brow (the
# brow is geometry: paint_brow), deep sockets, a wrinkled nose, snarl lines, the upper lip curled off
# the teeth, a crimson-to-black mouth and deep cracks across the face
EYE = np.array([.045, .573, .262])
EYE_TILT = np.radians(30)                     # the outer corner higher: a scowl
EYE_SIZE = (0.0135, 0.0052)                   # half length, half height: a narrow slit of an eye
NOSTRIL = np.array([.02, .541, .312])


def brow_y(x):
    """Under the brow ridge: low at the middle (the frown), high at the sides (the geometry's line)."""
    from scripts.gargoyle_model import BROW_INNER, BROW_OUTER
    (x0, y0), (x1, y1) = BROW_INNER, BROW_OUTER
    return y0 + (y1 - y0) * (np.clip(x, 0, .1) - x0) / (x1 - x0) - 0.012


def _face_cracks(p, seed):
    """A few deep cracks across the face, lit on the lower lip: cell edges, but warped so they wander
    and broken into short runs (straight unbroken ones read as a visor's seam: review round 1)."""
    w = np.stack([fbm3(p * 30, seed + 2), fbm3(p * 30 + 5, seed + 3), fbm3(p * 30 + 9, seed + 4)], 1) - .5
    q = (p + w * 0.018) * np.array([9, 14, 9])
    f1, f2, r, _ = worley3(q, seed)
    e = f2 - f1
    on = smooth(0.52, 0.66, fbm3(p * 14, seed + 1)) * (r > 0.3)
    return smooth(0.045, 0.0, e) * on, smooth(0.1, 0.045, e) * (1 - smooth(0.045, 0.0, e)) * on


def paint_skull(t):
    from scripts.gargoyle_model import HEAD_U
    LIP_U = HEAD_U[2]
    p = sym(t["pos"])
    u = t["u"]
    col, _ = stone(t, 6, chips=0.7, moss=0.6)
    face = smooth(0.215, 0.24, p[:, 2])
    # the socket: deep shadow from under the brow down round the eye, hooded over its top
    dx, dy = p[:, 0] - EYE[0], p[:, 1] - EYE[1]
    e1 = dx * np.cos(EYE_TILT) + dy * np.sin(EYE_TILT)
    e2 = -dx * np.sin(EYE_TILT) + dy * np.cos(EYE_TILT)
    d_eye = np.sqrt((e1 / EYE_SIZE[0]) ** 2 + (e2 / EYE_SIZE[1]) ** 2)       # 1 on the eye's rim
    by = brow_y(p[:, 0])
    # (review round 1: a thin dark band here read as a visor's seam: it is now a broad soft shadow under
    # the brow, kept to the eye's width, fading into the socket)
    under_brow = face * smooth(0.095, 0.07, p[:, 0]) * smooth(0.008, -0.006, p[:, 1] - by) * \
        smooth(-0.045, -0.018, p[:, 1] - by)
    col = mix(col, C["shadow"], 0.75 * under_brow)
    socket = face * smooth(3.2, 1.3, np.sqrt((e1 / EYE_SIZE[0]) ** 2 + (e2 / (EYE_SIZE[1] * 1.8)) ** 2))
    col = mix(col, C["socket"], 0.9 * socket)
    # a lit cheekbone under the socket, a dark crease under that
    cheek_d = p[:, 1] - (EYE[1] - 0.018 + 0.25 * (p[:, 0] - EYE[0]))
    cb = face * smooth(0.0, 0.03, p[:, 0]) * smooth(0.1, 0.07, p[:, 0])
    col = mix(col, C["light"], 0.4 * cb * np.exp(-(cheek_d / 0.007) ** 2))
    col = mix(col, C["dark"], 0.35 * cb * np.exp(-((cheek_d + 0.012) / 0.008) ** 2))
    # the frown: deep vertical furrows up the forehead from the bunched brows, and creases across it
    fore = smooth(0.035, 0.0, p[:, 0]) * smooth(0.585, 0.6, p[:, 1]) * smooth(0.27, 0.21, p[:, 2]) * \
        smooth(0.15, 0.2, p[:, 2])
    col = mix(col, C["crack"], 0.75 * fore * _bands(p[:, 0] + 0.006, 0.014, 0.18))
    acr = smooth(0.62, 0.635, p[:, 1]) * smooth(0.25, 0.2, p[:, 2]) * smooth(0.15, 0.19, p[:, 2]) * \
        smooth(0.085, 0.05, p[:, 0])
    col = mix(col, C["shadow"], 0.55 * acr * _bands(p[:, 1] + 0.3 * p[:, 0], 0.011, 0.2))
    # the nose wrinkled up into the frown: deep creases across the bridge
    bridge = face * smooth(0.035, 0.01, p[:, 0]) * smooth(0.548, 0.556, p[:, 1]) * smooth(0.592, 0.582, p[:, 1])
    wr = _bands(p[:, 1] - 0.15 * p[:, 0], 0.0075, 0.22)
    col = mix(col, C["crack"], 0.7 * bridge * wr)
    col = mix(col, C["hi"], 0.3 * bridge * (1 - wr))
    # a snarl line from each nostril's wing down and out to the corner of the mouth: a soft fold, shadow
    # below a lit edge (a thin dark line here read as a crease drawn on: review round 2)
    sn = np.hypot(p[:, 0] - .0, (p[:, 1] - .575) * 1.3)
    zone = smooth(0.255, 0.285, p[:, 2]) * smooth(0.012, 0.03, p[:, 0]) * smooth(0.56, 0.545, p[:, 1])
    col = mix(col, C["shadow"], 0.5 * zone * np.exp(-((sn - .058) / .007) ** 2))
    col = mix(col, C["light"], 0.25 * zone * np.exp(-((sn - .048) / .006) ** 2))
    # the curled upper lip: short deep creases along it, lit on their tops
    lipz = smooth(LIP_U - 0.16, LIP_U - 0.06, u) * smooth(LIP_U - 0.005, LIP_U - 0.03, u) * smooth(0.24, 0.27, p[:, 2])
    cr = _bands(p[:, 2] * 1.0 + p[:, 0] * 0.6, 0.009, 0.2)
    col = mix(col, C["crack"], 0.6 * lipz * cr)
    col = mix(col, C["hi"], 0.25 * lipz * (1 - cr))
    # the pug nose: flared nostrils, a lit pad
    d_n = np.linalg.norm((p - NOSTRIL) / np.array([1.0, 1.4, 1.0]), axis=1)
    col = mix(col, C["socket"], 0.95 * smooth(0.0095, 0.006, d_n))
    col = mix(col, C["hi"], 0.5 * smooth(0.008, 0.013, d_n) * smooth(0.019, 0.013, d_n) * (p[:, 1] > NOSTRIL[1]))
    # deep cracks across the face, readable from the high camera
    cline, clit = _face_cracks(p, 61)
    col = mix(col, C["crack"], 0.9 * cline * face)
    col = mix(col, C["hi"], 0.3 * clit * face)
    # the back of the skull: ridged
    back = smooth(0.17, 0.12, p[:, 2])
    col = mix(col, C["shadow"], 0.3 * back * _bands(p[:, 2] + p[:, 1] * 0.3, 0.025, 0.2))
    col = lit(col, t["nrm"], sheen=0.25)
    # the lip: a dark line, the teeth bared under it (uneven), then the crimson-to-black mouth
    col = mix(col, C["crack"], 0.9 * np.exp(-((u - LIP_U + 0.008) / 0.012) ** 2))
    k = p[:, 2] / 0.013
    tw = np.abs((k - np.floor(k)) - 0.5) * 2
    tl = 0.035 + 0.025 * _hash_noise(np.floor(k))
    tooth = (u > LIP_U + 0.004) & (u < LIP_U + 0.004 + tl * (1 - tw)) & (p[:, 2] > 0.2)
    inside = smooth(LIP_U - 0.004, LIP_U + 0.012, u)
    mouth = mix(C["mouth"], C["mouth_dark"], smooth(0.84, 0.98, u) + 0.4 * smooth(0.27, 0.2, p[:, 2]))
    mouth *= (0.8 + 0.3 * _bands(p[:, 2], 0.02, 0.3))[..., None]                 # ridges of the palate
    col = mix(col, mouth, inside)
    teeth = mix(C["tooth"] * 0.7, C["tooth"], 1 - tw)
    col = np.where(tooth[..., None], teeth * (0.85 + 0.15 * (u < LIP_U + 0.02))[..., None], col)
    # the eyes: a narrow slit burning hot amber-orange (they don't take the painted light)
    eye = face * smooth(1.0, 0.75, d_eye)
    glow = mix(C["eye_core"], C["eye"], smooth(0.0, 0.6, d_eye))
    glow = mix(glow, C["eye_edge"], smooth(0.5, 0.95, d_eye))
    slit = np.exp(-(e1 / 0.0022) ** 2) * smooth(0.9, 0.2, d_eye)
    glow = mix(glow, C["eye_edge"] * 0.5, 0.7 * slit)
    lid = smooth(0.0, 0.8, e2 / EYE_SIZE[1]) * 0.8                    # the hooded lid over its top
    glow = mix(glow, C["socket"] * 1.5, lid)
    return mix(col, glow * glow_comp(t, 0.15)[..., None], eye)


def _hash_noise(k):
    return ((np.sin(k * 12.9898) * 43758.5453) % 1.0 + 1.0) % 1.0


def paint_brow(t):
    """The heavy brow ridges: lit on top, deep shadow on the underside over the eye, furrowed across."""
    p = sym(t["pos"])
    col, _ = stone(t, 13, pits=0.6, chips=0.5, moss=0.3)
    nrm = np.array(t["nrm"], float)
    top = smooth(-0.3, 0.6, nrm[:, 1])
    col = mix(col * 0.72, col * 1.18, top)                 # a lit ridge, softly shaded below
    fur = _bands(p[:, 0] * 1.0 - p[:, 1] * 0.3, 0.012, 0.18)
    col = mix(col, C["crack"], 0.55 * fur)
    return lit(col, nrm, sheen=0.35)


def paint_jaw(t):
    u = t["u"]
    col, _ = stone(t, 7, chips=0.6, moss=0.5)
    p = sym(t["pos"])
    col = mix(col, C["shadow"], 0.4 * _bands(p[:, 2], 0.022, 0.2) * smooth(0.6, 0.9, u))    # chin folds
    cline, clit = _face_cracks(p, 71)
    col = mix(col, C["crack"], 0.8 * cline)
    col = lit(col, t["nrm"])
    # u: 0 the tongue bed (inside), 0.5 the lips, 1 the keel under the chin
    col = mix(col, C["crack"], 0.85 * np.exp(-((u - 0.47) / 0.025) ** 2))
    k = p[:, 2] / 0.013
    tw = np.abs((k - np.floor(k)) - 0.5) * 2
    tl = 0.06 + 0.04 * _hash_noise(np.floor(k) + 5)
    tooth = (u < 0.45) & (u > 0.45 - tl * (1 - tw)) & (p[:, 2] > 0.2)
    inside = smooth(0.44, 0.38, u)
    mouth = mix(C["tongue"], C["mouth_dark"], 0.35 + 0.65 * smooth(0.04, 0.3, u))
    mouth = mix(mouth, C["mouth_dark"], smooth(0.8, 0.3, t["v"]))           # black toward the throat
    col = mix(col, mouth, inside)
    teeth = mix(C["tooth"] * 0.7, C["tooth"], 1 - tw)
    return np.where(tooth[..., None], teeth, col)


def paint_throat(t):
    return C["mouth_dark"] + 0 * t["u"][:, None]


def paint_horn(t):
    along = np.clip(t["v"], 0, 1)
    col, _ = stone(t, 8, pits=0.4, moss=0.3, chips=0.5)
    col = mix(col * 0.9, col * 1.12, smooth(0.0, 0.6, along))
    col = mix(col, C["dark"], 0.5 * _bands(along, 0.14, 0.18) * smooth(0.0, 0.1, along))      # ridges
    col = mix(col, C["chip"], 0.5 * smooth(0.8, 1.0, along))
    return lit(col, t["nrm"], sheen=0.3, ambient=0.85)


def paint_ear(t):
    u, v = t["u"], t["v"]
    col, _ = stone(t, 9, pits=0.5, moss=0.4)
    # dark creased hollow in the middle, a lit rim along the edges
    edge = np.minimum(np.minimum(v, np.abs(u - 0.5 * v) * 2), np.abs(1 - u - 0.5 * (1 - v)) * 2)
    col = mix(col, C["shadow"], 0.55 * smooth(0.08, 0.25, edge))
    col = mix(col, C["dark"], 0.4 * _bands(v * 3 + u, 0.5, 0.15) * smooth(0.1, 0.25, edge))
    col = mix(col, C["hi"], 0.4 * smooth(0.06, 0.0, edge))
    return col * 0.95


def paint_fang(t):
    along = t["along"]
    col = mix(C["dark"], C["tooth"], smooth(0.0, 0.35, along))
    col = mix(col, C["hi"], smooth(0.7, 1.0, along))
    return lit(col, t["nrm"], sheen=0.4) * 0.6 + col * 0.45


def paint_membrane(t):
    from scripts.gargoyle_model import WING_UV, WING_UV_EXTENT
    u, v = t["u"], t["v"]
    ext = np.maximum(WING_UV_EXTENT, 1e-9)
    key = {k: np.array(q) / ext for k, q in WING_UV.items()}
    q = np.stack([u, v], -1)

    def seg(a, b):
        a, b = key[a], key[b]
        ab = b - a
        h = np.clip(((q - a) @ ab) / (ab @ ab), 0, 1)
        off = q - (a + h[..., None] * ab)
        return np.linalg.norm(off, axis=-1), h, off

    col, _ = stone(t, 10, pits=0.5, chips=0.4, moss=0.5, streaks=0.6)
    col = col * 0.88
    # thinner toward the hem: a little darker, with fine radiating creases from the wrist
    w = key["W"]
    ang = np.arctan2(v - w[1], u - w[0])
    col *= (0.96 + 0.04 * np.sin(ang * 12 + 3 * fbm3(sym(t["pos"]) * 20, 31)))[..., None]
    # the finger bones and the arm: raised ribs, lit along one side, shadowed along the other
    for a, b, width in (("W", "F1", .022), ("W", "F2", .022), ("W", "F3", .022), ("S", "E", .03),
                        ("E", "W", .028)):
        d, h, off = seg(a, b)
        taper = width * (1 - 0.5 * h)
        ab = key[b] - key[a]
        nrm2 = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
        sidev = off @ nrm2
        bone = smooth(taper, taper * 0.4, d)
        col = mix(col, C["light"], 0.6 * bone * (sidev < 0))
        col = mix(col, C["dark"], 0.55 * bone * (sidev >= 0))
        col = mix(col, C["shadow"], 0.5 * smooth(taper * 1.8, taper, d) * (1 - bone))
    # the scalloped hem: darker, with a thin lit edge
    for a, b in (("F1", "C12"), ("C12", "F2"), ("F2", "C23"), ("C23", "F3"), ("F3", "B")):
        d, _, _ = seg(a, b)
        col = mix(col, C["shadow"], 0.55 * smooth(0.06, 0.012, d))
        col = mix(col, C["light"], 0.4 * smooth(0.012, 0.0, d))
    return col


def paint_warm(t):
    col, _ = stone(t, 11, chips=0.5)
    col = mix(col, C["hi"], 0.4 * smooth(0.7, 1.0, t["along"]))                   # the wrist knob
    return lit(col, t["nrm"])


def paint_spade(t):
    u, v = t["u"], t["v"]
    col, _ = stone(t, 12, pits=0.5)
    edge = np.minimum(v, np.minimum(np.abs(u - 0.5 * v), np.abs(1 - u - 0.5 * (1 - v))))
    col = mix(col, C["hi"], 0.45 * smooth(0.06, 0.0, edge))
    col = mix(col, C["shadow"], 0.35 * smooth(0.1, 0.3, edge))
    return col


PAINTERS = {
    (BODY, "trunk"): paint_trunk, (BODY, "leg"): paint_leg, (BODY, "claw"): paint_claw,
    (BODY, "arm"): paint_arm, (BODY, "tail"): paint_tail,
    (HEAD, "skull"): paint_skull, (HEAD, "brow"): paint_brow, (HEAD, "jaw"): paint_jaw, (HEAD, "horn"): paint_horn,
    (HEAD, "ear"): paint_ear, (HEAD, "fang"): paint_fang, (HEAD, "throat"): paint_throat,
    (WING, "membrane"): paint_membrane, (WING, "warm"): paint_warm, (BODY, "spade"): paint_spade,
}


def bake(parts, occlusion=None, size=SIZE):
    """-> {texture name: 128 x 128 RGB image}."""
    if occlusion is None:
        occlusion = vertex_occlusion(parts, skip=("wings",), reach=0.08)
        # softened: the game lights him too, and full-strength folds turned the horns' undersides black
        occlusion = {k: 0.35 + 0.65 * v for k, v in occlusion.items()}
    return _bake(parts, size=size, occlusion=occlusion, atlas=ATLAS, painters=PAINTERS, local=island_local,
                 ring_islands=("trunk", "leg", "arm", "tail"))
