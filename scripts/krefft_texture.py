"""Krefft's three 128 x 128 skins (Krefft_Body, Krefft_Head, Krefft_Shell), painted by baking with the
Dragon's baker (scripts.dragon_texture.bake, on this ATLAS and these painters), the Gargoyle's way.

Painted like the game's own skins (shaded renders, not flat colour): the light painted in from above with a
sheen on the upper surfaces, the bake's occlusion darkening the crevices, everything from the bind pose's 3D
positions so it lands where it is meant to whatever the UVs. From Soles' photos of his turtle:

  skin     dark slate grey, pebbly all over the neck and limbs: every pebble a little dome lit on its top
           and shadowed under, dark grit between; bigger scales down the fronts of the forelegs; folds
           round the neck; the throat and the undersides paler
  head     a smooth grey head plate on top (outlined, a few fine fissures), pebbly sides; the wide smug
           mouth line curling up at the corners; a pale cream-yellow stripe from the mouth corner back
           along the jaw and down the neck, a faint pale stripe above and behind the eye; pale silvery-green
           eyes with a dark ring and pupil, heavy lids (smug); his RIGHT eye scarred and half shut (a pale
           seam of scar through the brow, the lid and the cheek, crossed by old stitches); two nostrils
  feet     dark webbing, five long black claws each with a lit ridge
  shell    olive-khaki scutes (vertebral, costal, marginal: a Voronoi of the real layout) with dark seams
           lit on their lower lip, a lighter mottled areola in each with faint growth rings, scuffs; the
           cracks the coins are wedged in; duckweed (bright little round leaves in clusters) and dark green
           slime in the seams and the back; the underside a pale dirty yellow with dark seams
  gold     the coins are the game's own coin (tex_Coinage, DK2's EngineTextures 'Coinage'); the chain and the
           tooth are painted in the game's gold (tex_Gold's palette): battered links, a chipped tooth

LICENCE: GPL-3.0.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.dragon_texture import SIZE, atlas_uv as _atlas_uv, bake as _bake, mix, smooth, vertex_occlusion
from scripts.gargoyle_texture import _bands, fbm3, sym, worley3
from scripts.krefft_model import BODY, HEADT as HEAD, SHELL

GAME_DATA = Path("/home/claude/video2/cgi/web/data")          # the game's textures, decoded

# texture -> island -> (x0, y0, x1, y1) in pixels of the 128 texture, transpose (v runs along x).
# v3: the body islands sized to their surfaces (about 300 texels per KMF unit, the neck no longer
# stretched 2.5 : 1); the head's islands are planar views (scripts.krefft_model.head_uv) with a 22-texel
# island for each eye.
ATLAS = {
    BODY: {
        "neck": ((0, 0, 84, 42), False),
        "sole": ((85, 0, 128, 20), False),
        "tail": ((85, 21, 128, 42), True),
        "fleg": ((0, 43, 59, 106), False),
        "hleg": ((60, 43, 119, 106), False),
        "ffoot": ((0, 107, 63, 128), False),
        "hfoot": ((64, 107, 128, 128), False),
    },
    HEAD: {
        "sideL": ((0, 0, 76, 46), False),
        "sideR": ((0, 47, 76, 93), False),
        "top": ((77, 0, 128, 58), False),
        "under": ((77, 59, 113, 99), False),
        "front": ((0, 94, 40, 118), False),
        "eyeL": ((41, 94, 63, 116), False),
        "eyeR": ((64, 94, 86, 116), False),
        "lid": ((87, 100, 113, 114), False),
        "tooth": ((114, 59, 128, 79), False),
        "mouth": ((114, 80, 128, 99), False),
        "chain": ((0, 119, 128, 128), False),
    },
    SHELL: {
        "carapace": ((0, 0, 86, 128), False),
        "under": ((87, 0, 128, 104), False),
        "coin": ((87, 105, 111, 128), False),
        "crack": ((112, 105, 128, 116), False),      # v4: the broken lips of the coin's crack
        "medal": ((112, 117, 128, 128), False),      # v4: the chain's medallion
    },
}


def island_local(part):
    return np.clip(np.array(part.uv, float), 0, 1)


def atlas_uv(part):
    return _atlas_uv(part, ATLAS, island_local)


# ------------------------------------------------------------------------------------------- palette

C = {k: np.array(v, float) for k, v in {
    "slate": (84, 90, 88), "slate_dk": (44, 48, 50), "slate_hi": (156, 162, 156), "grit": (16, 18, 19),
    "pale": (140, 142, 130), "plate": (104, 109, 108), "plate_hi": (146, 150, 146), "plate_dk": (62, 66, 68),
    "stripe": (214, 200, 140), "stripe_dk": (170, 152, 96),
    "iris": (178, 192, 168), "iris_dk": (104, 120, 104), "ring": (30, 34, 30), "pupil": (10, 12, 10),
    "scar": (170, 140, 132), "scar_dk": (84, 58, 56), "mouth": (22, 14, 12), "lid": (66, 70, 70),
    "claw": (18, 18, 20), "claw_hi": (120, 120, 126), "web": (44, 48, 50),
    "khaki": (104, 96, 60), "khaki_lt": (142, 130, 86), "khaki_dk": (54, 50, 30), "seam": (22, 20, 12),
    "areola": (150, 138, 92), "scuff": (170, 160, 120),
    "slime": (58, 76, 30), "weed": (112, 168, 36), "weed_hi": (168, 214, 70), "weed_dk": (50, 96, 22),
    "plastron": (200, 184, 122), "plastron_dk": (120, 106, 62),
    "gold_hi": (255, 236, 140), "gold": (214, 164, 48), "gold_dk": (120, 76, 14), "gold_shadow": (60, 36, 6),
    "sheen": (232, 228, 220),
}.items()}

LIGHT = np.array([-.25, 1.0, .35]) / np.linalg.norm([-.25, 1.0, .35])
HALF = (LIGHT + np.array([0, 0, 1.0])) / np.linalg.norm(LIGHT + np.array([0, 0, 1.0]))


def lit(col, nrm, sheen=0.18, power=8, ambient=0.72, mirrored=True):
    """The painted light: from above (a little from the front), a sheen on the upper surfaces."""
    nrm = np.array(nrm, float)
    if mirrored:
        nrm[:, 0] = np.abs(nrm[:, 0])
    k = ambient + 0.45 * (nrm @ LIGHT)
    spec = np.clip(nrm @ HALF, 0, 1) ** power * sheen
    return col * np.clip(k, 0.5, 1.22)[..., None] + (C["sheen"] - col * 0.6) * spec[..., None]


# ------------------------------------------------------------------------------------------- skin

def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


TOP, SIDE, UNDER = _hex("#545b56"), _hex("#3a403d"), _hex("#1f2422")     # the painted form: lit from above
PEB_HI = _hex("#7c8278")                     # a pebble's lit upper edge
CREASE = _hex("#101214")
JOINT = _hex("#1e2220")                      # the joint and ankle creases                     # deep creases: elbows, knees, where the neck and legs leave the shell
WEB = _hex("#1e2320")                        # the webbing between the toes
CLAW = _hex("#1e1f1c")
TOE = _hex("#3a403d")
STRIPE = _hex("#e6d48a")                     # the cream stripe from the mouth corner to the shell
STRIPE_EDGE = _hex("#2a2c28")
BROW_STRIPE = _hex("#c8c8a8")                # the faint stripe behind the eye
PLATE, PLATE_EDGE = _hex("#7a8474"), _hex("#4a5248")
BROW_SHADE = _hex("#2a2c2a")
SCAR, SCAR_SHADOW = _hex("#c8c2b0"), _hex("#8a8070")
CHEEK, HOLLOW = _hex("#5a6058"), _hex("#2a2c28")
SLIVER = _hex("#b8c4a8")
LIDC = _hex("#4a4e48")
IRIS, EYE_RING = _hex("#b8c4a8"), _hex("#1a1c1a")
LIP = _hex("#121312")
TOOTH, TOOTH_HI, TOOTH_LO = _hex("#f0c040"), _hex("#fff0a0"), _hex("#6a4a10")
COIN_FACE, COIN_RIM, COIN_RIM_LO, COIN_MID = _hex("#f0c040"), _hex("#fff4b0"), _hex("#a07818"), _hex("#c89a30")
CRACK, CAST = _hex("#101214"), _hex("#2a2418")
CHAIN, CHAIN_HI, CHAIN_GAP = _hex("#d8a830"), _hex("#fff0a0"), _hex("#1a1408")
WEED, WEED_FLECK, WEED_SHADOW, WEED_LO = _hex("#79a83c"), _hex("#a2c25a"), _hex("#2e3a18"), _hex("#3e6020")
WEED_HEART = _hex("#2f5a1e")
MARGINAL, SCUTE_SEAM, SCUTE_CENTRE = _hex("#a49a60"), _hex("#3a3a20"), _hex("#a8a070")
PLASTRON, PLASTRON_LO, PLASTRON_SEAM, PLASTRON_SHADE = _hex("#d8c890"), _hex("#8a7c50"), _hex("#9a8a50"), _hex("#5a5030")


def form(nrm, mirrored=True):
    """The painted light as colour: tops of forms TOP, sides SIDE, undersides and crevices UNDER."""
    n = np.array(nrm, float)
    if mirrored:
        n[:, 0] = np.abs(n[:, 0])
    t = 0.8 * n[:, 1] + 0.2 * (n @ LIGHT)          # mostly how far the surface faces up
    col = mix(UNDER, SIDE, smooth(-0.6, -0.2, t))
    return mix(col, TOP, smooth(0.15, 0.8, t))


def _n2(X, Y, seed, octaves=3):
    return fbm3(np.stack([X, Y, np.full_like(X, seed * 1.37)], 1), seed, octaves)


# v3 skin: his turtle's most recognisable trait is the very pebbly slate skin (crop0), painted here as the
# game's skins are (renders of a sculpt, lit from above): each pebble a little dome placed on the
# surface in 3D (so nothing tiles or stretches across a UV seam), its upper side catching the light,
# its lower side and the grit between dark; the pebbles gathered in clusters of bigger scales on the
# fronts of the limbs, finer granules in the folds and on the throat.
PEB_LIT = _hex("#6a716b")                    # a pebble's lit top
PEB_DARK = _hex("#1f2422")                   # the grit between pebbles


def _unitrows(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def pebbles3(p, nrm, size, seed, warp=0.4):
    """Domed pebbles about `size` KMF units across on the surface at p (normal nrm) ->
    (crevice 0..1, light -1..1: how much more (or less) the pebble's slope there faces the light than the
    surface does, tone 0..1 per pebble)."""
    size = np.asarray(size, float)
    q = p / (size[:, None] if size.ndim else size)
    w = np.stack([fbm3(q * .45 + 3.1 * k, seed + k, 2) for k in (1, 2, 3)], 1) - .5
    f1, f2, r, vec = worley3(q + w * warp * 2, seed, .9)
    vt = vec - nrm * np.sum(vec * nrm, 1)[:, None]
    slope = smooth(0.08, 0.62, f1)
    n2 = _unitrows(nrm + 1.25 * slope[:, None] * vt)
    light = n2 @ LIGHT - nrm @ LIGHT
    crev = smooth(0.15, 0.0, f2 - f1)
    return crev, light, r


def skin(nrm, pos, seed, size, amount, mirrored=True, base=None):
    """Slate turtle skin: the painted form (TOP / SIDE / UNDER by the way the surface faces), a broad
    mottle, and pebbles (`amount` 0..1 per texel: how strongly they show)."""
    n = np.array(nrm, float)
    p = np.array(pos, float)
    if mirrored:
        n[:, 0] = np.abs(n[:, 0])
        p[:, 0] = np.abs(p[:, 0])
    col = form(n, False) if base is None else base
    col = col * (0.9 + 0.2 * fbm3(p * 16, seed, 3))[..., None]                   # broad mottle
    amount = np.broadcast_to(np.asarray(amount, float), (len(p),))
    if np.any(amount > 0):
        crev, light, r = pebbles3(p, n, size, seed + 7)
        col = col * (1 + amount * 0.16 * (r - .5))[..., None]
        col = col * (1 + amount * 1.05 * np.clip(light, -.4, .4))[..., None]
        col = mix(col, PEB_LIT, amount * 0.4 * smooth(0.08, 0.35, light))
        col = mix(col, PEB_DARK, amount * 0.65 * crev)
    return col


def crease(col, d, width, strength=1.0, colour=None, lip=None):
    """A deep crease along d = 0, a soft lit roll of skin on its d < 0 side."""
    col = mix(col, CREASE if colour is None else colour, strength * np.exp(-(d / width) ** 2))
    lip = _hex("#4c524e") if lip is None else lip
    return mix(col, lip, 0.5 * strength * np.exp(-((d + width * 2.4) / (width * 1.3)) ** 2))


def stripe(col, d, half, edge, strength=1.0, colour=None):
    """The cream stripe (STRIPE) where |d| < half, a 1-texel STRIPE_EDGE outline either side."""
    colour = STRIPE if colour is None else colour
    a = np.abs(d)
    col = mix(col, STRIPE_EDGE, strength * smooth(half + edge * 1.1, half + edge * .6, a) * smooth(half - edge * .2, half + .2 * edge, a))
    return mix(col, colour, strength * smooth(half + .2 * edge, half - .3 * edge, a))


def neck_stripe_y(z, x):
    """Height of the middle of the cream stripe on the neck's side at (x, z): it leaves the head at the
    mouth corner's height and runs back down the neck's side to the shell's lip."""
    from scripts.krefft_model import HEAD_RINGS
    y_head = HEAD_RINGS[0][1] - HEAD_RINGS[0][4] * .55
    return y_head - (.262 - z) * .32


def paint_neck(t):
    u, v = t["u"], t["v"]
    p = np.array(t["pos"], float)
    throat = smooth(0.62, 0.85, u)                                       # the paler, finer throat
    amount = 0.95 - 0.35 * throat
    col = skin(t["nrm"], p, 1, .0088 + .004 * (1 - throat) * smooth(0.3, 0.0, np.abs(u - .45)), amount)
    col = mix(col, _hex("#4a4e46"), 0.35 * throat)
    col = col * mix(1.2, 0.8, smooth(0.25, 0.75, u))          # the form light: the top lit
    # folds: deep where the neck leaves the shell, a softer one round its middle and behind the head
    for k, (vc, st) in enumerate(((.12, 1.0), (.45, .6), (.84, .5))):
        d = v - (vc + .08 * np.abs(u - .45) + .02 * (_n2(u * 6, v * 2, 3 + k) - .5))
        broken = 0.4 + 0.6 * smooth(0.35, 0.55, _n2(u * 5, v * .5, 70 + k)) if k else 1.0
        col = crease(col, d * .14, .0028, st * broken * smooth(0.08, 0.3, u) * smooth(1.0, 0.8, u))
    col = mix(col, CREASE, smooth(0.08, 0.0, v))                          # into the shell
    # the cream stripe, 4-5 texels and outlined, from the mouth corner down to the shell
    d = p[:, 1] - neck_stripe_y(p[:, 2], p[:, 0])
    side = smooth(0.15, 0.35, np.abs(t["nrm"][:, 0]))
    col = mix(col, JAW_LOW, 0.8 * side * smooth(0.04, 0.12, v) * smooth(0.0035, 0.0015, np.abs(d + .0105)))
    col = stripe(col, d, .0095, .0034, side * smooth(0.04, 0.12, v),
                 colour=JAW_STRIPE * (0.92 + 0.12 * fbm3(p * 90, 4))[..., None])
    return col


# the painted form light on the limbs (top lit, underside in shadow); v5 (B's clips) softens it, because an
# animated limb turns its painted underside to the light (arms raised, on his back) and read as a black slab
LIMB_LIGHT = (1.55, 0.52)
LIMB_LIGHT_V5 = (1.32, 1.0)


def paint_leg(t, front):
    u, v = t["u"], t["v"]
    p = np.array(t["pos"], float)
    n = np.array(t["nrm"], float)
    # bigger scales down the front / top of the limb (crop0), fine granules on the back and underneath
    fronts = smooth(0.5, 0.2, u) * smooth(0.15, 0.35, v)
    under = smooth(0.6, 0.9, u)
    size = mix(.0125, .0062, smooth(0.45, 0.9, v))[:, 0]                 # bigger pebbles up the limb, fine at the wrist
    amt = (0.95 - 0.3 * under) * (1 - 0.2 * under)                       # quieter in the shadowed underside
    col = skin(n, p, 2 if front else 3, size, amt)
    big = skin(n, p, 12 if front else 13, .0165 if front else .0145, 1.0)
    col = mix(col, big, 0.85 * fronts)
    # the big painted form light: the top third of the limb lit, the underside in shadow
    col = col * mix(LIMB_LIGHT[0], LIMB_LIGHT[1], smooth(0.22, 0.8, u))
    # where the leg leaves the shell, the elbow / knee (a V wrapping round the joint), the ankle (a Y)
    col = mix(col, JOINT, smooth(0.12, 0.0, v))
    inner = smooth(0.2, 0.55, u)
    d = v - (.52 + .09 * np.abs(u - .55) + .03 * (_n2(u * 8, v * 3, 30) - .5))
    col = crease(col, d * .2, .0038, 1.0 * inner, colour=JOINT, lip=_hex("#8a938c"))
    d = v - (.6 - .07 * np.abs(u - .7) + .03 * (_n2(u * 8, v * 3, 31) - .5))
    col = crease(col, d * .2, .0025, 0.7 * inner * smooth(0.4, 0.6, _n2(u * 5, v * .5, 51)), colour=JOINT)
    d = v - (.82 + .06 * np.abs(u - .35) + .025 * (_n2(u * 9, v * 4, 40) - .5))
    col = crease(col, d * .2, .0034, 1.0, colour=JOINT, lip=_hex("#8a938c"))
    br = v - (.88 + .05 * (u - .5))                                       # the Y's short branch
    col = crease(col, br * .2, .0022, 0.7 * smooth(0.45, 0.6, u) * smooth(0.95, 0.8, u), colour=JOINT)
    d = v - (.93 + .04 * np.abs(u - .5) + .02 * (_n2(u * 9, v * 4, 41) - .5))
    col = crease(col, d * .2, .002, 0.75, colour=JOINT)
    return mix(col, _hex("#262b28"), 0.5 * smooth(0.8, 1.0, v))           # the ankle flowing into the paddle


def paint_foot(t):
    """Big dark webbed paddles: scaly toes, the web (#2a2e30) between, short blunt black claws."""
    from scripts.krefft_model import FFOOT
    u, v = t["u"], t["v"]
    p = np.array(t["pos"], float)
    length, claw = FFOOT[2], FFOOT[4]
    tip = length / (length + claw)
    across = u - 0.5
    ang = np.arctan2(across, np.maximum(v + .15, 1e-3))
    ray = np.cos(ang * 6.4) * 0.5 + 0.5                                  # 1 along each toe
    col = skin(t["nrm"], p, 7, .0085, 0.95 * smooth(0.3, 0.7, ray) + 0.4, base=TOE * np.ones((len(u), 3)))
    col = mix(col, WEB, 0.9 * smooth(0.55, 0.2, ray) * smooth(0.25, 0.55, v))
    col = mix(col, UNDER, 0.9 * smooth(0.3, 0.1, ray) * smooth(tip - .2, tip - .05, v))      # the web's notches
    for vc in (.12, .2):                                                   # wrinkles over the wrist
        col = crease(col, (v - vc - .02 * np.sin(u * 9)) * .09, .0016, 0.7)
    clawz = smooth(tip - .2, tip - .08, v) * smooth(0.4, 0.62, ray)
    col = mix(col, CLAW, clawz)                                            # five short black claws
    col = mix(col, _hex("#4a4c46"), 0.7 * clawz * smooth(0.8, 0.95, ray) * smooth(tip - .1, tip, v))         # a glint along the top
    col = mix(col, _hex("#7a6e54"), 0.8 * clawz * smooth(0.75, 0.92, ray) * smooth(tip + .06, tip + .14, v))  # worn tips (as in crop0)
    return col


def paint_sole(t):
    col = _hex("#2a2c2a") * (0.85 + 0.3 * _n2(t["u"] * 14, t["v"] * 14, 8))[..., None]
    return mix(col, CLAW, smooth(0.75, 0.9, t["v"]))


def paint_tail(t):
    col = skin(t["nrm"], t["pos"], 9, .008, 0.9)
    return mix(col, CREASE, smooth(0.2, 0.0, t["v"]))


# ------------------------------------------------------------------------------------------- head

HEAD_PLATE, HEAD_PLATE_HI, HEAD_PLATE_EDGE = _hex("#66715f"), _hex("#7e886f"), _hex("#56604f")
CROWN = _hex("#4a5347")
SNOUT_PALE = _hex("#9aa48f")
RIDGE_HI = _hex("#b4bea8")
OCC = _hex("#3a4138")
BROW_SH = _hex("#2f362c")
CHEEK, CHEEK_LO = _hex("#8a937e"), _hex("#76806c")
LIP_CREAM = _hex("#cfc6a0")
THROAT = _hex("#a4a28a")
IRIS, IRIS_HI = _hex("#a6b6a2"), _hex("#c4d4c0")
EYE_RING = _hex("#2a2f2b")
PUPIL = _hex("#0f1110")
GLINT = _hex("#ffffff")
LID_SKIN = _hex("#5a605a")
UPPER_LID = _hex("#4a5450")
LID_PLATE = _hex("#6f7a68")
LID_LINE = _hex("#1a1c1a")
SCAR, SCAR_SHADOW = _hex("#c9c3a8"), _hex("#3a4138")
BROW_STRIPE = _hex("#c9c9a8")
JAW_STRIPE, JAW_STRIPE_HI = _hex("#d8cf9e"), _hex("#f0e8c0")
JAW_LOW = _hex("#b0a676")
NECK_TONE = _hex("#3a403d")
MOUTH_LINE, MOUTH_HI = _hex("#2a2620"), _hex("#8c8a70")
LASH = _hex("#2a2e2c")
MOUTH_IN = _hex("#2a1412")
NOSTRILS = [np.array([.011, .1438, .4535]), np.array([-.011, .1438, .4535])]


def _lip_y(z):
    from scripts.krefft_model import HEAD_RINGS, SMILE
    zs = [r[0] for r in HEAD_RINGS]
    ys = [r[1] - r[4] + SMILE[i] for i, r in enumerate(HEAD_RINGS)]
    return np.interp(z, zs, ys)


def _eye_ab(p, s):
    """(along toward the snout, up) / EYE_R from eye s's centre, in its own plane."""
    from scripts.krefft_model import EYES, EYE_R
    c, nrm, along, up = EYES[s]
    d = p - c
    return d @ along / EYE_R, d @ up / EYE_R


def _scar(col, p):
    """The old scar on his right: a thick pale welt (#c8beb0, 3-4 texels) at 35 degrees from the brow
    (up, toward the snout) down over the lid and the cheek behind the eye, a #6a665e shadow under it."""
    from scripts.krefft_model import SCARRED
    a, b = _eye_ab(p, SCARRED)
    on = smooth(.018, .026, p[:, 0] * SCARRED) * smooth(2.3, 2.0, np.hypot(a, b))   # round the eye only
    ang = np.radians(55)
    al = (a + .0) * np.cos(ang) + (b + .05) * np.sin(ang)                 # along the welt
    ac = -(a + .0) * np.sin(ang) + (b + .05) * np.cos(ang) + .08 * np.sin(al * 1.4) + .02 * np.sin(al * 9.0)
    reach = on * smooth(1.55, 1.35, al) * smooth(-1.9, -1.65, al) * (0.75 + 0.25 * smooth(1.4, 0.4, np.abs(al)))
    w = 0.15 * (1 - 0.45 * smooth(0.3, 1.9, np.abs(al)))                  # thicker in the middle, tapering
    col = mix(col, SCAR_SHADOW, 0.95 * reach * smooth(w + .14, w + .04, np.abs(ac + w + .02)))
    col = mix(col, SCAR_SHADOW, 0.8 * reach * smooth(.1, .03, np.abs(ac - w - .05)))      # a dark line each side
    col = mix(col, SCAR, reach * smooth(w + .02, w - .06, np.abs(ac)))
    col = mix(col, _hex("#d4d2c4"), 0.5 * reach * smooth(0.06, 0.0, np.abs(ac - w * .45)))   # lit on its top
    for k in ():                                                          # (no stitch marks: they read as noise)
        col = mix(col, SCAR_SHADOW, 0.8 * on * smooth(0.06, 0.025, np.abs(al - k)) * smooth(w + .22, w + .12, np.abs(ac)) * smooth(w + .02, w + .08, np.abs(ac)))
    return col


def head_skin(t):
    """The skull's and the jaw's planar islands, painted where they are in 3D (so the sides, the plate,
    the snout and the chin meet up across the seams)."""
    from scripts.krefft_model import EYE_YZ, HEAD_RINGS, SCARRED
    p = np.array(t["pos"], float)
    n = np.array(t["nrm"], float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ax = np.abs(x)
    up = n[:, 1]
    lip = _lip_y(z)
    # the form, lit from above: the plate on top, the cheeks below, darker toward the jaw
    jaw = (np.array(t["along"], float) > .5)                              # the jaw's vertices carry along = 1
    nm = n.copy()
    nm[:, 0] = np.abs(nm[:, 0])
    # painted light on a smooth rounded head (an ellipsoid's normals), not on the facets: the DK2 way
    en = _unitrows(np.stack([x / .1 ** 2, (y - .14) / .075 ** 2, (z - .35) / .13 ** 2], 1))
    L2 = np.array([-.2, 1.0, .45]) / np.linalg.norm([-.2, 1.0, .45])
    en_m = en.copy()
    en_m[:, 0] = np.abs(en_m[:, 0])
    shade = np.clip(0.7 + 0.42 * (en_m @ L2) + 0.08 * (nm @ LIGHT), 0.42, 1.08)
    side = mix(CHEEK, CHEEK_LO, smooth(lip + .004, lip + .03, y) * 0.5)  # paler toward the jaw
    side = mix(side, HEAD_PLATE, smooth(lip + .02, lip + .045, y))
    # the head plate: everything above a line just over the eyes, sloping down the snout's sides
    pl_line = EYE_YZ[0] + .004 - .28 * np.maximum(z - EYE_YZ[1] - .01, 0) + .5 * np.maximum(.30 - z, 0)
    plate = smooth(-0.008, 0.01, y - pl_line)                            # fading, no hard edge
    pc = mix(HEAD_PLATE_EDGE, HEAD_PLATE, smooth(0.0, 0.012, y - pl_line))
    crown = smooth(.43, .37, z) * smooth(.085, .05, ax)                   # the crown, the top third: darker
    pc = mix(pc, CROWN, 0.95 * crown)
    csp = smooth(0.74, 0.8, fbm3(p * 240, 16, 1)) * smooth(0.5, 0.62, fbm3(p * 35, 19, 2)) * crown
    pc = mix(pc, _hex("#3f4944"), 0.9 * csp)
    pc = mix(pc, SNOUT_PALE, 0.7 * smooth(.39, .45, z))                   # the snout paler toward its tip
    pc = pc * (0.84 + 0.32 * fbm3(p * 45, 12, 3))[..., None]
    spots = smooth(0.58, 0.7, fbm3(p * 110, 13, 2))                       # a fine darker freckling
    pc = mix(pc, HEAD_PLATE_EDGE, 0.35 * spots)
    col = mix(side, pc, plate)
    # a few fine fissures in the plate, and fine granules on the cheeks and the back of the head
    fis = smooth(0.012, 0.0, np.abs(fbm3(p * np.array([30, 30, 12]), 14) - .5)) * smooth(0.5, 0.7, fbm3(p * 9, 15))
    col = mix(col, HEAD_PLATE_EDGE, 0.5 * fis * plate)
    gran = (1 - plate) * (0.1 + 0.45 * smooth(.33, .29, z))
    gran = np.maximum(gran, 0.85 * smooth(.3, .265, z))                  # into the neck's pebbles over 7 texels
    col = mix(col, NECK_TONE, 0.8 * smooth(.29, .265, z))
    col = skin(n, p, 11, .0068, gran * 0.7, mirrored=False, base=col)
    col = mix(col, SNOUT_PALE, 0.45 * (1 - plate) * smooth(lip + .03, lip + .008, y) * ~jaw)   # pale cheeks
    col = col * shade[..., None] / 0.95
    # the snout's ridge caught by the light, a few pale speckles on the cheeks
    col = mix(col, RIDGE_HI, 0.55 * np.exp(-(x / .018) ** 2 - ((z - .425) / .014) ** 2) * smooth(.5, .9, up) * ~jaw)
    clus = smooth(0.5, 0.65, fbm3(p * 40, 17, 2))                         # speckles in clusters
    spk = smooth(0.72, 0.8, fbm3(p * 260, 18, 1)) * clus * smooth(lip + .006, lip + .012, y) * ~jaw * smooth(.3, .33, z)
    col = mix(col, _hex("#6b7565"), 0.9 * spk)
    gr = skin(n, p, 23, .0058, 0.35 * plate, mirrored=False, base=col)     # a fine grain on the plate too
    col = gr
    for k_, zc_ in enumerate((.279, .288, .297)):                         # skin folds where the head meets the neck
        dz_ = z - (zc_ + .006 * np.sin(y * 300 + k_) + .01 * (ax / .08) ** 2)
        on_ = smooth(.03, .05, ax) * (1 - 0.5 * k_ / 2)
        col = mix(col, _hex("#a8b29c"), 0.5 * on_ * smooth(.0022, .0008, np.abs(dz_ - .0028)))
        col = mix(col, OCC, 0.8 * on_ * smooth(.0018, .0006, np.abs(dz_)))
    # soft occlusion behind the eye bulges and where the head meets the neck
    for s_ in (1, -1):
        a_, b_ = _eye_ab(p, s_)
        on_ = (x * s_ > .02)
        col = mix(col, OCC, 0.55 * on_ * np.exp(-((a_ + 1.25) ** 2 / .25 + b_ ** 2 / .8)))
        col = mix(col, BROW_SH, 0.75 * on_ * smooth(0.22, 0.08, np.abs(np.hypot(a_ * .9, b_) - 1.2)) * smooth(0.35, 0.75, b_))
    col = mix(col, OCC, 0.6 * smooth(.29, .27, z))
    # the back of the head: into the neck's skin; a dark line where the neck goes in
    col = mix(col, CREASE, 0.8 * smooth(.275, .262, z) * ~jaw)
    skull_col = _skull_marks(col.copy(), p, x, y, z, ax, lip)
    jaw_col = _jaw_marks(col.copy(), p, n, y, ax, lip, up)
    return np.where(jaw[:, None], jaw_col, skull_col)


def _skull_marks(col, p, x, y, z, ax, lip):
    from scripts.krefft_model import HEAD_RINGS
    # the upper lip: a lit ridge, then the mouth line (a 2-texel dark line) curling up at the corner
    if True:
        d = y - lip
        front = smooth(.326, .336, z)
        lipw = .0095 + .007 * smooth(.38, .34, z)                          # the cream lip broadens toward the corner
        lift = np.where(x > 0, .0065 * smooth(.365, .329, z), 0.0)        # the smirk: up on the tooth side only
        col = mix(col, JAW_STRIPE, 0.85 * front * smooth(lipw + lift, lipw + lift - .003, d) * smooth(lift + .003, lift + .0045, d))
        col = mix(col, MOUTH_LINE, 0.97 * front * smooth(0.0048, 0.0034, d - lift) * smooth(-.0035, -.0015, d - lift))
        col = mix(col, LIP_CREAM, 0.85 * front * (x > 0) * smooth(-.0015, -.003, d - lift) * smooth(0.0, 0.0012, d))
        # the smug curl: the line runs on past the corner, up and back (higher on the tooth side)
        cz = HEAD_RINGS[1][0] + .002
        curl_y = lip + .003 + np.where(x > 0, .0065, 0.0) + (cz - z) * np.where(x > 0, 1.9, .2)   # the smirk's curl
        on = smooth(cz + .003, cz - .001, z) * np.where(x > 0, smooth(cz - .026, cz - .02, z), smooth(cz - .008, cz - .004, z))
        col = mix(col, MOUTH_LINE, 0.95 * on * smooth(0.0045, 0.003, np.abs(y - curl_y)))
        tipz = cz - .021                                                   # the dimple at the corner's end
        dd = np.hypot((z - tipz) * .7, y - (lip + .0095 + .021 * 1.9))
        col = mix(col, _hex("#b9c2a8"), 0.7 * smooth(.0045, .002, np.hypot((z - tipz) * .7, y - (lip + .0095 + .021 * 1.9) - .005)) * (x > 0))
        col = mix(col, _hex("#2a302c"), 0.95 * smooth(.006, .0035, dd) * (x > 0))
        # the cream stripe from the mouth corner back along the jaw to the neck
        sy = neck_stripe_y(z, x)
        sy = np.where(z > .262, np.interp(z, [.262, cz], [neck_stripe_y(.262, 0), lip_corner_y()]), sy)
        st = smooth(cz + .004, cz - .002, z) * (ax > .02)
        col = stripe(col, y - sy, .011, .0034, st, colour=JAW_STRIPE)
        col = mix(col, JAW_LOW, 0.8 * st * smooth(0.0035, 0.0015, np.abs(y - sy + .0085)))
        # the faint pale stripe behind the eye
        for s in (1, -1):
            a, b = _eye_ab(p, s)
            on = (x * s > .02) * smooth(-1.2, -1.45, a) * smooth(-3.4, -2.8, a)
            col = mix(col, BROW_STRIPE, 0.9 * on * smooth(0.24, 0.14, np.abs(b + .05 - .05 * (a + 1.2))))
            # the socket: a dark ring round the dome, heavier under the brow
            r = np.hypot(a, b)
            ring_k = smooth(-0.5, 0.4, b - 0.4 * a)                    # broken under and in front of the eye
            col = mix(col, OCC, 0.8 * (x * s > .02) * ring_k * smooth(1.3, 1.02, r) * smooth(0.9, 1.0, r))
            fold = smooth(0.12, 0.04, np.abs(np.hypot((a + 1.7) * .8, b * 1.1) - .55)) * smooth(-1.2, -1.6, a)
            col = mix(col, OCC, 0.6 * (x * s > .02) * fold)
            col = mix(col, HOLLOW, 0.45 * (x * s > .02) * smooth(0.2, 0.08, np.abs(r - 1.25)) * smooth(0.3, 0.8, b))   # a soft crease over the eye
        for nn in NOSTRILS:
            q = p - nn
            dn = np.sqrt(q[:, 0] ** 2 / 1.9 + (q[:, 1] ** 2 + q[:, 2] ** 2) / 1.0)   # 3 wide x 2 high
            col = mix(col, _hex("#b4bea8"), 0.8 * smooth(.003, .0015, np.linalg.norm((q - np.array([0, .0032, -.0022])) * np.array([.55, 1, 1]), axis=1)))
            col = mix(col, _hex("#2a2f28"), 0.95 * smooth(.0028, .0016, dn))
        col = _scar(col, p)               # one welt: brow, across the lid, onto the cheek
    return col


def _jaw_marks(col, p, n, y, ax, lip, up):
    if True:
        # the jaw: a pale cream lower lip under the mouth line, grey below, the paler throat underneath
        d = lip - y
        col = mix(col, mix(col, THROAT, 0.7), smooth(-0.3, -0.8, up))
        col = stripe(col, d - .0098, .0082, .0032, smooth(0.3, 0.6, ax / .09 + 0.3),
                     colour=JAW_STRIPE)
        col = mix(col, JAW_LOW, 0.8 * smooth(0.003, 0.0012, np.abs(d - .0175)) * smooth(0.3, 0.6, ax / .09 + 0.3))
        col = mix(col, MOUTH_LINE, 0.95 * smooth(0.0022, 0.0008, d))
    return col


def lip_corner_y():
    from scripts.krefft_model import HEAD_RINGS
    return _lip_y(HEAD_RINGS[1][0]) - .004


def paint_eye(t, s):
    """The eye on its dome. The dome is lid skin (the head plate's olive grey, lit on top) with the eye
    opening in its middle: a silvery-green iris (#97a690 to #b4c2ac), a dark ring, a round pupil and a
    glint, a dark rim round the opening. On his left a heavy upper lid comes down over the top of the
    opening, sloping toward the snout (a glower); on his right the modelled lid covers 70% and only a
    sliver of the iris shows under it, the pupil low."""
    from scripts.krefft_model import SCARRED
    p = np.array(t["pos"], float)
    a, b = _eye_ab(p, s)
    r = np.hypot(a, b)
    skin_c = mix(_hex("#3a4138"), _hex("#7a846e"), smooth(-0.95, 0.6, b))  # a soft lit dome, shadowed under
    col = mix(IRIS, IRIS_HI, smooth(-0.4, 0.5, b))
    col = mix(col, EYE_RING, smooth(0.58, 0.68, r))
    opening = smooth(0.78, 0.72, r)
    if s == SCARRED:
        py = -.3
        col = mix(col, PUPIL, smooth(0.22, 0.15, np.hypot(a - .05, b - py)))
        col = mix(col, GLINT, 0.9 * smooth(0.09, 0.04, np.hypot(a - .2, b + .2)))
        lid = -.12 - .3 * a
    else:
        col = mix(col, PUPIL, smooth(0.27, 0.2, np.hypot(a - .05, b + .1)))
        col = mix(col, GLINT, 0.95 * smooth(0.1, 0.045, np.hypot(a - .22, b + .2)))
        lid = -.06 - .47 * a                                               # heavy, cutting the iris's top half, sloping to the snout
    opening = opening * smooth(lid + .05, lid - .03, b)
    if s == SCARRED:                                                       # in the half-shut lid's shadow
        skin_c = mix(skin_c, _hex("#3e463b"), smooth(lid - .1, lid + .1, b))
    col = mix(skin_c, col, opening)
    rim = smooth(0.1, 0.03, np.abs(r - .75)) + smooth(0.1, 0.03, np.abs(b - lid)) * smooth(0.85, 0.72, r)
    col = mix(col, LID_LINE, 0.9 * np.clip(rim, 0, 1))
    upper = smooth(lid, lid + .25, b) * smooth(0.95, 0.8, r) * (s != SCARRED)
    col = mix(col, _hex("#3e463b"), 0.85 * upper)                         # the heavy upper lid
    col = mix(col, _hex("#9aa48f"), 0.7 * smooth(0.07, 0.02, np.abs(b - lid - .1)) * smooth(0.7, 0.55, r) * (s != SCARRED))
    col = mix(col, _hex("#22281f"), 0.9 * smooth(0.09, 0.03, np.abs(b - lid)) * smooth(0.72, 0.6, r) * (s != SCARRED))
    col = mix(col, RIDGE_HI, 0.5 * smooth(0.75, 0.95, b) * smooth(1.0, 0.8, r))   # a highlight over the bulge
    return mix(col, HOLLOW, 0.6 * smooth(0.92, 1.0, r))


def paint_lid(t):
    """The scarred eye's modelled lid: heavy slate, lit on top, its rim a dark line, the welt across it."""
    p = np.array(t["pos"], float)
    v = np.array(t["along"], float)
    col = mix(HEAD_PLATE * .95, HEAD_PLATE * .72, smooth(0.0, 1.0, v))    # the head's own olive, darker to the rim
    col = skin(t["nrm"], p, 17, .0055, 0.35, mirrored=False, base=col)
    col = _scar(col, p)
    col = mix(col, LASH, smooth(0.72, 0.9, v))
    return mix(col, _hex("#22281f"), smooth(0.86, 0.96, v))


def paint_under(t):
    """The chin and the throat from below: pale."""
    p = np.array(t["pos"], float)
    col = THROAT * (0.9 + 0.15 * fbm3(p * 50, 19))[..., None]
    col = skin(t["nrm"], p, 19, .006, 0.45, mirrored=False, base=col)
    return mix(col, CREASE, 0.7 * smooth(.29, .27, p[:, 2]))


def paint_mouth(t):
    return MOUTH_IN + 0 * t["u"][:, None]


def gold(t, seed, dull=0.0):
    p = sym(t["pos"])
    g = fbm3(p * 60, seed)
    col = mix(C["gold_dk"], C["gold"], 0.35 + 0.65 * g)
    col = mix(col, C["gold_dk"] * 0.8, dull * smooth(0.5, 0.7, fbm3(p * 30, seed + 1)))
    return col


def paint_tooth(t):
    """Flat #f0c040 gold, a #fff0a0 top edge, the chipped end a dark #6a4a10 notch."""
    along = t["along"]
    nrm = np.array(t["nrm"], float)
    axis = np.array([.08, -1.0, .04]) / np.linalg.norm([.08, -1.0, .04])
    col = TOOTH + 0 * along[:, None]
    col = mix(col, TOOTH_HI, 0.8 * smooth(0.25, 0.0, along))
    col = mix(col, _hex("#fff2a0"), 0.9 * smooth(0.6, 0.85, nrm @ np.array([.6, .2, .77])))   # a highlight down the front
    col = mix(col, TOOTH * .85, 0.5 * smooth(0.2, -0.4, nrm @ LIGHT))
    return mix(col, TOOTH_LO, 0.95 * smooth(0.5, 0.8, nrm @ axis))


def paint_chain(t):
    """Battered gold chain round the neck: u along it (one ring = 0..1), v across (0 top). Oval links
    alternate flat (a gold ring round a dark hole) and edge-on (a narrow gold oval), overlapping, a thin
    dark gap between them; two dented, darker links for the wear."""
    u, v = t["u"], t["v"]
    n_links = 14
    k = u * n_links
    f = k - np.floor(k)
    odd = (np.floor(k) % 2) == 1
    ox, oy = (f - 0.5) / 0.62, (v - 0.5) / 0.46
    rr = np.hypot(ox, oy)
    flat = smooth(1.0, 0.86, rr)
    hole = smooth(0.48, 0.34, np.hypot(ox * 1.1, oy))
    edge = smooth(1.0, 0.8, np.hypot(ox * 2.2, oy))                       # an edge-on link: a narrow oval
    shape = np.where(odd, edge, flat)
    gold = mix(_hex("#b8902a"), _hex("#e0b040"), smooth(0.4, -0.6, oy))   # lit along the top
    gold = gold * (0.9 + 0.2 * fbm3(sym(t["pos"]) * 60, 41))[..., None]
    col = mix(CHAIN_GAP, gold, np.clip(shape, 0, 1))
    col = mix(col, _hex("#3a2a08"), np.where(odd, 0.0, hole))              # the flat links' holes
    hi = np.where(odd, np.exp(-((oy + .45) / .2) ** 2) * edge, flat * smooth(0.6, 0.85, rr) * smooth(0.1, -0.5, oy))
    col = mix(col, _hex("#f0d070"), 0.85 * np.clip(hi, 0, 1))
    worn = np.isin(np.floor(k).astype(int), (3, 10))
    col = mix(col, _hex("#6a5018"), 0.75 * worn * np.clip(shape, 0, 1))
    dent = smooth(0.62, 0.72, fbm3(sym(t["pos"]) * 90, 43))
    col = mix(col, _hex("#6a5018"), 0.45 * dent * shape)
    return lit(col, t["nrm"], sheen=0.4, power=6) * 0.95


# v4: one gold ramp for the coin, the chain and the medallion
GOLD_SH, GOLD_MID, GOLD_LT, GOLD_HI = _hex("#5a3a10"), _hex("#b07a22"), _hex("#e8c05a"), _hex("#fff0b0")


def paint_chain_v4(t):
    """v4: a fine gold chain (jewellery): u along the loop, v across. Interlocking oval links: flat links
    (a gold oval ring round a dark hole) and, between them, edge-on links (a narrow bright bar) laid
    over the flat links' ends; the dark gaps between them; a bright streak along each link's lit side."""
    from scripts.krefft_standing import CHAIN_LINKS
    u, v = t["u"], t["v"]
    s_ = u * CHAIN_LINKS                                   # in links
    cf = np.round((s_ - .5) / 2) * 2 + .5                  # the nearest flat link's centre
    ce = np.round((s_ - 1.5) / 2) * 2 + 1.5                # the nearest edge-on link's centre
    fx, fy = (s_ - cf) / .78, (v - .5) / .46
    ex, ey = (s_ - ce) / .74, (v - .5) / .17
    rf, re = np.hypot(fx, fy), np.hypot(ex, ey)
    ring = smooth(1.0, 0.86, rf) * smooth(0.4, 0.54, rf)
    bar = smooth(1.0, 0.8, re)
    G_HI, G_MID, G_DK = GOLD_HI, GOLD_LT, GOLD_MID
    flat = mix(G_DK, G_MID, smooth(0.7, -0.5, fy))                   # lit along its upper side
    flat = mix(flat, G_HI, 0.85 * smooth(-0.35, -0.6, fy) * smooth(0.95, 0.75, rf))
    edge = mix(G_MID, G_HI, smooth(0.6, 0.0, np.abs(ey + .25)))       # a streak along the bar
    edge = mix(edge, G_DK, 0.7 * smooth(0.55, 0.95, np.abs(ex)))       # darker where it dives under
    col = mix(_hex("#3a2606"), _hex("#6a4810"), smooth(0.9, 0.0, np.abs(v - .5) * 2)) + 0 * u[:, None]
    col = mix(col, flat, np.clip(ring, 0, 1))
    col = mix(col, _hex("#0e0a04"), 0.9 * smooth(0.5, 0.3, rf) * (1 - bar))     # the holes
    col = mix(col, _hex("#2a1c06"), 0.6 * smooth(1.25, 1.0, re) * (1 - bar))  # a shadow round each bar
    col = mix(col, edge, np.clip(bar, 0, 1))
    col = col * (0.94 + 0.12 * fbm3(sym(t["pos"]) * 70, 47))[..., None]
    return lit(col, t["nrm"], sheen=0.45, power=6) * 1.0


# ------------------------------------------------------------------------------------------- shell

R_MARGINAL = .8              # where the marginal scutes' ring starts (fraction of the outline)
N_MARGINAL = 12             # marginals a side


def _scute_seeds():
    """Vertebral (5 down the keel) and costal (4 a side, between them) scutes' centres (x, z)."""
    from scripts.krefft_model import SHELL_A, SHELL_B
    ka, kb = SHELL_A / .215, (SHELL_B[0] + SHELL_B[1]) / .57
    out = [(dx, z * kb) for dx, z in ((.004, .175), (-.006, .088), (.008, -.01), (-.004, -.098), (.006, -.19))]
    for s in (1, -1):
        for dz, z in zip((.006, -.008, .01, -.004) if s > 0 else (-.004, .01, -.006, .008), (.125, .035, -.06, -.155)):
            out.append((s * .112 * ka + .006 * s * (dz > 0), (z + dz) * kb))
    return np.array(out)


SEEDS = _scute_seeds()


def scutes(xz):
    """-> (distance to the nearest seam, distance to the scute's centre, scute id) on the top view: a
    Voronoi of the vertebrals and costals inside the marginal ring, the marginals as a ring of
    N_MARGINAL a side (the nuchal centred over the neck)."""
    from scripts.krefft_model import SHELL_A, SHELL_B, SHELL_Z
    x, z = xz[:, 0], xz[:, 1]
    dz = z - SHELL_Z
    b = np.where(dz >= 0, SHELL_B[0], SHELL_B[1])
    phi = np.arctan2(x / SHELL_A, dz / b)
    wide = 1 + .05 * -np.cos(phi)
    rn = np.hypot(x / (SHELL_A * wide), dz / b)
    R = SHELL_A                                                # about the shell's radius, for distances
    d = np.linalg.norm((xz[:, None, :] - SEEDS[None]) * np.array([.85, 1.0]), axis=2)
    o = np.argsort(d, axis=1)
    f1 = d[np.arange(len(d)), o[:, 0]]
    f2 = d[np.arange(len(d)), o[:, 1]]
    edge_in = np.minimum((f2 - f1) * .5, np.abs(rn - R_MARGINAL) * R)
    step = 2 * np.pi / (2 * N_MARGINAL)
    k = phi / step - .5
    dk = np.abs(k - np.round(k))
    edge_m = np.minimum(dk * step * R * rn, np.abs(rn - R_MARGINAL) * R)
    mid = np.hypot((rn - (1 + R_MARGINAL) / 2) * R, (0.5 - dk) * step * R)
    marg = rn > R_MARGINAL
    edge = np.where(marg, edge_m, edge_in)
    cen = np.where(marg, mid, f1)
    sid = np.where(marg, 100 + np.round(k).astype(int), o[:, 0])
    return edge, cen, sid


DUCKWEED = [(-.16, -.21, .046), (.13, -.24, .038), (-.02, -.285, .03)]  # near the edge and the seams   # (x, z, radius): big clumps, trailing off the edge


def paint_carapace(t, lips=True):
    from scripts.krefft_model import COINS
    p = np.array(t["pos"], float)
    xz = p[:, [0, 2]]
    # wobble the seams a little so they wander like the real ones
    w = np.stack([fbm3(p * 25, 51), fbm3(p * 25 + 7, 52)], 1) - 0.5
    edge, f1, idx = scutes(xz + w * 0.012)
    rnd = ((np.sin(idx * 12.9898) * 43758.5453) % 1.0 + 1.0) % 1.0
    col = mix(_hex("#5a5a38"), _hex("#8a8a5a"), 0.35 + 0.65 * fbm3(p * 9, 53))
    col *= (0.88 + 0.22 * rnd)[..., None]                                 # each scute its own shade
    # the areola: a lighter mottled patch in each scute, faint growth rings round it
    centre = smooth(0.06, 0.0, f1)
    col = mix(col, C["areola"], 0.45 * centre * smooth(0.35, 0.65, fbm3(p * 30, 54)))
    rings = _bands(f1 + 0.004 * fbm3(p * 50, 55), 0.011, 0.14)
    col = mix(col, C["khaki_dk"], 0.25 * rings * smooth(0.01, 0.03, f1))
    col = mix(col, C["khaki_lt"], 0.4 * smooth(0.5, 0.75, fbm3(p * 18, 56)))      # mottling
    col = mix(col, C["khaki_dk"], 0.35 * smooth(0.55, 0.8, fbm3(p * 26, 64)))     # darker blotches
    col *= (0.8 + 0.4 * fbm3(p * 110, 57, 2))[..., None]                          # fine pitted grain
    # each scute a shallow dome: darker toward its seams (the crevices), lit across its middle
    col *= (0.74 + 0.26 * smooth(0.0, 0.03, edge))[..., None]
    col = mix(col, C["khaki_lt"], 0.18 * smooth(0.035, 0.0, f1))
    # the marginals: lighter mottled patches (as in his photos)
    from scripts.krefft_model import SHELL_A as _A
    marg = idx >= 100
    col = mix(col, MARGINAL, 0.55 * marg * smooth(0.45, 0.7, fbm3(p * 30, 66)))
    # seams: dark, lit on the lower lip (a groove), slime caught in them
    seam = smooth(0.0035, 0.0, edge)
    lip = smooth(0.008, 0.0035, edge) * (1 - seam)
    col = mix(col, C["khaki_lt"] * 1.1, 0.35 * lip)
    slime_in = smooth(0.45, 0.65, fbm3(p * 14, 58))
    vary = 0.5 + 0.8 * fbm3(p * 40, 67)                                       # seams of varied width
    seam = smooth(0.0045 * vary, 0.0, edge)
    col = mix(col, mix(SCUTE_SEAM, C["slime"], 0.4 * slime_in), 0.9 * seam)
    col = mix(col, SCUTE_CENTRE, 0.35 * smooth(0.04, 0.0, f1))
    # scuffs: pale scratches
    sc = smooth(0.012, 0.0, np.abs(fbm3(p * np.array([60, 8, 22]), 59) - 0.5)) * smooth(0.55, 0.7, fbm3(p * 9, 60))
    col = mix(col, C["scuff"], 0.35 * sc)
    # the cracks the coins are wedged in: dark, jagged, lit on one lip
    import scripts.krefft_model as km
    if km.V4:
        col = _crack_v4(col, p, xz, lips)
    for (cx, cz), r, yaw, prop in (() if km.V4 else COINS):          # the crack each coin is wedged in, centred under it
        a = np.radians(yaw)
        ax = np.array([np.cos(a), -np.sin(a)])
        perp = np.array([-ax[1], ax[0]])
        d = xz - np.array([cx, cz])
        along = d @ ax
        across = d @ perp + 0.0028 * np.sin(along * 260) + 0.0016 * np.sin(along * 610 + 1.0)
        reach = smooth(r * 1.85, r * 1.5, np.abs(along))                  # out past the rim both ways
        width = 0.0062 * (1.0 - 0.45 * smooth(r * 1.1, r * 1.8, np.abs(along)))     # tapering to its ends
        col = mix(col, C["khaki_lt"] * 1.15, 0.5 * reach * smooth(width * 1.9, width * 1.2, np.abs(across + width * .9)))
        col = mix(col, CRACK, 0.97 * reach * smooth(width, width * .5, np.abs(across)))
        rr = np.hypot(d[:, 0], d[:, 1])                                    # a contact shadow round the rim
        col = mix(col, _hex("#4e5636"), 0.7 * smooth(r * 1.12, r * 0.98, rr))
    # slime: dark green smears toward the back and in patches
    slime = smooth(0.55, 0.72, fbm3(p * 7, 61)) * (0.5 + 0.5 * smooth(0.0, -0.2, p[:, 2]))
    col = mix(col, C["slime"], 0.55 * slime)
    col = lit(col, t["nrm"], sheen=0.1, mirrored=False) * 1.1               # the tops a little lighter
    # duckweed: clumps of little leaves (3-5 texels each, a dark heart, a lit edge) with a few
    # stragglers trailing off along the seams
    for ci, (cx, cz, rr) in enumerate(DUCKWEED):
        rng = np.random.default_rng(80 + ci)
        n_leaf = 9 + 3 * (ci % 2)
        for k in range(n_leaf):
            ang, dist = rng.uniform(0, 2 * np.pi), rr * rng.uniform(0, 1) ** 1.4 * (1.0 if k < n_leaf - 2 else 1.7)
            lx, lz = cx + dist * np.cos(ang), cz + dist * np.sin(ang)
            br = rng.uniform(.007, .011)
            dx, dz = p[:, 0] - lx, p[:, 2] - lz
            d = np.hypot(dx, dz * 1.15) / br
            body = smooth(1.0, 0.8, d)
            col = mix(col, WEED_SHADOW, 0.7 * smooth(1.35, 1.0, d) * (1 - body))
            lit_side = np.clip((dx * -.3 + dz * .9) / br, -1, 1)
            bc = mix(_hex("#5c6e2a"), _hex("#7a8f3a"), 0.35 + 0.6 * smooth(-0.8, 0.5, lit_side))
            bc = mix(bc, _hex("#3f6a20"), 0.7 * smooth(rr * .55, 0.0, np.hypot(p[:, 0] - cx, p[:, 2] - cz)))   # the clump's core
            col = mix(col, bc, body)
            col = mix(col, WEED_FLECK, 0.8 * body * smooth(0.3, 0.8, lit_side))
    return col


def _crack_v4(col, p, xz, lips=True):
    """v4: the one crack, splitting the plates right through the coin's scute: a dark split, widest
    where the coin stands in it (the gap the lips frame), running out jagged past the lips both ways
    with a short branch each end; a pale chipped edge on its lit lip; the coin's shadow on the shell
    on the side it leans to."""
    from scripts.krefft_model import COIN_V4, LIP_LEN
    (cx, cz), r, yaw = COIN_V4
    a = np.radians(yaw)
    ax = np.array([np.cos(a), -np.sin(a)])                # (x, z): along the crack (the model's ax)
    perp = np.array([-ax[1], ax[0]])
    d = xz - np.array([cx, cz])
    along = d @ ax
    jag = 0.0032 * np.sin(along * 240) + 0.0018 * np.sin(along * 570 + 1.0) + 0.0012 * np.sin(along * 1300)
    across = d @ perp + jag * smooth(r * .9, r * 1.4, np.abs(along))
    end = r * 3.1
    reach = smooth(end, end * .8, np.abs(along))
    width = 0.0085 * (1.0 - 0.75 * smooth(r * 1.1, end, np.abs(along)))          # tapering to hairline
    col = mix(col, C["khaki_lt"] * 1.18, 0.6 * reach * smooth(width * 2.1, width * 1.3, np.abs(across + width * 1.1)))
    col = mix(col, _hex("#3a3420"), 0.6 * reach * smooth(width * 2.4, width * 1.2, np.abs(across)))    # bruised edges
    col = mix(col, CRACK, 0.98 * reach * smooth(width, width * .55, np.abs(across)))
    for sgn in (1, -1):                                    # a short branch off each end
        o = np.array([cx, cz]) + ax * sgn * r * 1.9
        bd = np.array([np.cos(a + sgn * .7), np.sin(a + sgn * .7)])
        q = xz - o
        bal = q @ bd * sgn
        bac = q @ np.array([-bd[1], bd[0]]) + 0.002 * np.sin(bal * 400)
        on = smooth(0, .004, bal) * smooth(r * 1.1, r * .8, bal)
        col = mix(col, CRACK, 0.9 * on * smooth(0.0035, 0.0018, np.abs(bac)))
    # where the coin stands: the split opened wide and black, and the coin's shadow on the side it leans
    # to (the model leans it away from the keel)
    lean = np.sign(np.array([cx, cz + .02]) @ perp)       # (+ perp or - perp: away from the keel)
    if lips:
        hole = smooth(r * LIP_LEN, r * LIP_LEN * .7, np.abs(along))
        col = mix(col, _hex("#0a0806"), 0.95 * hole * smooth(0.016, 0.008, np.abs(across)))
    sh = smooth(r * 1.15, r * .6, np.abs(along)) * smooth(0, .004, across * lean) * smooth(r * .95, r * .3, across * lean)
    col = mix(col, _hex("#1e1a10"), 0.85 * sh)
    return col


from scripts.krefft_model import RIM_Y


def paint_shell_under(t):
    """The plastron and the bridge: #d8c890 falling off to #8a7c50 toward the back (the standing B's
    bottom), seams 2 texels #9a8a50 with a #5a5030 shadow side, a centre seam."""
    p = sym(t["pos"])
    col = mix(_hex("#b8ad6e"), _hex("#d6cc8e"), smooth(-0.2, 0.2, p[:, 2]))                    # top to bottom
    col = col * (0.94 + 0.12 * smooth(0.35, 0.65, fbm3(p * 22, 74)))[..., None]              # mottled
    col = mix(col, _hex("#a89a68"), 0.5 * smooth(0.6, 0.75, fbm3(p * 60, 75)))                 # wear
    col = mix(col, _hex("#a8975e"), 0.6 * smooth(0.6, 0.7, fbm3(p * 14, 76)))                  # soft stains
    col = mix(col, _hex("#8e8050"), 0.75 * smooth(-0.12, -0.26, p[:, 2]))                      # the bottom
    col = mix(col, _hex("#8a8058"), 0.9 * smooth(0.1, 0.18, p[:, 0]))                          # the outer 15%
    col *= (0.93 + 0.14 * fbm3(p * 30, 72, 2))[..., None]
    zc = p[:, 2] + 0.004 * (fbm3(p * 20, 73) - .5) - 0.12 * p[:, 0] ** 2 / .05   # seams bowing down
    for z0 in (.17, .09, .01, -.07, -.15):                                  # five transverse seams
        col = mix(col, _hex("#e0d6a4"), 0.7 * smooth(0.0075, 0.004, z0 - zc) * (zc < z0))     # lit lip below
        col = mix(col, _hex("#6e6340"), 0.95 * smooth(0.0048, 0.0025, np.abs(zc - z0)))
    col = mix(col, _hex("#d6c892"), 0.7 * smooth(0.014, 0.007, p[:, 0]))
    col = mix(col, _hex("#6b6136"), 0.97 * smooth(0.0065, 0.004, p[:, 0]))     # the midline, 2 texels
    for k, (x0, z0, ang) in enumerate(((.05, .1, .5), (.09, -.03, -.3), (.03, -.12, .9), (.12, .05, 1.3))):
        q = np.stack([p[:, 0] - x0, p[:, 2] - z0], 1)                     # old scratches
        al_ = q @ np.array([np.cos(ang), np.sin(ang)])
        ac_ = q @ np.array([-np.sin(ang), np.cos(ang)])
        col = mix(col, _hex("#e6dca8"), 0.85 * smooth(.04, .03, np.abs(al_)) * smooth(.0026, .0011, np.abs(ac_)))
    rimz = smooth(RIM_Y - .03, RIM_Y - .012, p[:, 1])
    col = mix(col, _hex("#7d7448"), 0.7 * rimz)                            # a darker rim at the bridge
    for k, (x0, z0) in enumerate(((.02, -.21), (.045, -.205), (.03, -.225), (.06, -.19))):   # a little duckweed
        dd = np.hypot(p[:, 0] - x0, p[:, 2] - z0) / .008
        col = mix(col, mix(WEED_LO, _hex("#6fae3a"), smooth(1.0, 0.2, dd)), smooth(1.0, 0.8, dd))
    return mix(col, _hex("#4a4a30"), 0.75 * smooth(RIM_Y - .018, RIM_Y - .003, p[:, 1]))


_COIN = None


def _coin_image():
    """The game's coin (tex_Coinage: two faces side by side, each 32 x 64 texels for a round coin)."""
    global _COIN
    if _COIN is None:
        im = Image.open(GAME_DATA / "tex_Coinage.png").convert("RGB")
        _COIN = np.asarray(im.crop((0, 0, 32, 64)).resize((64, 64), Image.LANCZOS), np.float32)
    return _COIN


COIN_HI, COIN_MIDT, COIN_SH = _hex("#ecc864"), _hex("#c9962a"), _hex("#6e4a0e")


def paint_coin(t):
    """A round gold coin on the 7-sided disc, shaded like the game's gold: lit from the upper left
    (#f6d46a) to #c8901e to a #6e4a0e shadow side, the game's own coin face (tex_Coinage) embossed in
    relief, a raised rim lit on its upper side and dark on its lower, a dark worn edge."""
    u, v = t["u"], t["v"]
    img = _coin_image().mean(2)
    img = img / max(img.max(), 1)
    x = np.clip((u * 63).astype(int), 0, 62)
    y = np.clip(((1 - v) * 63).astype(int), 1, 63)
    relief = img[y, x] - img[y - 1, x + 1]                                # the stamp, lit from the upper left
    rr = np.hypot(u - .5, v - .5) * 2
    lightdir = smooth(-0.7, 0.7, (v - .5) * 1.2 - (u - .5) * 0.6)
    col = mix(COIN_MIDT, COIN_HI, 0.5 + 0.5 * lightdir)
    col = mix(col, COIN_HI, np.clip(relief * 2.2, 0, 1) * smooth(0.72, 0.6, rr))
    col = mix(col, COIN_SH, np.clip(-relief * 2.2, 0, 1) * smooth(0.72, 0.6, rr))
    ring = smooth(0.7, 0.76, rr) * smooth(0.9, 0.84, rr)                 # the raised rim
    col = mix(col, mix(_hex("#8a6a1a"), _hex("#fff0a0"), lightdir), 0.9 * ring)
    col = mix(col, _hex("#fff0a0"), 0.95 * smooth(0.12, 0.06, np.hypot(u - .3, v - .72)))     # a glint, upper left
    col = mix(col, _hex("#8a6418"), 0.6 * smooth(0.6, 0.85, rr) * smooth(0.1, 0.5, (u - .5) - (v - .5)))   # rim shadow, lower right
    col = mix(col, _hex("#2a2618"), smooth(0.86, 0.9, rr))                 # the grime gap where it is wedged
    col = mix(col, COIN_SH, smooth(0.68, 0.72, rr) * smooth(0.76, 0.72, rr) * 0.6)   # inside the rim
    return mix(col, _hex("#2e2a16"), smooth(0.92, 0.97, rr))           # the hex's corners: the dark pocket


def paint_coin_v4(t, sunk=True):
    """v4: the coin standing in the crack (and the medallion): the game's coin face in relief, a
    raised rim lit on its upper side, a glint, a thin dark edge - no grime ring (the crack's own lips
    and shadow do that)."""
    u, v = t["u"], t["v"]
    img = _coin_image().mean(2)
    img = img / max(img.max(), 1)
    x = np.clip((u * 63).astype(int), 0, 62)
    y = np.clip(((1 - v) * 63).astype(int), 1, 63)
    relief = img[y, x] - img[y - 1, x + 1]
    rr = np.hypot(u - .5, v - .5) * 2
    lightdir = smooth(-0.7, 0.7, (v - .5) * 1.2 - (u - .5) * 0.6)
    col = mix(GOLD_MID, GOLD_LT, 0.3 + 0.6 * lightdir + (0.0 if sunk else 0.15))
    col = mix(col, GOLD_LT, np.clip(relief * 2.2, 0, 1) * smooth(0.74, 0.62, rr))
    col = mix(col, GOLD_SH, np.clip(-relief * 2.2, 0, 1) * smooth(0.74, 0.62, rr))
    col = mix(col, GOLD_SH, 0.55 * smooth(0.7, 0.74, rr) * smooth(0.79, 0.75, rr))       # inside the rim
    ring = smooth(0.75, 0.79, rr) * smooth(0.95, 0.9, rr)
    col = mix(col, mix(GOLD_MID, GOLD_HI, lightdir), 0.9 * ring)
    col = mix(col, GOLD_HI, 0.95 * smooth(0.13 if sunk else 0.2, 0.06, np.hypot(u - .3, v - .7)))
    col = mix(col, _hex("#4a3008"), smooth(0.94, 0.99, rr))
    if sunk:                     # its lower part: grimy and in the crack's shadow, down to black at the lips
        col = mix(col, _hex("#3a2a0c"), 0.75 * smooth(0.5, 0.3, v))
        col = mix(col, _hex("#0c0904"), 0.9 * smooth(0.36, 0.26, v))
    return col


def paint_crack(t):
    """v4: the broken lips of the coin's crack. Outside (v < .5): the shell's own paint there, a pale
    chipped ridge along the break (v .45); inside (v > .5): from a dark brown under the ridge to black
    down in the split."""
    v = t["v"]
    p = np.array(t["pos"], float)
    shell = paint_carapace(dict(t), lips=False)
    col = mix(shell, C["khaki_lt"] * 1.25, 0.85 * smooth(0.36, 0.45, v) * (v < .5))
    col = mix(col, _hex("#d8cc96"), 0.5 * smooth(0.42, 0.45, v) * (v < .5) * smooth(0.4, 0.6, fbm3(p * 200, 91)))
    inside = mix(_hex("#3a3018"), _hex("#060504"), smooth(0.56, 0.85, v))
    return np.where((v >= .5)[:, None], inside, col)


PAINTERS = {
    (BODY, "neck"): paint_neck, (BODY, "fleg"): lambda t: paint_leg(t, True),
    (BODY, "hleg"): lambda t: paint_leg(t, False), (BODY, "ffoot"): paint_foot, (BODY, "hfoot"): paint_foot,
    (BODY, "sole"): paint_sole, (BODY, "tail"): paint_tail,
    (HEAD, "sideL"): head_skin, (HEAD, "sideR"): head_skin, (HEAD, "top"): head_skin, (HEAD, "front"): head_skin,
    (HEAD, "under"): paint_under, (HEAD, "mouth"): paint_mouth,
    (HEAD, "eyeL"): lambda t: paint_eye(t, 1), (HEAD, "eyeR"): lambda t: paint_eye(t, -1),
    (HEAD, "lid"): paint_lid, (HEAD, "chain"): lambda t: paint_chain_v4(t) if _v4() else paint_chain(t), (HEAD, "tooth"): paint_tooth,
    (SHELL, "carapace"): paint_carapace, (SHELL, "under"): paint_shell_under, (SHELL, "coin"): lambda t: paint_coin_v4(t) if _v4() else paint_coin(t),
    (SHELL, "crack"): paint_crack,
    (SHELL, "medal"): lambda t: paint_coin_v4(t, sunk=False),
}


def _v4():
    import scripts.krefft_model as km
    return km.V4


def no_props(parts):
    """The parts without the eye domes, the lid and the tooth as occluders (they would darken whole
    faces of the head round them)."""
    import copy
    out = []
    for p in parts:
        q = copy.copy(p)
        q.tris = [t for t in p.tris if p.island[t[0]] not in ("eyeL", "eyeR", "lid", "tooth")]
        out.append(q)
    return out


def bake(parts, occlusion=None, size=SIZE, extra=()):
    """-> {texture name: 128 x 128 RGB image}. `extra`: more parts painted unoccluded that do not take
    part in the occlusion (v4: B's medallion, baked beside A's own parts)."""
    if occlusion is None:
        occlusion = vertex_occlusion(no_props(parts), skip=("coins", "chain"), reach=0.08)
        occlusion = {k: np.nan_to_num(v, nan=1.0) for k, v in occlusion.items()}     # (the props: open)
        occlusion = {k: (0.7 + 0.3 * v) if k in ("head", "jaw") else (0.4 + 0.6 * v) for k, v in occlusion.items()}
        for part in parts:                     # gold stays bright: no occlusion on the tooth
            for i, isl in enumerate(part.island):
                if isl in ("tooth", "coin"):
                    occlusion[part.name][i] = 1.0
    if extra:
        occlusion = dict(occlusion)
        for p in extra:
            occlusion[p.name] = np.ones(len(p.points))
    return _bake(list(parts) + list(extra), size=size, occlusion=occlusion, atlas=ATLAS, painters=PAINTERS,
                 local=island_local, ring_islands=("neck", "fleg", "hleg"))


def detail_metric(img, parts=None, texture=None):
    """Mean absolute horizontal difference of the greyscale skin (the vanilla skins: 9 - 11); with
    `parts` and `texture`, over the texels the model uses only (the bleed between islands is smooth)."""
    a = np.asarray(img.convert("L"), float)
    d = np.abs(np.diff(a, axis=1))
    if parts is None:
        return float(d.mean())
    from PIL import ImageDraw
    m = Image.new("L", a.shape[::-1], 0)
    dr = ImageDraw.Draw(m)
    for p in parts:
        if p.texture != texture:
            continue
        uv = atlas_uv(p) * np.array(a.shape[::-1])
        for t in p.tris:
            dr.polygon([tuple(uv[i]) for i in t], fill=255)
    mk = np.asarray(m) > 0
    return float(d[mk[:, 1:] & mk[:, :-1]].mean())
