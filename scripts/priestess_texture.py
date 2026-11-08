"""The Priestess's three 128 x 128 skins, painted by baking (lowpoly.bake), after her DK1 sprites and
portrait: golden-tan skin, lavender armour (top, briefs, long gloves) edged in darker violet with a
bronze belt and buckle, brown leather thigh-boots with turned-down cuffs, a blonde bob, silver
pauldrons, a pale cape that is dark on the inside, a wooden sceptre with a glowing lavender orb, and a
face with blue eyes and her portrait's violet lips.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_texture import fbm, mix, smooth
from scripts.lowpoly import bake
from scripts.priestess_model import BODY, CAPE, HEAD

ATLAS = {
    BODY: {"torso": ((0, 0, 56, 104), False), "bust": ((0, 106, 24, 128), False),
           "arm": ((58, 0, 90, 104), False), "leg": ((92, 0, 128, 104), False),
           "hand": ((26, 106, 56, 128), False), "foot": ((58, 106, 98, 128), False)},
    HEAD: {"face": ((0, 0, 64, 64), False), "hair": ((64, 0, 128, 80), False),
           "hair_in": ((66, 82, 128, 104), False), "scalp": ((66, 106, 128, 128), False),
           "pauldron": ((0, 66, 30, 96), False), "staff": ((32, 66, 64, 128), False),
           "orb": ((0, 98, 30, 128), False)},
    CAPE: {"cape": ((0, 0, 128, 92), False), "cape_in": ((0, 94, 128, 128), False)},
}

C = {k: np.array(v, float) for k, v in {
    "skin": (214, 160, 112), "skin_hi": (240, 196, 150), "skin_lo": (150, 98, 62),
    "armour": (156, 136, 196), "armour_hi": (214, 200, 240), "armour_lo": (84, 68, 122), "edge": (52, 38, 80),
    "bronze": (196, 150, 64), "bronze_hi": (244, 214, 124),
    "leather": (176, 112, 52), "leather_hi": (228, 166, 92), "leather_lo": (96, 58, 26),
    "hair": (232, 190, 74), "hair_hi": (255, 236, 150), "hair_lo": (160, 112, 34),
    "metal": (160, 162, 176), "metal_hi": (224, 226, 236), "metal_lo": (84, 84, 98),
    "cape": (196, 186, 154), "cape_hi": (230, 222, 194), "cape_lo": (128, 116, 90), "cape_in": (92, 66, 44),
    "wood": (122, 80, 40), "wood_lo": (70, 42, 18),
    "orb": (176, 146, 226), "orb_hi": (255, 214, 255), "orb_lo": (96, 70, 150),
    "eye": (70, 120, 200), "white": (240, 236, 230), "lips": (126, 92, 170), "brow": (150, 104, 40),
}.items()}


def light(nrm):
    return 0.82 + 0.26 * np.clip(nrm[:, 1], -1, 1) + 0.08 * np.clip(nrm[:, 2], -1, 1)


def armour(t, base=None):
    """Lavender plate: lit from above, a sheen on the parts facing up and forward."""
    nrm = t["nrm"]
    sheen = np.clip(nrm[:, 1] * .6 + nrm[:, 2] * .5, 0, 1) ** 2
    col = mix(C["armour"], C["armour_hi"], 0.55 * sheen)
    col = mix(col, C["armour_lo"], 0.35 * np.clip(-nrm[:, 1], 0, 1))
    return col


def skin(t):
    nrm = t["nrm"]
    col = mix(C["skin"], C["skin_hi"], 0.4 * np.clip(nrm[:, 1] * .5 + nrm[:, 2] * .5, 0, 1))
    return mix(col, C["skin_lo"], 0.3 * np.clip(-nrm[:, 1], 0, 1))


def band(y, lo, hi, soft=0.004):
    return smooth(lo - soft, lo + soft, y) * smooth(hi + soft, hi - soft, y)


def paint_torso(t):
    y, u, x = t["pos"][:, 1], t["u"], t["pos"][:, 0]
    front = u < 0.5
    col = skin(t)
    a = armour(t)
    # briefs (cut high over the hips), the belt and its buckle
    briefs = smooth(.505, .497, y) * np.where(np.abs(x) > .06, smooth(.47, .478, y + (np.abs(x) - .06) * .6), 1)
    col = mix(col, a, briefs)
    belt = band(y, .49, .512)
    col = mix(col, mix(C["bronze"], C["bronze_hi"], .4 * np.clip(t["nrm"][:, 2], 0, 1)), belt)
    buckle = (u < .1) & (np.abs(y - .501) < .012)
    col = np.where(buckle[:, None], C["bronze_hi"], col)
    col = np.where(((u < .055) & (np.abs(y - .501) < .006))[:, None], C["leather_lo"], col)
    # the top: armoured cups in front, a strap across the back, straps up over the shoulders
    top_front = front & (y > .628) & (y < .714 - .03 * smooth(.2, .0, u))
    top_back = (~front) & (y > .655) & (y < .69)
    straps = front & (np.abs(u - .3) < .055) & (y > .7) & (y < .78)
    over = (y > .755) & (np.abs(np.abs(x) - .07) < .014)
    col = np.where((top_front | top_back | straps | over)[:, None], a, col)
    # dark violet edging round every piece of armour
    edge = (np.abs(y - .628) < .004) & front | (np.abs(y - (.714 - .03 * smooth(.2, .0, u))) < .004) & front \
        | (np.abs(y - .497) < .003)
    # the cups' rims and a shine on each
    cup = front & (y > .63) & (y < .71) & (np.abs(u - .25) < .17)
    col = mix(col, C["armour_hi"], .3 * cup * np.exp(-((u - .22) / .06) ** 2 - ((y - .688) / .015) ** 2))
    col = np.where(edge[:, None], C["edge"], col)
    # the navel and a soft shadow under the bust
    col = mix(col, C["skin_lo"], 0.6 * np.exp(-((u / .03) ** 2) - ((y - .56) / .006) ** 2))
    col = mix(col, C["skin_lo"], 0.35 * front * np.exp(-((y - .622) / .01) ** 2))
    return col * light(t["nrm"])[:, None]


def paint_bust(t):
    col = armour(t)
    col = mix(col, C["edge"], smooth(.15, .0, t["along"]) * .8)
    return col * light(t["nrm"])[:, None]


def paint_arm(t):
    y = t["pos"][:, 1]
    col = mix(skin(t), armour(t), smooth(.676, .668, y))       # the long glove, above the elbow
    cuff = band(y, .664, .68)
    col = mix(col, mix(C["bronze"], C["bronze_hi"], .3), cuff)
    return col * light(t["nrm"])[:, None]


def paint_hand(t):
    return armour(t) * light(t["nrm"])[:, None]


def leather(t):
    nrm = t["nrm"]
    col = mix(C["leather"], C["leather_hi"], 0.5 * np.clip(nrm[:, 1] * .4 + nrm[:, 2] * .6, 0, 1))
    col = mix(col, C["leather_lo"], 0.35 * np.clip(-nrm[:, 2], 0, 1))
    return col * (0.92 + 0.12 * fbm(t["pos"][:, 0] * 90, t["pos"][:, 1] * 90, 41))[:, None]


def paint_leg(t):
    y, u = t["pos"][:, 1], t["u"]
    col = mix(skin(t), leather(t), smooth(.392, .384, y))
    cuff = band(y, .362, .392)
    col = mix(col, C["leather_hi"] * .95, cuff * .8)
    col = mix(col, C["leather_lo"], np.exp(-((y - .362) / .003) ** 2) * .9)
    seam = (np.abs(u - .5) < .03) & (y < .36)
    col = np.where(seam[:, None], C["leather_lo"], col)
    knee = np.exp(-((y - .245) / .02) ** 2) * (u < .3)
    col = mix(col, C["leather_hi"], .35 * knee)
    return col * light(t["nrm"])[:, None]


def paint_foot(t):
    col = leather(t)
    col = mix(col, C["leather_lo"] * .6, smooth(-.3, -.7, t["nrm"][:, 1]))     # the sole
    return col * light(t["nrm"])[:, None]


def paint_face(t):
    u, v = t["u"], t["v"]
    col = skin(t)
    # cheeks
    col = mix(col, np.array([226., 140, 110]), .25 * np.exp(-(((np.abs(u - .5) - .22) / .08) ** 2) - ((v - .6) / .07) ** 2))
    # eyes: a lid line, white, a blue iris
    for side in (-1, 1):
        cu = .5 + side * .19
        e = ((u - cu) / .085) ** 2 + ((v - .46) / .032) ** 2
        col = np.where((e < 1)[:, None], C["white"], col)
        iris = ((u - cu) / .036) ** 2 + ((v - .462) / .03) ** 2 < 1
        col = np.where(iris[:, None], C["eye"], col)
        pupil = ((u - cu) / .014) ** 2 + ((v - .462) / .014) ** 2 < 1
        col = np.where(pupil[:, None], C["edge"] * .4, col)
        lid = (np.abs(e - 1) < .45) & (v < .462)
        col = np.where(lid[:, None], C["edge"] * .6, col)
        brow = (np.abs(u - cu) < .1) & (np.abs(v - (.39 - .25 * (u - cu) * side * -1 * 0)) < .012)
        col = np.where(brow[:, None], C["brow"], col)
    # the nose's shadow and her violet lips
    col = mix(col, C["skin_lo"], .4 * np.exp(-((u - .5) / .03) ** 2 - ((v - .6) / .03) ** 2))
    lips = ((u - .5) / .1) ** 2 + ((v - .735) / .028) ** 2 < 1
    col = np.where(lips[:, None], mix(C["lips"], C["lips"] * 1.25, smooth(.735, .72, v)), col)
    col = np.where((lips & (np.abs(v - .735) < .005))[:, None], C["edge"], col)
    return col * light(t["nrm"])[:, None]


def paint_hair(t):
    u, v = t["u"], t["v"]
    strands = fbm(u * 26, v * 3, 51)
    col = mix(C["hair_lo"], C["hair"], .45 + .55 * strands)
    col = mix(col, C["hair_hi"], .6 * np.exp(-((v - .28) / .1) ** 2) * (.5 + .5 * strands))   # the sheen
    col = mix(col, C["hair_lo"], .45 * smooth(.75, 1.0, v))                                     # darker tips
    col = mix(col, C["hair_lo"] * .7, .55 * t.get("inner", np.zeros_like(u))[:])
    return col * light(t["nrm"])[:, None]


def paint_pauldron(t):
    nrm = t["nrm"]
    col = mix(C["metal"], C["metal_hi"], .6 * np.clip(nrm[:, 1] * .7 + nrm[:, 2] * .4, 0, 1) ** 1.5)
    col = mix(col, C["metal_lo"], .5 * smooth(.3, .0, t["along"]))
    return col * light(nrm)[:, None]


def paint_staff(t):
    along = t.get("along", np.zeros(len(t["u"])))
    grain = fbm(t["u"] * 8, t["pos"][:, 1] * 60, 61)
    col = mix(C["wood_lo"], C["wood"], .4 + .6 * grain)
    prong = along > 0.01
    col = np.where(prong[:, None], mix(C["bronze"], C["bronze_hi"], along), col)
    return col * light(t["nrm"])[:, None]


def paint_orb(t):
    nrm = t["nrm"]
    col = mix(C["orb_lo"], C["orb"], .5 + .5 * nrm[:, 1])
    spec = np.clip(nrm @ np.array([-.4, .7, .6]) / np.linalg.norm([-.4, .7, .6]), 0, 1) ** 6
    col = mix(col, C["orb_hi"], spec)
    return col


def paint_cape(t):
    u, v = t["u"], t["v"]
    face = t.get("face", np.ones_like(u))
    folds = .5 + .5 * np.sin(u * np.pi * 7 + fbm(u * 4, v * 2, 71) * 2) * smooth(.05, .5, v)
    col = mix(C["cape_lo"], C["cape_hi"], .3 + .55 * folds)
    col = mix(col, C["cape_lo"], .5 * smooth(.85, 1.0, v))           # the hem
    inside = mix(C["cape_in"] * .8, C["cape_in"], folds)
    col = np.where((face < 0)[:, None], inside, col)
    return col * (0.9 + 0.1 * np.clip(t["nrm"][:, 1], -1, 1))[:, None]


PAINTERS = {
    (BODY, "torso"): paint_torso, (BODY, "bust"): paint_bust, (BODY, "arm"): paint_arm,
    (BODY, "leg"): paint_leg, (BODY, "hand"): paint_hand, (BODY, "foot"): paint_foot,
    (HEAD, "face"): paint_face, (HEAD, "hair"): paint_hair, (HEAD, "hair_in"): paint_hair,
    (HEAD, "scalp"): lambda t: C["hair_lo"] * light(t["nrm"])[:, None],
    (HEAD, "pauldron"): paint_pauldron,
    (HEAD, "staff"): paint_staff, (HEAD, "orb"): paint_orb, (CAPE, "cape"): paint_cape,
    (CAPE, "cape_in"): paint_cape,
}


def bake_priestess(parts, occlusion=None):
    return bake(parts, ATLAS, PAINTERS, occlusion=occlusion)
