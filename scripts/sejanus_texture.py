"""Sejanus's three 128 x 128 skins (Sejanus_Body, Sejanus_Limbs, Sejanus_Head), painted by baking.

The Dragon's baker (dragon_texture.bake: every texel traced back to the bind-pose surface it covers, painted
by its island's painter, soft light from above and occlusion baked in, 4x supersampled, edges bled), with
Sejanus's own atlas and painters.

The look, after Soles' reference picture (13 Nov) and the game's own painted skins (the Horned Reaper's torso
and legs: every muscle drawn, with shading, sinew and highlights): blue-black hide over a lean, corded body,
each muscle edged by a deep purple crease, a dusty blue-grey sheen on the bulges; visible ribs, a ridged
spine, sinew down the neck, arms and legs; a faint violet glowing mark on the shoulder; a horse's face with
the mouth closed, rounded nostrils, one soft violet almond eye a side; a pale lilac-white mane, forelock,
tail and fetlock hair; pale horn hooves; short dark nails.
(Since Soles' second and third looks: near-black hide with purple only as small accents, the spike mane
back, the tail and fetlock hair charcoal-grey with dim purple ends.)

Painters work in each island's own coordinates (u round the loft: 0 on its dorsal or front line, 1 on the
ventral or back line; v along it) scaled to model units by EXTENT, so a muscle drawn as an ellipse of
model-unit radii keeps its shape. A painter builds a relief (1 on a muscle's crown, 0 off any muscle) that
is coloured by ramp() and embossed with light from above.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_texture import SIZE, atlas_uv as _atlas_uv, bake as _bake, fbm, mix, smooth, value_noise
from scripts.dragon_texture import vertex_occlusion  # noqa: F401  (re-exported for the build)

TEXTURES = ("Sejanus_Body", "Sejanus_Limbs", "Sejanus_Head")

# texture -> island -> ((x0, y0, x1, y1) pixels of the 128 texture, transpose): u across x, v down y.
# Sized so every island gets about the same texels per model unit both ways (tests/test_sejanus checks it):
# the trunk about 260, the limbs and tail about 220, the head 330.
ATLAS = {
    "Sejanus_Body": {
        "torso": ((0, 0, 64, 128), False),
        "neck": ((64, 0, 128, 128), False),
    },
    "Sejanus_Limbs": {
        "shank": ((0, 0, 22, 128), False),
        "arm": ((23, 0, 42, 128), False),
        "tail": ((43, 0, 70, 128), False),
        "thigh": ((71, 0, 128, 86), False),
        "finger": ((71, 88, 81, 116), False),
    },
    "Sejanus_Head": {
        "skull": ((0, 0, 58, 128), False),
        "jaw": ((59, 0, 90, 80), False),
        "mane": ((91, 0, 128, 60), False),
        "ear": ((91, 62, 106, 88), False),
    },
}
# model units per unit of island u and v (area-weighted, measured on the bind mesh: test_sejanus)
EXTENT = {"torso": (.240, .392), "neck": (.242, .372), "tail": (.102, .529), "skull": (.171, .394),
          "jaw": (.099, .206), "thigh": (.202, .306), "shank": (.097, .597), "arm": (.085, .573),
          "finger": (.019, .070), "mane": (.069, .112), "ear": (.042, .075)}


# islands of odd-sided lofts whose u runs past 1 over the straddling back face (sejanus_model.loft_islands)
WIDE_U = {"thigh": 1.2, "shank": 1.2}


def island_local(part):
    uv = np.array(part.uv, float)
    for isl, top in WIDE_U.items():
        m = np.array(part.island) == isl
        uv[m, 0] /= top
    return uv


def _unwide(isl, painter):
    """Give a wide island's painter the mirrored u it is written for (0 front .. 1 back)."""
    top = WIDE_U[isl]

    def run(t):
        u = t["u"] * top
        t = dict(t, u=np.where(u > 1, 2 - u, u))
        return painter(t)
    return run


def atlas_uv(part):
    return _atlas_uv(part, ATLAS, island_local)


# ------------------------------------------------------------------------------------------- palette

C = {k: np.array(v, float) for k, v in {
    "groove": (10, 9, 13), "crease": (40, 30, 52), "crease_hi": (90, 50, 130), "purplehide": (34, 32, 42),
    "hide": (38, 38, 46), "hide_dark": (15, 15, 19), "sheen": (88, 90, 102),
    "bone": (206, 200, 214), "bone_hi": (234, 230, 242), "bone_dark": (104, 98, 120), "bone_shadow": (56, 50, 68),
    "hair": (92, 94, 104), "hair_mid": (60, 61, 70), "hair_dark": (28, 28, 34),
    "glow": (176, 92, 255), "glow_core": (236, 200, 255), "glow_deep": (88, 36, 150),
    "socket": (8, 6, 12), "lip": (10, 9, 13), "muzzle": (56, 55, 64), "nostril": (6, 5, 8),
    "mouth": (58, 14, 20), "mouth_hi": (96, 30, 38), "tooth": (206, 198, 176), "tooth_dark": (110, 100, 84),
    "pupil": (40, 10, 60), "spike": (104, 98, 98), "spike_tip": (84, 52, 118),
    "nail": (16, 14, 20), "nail_hi": (70, 66, 84),
}.items()}
C["red"] = C["crease"]           # (the painters' muscle-edge colour)

RAMP_H = [0.0, 0.03, 0.09, 0.2, 0.4, 0.7, 1.0]
RAMP_C = [C["hide_dark"], C["groove"], C["crease"], C["crease"] * .85, C["purplehide"], C["hide"], C["sheen"]]


def ramp(h):
    """Relief (0 off any muscle .. 1 a muscle's crown) -> colour: plain blue-black hide off the muscles; at
    every muscle's edge a dark crease edged deep purple, then slate rising to a blue-grey sheen on the crown."""
    h = np.clip(h, 0, 1)
    return np.stack([np.interp(h, RAMP_H, [c[k] for c in RAMP_C]) for k in range(3)], -1)


def _light(nrm):
    up = np.clip(nrm[:, 1], -1, 1)
    front = np.clip(nrm[:, 2], -1, 1)
    return 0.80 + 0.28 * up + 0.06 * front


def blob(u, v, isl, cu, cv, ru, rv, ang=0.0, power=0.6):
    """A muscle: 1 at (cu, cv) falling to 0 at radii (ru, rv) model units (rotated by `ang` degrees)."""
    U, V = EXTENT[isl]
    du, dv = (u - cu) * U, (v - cv) * V
    a = np.radians(ang)
    x, y = du * np.cos(a) + dv * np.sin(a), -du * np.sin(a) + dv * np.cos(a)
    return np.clip(1 - (x / ru) ** 2 - (y / rv) ** 2, 0, 1) ** power


def emboss(hfun, u, v, isl, up=1.0, k=1.6):
    """Brightness from the relief's slope toward the light: `up` = +1 when v runs up the body, -1 down."""
    _, V = EXTENT[isl]
    e = 0.004 / V
    dh = (hfun(u, v + e) - hfun(u, v - e)) / 2
    return np.clip(1 - k * up * dh / 0.004 * 0.035, 0.55, 1.6)


def grain(u, v, isl, seed, scale=1.0):
    """Leathery hide: fine mottling and a few creases (multiplier about 1)."""
    U, V = EXTENT[isl]
    x, y = u * U * 260 * scale, v * V * 260 * scale
    m = 0.88 + 0.24 * fbm(x * .25, y * .25, seed)
    crease = np.abs(value_noise(x * .35, y * .3, seed + 5) - 0.5)
    return m * (1 - 0.18 * smooth(0.03, 0.0, crease))


def fibres(col, h, u, v, isl, seed, ang=0.0):
    """Muscle fibres: fine lines along v (turned `ang` degrees) on the muscles' crowns, darker between."""
    U, V = EXTENT[isl]
    a = np.radians(ang)
    x, y = u * U, v * V
    xr, yr = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    f = fbm(xr * 420, yr * 35, seed)
    k = smooth(.25, .6, h)
    return col * (1 + k * (0.65 * (f - 0.5)))[..., None]


def veins(col, u, v, isl, seed, amount=0.6, width=0.022):
    """A few branching veins, dark red, swollen over the muscles."""
    U, V = EXTENT[isl]
    n = np.abs(value_noise(u * U * 55, v * V * 22, seed) + 0.35 * value_noise(u * U * 140, v * V * 60, seed + 1)
               - 0.68)
    return mix(col, C["crease"] * .9, 0.6 * amount * smooth(width, 0.0, n))


def _shade(col, t, extra=1.0):
    return col * (_light(t["nrm"]) * extra)[..., None]


# ------------------------------------------------------------------------------------------- trunk

# the torso island's v runs from under the pelvis (0) to the chest ring (1); the neck island's from the chest
# ring (0) through the shoulders (.226) and the neck's base (.559) to the top of the neck (1).
# S: model units up the trunk from the chest ring (negative on the torso), continuous over the seam.

def _S(isl, v):
    return (v - 1) * EXTENT["torso"][1] if isl == "torso" else v * EXTENT["neck"][1]


def torso_relief(u, v):
    i = "torso"
    S = _S(i, v)
    h = np.zeros_like(u)
    # back: two long erector cords either side of the spine
    h = np.maximum(h, blob(u, v, i, .14, .62, .03, .17))
    # lats sweeping from the armpit down the back
    h = np.maximum(h, blob(u, v, i, .32, .86, .045, .1, ang=-25))
    # obliques on the waist
    h = np.maximum(h, blob(u, v, i, .55, .42, .05, .07, ang=10))
    # abdominals: three rows of plates beside the midline (u = 1 is the midline: mirrored)
    for cv in (.30, .47, .63):
        h = np.maximum(h, blob(u, v, i, .885, cv, .03, .033, power=.45))
    # the bottom of the pecs (they carry on into the neck island)
    h = np.maximum(h, blob(u, v, i, .8, 1.02, .065, .045, ang=-15))
    # ribs: ridges slanting down toward the front, on the sides of the ribcage
    phase = (S + 0.09 * u - 0.05 * (u - .55) ** 2) / 0.03
    rib = np.clip(1 - np.abs(((phase - np.floor(phase)) - .5) * 2) * 1.6, 0, 1)   # a ridge, wide gaps
    ribzone = smooth(.32, .45, u) * smooth(.84, .7, u) * smooth(.5, .68, v) * smooth(1.04, .92, v)
    h = np.maximum(h, rib ** 0.8 * ribzone * .8)
    # hips: the thigh muscles meet the pelvis (the legs cover most of it)
    h = np.maximum(h, blob(u, v, i, .45, .12, .07, .06))
    return h


def paint_torso(t):
    u, v = t["u"], t["v"]
    h = torso_relief(u, v)
    col = ramp(h) * grain(u, v, "torso", 3)[..., None]
    col = fibres(col, h, u, v, "torso", 4, ang=20 * (u - .5))
    col = veins(col, u, v, "torso", 5, .45)
    # the spine: a groove with a knob of bone every few centimetres
    S = _S("torso", v)
    knob = (0.5 + 0.5 * np.cos(2 * np.pi * S / 0.034)) ** 3 * np.exp(-(u / .035) ** 2) * smooth(.2, .4, v)
    col = mix(col, C["groove"], 0.6 * np.exp(-(u / .03) ** 2) * (1 - knob))
    col = mix(col, C["bone_dark"] * 0.9, 0.8 * knob)
    # the hip bones (iliac crests) showing through the gaunt flanks
    hip = blob(u, v, "torso", .5, .2, .03, .018)
    col = mix(col, C["bone_dark"], 0.7 * hip)
    # the linea alba down the belly, the groin darker
    col = mix(col, C["groove"], 0.7 * np.exp(-((u - 1) / .03) ** 2) * smooth(.15, .3, v) * smooth(.95, .8, v))
    col = mix(col, C["hide_dark"], 0.55 * smooth(.2, .0, v))
    col = col * emboss(torso_relief, u, v, "torso")[..., None]
    return _shade(col, t)


def neck_relief(u, v):
    i = "neck"
    h = np.zeros_like(u)
    # pecs: two slabs on the chest
    h = np.maximum(h, blob(u, v, i, .82, .06, .07, .075, ang=-20))
    # shoulder blades and the traps over them
    h = np.maximum(h, blob(u, v, i, .27, .1, .04, .07, ang=20))
    h = np.maximum(h, blob(u, v, i, .1, .25, .035, .09))
    # the deltoid cap (under the arm's top)
    h = np.maximum(h, blob(u, v, i, .5, .12, .04, .05))
    # the trapezius rising from the shoulders into the neck
    h = np.maximum(h, blob(u, v, i, .22, .38, .04, .09, ang=-10))
    # the neck: two long cords down each side (brachiocephalic, splenius), the windpipe ringed underneath
    h = np.maximum(h, blob(u, v, i, .4, .72, .035, .14, power=.5))
    h = np.maximum(h, blob(u, v, i, .62, .66, .03, .13, power=.5))
    h = np.maximum(h, 0.55 * smooth(.45, .6, v) * smooth(.15, .25, u) * smooth(.85, .75, u))
    h = np.maximum(h, 0.55 * smooth(.84, .92, u) * smooth(.3, .45, v) * smooth(.98, .9, v))   # the throat
    return h


def paint_neck(t):
    u, v = t["u"], t["v"]
    h = neck_relief(u, v)
    col = ramp(h) * grain(u, v, "neck", 7)[..., None]
    col = fibres(col, h, u, v, "neck", 8, ang=-15 * (v < .3))
    col = veins(col, u, v, "neck", 9, .5)
    # the jugular groove (deep red) between the cords and the windpipe
    col = mix(col, C["red"] * .8, 0.55 * np.exp(-((u - .76) / .025) ** 2) * smooth(.3, .45, v))
    # the spine of each shoulder blade: a bony ridge
    ridge = np.exp(-(((u - .25) - .25 * (v - .1)) / .018) ** 2) * smooth(.0, .04, v) * smooth(.22, .14, v)
    col = mix(col, C["bone_dark"], 0.6 * ridge)
    # the crest under the mane: hair roots, pale
    crest = smooth(.16, .06, u) * smooth(.15, .3, v)
    U, V = EXTENT["neck"]
    strands = fbm(u * U * 300, v * V * 40, 31)
    hair = mix(C["hair_dark"], C["hair_mid"], smooth(.4, .8, strands))
    col = mix(col, hair, crest)
    col = col * emboss(neck_relief, u, v, "neck")[..., None]
    return _shade(col, t)


def paint_tail(t):
    """Long pale horse hair (lilac-white), locks shaded lilac-grey between, greyer to the tip; hide at the root."""
    u, v = t["u"], t["v"]
    U, V = EXTENT["tail"]
    s1 = fbm(u * U * 420, v * V * 14, 41)
    s2 = fbm(u * U * 900, v * V * 30, 43)
    strands = 0.6 * s1 + 0.4 * s2
    col = mix(C["hair_mid"], C["hair"], smooth(.4, .75, strands))
    col = mix(col, C["hair_dark"], 0.55 * smooth(.55, .3, s2))         # gaps between locks
    col = mix(col, C["hair_mid"] * .85, 0.5 * smooth(.6, 1.0, v))       # greyer toward the tip
    col = mix(col, C["hair_dark"], 0.5 * smooth(.35, 1.0, u))          # the underside in shade
    col = mix(col, C["spike_tip"], 0.7 * smooth(.82, .98, v) * smooth(.4, .7, s1))   # dim purple ends
    hide = ramp(0.55 + 0.2 * fbm(u * 30, v * 30, 44)) * 0.9
    col = mix(hide, col, smooth(.08, .2, v))
    return _shade(col, t)


# ------------------------------------------------------------------------------------------- legs

def thigh_relief(u, v):
    i = "thigh"
    h = np.zeros_like(u)
    h = np.maximum(h, blob(u, v, i, .18, .62, .05, .12, ang=0))            # quadriceps (front)
    h = np.maximum(h, blob(u, v, i, .5, .45, .055, .1, ang=10))           # the outside of the thigh
    h = np.maximum(h, blob(u, v, i, .8, .5, .05, .13, ang=-5))            # hamstrings (behind)
    h = np.maximum(h, blob(u, v, i, .97, .62, .03, .1))                   # semitendinosus
    return h


def paint_thigh(t):
    u, v = t["u"], t["v"]
    h = thigh_relief(u, v)
    col = ramp(h) * grain(u, v, "thigh", 11)[..., None]
    col = fibres(col, h, u, v, "thigh", 13, ang=8)
    col = veins(col, u, v, "thigh", 14, .55)
    # the stifle's kneecap
    knee = blob(u, v, "thigh", .02, .97, .025, .025)
    col = mix(col, C["bone_dark"], 0.65 * knee)
    col = col * emboss(thigh_relief, u, v, "thigh", up=-1, k=.7)[..., None]
    return _shade(col, t)


def shank_relief(u, v):
    i = "shank"
    h = np.zeros_like(u)
    h = np.maximum(h, blob(u, v, i, .75, .17, .03, .09))    # gaskin
    h = np.maximum(h, blob(u, v, i, .15, .15, .025, .09))   # front of the shin
    # the cannon: the flexor tendons down the back, the bone down the front (no band across)
    h = np.maximum(h, blob(u, v, i, .9, .66, .02, .1, power=.8))
    h = np.maximum(h, 0.45 * blob(u, v, i, .1, .66, .02, .1, power=.8))
    return h


def paint_shank(t):
    u, v = t["u"], t["v"]
    h = shank_relief(u, v)
    col = ramp(h) * grain(u, v, "shank", 17)[..., None]
    col = fibres(col, h, u, v, "shank", 18)
    col = veins(col, u, v, "shank", 22, .5 * smooth(.6, .4, v))
    # the Achilles tendon to the point of the hock, the hock's bone
    ach = np.exp(-((u - .98) / .06) ** 2) * smooth(.15, .3, v) * smooth(.52, .46, v)
    col = mix(col, C["sheen"], 0.5 * ach)
    hock = blob(u, v, "shank", 1.0, .5, .025, .022) + blob(u, v, "shank", .5, .5, .02, .02) * .6
    col = mix(col, C["bone_dark"], 0.6 * np.clip(hock, 0, 1))
    fet = blob(u, v, "shank", .8, .8, .025, .02)
    col = mix(col, C["bone_dark"] * .8, 0.4 * fet)
    # pale feathering over the hoof, ragged at its lower edge
    U, V = EXTENT["shank"]
    s = fbm(u * U * 500, v * V * 60, 19)
    tips = .9 - 0.06 * s
    feather = smooth(.78, .83, v) * smooth(tips + .01, tips - .02, v)
    hair = mix(C["hair_dark"], C["hair"], smooth(.35, .75, s) * smooth(.78, .86, v))
    hair = mix(hair, C["spike_tip"], 0.6 * smooth(tips - .03, tips, v) * smooth(.45, .7, s))     # dim purple ends
    col = mix(col, hair, feather)
    # the hoof: pale horn, darker at the coronet, growth ridges down it, a dark sole edge
    hoof = smooth(tips - .01, tips + .01, v)
    rid = 0.85 + 0.15 * np.cos(2 * np.pi * u * 7 + 3 * fbm(u * 20, v * 40, 21))
    horn = mix(C["bone_dark"], C["bone"], smooth(.88, .95, v)) * rid[..., None]
    horn = mix(horn, C["bone_shadow"], 0.6 * smooth(.975, .99, v))
    horn = horn * (0.9 + 0.2 * fbm(u * 60, v * 200, 23))[..., None]
    col = mix(col, horn, hoof)
    col = col * emboss(shank_relief, u, v, "shank", up=-1, k=.7)[..., None]
    return _shade(col, t)


# ------------------------------------------------------------------------------------------- arms

def arm_relief(u, v):
    i = "arm"
    h = np.zeros_like(u)
    h = np.maximum(h, blob(u, v, i, .5, .13, .05, .06))     # deltoid
    h = np.maximum(h, blob(u, v, i, .15, .3, .03, .08))     # biceps
    h = np.maximum(h, blob(u, v, i, .82, .28, .035, .09))   # triceps
    h = np.maximum(h, blob(u, v, i, .3, .57, .03, .06))     # forearm: brachioradialis
    h = np.maximum(h, blob(u, v, i, .75, .58, .03, .06))    # forearm: flexors
    # tendons down the forearm to the wrist, then over the back of the hand
    ten = 0.5 + 0.5 * np.cos(2 * np.pi * (u - .12) / .3)
    h = np.maximum(h, (0.4 + 0.35 * ten ** 3) * smooth(.62, .72, v) * smooth(1.0, .97, v))
    return h


def paint_arm(t):
    u, v = t["u"], t["v"]
    h = arm_relief(u, v)
    col = ramp(h) * grain(u, v, "arm", 27)[..., None]
    col = fibres(col, h, u, v, "arm", 28)
    col = veins(col, u, v, "arm", 29, .6)
    elbow = blob(u, v, "arm", 1.0, .48, .02, .025)
    col = mix(col, C["bone_dark"], 0.7 * elbow)
    # knuckles (bony) and the fingers' dark red skin
    knuck = (0.5 + 0.5 * np.cos(2 * np.pi * u / .33)) ** 2 * np.exp(-((v - .97) / .012) ** 2)
    col = mix(col, C["bone_dark"], 0.55 * knuck)
    col = col * emboss(arm_relief, u, v, "arm", up=-1)[..., None]
    col = _shade(col, t)
    # the violet glowing mark on the outside of the shoulder (Soles' reference): a soft patch, brighter
    # veins of light through it; it glows, so it is not shaded
    U, V = EXTENT["arm"]
    du, dv = (u - .5) * U / .014, (v - .16) * V / .03
    r = np.sqrt(du ** 2 + dv ** 2)
    n = fbm(u * U * 300, v * V * 120, 91)
    patch = smooth(1.0, .45, r + .35 * (n - .5))
    streak = smooth(.04, .0, np.abs(value_noise(u * U * 220, v * V * 90, 92) - .5)) * smooth(1.1, .5, r)
    col = mix(col, mix(C["glow_deep"], C["glow"], n), 0.45 * patch)
    col = mix(col, C["glow"], 0.5 * streak)
    return col


def paint_finger(t):
    """Knobbly fingers: dark skin with lighter knuckle bumps, ending in a short dark glossy nail."""
    along = t["along"]
    col = mix(C["purplehide"], C["hide"], 0.5)
    knob = np.exp(-((along - .3) / .07) ** 2) + np.exp(-((along - .62) / .06) ** 2)
    col = mix(col, C["sheen"], 0.55 * np.clip(knob, 0, 1))
    col = mix(col, C["groove"], 0.6 * (np.exp(-((along - .46) / .04) ** 2) + np.exp(-((along - .76) / .03) ** 2)))
    nail = smooth(.8, .85, along)
    col = mix(col, mix(C["nail"], C["nail_hi"], 0.4 * (1 - t["u"])), nail)
    return _shade(col, t)


# ------------------------------------------------------------------------------------------- head

def skull_relief(u, v):
    i = "skull"
    h = np.full_like(u, 0.55)
    h = np.maximum(h, 0.85 * blob(u, v, i, .52, .2, .04, .07, ang=15))     # the flat cheek (masseter)
    h = np.maximum(h, 0.75 * blob(u, v, i, .15, .62, .035, .14))           # the long straight face
    h = np.maximum(h, 0.7 * blob(u, v, i, .25, .32, .03, .05))             # the broad forehead
    return h


EYE_UV = (.37, .33)


def paint_skull(t):
    u, v = t["u"], t["v"]
    U, V = EXTENT["skull"]
    h = skull_relief(u, v)
    col = ramp(h) * grain(u, v, "skull", 51, 1.3)[..., None]
    # the skull showing through: the nasal bone down the face, the facial crest down the cheek
    nasal = np.exp(-(u / .08) ** 2) * smooth(.35, .5, v) * smooth(.95, .8, v)
    col = mix(col, C["sheen"], 0.35 * nasal)
    crest = np.exp(-(((u - .46) + .2 * (v - .3)) / .025) ** 2) * smooth(.15, .25, v) * smooth(.62, .5, v)
    col = mix(col, C["sheen"] * .9, 0.45 * crest)
    # veins on the cheek and face
    vein = np.abs(value_noise(u * U * 160, v * V * 50, 53) - .5)
    col = mix(col, C["crease"], 0.45 * smooth(.03, .0, vein) * smooth(.25, .45, u) * smooth(.15, .3, v) * smooth(.75, .6, v))
    # the muzzle: soft and a little paler, the nostrils big, round and dark at its end
    muzzle = smooth(.78, .9, v)
    col = mix(col, C["muzzle"] * grain(u, v, "skull", 57, 2.0)[..., None], 0.7 * muzzle)
    du, dv = (u - .36) * U, (v - .955) * V
    n = (du / .012) ** 2 + ((dv - du * .5) / .016) ** 2
    col = mix(col, C["sheen"] * .75, 0.5 * smooth(2.4, 1.5, n) * smooth(.8, 1.3, n))      # flared rim
    col = mix(col, C["nostril"], smooth(1.15, .75, n))
    # the eye: one soft violet almond a side, in a dark lid, its glow bleeding a little onto the face
    du, dv = (u - EYE_UV[0]) * U, (v - EYE_UV[1]) * V
    e = (du / .0095) ** 2 + ((dv + du * .35) / .019) ** 2
    halo = smooth(9.0, 1.0, e)
    col = mix(col, C["glow_deep"], 0.35 * halo)
    col = mix(col, C["socket"], 0.85 * smooth(2.4, 1.2, e))
    glow = mix(C["glow_core"], C["glow"], smooth(0.0, 0.7, e))
    col = _shade(col, t)
    glow = mix(glow, C["pupil"], 0.7 * smooth(.12, .03, e))
    col = np.where((e < 1)[..., None], glow, col)
    # the mouth: the lip line low on the muzzle from its corner (two thirds down the face) to the front;
    # inside it (seen when the jaw drops) a row of teeth along the upper lip and a dark red palate
    lip = 2 / 3
    mouth = smooth(.64, .7, v) * smooth(.97, .93, v)        # (the front of the upper lip is hide)
    col = mix(col, C["lip"], 0.85 * np.exp(-((u - lip) / .025) ** 2) * mouth)
    k = v * V / .012
    tw = np.abs((k - np.floor(k)) - .5) * 2
    teeth = (u > lip + .01) & (u < lip + .09 * (1 - .5 * tw)) & (tw < .8) & (v > .72) & (v < .95)
    inside = mix(C["mouth"], C["mouth_hi"], smooth(1.0, .8, u))
    col = mix(col, inside, smooth(lip + .02, lip + .05, u) * mouth)
    col = np.where(teeth[..., None], mix(C["tooth"], C["tooth_dark"], .6 * tw), col)
    return col


def paint_jaw(t):
    """The short lower jaw: hide outside with the soft chin, the lip edge dark; inside (when it drops) lower
    teeth along the lip and a dark red tongue."""
    u, v = t["u"], t["v"]
    i = "jaw"
    U, V = EXTENT[i]
    h = np.maximum(0.55, 0.75 * blob(u, v, i, .95, .6, .02, .08))
    col = ramp(h) * grain(u, v, i, 61, 1.3)[..., None]
    col = mix(col, C["hide_dark"], 0.4 * smooth(.85, 1.0, u))      # the dark underside
    col = mix(col, C["muzzle"], 0.6 * smooth(.45, .75, v) * smooth(.5, .6, u))   # the soft chin
    lip = .5
    col = mix(col, C["lip"], 0.85 * np.exp(-((u - lip) / .03) ** 2) * smooth(.6, .5, v))
    k = v * V / .012 + .5
    tw = np.abs((k - np.floor(k)) - .5) * 2
    # (v > .5 is the chin, in front of the upper lip: hide all round, so the closed mouth shows nothing)
    front = smooth(.56, .5, v)
    teeth = (u < lip - .015) & (u > lip - .12 * (1 - .5 * tw)) & (tw < .8) & (v > .15) & (v < .47)
    inside = mix(C["mouth"], C["mouth_hi"], smooth(.0, .3, u))
    col = mix(col, inside, smooth(lip - .03, lip - .08, u) * front)
    col = np.where(teeth[..., None], mix(C["tooth"], C["tooth_dark"], .6 * tw), col)
    return _shade(col, t)


def paint_ear(t):
    """Upright horse ears: hide outside, a darker hollow, pale hair at the rim, a dark tip."""
    along, u = t["along"], t["u"]
    col = mix(C["purplehide"], C["hide"], smooth(.0, .5, along))
    col = mix(col, C["groove"], 0.6 * smooth(.45, .75, u) * smooth(.1, .3, along))
    col = mix(col, C["hair_mid"], 0.5 * np.exp(-((u - .4) / .08) ** 2) * smooth(.2, .5, along))
    col = mix(col, C["hide_dark"], smooth(.75, .95, along))
    return _shade(col, t)


def paint_mane(t):
    """The spike mane: charcoal at the root rising to dark bone, ridged, the tips fading to the dim purple
    glint of the eyes (not bright)."""
    u, along = t["u"], t["along"]
    col = mix(C["hide_dark"], C["spike"], smooth(.0, .45, along))
    col = col * (0.82 + 0.18 * np.abs(np.sin(np.pi * along * 6)))[..., None]       # growth ridges
    col = mix(col, col * .6, 0.5 * smooth(.3, .9, np.abs(u - .5) * 2))              # darker at the edges
    col = mix(col, C["spike_tip"], 0.75 * smooth(.6, .95, along))
    return _shade(col, t)


PAINTERS = {
    ("Sejanus_Body", "torso"): paint_torso, ("Sejanus_Body", "neck"): paint_neck,
    ("Sejanus_Limbs", "tail"): paint_tail, ("Sejanus_Limbs", "thigh"): _unwide("thigh", paint_thigh),
    ("Sejanus_Limbs", "shank"): _unwide("shank", paint_shank), ("Sejanus_Limbs", "arm"): paint_arm,
    ("Sejanus_Limbs", "finger"): paint_finger,
    ("Sejanus_Head", "skull"): paint_skull, ("Sejanus_Head", "jaw"): paint_jaw,
    ("Sejanus_Head", "ear"): paint_ear, ("Sejanus_Head", "mane"): paint_mane,
}


def bake(parts, occlusion=None, size=SIZE):
    """-> {texture name: PIL RGB image (128 x 128)}."""
    return _bake(parts, size=size, occlusion=occlusion, atlas=ATLAS, painters=PAINTERS, local=island_local,
                 ring_islands=())
