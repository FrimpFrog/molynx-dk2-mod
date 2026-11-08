"""Soles's three skins, shaded the DK2 way (priestess_texture.shade):

  Soles_Head   his likeness painted (no photo pixels): his eyes, brows, nose, thin moustache and mouth placed
               from his photo's proportions, a neutral expression; his dark wavy hair with a middle parting
  Soles_Body   the red T-shirt (front and back projections, like the game's own skins): soft folds, a ribbed
               collar, an original white flame emblem outlined in black, his silver chain, and a studded
               leather baldric across his chest
  Soles_Limbs  sleeves and arms, hands, black leather trousers, boots with straps, the red loincloth with
               a dark border, the belt and buckle, leather bracers, the steel pauldron

The emblem is original (a flame), not the print on his real shirt, which is a company's artwork.

LICENCE: GPL-3.0.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scripts.dragon_texture import fbm, mix, smooth
from scripts.lowpoly import bake
from scripts.priestess_texture import RGB, band, gauss, ret, shade
from scripts.soles_model import ARM, BODY, BOOT_TOP, HAIRLINE, HC, HEAD, LIMBS, split_front_back

ATLAS = {
    HEAD: {"front": ((0, 0, 84, 96), False), "back": ((86, 0, 128, 52), False)},
    BODY: {"front": ((0, 0, 64, 128), False), "back": ((64, 0, 128, 128), False)},
    LIMBS: {"arm": ((0, 0, 30, 96), False), "hand": ((0, 98, 30, 128), False),
            "leg": ((32, 0, 62, 128), False), "foot": ((64, 0, 92, 28), False),
            "cloth": ((64, 30, 96, 74), False), "cloth_in": ((64, 76, 96, 96), False),
            "belt": ((98, 0, 128, 18), False), "bracer": ((98, 20, 128, 50), False),
            "pauldron": ((98, 52, 128, 84), False)},
}

SKIN = RGB(206, 150, 124)
RED = RGB(190, 32, 38)
HAIR = RGB(70, 50, 38)
LEATHER_BLACK = RGB(40, 36, 40)
LEATHER = RGB(118, 72, 38)
STEEL = RGB(118, 122, 136)
BRONZE = RGB(196, 148, 66)

HEAD_BOX = (-.08, .826, .08, 1.004)            # x0, y0, x1, y1 of the head's front / back projections
BODY_BOX = (-.125, .425, .125, .875)


def planar(box, mirror=False):
    x0, y0, x1, y1 = box

    def uv(p):
        u = (p[0] - x0) / (x1 - x0)
        return (1 - u if mirror else u, (y1 - p[1]) / (y1 - y0))
    return uv


def project_uvs(parts):
    """Front and back projections on the head and the torso (the game's way)."""
    for p in parts:
        if p.name == "head":
            split_front_back(p, "front", "back", planar(HEAD_BOX), planar(HEAD_BOX, True), z_split=HC[2])
        elif p.name == "torso":
            split_front_back(p, "front", "back", planar(BODY_BOX), planar(BODY_BOX, True), z_split=0.0)


# ------------------------------------------------------------------------------------------- head

def hair_col(pos):
    x, y, z = pos[:, 0] - HC[0], pos[:, 1], pos[:, 2] - HC[2]
    ang = np.arctan2(x, -z)
    curls = fbm(ang * 5 + 40, y * 60, 5)
    wave = .5 + .5 * np.sin(ang * 22 + np.sin(y * 180 + ang * 3) * 1.6 + curls * 5)
    h = .004 * curls + .0016 * wave
    col = mix(HAIR * .38, HAIR * 1.15, .2 + .45 * curls + .3 * wave)
    return col, h


def hair_field(pos, t):
    col, h = hair_col(pos)
    return ret(col, h, spec=.3, gloss=12, soft=.4)


# His likeness, painted (no photo pixels): proportions and features measured from his photo, then drawn
# the way DK2's painted faces are (the Knight's, Loobinex's): x is from the middle of his face, y up.
EYE_Y, EYE_X = .909, .0265
BROW_Y = .9215
NOSE_Y = .882
MOUSTACHE_Y = .8705
MOUTH_Y = .8635
CHIN_Y = .842
EYES = RGB(112, 128, 120)          # grey-green
BROW = RGB(58, 40, 30)


def hairline(x):
    """The hairline's height across his forehead: a middle parting with a wave falling to his right."""
    ax = np.abs(x)
    base = .946 - .01 * smooth(.02, .05, ax) + .012 * smooth(.042, .055, ax)
    lock = .014 * gauss(x + .016, .012)                    # the wave over the right of his forehead
    return base - lock


def face_field(pos, t):
    x, y, z = pos[:, 0] - HC[0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    n = len(x)
    col = SKIN * np.ones((n, 1))
    h = np.zeros(n)
    # skin tone: a paler forehead, rosy cheeks and nose, a shadow under the jaw and at the temples
    col = mix(col, SKIN * RGB(1.06, 1.04, 1.03), .5 * smooth(.92, .94, y))
    col = mix(col, RGB(214, 128, 112), .32 * gauss(np.hypot(ax - .03, y - .888), .012))
    col = mix(col, RGB(214, 128, 112), .2 * gauss(np.hypot(x, y - NOSE_Y), .007))
    col = mix(col, SKIN * .62, .5 * smooth(.85, .835, y))
    col = mix(col, SKIN * .72, .35 * smooth(.04, .055, ax))
    col *= (.94 + .1 * fbm(x * 300, y * 300, 4))[:, None]                    # skin texture
    # bone structure
    h += .0018 * gauss(y - BROW_Y, .004) * band(ax, .008, .042, .006)           # brow ridge
    h -= .0018 * gauss(np.hypot(ax - EYE_X, (y - EYE_Y) * 1.3), .01)            # eye sockets
    h += .0014 * gauss(np.hypot(ax - .033, y - .89), .01)                       # cheekbones
    h += .0016 * gauss(np.hypot(x, (y - MOUTH_Y) * 1.6), .01)                   # the mouth's mound
    h += .0012 * gauss(np.hypot(x, y - .848), .008)                             # the chin
    h -= .0008 * gauss(np.hypot(ax - .02, y - .878), .006)                      # the nose-to-mouth lines
    col = mix(col, SKIN * .8, .45 * gauss(np.hypot(ax - .018, (y - .873) * .6), .004))
    col = mix(col, SKIN * .78, .35 * gauss(np.hypot(ax - EYE_X, y - EYE_Y + .006), .007))   # under the eyes
    # eyebrows: thick, dark, straight with a lift at the outer end
    by = BROW_Y + .0015 - .0015 * smooth(.01, .045, ax)          # straight, a touch higher inside
    brow = band(y - by, -.0028, .0028, .0008) * band(ax, .006, .046, .004)
    brow *= .7 + .3 * (.5 + .5 * np.sin(x * 2200 + y * 900))
    col = mix(col, BROW, np.clip(brow, 0, 1))
    h += .0005 * brow
    # eyes: almond, a heavy upper lid, white, grey-blue iris, pupil, a highlight
    ex, ey = (ax - EYE_X) / .0118, (y - EYE_Y) / .0055
    shape = ex ** 2 + (ey + .12 * ex) ** 2
    eye = shape < 1
    white = RGB(232, 226, 220)
    col = np.where(eye[:, None], white, col)
    iris = np.hypot((ax - EYE_X + .0006) / .0047, (y - EYE_Y + .0004) / .0047)
    col = np.where((eye & (iris < 1))[:, None], mix(EYES * .7, EYES * 1.15, smooth(1, .3, iris)), col)
    col = np.where((eye & (iris < .42))[:, None], RGB(14, 12, 14), col)
    hl = np.hypot(ax - EYE_X + .0022, y - EYE_Y - .0016) < .0011
    col = np.where((eye & hl)[:, None], RGB(255, 255, 255), col)
    upper = (np.abs(shape - 1) < .55) & (y > EYE_Y - .001)
    col = np.where(upper[:, None], RGB(40, 26, 24), col)
    crease = gauss(np.hypot(ax - EYE_X, (y - EYE_Y - .0085) * 2.2) - .012, .0015) * (y > EYE_Y)
    col = mix(col, SKIN * .62, .6 * crease)
    lower = (np.abs(shape - 1) < .4) & (y < EYE_Y - .001)
    col = mix(col, RGB(150, 92, 86), .55 * lower)
    # the nose: a lit bridge, shaded sides, nostrils
    col = mix(col, SKIN * 1.12, .45 * gauss(x, .003) * band(y, .885, .915, .004))
    col = mix(col, SKIN * .74, .35 * gauss(ax - .0085, .003) * band(y, .878, .905, .004))
    nos = gauss(np.hypot((ax - .0065) / 1.5, y - (NOSE_Y - .0055)), .0022)
    col = mix(col, RGB(70, 38, 34), .85 * nos)
    # his moustache: a thin, dark, slightly drooping line over the lip, thicker in the middle
    my = MOUSTACHE_Y - .004 * smooth(.006, .022, ax)
    thick = .0028 - .0016 * smooth(.004, .022, ax)
    mo = band(y - my, -thick, thick, .0008) * smooth(.025, .02, ax) * (1 - .6 * gauss(x, .0025))
    mo *= .75 + .25 * (.5 + .5 * np.sin(x * 2600))
    col = mix(col, RGB(66, 44, 32), np.clip(mo, 0, 1))
    h += .0004 * mo
    # lips: pale pink, a darker line between them, a little shine on the lower lip
    lips = ((x / .0155) ** 2 + ((y - MOUTH_Y) / .0042) ** 2) < 1
    lcol = mix(RGB(196, 120, 112), RGB(176, 100, 96), smooth(MOUTH_Y, MOUTH_Y + .004, y))
    lcol = mix(lcol, RGB(232, 170, 160), .35 * gauss(np.hypot(x * .5, y - MOUTH_Y + .0022), .0018))
    col = np.where(lips[:, None], lcol, col)
    col = mix(col, RGB(96, 50, 48), gauss(y - MOUTH_Y - .0004, .0007) * band(ax, 0, .015, .002))
    col = mix(col, SKIN * .75, .4 * gauss(np.hypot(x, y - (MOUTH_Y - .0085)), .004))   # under the lip
    # his hair over everything above the hairline and round the sides above the ears
    hl_y = hairline(x)
    hair = np.maximum(smooth(hl_y - .002, hl_y + .003, y),
                      smooth(.045, .051, ax) * smooth(.878, .89, y))           # down over his ears
    hcol, hh = hair_col(pos)
    col = mix(col, hcol, hair)
    h = h * (1 - hair) + hh * hair
    spec = .14 * (1 - hair) + .3 * hair + .55 * eye + .25 * lips
    m = ret(col, h, spec=spec, gloss=14 + 30 * eye, soft=.85 * (1 - hair) * (1 - eye))
    m["glow"] = np.where(eye[:, None], 18.0, 0.0) * np.ones((1, 3))
    return m


def head_front(t):
    return shade(t, face_field)


# ------------------------------------------------------------------------------------------- shirt

def _emblem():
    """An original emblem: three white flames outlined in black, with a red inner tongue in the middle one,
    as masks over the chest (x -.06 .. .06, y .57 .. .72 in model units)."""
    W, H = 240, 300

    def flame(cx, base, w, h, lean):
        """A flame: a round base and a tongue that narrows and curls over to one side."""
        ks = np.linspace(0, 1, 30)
        half = w * np.sin(np.pi * (.12 + .88 * (1 - ks)) / 1.0) ** .8 * (1 - ks) ** .35
        mid = cx + lean * w * ks ** 2.2 + .25 * w * np.sin(ks * np.pi * 1.5) * ks
        ys = base - h * ks
        right = list(zip(mid + half, ys))
        left = list(zip(mid - half, ys))[::-1]
        bottom = [(cx + w * .95 * np.cos(a), base + w * .35 * np.sin(a)) for a in np.linspace(np.pi, 0, 10)][::-1]
        return bottom + right + left

    shapes = [flame(120, 250, 56, 215, .45), flame(66, 258, 32, 120, -.9), flame(174, 258, 32, 120, .9)]
    fill = Image.new("L", (W, H), 0)
    for s in shapes:
        ImageDraw.Draw(fill).polygon(s, fill=255)
    outline = fill.filter(ImageFilter.MaxFilter(11))
    inner = Image.new("L", (W, H), 0)
    ImageDraw.Draw(inner).polygon(flame(122, 246, 22, 105, .4), fill=255)
    f = lambda im: np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    return f(fill), f(outline), f(inner)


EMBLEM = _emblem()
EMBLEM_BOX = (-.06, .57, .06, .72)


def emblem_at(pos):
    x0, y0, x1, y1 = EMBLEM_BOX
    fill, outline, inner = EMBLEM
    H, W = fill.shape
    u = ((pos[:, 0] - x0) / (x1 - x0) * W).astype(int)
    v = ((y1 - pos[:, 1]) / (y1 - y0) * H).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    return fill[v, u] * ok, outline[v, u] * ok, inner[v, u] * ok


def torso_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    front = smooth(-.01, .01, z)
    folds = fbm(x * 26, y * 8, 41)
    drape = (.5 + .5 * np.sin(y * 260 + ax * 90 + folds * 6)) * smooth(.62, .48, y)
    h = .0035 * folds + .0009 * drape
    col = mix(RED * .7, RED * 1.08, .3 + .5 * folds + .15 * drape)
    col = mix(col, RED * .55, .5 * smooth(.09, .115, ax))
    neckline = .816 - .016 * np.clip(1 - (ax / .04) ** 2, 0, 1) * front - .004 * (1 - front)
    collar = band(y, neckline - .007, neckline, .0015)
    col = mix(col, RED * .55, collar)
    h += .0006 * collar
    col = mix(col, SKIN * np.ones((len(x), 1)), smooth(neckline - .001, neckline + .001, y))
    fill, outline, inner = emblem_at(pos)
    col = mix(col, RGB(20, 12, 14), outline * front)
    col = mix(col, RGB(240, 236, 228) * (.9 + .1 * folds[:, None]), fill * front)
    col = mix(col, RED * 1.1, inner * front)
    line_y = .5 + (x + .08) / .17 * .28
    strap = band(y - line_y, -.012, .012, .002) * band(x, -.11, .1, .01)
    studs = gauss(np.hypot(np.mod(x + .006, .022) - .011, y - line_y), .003) * strap
    h += strap * .004 + studs * .0025
    col = mix(col, LEATHER * (.8 + .3 * fbm(x * 150, y * 150, 7)[:, None]), strap)
    col = mix(col, LEATHER * .5, strap * gauss(np.abs(y - line_y) - .011, .0018))
    col = mix(col, BRONZE * 1.1, smooth(.3, .7, studs))
    chain_y = .818 - .085 * smooth(.042, 0, ax) ** .7
    links = .5 + .5 * np.sin(np.hypot(x, y) * 1400)
    chain = gauss(y - chain_y, .0015) * band(ax, 0, .042, .003) * front
    pend = gauss(np.hypot(x, (y - .728) * .9), .0055) * front
    h += .002 * chain * links + .004 * pend
    metal = np.clip(chain * 1.5 + pend * 1.5 + studs, 0, 1)
    col = mix(col, RGB(214, 216, 224) * (.55 + .45 * links[:, None]), np.clip(chain * 1.3, 0, 1))
    col = mix(col, RGB(205, 205, 215), smooth(.4, .8, pend))
    return ret(col, h, spec=.1 + .9 * metal + .25 * strap, gloss=10 + 40 * metal, metal=.7 * metal, soft=.6)


# ------------------------------------------------------------------------------------------- limbs

def _along(pos, path):
    ys = np.array([c[1] for c, *_ in path])
    return np.interp(-pos[:, 1], -ys, np.linspace(0, 1, len(ys)))


def arm_field(pos, t):
    f = _along(pos, ARM)
    sleeve = smooth(.36, .31, f)
    hem = band(f, .29, .33, .006)
    folds = fbm(pos[:, 0] * 40, pos[:, 1] * 40, 51)
    skin = SKIN * (0.92 + .14 * fbm(pos[:, 0] * 90, pos[:, 1] * 90, 3))[:, None]
    col = mix(skin, mix(RED * .72, RED * 1.05, folds), sleeve)
    col = mix(col, RED * .55, hem)
    h = .003 * sleeve * folds + .0012 * hem + .0009 * gauss(f - .55, .05) * (1 - sleeve)
    col = mix(col, SKIN * .8, .3 * gauss(f - .55, .04) * (1 - sleeve))
    return ret(col, h, spec=.15, gloss=12, soft=.8 * (1 - sleeve))


def hand_field(pos, t):
    y = pos[:, 1]
    k = .5 + .5 * np.sin(y * 900)
    col = mix(SKIN, SKIN * .78, .5 * k * smooth(.45, .42, y))
    return ret(col * np.ones((len(y), 1)), .0007 * k * smooth(.45, .42, y), spec=.15, gloss=12, soft=.9)


def leg_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    boot = smooth(BOOT_TOP + .004, BOOT_TOP - .004, y)
    grain = fbm(x * 220, y * 220, 13)
    creases = (gauss(y - .27, .03) + gauss(y - .43, .02)) * (.5 + .5 * np.sin(y * 1100 + grain * 4))
    seam = gauss(np.abs(np.abs(x) - .07) - .044, .003)
    h = (1 - boot) * (.0025 * fbm(x * 40, y * 12, 19) - .0008 * creases) - .0005 * seam
    trousers = mix(LEATHER_BLACK * 1.3, LEATHER_BLACK * .6, .5 * creases + .3 * grain + .4 * seam)
    cuff = band(y, BOOT_TOP - .03, BOOT_TOP, .002)
    straps = band(y, .155, .166, .0015) + band(y, .105, .116, .0015)
    ank = gauss(y - .075, .012) * (.5 + .5 * np.sin(y * 1500 + grain * 4))
    bcol = mix(LEATHER, LEATHER * .55, .35 * grain + .45 * ank)
    bcol = mix(bcol, LEATHER * 1.18, cuff)
    bcol = mix(bcol, LEATHER_BLACK, straps)
    buckles = straps * gauss(x - np.sign(x) * .1, .008)
    bcol = mix(bcol, RGB(200, 202, 212), smooth(.3, .6, buckles))
    bh = .003 + .003 * cuff + .0018 * straps - .0006 * ank + .0003 * grain + .002 * buckles
    col = mix(trousers, bcol, boot)
    h = h * (1 - boot) + bh * boot
    return ret(col, h, spec=.45 * (1 - boot) + .3 * boot + .8 * buckles, gloss=22 - 6 * boot,
               metal=.15 * (1 - boot) + .6 * buckles)


def foot_field(pos, t):
    y = pos[:, 1]
    sole = smooth(.012, .004, y)
    grain = fbm(pos[:, 0] * 200, pos[:, 2] * 200, 17)
    col = mix(mix(LEATHER, LEATHER * .6, .4 * grain), LEATHER_BLACK * .6, sole)
    return ret(col, .0012 * sole + .0004 * grain, spec=.3, gloss=16)


def cloth_field(pos, t):
    x, y = pos[:, 0], pos[:, 1]
    face = t.get("face", np.ones(len(x)))
    u, v = t["u"], t["v"]
    folds = fbm(x * 40, y * 6, 61)
    border = np.maximum(smooth(.12, .06, np.minimum(u, 1 - u)), smooth(.86, .93, v))
    trim = gauss(np.minimum(u, 1 - u) - .09, .02) + gauss(v - .895, .02)
    col = mix(RGB(176, 22, 32) * .8, RGB(176, 22, 32) * 1.1, folds)
    col = mix(col, RGB(28, 18, 20), border)
    col = mix(col, RGB(200, 170, 110), np.clip(trim, 0, 1) * .8)
    h = .004 * folds + .001 * np.clip(trim, 0, 1)
    inside = RGB(90, 12, 18) * (.7 + .3 * folds[:, None])
    col = np.where((face < 0)[:, None], inside, col)
    return ret(col, h, spec=.12, gloss=10, soft=.6)


def belt_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    front = smooth(.04, .06, z)
    grain = fbm(x * 200, y * 200, 71)
    buckle = band(np.abs(x), 0, .016, .002) * band(y, .466, .496, .002) * front
    hole = band(np.abs(x), 0, .01, .0015) * band(y, .472, .49, .0015) * front
    edge = gauss(y - .464, .002) + gauss(y - .498, .002)
    col = mix(LEATHER * .7, LEATHER * .45, .4 * grain + .5 * edge)
    me = buckle * (1 - hole)
    col = mix(col, RGB(200, 202, 212), me)
    return ret(col, .001 * grain + .004 * me - .002 * edge, spec=.25 + .9 * me, gloss=16 + 30 * me, metal=.7 * me)


def bracer_field(pos, t):
    v, u = t["v"], t["u"]
    grain = fbm(pos[:, 0] * 200, pos[:, 1] * 200, 81)
    lace = gauss(np.abs(np.mod(v * 5, 1) - .5), .2) * gauss(u - 1, .08)
    studs = gauss(np.hypot(np.mod(u * 4, 1) - .5, np.mod(v * 3, 1) - .5), .12) * smooth(.9, .7, u)
    rim = gauss(v, .08) + gauss(v - 1, .08)
    col = mix(LEATHER * .9, LEATHER * .55, .4 * grain + .5 * rim)
    col = mix(col, RGB(60, 40, 26), lace)
    col = mix(col, BRONZE * 1.1, smooth(.4, .8, studs))
    return ret(col, .0012 * grain + .003 * studs + .001 * rim, spec=.3 + .8 * studs, gloss=18 + 30 * studs,
               metal=.6 * studs)


def pauldron_field(pos, t):
    along = t.get("along", np.zeros(len(pos)))
    rim = smooth(.18, .04, along)
    rivet = gauss(along - .3, .06) * (np.cos(np.arctan2(pos[:, 2] + .012, pos[:, 1] - .78) * 5) > .9)
    h = .002 * rim + .002 * rivet + .0006 * fbm(pos[:, 0] * 300, pos[:, 2] * 300, 91)
    col = mix(STEEL, BRONZE, rim)
    col = mix(col, BRONZE * 1.1, rivet)
    col = col * (.85 + .25 * fbm(pos[:, 0] * 120, pos[:, 1] * 120, 93))[:, None]
    return ret(col, h, spec=1.0, gloss=40, metal=.75)


PAINTERS = {
    (HEAD, "front"): head_front, (HEAD, "back"): lambda t: shade(t, hair_field),
    (BODY, "front"): lambda t: shade(t, torso_field), (BODY, "back"): lambda t: shade(t, torso_field),
    (LIMBS, "arm"): lambda t: shade(t, arm_field), (LIMBS, "hand"): lambda t: shade(t, hand_field),
    (LIMBS, "leg"): lambda t: shade(t, leg_field), (LIMBS, "foot"): lambda t: shade(t, foot_field),
    (LIMBS, "cloth"): lambda t: shade(t, cloth_field), (LIMBS, "cloth_in"): lambda t: shade(t, cloth_field),
    (LIMBS, "belt"): lambda t: shade(t, belt_field), (LIMBS, "bracer"): lambda t: shade(t, bracer_field),
    (LIMBS, "pauldron"): lambda t: shade(t, pauldron_field),
}


def bake_soles(parts, occlusion=None):
    ours = [p for p in parts if not getattr(p, "game_uv", False)]
    return bake(ours, ATLAS, PAINTERS, occlusion=occlusion, apply_ao=False)
