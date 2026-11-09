"""The Boss's three 128 x 128 skins, shaded the DK2 way (priestess_texture.shade):

  Boss_Body   the jacket and waistcoat: charcoal pinstripe, satin lapels, a white shirt, a blood-red tie
              with a gold pin, gold buttons, a pocket square, a gold watch chain over the paunch
  Boss_Limbs  sleeves, white cuffs with gold links, hands with gold rings, pinstripe trousers with a crease,
              polished black shoes, the jacket tails
  Boss_Head   the white box (printed mark on the front, a faint shadow of the seams), the neck, the cigar,
              the briefcase

The mark is original (MARK: a $ by default; any initials the user invents for his corporation).

LICENCE: GPL-3.0.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from scripts.boss_model import ARM, BODY, BOX_C, BOX_H, HEAD, LEG, LIMBS
from scripts.dragon_texture import fbm, mix, smooth
from scripts.lowpoly import bake
from scripts.priestess_texture import RGB, band, gauss, ret, shade
from scripts.soles_model import split_front_back

MARK = "$"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

ATLAS = {
    BODY: {"front": ((0, 0, 64, 128), False), "back": ((64, 0, 128, 128), False)},
    LIMBS: {"arm": ((0, 0, 30, 100), False), "cuff": ((0, 102, 30, 112), False), "hand": ((0, 114, 30, 128), False),
            "leg": ((32, 0, 70, 128), False), "shoe": ((72, 0, 100, 40), False),
            "tails": ((72, 42, 128, 100), False), "tails_in": ((102, 0, 128, 40), False)},
    HEAD: {"front": ((0, 0, 64, 64), False), "back": ((64, 0, 96, 32), False), "left": ((96, 0, 128, 32), False),
           "right": ((64, 34, 96, 66), False), "top": ((96, 34, 128, 66), False), "bottom": ((0, 66, 16, 82), False),
           "neck": ((18, 66, 34, 82), False), "cigar": ((36, 66, 64, 82), False),
           "case": ((0, 84, 64, 128), False), "handle": ((66, 84, 96, 128), False)},
}

SUIT = RGB(58, 58, 66)
SHIRT = RGB(236, 234, 228)
TIE = RGB(150, 16, 22)
GOLD = RGB(220, 172, 70)
SKIN = RGB(222, 178, 160)
BOX = RGB(238, 236, 230)
LEATHER = RGB(140, 84, 44)

BODY_BOX = (-.13, .425, .13, .86)


def planar(box, mirror=False):
    x0, y0, x1, y1 = box

    def uv(p):
        u = float(np.clip((p[0] - x0) / (x1 - x0), 0, 1))
        return (1 - u if mirror else u, float(np.clip((y1 - p[1]) / (y1 - y0), 0, 1)))
    return uv


def project_uvs(parts):
    for p in parts:
        if p.name == "torso":
            split_front_back(p, "front", "back", planar(BODY_BOX), planar(BODY_BOX, True), z_split=0.0)


def _mark():
    """The printed mark as a mask (1 = ink) over the box's front, 256 x 256."""
    im = Image.new("L", (256, 256), 0)
    d = ImageDraw.Draw(im)
    size = 190 if len(MARK) == 1 else int(200 / max(len(MARK), 1) * 1.6)
    f = ImageFont.truetype(FONT, size)
    bb = d.textbbox((0, 0), MARK, font=f)
    d.text(((256 - (bb[2] - bb[0])) / 2 - bb[0], (256 - (bb[3] - bb[1])) / 2 - bb[1] - 6), MARK, font=f, fill=255)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255


MARK_IMG = _mark()

# ------------------------------------------------------------------------------------------- jacket


def stripes(x, every=.016, w=.0016):
    return gauss(np.mod(x + every / 2, every) - every / 2, w)


def body_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    n = len(x)
    front = smooth(-.01, .01, z)
    folds = fbm(x * 30, y * 10, 3)
    cloth = SUIT * (.85 + .3 * folds)[:, None]
    col = mix(cloth, SUIT * 1.7, .35 * stripes(x))
    h = .002 * folds
    spec, metal = np.full(n, .08), np.zeros(n)
    # the open front: shirt in a V from the collar to the waistcoat's top button, the tie down it
    v_half = .006 + .55 * np.clip(y - .62, 0, None)
    shirt = smooth(v_half + .003, v_half - .003, ax) * band(y, .62, .83, .003) * front
    col = mix(col, SHIRT * (.9 + .1 * folds[:, None]), shirt)
    tie_w = .006 + .05 * np.clip(.79 - y, 0, None) * smooth(.64, .7, y) + .01 * smooth(.7, .64, y)
    tie = smooth(tie_w + .002, tie_w - .002, ax) * band(y, .6, .8, .003) * front
    knot = gauss(np.hypot(x, (y - .79) * 1.2), .008) * front
    col = mix(col, TIE * (.85 + .3 * folds[:, None]), np.clip(tie + smooth(.4, .7, knot), 0, 1))
    col = mix(col, TIE * .55, gauss(np.abs(np.mod(y * 70, 1) - .5), .08) * tie * .6)      # diagonal rep stripes
    pin = gauss(np.hypot(x, y - .7), .0035) * front
    col = mix(col, GOLD, smooth(.3, .7, pin))
    # satin lapels either side of the V
    lapel = band(ax - v_half, .002, .022, .002) * band(y, .66, .8, .004) * front
    col = mix(col, SUIT * .55, lapel)
    col = mix(col, SUIT * 1.6, lapel * gauss(ax - v_half - .004, .002))
    h += .002 * lapel
    # gold buttons down the waistcoat over the paunch, a pocket square, the watch chain
    buttons = sum(gauss(np.hypot(x, y - by), .0045) for by in (.6, .565, .53, .495)) * front
    col = mix(col, GOLD, smooth(.3, .7, buttons))
    sq = band(x - .07, -.015, .015, .002) * band(y, .7, .72, .002) * front
    col = mix(col, mix(SHIRT, GOLD, .3), sq)
    chain_y = .52 - 12 * (x - .03) ** 2
    chain = gauss(y - chain_y, .0018) * band(x, .0, .07, .003) * front
    col = mix(col, GOLD, chain)
    metal = metal + .8 * np.clip(buttons * 2 + chain + pin, 0, 1)
    spec = spec + 1.0 * np.clip(buttons * 2 + chain + pin, 0, 1)
    # shirt collar points over the lapels, a shadow under them
    collar = band(y - (.8 - .9 * (ax - .012)), -.006, .006, .002) * band(ax, .008, .04, .003) * front
    col = mix(col, SHIRT, collar)
    col = mix(col, SUIT * .25, .7 * gauss(y - (.79 - .9 * (ax - .012)), .004) * band(ax, .01, .045, .004) * front)
    # tension creases pulling from the buttons across the paunch
    for by in (.6, .565, .53):
        ang = np.arctan2(y - by, ax)
        cr = gauss(np.sin(ang * 3) * np.hypot(ax, y - by), .002) * smooth(.06, .015, np.hypot(ax, y - by))
        col = col * (1 - .35 * cr * front)[:, None]
        h -= .0008 * cr * front
    # flap pockets at the hips, a breast pocket, stitched seams
    flapm = band(ax, .05, .09, .002) * band(y, .49, .505, .002) * front
    col = mix(col, SUIT * .6, flapm)
    col = mix(col, SUIT * .2, gauss(y - .489, .0015) * band(ax, .05, .09, .002) * front)
    seam = (gauss(ax - .095, .002) * smooth(.48, .55, y) + gauss(x, .0025) * (1 - front) * smooth(.44, .5, y))
    col = col * (1 - .45 * seam)[:, None]
    # the back: a vent and a shaded drape
    back = 1 - front
    col = mix(col, SUIT * .3, back * gauss(x, .003) * band(y, .44, .5, .004))
    neck = smooth(.808, .816, y)
    col = mix(col, SKIN * np.ones((n, 1)), neck)
    # DK2-style painted shadow: dark sides, under the arms, under the paunch, under the lapels and tie
    occl = (.55 * smooth(.07, .12, ax) + .45 * smooth(.1, .125, ax) * smooth(.68, .75, y)
            + .35 * gauss(y - .47, .02) * front + .3 * lapel * gauss(ax - v_half, .004)
            + .25 * tie * gauss(ax - tie_w, .003) + .4 * smooth(.62, .5, y) * smooth(.04, .1, ax))
    col = col * (1 - .6 * np.clip(occl, 0, 1))[:, None]
    col = col * (1 + .25 * gauss(np.hypot(ax - .035, y - .7), .03) * front)[:, None]     # light on the chest
    return ret(col, h, spec=spec, gloss=12 + 30 * metal, metal=metal, soft=.4)


def _along(pos, path):
    ys = np.array([c[1] for c, *_ in path])
    return np.interp(-pos[:, 1], -ys, np.linspace(0, 1, len(ys)))


def arm_field(pos, t):
    x, y = pos[:, 0], pos[:, 1]
    folds = fbm(x * 40, y * 20, 5)
    col = SUIT * (.85 + .3 * folds)[:, None]
    col = mix(col, SUIT * 1.7, .3 * stripes(t["u"] * .1))
    f = _along(pos, ARM)
    elbow = gauss(f - .55, .1) * (.5 + .5 * np.sin(y * 420 + folds * 4))
    inner = smooth(.4, .9, t["u"])
    col = col * (1 - .45 * elbow - .45 * inner - .3 * gauss(f, .12))[:, None]
    seam = gauss(t["u"] - .95, .04)
    col = col * (1 - .4 * seam)[:, None]
    return ret(col, .002 * folds + .0015 * elbow, spec=.08, gloss=10)


def cuff_field(pos, t):
    link = gauss(t["u"] - .5, .12) * gauss(t["v"] - .5, .3)
    col = mix(SHIRT * np.ones((len(pos), 1)), GOLD, smooth(.3, .6, link))
    return ret(col, .001 * link, spec=.2 + link, gloss=20 + 30 * link, metal=.8 * link)


def hand_field(pos, t):
    v, u = t["v"], t["u"]
    ring = band(v, .55, .7, .03) * gauss(u - .2, .15)
    knuckle = .5 + .5 * np.cos(u * 4 * 2 * np.pi)
    col = SKIN * (.9 + .12 * knuckle)[:, None]
    col = mix(col, GOLD, smooth(.3, .6, ring))
    gem = ring * gauss(u - .2, .05)
    col = mix(col, RGB(200, 20, 40), smooth(.4, .8, gem))
    return ret(col, .0008 * knuckle + .0015 * ring, spec=.1 + ring, gloss=14 + 30 * ring, metal=.7 * ring, soft=.8)


def leg_field(pos, t):
    x, y = pos[:, 0], pos[:, 1]
    u = t["u"]
    folds = fbm(x * 40, y * 12, 7)
    col = SUIT * (.82 + .3 * folds)[:, None]
    col = mix(col, SUIT * 1.7, .3 * stripes(u * .12, .016, .0016))
    crease = gauss(u, .04) + gauss(u - 1, .04)
    col = mix(col, SUIT * 1.35, .5 * crease)
    break_ = gauss(y - .085, .016) * (.5 + .5 * np.sin(u * 30 + folds * 4))
    knee = gauss(y - .27, .03) * (.5 + .5 * np.sin(y * 500 + u * 9))
    inner = smooth(.4, .9, u)
    col = col * (1 - .4 * break_ - .45 * smooth(.42, .47, y) - .3 * knee - .4 * inner)[:, None]
    return ret(col, .002 * folds + .001 * crease, spec=.08, gloss=10)


def shoe_field(pos, t):
    y = pos[:, 1]
    sole = smooth(.012, .004, y)
    col = mix(RGB(26, 22, 24), RGB(10, 8, 10), sole) * np.ones((len(y), 1))
    lace = band(t["u"], 0, .2, .03) * band(t["v"], .2, .6, .05)
    col = mix(col, RGB(40, 34, 36), lace)
    return ret(col, .0008 * lace, spec=1.1 * (1 - sole), gloss=60, metal=.15)


def tails_field(pos, t):
    face = t.get("face", np.ones(len(pos)))
    folds = fbm(pos[:, 0] * 40, pos[:, 1] * 10, 9)
    col = SUIT * (.8 + .3 * folds)[:, None]
    col = mix(col, SUIT * 1.7, .3 * stripes(pos[:, 0]))
    col = np.where((face < 0)[:, None], TIE * .45, col)                          # a red silk lining
    return ret(col, .002 * folds, spec=.08 + .3 * (face < 0), gloss=10)


# ------------------------------------------------------------------------------------------- the box and the rest

def box_front(pos, t):
    u, v = t["u"], t["v"]
    H, W = MARK_IMG.shape
    ink = MARK_IMG[np.clip((v * H).astype(int), 0, H - 1), np.clip((u * W).astype(int), 0, W - 1)]
    grime = fbm(u * 6, v * 6, 21)
    card = BOX * (.88 + .12 * fbm(u * 40, v * 40, 11) - .25 * smooth(.5, .95, grime) * smooth(.3, 1.0, v))[:, None]
    shadow = smooth(.0, .12, np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)))     # the bevel's shade
    col = card * (.55 + .45 * shadow)[:, None] * (1 - .3 * smooth(.6, 1.0, v))[:, None]   # darker toward the jaw
    dent = gauss(np.hypot(u - .78, v - .25), .06) + gauss(np.hypot(u - .2, v - .8), .08)
    col = col * (1 - .2 * dent)[:, None]
    # the ink's printed drop shadow
    H, W = MARK_IMG.shape
    sh = MARK_IMG[np.clip(((v - .02) * H).astype(int), 0, H - 1), np.clip(((u - .02) * W).astype(int), 0, W - 1)]
    col = col * (1 - .45 * sh)[:, None]
    col = mix(col, GOLD * .95, ink)
    col = mix(col, RGB(70, 50, 20), .7 * gauss(ink - .5, .12))                            # a dark rim on the ink
    return ret(col, .0015 * ink, spec=.15 + .8 * ink, gloss=12 + 40 * ink, metal=.6 * ink, soft=.2)


def box_side(pos, t):
    u, v = t["u"], t["v"]
    grime = fbm(u * 6, v * 6, 23)
    card = BOX * (.86 + .1 * fbm(u * 40, v * 40, 13) - .3 * smooth(.45, .95, grime))[:, None]
    tape = band(u, .42, .58, .01) * (t["nrm"][:, 1] > .5)                         # packing tape over the top
    card = mix(card, RGB(200, 170, 110), .6 * tape)
    edge = smooth(.0, .14, np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)))
    card = card * (.6 + .4 * edge)[:, None] * (1 - .35 * smooth(.55, 1.0, v) * (t["nrm"][:, 1] < .5))[:, None]
    nrm = t["nrm"]
    lift = 1.0 + .2 * np.clip(np.abs(nrm[:, 0]), 0, 1)
    return ret(card * ((.86 + .14 * edge) * lift)[:, None], np.zeros(len(u)), spec=.15, gloss=12, soft=1.0)


def neck_field(pos, t):
    return ret(SKIN * np.ones((len(pos), 1)), np.zeros(len(pos)), spec=.05, gloss=10, soft=.8)


def cigar_field(pos, t):
    along = t.get("along", np.zeros(len(pos)))
    leaf = fbm(t["u"] * 8, along * 20, 15)
    col = mix(RGB(90, 52, 26), RGB(130, 80, 40), leaf)
    col = mix(col, RGB(200, 30, 20), gauss(along - .2, .04))                        # the band
    col = mix(col, GOLD, gauss(along - .2, .015))
    ash = smooth(.9, .97, along)
    col = mix(col, RGB(150, 146, 140), ash)
    m = ret(col, .0005 * leaf, spec=.1, gloss=10)
    m["glow"] = (smooth(.97, 1.0, along) * 255)[:, None] * np.array([1.0, .45, .1])
    return m


def case_field(pos, t):
    u, v = t["u"], t["v"]
    grain = fbm(u * 40, v * 40, 17)
    col = LEATHER * (.8 + .35 * grain)[:, None]
    stitch = gauss(np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)) - .05, .01)
    col = mix(col, LEATHER * .45, stitch)
    col = col * (.55 + .45 * smooth(.0, .15, np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v))))[:, None]
    clasp = sum(gauss(np.hypot(u - cu, v - .12), .04) for cu in (.3, .7))
    col = mix(col, GOLD, smooth(.3, .6, clasp))
    return ret(col, .001 * grain + .002 * clasp, spec=.3 + clasp, gloss=20 + 30 * clasp, metal=.7 * clasp)


def handle_field(pos, t):
    return ret(LEATHER * .6 * np.ones((len(pos), 1)), np.zeros(len(pos)), spec=.3, gloss=20)


S = lambda f: (lambda t: shade(t, f))
PAINTERS = {
    (BODY, "front"): S(body_field), (BODY, "back"): S(body_field),
    (LIMBS, "arm"): S(arm_field), (LIMBS, "cuff"): S(cuff_field), (LIMBS, "hand"): S(hand_field),
    (LIMBS, "leg"): S(leg_field), (LIMBS, "shoe"): S(shoe_field),
    (LIMBS, "tails"): S(tails_field), (LIMBS, "tails_in"): S(tails_field),
    (HEAD, "front"): S(box_front), **{(HEAD, k): S(box_side) for k in ("back", "left", "right", "top", "bottom")},
    (HEAD, "neck"): S(neck_field), (HEAD, "cigar"): S(cigar_field), (HEAD, "case"): S(case_field),
    (HEAD, "handle"): S(handle_field),
}


def bake_boss(parts, occlusion=None):
    return bake(parts, ATLAS, PAINTERS, occlusion=occlusion, apply_ao=True)
