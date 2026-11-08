"""Soles's two skins: Soles_Head (his face from his photo, projected from the front; his dark curly hair) and
Soles_Shirt (the red T-shirt, his neck, arms and hands, and the silver chain he wears), shaded the
DK2 way (priestess_texture.shade). The Dark Angel's parts keep the game's own skins.

The shirt is plain red: the print on the real shirt is someone else's artwork and is left off.

LICENCE: GPL-3.0.
"""
import numpy as np
from PIL import Image

from scripts.dragon_texture import fbm, mix, smooth
from scripts.lowpoly import bake
from scripts.priestess_texture import RGB, band, gauss, ret, shade
from scripts.soles_model import ARM_PATH, HC, HEAD, HRX, SHIRT

ATLAS = {
    HEAD: {"face": ((0, 0, 80, 96), False), "hair": ((82, 0, 128, 96), False),
           "scalp": ((0, 98, 60, 128), False)},
    SHIRT: {"torso": ((0, 0, 92, 128), False), "arm": ((94, 0, 128, 100), False),
            "hand": ((94, 102, 128, 128), False)},
}

SKIN = RGB(206, 152, 128)
RED = RGB(184, 30, 36)
HAIR = RGB(72, 52, 38)

# his photo (front, neutral): the eye line and the scale of the projection
PHOTO = None
PHOTO_EYES = (1484.0, 756.0)          # between his eyes, in photo pixels
PHOTO_K = 3040.0                      # photo pixels per model unit
EYE_Y = .911                          # the eye line on the model


def load_photo(path):
    global PHOTO
    PHOTO = np.asarray(Image.open(path).convert("RGB"), np.float32)


def photo_at(pos):
    x = PHOTO_EYES[0] + (pos[:, 0] - HC[0]) * PHOTO_K
    y = PHOTO_EYES[1] - (pos[:, 1] - EYE_Y) * PHOTO_K
    h, w = PHOTO.shape[:2]
    x = np.clip(x, 0, w - 2)
    y = np.clip(y, 0, h - 2)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    a, b = PHOTO[y0, x0], PHOTO[y0, x0 + 1]
    c, d = PHOTO[y0 + 1, x0], PHOTO[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def paint_face(t):
    pos, n = t["pos"], t["nrm"]
    col = photo_at(pos)
    # the photo is lit already: brighten its night-time exposure a little, and darken the sides of the
    # head (the projection stretches there) toward a plain skin tone
    col = col * 1.12 + 8
    side = smooth(.35, .75, np.abs(n[:, 0]))
    col = mix(col, SKIN * (0.75 + .25 * np.clip(n[:, 1] + .5, 0, 1))[:, None], side)
    return np.clip(col, 0, 255)


def hair_field(pos, t):
    x, y, z = pos[:, 0] - HC[0], pos[:, 1], pos[:, 2] - HC[2]
    ang = np.arctan2(x, -z)
    curls = fbm(ang * 5 + 40, y * 60, 5, octaves=3)
    wave = .5 + .5 * np.sin(ang * 22 + np.sin(y * 180 + ang * 3) * 1.6 + curls * 5)   # loose waves
    h = .004 * curls + .0016 * wave
    col = mix(HAIR * .45, HAIR * 1.45, .15 + .45 * curls + .4 * wave)
    return ret(col, h, spec=.35, gloss=12, soft=.4)


def scalp_field(pos, t):
    return ret(HAIR * .5 * np.ones((len(pos), 1)), np.zeros(len(pos)), spec=0.0)


def torso_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    front = smooth(-.01, .01, z)
    # the T-shirt: soft folds, a ribbed collar, darker under the arms and chest
    folds = fbm(x * 30, y * 9, 41)
    wr = .5 + .5 * np.sin(y * 300 + ax * 120 + folds * 6)
    h = .003 * folds + .0007 * wr * smooth(.6, .45, y)
    col = mix(RED * .78, RED * 1.08, .3 + .5 * folds + .15 * wr)
    neckline = .816 - .016 * np.clip(1 - (ax / .04) ** 2, 0, 1) * front - .004 * (1 - front)
    collar = band(y, neckline - .006, neckline, .0015)
    skin = smooth(neckline - .001, neckline + .001, y)
    h += .0005 * collar
    col = mix(col, RED * .62, collar)
    col = mix(col, SKIN * np.ones((len(x), 1)), skin)
    # the silver chain: a loop of links from the sides of the neck to a pendant on his chest
    chain_y = .806 - .1 * smooth(.045, 0, ax) ** .7
    links = .5 + .5 * np.sin(np.hypot(x, y) * 1400)
    chain = gauss(y - chain_y, .0016) * band(ax, 0, .045, .003) * front
    pend = gauss(np.hypot(x, (y - .697) * .9), .006) * front
    h += .002 * chain * links + .004 * pend
    metal = np.clip(chain * 1.5 + pend * 1.5, 0, 1)
    col = mix(col, RGB(210, 212, 220) * (.6 + .4 * links[:, None]), np.clip(chain * 1.3, 0, 1))
    col = mix(col, RGB(200, 200, 210), smooth(.4, .8, pend))
    return ret(col, h, spec=.12 + .9 * metal, gloss=12 + 40 * metal, metal=.7 * metal, soft=.6)


def _along(pos, path):
    ys = np.asarray(path, float)[:, 1]
    return np.interp(-pos[:, 1], -ys, np.linspace(0, 1, len(ys)))


def arm_field(pos, t):
    f = _along(pos, ARM_PATH)
    sleeve = smooth(.36, .32, f)
    hem = band(f, .3, .34, .006)
    folds = fbm(pos[:, 0] * 40, pos[:, 1] * 40, 51)
    col = mix(SKIN * (0.95 + .1 * fbm(pos[:, 0] * 80, pos[:, 1] * 80, 3))[:, None],
              mix(RED * .8, RED * 1.05, folds), sleeve)
    col = mix(col, RED * .7, hem)
    h = .0025 * sleeve * folds + .0012 * hem
    h += .0008 * gauss(f - .55, .05) * (1 - sleeve)                      # the elbow
    return ret(col, h, spec=.15, gloss=12, soft=.8)


def hand_field(pos, t):
    y = pos[:, 1]
    k = .5 + .5 * np.sin(y * 800)
    return ret(SKIN * np.ones((len(y), 1)), .0005 * k * smooth(.45, .43, y), spec=.15, gloss=12, soft=.9)


PAINTERS = {
    (HEAD, "face"): paint_face,
    (HEAD, "hair"): lambda t: shade(t, hair_field), (HEAD, "scalp"): lambda t: shade(t, scalp_field),
    (SHIRT, "torso"): lambda t: shade(t, torso_field), (SHIRT, "arm"): lambda t: shade(t, arm_field),
    (SHIRT, "hand"): lambda t: shade(t, hand_field),
}


def set_face_uvs(parts):
    """The face island's UVs: the same front projection the painter samples the photo with."""
    head = next(p for p in parts if p.name == "head")
    for i, (q, isl) in enumerate(zip(head.points, head.island)):
        if isl == "face":
            head.uv[i] = (np.clip((q[0] - HC[0]) / (HRX * 1.3) * .5 + .5, 0, 1),
                          np.clip((HC[1] + .075 - q[1]) / .16, 0, 1))


def bake_soles(parts, photo, occlusion=None):
    load_photo(photo)
    ours = [p for p in parts if not getattr(p, "game_uv", False)]
    return bake(ours, ATLAS, PAINTERS, occlusion=occlusion, apply_ao=False)
