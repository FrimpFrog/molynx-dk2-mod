"""The Priestess's four 128 x 128 skins, painted the way Dungeon Keeper 2's own skins were made: as renders
of a detailed model. Every texel is traced back to the surface it covers and SHADED there: a material
(albedo, a height field for the detail the low-poly mesh lacks, specular strength and gloss, how much it
reflects like metal) is lit by a key light from above the camera, a fill from below, ambient occlusion and,
for metal, a chrome-like environment (bright sky, dark horizon band, warm floor), as the Mistress's and the
Fairy's skins are.

The height fields carry what the mesh cannot: collarbones, the line of the stomach and the navel, the
shoulder blades; the armoured cups, their raised rims, engraved scrollwork and rivets; the studded bronze
belt and its jewelled buckle; the glove's wrinkles and cuff; the boot's leather creases, turned-down
cuff, seams and sole; hair strands; the cape's folds and stitched hem; the face's brow, eyelids, nose,
cheeks and lips.

UVs are mirrored left / right (the lofts' u), so all lighting is symmetric in x.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_texture import fbm, mix, smooth, value_noise
from scripts.lowpoly import bake
from scripts.priestess_model import ARM_PATH, BODY, CAPE, HC, HEAD, HRX, HRY, LEG_PATH, LIMBS

ATLAS = {
    BODY: {"torso": ((0, 0, 128, 128), False)},
    LIMBS: {"arm": ((0, 0, 40, 110), False), "leg": ((42, 0, 96, 110), False),
            "hand": ((98, 0, 128, 40), False), "foot": ((98, 42, 128, 110), False)},
    HEAD: {"face": ((0, 0, 72, 72), False), "hair": ((74, 0, 128, 76), False),
           "hair_in": ((74, 78, 128, 100), False), "scalp": ((74, 102, 128, 128), False),
           "pauldron": ((0, 74, 34, 100), False), "staff": ((36, 74, 72, 128), False),
           "orb": ((0, 102, 34, 128), False)},
    CAPE: {"cape": ((0, 0, 128, 96), False), "cape_in": ((0, 98, 128, 128), False)},
}

RGB = lambda *v: np.array(v, float)
SKIN = RGB(232, 176, 128)
SKIN_SHADE = RGB(150, 82, 56)
LAV = RGB(168, 146, 214)
LAV_DARK = RGB(70, 52, 108)
BRONZE = RGB(206, 156, 70)
SILVER = RGB(200, 202, 214)
LEATHER = RGB(150, 92, 44)
LEATHER_DARK = RGB(70, 38, 16)
BLONDE = RGB(236, 192, 84)
CLOTH = RGB(214, 204, 172)
CLOTH_IN = RGB(96, 70, 48)

KEY = np.array([0, .62, .78]) / np.linalg.norm([0, .62, .78])
FILL = np.array([0, -.5, .86]) / np.linalg.norm([0, -.5, .86])


def gauss(d, w):
    return np.exp(-(np.asarray(d) / w) ** 2)


def band(x, lo, hi, soft=.002):
    return smooth(lo - soft, lo + soft, x) * smooth(hi + soft, hi - soft, x)


# ------------------------------------------------------------------------------------------- shading

def env(n):
    """Chrome environment seen in direction n: lit sky above, a dark band at the horizon, warm floor."""
    y = n[:, 1]
    sky = mix(RGB(150, 150, 170), RGB(255, 255, 255), smooth(.1, .8, y))
    floor = mix(RGB(40, 30, 26), RGB(120, 92, 70), smooth(-.1, -.8, y))
    out = np.where((y > 0)[:, None], sky, floor)
    return mix(out, RGB(16, 14, 20), gauss(y - .02, .12))


def shade(t, field, eps=.0012, bump=1.0):
    """Light the material `field(pos, t) -> dict(albedo, height, spec, gloss, metal, glow)` at the
    texels `t` (lowpoly.bake samples)."""
    pos, n0 = t["pos"], t["nrm"]
    m = field(pos, t)
    # tangent frame and the height field's gradient
    a = np.where(np.abs(n0[:, 1:2]) < .9, np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
    t1 = np.cross(n0, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n0, t1)
    dh1 = (field(pos + t1 * eps, t)["height"] - field(pos - t1 * eps, t)["height"]) / (2 * eps)
    dh2 = (field(pos + t2 * eps, t)["height"] - field(pos - t2 * eps, t)["height"]) / (2 * eps)
    n = n0 - bump * (dh1[:, None] * t1 + dh2[:, None] * t2)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[:, 0] = np.abs(n[:, 0]) * np.sign(n0[:, 0] + 1e-9)        # keep the mirror symmetric
    alb = m["albedo"]
    # the light comes from the side being looked at (DK2 skins are a front render and a back render)
    side = np.where(n0[:, 2:3] >= -.15, 1.0, -1.0)
    key = KEY * np.concatenate([np.ones_like(side), np.ones_like(side), side], 1)
    fill = FILL * np.concatenate([np.ones_like(side), np.ones_like(side), side], 1)
    ndl = np.clip(np.sum(n * key, 1), 0, 1)
    wrap = np.clip((np.sum(n * key, 1) + .35) / 1.35, 0, 1)        # soft skin-like terminator
    diff = .3 + .62 * (wrap * m.get("soft", 0) + ndl * (1 - m.get("soft", 0))) + .16 * np.clip(np.sum(n * fill, 1), 0, 1)
    ao = t.get("ao", np.ones(len(pos)))
    ao = .55 + .45 * ao
    col = alb * (diff * ao)[:, None]
    metal = m.get("metal", 0.0)
    if np.any(metal):
        view = np.concatenate([np.zeros_like(side), np.zeros_like(side), side], 1)
        refl = env(2 * np.sum(n * view, 1)[:, None] * n - view)
        col = mix(col, refl * alb / 255 * 1.15 * ao[:, None], metal)
    h = key + n0
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    spec = m.get("spec", 0.0) * np.clip(np.sum(n * h, 1), 0, 1) ** m.get("gloss", 20.0)
    col = col + (spec * ao)[:, None] * 255
    rim = m.get("rim", 0.0)
    if np.any(rim):
        col = col + (rim * (1 - np.abs(n[:, 2])) ** 3)[:, None] * RGB(255, 230, 255)
    if "glow" in m:
        col = col + m["glow"]
    return np.clip(col, 0, 255)


def ret(albedo, height, **kw):
    return dict(albedo=albedo, height=height, **kw)


# ------------------------------------------------------------------------------------------- torso

def torso_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    front = smooth(-.01, .01, z)
    # skin and its anatomy
    h = np.zeros(len(x))
    clav = .776 + .1 * (ax - .02) ** 1.0 * .2
    h += .0012 * gauss(y - clav, .004) * band(ax, .014, .085, .006) * front
    h -= .0016 * gauss(ax, .006) * gauss(y - .782, .006) * front                  # the hollow of the throat
    h -= .0007 * gauss(ax, .005) * band(y, .52, .62, .01) * front                 # the line of the stomach
    h -= .0005 * gauss(ax - .036, .007) * band(y, .53, .6, .015) * front
    h -= .0025 * gauss(np.hypot(x, (y - .556) * 1.4), .0045) * front              # the navel
    h += .0012 * gauss(np.hypot(ax - .055, y - .512), .012) * front              # hip bones
    back = 1 - front
    h -= .0012 * gauss(ax, .006) * band(y, .5, .78, .01) * back                  # the spine
    h += .0022 * gauss(np.hypot(ax - .045, (y - .72) * .8), .018) * back          # shoulder blades
    h -= .0008 * gauss(y - .52, .006) * band(ax, .0, .05, .01) * back             # the small of the back
    skin_col = mix(SKIN, SKIN * RGB(1.02, .93, .9) / 1, .3 * fbm(x * 40, y * 40, 3))
    col = skin_col * np.ones((len(x), 1))
    # the shadow of the cleavage and under the bust
    col = mix(col, SKIN_SHADE, (.55 * gauss(ax, .006) * band(y, .655, .72, .01)
                                + .35 * gauss(y - .638, .008) * band(ax, .005, .07, .01)) * front)
    spec = np.full(len(x), .18)
    gloss = np.full(len(x), 14.0)
    metal = np.zeros(len(x))
    soft = np.ones(len(x))

    def put(mask, albedo, height, sp, gl, me, so=0.0):
        nonlocal col, h, spec, gloss, metal, soft
        mk = np.clip(mask, 0, 1)
        col = mix(col, albedo * np.ones((len(x), 1)), mk)
        h = h * (1 - mk) + height * mk
        spec = spec * (1 - mk) + sp * mk
        gloss = gloss * (1 - mk) + gl * mk
        metal = metal * (1 - mk) + me * mk
        soft = soft * (1 - mk) + so * mk

    # -- the armoured top: two cups joined at the breastbone, a band round the back, shoulder straps
    CX, CY, CR = .034, .676, .03
    r = np.hypot((ax - CX) / CR, (y - CY) / (CR * 1.05))
    cups = smooth(1.03, .97, r) * front
    dome = np.clip(1 - r * r, 0, 1) * front
    bridge = band(ax, 0, .01, .002) * band(y, .652, .668, .002) * front
    backband = band(y, .662, .69, .003) * (1 - front)
    strap = band(ax, .044, .056, .002) * band(y, .7, .79, .004)
    plate = np.clip(cups + bridge + backband + strap, 0, 1)
    rim_d = np.abs(r - .93)
    scroll = .5 + .5 * np.sin(np.arctan2(y - CY, ax - CX) * 4 + r * 9)
    inner = smooth(.75, .55, r) * cups
    ph = .004 + .009 * dome + .0012 * gauss(rim_d, .06) * front - .0008 * smooth(.7, .95, scroll) * inner
    acol = mix(LAV, LAV_DARK, .8 * gauss(rim_d, .055) * front)
    acol = mix(acol, LAV * 1.15, .5 * inner)
    acol = mix(acol, LAV_DARK, .5 * smooth(.7, .95, scroll) * inner)
    put(plate, acol, ph, .7, 26, .18)
    # a jewel where the cups meet
    gem = gauss(np.hypot(x, (y - .66) * 1.2), .0045) * front
    put(smooth(.3, .6, gem), RGB(150, 80, 230), .0075 + .004 * gem, 1.2, 60, .2)
    # -- the belt: bronze, studded, a jewelled buckle at the front
    belt = band(y, .49, .514, .0015)
    stud = gauss(np.hypot(np.mod(np.arctan2(x, z) * .06, .012) - .006, y - .502), .0022)
    put(belt, mix(BRONZE, BRONZE * .6, gauss(y - .49, .002) + gauss(y - .514, .002)),
        .004 + .0016 * stud, .9, 30, .65)
    buckle = band(ax, 0, .016, .002) * band(y, .487, .518, .002) * front
    put(buckle, BRONZE * 1.1, .0062 - .0018 * band(ax, 0, .009, .002) * band(y, .494, .511, .002),
        1.0, 40, .7)
    bgem = gauss(np.hypot(x, y - .5025), .0042) * front
    put(smooth(.3, .6, bgem), RGB(150, 80, 230), .0075 + .003 * bgem, 1.2, 60, .2)
    # -- the briefs: high on the hips, a V at the front, trimmed with a raised rim
    cut = .488 - .9 * np.clip(ax - .03, 0, None) * front - .5 * np.clip(ax - .04, 0, None) * (1 - front)
    cut = np.maximum(cut, .43)
    briefs = smooth(cut + .003, cut - .003, y) * smooth(.4, .412, y + .1)
    trim = gauss(y - cut, .0022) * briefs
    put(briefs, mix(LAV, LAV_DARK, .5 * trim + .25 * smooth(.47, .43, y)), .003 + .001 * trim, .8, 34, .5)
    out = ret(col, h, spec=spec, gloss=gloss, metal=metal, soft=soft * .8)
    return out


# ------------------------------------------------------------------------------------------- limbs

def _along(pos, path):
    """Height along a limb path (0 at its first point, 1 at its last), mirrored in x."""
    path = np.asarray(path, float)
    y = pos[:, 1]
    ys = path[:, 1]
    return np.interp(-y, -ys, np.linspace(0, 1, len(ys)))


def arm_field(pos, t):
    y = pos[:, 1]
    f = _along(pos, ARM_PATH)
    glove = smooth(.42, .46, f)
    wr = .5 + .5 * np.sin(y * 900 + fbm(pos[:, 0] * 60, y * 60, 9) * 4)
    creases = (gauss(f - .5, .06) + gauss(f - .96, .05)) * wr
    h = glove * (.002 - .0006 * creases)
    cuff = band(f, .44, .5, .006)
    h += .0016 * cuff + .0008 * gauss(f - .47, .012)
    gcol = mix(LAV, LAV_DARK, .4 * creases + .2 * smooth(.7, 1.0, f))
    col = mix(SKIN * np.ones((len(y), 1)), gcol, glove)
    col = mix(col, BRONZE, cuff)
    h += .0008 * gauss(f - .2, .08) * (1 - glove)                    # the shoulder's roundness
    return ret(col, h, spec=.2 + .55 * glove + .4 * cuff, gloss=14 + 20 * glove, metal=.2 * glove + .6 * cuff,
               soft=1 - glove)


def hand_field(pos, t):
    y = pos[:, 1]
    k = (y * 900) % (2 * np.pi)
    h = .0006 * np.sin(k) * smooth(.43, .41, y)                       # knuckles
    return ret(mix(LAV, LAV_DARK, .3) * np.ones((len(y), 1)), h, spec=.5, gloss=24, metal=.15)


def leg_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    f = _along(pos, LEG_PATH)
    boot = smooth(.395, .385, y)
    grain = fbm(x * 220, y * 220, 13)
    creases = (gauss(y - .095, .03) + gauss(y - .25, .02)) * (.5 + .5 * np.sin(y * 1300 + grain * 5))
    cuff = band(y, .352, .392, .002)
    h = boot * (.003 - .0007 * creases + .00025 * grain) + .0035 * cuff + .002 * gauss(y - .352, .003) * boot
    seam = gauss(np.abs(x) - np.interp(-y, -np.array(LEG_PATH)[:, 1], np.array(LEG_PATH)[:, 0]), .0025) \
        * smooth(.0, .01, z * 0 + 1)
    h -= .0006 * seam * boot
    lcol = mix(LEATHER, LEATHER_DARK, .45 * creases + .35 * seam + .25 * grain)
    lcol = mix(lcol, LEATHER * 1.12, cuff)
    lcol = mix(lcol, LEATHER_DARK * .7, gauss(y - .352, .0025))
    knee = gauss(np.hypot(y - .25, (z - .03) * 2), .02)
    h += .0014 * knee * boot
    col = mix(SKIN * np.ones((len(y), 1)), lcol, boot)
    return ret(col, h, spec=.3 + .25 * cuff, gloss=16, metal=0.0, soft=(1 - boot) * .8)


def foot_field(pos, t):
    y = pos[:, 1]
    sole = smooth(.012, .004, y)
    col = mix(LEATHER, LEATHER_DARK * .5, sole)
    h = .001 * sole + .0005 * fbm(pos[:, 0] * 200, pos[:, 2] * 200, 17)
    return ret(col * np.ones((len(y), 1)), h, spec=.35, gloss=18)


# ------------------------------------------------------------------------------------------- head

EYE_Y = .906
EYE_X = .0172


def face_field(pos, t):
    x, y, z = pos[:, 0] - HC[0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    col = SKIN * np.ones((len(x), 1))
    h = np.zeros(len(x))
    # bone structure
    h += .0016 * gauss(y - .918, .004) * band(ax, .006, .03, .006)                # brow ridge
    h -= .0016 * gauss(np.hypot(ax - EYE_X, (y - EYE_Y) * 1.3), .009)              # eye sockets
    h += .0012 * gauss(np.hypot(ax - .022, y - .892), .009)                        # cheekbones
    h += .002 * gauss(ax, .004) * band(y, .885, .912, .004)                        # the bridge of the nose
    h += .0018 * gauss(np.hypot(x, y - .868), .007)                                # the lips
    h -= .0012 * gauss(y - .868, .0012) * band(ax, 0, .009, .002)                  # between the lips
    h += .0012 * gauss(np.hypot(x, y - .852), .007)                                # the chin
    # make-up and colour
    col = mix(col, RGB(236, 150, 128), .3 * gauss(np.hypot(ax - .022, y - .885), .009))      # blush
    col = mix(col, RGB(150, 96, 170), .55 * gauss(np.hypot(ax - EYE_X, (y - EYE_Y - .006) * 1.6), .009)
              * smooth(EYE_Y - .002, EYE_Y + .004, y))                                         # eyeshadow
    brow = gauss(y - (.9175 + .03 * (ax - .017) * 0 - 18 * (ax - .017) ** 2), .0012) * band(ax, .007, .028, .003)
    col = mix(col, RGB(150, 100, 40), .9 * brow)
    # the eyes: almond, a dark upper lid with a wing, white, a blue iris with a pupil and a highlight
    ex, ey = (ax - EYE_X) / .0098, (y - EYE_Y) / .0048
    shape = ex ** 2 + (ey - .15 * ex) ** 2
    eye = shape < 1
    col = np.where(eye[:, None], RGB(240, 236, 230), col)
    iris = np.hypot((ax - EYE_X + .0008) / .0042, (y - EYE_Y) / .0044)
    col = np.where((eye & (iris < 1))[:, None], mix(RGB(40, 90, 190), RGB(110, 170, 240), smooth(1, .2, iris)), col)
    col = np.where((eye & (iris < .45))[:, None], RGB(10, 10, 20), col)
    hl = np.hypot(ax - EYE_X + .0022, y - EYE_Y - .0018) < .0011
    col = np.where((eye & hl)[:, None], RGB(255, 255, 255), col)
    lid = (np.abs(shape - 1) < .5) & (y > EYE_Y - .0005) | (np.hypot((ax - EYE_X - .011) / .004, (y - EYE_Y - .002 - (ax - EYE_X - .007) * .4) / .0012) < 1)
    col = np.where(lid[:, None], RGB(30, 18, 30), col)
    lower = (np.abs(shape - 1) < .35) & (y < EYE_Y - .001)
    col = mix(col, RGB(120, 70, 70), .5 * lower)
    # the nose: nostrils and the shade under it
    col = mix(col, SKIN_SHADE, .5 * gauss(np.hypot(ax - .005, y - .881), .0022))
    # her violet lips, a darker line between them and a highlight on the lower
    lips = ((x / .0125) ** 2 + ((y - .868) / .0052) ** 2) < 1
    lcol = mix(RGB(150, 100, 190), RGB(108, 70, 150), smooth(.868, .875, y))
    lcol = mix(lcol, RGB(60, 30, 70), gauss(y - .8685, .0009))
    lcol = mix(lcol, RGB(210, 180, 240), .5 * gauss(np.hypot(x * .5, y - .864), .002))
    col = np.where(lips[:, None], lcol, col)
    eyes_glow = np.where(eye[:, None], 1.0, 0.0)
    m = ret(col, h, spec=.22 + .5 * eye + .4 * lips, gloss=16 + 30 * eye, soft=.9 * (1 - eye))
    m["glow"] = eyes_glow * 25
    return m


def scalp_field(pos, t):
    return ret(BLONDE * .5 * np.ones((len(pos), 1)), np.zeros(len(pos)), spec=0.0)


def hair_field(pos, t):
    x, y, z = pos[:, 0] - HC[0], pos[:, 1], pos[:, 2] - HC[2]
    ang = np.arctan2(x, -z)
    inner = t.get("inner", np.zeros(len(x)))
    aa = np.abs(ang)
    clump = fbm(aa * 5, y * 1.5, 31)
    locks = .5 + .5 * np.sin(aa * 16 + clump * 4)                    # big locks falling straight
    strand = .5 + .5 * np.sin(aa * 90 + clump * 10)
    h = .003 * locks + .0005 * strand
    col = mix(BLONDE * .58, BLONDE * 1.08, .25 + .45 * locks + .15 * strand + .2 * clump)
    col = mix(col, BLONDE * .45, .45 * smooth(.3, .0, locks))
    col = mix(col, BLONDE * .5, .55 * smooth(.85, .79, y))                     # darker ends
    col = mix(col, BLONDE * .45, .6 * inner)
    return ret(col, h, spec=.25 * (1 - inner), gloss=10, rim=0.0, soft=.3)


def pauldron_field(pos, t):
    along = t.get("along", np.zeros(len(pos)))
    rim = smooth(.25, .08, along)
    rivet = gauss(along - .55, .06) * (.5 + .5 * np.cos(np.arctan2(pos[:, 2], pos[:, 1] - .78) * 6)) ** 8
    h = .002 * rim + .0015 * rivet + .0005 * np.sin(along * 30)
    col = mix(SILVER, SILVER * .5, .35 * rim)
    return ret(col * np.ones((len(pos), 1)), h, spec=1.2, gloss=50, metal=.8)


def staff_field(pos, t):
    y = pos[:, 1]
    along = t.get("along", np.zeros(len(y)))
    grain = fbm(t["u"] * 6, y * 120, 61)
    wood = mix(RGB(88, 54, 26), RGB(150, 100, 52), .3 + .7 * grain)
    h = .0004 * grain
    top = .432 + .33
    bands = band(y, top - .02, top, .002) + band(y, .45, .462, .002) + band(y, .4, .412, .002)
    wrap = band(y, .412, .45, .002) * (.5 + .5 * np.sin(y * 1400 + t["u"] * 9))
    col = mix(wood, RGB(108, 70, 120) * (0.7 + .3 * wrap[:, None]), band(y, .412, .45, .002))
    col = mix(col, BRONZE * 1.05, np.clip(bands, 0, 1))
    h += .0015 * np.clip(bands, 0, 1) + .0006 * wrap
    gold = along > .01
    col = np.where(gold[:, None], BRONZE * (.8 + .4 * along[:, None]), col)
    return ret(col, h, spec=.3 + .8 * np.clip(bands + gold, 0, 1), gloss=30,
               metal=.6 * np.clip(bands + gold, 0, 1))


ORB_C = None


def orb_field(pos, t):
    from scripts.priestess_model import build_sceptre  # noqa: F401  (centre from the part's own points)
    c = pos.mean(0) if ORB_C is None else ORB_C
    d = pos - c
    r = np.linalg.norm(d, axis=1)
    n = d / np.maximum(r[:, None], 1e-9)
    swirl = .5 + .5 * np.sin(np.arctan2(n[:, 0], n[:, 2]) * 3 + n[:, 1] * 6)
    col = mix(RGB(110, 70, 170), RGB(206, 170, 255), .4 + .5 * swirl)
    m = ret(col, np.zeros(len(r)), spec=1.5, gloss=60, metal=.15)
    m["glow"] = mix(RGB(40, 20, 70), RGB(110, 70, 150), swirl) * .9
    return m


def cape_field(pos, t):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    face = t.get("face", np.ones(len(x)))
    weave = .5 + .5 * np.sin(x * 2600) * np.sin(y * 2600)
    folds = fbm(x * 30, y * 4, 71)
    h = .004 * folds + .00015 * weave
    hem = smooth(.25, .22, y)
    edge = smooth(.012, .0, .2 - np.abs(x) - (.2 - .1) * (y - .2) / (.772 - .2) * 0 + 0) * 0
    trim = band(y, .222, .236, .0015)
    col = mix(CLOTH * .82, CLOTH * 1.06, .3 + .7 * folds)
    col = mix(col, RGB(140, 112, 180), trim)                                   # a lavender hem stripe
    col = mix(col, CLOTH * .6, hem * .35)
    h += .0008 * trim
    inside = mix(CLOTH_IN * .75, CLOTH_IN * 1.05, folds)
    col = np.where((face < 0)[:, None], inside, col)
    return ret(col, h, spec=.08, gloss=8, soft=.6)


FIELDS = {
    (BODY, "torso"): torso_field,
    (LIMBS, "arm"): arm_field, (LIMBS, "hand"): hand_field, (LIMBS, "leg"): leg_field,
    (LIMBS, "foot"): foot_field,
    (HEAD, "face"): face_field, (HEAD, "scalp"): scalp_field, (HEAD, "hair"): hair_field,
    (HEAD, "hair_in"): hair_field, (HEAD, "pauldron"): pauldron_field, (HEAD, "staff"): staff_field,
    (HEAD, "orb"): orb_field,
    (CAPE, "cape"): cape_field, (CAPE, "cape_in"): cape_field,
}
PAINTERS = {k: (lambda t, f=f: shade(t, f)) for k, f in FIELDS.items()}


def bake_priestess(parts, occlusion=None):
    global ORB_C
    sc = next(p for p in parts if p.name == "sceptre")
    orb = [q for q, isl in zip(sc.points, sc.island) if isl == "orb"]
    ORB_C = np.mean(orb, 0)
    return bake(parts, ATLAS, PAINTERS, occlusion=occlusion, apply_ao=False)
