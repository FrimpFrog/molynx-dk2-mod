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
from scripts.soles_model import ARM, BODY, BOOT_TOP, HAIR_TEX, HAIRLINE, HC, HEAD, LIMBS, split_front_back

ATLAS = {
    HEAD: {"front": ((0, 0, 128, 104), False), "back": ((0, 106, 128, 128), False)},
    HAIR_TEX: {"hair": ((0, 0, 128, 40), False), "lock": ((0, 42, 128, 104), False),
               "lock_in": ((0, 106, 128, 128), False)},
    BODY: {"front": ((0, 0, 64, 128), False), "back": ((64, 0, 128, 128), False)},
    LIMBS: {"arm": ((0, 0, 30, 96), False), "hand": ((0, 98, 30, 128), False),
            "leg": ((32, 0, 62, 128), False), "foot": ((64, 0, 92, 28), False),
            "cloth": ((64, 30, 96, 74), False), "cloth_in": ((64, 76, 96, 96), False),
            "belt": ((98, 0, 128, 18), False), "bracer": ((98, 20, 128, 50), False),
            "pauldron": ((98, 52, 128, 84), False), "spike": ((98, 86, 128, 100), False),
            "gorget": ((98, 102, 128, 116), False)},
}

SKIN = RGB(222, 178, 160)
RED = RGB(190, 32, 38)
HAIR = RGB(50, 33, 24)
LEATHER_BLACK = RGB(58, 50, 56)
LEATHER = RGB(118, 72, 38)
STEEL = RGB(118, 122, 136)
BRONZE = RGB(196, 148, 66)

HEAD_BOX = (-.08, .826, .08, 1.004)            # x0, y0, x1, y1 of the back of the head's projection
FACE_BOX = (-.058, .835, .058, .95)            # the face's projection: twice the pixels on his features
BODY_BOX = (-.125, .425, .125, .875)


def planar(box, mirror=False):
    x0, y0, x1, y1 = box

    def uv(p):
        u = np.clip((p[0] - x0) / (x1 - x0), 0, 1)
        return (1 - u if mirror else u, float(np.clip((y1 - p[1]) / (y1 - y0), 0, 1)))
    return uv


def project_uvs(parts):
    """Front and back projections on the head and the torso (the game's way)."""
    for p in parts:
        if p.name == "head":
            split_front_back(p, "front", "back", planar(FACE_BOX), planar(HEAD_BOX, True), z_split=HC[2])
        elif p.name == "torso":
            split_front_back(p, "front", "back", planar(BODY_BOX), planar(BODY_BOX, True), z_split=0.0)


# ------------------------------------------------------------------------------------------- head

_HAIR_MAP = None


def _hair_map():
    """Hair painted like DK2's: 12 wide locks round the head, each dark at its edges and bright along a
    ridge, falling in a slow wave from the parting, with a faint strand grain."""
    global _HAIR_MAP
    if _HAIR_MAP is not None:
        return _HAIR_MAP
    W, H = 720, 360
    rng = np.random.default_rng(21)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n_locks = 12
    width = W / n_locks
    sway = 14 * np.sin(yy / H * 3.0 + xx / W * 2 * np.pi * 2)            # the slow wave
    jitter = np.interp(xx[0], np.linspace(0, W, n_locks + 1), rng.uniform(-6, 6, n_locks + 1))[None, :]
    pos_in = np.mod(xx + sway + jitter, width) / width                     # 0..1 across each lock
    profile = np.sin(np.pi * pos_in) ** 1.4                                # dark crevice, bright ridge
    tone = .12 + .83 * profile
    grain = np.asarray(Image.fromarray((rng.random((H, W // 3)) * 255).astype(np.uint8))
                       .resize((W, H)).filter(ImageFilter.GaussianBlur((4, .6))), np.float32) / 255
    tone = tone * (.88 + .24 * grain)
    tone *= np.clip(yy / 30, .55, 1)                                       # the parting: no pole starburst
    _HAIR_MAP = np.clip(tone, 0, 1)
    return _HAIR_MAP


def hair_col(pos):
    """His hair: the painted strand map wrapped round the head (angle, height), a soft sheen across the
    crown, darker ends and deep shadow under the fringe."""
    x, y, z = pos[:, 0] - HC[0], pos[:, 1], pos[:, 2] - HC[2]
    ang = np.arctan2(x, -z)
    m = _hair_map()
    H, W = m.shape
    u = ((ang + np.pi) / (2 * np.pi) * W).astype(int) % W
    v = np.clip(((1.012 - y) / .2 * H).astype(int), 0, H - 1)
    tone = m[v, u]
    col = mix(HAIR * .35, HAIR * 1.45, tone)
    sheen = gauss(y - (.968 - .025 * (1 - np.cos(ang)) / 2), .016)
    col = mix(col, RGB(150, 104, 70), .35 * sheen * tone)
    col = mix(col, HAIR * .45, .3 * smooth(.87, .82, y))
    h = .002 * tone
    return col, h


def hair_field(pos, t):
    """The cap: strands combed out from the parting (on his left), the same browns as the locks."""
    x, y, z = pos[:, 0] - HC[0] - .02, pos[:, 1], pos[:, 2] - HC[2]
    ang = np.arctan2(x, -z)
    strands = .5 + .5 * np.sin(ang * 26 + fbm(ang * 4, y * 30, 7) * 3)
    col = mix(HAIR * .45, HAIR * 1.2, .3 + .5 * strands)
    col = mix(col, RGB(150, 104, 70), .3 * gauss(y - .975, .012) * strands)
    col = mix(col, HAIR * .3, .6 * gauss(x, .003) * smooth(.95, .99, y) * smooth(-.02, .02, -z + .04))   # the parting
    return ret(col, .002 * strands, spec=.3, gloss=14, soft=.5)


def lock_field(pos, t):
    """A hanging lock, painted like a DK2 hair card: dark at its edges, a bright ridge down the middle,
    strands running along it, a sheen near the top, darker toward the tip."""
    u, v = t["u"], t["v"]
    seed = t.get("lock", np.zeros(len(u)))
    ridge = np.sin(np.pi * np.clip(u, 0, 1)) ** 1.2
    strands = .5 + .5 * np.sin(u * 40 + seed * 3 + np.sin(v * 6 + seed) * 2)
    col = mix(HAIR * .35, HAIR * 1.25, .2 + .6 * ridge + .2 * strands)
    col = mix(col, RGB(150, 104, 70), .4 * ridge * gauss(v - .2, .15))
    col = mix(col, HAIR * .4, .4 * smooth(.6, 1.0, v) * (1 - ridge))
    return ret(col, .003 * ridge + .0006 * strands, spec=.3, gloss=14, soft=.5)


def lock_in_field(pos, t):
    m = lock_field(pos, t)
    m["albedo"] = m["albedo"] * .35
    return m


def hair_in_field(pos, t):
    col, h = hair_col(pos)
    return ret(col * .3, h, spec=.02, gloss=8, soft=.4)


# His likeness, painted (no photo pixels): proportions and features measured from his photo, then drawn
# the way DK2's painted faces are (the Knight's, Loobinex's): x is from the middle of his face, y up.
EYE_Y, EYE_X = .909, .0265
BROW_Y = .9215
NOSE_Y = .882
MOUSTACHE_Y = .8705
MOUTH_Y = .8635
CHIN_Y = .842
EYES = RGB(120, 140, 150)          # light blue-grey
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
    by = BROW_Y + .002 * smooth(.02, .008, ax)                  # straight, the inner ends a little raised
    brow = band(y - by, -.0034, .0034, .0008) * band(ax, .006, .046, .004)
    brow *= .9 + .1 * (.5 + .5 * np.sin(x * 2200 + y * 900))
    col = mix(col, BROW, np.clip(brow, 0, 1))
    h += .0005 * brow
    # painted light and shade, as DK2's faces have baked in: a lit forehead, bridge, cheekbones and chin;
    # shadow in the eye sockets, down the sides of the nose, under the nose, the cheekbones and the lip
    lit = (.5 * gauss(np.hypot(x, (y - .935) * .6), .02) + .5 * gauss(x, .004) * band(y, .885, .92, .004)
           + .4 * gauss(np.hypot(ax - .03, y - .893), .007) + .35 * gauss(np.hypot(x, y - .85), .006))
    dark = (.55 * gauss(np.hypot(ax - EYE_X, (y - EYE_Y - .002) * 1.2), .011)
            + .45 * gauss(ax - .009, .0035) * band(y, .878, .91, .004)
            + .6 * gauss(np.hypot(x * .7, y - (NOSE_Y - .008)), .005)
            + .35 * gauss(np.hypot(ax - .032, y - .874), .01)
            + .5 * gauss(np.hypot(x * .6, y - (MOUTH_Y - .008)), .004)
            + .4 * smooth(.85, .838, y) * smooth(.02, .045, ax))
    hl_ = hairline(x)
    dark = dark + .5 * band(y, hl_ - .012, hl_, .004) + .4 * smooth(.042, .05, ax) * smooth(.935, .925, y)
    col = col * (1 + .39 * lit - .75 * np.clip(dark, 0, 1))[:, None]
    # stubble: a faint shadow round the jaw, the chin and the upper lip
    stub = smooth(.875, .862, y) * smooth(.012, .03, ax) * smooth(.83, .838, y)
    stub = np.clip(stub, 0, 1) * (.6 + .4 * fbm(x * 900, y * 900, 33))
    col = mix(col, RGB(96, 82, 80), .06 * stub)
    # eyes: narrow, a heavy shadowed upper lid, little white showing, a grey-green iris under the lid
    ex, ey = (ax - EYE_X) / .011, (y - EYE_Y) / .006
    shape = ex ** 2 + (ey + .15 * ex) ** 2
    eye = shape < 1
    white = RGB(210, 196, 186) * (.8 + .2 * smooth(1, .2, ex ** 2))[:, None]
    col = np.where(eye[:, None], white, col)
    iris = np.hypot((ax - EYE_X + .0005) / .0045, (y - EYE_Y + .0006) / .0045)
    col = np.where((eye & (iris < 1))[:, None], mix(EYES * .55, EYES * 1.1, smooth(1, .35, iris)), col)
    col = np.where((eye & (iris < .4))[:, None], RGB(14, 12, 14), col)
    col = mix(col, RGB(60, 40, 36), (eye * smooth(EYE_Y + .0015, EYE_Y + .005, y)) * .2)   # lid's shadow
    hl = np.hypot(ax - EYE_X + .0018, y - EYE_Y - .0012) < .0009
    col = np.where((eye & hl)[:, None], RGB(240, 240, 240), col)
    upper = (np.abs(shape - 1) < .7) & (y > EYE_Y - .0008)
    col = np.where(upper[:, None], RGB(46, 30, 26), col)
    lid = gauss(np.hypot(ax - EYE_X, (y - EYE_Y - .0068) * 2.4) - .0115, .0018) * (y > EYE_Y)
    col = mix(col, SKIN * .55, .55 * lid)
    lower = (np.abs(shape - 1) < .45) & (y < EYE_Y - .0008)
    col = mix(col, RGB(140, 86, 78), .5 * lower)
    # the nose: nostrils and a highlight on the tip
    nos = gauss(np.hypot((ax - .0068) / 1.6, y - (NOSE_Y - .0058)), .0021)
    col = mix(col, RGB(62, 34, 30), .8 * nos)
    col = mix(col, SKIN * 1.2, .35 * gauss(np.hypot(x, y - NOSE_Y), .003))
    # his moustache: a thin dark line of hairs over the lip, drooping a little at the ends
    my = MOUSTACHE_Y - (MOUSTACHE_Y - MOUTH_Y - .001) * smooth(.008, .02, ax)    # ends drop to the mouth
    thick = .0032
    mo = band(y - my, -thick, thick, .0009) * smooth(.022, .017, ax)
    mo *= .85 + .15 * (.5 + .5 * np.sin(x * 3200 + y * 1500))
    col = mix(col, RGB(58, 40, 30), np.clip(mo, 0, 1))
    h += .0004 * mo
    # lips: a thin, darker upper lip, a fuller lower one with a soft light, shadowed corners
    up = ((x / .0145) ** 2 + ((y - MOUTH_Y - .0014) / .0018) ** 2) < 1
    lo = ((x / .0135) ** 2 + ((y - MOUTH_Y + .0022) / .0028) ** 2) < 1
    lips = up | lo
    col = np.where(up[:, None], RGB(150, 86, 80) * np.ones((n, 1)), col)
    col = np.where(lo[:, None], mix(RGB(184, 112, 104), RGB(214, 150, 140),
                                    gauss(np.hypot(x * .5, y - MOUTH_Y + .0018), .0016)), col)
    col = mix(col, RGB(80, 40, 38), gauss(y - MOUTH_Y, .0006) * band(ax, 0, .015, .002))
    col = mix(col, RGB(110, 60, 54), .5 * gauss(np.hypot(ax - .015, y - MOUTH_Y), .002))
    # his hair over everything above the hairline and round the sides above the ears
    hl_y = hairline(x)
    hair = smooth(hl_y - .002, hl_y + .003, y)
    hcol, hh = hair_col(pos)
    col = mix(col, hcol, hair)
    h = h * (1 - hair) + hh * hair
    spec = .04 * (1 - hair) + .3 * hair + .45 * eye + .12 * lips
    m = ret(col, h, spec=spec, gloss=14 + 30 * eye, soft=.85 * (1 - hair) * (1 - eye))
    m["glow"] = np.where(eye[:, None], 4.0, 0.0) * np.ones((1, 3))
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

    shapes = [flame(120, 250, 56, 215, .6), flame(66, 258, 32, 120, -1.3), flame(174, 258, 32, 120, 1.3)]
    fill = Image.new("L", (W, H), 0)
    for s in shapes:
        ImageDraw.Draw(fill).polygon(s, fill=255)
    outline = fill.filter(ImageFilter.MaxFilter(11))
    inner = Image.new("L", (W, H), 0)
    ImageDraw.Draw(inner).polygon(flame(122, 246, 22, 105, .4), fill=255)
    f = lambda im: np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    return f(fill), f(outline), f(inner)


EMBLEM = _emblem()


def _crest():
    """The Molynx crest: his face from his portrait in a round medallion (the user's own creature's art)."""
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "data" / "molynx_design" / "portrait_icon_A2.png"
    im = Image.open(src).convert("RGB").crop((60, 60, 420, 420)).resize((128, 128), Image.LANCZOS)
    return np.asarray(im, np.float32)


CREST = _crest()
CREST_C, CREST_R = (0.0, .69), .026            # on the breastplate, model units


def crest_at(pos):
    """-> (colour, inside the medallion 0..1, on its gold rim 0..1)."""
    dx, dy = pos[:, 0] - CREST_C[0], pos[:, 1] - CREST_C[1]
    r = np.hypot(dx, dy) / CREST_R
    H, W = CREST.shape[:2]
    u = np.clip(((dx / CREST_R) * .5 + .5) * W, 0, W - 1).astype(int)
    v = np.clip(((-dy / CREST_R) * .5 + .5) * H, 0, H - 1).astype(int)
    return CREST[v, u], smooth(.86, .8, r), band(r, .8, 1.0, .03)
EMBLEM_BOX = (-.022, .62, .022, .73)


def emblem_at(pos):
    x0, y0, x1, y1 = EMBLEM_BOX
    fill, outline, inner = EMBLEM
    H, W = fill.shape
    u = ((pos[:, 0] - x0) / (x1 - x0) * W).astype(int)
    v = ((y1 - pos[:, 1]) / (y1 - y0) * H).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    return fill[v, u] * ok, outline[v, u] * ok, inner[v, u] * ok


IRON = RGB(90, 90, 108)            # dark grey plate
SILVER = RGB(196, 200, 212)
WHITE = RGB(232, 228, 220)
BLOOD = RGB(150, 20, 26)


GOLD = RGB(214, 168, 72)


def scroll(x, y, f=1.0):
    """Engraved scrollwork: interlaced curls (0..1, 1 = a groove)."""
    a = np.sin(x * 420 * f + np.sin(y * 300 * f) * 2.2)
    b = np.sin(y * 420 * f + np.sin(x * 300 * f) * 2.2)
    return smooth(.82, .97, np.abs(a * b))


def torso_field(pos, t):
    """A dark grey breastplate rimmed in silver with gold engraved borders, a white and red heraldic
    panel down the centre, the Molynx crest in a gold medallion with a ruby above it, three faulds with
    gold edges; the backplate with engraved shoulder plates and rivets; a red arming coat at the sides."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    n = len(x)
    front = smooth(-.01, .01, z)
    grain = fbm(x * 120, y * 120, 41)
    folds = fbm(x * 30, y * 9, 44)
    col = mix(BLOOD * .55, BLOOD * .95, folds) * np.ones((n, 1))
    h = .003 * folds
    spec = np.full(n, .1)
    gloss = np.full(n, 10.0)
    metal = np.zeros(n)

    def put(mask, albedo, height, sp, gl, me):
        nonlocal col, h, spec, gloss, metal
        mk = np.clip(mask, 0, 1)
        col = mix(col, albedo * np.ones((n, 1)), mk)
        h, spec, gloss, metal = (h * (1 - mk) + height * mk, spec * (1 - mk) + sp * mk,
                                 gloss * (1 - mk) + gl * mk, metal * (1 - mk) + me * mk)

    plate_w = .092 + .025 * smooth(.6, .7, y)
    breast = smooth(plate_w + .004, plate_w - .004, ax) * band(y, .585, .79, .004) * front
    curve = gauss(np.hypot(ax - .04, (y - .69) * .8), .032)
    put(breast, IRON * (.8 + .25 * grain + .4 * curve)[:, None], .006 + .004 * curve, .45, 30, .55)
    # a gold engraved border inside the silver rim
    inset = breast * band(plate_w - ax, .006, .016, .002)
    eng = scroll(x, y)
    put(inset, mix(GOLD, GOLD * .45, eng), .0072 - .0008 * eng, .7, 34, .7)
    rim = breast * smooth(plate_w - .006, plate_w - .002, ax)
    put(rim, SILVER, .0078, .8, 40, .7)
    # the heraldic panel: white with a red chevron and gold edging
    panel = band(x, -.026, .026, .0015) * band(y, .6, .77, .002) * breast
    chev = band(y - (.64 + .9 * ax), -.008, .008, .0015) * panel
    put(panel, WHITE * (.88 + .12 * grain[:, None]), .0068, .3, 20, .0)
    col = mix(col, BLOOD * 1.15, chev)
    col = mix(col, GOLD, (gauss(np.abs(x) - .026, .0015) + gauss(y - .77, .0015) + gauss(y - .6, .0015))
              * band(x, -.028, .028, .001) * band(y, .598, .772, .001) * front)
    # the Molynx crest in a gold medallion, a ruby above it
    crest, inside, ring = crest_at(pos)
    col = mix(col, crest * 1.05, inside * front)
    put(ring * front, mix(GOLD * 1.1, GOLD * .5, scroll(x, y, 2)), .0082, .9, 40, .8)
    h += .001 * inside * front
    ruby = gauss(np.hypot(x, y - .735), .0045) * front
    put(smooth(.3, .6, ruby), RGB(200, 20, 40), .0085 + .002 * ruby, 1.4, 70, .1)
    # rivets along the rim
    riv = gauss(np.hypot(ax - (plate_w - .004), np.mod(y + .006, .022) - .011), .0022) * breast
    col = mix(col, GOLD * 1.15, smooth(.3, .7, riv))
    # under the breastplate's lower edge
    col = mix(col, RGB(28, 22, 26), gauss(y - .582, .003) * smooth(plate_w + .004, plate_w - .004, ax) * front)
    # faulds: three lames with gold edges and engraving
    fauld = smooth(.096, .088, ax) * band(y, .508, .585, .003)
    lame = np.mod(.585 - y, .026) / .026
    put(fauld, IRON * (.7 + .35 * (1 - lame)[:, None]) * (.9 + .2 * grain[:, None]), .005 + .003 * (1 - lame),
        .4, 28, .5)
    col = mix(col, GOLD, fauld * gauss(lame - .07, .05))
    col = mix(col, GOLD * .5, fauld * .5 * scroll(x * 1.5, y) * band(lame, .2, .7, .05))
    col = mix(col, RGB(28, 22, 26), fauld * gauss(lame - .97, .03))
    # the backplate
    back = (1 - front) * band(y, .56, .79, .004) * smooth(.1, .09, ax)
    blade = gauss(np.hypot(ax - .045, (y - .72) * .8), .03)
    put(back, IRON * (.75 + .25 * grain + .3 * blade + .3 * gauss(x, .005))[:, None], .006 + .003 * blade,
        .4, 28, .5)
    bl = gauss(np.hypot(ax - .045, (y - .72) * .8) - .028, .0022) * back
    col = mix(col, GOLD, bl)
    col = mix(col, GOLD * .55, .6 * back * scroll(x, y) * smooth(.028, .02, np.hypot(ax - .045, (y - .72) * .8)))
    rivets = gauss(np.hypot(np.mod(x + .01, .02) - .01, y - .575), .0025) * back
    col = mix(col, SILVER * 1.1, smooth(.3, .7, rivets))
    col = col * (1 - .35 * smooth(.1, .125, ax) * smooth(.7, .76, y))[:, None]
    neck = smooth(.8, .81, y)
    col = mix(col, SKIN * np.ones((n, 1)), neck)
    return ret(col, h, spec=spec, gloss=gloss, metal=metal, soft=.5 * neck)


# ------------------------------------------------------------------------------------------- limbs

def _along(pos, path):
    ys = np.array([c[1] for c, *_ in path])
    return np.interp(-pos[:, 1], -ys, np.linspace(0, 1, len(ys)))


def arm_field(pos, t):
    """Bare arms: painted muscle (deltoid, biceps, the elbow, the forearm), a gold arm ring."""
    f = _along(pos, ARM)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    u = t["u"]
    col = SKIN * (.95 + .08 * fbm(x * 200, y * 200, 3))[:, None]
    delt = gauss(f - .12, .08)
    bic = gauss(f - .3, .1) * gauss(u - .15, .25)
    elbow = gauss(f - .5, .04)
    fore = gauss(f - .62, .1) * gauss(u - .3, .3)
    h = .002 * delt + .002 * bic + .0015 * fore - .0008 * elbow
    col = col * (1 + .12 * (delt + bic + fore) - .2 * elbow - .25 * smooth(.6, 1, u) * smooth(.15, .4, f))[:, None]
    ring = band(f, .38, .42, .006)
    col = mix(col, mix(GOLD, GOLD * .5, scroll(x, y, 3)), ring)
    h += .002 * ring
    return ret(col, h, spec=.08 + .7 * ring, gloss=12 + 30 * ring, metal=.7 * ring, soft=.8 * (1 - ring))


def hand_field(pos, t):
    """Black leather gauntlet fingers, iron plates over the knuckles."""
    y = pos[:, 1]
    knuck = band(y, .438, .458, .003)
    fingers = .5 + .5 * np.cos(t["u"] * 4 * 2 * np.pi)
    col = mix(LEATHER_BLACK * 1.3, LEATHER_BLACK * .7, .3 * fingers)
    col = mix(col, IRON * 1.2, knuck)
    col = mix(col, SILVER, knuck * gauss(y - .458, .0025))
    return ret(col * np.ones((len(y), 1)), .0008 * fingers + .002 * knuck, spec=.2 + .4 * knuck, gloss=24,
               metal=.5 * knuck)


def leg_field(pos, t):
    """Plate legs: a cuisse over the thigh and a greave round the shin, dark grey with silver rims, over
    dark leather; the backs of the knees and thighs are leather."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    ax = np.abs(x)
    u = t["u"]
    grain = fbm(x * 120, y * 120, 13)
    col = mix(LEATHER_BLACK * 1.3, LEATHER_BLACK * .8, grain) * np.ones((len(x), 1))
    h = .001 * grain
    front = smooth(.75, .55, u)                                           # u: 0 front .. 1 back
    cuisse = band(y, .29, .46, .004) * front
    greave = band(y, .06, .225, .004)
    plate = np.clip(cuisse + greave, 0, 1)
    ridge = gauss(u, .08)
    pc = IRON * (.8 + .25 * grain + .35 * ridge)[:, None]
    col = mix(col, pc, plate)
    h = h * (1 - plate) + (.004 + .002 * ridge) * plate
    rim = (gauss(y - .29, .003) + gauss(y - .46, .003)) * front + gauss(y - .225, .003) + gauss(y - .06, .003)
    col = mix(col, GOLD, np.clip(rim, 0, 1) * .9)
    col = mix(col, GOLD * .55, .55 * plate * scroll(x * 1.5, y, 1.5) * gauss(u, .22))
    h += .0015 * np.clip(rim, 0, 1)
    # painted shadow under the loincloth and the knee cops
    col = col * (1 - .35 * smooth(.43, .47, y) - .25 * gauss(y - .25, .012))[:, None]
    me = plate * .5 + np.clip(rim, 0, 1) * .3
    return ret(col, h, spec=.12 + .35 * plate + .4 * np.clip(rim, 0, 1), gloss=14 + 16 * plate, metal=me)


def foot_field(pos, t):
    """Sabatons: dark grey plate in four broad lames over dark leather, a black sole."""
    y, z = pos[:, 1], pos[:, 2]
    sole = smooth(.014, .006, y)
    grain = fbm(pos[:, 0] * 160, z * 160, 17)
    lames = np.mod(z * 45, 1)
    col = mix(IRON * (.75 + .25 * grain[:, None]), IRON * .5, gauss(lames - .95, .06))
    col = mix(col, GOLD, gauss(lames - .05, .05) * .7)
    col = mix(col, LEATHER_BLACK * .6, sole)
    return ret(col, .0012 * sole + .0015 * (1 - lames) * (1 - sole), spec=.4 * (1 - sole), gloss=26,
               metal=.5 * (1 - sole))


def cloth_field(pos, t):
    x, y = pos[:, 0], pos[:, 1]
    face = t.get("face", np.ones(len(x)))
    u, v = t["u"], t["v"]
    folds = fbm(x * 40, y * 6, 61)
    border = np.maximum(smooth(.12, .06, np.minimum(u, 1 - u)), smooth(.86, .93, v))
    trim = gauss(np.minimum(u, 1 - u) - .09, .02) + gauss(v - .895, .02)
    col = mix(BLOOD * .7, BLOOD * 1.15, folds)
    col = mix(col, WHITE * .92, border)
    col = mix(col, GOLD, np.clip(trim, 0, 1) * .9)
    emb = scroll(x * 1.2, y * 1.2, 1.5) * smooth(.6, .75, v) * smooth(.86, .82, v)
    col = mix(col, GOLD * .8, .7 * emb)
    h = .004 * folds + .001 * np.clip(trim, 0, 1)
    inside = BLOOD * .45 * (.7 + .3 * folds[:, None])
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
    col = mix(col, GOLD * 1.05, me)
    return ret(col, .001 * grain + .004 * me - .002 * edge, spec=.25 + .9 * me, gloss=16 + 30 * me, metal=.7 * me)


def bracer_field(pos, t):
    """Plate vambraces: dark grey, silver rims, a ridge along the top."""
    v, u = t["v"], t["u"]
    grain = fbm(pos[:, 0] * 160, pos[:, 1] * 160, 81)
    rim = np.clip(gauss(v, .12) + gauss(v - 1, .12), 0, 1)
    ridge = gauss(u, .1)
    col = mix(IRON * (.8 + .25 * grain + .35 * ridge)[:, None], GOLD, rim)
    col = mix(col, GOLD * .55, .5 * scroll(pos[:, 0] * 2, pos[:, 1], 1.5) * (1 - rim) * gauss(u, .3))
    return ret(col, .001 * grain + .002 * rim + .002 * ridge, spec=.4 + .4 * rim, gloss=30, metal=.55 + .2 * rim)


def spike_field(pos, t):
    along = t.get("along", np.zeros(len(pos)))
    col = mix(IRON * .7, RGB(190, 190, 200), smooth(.5, 1.0, along))
    return ret(col, np.zeros(len(pos)), spec=1.2, gloss=50, metal=.8)


def gorget_field(pos, t):
    v = t["v"]
    grain = fbm(pos[:, 0] * 260, pos[:, 1] * 260, 83)
    col = mix(IRON * (.75 + .4 * grain)[:, None], BLOOD * 1.1, gauss(v - 1, .15))
    return ret(col, .0008 * grain, spec=.35, gloss=28, metal=.6)


def pauldron_field(pos, t):
    along = t.get("along", np.zeros(len(pos)))
    rim = smooth(.18, .04, along)
    rivet = gauss(along - .3, .06) * (np.cos(np.arctan2(pos[:, 2] + .012, pos[:, 1] - .78) * 5) > .9)
    h = .002 * rim + .002 * rivet + .0006 * fbm(pos[:, 0] * 300, pos[:, 2] * 300, 91)
    col = mix(IRON * 1.05, SILVER, rim)
    col = mix(col, GOLD * .7, .5 * scroll(pos[:, 0], pos[:, 2] * 1.3 + pos[:, 1], 1.4) * (1 - rim))
    col = mix(col, GOLD * 1.1, rivet)
    col = col * (.85 + .25 * fbm(pos[:, 0] * 120, pos[:, 1] * 120, 93))[:, None]
    return ret(col, h, spec=.45, gloss=30, metal=.55)


PAINTERS = {
    (HEAD, "front"): head_front, (HEAD, "back"): lambda t: shade(t, hair_field),
    (BODY, "front"): lambda t: shade(t, torso_field), (BODY, "back"): lambda t: shade(t, torso_field),
    (LIMBS, "arm"): lambda t: shade(t, arm_field), (LIMBS, "hand"): lambda t: shade(t, hand_field),
    (LIMBS, "leg"): lambda t: shade(t, leg_field), (LIMBS, "foot"): lambda t: shade(t, foot_field),
    (LIMBS, "cloth"): lambda t: shade(t, cloth_field), (LIMBS, "cloth_in"): lambda t: shade(t, cloth_field),
    (LIMBS, "belt"): lambda t: shade(t, belt_field), (LIMBS, "bracer"): lambda t: shade(t, bracer_field),
    (LIMBS, "pauldron"): lambda t: shade(t, pauldron_field),
    (LIMBS, "spike"): lambda t: shade(t, spike_field), (LIMBS, "gorget"): lambda t: shade(t, gorget_field),
    (HAIR_TEX, "hair"): lambda t: shade(t, hair_field), (HAIR_TEX, "lock"): lambda t: shade(t, lock_field),
    (HAIR_TEX, "lock_in"): lambda t: shade(t, lock_in_field),
}


def bake_soles(parts, occlusion=None):
    ours = [p for p in parts if not getattr(p, "game_uv", False)]
    return bake(ours, ATLAS, PAINTERS, occlusion=occlusion, apply_ao=True)
