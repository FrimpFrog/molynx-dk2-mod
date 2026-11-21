"""The Horned Reaper's skins, built on Quuz's DK1-style reaper textures (his "HD textures" pack for
Flame; credit: Quuz) and Horny's own:

  * HReaper_Face: Quuz's reaper face (white eyes, nose ring, long fangs), as he made it; only
    upscaled to 128 and its red matched to the body;
  * HReaper_Torso, HReaper_Back: Quuz's torso / back as he painted them: his shoulder plates (in
    shiny silver, none of it up the neck) and the chain between them (no 3D chains); the arm's gold
    stud painted out, the skin smoothed and lifted;
  * HReaper_Legs (from Horny's ReaperLegs): the iron greaves become bare red muscle, the hooves a
    dark olive horn like DK1's; Quuz's painted chain belt and the loin flap over the crotch from his
    legs texture;
  * HReaper_Scythe: Horny's scythe with no blood and a plain wooden handle;
  * HReaper_Gear: the shoulder spikes' horn (and steel).
Painting is done at 4x (512) and shrunk to 128 (necro_skin.finish).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

import scripts.reaper_grit as grit
from scripts.necro_skin import _noise, finish

UP = 4
SIZE = 128
N = SIZE * UP

BRONZE = np.array([0.46, 0.42, 0.20])
BRONZE_HI = np.array([0.80, 0.74, 0.44])
BRONZE_DARK = np.array([0.13, 0.11, 0.05])
HOOF = np.array([0.30, 0.28, 0.16])
IVORY = np.array([0.93, 0.88, 0.70])
HORN_ROOT = np.array([0.22, 0.17, 0.10])
IRON = np.array([0.42, 0.40, 0.36])
SILVER = np.array([0.80, 0.81, 0.84])
METAL = np.array([0.66, 0.60, 0.34])          # the sprites' olive bronze
METAL_HI = np.array([0.96, 0.92, 0.70])
IRON_HI = np.array([0.82, 0.80, 0.74])


def _big(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((N, N), Image.BICUBIC), float) / 255


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def _lum(a):
    return a @ np.array([0.30, 0.55, 0.15])


def _sat(a):
    mx, mn = a.max(-1), a.min(-1)
    return (mx - mn) / np.maximum(mx, 1e-6)


def _grid():
    v, u = np.mgrid[0:N, 0:N] / (N - 1)
    return u, v


def _feather(mask: np.ndarray, r: float = 3) -> np.ndarray:
    im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im, float) / 255


def deepen(a: np.ndarray, mask: np.ndarray, amount: float = 0.55) -> np.ndarray:
    """Cut the painted muscles deeper: local contrast on the skin."""
    im = _img(a).filter(ImageFilter.UnsharpMask(radius=10, percent=int(amount * 100), threshold=2))
    b = np.asarray(im, float) / 255
    m = _feather(mask.astype(float), 2)[..., None]
    return a * (1 - m) + b * m


GRIT = True                 # TEST (reaper-grit branch): the weathering pass (reaper_grit) on every skin
SMOOTH = 0.60 if GRIT else 0.72     # Soles: far less texture on the body (0.72); the grit test lets more
                                    # of Quuz's / Horny's own painted muscle back through
BRIGHT = 1.14               # ... and a slightly brighter red


def smooth_skin(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Soften the painted skin (keep the big shapes, lose the fine scratchy detail) and brighten it."""
    soft = np.asarray(_img(a).filter(ImageFilter.GaussianBlur(7)), float) / 255
    mid = np.asarray(_img(a).filter(ImageFilter.GaussianBlur(2)), float) / 255
    b = SMOOTH * (0.6 * soft + 0.4 * mid) + (1 - SMOOTH) * a
    b = b * BRIGHT
    b[..., 0] = np.clip(b[..., 0] * 1.04, 0, 1)
    m = _feather(mask.astype(float), 2)[..., None]
    return np.clip(a * (1 - m) + b * m, 0, 1)


def no_necklace(a: np.ndarray) -> np.ndarray:
    """Paint out Horny's necklace (the chain round the neck and the purple gem): non-skin texels in
    the neck band are filled from the skin around them."""
    u, v = _grid()
    band = (np.abs(u - 0.5) < 0.24) & (v > 0.40) & (v < 0.66) & ~_skin_mask(a) & (_lum(a) > 0.03)
    band &= ~((_sat(a) < 0.40) & (np.abs(u - 0.5) > 0.16))      # not the shoulder plates
    band = _feather(band.astype(float), 2) > 0.15
    skin = a.copy()
    skin[band] = 0
    w = (~band).astype(float)
    num = np.asarray(_img(skin).filter(ImageFilter.GaussianBlur(10)), float) / 255
    den = np.asarray(Image.fromarray((w * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10)), float)[..., None] / 255
    fill = num / np.maximum(den, 1e-3)
    m = _feather(band.astype(float), 2)[..., None]
    return a * (1 - m) + fill * m


def _skin_mask(a):
    """Red skin texels (Horny's red, not black background, metal, bone or gem)."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 0.22) & (r > 1.6 * g) & (r > 1.6 * b)


def _bronze(lum, seed):
    n = _noise(lum.shape, seed, 10) * 0.5 + _noise(lum.shape, seed + 1, 3) * 0.5
    s = np.clip(0.25 + 1.2 * lum, 0, 1.4)[..., None]
    c = BRONZE[None, None] * s * (0.85 + 0.3 * n[..., None])
    hi = np.clip((lum - 0.55) * 2.5, 0, 1)[..., None]
    return c * (1 - hi) + BRONZE_HI[None, None] * hi


def legs(original: Image.Image) -> Image.Image:
    a = _big(original)
    u, v = _grid()
    lum = _lum(a)
    out = a.copy()
    front = u < 0.5
    # the greaves: grey texels on the front of the shins and knees
    greave = front & (v > 0.20) & (v < 0.80) & ~_skin_mask(a) & (lum > 0.02)
    greave = _feather(greave.astype(float), 2.5) > 0.2
    # muscle drawn down from the front thigh (v 0.12 .. 0.33) over the shin (v 0.24 .. 0.80)
    src_v = 0.12 + (np.clip(v, 0.24, 0.80) - 0.24) / 0.56 * 0.21
    rows = np.clip((src_v * (N - 1)).round().astype(int), 0, N - 1)
    cols = np.arange(N)[None, :].repeat(N, 0)
    muscle = a[rows, cols]
    # keep it red where the thigh column had a dark gap (between the legs)
    red = np.array([0.62, 0.12, 0.06])
    bad = ~_skin_mask(muscle)
    fill = red[None, None] * (0.55 + 0.35 * _noise(lum.shape, 13, 4))[..., None]
    muscle[bad] = fill[bad]
    # shading: a kneecap highlight and a darker lower shin
    knee = np.exp(-((v - 0.47) / 0.035) ** 2)[..., None]
    shade = (1.0 - 0.35 * np.clip((v - 0.55) / 0.25, 0, 1))[..., None]
    soft = np.asarray(_img(muscle).filter(ImageFilter.GaussianBlur(2.5)), float) / 255
    muscle = 0.45 * muscle + 0.55 * soft
    muscle = np.clip(muscle * shade * (1 + 0.35 * knee), 0, 1)
    m = _feather(greave.astype(float), 2.5)[..., None]
    out = out * (1 - m) + muscle * m
    # hooves: the plates at the front foot and the hoof band at the back, dark olive horn
    hoof = ((front & (v > 0.76)) | (~front & (v > 0.86))) & (lum > 0.03)
    hcol = HOOF[None, None] * np.clip(0.35 + 1.4 * lum, 0, 1.5)[..., None]
    hcol *= (0.85 + 0.3 * _noise(lum.shape, 5, 6))[..., None]
    rim = np.exp(-((v - 0.965) / 0.012) ** 2)[..., None]
    hcol = hcol * (1 - 0.5 * rim)
    m = _feather(hoof.astype(float), 2)[..., None]
    out = out * (1 - m) + hcol * m
    # the belt and loin flap: painted out to skin (the chain belt is modelled)
    belt = (v < 0.12) & ~_skin_mask(a) & (lum > 0.02) & (np.abs(u - 0.25) < 0.22) | \
           (v < 0.14) & ~_skin_mask(a) & (lum > 0.02) & (np.abs(u - 0.75) < 0.22)
    flap = (((np.abs(u - 0.27) < 0.06) & (v > 0.08) & (v < 0.32)) |
            ((np.abs(u - 0.76) < 0.06) & (v > 0.10) & (v < 0.36))) & ~_skin_mask(a)
    paint = (belt | flap) & ~((u > 0.45) & (u < 0.61) & (v < 0.50))     # never the horn strip
    skin = np.array([0.56, 0.10, 0.05])[None, None] * (0.75 + 0.5 * _noise(lum.shape, 9, 5))[..., None]
    m = _feather(paint.astype(float), 2)[..., None]
    out = out * (1 - m) + skin * m
    out = smooth_skin(out, _skin_mask(out) & (v < 0.80))
    # Quuz's painted chain belt (front and back) and the loin flap under it at the front, from his
    # legs texture, over the painted-out belt
    q = _big(_quuz("ReaperLegs"))
    band = (((v < 0.11) & (u > 0.10) & (u < 0.44)) | ((v > 0.10) & (v < 0.20) & (u > 0.60) & (u < 0.93)) |
            ((v > 0.07) & (v < 0.34) & (np.abs(u - 0.27) < 0.065)))
    qm = band & ~_skin_mask(q) & (_lum(q) > 0.02)
    out = _overlay(out, q, qm)
    if GRIT:
        out = _grit_legs(out, qm, hoof)
    return finish(_img(out))


def _grit_legs(out, qm, hoof):
    """The grit test on the legs: red skin (crevices, mottle, grime, scars, dirt rising from the
    hooves), Quuz's chain belt and loin flap, the hooves; the head's horn strip left as Horny's."""
    u, v = _grid()
    horn_strip = (u > 0.45) & (u < 0.61) & (v < 0.50)
    qm = np.asarray(Image.fromarray((qm * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
                    .filter(ImageFilter.MinFilter(5))) > 0
    hoof = hoof & ~horn_strip
    skin = _skin_mask(out) & ~qm & ~hoof & ~horn_strip
    front = u < 0.5
    # the dirt line: mid-shin at the front (the hooves start at v 0.76), lower on the back of the leg
    # (hooves at 0.86), its edge broken into licks and splashes (reaper_grit.dirt_rise)
    dirt = np.where(front, 0.56, 0.60)
    scars = [(0.33, 0.27, 75, 0.07, 2, 0.018),         # old claw marks across the front of one thigh
             (0.80, 0.35, 100, 0.06, 1, 0.0)]
    # the stretched shin muscle carries a pixel speckle (its fill noise): a light median first, so the
    # grit reads as painted marks, not noise
    from scipy.ndimage import median_filter
    med = np.stack([median_filter(out[..., c], size=7) for c in range(3)], -1)
    sm = grit._soft(skin & front & (v > 0.30), 2)[..., None]
    out = out * (1 - sm) + med * sm
    out = grit.skin(out, skin, 300, scars=scars, dirt=dirt, fine=0.08)     # Horny's legs are busy already
    flap = qm & (np.abs(u - 0.27) < 0.07) & (v > 0.11)
    belt = qm & ~flap
    out = grit.metal_paint(out, belt, 310)
    out = grit.cloth(out, flap, 320, hem=grit._sstep(0.14, 0.34, v))
    v0 = np.where(front, 0.76, 0.86)
    out = grit.hooves(out, hoof, 330, v0)
    hs = horn_strip & (_lum(out) > 0.05)
    out = grit.horn(out, hs, 340, n_cracks=2)
    out = grit.grime(out, hs, 341, amount=0.55, where=1 - grit._sstep(0.03, 0.16, v), colour=grit.DIRT,
                     lo=0.30, hi=0.60, scale=(24, 8, 4))                 # the root, where it leaves the skull
    return out


def _quuz(name: str) -> Image.Image:
    """One of Quuz's "HD textures" (Quuz_HD pack for Flame; credit: Quuz)."""
    from scripts.dk2paths import QUUZ_HD_DIR
    return Image.open(QUUZ_HD_DIR / f"{name}.png").convert("RGB")


def _overlay(out: np.ndarray, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """src over out where mask (closed so the dark gaps between links come too; soft edge)."""
    m = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    m = _feather((np.asarray(m) > 0).astype(float), 1.2)[..., None]
    return out * (1 - m) + src * m


def _inpaint(a: np.ndarray, hole: np.ndarray, radius: float = 14) -> np.ndarray:
    """Fill `hole` from the texels around it (normalised blur), keeping a little of the hole's own
    light and shade so the filled skin is not flat."""
    good = ~hole & _skin_mask(a)                  # fill from skin only (never the black between islands)
    keep = a * good[..., None]
    w = good.astype(float)
    fill = np.zeros_like(a)
    got = np.zeros(a.shape[:2], bool)
    for rad in (radius, radius * 3, radius * 8):          # near skin first, then farther out
        num = np.asarray(_img(keep).filter(ImageFilter.GaussianBlur(rad)), float) / 255
        den = np.asarray(Image.fromarray((w * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rad)), float)[..., None] / 255
        ok = (den[..., 0] > 0.04) & ~got
        fill[ok] = (num / np.maximum(den, 1e-3))[ok]
        got |= ok
    shade = (0.93 + 0.10 * np.clip(_lum(a) / max(_lum(a)[hole].mean() if hole.any() else 1, 1e-3), 0.5, 1.5) - 0.10)[..., None]
    fill = np.clip(fill * shade, 0, 1)
    m = _feather(hole.astype(float), 2)[..., None]
    return a * (1 - m) + fill * m


def upper(original: Image.Image, seed: int) -> Image.Image:
    """Torso or back: Quuz's texture (his "HD textures" pack; credit: Quuz) as he painted it: his
    shoulder plates and the chain between them (Soles, 12 Nov 2026: the pads and chains painted, not
    3D); Horny's gold stud on the arm painted out; the skin smoothed and lifted like the legs'.
    `original` (Horny's own texture) only fills texels Quuz's leaves black."""
    q = _big(_quuz("ReaperTorso" if seed == 1 else "ReaperBack"))
    u, v = _grid()
    stud = (q[..., 1] > 0.45) & (q[..., 0] > 0.55) & (v < 0.30)
    stud = np.asarray(Image.fromarray((stud * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))) > 0
    # his painted shoulder plates in shiny silver (Soles, 12 Nov 2026: "recolour the shoulder pads more
    # shiny silver"): their own light and shade, brightened and turned to polished silver with a
    # glint on the raised parts; and none of it up the neck (those texels become skin)
    r_, g_ = q[..., 0], q[..., 1]
    chain = (v > 0.585) & (v < 0.675) & (u > 0.29) & (u < 0.69)
    plate = (v > 0.33) & (v < 0.70) & (r_ < 1.55 * g_ + 0.04) & (_lum(q) > 0.03) & ~chain
    neck = plate & (np.abs(u - 0.5) < 0.20) & (v < 0.56)
    out = _inpaint(q, (stud | neck) & (_lum(q) > 0.03), radius=10)
    plate &= ~neck
    pl = _lum(q)
    ref = np.percentile(pl[plate], 90) if plate.any() else 1.0
    t = np.clip(pl / max(ref, 1e-3), 0, 1.3)
    shine = SILVER[None, None] * np.clip(0.26 + 0.74 * t ** 1.15 + 0.30 * np.clip(t - 0.88, 0, None) * 4, 0, 1.2)[..., None]
    m = _feather(plate.astype(float), 1.0)[..., None]
    out = out * (1 - m) + np.clip(shine, 0, 1) * m
    out = smooth_skin(out, _skin_mask(out))
    if GRIT:
        out = _grit_upper(out, plate, chain & ~_skin_mask(q) & (_lum(q) > 0.03), seed)
    return finish(_img(out))


# old scars on the arms (the strip along the top) and the chest / back muscles: (u, v, angle, length)
# (u, v, angle, length, count, spacing): a few raised claw-mark scars, roughly parallel
SCARS = {1: [(0.61, 0.73, 35, 0.065, 3, 0.017), (0.24, 0.10, 15, 0.07, 1, 0.0)],
         2: [(0.63, 0.80, 140, 0.075, 2, 0.019)]}


def _grit_upper(out, plate, chain, seed):
    """The grit test on the torso / back: the red skin, Quuz's silver plates and his painted chain."""
    chain = np.asarray(Image.fromarray((chain * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
                       .filter(ImageFilter.MinFilter(5))) > 0
    skin = _skin_mask(out) & ~plate & ~chain
    u, v = _grid()
    if seed == 2:     # the back: calmer, and the neck column above the chain left as it was
        neck = (np.abs(u - 0.5) < 0.11) & (v > 0.22) & (v < 0.60)
        out = grit.skin(out, skin, 100 * seed, scars=SCARS[seed], broad=0.20, fine=0.15, calm=neck)
    else:
        out = grit.skin(out, skin, 100 * seed, scars=SCARS[seed])
    out = grit.drop_shadow(out, plate | chain, skin, dy=10, r=6, dark=0.45)
    out = grit.drop_shadow(out, chain, skin, dy=18, r=9, dark=0.40)      # the chain sags over the chest
    if seed == 1:
        # the strip under the chest chain (v 0.68 .. 0.73) is stretched over the upper chest's big
        # triangles and glowed flat orange in the light from above: shade it as the grimy hollow
        # under the chain
        u, v = _grid()
        band = (np.exp(-((v - 0.705) / 0.022) ** 2) * grit._sstep(0.27, 0.34, u) * grit._sstep(0.27, 0.34, 1 - u))
        band = band * (0.75 + 0.5 * grit.fbm(band.shape, 150, (20, 9, 5)))
        out = np.clip(out * (1 - 0.40 * band * grit._soft(skin, 1.5))[..., None], 0, 1)
    out = grit.plates(out, plate & ~chain, 100 * seed + 50)
    out = grit.metal_paint(out, chain, 100 * seed + 70)
    return out


def gear() -> Image.Image:
    """The gear texture (see reaper_build's UV boxes)."""
    u, v = _grid()
    out = np.zeros((N, N, 3)) + 0.06
    # --- the shoulder caps: a polished silver ramp across the box (u: a facet turned away from the
    # light, dark steel, to one facing it, bright silver with a white glint); each facet of the caps
    # takes one tone off it (reaper_build._facet_shade), so the polygons show like the DK1 render's
    box = (u < 0.50) & (v < 0.50)
    t = np.clip((u - 0.02) / 0.46, 0, 1)
    lum = 0.22 + 0.72 * t ** 1.3 + 0.22 * np.exp(-((t - 0.97) / 0.05) ** 2)
    grain = 0.97 + 0.05 * _noise((N, N), 17, 2)
    silver = SILVER[None, None] * np.clip(lum * grain, 0, 1.3)[..., None]
    out[box] = np.clip(silver, 0, 1)[box]
    # --- the horn spike (u 0.54 .. 0.98: around; v 0.02 (tip) .. 0.48 (root))
    hb = (u > 0.52) & (v < 0.50)
    t = np.clip((v - 0.02) / 0.46, 0, 1)                     # 0 tip, 1 root
    ring = 0.88 + 0.12 * np.cos(t * 2 * np.pi * 6)
    col = IVORY[None, None] * (1 - t[..., None] ** 1.5) + HORN_ROOT[None, None] * t[..., None] ** 1.5
    col = col * ring[..., None] * (0.9 + 0.2 * _noise((N, N), 31, 3))[..., None]
    shade = (0.75 + 0.25 * np.cos((u - 0.76) / 0.22 * np.pi))[..., None]
    out[hb] = (col * shade)[hb]
    # --- silver steel for the chain links and the nose ring (v across the wire: bright on top)
    sb = (u < 0.50) & (v > 0.52)
    vv = np.clip((v - 0.54) / 0.44, 0, 1)
    uu = np.clip((u - 0.02) / 0.46, 0, 1)
    lum = 0.50 + 0.60 * vv ** 1.3 + 0.35 * np.exp(-((vv - 0.82) / 0.07) ** 2) - 0.15 * (uu - 0.5) ** 2
    steel = SILVER[None, None] * np.clip(lum, 0, 1.3)[..., None] * (0.94 + 0.12 * _noise((N, N), 51, 3))[..., None]
    out[sb] = steel[sb]
    # --- the chest medallion (radial: centre 0.76, 0.76): a silver disc with a raised rim, a dark
    # ring and a domed boss, like the round medallion on the DK1 sprites' chest chain
    r = np.hypot(u - 0.76, v - 0.76) / 0.21
    md = r <= 1.05
    lum = 0.70 - 0.20 * r
    lum += 0.35 * (r > 0.80) * (0.6 + 0.4 * np.cos(np.arctan2(v - 0.76, u - 0.76) + 2.3))
    lum *= 1 - 0.65 * np.exp(-((r - 0.62) / 0.06) ** 2)
    boss = np.clip(1 - (r / 0.40) ** 2, 0, 1)
    lum = lum * (1 - boss) + boss * (0.55 + 0.6 * np.exp(-((u - 0.735) ** 2 + (v - 0.735) ** 2) / 0.0012))
    medal = SILVER[None, None] * np.clip(lum, 0.05, 1.3)[..., None]
    out[md] = np.clip(medal, 0, 1)[md]
    if GRIT:                     # the shoulder spikes' horn: deeper rings, cracks, a dirty root
        rings = 1 - 0.14 * np.clip(np.cos(np.clip((v - 0.02) / 0.46, 0, 1) * 2 * np.pi * 6), 0, 1) ** 4
        soft = grit._blur(out, 1.5)          # the base horn's fine noise off first (it read as static)
        out = np.where(hb[..., None], soft * rings[..., None], out)
        out = grit.horn(out, hb, 800)
        out = grit.grime(out, hb, 801, amount=0.55, where=grit._sstep(0.30, 0.48, v), colour=grit.DIRT,
                         lo=0.35, hi=0.65, scale=(24, 8, 4))
    return finish(_img(out))


FACE_TEXTURE = "ReaperFaceSmirk"   # Quuz's DK1-style reaper face (his "HD textures" pack, read from QUUZ_HD_DIR)


def all_skins(cache, ref) -> dict:
    """Every skin; ref is reaper_build.reference (the chains are painted through the reshaped body)."""
    from scripts.reaper_build import SOURCE_TEX
    return {"HReaper_Legs": legs(cache.get(SOURCE_TEX["reaperlegs"])),
            "HReaper_Torso": upper(cache.get(SOURCE_TEX["reapertorso"]), 1),
            "HReaper_Back": upper(cache.get(SOURCE_TEX["reaperback"]), 2),
            "HReaper_Gear": gear(),
            "HReaper_Scythe": scythe(cache.get(SOURCE_TEX["reaperschythe"])),
            "HReaper_Face": face()}


def face() -> Image.Image:
    """Quuz's reaper face, as he made it (credit: Quuz, the "HD textures" pack for Flame): it is
    already in Horny's face layout (ReaperFaceSmirk, 64 x 64), so it goes onto Horny's own head
    unchanged in shape. It is only brought to 128 x 128 (smooth upscale and a light sharpen) and its
    red lifted the same small amount as the body's skin, so face and body match."""
    src = _quuz(FACE_TEXTURE)
    big = src.resize((128, 128), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    a = np.asarray(big, float) / 255
    r, g = a[..., 0], a[..., 1]
    # how red a texel is (0 on the eyes, teeth, horn bone and nose ring; 1 on the skin), soft-edged
    w = np.clip((r / np.maximum(g, 1e-3) - 1.6) / 1.0, 0, 1) * np.clip((r - 0.12) / 0.15, 0, 1)
    lift = 1 + w[..., None] * (np.array(FACE_LIFT) - 1)
    a = np.clip(a * lift, 0, 1)
    if GRIT:                     # Quuz's face as he made it: only its crevices a little deeper, a faint mottle
        skin = w > 0.5
        a = grit.cavity(a, skin, radius=2.0, deep=0.30, ridge=0.08, brown=0.2)
        a = grit.mottle(a, skin, 900, amount=0.07, scales=(8, 4, 2), weights=(0.4, 0.4, 0.2))
        v, u = np.mgrid[0:128, 0:128] / 127
        jaw = grit._sstep(0.70, 0.92, v) * 0.28 * grit._soft(skin, 1)          # the jaw's underside in shadow
        a = a * (1 - jaw[..., None])
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


FACE_LIFT = (1.18, 0.96, 0.85)      # the face's red brought to the body skin's (mean red 140 -> ~165)


def scythe(original: Image.Image) -> Image.Image:
    """Horny's scythe (ReaperSchythe, 128 x 64) clean: no blood on the blade (plain steel), and a
    plain wooden handle (the gold fittings and purple gems become more of the shaft's wood)."""
    w0, h0 = original.size
    a = np.asarray(original.convert("RGB").resize((w0 * UP, h0 * UP), Image.BICUBIC), float) / 255
    H, W = a.shape[:2]
    v, u = np.mgrid[0:H, 0:W] / np.array([H - 1, W - 1])[:, None, None]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = _lum(a)
    # --- the blade (v < 0.48): fill the blood from the clean steel around it, then plain steel
    blade = (v < 0.48) & (lum > 0.04)
    blood = blade & (r > 1.35 * g) & (r > 0.18)
    clean = blade & ~blood
    num = np.asarray(_img(a * clean[..., None]).filter(ImageFilter.GaussianBlur(12)), float) / 255
    den = np.asarray(Image.fromarray((clean * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(12)), float)[..., None] / 255
    fill = num / np.maximum(den, 1e-3)
    out = a.copy()
    out[blood] = fill[blood]
    gl = _lum(out)
    # where the blood was, the steel's own brightness (its blur pulled in the dark under the blade and
    # left a dark patch, Soles 12 Nov): the clean steel's typical light, varied a little
    steel_lum = np.median(gl[clean & (gl > 0.25)])
    gl = np.where(blood, steel_lum * (0.92 + 0.16 * _noise((H, W), 72, 5)), gl)
    edge = np.exp(-((v - 0.06) / 0.05) ** 2)                  # the bright cutting edge along the top
    steel = np.array([0.70, 0.72, 0.76])[None, None] * np.clip(0.35 + 0.85 * gl + 0.35 * edge, 0, 1.25)[..., None]
    steel *= (0.94 + 0.12 * _noise((H, W), 71, 6))[..., None]
    out[blade] = np.clip(steel, 0, 1)[blade]
    # --- the handle strips: gold and gems -> the shaft's own wood from the same row
    gold = (v > 0.48) & (((r > 0.45) & (g > 0.30) & (b < 0.6 * g)) | ((b > r * 0.9) & (b > 0.25)))
    for row in np.flatnonzero(gold.any(1)):
        cols = np.flatnonzero(gold[row])
        wood_cols = np.flatnonzero(~gold[row] & (lum[row] > 0.08) & (r[row] > g[row]))
        if not len(wood_cols):
            continue
        src = wood_cols[np.arange(len(cols)) % len(wood_cols)]
        out[row, cols] = a[row, src]
    if GRIT:
        out = _grit_scythe(out, blade, (v > 0.48) & (_lum(out) > 0.06), u, v)
    small = _img(out).resize((w0, h0), Image.LANCZOS)
    return small.filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2))


def _grit_scythe(out, blade, wood, u, v):
    """The grit test on the scythe: darker, brushed steel with tarnish toward the spine and the
    heel, nicks in the cutting edge (the bright line along the top); the handle's grain and grime."""
    H, W = out.shape[:2]
    rng = np.random.default_rng(700)
    # brushed streaks along the blade (long in u, thin in v)
    cells = rng.random((H // 16 + 2, W // 90 + 2))
    streak = np.asarray(Image.fromarray((cells * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255
    st = grit._blur(out, 2.0) * (0.88 + 0.16 * streak)[..., None] * 0.88     # base speckle off first
    out = np.where(blade[..., None], st, out)
    out = grit.cavity(out, blade, radius=8, deep=0.4, ridge=0.10, brown=0.15)
    # tarnish toward the spine (v ~0.40) and the heel by the handle (u > 0.8)
    where = np.clip((v - 0.22) / 0.2, 0, 1) * 0.8 + np.clip((u - 0.75) / 0.2, 0, 1) * 0.6
    out = grit.grime(out, blade, 701, amount=0.60, where=np.clip(where, 0, 1), colour=grit.TARNISH,
                     lo=0.40, hi=0.70, scale=(40, 14, 6))
    out = grit.mottle(out, blade, 702, amount=0.06, scales=(40, 20, 12))
    # the honed cutting edge stays bright (a band under the blade's top line), the spine darker
    ys, xs = np.nonzero(blade)
    top = np.full(W, H)
    np.minimum.at(top, xs, ys)
    depth = (np.arange(H)[:, None] - top[None, :]).astype(float)
    hone = np.exp(-np.clip(depth - 2, 0, None) / 7.0) * blade
    out = np.clip(out * (1 + 0.30 * hone[..., None]) + 0.06 * hone[..., None], 0, 1)
    out = out * (1 - 0.40 * (grit._sstep(0.26, 0.46, v) * blade))[..., None]
    # a faint rust bloom at the heel by the handle (no blood: Soles wanted the blade clean of it)
    rust = grit._sstep(0.72, 0.95, u) * grit._sstep(0.10, 0.35, v) * blade
    rust = rust * grit._sstep(0.38, 0.62, grit.fbm((H, W), 705, (30, 12, 6)))
    rc = np.array([0.45, 0.24, 0.11])[None, None] * (0.6 + _lum(out))[..., None]
    out = out * (1 - 0.75 * rust[..., None]) + rc * 0.75 * rust[..., None]
    # three dull nicks in the edge: a shallow dark bite with a lit lower rim
    for x in (0.27, 0.55, 0.81):
        x = x * W + rng.uniform(-10, 10)
        t = top[int(np.clip(x, 0, W - 1))]
        if t >= H:
            continue
        d = rng.uniform(7, 11)
        pts = np.array([[x - d * 0.9, t - 2], [x - d * 0.2, t + d * 0.7], [x + d * 0.7, t - 2]])
        out = grit.cut(out, blade, pts, width=6, dark=0.72, lit=1.6, lit_off=(0, 4))
    out = grit.scratches(out, blade & (v > 0.12), 703, n=3, length=(80, 140), width=4.0, dark=0.6, lit=1.3,
                         ang=-0.35)                         # a few long diagonal scratches
    # the wood: deeper grain along the shaft, grime toward the ends, a worn lighter grip
    grain_cells = rng.random((H // 3 + 2, W // 40 + 2))
    grain = np.asarray(Image.fromarray((grain_cells * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), float) / 255
    wd = out * (0.78 + 0.40 * grain)[..., None]
    out = np.where(wood[..., None], wd, out)
    out = grit.contrast(out, wood, k=1.2, sat=1.0)
    ends = np.clip(1 - np.minimum(u, 1 - u) / 0.12, 0, 1)
    out = grit.grime(out, wood, 704, amount=0.6, where=ends, lo=0.35, hi=0.65, scale=(24, 8, 4))
    return out
