"""Grit pass for the Classic Reaper's skins (TEST, community feedback Nov 2026: "classic horny could
be a bit more gritty as well").

Hand-painted weathering in DK2's style (light from above, dark crevices, a few strong painted
marks rather than photographic noise), laid over the approved skins (reaper_skin) at the 512
painting resolution, before they are shrunk to 128:

  * red skin: crevices cut deeper (darker, a touch browner), a broken painterly mottle, grime
    patches, a few old scars (a dark cut with a lit lower lip), dirt rising up from the hooves;
  * Quuz's silver shoulder plates: darker, more contrasty steel, tarnish in the low parts, worn
    bright edges, chips and dents (dark upper wall, lit lower wall) and scratches; still silver;
  * Quuz's painted chains, belt and loin flap: more contrast, dark gaps, grime; the flap dirtier
    toward its hem;
  * hooves: growth ridges, cracks, chipped and dirty rims;
  * scythe: darker steel with brushed streaks, nicks in the cutting edge, tarnish near the spine;
    the handle's wood grain and grime at the ends;
  * the shoulder spikes' horn: deeper rings, cracks and a dirty root;
  * face: Quuz's face kept as he made it: only its crevices deepened and a faint mottle.

Every random mark comes from fixed seeds, so the build is reproducible.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scripts.necro_skin import _noise


# ------------------------------------------------------------------------------ helpers

def _blur(x: np.ndarray, r: float) -> np.ndarray:
    """Gaussian blur of a float array (2-D or RGB) in 0..1 (float precision, not 8-bit)."""
    from scipy.ndimage import gaussian_filter
    sig = (r, r) if x.ndim == 2 else (r, r, 0)
    return gaussian_filter(np.asarray(x, float), sig, mode="nearest")


def _lum(a):
    return a @ np.array([0.30, 0.55, 0.15])


def _sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def fbm(shape, seed, scales=(64, 24, 9), weights=(0.5, 0.3, 0.2)):
    n = sum(w * _noise(shape, seed + i, s) for i, (s, w) in enumerate(zip(scales, weights)))
    return n / sum(weights)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    im = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.MinFilter(2 * px + 1)), np.uint8) > 0


def _soft(mask, r=1.5):
    return _blur(mask.astype(float), r)


def _stroke(shape, pts, width) -> np.ndarray:
    """A painted line (0..1) along pts (x, y in pixels)."""
    h, w = shape
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).line([tuple(p) for p in pts], fill=255, width=int(round(width)), joint="curve")
    return np.asarray(im, float) / 255


def _path(rng, x, y, ang, length, step=6.0, wobble=0.18):
    pts = [(x, y)]
    for _ in range(max(2, int(length / step))):
        ang += rng.normal(0, wobble)
        x, y = x + step * np.cos(ang), y + step * np.sin(ang)
        pts.append((x, y))
    return np.array(pts)


def cut(a, mask, pts, width=6.0, dark=0.45, lit=1.30, lit_off=(0, 3), tint=None):
    """A painted scar / scratch: a dark groove and, just below it (light from above), its lit lip.
    Only inside mask."""
    sh = a.shape[:2]
    core = _blur(_stroke(sh, pts, width), 1.0)
    lip = _blur(_stroke(sh, pts + np.array(lit_off, float), max(2.0, width * 0.55)), 1.0)
    lip = np.clip(lip - core, 0, 1)
    m = _soft(mask, 1.0)
    core, lip = core * m, lip * m
    out = a * (1 - core[..., None] * (1 - dark))
    hi = np.clip(a * lit + 0.05, 0, 1)
    if tint is not None:
        hi = hi * 0.6 + np.asarray(tint)[None, None] * 0.4
    return out * (1 - lip[..., None]) + hi * lip[..., None]


def cavity(a, mask, radius=6.0, deep=0.55, ridge=0.18, brown=0.25, tint=(0.80, 0.70, 0.62)):
    """Deepen the painted crevices: texels darker than their surroundings get darker (and a little
    browner), the raised ones a touch lighter: DK2's painted light-from-above look."""
    L = _lum(a)
    d = L - _blur(L, radius)
    dn = np.clip(-d / 0.10, 0, 1)
    up = np.clip(d / 0.10, 0, 1)
    f = 1 - deep * dn + ridge * up
    m = _soft(mask, 1.5)[..., None]
    b = a * f[..., None]
    brn = np.array(tint)[None, None]
    b = b * (1 - brown * dn[..., None]) + b * brn * (brown * dn[..., None])
    return np.clip(a * (1 - m) + b * m, 0, 1)


def mottle(a, mask, seed, amount=0.10, scales=(22, 9, 5), weights=(0.35, 0.40, 0.25)):
    """Broken painterly mottle (blotches of a few texels at 128), multiplicative."""
    n = fbm(a.shape[:2], seed, scales, weights)
    n = (n - n.mean()) / max(n.std(), 1e-6)
    f = 1 + amount * np.clip(n, -2, 2) / 2
    m = _soft(mask, 1.5)[..., None]
    return np.clip(a * (1 - m) + a * f[..., None] * m, 0, 1)


GRIME = np.array([0.20, 0.11, 0.06])
DIRT = np.array([0.24, 0.16, 0.09])


def grime(a, mask, seed, amount=0.35, where=None, colour=GRIME, lo=0.52, hi=0.80, scale=(48, 18, 7), k=0.45):
    """Blotchy grime patches toward `colour` (where: an extra 0..1 weight field)."""
    n = fbm(a.shape[:2], seed, scale, (0.5, 0.3, 0.2))
    blot = _sstep(lo, hi, n)
    if where is not None:
        blot = blot * where
    w = amount * blot * _soft(mask, 1.5)
    dirty = a * (1 - k) + colour[None, None] * k * (0.7 + 1.2 * _lum(a))[..., None]
    return np.clip(a * (1 - w[..., None]) + dirty * w[..., None], 0, 1)


def chips(a, mask, seed, n=12, r=(4, 8), dark=0.35, lit=1.45, base_col=None):
    """Dents and chips: a dark spot (its upper wall in shadow) and a lit crescent below it."""
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(_erode(mask, 4))
    if not len(xs):
        return a
    h, w = mask.shape
    sh = np.zeros((h, w))
    li = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for i in rng.choice(len(xs), size=min(n, len(xs)), replace=False):
        cx, cy = xs[i], ys[i]
        rr = rng.uniform(*r)
        sx, sy = rng.uniform(0.7, 1.4), rng.uniform(0.6, 1.0)
        x0, x1, y0, y1 = int(max(cx - 3 * rr, 0)), int(min(cx + 3 * rr, w)), int(max(cy - 3 * rr, 0)), int(min(cy + 3 * rr, h))
        X, Y = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        # an irregular outline (a chip, not a drilled hole)
        th = np.arctan2(Y - cy, X - cx)
        k1, k2, p1, p2 = rng.integers(2, 4), rng.integers(4, 7), rng.uniform(0, 6.3), rng.uniform(0, 6.3)
        wob = 1 + 0.35 * np.sin(k1 * th + p1) + 0.18 * np.sin(k2 * th + p2)
        d0 = (((X - cx) / (rr * sx)) ** 2 + ((Y - cy) / (rr * sy)) ** 2) / wob ** 2
        d1 = (((X - cx) / (rr * sx)) ** 2 + ((Y - cy - 0.55 * rr) / (rr * sy)) ** 2) / wob ** 2
        spot = np.clip(1.0 - d0, 0, 1) ** 0.6 * (1 - 0.6 * np.clip(Y - cy, 0, None) / (rr + 1e-6)).clip(0.3, 1)
        ring = np.clip(1 - np.abs(d0 - 1.25) / 0.45, 0, 1)                 # a thin rim round the chip ...
        lip = np.maximum(ring * np.clip((Y - cy) / (rr * 0.5), 0, 1),       # ... lit on its lower side
                         np.clip(1.0 - d1, 0, 1) * (d0 > 0.45) * (Y > cy))
        sh[y0:y1, x0:x1] = np.maximum(sh[y0:y1, x0:x1], spot)
        li[y0:y1, x0:x1] = np.maximum(li[y0:y1, x0:x1], lip)
    m = _soft(mask, 1.0)
    sh, li = _blur(sh, 0.8) * m, _blur(li, 0.8) * m
    out = a * (1 - sh[..., None] * (1 - dark))
    if base_col is not None:
        out = out * (1 - 0.5 * sh[..., None]) + base_col[None, None] * 0.5 * sh[..., None] * _lum(a)[..., None]
    return np.clip(out * (1 - li[..., None]) + np.clip(a * lit + 0.06, 0, 1) * li[..., None], 0, 1)


def scratches(a, mask, seed, n=10, length=(25, 60), width=4.0, dark=0.55, lit=1.35, ang=None):
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(_erode(mask, 3))
    if not len(xs):
        return a
    for i in rng.choice(len(xs), size=min(n, len(xs)), replace=False):
        an = rng.uniform(0, np.pi) if ang is None else ang + rng.normal(0, 0.25)
        pts = _path(rng, xs[i], ys[i], an, rng.uniform(*length), step=5, wobble=0.10)
        a = cut(a, mask, pts, width=width, dark=dark, lit=lit, lit_off=(0, 2.5))
    return a


def contrast(a, mask, k=1.3, pivot=None, sat=1.0):
    m = _soft(mask, 1.2)[..., None]
    L = _lum(a)
    p = np.median(L[mask]) if pivot is None and mask.any() else (pivot or 0.4)
    b = (a - p) * k + p
    g = _lum(b)[..., None]
    b = g + (b - g) * sat
    return np.clip(a * (1 - m) + b * m, 0, 1)


def _grid(n):
    v, u = np.mgrid[0:n, 0:n] / (n - 1)
    return u, v


# ------------------------------------------------------------------------------ the red skin

DIRT_COL = np.array([0.24, 0.15, 0.08])


def _vnoise(shape, seed, sx, sy):
    """Value noise stretched: cells sx wide and sy tall (drips and splashes run down)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    cells = rng.random((max(2, h // sy + 2), max(2, w // sx + 2)))
    im = Image.fromarray((cells * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im, float) / 255


def dirt_rise(a, mask, edge, seed, depth=0.16):
    """Dirt rising from the hooves: brown-black grime below `edge` (per-texel v of the dirt line),
    its top edge broken into irregular upward licks, splashes above it, heaviest at the bottom."""
    h, w = a.shape[:2]
    u, v = _grid(h)
    lick = 0.6 * _vnoise((h, w), seed, 14, 400) + 0.4 * _vnoise((h, w), seed + 1, 6, 400)   # varies along u
    top = edge - depth * (lick - 0.3) * 1.6
    body = _sstep(top - 0.015, top + 0.05, v)
    deep = _sstep(top, top + 0.22, v)
    n = fbm((h, w), seed + 2, (40, 16, 8))
    w_ = np.clip(body * (0.55 + 0.9 * (n - 0.5)) + 0.45 * deep, 0, 1)
    # splatter: small blots in the band just above the line
    rng = np.random.default_rng(seed + 3)
    spl = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(mask & (v > top - 0.06) & (v < top - 0.01))
    for i in rng.choice(len(xs), size=min(11, len(xs)), replace=False) if len(xs) else []:
        rr = rng.uniform(5.0, 8.5)                       # a few larger, irregular flecks (not a pepper)
        y0, y1, x0, x1 = max(ys[i] - 16, 0), min(ys[i] + 17, h), max(xs[i] - 16, 0), min(xs[i] + 17, w)
        X, Y = xx[y0:y1, x0:x1] - xs[i], yy[y0:y1, x0:x1] - ys[i]
        th = np.arctan2(Y, X)
        wob = 1 + 0.35 * np.sin(3 * th + rng.uniform(0, 6.3)) + 0.2 * np.sin(5 * th + rng.uniform(0, 6.3))
        d = (X ** 2 + (Y * 0.8) ** 2) / (rr * wob) ** 2
        spl[y0:y1, x0:x1] = np.maximum(spl[y0:y1, x0:x1], np.clip(1.3 - d, 0, 1) * rng.uniform(0.6, 0.9))
    w_ = np.maximum(w_, _blur(spl, 0.8)) * _soft(mask, 1.5)
    L = _lum(a)
    dirty = DIRT_COL[None, None] * (0.55 + 1.3 * L)[..., None]
    return np.clip(a * (1 - 0.82 * w_[..., None]) + dirty * (0.82 * w_[..., None]), 0, 1)


CRIMSON_FLOOR = np.array([0.27, 0.055, 0.03])   # the darkest a crevice on the red skin goes (#450e08)


def scar(a, mask, pts, width=6.0):
    """An old healed scar, painted raised: a pale pink-tan ridge lit from above and its shadow
    just below it (not a black cut)."""
    sh = a.shape[:2]
    ridge = _blur(_stroke(sh, pts, width), 1.0)
    shadow = np.clip(_blur(_stroke(sh, pts + np.array([0.0, width * 0.8]), width * 0.8), 1.2) - ridge, 0, 1)
    m = _soft(mask, 1.0)
    ridge, shadow = ridge * m, shadow * m
    pale = np.clip(a * np.array([1.05, 1.9, 1.9])[None, None] + np.array([0.06, 0.04, 0.03])[None, None], 0, 1)
    pale = pale * 0.55 + np.array([0.72, 0.36, 0.28])[None, None] * 0.45
    out = a * (1 - shadow[..., None] * 0.50)
    return out * (1 - ridge[..., None] * 0.85) + pale * (ridge[..., None] * 0.85)


def skin(a, mask, seed, scars=(), dirt=None, fine=0.25, broad=0.28, calm=None):
    """The red skin's grit. scars: (u, v, angle in degrees, length, count, spacing) as shares of the
    texture: `count` roughly parallel scars, like old claw marks."""
    n = a.shape[0]
    # the clipped highlights (the lifted red ran into 255 on Quuz's upper chest) rolled off, so they
    # keep some painted form instead of glowing flat
    hi = np.where(a > 0.80, 0.80 + (a - 0.80) * 0.45, a)
    m = _soft(mask, 1.5)[..., None]
    a = a * (1 - m) + hi * m
    # the muscle grooves cut deeper, into a deep crimson (DK2's painted red skin), not black
    crimson = (1.0, 0.62, 0.62)
    # calm: 0 .. 1, where the grooves are left as they were (the back's neck column)
    cm = mask if calm is None else mask & (calm < 0.5)
    out = cavity(a, cm, radius=20, deep=broad, ridge=0.06, brown=0.45, tint=crimson)
    out = cavity(out, cm, radius=9, deep=fine, ridge=0.06, brown=0.45, tint=crimson)
    out = mottle(out, mask, seed, amount=0.08, scales=(40, 18, 10), weights=(0.5, 0.4, 0.1))
    # grime gathers in the crevices only (the raised muscle stays clean)
    L = _lum(out)
    low = np.clip(0.25 + (_blur(L, 12) - L) / 0.05, 0, 1)
    out = grime(out, cm, seed + 10, amount=0.55, where=_blur(low, 6), lo=0.50, hi=0.75, scale=(90, 40, 20))
    if dirt is not None:
        out = dirt_rise(out, mask, dirt, seed + 20)
    rng = np.random.default_rng(seed + 30)
    for (u, v, ang, ln, count, gap) in scars:
        th = np.radians(ang)
        nrm = np.array([-np.sin(th), np.cos(th)]) * gap * n
        for i in range(count):
            off = nrm * (i - (count - 1) / 2)
            pts = _path(rng, u * n + off[0], v * n + off[1], th, ln * n * rng.uniform(0.8, 1.1), step=5, wobble=0.06)
            out = scar(out, mask, pts, width=6.5)
    # no crevice goes black: the floor is a deep crimson
    out = np.where(m > 0.5, np.maximum(out, CRIMSON_FLOOR[None, None]), out)
    # roll the brightest red off again (the sharpen when shrinking to 128 lifts it a little more)
    hi = np.where(out > 0.76, 0.76 + (out - 0.76) * 0.40, out)
    return out * (1 - m) + hi * m


def drop_shadow(a, caster, receiver, dy=10, r=6.0, dark=0.45):
    """A painted shadow of `caster` on `receiver` just below it (light from above)."""
    sh = np.zeros(caster.shape)
    sh[dy:] = caster[:-dy]
    sh = _blur(sh, r) * receiver * ~caster
    sh = _blur(sh * 1.0, 1.0)
    return np.clip(a * (1 - dark * np.clip(sh, 0, 1))[..., None], 0, 1)


# ------------------------------------------------------------------------------ silver plates

STEEL_DARK = np.array([0.20, 0.20, 0.21])
TARNISH = np.array([0.30, 0.27, 0.22])


def plates(a, plate, seed):
    """Quuz's silver plates, worn: still silver (bright glints kept), but darker steel, more contrast,
    tarnish in the low parts, worn bright edges, chips, dents and scratches."""
    if not plate.any():
        return a
    L = _lum(a)
    m = _soft(plate, 0.8)[..., None]
    p50 = np.median(L[plate])
    # darker and more contrasty steel, the glints kept bright
    b = a * 0.86
    b = (b - p50 * 0.86) * 1.30 + p50 * 0.78
    # tarnish in the low parts (Quuz's dark folds), blotchy
    low = np.clip((np.percentile(L[plate], 45) - L) / 0.25, 0, 1)
    nt = fbm(a.shape[:2], seed, (30, 12, 5))
    tw = np.clip(low * (0.5 + nt), 0, 1) * 0.50
    b = b * (1 - tw[..., None]) + (TARNISH[None, None] * (0.5 + L[..., None])) * tw[..., None]
    out = np.clip(a * (1 - m) + b * m, 0, 1)
    # mottled wear
    out = mottle(out, plate, seed + 1, amount=0.10, scales=(28, 14, 8))
    # worn bright edges on top (a broken band), grime where the plate's lower edge meets the skin
    k = 6
    above_out = np.ones_like(plate)
    above_out[k:] = ~plate[:-k]
    below_out = np.ones_like(plate)
    below_out[:-k] = ~plate[k:]
    top_band = plate & above_out
    low_band = plate & below_out & ~top_band
    nb = _noise(a.shape[:2], seed + 2, 7)
    wv = _soft(top_band & (nb > 0.40), 1.0)[..., None] * 0.70
    out = out * (1 - wv) + np.clip(out * 1.35 + 0.12, 0, 1) * wv
    lv = _blur(low_band.astype(float), 2.5) * plate
    lv = (lv * (0.6 + 0.8 * fbm(a.shape[:2], seed + 7, (16, 8, 4))))[..., None] * 0.55
    out = out * (1 - lv) + (TARNISH[None, None] * 0.6) * lv
    # grime on the lower part of each plate (dirt settles; light from above)
    u, v = _grid(a.shape[0])
    out = grime(out, plate, seed + 3, amount=0.40, where=np.clip((v - 0.48) / 0.2, 0, 1), colour=TARNISH,
                lo=0.45, hi=0.75, scale=(24, 10, 5))
    out = chips(out, plate, seed + 4, n=8, r=(8, 13), dark=0.50, lit=1.6, base_col=STEEL_DARK)
    out = scratches(out, plate, seed + 5, n=8, length=(20, 50), width=4.5, dark=0.50, lit=1.35)
    # keep the polished glints (Soles: "shiny silver"): the brightest of Quuz's plate light comes back
    glint = _sstep(np.percentile(L[plate], 80), np.percentile(L[plate], 97), L) * _soft(plate, 0.8)
    glint = glint * (0.55 + 0.45 * _noise(a.shape[:2], seed + 6, 9))          # broken by the wear
    return np.clip(out * (1 - glint[..., None]) + np.maximum(out, a) * glint[..., None], 0, 1)


# ------------------------------------------------------------------------------ chains, belt, flap

def metal_paint(a, mask, seed, k=1.35):
    """Quuz's painted chain links: more contrast, darker gaps, a little less gold, some grime."""
    if not mask.any():
        return a
    out = contrast(a, mask, k=k, sat=0.80)
    L = _lum(out)
    gap = np.clip((0.30 - L) / 0.25, 0, 1) * mask
    out = out * (1 - 0.55 * _soft(gap, 0.8)[..., None])
    out = grime(out, mask, seed, amount=0.45, lo=0.45, hi=0.75, scale=(16, 7, 4), colour=GRIME)
    return out


def cloth(a, mask, seed, hem=None):
    """The loin flap: contrast, grime and stains, darker toward the hem."""
    if not mask.any():
        return a
    out = contrast(a, mask, k=1.25, sat=0.9)
    out = cavity(out, mask, radius=4, deep=0.45, ridge=0.15, brown=0.2)
    out = mottle(out, mask, seed, amount=0.20, scales=(12, 6, 3))
    out = grime(out, mask, seed + 1, amount=0.55, where=hem, lo=0.40, hi=0.70, scale=(16, 8, 4))
    return out


# ------------------------------------------------------------------------------ hooves

def hooves(a, mask, seed, v0):
    """Growth ridges, cracks, a chipped dirty rim. v0: per-texel top of the hoof (0..1 v)."""
    if not mask.any():
        return a
    n = a.shape[0]
    u, v = _grid(n)
    t = np.clip((v - v0) / np.maximum(1 - v0, 1e-3), 0, 1)
    wob = fbm(a.shape[:2], seed, (30, 10, 5)) - 0.5
    ridge = 0.5 + 0.5 * np.sin((t * 4 + wob * 1.2) * 2 * np.pi)
    out = a * (0.62 + 0.26 * ridge)[..., None] * _soft(mask)[..., None] + a * (1 - _soft(mask))[..., None]
    out = cavity(out, mask, radius=5, deep=0.5, ridge=0.2, brown=0.2)
    out = mottle(out, mask, seed + 1, amount=0.16, scales=(20, 10, 6))
    rng = np.random.default_rng(seed + 2)
    ys, xs = np.nonzero(_erode(mask, 3))
    for i in rng.choice(len(xs), size=min(9, len(xs)), replace=False):
        pts = _path(rng, xs[i], ys[i], np.radians(90 + rng.normal(0, 15)), rng.uniform(14, 34), step=4, wobble=0.25)
        out = cut(out, mask, pts, width=3.5, dark=0.35, lit=1.25, lit_off=(2, 0))
    out = grime(out, mask, seed + 3, amount=0.75, where=_sstep(0.35, 0.95, t), colour=DIRT, lo=0.30, hi=0.65,
                scale=(20, 8, 4))
    out = chips(out, mask, seed + 4, n=10, r=(3, 6), dark=0.35, lit=1.3)
    return out


# ------------------------------------------------------------------------------ horn

def horn(a, mask, seed, n_cracks=4):
    if not mask.any():
        return a
    out = cavity(a, mask, radius=5, deep=0.35, ridge=0.12, brown=0.25)
    out = mottle(out, mask, seed, amount=0.10, scales=(28, 14, 8), weights=(0.5, 0.4, 0.1))
    rng = np.random.default_rng(seed + 1)
    ys, xs = np.nonzero(_erode(mask, 3))
    for i in rng.choice(len(xs), size=min(n_cracks, len(xs)), replace=False):
        pts = _path(rng, xs[i], ys[i], np.radians(rng.uniform(-20, 20)), rng.uniform(10, 24), step=4, wobble=0.3)
        out = cut(out, mask, pts, width=3, dark=0.45, lit=1.2, lit_off=(0, 2))
    return out
