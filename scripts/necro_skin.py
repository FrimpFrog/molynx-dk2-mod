"""The Necromancer's skins: the Black Knight's two 128 x 128 skins (DKnightfrnt, DKnightBak) repainted
after Soles' approved preview (steel armour, deep-blue electric arms and belly, gold horns, trim and
hilt, glowing blue eyes, a small skull on the belt, a clean steel blade).

Every texel is given a body part from the reference pose (DRKnight-Idle1 frame 0): the triangles
are rasterised in UV space and each texel takes the 3D point under it, so the paint follows the
model, not guessed boxes. The original texel's brightness is kept as the shading, so the DK2
painted look (folds, rivets, plate edges) survives the recolour. Painting is done at 4x (512) and
shrunk to 128 like Molynx's skins.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

UP = 4                      # paint at 512, ship at 128
SIZE = 128
CX = 0.011                  # the Black Knight's centre line (x) in KMF units

STEEL = np.array([0.62, 0.66, 0.74])          # light, slightly cool steel
STEEL_DARK = np.array([0.20, 0.22, 0.28])
BLUE_DEEP = np.array([0.05, 0.12, 0.42])
BLUE_MID = np.array([0.10, 0.28, 0.72])
BOLT = np.array([0.62, 0.86, 1.00])
BOLT_CORE = np.array([0.90, 0.98, 1.00])
GOLD = np.array([0.86, 0.64, 0.18])
GOLD_DARK = np.array([0.40, 0.25, 0.05])
EYE = np.array([0.35, 0.85, 1.00])
BONE = np.array([0.92, 0.88, 0.76])


# ------------------------------------------------------------------------------ texel -> body point

def texel_maps(sprites, points, textures, tex: str, res: int):
    """For one texture: per texel the 3D point (res, res, 3; NaN where nothing maps), the sprite
    index (-1 none) and the triangle's mean normal facing (y of the outward normal)."""
    pos = np.full((res, res, 3), np.nan)
    spr = np.full((res, res), -1, int)
    for si, sp in enumerate(sprites):
        if textures[si] != tex:
            continue
        uv = np.array([v.uv for v in sp.vertices], float) / 32768.0 * res
        p = points[si]
        for a, b, c in sp.triangles[0]:
            xs, ys = uv[[a, b, c], 0], uv[[a, b, c], 1]
            x0, x1 = int(max(np.floor(xs.min()) - 1, 0)), int(min(np.ceil(xs.max()) + 1, res))
            y0, y1 = int(max(np.floor(ys.min()) - 1, 0)), int(min(np.ceil(ys.max()) + 1, res))
            if x1 <= x0 or y1 <= y0:
                continue
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
            w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
            w2 = 1 - w0 - w1
            pad = 0.6 / max(res / SIZE, 1)            # catch texels whose centre sits just outside an edge
            inside = (w0 >= -pad) & (w1 >= -pad) & (w2 >= -pad)
            q = w0[..., None] * p[a] + w1[..., None] * p[b] + w2[..., None] * p[c]
            blk = pos[y0:y1, x0:x1]
            take = inside & (np.isnan(blk[..., 0]) | ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)))
            blk[take] = q[take]
            spr[y0:y1, x0:x1][take] = si
    return pos, spr


def body_label(pos: np.ndarray, spr: np.ndarray, sword_sprite: int, horn_mask: np.ndarray) -> np.ndarray:
    """Per texel a part name (object array): sword, horn, helmet, pauldron, uparm, forearm, hand,
    chest, belly, belt, thigh, shin, boot, none."""
    x, y, h = pos[..., 0] - CX, pos[..., 1], -pos[..., 2]
    ax = np.abs(x)
    lab = np.full(pos.shape[:2], "none", object)
    ok = ~np.isnan(h)
    lab[ok & (h <= 0.06)] = "boot"
    lab[ok & (h > 0.06) & (h <= 0.17)] = "shin"
    lab[ok & (h > 0.17) & (h <= 0.315)] = "thigh"
    lab[ok & (h > 0.315) & (h <= 0.385)] = "belt"
    lab[ok & (h > 0.385) & (h <= 0.475)] = "belly"
    lab[ok & (h > 0.475)] = "chest"
    lab[ok & (h > 0.5) & (ax > 0.10)] = "pauldron"
    arm = ok & (ax > 0.118) & (h < 0.5) & (h > 0.20)
    lab[arm & (h >= 0.43)] = "uparm"
    lab[arm & (h < 0.43) & (h >= 0.315)] = "forearm"
    lab[arm & (h < 0.315)] = "hand"
    lab[ok & (h > 0.60) & (ax < 0.1)] = "helmet"
    lab[horn_mask] = "horn"
    lab[spr == sword_sprite] = "sword"
    return lab


# ------------------------------------------------------------------------------ paint helpers

def _lum(a):
    return a @ np.array([0.30, 0.55, 0.15])


def _tone(lum, colour, lo=0.25, gain=1.25, gamma=0.85):
    """Shade a flat colour with the original brightness (normalised so mid-grey keeps the colour)."""
    s = np.clip(lo + gain * np.power(np.clip(lum, 0, 1), gamma), 0, 1.6)
    return np.clip(colour[None, None] * s[..., None], 0, 1)


def _noise(shape, seed, scale):
    """Smooth value noise in 0..1 (bilinear blow-up of random cells)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    cells = rng.random((max(2, h // scale + 2), max(2, w // scale + 2)))
    im = Image.fromarray((cells * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im, float) / 255


def lightning(shape, seed: int, density: float = 1.0) -> np.ndarray:
    """A crackling electric pattern in 0..1 (1 = bolt core): random branching zig-zag bolts drawn
    at the painting resolution, then softened into a glow."""
    rng = np.random.default_rng(seed)
    h, w = shape
    img = np.zeros((h, w))
    n = int(14 * density * (h * w) / (512 * 512) * 4) + 6
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.15, 0.45) * min(h, w) * 0.5
        steps = int(length / 3)
        branch_at = set(rng.choice(range(max(steps, 2)), size=min(2, max(steps, 2)), replace=False).tolist())
        stack = [(x, y, ang, steps, 1.0)]
        while stack:
            x, y, ang, steps, wgt = stack.pop()
            for s in range(steps):
                ang += rng.normal(0, 0.55)
                x, y = x + 3 * np.cos(ang), y + 3 * np.sin(ang)
                xi, yi = int(x) % w, int(y) % h
                img[yi, xi] = max(img[yi, xi], wgt)
                img[yi, (xi + 1) % w] = max(img[yi, (xi + 1) % w], wgt * 0.8)
                if s in branch_at and wgt > 0.5:
                    stack.append((x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1), steps // 2, wgt * 0.7))
    core = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)), float) / 255
    glow = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3.0)), float) / 255
    return np.clip(np.maximum(core * 1.6, glow * 2.2), 0, 1)


def electric(lum, seed, density=1.0):
    """Deep blue, shaded, with light-blue crackling bolts (the arms and belly of the preview)."""
    h, w = lum.shape
    n = _noise((h, w), seed + 1, 24)
    base = BLUE_DEEP[None, None] * (1 - n[..., None]) + BLUE_MID[None, None] * n[..., None]
    base = np.clip(base * (0.55 + 0.9 * np.clip(lum, 0, 1))[..., None], 0, 1)
    bolt = lightning((h, w), seed, density)
    col = base * (1 - bolt[..., None]) + (BOLT[None, None] * (1 - bolt[..., None] ** 3) + BOLT_CORE[None, None] * bolt[..., None] ** 3) * bolt[..., None]
    return np.clip(col, 0, 1)


def steel(lum, contrast=1.15, tint=STEEL):
    """Original shading pushed to bright steel (the Black Knight's gunmetal is much darker)."""
    l = np.clip((lum - 0.06) * 2.1, 0, 1)
    l = np.clip(0.5 + (l - 0.5) * contrast, 0, 1)
    return np.clip(STEEL_DARK[None, None] * (1 - l[..., None]) + tint[None, None] * l[..., None] * 1.25, 0, 1)


def gold(lum):
    l = np.clip((lum - 0.05) * 1.9, 0, 1)
    return np.clip(GOLD_DARK[None, None] * (1 - l[..., None]) + GOLD[None, None] * (0.6 + 0.7 * l[..., None]) * l[..., None]
                   + GOLD_DARK[None, None] * 0.3, 0, 1)


def blue_mottle(col, lum, mask, seed, amount=0.55):
    """Blue patches over steel (the preview's legs)."""
    n = _noise(mask.shape, seed, 30)
    m = mask & (n > 0.55)
    k = np.clip((n - 0.55) / 0.2, 0, 1)[..., None] * amount
    blue = BLUE_MID[None, None] * (0.6 + 0.8 * np.clip(lum, 0, 1))[..., None]
    out = col.copy()
    out[m] = (col * (1 - k) + blue * k)[m]
    return out


def gold_flecks(col, lum, mask, seed, amount=0.12):
    rng = np.random.default_rng(seed)
    r = rng.random(mask.shape)
    m = mask & (r < amount) & (lum > 0.15)
    out = col.copy()
    out[m] = gold(lum)[m]
    return out


# the bare fists of the Black Knight (tan with knuckles): steel gauntlets on the Necromancer
FIST_UV = {"dknightfrnt": [(0.60, 0.78, 0.78, 0.95)], "dknightbak": [(0.0, 0.16, 0.13, 0.31)]}

SKULL = ["..XXXX..",
         ".XXXXXX.",
         "XXXXXXXX",
         "X..XX..X",
         "XXXXXXXX",
         ".XX..XX.",
         "..XXXX..",
         "..X.X.X."]


def paint_skull(col, centre, px, up_dir):
    """A small bone skull (8 x 8 texels at 128) centred on `centre` (row, col at paint resolution),
    with the skull's top towards `up_dir` (unit vector in (row, col))."""
    up = np.array(up_dir, float) / np.linalg.norm(up_dir)
    right = np.array([up[1], -up[0]])          # row, col
    s = px                                     # paint pixels per skull cell
    h, w = col.shape[:2]
    out = col.copy()
    for r, line in enumerate(SKULL):
        for c, ch in enumerate(line):
            if ch != "X":
                continue
            # cell centre relative to the skull centre, in cells
            dv, du = (r - 3.5), (c - 3.5)
            for a in range(s):
                for b in range(s):
                    off = -up * (dv + (a + 0.5) / s - 0.5) + right * (du + (b + 0.5) / s - 0.5)
                    rr, cc = int(round(centre[0] + off[0] * s)), int(round(centre[1] + off[1] * s))
                    if 0 <= rr < h and 0 <= cc < w:
                        shade = 0.85 + 0.15 * (1 - r / 8)
                        out[rr, cc] = BONE * shade
    # dark outline under the skull so it reads at 128
    return out


# ------------------------------------------------------------------------------ the two skins

def paint(tex_name: str, original: Image.Image, pos, spr, lab, seed: int) -> Image.Image:
    """The Necromancer's version of one skin (128 x 128)."""
    big = SIZE * UP
    src = np.asarray(original.convert("RGB").resize((big, big), Image.BICUBIC), float) / 255
    src_near = np.asarray(original.convert("RGB").resize((big, big), Image.NEAREST), float) / 255
    lum = _lum(src)
    r, g, b = src_near[..., 0], src_near[..., 1], src_near[..., 2]
    # the label maps come at paint resolution
    out = steel(lum)
    L = lambda *names: np.isin(lab, names)

    # bare skin (tan) and the straps over it, wherever they are, become the electric blue
    tan = (r > g * 1.08) & (g > b * 1.05) & (r > 0.25)
    red_eye = (r > 0.45) & (g < 0.25) & (b < 0.25)
    gold_stud = (r > 0.45) & (g > 0.33) & (b < 0.25) & (r - b > 0.3) & np.isin(lab, ["chest"])
    front = tex_name == "dknightfrnt"
    uu = np.mgrid[0:big, 0:big][1] / big
    vv = np.mgrid[0:big, 0:big][0] / big
    strip = front & (uu < 0.165)                      # the sword strip: blade, guard, grip, pommel

    arms = L("uparm", "forearm") & ~strip
    out[arms] = electric(lum, seed + 10, 4.0)[arms]
    belly = L("belly")
    out[belly] = electric(lum, seed + 20, 2.6)[belly]
    pa = L("pauldron")
    out[pa] = np.clip(_tone(lum, BLUE_MID * 0.9, lo=0.15, gain=1.6) + 0.05, 0, 1)[pa]
    # hands: steel gauntlets (any tan on them is a bare hand in the original)
    fist = np.zeros_like(tan)
    for u0, v0, u1, v1 in FIST_UV.get(tex_name, ()):
        fist |= (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)
    hands = (L("hand") | fist) & ~strip
    out[hands] = steel(np.clip(lum * 1.25 + 0.05, 0, 1))[hands]
    # leftover tan skin elsewhere (wrists, neck seams): electric too
    # bone-coloured texels on or beside the helmet are the horns' (both skins): gold
    hornish = tan & L("helmet", "horn", "none") & ~strip
    m8 = Image.fromarray((hornish * 255).astype(np.uint8))
    closed = np.asarray(m8.filter(ImageFilter.MaxFilter(2 * UP + 1)).filter(ImageFilter.MinFilter(2 * UP + 1)), np.uint8) > 0
    hornish = (closed | hornish | L("horn")) & ~strip & ~L("chest", "pauldron", "uparm", "forearm", "hand")
    out[hornish] = gold(np.clip(0.25 + lum * 1.2, 0, 1))[hornish]
    warm = (r > g) & (g > b) & ((r - b) > 0.12)
    studs = L("chest") & warm & front & (vv < 0.25)
    stray = tan & ~L("horn", "sword", "hand", "helmet", "none", "chest") & ~arms & ~strip & ~hornish
    out[stray] = electric(lum, seed + 30, 1.0)[stray]

    # legs and belt: steel with blue mottling and a few gold flecks, the cloth between plates blue
    legs = L("thigh", "shin", "boot") & ~strip
    soft = np.asarray(Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(UP * 0.6)), float) / 255
    out[legs] = steel(0.55 * lum + 0.45 * soft, contrast=0.75)[legs]
    out = blue_mottle(out, lum, legs & ~L("boot"), seed + 40, 0.6)
    out = gold_flecks(out, lum, L("thigh") | L("belt"), seed + 41, 0.05)
    cloth = (L("thigh", "belt")) & (r > b * 1.35) & (r > 0.12) & ~tan     # the red-brown cloth
    out[cloth] = _tone(lum, BLUE_MID, lo=0.2, gain=1.4)[cloth]

    # chest: steel, the ribbing under the pecs turned gold trim, the studs bright gold
    chest = L("chest")
    rib = chest & (lum < 0.18) & (lum > 0.06) & front & (vv < 0.32)
    out[rib] = (0.5 * gold(lum * 2.2) + 0.5 * out)[rib]
    out[gold_stud | studs] = gold(np.clip(lum * 1.5, 0, 1))[gold_stud | studs]

    # horns gold, the eyes a glowing blue
    horn = L("horn")
    out[horn] = gold(np.clip(lum * 1.4, 0, 1))[horn]
    eyes = red_eye & L("helmet")
    glow = np.asarray(Image.fromarray((eyes * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(UP * 0.8)), float) / 255
    out = np.clip(out * (1 - 0.6 * glow[..., None]) + EYE[None, None] * glow[..., None] * 1.1, 0, 1)
    out[eyes] = np.clip(EYE * 1.15, 0, 1)

    # the sword: a clean steel blade (the blood is taken out), gold guard and grip with dark bands
    sw = strip
    if sw.any():
        blade = sw & (vv < 0.555)
        bl = np.clip(src[..., 1] * 0.6 + src[..., 2] * 0.4, 0, 1)        # blood is red: green / blue show the steel
        centre_u = 0.075
        edge = np.clip(1 - np.abs(uu - centre_u) / 0.06, 0, 1)
        bl = 0.35 + 0.45 * bl + 0.25 * edge
        out[blade] = np.clip(np.array([0.70, 0.73, 0.76])[None, None] * bl[..., None], 0, 1)[blade]
        guard = sw & (vv >= 0.555) & (vv < 0.645)
        out[guard] = gold(lum * 1.3)[guard]
        grip = sw & (vv >= 0.645)
        bands = (np.floor((vv - 0.645 + 0.6 * np.abs(uu - 0.06)) * 128 / 2.5) % 2 == 0)     # chevrons
        gg = gold(lum * 1.2)
        gg[bands] *= 0.45
        out[grip] = gg[grip]
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


def finish(img: Image.Image) -> Image.Image:
    """512 -> 128 the way Molynx's skins were made: a good downscale and a light sharpen."""
    small = img.resize((SIZE, SIZE), Image.LANCZOS)
    return small.filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2))
