"""Texture atlas of the custom Behemoth variant. LICENCE: GPL-3.0 (base texture: MegaGlest, CC BY-SA 3.0).

The 256x256 Behemoth texture sits in the left half of a 512x256 atlas, edited: the tribal tattoos (the team-colour strokes) are removed from
the body but kept on the face, and the skin is regraded to a deep blood red. The right half holds new materials for the added parts:
dark armour plate, steel (collar and chain) and bone (spikes).
"""
import colorsys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

FACE = (186, 0, 256, 105)                 # x0, y0, x1, y1 in the 256x256 texture: the head island (its tattoo is kept)
LIGHT_PARTS = [(136, 88, 254, 128), (0, 212, 52, 256), (46, 178, 134, 236)]    # loincloth trims and the horn strip keep their cream
CLUB = (0, 0, 136, 126)                   # the club and its zigzag strip stay as they are
ATLAS = (512, 256)
PLATE = (256, 0, 384, 128)                # armour plate
STEEL = (384, 0, 512, 128)                # collar, chain
BONE = (256, 128, 384, 256)               # spikes
WHITE_DOT = (384, 128, 512, 256)          # unused spare


def remove_body_tattoos(rgba: Image.Image) -> Image.Image:
    """Lift the alpha of the tattoo strokes to the surrounding body value and paint them over with the nearby skin colour."""
    a = np.asarray(rgba, float).copy()
    alpha = a[..., 3]
    body = ndimage.median_filter(alpha, size=9)                       # the usual alpha of the area around a stroke
    stroke = (alpha < body - 28) & (alpha < 200)
    stroke = ndimage.binary_dilation(stroke, iterations=2)
    stroke[FACE[1]:FACE[3], FACE[0]:FACE[2]] = False
    stroke[0:CLUB[3], 45:CLUB[2]] = False
    stroke[0:62, 0:45] = False
    valid = ~stroke & (alpha > 150)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = a[idx[0], idx[1]]
    for c in range(3):
        filled[..., c] = ndimage.gaussian_filter(filled[..., c], 1.2)
    out = a.copy()
    out[stroke, :3] = filled[stroke, :3]
    out[stroke, 3] = body[stroke]
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def bake(rgba: Image.Image, team=(150, 36, 28)) -> Image.Image:
    back = Image.new("RGBA", rgba.size, (*team, 255))
    back.alpha_composite(rgba)
    return back.convert("RGB")


def regrade_red(img: Image.Image) -> Image.Image:
    """Deep blood-red skin; horns, loincloth, leather, studs, claws and the club keep their own colours."""
    px = np.asarray(img, float) / 255.0
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    mx, mn = px.max(-1), px.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    lum = 0.30 * r + 0.59 * g + 0.11 * b
    cream = (g > r * 0.82) & (lum > 0.26) & (b < g * 0.95)           # horns, loincloth, the olive-cream parts
    leather = (r > g * 1.35) & (g > b * 1.25) & (sat > 0.35) & (lum < 0.5)   # strap and cuffs (orange-brown)
    metal = (sat < 0.10) & (lum > 0.35)                                # rings, studs
    yy, xx = np.mgrid[0:img.height, 0:img.width]
    trim = np.zeros_like(cream)
    for x0, y0, x1, y1 in LIGHT_PARTS:
        trim[y0:y1, x0:x1] = True
    keep = (cream & trim) | leather | metal
    keep |= ((xx >= 45) & (xx < CLUB[2]) & (yy < CLUB[3])) | ((xx < 45) & (yy < 62))
    t = np.clip((lum - 0.04) / 0.45, 0, 1) ** 0.9                       # tonal ramp of the old skin
    dark = np.array([0.10, 0.015, 0.02])
    mid = np.array([0.34, 0.045, 0.04])
    light = np.array([0.64, 0.13, 0.09])
    ramp = np.where(t[..., None] < 0.5, dark + (mid - dark) * (t[..., None] / 0.5), mid + (light - mid) * ((t[..., None] - 0.5) / 0.5))
    soft = ndimage.gaussian_filter((~keep).astype(float), 1.0)[..., None]   # blend at the edges of the kept areas
    out = px * (1 - soft) + ramp * soft
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), "RGB")


def _noise(shape, scale, rng):
    n = rng.random(shape)
    return ndimage.gaussian_filter(n, scale) * (scale * 3.5)


def material_plate(size=(128, 128), seed=1) -> Image.Image:
    """Dark tough plate: gunmetal, brushed grain, panel seams, rivets, a red-black rim heat."""
    rng = np.random.default_rng(seed)
    w, h = size
    base = np.array([46, 40, 42], float)
    grain = ndimage.gaussian_filter(rng.random((h, w)), (0.4, 6)) - 0.5
    blot = ndimage.gaussian_filter(rng.random((h, w)), 5) - 0.5
    img = base[None, None] * (1 + 0.55 * grain[..., None] + 0.5 * blot[..., None])
    for x in range(0, w, 32):
        img[:, x:x + 2] *= 0.45                                       # vertical seams
    for y in range(0, h, 40):
        img[y:y + 2] *= 0.45
    yy, xx = np.mgrid[0:h, 0:w]
    for cx in range(8, w, 32):
        for cy in range(8, h, 40):
            d = np.hypot(xx - cx, yy - cy)
            img[d < 3.2] = (120, 112, 112)                               # rivet
            img[(d >= 3.2) & (d < 4.4)] *= 0.5
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    img[edge < 5] = img[edge < 5] * 0.6 + np.array([60, 10, 8]) * 0.4   # dark-red rim
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def material_steel(size=(128, 128), seed=2) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    base = np.array([108, 110, 120], float)
    grain = ndimage.gaussian_filter(rng.random((h, w)), (5, 0.5)) - 0.5
    blot = ndimage.gaussian_filter(rng.random((h, w)), 4) - 0.5
    img = base[None, None] * (1 + 0.5 * grain[..., None] + 0.45 * blot[..., None])
    rust = np.clip(ndimage.gaussian_filter(rng.random((h, w)), 3) - 0.52, 0, 1) * 6
    img = img * (1 - 0.5 * rust[..., None]) + rust[..., None] * np.array([90, 40, 20]) * 0.5
    for _ in range(14):                                               # scratches
        y = rng.integers(0, h)
        x0 = rng.integers(0, w - 30)
        img[y, x0:x0 + rng.integers(10, 30)] *= 1.35
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def material_bone(size=(128, 128), seed=3) -> Image.Image:
    """Chin spikes: ivory fading to a dark base and a pale point, like the Behemoth's own horns."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy = np.linspace(0, 1, h)[:, None]
    base = np.array([196, 188, 150], float)
    dark = np.array([70, 55, 40], float)
    g = np.clip(yy * 1.0, 0, 1)
    img = dark[None] * (1 - g[..., None]) * 0 + (dark * (1 - np.clip(1 - yy * 2.0, 0, 1)[..., None]) + base * np.clip(1 - yy * 2.0, 0, 1)[..., None]) * 0 + 0
    ramp = np.clip(yy * 1.2, 0, 1)
    img = np.repeat(dark[None, None] * (1 - ramp[..., None]) + base[None, None] * ramp[..., None], w, axis=1)
    img = img * (1 + 0.25 * (ndimage.gaussian_filter(rng.random((h, w)), (6, 0.8)) - 0.5)[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def build_atlas(base_png) -> Image.Image:
    rgba = Image.open(base_png).convert("RGBA")
    body = regrade_red(bake(remove_body_tattoos(rgba)))
    atlas = Image.new("RGB", ATLAS, (0, 0, 0))
    atlas.paste(body, (0, 0))
    atlas.paste(material_plate(), PLATE[:2])
    atlas.paste(material_steel(), STEEL[:2])
    atlas.paste(material_bone(), BONE[:2])
    return atlas


def uv_rect(rect, u, v):
    """(u, v in 0..1 inside a material rectangle) -> atlas UV (v measured top-down like the viewer's texture)."""
    x0, y0, x1, y1 = rect
    return np.stack([(x0 + u * (x1 - x0)) / ATLAS[0], (y0 + v * (y1 - y0)) / ATLAS[1]], -1)


def material_cuff(body: Image.Image) -> Image.Image:
    """The studded leather arm band of the Behemoth, cut out of its skin texture and levelled (for the ankle bands)."""
    crop = body.convert("RGB").crop((80, 120, 160, 200)).resize((640, 640), Image.BICUBIC)
    strip = crop.rotate(114, resample=Image.BICUBIC).crop((250, 242, 620, 342))
    return strip.resize((256, 68), Image.LANCZOS)


def materials(base_png) -> dict:
    """name -> image of every texture of the custom character: the red body, plate armour, steel, bone and the leather band."""
    atlas = build_atlas(base_png)
    body = atlas.crop((0, 0, 256, 256)).convert("RGB")
    return {"body": body, "plate": material_plate().convert("RGB"), "steel": material_steel().convert("RGB"),
            "bone": material_bone().convert("RGB"), "cuff": material_cuff(body)}
