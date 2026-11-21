"""Procedural DK2-style textures for custom creatures, plus UV guide sheets for hand repainting.

Textures keep the size and UV layout of the creature they are made from (128x128 for the Giant), so
the original UVs stay valid. Everything is driven by the creature spec's "textures" section and by
which triangles (by position in the creature's pose) cover which part of the texture.
LICENCE: GPL-3.0 when combined with the rest of dk2_tools (texture decoder).
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scripts.custom_creature import region_weights

SS = 4   # supersampling for UV masks


def uv_mask(size, uv, indices, tri_keep=None) -> np.ndarray:
    """Float mask (H, W) of the texels covered by the selected triangles' UV polygons."""
    w, h = size
    img = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(img)
    tris = np.asarray(indices).reshape(-1, 3)
    for k, t in enumerate(tris):
        if tri_keep is not None and not tri_keep[k]:
            continue
        d.polygon([(float(uv[i][0]) * w * SS, float(uv[i][1]) * h * SS) for i in t], fill=255)
    return np.asarray(img.resize((w, h), Image.BOX), float) / 255.0


def tri_in_region(points, indices, region) -> np.ndarray:
    idx = np.asarray(indices).reshape(-1, 3)
    return region_weights(np.asarray(points)[idx].mean(1), region) > 0


def uv_guide(img: Image.Image, uv, indices, scale=4, label=None) -> Image.Image:
    """The texture enlarged with the UV triangles drawn on it (for repainting by hand)."""
    big = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(big)
    for t in np.asarray(indices).reshape(-1, 3):
        pts = [(float(uv[i][0]) * big.width, float(uv[i][1]) * big.height) for i in t]
        d.polygon(pts, outline=(0, 255, 255))
    if label:
        d.text((4, 4), label, fill=(255, 255, 0))
    return big


# ------------------------------------------------------------------ colour helpers
def ramp(lum: np.ndarray, stops) -> np.ndarray:
    """Gradient map: lum in 0..1 -> RGB float (H, W, 3); stops = [(pos, (r, g, b)), ...]."""
    pos = np.array([s[0] for s in stops], float)
    cols = np.array([s[1] for s in stops], float)
    return np.stack([np.interp(lum, pos, cols[:, c]) for c in range(3)], -1)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.30 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]) / 255.0


def stretch(lum: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(lum[valid], [3, 97]) if valid.any() else (0, 1)
    return np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)


def _hsv(rgb: np.ndarray):
    x = rgb / 255.0
    mx, mn = x.max(-1), x.min(-1)
    v, d = mx, mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(d == 0, 0, np.where(mx == r, ((g - b) / np.maximum(d, 1e-9)) % 6,
                                     np.where(mx == g, (b - r) / np.maximum(d, 1e-9) + 2,
                                              (r - g) / np.maximum(d, 1e-9) + 4))) / 6.0
    return h, s, v


def _blur(mask, r=0.8):
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), float) / 255


def _mix(base, new, w):
    return base * (1 - w[..., None]) + new * w[..., None]


# ------------------------------------------------------------------ patterns
def plates(shape, cell=(15, 11), seed=1, palette=None):
    """Overlapping chiton-style plates -> (rgb float, strength). Plate colour varies pearly per plate."""
    h, w = shape
    rng = np.random.default_rng(seed)
    palette = palette or [(84, 62, 140), (48, 104, 124), (120, 70, 124), (44, 78, 140), (136, 92, 116)]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cw, ch = cell
    row = np.floor(yy / ch)
    xs = xx + (row % 2) * cw / 2
    col = np.floor(xs / cw)
    lx, ly = (xs / cw) % 1.0, (yy / ch) % 1.0
    boundary = 0.72 + 0.28 * (2 * lx - 1) ** 2        # plate's rounded lower edge
    d = boundary - ly                                  # >0 inside the plate
    tint = np.array([palette[int(i)] for i in rng.integers(0, len(palette), (int(row.max()) + 2) * (int(col.max()) + 3))])
    cid = (row.astype(int) * (int(col.max()) + 3) + col.astype(int) + 1)
    base = tint[cid % len(tint)]
    shade = np.clip(0.55 + 0.75 * np.clip(d / 0.6, 0, 1) ** 0.8, 0, 1.3)
    shade = np.where(d < 0.10, 0.32, shade)           # dark overlap line
    rim = ((d > 0.10) & (d < 0.22)) * 0.35            # lit rim above the line
    rgb = np.clip(base * shade[..., None] + rim[..., None] * 70, 0, 255)
    return rgb, np.ones(shape)


def spiral(shape, centre, turns=1.6, bands=7, palette=None):
    """Logarithmic nautilus spiral with ridges -> (rgb, strength)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = xx - centre[0], yy - centre[1]
    r = np.hypot(dx, dy) + 1.0
    th = np.arctan2(dy, dx)
    phase = (np.log(r) * bands / turns - th / (2 * np.pi) * bands / 2.5) % 1.0
    ridge = np.abs(np.sin(np.pi * phase))
    palette = palette or [(34, 22, 70), (86, 62, 128), (150, 116, 170)]
    t = np.clip(phase, 0, 1)
    base = ramp(t, [(0, palette[0]), (0.55, palette[1]), (1.0, palette[2])])
    rgb = base * (0.45 + 0.6 * ridge[..., None] ** 0.6)
    return np.clip(rgb, 0, 255), np.ones(shape)


def suckers(shape, spacing=9.0, radius=3.2, base=(168, 148, 198)):
    """Hex lattice of sucker dots on a pale lilac skin -> (rgb, strength)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rowh = spacing * 0.866
    row = np.round(yy / rowh)
    cx = (np.round((xx - (row % 2) * spacing / 2) / spacing)) * spacing + (row % 2) * spacing / 2
    cy = row * rowh
    dist = np.hypot(xx - cx, yy - cy)
    rgb = np.zeros((h, w, 3)) + np.array(base)
    ring = (dist > radius * 0.55) & (dist < radius)
    inner = dist <= radius * 0.55
    rgb[ring] = np.array(base) * 0.55
    rgb[inner] = np.array(base) * 0.8 + 40
    rgb[dist >= radius] = np.array(base) * (0.92 + 0.08 * np.sin(xx[dist >= radius] * 0.7))[:, None]
    return np.clip(rgb, 0, 255), np.ones(shape)


def paint_eye(img: np.ndarray, centre, width=4.2, height=1.8, colour=(255, 214, 60)):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    e = ((xx - centre[0]) / width) ** 2 + ((yy - centre[1]) / height) ** 2
    glow = np.clip(1 - e, 0, 1)
    img[:] = _mix(img, np.zeros_like(img) + colour, glow ** 0.5 * 0.95)
    slit = (np.abs(xx - centre[0]) < 0.6) & (np.abs(yy - centre[1]) < height * 0.9)
    img[slit] = (20, 10, 30)


def barnacles(shape, count=40, seed=3, base=(225, 200, 190)):
    h, w = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rgb = np.zeros((h, w, 3)) + np.array(base) * 0.6
    strength = np.zeros(shape)
    for _ in range(count):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(1.4, 3.0)
        d = np.hypot(xx - cx, yy - cy)
        rgb[d < r] = np.array(base) * (0.75 + 0.5 * (1 - d[d < r] / r))[:, None]
        rgb[(d < r * 0.4)] = (60, 40, 60)
        strength[d < r + 0.6] = 1.0
    return np.clip(rgb, 0, 255), strength


PATTERNS = {"plates": plates, "spiral": spiral, "suckers": suckers, "barnacles": barnacles}

BODY_SKIN = [(0, (14, 10, 34)), (0.35, (40, 52, 100)), (0.7, (70, 130, 152)), (1, (176, 218, 216))]
BODY_CLOTH = [(0, (10, 6, 22)), (0.5, (54, 26, 82)), (1, (142, 88, 154))]
BODY_CREST = [(0, (6, 40, 52)), (0.5, (22, 150, 150)), (1, (200, 255, 235))]
PEARL = [(0, (74, 30, 62)), (0.5, (224, 142, 132)), (1, (252, 236, 226))]
ABYSS = [(0, (10, 8, 24)), (0.5, (40, 50, 92)), (1, (132, 192, 192))]
FLESH = [(0, (60, 20, 90)), (0.6, (150, 70, 160)), (1, (240, 170, 200))]


def recolour(img: Image.Image, kind: str) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), float)
    lum_raw = luminance(rgb)
    valid = lum_raw > 0.03
    lum = stretch(lum_raw, valid)
    if kind == "body":
        h, s, v = _hsv(rgb)
        hair = _blur(((h < 0.10) & (s > 0.6) & (v > 0.7)).astype(float))
        skin = _blur(((s < 0.55) & (v > 0.35)).astype(float)) * (1 - hair)
        cloth = np.clip(1 - hair - skin, 0, 1)
        out = (ramp(lum, BODY_SKIN) * skin[..., None] + ramp(lum, BODY_CLOTH) * cloth[..., None]
               + ramp(lum, BODY_CREST) * hair[..., None])
    elif kind in ("pearl", "abyss", "flesh"):
        out = ramp(lum, {"pearl": PEARL, "abyss": ABYSS, "flesh": FLESH}[kind])
    else:
        raise ValueError(f"unknown recolour {kind!r}")
    out[~valid] = rgb[~valid] * 0.4          # keep the (black) unused texels dark
    return out


def make_texture(src: Image.Image, geom: dict, recipe: dict) -> Image.Image:
    out = recolour(src, recipe.get("recolour", "body"))
    size = src.size
    for ov in recipe.get("overlays", []):
        keep = tri_in_region(geom["points"], geom["indices"], ov["region"]) if "region" in ov else None
        mask = uv_mask(size, geom["uv"], geom["indices"], keep)
        if not mask.any():
            continue
        args = dict(ov.get("args", {}))
        if ov["pattern"] == "spiral" and "centre" not in args:
            ys, xs = np.nonzero(mask > 0.5)
            args["centre"] = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
        rgb, strength = PATTERNS[ov["pattern"]](src.size[::-1], **args)
        out = _mix(out, rgb, _blur(mask, 0.6) * strength * ov.get("strength", 1.0))
    for eye in recipe.get("eyes", []):
        paint_eye(out, eye)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def generate_textures(spec: dict, built: dict, cache, custom_dir=None):
    """-> {texture name: (PIL image, uv guide image)}. Hand-painted PNGs in <custom_dir>/textures/<short>.png
    replace the generated ones (same size required)."""
    out = {}
    prefix = f"custom/{spec['id']}/"
    for name, geom in built["geometry"].items():
        short = name[len(prefix):]
        src = cache.get(built["sources"][name][1]).convert("RGB")
        img = make_texture(src, geom, spec["textures"][short])
        if custom_dir is not None:
            hand = Path(custom_dir) / "textures" / f"{short}.png"
            if hand.exists():
                painted = Image.open(hand).convert("RGB")
                if painted.size != img.size:
                    raise ValueError(f"{hand} is {painted.size}, expected {img.size}")
                img = painted
        out[name] = (img, uv_guide(img, geom["uv"], geom["indices"], label=short))
    return out
