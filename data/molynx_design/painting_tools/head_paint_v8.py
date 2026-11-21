"""DESIGN v8 textures. Version A ('rock'): his whole body in the head's red rock (cracks over every skin texel, keeping
the muscle shading, chest, belly, claws, horns, teeth, belt, loincloth, bands). Version B ('smooth'): the head in exactly
his body skin (same grain at the same size, same colour), only the face's form and features painted on."""
import sys; sys.path.insert(0, '/home/claude/molynx_design/tools')
import numpy as np
from PIL import Image
from scipy import ndimage
from golem_src import G
from head_paint import K, FACE, _grid, blend, soft, angry_face, teeth, clean_patch

ROCK_BOX = (100, 60, 200, 150)            # plain rock cells in the Golem texture (no face), 256 px
BODY_PX_PER_UNIT = 116.5                  # his body texture (512) texels per model unit
HEAD_PX_PER_UNIT = 117.5                  # the Golem head texture (256) texels per model unit (after the 0.44 scale)

def golem_filled():
    raw = np.asarray(Image.open(G / 'texture_golem.png').convert('RGB'), float)
    empty = (raw @ np.array([0.3, 0.59, 0.11])) < 10
    near = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    return raw[near[0], near[1]]

def rock_relief(scale_px):
    """The Golem's rock as a relief map (1 = flat, < 1 in the cracks), tiled with mirroring; scale_px = output px per Golem px."""
    raw = golem_filled()
    x0, y0, x1, y1 = ROCK_BOX
    L = raw[y0:y1, x0:x1] @ np.array([0.3, 0.59, 0.11])
    rel = L / np.maximum(ndimage.gaussian_filter(L, 4.0), 1)
    return rel

def tile(rel, H, W, scale_px):
    yy, xx = np.mgrid[0:H, 0:W] / scale_px
    h, w = rel.shape
    u = np.abs((xx % (2 * w)) - w).astype(int).clip(0, w - 1)
    v = np.abs((yy % (2 * h)) - h).astype(int).clip(0, h - 1)
    return ndimage.gaussian_filter(rel[v, u], 0.5 * scale_px)

def skin_mask(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = a @ np.array([0.3, 0.59, 0.11])
    m = (g < 0.42 * r) & (r > 45) & (lum > 24)                 # red skin; not belt, cloth, horns, teeth, bands; not the black claws
    m = ndimage.binary_opening(m, iterations=1)
    return np.clip(ndimage.gaussian_filter(m.astype(float), 1.0) * 1.3, 0, 1)

def rockify(img_rgb: Image.Image, px_per_unit: float, strength: float = 0.75):
    """Cracks over the skin of any texture whose texel density is px_per_unit (512-texture texels per model unit)."""
    a = np.asarray(img_rgb.convert('RGB'), float)
    rel = rock_relief(1)
    scale = px_per_unit / (HEAD_PX_PER_UNIT * 2)              # output px per Golem px, so the cells match the head's
    R = tile(rel, a.shape[0], a.shape[1], scale * 2)
    factor = np.clip(1 + (R - 1) * strength, 0.35, 1.25)
    m = skin_mask(a)
    out = a * (1 - m[..., None]) + a * factor[..., None] * m[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def head_texture(body_tex: Image.Image, skin_box, mode: str) -> Image.Image:
    raw = golem_filled()
    src = np.asarray(Image.fromarray(raw.astype(np.uint8)).resize((256 * K, 256 * K), Image.LANCZOS), float)
    xx, yy = _grid()
    L = src @ np.array([0.3, 0.59, 0.11])
    body = np.asarray(body_tex.convert('RGB'), float)
    sm = skin_mask(body) > 0.9
    base = body[sm].mean(0)
    in_face = np.zeros(L.shape); x0, y0, x1, y1 = FACE
    in_face[int(y0 * K):int(y1 * K), int(x0 * K):int(x1 * K)] = 1
    in_face = ndimage.gaussian_filter(in_face, 3 * K)
    if mode == 'smooth':
        # his own skin at his own grain size: the body's lumpy muscle (125..245 x 262..330) at 512 px -> head canvas px
        k = (HEAD_PX_PER_UNIT * K) / BODY_PX_PER_UNIT          # canvas px per body px (about 4)
        patch = clean_patch(body[262:330, 125:245])
        ph, pw = patch.shape[:2]
        u = np.abs(((xx * K / k) % (2 * pw)) - pw).astype(int).clip(0, pw - 1)
        v = np.abs(((yy * K / k) % (2 * ph)) - ph).astype(int).clip(0, ph - 1)
        skin = patch[v, u]
        skin = skin / skin.reshape(-1, 3).mean(0) * base          # exactly his skin colour on average
        Lf = ndimage.gaussian_filter(L, 2.2 * K)                  # only the big forms of the Golem's face (brow, cheeks, nose, chin)
        m = Lf.mean()
        form = np.clip(1 + (Lf - m) / m * (0.25 + 0.30 * in_face), 0.6, 1.25)
        img = skin * form[..., None]
    else:
        Ls = ndimage.gaussian_filter(L, 0.35 * K)
        m = Ls.mean()
        form = np.clip(1 + (Ls - m) / m * (0.55 + 0.30 * in_face), 0.35, 1.18)
        g = ndimage.gaussian_filter(np.random.default_rng(3).random(L.shape), 0.8 * K)
        img = np.clip(base * form[..., None] * (0.92 + 0.16 * g)[..., None] * 1.02, 0, 255)
    u0, v0, u1, v1 = [c * 256 for c in skin_box]
    box = (xx >= u0 - 2) & (xx <= u1 + 2) & (yy >= v0 - 2) & (yy <= v1 + 2)
    img[box] = base
    # jaw definition: a shadow under the jutting lower jaw, along the chin's lower edge
    chin = (np.abs(xx - 62.5) < 20) & (yy > 202) & (yy < 207)
    img = blend(img, soft(chin, 1.2), img * 0.55, 0.8)
    img = angry_face(img)
    img = teeth(img)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)
