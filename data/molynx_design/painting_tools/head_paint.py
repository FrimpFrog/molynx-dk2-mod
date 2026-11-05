"""DESIGN v7: the Golem head's texture in Molynx's skin, with his angry face and teeth with a slight overbite.
Coordinates below are Golem texture pixels (256 x 256); the result is painted at 4x and returned at 512 x 512."""
import sys; sys.path.insert(0, '/home/claude/molynx_design/tools')
import numpy as np
from PIL import Image
from scipy import ndimage
from golem_src import G

K = 4                                                      # paint scale (1024 px)
FACE = (40, 150, 92, 206)                                  # face area: full contrast is kept here

def _grid():
    yy, xx = np.mgrid[0:256 * K, 0:256 * K].astype(float) / K
    return xx, yy

def clean_patch(p):
    """Skin only: any pale or brown fleck in a borrowed skin patch takes the nearest red skin texel (no repeating flecks)."""
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    bad = ~((g < 0.42 * r) & (r > 30))
    bad = ndimage.binary_dilation(bad, iterations=1)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return p[idx[0], idx[1]]

def blend(img, mask, rgb, a=1.0):
    m = (np.clip(mask, 0, 1) * a)[..., None]
    return img * (1 - m) + np.asarray(rgb, float) * m

def soft(mask, s=0.6):
    return np.clip(ndimage.gaussian_filter(mask.astype(float), s * K), 0, 1)

def paint_head(body_tex: Image.Image, skin_box) -> Image.Image:
    raw = np.asarray(Image.open(G / 'texture_golem.png').convert('RGB'), float)
    empty = (raw @ np.array([0.3, 0.59, 0.11])) < 10                          # the black gaps between the Golem's texture pieces
    near = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    raw = raw[near[0], near[1]]                                               # filled with the nearest real texel: no dark seams
    src = np.asarray(Image.fromarray(raw.astype(np.uint8)).resize((256 * K, 256 * K), Image.LANCZOS), float)
    xx, yy = _grid()
    L = src @ np.array([0.3, 0.59, 0.11])
    body = np.asarray(body_tex.convert('RGB'), float)
    # Molynx's skin colour and grain, measured on his own body texture (red texels only)
    r, g, b = body[..., 0], body[..., 1], body[..., 2]
    skin = (g < 0.42 * r) & (r > 60)
    base = body[skin].mean(0)
    patch = Image.fromarray(body[262:332, 125:195].astype(np.uint8)).resize((70 * K // 2, 70 * K // 2), Image.LANCZOS)
    pa = np.asarray(patch, float)
    ph, pw = pa.shape[:2]
    grain = pa[(yy * K).astype(int) % ph, (xx * K).astype(int) % pw]
    gl = grain @ np.array([0.3, 0.59, 0.11]); gl = gl / gl.mean()
    gl = np.clip(gl, 0.7, 1.2)                                               # the patch has one bright speck: it must not repeat as streaks
    # rock -> skin: the Golem's light and shade (its form), softened outside the face, times Molynx's red and grain
    in_face = np.zeros(L.shape); x0, y0, x1, y1 = FACE
    in_face[int(y0 * K):int(y1 * K), int(x0 * K):int(x1 * K)] = 1
    in_face = ndimage.gaussian_filter(in_face, 3 * K)
    Ls = ndimage.gaussian_filter(L, 0.35 * K)
    m = Ls.mean()
    contrast = 0.28 + 0.42 * in_face                         # the Golem's cracks only faintly: his body skin is smooth muscle
    form = np.clip(1 + (Ls - m) / m * contrast, 0.45, 1.12)                # no hot highlights: they read as cuts in red skin
    img = np.clip(base * form[..., None] * (0.75 + 0.25 * gl)[..., None] * 1.05, 0, 255)
    # the skirt's skin box: plain grain
    u0, v0, u1, v1 = [c * 256 for c in skin_box]
    box = (xx >= u0 - 2) & (xx <= u1 + 2) & (yy >= v0 - 2) & (yy <= v1 + 2)
    img[box] = np.clip(base * (0.8 + 0.25 * gl[box])[..., None], 0, 255)
    return img

def angry_face(img):
    xx, yy = _grid()
    # deep scowl: dark sockets that slant down to the nose (inner corners lower), heavy brows over them
    for side in (-1, 1):
        cx = 62.5 + side * 9.0                                  # eye centres (left 53.5, right 71.5)
        tilt = side * -0.32                                     # inner end (towards the nose) lower
        d_y = yy - (170.6 + tilt * (xx - cx))
        sock = (np.abs(xx - cx) < 6.2) & (np.abs(d_y) < 2.6 - 0.12 * np.abs(xx - cx))
        img = blend(img, soft(sock, 0.7), [28, 4, 4], 0.95)
        # brow: a dark-shadowed ridge above, same slant, a bit longer
        brow = (np.abs(xx - cx) < 7.5) & (d_y > -5.4) & (d_y < -2.4)
        img = blend(img, soft(brow, 0.5), [70, 10, 8], 0.75)
        ridge = (np.abs(xx - cx) < 7.0) & (d_y > -6.6) & (d_y < -5.0)
        img = blend(img, soft(ridge, 0.5), [190, 52, 38], 0.55)   # light catches the top of the brow
        # the eye: a narrow glowing slit like Molynx's (red with a hot centre)
        eye = (np.abs(xx - cx) < 3.4) & (np.abs(d_y + 0.2) < 0.75)
        img = blend(img, soft(eye, 0.35), [215, 40, 20])
        hot = (np.abs(xx - cx - side * 0.6) < 1.1) & (np.abs(d_y + 0.25) < 0.45)
        img = blend(img, soft(hot, 0.3), [255, 190, 90])
    # furrow between the brows (the frown lines over the nose)
    for side in (-1, 1):
        line = (np.abs(xx - (62.5 + side * 1.1)) < 0.45) & (yy > 161.5) & (yy < 168.5)
        img = blend(img, soft(line, 0.4), [40, 6, 5], 0.8)
    # flared nostrils and a snarl crease from the nose to the mouth corners
    for side in (-1, 1):
        nos = np.hypot(xx - (62.5 + side * 2.4), (yy - 184.0) * 1.6) < 1.2
        img = blend(img, soft(nos, 0.4), [30, 4, 4], 0.9)
        crease = (np.abs((xx - 62.5) * side - (4.0 + (yy - 182.0) * 0.9)) < 0.5) & (yy > 182) & (yy < 190)
        img = blend(img, soft(crease, 0.4), [55, 8, 6], 0.7)
    return img

def teeth(img):
    """v8: the mouth of the original Molynx: big BOTTOM teeth that rise from the lower jaw and stand slightly over the
    top lip (two big square front teeth, smaller ones beside them, a pointed tusk at each corner); upper teeth small."""
    xx, yy = _grid()
    dx_ = xx - 62.5
    top = 189.0 + 0.010 * dx_ ** 2                            # upper lip edge (corners pulled down: a snarl)
    line = 192.6 + 0.010 * dx_ ** 2                           # lower lip / gum line, where the bottom teeth grow from
    mouth = (np.abs(dx_) < 16.5) & (yy > top) & (yy < line + 0.4)
    img = blend(img, soft(mouth, 0.4), [22, 3, 3])
    ivory = np.array([238.0, 228.0, 196.0])
    # small upper teeth, mostly hidden behind the lower ones
    for dx in (-6.0, -2.0, 2.0, 6.0, -10.0, 10.0):
        t0 = 189.0 + 0.010 * dx ** 2
        up = (np.abs(xx - 62.5 - dx) < 1.3) & (yy > t0 - 0.1) & (yy < t0 + 1.6)
        img = blend(img, soft(up, 0.25), ivory * 0.85, 0.9)
    # bottom teeth: (offset, half width, height, pointed?)
    lower = [(-2.3, 2.1, 5.6, False), (2.3, 2.1, 5.6, False), (-6.3, 1.6, 4.6, False), (6.3, 1.6, 4.6, False),
             (-9.8, 1.3, 4.0, False), (9.8, 1.3, 4.0, False), (-13.0, 1.4, 6.4, True), (13.0, 1.4, 6.4, True)]
    for dx, hw, ht, pointed in lower:
        cx = 62.5 + dx
        base = 192.6 + 0.010 * dx ** 2 + 0.6                   # rooted in the lower gum
        h = base - yy                                          # height above the root
        w = hw * (np.clip(1.0 - h / ht, 0.05, 1) ** 0.8 if pointed else np.clip(1.0 - 0.12 * h / ht, 0.6, 1))
        tooth = (np.abs(xx - cx) < w) & (h > 0) & (h < ht)
        shade = np.clip(0.8 + 0.3 * h / ht, 0.8, 1.08)          # lit at the tips, darker at the gum
        img = blend(img, soft(tooth, 0.25), ivory * shade[..., None])
        edge = (np.abs(np.abs(xx - cx) - w) < 0.2) & (h > 0) & (h < ht)
        img = blend(img, soft(edge, 0.2), [118, 92, 70], 0.6)
        tip = (np.abs(xx - cx) < w) & (h > ht - 0.7) & (h < ht)
        img = blend(img, soft(tip, 0.2), [250, 244, 222], 0.5)
    # the lower lip/gum below the teeth
    lip = (np.abs(dx_) < 15.5) & (yy > line + 0.7) & (yy < line + 2.0)
    img = blend(img, soft(lip, 0.4), [130, 24, 18], 0.55)
    return img

def build_head_texture(body_tex, skin_box):
    img = paint_head(body_tex, skin_box)
    img = angry_face(img)
    img = teeth(img)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)


def strip_texture(body_tex: Image.Image, length: float, depth: float, density: float = 233.0, crease: float = 0.55):
    """The head's sides/back/underside: his own lumpy muscle skin (body texture 125..245 x 262..330), mirrored into a strip
    `length` x `depth` model units at the body's texel density x2. A soft shadow along the face's edge (v = 0) defines the jaw
    line; the back (v = 1) darkens a little where it meets the neck."""
    W, H = int(round(length * density)), int(round(depth * density))
    patch = clean_patch(np.asarray(body_tex.convert('RGB').crop((125, 262, 245, 330)), float))
    patch = np.asarray(Image.fromarray(patch.astype(np.uint8)).resize((patch.shape[1] * 2, patch.shape[0] * 2), Image.LANCZOS), float)
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    u = np.abs(((xx) % (2 * pw)) - pw + 0.5).astype(int).clip(0, pw - 1)
    v = np.abs(((yy) % (2 * ph)) - ph + 0.5).astype(int).clip(0, ph - 1)
    img = patch[v, u]
    t = yy / max(H - 1, 1)
    shade = 1 - crease * np.exp(-t / 0.06) - 0.18 * np.clip((t - 0.6) / 0.4, 0, 1)
    img = img * shade[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
