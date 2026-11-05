"""DESIGN COPY (v6: harness removed, Battlemage belt). Paint the custom character onto the Behemoth's own texture, using real Dungeon Keeper 2 art.
LICENCE: GPL-3.0 (base model: MegaGlest, CC BY-SA 3.0; art crops: Dungeon Keeper 2 textures, kept in data/character_art/).

Nothing here adds geometry. Every change is painted onto the model's existing polygons:
  * dark riveted armour on the chest, back and shoulders      <- the Horned Reaper's pauldron and gorget textures
  * a steel collar (shackle) around the neck with a padlock   <- the Reaper's gorget plate and DK2's door-chain padlock
  * two chains hanging down the chest                         <- the DK2 door chain, recoloured gunmetal
  * the studded leather arm bands again on both ankles        <- cut from the Behemoth's own texture
The painter works in 3D: it first rasterises the UV layout into a map of "which 3D point is this texel", so a band, a chain or a
plate can be described in model space (heights, distances) and lands on the right polygons wherever their UV islands are.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.behemoth_rig import CLUB_VERTICES, behemoth_joints, behemoth_labels, behemoth_rest
from scripts.character_texture import material_cuff
from scripts.character_paint import ART  # the art crops stay in the repo (read only)


SIZE = 512
HORNS = [(277, 296), (463, 482)]          # the cream horn strip (vertex ranges of the body mesh)


# ---------------------------------------------------------------------------------------------------------------- UV -> 3D map
def position_map(rest, uv, tris, labels, size=SIZE):
    """Rasterise the UV layout. -> dict of (size,size) arrays: P (3D point), N (face normal), T (triangle number, -1 = empty), L (label)."""
    P = np.zeros((size, size, 3))
    N = np.zeros((size, size, 3))
    T = np.full((size, size), -1, int)
    L = np.zeros((size, size), int)
    px = uv * size
    for k, t in enumerate(tris):
        a, b, c = px[t]
        x0, x1 = int(np.floor(min(a[0], b[0], c[0]))), int(np.ceil(max(a[0], b[0], c[0])))
        y0, y1 = int(np.floor(min(a[1], b[1], c[1]))), int(np.ceil(max(a[1], b[1], c[1])))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size), min(y1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-9:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / d
        w1 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / d
        w2 = 1 - w0 - w1
        m = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
        if not m.any():
            continue
        pa, pb, pc = rest[t]
        pt = w0[..., None] * pa + w1[..., None] * pb + w2[..., None] * pc
        n = np.cross(pb - pa, pc - pa)
        n = n / max(np.linalg.norm(n), 1e-9)
        sl = (slice(y0, y1), slice(x0, x1))
        P[sl][m] = pt[m]
        N[sl][m] = n
        T[sl][m] = k
        L[sl][m] = np.bincount(labels[t]).argmax()
    return {"P": P, "N": N, "T": T, "L": L}


def pad_map(mp, px=3):
    """Let every map entry spill `px` texels into the empty margin around its island, so painted edges survive mip-mapping."""
    empty = mp["T"] < 0
    idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    dist = ndimage.distance_transform_edt(empty)
    out = {k: v[idx[0], idx[1]] for k, v in mp.items()}
    out["valid"] = ~empty
    out["pad"] = empty & (dist <= px)
    return out


# ---------------------------------------------------------------------------------------------------------------------- art
def _load(name):
    return Image.open(ART / name)


def gunmetal(rgb, lift=1.0, tint=(0.97, 1.0, 1.08)):
    """Any picture -> grey steel keeping its light and dark detail (the rusty brown chain becomes gunmetal)."""
    a = np.asarray(rgb, float)
    lum = a @ np.array([0.3, 0.59, 0.11])
    lum = np.clip((lum - 20) * 1.35 * lift + 28, 0, 255)
    return np.clip(lum[..., None] * np.array(tint), 0, 255)


def tiled(img, u, v, scale=1.0):
    """Sample `img` (PIL, RGB) at float pixel coordinates (u, v) arrays with mirrored tiling (no visible seams)."""
    a = np.asarray(img.convert("RGB"), float)
    h, w = a.shape[:2]
    uu = np.abs(((u * scale) % (2 * w)) - w) - 0.5
    vv = np.abs(((v * scale) % (2 * h)) - h) - 0.5
    return a[np.clip(vv, 0, h - 1).astype(int), np.clip(uu, 0, w - 1).astype(int)]


def blend(img, mask, rgb, alpha=1.0):
    a = (mask * alpha)[..., None] if np.ndim(alpha) == 0 else (mask * alpha)[..., None]
    return img * (1 - a) + rgb * a


# ------------------------------------------------------------------------------------------------------------------- painter
def zones(mp, tris, lab, joints, bad_tri):
    """Boolean texel masks for every painted feature, judged from one 3D map. Every rule is the same for the left and right side
    (the Behemoth's arms and legs share one UV island), so a texel is painted only if all triangles that use it agree."""
    P, Lb, valid = mp["P"], mp["L"], mp["valid"]
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    ok = valid & (~bad_tri)[np.maximum(mp["T"], 0)]
    n = mp["N"].copy()
    out = np.stack([x - 0.05, np.zeros_like(x), z - 0.25], -1)              # horizontal direction away from the spine
    n[(n * out).sum(-1) < 0] *= -1                                           # triangle winding is not consistent: face outward
    front = (n[..., 2] > -0.05) & (z > 0.3)
    back = (n[..., 2] < 0.05) & (z < 0.15)
    front2 = (n[..., 2] > -0.1) & (z > 0.02)                                 # the front, a bit further round the shoulders
    torso = ok & (Lb == 0)
    collar = ok & np.isin(Lb, (0, 1)) & (np.hypot((x - 0.05) / 0.52, (z - 0.42) / 0.62) < 1.0) & (y > 3.10) & (y < 3.42)
    legs = ok & np.isin(Lb, (4, 5, 8, 9)) & (y > 0.40) & (y < 0.66)
    return {"ok": ok, "front": front, "front2": front2, "back": back, "torso": torso, "collar": collar, "legs": legs}


def remove_arm_bands(body: Image.Image) -> Image.Image:
    """The studded leather arm bands come off the arms: their texels are painted over with the surrounding skin."""
    a = np.asarray(body.convert("RGB"), float).copy()
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    zone = np.zeros(r.shape, bool)
    zone[138:204, 94:140] = True                                           # the strip of the band in the 256 texture
    skin = (g < 0.42 * r) & (r > 45)
    band = zone & ~skin
    band = ndimage.binary_dilation(band, iterations=2) & zone
    valid = ~band
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    fill = a[idx[0], idx[1]]
    for c in range(3):
        fill[..., c] = ndimage.gaussian_filter(fill[..., c], 1.5)
    a[band] = fill[band]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


# ------------------------------------------------------------------------------------------- DESIGN v6: the Battlemage belt
BM_TEX = Path(__file__).resolve().parent.parent.parent / "molynx" / "megaglest_battlemage" / "models" / "texture_battle_mage.png"
BELT_HEIGHT = 0.18        # model units (his old belt was 0.14: "slightly bigger")
BELT_TOP = 2.49 + BELT_HEIGHT / 2
WAIST_RADIUS = 0.66       # median distance of the waist vertices from the spine axis (x 0.05, z 0.20)
SCALE = 4                 # the belt art is drawn at 4x the Battlemage's texels
BAND_PX = 9 * SCALE       # the band is 9 texels tall in the Battlemage texture


def battlemage_belt():
    """One strip of belt in RGBA: the Battlemage's back strap (4 studs) repeated, with his front piece (2 studs each side and the
    round silver buckle) in the middle. Rows outside the band are transparent except the buckle, which stands proud of it."""
    global BAND_ROWS, BUCKLE_X
    src = Image.open(BM_TEX).convert("RGBA")
    back = src.crop((17, 130, 60, 139)).convert("RGB")                       # 43 x 9: the plain strap of his back
    front = src.crop((133, 115, 193, 133)).convert("RGB")                    # 60 x 18: front strap with the buckle (band rows 4..12)
    back = np.asarray(back.resize((back.width * SCALE, back.height * SCALE), Image.LANCZOS), float)
    front = np.asarray(front.resize((front.width * SCALE, front.height * SCALE), Image.LANCZOS), float)
    pad = 6 * SCALE                                                          # rows above and below the band (room for the buckle)
    H = BAND_PX + 2 * pad
    W = back.shape[1] * 5                                                    # about his waist's length of strap
    out = np.zeros((H, W, 4))
    band = np.tile(back, (1, 5, 1))[:BAND_PX].copy()
    band[:SCALE] *= 0.78                                                     # tone the top highlight down to the leather
    out[pad:pad + BAND_PX, :, :3] = band
    out[pad:pad + BAND_PX, :, 3] = 255
    # the front piece: band rows (src rows 119..127 = crop rows 4..12) plus the buckle disc (centre 163.5, 124; radius 7.5)
    fy, fx = np.mgrid[0:front.shape[0], 0:front.shape[1]] / SCALE
    band_rows = (fy >= 4.0) & (fy < 12.6)                                  # from src row 119: no green tunic above the strap
    buckle = np.hypot(fx - 30.0, fy - 9.6) < 6.6                           # the disc only, not the shadow above it
    keep = band_rows | buckle
    cx = W // 2
    x0 = cx - front.shape[1] // 2
    y0 = pad - int(round(3.6 * SCALE))
    for c in range(3):
        sub = out[y0:y0 + front.shape[0], x0:x0 + front.shape[1], c]
        sub[keep] = front[..., c][keep]
    sub = out[y0:y0 + front.shape[0], x0:x0 + front.shape[1], 3]
    sub[keep] = 255
    BAND_ROWS = (pad, pad + BAND_PX)
    BUCKLE_X = cx
    return out



SKIN = np.array([136.0, 24.0, 18.0])


def chain_pattern(marr, along, across, scale):
    """Colour and alpha of the chain motif at texel coordinates (along the chain, across it), in motif pixels."""
    mh, mw = marr.shape[:2]
    px = marr[np.clip((across * scale + mh / 2).astype(int), 0, mh - 1), (along * scale).astype(int) % mw]
    inside = (across * scale + mh / 2 >= 0) & (across * scale + mh / 2 < mh)
    return px[..., :3], (px[..., 3] > 90) & inside


def paint_teeth(img, unit_dir):
    """The two big teeth (and the small ones) of the Behemoth's face: bone white. Their texels are found from the original texture,
    where they are the pale spots in the mouth."""
    raw = np.asarray(Image.open(Path(unit_dir) / "models" / "texture_daemon_giant.png").convert("RGB"), float)
    lum = raw @ np.array([0.3, 0.59, 0.11])
    zone = np.zeros(lum.shape, bool)
    zone[60:68, 195:212] = True
    small = zone & (lum > 105)                                             # every pale spot in the mouth
    big = np.zeros(lum.shape, bool)
    big[61:66, 198:201] = True                                             # the two big front teeth, widened and lengthened a little
    big[61:66, 202:205] = True
    out = img.copy()
    for mask, strength in ((small, 0.8), (big, 1.0)):
        m = np.kron(mask, np.ones((2, 2))).astype(bool)
        grad = np.clip(1.05 - 0.22 * ((np.arange(SIZE)[:, None] - 122.0) / 12.0), 0.7, 1.05) * np.ones((SIZE, SIZE))
        ivory = np.array([238.0, 230.0, 200.0]) * grad[..., None]
        out = blend(out, m, ivory, strength)
    return out


def paint_character(unit_dir, base_body: Image.Image, uv, tris, debug=False) -> Image.Image:
    """base_body: the 256x256 red body texture (tattoos already removed). uv: viewer-convention (v top-down) UVs of the 728 vertices."""
    unit_dir = Path(unit_dir)
    rest, _, _ = behemoth_rest(unit_dir)
    lab = behemoth_labels(rest)
    joints = behemoth_joints(rest, lab)
    hand_v = (lab == 1) & (rest[:, 2] > 0.8)              # claws of the raised club hand are labelled "head" by height
    bad_tri = np.zeros(len(tris), bool)
    for a, b in HORNS:
        bad_tri |= np.all((tris >= a) & (tris <= b), axis=1)
    bad_tri |= np.all(tris < CLUB_VERTICES, axis=1)
    bad_tri |= hand_v[tris].any(axis=1)
    mp = pad_map(position_map(rest, uv, tris, lab))
    mp2 = pad_map(position_map(rest, uv, tris[::-1], lab))                  # same layout, other triangle order: shows shared texels
    mp2["T"] = np.where(mp2["T"] >= 0, len(tris) - 1 - mp2["T"], -1)
    z1, z2 = zones(mp, tris, lab, joints, bad_tri), zones(mp2, tris, lab, joints, bad_tri)
    Z = {k: (z1[k] & z2[k]) for k in z1}
    P, Lb = mp["P"], mp["L"]
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    ok, front, torso = Z["ok"], Z["front"], Z["torso"]
    img = np.asarray(remove_arm_bands(base_body).resize((SIZE, SIZE), Image.LANCZOS), float).copy()
    rng = np.random.default_rng(7)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    lum = 0.30 * r + 0.59 * g + 0.11 * b
    strap = (g > 0.45 * r) & (b < 1.0 * g) & (r < 175) & (r > 24) & (lum < 150)           # the Behemoth's own leather harness
    okU = z1["ok"] | z2["ok"]
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    u = x - 0.05
    front_t, back_t = Z["front2"] & torso, Z["back"] & torso
    front_any = Z["front"] & Z["ok"]

    cloth = ndimage.binary_closing((lum > 140) & (g > 0.75 * r), iterations=4)                             # the pale loincloth
    holes, nh = ndimage.label(ndimage.binary_fill_holes(cloth) & ~cloth)
    small = np.isin(holes, [i for i in range(1, nh + 1) if (holes == i).sum() < 250])                      # its darker stains inside, not the big gaps
    cloth = ndimage.binary_dilation(cloth | small, iterations=2)

    # ------------------------------------------------------------- 1. the original harness comes off: its texels get the body's own skin texture
    # DESIGN v6: the Behemoth's own harness must vanish completely now (nothing is drawn over it any more), so the mask takes the
    # whole strap with its dark edges, and the fill is the surrounding skin's colour plus skin grain, feathered at the edge.
    body_lab = okU & np.isin(Lb, (0, 1, 2, 3)) & (y > 1.72) & ((Lb != 1) | (y < 3.45)) & ~cloth
    core = body_lab & strap
    core = ndimage.binary_opening(core, iterations=1) | (core & ndimage.binary_erosion(core, iterations=1))
    core = ndimage.binary_closing(core, iterations=3) & body_lab
    harness = ndimage.binary_dilation(core, iterations=5) & body_lab
    belly = torso & (y > 1.72) & (y < 2.45) & ~cloth
    refill = harness | belly
    skin_src = okU & ~refill & ~cloth & (g < 0.42 * r) & (r > 45)
    idx_n = ndimage.distance_transform_edt(~skin_src, return_distances=False, return_indices=True)
    padded = img[idx_n[0], idx_n[1]]                                          # every texel takes its nearest real skin colour
    low = np.stack([ndimage.gaussian_filter(padded[..., c], 5.0) for c in range(3)], -1)
    patch = Image.fromarray(np.clip(img[262:332, 125:195], 0, 255).astype(np.uint8))      # lumpy muscle, cut from the body's own texture
    lumps = tiled(patch, xx * 0.9, yy * 0.9)
    detail = lumps - np.stack([ndimage.gaussian_filter(lumps[..., c], 5.0) for c in range(3)], -1)
    skin_hp = img - np.stack([ndimage.gaussian_filter(img[..., c], 5.0) for c in range(3)], -1)
    amp = np.sqrt(ndimage.gaussian_filter((skin_hp ** 2).mean(-1) * skin_src, 12) / np.maximum(ndimage.gaussian_filter(skin_src.astype(float), 12), 1e-3))
    amp_l = np.sqrt((detail ** 2).mean())
    fill = np.clip(low + detail * (amp / max(amp_l, 1e-3))[..., None], 0, 255)
    soft = np.clip(ndimage.gaussian_filter(refill.astype(float), 1.5) * 1.6, 0, 1)
    soft = np.maximum(soft, ndimage.binary_erosion(refill, iterations=1).astype(float)) * okU
    img = img * (1 - soft[..., None]) + fill * soft[..., None]

    # ------------------------------------------------------------------------ 2. DESIGN v6: no harness at all (no straps, no skull medallion,
    # no chest chain, no bone belt plate). Only the plain bone tile is kept: the loincloth below still uses it.
    bone_src = Image.open(ART / "mg_skull.png").convert("RGB")
    bone_tile = bone_src.crop((0, 0, 18, 64))                                # plain bone, from MegaGlest's Tower of Souls texture
    bt = tiled(bone_tile.resize((72, 256), Image.LANCZOS), xx * 1.3, yy * 1.3)
    bt = bt @ np.array([0.3, 0.59, 0.11])
    bone = np.clip(bt[..., None] / bt.mean() * np.array([212.0, 200.0, 164.0]), 0, 255)

    # the belt: MegaGlest's Battlemage belt (brown leather, steel studs, round silver buckle), wrapped round the waist
    belt_img = battlemage_belt()                                              # (H, W, 4) float, band rows BAND_ROWS, buckle at BUCKLE_X
    H, W = belt_img.shape[:2]
    yc = BELT_TOP - BELT_HEIGHT / 2
    wv = (lab == 0) & (np.abs(rest[:, 1] - yc) < 0.15)                     # torso vertices at belt height (his texture is mirrored
    wcx = 0.05                                                                # left/right, so the centre is the spine plane x 0.05)
    wcz = (rest[wv, 2].min() + rest[wv, 2].max()) / 2
    wax = np.abs(rest[wv, 0] - wcx).max()
    waz = (rest[wv, 2].max() - rest[wv, 2].min()) / 2
    ang = np.arctan2((x - wcx) / wax, (z - wcz) / waz)                        # 0 = straight ahead (+z is his front)
    px_per_unit = BAND_PX / BELT_HEIGHT
    bu = BUCKLE_X + ang * (wax + waz) / 2 * px_per_unit                       # belt pixels along the waist, buckle at the front
    print(f"waist centre x {wcx:.3f} z {wcz:.3f}, half-widths {wax:.3f} x {waz:.3f}")
    bv = BAND_ROWS[0] + (BELT_TOP - y) * px_per_unit                          # belt pixels downwards from the band's top edge
    inside = torso & (bv >= 0) & (bv < H - 0.5)
    bu_w = np.mod(bu, W)
    col = belt_img[np.clip(bv.astype(int), 0, H - 1), np.clip(bu_w.astype(int), 0, W - 1)]
    a_ = np.where(inside, col[..., 3] / 255.0, 0.0)
    img = img * (1 - a_[..., None]) + col[..., :3] * a_[..., None]
    belt_l = inside & (a_ > 0.5)

    # the loincloth becomes bone, like the Beastmaster's pelvis plate
    cloth_core = okU & (lum > 120) & (g > 0.7 * r) & (y < 2.2) & (y > 1.2) & ~(np.isin(Lb, (8, 9)))
    cm = ndimage.gaussian_filter(cloth_core.astype(float), 0.8)
    img = blend(img, cm > 0.5, np.clip(0.45 * img + 0.55 * bone, 0, 255))
    for sx in (-1, 1):
        for cy, (rx, ry) in ((1.86, (0.075, 0.10)),):
            hole = cloth_core & front_any & (((u - sx * 0.17) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1)
            img = blend(img, hole, np.array([30.0, 22.0, 18.0]))

    # ------------------------------------------------------------------------------------------------------ ankle bands
    cuff = material_cuff(Image.fromarray(np.asarray(base_body.convert("RGB"), np.uint8)))
    cuff_big = cuff.resize((cuff.width * 2, cuff.height * 2), Image.LANCZOS)
    from scripts.character_parts import slice_at
    centres = {}
    for ls in ((4, 8), (5, 9)):
        t_leg = np.array([np.isin(lab[t], ls).all() for t in tris])
        pts = slice_at(rest, tris[t_leg], 0.52)
        c = ((pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 2].min() + pts[:, 2].max()) / 2)
        for l in ls:
            centres[l] = c
    cxs = np.vectorize(lambda l: centres.get(l, (0, 0))[0])(Lb)
    czs = np.vectorize(lambda l: centres.get(l, (0, 0))[1])(Lb)
    legs = Z["legs"]
    angl = np.arctan2(z - czs, x - cxs)
    uu = (angl / (2 * np.pi) + 0.5) * 2.0 * cuff_big.width
    vv = (0.66 - y) / 0.26 * cuff_big.height
    img = blend(img, legs, tiled(cuff_big, uu, vv))
    rim = legs & (((y > 0.40) & (y < 0.415)) | ((y > 0.645) & (y < 0.66)))
    img = blend(img, rim, np.array([34, 18, 12]), 0.8)

    img = paint_teeth(img, unit_dir)
    if debug:
        dbg = np.asarray(base_body.convert("RGB").resize((SIZE, SIZE)), float).copy()
        for mk, col_ in ((belly, (0, 255, 0)), (harness, (0, 0, 255)), (belt_l, (255, 255, 0))):
            dbg = blend(dbg, mk, np.array(col_), 0.8)
        return Image.fromarray(np.clip(dbg, 0, 255).astype(np.uint8), "RGB")
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
