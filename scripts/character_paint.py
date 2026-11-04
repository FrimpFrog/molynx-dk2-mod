"""Paint the custom character onto the Behemoth's own texture, using real Dungeon Keeper 2 art.
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

ART = Path(__file__).resolve().parent.parent / "data" / "character_art"
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
    harness = okU & np.isin(Lb, (0, 1, 2, 3)) & strap & (y > 1.72) & ((Lb != 1) | (y < 3.45)) & ~cloth
    harness = ndimage.binary_dilation(harness, iterations=1) & okU & ~cloth
    belly = torso & (y > 1.72) & (y < 2.45) & ~cloth
    refill = harness | belly
    src = ~refill & ~cloth & (g < 0.42 * r) & (r > 70) & okU
    w = ndimage.gaussian_filter(src.astype(float), 9)
    low = np.stack([ndimage.gaussian_filter(img[..., c] * src, 9) for c in range(3)], -1) / np.maximum(w, 1e-3)[..., None]
    low = np.where((w > 0.02)[..., None], low, np.array([108.0, 19.0, 14.0]))
    patch = Image.fromarray(np.clip(img[262:332, 125:195], 0, 255).astype(np.uint8))      # lumpy muscle, cut from the body's own texture
    lumps = tiled(patch, xx * 0.9, yy * 0.9)
    pl = lumps @ np.array([0.3, 0.59, 0.11])
    rel = np.clip(pl / np.maximum(ndimage.gaussian_filter(pl, 14), 1.0), 0.45, 1.9)
    fill = np.clip(low * (0.55 + 0.45 * rel[..., None]) * (0.9 + 0.25 * rel[..., None]), 0, 255)
    img = blend(img, refill, fill)

    # ------------------------------------------------------------------------ 2. the new harness: black leather, bone stitches, a bone skull
    bone_src = Image.open(ART / "mg_skull.png").convert("RGB")
    bone_tile = bone_src.crop((0, 0, 18, 64))                                # plain bone, from MegaGlest's Tower of Souls texture
    bt = tiled(bone_tile.resize((72, 256), Image.LANCZOS), xx * 1.3, yy * 1.3)
    bt = bt @ np.array([0.3, 0.59, 0.11])
    bone = np.clip(bt[..., None] / bt.mean() * np.array([212.0, 200.0, 164.0]), 0, 255)

    def seg(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
        qx, qy = ax + t * dx, ay + t * dy
        side = np.sign((px - ax) * dy - (py - ay) * dx)
        return np.hypot(px - qx, py - qy), t * np.sqrt(L2), side

    def draw_straps(region, lines, half=0.052):
        nonlocal img
        for line in lines:
            run = 0.0
            for (ax, ay), (bx, by) in zip(line[:-1], line[1:]):
                d, along, side = seg(u, y, ax, ay, bx, by)
                m = region & (d < half)
                lat = d / half
                leather = np.array([26.0, 21.0, 19.0]) * (0.8 + 0.5 * ndimage.gaussian_filter(rng.random(r.shape), 0.8))[..., None]
                leather = leather * np.where(lat > 0.78, 0.55, 1.0)[..., None]
                leather = leather + np.where((lat > 0.62) & (lat <= 0.78), 22.0, 0.0)[..., None] * np.array([1.0, 0.9, 0.8])
                img = blend(img, m, leather)
                s_ = (along + run) / 0.082
                fr = np.abs((s_ % 1.0) - 0.5)
                staple = m & (fr < 0.11) & (lat < 0.8)
                rim = m & (fr >= 0.11) & (fr < 0.19) & (lat < 0.86)
                img = blend(img, rim & ~staple, np.array([8.0, 6.0, 6.0]))
                img = blend(img, staple, np.clip(bone * (0.8 + 0.3 * (1 - lat))[..., None] if bone.ndim == 2 else bone, 0, 255))
                run += np.hypot(bx - ax, by - ay)

    Y_SK = 2.95
    shoulder = [[(-0.40, 3.28), (-0.05, Y_SK + 0.02)], [(0.40, 3.28), (0.05, Y_SK + 0.02)], [(0.0, Y_SK), (0.0, 2.40)]]
    draw_straps(front_t, shoulder)
    draw_straps(back_t, shoulder)
    # a stitched strap from the skull across the belly to the opposite hip, like the Beastmaster's
    draw_straps(front_t, [[(-0.02, Y_SK - 0.08), (0.36, 2.52)]], half=0.04)

    # belt: black leather all round, a row of chain under it, and a bone plate in front
    belt_l = torso & (y > 2.42) & (y < 2.56)
    leather = np.array([28.0, 22.0, 20.0]) * (0.8 + 0.5 * ndimage.gaussian_filter(rng.random(r.shape), 0.8))[..., None]
    img = blend(img, belt_l, leather)
    img = blend(img, belt_l & ((y > 2.54) | (y < 2.44)), np.array([60.0, 50.0, 42.0]))
    stud = belt_l & (np.abs(((np.arctan2(x - 0.05, z - 0.20) * 147.0 / 20.0) % 1.0) - 0.5) < 0.12) & (np.abs(y - 2.49) < 0.02)
    img = blend(img, stud, np.array([170.0, 160.0, 130.0]))
    ang = np.arctan2(x - 0.05, z - 0.20)
    chain_band = torso & (y > 2.33) & (y <= 2.42)
    col, alpha = chain_pattern(motif := np.asarray(_load("chain_arm.png").convert("RGBA").crop((16, 17, 58, 32)), float), ang * 147.0, (2.375 - y) * 210.0, 1.0)
    img = blend(img, chain_band & ~alpha, img * 0.5)
    img = blend(img, chain_band & alpha, np.clip(gunmetal(col, lift=1.5), 0, 255))
    def rrect(cx, cy, hw, hh, k=5.0):
        return (np.abs((u - cx) / hw) ** k + np.abs((y - cy) / hh) ** k) < 1.0

    plate = front_t & rrect(0, 2.46, 0.27, 0.165, 3.5)
    img = blend(img, plate, np.array([30.0, 24.0, 20.0]))
    inner = front_t & rrect(0, 2.46, 0.24, 0.14, 3.5)
    shade_p = 1.0 - 0.3 * np.clip((np.abs(y - 2.46) - 0.06) / 0.08, 0, 1)
    grain_l = ndimage.gaussian_filter(rng.random(r.shape), 0.9) - 0.5
    crease = ndimage.gaussian_filter(rng.random(r.shape), 3.0) - 0.5
    leather_t = np.clip(np.array([52.0, 40.0, 33.0]) * (1.0 + 0.9 * grain_l + 1.4 * crease)[..., None], 0, 255)
    img = blend(img, inner, leather_t * shade_p[..., None])
    edge_in = inner & ~(front_t & rrect(0, 2.46, 0.222, 0.122, 3.5))
    dash = edge_in & (np.abs(((u * 60.0 + y * 60.0) % 1.0) - 0.5) < 0.28)
    img = blend(img, dash, np.array([150.0, 138.0, 110.0]), 0.85)             # bone-coloured stitching round the edge
    img = blend(img, inner & (np.abs(u) < 0.012), np.array([20.0, 15.0, 12.0]))               # a seam down the middle of the plate
    for sx in (-1, 1):
        for sy in (-1, 1):
            riv = front_t & (np.hypot(u - sx * 0.205, y - (2.46 + sy * 0.10)) < 0.017)
            img = blend(img, riv, np.array([80.0, 70.0, 58.0]))

    # the horned skull from MegaGlest's Tower of Souls texture, on a dark leather medallion over the breastbone
    sk = bone_src.crop((18, 5, 76, 63)).resize((232, 232), Image.LANCZOS)
    rr = np.hypot(u / 0.26, (y - Y_SK) / 0.235)
    med = front_t & (rr < 1.0)
    img = blend(img, med, np.array([24.0, 19.0, 17.0]))
    su = (u / 0.34 + 0.5) * 232
    sv = (0.5 - (y - Y_SK) / 0.34) * 232
    skc = tiled(sk, su, sv)
    sk_in = med & (rr < 0.86) & (su > 0) & (su < 232) & (sv > 0) & (sv < 232)
    sl = np.clip(((skc @ np.array([0.3, 0.59, 0.11])) - 120.0) * 1.5 + 128.0, 0, 255) / 255.0    # the skull, embossed into dark leather
    emb = np.clip(np.array([26.0, 20.0, 17.0]) + sl[..., None] * np.array([92.0, 76.0, 60.0]), 0, 255) * (1.0 + 0.5 * grain_l)[..., None]
    img = blend(img, sk_in, emb)
    img = blend(img, med & (rr > 0.86), np.array([14.0, 11.0, 9.0]))

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
