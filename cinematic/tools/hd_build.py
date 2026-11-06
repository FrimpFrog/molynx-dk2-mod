"""Molynx, cinematic high-detail model, built from the approved game design.

Body, head and neck are Loop-subdivided from the design mesh, then sculpted by displacement:
rock plates and cracks (from the rock skin texture), a raised belt with domed studs, cloth folds,
a heavy brow ridge, sunken eye sockets, a nose bridge, a jutting lower lip. Added geometry: 3D
bottom teeth with corner fangs (placed from the painted teeth), ivory horns with growth rings,
dark glossy claws, and a rebuilt mace (iron spiked ball, wooden shaft, leather grip, iron bands)
on the axis and size of the game mace. Eyes glow (emissive). Materials are split per region
(rock skin, leather, iron studs, cloth, horn, claw).

Usage: python3 hd_build.py <pose clip> <out.glb> [body levels]
"""
import io
import pickle
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

sys.path.insert(0, '/home/claude/molynx_cinematic/tools')
sys.path.insert(0, '/home/claude/molynx')
from subdiv import weld, drop_degenerate, loop_subdivide, vertex_normals, split_by_uv, sample_bilinear, laplacian_smooth
from glb import GLB
from rock3d import rock

ROCK_CELL, ROCK_SEED, ROCK_FINE = 0.17, 3, 0.07


def HS_eye_tex():
    import head_sdf
    return head_sdf.eye_texture()


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def bake_positions(V, F, UVc, size):
    """Surface position at every texel (texture space rasterization of the mesh), gaps filled
    with the nearest baked texel so filtering never reads empty space."""
    pos = np.zeros((size, size, 3)); have = np.zeros((size, size), bool)
    U = UVc * size - 0.5
    for t in range(len(F)):
        u, v = U[t, :, 0], U[t, :, 1]
        x0, x1 = int(np.floor(u.min())), int(np.ceil(u.max())); y0, y1 = int(np.floor(v.min())), int(np.ceil(v.max()))
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = (v[1] - v[2]) * (u[0] - u[2]) + (u[2] - u[1]) * (v[0] - v[2])
        if abs(d) < 1e-12:
            continue
        a = ((v[1] - v[2]) * (xs - u[2]) + (u[2] - u[1]) * (ys - v[2])) / d
        b = ((v[2] - v[0]) * (xs - u[2]) + (u[0] - u[2]) * (ys - v[2])) / d
        c = 1 - a - b
        inside = (a >= -1e-3) & (b >= -1e-3) & (c >= -1e-3)
        if not inside.any():
            continue
        P = a[inside, None] * V[F[t, 0]] + b[inside, None] * V[F[t, 1]] + c[inside, None] * V[F[t, 2]]
        pos[ys[inside], xs[inside]] = P; have[ys[inside], xs[inside]] = True
    idx = ndi.distance_transform_edt(~have, return_distances=False, return_indices=True)
    pos = pos[idx[0], idx[1]]
    return pos, have

SRC = '/home/claude/dk2data/apply/build_source_design_v2.pkl'
DESIGN = '/home/claude/molynx/data/molynx_design/'
MIRROR = np.array([-1.0, 1.0, 1.0])      # game handedness (mace on his right shoulder)
TEX = 2048
TOPO = None
rng = np.random.default_rng(7)


# ------------------------------------------------------------------ helpers
POSE_PARTS = {}      # pose clip -> index of body, mace, head, neck in its parts (topology always the stand's)


def part_mesh(clip, k, frame=0):
    p = TOPO['parts'][k]
    V = np.asarray(clip['frames'][frame][POSE_PARTS.get(id(clip), [0, 1, 2, 3])[k]], float) * MIRROR
    F = np.asarray(p['indices'], int).reshape(-1, 3)[:, ::-1]
    UVc = np.asarray(p['uv'], float)[F]
    Vw, Fw = weld(V, F)
    Fw, UVc = drop_degenerate(Fw, UVc)
    return Vw, Fw, UVc, F


def subdivide(V, F, UVc, levels):
    for _ in range(levels):
        V, F, UVc = loop_subdivide(V, F, UVc)
    return V, F, UVc


def vertex_uv_mean(V, F, UVc, img):
    """Per welded vertex: the mean of `img` sampled at each of its corner UVs (seams agree)."""
    vals = sample_bilinear(img, UVc.reshape(-1, 2))
    acc = np.zeros((len(V),) + vals.shape[1:]); cnt = np.zeros(len(V))
    np.add.at(acc, F.reshape(-1), vals); np.add.at(cnt, F.reshape(-1), 1)
    return acc / np.maximum(cnt, 1).reshape((-1,) + (1,) * (vals.ndim - 1))


def img_bytes(arr_or_im, fmt='JPEG', q=90):
    im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(np.clip(arr_or_im, 0, 255).astype(np.uint8))
    b = io.BytesIO()
    if fmt == 'JPEG':
        im.convert('RGB').save(b, 'JPEG', quality=q, optimize=True)
    else:
        im.save(b, 'PNG', optimize=True)
    return b.getvalue(), 'image/jpeg' if fmt == 'JPEG' else 'image/png'


def upscale(im, size):
    big = im.convert('RGB').resize((size, size), Image.LANCZOS)
    return big.filter(ImageFilter.UnsharpMask(radius=3, percent=60, threshold=2))


def fbm(shape, scales=(4, 8, 16, 32, 64), seed=0):
    r = np.random.default_rng(seed); out = np.zeros(shape)
    for i, s in enumerate(scales):
        n = r.standard_normal((shape[0] // s + 2, shape[1] // s + 2))
        n = np.asarray(Image.fromarray(n.astype(np.float32)).resize((shape[1] + 2 * s, shape[0] + 2 * s), Image.BICUBIC))[s:s + shape[0], s:s + shape[1]]
        out += n * (s / scales[-1]) ** 0.7
    return out / out.std()


def hsv(a):
    a = a / 255.0
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn + 1e-9
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, d / (mx + 1e-9), 0)
    return h, s, mx


def mr_map(rough, metal):
    """glTF metallicRoughness image: G = roughness, B = metalness."""
    z = np.zeros_like(rough)
    return np.stack([z, rough, metal], -1) * 255


def height_to_normal(h, strength):
    """Tangent-space normal map (glTF/OpenGL: +X = +u, +Y = up the texture = -v)."""
    gy, gx = np.gradient(h)
    n = np.stack([-gx * strength, gy * strength, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5) * 255


# ------------------------------------------------------------------ body textures
def body_maps():
    src = Image.open(DESIGN + 'body_A2.png').convert('RGB')
    a512 = np.asarray(src).astype(float)
    h, s, v = hsv(a512)
    cloth = (s < 0.5) & (v > 0.55) & (h > 25) & (h < 75)
    metal = (s < 0.22) & (v > 0.33) & ~cloth
    leather = (h > 12) & (h < 50) & (s > 0.3) & (v > 0.15) & (v < 0.75) & ~cloth & ~metal
    cls = np.zeros(a512.shape[:2], np.uint8)           # 0 skin, 1 leather, 2 metal, 3 cloth
    cls[leather] = 1; cls[metal] = 2; cls[cloth] = 3
    cls = ndi.median_filter(cls, 3)
    cls_big = np.asarray(Image.fromarray(cls).resize((TEX, TEX), Image.NEAREST))
    soft = {k: ndi.gaussian_filter((cls_big == k).astype(float), 2.0) for k in range(4)}
    albedo = np.asarray(upscale(src, TEX)).astype(float)
    L = albedo.mean(-1) / 255.0
    grain = fbm((TEX, TEX), (2, 4, 8), seed=1)
    # rock: plates (bright cells) raised, cracks deep; fine grain on top
    plates = ndi.gaussian_filter(L, 5.0)
    plates = (plates - plates.mean()) / (plates.std() + 1e-9)
    crack = np.clip(0.18 - ndi.gaussian_filter(L, 1.5), 0, None) * 6
    rock_h = 0.5 * np.tanh(0.8 * plates) - 0.55 * crack + 0.06 * grain
    belt_h = 0.9 + 0.06 * fbm((TEX, TEX), (4, 8, 16), seed=2)
    stud = ndi.gaussian_filter((cls_big == 2).astype(float), 6.0); stud = stud / (stud.max() + 1e-9)
    cloth_h = 0.25 * fbm((TEX, TEX), (32, 64, 128), seed=3) + 0.1 * grain
    H = soft[0] * rock_h + soft[1] * belt_h + soft[2] * (belt_h + 1.6 * stud) + soft[3] * cloth_h
    # albedo: cinematic grading - deep shadows in cracks, warm saturated plates, grain variation
    skin_tint = albedo * (0.82 + 0.10 * grain[..., None]) * (1 - 0.35 * np.clip(crack, 0, 1))[..., None]
    skin_tint[..., 0] *= 1.08; skin_tint[..., 1] *= 0.92; skin_tint[..., 2] *= 0.85
    leather_c = albedo * (0.9 + 0.08 * grain[..., None])
    alb = skin_tint * (soft[0])[..., None] + leather_c * (soft[1] + soft[3])[..., None] + albedo * soft[2][..., None] * 1.1
    rough = soft[0] * (0.42 + 0.18 * np.clip(crack, 0, 1)) + soft[1] * 0.55 + soft[2] * 0.28 + soft[3] * 0.85
    metal_m = soft[2] * 0.95
    return dict(albedo=alb, H=H, rough=rough, metal=metal_m, cls=cls_big, crack=crack)


# ------------------------------------------------------------------ head / face
FACE = dict(brow_y=(318, 336), eye_y=(336, 348), teeth_box=(372, 395, 90, 160))


def head_maps():
    src = Image.open(DESIGN + 'head_A2.png').convert('RGB')
    a = np.asarray(src).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    box = np.zeros(R.shape, bool); box[300:420, 80:175] = True
    teeth = box & (R > 150) & (G > 140) & (B > 120)
    glow = box & (R > 200) & (G > 70) & (G < 200) & (B < 120) & ~teeth
    yy = np.arange(512)[:, None] * np.ones((1, 512))
    brows = glow & (yy < FACE['brow_y'][1])
    eyes = glow & (yy >= FACE['eye_y'][0]) & (yy < FACE['eye_y'][1])
    s = TEX // 512
    up = lambda m, sig: ndi.gaussian_filter(np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((TEX, TEX), Image.BILINEAR)).astype(float) / 255, sig)
    brow_f = up(ndi.binary_dilation(brows, iterations=5), 5 * s)
    brow_f /= brow_f.max()
    eye_f = up(ndi.binary_dilation(eyes, iterations=1), 1.0 * s)
    socket = up(ndi.binary_dilation(eyes, iterations=6), 4 * s); socket /= socket.max()
    t_y0, t_y1, t_x0, t_x1 = FACE['teeth_box']
    mouth = np.zeros_like(teeth); mouth[t_y0 - 2:t_y1 + 2, t_x0 + 2:t_x1 - 2] = True
    mouth_f = up(mouth, 4 * s); mouth_f /= mouth_f.max()
    lip = np.zeros_like(teeth); lip[t_y1 + 4:t_y1 + 22, t_x0 - 6:t_x1 + 6] = True       # the jutting lower lip / chin
    lip_f = up(lip, 8 * s); lip_f /= lip_f.max()
    nose = np.zeros_like(teeth); nose[336:368, 116:136] = True
    nose_f = up(nose, 6 * s); nose_f /= nose_f.max()
    albedo = np.asarray(upscale(src, TEX)).astype(float)
    L = albedo.mean(-1) / 255.0
    grain = fbm((TEX, TEX), (2, 4, 8), seed=11)
    plates = ndi.gaussian_filter(L, 5.0); plates = (plates - plates.mean()) / (plates.std() + 1e-9)
    crack = np.clip(0.18 - ndi.gaussian_filter(L, 1.5), 0, None) * 6
    rock_h = 0.3 * np.tanh(0.8 * plates) - 0.35 * crack + 0.05 * grain
    feat = 3.4 * brow_f - 2.4 * socket + 1.3 * nose_f - 1.8 * mouth_f + 1.9 * lip_f
    H = rock_h * (1 - 0.7 * np.clip(socket + mouth_f + brow_f, 0, 1)) + feat
    # colour: dark mouth interior where the painted teeth were, ember-dark sockets, glowing eyes
    alb = albedo * (0.85 + 0.08 * grain[..., None])
    alb[..., 0] *= 1.06; alb[..., 2] *= 0.85
    mouth_dark = np.clip(mouth_f * 1.3, 0, 1)[..., None]
    alb = alb * (1 - 0.85 * mouth_dark) + np.array([40, 6, 4]) * 0.85 * mouth_dark
    sock = np.clip(socket * 1.2 - eye_f, 0, 1)[..., None]
    alb = alb * (1 - 0.55 * sock)
    emis = np.zeros((TEX, TEX, 3))
    core = np.clip(eye_f * 1.8, 0, 1)
    emis += core[..., None] * np.array([255, 205, 70])
    halo = np.clip(ndi.gaussian_filter(eye_f, 6 * s) * 3.0, 0, 1)
    emis += (halo * (1 - core))[..., None] * np.array([255, 90, 10]) * 0.6
    alb = alb * (1 - core[..., None]) + core[..., None] * np.array([255, 220, 120])
    rough = 0.42 + 0.18 * np.clip(crack, 0, 1) + 0.2 * mouth_f
    return dict(albedo=alb, H=H, rough=rough, emis=emis, teeth=teeth)


# ------------------------------------------------------------------ geometry builders
def find_on_surface(V, F, UVc, uv_pts):
    """3D points (and triangle normals) on a mesh at texture coordinates uv_pts (k,2)."""
    out_p, out_n = [], []
    a, b, c = UVc[:, 0], UVc[:, 1], UVc[:, 2]
    v0, v1 = b - a, c - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01 + 1e-18
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    for p in uv_pts:
        v2 = p - a
        d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
        bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den; bu = 1 - bv - bw
        inside = (bu >= -1e-6) & (bv >= -1e-6) & (bw >= -1e-6)
        if not inside.any():
            score = np.minimum(np.minimum(bu, bv), bw); i = int(np.argmax(score))
        else:
            i = int(np.nonzero(inside)[0][0])
        P = V[F[i, 0]] * bu[i] + V[F[i, 1]] * bv[i] + V[F[i, 2]] * bw[i]
        out_p.append(P); out_n.append(fn[i])
    return np.array(out_p), np.array(out_n)


def tooth_mesh(base, up, out, width, height, depth, fang=False, seg=10):
    """A tooth: a tapered, rounded wedge (or a curved cone for a fang), base at `base`."""
    side = np.cross(up, out); side /= np.linalg.norm(side)
    rings = []
    nr = 7
    for k in range(nr):
        t = k / (nr - 1)
        if fang:
            rw = width * 0.5 * (1 - t) ** 1.2 + 0.002
            rd = depth * 0.5 * (1 - t) ** 1.2 + 0.002
        else:
            rw = width * 0.5 * (1 - 0.12 * t ** 2) * (1 - 0.35 * t ** 8)
            rd = depth * 0.5 * (1 - 0.35 * t ** 1.5) * (1 - 0.3 * t ** 8)
        centre = base + up * height * t + out * (0.25 * height * t ** 2 if fang else 0.12 * height * t)
        ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        # rounded-rectangle-ish cross section
        cx = np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** (0.6 if not fang else 1.0)
        cy = np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** (0.6 if not fang else 1.0)
        rings.append(centre + np.outer(cx * rw, side) + np.outer(cy * rd, out))
    cap = base + up * height * (1.0 if fang else 0.97) + out * (0.25 * height if fang else 0.12 * height)
    P = np.vstack(rings + [cap])
    tip = len(P) - 1
    F = []
    for k in range(nr - 1):
        for j in range(seg):
            a0, a1 = k * seg + j, k * seg + (j + 1) % seg
            b0, b1 = a0 + seg, a1 + seg
            F += [(a0, b0, a1), (a1, b0, b1)]
    for j in range(seg):
        a0, a1 = (nr - 1) * seg + j, (nr - 1) * seg + (j + 1) % seg
        F.append((a0, tip, a1))
    F = np.array(F)
    return P, F


def lathe(profile, axis_o, axis_d, seg=48, phase=0.0):
    """Surface of revolution: profile [(t along axis, radius)] -> (P, F, UV)."""
    axis_d = axis_d / np.linalg.norm(axis_d)
    a = np.cross(axis_d, [0, 1, 0]);
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(axis_d, [1, 0, 0])
    a /= np.linalg.norm(a); b = np.cross(axis_d, a)
    ang = np.linspace(0, 2 * np.pi, seg + 1) + phase
    P, UV = [], []
    tot = np.r_[0, np.cumsum(np.hypot(np.diff([p[0] for p in profile]), np.diff([p[1] for p in profile])))]
    for k, (t, r) in enumerate(profile):
        for j, an in enumerate(ang):
            P.append(axis_o + axis_d * t + r * (np.cos(an) * a + np.sin(an) * b))
            UV.append((j / seg, tot[k] / max(tot[-1], 1e-9)))
    P = np.array(P); UV = np.array(UV); F = []
    n = seg + 1
    for k in range(len(profile) - 1):
        for j in range(seg):
            a0, a1 = k * n + j, k * n + j + 1
            b0, b1 = a0 + n, a1 + n
            F += [(a0, a1, b0), (a1, b1, b0)]
    return P, np.array(F), UV


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
                  (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(level):
        V, F, _ = loop_subdivide(V, F)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, F


def mace_geometry(m_pts):
    """Rebuild the mace on the game mace's axis: (dict name -> (P, F, UV))."""
    c = m_pts.mean(0); _, _, vt = np.linalg.svd(m_pts - c); ax = vt[0]
    t = (m_pts - c) @ ax
    if t.max() < -t.min():
        ax, t = -ax, -t
    r = np.linalg.norm((m_pts - c) - np.outer(t, ax), axis=1)
    ball_sel = t < t.min() + 0.9
    ball_c = c + ax * t[ball_sel].mean()
    dist = np.linalg.norm(m_pts - ball_c, axis=1)
    tips = m_pts[ball_sel & (dist > np.percentile(dist[ball_sel], 80))]
    end = c + ax * t.max()
    shaft_len = np.linalg.norm(end - ball_c)
    core_r = 0.27
    out = {}
    # iron ball
    V, F = icosphere(4)
    V = V * core_r
    V = V * (1 + 0.012 * np.sin(V[:, 0] * 60) * np.sin(V[:, 1] * 50))[:, None]      # hammered dents
    out['ball'] = (V + ball_c, F, None)
    # spikes: one per spike direction (cluster the far vertices by direction)
    dirs = tips - ball_c; dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    picked = []
    for d in dirs[np.argsort(-np.linalg.norm(tips - ball_c, axis=1))]:
        if all(np.dot(d, p) < 0.9 for p in picked):
            picked.append(d)
    spk_len = np.linalg.norm(tips - ball_c, axis=1).max()
    SP, SF = [], []
    for d in picked:
        prof = [(core_r * 0.85, 0.085), (core_r + 0.02, 0.075), (core_r + 0.07, 0.05), (spk_len * 0.85, 0.012), (spk_len, 0.0)]
        P, F, _ = lathe(prof, ball_c, d, seg=16)
        SF.append(F + sum(len(p) for p in SP)); SP.append(P)
    out['spikes'] = (np.vstack(SP), np.vstack(SF), None)
    # wooden shaft, leather grip, iron collar and pommel
    shaft = [(0.0, 0.07), (shaft_len * 0.2, 0.066), (shaft_len * 0.6, 0.064), (shaft_len * 0.97, 0.07), (shaft_len + 0.02, 0.07)]
    out['shaft'] = lathe(shaft, ball_c, ax, seg=32)
    grip_a, grip_b = shaft_len * 0.62, shaft_len * 0.92
    gp = [(grip_a, 0.064)]
    for k in range(1, 40):
        tt = grip_a + (grip_b - grip_a) * k / 40
        gp.append((tt, 0.074 + 0.006 * np.cos(k * np.pi)))
    gp.append((grip_b, 0.07))
    out['grip'] = lathe(gp, ball_c, ax, seg=32)
    collar = [(core_r * 0.75, 0.0), (core_r * 0.78, 0.11), (core_r + 0.03, 0.11), (core_r + 0.05, 0.095), (core_r + 0.11, 0.09),
              (core_r + 0.13, 0.075), (core_r + 0.13, 0.0)]
    pom = [(shaft_len - 0.02, 0.0), (shaft_len - 0.02, 0.085), (shaft_len + 0.04, 0.095), (shaft_len + 0.1, 0.08),
           (shaft_len + 0.14, 0.04), (shaft_len + 0.15, 0.0)]
    bands = []
    for tt in (shaft_len * 0.35, shaft_len * 0.55):
        bands += [(tt - 0.03, 0.0), (tt - 0.03, 0.074), (tt + 0.03, 0.074), (tt + 0.03, 0.0)]
    PC, FC = [], []
    for prof in (collar, pom, bands[:4], bands[4:]):
        P, F, _ = lathe(prof, ball_c, ax, seg=32)
        FC.append(F + sum(len(p) for p in PC)); PC.append(P)
    out['iron'] = (np.vstack(PC), np.vstack(FC), None)
    return out


def horn_geometry(V, F, UVc, horn_mask, levels=4):
    """The horn triangles of the base mesh, subdivided, with growth rings along their length."""
    keep = np.all(horn_mask[F], axis=1)
    Fh = F[keep]; UVh = UVc[keep]
    used = np.unique(Fh); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    Vh, Fh = V[used], remap[Fh]
    Vh, Fh, UVh = subdivide(Vh, Fh, UVh, levels)
    return Vh, Fh, UVh


def spurs(V, N, mace_pts, seed=9):
    """Jagged stone shards breaking out of the body: on the shoulders, down the spine and on the
    elbows (positions picked from the body surface, kept clear of the mace). Faceted, dark rock,
    glowing faintly where they split from the skin. -> (P, N, F, colour, glow)"""
    rng = np.random.default_rng(seed)
    c = mace_pts.mean(0); _, _, vt = np.linalg.svd(mace_pts - c); ax = vt[0]
    off = V - c; dmace = np.linalg.norm(off - np.outer(off @ ax, ax), axis=1)
    clear = dmace > 0.32
    picks = []
    def choose(mask, n, spacing, score):
        idx = np.nonzero(mask & clear)[0]
        idx = idx[np.argsort(-score[idx])]
        got = []
        for i in idx:
            if all(np.linalg.norm(V[i] - V[j]) > spacing for j in got + picks):
                got.append(i)
            if len(got) == n:
                break
        picks.extend(got)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    # shoulders: tops of the shoulders, pointing up and out
    for sx in (-1, 1):
        m = (sx * x > 0.5) & (sx * x < 1.0) & (y > 2.85) & (y < 3.5) & (N[:, 1] > 0.3)
        choose(m, 2, 0.22, N[:, 1] + 0.6 * sx * N[:, 0] + rng.uniform(0, 0.15, len(V)))
    # spine: a row down the upper back
    m = (np.abs(x) < 0.18) & (y > 2.75) & (y < 3.55) & (N[:, 2] < -0.55)
    choose(m, 4, 0.2, -N[:, 2] + rng.uniform(0, 0.1, len(V)))
    # elbows: the backs of the elbows
    for sx in (-1, 1):
        m = (sx * x > 0.55) & (y > 1.9) & (y < 2.7) & (N[:, 2] < -0.4)
        choose(m, 1, 0.14, -N[:, 2] + rng.uniform(0, 0.1, len(V)))
    P_all, F_all, C_all, G_all = [], [], [], []
    for i in picks:
        n0 = N[i]
        up = n0 + np.array([0, 0.35, 0]) + rng.normal(0, 0.15, 3); up /= np.linalg.norm(up)
        a = np.cross(up, [0.3, 0.9, 0.1]); a /= np.linalg.norm(a); b = np.cross(up, a)
        h = rng.uniform(0.2, 0.36); r0 = h * rng.uniform(0.42, 0.55)
        k = rng.integers(5, 8)
        ang = np.sort(rng.uniform(0, 2 * np.pi, k)); rad = r0 * rng.uniform(0.7, 1.15, k)
        base = V[i] - up * h * 0.3
        ring0 = base + (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b) * rad[:, None]
        mid_c = base + up * h * 0.55 + (a * rng.normal(0, 0.15) + b * rng.normal(0, 0.15)) * r0
        ring1 = mid_c + (np.cos(ang + 0.3)[:, None] * a + np.sin(ang + 0.3)[:, None] * b) * (rad * rng.uniform(0.35, 0.55, k))[:, None]
        tip = base + up * h + (a * rng.normal(0, 0.25) + b * rng.normal(0, 0.25)) * r0
        tris = []
        for j in range(k):
            j2 = (j + 1) % k
            tris += [(ring0[j], ring0[j2], ring1[j]), (ring0[j2], ring1[j2], ring1[j]), (ring1[j], ring1[j2], tip)]
        T = np.array(tris)                                     # (t, 3, 3): faceted, every triangle its own vertices
        P = T.reshape(-1, 3)
        F = np.arange(len(P)).reshape(-1, 3)
        # outward winding: away from the shard's axis
        cen = T.mean(1); fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        axis_pt = base + np.outer(((cen - base) @ up), up)
        flip = np.sum(fn * (cen - axis_pt), 1) < 0
        F[flip] = F[flip][:, ::-1]
        hgt = np.clip(((P - base) @ up) / h, 0, 1)
        col = (np.array([0.03, 0.006, 0.004]) * (1 - hgt[:, None]) + np.array([0.075, 0.016, 0.01]) * hgt[:, None]) * rng.uniform(0.8, 1.2)
        glo = np.clip(1 - hgt / 0.05, 0, 1) ** 2 * 0.6
        F_all.append(F + sum(len(q) for q in P_all)); P_all.append(P); C_all.append(col); G_all.append(glo)
    P = np.vstack(P_all); F = np.vstack(F_all)
    Nf = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]); Nf /= np.linalg.norm(Nf, axis=1, keepdims=True) + 1e-12
    Nv = np.repeat(Nf, 3, axis=0)
    Nv = Nv[np.argsort(F.reshape(-1))]
    return P, Nv, F, np.vstack(C_all), np.concatenate(G_all)


# ------------------------------------------------------------------ main
def main(pose='stand', out='molynx_idle.glb', levels=4):
    from scripts.character_paint import HORNS
    global TOPO
    data, tex = pickle.load(open(SRC, 'rb'))
    clip = data[pose]
    TOPO = data['stand']            # every pose shares the stand's triangles and UVs (the game files' layout)
    order = [p['texture'] for p in clip['parts']]
    want = ['custom_character/body', 'custom_character/mace', 'custom_character/golem_head', 'custom_character/head_neck']
    POSE_PARTS[id(clip)] = [order.index(w) for w in want]
    print('pose', pose, 'parts', POSE_PARTS[id(clip)])
    g = GLB()
    stats = {}

    # ---- body (horns and claws split off; the club arm's hand stays)
    V, F, UVc, _ = part_mesh(clip, 0)
    raw = np.asarray(clip['frames'][0][POSE_PARTS[id(clip)][0]], float) * MIRROR
    # welded index -> original vertices, to find horns/claws
    key = np.round(raw / 1e-6).astype(np.int64)
    wkey = np.round(V / 1e-6).astype(np.int64)
    lookup = {tuple(k): i for i, k in enumerate(wkey)}
    horn = np.zeros(len(V), bool); claw = np.zeros(len(V), bool)
    for a, c in HORNS:
        for i in range(a, c + 1):
            horn[lookup[tuple(key[i])]] = True
    for grp in ([244, 571, 572, 573, 574], [585, 586, 587], [706, 707, 708]):
        for i in grp:
            claw[lookup[tuple(key[i])]] = True
    tri_horn = np.all(horn[F], axis=1); tri_claw = np.all(claw[F], axis=1)
    # horns
    Hv, Hf, Huv = horn_geometry(V, F, UVc, horn, levels)
    # body proper (everything but horn and claw triangles); subdivided together so seams stay closed
    Fb, UVb = F[~tri_horn & ~tri_claw], UVc[~tri_horn & ~tri_claw]
    Vb3, Fb3, UVb3 = subdivide(V, Fb, UVb, levels - 1)
    Vb, Fb, UVb = loop_subdivide(Vb3, Fb3, UVb3)
    maps = body_maps()
    # rock skin: the same 3D cracked-stone field as the head, cut into the geometry and baked
    # (crisp) into the textures through a position map of the surface
    skin_soft = ndi.gaussian_filter((maps['cls'] == 0).astype(float), 4.0)
    skin_w = vertex_uv_mean(Vb, Fb, UVb, skin_soft)
    N = vertex_normals(Vb, Fb)
    r = rock(Vb, cell=ROCK_CELL, seed=ROCK_SEED, crack_w=0.10, fine_cell=ROCK_FINE)
    h_other = vertex_uv_mean(Vb, Fb, UVb, maps['H'])
    Vb = Vb + N * (skin_w * r['height'] * 0.017 + (1 - skin_w) * h_other * 0.012)[:, None]
    Vb = laplacian_smooth(Vb, Fb, 1, 0.2)
    N = vertex_normals(Vb, Fb)
    pos, have = bake_positions(Vb3, Fb3, UVb3, TEX)
    rt = rock(pos.reshape(-1, 3), cell=ROCK_CELL, seed=ROCK_SEED, crack_w=0.10, fine_cell=ROCK_FINE)
    rh = rt['height'].reshape(TEX, TEX); rc = rt['crack'].reshape(TEX, TEX)
    from head_build import skin_colour
    lin = skin_colour(pos.reshape(-1, 3), rt).reshape(TEX, TEX, 3)
    rock_alb = to_srgb(lin) * 255
    alb = maps['albedo'] * (1 - skin_soft[..., None]) + rock_alb * skin_soft[..., None]
    rough = maps['rough'] * (1 - skin_soft) + (0.74 + 0.18 * rc) * skin_soft
    rhn = (rh - np.percentile(rh, 1)) / (np.percentile(rh, 99) - np.percentile(rh, 1) + 1e-9)
    Hb = maps['H'] / (np.ptp(maps['H']) + 1e-9)
    Htex = Hb * (1 - skin_soft) + rhn * skin_soft
    t_alb = g.image(*img_bytes(alb))
    ember = np.clip((rc - 0.82) / 0.18, 0, 1) ** 1.5 * skin_soft
    ember = ndi.gaussian_filter(ember, 0.7)
    t_em = g.image(*img_bytes(ember[..., None] * np.array([255, 92, 18])))
    t_mr = g.image(*img_bytes(mr_map(np.clip(rough, 0, 1), maps['metal'])))
    Hn = ndi.gaussian_filter(Htex, 0.8)
    t_bump = g.image(*img_bytes(np.clip(Hn, 0, 1) * 255, q=95))
    t_nrm = g.image(*img_bytes(height_to_normal(Hn * 40, 1.0), q=95))
    # split triangles by region (rock skin / leather+cloth / iron) so each gets its own shader
    cuv = UVb.mean(1)
    cls = maps['cls'][np.clip((cuv[:, 1] * TEX).astype(int), 0, TEX - 1), np.clip((cuv[:, 0] * TEX).astype(int), 0, TEX - 1)]
    regions = {'skin': cls == 0, 'leather': (cls == 1) | (cls == 3), 'iron': cls == 2}
    mats = {
        'skin': g.material('rock skin', t_alb, t_mr, t_nrm, roughness=1.0, emissive_tex=t_em, emissive=(1, 1, 1), extras={'bump': t_bump, 'bumpScale': 5.0, 'clearcoat': 0.08, 'clearcoatRoughness': 0.6, 'emissiveIntensity': 0.9}),
        'leather': g.material('leather and cloth', t_alb, t_mr, t_nrm, roughness=1.0, extras={'bump': t_bump, 'bumpScale': 2.0, 'sheen': 0.4, 'sheenColor': [0.6, 0.5, 0.35]}),
        'iron': g.material('studs', t_alb, t_mr, t_nrm, metallic=1.0, roughness=1.0, extras={'bump': t_bump, 'bumpScale': 2.0}),
    }
    for nm, sel in regions.items():
        if not sel.any():
            continue
        P, UV, idx, Nr = split_by_uv(Vb, Fb[sel], UVb[sel], N)
        g.mesh('body_' + nm, P, Nr, idx, mats[nm], UV=UV)
        stats['body_' + nm] = len(idx)

    # stone spurs breaking out of his shoulders, back and elbows
    mpts0 = np.asarray(clip['frames'][0][POSE_PARTS[id(clip)][1]], float) * MIRROR
    SP, SN, SF, SC, SG = spurs(Vb, N, mpts0)
    m_spur = g.material('stone spurs', roughness=0.8, extras={'vertexColors': True})
    g.mesh('spurs', SP, SN, SF, m_spur, color=SC)
    stats['spurs'] = len(SF)

    # horns: ivory, darker at the root, growth rings
    Nh = vertex_normals(Hv, Hf)
    root_c = []
    hv_rest = Hv
    # distance from the skull: the horn root is the end nearest the head centre
    head_c = np.array([0.0, 3.62, 0.45]) * MIRROR
    dist = np.linalg.norm(Hv - head_c, axis=1)
    tpar = (dist - dist.min()) / (np.ptp(dist) + 1e-9)
    ridge = np.sin(tpar * 60) + 0.35 * np.sin(tpar * 160 + 1.3)
    Hv = Hv + Nh * (0.0055 * ridge * (1 - tpar) ** 0.6)[:, None]
    Nh = vertex_normals(Hv, Hf)
    # linear-light: dark brown root, weathered bone, a pale tip; streaked along the ridges
    hcol = (np.array([0.05, 0.028, 0.014])[None] * (1 - tpar[:, None]) ** 1.3 + np.array([0.52, 0.42, 0.26])[None] * (1 - (1 - tpar[:, None]) ** 1.3))
    hcol = hcol * (0.72 + 0.28 * (0.5 + 0.5 * np.sin(tpar * 60))[:, None])
    m_horn = g.material('horn', roughness=0.55, extras={'vertexColors': True, 'clearcoat': 0.15, 'clearcoatRoughness': 0.5})
    g.mesh('horns', Hv, Nh, Hf, m_horn, color=hcol)
    stats['horns'] = len(Hf)

    # claws: dark glossy keratin, subdivided
    for grp_name, sel in (('claws', tri_claw),):
        Fc = F[sel]; used = np.unique(Fc); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
        Vc, Fc2, _ = subdivide(V[used], remap[Fc], UVc[sel], 2)
        g.mesh('claws', Vc, vertex_normals(Vc, Fc2), Fc2, g.material('claw', color=(0.08, 0.05, 0.04, 1), roughness=0.22, extras={'clearcoat': 1.0, 'clearcoatRoughness': 0.1}))
        stats['claws'] = len(Fc2)

    # ---- head: the sculpted cinematic head, both expressions (the viewer shows one at a time)
    from head_build import build_head
    m_hskin = g.material('head rock', roughness=0.76, extras={'vertexColors': True, 'clearcoat': 0.08, 'clearcoatRoughness': 0.6})
    m_teeth = g.material('teeth', roughness=0.38, extras={'vertexColors': True, 'clearcoat': 0.45, 'clearcoatRoughness': 0.3})
    m_eye = None
    for expr in ('grin', 'closed'):
        for m in build_head(expr):
            if m['kind'] == 'eye' and m_eye is None:
                col_t, em_t = HS_eye_tex()
                m_eye = g.material('eye', g.image(*img_bytes(col_t)), roughness=0.1, emissive_tex=g.image(*img_bytes(em_t)), emissive=(1, 1, 1),
                                   extras={'clearcoat': 1.0, 'clearcoatRoughness': 0.04, 'emissiveIntensity': 1.8})
            mat = {'skin': m_hskin, 'teeth': m_teeth, 'eye': m_eye}[m['kind']]
            g.mesh(f'head_{expr}_' + m['name'], m['P'] * MIRROR, m['N'] * MIRROR, m['F'][:, ::-1], mat, UV=m.get('UV'), color=m.get('color'))
            stats[f'head_{expr}_' + m['name']] = len(m['F'])

    # ---- mace
    mpts = np.asarray(clip['frames'][0][POSE_PARTS[id(clip)][1]], float) * MIRROR
    mg = mace_geometry(mpts)
    wood = fbm((1024, 256), (4, 8, 16, 32), seed=21)
    cols = np.linspace(0, 1, 256)[None, :] * np.ones((1024, 1))
    grain = np.sin(cols * 2 * np.pi * 14 + 2.5 * wood) * 0.5 + 0.5      # lines along the shaft
    wood_c = (np.array([130, 84, 46])[None, None] * (0.72 + 0.4 * grain[..., None])).astype(float)
    t_wood = g.image(*img_bytes(wood_c))
    leather_g = (np.array([60, 36, 22])[None, None] * (0.8 + 0.25 * (np.sin(np.linspace(0, 80, 512))[:, None] > 0.6)[..., None] * np.ones((1, 128, 1)))).astype(float)
    t_grip = g.image(*img_bytes(leather_g))
    m_iron = g.material('iron', color=(0.22, 0.21, 0.2, 1), metallic=0.9, roughness=0.55)
    m_spk = g.material('spikes', color=(0.3, 0.29, 0.27, 1), metallic=0.9, roughness=0.45)
    m_wood = g.material('wood', t_wood, roughness=0.7)
    m_grip = g.material('grip', t_grip, roughness=0.7, extras={'sheen': 0.5, 'sheenColor': [0.5, 0.35, 0.25]})
    for nm, mat in (('ball', m_iron), ('spikes', m_spk), ('iron', m_iron), ('shaft', m_wood), ('grip', m_grip)):
        P, Fm, UVm = mg[nm]
        Fm = np.asarray(Fm)
        g.mesh('mace_' + nm, P, vertex_normals(P, Fm), Fm, mat, UV=UVm)
        stats['mace_' + nm] = len(Fm)
    import head_sdf
    g.j['extras'] = {'eyeGlow': [((e + head_sdf.O) * MIRROR + np.array([0, 0, head_sdf.EYE_R * 0.9])).round(4).tolist() for e in head_sdf.EYES]}
    size = g.save(out)
    print(stats, 'total tris', sum(v if isinstance(v, int) else v[1] for v in stats.values()), 'bytes', size)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
