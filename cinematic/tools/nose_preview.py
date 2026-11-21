"""Quick clay preview of the head sculpt (sphere-traced SDF, no Blender): front, 3/4 and side views,
old human nose vs the golem nose.  python3 nose_preview.py out.png"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/claude/molynx_cinematic/tools')
import head_sdf as HS


def look(az, el):
    a, e = np.radians(az), np.radians(el)
    fwd = -np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])      # from camera towards the face (+z is front)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    return fwd, right, up


def render(expr, az, el, res=300, half=0.2, centre=(0, -0.01, 0.15)):
    fwd, right, up = look(az, el)
    c = np.array(centre)
    u, v = np.meshgrid(np.linspace(-half, half, res), np.linspace(half, -half, res))
    o = c - fwd * 0.8 + u[..., None] * right + v[..., None] * up          # orthographic
    o = o.reshape(-1, 3); t = np.zeros(len(o)); hit = np.zeros(len(o), bool); alive = np.ones(len(o), bool)
    for _ in range(140):
        idx = np.where(alive)[0]
        if not len(idx):
            break
        dd = HS.head_sdf(o[idx] + fwd * t[idx, None], expr)
        t[idx] += dd * 0.8
        h = dd < 4e-4; hit[idx[h]] = True; alive[idx[h]] = False
        alive[idx[t[idx] > 1.6]] = False
    P = o + fwd * t[:, None]
    e = 1e-3; img = np.zeros(len(o))
    Ph = P[hit]
    g = np.stack([HS.head_sdf(Ph + np.eye(3)[i] * e, expr) - HS.head_sdf(Ph - np.eye(3)[i] * e, expr) for i in range(3)], 1)
    N = g / np.linalg.norm(g, axis=1, keepdims=True)
    L1 = np.array([-0.5, 0.75, 0.6]); L1 /= np.linalg.norm(L1)
    L2 = np.array([0.7, 0.1, 0.3]); L2 /= np.linalg.norm(L2)
    ao = np.clip(HS.head_sdf(Ph + N * 0.012, expr) / 0.012, 0, 1) ** 0.7
    img[hit] = (0.85 * np.clip(N @ L1, 0, 1) + 0.25 * np.clip(N @ L2, 0, 1) + 0.12) * (0.4 + 0.6 * ao)
    rgb = np.stack([img * 205, img * 120, img * 100], -1)
    rgb[~hit] = (20, 22, 28)
    return np.clip(rgb.reshape(res, res, 3), 0, 255).astype(np.uint8)


if __name__ == '__main__':
    out = sys.argv[1]; expr = sys.argv[2] if len(sys.argv) > 2 else 'grin'
    rows = []
    for th in ('old', 'trim'):
        HS.THROAT = th
        rows.append(np.hstack([render(expr, az, el, half=0.28, centre=(0, -0.15, 0.1)) for az, el in ((0, 8), (35, -5), (88, 0))]))
    Image.fromarray(np.vstack(rows)).save(out)
    print('saved', out)
