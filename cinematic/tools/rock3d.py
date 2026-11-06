"""Cracked stone plates as a 3D field (cellular / Voronoi), so head and body share one rock look
and texture seams don't show: every query is a 3D point.

rock(points, cell) -> dict(height, crack, tone, cell_id)
  height: stepped plates (each plate its own tilt and level, domed a little), cracks cut in
  crack : 0..1, 1 in the middle of a crack between plates
  tone  : -1..1 per-plate colour variation
"""
import numpy as np


def _hash3(ijk, seed):
    """Deterministic pseudo-random numbers per integer cell (k,3) -> (k,4) in [0,1)."""
    h = (ijk[..., 0].astype(np.int64) * 73856093) ^ (ijk[..., 1].astype(np.int64) * 19349663) ^ (ijk[..., 2].astype(np.int64) * 83492791) ^ (seed * 2654435761)
    out = []
    for k in range(4):
        h = (h * 6364136223846793005 + 1442695040888963407 + k) & 0xFFFFFFFFFFFF
        out.append(((h >> 16) & 0xFFFFFF) / float(1 << 24))
    return np.stack(out, -1)


def voronoi(P, cell, seed=0, jitter=0.85):
    """F1, F2 distances (in cell units), the nearest cell's random numbers, chunked."""
    P = np.asarray(P, float) / cell
    n = len(P)
    F1 = np.full(n, 1e9); F2 = np.full(n, 1e9); R1 = np.zeros((n, 4))
    base = np.floor(P).astype(np.int64)
    offs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    CH = 400000
    for s in range(0, n, CH):
        b = base[s:s + CH]; p = P[s:s + CH]
        f1 = np.full(len(p), 1e9); f2 = np.full(len(p), 1e9); r1 = np.zeros((len(p), 4))
        for o in offs:
            c = b + o
            rnd = _hash3(c, seed)
            feat = c + 0.5 + (rnd[:, :3] - 0.5) * jitter
            d = np.linalg.norm(p - feat, axis=1)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            r1 = np.where(closer[:, None], rnd, r1)
            f1 = np.where(closer, d, f1)
        F1[s:s + CH] = f1; F2[s:s + CH] = f2; R1[s:s + CH] = r1
    return F1, F2, R1


def rock(P, cell=0.16, seed=3, crack_w=0.10, fine_cell=None):
    P = np.asarray(P, float)
    F1, F2, R = voronoi(P, cell, seed)
    edge = F2 - F1                                   # 0 on a crack
    crack = np.clip(1 - edge / crack_w, 0, 1) ** 1.6
    level = (R[:, 0] - 0.5)                          # each plate its own height
    dome = -0.6 * F1 ** 2                            # plates bulge a little in the middle
    tilt = (R[:, 1] - 0.5) * 0.6 * (P[:, 1] / cell - np.floor(P[:, 1] / cell))
    height = 0.6 * level + dome + 0.25 * tilt - 1.6 * crack
    # chipped, pitted stone on every plate
    height = height + 0.22 * fbm3(P, cell * 0.35, 4, seed + 5) * (1 - crack)
    tone = (R[:, 2] - 0.5) * 2
    if fine_cell:
        f1, f2, r = voronoi(P, fine_cell, seed + 11)
        fc = np.clip(1 - (f2 - f1) / (crack_w * 1.2), 0, 1) ** 2
        height = height - 0.45 * fc + 0.12 * (r[:, 0] - 0.5)
        crack = np.maximum(crack, 0.55 * fc)
    return dict(height=height, crack=crack, tone=tone, cell=R[:, 3])


def value_noise(P, scale, seed=0):
    """Smooth 3D value noise in -1..1 (trilinear over a hashed lattice)."""
    q = np.asarray(P, float) / scale
    b = np.floor(q).astype(np.int64); f = q - b
    f = f * f * (3 - 2 * f)
    out = np.zeros(len(q))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out += w * (_hash3(b + np.array([dx, dy, dz]), seed)[:, 0] * 2 - 1)
    return out


def fbm3(P, scale, octaves=4, seed=0):
    tot = 0; amp = 1; norm = 0
    for o in range(octaves):
        tot = tot + amp * value_noise(P, scale / 2 ** o, seed + o); norm += amp; amp *= 0.5
    return tot / norm
