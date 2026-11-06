"""Loop subdivision of a triangle mesh with per-corner UVs, plus mesh helpers.

Positions are welded (shared) so the surface stays closed across UV seams; UVs ride on the face
corners and are split linearly (a seam stays a seam). Boundary and non-manifold edges follow the
crease rules (edge midpoints, boundary vertices smoothed only along their boundary).
"""
import numpy as np


def weld(V, F, tol=1e-6):
    """(V2, F2): duplicate positions merged."""
    key = np.round(np.asarray(V, float) / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    return np.asarray(V, float)[first], inv[np.asarray(F, int)]


def drop_degenerate(F, UVc=None):
    F = np.asarray(F, int)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return (F[ok], None if UVc is None else UVc[ok])


def loop_subdivide(V, F, UVc=None, crease_extra=None):
    """One Loop step. V (n,3) float, F (m,3) int, UVc (m,3,2) corner UVs or None.
    crease_extra: optional set of (a,b) edges (sorted) kept sharp like boundaries."""
    V = np.asarray(V, float); F = np.asarray(F, int)
    n, m = len(V), len(F)
    e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)  # face edge k = corner k -> k+1
    es = np.sort(e, 1)
    uniq, einv, ecount = np.unique(es, axis=0, return_inverse=True, return_counts=True)
    einv = einv.reshape(-1)
    ne = len(uniq)
    opp = F[np.repeat(np.arange(m), 3), np.tile([2, 0, 1], m)]   # the corner opposite each face edge
    crease = ecount[einv] != 2
    if crease_extra:
        cx = np.array([tuple(x) in crease_extra for x in map(tuple, uniq)])
        crease_e = (ecount != 2) | cx
    else:
        crease_e = ecount != 2
    # edge points
    a, b = uniq[:, 0], uniq[:, 1]
    osum = np.zeros((ne, 3)); np.add.at(osum, einv, V[opp])
    epts = np.where(crease_e[:, None], 0.5 * (V[a] + V[b]), 0.375 * (V[a] + V[b]) + 0.125 * osum)
    # vertex points
    nb_sum = np.zeros((n, 3)); val = np.zeros(n)
    np.add.at(nb_sum, a, V[b]); np.add.at(nb_sum, b, V[a]); np.add.at(val, a, 1); np.add.at(val, b, 1)
    cr_sum = np.zeros((n, 3)); cr_cnt = np.zeros(n)
    ca, cb = a[crease_e], b[crease_e]
    np.add.at(cr_sum, ca, V[cb]); np.add.at(cr_sum, cb, V[ca]); np.add.at(cr_cnt, ca, 1); np.add.at(cr_cnt, cb, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(val > 3, (1 / val) * (0.625 - (0.375 + 0.25 * np.cos(2 * np.pi / val)) ** 2), 3 / 16)
        smooth = (1 - val * beta)[:, None] * V + beta[:, None] * nb_sum
        bnd = 0.75 * V + 0.125 * cr_sum
    vpts = np.where((cr_cnt == 2)[:, None], bnd, np.where((cr_cnt > 2)[:, None] | (cr_cnt == 1)[:, None], V, smooth))
    V2 = np.vstack([vpts, epts])
    ei = (n + einv).reshape(m, 3)          # edge point index of face edge k (corner k -> k+1)
    c0, c1, c2 = F[:, 0], F[:, 1], F[:, 2]
    e01, e12, e20 = ei[:, 0], ei[:, 1], ei[:, 2]
    F2 = np.concatenate([np.stack([c0, e01, e20], 1), np.stack([c1, e12, e01], 1),
                         np.stack([c2, e20, e12], 1), np.stack([e01, e12, e20], 1)])
    if UVc is None:
        return V2, F2, None
    u0, u1, u2 = UVc[:, 0], UVc[:, 1], UVc[:, 2]
    m01, m12, m20 = (u0 + u1) / 2, (u1 + u2) / 2, (u2 + u0) / 2
    UV2 = np.concatenate([np.stack([u0, m01, m20], 1), np.stack([u1, m12, m01], 1),
                          np.stack([u2, m20, m12], 1), np.stack([m01, m12, m20], 1)])
    return V2, F2, UV2


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    L = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(L, 1e-12)


def laplacian_smooth(V, F, iters=1, lam=0.5, mask=None):
    """Umbrella smoothing; mask (n,) weights 0..1 of how much each vertex may move."""
    V = V.copy()
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    for _ in range(iters):
        s = np.zeros_like(V); c = np.zeros(len(V))
        np.add.at(s, e[:, 0], V[e[:, 1]]); np.add.at(s, e[:, 1], V[e[:, 0]])
        np.add.at(c, e[:, 0], 1); np.add.at(c, e[:, 1], 1)
        avg = s / np.maximum(c, 1)[:, None]
        w = lam if mask is None else lam * mask[:, None]
        V = V + w * (avg - V)
    return V


def split_by_uv(V, F, UVc, N=None):
    """Render-ready arrays: one vertex per distinct (position, uv) pair -> (P, UV, idx[, Nrm])."""
    corners = F.reshape(-1)
    uvq = np.round(UVc.reshape(-1, 2) * 1e6).astype(np.int64)
    key = np.concatenate([corners[:, None], uvq], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    P = V[corners[first]]
    UV = UVc.reshape(-1, 2)[first]
    out = [P, UV, inv.reshape(-1, 3)]
    if N is not None:
        out.append(N[corners[first]])
    return out


def sample_bilinear(img, uv, wrap=False):
    """img (H,W[,C]) float; uv (k,2) in 0..1 (v down, like the textures here)."""
    H, W = img.shape[:2]
    x = uv[:, 0] * W - 0.5; y = uv[:, 1] * H - 0.5
    if wrap:
        x = np.mod(x, W); y = np.mod(y, H)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = x - x0; fy = y - y0
    def g(yy, xx):
        if wrap:
            return img[np.mod(yy, H), np.mod(xx, W)]
        return img[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)]
    if img.ndim == 3:
        fx = fx[:, None]; fy = fy[:, None]
    return (g(y0, x0) * (1 - fx) * (1 - fy) + g(y0, x0 + 1) * fx * (1 - fy)
            + g(y0 + 1, x0) * (1 - fx) * fy + g(y0 + 1, x0 + 1) * fx * fy)
