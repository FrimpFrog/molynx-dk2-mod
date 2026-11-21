"""Molynx's cinematic body sculpt.

The game body (smooth, Loop level 3) becomes a signed distance field, so the silhouette and pose
stay exactly DK2's. On top of it:
  - anatomy: pecs, traps, deltoids, shoulder blades, spine groove, biceps/triceps, elbows, forearm
    muscles, quads, kneecaps, calves, ankle bones
  - the mitten hands are carved off and replaced: a real fist wrapped round the mace grip (right)
    and an open clawed hand (left)
  - the stump toes are carved off and replaced with three clawed toes on a forefoot pad
  - belt, straps and loincloth stand proud of the skin (real thickness, ragged hem from the paint)
Polygonized narrow-band (marching cubes), then UVs/material class transferred from the game mesh.
Coordinates: parts space (game, x mirrored), y up, +z forward.
"""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

sys.path.insert(0, '/home/claude/molynx_cinematic/tools')
from head_sdf import smin, smax, ellipsoid, sphere, capsule
from subdiv import loop_subdivide, vertex_normals, laplacian_smooth

P = '/home/claude/molynx_blender/parts/'
VOX = 0.0075
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


def nrm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- the game body as a distance field
b = np.load(P + 'body.npz')
V0, F0, UV0, CLS0 = b['V'], b['F'], b['UV'], b['cls']


def drop_small_islands(V, F, cls, min_area=0.01):
    """tiny painted leather/cloth specks on the skin become skin (only real straps, belt, bands, loincloth stay)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cls = cls.copy(); m = np.where(cls != 0)[0]
    e = np.sort(np.stack([F[m][:, [0, 1]], F[m][:, [1, 2]], F[m][:, [2, 0]]], 1).reshape(-1, 2), 1); fid = np.repeat(np.arange(len(m)), 3)
    key = e[:, 0] * 1000003 + e[:, 1]; o = np.argsort(key); k = key[o]; same = np.where(k[1:] == k[:-1])[0]
    A = coo_matrix((np.ones(len(same)), (fid[o][same], fid[o][same + 1])), shape=(len(m), len(m)))
    _, lab = connected_components(A, directed=False)
    T = V[F[m]]; area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    small = np.bincount(lab, area)[lab] < min_area
    cls[m[small]] = 0
    return cls


CLS0 = drop_small_islands(V0, F0, CLS0)


def capped(V, F):
    """close the holes left where the horn and claw triangles were taken out (fan from each loop's centre)"""
    he = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2)
    key = {tuple(e) for e in he}
    bnd = [tuple(e) for e in he if (e[1], e[0]) not in key]
    nxt = {a: b2 for a, b2 in bnd}
    V = list(V); newF = []
    seen = set()
    for a, _ in bnd:
        if a in seen:
            continue
        loop = [a]; seen.add(a); c = nxt[a]
        while c != a:
            loop.append(c); seen.add(c); c = nxt[c]
        ci = len(V); V.append(np.mean([V[i] for i in loop], 0))
        for i, j in zip(loop, loop[1:] + loop[:1]):
            newF.append([j, i, ci])
    return np.array(V), np.vstack([F, np.array(newF, int)]) if newF else F


Vc_, Fc_ = capped(V0, F0)
Vs, Fs, _ = loop_subdivide(Vc_, Fc_)
Ns = vertex_normals(Vs, Fs)
tri = Vs[Fs]
fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
# per-subdivided-face class: Loop children come in 4 blocks of m (child k of face f is k*m+f)
cls_s = np.tile(np.concatenate([CLS0, np.zeros(len(Fc_) - len(F0), int)]), 4) if len(Fs) == 4 * len(Fc_) else None
assert cls_s is not None, 'unexpected loop_subdivide face order'
raised = np.isin(cls_s, [1, 2, 3]).astype(float)
v_raised = np.zeros(len(Vs)); np.maximum.at(v_raised, Fs.ravel(), np.repeat(raised, 3))
SP = np.vstack([Vs, tri.mean(1)]); SN = np.vstack([Ns, fn]); SR = np.concatenate([v_raised, raised])
TREE = cKDTree(SP)
log('samples', len(SP))


# ---- inside/outside by winding number along vertical (z) lines: robust with overlapping shells
TV = Vs[Fs]
TN = np.cross(TV[:, 1] - TV[:, 0], TV[:, 2] - TV[:, 0])
TS = -np.sign(TN[:, 2])                              # +1 entering a shell when travelling +z
BK = 0.02
_bx0 = np.floor(TV[:, :, 0].min(1) / BK).astype(int); _bx1 = np.floor(TV[:, :, 0].max(1) / BK).astype(int)
_by0 = np.floor(TV[:, :, 1].min(1) / BK).astype(int); _by1 = np.floor(TV[:, :, 1].max(1) / BK).astype(int)
BUCKET = {}
for t in range(len(TV)):
    for i in range(_bx0[t], _bx1[t] + 1):
        for j in range(_by0[t], _by1[t] + 1):
            BUCKET.setdefault((i, j), []).append(t)
BUCKET = {k: np.array(v) for k, v in BUCKET.items()}


def column_hits(x, y, cand):
    """z and sign of every crossing of the vertical line (x, y) with triangles `cand`"""
    T = TV[cand]
    a, b2, c = T[:, 0, :2], T[:, 1, :2], T[:, 2, :2]
    q = np.array([x, y])
    def cr(u, v, w):
        return (v[:, 0] - u[:, 0]) * (w[1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[0] - u[:, 0])
    w0 = cr(b2, c, q); w1 = cr(c, a, q); w2 = cr(a, b2, q)
    ins = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
    tot = w0 + w1 + w2
    ins &= np.abs(tot) > 1e-14
    z = (w0 * T[:, 0, 2] + w1 * T[:, 1, 2] + w2 * T[:, 2, 2])[ins] / tot[ins]
    return z, TS[cand][ins]


def winding_sign(p):
    p = p.reshape(-1, 3); out = np.ones(len(p))
    for n, (x, y, z) in enumerate(p):
        cand = BUCKET.get((int(np.floor(x / BK)), int(np.floor(y / BK))))
        if cand is None:
            continue
        zz, ss = column_hits(x, y, cand)
        if ss[zz < z].sum() > 0:
            out[n] = -1.0
    return out


def winding_grid(lo, n):
    """inside (-1) / outside (+1) for every voxel of the grid, one vertical line per (x, y) column"""
    D = np.zeros((n[0], n[1], n[2] + 1), np.int16)
    xs = lo[0] + np.arange(n[0]) * VOX; ys = lo[1] + np.arange(n[1]) * VOX
    for i, x in enumerate(xs):
        bi = int(np.floor(x / BK))
        for j, y in enumerate(ys):
            cand = BUCKET.get((bi, int(np.floor(y / BK))))
            if cand is None:
                continue
            zz, ss = column_hits(x, y, cand)
            if len(zz):
                kz = np.clip(np.ceil((zz - lo[2]) / VOX).astype(int), 0, n[2])
                np.add.at(D[i, j], kz, ss.astype(np.int16))
    W = np.cumsum(D[:, :, :-1], axis=2)
    return np.where(W > 0, -1, 1).astype(np.int8)


def mesh_sdf(p, k=6, sgn=None):
    p = p.reshape(-1, 3)
    dist, idx = TREE.query(p, k=k, workers=-1)
    q = SP[idx]; n = SN[idx]
    plane = np.einsum('nkj,nkj->nk', p[:, None] - q, n)
    w = 1.0 / (dist ** 2 + 1e-6)
    dp = (plane * w).sum(1) / w.sum(1)
    dmin = dist[:, 0]
    t = np.clip((dmin - 0.03) / 0.03, 0, 1)
    if sgn is None:
        sgn = winding_sign(p)
    d = sgn * ((1 - t) * np.abs(dp) + t * dmin)
    rz = (SR[idx] * w).sum(1) / w.sum(1)
    return d, rz


def surface(inside, direction, step=0.004, maxd=0.8):
    """first exit point of the game surface along a ray from an inside point, and its normal"""
    o = np.asarray(inside, float); dvec = nrm(direction)
    ts = np.arange(0, maxd, step); pts = o + ts[:, None] * dvec
    d, _ = mesh_sdf(pts)
    j = np.argmax(d > 0)
    assert d[0] < 0 and d[j] > 0, f'ray from {inside} did not start inside / exit'
    t = ts[j - 1] + step * (-d[j - 1]) / (d[j] - d[j - 1])
    S = o + t * dvec
    _, i = TREE.query(S)
    return S, SN[i]


# ---------------------------------------------------------------- sculpt features
def frame(n, hint):
    n = nrm(n); t1 = np.asarray(hint, float) - np.dot(hint, n) * n; t1 = nrm(t1); return n, t1, np.cross(n, t1)


class Bulge:
    """oriented ellipsoid pushed out through the surface by `lift` (smooth-unioned)"""
    def __init__(self, inside, direction, radii, lift, hint=(0, 1, 0), k=0.05, sink=None):
        S, N = surface(inside, direction)
        N = nrm(0.5 * N + 0.5 * nrm(direction))
        self.n, self.t1, self.t2 = frame(N, hint)
        self.r = np.array(radii, float); self.c = S - self.n * (self.r[0] - lift); self.k = k
        self.R = self.r.max() + k + 0.02

    def apply(self, p, d):
        q = p - self.c
        m = np.einsum('ij,ij->i', q, q) < self.R ** 2
        if not m.any():
            return d
        ql = np.stack([q[m] @ self.n, q[m] @ self.t1, q[m] @ self.t2], 1)
        d = d.copy(); d[m] = smin(d[m], ellipsoid(ql, np.zeros(3), self.r), self.k)
        return d


class Groove:
    """a channel carved along a surface path"""
    def __init__(self, pts_inside, direction, r, depth, k=0.012):
        self.S = []
        for pi in pts_inside:
            S, N = surface(pi, direction); self.S.append(S - nrm(N) * 0 + nrm(direction) * (r - depth))
        self.r, self.k = r, k

    def apply(self, p, d):
        d = d.copy()
        for a, b in zip(self.S[:-1], self.S[1:]):
            c = 0.5 * (a + b); R = np.linalg.norm(b - a) / 2 + self.r + self.k + 0.01
            m = np.einsum('ij,ij->i', p - c, p - c) < R * R
            if m.any():
                d[m] = smax(d[m], -capsule(p[m], a, b, self.r), self.k)
        return d


class Carve:
    """remove the game geometry inside a capsule region beyond a cutting plane (old hands/toes)"""
    def __init__(self, a, b, r, plane_pt, plane_n, k=0.015):
        self.a, self.b, self.r = np.asarray(a, float), np.asarray(b, float), r
        self.pp, self.pn, self.k = np.asarray(plane_pt, float), nrm(plane_n), k

    def region(self, p):
        return np.maximum(capsule(p, self.a, self.b, self.r), -((p - self.pp) @ self.pn))


class Solid:
    """arbitrary SDF added with smooth union inside a bounding sphere"""
    def __init__(self, f, centre, radius, k):
        self.f, self.c, self.R, self.k = f, np.asarray(centre, float), radius, k

    def apply(self, p, d):
        m = np.einsum('ij,ij->i', p - self.c, p - self.c) < self.R ** 2
        if m.any():
            d = d.copy(); d[m] = smin(d[m], self.f(p[m]), self.k)
        return d


CLAWS = []          # (base, direction, curl direction, length, radius)


def finger_chain(p, joints, radii):
    d = np.full(len(p), 1e9)
    for (a, b), r in zip(zip(joints[:-1], joints[1:]), radii):
        d = np.minimum(d, capsule(p, a, b, r))
    for j, r in zip(joints[1:-1], radii[1:]):      # knuckle bumps on every joint
        d = smin(d, sphere(p, j, r * 1.12), 0.012)
    return d


def bend(v, toward, ang):
    v = nrm(v); t = nrm(np.asarray(toward) - np.dot(toward, v) * v)
    return np.cos(ang) * v + np.sin(ang) * t


# ---------- left hand: open, hanging, palm toward the thigh, heavy clawed fingers
WL = np.array([-0.872, 1.775, 0.02])               # wrist
aL = nrm([0.03, -0.95, 0.31])                      # down the hand
nL = nrm(np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0, 0], aL) * aL)   # palm normal (toward the body)
wL = np.cross(aL, nL)                              # across the hand (thumb side = forward)


def left_hand_parts():
    parts = []
    palm_c = WL + aL * 0.13
    parts.append(lambda p: ellipsoid(np.stack([(p - palm_c) @ aL, (p - palm_c) @ nL, (p - palm_c) @ wL], 1), np.zeros(3), np.array([0.135, 0.065, 0.125])))
    parts.append(lambda p: capsule(p, WL - aL * 0.06, WL + aL * 0.06, 0.085))          # wrist
    fingers = []
    for i, (w, L, r) in enumerate(((-0.088, 0.86, 0.036), (-0.03, 1.0, 0.04), (0.03, 0.97, 0.04), (0.088, 0.82, 0.036))):
        base = WL + aL * 0.235 + wL * w - nL * 0.01
        dirn = nrm(aL + wL * w * 0.8)
        segs = np.array([0.085, 0.065, 0.05]) * L * 1.15
        js = [base]; dv = dirn
        for s, ang in zip(segs, (0.25, 0.42, 0.42)):
            dv = bend(dv, nL, ang); js.append(js[-1] + dv * s)
        fingers.append((js, [r, r * 0.9, r * 0.8]))
        CLAWS.append((js[-1] - dv * 0.012, bend(dv, nL, 0.25), nL, 0.075 * L, r * 0.72))
    # thumb: from the side of the palm, forward and across
    tb = WL + aL * 0.07 + wL * 0.11 + nL * 0.02
    tdir = nrm(aL * 0.6 + wL * 0.6 + nL * 0.35)
    tj = [tb, tb + tdir * 0.075]
    td = bend(tdir, nL, 0.45); tj.append(tj[-1] + td * 0.06)
    fingers.append((tj, [0.04, 0.034]))
    CLAWS.append((tj[-1] - td * 0.01, bend(td, nL, 0.2), nL, 0.06, 0.026))
    for js, rs in fingers:
        parts.append(lambda p, js=js, rs=rs: finger_chain(p, js, rs))
    # knuckle row on the back of the hand
    for w in (-0.088, -0.03, 0.03, 0.088):
        kc = WL + aL * 0.23 + wL * w - nL * 0.035
        parts.append(lambda p, kc=kc: sphere(p, kc, 0.03))
    return parts


# ---------- right hand: a fist around the mace grip
GC = np.array([0.249, 2.991, 0.946]); GA = nrm([-0.189, -0.391, 0.901]); GR = 0.07
TH = -0.085                                        # hand centre along the grip axis
HC = GC + GA * TH
ELB = np.array([0.9, 2.35, 0.3])
uR = ELB - HC; uR = nrm(uR - np.dot(uR, GA) * GA)  # from the shaft toward the wrist
vR = np.cross(GA, uR)
if np.dot(vR, nrm([1.0, 0.25, 0.6])) < 0:          # knuckles face away from the body
    vR = -vR
WR = HC + uR * 0.2 + GA * 0.0                      # wrist


def ring_pt(t, ang, rad):
    return HC + GA * t + rad * (np.cos(ang) * uR + np.sin(ang) * vR)


def right_hand_parts():
    parts = []
    # palm wrapped against the wrist side of the shaft, joined to the forearm
    pc = HC + uR * (GR + 0.055)
    parts.append(lambda p: ellipsoid(np.stack([(p - pc) @ uR, (p - pc) @ vR, (p - pc) @ GA], 1), np.zeros(3), np.array([0.07, 0.115, 0.13])))
    parts.append(lambda p: capsule(p, WR + uR * 0.06, pc + uR * 0.01, 0.085))
    # four fingers curl round the grip; the index sits nearest the mace head (-GA)
    for t, r in ((-0.083, 0.039), (-0.028, 0.041), (0.027, 0.04), (0.08, 0.035)):
        rad = GR + r * 0.95
        angs = [np.radians(a) for a in (55, 112, 175, 232, 268)]
        js = [ring_pt(t, a, rad + (0.022 if i == 1 else 0.0)) for i, a in enumerate(angs)]
        js[0] = ring_pt(t, angs[0], rad + 0.04)
        parts.append(lambda p, js=js, r=r: finger_chain(p, js, [r * 1.05, r, r * 0.92, r * 0.85]))
        tip = js[-1]; tdir = nrm(js[-1] - js[-2])
        CLAWS.append((tip - tdir * 0.01, tdir, -uR, 0.06, r * 0.7))
        kc = ring_pt(t, np.radians(108), rad + 0.03)                 # big knuckle
        parts.append(lambda p, kc=kc, r=r: sphere(p, kc, r * 1.05))
    # thumb: from the palm, over the index and middle fingers
    tj = [ring_pt(-0.11, np.radians(-35), GR + 0.065), ring_pt(-0.115, np.radians(-90), GR + 0.05),
          ring_pt(-0.085, np.radians(-140), GR + 0.04)]
    tj.append(ring_pt(-0.045, np.radians(-178), GR + 0.04))
    parts.append(lambda p: finger_chain(p, tj, [0.045, 0.04, 0.036]))
    td = nrm(tj[-1] - tj[-2]); CLAWS.append((tj[-1] - td * 0.01, td, -uR, 0.05, 0.025))
    return parts


# ---------- feet: three clawed toes on a forefoot pad
FEET = [  # (heel-toe centre in xz, direction in xz)
    (np.array([-0.595, -0.28]), nrm([-0.362, 0.932])),
    (np.array([0.606, 0.167]), nrm([0.275, 0.961])),
]


def foot_parts(c, f):
    F3 = np.array([f[0], 0, f[1]]); W3 = np.array([-f[1], 0, f[0]]); Y = np.array([0, 1.0, 0])
    B = np.array([c[0], 0.0, c[1]]) + F3 * 0.12
    parts = [lambda p: ellipsoid(np.stack([(p - (B + Y * 0.075)) @ F3, (p - (B + Y * 0.075)) @ Y, (p - (B + Y * 0.075)) @ W3], 1), np.zeros(3), np.array([0.12, 0.08, 0.2]))]
    for w, L, r in ((-0.13, 0.85, 0.05), (0.0, 1.0, 0.058), (0.13, 0.88, 0.052)):
        b0 = B + W3 * w + Y * 0.07 + F3 * 0.02
        dv = nrm(F3 + W3 * w * 0.5)
        k1 = b0 + dv * 0.1 * L + Y * 0.02
        tip = k1 + dv * 0.085 * L - Y * 0.045
        js = [b0, k1, tip]
        parts.append(lambda p, js=js, r=r: finger_chain(p, js, [r, r * 0.85]))
        td = nrm(tip - k1); CLAWS.append((tip - td * 0.012, nrm(td + Y * 0.1), -Y, 0.085 * L, r * 0.7))
    return parts, B, F3


# ---------------------------------------------------------------- assemble
def build_features():
    S = []
    X = np.array([1.0, 0, 0])
    for s in (-1, 1):
        S.append(Bulge((s * 0.23, 3.02, -0.12), (s * 0.28, 0.08, 1), (0.1, 0.21, 0.15), 0.065, hint=X, k=0.045))     # pecs (heavier)
        S.append(Bulge((s * 0.24, 3.36, -0.25), (s * 0.25, 1, -0.25), (0.08, 0.2, 0.13), 0.035, hint=X, k=0.06))     # traps
        S.append(Bulge((s * 0.42, 3.16, -0.22), (s * 1, 0.55, 0.05), (0.085, 0.17, 0.2), 0.04, hint=(0, 1, 0), k=0.05))  # deltoids
        S.append(Bulge((s * 0.26, 3.0, -0.3), (s * 0.25, 0.05, -1), (0.06, 0.17, 0.2), 0.025, hint=(0, 1, 0), k=0.05))  # shoulder blades
    S.append(Groove([(0, y, -0.3) for y in (3.25, 3.0, 2.75, 2.5, 2.3)], (0, 0, -1), 0.03, 0.016))                    # spine
    S.append(Groove([(x, 2.88, -0.05) for x in (-0.36, -0.12, 0.0)], (0, -0.25, 1), 0.016, 0.01))                     # under the left pec
    S.append(Groove([(0.0, 2.88, -0.05), (0.15, 2.88, -0.05)], (0, -0.25, 1), 0.016, 0.01))                           # under the right pec
    # ---- torso definition: chest and stomach
    S.append(Groove([(x, 2.875, -0.05) for x in (-0.33, -0.15, 0.0, 0.15, 0.3)], (0, -0.25, 1), 0.03, 0.03))          # hard line under the pecs
    S.append(Groove([(0, y, 0.0) for y in (3.14, 3.02, 2.9)], (0, 0, 1), 0.026, 0.024))                              # breastbone
    S.append(Groove([(0, y, 0.0) for y in (2.86, 2.7, 2.53)], (0, 0, 1), 0.024, 0.022))                               # centre line (above the belt)
    S.append(Groove([(0, y, 0.0) for y in (2.36, 2.2, 2.0)], (0, 0, 1), 0.024, 0.022))                                # centre line (below the belt)
    for s in (-1, 1):
        for y in (2.79, 2.65, 2.27, 2.12):                                                                            # ab blocks
            S.append(Bulge((s * 0.08, y, 0.0), (s * 0.18, 0, 1), (0.045, 0.07, 0.064), 0.036, hint=X, k=0.018))
        S.append(Groove([(s * 0.05, 2.85, 0.0), (s * 0.17, 2.79, 0.0), (s * 0.28, 2.66, 0.0)], (s * 0.3, -0.1, 1), 0.024, 0.018))  # ribcage arch
        S.append(Bulge((s * 0.29, 2.22, -0.05), (s * 1, 0, 0.55), (0.06, 0.17, 0.1), 0.042, hint=(0, 1, 0), k=0.035))   # obliques
        for i, y in enumerate((2.97, 2.87, 2.77)):                                                                   # serratus along the ribs
            S.append(Bulge((s * 0.3, y, 0.0), (s * 0.85, -0.1, 0.6), (0.026, 0.05, 0.032), 0.02, hint=(s * 0.3, -0.35, 1), k=0.012))
    # left arm (hanging, swung back)
    S.append(Bulge((-0.84, 2.5, -0.52), (0, 0.15, 1), (0.065, 0.2, 0.11), 0.03, hint=(0, 1, 0)))                  # biceps
    S.append(Bulge((-0.84, 2.55, -0.52), (0, 0, -1), (0.07, 0.22, 0.12), 0.03, hint=(0, 1, 0)))                    # triceps
    S.append(Bulge((-0.85, 2.13, -0.26), (0, -0.25, -1), (0.045, 0.055, 0.055), 0.025, hint=X, k=0.03))           # elbow
    S.append(Bulge((-0.86, 1.98, -0.14), (-1, 0, 0.15), (0.06, 0.15, 0.08), 0.025, hint=(0, -0.75, 0.66)))         # forearm
    S.append(Bulge((-0.86, 1.96, -0.12), (0.2, 0, 1), (0.05, 0.13, 0.07), 0.02, hint=(0, -0.75, 0.66)))
    # right arm (raised, carrying the mace)
    S.append(Bulge((0.88, 2.42, 0.25), (0.25, -1, -0.35), (0.045, 0.06, 0.06), 0.025, hint=X, k=0.03))            # elbow
    S.append(Bulge((0.62, 2.66, 0.55), (0.6, 0.25, 0.4), (0.06, 0.16, 0.08), 0.025, hint=nrm(HC - ELB)))           # forearm
    S.append(Bulge((0.72, 2.75, 0.05), (0.25, 0.3, 1), (0.065, 0.19, 0.11), 0.03, hint=(0.3, 1, -0.3)))            # biceps
    S.append(Bulge((0.75, 2.75, -0.05), (0.5, 0.0, -1), (0.07, 0.2, 0.12), 0.03, hint=(0.3, 1, -0.3)))             # triceps
    # legs
    for cx, cz in ((0.38, 0.06), (-0.41, -0.29)):
        S.append(Bulge((cx, 0.96, cz), (0, 0.05, 1), (0.05, 0.075, 0.07), 0.03, hint=X, k=0.035))                # kneecap
        S.append(Bulge((cx - np.sign(cx) * 0.04, 1.3, cz - 0.02), (0, 0.05, 1), (0.07, 0.22, 0.13), 0.03, hint=(0, 1, 0)))   # quads
        S.append(Bulge((cx + np.sign(cx) * 0.03, 0.62, cz - 0.07), (0, 0.1, -1), (0.07, 0.17, 0.11), 0.035, hint=(0, 1, 0)))  # calves
    for (cx, cz), sx in (((0.44, -0.13), 1), ((0.44, -0.13), -1), ((-0.41, -0.54), 1), ((-0.41, -0.54), -1)):
        S.append(Bulge((cx, 0.27, cz), (sx, 0, 0), (0.03, 0.035, 0.035), 0.02, hint=(0, 1, 0), k=0.02))             # ankle bones
    # hands and feet
    carves = [Carve(WL + aL * 0.02, WL + aL * 0.55, 0.2, WL + aL * 0.02, aL),
              Carve(HC - GA * 0.3, HC + GA * 0.25, 0.2, HC, -uR, k=0.012),
              Carve(HC - GA * 0.3, HC + GA * 0.25, 0.135, HC + uR * 0.135, uR, k=0.012)]
    solids = [Solid(lambda p, parts=left_hand_parts(): np.min([f(p) for f in parts], 0), WL + aL * 0.3, 0.5, 0.025),
              Solid(lambda p, parts=right_hand_parts(): np.min([f(p) for f in parts], 0), HC, 0.45, 0.022)]
    for c, f in FEET:
        parts, B, F3 = foot_parts(c, f)
        carves.append(Carve(B + F3 * 0.0, B + F3 * 0.4, 0.26, B + F3 * 0.04, F3))
        solids.append(Solid(lambda p, parts=parts: np.min([g(p) for g in parts], 0), B + F3 * 0.15, 0.45, 0.04))
    return S, carves, solids


def body_sdf(p, feats, sgn=None):
    S, carves, solids = feats
    d, rz = mesh_sdf(p, sgn=sgn)
    d = d - 0.011 * np.clip((rz - 0.35) / 0.3, 0, 1)            # straps, belt and loincloth stand proud
    for c in carves:
        d = smax(d, -c.region(p), c.k)
    for f in S:
        d = f.apply(p, d)
    for s in solids:
        d = s.apply(p, d)
    return d


def polygonize(feats):
    lo = np.array([-1.12, -0.03, -1.2]); hi = np.array([1.15, 3.95, 1.52])
    n = np.ceil((hi - lo) / VOX).astype(int) + 1
    log('grid', n, n.prod() / 1e6, 'M voxels')
    # coarse pass picks the narrow band
    B = 12
    nb = np.ceil(n / B).astype(int)
    cc = lo + (np.stack(np.meshgrid(*[np.arange(k) for k in nb], indexing='ij'), -1).reshape(-1, 3) + 0.5) * B * VOX
    WG = winding_grid(lo, n)
    log('winding grid done; inside voxels', int((WG < 0).sum()))
    ci = np.clip(np.round((cc - lo) / VOX).astype(int), 0, n - 1)
    dc = body_sdf(cc, feats, sgn=WG[ci[:, 0], ci[:, 1], ci[:, 2]].astype(float))
    half = B * VOX * 0.5 * np.sqrt(3) + 0.06
    active = np.abs(dc) < half
    log('active blocks', active.sum(), '/', len(cc))
    G = np.empty(n, np.float32)
    inside = (dc < 0).reshape(nb)
    # fill everything with the coarse sign first
    for i in range(nb[0]):
        for j in range(nb[1]):
            for k in range(nb[2]):
                G[i * B:(i + 1) * B, j * B:(j + 1) * B, k * B:(k + 1) * B] = -1.0 if inside[i, j, k] else 1.0
    ids = np.argwhere(active.reshape(nb))
    li = np.arange(B)
    loc = np.stack(np.meshgrid(li, li, li, indexing='ij'), -1).reshape(-1, 3)
    CH = 64
    for s in range(0, len(ids), CH):
        blk = ids[s:s + CH]
        idx = (blk[:, None, :] * B + loc[None]).reshape(-1, 3)
        ok = np.all(idx < n, 1); idx = idx[ok]
        d = body_sdf(lo + idx * VOX, feats, sgn=WG[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float))
        G[idx[:, 0], idx[:, 1], idx[:, 2]] = d
        if s % (CH * 20) == 0:
            log('block', s, '/', len(ids))
    log('marching cubes')
    Vm, Fm, _, _ = marching_cubes(G, 0.0, spacing=(VOX,) * 3)
    return Vm + lo, Fm


# ---------------------------------------------------------------- claws
def claw_mesh(base, dirn, curl, length, r, nring=10, nseg=12):
    dirn = nrm(dirn); curl = nrm(np.asarray(curl) - np.dot(curl, dirn) * dirn); side = np.cross(dirn, curl)
    Pts, Fs_ = [], []
    for i in range(nseg + 1):
        t = i / nseg
        c = base + dirn * length * t + curl * length * 0.35 * t * t
        tang = nrm(dirn + curl * 0.7 * t)
        a1 = nrm(curl - np.dot(curl, tang) * tang); a2 = np.cross(tang, a1)
        rr = r * (1 - t) ** 0.85 + 1e-4
        for j in range(nring):
            ang = 2 * np.pi * j / nring
            Pts.append(c + rr * (np.cos(ang) * a1 * 0.8 + np.sin(ang) * a2))   # slightly flattened
    for i in range(nseg):
        for j in range(nring):
            a, b2 = i * nring + j, i * nring + (j + 1) % nring
            Fs_ += [[a, b2, a + nring], [b2, b2 + nring, a + nring]]
    Pts = np.array(Pts)
    cap = len(Pts); Pts = np.vstack([Pts, base - dirn * 0.01])
    for j in range(nring):
        Fs_.append([cap, (j + 1) % nring, j])
    return Pts, np.array(Fs_)


def transfer(Vn, Fn):
    """material class and UVs for the new faces from the nearest game face (chart-consistent per face)"""
    cent0 = V0[F0].mean(1); t0 = cKDTree(cent0)
    cn = Vn[Fn].mean(1)
    _, fi = t0.query(cn, k=1, workers=-1)
    cls = CLS0[fi]
    A, Bv, C = V0[F0[fi, 0]], V0[F0[fi, 1]], V0[F0[fi, 2]]
    e0, e1 = Bv - A, C - A
    d00 = (e0 * e0).sum(1); d01 = (e0 * e1).sum(1); d11 = (e1 * e1).sum(1); den = d00 * d11 - d01 ** 2 + 1e-12
    UV = np.zeros((len(Fn), 3, 2))
    for c in range(3):
        q = Vn[Fn[:, c]] - A
        d20 = (q * e0).sum(1); d21 = (q * e1).sum(1)
        v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den; u = 1 - v - w
        UV[:, c] = u[:, None] * UV0[fi, 0] + v[:, None] * UV0[fi, 1] + w[:, None] * UV0[fi, 2]
    return cls, UV


def main():
    feats = build_features()
    log('features ready; claws', len(CLAWS))
    Vn, Fn = polygonize(feats)
    log('mc', Vn.shape, Fn.shape)
    Fn = Fn[:, ::-1]                                 # outward winding (skimage gives inward for this layout)
    # keep only the big pieces (drops scraps of the game's horn strips)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.concatenate([Fn[:, [0, 1]], Fn[:, [1, 2]], Fn[:, [2, 0]]])
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(Vn), len(Vn))), directed=False)
    cnt = np.bincount(lab[Fn[:, 0]])
    keep = cnt[lab[Fn[:, 0]]] >= 20000
    log('components', len(cnt), 'kept sizes', sorted(cnt[cnt >= 20000].tolist()), 'dropped faces', int((~keep).sum()))
    Fn = Fn[keep]; used = np.unique(Fn); remap = -np.ones(len(Vn), int); remap[used] = np.arange(len(used)); Vn = Vn[used]; Fn = remap[Fn]
    Vn = laplacian_smooth(Vn, Fn, 2, 0.35)
    Nn = vertex_normals(Vn, Fn)
    # winding check: normals should point away from the centroid on average
    if np.mean(np.einsum('ij,ij->i', Nn, Vn - Vn.mean(0))) < 0:
        Fn = Fn[:, ::-1]; log('flipped winding')
    cls, UV = transfer(Vn, Fn)
    np.savez_compressed(P + 'body_sculpt.npz', V=Vn, F=Fn, UV=UV, cls=cls)
    CP, CF = [], []
    for base, dirn, curl, L, r in CLAWS:
        p, f = claw_mesh(base, dirn, curl, L, r)
        CF.append(f + sum(len(x) for x in CP)); CP.append(p)
    np.savez_compressed(P + 'claws_sculpt.npz', V=np.vstack(CP), F=np.vstack(CF))
    log('saved', len(Vn), 'verts', len(Fn), 'faces; class counts', np.bincount(cls))


if __name__ == '__main__':
    main()
