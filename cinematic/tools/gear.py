"""Molynx's gear, fitted to the sculpted body by ray casting (parts coordinates: y up, +z front, the mace
hand on +x):

  belt     a complete leather band round the waist, iron buckle at the front, iron studs
  skirt    a cloth skirt slung on the hips (lower at the front), flared over the thighs, folds and a
           tattered hem; in the animation it is simulated cloth pinned along its top
  armour   iron vambraces on both forearms (rolled rims, ridges, rivets) and a spiked iron band round
           each upper arm

build(V, F) -> list of dict(name, V, F, mat, bone, attrs)    bone: the bone it rides rigidly, or None
(skinned from the body) / 'cloth'.
"""
import math

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def nrm(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


class Caster:
    def __init__(self, V, F, kd=False):
        self.t = BVHTree.FromPolygons([tuple(v) for v in np.asarray(V, float)], np.asarray(F).tolist(), all_triangles=True)
        if kd:
            from mathutils.kdtree import KDTree
            self.kd = KDTree(len(V))
            for i, v in enumerate(np.asarray(V, float)):
                self.kd.insert(v, i)
            self.kd.balance()

    def inside(self, p):
        org = Vector(p); d = Vector((0.0123, 1.0, 0.0371)).normalized(); n = 0
        for _ in range(40):
            loc, nn, i, dist = self.t.ray_cast(org, d, 6.0)
            if loc is None:
                break
            n += 1; org = loc + d * 1e-4
        return n % 2 == 1

    def first(self, o, d, maxd=3.0):
        loc, n, i, dist = self.t.ray_cast(Vector(o), Vector(d), maxd)
        return None if loc is None else dist

    def last(self, o, d, maxd=3.0):
        """distance to the farthest surface crossing along the ray (outermost skin)"""
        o = Vector(o); d = Vector(d).normalized(); tot = 0.0; best = None
        for _ in range(12):
            loc, n, i, dist = self.t.ray_cast(o, d, maxd - tot)
            if loc is None:
                break
            tot += dist + 1e-4; best = tot; o = loc + d * 1e-4
        return best


def smooth_circ(a, n=3, axis=0):
    for _ in range(n):
        a = 0.25 * np.roll(a, 1, axis) + 0.5 * a + 0.25 * np.roll(a, -1, axis)
    return a


def grid_faces(nu, nv, closed_u=True, closed_v=False):
    """quads of a (nu x nv) grid of vertices, index = i * nv + j, as triangles"""
    F = []
    for i in range(nu if closed_u else nu - 1):
        i2 = (i + 1) % nu
        for j in range(nv if closed_v else nv - 1):
            j2 = (j + 1) % nv
            a, b, c, d = i * nv + j, i2 * nv + j, i2 * nv + j2, i * nv + j2
            F += [(a, b, c), (a, c, d)]
    return np.array(F, np.int64)


def box(c, ax, half):
    """oriented box: centre, 3 unit axes, half sizes"""
    c = np.asarray(c, float); P = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                P.append(c + ax[0] * half[0] * sx + ax[1] * half[1] * sy + ax[2] * half[2] * sz)
    F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return np.array(P), np.array(F)


def dome(c, n, r, h, seg=10):
    """a rivet/stud: a flattened half-sphere sitting on the surface with normal n"""
    n = nrm(n); t1 = nrm(np.cross(n, [0.3, 1, 0.2])); t2 = np.cross(n, t1)
    F = []
    rings = 4
    P = [np.asarray(c) + n * h]
    for k in range(1, rings + 1):
        a = k / rings * math.pi / 2
        for s in range(seg):
            ph = 2 * math.pi * s / seg
            P.append(np.asarray(c) + n * h * math.cos(a) + (t1 * math.cos(ph) + t2 * math.sin(ph)) * r * math.sin(a))
    P.append(np.asarray(c) - n * 0.006)
    for s in range(seg):
        F.append((0, 1 + s, 1 + (s + 1) % seg))
    for k in range(rings - 1):
        b0 = 1 + k * seg; b1 = b0 + seg
        for s in range(seg):
            s2 = (s + 1) % seg
            F += [(b0 + s, b1 + s, b1 + s2), (b0 + s, b1 + s2, b0 + s2)]
    last = 1 + (rings - 1) * seg; bot = len(P) - 1
    for s in range(seg):
        F.append((last + (s + 1) % seg, last + s, bot))
    return np.array(P), np.array(F)


def cone(base, tip, r, seg=10):
    base = np.asarray(base, float); tip = np.asarray(tip, float); a = nrm(tip - base)
    t1 = nrm(np.cross(a, [0.3, 1, 0.2])); t2 = np.cross(a, t1)
    P = [tip] + [base + (t1 * math.cos(2 * math.pi * s / seg) + t2 * math.sin(2 * math.pi * s / seg)) * r for s in range(seg)] + [base - a * 0.01]
    F = [(0, 1 + s, 1 + (s + 1) % seg) for s in range(seg)] + [(1 + (s + 1) % seg, 1 + s, seg + 1) for s in range(seg)]
    return np.array(P), np.array(F)


def merge(parts):
    V, F, o = [], [], 0
    for P, Fp in parts:
        V.append(P); F.append(np.asarray(Fp) + o); o += len(P)
    return np.vstack(V), np.vstack(F)


def noise1(x, seed):
    rng = np.random.default_rng(seed); ph = rng.uniform(0, 2 * np.pi, 4); fr = np.array([1, 2.3, 4.7, 9.1])
    return sum(np.sin(f * x + p) / f for f, p in zip(fr, ph)) / 1.6


# ---------------------------------------------------------------- belt
def belt(C, V):
    m = (V[:, 1] > 2.38) & (V[:, 1] < 2.62) & (np.abs(V[:, 0]) < 0.55) & (np.abs(V[:, 2]) < 0.6)
    cx, cz = V[m, 0].mean(), V[m, 2].mean()
    N = 192; th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dirs = np.stack([np.sin(th), np.zeros(N), np.cos(th)], 1)
    R = np.zeros(N)
    for i, d in enumerate(dirs):
        hs = [C.last((cx, h, cz), d, 1.0) for h in np.linspace(2.4, 2.6, 9)]
        hs = [x for x in hs if x is not None]
        R[i] = max(hs) if hs else np.nan
    R = np.where(np.isnan(R), np.nanmedian(R), R); R = smooth_circ(R, 6)
    y0, y1 = 2.405, 2.6
    # cross-section (radial offset, height), a rounded strap
    prof = [(-0.006, y0 + 0.004), (0.016, y0), (0.03, y0 + 0.008), (0.033, (y0 + y1) / 2), (0.03, y1 - 0.008), (0.016, y1), (-0.006, y1 - 0.004)]
    P = []
    for i in range(N):
        for off, y in prof:
            P.append((cx + dirs[i, 0] * (R[i] + off), y, cz + dirs[i, 2] * (R[i] + off)))
    P = np.array(P); Fb = grid_faces(N, len(prof), True, True)
    parts = [dict(name='belt', V=P, F=Fb, mat='leather', bone=None)]
    # iron buckle at the front, studs round the band
    i0 = 0; out = dirs[i0]; up = np.array([0, 1.0, 0]); tg = np.cross(up, out)
    bc = np.array([cx, (y0 + y1) / 2, cz]) + out * (R[i0] + 0.04)
    bw, bh, bt, bd = 0.12, 0.13, 0.026, 0.02
    bx = [box(bc + up * (bh - bt / 2), (tg, up, out), (bw, bt / 2, bd)), box(bc - up * (bh - bt / 2), (tg, up, out), (bw, bt / 2, bd)),
          box(bc + tg * (bw - bt / 2), (tg, up, out), (bt / 2, bh, bd)), box(bc - tg * (bw - bt / 2), (tg, up, out), (bt / 2, bh, bd)),
          box(bc - tg * 0.02, (tg, up, out), (0.012, bh - bt, bd * 0.7)),                           # centre bar
          box(bc + tg * 0.035 + out * 0.012, (tg, up, out), (0.05, 0.009, 0.008))]                  # prong
    studs = []
    for k in range(16):
        ang = 2 * np.pi * (k + 0.5) / 16
        if min(ang, 2 * np.pi - ang) < 0.32:
            continue
        i = int(round(ang / (2 * np.pi) * N)) % N
        for y in (2.455, 2.55):
            c = np.array([cx, y, cz]) + dirs[i] * (R[i] + 0.03)
            studs.append(dome(c, dirs[i], 0.017, 0.012))
    V2, F2 = merge(bx + studs)
    parts.append(dict(name='belt iron', V=V2, F=F2, mat='iron', bone=None))
    return parts, (cx, cz, R, th)


# ---------------------------------------------------------------- skirt
def skirt(C, V, cx, cz):
    N, NT = 160, 28
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    front = np.clip(np.cos(th), 0, 1) ** 2
    ytop = 2.07 - 0.19 * front
    L = 0.64 + 0.05 * noise1(th, 3)
    # tattered hem: strips of slightly different length, a few deep notches
    strip = np.floor(th / (2 * np.pi) * 34)
    rng = np.random.default_rng(7); sl = rng.uniform(-1, 1, 35)
    L = L * (1 - 0.3 * (sl[strip.astype(int)] + 1) / 2)           # strips torn to very different lengths
    for c0 in rng.uniform(0, 2 * np.pi, 22):
        L *= 1 - 0.45 * np.exp(-(np.angle(np.exp(1j * (th - c0))) / 0.028) ** 2)
    L *= 1 + 0.05 * np.sin(53 * th) * np.sin(7 * th)
    dirs = np.stack([np.sin(th), np.zeros(N), np.cos(th)], 1)
    rtop = np.zeros(N)
    for i, d in enumerate(dirs):
        hs = [C.last((cx, ytop[i] + dy, cz), d, 1.0) for dy in (-0.03, 0.0, 0.03)]
        hs = [x for x in hs if x is not None]
        rtop[i] = max(hs) if hs else np.nan
    rtop = np.where(np.isnan(rtop), np.nanmedian(rtop), rtop); rtop = smooth_circ(rtop, 5) + 0.014
    ts = np.linspace(0, 1, NT)
    Rg = np.zeros((N, NT)); Yg = np.zeros((N, NT))
    for j, t in enumerate(ts):
        y = ytop - L * t; Yg[:, j] = y
        base = rtop + 0.03 * t ** 1.2
        body = np.array([C.last((cx, y[i], cz), dirs[i], 1.2) or 0.0 for i in range(N)])
        body = smooth_circ(body, 3)
        Rg[:, j] = np.maximum(base, body + 0.014 + 0.008 * t)
    Rg = smooth_circ(Rg, 2)
    for _ in range(2):                                   # smooth down the skirt too (no kinks where it meets a thigh)
        Rg[:, 1:-1] = 0.25 * Rg[:, :-2] + 0.5 * Rg[:, 1:-1] + 0.25 * Rg[:, 2:]
    # folds: deeper toward the hem
    fold = (np.sin(17 * th + 1.9 * np.sin(3 * th + 0.5)) + 0.5 * np.sin(29 * th + 1.3))[:, None] * (0.004 + 0.028 * ts[None, :] ** 1.1)
    Rg = Rg + fold
    P = np.zeros((N, NT, 3))
    P[..., 0] = cx + dirs[:, None, 0] * Rg; P[..., 1] = Yg; P[..., 2] = cz + dirs[:, None, 2] * Rg
    F = grid_faces(N, NT, True, False)
    # tears: columns of faces removed from the hem upward, so the cloth hangs in separate ragged strips
    rt = np.random.default_rng(23)
    cols = rt.choice(N, 18, replace=False); hts = rt.uniform(0.25, 0.65, 18)
    fi = np.arange(len(F)); ci = fi // (2 * (NT - 1)); rj = (fi // 2) % (NT - 1)
    drop = np.zeros(len(F), bool)
    for c_, h_ in zip(cols, hts):
        drop |= (ci == c_) & (rj >= (NT - 1) * (1 - h_))
    F = F[~drop]
    tpar = np.tile(ts, N)
    return dict(name='skirt', V=P.reshape(-1, 3), F=F, mat='cloth', bone='cloth', attrs={'t': tpar}, grid=(N, NT))


# ---------------------------------------------------------------- arm armour
def sleeve(C, VV, a, b, t0, t1, lateral, name, bone, spikes=0, flare=0.0, nt=16, nth=72):
    a = np.asarray(a, float); b = np.asarray(b, float); ax = b - a; Lx = np.linalg.norm(ax); axn = ax / Lx
    u = nrm(np.asarray(lateral, float) - np.dot(lateral, axn) * axn); w = np.cross(axn, u)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False); ts = np.linspace(t0, t1, nt)
    # the real centre of the limb at each station (the bone line can run outside a swung-back arm): the
    # point inside the skin farthest from it, searched in the cross-section plane near the bone line
    cen = []
    g = np.arange(-0.27, 0.271, 0.015)
    for t in ts:
        o = a + ax * t; best, bs = None, -1e9
        for gu in g:
            for gw in g:
                r2 = gu * gu + gw * gw
                if r2 > 0.27 ** 2:
                    continue
                p = o + u * gu + w * gw
                clr = C.kd.find(p)[2]
                if clr < 0.03:
                    continue
                sc = clr - 0.12 * math.sqrt(r2)
                if sc > bs and C.inside(p):
                    best, bs = p, sc
        cen.append(best if best is not None else o)
    cen = np.array(cen)
    for _ in range(2):
        cen[1:-1] = 0.25 * cen[:-2] + 0.5 * cen[1:-1] + 0.25 * cen[2:]
    D = np.zeros((nt, nth))
    for j, t in enumerate(ts):
        o = cen[j]
        for i, ang in enumerate(th):
            d = u * math.cos(ang) + w * math.sin(ang)
            hit = C.first(o, d, 0.4)
            D[j, i] = hit if hit is not None else np.nan
    D = np.where(np.isnan(D), np.nanmedian(D), D)
    med = np.median(D, axis=1, keepdims=True)
    D = np.minimum(D, 1.6 * med)                         # ignore stray far hits (fingers, the mace)
    Draw = D.copy()
    D = smooth_circ(D, 4, axis=1)
    for _ in range(3):
        D[1:-1] = 0.25 * D[:-2] + 0.5 * D[1:-1] + 0.25 * D[2:]
    Dn = D.copy()                                        # never inside the skin: take the outer of raw and smoothed,
    for _ in range(3):                                   # dilated around the ring and along the limb
        Dn = np.maximum(Dn, np.maximum(np.roll(Draw, 1, 1), np.roll(Draw, -1, 1)))
        Dn = np.maximum(Dn, Draw)
    Dn[1:] = np.maximum(Dn[1:], Draw[:-1]); Dn[:-1] = np.maximum(Dn[:-1], Draw[1:])
    D = smooth_circ(Dn, 2, axis=1)
    s = (ts - t0) / (t1 - t0)
    rin = D + 0.024                                       # clearance over the rock surface
    rout = rin + 0.02 + flare * s[:, None] ** 2
    rim = np.exp(-(s / 0.07) ** 2) + np.exp(-((1 - s) / 0.07) ** 2)
    rout = rout + 0.012 * rim[:, None]
    ridge = np.exp(-((np.angle(np.exp(1j * th[None, :])) / 0.09) ** 2)) * (1 - rim[:, None]) * 0.01   # a ridge down the outside
    rout = rout + ridge
    # vertices: outer surface t0->t1, then inner surface t1->t0 (closed tube profile)
    rows = [(j, rout[j]) for j in range(nt)] + [(j, rin[j]) for j in range(nt - 1, -1, -1)]
    P = []
    for i, ang in enumerate(th):
        d = u * math.cos(ang) + w * math.sin(ang)
        for j, r in rows:
            P.append(cen[j] + d * r[i])
    P = np.array(P); F = grid_faces(nth, len(rows), True, True)
    parts = [(P, F)]
    # rivets round both rims
    for t, j in ((ts[1], 1), (ts[-2], nt - 2)):
        for k in range(8):
            i = int(k * nth / 8); ang = th[i]
            d = u * math.cos(ang) + w * math.sin(ang)
            parts.append(dome(cen[j] + d * rout[j, i], d, 0.013, 0.009))
    for k in range(spikes):
        i = int((k - (spikes - 1) / 2) * 0.35 / (2 * np.pi) * nth) % nth; ang = th[i]
        d = u * math.cos(ang) + w * math.sin(ang); jm = nt // 2
        base = cen[jm] + d * (rout[jm, i] - 0.004)
        parts.append(cone(base, base + d * 0.09 - axn * 0.015, 0.024))
    V2, F2 = merge(parts)
    return dict(name=name, V=V2, F=F2, mat='armour', bone=bone)


def build(V, F):
    V = np.asarray(V, float); F = np.asarray(F); C = Caster(V, F, kd=True)
    # torso and legs only (no arms, no hanging hand) for fitting the belt and the skirt
    cen = V[F].mean(1)
    keep = (np.abs(cen[:, 0]) < 0.64) & ~((cen[:, 1] > 2.25) & (np.abs(cen[:, 0]) > 0.5))
    Ct = Caster(V, F[keep])
    parts, (cx, cz, R, th) = belt(Ct, V)
    parts.append(skirt(Ct, V, cx, cz))
    # arms (same landmarks as the rig)
    GC = np.array([0.249, 2.991, 0.946]); GA = nrm([-0.189, -0.391, 0.901]); HC = GC + GA * -0.085
    ELB_R = np.array([0.9, 2.35, 0.3]); uR = ELB_R - HC; uR = nrm(uR - np.dot(uR, GA) * GA); WR = HC + uR * 0.2
    ELB_L = np.array([-0.85, 2.13, -0.27]); WL = np.array([-0.872, 1.775, 0.02])
    SH_L = np.array([-0.6, 3.05, -0.3]); SH_R = np.array([0.58, 3.0, -0.12])
    parts.append(sleeve(C, V, ELB_L, WL, 0.22, 0.86, (-1, 0, 0), 'vambrace L', 'fore.L', flare=0.012))
    parts.append(sleeve(C, V, ELB_R, WR, 0.36, 0.9, (0.5, -0.6, -0.3), 'vambrace R', 'fore.R', flare=0.01))
    parts.append(sleeve(C, V, SH_L, ELB_L, 0.3, 0.44, (-1, 0, 0), 'arm band L', 'upper.L', spikes=3, nt=8))
    parts.append(sleeve(C, V, SH_R, ELB_R, 0.36, 0.5, (1, 0.2, -0.3), 'arm band R', 'upper.R', spikes=3, nt=8))
    return parts
