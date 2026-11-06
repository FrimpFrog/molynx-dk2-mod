"""Molynx's cinematic head: a sculpt built from blended 3D shapes (signed distance field),
polygonized densely, then covered in cracked stone plates. Two expressions:

  grin   - lips drawn back in a snarl, the lower teeth up over the upper lip, fangs at the corners
  closed - mouth shut, lips together, only the two lower tusks peek up over the upper lip

Bulging glossy eyeballs sit in deep sockets under heavy angry lids and the brow ridge.
Coordinates: the game data's (before the x mirror), placed on the design head's landmarks.
"""
import numpy as np
from skimage.measure import marching_cubes

O = np.array([0.0275, 3.60, 0.45])          # local origin: face midline, between eyes and mouth
EYE_R = 0.04
EYES = [np.array([-0.072, 0.045, 0.152]), np.array([0.072, 0.045, 0.152])]


# ---------------------------------------------------------------- SDF primitives
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / r, axis=-1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def sphere(p, c, r):
    return np.linalg.norm(p - c, axis=-1) - r


def capsule(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def mirror_x(f):
    """f evaluated on both sides (left/right symmetric feature) as a union."""
    return lambda p: np.minimum(f(p), f(p * np.array([-1, 1, 1])))


def cone(p, a, b, ra, rb):
    """a capsule whose radius tapers from ra at a to rb at b (approximate SDF, smooth enough here)"""
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - (ra + (rb - ra) * h)


def plane(p, p0, n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    return (p - p0) @ n


NOSE = 'golem'
THROAT = 'trim'


def human_nose(d, p, q):
    """the earlier nose: small, human - a short bridge, a hooked tip and flared wings"""
    d = smin(d, capsule(p, np.array([0, 0.03, 0.226]), np.array([0, -0.028, 0.248]), 0.019), 0.02)
    d = smin(d, ellipsoid(p, np.array([0, -0.046, 0.256]), np.array([0.026, 0.02, 0.02])), 0.016)
    d = smin(d, ellipsoid(q, np.array([0.03, -0.058, 0.238]), np.array([0.024, 0.016, 0.018])), 0.012)
    d = smax(d, -sphere(q, np.array([0.02, -0.066, 0.258]), 0.0095), 0.005)
    d = smax(d, -capsule(q, np.array([0.05, -0.042, 0.232]), np.array([0.054, -0.066, 0.228]), 0.0045), 0.005)
    for yy in (0.004, 0.018):
        d = smax(d, -capsule(p, np.array([-0.02, yy, 0.244 - yy * 0.4]), np.array([0.02, yy, 0.244 - yy * 0.4]), 0.004), 0.004)
    return d


def golem_nose(d, p, q):
    """a big blunt slab of rock growing straight out of the brow knot: a faceted wedge (flat front,
    flat cheeks, a squared-off underside) that widens into heavy nostril lobes - no human bridge or wings"""
    n = cone(p, np.array([0, 0.058, 0.236]), np.array([0, -0.052, 0.272]), 0.026, 0.052)
    n = smin(n, ellipsoid(q, np.array([0.044, -0.066, 0.256]), np.array([0.037, 0.03, 0.035])), 0.02)    # nostril lobes
    # chisel the facets: a flat front, two flat side planes (a wedge in cross-section), a squared base
    # and a bevel across the tip so it ends blunt, not pointed
    n = smax(n, plane(p, np.array([0, -0.01, 0.291]), (0, 0.3, 1.0)), 0.012)
    n = smax(n, plane(q, np.array([0.056, 0.0, 0.268]), (0.95, 0.05, 0.42)), 0.012)
    n = smax(n, plane(p, np.array([0, -0.092, 0.27]), (0, -1.0, 0.1)), 0.008)
    n = smax(n, plane(p, np.array([0, -0.074, 0.296]), (0, -0.75, 0.66)), 0.01)
    d = smin(d, n, 0.022)
    # deep nostril pits in the underside, angled out and back
    d = smax(d, -capsule(q, np.array([0.026, -0.1, 0.27]), np.array([0.04, -0.082, 0.254]), 0.0115), 0.006)
    # a chipped crack across the bridge and a notch down one side (stone, not skin)
    d = smax(d, -capsule(p, np.array([-0.034, 0.014, 0.264]), np.array([0.03, 0.0, 0.272]), 0.0055), 0.004)
    d = smax(d, -capsule(p, np.array([0.032, -0.012, 0.272]), np.array([0.056, -0.054, 0.27]), 0.0045), 0.004)
    return d


# ---------------------------------------------------------------- the sculpt
def head_sdf(p, expr):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    ax = np.abs(x)
    q = np.stack([ax, y, z], -1)                        # left/right symmetric shapes use |x|
    d = ellipsoid(p, np.array([0, 0.095, -0.07]), np.array([0.262, 0.215, 0.27]))        # cranium: broad, low
    d = smin(d, ellipsoid(p, np.array([0, 0.02, 0.035]), np.array([0.215, 0.165, 0.15])), 0.05)  # face mass
    # brow ridge: massive, low and furious - a V crushing down onto the eyes, with bony knots
    # (lowered and pushed forward: a hard shelf of rock that shades the eyes)
    brow = capsule(q, np.array([0.198, 0.1, 0.128]), np.array([0.028, 0.058, 0.242]), 0.055)
    brow = smin(brow, sphere(p, np.array([0, 0.052, 0.24]), 0.042), 0.03)            # the knot between the brows
    brow = smin(brow, sphere(q, np.array([0.12, 0.1, 0.182]), 0.028), 0.04)            # bony knot above each eye (subtle)
    d = smin(d, brow, 0.03)
    # frown furrows between the brows and creases across the forehead
    for sx in (-1, 1):
        d = smax(d, -capsule(p, np.array([sx * 0.016, 0.045, 0.272]), np.array([sx * 0.026, 0.115, 0.245]), 0.0065), 0.006)
    for yy, zz, w in ((0.152, 0.205, 0.13), (0.185, 0.18, 0.12)):
        pts = [np.array([-w, yy + 0.012, zz - 0.05]), np.array([-w * 0.45, yy, zz - 0.008]), np.array([w * 0.45, yy, zz - 0.008]), np.array([w, yy + 0.012, zz - 0.05])]
        for a, b in zip(pts[:-1], pts[1:]):
            d = smax(d, -capsule(p, a, b, 0.0055), 0.006)
    # cheekbones (sharp), sunken cheeks under them, heavy jaw muscles
    # (flattened: broad stone planes, not two round balls beside the eyes)
    d = smin(d, ellipsoid(q, np.array([0.13, -0.035, 0.125]), np.array([0.095, 0.042, 0.07])), 0.055)
    d = smax(d, -ellipsoid(q, np.array([0.15, -0.095, 0.17]), np.array([0.05, 0.04, 0.022])), 0.04)
    d = smin(d, ellipsoid(q, np.array([0.165, -0.13, 0.04]), np.array([0.075, 0.115, 0.12])), 0.05)
    # chisel: flat planes cut across the cheeks and jaw corners (hewn rock, not soft flesh)
    for p0, n_ in (((0.168, -0.04, 0.17), (0.82, 0.0, 0.57)), ((0.15, -0.2, 0.18), (0.66, -0.42, 0.62)),
                   ((0.11, -0.075, 0.214), (0.22, -0.3, 0.93)), ((0.2, 0.06, 0.12), (0.9, 0.25, 0.36))):
        n_ = np.array(n_) / np.linalg.norm(n_)
        d = smax(d, plane(q, np.array(p0), n_), 0.012)
    if NOSE == 'human':
        d = human_nose(d, p, q)
    else:
        d = golem_nose(d, p, q)
    # upper lip / muzzle (kept behind the nose tip)
    grin = expr == 'grin'
    up_lip_y = -0.104 if grin else -0.114
    d = smin(d, ellipsoid(p, np.array([0, up_lip_y, 0.165]), np.array([0.13, 0.046, 0.095])), 0.03)
    # snarl folds from the nose wings down to the mouth corners
    for sx in (-1, 1):
        d = smax(d, -capsule(p, np.array([sx * 0.058, -0.06, 0.235]), np.array([sx * 0.118, -0.135, 0.2]), 0.0065), 0.007)
    # lower jaw: jutting forward past the upper lip (underbite), square chin with a cleft
    jaw_y = -0.192 if grin else -0.174
    d = smin(d, ellipsoid(p, np.array([0, jaw_y, 0.15]), np.array([0.175, 0.075, 0.15])), 0.04)
    if THROAT == 'old':
        d = smin(d, ellipsoid(p, np.array([0, jaw_y - 0.045, 0.205]), np.array([0.095, 0.05, 0.075])), 0.035)
        d = smax(d, -capsule(p, np.array([0, jaw_y - 0.06, 0.28]), np.array([0, jaw_y - 0.02, 0.29]), 0.006), 0.006)
        d = smin(d, capsule(p, np.array([0, -0.12, -0.05]), np.array([0, -0.33, -0.16]), 0.19), 0.08)
    else:
        # square chin, tucked up under the jaw (no hanging chin)
        d = smin(d, ellipsoid(p, np.array([0, jaw_y - 0.03, 0.2]), np.array([0.09, 0.034, 0.07])), 0.03)
        d = smax(d, -capsule(p, np.array([0, jaw_y - 0.045, 0.27]), np.array([0, jaw_y - 0.01, 0.28]), 0.006), 0.006)
        # neck: from the back of the skull down into the hump of his back (as in the game: the head juts
        # forward off the shoulders, nothing underneath the jaw)
        d = smin(d, capsule(p, np.array([0, -0.02, -0.16]), np.array([0, -0.38, -0.36]), 0.17), 0.06)
        # cut everything under the jaw line away, back to the neck: the jaw sits straight on the chest
        y0 = jaw_y - 0.06
        cut = np.maximum(y - y0, (-0.1 + 0.35 * (y - y0)) - z)
        d = smax(d, -cut, 0.02)
    # eyes: a narrow, slanted almond slit cut into the face under the brow; the glowing eye sits
    # behind it (Horned-Reaper-like glare), the lids are the face itself
    for e in EYES:
        sx = np.sign(e[0])
        dx = x - e[0]
        inward = dx * (-sx)                                         # + towards the nose
        upper = e[1] + 0.006 - 0.36 * inward - 5.5 * dx ** 2         # slants down towards the nose
        lower = e[1] - 0.012 - 0.06 * inward + 5.0 * dx ** 2
        slit = np.maximum.reduce([y - upper, lower - y, np.abs(dx) - 0.047, (e[2] + 0.012) - z])
        d = smax(d, -slit, 0.005)
        d = smax(d, -sphere(p, e, EYE_R + 0.004), 0.004)           # room for the eye behind the slit
    # mouth
    if grin:
        # a snarl: the cavity widest under the canines, corners dragged down and back
        cav = ellipsoid(p, np.array([0, -0.142, 0.21]), np.array([0.115, 0.02, 0.085]))
        cav = smin(cav, ellipsoid(q, np.array([0.06, -0.132, 0.215]), np.array([0.035, 0.03, 0.06])), 0.02)
        cav = smin(cav, ellipsoid(q, np.array([0.105, -0.152, 0.18]), np.array([0.035, 0.02, 0.05])), 0.02)
        d = smax(d, -cav, 0.008)
    else:
        # shut in a hard scowl: the line dips at the corners
        for a, b in ((np.array([-0.11, -0.156, 0.205]), np.array([-0.05, -0.143, 0.235])), (np.array([-0.05, -0.143, 0.235]), np.array([0.05, -0.143, 0.235])),
                     (np.array([0.05, -0.143, 0.235]), np.array([0.11, -0.156, 0.205]))):
            d = smax(d, -capsule(p, a, b, 0.0045), 0.004)
    return d


def polygonize(expr, voxel=0.0048):
    lo = np.array([-0.32, -0.6, -0.40]); hi = np.array([0.32, 0.40, 0.33])
    n = np.ceil((hi - lo) / voxel).astype(int) + 1
    xs, ys, zs = [np.linspace(lo[i], lo[i] + voxel * (n[i] - 1), n[i]) for i in range(3)]
    vol = np.empty(n, np.float32)
    for i, xv in enumerate(xs):                       # slab by slab (memory)
        g = np.stack(np.meshgrid([xv], ys, zs, indexing='ij'), -1).reshape(-1, 3)
        vol[i] = head_sdf(g, expr).reshape(n[1], n[2])
    V, F, Nm, _ = marching_cubes(vol, 0.0, spacing=(voxel,) * 3)
    V = V + lo
    # outward winding
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.mean(np.sum(fn * (V[F].mean(1) - np.array([0, 0.05, 0])), 1)) < 0:
        F = F[:, ::-1]
    return V, F


def eyeball(e, seg=48, rings=32):
    """A sphere with UVs: the iris at the front pole (u around, v from the front)."""
    th = np.linspace(0, np.pi, rings + 1); ph = np.linspace(0, 2 * np.pi, seg + 1)
    T, Pp = np.meshgrid(th, ph, indexing='ij')
    look = np.array([0.0, -0.12, 1.0]); look /= np.linalg.norm(look)
    a = np.cross(look, [0, 1, 0]); a /= np.linalg.norm(a); b = np.cross(look, a)
    dirs = (np.cos(T)[..., None] * look + np.sin(T)[..., None] * (np.cos(Pp)[..., None] * a + np.sin(Pp)[..., None] * b))
    V = (e + EYE_R * dirs).reshape(-1, 3)
    UV = np.stack([Pp / (2 * np.pi), T / np.pi], -1).reshape(-1, 2)
    F = []
    W = seg + 1
    for i in range(rings):
        for j in range(seg):
            a0, a1, b0, b1 = i * W + j, i * W + j + 1, (i + 1) * W + j, (i + 1) * W + j + 1
            F += [(a0, b0, a1), (a1, b0, b1)]
    return V, np.array(F), UV


def eye_texture(size=512):
    """Horned-Reaper-style eyes: solid glowing yellow, white-hot in the middle, burning orange at
    the rim, no pupil. (colour, emission) images; v = angle from the front pole."""
    v = (np.arange(size) + 0.5)[:, None] / size * np.pi * np.ones((1, size))
    t = np.clip(v / 1.2, 0, 1)
    core = np.array([255, 250, 190]); mid = np.array([255, 205, 45]); rim = np.array([235, 95, 5])
    c = np.where(t[..., None] < 0.45, core * (1 - t[..., None] / 0.45) + mid * (t[..., None] / 0.45),
                 mid * (1 - (t[..., None] - 0.45) / 0.55) + rim * ((t[..., None] - 0.45) / 0.55))
    u = (np.arange(size) + 0.5)[None, :] / size * 2 * np.pi * np.ones((size, 1))
    c = c * (0.93 + 0.07 * np.sin(u * 17 + v * 9)[..., None])           # a little flicker in the fire
    return c, c * 0.95


def teeth_set(expr):
    """Lower teeth along the jaw: (list of (base, up, out, width, height, depth, fang))."""
    out = []
    if expr == 'grin':
        rng = np.random.default_rng(4)
        xs = [-0.1, -0.066, -0.033, 0.0, 0.033, 0.066, 0.1]
        for k, x in enumerate(xs):
            tusk = k in (0, len(xs) - 1)
            zf = 0.262 - 3.0 * x * x
            base = np.array([x, -0.172, zf - 0.014])
            lean = rng.normal(0, 0.05)
            up = np.array([lean, 1.0, 0.18]); up /= np.linalg.norm(up)
            o = np.array([x * 2.2, 0, 1.0]); o /= np.linalg.norm(o)
            if tusk:
                up = np.array([-np.sign(x) * 0.08, 1.0, 0.2]); up /= np.linalg.norm(up)
                out.append((base + np.array([0, -0.01, 0]), up, o, 0.042, 0.072, 0.036, True))
            else:
                # square, worn, chipped blocks of stone (blunt tops), close together
                out.append((base, up, o, 0.033 * rng.uniform(0.95, 1.05), 0.046 * rng.uniform(0.85, 1.1), 0.026, False))
        # a short upper row, mostly hidden under the lip
        for x in (-0.085, -0.051, -0.017, 0.017, 0.051, 0.085):
            zf = 0.245 - 3.0 * x * x
            out.append((np.array([x, -0.1, zf - 0.016]), np.array([0, -1.0, 0.05]) / np.linalg.norm([0, -1, 0.05]),
                        np.array([x, 0, 1.0]) / np.linalg.norm([x, 0, 1]), 0.031, 0.04, 0.02, False))
    else:
        for x in (-0.092, 0.092):
            zf = 0.25 - 3.0 * x * x
            up = np.array([-np.sign(x) * 0.06, 1.0, 0.05]); up /= np.linalg.norm(up)
            o = np.array([x * 2.2, 0, 1.0]); o /= np.linalg.norm(o)
            out.append((np.array([x, -0.165, zf + 0.04]), up, o, 0.03, 0.095, 0.026, True))
    return out
