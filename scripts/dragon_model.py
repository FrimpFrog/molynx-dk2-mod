"""The Dragon (from Dungeon Keeper 1) as a low-poly Dungeon Keeper 2 creature: skeleton, mesh, skin weights, UVs.

Built from scratch (no donor model), to the proportions of the DK1 dragon sprites (CREATURE.JTY entries
6738-6943; the side view 6800 and the front view 6798 set the body plan): an upright wyvern whose ribbed
belly runs from the hips up a long S-shaped neck, a long crocodile head with two swept-back horns, folded
bat wings that hang like a cloak, two thick digitigrade legs with three clawed toes and a long spiked tail
that lies on the ground behind it.

Model space: Y up, Z forward (the way the dragon faces), X to the dragon's LEFT (right-handed); the top
of the head is at about y = 1. Triangles are wound counter-clockwise seen from outside here; the KMF
export mirrors x (molynx_assets does the same), which turns that into the game's winding (outward normal
= -(b - a) x (c - a) in KMF axes).

Budget, from the game's own creatures: idle meshes have 200-290 triangles (Bile Demon 212, Knight 290)
and the largest, the Horned Reaper, 406; skins are 128 x 128. The dragon is the size of the Reaper's
budget (about 430 triangles) on three 128 x 128 skins (body, head, wings).

Parts (each a list of triangles over its own vertices; `Part.texture` names its skin):
  trunk   one 8-sided loft from the tail tip to the top of the neck (belly plates on its ventral faces)
  head    skull and snout (6-sided loft, flat palate), two horns, upper fangs
  jaw     lower jaw (4-sided loft) and lower fangs
  legs    5-sided loft hip-knee-ankle-ball, three toes and a back spur each
  wings   double-sided membranes (13 triangles a side) on an arm and three two-bone fingers, a thumb claw
  spikes  double-sided fins along the back of the neck, the back and the tail

UVs mirror across each part's own symmetry plane (u = |angle| / pi round a loft, 0 on the dorsal or front
line, 1 on the ventral or back line), so no loft needs a seam and left and right share texels; both legs
and both wings share one island each, and a membrane's two sides share theirs.

LICENCE: GPL-3.0.
"""
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------------------------------- skeleton

# name, parent, head (bind position), tail (end of its segment, for weights)
_BONES_L = [
    ("root", None, (0, .33, -.04), (0, .42, .00)),
    ("spine1", "root", (0, .42, .01), (0, .56, .02)),
    ("chest", "spine1", (0, .56, .02), (0, .68, -.02)),
    ("neck1", "chest", (0, .68, -.02), (0, .78, -.065)),
    ("neck2", "neck1", (0, .78, -.065), (0, .86, -.05)),
    ("neck3", "neck2", (0, .86, -.05), (0, .91, -.005)),
    ("head", "neck3", (0, .91, -.005), (0, .85, .37)),
    ("jaw", "head", (0, .893, .00), (0, .85, .33)),
    ("tail1", "root", (0, .26, -.12), (0, .16, -.26)),
    ("tail2", "tail1", (0, .16, -.26), (0, .11, -.40)),
    ("tail3", "tail2", (0, .11, -.40), (0, .13, -.54)),
    ("tail4", "tail3", (0, .13, -.54), (0, .24, -.74)),
    ("thighL", "root", (.10, .34, -.03), (.135, .195, .065)),
    ("shinL", "thighL", (.135, .195, .065), (.15, .075, -.03)),
    ("footL", "shinL", (.15, .075, -.03), (.16, .024, .055)),
    ("toesL", "footL", (.16, .024, .055), (.165, .01, .13)),
    ("wing1L", "chest", (.065, .70, -.085), (.27, .83, -.15)),
    ("wing2L", "wing1L", (.27, .83, -.15), (.47, .90, -.17)),
    ("wing3L", "wing2L", (.47, .90, -.17), (.52, .88, -.18)),
    ("f1aL", "wing3L", (.47, .90, -.17), (.66, .98, -.19)),
    ("f1bL", "f1aL", (.66, .98, -.19), (.86, 1.04, -.21)),
    ("f2aL", "wing3L", (.47, .90, -.17), (.66, .80, -.22)),
    ("f2bL", "f2aL", (.66, .80, -.22), (.84, .68, -.26)),
    ("f3aL", "wing3L", (.47, .90, -.17), (.56, .62, -.22)),
    ("f3bL", "f3aL", (.56, .62, -.22), (.60, .40, -.26)),
]


def _mirror_name(n):
    return n[:-1] + "R" if n and n.endswith("L") else n


def _build_bones():
    out = []
    for name, parent, head, tail in _BONES_L:
        out.append((name, parent, np.array(head, float), np.array(tail, float)))
        if name.endswith("L"):
            m = np.array([-1.0, 1, 1])
            out.append((_mirror_name(name), _mirror_name(parent), np.array(head) * m, np.array(tail) * m))
    return out


BONES = _build_bones()
BONE_NAMES = [b[0] for b in BONES]
BONE_INDEX = {n: i for i, n in enumerate(BONE_NAMES)}
PARENT = [None if b[1] is None else BONE_INDEX[b[1]] for b in BONES]
HEAD = np.array([b[2] for b in BONES])
TAIL = np.array([b[3] for b in BONES])
for _i, _p in enumerate(PARENT):        # parents come first: forward kinematics runs in list order
    assert _p is None or _p < _i


# ------------------------------------------------------------------------------------------- parts

@dataclass
class Part:
    name: str
    texture: str
    points: list = field(default_factory=list)     # bind positions
    uv: list = field(default_factory=list)         # island-local (0..1, 0..1); placed by the atlas later
    weights: list = field(default_factory=list)    # {bone index: weight}
    tris: list = field(default_factory=list)       # (a, b, c), counter-clockwise from outside
    island: list = field(default_factory=list)     # island name of each vertex
    tag: list = field(default_factory=list)        # painter hints per vertex (dict)

    def add(self, p, uv, w, island, **tag):
        self.points.append(np.asarray(p, float))
        self.uv.append(tuple(map(float, uv)))
        self.weights.append(_norm_weights(w))
        self.island.append(island)
        self.tag.append(tag)
        return len(self.points) - 1

    def tri(self, a, b, c, inside=None):
        """Add a triangle; with `inside`, wind it so its normal points away from that point."""
        if inside is not None:
            pa, pb, pc = (self.points[i] for i in (a, b, c))
            n = np.cross(pb - pa, pc - pa)
            if np.dot(n, (pa + pb + pc) / 3 - np.asarray(inside, float)) < 0:
                b, c = c, b
        self.tris.append((a, b, c))


def _norm_weights(w):
    if isinstance(w, str):
        return {BONE_INDEX[w]: 1.0}
    w = {BONE_INDEX[k] if isinstance(k, str) else k: float(v) for k, v in w.items() if v > 1e-4}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def auto_weights(p, bones, power=4.0, keep=2):
    """Inverse-distance weights to the nearest `keep` bone segments among `bones`."""
    d = np.array([max(_seg_dist(p, HEAD[BONE_INDEX[b]], TAIL[BONE_INDEX[b]]), 1e-4) for b in bones])
    order = np.argsort(d)[:keep]
    w = {bones[i]: d[i] ** -power for i in order}
    return w


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frames(path, hint):
    """Per point of a polyline: (tangent, dorsal) with dorsal the hint made perpendicular."""
    path = np.asarray(path, float)
    out = []
    for i in range(len(path)):
        a, b = path[max(i - 1, 0)], path[min(i + 1, len(path) - 1)]
        t = _unit(b - a)
        h = np.asarray(hint[i] if np.ndim(hint) == 2 else hint, float)
        d = _unit(h - np.dot(h, t) * t)
        out.append((t, d))
    return out


def loft(part, island, path, radii, sides, hint, weights, *, start_point=None, end_point=None,
         shape=None, v_range=(0.0, 1.0), tag=None, side_axis_sign=1.0):
    """Rings of `sides` vertices round `path` (radii (dorsal, lateral) each), mirrored-u UVs.
    Angle 0 is the dorsal side (`hint`); the ring's lateral axis is t x d (the model's +x for an upward
    loft facing back). `weights(i, point)` -> weights of ring i's vertex. Optional caps: a single point
    before the first ring / after the last. `shape(i, theta) -> (dorsal scale, lateral scale)`."""
    path = np.asarray(path, float)
    frames = _frames(path, hint)
    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    total = seglen[-1]
    extra0 = np.linalg.norm(path[0] - start_point) if start_point is not None else 0.0
    extra1 = np.linalg.norm(end_point - path[-1]) if end_point is not None else 0.0
    full = extra0 + total + extra1
    vmap = lambda s: v_range[0] + (v_range[1] - v_range[0]) * s / full
    rings = []
    for i, (c, (t, d)) in enumerate(zip(path, frames)):
        lat = side_axis_sign * _unit(np.cross(t, d))
        ring = []
        for k in range(sides):
            th = 2 * np.pi * k / sides
            sd, sl = shape(i, th) if shape else (1.0, 1.0)
            p = c + radii[i][0] * sd * np.cos(th) * d + radii[i][1] * sl * np.sin(th) * lat
            u = (th if th <= np.pi else 2 * np.pi - th) / np.pi
            ring.append(part.add(p, (u, vmap(extra0 + seglen[i])), weights(i, p), island,
                                 **(tag(i, th) if tag else {})))
        rings.append(ring)
    for i in range(len(rings) - 1):
        mid = (path[i] + path[i + 1]) / 2
        for k in range(sides):
            a, b = rings[i][k], rings[i][(k + 1) % sides]
            c_, d_ = rings[i + 1][k], rings[i + 1][(k + 1) % sides]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
    if start_point is not None:
        s = part.add(start_point, (0.5, v_range[0]), weights(-1, start_point), island,
                     **(tag(-1, 0.0) if tag else {}))
        inner = path[0] + (path[1] - path[0]) * 0.5
        for k in range(sides):
            part.tri(s, rings[0][k], rings[0][(k + 1) % sides], inside=inner)
    if end_point is not None:
        e = part.add(end_point, (0.5, v_range[1]), weights(len(path), end_point), island,
                     **(tag(len(path), 0.0) if tag else {}))
        inner = path[-1] + (path[-2] - path[-1]) * 0.5
        for k in range(sides):
            part.tri(e, rings[-1][(k + 1) % sides], rings[-1][k], inside=inner)
    return rings


def cone(part, island, base, tip, radius, up, weights, sides=3, uv_box=(0, 0, 1, 1), tag=None):
    """An open cone (its base hidden inside the body)."""
    t = _unit(np.asarray(tip, float) - base)
    d = _unit(np.asarray(up, float) - np.dot(up, t) * t)
    lat = np.cross(t, d)
    u0, v0, u1, v1 = uv_box
    ids = []
    for k in range(sides):
        th = 2 * np.pi * k / sides
        p = np.asarray(base, float) + radius * (np.cos(th) * d + np.sin(th) * lat)
        u = (th if th <= np.pi else 2 * np.pi - th) / np.pi
        ids.append(part.add(p, (u0 + (u1 - u0) * u, v0), weights, island, **(tag or {}), along=0.0))
    tp = part.add(tip, ((u0 + u1) / 2, v1), weights, island, **(tag or {}), along=1.0)
    base = np.asarray(base, float)
    for k in range(sides):
        cen = (part.points[ids[k]] + part.points[ids[(k + 1) % sides]] + part.points[tp]) / 3
        axis_pt = base + np.dot(cen - base, t) * t
        part.tri(ids[k], ids[(k + 1) % sides], tp, inside=axis_pt)
    return ids + [tp]


def fin(part, island, a, b, tip, side, weights_ab, weights_tip, uv_box=(0, 0, 1, 1), tag=None):
    """A double-sided triangle fin (a, b on the surface, tip outward); `side` = its lateral normal."""
    u0, v0, u1, v1 = uv_box
    corners = [(a, (u0, v0), weights_ab, 0.0), (b, (u1, v0), weights_ab, 0.0),
               (tip, ((u0 + u1) / 2, v1), weights_tip, 1.0)]
    one = [part.add(p, uv, w, island, **(tag or {}), along=al) for p, uv, w, al in corners]
    two = [part.add(p, uv, w, island, **(tag or {}), along=al) for p, uv, w, al in corners]
    n = np.cross(np.asarray(b) - a, np.asarray(tip) - a)
    if np.dot(n, side) >= 0:
        part.tri(*one)
        part.tri(two[0], two[2], two[1])
    else:
        part.tri(one[0], one[2], one[1])
        part.tri(*two)


# ------------------------------------------------------------------------------------------- body plan

TRUNK_PATH = np.array([
    (0, .30, -.60), (0, .19, -.50), (0, .15, -.36), (0, .21, -.19),   # tail (its tip curls up)
    (0, .33, -.03), (0, .47, .03), (0, .615, .0),                    # pelvis, belly (bulging), chest
    (0, .725, -.045), (0, .825, -.07), (0, .895, -.035),             # neck: back, then forward (DK1 S)
])
# (dorsal-ventral, lateral) half sizes
TRUNK_RADII = [(.032, .03), (.05, .046), (.068, .062), (.088, .08),
               (.118, .11), (.128, .114), (.11, .102),
               (.1, .085), (.092, .078), (.082, .07)]
TRUNK_TIP = np.array([0, .42, -.66])
TRUNK_TOP = np.array([0, .945, -.015])
TRUNK_BONES = ["root", "spine1", "chest", "neck1", "neck2", "neck3", "tail1", "tail2", "tail3", "tail4"]
BELLY_RINGS = range(1, 10)  # rings whose ventral faces carry belly plates (tail underside too)


def _trunk_hint(path):
    # dorsal: the back (-z) along the upright body, up (+y) along the tail
    return [(0, 1, -1.0) if p[2] < -0.1 else (0, 0.35, -1.0) for p in path]


def trunk_shape(i, th):
    c = np.cos(th)
    # flatter, broader belly; the back ridged a little
    if c < -0.3:
        return 0.86, 1.08
    return 1.0, 1.0


def build_trunk(parts):
    part = Part("trunk", "Dragon_Body")
    path = TRUNK_PATH
    w = lambda i, p: auto_weights(p, TRUNK_BONES)
    loft(part, "trunk", path, TRUNK_RADII, 8, _trunk_hint(path), w, start_point=TRUNK_TIP,
         end_point=TRUNK_TOP, shape=trunk_shape,
         tag=lambda i, th: {"theta": th, "ring": i})
    parts.append(part)
    return part


HEAD_RINGS = [  # centre, lateral, up, down (to the lip line)
    ((0, .952, -.035), .066, .056, .054),     # broad jaw muscles behind the eyes
    ((0, .958, .055), .078, .072, .056),      # the brow: the deepest part of the head
    ((0, .926, .165), .058, .054, .044),
    ((0, .9, .25), .05, .052, .038),          # a broad nose with a raised, bulbous top
]
SNOUT_TIP = np.array([0, .9, .283])           # blunt: the nose ends in a flat, rounded front
SKULL_BACK = np.array([0, .965, -.085])
HEAD_AXIS_HINT = (0, 1, 0)


def _head_section(c, lat, up, down, axis):
    """6 points: top, upper-left, lip-left, palate, lip-right, upper-right (theta 0, 60 .. 300)."""
    t = _unit(axis)
    u = _unit(np.array(HEAD_AXIS_HINT) - np.dot(HEAD_AXIS_HINT, t) * t)
    x = np.cross(u, t)  # +x (left) for a forward-looking head: up x forward = left? (0,1,0)x(0,0,1) = (1,0,0)
    c = np.asarray(c, float)
    return [c + up * u,
            c + .86 * lat * x + .55 * up * u,
            c + lat * x - .55 * down * u,
            c - .30 * down * u,
            c - lat * x - .55 * down * u,
            c - .86 * lat * x + .55 * up * u]


def build_head(parts):
    part = Part("head", "Dragon_Head")
    centres = [np.array(r[0], float) for r in HEAD_RINGS]
    axes = [centres[min(i + 1, 3)] - centres[max(i - 1, 0)] for i in range(4)]
    length = np.linalg.norm(SNOUT_TIP - SKULL_BACK)
    rings, lips = [], []
    for i, ((c, lat, up, down), ax) in enumerate(zip(HEAD_RINGS, axes)):
        pts = _head_section(c, lat, up, down, ax)
        v = np.linalg.norm(np.asarray(c) - SKULL_BACK) / length
        ring = []
        for k, p in enumerate(pts):
            u = [0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3][k]
            ring.append(part.add(p, (u, v), "head", "skull", ring=i, k=k))
        rings.append(ring)
        lips.append((pts[2], pts[4]))
    inner = [np.mean([part.points[j] for j in r], 0) for r in rings]
    for i in range(3):
        mid = (inner[i] + inner[i + 1]) / 2
        for k in range(6):
            a, b = rings[i][k], rings[i][(k + 1) % 6]
            c_, d_ = rings[i + 1][k], rings[i + 1][(k + 1) % 6]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
    tip = part.add(SNOUT_TIP, (0.5, 1.0), "head", "skull", ring=4, k=-1)
    for k in range(6):
        part.tri(rings[3][k], rings[3][(k + 1) % 6], tip, inside=inner[3] - (SNOUT_TIP - inner[3]))
    back = part.add(SKULL_BACK, (0.5, 0.0), "head", "skull", ring=-1, k=-1)
    for k in range(6):
        part.tri(rings[0][(k + 1) % 6], rings[0][k], back, inside=inner[0] + (inner[0] - SKULL_BACK))
    # horns: from the back of the skull, swept back, out and a little up
    for s in (1, -1):
        base = np.array([.04 * s, .99, -.02])
        tip_ = base + np.array([.085 * s, .05, -.095])
        cone(part, "horn", base, tip_, .02, (0, 1, 0.3), "head", sides=3)
    # heavy brow ridges overhanging the eyes, low at the front: the DK1 dragon's scowl (its portrait)
    for s in (1, -1):
        a = np.array([.06 * s, 1.012, .0])
        b = np.array([.052 * s, .99, .115])
        t = np.array([.092 * s, 1.035, -.03])
        fin(part, "brow", a, b, t, (s, 0.6, 0), "head", "head")
    # cheek frills: two spikes on each side fanning back from behind the jaw (the portrait's ear fins)
    for s in (1, -1):
        for base, d in (((.062 * s, .935, -.035), (.075 * s, -.012, -.085)),
                        ((.06 * s, .965, -.06), (.085 * s, .03, -.075))):
            base = np.array(base)
            d = np.array(d)
            fin(part, "spike", base + (0, -.012, .012), base + (0, .012, -.012), base + d, (s, .2, 0),
                "head", "head")
    # the crown: three spikes fanning back from the back of the skull (DK1 sprite 6802)
    for base, d in (((0, .99, -.055), (0, .045, -.085)), ((.032, .985, -.04), (.05, .04, -.07)),
                    ((-.032, .985, -.04), (-.05, .04, -.07))):
        base = np.array(base)
        d = np.array(d)
        t = _unit(d)
        side = np.cross(t, (0, 0, 1.0)) if abs(t[2]) < .9 else np.array([1.0, 0, 0])
        a = base - t * .012 + np.array([0, -.012, .0])
        b = base + np.array([0, -.012, .022])
        fin(part, "spike", a, b, base + d, (1, 0, 0) if abs(base[0]) < 1e-6 else (np.sign(base[0]), .3, 0),
            "head", "head")
    # upper fangs (double-sided)
    for s in (1, -1):
        zs = [HEAD_RINGS[i][0][2] for i in (1, 2, 3)]
        for z, l in ((.225, .05), (.16, .036), (.095, .03)):
            c = np.array([0, .0, z])
            c[1] = np.interp(z, zs, [lips[1][0][1], lips[2][0][1], lips[3][0][1]]) - .002
            c[0] = s * np.interp(z, zs, [abs(lips[1][0][0]), abs(lips[2][0][0]), abs(lips[3][0][0])]) * .93
            fin(part, "fang", c + (0, 0, -.009), c + (0, 0, .009), c + (0, -l, .004), (s, 0, 0), "head", "head")
    parts.append(part)
    return part, rings


def build_jaw(parts, head_rings):
    part = Part("jaw", "Dragon_Head")
    hp = None
    rings = []
    depths = [.034, .048, .04, .03]
    for i in range(4):
        # the lips of the upper ring (closed mouth: they meet), the tongue groove, the keel
        ring_ids = head_rings[i]
        P = [None] * 4
        lipL = np.asarray(_ring_point(parts, "head", ring_ids[2]))
        lipR = np.asarray(_ring_point(parts, "head", ring_ids[4]))
        mid = (lipL + lipR) / 2
        P[0] = mid + (0, -.012, 0)            # tongue (inside)
        P[1] = lipL + (lipR - lipL) * .04 + (0, -.002, 0)
        P[2] = mid + (0, -depths[i] - .01, 0)  # keel
        P[3] = lipR + (lipL - lipR) * .04 + (0, -.002, 0)
        v = i / 4.0
        ring = []
        for k, p in enumerate(P):
            u = [0, .5, 1, .5][k]
            ring.append(part.add(p, (u, v), "jaw", "jaw", ring=i, k=k))
        rings.append(ring)
    # inside each ring: halfway between the lips and the keel (the tongue sits only just below the lips,
    # so the ring's mean can lie above it near the chin)
    inner = [((part.points[r[1]] + part.points[r[3]]) / 2 + part.points[r[2]]) / 2 for r in rings]
    for i in range(3):
        mid = (inner[i] + inner[i + 1]) / 2
        for k in range(4):
            a, b = rings[i][k], rings[i][(k + 1) % 4]
            c_, d_ = rings[i + 1][k], rings[i + 1][(k + 1) % 4]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
    chin_p = np.array([0, .866, .262])
    chin = part.add(chin_p, (.5, 1.0), "jaw", "jaw", ring=4, k=-1)
    for k in range(4):
        part.tri(rings[3][k], rings[3][(k + 1) % 4], chin, inside=inner[3] - (chin_p - inner[3]))
    back_p = np.array([0, .885, -.05])
    back = part.add(back_p, (.5, 0.0), "jaw", "jaw", ring=-1, k=-1)
    for k in range(4):
        part.tri(rings[0][(k + 1) % 4], rings[0][k], back, inside=inner[0] + (inner[0] - back_p))
    # lower fangs
    for s in (1, -1):
        for r, dz, l in ((2, .01, .036), (1, .03, .026)):
            p = _ring_point(part, None, rings[r][1 if s > 0 else 3])
            c = np.asarray(p) + (-.004 * s, 0, dz)
            fin(part, "fang", c + (0, 0, -.008), c + (0, 0, .008), c + (0, l, .0), (s, 0, 0), "jaw", "jaw")
    parts.append(part)
    return part


def _ring_point(parts, name, idx):
    part = parts if isinstance(parts, Part) else next(p for p in parts if p.name == name)
    return part.points[idx]


LEG_PATH = np.array([(.095, .37, -.03), (.115, .28, .01), (.135, .195, .065), (.15, .075, -.03),
                     (.16, .024, .055)])
LEG_RADII = [(.1, .086), (.098, .084), (.062, .056), (.04, .036), (.03, .042)]
LEG_BONES = ["thighL", "shinL", "footL", "toesL"]


def build_legs(parts):
    part = Part("legs", "Dragon_Body")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        path = LEG_PATH * m
        bones = LEG_BONES if s > 0 else [_mirror_name(b) for b in LEG_BONES]
        fwd = [(0, 0, 1.0), (0, 0, 1.0), (0, 0.2, 1.0), (0, 0.6, 1.0), (0, 1.0, 0.3)]

        def w(i, p, bones=bones):
            if i <= 1:
                return {bones[0]: 1.0}
            return auto_weights(p, bones)
        start = len(part.points)
        loft(part, "leg", path, LEG_RADII, 6, fwd, w, end_point=path[-1] + (0, -.012, .03),
             tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
        # toes and the back spur
        ball = path[-1] + (0, -.004, .0)
        toe = bones[3]
        for dx, l in ((0.0, .085), (.45, .07), (-.45, .07)):
            d = _unit(np.array([dx * s, -.08, 1.0]))
            cone(part, "toe", ball + np.array([dx * s * .02, 0, .01]), ball + d * l + (0, -.012, 0), .016,
                 (0, 1, 0), toe, sides=3, tag={"claw": True})
        ankle = path[3]
        cone(part, "toe", ankle + (0, -.01, -.02), ankle + (0, -.07, -.06), .014, (0, 0, -1), bones[2], sides=3,
             tag={"claw": True})
        if s < 0:   # mirrored lofts come out inside-out when the lateral axis flips: keep them outward
            pass
    parts.append(part)
    return part


# membrane points in the spread (bind) pose, left wing
WING_POINTS = {
    "S": (.065, .70, -.085), "E": (.27, .83, -.15), "W": (.47, .90, -.17),
    "M1": (.66, .98, -.19), "F1": (.86, 1.04, -.21),
    "M2": (.66, .80, -.22), "F2": (.84, .68, -.26),
    "M3": (.56, .62, -.22), "F3": (.60, .40, -.26),
    "C12": (.70, .86, -.23), "C23": (.62, .56, -.25), "C3B": (.36, .42, -.21),
    "B": (.06, .44, -.12),
}
WING_WEIGHTS = {
    "S": {"wing1L": .6, "chest": .4}, "E": {"wing1L": .4, "wing2L": .6}, "W": {"wing3L": 1},
    "M1": {"f1aL": .5, "f1bL": .5}, "F1": {"f1bL": 1}, "M2": {"f2aL": .5, "f2bL": .5}, "F2": {"f2bL": 1},
    "M3": {"f3aL": .5, "f3bL": .5}, "F3": {"f3bL": 1},
    "C12": {"f1bL": .5, "f2bL": .5}, "C23": {"f2bL": .5, "f3bL": .5},
    "C3B": {"f3bL": .55, "root": .45}, "B": {"root": .6, "spine1": .4},
}
WING_TRIS = [("S", "E", "W"), ("S", "W", "B"),
             ("W", "M3", "B"), ("M3", "C3B", "B"), ("M3", "F3", "C3B"),
             ("W", "M2", "M3"), ("M2", "C23", "M3"), ("M2", "F2", "C23"), ("M3", "C23", "F3"),
             ("W", "M1", "M2"), ("M1", "C12", "M2"), ("M1", "F1", "C12"), ("M2", "C12", "F2")]


def _wing_uv():
    """Planar UVs of the spread membrane (its best-fit plane), 0..1 on the longer side."""
    P = np.array(list(WING_POINTS.values()), float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    a, b = vt[0], vt[1]
    q = np.stack([(P - c) @ a, (P - c) @ b], 1)
    lo, hi = q.min(0), q.max(0)
    span = (hi - lo).max()
    q = (q - lo) / span
    return {k: tuple(q[i]) for i, k in enumerate(WING_POINTS)}, (hi - lo) / span


WING_UV, WING_UV_EXTENT = _wing_uv()


def build_wings(parts):
    part = Part("wings", "Dragon_Wing")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        names = {k: (_mirror_name(b) if s < 0 else b) for k, w in WING_WEIGHTS.items() for b in w}
        pts = {k: np.array(v) * m for k, v in WING_POINTS.items()}
        # outward normal of the spread membrane's back face: -z (behind), front face +z
        for face, side in (("back", -1.0), ("front", 1.0)):
            ids = {}
            for k, p in pts.items():
                w = {(_mirror_name(b) if s < 0 else b): x for b, x in WING_WEIGHTS[k].items()}
                ids[k] = part.add(p, WING_UV[k], w, "membrane", face=face, key=k)
            for a, b, c in WING_TRIS:
                pa, pb, pc = pts[a], pts[b], pts[c]
                n = np.cross(pb - pa, pc - pa)
                if (n[2] * side) >= 0:
                    part.tri(ids[a], ids[b], ids[c])
                else:
                    part.tri(ids[a], ids[c], ids[b])
        # the arm: a thin 3-sided tube shoulder-elbow-wrist and a thumb claw
        arm = np.array([pts["S"], pts["E"], pts["W"]])
        b1, b2, b3 = ("wing1L", "wing2L", "wing3L") if s > 0 else ("wing1R", "wing2R", "wing3R")
        aw = [{b1: 1.0, "chest": 0.0}, {b1: .4, b2: .6}, {b3: 1.0}]
        loft(part, "arm", arm, [(.024, .024), (.02, .02), (.016, .016)], 4, (0, 0, 1),
             lambda i, p, aw=aw: aw[min(max(i, 0), 2)],
             end_point=pts["W"] + (pts["W"] - pts["E"]) * .12, tag=lambda i, th: {"arm": True},
             side_axis_sign=1.0)
        cone(part, "toe", pts["W"] + (0, .005, .0), pts["W"] + np.array([.0 * s, .06, .045]), .011, (0, 0, 1),
             b3, sides=3, tag={"claw": True})
    parts.append(part)
    return part


# dorsal spikes: (ring index along the trunk path as a float, length, lean back)
SPIKES = [(1.0, .03), (1.6, .04), (2.3, .05), (3.0, .055), (3.7, .05),
          (4.6, .045), (5.4, .05), (6.3, .06), (7.0, .075), (7.6, .085), (8.25, .08), (8.9, .075)]


def build_spikes(parts):
    part = Part("spikes", "Dragon_Head")
    path = TRUNK_PATH
    hints = _trunk_hint(path)
    frames = _frames(path, hints)
    for f, length in SPIKES:
        i = int(np.floor(f))
        u = f - i
        j = min(i + 1, len(path) - 1)
        c = path[i] * (1 - u) + path[j] * u
        t = _unit(frames[i][0] * (1 - u) + frames[j][0] * u)
        d = _unit(frames[i][1] * (1 - u) + frames[j][1] * u)
        r = TRUNK_RADII[i][0] * (1 - u) + TRUNK_RADII[j][0] * u
        base = c + d * r * .9
        half = max(.018, length * .45)
        a, b = base - t * half, base + t * half
        tip = base + d * length - t * length * .55     # leaning toward the tail
        w = auto_weights(base, TRUNK_BONES)
        fin(part, "spike", a, b, tip, (1, 0, 0), w, w)
    parts.append(part)
    return part


def build_dragon():
    """-> list of Parts in bind pose (wings spread)."""
    parts = []
    build_trunk(parts)
    head, rings = build_head(parts)
    build_jaw(parts, rings)
    build_legs(parts)
    build_wings(parts)
    build_spikes(parts)
    return parts


def counts(parts):
    return {p.name: (len(p.points), len(p.tris)) for p in parts}


# ------------------------------------------------------------------------------------------- posing

def _rotmat(rv):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(rv).as_matrix()


def forward(local, root_offset=(0, 0, 0)):
    """local: (B, 3, 3) local rotations (bind = identity, world-aligned frames).
    -> (world rotations (B,3,3), world heads (B,3))."""
    B = len(BONES)
    Rw = np.zeros((B, 3, 3))
    Pw = np.zeros((B, 3))
    for i in range(B):
        p = PARENT[i]
        if p is None:
            Rw[i] = local[i]
            Pw[i] = HEAD[i] + np.asarray(root_offset, float)
        else:
            Rw[i] = Rw[p] @ local[i]
            Pw[i] = Pw[p] + Rw[p] @ (HEAD[i] - HEAD[p])
    return Rw, Pw


def skin(part, Rw, Pw):
    P = np.array(part.points)
    out = np.zeros_like(P)
    for v, w in enumerate(part.weights):
        acc = np.zeros(3)
        for b, x in w.items():
            acc += x * (Pw[b] + Rw[b] @ (P[v] - HEAD[b]))
        out[v] = acc
    return out


class Skinner:
    """Vectorised linear blend skinning of all parts."""

    def __init__(self, parts):
        self.parts = parts
        self.sizes = [len(p.points) for p in parts]
        P = np.concatenate([np.array(p.points) for p in parts])
        W = np.zeros((len(P), len(BONES)))
        k = 0
        for p in parts:
            for w in p.weights:
                for b, x in w.items():
                    W[k, b] = x
                k += 1
        self.P, self.W = P, W
        self.rel = P[:, None, :] - HEAD[None, :, :]       # (V, B, 3)

    def __call__(self, Rw, Pw):
        moved = np.einsum("bij,vbj->vbi", Rw, self.rel) + Pw[None]
        out = np.einsum("vb,vbi->vi", self.W, moved)
        return np.split(out, np.cumsum(self.sizes)[:-1])
