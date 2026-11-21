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

Budget, from the game's own creatures: idle meshes have 190-290 triangles (Imp 187, Salamander 218,
Bile Demon 212, Dark Angel 238 with its wings, Fairy 280, Knight 290); skins are 128 x 128. Dragon v2
(Soles' community feedback, 15 Nov: "feels too smooth compared to the other creatures", "could use less
polygons") is cut from 574 to the game's own budget: faceted 6- and 4-sided lofts, every ring that lay on
a straight line removed, the wings' membranes fanned from the wrist (scalloped hem kept) and no arm tube.

Parts (each a list of triangles over its own vertices; `Part.texture` names its skin):
  trunk   one 6-sided loft from the tail tip to the top of the neck (belly plates on its ventral faces)
  head    skull and snout (6-sided loft, 3 rings, flat palate), raised 3D nostrils, two horns, brows,
          cheek fins, a crown spike, two single-sided upper fangs
  jaw     lower jaw (3-sided loft: lips and keel; the tongue is its top face, edged with painted teeth),
          two single-sided lower fangs
  legs    4-sided loft hip-knee-ankle, a wedge foot to the ball, three toe claws (2-faced tents) each
  wings   double-sided membranes (8 triangles a side, fanned from the wrist) on three fingers, a thumb claw
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


class Skeleton:
    """Another creature's skeleton (same conventions as BONES): `forward`, `Skinner` and `Part` take one
    to work on it instead of the dragon's."""

    def __init__(self, bones):
        self.bones = bones
        self.names = [b[0] for b in bones]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parent = [None if b[1] is None else self.index[b[1]] for b in bones]
        self.head = np.array([b[2] for b in bones])
        self.tail = np.array([b[3] for b in bones])
        assert all(p is None or p < i for i, p in enumerate(self.parent))


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
    bone_index = None                              # a subclass's own {bone name: index} (default: the dragon's)
    post = None                                    # posed points -> posed points after skinning (Skinner)

    def add(self, p, uv, w, island, **tag):
        self.points.append(np.asarray(p, float))
        self.uv.append(tuple(map(float, uv)))
        self.weights.append(_norm_weights(w, self.bone_index))
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


def _norm_weights(w, index=None):
    index = BONE_INDEX if index is None else index
    if isinstance(w, str):
        return {index[w]: 1.0}
    w = {index[k] if isinstance(k, str) else k: float(v) for k, v in w.items() if v > 1e-4}
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
         shape=None, v_range=(0.0, 1.0), tag=None, side_axis_sign=1.0, start_inside=None):
    """Rings of `sides` vertices round `path` (radii (dorsal, lateral) each), mirrored-u UVs.
    Angle 0 is the dorsal side (`hint`); the ring's lateral axis is t x d (the model's +x for an upward
    loft facing back). `weights(i, point)` -> weights of ring i's vertex. Optional caps: a single point
    before the first ring / after the last. `shape(i, theta) -> (dorsal scale, lateral scale)`.
    `start_inside`: a point inside the start cap, to wind its triangles by (default: halfway between the
    first two rings, which is outside a long cap that bends away from the loft's first segment)."""
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
        inner = path[0] + (path[1] - path[0]) * 0.5 if start_inside is None else np.asarray(start_inside, float)
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
    (0, .17, -.42), (0, .215, -.205),                                # tail (it lies on the ground)
    (0, .335, -.03), (0, .48, .03), (0, .625, -.005),                # pelvis, belly (bulging), chest
    (0, .785, -.065), (0, .895, -.035),                              # neck: back, then forward (DK1 S)
])
# (dorsal-ventral, lateral) half sizes
TRUNK_RADII = [(.062, .058), (.092, .084),
               (.122, .114), (.13, .118), (.11, .102),
               (.098, .082), (.084, .072)]
TRUNK_TIP = np.array([0, .33, -.67])           # the tail's tip, curled up off the floor
TRUNK_TOP = np.array([0, .985, .0])           # the neck's cap, steep and up inside the head
PELVIS_RING = 2
TRUNK_SIDES = 6
TRUNK_BONES = ["root", "spine1", "chest", "neck1", "neck2", "neck3", "tail1", "tail2", "tail3", "tail4"]
BELLY_RINGS = range(0, 7)  # rings whose ventral faces carry belly plates (tail underside too)


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
    # v2d: the tail's tip cone is wound from a point on its own axis. It rises from the first ring toward
    # the curled-up tip, so the old reference (halfway to the second ring) lay above its two dorsal faces:
    # they were wound inward, the game culled them, and the tip showed a hole along its top ("the minor
    # missing texture on the end of his tail") and read as a flat needle
    loft(part, "trunk", path, TRUNK_RADII, TRUNK_SIDES, _trunk_hint(path), w, start_point=TRUNK_TIP,
         end_point=TRUNK_TOP, shape=trunk_shape, start_inside=path[0] + (TRUNK_TIP - path[0]) * 0.25,
         tag=lambda i, th: {"theta": th, "ring": i})
    parts.append(part)
    return part


HEAD_RINGS = [  # centre, lateral, up, down (to the lip line)
    ((0, .952, -.035), .066, .056, .054),     # broad jaw muscles behind the eyes
    ((0, .958, .06), .078, .072, .056),       # the brow: the deepest part of the head
    ((0, .882, .32), .042, .038, .03),        # the long snout (v1's middle ring lay on this line: cut)
]
SNOUT_TIP = np.array([0, .872, .368])
NOSTRILS = [(np.array([.019, .905, .332]), np.array([.35, .55, .76]))]   # left; mirrored for the right
SKULL_BACK = np.array([0, .965, -.085])
HEAD_AXIS_HINT = (0, 1, 0)
JAW_DEPTHS = [.034, .048, .03]                # keel below the lips, per head ring


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


def _lip_at(lips, z, side):
    """Point on the (left, side=+1, or right) lip line at depth z, interpolated between the rings."""
    zs = [HEAD_RINGS[i][0][2] for i in range(len(HEAD_RINGS))]
    pts = np.array([l[0 if side > 0 else 1] for l in lips])
    return np.array([np.interp(z, zs, pts[:, 0]), np.interp(z, zs, pts[:, 1]), z])


def build_head(parts):
    part = Part("head", "Dragon_Head")
    n = len(HEAD_RINGS)
    centres = [np.array(r[0], float) for r in HEAD_RINGS]
    axes = [centres[min(i + 1, n - 1)] - centres[max(i - 1, 0)] for i in range(n)]
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
    for i in range(n - 1):
        mid = (inner[i] + inner[i + 1]) / 2
        for k in range(6):
            a, b = rings[i][k], rings[i][(k + 1) % 6]
            c_, d_ = rings[i + 1][k], rings[i + 1][(k + 1) % 6]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
    tip = part.add(SNOUT_TIP, (0.5, 1.0), "head", "skull", ring=n, k=-1)
    for k in range(6):
        part.tri(rings[-1][k], rings[-1][(k + 1) % 6], tip, inside=inner[-1] - (SNOUT_TIP - inner[-1]))
    back = part.add(SKULL_BACK, (0.5, 0.0), "head", "skull", ring=-1, k=-1)
    for k in range(6):
        part.tri(rings[0][(k + 1) % 6], rings[0][k], back, inside=inner[0] + (inner[0] - SKULL_BACK))
    # nostrils: two raised square tubes at the end of the snout, as v1 (Soles likes them), each capped by a
    # dark hole: 4-sided, 10 triangles each (v1's sunk hole fan becomes a flat 2-triangle cap)
    for s in (1, -1):
        for base, d in NOSTRILS:
            base = base * (s, 1, 1)
            d = _unit(d * (s, 1, 1))
            up = np.array([0, 1.0, 0])
            side = _unit(np.cross(d, up))
            up2 = np.cross(side, d)
            sink = base - d * .014
            rim = base + d * .017
            ids, top = [], []
            for k in range(4):
                th = np.pi / 4 + k * np.pi / 2
                off = np.cos(th) * side + np.sin(th) * up2
                ids.append((part.add(sink + off * .02, (k / 4, 0), "head", "nostril", along=0.0),
                            part.add(rim + off * .019, (k / 4, .6), "head", "nostril", along=0.55)))
                top.append(part.add(rim + off * .019 - d * .003, [(.25, .75), (.75, .75), (.75, 1.0), (.25, 1.0)][k],
                                    "head", "nostril", along=1.0))
            for k in range(4):
                a0, a1 = ids[k]
                b0, b1 = ids[(k + 1) % 4]
                part.tri(a0, b0, a1, inside=base - d * .01)
                part.tri(b0, b1, a1, inside=base - d * .01)
            part.tri(top[0], top[1], top[2], inside=base - d * .03)
            part.tri(top[0], top[2], top[3], inside=base - d * .03)
    # horns: from the back of the skull, swept back, out and a little up
    for s in (1, -1):
        base = np.array([.04 * s, .99, -.02])
        tip_ = base + np.array([.085 * s, .05, -.095])
        cone(part, "horn", base, tip_, .022, (0, 1, 0.3), "head", sides=3)
    # heavy brow ridges overhanging the eyes, low at the front: the DK1 dragon's scowl (its portrait)
    for s in (1, -1):
        a = np.array([.06 * s, 1.012, .0])
        b = np.array([.052 * s, .99, .115])
        t = np.array([.092 * s, 1.035, -.03])
        fin(part, "brow", a, b, t, (s, 0.6, 0), "head", "head")
    # cheek frills: one spike on each side fanning back from behind the jaw (the portrait's ear fins)
    for s in (1, -1):
        base, d = np.array((.06 * s, .955, -.05)), np.array((.09 * s, .012, -.085))
        fin(part, "spike", base + (0, -.016, .014), base + (0, .014, -.014), base + d, (s, .2, 0),
            "head", "head")
    # the crown: one spike back from the back of the skull (DK1 sprite 6802)
    base, d = np.array((0, .99, -.055)), np.array((0, .05, -.09))
    t = _unit(d)
    fin(part, "spike", base - t * .012 + np.array([0, -.012, .0]), base + np.array([0, -.012, .022]), base + d,
        (1, 0, 0), "head", "head")
    # upper fangs: one big one on each side, a single triangle facing out (the inside of the mouth never
    # shows it edge-on at the game's distance); the rest of the row is painted (v2b: the two smaller rear
    # fangs went to the neck spikes)
    for s in (1, -1):
        for z, l in ((.205, .05),):
            c = _lip_at(lips, z, s) + (0, -.002, 0)
            c[0] *= .95
            fang(part, c, c + (0, -l, .004), s, "head")
    parts.append(part)
    return part, rings


def build_jaw(parts, head_rings):
    """The lower jaw: a 3-sided loft (left lip, keel, right lip) whose top face is the tongue; that face
    has its own vertices (u = 0, the mouth's red) so the lips keep their colour."""
    part = Part("jaw", "Dragon_Head")
    n = len(head_rings)
    rings, tongue = [], []
    for i in range(n):
        ring_ids = head_rings[i]
        lipL = np.asarray(_ring_point(parts, "head", ring_ids[2]))
        lipR = np.asarray(_ring_point(parts, "head", ring_ids[4]))
        mid = (lipL + lipR) / 2
        P = [lipL + (lipR - lipL) * .04 + (0, -.002, 0),
             mid + (0, -JAW_DEPTHS[i] - .01, 0),           # keel
             lipR + (lipL - lipR) * .04 + (0, -.002, 0)]
        v = i / float(n)
        ring = [part.add(p, (u, v), "jaw", "jaw", ring=i, k=k) for k, (p, u) in enumerate(zip(P, (.5, 1, .5)))]
        rings.append(ring)
        tongue.append([part.add(P[0], (0.0, v), "jaw", "jaw", ring=i, k=3),
                       part.add(P[2], (0.3, v), "jaw", "jaw", ring=i, k=4)])
    inner = [((part.points[r[0]] + part.points[r[2]]) / 2 + part.points[r[1]]) / 2 for r in rings]
    for i in range(n - 1):
        mid = (inner[i] + inner[i + 1]) / 2
        for k in range(2):
            a, b = rings[i][k], rings[i][k + 1]
            c_, d_ = rings[i + 1][k], rings[i + 1][k + 1]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
        a, b = tongue[i]
        c_, d_ = tongue[i + 1]
        part.tri(a, b, c_, inside=mid)
        part.tri(b, d_, c_, inside=mid)
    chin_p = np.array([0, .856, .342])
    chin = part.add(chin_p, (.5, 1.0), "jaw", "jaw", ring=n, k=-1)
    chin_t = part.add(chin_p, (0.15, 1.0), "jaw", "jaw", ring=n, k=-1)
    last = inner[-1] - (chin_p - inner[-1])
    part.tri(rings[-1][0], rings[-1][1], chin, inside=last)
    part.tri(rings[-1][1], rings[-1][2], chin, inside=last)
    part.tri(tongue[-1][1], tongue[-1][0], chin_t, inside=last)
    # (no back cap: the jaw's back end sits up inside the neck and the skull at every opening of the mouth)
    # one lower fang each side, single triangles facing out (the rest of the row is painted on the lips
    # and along both edges of the tongue face, seen when it drops)
    for s, k in ((1, 0), (-1, 2)):
        p0 = np.asarray(part.points[rings[1][k]]) * .45 + np.asarray(part.points[rings[2][k]]) * .55
        c = p0 + (-.003 * s, -.004, 0)
        fang(part, c, c + (0, .032, .002), s, "jaw")
    parts.append(part)
    return part


def fang(part, root, tip, side, weights, half=.009):
    """A single-sided fang: one triangle from a root edge along the jaw to its tip, facing outward
    (toward +x for side 1, -x for side -1). UVs as `fin`'s, island 'fang'."""
    root = np.asarray(root, float)
    a = part.add(root + (0, 0, -half), (0.0, 0.0), weights, "fang", along=0.0)
    b = part.add(root + (0, 0, half), (1.0, 0.0), weights, "fang", along=0.0)
    c = part.add(tip, (0.5, 1.0), weights, "fang", along=1.0)
    part.tri(a, b, c, inside=root - np.array([side * .03, 0, 0]))


def _ring_point(parts, name, idx):
    part = parts if isinstance(parts, Part) else next(p for p in parts if p.name == name)
    return part.points[idx]


LEG_PATH = np.array([(.115, .28, .01), (.135, .195, .065), (.15, .075, -.03)])    # thigh, knee, ankle
LEG_RADII = [(.112, .1), (.07, .066), (.052, .05)]   # a diamond: 4 sides, sized up
LEG_HIP = np.array([.095, .43, -.03])        # the thigh's cap, up inside the body
LEG_BALL = np.array([.16, .026, .07])        # the foot: a wedge from the ankle to the ball of the toes
LEG_BALL_HALF = .034
LEG_BONES = ["thighL", "shinL", "footL", "toesL"]
LEG_SIDES = 4


def claw(part, base, tip, radius, weights):
    """A toe claw as a two-faced tent (its open underside lies on the floor): a ridge on top and two
    flanks, 2 triangles where a 3-sided cone takes 3. UVs as `cone`'s (along 0 at the base, 1 at the tip)."""
    base = np.asarray(base, float)
    t = _unit(np.asarray(tip, float) - base)
    side = _unit(np.cross(t, (0, 1.0, 0)))
    up = np.cross(side, t)
    a = part.add(base + side * radius, (0.0, 0.0), weights, "toe", claw=True, along=0.0)
    r = part.add(base + up * radius * 1.1, (0.5, 0.0), weights, "toe", claw=True, along=0.0)
    b = part.add(base - side * radius, (1.0, 0.0), weights, "toe", claw=True, along=0.0)
    tp = part.add(tip, (0.5, 1.0), weights, "toe", claw=True, along=1.0)
    below = base - up * radius
    part.tri(a, r, tp, inside=below)
    part.tri(r, b, tp, inside=below)


def build_legs(parts):
    part = Part("legs", "Dragon_Body")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        path = LEG_PATH * m
        bones = LEG_BONES if s > 0 else [_mirror_name(b) for b in LEG_BONES]
        fwd = [(0, 0, 1.0), (0, 0.2, 1.0), (0, 0.6, 1.0), (0, 1.0, 0.3)]

        def w(i, p, bones=bones):
            if i <= 0:
                return {bones[0]: 1.0}
            return auto_weights(p, bones)
        # Soles, 13 Nov: "ensure the legs don't have this missing hollow part" - the thigh tube is capped
        # with a point up inside the body, so a swung thigh never shows its inside
        rings = loft(part, "leg", path, LEG_RADII, LEG_SIDES, fwd, w, start_point=LEG_HIP * m,
                     tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
        # the foot: a wedge from the ankle ring to the ball, as broad as the toes (loft's own end cap judges
        # its winding from the shin, which bends the other way: wound here from the foot's own centre)
        ballp = LEG_BALL * m
        ring = rings[-1]
        R = np.array([part.points[i] for i in ring])
        fr, bk = ring[0], ring[2]                                   # theta 0 (front) and pi (back)
        lat, med = (ring[1], ring[3]) if (R[1][0] - path[-1][0]) * s > 0 else (ring[3], ring[1])
        bl = part.add(ballp + (LEG_BALL_HALF * s, 0, -.004), (0.25, 1.0), w(len(path), ballp), "leg", theta=0.0,
                      ring=len(path), side=s)
        bm = part.add(ballp + (-LEG_BALL_HALF * s, 0, -.004), (0.25, 1.0), w(len(path), ballp), "leg", theta=0.0,
                      ring=len(path), side=s)
        inner = np.mean([part.points[i] for i in (fr, bk, lat, med, bl, bm)], 0)
        for a, b, c in ((fr, lat, bl), (fr, bl, bm), (fr, bm, med), (bk, bl, lat), (bk, bm, bl), (bk, med, bm)):
            part.tri(a, b, c, inside=inner)
        # three toe claws from the ball's corners and middle
        toe = bones[3]
        for base, dx, l in ((ballp, 0.0, .085), (part.points[bl], .4, .07), (part.points[bm], -.4, .07)):
            d = _unit(np.array([dx * s, -.08, 1.0]))
            base = np.asarray(base) + (0, -.002, -.006)
            claw(part, base, base + d * l + (0, -.01, 0), .018, toe)
    parts.append(part)
    return part


# membrane points in the spread (bind) pose, left wing: shoulder, elbow, wrist, the three finger tips,
# the hem's cusps between them and the membrane's root on the flank. v2: the fingers' middle joints are
# no longer vertices (the membrane fans from the wrist), the hem keeps its scallops.
_WING_FRAME = {
    "S": (.065, .70, -.085), "E": (.27, .83, -.15), "W": (.47, .90, -.17),
    "F1": (.86, 1.04, -.21), "F2": (.84, .68, -.26), "F3": (.60, .40, -.26),
    "B": (.06, .44, -.12),
}
# v2c: each hem cusp is not skinned but built in every frame from its panel's corners (WingHem): on the
# panel's own plane (wrist, finger tip, next finger tip or the flank), so a panel is always one flat sheet
# whatever the fingers do, at `depth` x the chord between the two corners in from the chord (a gentle bat
# scallop that stays the same shape folded, spread or splayed: no deep V when the fingers close, no long
# needle when they open) and `along` x the chord off its middle. v2b skinned the cusps (fitted weights),
# which left two needle triangles per panel in the folded idle wing and slivers in the lying clips.
WING_HEM = {"C12": ("F1", "F2", .20, .04), "C23": ("F2", "F3", .18, .04), "C3B": ("F3", "B", .09, -.04)}


def hem_point(W, A, B, depth, along):
    """The cusp between corners A and B of the panel with apex W (arrays (..., 3)): on the plane W, A, B,
    `depth` x |AB| in from AB's middle toward W, `along` x |AB| from the middle toward B; never deeper than
    40 % of W's height over AB (so the panel cannot turn over)."""
    W, A, B = (np.asarray(x, float) for x in (W, A, B))
    M = (A + B) / 2
    ab = B - A
    ch = np.linalg.norm(ab, axis=-1, keepdims=True)
    u = ab / np.maximum(ch, 1e-9)
    to_w = W - M
    v = to_w - (to_w * u).sum(-1, keepdims=True) * u
    h = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v / np.maximum(h, 1e-9)
    d = np.minimum(depth * ch, 0.4 * h)
    return M + d * v + along * ch * u


WING_POINTS = {k: (_WING_FRAME[k] if k in _WING_FRAME else
                    tuple(hem_point(_WING_FRAME["W"], _WING_FRAME[WING_HEM[k][0]], _WING_FRAME[WING_HEM[k][1]],
                                    *WING_HEM[k][2:])))
               for k in ("S", "E", "W", "F1", "F2", "F3", "C12", "C23", "C3B", "B")}
WING_WEIGHTS = {
    "S": {"wing1L": .6, "chest": .4}, "E": {"wing1L": .4, "wing2L": .6}, "W": {"wing3L": 1},
    "F1": {"f1bL": 1}, "F2": {"f2bL": 1}, "F3": {"f3bL": 1},
    # the cusps' weights only matter where nothing re-places them (WingHem does, in every frame)
    "C12": {"f1bL": .5, "f2bL": .5}, "C23": {"f2bL": .5, "f3bL": .5},
    "C3B": {"f3bL": .5, "root": .5}, "B": {"root": .6, "spine1": .4},
}
WING_TRIS = [("S", "E", "W"), ("S", "W", "B"), ("W", "C3B", "B"), ("W", "F3", "C3B"),
             ("W", "C23", "F3"), ("W", "F2", "C23"), ("W", "C12", "F2"), ("W", "F1", "C12")]
# the finger bones, drawn on the skin: wrist -> tip
WING_FINGERS = [("W", "F1"), ("W", "F2"), ("W", "F3")]


# v2c: the elbow's place on the skin. E is only in the arm triangle (shoulder, elbow, wrist), a 7 degree
# sliver when spread but open (its elbow 0.35 x |SW| off the shoulder-wrist line) whenever the wing is
# held folded, so a planar layout of the spread wing smeared the painted arm bone across that whole patch.
# The patch is laid out at its typical folded shape instead: E `ARM_UV` = (along SW, out from SW) x |SW|.
ARM_UV = (.55, .24)


def _wing_uv():
    """Planar UVs of the spread membrane (its best-fit plane), 0..1 on the longer side; the elbow off the
    shoulder-wrist edge at ARM_UV (the arm triangle's own undistorted patch, away from the membrane)."""
    keys = list(WING_POINTS)
    P = np.array([WING_POINTS[k] for k in keys], float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    a, b = vt[0], vt[1]
    q = np.stack([(P - c) @ a, (P - c) @ b], 1)
    i = {k: n for n, k in enumerate(keys)}
    S, W, B = q[i["S"]], q[i["W"]], q[i["B"]]
    sw = W - S
    n = np.array([-sw[1], sw[0]])
    if np.dot(B - S, n) > 0:
        n = -n                                   # away from the membrane
    q[i["E"]] = S + ARM_UV[0] * sw + ARM_UV[1] * n
    lo, hi = q.min(0), q.max(0)
    span = (hi - lo).max()
    q = (q - lo) / span
    return {k: tuple(q[i[k]]) for k in keys}, (hi - lo) / span


WING_UV, WING_UV_EXTENT = _wing_uv()


class WingHem:
    """Re-places the hem cusps of a posed wings part (see WING_HEM): `fix(points)` takes the part's posed
    points (V, 3) and returns them with every copy of every cusp (both faces, crease copies) moved onto its
    panel. In the bind pose it gives back the bind points."""

    def __init__(self, part):
        P = np.array(part.points)
        corner = {}
        for i, t in enumerate(part.tag):
            k = t.get("key")
            if k in ("W", "F1", "F2", "F3", "B"):
                corner.setdefault((k, P[i, 0] > 0), i)
        self.rows = []
        for i, t in enumerate(part.tag):
            k = t.get("key")
            if k in WING_HEM:
                a, b, depth, along = WING_HEM[k]
                s = P[i, 0] > 0
                self.rows.append((i, corner[("W", s)], corner[(a, s)], corner[(b, s)], depth, along))

    def __call__(self, points):
        out = np.array(points, float)
        for i, w, a, b, depth, along in self.rows:
            out[i] = hem_point(out[w], out[a], out[b], depth, along)
        return out


def build_wings(parts):
    part = Part("wings", "Dragon_Wing")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
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
        b3 = "wing3L" if s > 0 else "wing3R"
        claw(part, pts["W"] + (0, .005, .0), pts["W"] + np.array([.0 * s, .065, .05]), .014, {b3: 1.0})
    parts.append(part)
    return part


# dorsal spikes: (ring index along the trunk path as a float, length)
# (index -1 .. 0 runs from the tail's tip to the first ring). v2b: six on the back and neck again (v1 had
# seven), paid for by the two rear upper fangs: 3 on the tail, 6 from between the shoulders to the head
SPIKES = [(-0.45, .06), (0.25, .075), (0.95, .075),
          (3.1, .06), (3.95, .07), (4.6, .08), (5.1, .085), (5.55, .08), (5.97, .07)]


def build_spikes(parts):
    part = Part("spikes", "Dragon_Head")
    path = np.vstack([TRUNK_TIP, TRUNK_PATH])
    radii = [(0.012, 0.012)] + list(TRUNK_RADII)
    hints = _trunk_hint(path)
    frames = _frames(path, hints)
    for f, length in SPIKES:
        f = f + 1
        i = int(np.floor(f))
        u = f - i
        j = min(i + 1, len(path) - 1)
        c = path[i] * (1 - u) + path[j] * u
        t = _unit(frames[i][0] * (1 - u) + frames[j][0] * u)
        d = _unit(frames[i][1] * (1 - u) + frames[j][1] * u)
        r = radii[i][0] * (1 - u) + radii[j][0] * u
        base = c + d * r * .9
        half = max(.02, length * .5)
        a, b = base - t * half, base + t * half
        tip = base + d * length - t * length * .55     # leaning toward the tail
        w = auto_weights(base, TRUNK_BONES)
        fin(part, "spike", a, b, tip, (1, 0, 0), w, w)
    parts.append(part)
    return part


def crease_split(part, angle):
    """Hard edges: give a vertex one copy per fan of its faces that meet at less than `angle` degrees
    (bind pose), so the game's per-vertex normals stay smooth across a gentle bend and break across a
    crease, the way the game's own creatures are shaded. Copies share position, UV, weights and tags, so
    the files keep one geometry row for them (no gaps)."""
    if not angle:
        return part
    P = np.array(part.points)
    T = [list(t) for t in part.tris]
    fn = np.array([np.cross(P[b] - P[a], P[c] - P[a]) for a, b, c in T])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    cos_lim = np.cos(np.radians(angle))
    faces_of = {}
    for f, t in enumerate(T):
        for v in t:
            faces_of.setdefault(v, []).append(f)
    for v, fs in faces_of.items():
        parent = {f: f for f in fs}

        def root(f):
            while parent[f] != f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f
        for i, f in enumerate(fs):
            for g in fs[i + 1:]:
                if len(set(T[f]) & set(T[g])) >= 2 and np.dot(fn[f], fn[g]) >= cos_lim:
                    parent[root(f)] = root(g)
        groups = {}
        for f in fs:
            groups.setdefault(root(f), []).append(f)
        for gi, group in enumerate(sorted(groups.values(), key=min)):
            if gi == 0:
                continue
            nv = len(part.points)
            part.points.append(part.points[v].copy())
            part.uv.append(part.uv[v])
            part.weights.append(dict(part.weights[v]))
            part.island.append(part.island[v])
            part.tag.append(dict(part.tag[v]))
            for f in group:
                T[f] = [nv if x == v else x for x in T[f]]
    part.tris = [tuple(t) for t in T]
    return part


# degrees between faces where the shading breaks, per part (0: smooth): the game's creatures have angular
# silhouettes but smooth shading inside them, with hard edges on the head's planes and the jaw line
CREASE_ANGLE = {"trunk": 0, "legs": 0, "head": 70, "jaw": 70, "wings": 70, "spikes": 70}


def build_dragon(crease=None):
    """-> list of Parts in bind pose (wings spread)."""
    parts = []
    build_trunk(parts)
    head, rings = build_head(parts)
    build_jaw(parts, rings)
    build_legs(parts)
    build_wings(parts)
    build_spikes(parts)
    for p in parts:
        crease_split(p, CREASE_ANGLE.get(p.name, 0) if crease is None else crease)
        if p.name == "wings":
            p.post = WingHem(p)          # v2c: the hem cusps are built per frame (Skinner applies it)
    return parts


def counts(parts):
    return {p.name: (len(p.points), len(p.tris)) for p in parts}


# ------------------------------------------------------------------------------------------- posing

def _rotmat(rv):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(rv).as_matrix()


def forward(local, root_offset=(0, 0, 0), skel=None):
    """local: (B, 3, 3) local rotations (bind = identity, world-aligned frames).
    -> (world rotations (B,3,3), world heads (B,3)). `skel`: a Skeleton (default the dragon's)."""
    head, parent = (HEAD, PARENT) if skel is None else (skel.head, skel.parent)
    B = len(parent)
    Rw = np.zeros((B, 3, 3))
    Pw = np.zeros((B, 3))
    for i in range(B):
        p = parent[i]
        if p is None:
            Rw[i] = local[i]
            Pw[i] = head[i] + np.asarray(root_offset, float)
        else:
            Rw[i] = Rw[p] @ local[i]
            Pw[i] = Pw[p] + Rw[p] @ (head[i] - head[p])
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
    """Vectorised linear blend skinning of all parts (`skel`: a Skeleton, default the dragon's)."""

    def __init__(self, parts, skel=None):
        head = HEAD if skel is None else skel.head
        self.parts = parts
        self.sizes = [len(p.points) for p in parts]
        P = np.concatenate([np.array(p.points) for p in parts])
        W = np.zeros((len(P), len(head)))
        k = 0
        for p in parts:
            for w in p.weights:
                for b, x in w.items():
                    W[k, b] = x
                k += 1
        self.P, self.W = P, W
        self.rel = P[:, None, :] - head[None, :, :]       # (V, B, 3)

    def __call__(self, Rw, Pw):
        moved = np.einsum("bij,vbj->vbi", Rw, self.rel) + Pw[None]
        out = np.einsum("vb,vbi->vi", self.W, moved)
        out = np.split(out, np.cumsum(self.sizes)[:-1])
        # a part's own per-frame corrective after the skinning (the dragon's wing hem, WingHem)
        return [p.post(o) if getattr(p, "post", None) is not None else o for p, o in zip(self.parts, out)]
