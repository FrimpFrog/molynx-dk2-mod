"""The Gargoyle (Soles' Creatures Mod) as a low-poly Dungeon Keeper 2 creature: skeleton, mesh, skin
weights, UVs. Built the way the Dragon is (scripts.dragon_model: its Part, loft, cone,
fin, forward and Skinner, here on the gargoyle's own Skeleton).

Soles: "make the small winged gargoyle but he will just be a cheap fighter like the goblin." So he is
SMALL (an Imp's to a Goblin's height in the game's files) and wiry, nothing like Molynx's big rocky golem:
the cathedral gargoyle, crouched on his haunches and hunched forward, a big grotesque head on a short neck
(a heavy brow over deep glowing eye sockets, a wide snarling mouth held open with an underbite of tusks
and two upper fangs, big pointed bat ears, two horns curling up and back), thin clawed human-like arms
hanging in front of his knees, digitigrade legs on clawed talons, a short tail with a spade tip and bat
wings (folded down his back when he stands; they open for hops and attacks).

Model space as the Dragon's: Y up, Z forward, X to the creature's LEFT; triangles counter-clockwise seen
from outside (the KMF export mirrors x into the game's winding). About 0.8 units to the wing tops in the
crouch; gargoyle_build scales his files. The bind pose is the crouch with the wings SPREAD (the membrane's
UVs are planar in that pose); the stand folds them (gargoyle_anims).

Budget, from the game's own small creatures: the Goblin's files have 216 triangles, the Imp's 187; the
gargoyle has 355 (his wings are 48 of them; 25 close the bases of his claws, fangs, horns and limbs so no
open tube can show; the angrier face's brow ridges and extra teeth 24) on three 128 x 128 skins
(Gargoyle_Body, Gargoyle_Head, Gargoyle_Wing).

Parts:
  trunk   one 5-sided loft (a ridge down the spine) from under the pelvis (capped) up the hunched back to
          the neck (inside the head, capped flat)
  tail    a 3-sided loft from inside the pelvis, a point at its tip and a double-sided spade over it
  legs    two 4-sided lofts hip (domed) - knee - ankle, tapering to the ball of the foot; three splayed
          talons rooted along each foot
  arms    two 3-sided lofts shoulder - elbow - wrist - the hand, three claws each
  head    skull (6-sided loft, the palate its floor; the upper lip curled up off the teeth), two heavy
          brow ridges in a V (3-sided lofts sat on the face), two horns hooking forward (3-sided lofts),
          two ears swept back (double-sided), two long uneven fangs and two small teeth
  jaw     the lower jaw (4-sided, built open: JAW_OPEN), two tusks, the throat closing the mouth
  wings   double-sided bat membranes (7 triangles a side) on an arm and three fingers; the forearm a
          3-sided tube ending in the wrist's knob

Lofts go through `gloft` (caps wound against their own ring; odd-sided rings get their straddling face
unfolded in UV, `unfold_odd`), cones through `ccone` (base closed), and `orient` makes every closed piece's
winding consistent and outward at the end.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_model import Part, Skeleton, Skinner, _head_section, _mirror_name, _seg_dist, _unit, \
    cone, forward, loft

# ------------------------------------------------------------------------------------------- skeleton

# name, parent, head (bind position), tail (end of its segment, for weights)
_BONES_L = [
    ("root", None, (0, .24, -.07), (0, .31, -.05)),
    ("spine", "root", (0, .31, -.05), (0, .39, -.01)),
    ("chest", "spine", (0, .39, -.01), (0, .465, .09)),
    ("neck", "chest", (0, .465, .09), (0, .50, .15)),
    ("head", "neck", (0, .50, .15), (0, .52, .34)),
    ("jaw", "head", (0, .49, .17), (0, .46, .32)),
    ("tail1", "root", (0, .22, -.09), (0, .14, -.19)),
    ("tail2", "tail1", (0, .14, -.19), (0, .11, -.275)),
    ("tail3", "tail2", (0, .11, -.275), (0, .17, -.37)),
    ("thighL", "root", (.085, .25, -.05), (.13, .25, .10)),
    ("shinL", "thighL", (.13, .25, .10), (.125, .09, -.03)),
    ("footL", "shinL", (.125, .09, -.03), (.13, .025, .07)),
    ("toesL", "footL", (.13, .025, .07), (.135, .01, .13)),
    ("upperAL", "chest", (.10, .45, .06), (.158, .33, .03)),
    ("foreAL", "upperAL", (.158, .33, .03), (.15, .20, .15)),
    ("handAL", "foreAL", (.15, .20, .15), (.15, .15, .19)),
    ("wing1L", "chest", (.06, .46, -.04), (.23, .52, -.07)),
    ("wing2L", "wing1L", (.23, .52, -.07), (.42, .66, -.10)),
    ("f1L", "wing2L", (.42, .66, -.10), (.68, .64, -.12)),
    ("f2L", "wing2L", (.42, .66, -.10), (.62, .42, -.13)),
    ("f3L", "wing2L", (.42, .66, -.10), (.44, .24, -.12)),
]


def _build_bones():
    out = []
    for name, parent, head, tail in _BONES_L:
        out.append((name, parent, np.array(head, float), np.array(tail, float)))
        if name.endswith("L"):
            m = np.array([-1.0, 1, 1])
            out.append((_mirror_name(name), _mirror_name(parent), np.array(head) * m, np.array(tail) * m))
    return out


SKEL = Skeleton(_build_bones())
BONES, BONE_NAMES, BONE_INDEX = SKEL.bones, SKEL.names, SKEL.index
PARENT, HEAD, TAIL = SKEL.parent, SKEL.head, SKEL.tail
NB_BONES = len(BONES)
LEGS = {s: tuple(b + s for b in ("thigh", "shin", "foot", "toes")) for s in "LR"}
ARMS = {s: tuple(b + s for b in ("upperA", "foreA", "handA")) for s in "LR"}
WINGS = {s: tuple(b + s for b in ("wing1", "wing2", "f1", "f2", "f3")) for s in "LR"}


class GargoylePart(Part):
    bone_index = BONE_INDEX


def auto_weights(p, bones, power=4.0, keep=2):
    """Inverse-distance weights to the nearest `keep` bone segments among `bones` (the gargoyle's)."""
    d = np.array([max(_seg_dist(p, HEAD[BONE_INDEX[b]], TAIL[BONE_INDEX[b]]), 1e-4) for b in bones])
    order = np.argsort(d)[:keep]
    return {bones[i]: d[i] ** -power for i in order}


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def gloft(part, island, path, radii, sides, hint, weights, **kw):
    """`loft`, with every cap triangle wound away from a point inside its ring (the ring's centre pushed
    away from the cap's point): the Dragon's loft judges a cap from halfway along the next segment, which
    turns it inside out where the limb bends sharply into the cap (a foot forward off the shin, a dome on
    a thigh)."""
    n0 = len(part.points)
    t0 = len(part.tris)
    rings = loft(part, island, path, radii, sides, hint, weights, **kw)
    caps = range(n0 + len(rings) * sides, len(part.points))       # the start / end points, in that order
    for cap in caps:
        ring = rings[0] if (kw.get("start_point") is not None and cap == caps[0]) else rings[-1]
        centre = np.mean([part.points[j] for j in ring], 0)
        inside = centre - (part.points[cap] - centre) * 0.5
        for ti in range(t0, len(part.tris)):
            a, b, c = part.tris[ti]
            if cap in (a, b, c):
                pa, pb, pc = (part.points[j] for j in (a, b, c))
                if np.dot(np.cross(pb - pa, pc - pa), (pa + pb + pc) / 3 - inside) < 0:
                    part.tris[ti] = (a, c, b)
    if sides % 2:
        unfold_odd(part, rings, t0)
    return rings


def unfold_odd(part, rings, t0, u_of=lambda u: u, inv=lambda u: u):
    """Odd-sided rings with mirrored UVs (u = |theta| / pi) give the face straddling theta = pi zero
    width in u (both its corners map to the same u): a smear. Unfold it: u = theta / theta_max on the
    left (theta_max = pi + pi / sides), the right side mirrored onto the same texels, and the straddling
    face's right-hand corner duplicated at u = 1, so that face gets its own strip of the island.
    `u_of` / `inv` map between the loft's 0..1 u and the stored u (a cone's uv_box)."""
    n = len(rings[0])
    tmax = np.pi + np.pi / n
    kl, kr = (n - 1) // 2, (n + 1) // 2
    dup = {}
    for ring in rings:
        for k, vid in enumerate(ring):
            th = 2 * np.pi * k / n
            u = (th if th <= np.pi else 2 * np.pi - th) / tmax
            part.uv[vid] = (u_of(u), part.uv[vid][1])
        r = ring[kr]
        part.points.append(part.points[r].copy())
        part.uv.append((u_of(1.0), part.uv[r][1]))
        part.weights.append(dict(part.weights[r]))
        part.island.append(part.island[r])
        part.tag.append(dict(part.tag[r]))
        dup[r] = len(part.points) - 1
    left = {ring[kl] for ring in rings}
    for ti in range(t0, len(part.tris)):
        t = part.tris[ti]
        if any(v in left for v in t) and any(v in dup for v in t):
            part.tris[ti] = tuple(dup.get(v, v) for v in t)


def flat_cap(part, ring, away):
    """Close a loft's open ring with a flat fan (sides - 2 triangles), wound to face away from `away`."""
    cen = np.mean([part.points[j] for j in ring], 0)
    inside = cen + (np.asarray(away, float) - cen) * 0.5
    for k in range(1, len(ring) - 1):
        part.tri(ring[0], ring[k], ring[k + 1], inside=inside)


def ccone(part, island, base, tip, radius, up, weights, sides=3, tag=None):
    """`cone` with its base closed (one flat triangle for three sides), for claws, fangs and talons whose
    base is not wholly inside the body."""
    t0 = len(part.tris)
    ids = cone(part, island, base, tip, radius, up, weights, sides=sides, tag=tag)
    flat_cap(part, ids[:sides], tip)
    if sides % 2:
        unfold_odd(part, [ids[:sides]], t0)
    return ids


def dside(part, island, a, b, tip, side, w_ab, w_tip, extra=None, tag=None):
    """A double-sided flat piece: the triangle (a, b, tip) and, with `extra` = (point, weights), a
    second one (b, extra, tip) making a quad-ish blade; `side` is the lateral direction its first face
    looks. Every vertex carries `along` (0 at the base, 1 at the tip) and its own (u, v)."""
    pts = [(a, (0.0, 0.0), w_ab, 0.0), (b, (1.0, 0.0), w_ab, 0.0), (tip, (0.5, 1.0), w_tip, 1.0)]
    if extra is not None:
        pts.append((extra[0], (1.0, 0.55), extra[1], 0.55))
    tris = [(0, 1, 2)] + ([(1, 3, 2)] if extra is not None else [])
    for face in (1, -1):
        ids = [part.add(p, uv, w, island, **(tag or {}), along=al, face=face) for p, uv, w, al in pts]
        for i, j, k in tris:
            pa, pb, pc = (np.asarray(pts[q][0], float) for q in (i, j, k))
            n = np.cross(pb - pa, pc - pa)
            if np.dot(n, side) * face >= 0:
                part.tri(ids[i], ids[j], ids[k])
            else:
                part.tri(ids[i], ids[k], ids[j])


# ------------------------------------------------------------------------------------------- trunk

TRUNK_PATH = np.array([
    (0, .245, -.075),          # the pelvis, low between the haunches
    (0, .385, -.015),          # the chest: the hunch of the back, leaning forward
    (0, .465, .085),           # the shoulders
    (0, .495, .15),            # the neck (inside the head's back)
])
# (dorsal-ventral, lateral) half sizes: a thin, wiry body with a hunched back
TRUNK_RADII = [(.075, .08), (.09, .1), (.07, .095), (.045, .05)]
TRUNK_BOTTOM = np.array([0, .175, -.07])
TRUNK_SIDES = 5                # a ridge down the spine (theta 0), a flat belly
TRUNK_BONES = ["root", "spine", "chest", "neck"]


def _trunk_hint(path):
    return [(0, .25, -1.0)] * len(path)


def trunk_shape(i, th):
    c = np.cos(th)
    if c > 0.9 and i in (1, 2):            # the hunch: the spine ridges behind the shoulders
        return 1.15, 1.0
    return 1.0, 1.0


def build_trunk(parts):
    part = GargoylePart("trunk", "Gargoyle_Body")
    rings = gloft(part, "trunk", TRUNK_PATH, TRUNK_RADII, TRUNK_SIDES, _trunk_hint(TRUNK_PATH),
                  lambda i, p: auto_weights(p, TRUNK_BONES), start_point=TRUNK_BOTTOM, shape=trunk_shape,
                  tag=lambda i, th: {"theta": th, "ring": i})
    flat_cap(part, rings[-1], TRUNK_PATH[-2])            # the neck, in case the head turns off it
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- tail

TAIL_PATH = np.array([(0, .22, -.09), (0, .14, -.19), (0, .11, -.275)])
TAIL_RADII = [(.028, .028), (.02, .02), (.014, .014)]
TAIL_END = np.array([0, .145, -.345])
TAIL_BONES = ["tail1", "tail2", "tail3"]
# the spade: the arrowhead at the tail's tip (left barb; mirrored), lying across the tail so the high
# camera sees it
SPADE_TIP = np.array([0, .195, -.405])
SPADE_BARB = np.array([.045, .14, -.325])
SPADE_NOTCH = np.array([0, .148, -.338])


def build_tail(parts):
    part = GargoylePart("tail", "Gargoyle_Body")

    def w(i, p):
        if i <= 0:
            return {"tail1": 1.0}
        if i >= len(TAIL_PATH):
            return {"tail3": 1.0}
        return auto_weights(p, TAIL_BONES)
    gloft(part, "tail", TAIL_PATH, TAIL_RADII, 3, (0, 1, 0), w, end_point=TAIL_END,
         tag=lambda i, th: {"theta": th, "ring": i})
    # the spade (double-sided: two triangles a side, tip - barb - notch)
    up = _unit(np.cross(SPADE_TIP - SPADE_NOTCH, (1.0, 0, 0)))
    up = up if up[1] > 0 else -up
    for s in (1, -1):
        barb = SPADE_BARB * (s, 1, 1)
        dside(part, "spade", SPADE_NOTCH, barb, SPADE_TIP, up, "tail3", "tail3")
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- head

HEAD_RINGS = [  # centre, lateral, up, down (to the lip line): a big, broad, pug-nosed head
    ((0, .535, .13), .07, .065, .045),          # the back of the skull
    ((0, .548, .215), .104, .1, .066),          # the brow: the widest, deepest part
    ((0, .515, .287), .094, .052, .05),         # the short, broad, flat muzzle (a wide mouth)
]
SNOUT_TIP = np.array([0, .51, .322])
SKULL_BACK = np.array([0, .55, .09])
JAW_OPEN = 9.0               # a snarl, not a gape (Soles, 13 Nov: "angrier and scarier")
THROAT_RISE = .04            # how far the throat's top edge reaches up inside the head
LIP_CURL = .016              # the upper lip at the muzzle curled up off the teeth
# u of the head section's six points (top, upper side, lip, palate, ...): the face gets most of the
# island, the palate (inside the mouth, mostly hidden) the last fifth
HEAD_U = [0, .4, .8, 1, .8, .4]              # degrees: the bind pose's snarl
HINGE = HEAD[BONE_INDEX["jaw"]]
# horns (left; mirrored): base on top of the head above the brow, sweeping up, out and back, curling
# (angled forward like a bull's: up and out off the skull, hooking forward over the brow)
HORN_PATH = np.array([(.048, .612, .195), (.098, .68, .19)])
HORN_RADII = [(.03, .028), (.018, .017)]
HORN_TIP = np.array([.128, .705, .275])
# ears (left; mirrored): a big pointed bat's ear from the side of the skull, out, up and back
# (swept back flat along the skull, as an angry cat's)
EAR_FRONT = np.array([.078, .582, .19])
EAR_BACK = np.array([.064, .558, .12])
EAR_TIP = np.array([.175, .615, .005])
EAR_LOBE = np.array([.088, .528, .145])
# upper fangs (z, length on the left, on the right): long and uneven at the corners, small ones in front
UPPER_FANGS = ((.27, .074, .062), (.3, .036, .04))
TUSKS = (.072, .06)                              # the lower tusks, left and right
# the brow: a heavy ridge each side, low and bunched over the nose (the frown), rising over the eye
BROW_INNER = (.012, .585)                         # (x, y) on the face; z found on the skull's surface
BROW_OUTER = (.08, .603)
BROW_RADII = [(.017, .015), (.008, .007)]


def build_head(parts):
    part = GargoylePart("head", "Gargoyle_Head")
    centres = [np.array(r[0], float) for r in HEAD_RINGS]
    n = len(HEAD_RINGS)
    axes = [centres[min(i + 1, n - 1)] - centres[max(i - 1, 0)] for i in range(n)]
    length = np.linalg.norm(SNOUT_TIP - SKULL_BACK)
    rings, lips = [], []
    for i, ((c, lat, up, down), ax) in enumerate(zip(HEAD_RINGS, axes)):
        pts = _head_section(c, lat, up, down, ax)
        v = np.linalg.norm(np.asarray(c) - SKULL_BACK) / length
        lips.append((pts[2].copy(), pts[4].copy()))
        if i == n - 1:                                     # the snarl: the upper lip curled up
            for k in (2, 4):
                pts[k] = pts[k] + np.array([0, LIP_CURL, .004])
        ring = []
        for k, p in enumerate(pts):
            ring.append(part.add(p, (HEAD_U[k], v), "head", "skull", ring=i, k=k))
        rings.append(ring)
    part.lips = lips                                       # the uncurled lip line: the jaw closes on it
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
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        # horns: a 3-sided loft curling up and back, a point at the tip
        hrings = gloft(part, "horn", HORN_PATH * m, HORN_RADII, 3, (0, 0, 1.0), lambda i, p: "head",
             end_point=HORN_TIP * m, tag=lambda i, th: {"along": min(max(i, 0), 2) / 2.0, "theta": th},
             side_axis_sign=s)
        flat_cap(part, hrings[0], HORN_PATH[1] * m)
        # ears: double-sided, a front and a back corner on the skull, the point, and the lobe below
        dside(part, "ear", EAR_FRONT * m, EAR_BACK * m, EAR_TIP * m, (0, 0.3, 1.0), "head", "head",
              extra=None)
        dside(part, "ear", EAR_LOBE * m, EAR_FRONT * m, EAR_TIP * m, (0, 0.3, 1.0), "head", "head")
        # the upper fangs along the (uncurled) lip line, pointing down over the lower lip: long, uneven,
        # a little crooked
        zs = [r[0][2] for r in HEAD_RINGS]
        for j, (z, ll, lr) in enumerate(UPPER_FANGS):
            l = ll if s > 0 else lr
            c = np.array([0, .0, z])
            c[1] = np.interp(z, zs, [lp[0][1] for lp in lips]) + .012
            c[0] = s * np.interp(z, zs, [abs(lp[0][0]) for lp in lips]) * (.82 if j == 0 else .5)
            r = .015 if j == 0 else .01
            ccone(part, "fang", c, c + np.array([(-.008 if j == 0 else .004) * s, -l, .014 - .006 * j]), r,
                  (s, 0, 0), "head", sides=3)
    # the brows: 3-sided ridges sat on the face (their centre line just proud of the skull's surface)
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        a = np.array([BROW_INNER[0], BROW_INNER[1], 0])
        b = np.array([BROW_OUTER[0], BROW_OUTER[1], 0])
        a[2] = _front_z(part, a[0], a[1]) + .001
        b[2] = _front_z(part, b[0], b[1]) - .003
        path = np.array([a, b]) * m
        brings = gloft(part, "brow", path, BROW_RADII, 3, (0, .6, 1.0), lambda i, p: "head",
                       tag=lambda i, th: {"along": float(min(max(i, 0), 1)), "theta": th}, side_axis_sign=s)
        flat_cap(part, brings[0], path[1])
        flat_cap(part, brings[1], path[0])
    parts.append(part)
    return part, rings


def _front_z(part, x, y):
    """z of the part's frontmost surface at (x, y) (a ray along -z from in front)."""
    P = np.array(part.points)
    best = None
    for a, b, c in part.tris:
        if part.island[a] != "skull":
            continue
        A, B, C = P[a], P[b], P[c]
        d = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        if abs(d) < 1e-12:
            continue
        w0 = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / d
        w1 = ((C[1] - A[1]) * (x - C[0]) + (A[0] - C[0]) * (y - C[1])) / d
        w2 = 1 - w0 - w1
        if min(w0, w1, w2) >= -1e-9:
            z = w0 * A[2] + w1 * B[2] + w2 * C[2]
            best = z if best is None else max(best, z)
    return best


def build_jaw(parts, head_rings):
    """A closed lower jaw under the head's lip line (like the Dragon's), turned open by JAW_OPEN about
    the hinge; a jutting underbite with two tusks."""
    head = parts[-1]
    part = GargoylePart("jaw", "Gargoyle_Head")
    R = _rx(JAW_OPEN)
    op = lambda p: HINGE + R @ (np.asarray(p, float) - HINGE)
    rings = []
    depths = [.05, .045]
    for i, hr in enumerate((1, 2)):
        lipL, lipR = (np.asarray(q) for q in head.lips[hr])
        mid = (lipL + lipR) / 2
        P = [mid + (0, -.018, 0),                                     # the tongue bed
             lipL + (lipR - lipL) * .03 + (0, -.002, .0),
             mid + (0, -depths[i] - .012, 0),                         # the keel
             lipR + (lipL - lipR) * .03 + (0, -.002, .0)]
        ring = [part.add(op(p), ([0, .5, 1, .5][k], i / 2.0 + .15), "jaw", "jaw", ring=i, k=k)
                for k, p in enumerate(P)]
        rings.append(ring)
    inner = [((part.points[r[1]] + part.points[r[3]]) / 2 + part.points[r[2]]) / 2 for r in rings]
    mid = (inner[0] + inner[1]) / 2
    for k in range(4):
        a, b = rings[0][k], rings[0][(k + 1) % 4]
        c_, d_ = rings[1][k], rings[1][(k + 1) % 4]
        part.tri(a, b, c_, inside=mid)
        part.tri(b, d_, c_, inside=mid)
    chin_p = op((0, .462, .335))                     # the jutting chin of the underbite
    chin = part.add(chin_p, (.5, 1.0), "jaw", "jaw", ring=2, k=-1)
    for k in range(4):
        part.tri(rings[1][k], rings[1][(k + 1) % 4], chin, inside=inner[1] - (chin_p - inner[1]) * .5)
    back_p = op((0, .485, .13))
    back = part.add(back_p, (.5, 0.0), "jaw", "jaw", ring=-1, k=-1)
    for k in range(4):
        part.tri(rings[0][(k + 1) % 4], rings[0][k], back, inside=inner[0] + (inner[0] - back_p) * .5)
    # tusks: pointing up from the front corners of the jaw, outside the upper lip
    up = R @ np.array([0, 1.0, 0])
    for s in (1, -1):
        p = np.asarray(part.points[rings[1][1 if s > 0 else 3]])
        c = p + np.array([-.012 * s, -.004, .004])
        l = TUSKS[0] if s > 0 else TUSKS[1]
        ccone(part, "fang", c, c + up * l + np.array([.014 * s, 0, .01]), .015, (s, 0, 0), "jaw", sides=3)
    # the throat: a dark wall closing the back of the mouth, from the jaw's lips at the brow ring up into
    # the head (its top edge hidden inside the skull, THROAT_RISE above the head's lips). It moves with
    # the jaw alone, so opening the mouth turns it instead of stretching it (it used to run from the
    # head's lips to the jaw's and stretched 3x in a screech)
    hl, hr = (np.asarray(head.points[head_rings[1][k]]) + np.array([0, THROAT_RISE, -.012]) for k in (2, 4))
    hl, hr = hl * np.array([.8, 1, 1]), hr * np.array([.8, 1, 1])
    jl, jr = (np.asarray(part.points[rings[0][k]]) for k in (1, 3))
    ids = [part.add(p, uv, "jaw", "throat") for p, uv in ((hl, (0, 0)), (hr, (1, 0)), (jr, (1, 1)), (jl, (0, 1)))]
    front = np.array([0, 0, 1.0])
    for a, b, c in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])):
        pa, pb, pc = (part.points[j] for j in (a, b, c))
        part.tri(a, b, c, inside=(pa + pb + pc) / 3 - front)
    parts.append(part)
    return part, rings


# ------------------------------------------------------------------------------------------- limbs

# legs (left; mirrored): hip, knee (high and forward: the crouch), ankle (back, raised), the ball
LEG_PATH = np.array([(.085, .25, -.05), (.13, .25, .10), (.125, .09, -.03)])
LEG_RADII = [(.058, .052), (.04, .038), (.026, .026)]
LEG_TOP = np.array([.07, .30, -.075])
LEG_END = np.array([.13, .03, .075])            # the ball of the foot: the talons spring from it
# talons: three splayed toes springing from the foot (left; mirrored): (sideways offset of the root,
# tip on the floor); the roots sit along the foot so the toes read as part of it, a bird's foot
TALON_ROOT = 0.5                    # of the way from the ankle to the ball
TALONS = [(-.016, (.095, .002, .118)), (0.0, (.132, .002, .142)), (.016, (.172, .002, .118))]
# arms (left; mirrored): shoulder (in the chest), elbow (hanging, out and back), wrist (forward)
ARM_PATH = np.array([(.09, .45, .06), (.158, .33, .03), (.15, .20, .15)])
ARM_RADII = [(.032, .032), (.024, .024), (.017, .019)]
ARM_END = np.array([.15, .165, .185])
HAND_CLAWS = [(-.55, .055), (0.0, .062), (.55, .05)]


def _limb_weights(bones, n_rings):
    def w(i, p):
        if i <= 0:
            return {bones[0]: 1.0}
        if i >= n_rings:
            return {bones[-1]: 1.0}
        return auto_weights(p, list(bones))
    return w


def build_legs(parts):
    part = GargoylePart("legs", "Gargoyle_Body")
    for s, side in ((1, "L"), (-1, "R")):
        m = np.array([s, 1, 1.0])
        thigh, shin, foot, toes = LEGS[side]
        path = LEG_PATH * m
        # one dorsal hint (forward) for the whole limb
        gloft(part, "leg", path, LEG_RADII, 4, (0, .2, 1.0), _limb_weights((thigh, shin, foot), 3),
             start_point=LEG_TOP * m, end_point=LEG_END * m,
             tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
        root = LEG_PATH[2] + (LEG_END - LEG_PATH[2]) * TALON_ROOT
        for dx, tip in TALONS:
            base = (root + np.array([dx, -.006, 0])) * m
            ccone(part, "claw", base, np.array(tip) * m, .016, (0, 1, 0), toes, sides=3)
    parts.append(part)
    return part


def build_arms(parts):
    part = GargoylePart("arms", "Gargoyle_Body")
    for s, side in ((1, "L"), (-1, "R")):
        m = np.array([s, 1, 1.0])
        upper, fore, hand = ARMS[side]
        path = ARM_PATH * m
        rings = gloft(part, "arm", path, ARM_RADII, 3, (0, .3, 1.0), _limb_weights((upper, fore, hand), 3),
                      end_point=ARM_END * m, tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s},
                      side_axis_sign=s)
        flat_cap(part, rings[0], path[1])
        knuckle = ARM_END * m + np.array([0, .004, -.008])
        for dx, l in HAND_CLAWS:
            d = _unit(np.array([dx * s, -1.1, 1.0]))
            base = knuckle + np.array([dx * s * .022, 0, 0])
            ccone(part, "claw", base, base + d * l, .011, (0, 1, 0), hand, sides=3)
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- wings

# membrane points in the spread (bind) pose, left wing
WING_POINTS = {
    "S": (.06, .46, -.04), "E": (.23, .52, -.07), "W": (.42, .66, -.10),     # (the elbow bent: review 1)
    "F1": (.68, .64, -.12), "F2": (.62, .42, -.13), "F3": (.44, .24, -.12),
    "C12": (.53, .55, -.13), "C23": (.45, .38, -.13), "B": (.07, .30, -.10),
}
WING_WEIGHTS = {
    "S": {"wing1L": .5, "chest": .5}, "E": {"wing1L": .5, "wing2L": .5}, "W": {"wing2L": 1},
    "F1": {"f1L": 1}, "F2": {"f2L": 1}, "F3": {"f3L": 1},
    "C12": {"f1L": .5, "f2L": .5}, "C23": {"f2L": .5, "f3L": .5}, "B": {"spine": .6, "root": .4},
}
WING_TRIS = [("S", "E", "W"), ("S", "W", "B"), ("W", "F3", "B"),
             ("W", "F1", "C12"), ("W", "C12", "F2"), ("W", "F2", "C23"), ("W", "C23", "F3")]
FOREARM_RADII = [(.016, .016), (.012, .012)]


def _wing_uv():
    """Planar UVs of the spread membrane (its best-fit plane), 0..1 on the longer side."""
    P = np.array(list(WING_POINTS.values()), float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    a, b = vt[0], vt[1]
    q = np.stack([(P - c) @ a, (P - c) @ b], 1)
    if q[list(WING_POINTS).index("W"), 1] < q[list(WING_POINTS).index("F3"), 1]:
        q[:, 1] = -q[:, 1]                  # the arm along the top (v small), the hem below
        q[:, 1] = q[:, 1]
    lo, hi = q.min(0), q.max(0)
    span = (hi - lo).max()
    q = (q - lo) / span
    return {k: tuple(q[i]) for i, k in enumerate(WING_POINTS)}, (hi - lo) / span


WING_UV, WING_UV_EXTENT = _wing_uv()


def build_wings(parts):
    part = GargoylePart("wings", "Gargoyle_Wing")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        pts = {k: np.array(v) * m for k, v in WING_POINTS.items()}
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
        # the forearm: a thin 3-sided tube elbow - wrist, ending in the wrist's knob (a claw-like point)
        b1, b2 = ("wing1L", "wing2L") if s > 0 else ("wing1R", "wing2R")
        arm = np.array([pts["E"], pts["W"]])
        knob = pts["W"] + (pts["W"] - pts["E"]) * .22 + np.array([0, .02, 0])
        rings = gloft(part, "warm", arm, FOREARM_RADII, 3, (0, 0, 1), lambda i, p, b2=b2: {b2: 1.0},
                      end_point=knob, tag=lambda i, th: {"along": min(max(i, 0), 2) / 2.0, "theta": th},
                      side_axis_sign=1.0)
        flat_cap(part, rings[0], pts["W"])
    parts.append(part)
    return part


DOUBLE_SIDED = ("ear", "spade", "membrane", "throat")      # islands that are open on purpose


def welded(part):
    """-> each vertex's index of the first vertex at the same position (UV seams duplicate vertices)."""
    first = {}
    return [first.setdefault(tuple(np.round(p, 7)), i) for i, p in enumerate(part.points)]


def orient(part):
    """Wind every closed piece of `part` consistently and outward: a flood fill over shared edges makes
    neighbours agree (the loft's own test, from the midpoint between two rings, can misjudge a face at a
    sharp bend such as the knee), then a piece whose signed volume is negative is turned round."""
    P = np.array(part.points)
    weld = welded(part)                    # UV seams duplicate vertices: join pieces by position
    idx = [i for i, t in enumerate(part.tris) if part.island[t[0]] not in DOUBLE_SIDED]
    edges = {}
    for i in idx:
        a, b, c = (weld[v] for v in part.tris[i])
        for e in ((a, b), (b, c), (c, a)):
            edges.setdefault(frozenset(e), []).append(i)
    seen = set()
    for start in idx:
        if start in seen:
            continue
        comp, todo = [], [start]
        seen.add(start)
        while todo:
            i = todo.pop()
            comp.append(i)
            a, b, c = (weld[v] for v in part.tris[i])
            for e in ((a, b), (b, c), (c, a)):
                for j in edges[frozenset(e)]:
                    if j in seen:
                        continue
                    x, y, z = (weld[v] for v in part.tris[j])
                    if e in ((x, y), (y, z), (z, x)):          # same direction: j disagrees with i
                        x, y, z = part.tris[j]
                        part.tris[j] = (x, z, y)
                    seen.add(j)
                    todo.append(j)
        vol = sum(np.dot(P[a], np.cross(P[b], P[c])) for a, b, c in (part.tris[i] for i in comp))
        if vol < 0:
            for i in comp:
                a, b, c = part.tris[i]
                part.tris[i] = (a, c, b)


def build_gargoyle():
    """-> list of Parts in bind pose (the crouch, wings spread)."""
    parts = []
    build_trunk(parts)
    build_tail(parts)
    build_legs(parts)
    build_arms(parts)
    head, rings = build_head(parts)
    build_jaw(parts, rings)
    build_wings(parts)
    for p in parts:
        orient(p)
    return parts


def counts(parts):
    return {p.name: (len(p.points), len(p.tris)) for p in parts}


def gargoyle_skinner(parts):
    return Skinner(parts, SKEL)


def gargoyle_forward(local, root_offset=(0, 0, 0)):
    return forward(local, root_offset, SKEL)
