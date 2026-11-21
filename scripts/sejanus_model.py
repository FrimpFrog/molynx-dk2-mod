"""Sejanus, the horse demon (Soles' design, 13 Nov), as a low-poly Dungeon Keeper 2 creature: skeleton, mesh,
skin weights, UVs.

Soles: "A demonic horse like creature, standing upright with human arms, leaning very forward and arms
dangling down, a skinnier muscly creature." He is the Hunter: the fastest runner in the dungeon. So the body
is built pitched forward (the spine about 40 degrees from vertical), shoulders hunched, a long thick arched
neck carrying a real horse's head (after Soles' reference picture: a long straight face, a rounded blunt
muzzle, the mouth closed, upright pointed ears, no horns) with a pale mane swept back along its crest; a
gaunt ribbed torso; long sinewy human arms whose knobbly hands hang to the stifle; horse hind legs (stifle
forward, hock back) on heavy hooves; a ragged horse tail.

Built from scratch on the Dragon's machinery (dragon_model: Part, loft, cone, fin, forward, Skinner), with its
conventions: model space Y up, Z forward (the way he faces), X to his LEFT; the tips of the ears at about
y = 1; triangles counter-clockwise seen from outside (the KMF export mirrors x into the game's winding).
The bind pose IS his standing pose (arms hanging, mouth closed), so weights and UVs are laid out on the shape
the player sees most.

Budget, from the game's own creatures: the Horned Reaper's PoseFrame has 406 triangles on 128 x 128 skins;
Sejanus has 416 on three 128 x 128 skins (Sejanus_Body: torso and neck; Sejanus_Limbs: legs, arms, fingers
and tail; Sejanus_Head: skull, jaw, ears, mane).

Parts (each a list of triangles over its own vertices; `Part.texture` names its skin):
  trunk   one 8-sided loft from under the pelvis up the forward-thrust neck (capped both ends), its UVs
          two islands (torso, neck: loft_islands) so neither stretches
  tail    a 4-sided loft hanging from the base of the spine, its tip a ragged point, two loose locks
  head    the horse skull (6-sided loft, palate; closed deep jowl at the back), two ears (closed cones), and
          the spike mane (double-sided spikes swept back along the neck and on down the spine)
  jaw     the lower jaw (4-sided loft): short and slim under the muzzle, hinged behind the mouth's corner
  legs    5-sided loft hip-stifle-hock-fetlock-hoof, capped at the hip (a dome into the body) and the flat
          sole; UV islands thigh and shank
  arms    4-sided loft shoulder-elbow-wrist-knuckles (capped both ends), three knobbly fingers rooted inside
          the hand

UVs mirror across each part's own symmetry plane (dragon_model.loft: u = |angle| / pi round a loft), so no
loft needs a seam and left and right share texels.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_model import Part, Skeleton, Skinner as _Skinner, _frames, _seg_dist, _unit, cone, fin, loft
from scripts.dragon_model import forward as _forward

# ------------------------------------------------------------------------------------------- lean

# The upper body is drawn at about 35 degrees from vertical, then pitched LEAN degrees further forward about
# the waist (Soles: "leaning very forward"); the arms keep hanging straight down from the moved shoulders.
LEAN = 8.0
WAIST = np.array([0, .585, -.05])


def lean(p):
    p = np.asarray(p, float)
    a = np.radians(LEAN)
    d = p - WAIST
    return WAIST + np.array([d[0], d[1] * np.cos(a) - d[2] * np.sin(a), d[1] * np.sin(a) + d[2] * np.cos(a)])


SHOULDER = np.array([.118, .785, .082])
ARM_SHIFT = lean(SHOULDER) - SHOULDER      # the arms move with the shoulders, still hanging


# ------------------------------------------------------------------------------------------- head geometry

# Soles' reference (13 Nov): a real horse's head, not a wolf's snout: a long straight face from the forehead to
# a rounded, blunt muzzle, the mouth closed, a flat cheek, two upright pointed ears, no horns. It hangs from
# the end of the arched neck about 40 degrees below level.
SKULL_BACK = lean([0, .972, .262])
HEAD_AXIS = np.array([0, -np.sin(np.radians(40)), np.cos(np.radians(40))])
HEAD_UP = _unit(np.array([0, 1.0, 0]) - np.dot([0, 1.0, 0], HEAD_AXIS) * HEAD_AXIS)
_HEAD_LEN = .345
SNOUT_TIP = SKULL_BACK + HEAD_AXIS * _HEAD_LEN - HEAD_UP * .006   # blunt: just ahead of the muzzle ring
HEAD_RINGS = [  # distance along the skull from its back, lateral, up, down, closed
    # Soles (13 Nov, second look): the long hinged jaw read as a sock puppet's. Now the back of the head is
    # closed (the deep round jowl is part of the skull) and the mouth is a horse's: its corner low and two
    # thirds of the way down the face, a short slim lower jaw under the muzzle only.
    (.055, .052, .048, .10, True),      # cranium and the deep, flat cheek down to the jowl
    (.14, .05, .044, .082, True),       # the eyes: the broad forehead, the jowl's front
    (.235, .036, .036, .04, False),     # the long straight face; the corner of the mouth
    (.31, .041, .038, .046, False),     # the rounded, deep muzzle: the upper lip
]
# (along, lip width (of the face's), lip, tongue, keel): behind the mouth's corner (inside the head), then
# under the upper lip; the tongue above the lips, so closed it hides inside the head
JAW = [(.2, .95, -.022, -.012, -.064),
       (.31, .93, None, -.02, -.075)]
JAW_CHIN = (.335, -.064)                  # the round lower lip and chin
JAW_BACK = (.185, -.04)
JAW_HINGE = (.2, -.026)
EAR = (.05, .028, .025)             # base on the poll: along, up, out
EAR_TIP = (.014, .07, .02)          # upright, a little forward and out  # the jaw below the lip line: the round jowl, then the chin


def head_point(s, up=0.0, x=0.0):
    """A point on the skull's axis `s` from its back, `up` along its up and `x` to the left."""
    return SKULL_BACK + HEAD_AXIS * s + HEAD_UP * up + np.array([x, 0, 0])


# ------------------------------------------------------------------------------------------- skeleton

# name, parent, head (bind position), tail (end of its segment, for weights)
_BONES_L = [
    ("root", None, (0, .50, -.10), (0, .585, -.05)),
    ("spine1", "root", (0, .585, -.05), (0, .665, .005)),
    ("chest", "spine1", (0, .665, .005), (0, .79, .10)),
    ("neck1", "chest", (0, .79, .10), (0, .865, .19)),
    ("neck2", "neck1", (0, .865, .19), (0, .93, .275)),
    ("head", "neck2", (0, .93, .275), tuple(head_point(_HEAD_LEN))),        # (head and jaw: already leaned)
    ("jaw", "head", tuple(head_point(*JAW_HINGE)), tuple(head_point(*JAW_CHIN))),
    ("earL", "head", tuple(head_point(*EAR)), tuple(head_point(*EAR) + EAR_TIP)),
    ("tail1", "root", (0, .53, -.15), (0, .42, -.25)),
    ("tail2", "tail1", (0, .42, -.25), (0, .30, -.29)),
    ("tail3", "tail2", (0, .30, -.29), (0, .15, -.30)),
    ("clavL", "chest", (.02, .775, .075), (.115, .785, .085)),
    ("armL", "clavL", (.115, .785, .085), (.155, .57, .065)),
    ("forearmL", "armL", (.155, .57, .065), (.17, .37, .115)),
    ("handL", "forearmL", (.17, .37, .115), (.175, .25, .16)),
    ("thighL", "root", (.08, .50, -.085), (.10, .33, .03)),
    ("shinL", "thighL", (.10, .33, .03), (.10, .17, -.11)),
    ("footL", "shinL", (.10, .17, -.11), (.10, .055, -.075)),
    ("toesL", "footL", (.10, .055, -.075), (.10, .0, -.015)),
]


def _mirror_name(n):
    return n[:-1] + "R" if n and n.endswith("L") else n


_UPPER = {"spine1": "tail", "chest": "both", "neck1": "both", "neck2": "both", "head": "head", "clavL": "both"}
_ARM = ("armL", "forearmL", "handL")


def _leaned(name, head, tail):
    head, tail = np.array(head, float), np.array(tail, float)
    how = _UPPER.get(name)
    if how == "tail":
        return head, lean(tail)
    if how == "both":
        return lean(head), lean(tail)
    if how == "head":
        return lean(head), tail
    if name in _ARM:
        return head + ARM_SHIFT, tail + ARM_SHIFT
    return head, tail


def _build_bones():
    out = []
    for name, parent, head, tail in _BONES_L:
        head, tail = _leaned(name, head, tail)
        out.append((name, parent, np.array(head, float), np.array(tail, float)))
        if name.endswith("L"):
            m = np.array([-1.0, 1, 1])
            out.append((_mirror_name(name), _mirror_name(parent), np.array(head) * m, np.array(tail) * m))
    return out


BONES = _build_bones()
SKEL = Skeleton(BONES)
BONE_NAMES = SKEL.names
BONE_INDEX = SKEL.index
PARENT = SKEL.parent
HEAD = SKEL.head
TAIL = SKEL.tail


class SPart(Part):
    """A Part weighted on Sejanus's bones."""
    bone_index = BONE_INDEX


def auto_weights(p, bones, power=4.0, keep=2):
    """Inverse-distance weights to the nearest `keep` bone segments among `bones`."""
    d = np.array([max(_seg_dist(p, HEAD[BONE_INDEX[b]], TAIL[BONE_INDEX[b]]), 1e-4) for b in bones])
    order = np.argsort(d)[:keep]
    return {bones[i]: d[i] ** -power for i in order}


def capped_cone(part, island, base, tip, radius, up, weights, sides=3, tag=None):
    """dragon_model.cone with its base closed (one triangle for three sides), so it never shows a hole."""
    ids = cone(part, island, base, tip, radius, up, weights, sides=sides, tag=tag)
    ring, t = ids[:-1], ids[-1]
    for k in range(1, sides - 1):
        part.tri(ring[0], ring[k], ring[k + 1], inside=part.points[t])
    return ids


# ------------------------------------------------------------------------------------------- trunk

# from under the pelvis up the neck: (centre, (dorsal-ventral half, lateral half))
TRUNK = [
    ((0, .505, -.10), (.075, .084)),     # hips
    ((0, .585, -.05), (.056, .066)),     # gaunt waist
    ((0, .665, .005), (.074, .088)),     # lower ribs
    ((0, .735, .05), (.084, .104)),      # chest
    ((0, .79, .10), (.074, .098)),       # hunched shoulders
    ((0, .865, .19), (.076, .058)),      # neck: thick and deep (a horse's), narrower side to side, arched up
    ((0, .93, .275), (.062, .05)),       # neck top, into the back of the skull
]
TRUNK_PATH = np.array([c if i < 2 else lean(c) for i, (c, _) in enumerate(TRUNK)], float)
TRUNK_RADII = [r for _, r in TRUNK]
TRUNK_BOTTOM = np.array([0, .452, -.122])
TRUNK_TOP = lean([0, .955, .315])
TRUNK_BONES = ["root", "spine1", "chest", "neck1", "neck2"]
TRUNK_HINT = (0, 1.0, -0.8)       # dorsal: the back


def trunk_shape(i, th):
    c, s = np.cos(th), abs(np.sin(th))
    if i == 4 and c > 0.5:            # the hump of the hunched shoulders
        return 1.25, 1.0
    if i == 3 and c < -0.5:           # a deep, keeled ribcage
        return 1.12, 1.0
    if i == 1 and c < -0.5:           # the gut sucked in under the ribs
        return 0.85, 1.0
    if i in (5, 6) and c > 0.5:       # the crest of the neck under the mane
        return 1.1, 1.0
    return 1.0, 1.0


NECK_RING = 3      # the trunk's UVs are two islands, torso (rings 0..3) and neck (3..6), so neither stretches


def loft_islands(part, ranges, path, radii, sides, hint, weights, *, start_point=None, end_point=None, shape=None,
                 tag=None, side_axis_sign=1.0):
    """dragon_model.loft, laid out as several UV islands along its length: `ranges` = [(island, first ring,
    last ring)], consecutive ranges sharing their boundary ring (its vertices doubled, one copy per island;
    no extra triangles). Frames come from the whole path, so the halves meet exactly. Each island's v runs
    0..1 over its own arc length (with the cap that ends it).

    With an odd number of sides one face straddles the ventral / back line, where mirrored u (|angle| / pi)
    would give both its edges the same u and smear the texture across it in bands. Its far edge gets its own
    copy of the ring's vertices with u = angle / pi (past 1), so it spans the back line properly: the island's
    u then runs 0..U_MAX(sides) (see straddle_u)."""
    path = np.asarray(path, float)
    frames = _frames(path, hint)
    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    extra0 = np.linalg.norm(path[0] - start_point) if start_point is not None else 0.0
    extra1 = np.linalg.norm(end_point - path[-1]) if end_point is not None else 0.0
    n = len(path)
    for isl, a, b in ranges:
        lo = seglen[a] - (extra0 if a == 0 else 0.0)
        hi = seglen[b] + (extra1 if b == n - 1 else 0.0)
        vmap = lambda x: (x - lo) / (hi - lo)
        rings = []
        for i in range(a, b + 1):
            c = path[i]
            t, d = frames[i]
            lat = side_axis_sign * _unit(np.cross(t, d))
            ring = []
            for k in range(sides):
                th = 2 * np.pi * k / sides
                sd, sl = shape(i, th) if shape else (1.0, 1.0)
                p = c + radii[i][0] * sd * np.cos(th) * d + radii[i][1] * sl * np.sin(th) * lat
                u = (th if th <= np.pi else 2 * np.pi - th) / np.pi
                ring.append(part.add(p, (u, vmap(seglen[i])), weights(i, p), isl, **(tag(i, th) if tag else {})))
            if sides % 2:            # the straddling face's far edge: u past 1
                kb = (sides + 1) // 2
                th = 2 * np.pi * kb / sides
                q = part.points[ring[kb]]
                ring.append(part.add(q, (th / np.pi, vmap(seglen[i])), weights(i, q), isl,
                                     **(tag(i, th) if tag else {})))
            rings.append(ring)
        kb = (sides + 1) // 2 if sides % 2 else None
        for j in range(len(rings) - 1):
            mid = (path[a + j] + path[a + j + 1]) / 2
            for k in range(sides):
                k1 = sides if (kb is not None and (k + 1) % sides == kb) else (k + 1) % sides
                p0, p1 = rings[j][k], rings[j][k1]
                q0, q1 = rings[j + 1][k], rings[j + 1][k1]
                part.tri(p0, p1, q0, inside=mid)
                part.tri(p1, q1, q0, inside=mid)
        if a == 0 and start_point is not None:
            sv = part.add(start_point, (0.5, 0.0), weights(-1, start_point), isl, **(tag(-1, 0.0) if tag else {}))
            inner = path[0] + (path[1] - path[0]) * 0.5
            for k in range(sides):
                part.tri(sv, rings[0][k], rings[0][(k + 1) % sides], inside=inner)
        if b == n - 1 and end_point is not None:
            ev = part.add(end_point, (0.5, 1.0), weights(n, end_point), isl, **(tag(n, 0.0) if tag else {}))
            inner = path[-1] + (path[-2] - path[-1]) * 0.5
            for k in range(sides):
                part.tri(ev, rings[-1][(k + 1) % sides], rings[-1][k], inside=inner)


def straddle_u(sides):
    """The largest u of an odd-sided loft_islands loft (1 for an even one)."""
    return 2 * ((sides + 1) // 2) / sides if sides % 2 else 1.0


def build_trunk(parts):
    part = SPart("trunk", "Sejanus_Body")
    w = lambda i, p: auto_weights(p, TRUNK_BONES)
    loft_islands(part, [("torso", 0, NECK_RING), ("neck", NECK_RING, len(TRUNK_PATH) - 1)], TRUNK_PATH,
                 TRUNK_RADII, 8, TRUNK_HINT, w, start_point=TRUNK_BOTTOM, end_point=TRUNK_TOP, shape=trunk_shape,
                 tag=lambda i, th: {"theta": th, "ring": i})
    parts.append(part)
    return part


TAIL_PATH = np.array([(0, .535, -.15), (0, .43, -.245), (0, .27, -.29)])
TAIL_RADII = [(.026, .028), (.036, .036), (.04, .036)]      # the hair thickens, then thins to a ragged point
TAIL_BASE = np.array([0, .58, -.105])                        # a dome up into the rump
TAIL_TIP = np.array([0, .11, -.275])
# loose locks hanging from the end of the tail: (start, tip)
TAIL_LOCKS = [((.028, .25, -.29), (.05, .12, -.31)), ((-.026, .23, -.275), (-.045, .1, -.25))]
TAIL_BONES = ["root", "tail1", "tail2", "tail3"]


def build_tail(parts):
    part = SPart("tail", "Sejanus_Limbs")

    def w(i, p):
        if i < 0:
            return {"root": 1.0}
        if i >= len(TAIL_PATH):
            return {"tail3": 1.0}
        return auto_weights(p, TAIL_BONES[1:]) if i else {"root": .5, "tail1": .5}

    # dorsal: the back of the hanging tail (-z, up)
    loft(part, "tail", TAIL_PATH, TAIL_RADII, 4, (0, .3, -1.0), w, start_point=TAIL_BASE, end_point=TAIL_TIP,
         shape=lambda i, th: (1.0, 1.0) if i < 2 else ((1.15, .9) if np.cos(th) > .5 else (1.0, 1.1)),
         tag=lambda i, th: {"theta": th, "ring": i})
    for a, tip in TAIL_LOCKS:
        a, tip = np.array(a), np.array(tip)
        d = _unit(tip - a)
        side = np.array([np.sign(a[0]), 0, 0])
        w = auto_weights(a, TAIL_BONES[2:])
        fin(part, "tail", a - (0, -.03, 0) - side * .01, a + (0, -.03, 0) + side * .01 + d * .0, tip,
            np.cross(d, (1.0, 0, 0)), w, {"tail3": 1.0}, uv_box=(0, .55, 1, 1))
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- head

def _head_section(c, lat, up, down, closed):
    """6 points: top, upper-left, lip-left, palate, lip-right, upper-right (u 0, 1/3, 2/3, 1, ...). A closed
    ring's 'lips' are the sides of the jowl and its 'palate' the jowl's underside."""
    u, x = HEAD_UP, np.array([1.0, 0, 0])
    if closed:
        return [c + up * u,
                c + .72 * lat * x + .6 * up * u,
                c + .95 * lat * x - .55 * down * u,
                c - down * u,
                c - .95 * lat * x - .55 * down * u,
                c - .72 * lat * x + .6 * up * u]
    return [c + up * u,
            c + .72 * lat * x + .6 * up * u,
            c + lat * x - .7 * down * u,
            c - .35 * down * u,
            c - lat * x - .7 * down * u,
            c - .72 * lat * x + .6 * up * u]


def build_head(parts):
    part = SPart("head", "Sejanus_Head")
    rings, lips = [], []
    for i, (s, lat, up, down, closed) in enumerate(HEAD_RINGS):
        c = head_point(s)
        pts = _head_section(c, lat, up, down, closed)
        ring = []
        for k, p in enumerate(pts):
            u = [0, 1 / 3, 2 / 3, 1, 2 / 3, 1 / 3][k]
            ring.append(part.add(p, (u, s / _HEAD_LEN), "head", "skull", ring=i, k=k))
        rings.append(ring)
        lips.append((pts[2], pts[4]))
    inner = [np.mean([part.points[j] for j in r], 0) for r in rings]
    n = len(rings)
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
    # two upright, pointed horse ears on the poll (closed cones, sunk into the skull)
    for s in (1, -1):
        base = head_point(EAR[0], EAR[1], EAR[2] * s)
        tip_ = base + np.array([EAR_TIP[0] * s, EAR_TIP[1], EAR_TIP[2]])
        capped_cone(part, "ear", base, tip_, .016, (s, 0, .3), "earL" if s > 0 else "earR", sides=3)
    build_mane(part)
    parts.append(part)
    return part, rings



# The mane (Soles, 13 Nov, third look: "swap the pale lilac hair mane for the spike mane from his earlier
# version"): a row of short spikes swept back along the crest of the neck, leaning to one side then the
# other (so the game's high camera sees them), carried on down the spine by three smaller ones.
# (place along the crest 0 = shoulders .. 1 = poll, length, side lean)
MANE = [(0.0, .11, .7), (0.15, .13, -.75), (0.3, .135, .75), (0.45, .135, -.7), (0.6, .13, .7),
        (0.75, .12, -.65), (0.9, .1, .6)]
# down the spine: (place along the trunk path between rings, length, side lean)
SPINE_SPIKES = [(3.5, .095, -.6), (2.7, .08, .55), (1.9, .065, -.5)]


def _crest(f):
    """A point on the top of the neck (bind) at `f` (0 = the shoulder ring, 1 = the poll), and its tangent."""
    tops = []
    frames = _frames(TRUNK_PATH, TRUNK_HINT)
    for i in (4, 5, 6):
        _, d = frames[i]
        sd, _ = trunk_shape(i, 0.0)
        tops.append(TRUNK_PATH[i] + d * TRUNK_RADII[i][0] * sd)
    tops.append(head_point(.035, .05))
    tops = np.array(tops)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tops, axis=0), axis=1))]
    x = f * seg[-1]
    j = min(int(np.searchsorted(seg, x, side="right")) - 1, len(tops) - 2)
    u = (x - seg[j]) / (seg[j + 1] - seg[j])
    return tops[j] * (1 - u) + tops[j + 1] * u, _unit(tops[j + 1] - tops[j]), j


def _spike(part, base, t, length, lean, w):
    """One double-sided spike rooted at `base` on the dorsal line (tangent `t`, toward the head), swept back
    along it and leaning to the side `lean`."""
    up = _unit(np.cross(np.array([1.0, 0, 0]), t))
    if up[1] < 0:
        up = -up
    base = base - up * .008
    tip = base + up * length * .5 - t * length * .72 + np.array([lean * length * .6, 0, 0])
    fin(part, "mane", base - t * .034 - (lean * .008, 0, 0), base + t * .034 + (lean * .008, 0, 0), tip,
        (lean, .2, 0), w, w)


def build_mane(part):
    for f, length, lean in MANE:
        p, t, j = _crest(f)
        bones = ["chest", "neck1", "neck2", "head"]
        w = auto_weights(p, bones[max(j - 1, 0):j + 2]) if j < 3 else "head"
        if f >= 0.9:
            w = {"neck2": .5, "head": .5}
        _spike(part, p, t, length, lean, w)
    frames = _frames(TRUNK_PATH, TRUNK_HINT)
    for f, length, lean in SPINE_SPIKES:
        i = int(np.floor(f))
        u = f - i
        c = TRUNK_PATH[i] * (1 - u) + TRUNK_PATH[i + 1] * u
        d = _unit(frames[i][1] * (1 - u) + frames[i + 1][1] * u)
        rd = TRUNK_RADII[i][0] * (1 - u) + TRUNK_RADII[i + 1][0] * u
        p = c + d * rd
        _spike(part, p, _unit(TRUNK_PATH[i + 1] - TRUNK_PATH[i]), length, lean,
               auto_weights(p, ["root", "spine1", "chest"]))


def build_jaw(parts, head_rings):
    """The lower jaw: short and slim under the muzzle, from just behind the corner of the mouth (its back end
    tucked inside the head, where it hinges) to the chin; closed, its lips meet the upper lip's."""
    part = SPart("jaw", "Sejanus_Head")
    head = next(p for p in parts if p.name == "head")
    x = np.array([1.0, 0, 0])
    rings = []
    lipL3 = np.asarray(head.points[head_rings[3][2]])
    for n, (along, wide, lip, tongue, keel) in enumerate(JAW):
        lat = np.interp(along, [r[0] for r in HEAD_RINGS], [r[1] for r in HEAD_RINGS]) * wide
        if lip is None:                 # under the muzzle: just inside the upper lip, so they meet
            pL = lipL3 * np.array([wide, 1, 1]) - HEAD_UP * .002
            pR = pL * np.array([-1, 1, 1])
        else:
            pL, pR = head_point(along, lip, lat), head_point(along, lip, -lat)
        P = [head_point(along, tongue), pL, head_point(along, keel), pR]
        ring = []
        for k, p in enumerate(P):
            ring.append(part.add(p, ([0, .5, 1, .5][k], n / len(JAW)), "jaw", "jaw", ring=n, k=k))
        rings.append(ring)
    inner = [((part.points[r[1]] + part.points[r[3]]) / 2 + part.points[r[2]]) / 2 for r in rings]
    for i in range(len(rings) - 1):
        mid = (inner[i] + inner[i + 1]) / 2
        for k in range(4):
            a, b = rings[i][k], rings[i][(k + 1) % 4]
            c_, d_ = rings[i + 1][k], rings[i + 1][(k + 1) % 4]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
    chin_p = head_point(*JAW_CHIN)
    chin = part.add(chin_p, (.5, 1.0), "jaw", "jaw", ring=len(rings), k=-1)
    for k in range(4):
        part.tri(rings[-1][k], rings[-1][(k + 1) % 4], chin, inside=inner[-1] - (chin_p - inner[-1]))
    back_p = head_point(*JAW_BACK)
    back = part.add(back_p, (.5, 0.0), "jaw", "jaw", ring=-1, k=-1)
    for k in range(4):
        part.tri(rings[0][(k + 1) % 4], rings[0][k], back, inside=inner[0] + (inner[0] - back_p))
    # the cheeks of the open mouth: from the mouth's corner (on the skull) to the front of the upper lip
    # (skull) and of the lower lip (jaw). Closed, the two lips meet and it folds flat along the lip line;
    # open, it is the dark red wedge of the mouth seen from the side.
    for side, kl in ((1, 1), (-1, 3)):
        corner = np.asarray(head.points[head_rings[2][2 if side > 0 else 4]])
        upper = np.asarray(head.points[head_rings[3][2 if side > 0 else 4]])
        lower = part.points[rings[-1][kl]]
        ids = [part.add(corner, (.12, .2), "head", "jaw", cheek=True),
               part.add(upper, (.12, .45), "head", "jaw", cheek=True),
               part.add(lower, (.04, .45), "jaw", "jaw", cheek=True)]
        part.tri(*ids, inside=head_point(.27, -.02))
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- legs

LEG_PATH = np.array([(.085, .50, -.085), (.10, .33, .03), (.10, .17, -.11), (.10, .055, -.075),
                     (.10, .004, -.012)])
LEG_RADII = [(.08, .064), (.042, .036), (.03, .021), (.023, .021), (.047, .042)]   # heavy hooves
LEG_BONES = ["thighL", "shinL", "footL", "toesL"]


def leg_shape(i, th):
    c = np.cos(th)
    if i == 2 and c < -0.5:      # the point of the hock
        return 1.35, 1.0
    if i == 0 and c < -0.3:      # the hamstrings bulge behind the thigh
        return 1.12, 1.0
    return 1.0, 1.0


def build_legs(parts):
    part = SPart("legs", "Sejanus_Limbs")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        path = LEG_PATH * m
        bones = LEG_BONES if s > 0 else [_mirror_name(b) for b in LEG_BONES]
        fwd = [(0, 0, 1.0), (0, 0.3, 1.0), (0, 0, 1.0), (0, 0.3, 1.0), (0, 0.2, 1.0)]

        def w(i, p, bones=bones):
            if i <= 0:
                # (more of the pelvis at the top of the thigh: the hip crease bends, not pinches)
                return {bones[0]: .7, "root": .3} if i == 0 else {bones[0]: .45, "root": .55}
            if i >= len(LEG_PATH) - 1:
                return {bones[3]: 1.0}
            return auto_weights(p, bones)
        # the hip end is a dome up into the body (the Dragon's lesson: no hollow thigh top seen from behind)
        loft_islands(part, [("thigh", 0, 1), ("shank", 1, len(LEG_PATH) - 1)], path, LEG_RADII, 5, fwd, w,
             start_point=path[0] + np.array([-.012 * s, .07, -.005]),
             end_point=path[-1] + (0, -.004, .0), shape=leg_shape,
             tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
    # a flat sole: the hoof ring (tilted with the pastern) and its cap sit on the floor
    last = len(LEG_PATH) - 1
    for v, tg in enumerate(part.tag):
        if tg.get("ring") in (last, last + 1):
            part.points[v] = part.points[v] * (1, 0, 1) + (0, .0 if tg["ring"] > last else .002, 0)
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- arms

ARM_PATH = np.array([SHOULDER, (.155, .57, .065), (.17, .37, .115), (.175, .305, .14)]) + ARM_SHIFT
ARM_RADII = [(.04, .042), (.024, .026), (.019, .022), (.031, .015)]   # the hand flat, palm to the thigh
ARM_BONES = ["armL", "forearmL", "handL"]
FINGERS = [(-.0095, .062), (0.0, .074), (.0095, .066)]      # (offset along z at the knuckles, length)


_D = _unit(ARM_PATH[-1] - ARM_PATH[-2])
FINGER_TIP = ARM_PATH[-1] + _D * FINGERS[1][1] + np.array([-.01, 0, .006])   # the middle finger's tip (left)


def build_arms(parts):
    part = SPart("arms", "Sejanus_Limbs")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        path = ARM_PATH * m
        nm = (lambda b: b) if s > 0 else _mirror_name
        bones = [nm(b) for b in ARM_BONES]
        clav = nm("clavL")

        def w(i, p, bones=bones, clav=clav):
            if i < 0:
                return {clav: 1.0}
            if i == 0:                  # the shoulder: half on the collarbone, so a raised arm doesn't pinch
                return {bones[0]: .55, clav: .45}
            if i >= 3:
                return {bones[2]: 1.0}
            return auto_weights(p, bones)
        loft(part, "arm", path, ARM_RADII, 4, (0, 0, 1.0), w,
             start_point=path[0] + np.array([-.035 * s, .03, -.005]),
             end_point=path[-1] + _unit(path[-1] - path[-2]) * .012,
             shape=lambda i, th: (1.0, 1.0),
             tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
        # three knobbly fingers with short dark nails, hanging from the knuckles, curled a little toward the
        # palm; rooted inside the hand, so their open bases never show
        k = path[-1]
        d = _unit(path[-1] - path[-2])
        for dz, length in FINGERS:
            base = k + np.array([0, 0, dz]) - d * .016
            tip = base + d * (length + .016) + np.array([-.01 * s, 0, .006 + dz * .4])
            cone(part, "finger", base, tip, .0085, (s, 0, 0), bones[2], sides=3, tag={"finger": True})
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- all

def build_sejanus():
    """-> list of Parts in bind pose (his standing pose)."""
    parts = []
    build_trunk(parts)
    build_tail(parts)
    head, rings = build_head(parts)
    build_jaw(parts, rings)
    build_legs(parts)
    build_arms(parts)
    return parts


def counts(parts):
    return {p.name: (len(p.points), len(p.tris)) for p in parts}


def forward(local, root_offset=(0, 0, 0)):
    return _forward(local, root_offset, SKEL)


def Skinner(parts):
    return _Skinner(parts, SKEL)
