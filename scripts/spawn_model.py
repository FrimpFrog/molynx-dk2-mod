"""The Demon Spawn (from Dungeon Keeper 1) as a low-poly Dungeon Keeper 2 creature: skeleton, mesh, skin
weights, UVs. Built the way the Dragon is (scripts.dragon_model: its Part, loft, cone, fin, forward and
Skinner, here on the spawn's own Skeleton).

To the DK1 sprites (CREATURE.JTY, Stand 666, Ambulate 664, Attack 668 ...; demonspawn.cfg), read as the baby
dragon it is: a squat, hunched BIPED, like a little toad-lizard T-rex. A wide, heavy, domed body tilted
forward over two thick digitigrade hind legs (thigh, shin, a long raised foot, three forward toes and a
backward spur at each ankle), the long tapering tail lying on the floor behind as the counterweight. The
front limbs are arms: they hang from the shoulders under the chest, reaching forward, long clawed hands
just off the floor. A row of spikes from the crown down the spine and the tail; a short broad head held low
with big bulging toad's eyes on top (raised domes, glowing orange-red) and a big gaping, glowing maw (the
jaw is built open, JAW_OPEN) with big fangs; a red tongue lolling out (the eating clip shoots it at the
chicken, see spawn_build).

Model space as the Dragon's: Y up, Z forward, X to the creature's LEFT; triangles counter-clockwise seen
from outside (the KMF export mirrors x into the game's winding). About 1.0 unit from nose to tail tip and
0.5 tall at the hump; spawn_build scales its files.

Parts (two 128 x 128 skins: DemonSpawn_Body, DemonSpawn_Head):
  trunk   one 8-sided loft from the tail tip to the neck (belly plates on its ventral faces)
  legs    two 5-sided lofts hip-knee-ankle-ball, three toe claws and an ankle spur each
  arms    two 4-sided lofts shoulder-elbow-wrist-hand, three long claws each
  head    skull (6-sided loft, the palate its floor), two eye domes, the crown spike, upper fangs (cones)
  jaw     lower jaw (5-sided loft, U-shaped, built open), lower fangs (cones), the throat
  tongue  a flat 4-sided loft lying in the jaw, its tip hanging out
  spikes  double-sided fins along the back and the tail

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_model import Part, Skeleton, Skinner, _frames, _head_section, _mirror_name, _seg_dist, \
    _unit, cone, fin, forward, loft

# ------------------------------------------------------------------------------------------- skeleton

# name, parent, head (bind position), tail (end of its segment, for weights)
_BONES_L = [
    ("root", None, (0, .335, -.09), (0, .345, .03)),
    ("spine", "root", (0, .345, .03), (0, .31, .15)),
    ("chest", "spine", (0, .31, .15), (0, .28, .23)),
    ("neck", "chest", (0, .28, .23), (0, .285, .28)),
    ("head", "neck", (0, .285, .28), (0, .255, .43)),
    ("jaw", "head", (0, .24, .22), (0, .165, .42)),
    ("tail1", "root", (0, .27, -.21), (0, .17, -.32)),
    ("tail2", "tail1", (0, .17, -.32), (0, .11, -.42)),
    ("tail3", "tail2", (0, .11, -.42), (0, .07, -.52)),
    ("tail4", "tail3", (0, .07, -.52), (0, .04, -.63)),
    ("upperAL", "chest", (.15, .28, .17), (.24, .2, .15)),
    ("foreAL", "upperAL", (.24, .2, .15), (.23, .17, .28)),
    ("handAL", "foreAL", (.23, .17, .28), (.24, .15, .33)),
    ("thighL", "root", (.15, .3, -.07), (.26, .19, .07)),
    ("shinL", "thighL", (.26, .19, .07), (.22, .09, -.08)),
    ("footL", "shinL", (.22, .09, -.08), (.23, .02, .0)),
    ("toesL", "footL", (.23, .02, .0), (.235, .006, .08)),
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


class SpawnPart(Part):
    bone_index = BONE_INDEX


def auto_weights(p, bones, power=4.0, keep=2):
    """Inverse-distance weights to the nearest `keep` bone segments among `bones` (the spawn's)."""
    d = np.array([max(_seg_dist(p, HEAD[BONE_INDEX[b]], TAIL[BONE_INDEX[b]]), 1e-4) for b in bones])
    order = np.argsort(d)[:keep]
    return {bones[i]: d[i] ** -power for i in order}


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


# ------------------------------------------------------------------------------------------- trunk

TRUNK_PATH = np.array([
    (0, .115, -.4), (0, .22, -.26),                            # the tail, sloping down to the floor
    (0, .335, -.1), (0, .345, .03), (0, .31, .15),             # the pelvis (high), the hump, the shoulders
    (0, .28, .23),                                             # the neck (inside the head's back)
])
# (dorsal-ventral, lateral) half sizes: a squat, wide, heavy, domed body
TRUNK_RADII = [(.05, .058), (.09, .1), (.14, .165), (.155, .2), (.135, .18), (.1, .14)]
PELVIS_RING = 2
TRUNK_TIP = np.array([0, .04, -.64])
TRUNK_BONES = ["root", "spine", "chest", "neck", "tail1", "tail2", "tail3", "tail4"]


def _trunk_hint(path):
    return [(0, 1, -.3) if p[2] < -.2 else (0, 1, 0.0) for p in path]


def trunk_shape(i, th):
    c = np.cos(th)
    if c < -0.3:                 # a broad, low-slung belly
        return 0.95, 1.06
    if c > 0.6 and i >= PELVIS_RING:       # a high ridge along the hump
        return 1.06, 1.0
    return 1.0, 1.0


def build_trunk(parts):
    part = SpawnPart("trunk", "DemonSpawn_Body")
    loft(part, "trunk", TRUNK_PATH, TRUNK_RADII, 8, _trunk_hint(TRUNK_PATH),      # its neck end is in the head
         lambda i, p: auto_weights(p, TRUNK_BONES), start_point=TRUNK_TIP,
         shape=trunk_shape, tag=lambda i, th: {"theta": th, "ring": i})
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- head

HEAD_RINGS = [  # centre, lateral, up, down (to the lip line): a short, broad, toad-like head
    ((0, .295, .23), .15, .10, .07),           # behind the eyes: the jaw muscles
    ((0, .295, .32), .165, .09, .08),          # the brow: the widest part, the eyes on top
    ((0, .27, .39), .125, .06, .07),           # the blunt snout
]
SNOUT_TIP = np.array([0, .26, .43])
SKULL_BACK = np.array([0, .325, .17])
JAW_OPEN = 27.0              # degrees: the bind pose's gaping mouth
HINGE = HEAD[BONE_INDEX["jaw"]]
# the eyes: big bulging domes on top of the head (left; mirrored): base centre, outward axis
EYE_BASE = np.array([.088, .358, .30])
EYE_AXIS = np.array([.55, .75, .38])
EYE_RADII = ((.034, .034), (.042, .042))      # base ring, the bulge
EYE_STEPS = (0.0, .03, .058)                  # along the axis: base, bulge, apex
UPPER_FANGS = ((.385, .085), (.325, .065))               # (z, length), each side
LOWER_FANGS = ((2, .062),)                                # (jaw ring, length)


def build_head(parts):
    part = SpawnPart("head", "DemonSpawn_Head")
    centres = [np.array(r[0], float) for r in HEAD_RINGS]
    n = len(HEAD_RINGS)
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
    # the crown spike on top of the head (the front view's single spike), between the eyes
    fin(part, "spike", np.array([0, .38, .22]), np.array([0, .386, .28]), np.array([0, .445, .19]),
        (1, 0, 0), "head", "head")
    # the eyes: bulging domes (5-sided: the lid ring sunk in the skin, the bulge, the apex)
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        axis = _unit(EYE_AXIS * m)
        base = EYE_BASE * m
        path = np.array([base + axis * EYE_STEPS[0], base + axis * EYE_STEPS[1]])
        loft(part, "eye", path, list(EYE_RADII), 5, (0, .3, 1.0), lambda i, p: "head",
             end_point=base + axis * EYE_STEPS[2], tag=lambda i, th: {"along": min(i, 2) / 2.0},
             side_axis_sign=s)
    # big upper fangs along the lip line, pointing down and a little in (3-sided cones)
    zs = [r[0][2] for r in HEAD_RINGS]
    for s in (1, -1):
        for z, l in UPPER_FANGS:
            c = np.array([0, .0, z])
            c[1] = np.interp(z, zs, [lp[0][1] for lp in lips]) + .006
            c[0] = s * np.interp(z, zs, [abs(lp[0][0]) for lp in lips]) * .86
            cone(part, "fang", c, c + np.array([-.012 * s, -l, .008]), .016, (s, 0, 0), "head", sides=3)
    parts.append(part)
    return part, rings


def build_jaw(parts, head_rings):
    """A closed lower jaw under the head's lip line (like the Dragon's), turned open by JAW_OPEN about
    the hinge."""
    head = parts[-1]
    part = SpawnPart("jaw", "DemonSpawn_Head")
    R = _rx(JAW_OPEN)
    op = lambda p: HINGE + R @ (np.asarray(p, float) - HINGE)
    rings = []
    depths = [.06, .065, .05]
    for i in range(len(HEAD_RINGS)):
        lipL = np.asarray(head.points[head_rings[i][2]])
        lipR = np.asarray(head.points[head_rings[i][4]])
        mid = (lipL + lipR) / 2
        keel = mid + (0, -depths[i] - .012, 0)
        P = [mid + (0, -.02, 0),                                     # the tongue bed
             lipL + (lipR - lipL) * .04 + (0, -.003, 0),
             keel + (lipL - lipR) * .28,                             # a U, not a V: two keel edges
             keel + (lipR - lipL) * .28,
             lipR + (lipL - lipR) * .04 + (0, -.003, 0)]
        ring = [part.add(op(p), ([0, .5, 1, 1, .5][k], i / 3.0), "jaw", "jaw", ring=i, k=k)
                for k, p in enumerate(P)]
        rings.append(ring)
    inner = [((part.points[r[1]] + part.points[r[4]]) / 2 + (part.points[r[2]] + part.points[r[3]]) / 2) / 2
             for r in rings]
    for i in range(len(rings) - 1):
        mid = (inner[i] + inner[i + 1]) / 2
        for k in range(5):
            a, b = rings[i][k], rings[i][(k + 1) % 5]
            c_, d_ = rings[i + 1][k], rings[i + 1][(k + 1) % 5]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
    chin_p = op((0, .215, .425))
    chin = part.add(chin_p, (.5, 1.0), "jaw", "jaw", ring=3, k=-1)
    for k in range(5):
        part.tri(rings[-1][k], rings[-1][(k + 1) % 5], chin, inside=inner[-1] - (chin_p - inner[-1]))
    back_p = op((0, .225, .18))
    back = part.add(back_p, (.5, 0.0), "jaw", "jaw", ring=-1, k=-1)
    for k in range(5):
        part.tri(rings[0][(k + 1) % 5], rings[0][k], back, inside=inner[0] + (inner[0] - back_p))
    # big lower fangs, pointing up (3-sided cones)
    up = R @ np.array([0, 1.0, 0])
    for s in (1, -1):
        for r, l in LOWER_FANGS:
            p = np.asarray(part.points[rings[r][1 if s > 0 else 4]])
            c = p + (-.01 * s, -.004, -.006)
            cone(part, "fang", c, c + up * l + np.array([-.008 * s, 0, 0]), .014, (s, 0, 0), "jaw", sides=3)
    # the throat: a glowing wall closing the back of the gaping mouth (head lips to jaw lips, ring 0)
    hl, hr = (np.asarray(head.points[head_rings[0][k]]) for k in (2, 4))
    jl, jr = (np.asarray(part.points[rings[0][k]]) for k in (1, 4))
    hp = np.asarray(head.points[head_rings[0][3]])
    ids = [part.add(p, uv, w, "throat") for p, uv, w in
           ((hl, (0, 0), "head"), (hr, (1, 0), "head"), (jr, (1, 1), "jaw"), (jl, (0, 1), "jaw"),
            (hp, (.5, 0), "head"))]
    front = np.array([0, 0, 1.0])
    for a, b, c in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3]), (ids[0], ids[4], ids[1])):
        pa, pb, pc = (part.points[j] for j in (a, b, c))
        part.tri(a, b, c, inside=(pa + pb + pc) / 3 - front)
    parts.append(part)
    return part, rings


# ------------------------------------------------------------------------------------------- tongue

def tongue_path_rest():
    """The tongue lying in the open jaw, its tip lolling out over the chin (bind)."""
    R = _rx(JAW_OPEN)
    closed = np.array([(0, .225, .27), (0, .215, .40)])
    p = np.array([HINGE + R @ (c - HINGE) for c in closed])
    tip = p[-1] + np.array([0, -.05, .03])
    return np.vstack([p, tip])


TONGUE_RADII = [(.01, .045), (.009, .036)]


def build_tongue(parts, path=None, weights=None):
    """path: 2 ring centres + the tip (default the rest pose). Same vertex order for any path (the eating
    clip rebuilds it along the flying tongue)."""
    path = tongue_path_rest() if path is None else np.asarray(path, float)
    part = SpawnPart("tongue", "DemonSpawn_Head")
    w = weights or (lambda i, p: "jaw")
    hint = np.cross(_unit(path[-2] - path[0]), (1.0, 0, 0))
    hint = -hint if hint[1] < 0 else hint
    loft(part, "tongue", path[:2], TONGUE_RADII, 4, hint, w, end_point=path[2],
         tag=lambda i, th: {"ring": i})
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- limbs

# hind legs (left; mirrored): hip, knee (forward and out), ankle (back, raised), ball of the foot
LEG_PATH = np.array([(.15, .3, -.07), (.26, .19, .07), (.22, .09, -.08), (.23, .036, .0)])
LEG_END = np.array([.232, .024, .04])
LEG_RADII = [(.115, .105), (.085, .078), (.056, .05), (.038, .05)]
TOE_CLAWS = [(-.45, .075), (0.0, .085), (.45, .07)]       # (spread, length), forward from the ball
SPUR = (np.array([0, -.004, -.03]), np.array([0, -.03, -.085]))   # from the ankle: base offset, tip offset
# arms (left; mirrored): shoulder, elbow (down and back), wrist (forward): carried bent in front of the
# chest like a T-rex's, the long claws hooked down, the hands clear of the floor
ARM_PATH = np.array([(.15, .28, .17), (.24, .2, .15), (.23, .17, .28)])
ARM_END = np.array([.24, .155, .325])
ARM_RADII = [(.06, .055), (.045, .042), (.03, .034)]
HAND_CLAWS = [(-.5, .085), (0.0, .1), (.5, .08)]


def _limb_weights(bones, n_rings):
    def w(i, p):
        if i <= 0:
            return {bones[0]: 1.0}
        if i >= n_rings:
            return {bones[-1]: 1.0}
        return auto_weights(p, list(bones))
    return w


def build_legs(parts):
    part = SpawnPart("legs", "DemonSpawn_Body")
    for s, side in ((1, "L"), (-1, "R")):
        m = np.array([s, 1, 1.0])
        thigh, shin, foot, toes = LEGS[side]
        path = LEG_PATH * m
        # one dorsal hint for the whole limb (forward): a hint that turns between rings twists the loft,
        # and one parallel to a ring's tangent (up, at the knee, whose tangent runs hip to ankle) collapses it
        loft(part, "leg", path, LEG_RADII, 5, (0, .2, 1.0), _limb_weights((thigh, shin, foot, toes), 4),
             end_point=LEG_END * m, tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s},
             side_axis_sign=s)
        ball = path[3] + np.array([0, -.014, 0])           # the toes spring from the sole
        for dx, l in TOE_CLAWS:
            d = _unit(np.array([dx * s, -.12, 1.0]))
            base = ball + np.array([dx * s * .03, .002, .01])
            cone(part, "claw", base, base + d * l + (0, -.0135, 0), .016, (0, 1, 0), toes, sides=3)
        ankle = path[2]
        cone(part, "claw", ankle + SPUR[0], ankle + SPUR[1], .014, (0, 1, 0), foot, sides=3)   # the spur
    parts.append(part)
    return part


def build_arms(parts):
    part = SpawnPart("arms", "DemonSpawn_Body")
    for s, side in ((1, "L"), (-1, "R")):
        m = np.array([s, 1, 1.0])
        upper, fore, hand = ARMS[side]
        path = ARM_PATH * m
        loft(part, "arm", path, ARM_RADII, 4, (0, .3, 1.0), _limb_weights((upper, fore, hand), 3),
             end_point=ARM_END * m, tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s},
             side_axis_sign=s)
        knuckle = ARM_END * m + np.array([0, .004, -.012])
        for dx, l in HAND_CLAWS:
            d = _unit(np.array([dx * s + .2 * s, -.9, 1.0]))
            base = knuckle + np.array([dx * s * .03, 0, 0])
            cone(part, "claw", base, base + d * l, .015, (0, 1, 0), hand, sides=3)
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- spikes

# (ring index along the trunk path as a float, length)
SPIKES = [(.1, .036), (.8, .048), (1.5, .058), (2.2, .068), (2.85, .074), (3.5, .072), (4.2, .066),
          (4.85, .06)]


def build_spikes(parts):
    part = SpawnPart("spikes", "DemonSpawn_Head")
    path = TRUNK_PATH
    frames = _frames(path, _trunk_hint(path))
    for f, length in SPIKES:
        i = int(np.floor(f))
        u = f - i
        j = min(i + 1, len(path) - 1)
        c = path[i] * (1 - u) + path[j] * u
        t = _unit(frames[i][0] * (1 - u) + frames[j][0] * u)
        d = _unit(frames[i][1] * (1 - u) + frames[j][1] * u)
        r = TRUNK_RADII[i][0] * (1 - u) + TRUNK_RADII[j][0] * u
        base = c + d * r * (1.0 if i >= PELVIS_RING else .95)
        half = max(.014, length * .42)
        a, b = base - t * half, base + t * half
        tip = base + d * length - t * length * .5          # leaning toward the tail
        w = auto_weights(base, TRUNK_BONES)
        fin(part, "spike", a, b, tip, (1, 0, 0), w, w)
    parts.append(part)
    return part


def build_spawn():
    """-> list of Parts in bind pose."""
    parts = []
    build_trunk(parts)
    build_legs(parts)
    build_arms(parts)
    head, rings = build_head(parts)
    build_jaw(parts, rings)
    build_tongue(parts)
    build_spikes(parts)
    return parts


def counts(parts):
    return {p.name: (len(p.points), len(p.tris)) for p in parts}


def spawn_skinner(parts):
    return Skinner(parts, SKEL)


def spawn_forward(local, root_offset=(0, 0, 0)):
    return forward(local, root_offset, SKEL)
