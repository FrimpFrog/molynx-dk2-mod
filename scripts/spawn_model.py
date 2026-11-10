"""The Demon Spawn (from Dungeon Keeper 1) as a low-poly Dungeon Keeper 2 creature: skeleton, mesh, skin
weights, UVs. Built the way the Dragon is (scripts.dragon_model: its Part, loft, cone, fin, forward and
Skinner, here on the spawn's own Skeleton).

To the DK1 sprites (CREATURE.JTY, Stand 666, Ambulate 664, Attack 668 ...; demonspawn.cfg): the baby dragon,
a squat four-legged olive-green reptile. A wide, heavy, domed body hunched forward, a row of pale spikes
from the crown of the head down the spine and the long tapering tail that lies on the ground behind it; a
short broad head held low with small orange-red eyes on top and a big gaping, glowing red-orange maw (the
jaw is built open, JAW_OPEN, and the mouth's inside is painted as a glow); splayed front legs with long
bone-coloured claws, thick back legs; a red tongue that lolls out of the mouth (the eating clip shoots it
at the chicken, see spawn_build).

Model space as the Dragon's: Y up, Z forward, X to the creature's LEFT; triangles counter-clockwise seen
from outside (the KMF export mirrors x into the game's winding). The model is about 1.0 unit from nose
to tail tip and 0.41 tall at the hump (spikes 0.47); spawn_build scales its files. 327 triangles.

Parts (two 128 x 128 skins: DemonSpawn_Body, DemonSpawn_Head):
  trunk   one 8-sided loft from the tail tip to the neck (belly plates on its ventral faces)
  legs    four 4-sided lofts shoulder-elbow-wrist-hand (hip-knee-ankle-foot), three claws each
  head    skull (6-sided loft, the palate its floor), upper fangs, the crown spike, the throat
  jaw     lower jaw (5-sided loft, U-shaped, built open), lower fangs, the throat
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
    ("root", None, (0, 0.215, -0.12), (0, 0.245, 0.03)),
    ("spine", "root", (0, 0.245, 0.03), (0, 0.235, 0.16)),
    ("chest", "spine", (0, 0.235, 0.16), (0, 0.225, 0.24)),
    ("neck", "chest", (0, 0.225, 0.24), (0, 0.235, 0.28)),
    ("head", "neck", (0, 0.235, 0.28), (0, 0.215, 0.43)),
    ("jaw", "head", (0, 0.2, 0.22), (0, 0.125, 0.42)),
    ("tail1", "root", (0, 0.165, -0.22), (0, 0.115, -0.32)),
    ("tail2", "tail1", (0, 0.115, -0.32), (0, 0.075, -0.42)),
    ("tail3", "tail2", (0, 0.075, -0.42), (0, 0.045, -0.51)),
    ("tail4", "tail3", (0, 0.045, -0.51), (0, 0.025, -0.6)),
    ("upperFL", "chest", (0.16, 0.195, 0.16), (0.3, 0.155, 0.2)),
    ("foreFL", "upperFL", (0.3, 0.155, 0.2), (0.33, 0.035, 0.25)),
    ("handFL", "foreFL", (0.33, 0.035, 0.25), (0.35, 0.012, 0.31)),
    ("thighBL", "root", (0.15, 0.195, -0.12), (0.29, 0.145, -0.03)),
    ("shinBL", "thighBL", (0.29, 0.145, -0.03), (0.29, 0.042, -0.15)),
    ("footBL", "shinBL", (0.29, 0.042, -0.15), (0.31, 0.012, -0.08)),
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
LEGS = {"FL": ("upperFL", "foreFL", "handFL"), "FR": ("upperFR", "foreFR", "handFR"),
        "BL": ("thighBL", "shinBL", "footBL"), "BR": ("thighBR", "shinBR", "footBR")}


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
    (0, .06, -.49), (0, .095, -.37), (0, .148, -.245),         # the tail, lying on the ground
    (0, .215, -.12), (0, .245, .03), (0, .235, .16),           # pelvis, the hump, the shoulders
    (0, .23, .24),                                             # the neck (inside the head's back)
])
# (dorsal-ventral, lateral) half sizes: a squat, wide, heavy, domed body
TRUNK_RADII = [(.028, .032), (.05, .058), (.085, .1), (.16, .205), (.165, .235), (.14, .21), (.1, .15)]
TRUNK_TIP = np.array([0, .04, -.6])
TRUNK_BONES = ["root", "spine", "chest", "neck", "tail1", "tail2", "tail3", "tail4"]
TAIL_RINGS = 3            # rings 0..2 are the tail


def _trunk_hint(path):
    return [(0, 1, -.3) if p[2] < -.2 else (0, 1, 0.0) for p in path]


def trunk_shape(i, th):
    c = np.cos(th)
    if c < -0.3:                 # a broad, low-slung belly
        return 0.95, 1.06
    if c > 0.6 and i >= 3:       # a high ridge along the hump
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
    ((0, .245, .23), .15, .10, .07),           # behind the eyes: the jaw muscles
    ((0, .245, .32), .165, .09, .08),          # the brow: the widest part, eyes on top
    ((0, .22, .39), .125, .06, .07),           # the blunt snout
]
SNOUT_TIP = np.array([0, .21, .43])
SKULL_BACK = np.array([0, .275, .17])
JAW_OPEN = 27.0              # degrees: the bind pose's gaping mouth
HINGE = HEAD[BONE_INDEX["jaw"]]


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
    # the crown spike on top of the head (the front view's single spike)
    fin(part, "spike", np.array([0, .33, .26]), np.array([0, .332, .33]), np.array([0, .395, .23]),
        (1, 0, 0), "head", "head")
    # upper fangs along the lip line, pointing down (double-sided)
    zs = [r[0][2] for r in HEAD_RINGS]
    for s in (1, -1):
        for z, l in ((.385, .055), (.33, .045)):
            c = np.array([0, .0, z])
            c[1] = np.interp(z, zs, [lp[0][1] for lp in lips]) + .002
            c[0] = s * np.interp(z, zs, [abs(lp[0][0]) for lp in lips]) * .88
            fin(part, "fang", c + (0, 0, -.009), c + (0, 0, .009), c + (0, -l, .006), (s, 0, 0), "head", "head")
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
    chin_p = op((0, .165, .425))
    chin = part.add(chin_p, (.5, 1.0), "jaw", "jaw", ring=3, k=-1)
    for k in range(5):
        part.tri(rings[-1][k], rings[-1][(k + 1) % 5], chin, inside=inner[-1] - (chin_p - inner[-1]))
    back_p = op((0, .175, .18))
    back = part.add(back_p, (.5, 0.0), "jaw", "jaw", ring=-1, k=-1)
    for k in range(5):
        part.tri(rings[0][(k + 1) % 5], rings[0][k], back, inside=inner[0] + (inner[0] - back_p))
    # lower fangs, pointing up
    for s in (1, -1):
        p = np.asarray(part.points[rings[2][1 if s > 0 else 4]])
        c = p + (-.006 * s, 0, -.012)
        up = R @ np.array([0, 1.0, 0])
        fin(part, "fang", c + (0, 0, -.009), c + (0, 0, .009), c + up * .04, (s, 0, 0), "jaw", "jaw")
    # the throat: a glowing wall closing the back of the gaping mouth (head lips to jaw lips, ring 0)
    hl, hr = (np.asarray(head.points[head_rings[0][k]]) for k in (2, 4))
    jl, jr = (np.asarray(part.points[rings[0][k]]) for k in (1, 4))
    hp = np.asarray(head.points[head_rings[0][3]])
    ids = [part.add(p, uv, w, "throat") for p, uv, w in
           ((hl, (0, 0), "head"), (hr, (1, 0), "head"), (jr, (1, 1), "jaw"), (jl, (0, 1), "jaw"),
            (hp, (.5, -.2), "head"))]
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
    closed = np.array([(0, .175, .27), (0, .165, .40)])
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


# ------------------------------------------------------------------------------------------- legs

LEG_PATHS = {   # left side; mirrored for the right
    "F": np.array([(.16, .195, .16), (.30, .155, .20), (.33, .035, .25)]),
    "B": np.array([(.15, .195, -.12), (.29, .145, -.03), (.29, .042, -.15)]),
}
LEG_ENDS = {"F": np.array([.343, .014, .285]), "B": np.array([.30, .014, -.115])}
LEG_RADII = {"F": [(.08, .075), (.06, .056), (.04, .04)],
             "B": [(.105, .095), (.072, .066), (.04, .04)]}
# one dorsal hint per leg (the front of the limb): a hint that turns between rings twists the 4-sided loft
LEG_HINTS = {"F": (0, .4, 1.0), "B": (0, .4, 1.0)}
CLAWS = {"F": [(-.55, .095), (0.0, .11), (.55, .09)], "B": [(-.3, .08), (.35, .08)]}


def build_legs(parts):
    part = SpawnPart("legs", "DemonSpawn_Body")
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        for fb in "FB":
            bones = LEGS[fb + ("L" if s > 0 else "R")]
            path = LEG_PATHS[fb] * m

            def w(i, p, bones=bones):
                if i == 0:
                    return {bones[0]: 1.0}
                if i >= 3:
                    return {bones[2]: 1.0}
                return auto_weights(p, list(bones))
            loft(part, "leg", path, LEG_RADII[fb], 4, LEG_HINTS[fb], w, end_point=LEG_ENDS[fb] * m,
                 tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
            ball = LEG_ENDS[fb] * m
            for dx, l in CLAWS[fb]:
                d = _unit(np.array([dx * s + (.35 * s if fb == "F" else 0), -.15, 1.0]))
                base = ball + np.array([dx * s * .028, .006, -.01])
                cone(part, "claw", base, base + d * l + (0, -.005, 0), .015, (0, 1, 0), bones[2], sides=3,
                     tag={"claw": True})
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- spikes

# (ring index along the trunk path as a float, length)
SPIKES = [(.4, .026), (1.1, .036), (1.8, .046), (2.5, .056), (3.2, .066), (3.85, .072), (4.5, .07),
          (5.2, .064), (5.85, .058)]


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
        base = c + d * r * (1.0 if i >= 3 else .95)
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
