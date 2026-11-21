"""The Necromancer's own animations and his lightning, posed by hand on a rig of the Black Knight's body.

Soles' brief (10 Nov): no ranged lightning attack; the lightning crackling from his hand is the
Necromancer's signature, shown in a new idle and in some (not all) of his other animations.

  * Storm (his Idle2, 5 s, plays once like the Black Knight's fidget): he lifts his free hand,
    summons lightning crackling in the palm and watches it for a moment, then closes his hand on
    it and settles back into his stance.
  * Walk and Run (loops; Soles: a different walk and run from the Black Knight's): an upright,
    measured march with the sword resting on his shoulder, blade pointing back, and a leaning run
    with the sword held low and forward like a charge, the free arm pumping. No Start / End files:
    the record clears those flags, as Molynx's.
  * Research (animResearchResource, loops, 4 s; the Black Knight never researches): his sword
    lowered at his side, a floating tome open over his free palm; he reads, nods, sweeps his hand
    over the pages, and now and then sparks jump over them.
  * Lightning added to the Black Knight's own motion in Angry (crackling in his shaking fist) and
    Entrance (his hand crackles as he comes out of the portal). The lightning only ever comes from
    his free palm, never the sword. Every other clip (walking, fighting, sleeping, ...) is plain.

Rig. Bones (humanoid_retarget): root, torso, head, armN/P, foreN/P, legN/P, shinN/P; N is the
sword side (-x), P the free side (+x). Every reference place gets a bone from its spot on the body
(the pose is DRKnight-Idle1 frame 0), the labels are blurred one edge for smooth joints, the blade
rides rigidly on the sword hand and the shoulder plates half on the arm, half on the body. Posing
is done in a Y-up frame: (X, Y, Z) = (x, -z, y) of the KMF (a proper rotation of it).

Angles in the key poses are rotation vectors in degrees (keypose), applied in the parent's frame:
+rx tips a bone's top forward (a hanging arm swings back), so an arm raised forward is -rx;
+rz on the P arm lifts it out to the side, -rz on the N arm.

The lightning is geometry: thin jagged ribbons (two crossed, so they show from every side) on the
bright streak of Necro_Gear, regenerated every frame so it flickers, and collapsed to a point in
frames where it is off.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import numpy as np

from scripts.humanoid_retarget import HUMANOID_BONES, Rig, keypose
from scripts.retarget import smooth_weights

BONES = [b for b in HUMANOID_BONES if b != "cape"]
PARENTS = {"root": None, "torso": "root", "head": "torso", "armN": "torso", "armP": "torso", "legN": "root",
           "legP": "root", "foreN": "armN", "foreP": "armP", "shinN": "legN", "shinP": "legP"}
FPS = 30
RESEARCH_FRAMES = 120           # loops: the last frame is a copy of the first (DK2 wrap files)
STORM_FRAMES = 150
WALK_FRAMES = 25                # loops (24 + the first again): 0.8 s a cycle of two steps
RUN_FRAMES = 19                 # loops (18 + 1): 0.6 s a cycle


def to_yup(p: np.ndarray) -> np.ndarray:
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def from_yup(q: np.ndarray) -> np.ndarray:
    return np.stack([q[..., 0], q[..., 2], -q[..., 1]], -1)


# The arms' texture islands: the arm strip of the back skin (upper arm, forearm, fist, the shoulder
# plate) and the forearm / fist block and shoulder-plate corner of the front skin.
ARM_UV = {"dknightbak": [(0.0, 0.0, 0.62, 0.33)],
          "dknightfrnt": [(0.24, 0.74, 0.83, 1.0), (0.16, 0.0, 0.43, 0.21)]}


HILT_UV = {"dknightfrnt": [(0.0, 0.55, 0.17, 1.0)]}        # the guard and grip below the blade


def _places_in(ref, table) -> np.ndarray:
    c = ref.clip
    out = np.zeros(len(ref.places), bool)
    for s in c.body:
        boxes = table.get(c.texture(s), [])
        uv = c.uv(s)
        for i, pid in enumerate(c.place[s]):
            for u0, v0, u1, v1 in boxes:
                if u0 <= uv[i, 0] <= u1 and v0 <= uv[i, 1] <= v1:
                    out[pid] = True
    return out


def _arm_places(ref) -> np.ndarray:
    P = to_yup(ref.places)
    near_neck = (P[:, 1] > 0.5) & (np.abs(P[:, 0] - 0.011) < 0.075)
    return _places_in(ref, ARM_UV) & ~near_neck


def labels(ref) -> np.ndarray:
    """Bone index (into BONES) per reference place."""
    P = to_yup(ref.places)
    x, y, z = P[:, 0] - 0.011, P[:, 1], P[:, 2]
    b = {n: i for i, n in enumerate(BONES)}
    lab = np.full(len(P), b["torso"])
    side = np.where(x < 0, "N", "P")
    arm = _arm_places(ref) & (y > 0.2)
    arm &= (np.abs(x) > 0.085) | (y < 0.345)      # waist and armpit places inside the body stay with it (fixes the waist
                                                  # stretching up to the bent sword arm)
    leg = (y < 0.325) & ~arm
    for i in np.flatnonzero(leg):
        lab[i] = b[("shin" if y[i] < 0.175 else "leg") + side[i]]
    for i in np.flatnonzero(arm):
        lab[i] = b[("fore" if y[i] < 0.435 else "arm") + side[i]]
    lab[(y > 0.60) & (np.abs(x) < 0.1)] = b["head"]
    sword = np.array(sorted(ref.sword_places()))
    lab[sword] = b["foreN"]
    lab[_places_in(ref, HILT_UV)] = b["foreN"]
    return lab


JOINTS_YUP = {"root": (0.011, 0.0, 0.03), "torso": (0.011, 0.36, 0.03), "head": (0.011, 0.60, 0.035),
              "armN": (-0.118, 0.550, 0.025), "armP": (0.145, 0.550, 0.025),
              "foreN": (-0.130, 0.432, 0.030), "foreP": (0.150, 0.432, 0.030),
              "legN": (-0.045, 0.31, 0.030), "legP": (0.065, 0.31, 0.030),
              "shinN": (-0.045, 0.165, 0.04), "shinP": (0.066, 0.165, 0.04)}


def build_rig(ref) -> tuple[Rig, np.ndarray]:
    """(the rig, its triangles as place ids)."""
    c = ref.clip
    tris = []
    for s in c.body:
        for t in c.sprites()[s].triangles[0]:
            tris.append([c.place[s][v] for v in t])
    tris = np.array(tris)
    hard = labels(ref)
    w = smooth_weights(hard, tris, len(BONES), 1)
    rigid = np.isin(hard, [BONES.index(n) for n in ("foreN", "foreP", "head", "shinN", "shinP")])
    onehot = np.eye(len(BONES))[hard]
    w[rigid] = 0.85 * onehot[rigid] + 0.15 * w[rigid]
    P_ = to_yup(ref.places)
    for side in "NP":
        fi, ai = BONES.index("fore" + side), BONES.index("arm" + side)
        ey = JOINTS_YUP["fore" + side][1]
        for i in np.flatnonzero((hard == fi) & (P_[:, 1] > ey - 0.025)):
            k = 0.5 if P_[i, 1] > ey - 0.012 else 0.25     # how much the upper arm takes
            w[i] = 0
            w[i, ai], w[i, fi] = k, 1 - k
    sword = np.array(sorted(ref.sword_places() | set(np.flatnonzero(_places_in(ref, HILT_UV)).tolist())))
    w[sword] = np.eye(len(BONES))[BONES.index("foreN")]
    # shoulder plates: half arm, half torso, so a raised arm lifts the plate without tearing it off
    P = to_yup(ref.places)
    plate = (P[:, 1] > 0.50) & (np.abs(P[:, 0] - 0.011) > 0.085)
    plate[sword] = False
    for i in np.flatnonzero(plate):
        arm = BONES.index("armN" if P[i, 0] < 0.011 else "armP")
        w[i] = 0
        w[i, arm] = 0.6
        w[i, BONES.index("torso")] = 0.4
    w /= w.sum(1, keepdims=True)
    rig = Rig(rest=P, indices=tris, joints={k: np.array(v, float) for k, v in JOINTS_YUP.items()}, weights=w,
              height=float(P[:, 1].max() - P[:, 1].min()), floor=float(P[:, 1].min()), bones=list(BONES),
              parents=dict(PARENTS))
    return rig, tris


# ------------------------------------------------------------------------------ key poses
# (t in 0..1, {bone: rotvec degrees}, root shift (X, Y, Z))

STANCE = {"armN": (0, 0, -4), "armP": (0, 0, 4)}


def research_keys():
    """Reading from a floating tome over the free hand; the sword held low at his side."""
    read = {"armP": (-38, 0, 14), "foreP": (-62, -18, 6), "head": (22, -14, 0), "torso": (6, -6, 0),
            "armN": (4, 0, -6), "foreN": (-8, 0, 0)}
    nod = dict(read, head=(30, -10, 0))
    sweep = dict(read, armP=(-46, 10, 20), foreP=(-72, 26, 4), head=(18, -4, 0), torso=(4, 4, 0))
    turn = dict(read, armP=(-34, -10, 10), foreP=(-58, -30, 8), head=(24, -18, 2))
    return [(0.00, read, (0, 0, 0)),
            (0.18, nod, (0, -0.002, 0)),
            (0.34, read, (0, 0, 0)),
            (0.50, sweep, (0, 0.002, 0)),
            (0.62, turn, (0, 0, 0)),
            (0.80, nod, (0, -0.002, 0))]


def storm_keys():
    """The storm idle: he lifts his free hand, summons lightning in the palm, watches it for a
    moment (a small tilt of the head), closes his hand on it and settles back."""
    look = {"armP": (-40, 8, 42), "foreP": (-62, -8, 0), "head": (18, 26, 4), "torso": (2, 8, 0)}
    watch = dict(look, foreP=(-68, -12, 0), head=(14, 32, 8))
    tilt = dict(look, foreP=(-66, -4, 0), head=(16, 24, -6))
    close = dict(look, armP=(-30, 6, 30), foreP=(-58, 0, 0), head=(12, 14, 0), torso=(2, 4, 0))
    rest = {}
    return [(0.00, rest, (0, 0, 0)),
            (0.16, look, (0, 0, 0)),
            (0.36, watch, (0, 0.001, 0)),
            (0.56, tilt, (0, 0.001, 0)),
            (0.70, close, (0, 0, 0)),
            (1.00, rest, (0, 0, 0))]


MAGIC_FRAMES = 60               # plays once: 2 s
MAGIC_RELEASE = 34              # the frame his palm drives down at the ground (the spell's shot leaves)


def magic_keys():
    """Raise Dead: he lifts his free hand and gathers lightning in the palm, then drives the palm
    down at the ground in front of him (the dead rise there) and straightens up."""
    gather = {"armP": (-40, 8, 42), "foreP": (-66, -8, 0), "head": (14, 22, 4), "torso": (2, 8, 0),
              "armN": (8, 0, -8)}
    lift = dict(gather, armP=(-70, 6, 30), foreP=(-40, -6, 0), head=(-6, 10, 0), torso=(-4, 6, 0))
    drive = {"armP": (-34, 0, 10), "foreP": (-6, 0, 0), "head": (22, 0, 0), "torso": (16, 0, 0),
             "armN": (14, 0, -10), "legN": (-12, 0, 0), "shinN": (20, 0, 0), "legP": (-8, 0, 0),
             "shinP": (16, 0, 0)}
    hold = dict(drive, armP=(-30, 0, 10), torso=(14, 0, 0), head=(26, 0, 0))
    return [(0.00, {}, (0, 0, 0)),
            (0.18, gather, (0, 0, 0)),
            (0.40, lift, (0, 0.004, 0)),
            (MAGIC_RELEASE / (MAGIC_FRAMES - 1), drive, (0, -0.012, 0)),
            (0.75, hold, (0, -0.010, 0)),
            (1.00, {}, (0, 0, 0))]


def magic_strength(f: int) -> float:
    """Palm lightning in the cast: it gathers while the hand rises, flares as the palm comes down."""
    if f % 4 == 3:
        return 0.0
    if 8 <= f < MAGIC_RELEASE - 4:
        return 0.3 + 0.5 * (f - 8) / (MAGIC_RELEASE - 12)
    if MAGIC_RELEASE - 4 <= f < MAGIC_RELEASE + 8:
        return 1.0
    if MAGIC_RELEASE + 8 <= f < MAGIC_RELEASE + 14:
        return 0.3
    return 0.0


def _step(lead: str, trail: str, swing: float, knee: float, push: float):
    """Contact pose: `lead` leg forward with its heel down, `trail` leg back on its toe."""
    return {"leg" + lead: (-swing, 0, 0), "shin" + lead: (4, 0, 0),
            "leg" + trail: (push, 0, 0), "shin" + trail: (knee, 0, 0)}


def _mirror(pose: dict) -> dict:
    """The same pose on the other side: N <-> P bones, side tilt and twist flipped."""
    out = {}
    for k, (x, y, z) in pose.items():
        k2 = k[:-1] + {"N": "P", "P": "N"}[k[-1]] if k[-1] in "NP" else k
        out[k2] = (x, -y, -z)
    return out


def walk_keys():
    """His own walk (not the Black Knight's): a heavy knight's stride with the sword resting on his
    shoulder (blade pointing back), the free arm swinging. Eight keys a cycle (contact, down,
    passing, up, for each foot), played as a smooth spline so it never stops at a key: hips sway
    over the standing leg and turn with the stride, the chest turns against them, the knees give
    on each landing and the head stays level."""
    def legs(sl, ss, wl, ws):            # stance leg / shin, swing leg / shin (N = sword side stands)
        return {"legN": (sl, 0, 0), "shinN": (ss, 0, 0), "legP": (wl, 0, 0), "shinP": (ws, 0, 0)}
    # N-side half cycle: the sword-side leg lands and carries him; the free arm swings forward
    contact = {**legs(-26, 6, 22, 22), "root": (0, 6, 0), "torso": (6, -12, 0), "head": (-6, 8, 0),
               "armP": (-30, 0, 8), "foreP": (-26, 0, 0)}
    down = {**legs(-16, 26, 16, 34), "root": (0, 4, -2), "torso": (9, -8, 2), "head": (-8, 5, 0),
            "armP": (-22, 0, 8), "foreP": (-30, 0, 0)}
    passing = {**legs(-2, 8, -18, 62), "root": (0, 0, -3), "torso": (6, 0, 2), "head": (-6, 0, 0),
               "armP": (-4, 0, 8), "foreP": (-22, 0, 0)}
    up = {**legs(12, 4, -30, 34), "root": (0, -4, -2), "torso": (5, 8, 1), "head": (-5, -4, 0),
          "armP": (12, 0, 8), "foreP": (-14, 0, 0)}
    half = [(contact, (0.000, -0.004, 0)), (down, (-0.004, -0.012, 0)),
            (passing, (-0.006, 0.006, 0)), (up, (-0.003, 0.004, 0))]
    keys = []
    for i, (pose, shift) in enumerate(half):
        keys.append((i / 8, pose, shift))
    for i, (pose, (x, y, z)) in enumerate(half):          # the other foot: mirrored legs, hips and body
        m = _mirror({k: v for k, v in pose.items() if not k.endswith("P") or k[:-1] not in ("arm", "fore")})
        # the free arm swings back as the other leg comes forward (opposite to the first half)
        m["armP"] = (-pose["armP"][0] - 18, 0, 8)
        m["foreP"] = (min(pose["foreP"][0], -10) * 0.6 - 8, 0, 0)
        keys.append((0.5 + i / 8, m, (-x, y, z)))
    # the sword arm: blade on the shoulder, bobbing a little with each step, never mirrored
    for k, (t, pose, shift) in enumerate(keys):
        bob = (-22 + 3 * np.cos(4 * np.pi * t), 0, -20)
        keys[k] = (t, {**pose, "armN": bob, "foreN": (-92, 0, 0)}, shift)
    return keys


def run_keys():
    """His own run: leaning in, the sword held low and forward like a charge, the free arm
    pumping."""
    sword = {"armN": (34, 0, -14), "foreN": (-8, 0, 0)}              # blade forward at hip height
    body = {"torso": (16, 0, 0), "head": (-12, 0, 0)}
    c1 = {**body, **sword, "legN": (-40, 0, 0), "shinN": (14, 0, 0), "legP": (30, 0, 0), "shinP": (60, 0, 0),
          "armP": (-46, 0, 8), "foreP": (-70, 0, 0), "torso": (16, -8, 0)}
    a1 = {**body, **sword, "legN": (-4, 0, 0), "shinN": (20, 0, 0), "legP": (-30, 0, 0), "shinP": (95, 0, 0),
          "armP": (-10, 0, 8), "foreP": (-80, 0, 0)}
    c2 = {**body, **sword, "legP": (-40, 0, 0), "shinP": (14, 0, 0), "legN": (30, 0, 0), "shinN": (60, 0, 0),
          "armP": (36, 0, 8), "foreP": (-80, 0, 0), "torso": (16, 8, 0), "armN": (28, 0, -14)}
    a2 = {**body, **sword, "legP": (-4, 0, 0), "shinP": (20, 0, 0), "legN": (-30, 0, 0), "shinN": (95, 0, 0),
          "armP": (-10, 0, 8), "foreP": (-80, 0, 0)}
    return [(0.00, c1, (0, -0.008, 0)), (0.25, a1, (0, 0.012, 0)),
            (0.50, c2, (0, -0.008, 0)), (0.75, a2, (0, 0.012, 0))]


def storm_strength(f: int) -> float:
    """Palm lightning strength in storm frame f (0 = off), flickering: it sparks up, holds while he
    watches it, and dies as his hand closes."""
    t = f / (STORM_FRAMES - 1)
    if f % 4 == 3:
        return 0.0
    if 0.17 <= t < 0.26:
        return 0.3 + 0.4 * (t - 0.17) / 0.09
    if 0.26 <= t < 0.64:
        return 0.7 + 0.08 * np.sin(f * 1.7)
    if 0.64 <= t < 0.70:
        return 0.3
    return 0.0


def posed(rig: Rig, keys, n: int, loop: bool, spline: bool = False) -> np.ndarray:
    """(n, places, 3) in the KMF frame. A loop's last frame is its first again."""
    if loop:
        out = keypose(rig, keys, n - 1, loop=True, ground=True, spline=spline)
        out = np.concatenate([out, out[:1]])
    else:
        out = keypose(rig, keys, n, loop=False, ground=True)
    return from_yup(out)


# ------------------------------------------------------------------------------ props

def tome(_unused, palm: np.ndarray, f: int, n: int):
    """The floating tome for frame f: points (KMF), triangles, UVs. It hovers over the free palm,
    open and tipped towards his face, bobbing and slowly turning; drawn from both sides."""
    t = f / max(n, 1)
    bob = 0.007 * np.sin(2 * np.pi * 2 * t)
    spin = np.radians(12 * np.sin(2 * np.pi * t))
    w, d, lift = 0.046, 0.034, 0.040
    centre = palm + np.array([0.0, 0.0, -(lift + bob)])
    ang = np.radians(24)                     # pages open in a shallow V
    tip = np.radians(-35)                    # the far edge raised: the pages face him (-y is ahead of him)
    cz, sz = np.cos(spin), np.sin(spin)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    cx, sx = np.cos(tip), np.sin(tip)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot = rz @ rx

    def P(x, y, z):
        return centre + rot @ np.array([x, y, z])
    hz = -np.sin(ang) * w
    ow = w * np.cos(ang)
    pts = np.array([
        P(0, -d, 0), P(0, d, 0), P(-ow, d, hz), P(-ow, -d, hz),          # left page
        P(0, -d, 0), P(0, d, 0), P(ow, d, hz), P(ow, -d, hz),            # right page
        P(0, -d * 1.08, 0.005), P(0, d * 1.08, 0.005), P(-ow * 1.06, d * 1.08, hz + 0.005),
        P(-ow * 1.06, -d * 1.08, hz + 0.005), P(ow * 1.06, d * 1.08, hz + 0.005), P(ow * 1.06, -d * 1.08, hz + 0.005),
    ])
    one = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [8, 10, 9], [8, 11, 10], [8, 9, 12], [8, 12, 13]]
    tris = np.array(one + [[a, c, b] for a, b, c in one])
    uv = np.array([[0.50, 0.00], [0.50, 0.50], [0.00, 0.50], [0.00, 0.00],
                   [0.50, 0.00], [0.50, 0.50], [1.00, 0.50], [1.00, 0.00],
                   [0.50, 0.52], [0.50, 1.00], [0.00, 1.00], [0.00, 0.52], [1.00, 1.00], [1.00, 0.52]])
    return pts, tris, uv * np.array([0.5, 1.0])            # the tome uses the left half of Necro_Gear


def _ribbons(paths, width: float, pts, tris, uvs):
    """Two crossed ribbons along each jagged path (a list of points)."""
    for path in paths:
        path = np.asarray(path)
        seg = len(path) - 1
        d = path[-1] - path[0]
        if np.linalg.norm(d) < 1e-9:
            d = np.array([0.0, 0.0, -1.0])
        d = d / np.linalg.norm(d)
        a = np.cross(d, [0.0, 0.0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.array([1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        for across in (a, b):
            base = len(pts)
            for k, p in enumerate(path):
                off = across * (width * (1 - 0.7 * k / max(seg, 1)))
                pts += [p - off, p + off]
                v = k / max(seg, 1)
                uvs += [[0.62, v], [0.88, v]]
            for k in range(seg):
                q = base + 2 * k
                tris += [[q, q + 2, q + 1], [q + 1, q + 2, q + 3], [q, q + 1, q + 2], [q + 1, q + 3, q + 2]]


def _jagged(rng, start, dirn, length, seg, jit):
    side = np.cross(dirn, [0.0, 0.0, 1.0])
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    up = np.cross(side, dirn)
    path = [np.asarray(start, float)]
    for _ in range(seg):
        path.append(path[-1] + dirn * length / seg + rng.normal(0, jit) * side + rng.normal(0, jit) * up)
    return path


BOLTS = 4          # bolts in a palm crackle
BLADE_ARCS = 3     # arcs along a blade
SEG = 5


def palm_crackle(palm, direction, f: int, strength: float, seed: int = 3):
    """Bolts leaping from the palm along `direction` (spread wide when weak, a tight burst when
    strong). Always the same number of points, collapsed to the palm when strength is 0."""
    rng = np.random.default_rng(seed + 31 * f)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    paths = []
    for _ in range(BOLTS):
        spread = 0.9 - 0.5 * min(strength, 1.0)
        v = d + rng.normal(0, spread, 3)
        v /= np.linalg.norm(v)
        length = rng.uniform(0.05, 0.10) + 0.12 * strength
        paths.append(_jagged(rng, palm, v, length * (strength > 0), SEG, 0.010 * (strength > 0)))
    pts, tris, uvs = [], [], []
    _ribbons(paths, 0.011 * min(1.0, 0.4 + strength), pts, tris, uvs)
    pts = np.array(pts)
    if strength <= 0:
        pts[:] = palm
    return pts, np.array(tris), np.array(uvs)


def blade_arcs(base, tip, f: int, strength: float, seed: int = 9):
    """Arcs crawling along the blade from guard to tip (always the same points; collapsed when off)."""
    rng = np.random.default_rng(seed + 17 * f)
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    d = tip - base
    paths = []
    for _ in range(BLADE_ARCS):
        a, b = sorted(rng.uniform(0, 1, 2))
        b = max(b, a + 0.35)
        start = base + d * a
        L = np.linalg.norm(d) * (min(b, 1.0) - a)
        paths.append(_jagged(rng, start, d / np.linalg.norm(d), L, SEG, 0.012))
    pts, tris, uvs = [], [], []
    _ribbons(paths, 0.008, pts, tris, uvs)
    pts = np.array(pts)
    if strength <= 0:
        pts[:] = base
    return pts, np.array(tris), np.array(uvs)


def gear_texture(size: int = 64):
    """Necro_Gear (64 x 64): left half the tome (pages with glowing blue runes over a dark leather
    cover with a gold skull), right half a lightning streak (white-hot core, blue glow)."""
    from PIL import Image, ImageDraw, ImageFilter
    S = size * 8
    im = Image.new("RGB", (S, S), (0, 0, 0))
    d = ImageDraw.Draw(im)
    half = S // 2
    # pages (top of the left half): parchment with rune lines
    d.rectangle([0, 0, half - 1, half - 1], fill=(206, 190, 150))
    rng = np.random.default_rng(5)
    for row in range(6):
        y = 30 + row * 38
        x = 24
        while x < half - 40:
            wdt = int(rng.integers(10, 30))
            colour = (40, 120, 230) if rng.random() < 0.35 else (60, 45, 30)
            d.line([(x, y), (x + wdt, y + int(rng.integers(-4, 5)))], fill=colour, width=6)
            x += wdt + int(rng.integers(8, 18))
    d.line([(half - 4, 0), (half - 4, half)], fill=(120, 100, 70), width=8)          # the spine's gutter
    # cover (bottom of the left half): dark leather, gold corners and skull
    d.rectangle([0, half, half - 1, S - 1], fill=(48, 26, 30))
    for cx, cy in ((12, half + 12), (half - 30, half + 12), (12, S - 30), (half - 30, S - 30)):
        d.rectangle([cx, cy, cx + 18, cy + 18], fill=(200, 150, 40))
    sx, sy = half // 2, half + half // 2
    d.ellipse([sx - 40, sy - 46, sx + 40, sy + 26], fill=(214, 168, 52))
    d.rectangle([sx - 24, sy + 14, sx + 24, sy + 40], fill=(214, 168, 52))
    d.ellipse([sx - 26, sy - 22, sx - 6, sy - 2], fill=(30, 20, 20))
    d.ellipse([sx + 6, sy - 22, sx + 26, sy - 2], fill=(30, 20, 20))
    # the bolt (right half): blue glow with a white core along the middle
    bolt = Image.new("L", (half, S), 0)
    bd = ImageDraw.Draw(bolt)
    bd.rectangle([half // 2 - 22, 0, half // 2 + 22, S], fill=255)
    glow = bolt.filter(ImageFilter.GaussianBlur(26))
    g = np.clip(np.asarray(glow, float) / 255 * 2.2, 0, 1)
    core = np.asarray(bolt.filter(ImageFilter.GaussianBlur(6)), float) / 255
    col = np.stack([0.30 * g + 0.85 * core, 0.78 * g + core, 1.0 * g + core], -1)
    col = np.clip(col * 1.3, 0, 1)
    im.paste(Image.fromarray((col * 255).astype(np.uint8)), (half, 0))
    return im.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------------------ the two clips

def _hand_sets(ref):
    lab = labels(ref)
    P = to_yup(ref.places)
    fore = lab == BONES.index("foreP")
    hand = np.flatnonzero(fore & (P[:, 1] < 0.345))
    elbow = np.flatnonzero(fore & (P[:, 1] > 0.39))
    return hand, elbow


def blade_ends(ref):
    """(guard places, tip place) of the blade."""
    sw = np.array(sorted(ref.sword_places()))
    P = ref.places[sw]
    hilt = P[np.argmin(-P[:, 2])]                      # the lowest blade point is at the guard
    far = np.linalg.norm(P - hilt, axis=1)
    return sw[far < 0.03], int(sw[np.argmax(far)])


def own_clips(ref):
    """{'Research': (Clip, [Moving], wrap), 'Storm': (...)} on the reference's sprites."""
    from scripts.necro_build import Moving, clip_from_places
    rig, _ = build_rig(ref)
    hand, elbow = _hand_sets(ref)
    guard, tip = blade_ends(ref)
    out = {}
    # research: the tome over the free palm, sparks over its pages now and then
    pose = posed(rig, research_keys(), RESEARCH_FRAMES, loop=True)
    book, sparks, tris, uv, stri, suv = [], [], None, None, None, None
    for f in range(len(pose)):
        palm = pose[f][hand].mean(0)
        g = f % (RESEARCH_FRAMES - 1)
        pts, tris, uv = tome(None, palm, g, RESEARCH_FRAMES - 1)
        book.append(pts)
        on = 0.35 if (40 <= g < 54 or 86 <= g < 98) and g % 3 != 2 else 0.0
        sp, stri, suv = palm_crackle(pts[:8].mean(0) + np.array([0, 0, -0.01]), [0, 0, -1], g, on, seed=21)
        sparks.append(sp)
    book[-1], sparks[-1] = book[0].copy(), sparks[0].copy()
    out["Research"] = (clip_from_places(ref, pose, "Research"),
                       [Moving("tome", "Necro_Gear", tris, uv, book), Moving("sparks", "Necro_Gear", stri, suv, sparks)], True)
    # his own walk and run (loops)
    for name, keys, n in (("Walk", walk_keys(), WALK_FRAMES), ("Run", run_keys(), RUN_FRAMES)):
        pose = posed(rig, keys, n, loop=True, spline=True)
        out[name] = (clip_from_places(ref, pose, name), [], True)
    # storm idle: lightning summoned in the free palm
    pose = posed(rig, storm_keys(), STORM_FRAMES, loop=False)
    pf, ptri, puv = [], None, None
    for f in range(len(pose)):
        palm = pose[f][hand].mean(0)
        up = palm - pose[f][elbow].mean(0)
        up = 0.3 * up / np.linalg.norm(up) + np.array([0.0, 0.0, -1.0])
        p, ptri, puv = palm_crackle(palm + np.array([0, 0, -0.012]), up, f, storm_strength(f))
        pf.append(p)
    out["Storm"] = (clip_from_places(ref, pose, "Storm"), [Moving("palm", "Necro_Gear", ptri, puv, pf)], False)
    # magic (Raise Dead): lightning gathered in the palm, then driven down at the ground
    pose = posed(rig, magic_keys(), MAGIC_FRAMES, loop=False)
    pf = []
    for f in range(len(pose)):
        palm = pose[f][hand].mean(0)
        along = palm - pose[f][elbow].mean(0)
        along = along / np.linalg.norm(along)
        if f < MAGIC_RELEASE - 4:
            d = 0.3 * along + np.array([0.0, 0.0, -1.0])        # up out of the open palm
        else:
            d = 0.7 * along + np.array([0.0, 0.0, 1.0])         # down at the ground
        p, ptri, puv = palm_crackle(palm, d, f, magic_strength(f), seed=57)
        pf.append(p)
    out["Magic"] = (clip_from_places(ref, pose, "Magic"), [Moving("palm", "Necro_Gear", ptri, puv, pf)], False)
    return out


# ------------------------------------------------------------------------------ lightning on his other clips

# Necromancer file -> (palm windows, blade windows) as (first, last) frames; strength flickers inside
# Only ever from his free palm: the sword never sparks (Soles, 10 Nov).
LIGHTNING_IN = {
    "Necro-Angry": (((10, 30),), ()),
    "Necro-Entrance": (((70, 124),), ()),
}


def _track(ref, clip, places):
    """Per frame, the mean clip position of the given reference places (None if the clip lacks them)."""
    want = set(int(p) for p in np.atleast_1d(places))
    sel = [(s, np.isin(clip.place[s], list(want))) for s in clip.body]
    sel = [(s, m) for s, m in sel if m.any()]
    if not sel:
        return None
    return [np.concatenate([fr[s][m] for s, m in sel]).mean(0) for fr in clip.frames]


def lightning_for(ref, clip, name: str):
    """The moving lightning props for one of the Black Knight's clips (empty when it has none)."""
    from scripts.necro_build import Moving, match
    if name not in LIGHTNING_IN:
        return []
    match(ref, clip)
    palm_w, blade_w = LIGHTNING_IN[name]
    hand, elbow = _hand_sets(ref)
    guard, tip = blade_ends(ref)
    n = len(clip.frames)
    out = []
    if palm_w:
        ph, pe = _track(ref, clip, hand), _track(ref, clip, elbow)
        frames, tri, uv = [], None, None
        for f in range(n):
            on = any(a <= f <= b for a, b in palm_w) and f % 4 != 3
            d = ph[f] - pe[f]
            d = 0.5 * d / np.linalg.norm(d) + np.array([0.0, 0.0, -1.0])
            p, tri, uv = palm_crackle(ph[f], d, f, 0.8 if on else 0.0, seed=41)
            frames.append(p)
        out.append(Moving("palm", "Necro_Gear", tri, uv, frames))
    if blade_w:
        g, t = _track(ref, clip, guard), _track(ref, clip, tip)
        if g is not None and t is not None:
            frames, tri, uv = [], None, None
            for f in range(n):
                on = any(a <= f <= b for a, b in blade_w) and f % 5 != 4
                b, tri, uv = blade_arcs(g[f], t[f], f, 1.0 if on else 0.0, seed=43)
                frames.append(b)
            out.append(Moving("blade", "Necro_Gear", tri, uv, frames))
    return out
