"""The Demon Spawn's clips, keyframed from scratch after the DK1 spawn's own animations (CREATURE.JTY;
demonspawn.cfg [sprites]): the waddling walk 664, the coil-rear-and-pounce bite 668, the rear-up roar of
Smoke/Relax/Scream 672, the flop onto its back of DropDead 674, the legs dangling from the hand 676, the
rearing, front-pawing celebration 678, curled asleep among the lair's bones 680, the squashed recoil 670
and the cowering crouch of GotSlapped 682, the tongue lashing out at the chicken 684. Extended to every
slot the Dragon covers (scripts.dragon_anims.CLIPS), plus a dance (PrettyDance) and an entrance.

Built with the Dragon's machinery (its Track, keys and channel adder, its Pose and blend) on the spawn's
own skeleton. Channels:
  <bone>.x / .y / .z   local rotation in degrees (x pitch: + lowers a forward bone and raises a tail bone;
                       y yaw: + turns to its left; z roll) of root's children spine, chest, neck, head, jaw,
                       tail1..4
  body.*, front.*, tail.*  spread over spine + chest, neck + head, tail1..4
  jaw                  the jaw's extra opening in degrees (-JAW_OPEN closes the mouth)
  root.x/y/z           pelvis offset (model units); rootr.x/y/z the pelvis rotation (the whole body)
  FL.x/y/z, FR.*, BL.*, BR.*  each paw's offset from where it stands (IK keeps the paws planted);
                       FL.p a paw's pitch; ik 0..1 (or ikFL ...) blends a leg to FK: upperFL.x ... footBR.z
                       (the R side's y and z are mirrored by `fk`)
  ground               0..1: lift or sink the whole creature so its lowest point is on the floor
  tailfree             1 stops the tail's base turning to keep the tail on the floor (see build_pose)
  headfree             1 lets the head go below the floor (frames_of otherwise lifts it clear)
Every clip is built at 30 fps; a looping clip's last frame is a copy of its first.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_anims import Track, add
from scripts.dragon_pose import Pose, _frame, euler
from scripts.spawn_model import BONE_INDEX, HEAD, JAW_OPEN, LEGS, NB_BONES, PARENT, build_spawn, spawn_forward, \
    spawn_skinner

FPS = 30
CLOSE = -JAW_OPEN * 0.9         # a "jaw" value that shuts the mouth


class SpawnPose(Pose):
    def __init__(self, local=None, root=(0, 0, 0)):
        self.local = np.tile(np.eye(3), (NB_BONES, 1, 1)) if local is None else np.array(local)
        self.root = np.asarray(root, float)

    def world(self):
        return spawn_forward(self.local, self.root)


# ------------------------------------------------------------------------------------------- legs

REST_PAWS = {k: HEAD[BONE_INDEX[b[2]]].copy() for k, b in LEGS.items()}
SIDE = {"FL": 1, "BL": 1, "FR": -1, "BR": -1}
KNEE_HINT = {"F": np.array([1.0, .25, -.15]), "B": np.array([1.0, .2, .55])}


def _yaw_only(R):
    f = R @ np.array([0, 0, 1.0])
    return euler(0, np.degrees(np.arctan2(f[0], f[2])), 0)


def leg_ik(P, leg, target, hint, paw_world):
    """Two-bone IK of `leg` ("FL", ...): the paw bone's head reaches `target` (world), the knee/elbow in
    the plane of the hint; the paw turned to `paw_world`. As the Dragon's leg_ik, on the spawn's legs."""
    a, b, c = (BONE_INDEX[n] for n in LEGS[leg])
    Rw, Pw = P.world()
    hip = Pw[a]
    l1 = np.linalg.norm(HEAD[b] - HEAD[a])
    l2 = np.linalg.norm(HEAD[c] - HEAD[b])
    d = np.asarray(target, float) - hip
    dist = np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dn = d / np.linalg.norm(d)
    h = np.asarray(hint, float) - np.dot(hint, dn) * dn
    h /= max(np.linalg.norm(h), 1e-9)
    along = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    knee = hip + dn * along + h * np.sqrt(max(l1 * l1 - along * along, 0))
    ankle = hip + dn * dist
    bind1, bind2 = HEAD[b] - HEAD[a], HEAD[c] - HEAD[b]
    bn = np.cross(bind1, bind2)
    nn = np.cross(knee - hip, ankle - knee)
    R1 = _frame(knee - hip, np.cross(nn, knee - hip)) @ _frame(bind1, np.cross(bn, bind1)).T
    R2 = _frame(ankle - knee, np.cross(nn, ankle - knee)) @ _frame(bind2, np.cross(bn, bind2)).T
    P.local[a] = Rw[PARENT[a]].T @ R1
    P.local[b] = R1.T @ R2
    P.local[c] = R2.T @ paw_world
    return P


def fk(leg, x=0.0, y=0.0, z=0.0):
    """FK channels of one leg's upper bone (the R side mirrored)."""
    s = SIDE[leg]
    return {f"{LEGS[leg][0]}.x": x, f"{LEGS[leg][0]}.y": y * s, f"{LEGS[leg][0]}.z": z * s}


def fk_leg(leg, upper=(0, 0, 0), lower=(0, 0, 0), paw=(0, 0, 0)):
    s = SIDE[leg]
    out = {}
    for bone, (x, y, z) in zip(LEGS[leg], (upper, lower, paw)):
        out.update({f"{bone}.x": x, f"{bone}.y": y * s, f"{bone}.z": z * s})
    return out


def paws(**kw):
    """paws(F=(x, y, z), B=...) or FL=..., FR=...: paw offsets, x mirrored for the right side
    (+x = outward)."""
    out = {}
    for k, v in kw.items():
        legs = [k + "L", k + "R"] if len(k) == 1 else [k]
        for leg in legs:
            x, y, z = v
            out.update({f"{leg}.x": x * SIDE[leg], f"{leg}.y": y, f"{leg}.z": z})
    return out


# ------------------------------------------------------------------------------------------- poses

MACROS = {
    "body": (("spine", .5), ("chest", .5)),
    "front": (("neck", .45), ("head", .55)),
    "tail": (("tail1", .25), ("tail2", .25), ("tail3", .25), ("tail4", .25)),
}
TRUNK = ["spine", "chest", "neck", "head", "tail1", "tail2", "tail3", "tail4"]


def build_pose(ch):
    """Channel values -> SpawnPose."""
    P = SpawnPose()
    rot = {b: np.zeros(3) for b in TRUNK + ["jaw"]}
    for k, v in ch.items():
        if "." not in k:
            continue
        name, ax = k.split(".")
        if ax not in ("x", "y", "z"):
            continue
        i = "xyz".index(ax)
        if name in MACROS:
            for b, w in MACROS[name]:
                rot[b][i] += v * w
        elif name in rot:
            rot[name][i] += v
    rot["jaw"][0] += ch.get("jaw", 0.0)
    # the tail lies on the floor: when the body rears (nose up) or crouches, its base turns to keep it
    # there (tailfree 1 turns this off, for the rolls and the hand)
    keep = 1.0 - float(np.clip(ch.get("tailfree", 0.0), 0, 1))
    rot["tail1"][0] += keep * (max(0.0, -ch.get("rootr.x", 0.0)) * 0.9
                               + np.degrees(max(0.0, -ch.get("root.y", 0.0)) / 0.42))
    for b, r in rot.items():
        P.local[BONE_INDEX[b]] = euler(*r)
    P.local[BONE_INDEX["root"]] = euler(ch.get("rootr.x", 0), ch.get("rootr.y", 0), ch.get("rootr.z", 0))
    P.root = np.array([ch.get("root.x", 0), ch.get("root.y", 0), ch.get("root.z", 0)])
    fkp = {}
    for leg, bones in LEGS.items():
        for b in bones:
            fkp[BONE_INDEX[b]] = euler(*[ch.get(f"{b}.{a}", 0.0) for a in "xyz"])
    Rw, _ = P.world()
    yaw = _yaw_only(Rw[BONE_INDEX["root"]])
    for leg, bones in LEGS.items():
        ik = float(np.clip(ch.get("ik" + leg, ch.get("ik", 1.0)), 0, 1))
        idx = [BONE_INDEX[b] for b in bones]
        if ik <= 0:
            for i in idx:
                P.local[i] = fkp[i]
            continue
        off = np.array([ch.get(f"{leg}.{a}", 0.0) for a in "xyz"])
        hint = yaw @ (KNEE_HINT[leg[0]] * [SIDE[leg], 1, 1])
        paw = yaw @ euler(ch.get(f"{leg}.p", 0.0), 0, 0)
        leg_ik(P, leg, REST_PAWS[leg] + off, hint, paw)
        if ik < 1:
            F = P.copy()
            for i in idx:
                F.local[i] = fkp[i]
            from scripts.dragon_pose import blend
            mixed = blend(F, P, ik)
            for i in idx:
                P.local[i] = mixed.local[i]
    return P


_SKINNER = None


def skinner():
    global _SKINNER
    if _SKINNER is None:
        _SKINNER = spawn_skinner(build_spawn())
    return _SKINNER


def frames_of(track, extra=None):
    """Sample a Track at 30 fps -> list of SpawnPoses (a loop gets its first frame appended)."""
    n = int(round(track.duration * FPS))
    times = np.arange(n) / FPS if track.wrap else np.linspace(0, track.duration, n)
    out = []
    for t in times:
        ch = track(t)
        if extra:
            extra(t, ch)
        P = build_pose(ch)
        g = ch.get("ground", 0.0)
        if g:                     # the body's lowest point (the head's too when it may go low) on the floor
            free = ch.get("headfree", 0.0) >= 1
            low = min(p[:, 1].min() for p, part in zip(skinner()(*P.world()), skinner().parts)
                      if free or part.name not in HEAD_PARTS + ("tongue",))
            P.root = P.root + np.array([0, -g * low, 0])
        P = _head_above_floor(P, ch)
        out.append(P)
    if track.wrap:
        out.append(out[0].copy())
    return out


HEAD_PARTS = ("head", "jaw")
FLOOR = 0.003


def _head_above_floor(P, ch, step=1.5, tries=30):
    """A head bowed low with the jaw hanging open would put the chin through the floor: lift the head
    (less neck and head pitch) just enough to keep it above (headfree 1 turns this off)."""
    if ch.get("headfree", 0.0) >= 1:
        return P
    sk = skinner()
    idx = [i for i, p in enumerate(sk.parts) if p.name in HEAD_PARTS]
    for _ in range(tries):
        pts = sk(*P.world())
        if min(pts[i][:, 1].min() for i in idx) >= FLOOR:
            break
        P = P.copy()
        for b in ("neck", "head"):
            P.local[BONE_INDEX[b]] = euler(-step, 0, 0) @ P.local[BONE_INDEX[b]]
    return P


def _loop(keys, d, extra=None):
    return frames_of(Track(keys, d, True), extra), True


def _once(keys, d, extra=None):
    return frames_of(Track(keys, d, False), extra), False


# ------------------------------------------------------------------------------------------- clips

def _breath(t, ch, period=2.0, amount=1.0):
    s = np.sin(2 * np.pi * t / period)
    add(ch, **{"spine__x": -1.2 * amount * s, "chest__x": 1.0 * amount * s, "root__y": 0.003 * amount * s,
               "head__x": -0.8 * amount * s})


def stand():
    d = 4.0
    keys = [(0, {}), (1.0, {"front.y": 8, "head.x": -3, "tail.y": -6}),
            (1.7, {"jaw": 8}), (2.1, {"front.y": 0, "head.x": 2, "jaw": 0, "tail.y": 0}),
            (3.0, {"front.y": -8, "head.x": -3, "tail.y": 6}), (4.0, {})]

    def extra(t, ch):
        _breath(t, ch, 2.0)
        add(ch, tail4__y=6 * np.sin(2 * np.pi * t / 2.0), tail3__y=3 * np.sin(2 * np.pi * t / 2.0 - 1))
    return _loop(keys, d, extra)


def fidgeting():
    """Looks round, then rears up and roars (DK1 672: Smoke / Relax / Scream)."""
    d = 5.0
    rear = {"rootr.x": -16, "root.y": .03, "front.x": -22, "jaw": 26, "tail.x": -6,
            **paws(F=(0.01, .06, .03))}
    keys = [(0, {}),
            (0.6, {"front.y": 32, "head.x": -8, "tail.y": -14}),
            (1.3, {"front.y": 34, "head.x": 4}),
            (1.8, {"front.y": -30, "head.x": -8, "tail.y": 14}),
            (2.5, {"front.y": -32, "head.x": 4}),
            (2.9, {"front.y": 0, "head.x": 0, "tail.y": 0}),
            (3.3, rear), (3.5, {"jaw": 34, "head.z": 8}), (3.7, {"head.z": -8}), (3.9, {"head.z": 0, "jaw": 30}),
            (4.4, {"rootr.x": 0, "root.y": 0, "front.x": 0, "jaw": 0, "tail.x": 0, **paws(F=(0, 0, 0))}),
            (5.0, {})]
    return _loop(keys, d, lambda t, ch: _breath(t, ch, 2.5))


def _gait(t, ch, period, stride, lift, bob, sway, phases, duty, lean=0.0):
    """Four paws stepping on the spot (each slides back on the ground for `duty` of its cycle, then
    swings forward lifted); the spine bends from side to side like a lizard's, the tail following."""
    ph = (t / period) % 1.0
    for leg, off in phases.items():
        p = (ph + off) % 1.0
        if p < duty:
            u = p / duty
            z, y, pitch = stride * (0.5 - u), 0.0, -10 * max(0, (u - .8) / .2)
        else:
            u = (p - duty) / (1 - duty)
            z = stride * (-0.5 + (1 - np.cos(np.pi * u)) / 2)
            y = lift * np.sin(np.pi * u)
            pitch = -25 * np.sin(np.pi * u)
        add(ch, **{f"{leg}__z": z, f"{leg}__y": y, f"{leg}__p": pitch})
    w = 2 * np.pi * ph
    add(ch, **{"root__y": -bob * (0.5 + 0.5 * np.cos(2 * w)), "rootr__z": .4 * sway * np.sin(w),
               "rootr__x": lean, "rootr__y": .8 * sway * np.sin(w),
               "spine__y": -1.2 * sway * np.sin(w), "chest__y": -1.0 * sway * np.sin(w - .3),
               "neck__y": .8 * sway * np.sin(w - .3), "head__y": .9 * sway * np.sin(w - .6),
               "head__x": 2.5 * np.cos(2 * w),
               "tail1__y": 1.4 * sway * np.sin(w + .5), "tail2__y": 1.8 * sway * np.sin(w + 1.0),
               "tail3__y": 2.2 * sway * np.sin(w + 1.6), "tail4__y": 2.6 * sway * np.sin(w + 2.2),
               "root__x": -0.01 * np.sin(w)})


WALK_PHASES = {"BL": 0.0, "FL": 0.25, "BR": 0.5, "FR": 0.75}        # a lizard's lateral sequence
TROT_PHASES = {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}


def walk():
    period = 1.0
    return _loop([(0, {"head.x": 3})], period,
                 lambda t, ch: _gait(t, ch, period, stride=.16, lift=.05, bob=.01, sway=6, phases=WALK_PHASES,
                                     duty=.68))


def running():
    period = 0.6

    def extra(t, ch):
        _gait(t, ch, period, stride=.26, lift=.075, bob=.02, sway=7, phases=TROT_PHASES, duty=.48, lean=3)
        w = 4 * np.pi * t / period
        add(ch, **{"jaw": 8 + 6 * np.sin(w), "body__x": 3 * np.sin(w), "tail__x": 4 * np.sin(w + 1)})
    return _loop([(0, {"front.x": 6})], period, extra)


def patrolling():
    period = 1.4

    def extra(t, ch):
        _gait(t, ch, period, stride=.12, lift=.04, bob=.007, sway=5, phases=WALK_PHASES, duty=.7)
        add(ch, **{"neck__y": 16 * np.sin(2 * np.pi * t / period), "head__y": 12 * np.sin(2 * np.pi * t / period - .5),
                   "head__x": -4})
    return _loop([(0, {})], period, extra)


def attack():
    """DK1 668: coils, rears with the front paws up, pounces forward with the jaws wide, bites, lands."""
    d = 1.1
    rest = {"rootr.x": 0, "root.y": 0, "root.z": 0, "front.x": 0, "head.x": 0, "jaw": 0, "tail.x": 0,
            "body.x": 0, **paws(F=(0, 0, 0), B=(0, 0, 0))}
    keys = [(0, rest),
            (0.22, {"rootr.x": 6, "root.y": -.035, "root.z": -.04, "front.x": 10, "jaw": -8, "tail.x": -6,
                    "body.x": 4}),
            (0.42, {"rootr.x": -26, "root.y": .03, "root.z": -.01, "front.x": -16, "head.x": -6, "jaw": 24,
                    "tail.x": 10, "body.x": -6, **paws(F=(.02, .13, .07))}),
            (0.58, {"rootr.x": 10, "root.y": .02, "root.z": .1, "front.x": 8, "head.x": 8, "jaw": 42,
                    "tail.x": 18, "body.x": 4, **paws(F=(.02, .05, .2), B=(0, 0, .03))}),
            (0.66, {"jaw": CLOSE, "head.x": 14}),
            (0.8, {"rootr.x": 4, "root.y": -.02, "root.z": .08, "front.x": 6, "head.x": 6, "jaw": 4,
                   **paws(F=(.01, 0, .15))}),
            (1.1, rest)]
    return _once(keys, d)


def breathe_fire():
    """Spits fire: rears its head back, swells, then thrusts the open maw forward and heaves."""
    d = 1.6
    rest = {"rootr.x": 0, "root.y": 0, "root.z": 0, "front.x": 0, "head.x": 0, "jaw": 0, "body.x": 0,
            "tail.x": 0, "front.y": 0}
    keys = [(0, rest),
            (0.4, {"rootr.x": -12, "root.y": .025, "root.z": -.03, "front.x": -24, "head.x": -12, "jaw": -6,
                   "body.x": -6, "tail.x": 8}),
            (0.55, {"rootr.x": 8, "root.y": -.01, "root.z": .04, "front.x": 12, "head.x": -14, "jaw": 40,
                    "body.x": 6, "tail.x": -4}),
            (0.85, {"front.y": 8}), (1.1, {"front.y": -8}), (1.25, {"front.y": 0, "jaw": 38}),
            (1.35, {"jaw": 4, "front.x": 4}),
            (1.6, rest)]

    def extra(t, ch):
        if 0.55 < t < 1.3:
            add(ch, **{"chest__x": 2.5 * np.sin(t * 42), "jaw": 4 * np.sin(t * 35), "root__z": .006 * np.sin(t * 42)})
    return _once(keys, d, extra)


def _rear(height=1.0, paw=(.03, .16, .06)):
    return {"rootr.x": -34 * height, "root.y": .035 * height, "root.z": -.03 * height, "front.x": -10 * height,
            **paws(F=paw)}


def happy():
    """Two little rears with the front paws up and a wagging tail (DK1 678)."""
    d = 1.6
    down = {"rootr.x": 0, "root.y": 0, "root.z": 0, "front.x": 0, "tail.x": 0, "jaw": 0, "head.x": 0,
            **paws(F=(0, 0, 0))}
    keys = [(0, down),
            (0.3, {**_rear(.55, (.02, .1, .05)), "jaw": 22, "head.x": -12}),
            (0.55, {**down, "root.y": -.015}),
            (0.85, {**_rear(.6, (.03, .11, .05)), "jaw": 26, "head.x": -14}),
            (1.15, {**down, "root.y": -.015}),
            (1.6, down)]

    def extra(t, ch):
        w = 2 * np.pi * t / 0.4
        add(ch, **{"tail2__y": 10 * np.sin(w), "tail3__y": 16 * np.sin(w - .7), "tail4__y": 20 * np.sin(w - 1.4)})
    return _loop(keys, d, extra)


def angry():
    """Stamps its front paws in turn, snapping and roaring, the tail lashing."""
    d = 1.8
    base = {"front.x": 8, "head.x": -6, "jaw": 20, "tail.x": 6, "rootr.x": 4, "root.y": -.01}
    keys = [(0, {**base, **paws(F=(0, 0, 0))}),
            (0.25, {"FL.y": .09, "FL.z": .03, "rootr.z": -4, "jaw": 34}),
            (0.4, {"FL.y": 0, "FL.z": .02, "root.y": -.025, "rootr.z": 0, "jaw": 10}),
            (0.7, {"FR.y": .09, "FR.z": .03, "rootr.z": 4, "jaw": 36, "root.y": -.01}),
            (0.85, {"FR.y": 0, "FR.z": .02, "root.y": -.025, "rootr.z": 0, "jaw": 8}),
            (1.15, {"front.x": -14, "head.x": -16, "jaw": 42, "rootr.x": -10, "root.y": .01}),
            (1.45, {"front.x": -10, "jaw": 40}),
            (1.8, {**base, **paws(F=(0, 0, 0))})]

    def extra(t, ch):
        w = 2 * np.pi * t / 0.6
        add(ch, **{"tail2__y": 14 * np.sin(w), "tail3__y": 20 * np.sin(w - 1), "tail4__y": 24 * np.sin(w - 2),
                   "head__z": 5 * np.sin(2 * np.pi * t / 0.45)})
    return _loop(keys, d, extra)


def celebrating():
    """Rears high on its back legs, pawing the air and roaring (DK1 678 Celebrate / Roar)."""
    period = 1.4

    def extra(t, ch):
        ph = 2 * np.pi * t / period
        h = 0.5 - 0.5 * np.cos(ph)                     # 0 down .. 1 reared
        up = _rear(0.3 + 0.7 * h, (0, 0, 0))
        add(ch, **{k.replace(".", "__"): v for k, v in up.items() if not k[:2] in ("FL", "FR")})
        paw = 0.05 + 0.15 * h
        add(ch, **{"FL__y": paw + .04 * np.sin(2 * ph), "FR__y": paw - .04 * np.sin(2 * ph),
                   "FL__z": .05 * h + .03 * np.sin(2 * ph + 1), "FR__z": .05 * h - .03 * np.sin(2 * ph + 1),
                   "FL__x": .03 * h, "FR__x": -.03 * h,
                   "jaw": 10 + 30 * h, "head__x": 8 * h, "tail__y": 14 * np.sin(ph * 2),
                   "head__z": 6 * np.sin(ph)})
    return _loop([(0, {})], period, extra)


def dance():
    """A pretty dance (DK1 PrettyDance 678): swaying side to side, stepping with the front paws in turn,
    the head bobbing and the tail swinging."""
    period = 2.0

    def extra(t, ch):
        ph = 2 * np.pi * t / period
        s = np.sin(ph)
        add(ch, **{"root__x": .035 * s, "rootr__z": -7 * s, "rootr__y": 6 * s, "body__y": -10 * s,
                   "front__y": 14 * s, "front__z": 8 * s, "front__x": -6 + 6 * np.sin(2 * ph),
                   "tail__y": -26 * np.sin(ph + .6),
                   "FL__y": .07 * max(0, np.sin(2 * ph)), "FR__y": .07 * max(0, -np.sin(2 * ph)),
                   "FL__x": .035 * s, "FR__x": .035 * s, "BL__x": .02 * s, "BR__x": .02 * s,
                   "root__y": .012 * max(0, np.sin(2 * ph)) - .008, "jaw": 12 + 10 * np.sin(2 * ph)})
    return _loop([(0, {})], period, extra)


def getting_hit():
    """Squashed flat by the blow, legs splaying (DK1 670), then back up."""
    d = 0.5
    keys = [(0, {}),
            (0.09, {"root.y": -.05, "root.z": -.03, "rootr.x": -8, "front.x": -16, "jaw": 24, "tail.x": 10,
                    **paws(F=(.04, 0, .02), B=(.03, 0, -.01))}),
            (0.5, {"root.y": 0, "root.z": 0, "rootr.x": 0, "front.x": 0, "jaw": 0, "tail.x": 0,
                   **paws(F=(0, 0, 0), B=(0, 0, 0))})]
    return _once(keys, d)


def being_slapped():
    """Cowers: ducks low, the head turned away and tucked in (DK1 682), then shakes it off."""
    d = 0.6
    keys = [(0, {}),
            (0.1, {"root.y": -.06, "rootr.z": 10, "rootr.y": -10, "front.x": 22, "front.y": 26, "jaw": CLOSE,
                   "tail.y": 20, "tail.x": -6, **paws(F=(.02, 0, -.02))}),
            (0.35, {"front.y": 10, "rootr.z": 4, "root.y": -.03}),
            (0.6, {"root.y": 0, "rootr.z": 0, "rootr.y": 0, "front.x": 0, "front.y": 0, "jaw": 0, "tail.y": 0,
                   "tail.x": 0, **paws(F=(0, 0, 0))})]
    return _once(keys, d)


def staggering():
    """Drunk: swaying, lurching steps, the head lolling, the tongue out."""
    d = 3.0

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"rootr__z": 9 * np.sin(w), "rootr__x": 4 * np.sin(2 * w), "root__x": .025 * np.sin(w),
                   "front__z": -12 * np.sin(w - .7), "front__y": 14 * np.sin(2 * w + .3),
                   "front__x": 10 + 6 * np.sin(2 * w), "jaw": 10 + 6 * np.sin(3 * w),
                   "tail__y": 16 * np.sin(w + 1),
                   "FL__y": .04 * max(0, np.sin(w + 1.6)), "FR__y": .04 * max(0, -np.sin(w + 1.6)),
                   "FL__x": .03 * np.sin(w), "FR__x": .03 * np.sin(w),
                   "BL__y": .03 * max(0, -np.sin(w + .2)), "BR__y": .03 * max(0, np.sin(w + .2)),
                   "root__y": -.015})
    return _loop([(0, {})], d, extra)


LOW = {"rootr.x": 6, "root.y": -.02, "front.x": 14, "head.x": 8}


def drinking():
    d = 2.0
    keys = [(0, {**LOW, "jaw": 0})]
    for i, t in enumerate(np.arange(0.2, 1.8, 0.2)):
        keys.append((t, {"jaw": 8 if i % 2 == 0 else -12, "head.x": 12 if i % 2 == 0 else 6}))
    keys.append((1.85, {"jaw": 0, "head.x": 10}))
    return _loop(keys, d, lambda t, ch: _breath(t, ch, 1.0, .6))


EAT_TIMES = {"shoot": 0.42, "hit": 0.58, "reel": 0.95, "gone": 1.35, "back": 2.0}


def eating():
    """Eyes the chicken, flicks its tongue out to it, reels it in and swallows it whole (DK1 684). The
    tongue's and the chicken's flight are added by spawn_build."""
    d = 2.4
    look = {"front.x": 10, "head.x": 4, "rootr.x": 4, "root.y": -.01, "jaw": -4}
    keys = [(0, look), (0.3, {**look, "front.x": 14, "root.z": -.015}),
            (0.42, {"jaw": 30, "front.x": 6, "root.z": .01}),
            (0.6, {"jaw": 34}),
            (0.95, {"jaw": 30, "front.x": 4, "root.z": 0}),
            (1.05, {"jaw": CLOSE, "head.x": -6}),                          # chomp
            (1.15, {"jaw": 10}), (1.25, {"jaw": CLOSE}),
            (1.45, {"front.x": -16, "head.x": -14, "jaw": -10}),            # head up: gulp
            (1.7, {"front.x": -4, "head.x": 0, "jaw": 4}),
            (2.4, look)]

    def extra(t, ch):
        if 1.45 < t < 1.9:
            u = (t - 1.45) / .45
            add(ch, **{"neck__x": 6 * np.sin(np.pi * u), "chest__x": 3 * np.sin(np.pi * max(0, u - .3) / .7)})
    return _loop(keys, d, extra)


def manufacturing():
    """Hammers at the work with its front claws in turn."""
    d = 1.2
    keys = [(0, {"front.x": 14, "rootr.x": 6, "root.y": -.01, "FL.y": .0, "FR.y": .1, "FR.z": .04, "jaw": 4}),
            (0.3, {"FL.y": .1, "FL.z": .04, "FR.y": 0, "FR.z": .02, "rootr.z": 3}),
            (0.6, {"FL.y": 0, "FL.z": .02, "FR.y": .1, "FR.z": .04, "rootr.z": 0, "head.x": 6}),
            (0.9, {"FL.y": .1, "FL.z": .04, "FR.y": 0, "FR.z": .02, "rootr.z": -3, "head.x": 0}),
            (1.2, {})]
    return _loop(keys, d)


def praying():
    """Bows low with its chin to the floor, then rears its head up to the heavens."""
    d = 3.0
    up = {"rootr.x": -14, "root.y": .02, "front.x": -18, "head.x": -16, "jaw": 12, "tail.x": 0,
          **paws(F=(0, .03, .02))}
    bow = {"rootr.x": 9, "root.y": -.03, "front.x": 20, "head.x": 10, "jaw": CLOSE * .9, "tail.x": 8,
           **paws(F=(.03, 0, .03))}
    keys = [(0, up), (0.4, up), (1.3, bow), (2.1, bow), (3.0, up)]
    return _loop(keys, d, lambda t, ch: _breath(t, ch, 1.5, .5))


def interacting():
    """Sniffs at the thing on the floor, nudges it, paws at it."""
    d = 3.0
    keys = [(0, {**LOW, "front.y": 16}),
            (0.6, {"front.y": -16, "jaw": 8}), (0.75, {"jaw": 0}),
            (1.2, {"front.y": 4}),
            (1.5, {"FL.z": .07, "FL.y": .04, "FL.p": -10}), (1.7, {"FL.z": -.02, "FL.y": 0, "FL.p": 0}),
            (1.9, {"FL.z": .07, "FL.y": .04}), (2.1, {"FL.z": 0, "FL.y": 0}),
            (2.5, {"front.y": 18, "jaw": 8}), (2.65, {"jaw": 0}),
            (3.0, {**LOW, "front.y": 16})]
    return _loop(keys, d, lambda t, ch: add(ch, head__z=3 * np.sin(2 * np.pi * t / 0.5)))


# ---- lying, falling and held ----------------------------------------------------------------------

SPRAWL = {"root.y": -.085, **paws(F=(.08, 0, .02), B=(.06, 0, -.03)), "FL.p": 0}
CURL = {"tail1.y": 16, "tail2.y": 26, "tail3.y": 30, "tail4.y": 30}


def sleeping():
    """Flat on its belly in the bed, legs sprawled, the tail curled round, the head on the floor (DK1 680)."""
    d = 4.0
    lie = {**SPRAWL, "ground": 1, **CURL, "front.y": 22, "front.x": 30, "head.z": 12, "jaw": CLOSE,
           "rootr.y": -8}

    def extra(t, ch):
        s = np.sin(2 * np.pi * t / d)
        add(ch, **{"chest__x": 1.8 * s, "spine__x": -1.2 * s, "jaw": 2 + 2 * s, "head__x": -1.2 * s,
                   "tail4__y": 6 * np.sin(2 * np.pi * t / 2.0) * max(0, np.sin(2 * np.pi * t / d - 1))})
    return _loop([(0, lie)], d, extra)


# lying on its right side: the upper (left) legs flop down across the body, the lower ones splay
LIMP = {"ik": 0, **fk_leg("FL", (15, 0, -35), (0, 0, -25)), **fk_leg("FR", (-10, 0, 15), (0, 0, -20)),
        **fk_leg("BL", (-20, 0, -30), (10, 0, -20)), **fk_leg("BR", (10, 0, 15), (0, 0, -25))}
FALLEN = {"rootr.z": 82, "ground": 1, "tailfree": 1, "headfree": 1, **LIMP, "front.z": -18, "front.x": 8, "jaw": 18, "tail.y": 18,
          "tail.z": -10}


def falling():
    d = 1.0
    keys = [(0, {"ground": 1, "ik": 1, "tailfree": 0, "headfree": 0}),
            (0.15, {"front.x": -18, "jaw": 30, "rootr.x": -10, "root.y": .02}),
            (0.45, {**FALLEN, "rootr.z": 55, "rootr.x": -6}),
            (0.62, {**FALLEN, "rootr.x": 0}),
            (0.75, {"front.z": -6, "tail.y": 26}),
            (1.0, FALLEN)]
    return _once(keys, d)


def waking_up():
    d = 1.6
    zero = {k: 0 for k in LIMP if k != "ik"}
    keys = [(0, FALLEN),
            (0.5, {**FALLEN, "rootr.z": 40, "ground": 1, "front.z": 0, **fk_leg("FL", (0, 0, -10), (0, 0, -40))}),
            (0.8, {**zero, "ik": 1, "rootr.z": 0, "tailfree": 0, "headfree": 0, "root.y": -.05, "front.x": 10, "jaw": 4,
                   "tail.y": 10, "tail.z": 0}),
            (1.05, {"root.y": 0, "front.x": 0, "front.z": 14}), (1.2, {"front.z": -14}),
            (1.35, {"front.z": 8}), (1.6, {"front.z": 0, "tail.y": 0, "jaw": 0})]
    return _once(keys, d)


# on its back, the legs sticking up into the air, stiff and splayed (DK1 674)
ON_BACK = {"rootr.z": 178, "ground": 1, "ik": 0, "tailfree": 1, "headfree": 1,
           **fk_leg("FL", (25, 0, -5), (0, 0, -30)), **fk_leg("FR", (15, 0, 5), (0, 0, -25)),
           **fk_leg("BL", (-20, 0, -5), (10, 0, -35)), **fk_leg("BR", (-30, 0, 0), (10, 0, -25)),
           "front.x": -24, "jaw": 30, "tail.x": 20, "tail.y": 10}


def die():
    """Rears up with a last roar, then topples over onto its back, the legs in the air (DK1 674)."""
    d = 2.4
    keys = [(0, {"ik": 1, "ground": 1, "tailfree": 0, "headfree": 0}),
            (0.4, {**_rear(.8, (.02, .12, .04)), "jaw": 40, "front.x": -24}),
            (0.85, {"jaw": 44, "head.z": 10}),
            (1.15, {"rootr.z": 60, "rootr.x": -10, "root.y": .02, "ik": 0, "tailfree": 1, "headfree": 1,
                    **{k: v * .5 for k, v in ON_BACK.items()
                       if k not in ("rootr.z", "ground", "ik", "tailfree", "headfree")},
                    **paws(F=(0, 0, 0)), "head.z": 0}),
            (1.5, {**ON_BACK, "rootr.x": 0, "root.y": 0, "root.z": 0}),
            (1.7, {**fk_leg("FL", (35, 0, -20), (0, 0, -10)), **fk_leg("BR", (-40, 0, -15), (10, 0, -10))}),
            (1.9, {**fk_leg("FL", (20, 0, 0), (0, 0, -35)), **fk_leg("BR", (-25, 0, 5), (10, 0, -30))}),
            (2.4, {**ON_BACK, "jaw": 34, "front.x": -28})]
    return _once(keys, d)


def being_picked_up():
    """Held by the scruff: the body hangs nose-down, legs dangling and paddling, the tail swinging (DK1 676)."""
    d = 2.0
    hang = {"ik": 0, "tailfree": 1, "headfree": 1, "rootr.x": 22, "front.x": 18, "jaw": 16, "tail.x": -24,
            **fk_leg("FL", (0, 0, -15), (0, 0, 10)), **fk_leg("FR", (0, 0, -15), (0, 0, 10)),
            **fk_leg("BL", (0, 0, -10), (0, 0, 5)), **fk_leg("BR", (0, 0, -10), (0, 0, 5))}

    def extra(t, ch):
        w = 2 * np.pi * t / d
        k = 3 * w
        add(ch, **{"rootr__z": 8 * np.sin(w), "rootr__x": 5 * np.sin(w + 1),
                   "upperFL__x": -25 * np.sin(k), "upperFR__x": 25 * np.sin(k),
                   "thighBL__x": 22 * np.sin(k + 1), "thighBR__x": -22 * np.sin(k + 1),
                   "foreFL__x": 15 * np.sin(k + .6), "foreFR__x": -15 * np.sin(k + .6),
                   "tail__y": 20 * np.sin(w + .5), "tail__x": 6 * np.sin(2 * w), "front__y": 12 * np.sin(w - .7),
                   "head__z": 8 * np.sin(w - 1.2), "jaw": 8 * np.sin(k)})
    return _loop([(0, hang)], d, extra)


def torture():
    """Strapped spread-eagled to the wheel, writhing."""
    d = 2.4
    spread = {"ik": 0, "tailfree": 1, "headfree": 1, **fk_leg("FL", (-30, 0, 35), (0, 0, -10)), **fk_leg("FR", (-30, 0, 35), (0, 0, -10)),
              **fk_leg("BL", (30, 0, 35), (0, 0, -10)), **fk_leg("BR", (30, 0, 35), (0, 0, -10)),
              "front.x": -14, "jaw": 30, "tail.x": -10}

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"body__y": 10 * np.sin(w), "body__z": 6 * np.sin(2 * w), "front__y": -22 * np.sin(w + .5),
                   "front__x": 8 * np.sin(2 * w + 1), "head__z": 16 * np.sin(w + 1.3), "jaw": 14 * np.sin(3 * w),
                   "tail__y": 24 * np.sin(w + 2),
                   "upperFL__x": 14 * np.sin(2 * w), "upperFR__x": -14 * np.sin(2 * w),
                   "thighBL__x": -12 * np.sin(2 * w + 1), "thighBR__x": 12 * np.sin(2 * w + 1)})
    return _loop([(0, spread)], d, extra)


def entrance():
    """Comes into the dungeon: drops in, lands squashed, then rears up with a roar and settles."""
    d = 1.8
    air = {"root.y": .3, "ik": 0, "rootr.x": -6, "front.x": -10, "jaw": 20, "tail.x": 8, "tailfree": 1,
           **fk_leg("FL", (-15, 0, 5), (0, 0, 0)), **fk_leg("FR", (-15, 0, 5), (0, 0, 0)),
           **fk_leg("BL", (15, 0, 5), (0, 0, 0)), **fk_leg("BR", (15, 0, 5), (0, 0, 0))}
    zero_fk = {k: 0 for k in air if k.endswith((".x", ".y", ".z")) and k[:-2] not in ("root", "rootr", "front", "tail")}
    keys = [(0, air),
            (0.2, {**zero_fk, "ik": 1, "root.y": .035, "tailfree": .5}),        # paws down first
            (0.3, {"ik": 1, "tailfree": 0, "root.y": -.05, "rootr.x": 6, "front.x": 12, "jaw": 0,
                   "tail.x": 0}),
            (0.5, {"root.y": 0, "rootr.x": 0, "front.x": 0}),
            (0.85, {**_rear(.7, (.02, .1, .05)), "jaw": 40, "front.x": -26}),
            (1.2, {"jaw": 44, "head.z": 8}),
            (1.5, {"rootr.x": 0, "root.y": 0, "root.z": 0, "front.x": 0, "tail.x": 0, "jaw": 4, "head.z": 0,
                   **paws(F=(0, 0, 0))}),
            (1.8, {"jaw": 0})]
    return _once(keys, d)


CLIPS = {
    "stand": stand, "fidgeting": fidgeting, "walk": walk, "running": running, "patrolling": patrolling,
    "attack": attack, "breathe_fire": breathe_fire, "happy": happy, "angry": angry, "celebrating": celebrating,
    "dance": dance, "getting_hit": getting_hit, "being_slapped": being_slapped, "staggering": staggering,
    "drinking": drinking, "eating": eating, "manufacturing": manufacturing, "praying": praying,
    "interacting_with_objects": interacting, "sleeping": sleeping, "falling": falling,
    "waking_up": waking_up, "die": die, "being_picked_up": being_picked_up, "torture": torture,
    "entrance": entrance,
}


def build_clips(names=None):
    """-> {name: (list of SpawnPoses, wrap)}."""
    return {n: CLIPS[n]() for n in (names or CLIPS)}
