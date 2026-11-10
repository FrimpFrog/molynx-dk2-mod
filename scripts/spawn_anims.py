"""The Demon Spawn's clips: a hunched little biped, keyframed from scratch after the DK1 spawn's own
animations (CREATURE.JTY; demonspawn.cfg [sprites]): the waddling two-legged walk 664 (arms hanging, the
body rocking over each foot), the crouch-and-pounce of the attack 668 (it springs off its hind legs, claws
forward, and bites), the rear-up roar of Smoke/Relax/Scream 672, the flop onto its back of DropDead 674,
the legs and arms dangling from the hand 676, the rearing, arm-waving celebration 678, curled asleep in
the lair 680, the recoil 670 and the cowering crouch of GotSlapped 682, the tongue lashing out at the
chicken 684. Extended to every slot the Dragon covers (scripts.dragon_anims.CLIPS), plus a dance
(PrettyDance) and an entrance.

Built with the Dragon's machinery (its Track, keys and channel adder, its Pose and blend) on the spawn's
own skeleton. Channels:
  <bone>.x / .y / .z   local rotation in degrees (x pitch: + lowers a forward bone and raises a tail bone;
                       y yaw: + turns to its left; z roll) of spine, chest, neck, head, jaw, tail1..4 and
                       the arms' bones (upperAL, foreAL, handAL ...: x + swings back, - forward)
  body.*, front.*, tail.*  spread over spine + chest, neck + head, tail1..4
  arms.*, fores.*, hands.*  both arms' upper, fore and hand bones at once (y and z mirrored on the right:
                       z + spreads the arms outward and up)
  jaw                  the jaw's extra opening in degrees (CLOSE shuts the mouth)
  root.x/y/z           pelvis offset (model units); rootr.x/y/z the pelvis rotation (the whole body)
  L.x/y/z, R.*         each hind foot's ankle offset from where it stands (IK keeps the feet planted;
                       x + outward); L.p the foot's pitch, L.t the toes'; ik 0..1 (or ikL, ikR) blends a
                       leg to FK: thighL.x ... toesR.z (see fk_leg)
  ground               0..1: lift or sink the whole creature so its body's lowest point is on the floor
  tailfree             1 stops the tail's base turning to keep the tail on the floor (see build_pose)
  headfree, armsfree   1 let the head / the hands go below the floor (frames_of otherwise lifts them)
Every clip is built at 30 fps; a looping clip's last frame is a copy of its first.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_anims import Track, add
from scripts.dragon_pose import Pose, _frame, blend, euler
from scripts.spawn_model import ARMS, BONE_INDEX, HEAD, JAW_OPEN, LEGS, NB_BONES, PARENT, build_spawn, \
    spawn_forward, spawn_skinner

FPS = 30
CLOSE = -JAW_OPEN * 0.9         # a "jaw" value that shuts the mouth
SIDE = {"L": 1, "R": -1}


class SpawnPose(Pose):
    def __init__(self, local=None, root=(0, 0, 0)):
        self.local = np.tile(np.eye(3), (NB_BONES, 1, 1)) if local is None else np.array(local)
        self.root = np.asarray(root, float)

    def world(self):
        return spawn_forward(self.local, self.root)


# ------------------------------------------------------------------------------------------- legs

REST_ANKLES = {s: HEAD[BONE_INDEX[LEGS[s][2]]].copy() for s in "LR"}
KNEE_HINT = np.array([.35, .1, 1.0])


def _yaw_only(R):
    f = R @ np.array([0, 0, 1.0])
    return euler(0, np.degrees(np.arctan2(f[0], f[2])), 0)


def leg_ik(P, side, target, hint, foot_world, toe_pitch=0.0):
    """Two-bone IK of a hind leg (as the Dragon's leg_ik, on the spawn's legs): thigh and shin bend so the
    ankle reaches `target` (world), the knee in the plane of the hint; the foot turned to `foot_world`, the
    toes pitched by `toe_pitch`."""
    a, b, c, d = (BONE_INDEX[n] for n in LEGS[side])
    Rw, Pw = P.world()
    hip = Pw[a]
    l1 = np.linalg.norm(HEAD[b] - HEAD[a])
    l2 = np.linalg.norm(HEAD[c] - HEAD[b])
    v = np.asarray(target, float) - hip
    dist = np.clip(np.linalg.norm(v), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dn = v / np.linalg.norm(v)
    h = np.asarray(hint, float) - np.dot(hint, dn) * dn
    h /= max(np.linalg.norm(h), 1e-9)
    along = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    knee = hip + dn * along + h * np.sqrt(max(l1 * l1 - along * along, 0))
    ankle = hip + dn * dist
    bind1, bind2 = HEAD[b] - HEAD[a], HEAD[c] - HEAD[b]
    bn = np.cross(bind1, bind2)
    nn = np.cross(knee - hip, ankle - knee)
    if np.dot(bn, nn) < 0:          # keep the knee bending the way it does in the bind pose
        nn = -nn
    R1 = _frame(knee - hip, np.cross(nn, knee - hip)) @ _frame(bind1, np.cross(bn, bind1)).T
    R2 = _frame(ankle - knee, np.cross(nn, ankle - knee)) @ _frame(bind2, np.cross(bn, bind2)).T
    P.local[a] = Rw[PARENT[a]].T @ R1
    P.local[b] = R1.T @ R2
    P.local[c] = R2.T @ foot_world
    P.local[d] = euler(toe_pitch, 0, 0)
    return P


def fk_leg(side, thigh=(0, 0, 0), shin=(0, 0, 0), foot=(0, 0, 0), toes=(0, 0, 0)):
    """FK channels of one hind leg (the R side's y and z mirrored)."""
    s = SIDE[side]
    out = {}
    for bone, (x, y, z) in zip(LEGS[side], (thigh, shin, foot, toes)):
        out.update({f"{bone}.x": x, f"{bone}.y": y * s, f"{bone}.z": z * s})
    return out


def arm(side, upper=(0, 0, 0), fore=(0, 0, 0), hand=(0, 0, 0)):
    """Channels of one arm (the R side's y and z mirrored)."""
    s = SIDE[side]
    out = {}
    for bone, (x, y, z) in zip(ARMS[side], (upper, fore, hand)):
        out.update({f"{bone}.x": x, f"{bone}.y": y * s, f"{bone}.z": z * s})
    return out


def feet(**kw):
    """feet(B=(x, y, z)) for both, or L=..., R=...: ankle offsets, x mirrored for the right (+ outward)."""
    out = {}
    for k, v in kw.items():
        for side in ("L", "R") if k == "B" else (k,):
            x, y, z = v
            out.update({f"{side}.x": x * SIDE[side], f"{side}.y": y, f"{side}.z": z})
    return out


# ------------------------------------------------------------------------------------------- poses

MACROS = {
    "body": (("spine", .5), ("chest", .5)),
    "front": (("neck", .45), ("head", .55)),
    "tail": (("tail1", .25), ("tail2", .25), ("tail3", .25), ("tail4", .25)),
}
TRUNK = ["spine", "chest", "neck", "head", "tail1", "tail2", "tail3", "tail4"]
ARM_MACROS = {"arms": 0, "fores": 1, "hands": 2}


def build_pose(ch):
    """Channel values -> SpawnPose."""
    P = SpawnPose()
    rot = {b: np.zeros(3) for b in TRUNK + ["jaw"] + [b for s in "LR" for b in ARMS[s]]}
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
        elif name in ARM_MACROS:
            for s in "LR":
                rot[ARMS[s][ARM_MACROS[name]]][i] += v * (SIDE[s] if i else 1)
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
    fkp = {BONE_INDEX[b]: euler(*[ch.get(f"{b}.{a}", 0.0) for a in "xyz"]) for s in "LR" for b in LEGS[s]}
    Rw, _ = P.world()
    yaw = _yaw_only(Rw[BONE_INDEX["root"]])
    for side, bones in LEGS.items():
        ik = float(np.clip(ch.get("ik" + side, ch.get("ik", 1.0)), 0, 1))
        idx = [BONE_INDEX[b] for b in bones]
        if ik <= 0:
            for i in idx:
                P.local[i] = fkp[i]
            continue
        off = np.array([ch.get(f"{side}.{a}", 0.0) for a in "xyz"])
        hint = yaw @ (KNEE_HINT * [SIDE[side], 1, 1])
        leg_ik(P, side, REST_ANKLES[side] + off, hint, yaw @ euler(ch.get(f"{side}.p", 0.0), 0, 0),
               ch.get(f"{side}.t", 0.0))
        if ik < 1:
            F = P.copy()
            for i in idx:
                F.local[i] = fkp[i]
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


HEAD_PARTS = ("head", "jaw")
FLOOR = 0.003


def _part_index(name):
    return next(i for i, p in enumerate(skinner().parts) if p.name == name)


_ARM_SIDES = None


def _arm_sides():
    """Vertex indices of the arms part on each side (by their bind x)."""
    global _ARM_SIDES
    if _ARM_SIDES is None:
        P = np.array(skinner().parts[_part_index("arms")].points)
        _ARM_SIDES = {"L": np.flatnonzero(P[:, 0] > 0), "R": np.flatnonzero(P[:, 0] < 0)}
    return _ARM_SIDES


def _lift_clear(P, ch):
    """Keep the head and the hands above the floor: a head bowed low with the jaw hanging open, or a
    hand hanging in a crouch, is lifted just enough (less neck and head pitch; the arm swung forward)."""
    sk = skinner()
    if ch.get("headfree", 0.0) < 1:
        idx = [i for i, p in enumerate(sk.parts) if p.name in HEAD_PARTS]
        for _ in range(45):
            pts = sk(*P.world())
            if min(pts[i][:, 1].min() for i in idx) >= FLOOR:
                break
            P = P.copy()
            for b in ("neck", "head"):
                P.local[BONE_INDEX[b]] = euler(-1.5, 0, 0) @ P.local[BONE_INDEX[b]]
    if ch.get("armsfree", 0.0) < 1:
        ai = _part_index("arms")
        for s, vs in _arm_sides().items():
            for _ in range(40):
                pts = sk(*P.world())
                if pts[ai][vs, 1].min() >= FLOOR:
                    break
                P = P.copy()
                b = BONE_INDEX[ARMS[s][0]]
                P.local[b] = P.local[b] @ euler(-2.0, 0, 0)
    return P


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
        if g:                     # the body's lowest point (the head's and arms' too when they may go low)
            skip = {"tongue"} | (set() if ch.get("headfree", 0.0) >= 1 else set(HEAD_PARTS)) \
                | (set() if ch.get("armsfree", 0.0) >= 1 else {"arms"})
            low = min(p[:, 1].min() for p, part in zip(skinner()(*P.world()), skinner().parts)
                      if part.name not in skip)
            P.root = P.root + np.array([0, -g * low, 0])
        out.append(_lift_clear(P, ch))
    if track.wrap:
        out.append(out[0].copy())
    return out


def _loop(keys, d, extra=None):
    return frames_of(Track(keys, d, True), extra), True


def _once(keys, d, extra=None):
    return frames_of(Track(keys, d, False), extra), False


# ------------------------------------------------------------------------------------------- clips

def _breath(t, ch, period=2.0, amount=1.0):
    s = np.sin(2 * np.pi * t / period)
    add(ch, **{"spine__x": -1.2 * amount * s, "chest__x": 1.0 * amount * s, "root__y": 0.003 * amount * s,
               "head__x": -0.8 * amount * s, "arms__x": 1.5 * amount * s, "fores__x": -1.5 * amount * s})


def stand():
    d = 4.0
    keys = [(0, {}), (1.0, {"front.y": 8, "head.x": -3, "tail.y": -6}),
            (1.7, {"jaw": 8}), (2.1, {"front.y": 0, "head.x": 2, "jaw": 0, "tail.y": 0}),
            (3.0, {"front.y": -8, "head.x": -3, "tail.y": 6}), (4.0, {})]

    def extra(t, ch):
        _breath(t, ch, 2.0)
        add(ch, tail4__y=6 * np.sin(2 * np.pi * t / 2.0), tail3__y=3 * np.sin(2 * np.pi * t / 2.0 - 1),
            handAL__x=4 * np.sin(2 * np.pi * t / 4.0), handAR__x=4 * np.sin(2 * np.pi * t / 4.0 + 1))
    return _loop(keys, d, extra)


def _roar(h=1.0):
    """Rearing up on its hind legs, arms spread and claws up, head back, maw wide."""
    return {"rootr.x": -26 * h, "root.y": .03 * h, "root.z": -.02 * h, "front.x": -16 * h, "jaw": 34 * h,
            "arms.x": -55 * h, "arms.z": 30 * h, "fores.x": -30 * h, "hands.x": -20 * h}


def fidgeting():
    """Looks round, scratches its belly, then rears up and roars (DK1 672: Smoke / Relax / Scream)."""
    d = 5.0
    rest = {k: 0 for k in _roar()}
    keys = [(0, {}),
            (0.6, {"front.y": 32, "head.x": -8, "tail.y": -14}),
            (1.3, {"front.y": 34, "head.x": 4}),
            (1.8, {"front.y": -30, "head.x": -8, "tail.y": 14}),
            (2.5, {"front.y": -32, "head.x": 4, **arm("L", (-30, 0, -10), (-60, 0, 0))}),
            (2.65, arm("L", (-20, 0, -10), (-40, 0, 0))), (2.8, arm("L", (-30, 0, -10), (-60, 0, 0))),
            (2.95, arm("L", (-20, 0, -10), (-40, 0, 0))),
            (3.1, {"front.y": 0, "head.x": 0, "tail.y": 0, **arm("L")}),
            (3.5, _roar()), (3.7, {"jaw": 40, "head.z": 8}), (3.9, {"head.z": -8}), (4.1, {"head.z": 0, "jaw": 32}),
            (4.6, rest), (5.0, {})]
    return _loop(keys, d, lambda t, ch: _breath(t, ch, 2.5))


def _gait(t, ch, period, stride, lift, bob, sway, duty, lean=0.0, arm_swing=14.0):
    """Two feet stepping on the spot (each slides back on the floor for `duty` of its cycle, then swings
    forward lifted); the body rocks over the planted foot, the tail and the arms swing against the legs."""
    ph = (t / period) % 1.0
    for side, off in (("L", 0.0), ("R", 0.5)):
        p = (ph + off) % 1.0
        if p < duty:
            u = p / duty
            z, y, pitch = stride * (0.5 - u), 0.0, -12 * max(0, (u - .7) / .3)
        else:
            u = (p - duty) / (1 - duty)
            z = stride * (-0.5 + (1 - np.cos(np.pi * u)) / 2)
            y = lift * np.sin(np.pi * u)
            pitch = 30 * np.sin(np.pi * u) - 10 * u
        add(ch, **{f"{side}__z": z, f"{side}__y": y, f"{side}__p": pitch, f"{side}__t": -0.5 * pitch})
    w = 2 * np.pi * ph
    add(ch, **{"root__y": -bob * (0.5 + 0.5 * np.cos(2 * w)), "root__x": .018 * np.sin(w + np.pi),
               "rootr__z": -sway * np.sin(w), "rootr__y": .6 * sway * np.sin(w), "rootr__x": lean,
               "body__y": -.8 * sway * np.sin(w), "front__y": -.6 * sway * np.sin(w - .4),
               "front__z": .8 * sway * np.sin(w), "head__x": 2.5 * np.cos(2 * w),
               "tail1__y": 1.4 * sway * np.sin(w + .5), "tail2__y": 1.8 * sway * np.sin(w + 1.0),
               "tail3__y": 2.2 * sway * np.sin(w + 1.6), "tail4__y": 2.6 * sway * np.sin(w + 2.2),
               "upperAL__x": arm_swing * np.sin(w), "upperAR__x": -arm_swing * np.sin(w),
               "foreAL__x": -.6 * arm_swing * np.sin(w - .5), "foreAR__x": .6 * arm_swing * np.sin(w - .5)})


def walk():
    """The waddle (DK1 664): short steps, the heavy body rocking from foot to foot."""
    period = 1.0
    return _loop([(0, {"head.x": 3})], period,
                 lambda t, ch: _gait(t, ch, period, stride=.17, lift=.06, bob=.012, sway=7, duty=.6))


def running():
    period = 0.56

    def extra(t, ch):
        _gait(t, ch, period, stride=.3, lift=.09, bob=.025, sway=6, duty=.42, lean=8, arm_swing=26)
        w = 4 * np.pi * t / period
        add(ch, **{"jaw": 10 + 6 * np.sin(w), "body__x": 3 * np.sin(w), "tail__x": 3 * np.sin(w + 1),
                   "root__y": .02 * max(0, np.sin(w)), "fores__x": -25})
    return _loop([(0, {"front.x": 4})], period, extra)


def patrolling():
    period = 1.3

    def extra(t, ch):
        _gait(t, ch, period, stride=.13, lift=.05, bob=.008, sway=5, duty=.62)
        add(ch, **{"neck__y": 18 * np.sin(2 * np.pi * t / period), "head__y": 12 * np.sin(2 * np.pi * t / period - .5),
                   "head__x": -4})
    return _loop([(0, {})], period, extra)


def attack():
    """DK1 668: crouches, springs forward off its hind legs with both claws raking forward, bites, lands."""
    d = 1.1
    rest = {"rootr.x": 0, "root.y": 0, "root.z": 0, "front.x": 0, "head.x": 0, "jaw": 0, "tail.x": 0,
            "body.x": 0, "arms.x": 0, "arms.z": 0, "fores.x": 0, "hands.x": 0, **feet(L=(0, 0, 0), R=(0, 0, 0))}
    keys = [(0, rest),
            (0.25, {"rootr.x": 10, "root.y": -.05, "root.z": -.04, "front.x": 8, "jaw": -6, "tail.x": 4,
                    "body.x": 6, "arms.x": 25, "arms.z": 10, "fores.x": -10}),
            (0.45, {"rootr.x": -16, "root.y": .07, "root.z": .06, "front.x": -14, "head.x": -6, "jaw": 36,
                    "tail.x": 16, "body.x": -6, "arms.x": -95, "arms.z": 25, "fores.x": -40, "hands.x": -30,
                    **feet(L=(0, .05, -.05), R=(0, .03, -.08))}),
            (0.58, {"rootr.x": 10, "root.y": .01, "root.z": .12, "front.x": 6, "head.x": 4, "jaw": 36,
                    "tail.x": 20, "body.x": 6, "arms.x": -40, "arms.z": 15, "fores.x": -20, "hands.x": 10,
                    **feet(L=(0, 0, .1), R=(0, 0, .02))}),
            (0.66, {"jaw": CLOSE, "head.x": 16, "arms.x": -10, "fores.x": 0}),
            (0.82, {"rootr.x": 6, "root.y": -.02, "root.z": .08, "front.x": 6, "head.x": 6, "jaw": 6,
                    "arms.x": 0, "hands.x": 0}),
            (1.1, rest)]
    return _once(keys, d)


def breathe_fire():
    """Spits fire: rears its head back, swells, then thrusts the open maw forward and heaves."""
    d = 1.6
    rest = {"rootr.x": 0, "root.y": 0, "root.z": 0, "front.x": 0, "head.x": 0, "jaw": 0, "body.x": 0,
            "tail.x": 0, "front.y": 0, "arms.x": 0, "arms.z": 0, "fores.x": 0}
    keys = [(0, rest),
            (0.4, {"rootr.x": -16, "root.y": .025, "root.z": -.03, "front.x": -24, "head.x": -12, "jaw": -6,
                   "body.x": -6, "tail.x": 8, "arms.x": 20, "arms.z": 20, "fores.x": -20}),
            (0.55, {"rootr.x": 8, "root.y": -.01, "root.z": .04, "front.x": 10, "head.x": -12, "jaw": 34,
                    "body.x": 6, "tail.x": -4, "arms.x": 10, "arms.z": 30, "fores.x": -10}),
            (0.85, {"front.y": 8}), (1.1, {"front.y": -8}), (1.25, {"front.y": 0, "jaw": 32}),
            (1.35, {"jaw": 4, "front.x": 4}),
            (1.6, rest)]

    def extra(t, ch):
        if 0.55 < t < 1.3:
            add(ch, **{"chest__x": 2.5 * np.sin(t * 42), "jaw": 4 * np.sin(t * 35), "root__z": .006 * np.sin(t * 42)})
    return _once(keys, d, extra)


def _hop(h, arms_up=1.0):
    return {"root.y": h, **feet(B=(0, h, 0)), "L.p": 0, "R.p": 0, "arms.x": -70 * arms_up, "arms.z": 30 * arms_up,
            "fores.x": -40 * arms_up}


def happy():
    """Two little hops on both feet, claws in the air, tail wagging (DK1 678)."""
    d = 1.6
    down = {"root.y": 0, **feet(B=(0, 0, 0)), "arms.x": 0, "arms.z": 0, "fores.x": 0, "jaw": 0, "front.x": 0}
    keys = [(0, down),
            (0.18, {**down, "root.y": -.04, "arms.x": 10}),
            (0.38, {**_hop(.07), "jaw": 26, "front.x": -12}),
            (0.58, {**down, "root.y": -.035, "arms.x": -20, "arms.z": 15}),
            (0.8, {**_hop(.07), "jaw": 30, "front.x": -14}),
            (1.0, {**down, "root.y": -.035, "arms.x": -10}),
            (1.3, {**down, "root.y": 0}),
            (1.6, down)]

    def extra(t, ch):
        w = 2 * np.pi * t / 0.4
        add(ch, **{"tail2__y": 10 * np.sin(w), "tail3__y": 16 * np.sin(w - .7), "tail4__y": 20 * np.sin(w - 1.4)})
    return _loop(keys, d, extra)


def angry():
    """Stamps its feet in turn, claws raised and spread, snapping and roaring, the tail lashing."""
    d = 1.8
    base = {"front.x": 4, "head.x": -6, "jaw": 20, "tail.x": 6, "rootr.x": -6, "root.y": .0,
            "arms.x": -45, "arms.z": 35, "fores.x": -35, "hands.x": -20, **feet(L=(0, 0, 0), R=(0, 0, 0))}
    keys = [(0, base),
            (0.25, {"L.y": .08, "L.z": .03, "rootr.z": 5, "jaw": 34, "arms.z": 45}),
            (0.4, {"L.y": 0, "L.z": .02, "root.y": -.03, "rootr.z": 0, "jaw": 10, "arms.z": 30}),
            (0.7, {"R.y": .08, "R.z": .03, "rootr.z": -5, "jaw": 36, "root.y": 0, "arms.z": 45}),
            (0.85, {"R.y": 0, "R.z": .02, "root.y": -.03, "rootr.z": 0, "jaw": 8, "arms.z": 30}),
            (1.15, {"front.x": -16, "head.x": -16, "jaw": 42, "rootr.x": -16, "root.y": .02, "arms.x": -80,
                    "arms.z": 40}),
            (1.45, {"front.x": -12, "jaw": 40}),
            (1.8, base)]

    def extra(t, ch):
        w = 2 * np.pi * t / 0.6
        add(ch, **{"tail2__y": 14 * np.sin(w), "tail3__y": 20 * np.sin(w - 1), "tail4__y": 24 * np.sin(w - 2),
                   "head__z": 5 * np.sin(2 * np.pi * t / 0.45),
                   "handAL__z": 10 * np.sin(2 * np.pi * t / 0.3), "handAR__z": -10 * np.sin(2 * np.pi * t / 0.3)})
    return _loop(keys, d, extra)


def celebrating():
    """Rears up tall on its hind legs, pumping its claws in the air and roaring (DK1 678 Celebrate / Roar)."""
    period = 1.4

    def extra(t, ch):
        ph = 2 * np.pi * t / period
        h = 0.5 - 0.5 * np.cos(ph)
        add(ch, **{"rootr__x": -30 * (0.3 + 0.7 * h), "root__y": .035 * h, "root__z": -.03 * h,
                   "front__x": -8 * h, "jaw": 12 + 30 * h, "tail__y": 14 * np.sin(2 * ph), "head__z": 6 * np.sin(ph),
                   "upperAL__x": -120 * h - 30 + 25 * np.sin(2 * ph), "upperAR__x": -120 * h - 30 - 25 * np.sin(2 * ph),
                   "arms__z": 25 + 15 * h, "fores__x": -30, "hands__x": -20,
                   "L__y": .02 * max(0, np.sin(2 * ph)), "R__y": .02 * max(0, -np.sin(2 * ph))})
    return _loop([(0, {})], period, extra)


def dance():
    """A pretty dance (DK1 PrettyDance 678): stepping from foot to foot, swaying, the arms swinging side to
    side, the head bobbing and the tail swinging."""
    period = 2.0

    def extra(t, ch):
        ph = 2 * np.pi * t / period
        s = np.sin(ph)
        add(ch, **{"root__x": .03 * s, "rootr__z": -8 * s, "rootr__y": 8 * s, "body__y": -10 * s,
                   "front__y": 14 * s, "front__z": 8 * s, "front__x": -6 + 6 * np.sin(2 * ph),
                   "tail__y": -26 * np.sin(ph + .6),
                   "L__y": .06 * max(0, np.sin(2 * ph)), "R__y": .06 * max(0, -np.sin(2 * ph)),
                   "L__x": .03 * s, "R__x": -.03 * s,
                   "arms__x": -50, "fores__x": -40, "arms__y": 25 * s, "arms__z": 20 + 15 * np.sin(2 * ph),
                   "root__y": .012 * np.abs(np.sin(2 * ph)) - .008, "jaw": 12 + 10 * np.sin(2 * ph)})
    return _loop([(0, {})], period, extra)


def getting_hit():
    """Knocked back by the blow, squashed down, arms flung out (DK1 670), then back up."""
    d = 0.5
    keys = [(0, {}),
            (0.09, {"root.y": -.045, "root.z": -.035, "rootr.x": -10, "front.x": -16, "jaw": 24, "tail.x": 10,
                    "arms.x": -30, "arms.z": 35, "fores.x": -20, **feet(B=(.02, 0, 0))}),
            (0.5, {"root.y": 0, "root.z": 0, "rootr.x": 0, "front.x": 0, "jaw": 0, "tail.x": 0, "arms.x": 0,
                   "arms.z": 0, "fores.x": 0, **feet(B=(0, 0, 0))})]
    return _once(keys, d)


def being_slapped():
    """Cowers: ducks low, the head turned away and tucked in, the arms up over its head (DK1 682)."""
    d = 0.6
    cower = {"root.y": -.06, "rootr.z": 10, "rootr.y": -10, "front.x": 22, "front.y": 26, "jaw": CLOSE,
             "tail.y": 20, "tail.x": -6, "arms.x": -85, "arms.y": -20, "arms.z": 5, "fores.x": -70}
    keys = [(0, {}), (0.1, cower), (0.35, {"front.y": 10, "rootr.z": 4, "root.y": -.03}),
            (0.6, {k: 0 for k in cower})]
    return _once(keys, d)


def staggering():
    """Drunk: swaying, lurching steps, the arms dangling, the head lolling, the tongue out."""
    d = 3.0

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"rootr__z": 9 * np.sin(w), "rootr__x": 4 * np.sin(2 * w), "root__x": .025 * np.sin(w),
                   "front__z": -12 * np.sin(w - .7), "front__y": 14 * np.sin(2 * w + .3),
                   "front__x": 10 + 6 * np.sin(2 * w), "jaw": 10 + 6 * np.sin(3 * w),
                   "tail__y": 16 * np.sin(w + 1),
                   "L__y": .05 * max(0, np.sin(w + 1.6)), "R__y": .05 * max(0, -np.sin(w + 1.6)),
                   "L__x": .03 * np.sin(w), "R__x": .03 * np.sin(w), "L__z": .03 * np.sin(2 * w),
                   "upperAL__x": 18 * np.sin(w + 1), "upperAR__x": -18 * np.sin(w + 1),
                   "arms__z": 8 + 8 * np.sin(2 * w), "root__y": -.015})
    return _loop([(0, {})], d, extra)


LOW = {"rootr.x": 8, "root.y": -.03, "front.x": 14, "head.x": 8, "arms.x": -15, "fores.x": -10}


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
    look = {"front.x": 10, "head.x": 4, "rootr.x": 4, "root.y": -.01, "jaw": -4, "arms.x": -10, "fores.x": -10}
    keys = [(0, look), (0.3, {**look, "front.x": 14, "root.z": -.015, "arms.x": 0}),
            (0.42, {"jaw": 30, "front.x": 6, "root.z": .01}),
            (0.6, {"jaw": 34}),
            (0.95, {"jaw": 30, "front.x": 4, "root.z": 0, "arms.x": -40, "fores.x": -40}),     # claws ready
            (1.05, {"jaw": CLOSE, "head.x": -6, "arms.x": -30}),                               # chomp
            (1.15, {"jaw": 10}), (1.25, {"jaw": CLOSE}),
            (1.45, {"front.x": -16, "head.x": -14, "jaw": -10, "arms.x": -20}),                # head up: gulp
            (1.7, {"front.x": -4, "head.x": 0, "jaw": 4}),
            (2.4, look)]

    def extra(t, ch):
        if 1.45 < t < 1.9:
            u = (t - 1.45) / .45
            add(ch, **{"neck__x": 6 * np.sin(np.pi * u), "chest__x": 3 * np.sin(np.pi * max(0, u - .3) / .7)})
    return _loop(keys, d, extra)


def manufacturing():
    """Hammers at the work with its claws, one arm then the other."""
    d = 1.2
    up = {"upper": (-90, 0, 15), "fore": (-50, 0, 0)}
    down = {"upper": (-25, 0, 10), "fore": (-5, 0, 0)}
    keys = [(0, {"front.x": 12, "rootr.x": 8, "root.y": -.015, **arm("L", **down), **arm("R", **up), "jaw": 4}),
            (0.3, {**arm("L", **up), **arm("R", **down), "rootr.z": 3}),
            (0.6, {**arm("L", **down), **arm("R", **up), "rootr.z": 0, "head.x": 6}),
            (0.9, {**arm("L", **up), **arm("R", **down), "rootr.z": -3, "head.x": 0}),
            (1.2, {})]
    return _loop(keys, d)


def praying():
    """Raises its claws to the heavens with its head thrown back, then bows low, claws to the floor."""
    d = 3.0
    up = {"rootr.x": -14, "root.y": .02, "front.x": -18, "head.x": -16, "jaw": 12, "tail.x": 0,
          "arms.x": -120, "arms.z": 20, "fores.x": -20, "hands.x": -20}
    bow = {"rootr.x": 10, "root.y": -.035, "front.x": 20, "head.x": 10, "jaw": CLOSE * .9, "tail.x": 8,
           "arms.x": -35, "arms.z": 10, "fores.x": -15, "hands.x": 10}
    keys = [(0, up), (0.4, up), (1.3, bow), (2.1, bow), (3.0, up)]
    return _loop(keys, d, lambda t, ch: _breath(t, ch, 1.5, .5))


def interacting():
    """Sniffs at the thing on the floor, nudges it, paws at it with a claw."""
    d = 3.0
    keys = [(0, {**LOW, "front.y": 16}),
            (0.6, {"front.y": -16, "jaw": 8}), (0.75, {"jaw": 0}),
            (1.2, {"front.y": 4}),
            (1.45, arm("L", (-60, 0, 5), (-40, 0, 0), (-20, 0, 0))),
            (1.65, arm("L", (-20, 0, 5), (0, 0, 0), (20, 0, 0))),
            (1.85, arm("L", (-60, 0, 5), (-40, 0, 0), (-20, 0, 0))),
            (2.1, {**arm("L", (-15, 0, 0), (-10, 0, 0))}),
            (2.5, {"front.y": 18, "jaw": 8}), (2.65, {"jaw": 0}),
            (3.0, {**LOW, "front.y": 16})]
    return _loop(keys, d, lambda t, ch: add(ch, head__z=3 * np.sin(2 * np.pi * t / 0.5)))


# ---- lying, falling and held ----------------------------------------------------------------------

CURL = {"tail1.y": 16, "tail2.y": 26, "tail3.y": 30, "tail4.y": 30}


def sleeping():
    """Flat on its belly in the bed, the hind legs folded under, the arms tucked forward, the tail curled
    round and the head down (DK1 680)."""
    d = 4.0
    lie = {"root.y": -.13, "ground": 1, **feet(B=(.05, 0, -.04)), "L.p": 30, "R.p": 30, **CURL, "front.y": 22,
           "front.x": 30, "head.z": 12, "jaw": CLOSE, "rootr.y": -8, "arms.x": -65, "arms.z": 35,
           "fores.x": 20, "hands.x": 20}

    def extra(t, ch):
        s = np.sin(2 * np.pi * t / d)
        add(ch, **{"chest__x": 1.8 * s, "spine__x": -1.2 * s, "jaw": 2 + 2 * s, "head__x": -1.2 * s,
                   "tail4__y": 6 * np.sin(2 * np.pi * t / 2.0) * max(0, np.sin(2 * np.pi * t / d - 1))})
    return _loop([(0, lie)], d, extra)


# lying on its right side: the upper (left) limbs flop down across the body, the lower ones splay
LIMP = {"ik": 0, **fk_leg("L", (-20, 0, -25), (10, 0, 0)), **fk_leg("R", (-30, 0, 15), (20, 0, 0)),
        **arm("L", (-20, 0, -35), (-20, 0, 0)), **arm("R", (-40, 0, 20), (-10, 0, 0))}
FALLEN = {"rootr.z": 82, "ground": 1, "tailfree": 1, "headfree": 1, "armsfree": 1, **LIMP, "front.z": -18,
          "front.x": 8, "jaw": 18, "tail.y": 18, "tail.z": -10}


def falling():
    d = 1.0
    keys = [(0, {"ground": 1, "ik": 1, "tailfree": 0, "headfree": 0, "armsfree": 0}),
            (0.15, {"front.x": -18, "jaw": 30, "rootr.x": -10, "root.y": .02, "arms.x": -40, "arms.z": 40}),
            (0.45, {**FALLEN, "rootr.z": 55, "rootr.x": -6}),
            (0.62, {**FALLEN, "rootr.x": 0}),
            (0.75, {"front.z": -6, "tail.y": 26}),
            (1.0, FALLEN)]
    return _once(keys, d)


def waking_up():
    d = 1.6
    zero = {k: 0 for k in LIMP if k != "ik"}
    keys = [(0, FALLEN),
            (0.5, {**FALLEN, "rootr.z": 40, "front.z": 0, **arm("R", (-60, 0, 40), (-20, 0, 0))}),
            (0.8, {**zero, "ik": 1, "rootr.z": 0, "tailfree": 0, "headfree": 0, "armsfree": 0, "root.y": -.05,
                   "front.x": 10, "jaw": 4, "tail.y": 10, "tail.z": 0}),
            (1.05, {"root.y": 0, "front.x": 0, "front.z": 14}), (1.2, {"front.z": -14}),
            (1.35, {"front.z": 8}), (1.6, {"front.z": 0, "tail.y": 0, "jaw": 0})]
    return _once(keys, d)


# on its back, legs and arms sticking up into the air, stiff and splayed (DK1 674)
ON_BACK = {"rootr.z": 178, "ground": 1, "ik": 0, "tailfree": 1, "headfree": 1, "armsfree": 1,
           **fk_leg("L", (-30, 0, -10), (40, 0, 0), (-20, 0, 0)), **fk_leg("R", (-45, 0, 0), (30, 0, 0), (-10, 0, 0)),
           **arm("L", (-20, 0, -10), (-30, 0, 0)), **arm("R", (-10, 0, 0), (-20, 0, 0)),
           "front.x": -24, "jaw": 30, "tail.x": 20, "tail.y": 10}


def die():
    """Rears up with a last roar, then topples over onto its back, the legs in the air (DK1 674)."""
    d = 2.4
    keys = [(0, {"ik": 1, "ground": 1, "tailfree": 0, "headfree": 0, "armsfree": 0}),
            (0.4, {**_roar(.9), "jaw": 40, "front.x": -24}),
            (0.85, {"jaw": 44, "head.z": 10}),
            (1.15, {"rootr.z": 60, "rootr.x": -10, "root.y": .02, "ik": 0, "tailfree": 1, "headfree": 1,
                    "armsfree": 1, **{k: v * .5 for k, v in ON_BACK.items()
                                      if k not in ("rootr.z", "ground", "ik", "tailfree", "headfree", "armsfree")},
                    "head.z": 0, "root.z": 0}),
            (1.5, {**ON_BACK, "rootr.x": 0, "root.y": 0, "arms.x": 0, "arms.z": 0, "fores.x": 0, "hands.x": 0}),
            (1.7, {**fk_leg("L", (-50, 0, -20), (20, 0, 0), (-30, 0, 0)), **arm("L", (-40, 0, -20), (-10, 0, 0))}),
            (1.9, {**fk_leg("L", (-25, 0, -5), (45, 0, 0), (-15, 0, 0)), **arm("L", (-15, 0, -5), (-35, 0, 0))}),
            (2.4, {**ON_BACK, "jaw": 34, "front.x": -28})]
    return _once(keys, d)


def being_picked_up():
    """Held by the scruff: the body hangs nose-down, legs and arms dangling and kicking, the tail swinging
    (DK1 676)."""
    d = 2.0
    hang = {"ik": 0, "tailfree": 1, "headfree": 1, "armsfree": 1, "rootr.x": 28, "front.x": 18, "jaw": 16,
            "tail.x": -24, **fk_leg("L", (-35, 0, -8), (20, 0, 0), (10, 0, 0)),
            **fk_leg("R", (-35, 0, -8), (20, 0, 0), (10, 0, 0)), "arms.x": 10, "arms.z": -5, "fores.x": 10}

    def extra(t, ch):
        w = 2 * np.pi * t / d
        k = 3 * w
        add(ch, **{"rootr__z": 8 * np.sin(w), "rootr__x": 5 * np.sin(w + 1),
                   "thighL__x": 22 * np.sin(k), "thighR__x": -22 * np.sin(k),
                   "shinL__x": 15 * np.sin(k + .6), "shinR__x": -15 * np.sin(k + .6),
                   "upperAL__x": -20 * np.sin(k + 1), "upperAR__x": 20 * np.sin(k + 1),
                   "foreAL__x": -15 * np.sin(k + 1.5), "foreAR__x": 15 * np.sin(k + 1.5),
                   "tail__y": 20 * np.sin(w + .5), "tail__x": 6 * np.sin(2 * w), "front__y": 12 * np.sin(w - .7),
                   "head__z": 8 * np.sin(w - 1.2), "jaw": 8 * np.sin(k)})
    return _loop([(0, hang)], d, extra)


def torture():
    """Strapped spread-eagled to the wheel, writhing."""
    d = 2.4
    spread = {"ik": 0, "tailfree": 1, "headfree": 1, "armsfree": 1,
              **fk_leg("L", (10, 0, 40), (20, 0, 0)), **fk_leg("R", (10, 0, 40), (20, 0, 0)),
              "arms.x": -20, "arms.z": 70, "fores.x": -10, "front.x": -14, "jaw": 30, "tail.x": -10}

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"body__y": 10 * np.sin(w), "body__z": 6 * np.sin(2 * w), "front__y": -22 * np.sin(w + .5),
                   "front__x": 8 * np.sin(2 * w + 1), "head__z": 16 * np.sin(w + 1.3), "jaw": 14 * np.sin(3 * w),
                   "tail__y": 24 * np.sin(w + 2),
                   "upperAL__x": 14 * np.sin(2 * w), "upperAR__x": -14 * np.sin(2 * w),
                   "foreAL__x": -20 * np.sin(3 * w), "foreAR__x": 20 * np.sin(3 * w),
                   "thighL__x": -12 * np.sin(2 * w + 1), "thighR__x": 12 * np.sin(2 * w + 1)})
    return _loop([(0, spread)], d, extra)


def entrance():
    """Comes into the dungeon: drops in, lands on both feet squashed, then rears up with a roar, claws
    spread, and settles."""
    d = 1.8
    air = {"root.y": .3, "ik": 0, "rootr.x": -6, "front.x": -10, "jaw": 20, "tail.x": 8, "tailfree": 1,
           **fk_leg("L", (-15, 0, 5)), **fk_leg("R", (-15, 0, 5)), "arms.x": -50, "arms.z": 40, "fores.x": -20}
    zero_fk = {k: 0 for k in air if k.split(".")[0] in LEGS["L"] + LEGS["R"]}
    keys = [(0, air),
            (0.2, {**zero_fk, "ik": 1, "root.y": .035, "tailfree": .5}),        # feet down first
            (0.3, {"ik": 1, "tailfree": 0, "root.y": -.05, "rootr.x": 6, "front.x": 12, "jaw": 0,
                   "tail.x": 0, "arms.x": 10, "arms.z": 10, "fores.x": 0}),
            (0.5, {"root.y": 0, "rootr.x": 0, "front.x": 0}),
            (0.85, {**_roar(.9), "jaw": 40, "front.x": -24}),
            (1.2, {"jaw": 44, "head.z": 8}),
            (1.5, {**{k: 0 for k in _roar()}, "tail.x": 0, "jaw": 4, "head.z": 0}),
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
