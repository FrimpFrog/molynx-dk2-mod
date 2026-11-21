"""The Dragon's clips, keyframed from scratch after the DK1 dragon's own animations (CREATURE.JTY
6738-6943: the lumbering upright walk 6738, the coiled bite 6808, the head-up roar 6868, the rear-and-
collapse death 6876, the curled dangle in the hand 6882, the chicken caught in the air 6928, lying asleep
6902), extended to every slot a DK2 creature plays.

A clip is a set of keys over named channels, interpolated with monotone cubics (no overshoot; loops wrap),
plus procedural layers (breathing, the walk's stride). Channels:
  <bone>.x / .y / .z   local rotation in degrees (x pitch, y yaw, z roll) of any trunk, head or jaw bone
  neck.*, spine.*, tail.*  spread over the neck bones (neck1..3), spine1 + chest, tail1..4
  head.*, jaw          the head; the jaw's opening in degrees
  root.x/y/z           pelvis offset (model units); rootr.x/y/z the pelvis rotation (the whole body)
  wingL / wingR / wing 0 folded, 1 half open, 2 spread, 3 raised; flapL / flapR / flap: 0..1 toward the
                       down stroke; foldL... extra
  footL.x/y/z, footR.* ankle offsets from where it stands (IK keeps the feet on the ground otherwise);
                       footL.p foot pitch; legs "ik" 0 for free (FK) legs: thighL.x etc.
Every clip is built at 30 fps; a looping clip's last frame is a copy of its first.

LICENCE: GPL-3.0.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from scripts.dragon_model import BONE_INDEX, HEAD
from scripts.dragon_pose import Pose, ankle_rest, blend, euler, leg_ik, pose

FPS = 30

# ------------------------------------------------------------------------------------------- presets

FOLD = {
    "wing1L": ((.42, .55, -.72), (1, 0, .35)),
    "wing2L": ((.08, -.55, -.83), (1, 0, .12)),
    "wing3L": ((.06, -.7, -.7), (1, 0, .1)),
    "f1aL": ((.06, -.6, -.8), (1, 0, .1)), "f1bL": ((.04, -.8, -.6), (1, 0, .1)),
    "f2aL": ((.06, -.8, -.6), (1, 0, .1)), "f2bL": ((.04, -.95, -.3), (1, 0, .1)),
    "f3aL": ((.06, -.95, -.3), (1, 0, .1)), "f3bL": ((.02, -.95, .1), (1, 0, .1)),
}
HALF = {
    "wing1L": ((.7, .55, -.45), (.45, 0, 1)),
    "wing2L": ((.45, -.25, -.85), (.6, 0, .6)),
    "wing3L": ((.45, -.4, -.8), (.6, 0, .6)),
    "f1aL": ((.55, -.2, -.8), (.6, 0, .6)), "f1bL": ((.55, -.45, -.7), (.6, 0, .6)),
    "f2aL": ((.45, -.55, -.7), (.6, 0, .6)), "f2bL": ((.4, -.75, -.5), (.6, 0, .6)),
    "f3aL": ((.3, -.85, -.4), (.6, 0, .6)), "f3bL": ((.2, -.95, -.1), (.6, 0, .6)),
}
SPREAD = {
    "wing1L": ((.83, .5, -.26), (.1, 0, 1)),
    "wing2L": ((.95, .3, -.12), (.1, 0, 1)),
    "wing3L": ((.95, -.3, -.12), (.1, 0, 1)),
    "f1aL": ((.9, .38, -.1), (.1, 0, 1)), "f1bL": ((.93, .3, -.12), (.1, 0, 1)),
    "f2aL": ((.9, -.45, -.25), (.1, 0, 1)), "f2bL": ((.8, -.6, -.3), (.1, 0, 1)),
    "f3aL": ((.4, -.9, -.25), (.1, 0, 1)), "f3bL": ((.15, -.95, -.3), (.1, 0, 1)),
}
RAISED = {
    "wing1L": ((.5, .82, -.28), (0, -.3, 1)),
    "wing2L": ((.55, .82, -.12), (0, -.3, 1)),
    "wing3L": ((.6, .7, -.2), (0, -.3, 1)),
    "f1aL": ((.45, .88, -.1), (0, -.3, 1)), "f1bL": ((.4, .9, -.1), (0, -.3, 1)),
    "f2aL": ((.85, .35, -.3), (0, -.3, 1)), "f2bL": ((.9, .2, -.35), (0, -.3, 1)),
    "f3aL": ((.8, -.3, -.4), (0, -.3, 1)), "f3bL": ((.55, -.75, -.35), (0, -.3, 1)),
}
DOWN = {
    "wing1L": ((.8, -.1, -.1), (.1, .4, 1)),
    "wing2L": ((.75, -.6, .05), (.1, .5, 1)),
    "wing3L": ((.6, -.75, .1), (.1, .5, 1)),
    "f1aL": ((.75, -.6, .1), (.1, .5, 1)), "f1bL": ((.6, -.75, .15), (.1, .5, 1)),
    "f2aL": ((.45, -.88, .05), (.1, .5, 1)), "f2bL": ((.3, -.95, .05), (.1, .5, 1)),
    "f3aL": ((.15, -.98, -.1), (.1, .5, 1)), "f3bL": ((.05, -.95, -.2), (.1, .5, 1)),
}
# folded flat along the back, pointing down the body (for lying on the belly)
TUCK = {
    "wing1L": ((.5, -.75, -.3), (.2, -.3, -1)),
    "wing2L": ((.15, -.95, -.25), (.2, 0, -1)),
    "wing3L": ((.12, -.96, -.2), (.2, 0, -1)),
    "f1aL": ((.1, -.97, -.2), (.2, 0, -1)), "f1bL": ((.06, -.98, -.15), (.2, 0, -1)),
    "f2aL": ((.06, -.98, -.15), (.2, 0, -1)), "f2bL": ((.02, -.99, -.1), (.2, 0, -1)),
    "f3aL": ((.0, -.99, -.1), (.2, 0, -1)), "f3bL": ((-.06, -.99, -.05), (.2, 0, -1)),
}
# v2c: asleep (Sleep only, channel "rest"). TUCK lays the arm along the body and closes the fingers to
# 4 degrees apart, so the folded wing read as a bundle of shards. REST swings the upper arm and forearm
# out (the elbow opened to about 150 degrees, so the arm triangle keeps its height) and lifts the wing off
# the back: the shoulder-wrist-flank and inner panels stand as broad flaps (as v1's sleeping wings do),
# the three fingers fanned back along the tail 18-20 degrees apart, clear of the curled trunk and the
# bed. Membrane area 85 % of the folded idle wing's (TUCK's 55 %), broad from every side. The clips that
# blend TUCK with FOLD keep TUCK.
def _rested(tuck, arm=1.2, fore=.4, droop=0.0, fan=(.6, .2, -.2), lift=-.25, normal=(.2, 0, -1)):
    """A folded-on-the-back wing: `arm` / `fore` the upper arm's / forearm's sideways lean (pointing down
    the body), `droop` how far the outer bones bend toward the belly (over the flank), `fan` each
    finger's sideways lean (f1, f2, f3), `lift` off the back."""
    def unit(v):
        v = np.asarray(v, float)
        return tuple(v / np.linalg.norm(v))
    out = dict(tuck)
    out["wing1L"] = (unit((arm, -.7, -.3)), (.2, -.3, -1))
    out["wing2L"] = (unit((fore, -.8, lift + droop * .5)), normal)
    out["wing3L"] = (unit((fore * .8, -.85, lift + droop * .5)), normal)
    for f, x in zip(("f1", "f2", "f3"), fan):
        out[f + "aL"] = (unit((x * .8, -1, lift * .6 + droop)), normal)
        out[f + "bL"] = (unit((x, -1, lift * .5 + droop)), normal)
    return out


REST = _rested(TUCK)
WING_BONES = ["wing1", "wing2", "wing3", "f1a", "f1b", "f2a", "f2b", "f3a", "f3b"]
_PRESETS = None


def presets():
    global _PRESETS
    if _PRESETS is None:
        _PRESETS = [pose({"aim": a}) for a in (FOLD, HALF, SPREAD, RAISED, DOWN, TUCK, REST)]
    return _PRESETS


def wing_locals(level, flap, side, tuck=0.0, rest=0.0):
    """Local rotations of one wing's bones: the preset path at `level`, then `flap` toward DOWN,
    `tuck` toward TUCK and `rest` toward REST."""
    ps = presets()
    level = float(np.clip(level, 0, 3))
    i = min(int(np.floor(level)), 2)
    P = blend(ps[i], ps[i + 1], level - i)
    if flap:
        P = blend(P, ps[4], float(np.clip(flap, 0, 1)))
    if tuck:
        P = blend(P, ps[5], float(np.clip(tuck, 0, 1)))
    if rest:
        P = blend(P, ps[6], float(np.clip(rest, 0, 1)))
    return {b + side: P.local[BONE_INDEX[b + side]] for b in WING_BONES}


REST_FEET = {s: ankle_rest(s) for s in "LR"}

MACROS = {
    "neck": (("neck1", .3), ("neck2", .35), ("neck3", .35)),
    "spine": (("spine1", .5), ("chest", .5)),
    "tail": (("tail1", .25), ("tail2", .25), ("tail3", .25), ("tail4", .25)),
}
TRUNK = ["spine1", "chest", "neck1", "neck2", "neck3", "head", "tail1", "tail2", "tail3", "tail4"]


def build_pose(ch):
    """Channel values -> Pose."""
    P = Pose()
    rot = {b: np.zeros(3) for b in TRUNK + ["jaw"]}
    for k, v in ch.items():
        if "." not in k:
            continue
        name, ax = k.split(".")
        if ax not in "xyz" or len(ax) != 1:
            continue
        i = "xyz".index(ax)
        if name in MACROS:
            for b, w in MACROS[name]:
                rot[b][i] += v * w
        elif name in rot:
            rot[name][i] += v
    rot["jaw"][0] += ch.get("jaw", 0.0)
    for b, r in rot.items():
        P.local[BONE_INDEX[b]] = euler(*r)
    P.local[BONE_INDEX["root"]] = euler(ch.get("rootr.x", 0), ch.get("rootr.y", 0), ch.get("rootr.z", 0))
    P.root = np.array([ch.get("root.x", 0), ch.get("root.y", 0), ch.get("root.z", 0)])
    for s in "LR":
        lv = ch.get("wing" + s, ch.get("wing", 0.0))
        fl = ch.get("flap" + s, ch.get("flap", 0.0))
        tk = ch.get("tuck" + s, ch.get("tuck", 0.0))
        rs = ch.get("rest" + s, ch.get("rest", 0.0))
        for b, R in wing_locals(lv, fl, s, tk, rs).items():
            extra = [ch.get(f"{b}.{a}", 0.0) for a in "xyz"]
            P.local[BONE_INDEX[b]] = R @ euler(*extra) if any(extra) else R
    ik = float(np.clip(ch.get("ik", 1.0), 0, 1))
    legs = [BONE_INDEX[b + s] for s in "LR" for b in ("thigh", "shin", "foot", "toes")]
    fk = {}
    for s in "LR":
        for b in ("thigh", "shin", "foot", "toes"):
            key = "ankle" if b == "foot" else b        # footL.x etc. are the IK offsets; FK uses ankleL.x
            r = [ch.get(f"{key}{s}.{a}", 0.0) for a in "xyz"]
            fk[BONE_INDEX[b + s]] = euler(*r)
    if ik > 0:
        Rw, _ = P.world()
        yaw = np.degrees(np.arctan2(*(Rw[BONE_INDEX["root"]] @ [0, 0, 1.0])[[0, 2]]))
        for s, sign in (("L", 1), ("R", -1)):
            off = np.array([ch.get(f"foot{s}.{a}", 0.0) for a in "xyz"])
            target = REST_FEET[s] + off
            hint = euler(0, yaw, 0) @ np.array([0.35 * sign, 0, 1.0])
            fw = euler(ch.get(f"foot{s}.p", 0.0), yaw, 0)
            leg_ik(P, s, target, hint, foot_world=fw, toe_pitch=ch.get(f"toe{s}.p", 0.0))
        if ik < 1:
            F = P.copy()
            for b in legs:
                F.local[b] = fk[b]
            mixed = blend(F, P, ik)
            for b in legs:
                P.local[b] = mixed.local[b]
    else:
        for b in legs:
            P.local[b] = fk[b]
    return P


# ------------------------------------------------------------------------------------------- tracks

class Track:
    """Keys [(seconds, {channel: value})]: a channel unstated at a key holds its previous value (0 before
    its first). Monotone cubic interpolation; `wrap` makes the clip periodic over `duration`."""

    def __init__(self, keys, duration, wrap):
        keys = sorted(keys, key=lambda k: k[0])
        self.duration, self.wrap = duration, wrap
        names = sorted({c for _, d in keys for c in d})
        times = np.array([k[0] for k in keys], float)
        self.interp = {}
        for c in names:
            vals, cur = [], 0.0
            for _, d in keys:
                cur = d.get(c, cur)
                vals.append(cur)
            vals = np.array(vals, float)
            if wrap:
                # periodic: the clip's last key must equal its first (t=0 and t=duration)
                if times[-1] < duration - 1e-9:
                    t = np.r_[times, duration]
                    v = np.r_[vals, vals[0]]
                else:
                    t, v = times, vals.copy()
                    v[-1] = v[0]
                tt = np.r_[t[:-1] - duration, t, t[1:] + duration]
                vv = np.r_[v[:-1], v, v[1:]]
                self.interp[c] = PchipInterpolator(tt, vv)
            elif len(times) == 1:
                self.interp[c] = (lambda val: (lambda x: np.full_like(np.asarray(x, float), val)))(vals[0])
            else:
                self.interp[c] = PchipInterpolator(times, vals, extrapolate=True)
                self.interp[c].tmax = times[-1]
                self.interp[c].tmin = times[0]

    def __call__(self, t):
        out = {}
        for c, f in self.interp.items():
            x = t
            if not self.wrap and hasattr(f, "tmax"):
                x = min(max(t, f.tmin), f.tmax)
            out[c] = float(f(x))
        return out


_SKINNER = None


def skinner():
    global _SKINNER
    if _SKINNER is None:
        from scripts.dragon_model import Skinner, build_dragon
        _SKINNER = Skinner(build_dragon())
    return _SKINNER


def add(ch, **kw):
    for k, v in kw.items():
        k = k.replace("__", ".")
        ch[k] = ch.get(k, 0.0) + v
    return ch


def frames_of(track, extra=None, frames=None):
    """Sample a Track at 30 fps -> list of Poses (a loop gets its first frame appended)."""
    n = frames or int(round(track.duration * FPS))
    if track.wrap:
        times = np.arange(n) / FPS
    else:
        times = np.linspace(0, track.duration, n)
    out = []
    for t in times:
        ch = track(t)
        if extra:
            extra(t, ch)
        P = build_pose(ch)
        g = ch.get("ground", 0.0)
        if g:
            low = min(p[:, 1].min() for p in skinner()(*P.world()))
            P.root = P.root + np.array([0, -g * low, 0])
        out.append(P)
    if track.wrap:
        out.append(out[0].copy())
    return out


# ------------------------------------------------------------------------------------------- clips

def _breath(t, ch, period=2.6, amount=1.0):
    s = np.sin(2 * np.pi * t / period)
    add(ch, **{"chest__x": -1.2 * amount * s, "spine1__x": 0.6 * amount * s, "neck1__x": 0.8 * amount * s,
               "head__x": -0.5 * amount * s, "root__y": 0.002 * amount * s})


def stand():
    d = 4.0
    keys = [(0, {"neck.x": 0, "neck.y": 0, "head.x": 0, "head.y": 0, "tail.y": 0, "jaw": 0}),
            (1.0, {"neck.y": 6, "head.y": 4, "tail.y": -5, "head.x": -3}),
            (2.0, {"neck.y": 0, "head.y": 0, "tail.y": 0, "jaw": 6, "head.x": -6}),
            (2.4, {"jaw": 0}),
            (3.0, {"neck.y": -6, "head.y": -4, "tail.y": 5, "head.x": 0}),
            (4.0, {"neck.y": 0, "head.y": 0, "tail.y": 0})]
    tr = Track(keys, d, True)

    def extra(t, ch):
        _breath(t, ch, period=2.0)
        add(ch, tail4__y=4 * np.sin(2 * np.pi * t / 4.0 * 2), tail3__y=2 * np.sin(2 * np.pi * t / 4.0 * 2 - 1))
    return frames_of(tr, extra), True


def fidgeting():
    d = 5.0
    keys = [(0, {}),
            (0.6, {"neck.y": 30, "head.y": 25, "head.x": -6, "neck.x": 4, "tail.y": -10}),
            (1.4, {"neck.y": 30, "head.y": 30, "head.x": 4}),
            (1.9, {"neck.y": -28, "head.y": -25, "head.x": -6, "tail.y": 10}),
            (2.6, {"neck.y": -28, "head.y": -30, "head.x": 6}),
            (3.0, {"neck.y": 0, "head.y": 0, "head.x": 0, "tail.y": 0, "wing": 0}),
            (3.35, {"wing": 1.4, "jaw": 14, "head.x": -14, "neck.x": -6}),
            (3.55, {"wing": 1.1, "jaw": 4}),
            (3.75, {"wing": 1.5, "head.z": 14}),
            (3.95, {"wing": 1.1, "head.z": -14}),
            (4.15, {"head.z": 8}),
            (4.5, {"wing": 0, "jaw": 0, "head.z": 0, "head.x": 0, "neck.x": 0}),
            (5.0, {})]
    tr = Track(keys, d, True)
    return frames_of(tr, lambda t, ch: _breath(t, ch, period=2.5)), True


def _gait(t, ch, period, stride, lift, bob, sway, lean=0.0, duty=0.62):
    """Bipedal stride on the spot: each foot slides back on the ground for `duty` of the cycle, then
    swings forward lifted. The pelvis bobs, rolls and yaws; chest, neck and tail counter-swing."""
    ph = (t / period) % 1.0
    for s, off in (("L", 0.0), ("R", 0.5)):
        p = (ph + off) % 1.0
        if p < duty:                    # stance: front to back
            u = p / duty
            z = stride * (0.5 - u)
            y = 0.0
            pitch = -8 * max(0, (u - 0.75) / 0.25)    # heel lifts at the end of the stance
        else:                           # swing: back to front, lifted
            u = (p - duty) / (1 - duty)
            z = stride * (-0.5 + (1 - np.cos(np.pi * u)) / 2)
            y = lift * np.sin(np.pi * u)
            pitch = 25 * np.sin(np.pi * u) - 10 * u
        add(ch, **{f"foot{s}__z": z, f"foot{s}__y": y, f"foot{s}__p": pitch, f"toe{s}__p": -0.5 * pitch})
    w = 2 * np.pi * ph
    add(ch, **{"root__y": -bob * (0.5 + 0.5 * np.cos(2 * w)), "rootr__z": sway * np.sin(w),
               "rootr__y": 0.6 * sway * np.sin(w), "rootr__x": lean,
               "chest__y": -0.9 * sway * np.sin(w), "neck__y": -0.6 * sway * np.sin(w + 0.4),
               "head__y": 0.4 * sway * np.sin(w + 0.8), "neck__x": 2.0 * np.cos(2 * w),
               "tail1__y": 1.4 * sway * np.sin(w + 0.3), "tail2__y": 1.6 * sway * np.sin(w + 0.9),
               "tail3__y": 2.0 * sway * np.sin(w + 1.5), "tail4__y": 2.2 * sway * np.sin(w + 2.1),
               "root__x": -0.012 * np.sin(w)})


def walk():
    period = 1.2
    tr = Track([(0, {"head.x": 4, "neck.x": 2}), (period, {})], period, True)

    def extra(t, ch):
        _gait(t, ch, period, stride=0.2, lift=0.06, bob=0.018, sway=5.0, lean=3.0)
        add(ch, wing=0.04 + 0.04 * np.sin(4 * np.pi * t / period + 1))
    return frames_of(tr, extra), True


def running():
    period = 0.72
    tr = Track([(0, {"neck.x": 22, "head.x": -12, "tail.x": 6, "spine.x": 8}), (period, {})], period, True)

    def extra(t, ch):
        _gait(t, ch, period, stride=0.30, lift=0.09, bob=0.03, sway=6.0, lean=14.0, duty=0.5)
        w = 4 * np.pi * t / period
        add(ch, wing=1.0 + 0.25 * np.sin(w), flap=0.25 + 0.25 * np.sin(w + 1.5), jaw=6 + 6 * np.sin(w))
    return frames_of(tr, extra), True


def patrolling():
    period = 1.6
    tr = Track([(0, {})], period, True)

    def extra(t, ch):
        _gait(t, ch, period, stride=0.15, lift=0.05, bob=0.012, sway=4.0, lean=2.0)
        add(ch, **{"neck__y": 16 * np.sin(2 * np.pi * t / period), "head__y": 14 * np.sin(2 * np.pi * t / period - .5),
                   "head__x": -4, "wing": 0.03})
    return frames_of(tr, extra), True


def attack():
    d = 1.1
    keys = [(0, {}),
            (0.30, {"neck1.x": -18, "neck2.x": -20, "neck3.x": -10, "head.x": -15, "spine.x": -8, "jaw": 10,
                    "wing": 0.9, "tail.x": -6, "root.z": -.02}),
            (0.42, {"jaw": 34, "footL.y": .05, "footL.z": .04}),          # steps into the bite
            (0.52, {"neck1.x": 28, "neck2.x": 26, "neck3.x": 12, "head.x": 18, "spine.x": 14, "jaw": 36,
                    "wing": 1.5, "tail.x": 8, "root.z": .03, "root.y": -.02, "footL.y": 0, "footL.z": .07}),
            (0.6, {"jaw": 0, "head.x": 24}),
            (0.75, {"neck1.x": 18, "neck2.x": 14, "head.x": 10, "wing": 1.0}),
            (0.8, {"footL.y": .04, "footL.z": .035}), (0.95, {"footL.y": 0, "footL.z": 0}),
            (1.1, {"neck1.x": 0, "neck2.x": 0, "neck3.x": 0, "head.x": 0, "spine.x": 0, "jaw": 0, "wing": 0,
                   "tail.x": 0, "footL.z": 0, "root.z": 0, "root.y": 0})]
    tr = Track(keys, d, False)
    return frames_of(tr), False


def breathe_fire():
    d = 1.8
    keys = [(0, {}),
            (0.45, {"neck1.x": -14, "neck2.x": -22, "neck3.x": -18, "head.x": -26, "spine.x": -12,
                    "chest.x": -6, "wing": 1.6, "jaw": 8, "tail.x": -5, "root.z": -.03}),
            (0.54, {"footL.y": .05, "footL.z": .03}),
            (0.62, {"neck1.x": 14, "neck2.x": 8, "neck3.x": -4, "head.x": -18, "spine.x": 10, "jaw": 42,
                    "wing": 2.0, "root.z": .03, "footL.y": 0, "footL.z": .05, "tail.x": 6}),
            (0.9, {"neck.y": 10, "head.y": 6}),
            (1.2, {"neck.y": -10, "head.y": -6}),
            (1.38, {"neck.y": 0, "head.y": 0, "jaw": 40}),
            (1.5, {"jaw": 6, "neck1.x": 10, "neck2.x": 8, "head.x": 0}),
            (1.6, {"footL.y": .04, "footL.z": .025}),
            (1.8, {"neck1.x": 0, "neck2.x": 0, "neck3.x": 0, "head.x": 0, "spine.x": 0, "chest.x": 0,
                   "wing": 0, "jaw": 0, "tail.x": 0, "root.z": 0, "footL.y": 0, "footL.z": 0})]
    tr = Track(keys, d, False)

    def extra(t, ch):
        if 0.62 < t < 1.38:     # the heave of the flame
            add(ch, **{"chest__x": 2 * np.sin(t * 40), "jaw": 3 * np.sin(t * 33)})
    return frames_of(tr, extra), False


def roar_keys(t0, wing=3.0, hold=0.6):
    return [(t0 + 0.35, {"neck1.x": -16, "neck2.x": -26, "neck3.x": -22, "head.x": -40, "spine.x": -10,
                         "jaw": 42, "wing": wing, "tail.x": -8}),
            (t0 + 0.35 + hold, {"neck2.x": -30, "head.x": -44, "jaw": 46})]


def happy():
    d = 1.6
    keys = [(0, {}),
            (0.2, {"root.y": -.03, "footL.y": 0, "spine.x": 6, "wing": 0.6}),
            (0.42, {"root.y": .06, "footL.y": .06, "footR.y": .06, "wing": 2.6, "flap": 0, "neck.x": -12,
                    "head.x": -20, "jaw": 30}),
            (0.6, {"root.y": -.02, "footL.y": 0, "footR.y": 0, "wing": 2.2, "flap": .8, "spine.x": 2}),
            (0.82, {"root.y": .05, "footL.y": .05, "footR.y": .05, "wing": 2.8, "flap": 0, "jaw": 34}),
            (1.0, {"root.y": -.015, "footL.y": 0, "footR.y": 0, "flap": .6}),
            (1.3, {"wing": 0.4, "flap": 0, "neck.x": 0, "head.x": 0, "jaw": 0, "spine.x": 0, "root.y": 0}),
            (1.6, {"wing": 0})]
    tr = Track(keys, d, True)
    return frames_of(tr), True


def angry():
    d = 1.8
    keys = [(0, {}),
            (0.3, {"wing": 2.2, "neck1.x": 10, "neck2.x": 16, "neck3.x": 6, "head.x": 10, "jaw": 30,
                   "spine.x": 8, "tail.x": -10}),
            (0.55, {"footL.y": .07, "rootr.z": -4}),
            (0.7, {"footL.y": 0, "root.y": -.015, "rootr.z": 0, "jaw": 40, "head.z": 10}),
            (0.95, {"footR.y": .07, "rootr.z": 4, "head.z": -10, "root.y": 0}),
            (1.1, {"footR.y": 0, "root.y": -.015, "rootr.z": 0, "head.z": 0, "jaw": 44}),
            (1.4, {"wing": 2.0, "jaw": 20, "root.y": 0}),
            (1.8, {"wing": 0, "neck1.x": 0, "neck2.x": 0, "neck3.x": 0, "head.x": 0, "jaw": 0, "spine.x": 0,
                   "tail.x": 0})]
    tr = Track(keys, d, True)

    def extra(t, ch):
        add(ch, **{"tail3__y": 18 * np.sin(2 * np.pi * t / 0.6), "tail4__y": 22 * np.sin(2 * np.pi * t / 0.6 - 1)})
    return frames_of(tr, extra), True


def celebrating():
    period = 1.4
    tr = Track([(0, {})], period, True)

    def extra(t, ch):
        ph = 2 * np.pi * t / period
        hop = max(0.0, np.sin(2 * ph))
        side = np.sin(ph)
        add(ch, **{"root__y": 0.04 * hop - 0.01, "root__x": 0.035 * side, "rootr__z": -6 * side,
                   "footL__y": 0.05 * max(0, np.sin(ph)) , "footR__y": 0.05 * max(0, -np.sin(ph)),
                   "neck__z": 8 * side, "head__z": 8 * side, "neck__x": -8 - 6 * np.sin(2 * ph),
                   "head__x": -14, "jaw": 18 + 14 * np.sin(2 * ph), "tail__y": -14 * side,
                   "wing": 2.4 + 0.4 * np.sin(2 * ph), "flap": 0.5 + 0.5 * np.sin(2 * ph + 1.2)})
    return frames_of(tr, extra), True


def getting_hit():
    d = 0.55
    keys = [(0, {}),
            (0.1, {"spine.x": -14, "neck.x": -18, "head.x": -16, "jaw": 24, "wing": 1.2, "root.z": -.04,
                   "tail.x": 8}),
            (0.55, {"spine.x": 0, "neck.x": 0, "head.x": 0, "jaw": 0, "wing": 0, "root.z": 0, "tail.x": 0})]
    return frames_of(Track(keys, d, False)), False


def being_slapped():
    d = 0.55
    keys = [(0, {}),
            (0.1, {"spine.y": 14, "spine.x": 10, "neck.x": 22, "neck.y": -20, "head.x": 14, "jaw": 18,
                   "wing": 0.8, "root.z": .03, "tail.y": 14}),
            (0.55, {"spine.y": 0, "spine.x": 0, "neck.x": 0, "neck.y": 0, "head.x": 0, "jaw": 0, "wing": 0,
                    "root.z": 0, "tail.y": 0})]
    return frames_of(Track(keys, d, False)), False


def staggering():
    d = 3.2
    tr = Track([(0, {"neck.x": 16, "head.x": 18, "wing": 0.7, "root.y": -.02, "jaw": 10})], d, True)

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"rootr__z": 8 * np.sin(w), "rootr__x": 4 * np.sin(2 * w), "root__x": .02 * np.sin(w),
                   "neck__z": -10 * np.sin(w - .6), "head__z": -12 * np.sin(w - 1.1),
                   "neck__y": 10 * np.sin(w * 2 + .3), "tail__y": 12 * np.sin(w + 1),
                   "wingL": 0.5 * np.sin(w), "wingR": -0.5 * np.sin(w)})
        _gait(t, ch, 1.6, stride=0.12, lift=0.04, bob=0.015, sway=0, duty=0.65)
    return frames_of(tr, extra), True


def drinking():
    d = 2.0
    base = {**LEAN, "rootr.x": 44, "root.y": -.07, "head.x": 34, "neck2.x": 18}
    keys = [(0, {**base, "jaw": 4})]
    for i, t in enumerate(np.arange(0.2, 1.7, 0.25)):
        keys.append((t, {"jaw": 20 if i % 2 == 0 else 2, "head.x": 22 if i % 2 == 0 else 16}))
    keys.append((1.85, {"jaw": 4, "head.x": 18}))
    tr = Track(keys, d, True)
    return frames_of(tr, lambda t, ch: _breath(t, ch, 1.0, 0.6)), True


def eating():
    d = 2.4
    down = {**LEAN, "rootr.x": 26, "head.x": 10, "neck2.x": 6, "root.y": -.03}
    keys = [(0, {**down, "jaw": 12}),                 # the chicken in its jaws
            (0.35, {**down, "jaw": 10}),
            (0.7, {"neck1.x": -10, "neck2.x": -20, "neck3.x": -16, "head.x": -40, "spine.x": -6, "root.y": .01,
                   "tail.x": 0, "jaw": 8, "rootr.x": -4, "root.z": 0}),            # tosses it up
            (0.78, {"jaw": 40}),
            (1.05, {"neck2.x": -26, "head.x": -48, "jaw": 44}),                   # catches it
            (1.18, {"jaw": 0}),
            (1.45, {"neck.x": 2, "head.x": -20, "jaw": 6}),                        # gulp
            (1.6, {"jaw": 0, "neck.x": -4}),
            (1.75, {"jaw": 4}),
            (2.4, {**down, "jaw": 12})]
    tr = Track(keys, d, True)

    def extra(t, ch):
        if 1.35 < t < 1.9:   # swallow ripple down the neck
            u = (t - 1.35) / 0.55
            add(ch, **{"neck3__x": 6 * np.sin(np.pi * u), "neck2__x": 6 * np.sin(np.pi * max(0, u - .2) / .8),
                       "neck1__x": 5 * np.sin(np.pi * max(0, u - .4) / .6)})
    return frames_of(tr, extra), True


def manufacturing():
    d = 1.2
    keys = [(0, {"neck1.x": -10, "neck2.x": -14, "head.x": -12, "spine.x": -6, "wing": 0.3}),
            (0.45, {"neck1.x": -18, "neck2.x": -22, "head.x": -24, "spine.x": -10, "jaw": 10, "wing": 0.6,
                    "rootr.x": -6}),
            (0.62, {**LEAN, "rootr.x": 34, "neck1.x": 14, "neck2.x": 16, "head.x": 30, "jaw": 0, "wing": 0.3,
                    "root.y": -.07}),
            (0.72, {"rootr.x": 30, "head.x": 24, "root.y": -.06}),
            (1.2, {"neck1.x": -10, "neck2.x": -14, "neck3.x": 0, "head.x": -12, "spine.x": -6, "wing": 0.3,
                   "root.y": 0, "rootr.x": 0, "root.z": 0, "tail.x": 0})]
    tr = Track(keys, d, True)
    return frames_of(tr), True


def praying():
    d = 3.0
    up = {"neck1.x": -8, "neck2.x": -14, "neck3.x": -10, "head.x": -26, "wing": 2.9, "spine.x": -6,
          "root.y": 0, "jaw": 8}
    bow = {"neck1.x": 30, "neck2.x": 36, "neck3.x": 24, "head.x": 20, "wing": 2.1, "flap": 0.6,
           "spine.x": 28, "root.y": -.09, "root.z": -.03, "jaw": 0, "tail.x": 0}
    keys = [(0, up), (0.4, up), (1.3, {**bow}), (2.1, bow), (3.0, {**up, "flap": 0, "tail.x": 0, "root.z": 0})]
    return frames_of(Track(keys, d, True), lambda t, ch: _breath(t, ch, 1.5, 0.5)), True


def interacting():
    """Research (the Library): nearly upright, looking up along the shelves, two quick nose-flicks of a page."""
    d = 3.0
    base = {"rootr.x": 5, "neck.x": -10, "head.x": -15, "wing": 0.15}
    keys = [(0, {**base, "neck.y": 22, "head.y": 8}),
            (0.95, {"neck.y": -4, "head.y": 0}), (1.0, {"head.y": 15, "jaw": 8}), (1.12, {"head.y": -4, "jaw": 0}),
            (1.5, {"neck.y": -22, "head.y": -8}),
            (2.2, {"neck.y": 0, "head.y": 0}), (2.25, {"head.y": -15, "jaw": 8}), (2.37, {"head.y": 2, "jaw": 0}),
            (3.0, {**base, "neck.y": 22, "head.y": 8})]
    return frames_of(Track(keys, d, True), lambda t, ch: _breath(t, ch, 1.5, 0.6)), True


# ---- the lying, falling and held clips ------------------------------------------------------------

LEGS_TUCKED = {"ik": 0, "thighL.x": -110, "thighR.x": -110, "shinL.x": 150, "shinR.x": 150,
               "ankleL.x": -40, "ankleR.x": -40}
LIE = {"rootr.x": 80, "ground": 1, **LEGS_TUCKED, "tuck": 1, "neck1.x": -10, "neck2.x": -5, "neck3.x": 5,
       "head.x": 15, "tail.x": -70, "tail1.x": -15, "neck.y": 40, "head.y": 30, "head.z": 15, "tail.y": -60}


def sleeping():
    d = 4.0
    tr = Track([(0, {**LIE, "rest": 1})], d, True)          # v2c: the fingers fanned (REST)

    def extra(t, ch):
        s = np.sin(2 * np.pi * t / 4.0)
        add(ch, **{"chest__x": 2.2 * s, "spine1__x": -1.2 * s, "jaw": 1.5 + 1.5 * s, "head__x": -1.5 * s,
                   "tail4__y": 6 * np.sin(2 * np.pi * t / 2.0) * max(0, np.sin(2 * np.pi * t / 4.0 - 1))})
    return frames_of(tr, extra), True


SIDE = {"rootr.z": 85, "ground": 1, "ik": 0, "thighL.x": -30, "shinL.x": 40, "thighR.x": -60, "shinR.x": 70,
        "ankleL.x": -20, "ankleR.x": -20, "thighL.z": -20, "wingL": 0, "tuckL": 0.7, "wingR": 0, "tuckR": 0.3}
DEAD = {**SIDE, "neck1.z": -15, "neck2.z": -20, "neck3.z": -10, "neck.x": 15, "head.z": -25, "jaw": 24,
        "tail.z": -10, "tail.y": 15}
# v2c: both wings less tucked (upper 0.5 -> 0, lower 0.3 -> 0.1), so their fingers open 16-20 degrees
# apart (were 9-13): broad panels on the fallen and stunned dragon instead of long thin blades
FALLEN = {**SIDE, "rootr.x": -20, "wingL": 0.5, "tuckL": 0.0, "tuckR": 0.1, "neck1.z": 10, "neck2.z": 10, "neck.x": 10,
          "head.z": 5, "jaw": 10, "tail.y": 15}
LEAN = {"rootr.x": 38, "root.y": -.05, "root.z": -.05, "spine.x": 10, "neck1.x": 8, "neck2.x": 12,
        "neck3.x": 10, "head.x": 24, "tail.x": -25}
_ZERO_SIDE = {k: 0 for k in SIDE if k not in ("ik",)}


def falling():
    d = 1.0
    still = {"spine.x": 0, "head.x": 0, "root.z": 0}
    keys = [(0, {"ground": 1, "ik": 1}),
            (0.18, {"ik": 1, "spine.x": -12, "neck.x": -20, "head.x": -20, "wingL": 1.4, "wingR": 1.4, "jaw": 30,
                    "root.z": -.03, "rootr.x": -8}),
            (0.5, {**FALLEN, "rootr.z": 60, "wingL": 1.4, "wing": 1.4, "jaw": 30}),
            (0.68, {**FALLEN, **still}),
            (0.78, {"neck1.z": 22, "head.z": 15}),
            (0.9, {**FALLEN, **still}),
            (1.0, {**FALLEN, **still})]
    return frames_of(Track(keys, d, False)), False


def waking_up():
    d = 1.8
    belly = {**LIE, "neck.y": 0, "head.y": 0, "head.z": 0, "tail.y": 0, "neck1.x": -40, "neck2.x": -30,
             "neck3.x": -10, "head.x": 10, "tuck": 0.3, "wing": 0.6, "wingL": 0.6, "wingR": 0.6, "tuckL": 0.3,
             "tuckR": 0.3}       # v2c: tuck 0.6 -> 0.3, the fingers 14 degrees apart on the way up (were 7)
    keys = [(0, FALLEN), (0.12, FALLEN),
            (0.6, {**belly, "rootr.z": 0}),
            (1.0, {**_ZERO_SIDE, "ik": 1, "ground": 1, "rootr.x": 34, "root.y": -.1, "root.z": -.04,
                   "neck1.x": -10, "neck2.x": -6, "neck3.x": 0, "head.x": 10, "spine.x": 8, "tuck": 0,
                   "wing": 1.2, "wingL": 1.2, "wingR": 1.2, "tail.x": -20, "tail1.x": 0, "jaw": 4}),
            (1.4, {"ground": 1, "rootr.x": 0, "root.y": .01, "root.z": 0, "neck1.x": -4, "neck2.x": -4,
                   "head.x": -8, "spine.x": -3, "wing": 0.6, "wingL": 0.6, "wingR": 0.6, "tail.x": 0, "jaw": 12}),
            (1.8, {"ground": 0, "root.y": 0, "neck1.x": 0, "neck2.x": 0, "neck.x": 0, "neck1.z": 0, "neck2.z": 0,
                   "head.x": 0, "spine.x": 0, "wing": 0, "wingL": 0, "wingR": 0, "jaw": 0})]
    return frames_of(Track(keys, d, False)), False


def die():
    d = 2.6
    # v2c: the upper wing tucked 0.2 (was 0.7: its fingers closed to 4 degrees and crossed, a panel
    # turned over in Die's end and DiePose); now 14-15 degrees apart
    dead = {**DEAD, "rootr.z": -85, "thighL.z": 0, "thighR.z": 20, "tuckL": 0.3, "tuckR": 0.2,
            "neck1.z": 15, "neck2.z": 20, "neck3.z": 10, "head.z": 25, "tail.z": 10, "tail.y": -15}
    keys = [(0, {"ik": 1, "ground": 1}),
            (0.45, {"neck1.x": -16, "neck2.x": -26, "neck3.x": -22, "head.x": -44, "spine.x": -12, "jaw": 46,
                    "wing": 3.0, "wingL": 3.0, "wingR": 3.0, "tail.x": -8, "root.y": .02}),
            (0.95, {"neck2.x": -30, "head.x": -48, "jaw": 48, "spine.x": -14}),
            (1.25, {"neck1.x": 10, "neck2.x": 10, "neck3.x": 0, "head.x": 10, "jaw": 20, "spine.x": 10,
                    "wing": 1.4, "wingL": 1.4, "wingR": 1.4, "rootr.z": -12, "root.y": -.05, "footR.y": .0,
                    "footL.y": .05, "ik": 1}),
            (1.75, {**dead, "rootr.z": -60, "wing": 1.0, "neck2.x": 0, "neck1.x": 0,
                    "neck3.x": 0, "head.x": 0, "spine.x": 0, "root.y": 0}),
            (1.95, {**dead, "ground": 1}),
            (2.1, {"neck2.z": 26, "head.z": 30, "jaw": 30}),
            (2.6, dead)]
    return frames_of(Track(keys, d, False)), False


def being_picked_up():
    """Held up by the scruff: the shoulders at the hand, body hanging, legs and tail swinging."""
    d = 2.0
    tr = Track([(0, {"ik": 0, "neck.x": 20, "head.x": 18, "wing": 1.0, "jaw": 14, "tail.x": 14,
                     "thighL.x": -10, "thighR.x": -10, "shinL.x": -20, "shinR.x": -20,
                     "ankleL.x": 40, "ankleR.x": 40})], d, True)

    def extra(t, ch):
        w = 2 * np.pi * t / d
        k = 2 * np.pi * t / (d / 2)
        add(ch, **{"rootr__z": 10 * np.sin(w), "rootr__x": 6 * np.sin(w + 1),
                   "thighL__x": -15 * np.sin(k), "thighR__x": 15 * np.sin(k), "shinL__x": -10 * (1 + np.sin(k + 1)),
                   "shinR__x": -10 * (1 - np.sin(k + 1)), "tail__y": 22 * np.sin(w + .5), "tail__x": 10 * np.sin(2 * w),
                   "neck__y": 16 * np.sin(w - .7), "head__z": 10 * np.sin(w - 1.2), "jaw": 5 * np.sin(k),
                   "wing": 0.8 * np.sin(k * 0.5) ** 2, "flap": 0.2 * max(0, np.sin(k))})
    return frames_of(tr, extra), True


def torture():
    d = 2.4
    tr = Track([(0, {"wing": 2.4, "ik": 0, "thighL.z": 22, "thighR.z": -22, "thighL.x": -6, "thighR.x": -6,
                     "shinL.x": -16, "shinR.x": -16, "neck.x": -6, "head.x": -20, "jaw": 30, "tail.x": 20})],
               d, True)

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"spine__y": 12 * np.sin(w), "spine__z": 6 * np.sin(2 * w), "neck__y": -22 * np.sin(w + .5),
                   "neck__x": 10 * np.sin(2 * w + 1), "head__z": 18 * np.sin(w + 1.3), "jaw": 16 * np.sin(3 * w),
                   "tail__y": 25 * np.sin(w + 2), "flap": 0.5 + 0.5 * np.sin(2 * w),
                   "thighL__x": 20 * np.sin(2 * w), "thighR__x": -20 * np.sin(2 * w)})
    return frames_of(tr, extra), True


# ---- the slots a full DK2 creature also plays (Soles, 13 Nov: "finish the dragon's animations fully") ----

def entrance():
    """Out of the Portal: crouched with the wings wrapped, he rears up, throws the wings wide, roars,
    beats them twice and settles."""
    d = 4.4
    crouch = {"root.y": -.11, "spine.x": 26, "neck1.x": 22, "neck2.x": 26, "neck3.x": 12, "head.x": 18,
              "wing": 0, "tuck": 0.3, "tail.x": 10, "jaw": 0}     # v2c: tuck 0.5 -> 0.3 (fingers 8 -> 12 degrees)
    keys = [(0, crouch), (0.35, {**crouch, "neck1.x": 14, "head.x": 6}),
            (0.75, {"root.y": .02, "spine.x": -10, "neck1.x": -12, "neck2.x": -18, "neck3.x": -12,
                    "head.x": -20, "tuck": 0, "wing": 3.0, "jaw": 14, "tail.x": -4}),
            *roar_keys(0.75, wing=3.0, hold=0.45),
            (1.75, {"jaw": 20, "flap": 0.9, "wing": 2.6, "root.y": .01}),
            (2.05, {"flap": 0, "wing": 3.0}),
            (2.3, {"flap": 0.8, "wing": 2.4, "jaw": 8}),
            (2.8, {"root.y": 0, "spine.x": 0, "neck1.x": 0, "neck2.x": 0, "neck3.x": 0, "head.x": 0,
                   "wing": 0.5, "flap": 0, "jaw": 0, "tail.x": 0}),
            (3.2, {"neck.y": 26, "head.y": 20, "head.x": -6, "wing": 0.2}),     # sizes up his new dungeon
            (3.65, {"neck.y": -26, "head.y": -20}),
            (3.9, {"neck.y": 0, "head.y": 0, "head.z": 12, "jaw": 10}),         # a satisfied snort
            (4.05, {"head.z": -10}), (4.2, {"head.z": 0, "jaw": 0}),
            (4.4, {"neck.y": 0, "head.y": 0, "head.x": 0, "wing": 0})]

    def extra(t, ch):
        if 0.9 < t < 1.65:      # the roar shakes him
            add(ch, **{"chest__x": 2 * np.sin(t * 46), "jaw": 3 * np.sin(t * 37)})
    return frames_of(Track(keys, d, False), extra), False


def electrocute():
    """Caught in lightning: rigid, wings half out, shaking hard (a loop)."""
    d = 0.8
    tr = Track([(0, {"wing": 2.2, "jaw": 34, "neck.x": -10, "head.x": -22, "spine.x": -14, "tail.x": -12,
                     "root.y": .015})], d, True)

    def extra(t, ch):           # 4 to 6 jolts a loop: reads as shaking, not noise, at 30 fps
        w = 2 * np.pi * t / d
        add(ch, **{"rootr__z": 5 * np.sin(4 * w), "rootr__x": 3 * np.sin(4 * w + 1), "neck__y": 8 * np.sin(5 * w),
                   "head__z": 10 * np.sin(5 * w + .5), "jaw": 8 * np.sin(6 * w), "flap": 0.12 + 0.08 * np.sin(4 * w),
                   "tail__y": 10 * np.sin(4 * w + 2), "wingL": 2.2 + 0.15 * np.sin(4 * w),
                   "wingR": 2.2 - 0.15 * np.sin(4 * w + 1)})
    return frames_of(tr, extra), True


def drunk_idle():
    """Standing drunk: feet planted, swaying, head lolling, one hiccup (a loop)."""
    d = 3.6
    keys = [(0, {"neck.x": 14, "head.x": 16, "wing": 0.6, "root.y": -.02, "jaw": 8}),
            (2.1, {"jaw": 8, "head.x": 16}), (2.2, {"jaw": 26, "head.x": -8, "neck.x": 4, "root.y": .01}),
            (2.4, {"jaw": 6, "head.x": 18, "neck.x": 16, "root.y": -.02}), (3.6, {})]
    tr = Track(keys, d, True)

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"rootr__z": 6 * np.sin(w), "rootr__x": 3 * np.sin(2 * w), "neck__z": -9 * np.sin(w - .7),
                   "head__z": -12 * np.sin(w - 1.2), "neck__y": 8 * np.sin(w + .4), "tail__y": 10 * np.sin(w + 1),
                   "wingL": 0.3 * np.sin(w), "wingR": -0.3 * np.sin(w)})
    return frames_of(tr, extra), True


def walkback():
    """Backing away: the stride runs in reverse, head up and watchful."""
    period = 1.4
    tr = Track([(0, {"head.x": -8, "neck.x": -6, "jaw": 4})], period, True)

    def extra(t, ch):
        _gait(t, ch, period, stride=-0.15, lift=0.05, bob=0.014, sway=4.0, lean=-3.0)
        add(ch, wing=0.5 + 0.1 * np.sin(4 * np.pi * t / period))
    return frames_of(tr, extra), True


def dejected():
    """Unhappy: head hung low, wings drooping, tail down (the DejectedPose comes from frame 0)."""
    d = 3.0
    tr = Track([(0, {"neck1.x": 20, "neck2.x": 22, "neck3.x": 12, "head.x": 22, "spine.x": 10, "wing": 0.35,
                     "tuck": 0.15, "tail.x": 16, "root.y": -.02})], d, True)   # v2c: tuck 0.35 -> 0.15, fingers 16 deg
    return frames_of(tr, lambda t, ch: _breath(t, ch, 3.0, 0.5)), True


def _ease(u):
    return u * u * (3 - 2 * u)


def transition(a, b, seconds):
    """A start or end clip: pose `a` eased into pose `b` (DK2 plays these between a loop and the idle)."""
    from scripts.dragon_pose import blend
    n = max(2, int(round(seconds * FPS)) + 1)
    return [blend(a, b, _ease(i / (n - 1))) for i in range(n)], False


def _first(fn):
    return fn()[0][0]


def _between(loop_fn, seconds, end=False):
    idle, loop = _first(stand), _first(loop_fn)
    return transition(loop, idle, seconds) if end else transition(idle, loop, seconds)


# lengths after the Bile Demon's own Start / End files (40/40, 18/12, 56/55, 45/32 frames)
TRANSITIONS = {
    "praying_start": (praying, 1.6, False), "praying_end": (praying, 1.6, True),
    "manufacturing_start": (manufacturing, 1.3, False), "manufacturing_end": (manufacturing, 1.0, True),
}


GAITS = {   # the loop's own stride, eased in from the idle (Start) or out to it (End): feet lift as they move
    "walk": dict(track={"head.x": 4, "neck.x": 2}, period=1.2, stride=0.2, lift=0.06, bob=0.018, sway=5.0, lean=3.0,
                 wing=lambda t, p: 0.04 + 0.04 * np.sin(4 * np.pi * t / p + 1)),
    "walkback": dict(track={"head.x": -8, "neck.x": -6, "jaw": 4}, period=1.4, stride=-0.15, lift=0.05, bob=0.014,
                     sway=4.0, lean=-3.0, wing=lambda t, p: 0.5 + 0.1 * np.sin(4 * np.pi * t / p)),
}


def gait_transition(which, seconds, end):
    g = GAITS[which]
    n = max(2, int(round(seconds * FPS)) + 1)
    out = []
    for i in range(n):
        t = seconds * i / (n - 1)
        e = _ease(1 - t / seconds) if end else _ease(t / seconds)
        tg = t if end else t - seconds          # the Start ends, and the End begins, on the loop's frame 0
        full = dict(g["track"])                 # the loop's own channels at this moment, then scaled
        _gait(tg, full, g["period"], stride=g["stride"], lift=g["lift"], bob=g["bob"], sway=g["sway"],
              lean=g["lean"])
        add(full, wing=g["wing"](tg, g["period"]))
        ch = {k: (v if k.endswith(".y") and k.startswith("foot") else v * e) for k, v in full.items()}
        out.append(build_pose(ch))
    return out, False


CLIPS = {
    "stand": stand, "fidgeting": fidgeting, "walk": walk, "running": running, "patrolling": patrolling,
    "attack": attack, "breathe_fire": breathe_fire, "happy": happy, "angry": angry, "celebrating": celebrating,
    "getting_hit": getting_hit, "being_slapped": being_slapped, "staggering": staggering,
    "drinking": drinking, "eating": eating, "manufacturing": manufacturing, "praying": praying,
    "interacting_with_objects": interacting, "sleeping": sleeping, "falling": falling,
    "waking_up": waking_up, "die": die, "being_picked_up": being_picked_up, "torture": torture,
    "entrance": entrance, "electrocute": electrocute, "drunk_idle": drunk_idle, "walkback": walkback,
    "dejected": dejected,
}
for _name, (_fn, _sec, _end) in TRANSITIONS.items():
    CLIPS[_name] = (lambda fn=_fn, sec=_sec, end=_end: _between(fn, sec, end))
for _name, (_which, _sec, _end) in {"walk_start": ("walk", 0.9, False), "walk_end": ("walk", 0.9, True),
                                     "walkback_start": ("walkback", 0.6, False),
                                     "walkback_end": ("walkback", 0.55, True)}.items():
    CLIPS[_name] = (lambda w=_which, sec=_sec, end=_end: gait_transition(w, sec, end))


def build_clips(names=None):
    """-> {name: (list of Poses, wrap)}."""
    return {n: CLIPS[n]() for n in (names or CLIPS)}
