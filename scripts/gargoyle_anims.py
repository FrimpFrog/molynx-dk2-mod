"""The Gargoyle's clips, keyframed with the Dragon's machinery (scripts.dragon_anims: Track, add) on the
gargoyle's own skeleton (scripts.gargoyle_pose).

Stage 2 (Soles approved the look, 13 Nov; the face made angrier): the test set
  stand      Idle1   crouched and breathing, the head twitching from side to side, the tail flicking, the
                     wings shifting on his back
  statue     Idle2   he freezes into a perched cathedral-statue pose (wings half raised, hands on his
                     knees, chin out, a frozen grimace) and holds stone-still, then snaps back to life with
                     a quick head turn and a hiss
  walk       Walk    a hunched, scuttling, slightly hopping walk on the spot
  running    Run     bounding hops on the spot, both feet together, the wings beating
  attack     Melee   a vicious claw swipe, then a lunging bite
  angry      Angry   rears up, wings flared high and wide, screeches
  entrance   Entrance  bursts up out of a tight crouch with a wing flap (out of the Portal)
Like the game's own walk and run files, Walk and Run stay on the spot: a foot on the ground moves straight
back at a steady speed (the walk's TREADMILL_SPEED, the run's) while the game carries him forward.

Channels:
  <bone>.x / .y / .z   local rotation in degrees (x pitch: + lowers a forward-pointing bone and swings a
                       hanging arm back; y yaw; z roll) of spine, chest, neck, head, jaw, tail1..3 and the
                       arms' bones (upperAL, foreAL, handAL ...)
  body.*, front.*, tail.*  spread over spine + chest, neck + head, tail1..3
  arms.*, fores.*, hands.*  both arms' upper, fore and hand bones at once (y and z mirrored on the right:
                       z + spreads them outward)
  jaw                  the jaw's extra opening in degrees (CLOSE shuts the mouth)
  root.x/y/z           pelvis offset (model units); rootr.x/y/z the pelvis rotation (x + leans forward)
  wing, wingL, wingR   0 folded down the back (the stand), 1 half open, 2 spread (the bind), 3 raised high
  flap, flapL, flapR   0..1 toward the down stroke
  L.x/y/z, R.*         each foot's ankle offset from where it stands (IK keeps the feet planted; x +
                       outward); L.p the foot's pitch (+ toes down)
Every clip is built at 30 fps; a looping clip's last frame is a copy of its first.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_anims import Track, add
from scripts.gargoyle_model import ARMS, BONE_INDEX, JAW_OPEN, LEGS, WINGS, build_gargoyle, gargoyle_skinner
from scripts.gargoyle_pose import GargoylePose, ankle_rest, blend, euler, leg_ik, pose, yaw_only

FPS = 30
CLOSE = -JAW_OPEN * 0.9
SIDE = {"L": 1, "R": -1}

# ------------------------------------------------------------------------------------------- wings

# folded: the arm up and back from the shoulder blade, the wrist a knob just above and behind the
# shoulders, the fingers hanging down and back so the membrane drapes over the back like a cloak
FOLD = {
    "wing1L": ((.32, .62, -.72), (1, 0, .35)),
    "wing2L": ((.02, .9, .42), (1, 0, .2)),
    "f1L": ((.12, -.55, -.83), (1, 0, .2)),
    "f2L": ((.14, -.74, -.66), (1, 0, .2)),
    "f3L": ((.12, -.92, -.38), (1, 0, .2)),
}
# The open wing keeps the bind's finger fan: spread, raised and the down stroke turn the whole open wing
# about the shoulder (a rigid turn of wing1) rather than aiming each finger, so no membrane triangle
# collapses or flips (review round 1). Spread is tilted so the membrane faces up 30 degrees more and the
# tips come 10 degrees forward: DK2's high camera sees it rather than its edge. (Euler degrees, left.)
SPREAD_TURN = (-30, -10, 0)
RAISED_TURN = (12, -10, 38)        # (upright, unlike spread: rearing tips it back; it reads from behind too)
DOWN_TURN = (-12, -8, -42)        # (tips further forward sweep the membrane through the legs)
HALF_AT = 0.55                    # the half-open wing: this far from folded to spread
_PRESETS = None
WING_BONES = [b for s in "LR" for b in WINGS[s]]


def _turned(rx, ry, rz):
    return pose({"wing1L": (rx, ry, rz)})


def presets():
    """[folded, half open, spread, raised, down stroke] as poses."""
    global _PRESETS
    if _PRESETS is None:
        fold, spread = pose({"aim": FOLD}), _turned(*SPREAD_TURN)
        _PRESETS = [fold, blend(fold, spread, HALF_AT), spread, _turned(*RAISED_TURN), _turned(*DOWN_TURN)]
    return _PRESETS


def wing_locals(level, side, flap=0.0):
    ps = presets()
    level = float(np.clip(level, 0, 3))
    i = min(int(np.floor(level)), 2)
    P = blend(ps[i], ps[i + 1], level - i)
    if flap:
        P = blend(P, ps[4], float(np.clip(flap, 0, 1)))
    return {b: P.local[BONE_INDEX[b]] for b in WINGS[side]}


# ------------------------------------------------------------------------------------------- poses

MACROS = {
    "body": (("spine", .5), ("chest", .5)),
    "front": (("neck", .4), ("head", .6)),
    "tail": (("tail1", 1 / 3), ("tail2", 1 / 3), ("tail3", 1 / 3)),
}
TRUNK = ["spine", "chest", "neck", "head", "tail1", "tail2", "tail3"]
ARM_MACROS = {"arms": 0, "fores": 1, "hands": 2}
REST_ANKLES = {s: ankle_rest(s) for s in "LR"}
KNEE_HINT = np.array([.3, .4, 1.0])

# his posture under every clip: the neck thrust forward and the head held low, looking up from under
# the brow (Soles, 13 Nov: angrier and scarier)
# and (review round 1) the arms carried out clear of the thighs, the forearms a little forward, the
# mouth closed a little at rest (the snarl and the bared fangs stay), so a bite or a screech can open it
POSTURE = {"chest.x": 4.0, "neck.x": 18.0, "head.x": -12.0, "arms.z": 16.0, "fores.x": -10.0, "jaw": -6.0}


def build_pose(ch):
    """Channel values -> GargoylePose."""
    ch = dict(ch)
    for k, v in POSTURE.items():
        ch[k] = ch.get(k, 0.0) + v
    P = GargoylePose()
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
    for b, r in rot.items():
        P.local[BONE_INDEX[b]] = euler(*r)
    P.local[BONE_INDEX["root"]] = euler(ch.get("rootr.x", 0), ch.get("rootr.y", 0), ch.get("rootr.z", 0))
    P.root = np.array([ch.get("root.x", 0), ch.get("root.y", 0), ch.get("root.z", 0)])
    for s in "LR":
        lv = ch.get("wing" + s, 0.0) + ch.get("wing", 0.0)
        fl = ch.get("flap" + s, 0.0) + ch.get("flap", 0.0)
        for b, R in wing_locals(lv, s, fl).items():
            extra = [ch.get(f"{b}.{a}", 0.0) for a in "xyz"]
            P.local[BONE_INDEX[b]] = R @ euler(*extra) if any(extra) else R
    Rw, _ = P.world()
    yaw = yaw_only(Rw[BONE_INDEX["root"]])
    for side in "LR":
        off = np.array([ch.get(f"{side}.{a}", 0.0) for a in "xyz"]) * [SIDE[side], 1, 1]
        hint = yaw @ (KNEE_HINT * [SIDE[side], 1, 1])
        # the foot keeps its own (world) heading: a planted foot must not twist when the pelvis turns
        leg_ik(P, side, REST_ANKLES[side] + off, hint, euler(ch.get(f"{side}.p", 0.0), 0, 0),
               ch.get(f"{side}.t", 0.0))
    # free legs (lying, hanging, on the wheel): `ik` 0 gives every leg bone its own channels (thighL.x ...,
    # bind-relative; fk_legs mirrors them), 0..1 blends from the planted legs to those
    for side in "LR":
        ik = float(np.clip(ch.get("ik" + side, ch.get("ik", 1.0)), 0, 1))
        if ik >= 1:
            continue
        idx = [BONE_INDEX[b] for b in LEGS[side]]
        F = P.copy()
        for i, b in zip(idx, LEGS[side]):
            F.local[i] = euler(*[ch.get(f"{b}.{a}", 0.0) for a in "xyz"])
        mixed = blend(F, P, ik)
        for i in idx:
            P.local[i] = mixed.local[i]
    return P


def fk_legs(thigh=(0, 0, 0), shin=(0, 0, 0), foot=(0, 0, 0), toes=(0, 0, 0), sides="LR", **kw):
    """FK channels of the legs (the R side's y and z mirrored); kw: per-side overrides thighR=(..) etc."""
    out = {}
    for side in sides:
        s = SIDE[side]
        for stem, v in zip(("thigh", "shin", "foot", "toes"), (thigh, shin, foot, toes)):
            v = kw.get(stem + side, v)
            out.update({f"{stem}{side}.x": v[0], f"{stem}{side}.y": v[1] * s, f"{stem}{side}.z": v[2] * s})
    return out


_SKINNER = None


def skinner():
    global _SKINNER
    if _SKINNER is None:
        _SKINNER = gargoyle_skinner(build_gargoyle())
    return _SKINNER


GROUND = 0.002          # where a lying body's lowest point rests (the feet's own floor height)


def frames_of(track, extra=None):
    """Sample a Track at 30 fps -> list of GargoylePoses (a loop gets its first frame appended)."""
    n = int(round(track.duration * FPS))
    times = np.arange(n) / FPS if track.wrap else np.linspace(0, track.duration, n + 1)
    out = []
    for t in times:
        ch = track(t)
        if extra:
            extra(t, ch)
        P = build_pose(ch)
        g = float(ch.get("ground", 0.0))
        if g:                  # rest the lowest point on the floor (lying, collapsing): free legs only
            low = min(q[:, 1].min() for q in skinner()(*P.world()))
            P.root = P.root + np.array([0, g * (GROUND - low), 0])
        if ch.get("floor", 0.0) > 0:      # never below the floor (between keys that change his shape a lot)
            low = min(q[:, 1].min() for q in skinner()(*P.world()))
            if low < GROUND:
                P.root = P.root + np.array([0, GROUND - low, 0])
        out.append(P)
    if track.wrap:
        out.append(out[0].copy())
    return out


def _full(keys):
    """Keys as whole poses: a channel a key leaves out is at rest (0) there. (The Dragon's Track holds a
    channel's last value instead; these clips are written pose by pose.)"""
    names = sorted({c for _, d in keys for c in d})
    return [(t, {c: d.get(c, REST.get(c, 0.0)) for c in names}) for t, d in keys]


REST = {"ik": 1.0}       # channels whose rest is not 0 (planted legs)


def _loop(keys, d, extra=None):
    return frames_of(Track(_full(keys), d, True), extra), True


def _once(keys, d, extra=None):
    return frames_of(Track(_full(keys), d, False), extra), False


def _ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def _env(t, a, b, c, d):
    """0 before a, eased up to 1 by b, held to c, eased down to 0 by d."""
    if t <= a or t >= d:
        return 0.0
    if t < b:
        return _ease((t - a) / (b - a))
    if t <= c:
        return 1.0
    return 1.0 - _ease((t - c) / (d - c))


# ------------------------------------------------------------------------------------------- layers

def _breath(t, ch, period=2.0, amount=1.0):
    s = np.sin(2 * np.pi * t / period)
    add(ch, **{"body__x": -1.5 * amount * s, "root__y": 0.003 * amount * s, "head__x": 1.0 * amount * s,
               "arms__x": 1.5 * amount * s, "fores__x": -1.5 * amount * s, "wing1L__x": 1.5 * amount * s,
               "wing1R__x": 1.5 * amount * s})


TREADMILL_SPEED = {}          # clip -> model units per second a planted foot moves back (set by the gaits)


def _walk_feet(t, ch, period, stride, lift, duty):
    """Bipedal stride on the spot: each foot moves straight back on the ground for `duty` of the cycle
    (at a steady stride / (duty * period)), then swings forward lifted, toes trailing. -> (phase, the
    feet's forward offsets normalised to -1..1)."""
    ph = (t / period) % 1.0
    fwd = {}
    for s, off in (("L", 0.0), ("R", 0.5)):
        p = (ph + off) % 1.0
        if p < duty:                                 # stance: front to back, flat on the floor
            u = p / duty
            z, y, pitch = stride * (0.5 - u), 0.0, 0.0
        else:                                        # swing: back to front, lifted
            u = (p - duty) / (1 - duty)
            z = stride * (-0.5 + (1 - np.cos(np.pi * u)) / 2)
            y = lift * np.sin(np.pi * u)
            pitch = 8 * np.sin(np.pi * u)                # (more and the talons drag on the floor)
        add(ch, **{f"{s}__z": z, f"{s}__y": y, f"{s}__p": pitch})
        fwd[s] = z / (stride / 2)
    return ph, fwd


def _sampler(track, extra):
    def at(t):
        ch = track(t)
        if extra:
            extra(t, ch)
        return ch
    return at


def _channels_at(clip, t):
    """A clip's channel values at time t (for the Start / End clips that bridge into and out of it)."""
    return _SAMPLERS[clip]()(t)


FEET = ("L.x", "L.y", "L.z", "R.x", "R.y", "R.z")


def bridge(a, b, frames, steps=(("L", 0.0, 0.6), ("R", 0.4, 1.0)), lift=0.05, hop=0.0, splay=0.0):
    """A short clip from channel values `a` to `b` (both whole poses) over `frames` frames: everything
    eases across; each foot whose place changes steps there in an arc in its own window (`steps`: side,
    start and end as fractions of the clip, and optionally its lift: 0 keeps that foot on the floor,
    moving back under him as a planted foot does in the walk; both feet in one window is a little hop;
    and optionally its curve: "in" / "out" for a planted foot speeding up into a gait / slowing out of it)."""
    names = sorted(set(a) | set(b))
    out = []
    for i in range(frames + 1):
        u = i / frames
        e = _ease(u)
        ch = {c: a.get(c, 0.0) * (1 - e) + b.get(c, 0.0) * e for c in names if c not in FEET}
        for side, s0, s1, *own in steps:
            up = own[0] if own else lift
            k = float(np.clip((u - s0) / (s1 - s0), 0, 1))
            curve = own[1] if len(own) > 1 else "ease"
            # a planted foot ramping its slide up to the gait's treadmill speed ("in") or down from it ("out")
            ke = k * k if curve == "in" else 1 - (1 - k) ** 2 if curve == "out" else _ease(k)
            for ax in "xyz":
                c = f"{side}.{ax}"
                ch[c] = a.get(c, 0.0) * (1 - ke) + b.get(c, 0.0) * ke
            moved = abs(b.get(f"{side}.z", 0) - a.get(f"{side}.z", 0)) + abs(b.get(f"{side}.x", 0) - a.get(f"{side}.x", 0))
            if moved > 0.005 and up > 0:
                ch[f"{side}.y"] += up * np.sin(np.pi * k)
                ch[f"{side}.x"] += splay * np.sin(np.pi * k)       # (out round the knee of a high step)
                ch[f"{side}.p"] = ch.get(f"{side}.p", 0.0) + 8 * np.sin(np.pi * k)
        if hop:
            ch["root.y"] = ch.get("root.y", 0.0) + hop * np.sin(np.pi * u)
        out.append(build_pose(ch))
    return out


# ------------------------------------------------------------------------------------------- clips

def _stand_track():
    """Crouched, breathing; the head turns sharply one way, then the other (each turn eased over five
    frames), the tail flicks, the wings shift and resettle on his back, a little snarl."""
    d = 4.0
    keys = [(0, {}), (0.55, {}), (0.72, {"front.y": 26, "head.z": -6}), (1.35, {"front.y": 26, "head.z": -6}),
            (1.52, {"front.y": 10, "head.x": -6}), (1.9, {"front.y": 10, "head.x": -6}),
            (2.05, {"front.y": 10, "head.x": -6, "jaw": 9}), (2.35, {"front.y": 6, "head.x": -3}),
            (2.52, {"front.y": -24, "head.z": 6}), (3.25, {"front.y": -24, "head.z": 6}),
            (3.42, {}), (4.0, {})]

    def extra(t, ch):
        _breath(t, ch, 2.0)
        flick = np.exp(-((t - 1.0) / 0.09) ** 2) - np.exp(-((t - 3.0) / 0.09) ** 2)
        add(ch, tail3__y=6 * np.sin(2 * np.pi * t / 4.0) + 28 * flick, tail2__y=12 * flick,
            tail3__x=-10 * abs(flick))
        add(ch, wingL=0.22 * _env(t, 1.5, 1.7, 1.8, 2.2), wingR=0.18 * _env(t, 1.65, 1.85, 1.95, 2.35))
        add(ch, handAL__x=3 * np.sin(2 * np.pi * t / 4.0), handAR__x=-3 * np.sin(2 * np.pi * t / 4.0))
    return Track(_full(keys), d, True), extra


def stand():
    tr, ex = _stand_track()
    return frames_of(tr, ex), True


STATUE = {"root.y": -.022, "body.x": 2, "front.x": -22, "head.x": -8, "jaw": 9, "wing": 1.3,
          "arms.x": -34, "fores.x": -30, "hands.x": 20, "arms.z": -6, "tail.y": 30, "tail3.x": -20}


def statue():
    """Idle2: settles into a perched cathedral-statue pose and holds stone-still (no breath); then the
    stone cracks (a hard tremble), and he snaps back to life: a quick head turn and a hiss, the wings
    folding back over eight frames, a shake, and back to the crouch."""
    d = 5.4
    hiss = {"front.y": 38, "head.z": -10, "jaw": 30, "wing": 0.7, "root.y": .004, "body.x": -4,
            "arms.x": -10, "fores.x": -10}
    keys = [(0, {}), (0.8, dict(STATUE)), (2.8, dict(STATUE)), (2.93, dict(STATUE)),
            (3.12, hiss), (3.45, dict(hiss, jaw=34, wing=0.45, **{"front.y": 34})),
            (3.8, {"front.y": 8, "jaw": 4, "wing": 0.15}), (4.5, {"front.y": -6}), (5.4, {})]

    def extra(t, ch):
        still = _env(t, 0.5, 0.8, 2.8, 2.95)              # no breathing while he is stone
        _breath(t, ch, 2.7, amount=1 - still)
        if 2.8 <= t <= 2.95:                                # the stone cracks: a hard, quick tremble
            k = np.sin(np.pi * (t - 2.8) / 0.15)
            add(ch, body__z=3.5 * k * np.sin(2 * np.pi * (t - 2.8) * 22), head__y=4 * k * np.sin(2 * np.pi * (t - 2.8) * 19),
                root__y=.004 * k * np.sin(2 * np.pi * (t - 2.8) * 25), wing=0.08 * k)
        if 3.12 <= t <= 3.7:                                # the hiss: a quick shake of the head
            add(ch, head__y=5 * np.sin(2 * np.pi * (t - 3.12) * 9) * (1 - (t - 3.12) / 0.58))
    return _loop(keys, d, extra)


WALK = {"period": 0.8, "duty": 0.55, "speed": 0.545}          # 24 frames; the same ground speed as before


def _walk_track():
    period, duty = WALK["period"], WALK["duty"]
    stride = WALK["speed"] * duty * period
    TREADMILL_SPEED["walk"] = stride / (duty * period)
    tr = Track([(0, {"body.x": 6, "front.x": -4}), (period, {})], period, True)

    def extra(t, ch):
        ph, fwd = _walk_feet(t, ch, period, stride, lift=0.07, duty=duty)
        w = 2 * np.pi * ph
        hop = abs(np.sin(w))                               # up off each push-off
        # the arms swing against the legs: an arm goes back as its own leg comes forward
        add(ch, **{"root__y": -0.01 + 0.018 * hop, "rootr__z": 4 * np.sin(w), "rootr__y": 4 * np.sin(w),
                   "body__y": -4 * np.sin(w), "front__y": 3 * np.sin(w + .5), "front__x": 3 * np.cos(2 * w),
                   "tail__y": 10 * np.sin(w + 1.2), "upperAL__x": 20 * fwd["L"], "upperAR__x": 20 * fwd["R"],
                   "foreAL__x": -6 - 8 * fwd["L"], "foreAR__x": -6 - 8 * fwd["R"],
                   "wing": 0.06 + 0.06 * np.cos(2 * w)})
    return tr, extra


def walk():
    tr, ex = _walk_track()
    return frames_of(tr, ex), True


RUN = {"period": 0.5, "stride": 0.26, "duty": 0.42}


def _run_track():
    """Bounding hops on the spot, wing-assisted: both feet together push off as the wings beat down,
    he flies low with the legs tucked forward, the wings rise, and he lands into the next push."""
    period, stride, duty = RUN["period"], RUN["stride"], RUN["duty"]
    TREADMILL_SPEED["running"] = stride / (duty * period)
    tr = Track([(0, {"body.x": 14, "front.x": -10}), (period, {})], period, True)

    def extra(t, ch):
        ph = (t / period) % 1.0
        if ph < duty:                                      # both feet planted, moving back; crouch & push
            u = ph / duty
            z, y, pitch = stride * (0.5 - u), 0.0, 0.0
            root_y = -0.035 * np.sin(np.pi * u) + 0.012 * u
        else:                                              # airborne: the feet tuck up and swing forward
            u = (ph - duty) / (1 - duty)
            z = stride * (-0.5 + (1 - np.cos(np.pi * u)) / 2)
            y = 0.045 * np.sin(np.pi * u)               # (a higher tuck drives the knees into the arms)
            pitch = 30 * np.sin(np.pi * u)
            root_y = 0.012 + 0.1 * np.sin(np.pi * u) - 0.012 * u
        for s in "LR":
            add(ch, **{f"{s}__z": z, f"{s}__y": y + max(root_y, 0) * (ph >= duty), f"{s}__p": pitch})
        w = 2 * np.pi * ph
        beat = np.sin(w - 2.0)                           # the down stroke peaks in the air (ph ~0.57)
        add(ch, **{"root__y": root_y, "rootr__x": 6 - 6 * np.cos(w), "wing": 1.9 + 0.8 * max(-beat, 0),
                   "flap": max(beat, 0) * 0.9, "tail__x": -8 * np.cos(w), "arms__x": -22 - 12 * np.sin(w),
                   "arms__z": 22,                        # wide of the knees in the crouch
                   "upperAL__x": -4 * np.sin(w + 1), "upperAR__x": 4 * np.sin(w + 1),
                   "fores__x": -25, "front__x": 4 * np.sin(w), "jaw": 6 + 6 * np.sin(w)})
    return tr, extra


def running():
    tr, ex = _run_track()
    return frames_of(tr, ex), True


def _stand_ch():
    return _channels_at("stand", 0.0)


def walk_start():
    """WalkStart: from the crouch into the stride: the left foot steps forward while the right stays on
    the floor and moves back under him as he sets off (one foot always down)."""
    return bridge(_stand_ch(), _channels_at("walk", 0.0), 10, steps=(("L", 0.0, 0.6), ("R", 0.0, 1.0, 0.0))), False


def walk_end():
    """WalkEnd: out of the stride back to the crouch: the back foot steps up while the front one stays on
    the floor, moving back under him as he stops."""
    return bridge(_channels_at("walk", 0.0), _stand_ch(), 12, steps=(("R", 0.0, 0.6), ("L", 0.0, 1.0, 0.0))), False


def run_start():
    """RunStart: the wings unfurl as he drops into the first bound, both feet hopping forward."""
    return bridge(_stand_ch(), _channels_at("running", 0.0), 10, steps=(("L", 0.1, 0.8), ("R", 0.1, 0.8)),
                  lift=0.04), False


def run_end():
    """RunEnd: lands out of the bound and folds the wings back down his back."""
    return bridge(_channels_at("running", 0.0), _stand_ch(), 10, steps=(("L", 0.0, 0.7), ("R", 0.0, 0.7)),
                  lift=0.04), False


def attack():
    """Melee: twists back and cocks the right claw up and behind his shoulder, rakes it forward and down
    across in front of his face and follows through low to the left (the left arm pulls back against it),
    drifts back as he cocks his head, then lunges in with the jaws wide open and snaps them shut with a
    jerk of the head."""
    d = 1.2
    # The right arm's swing: with its z turn held at about -85 the arm's swing plane lies flat, so the x
    # turn alone sweeps it round at shoulder height (behind -> out -> in front) without twisting it
    # (Euler keys that change several turns at once twist the arm mid-swing: review round 1)
    wind = {"rootr.y": -22, "rootr.x": -6, "body.y": -12, "root.z": -.02, "upperAR.x": 40, "upperAR.z": -85,
            "foreAR.x": 0, "foreAR.y": -70,           # (the forearm cocked up: the claw high over his shoulder)
            "handAR.x": -30, "upperAL.x": -25, "foreAL.x": -20, "wing": 0.7, "jaw": 10,
            "front.y": 16, "front.x": -14}             # head up and turned away from the cocked claw
    rake = {"rootr.y": -4, "rootr.x": 8, "body.y": -2, "root.z": .025, "upperAR.x": -35, "upperAR.z": -85,
            "foreAR.x": -20, "handAR.x": 15, "upperAL.x": 5, "foreAL.x": -12, "wing": 0.85, "jaw": 4,
            "front.y": 14, "front.x": -18}
    swipe = {"rootr.y": 18, "rootr.x": 24, "body.y": 10, "root.z": .075, "root.y": -.012, "upperAR.x": -75,
             "upperAR.z": -30, "upperAR.y": 25, "foreAR.x": -5, "handAR.x": 10, "upperAL.x": 28, "foreAL.x": -8,
             "wing": 0.9, "jaw": 0, "front.y": 6, "front.x": -22}       # (the face kept up out of the claw's way)
    follow = dict(swipe, **{"rootr.y": 22, "body.y": 13, "upperAR.x": -55, "upperAR.z": -10, "upperAR.y": 40,
                            "jaw": 12, "front.y": -14, "front.x": -12})
    cock = {"rootr.y": 6, "rootr.x": -4, "root.z": -.015, "front.x": 12, "head.x": -4, "jaw": 30,
            "upperAR.x": -15, "upperAR.z": 12, "upperAL.x": -12, "foreAL.x": -28, "foreAR.x": -18, "wing": 1.0}
    lunge = {"rootr.x": 18, "root.z": .06, "root.y": -.012, "body.x": 10, "front.x": -24, "neck.x": 6,
             "jaw": 66, "upperAL.x": -40, "upperAR.x": -22, "foreAL.x": -22, "foreAR.x": -36, "arms.z": 10,
             "wing": 1.4}
    lunge["head.x"] = -36                       # the head tipped up as the jaws open: the high camera sees in
    snap = dict(lunge, jaw=CLOSE, **{"front.x": -38, "root.z": .072, "head.x": 8})
    keys = [(0, {}), (0.2, wind), (0.3, dict(wind, **{"upperAR.x": 48, "rootr.y": -26})), (0.36, rake),
            (0.42, swipe), (0.5, follow), (0.62, cock), (0.74, lunge), (0.78, snap),
            (0.86, dict(snap, jaw=4, **{"front.x": -16, "head.x": 0, "root.z": .05})),
            (1.0, {"jaw": 6, "root.z": .02, "wing": 0.4, "upperAL.x": -8, "upperAR.x": 6}), (1.2, {})]

    def extra(t, ch):
        _breath(t, ch, 1.2, 0.6)
        if 0.79 <= t <= 0.88:                             # the bite lands: a quick shake of the head
            add(ch, head__y=9 * np.sin(2 * np.pi * (t - 0.79) * 16) * (1 - (t - 0.79) / 0.09))
        # never quite still: a slow drift of the body and the head through the holds
        add(ch, body__y=2.5 * np.sin(2 * np.pi * t / 1.2 * 2 + .5), front__y=3 * np.sin(2 * np.pi * t / 1.2 * 3))
    return _loop(keys, d, extra)


def angry():
    """Rears up on his haunches, arms flung out wide and back with the claws spread, wings flared high,
    head thrown up and forward with the jaws wide in a screech that shakes his whole body, then drops
    back into the crouch."""
    d = 2.0
    rear = {"rootr.x": -24, "root.y": .03, "root.z": -.02, "tail1.x": 38, "tail.x": 8, "body.x": -6,
            "neck.x": 14, "head.x": -36, "jaw": 58, "wing": 2.9, "arms.x": 25, "arms.z": 75, "fores.x": -25,
            "hands.x": -30}
    keys = [(0, {}), (0.2, {"root.y": -.02, "wing": 0.6, "body.x": 6, "jaw": 10}), (0.46, rear),
            (1.3, dict(rear, jaw=52, wing=2.8)), (1.62, {"jaw": 8, "wing": 0.6, "root.y": -.006}), (2.0, {})]

    def extra(t, ch):
        k = _env(t, 0.46, 0.56, 1.22, 1.36)                # the screech shakes him
        f = 2 * np.pi * t
        add(ch, head__y=5 * k * np.sin(f * 11), head__x=3 * k * np.sin(f * 8 + 1), body__z=3 * k * np.sin(f * 11 + 1),
            body__x=2 * k * np.sin(f * 7), root__y=0.004 * k * np.sin(f * 13), jaw=7 * k * np.sin(f * 6),
            hands__x=12 * k * np.sin(f * 9), fores__x=5 * k * np.sin(f * 7 + 2),
            flap=0.25 * k * (0.5 + 0.5 * np.sin(f * 5.5)))
        _breath(t, ch, 2.0, 0.4)
    return _once(keys, d, extra)


ENTRANCE = {"depth": -1.0, "forward": 0.6, "d": 4.4, "land": 2.6}   # (land: his feet are down by then)
HANG, TOES = -0.025, 12.0       # the feet in flight: below where the crouch holds them; the toes' droop


def entrance():
    """Out of the Portal: he rises from under the floor beating his wings (three beats), climbs out
    tucked, glides forward and lands ENTRANCE["forward"] ahead with a bounce, folds his wings and snarls
    into his crouch there."""
    d, depth, fwd = ENTRANCE["d"], ENTRANCE["depth"], ENTRANCE["forward"]
    fly = {"wing": 2.2, "arms.x": -40, "arms.z": 26, "fores.x": -35, "body.x": 10, "front.x": -6, "jaw": 20,
           "tail.x": 15}
    keys = [(0, dict(fly, **{"root.y": depth, "root.z": 0.0})),
            (1.5, dict(fly, **{"root.y": -.12, "root.z": fwd * 0.25})),
            (2.1, dict(fly, **{"root.y": .13, "root.z": fwd * 0.7, "jaw": 30})),
            (2.55, dict(fly, wing=2.0, **{"root.y": .06, "root.z": fwd * 0.96, "jaw": 24})),
            (2.75, {"root.y": -.035, "root.z": fwd, "body.x": 10, "wing": 1.2, "jaw": 14, "arms.x": -20}),
            (3.0, {"root.y": .004, "root.z": fwd, "wing": 0.6, "jaw": 20, "front.y": 12}),
            (3.5, {"root.z": fwd, "jaw": 30, "front.x": -10, "head.x": -8, "wing": 0.25}),
            (3.9, {"root.z": fwd, "jaw": 6}), (d, {"root.z": fwd})]

    def extra(t, ch):
        # wing beats while he climbs and flies (three down strokes), easing out before he lands
        k = 1 - _ease((t - 2.2) / 0.35)
        ph = t / 0.62
        beat = np.sin(2 * np.pi * ph)
        add(ch, flap=k * 0.85 * max(beat, 0), wing=k * 0.7 * max(-beat, 0), root__y=k * 0.025 * -beat)
        # the feet: tucked under him while he flies (they ride with the body), set down for the landing
        air = 1 - _ease((t - 2.45) / 0.3)
        for s in "LR":   # hanging a little stretched (tucked up, the thighs drive into the belly); they
            #              reach the landing spot before the body does
            y = air * (ch.get("root.y", 0.0) + HANG)
            y = max(y, 0.0) if t > 2.3 else y                # (above the floor only once he is out)
            add(ch, **{f"{s}__y": y,
                       f"{s}__z": min(ch.get("root.z", 0.0) + 0.08 * _ease((t - 2.0) / 0.4), fwd),
                       f"{s}__p": TOES * (1 - _ease((t - 2.2) / 0.3))})    # level before they touch down
        _breath(t, ch, 2.2, 1 - air)                    # (two whole breaths by the end: it ends on the stand)
    return _once(keys, d, extra)


# ------------------------------------------------------------------------------------------- the full set
# (Soles, 14 Nov: every slot a DK2 creature plays, in character: a cheap, dim, scrappy little stone brute)

def _ltrack(keys, d, extra=None):
    return Track(_full(keys), d, True), extra


def _otrack(keys, d, extra=None):
    return Track(_full(keys), d, False), extra


def _frames(tr_ex):
    tr, ex = tr_ex
    return frames_of(tr, ex), tr.wrap


def _shake(t, ch, t0, t1, amount, freq, **chan):
    """A tremble between t0 and t1 (eased in and out) on the given channels (name: weight)."""
    k = _env(t, t0, t0 + 0.08, t1 - 0.08, t1) * amount
    for i, (c, w) in enumerate(chan.items()):
        ch[c] = ch.get(c, 0.0) + k * w * np.sin(2 * np.pi * freq * t + 1.7 * i)


# the right arm cocked for a rake and the rake itself (the flat swing plane, see attack); mirrored left
def _rake_arm(side, phase):
    s = SIDE[side]
    a = f"upperA{side}"
    f = f"foreA{side}"
    h = f"handA{side}"
    if phase == "cock":
        return {a + ".x": 40, a + ".z": 85 * s, f + ".y": 70 * s, h + ".x": -30}
    if phase == "out":
        return {a + ".x": -35, a + ".z": 85 * s, f + ".x": -20, h + ".x": 15}
    return {a + ".x": -75, a + ".z": 30 * s, a + ".y": -25 * s, f + ".x": -5, h + ".x": 10}


_ACROSS = {"rootr.x": 24, "root.z": .075, "root.y": -.012, "front.x": -22, "jaw": 0}   # (as Melee's swipe)


def melee2():
    """Melee2 (and the Training Room's dummy): a flurry of claw rakes, right, left, right, then a two-
    handed slash down, the head turned from each cocked claw and snapping at the target between."""
    d = 2.3
    R = lambda ph, **kw: dict(_rake_arm("R", ph), **kw)
    L = lambda ph, **kw: dict(_rake_arm("L", ph), **kw)
    keys = [(0, {}),
            (0.22, R("cock", **{"rootr.y": -18, "body.y": -10, "front.y": 16, "front.x": -14, "wing": 0.6})),
            (0.3, R("out", **{"rootr.y": -4, "front.y": 12, "front.x": -16, "root.z": .02, "wing": 0.7})),
            (0.38, R("across", **_ACROSS, **{"rootr.y": 16, "body.y": 8, "front.y": 4, "wing": 0.8})),
            (0.62, L("cock", **{"rootr.y": 18, "body.y": 10, "front.y": -16, "front.x": -14, "wing": 0.7})),
            (0.7, L("out", **{"rootr.y": 4, "front.y": -12, "front.x": -16, "root.z": .02, "wing": 0.75})),
            (0.78, L("across", **_ACROSS, **{"rootr.y": -16, "body.y": -8, "front.y": -4, "wing": 0.85})),
            (1.02, R("cock", **{"rootr.y": -16, "body.y": -8, "front.y": 14, "front.x": -14, "wing": 0.7})),
            (1.1, R("out", **{"rootr.y": -2, "front.y": 10, "front.x": -16, "root.z": .02, "wing": 0.8})),
            (1.18, R("across", **_ACROSS, **{"rootr.y": 16, "body.y": 8, "front.y": 4, "wing": 0.9})),
            # both claws flung out wide, then slashed in together before him (overhead they would pass
            # through his low-carried head)
            (1.45, {"arms.x": -20, "arms.z": 95, "fores.x": -30, "rootr.x": -10, "root.z": -.02, "front.x": -20,
                    "jaw": 26, "wing": 1.4, "body.x": -6}),
            (1.6, {"arms.x": -70, "arms.z": 20, "fores.x": -10, "rootr.x": 20, "root.z": .05, "root.y": -.015,
                   "front.x": -22, "jaw": 4, "wing": 0.9, "body.x": 12}),
            (1.85, {"arms.x": -30, "arms.z": 14, "rootr.x": 10, "root.z": .025, "jaw": 10, "wing": 0.5}), (2.3, {})]

    def extra(t, ch):
        _breath(t, ch, 2.3, 0.5)
        add(ch, body__y=2 * np.sin(2 * np.pi * t / 2.3 * 3))
    return _frames(_otrack(keys, d, extra))


def magic():
    """Magic (he has no spell): a stone screech: he rocks back, flares his wings, then thrusts his head at
    the enemy with the jaws wide, the screech shaking him, and snaps back into the crouch."""
    d = 1.7
    back = {"rootr.x": -10, "root.z": -.02, "front.x": -10, "head.x": -10, "jaw": 12, "wing": 1.8, "arms.z": 10}
    thrust = {"rootr.x": 16, "root.z": .05, "root.y": -.01, "body.x": 6, "neck.x": 8, "front.x": -18,
              "head.x": -26, "jaw": 56, "wing": 2.4, "wing1L.x": -40, "wing1R.x": -40, "arms.x": 20, "arms.z": 60,
              "fores.x": -20, "hands.x": -30}
    # (review 4: the flared wings tipped back so DK2's high camera sees their membranes, not their edges,
    # as Angry's)
    keys = [(0, {}), (0.3, back), (0.45, thrust), (1.15, dict(thrust, jaw=50, wing=2.3)),
            (1.4, {"jaw": 6, "wing": 0.6, "root.z": .01}), (1.7, {})]

    def extra(t, ch):
        _shake(t, ch, 0.45, 1.2, 1.0, 11, **{"head.y": 4, "body.z": 2.5, "jaw": 5, "hands.x": 10})
        _breath(t, ch, 1.7, 0.4)
    return _frames(_otrack(keys, d, extra))


def _hop(h, wing=1.6, flap=0.0, **kw):
    """Both feet off the floor by h (the body h higher), the feet tucked a little."""
    return dict({"root.y": h, "L.y": h * 0.9, "R.y": h * 0.9, "L.p": 10, "R.p": 10, "wing": wing,
                 "flap": flap}, **kw)


def happy():
    """Happy: two gleeful hops with a flutter of the wings, clapping his claws together, the tail
    wagging and the jaws snapping in a grin."""
    d = 2.4
    clap = {"arms.x": -45, "arms.z": 2, "fores.x": -15, "hands.x": -10, "front.x": -14}
    open_ = {"arms.x": -45, "arms.z": 30, "fores.x": -15, "hands.x": -10, "front.x": -14}
    keys = [(0, {}), (0.2, dict(open_, **{"root.y": -.03, "jaw": 10, "wing": 0.6})),
            (0.4, _hop(.07, 1.8, 0.0, **clap, jaw=18)), (0.52, _hop(.08, 1.9, 0.9, **open_, jaw=8)),
            (0.66, dict(clap, **{"root.y": -.02, "jaw": 16, "wing": 1.0})),
            (0.86, _hop(.07, 1.8, 0.0, **clap, jaw=18)), (0.98, _hop(.08, 1.9, 0.9, **open_, jaw=8)),
            (1.14, dict(clap, **{"root.y": -.02, "jaw": 16, "wing": 0.9})),
            (1.5, dict(open_, **{"jaw": 20, "wing": 0.6, "front.x": -10})), (1.8, dict(clap, jaw=12)),
            (2.4, {})]

    def extra(t, ch):
        add(ch, tail__y=14 * np.sin(2 * np.pi * t * 2.2) * _env(t, 0.1, 0.3, 2.0, 2.3))
        _breath(t, ch, 2.4, 0.5)
    return _frames(_otrack(keys, d, extra))


def dance():
    """Dance: a wing-flapping jig: from his folded crouch he bounds up with his wings flung open and claws
    wide, kicks out one foot then the other, two more bounds with a twist and a wing beat each, and folds
    back down into the crouch (review 4: three real hops a loop, the feet well clear of the floor)."""
    d = 2.0
    s0 = _stand_ch()
    wide = {"arms.x": -20, "arms.z": 95, "fores.x": 0, "hands.x": -20, "front.x": -16}      # flung wide
    crouch = lambda **kw: dict(wide, **dict({"root.y": -.04, "jaw": 14, "wing": 1.8}, **kw))
    H = DANCE["hop"]
    keys = [(0, s0),
            (0.18, dict(s0, **{"root.y": -.04, "wing": 1.0, "arms.z": 50, "arms.x": -20, "jaw": 10})),
            (0.33, _hop(H, 2.2, 0.0, **wide, **{"rootr.y": 4, "jaw": 20})),
            (0.48, crouch(flap=0.6)), (0.54, crouch(flap=0.3)),
            (0.65, crouch(**{"root.y": -.01, "L.y": DANCE["kick"], "L.x": .12, "L.z": -.06, "L.p": 14, "rootr.z": 8,
                             "front.y": 12, "wing": 2.0})),
            (0.73, crouch(**{"rootr.z": 5, "L.y": .035, "L.x": .01, "L.p": 4})),   # (set down, not slid)
            (0.78, crouch(**{"rootr.z": 3})),
            (0.9, crouch(**{"root.y": -.01, "R.y": DANCE["kick"], "R.x": .12, "R.z": -.06, "R.p": 14, "rootr.z": -8,
                            "front.y": -12, "wing": 2.0})),
            (0.98, crouch(**{"rootr.z": -5, "R.y": .035, "R.x": .01, "R.p": 4})),
            (1.04, crouch(**{"rootr.z": -3})),
            (1.2, _hop(H, 2.2, 0.0, **wide, **{"rootr.y": -10, "jaw": 22})),
            (1.35, crouch(flap=0.6)),
            (1.52, _hop(H, 2.2, 0.0, **wide, **{"rootr.y": 6, "jaw": 18})),
            (1.67, crouch(wing=1.0, flap=0.3)),
            (2.0, s0)]

    def extra(t, ch):
        add(ch, tail__y=16 * np.sin(2 * np.pi * t / 0.8 + 1) * _env(t, 0.1, 0.3, 1.7, 1.9))
    return _frames(_ltrack(keys, d, extra))


DANCE = {"hop": 0.13, "kick": 0.11}


def getting_hit():
    """RecoilHff (hit from the front): jerked back, head snapping back and wings flaring; he staggers a step
    back, shakes his head clear with a snarl and steps up into his crouch again (review 4: moving all
    through, not a jolt and a hold)."""
    d = 1.1
    hit = {"rootr.x": -20, "root.z": -.04, "root.y": .01, "front.x": -20, "head.x": -14, "jaw": 26,
           "wing": 1.2, "arms.x": -40, "arms.z": 45, "fores.x": -30, "tail.x": 15}
    back = dict(hit, **{"rootr.x": -12, "root.z": -.1, "root.y": -.01, "front.x": -6, "jaw": 20, "wing": 0.9,
                        "L.z": -.13, "arms.x": -15, "arms.z": 30, "fores.x": -20})
    keys = [(0, {}), (0.08, hit), (0.16, dict(hit, **{"root.z": -.06, "L.y": .09, "L.z": -.06, "L.p": 8})),
            (0.26, back), (0.42, dict(back, **{"rootr.x": 0, "front.x": 8, "front.y": 14, "jaw": 30,
                                               "wing": 0.7})),
            (0.58, dict(back, **{"rootr.x": 4, "front.x": 6, "front.y": -14, "jaw": 14, "wing": 0.5})),
            (0.7, dict(back, **{"rootr.x": 6, "front.y": 4, "jaw": 24, "wing": 0.4})),
            (0.8, {"root.z": -.05, "L.y": .09, "L.z": -.06, "L.p": 8, "jaw": 16, "wing": 0.3, "rootr.x": 4}),
            (0.9, {"jaw": 10, "wing": 0.2, "front.x": 4}), (1.1, {})]

    def extra(t, ch):
        _shake(t, ch, 0.3, 0.75, 1.0, 5, **{"head.y": 8, "body.z": 3})
    return _frames(_otrack(keys, d, extra))


def being_slapped():
    """RecoilHfb (hit from behind): knocked headlong, he stumbles a long step forward with the claws flung
    out, wings flaring, glares back over his shoulder with a snarl, and steps back into his crouch (review
    4: as big as the Goblin's lurch, and moving all through)."""
    d = 1.25
    hit = {"rootr.x": 30, "root.z": .06, "root.y": -.02, "front.x": 18, "jaw": 24, "wing": 1.4, "tail.x": 35,
           "arms.x": -45, "arms.z": 40, "fores.x": -20}
    lurch = dict(hit, **{"rootr.x": 22, "root.z": .14, "root.y": -.03, "front.x": 10, "R.z": .16, "R.x": .05})
    glare = dict(lurch, **{"rootr.x": 6, "root.z": .1, "root.y": -.01, "front.x": -6, "front.y": 36,
                           "rootr.y": 14, "jaw": 30, "wing": 0.8, "arms.x": -25, "arms.z": 25, "tail.x": 10})
    keys = [(0, {}), (0.08, hit), (0.17, dict(hit, **{"root.z": .1, "R.y": .09, "R.z": .08, "R.x": .1, "R.p": 8, "rootr.x": 24})),
            (0.27, lurch), (0.5, glare), (0.68, dict(glare, **{"front.y": 28, "jaw": 14, "rootr.y": 10})),
            (0.8, dict(glare, **{"front.y": 8, "rootr.y": 2, "jaw": 20, "wing": 0.5})),
            (0.92, {"root.z": .05, "R.y": .09, "R.z": .08, "R.x": .06, "R.p": 8, "jaw": 12, "wing": 0.3}),
            (1.02, {"jaw": 8, "wing": 0.2}), (1.25, {})]

    def extra(t, ch):
        _shake(t, ch, 0.45, 0.8, 1.0, 6, **{"head.y": 6, "body.z": 2})
    return _frames(_otrack(keys, d, extra))


ELEC = {"arms.x": -40, "arms.z": 70, "fores.x": -10, "hands.x": -30, "wing": 1.6, "jaw": 40, "front.x": -10,
        "head.x": -24, "rootr.x": -6, "root.y": .01, "tail.x": 25}


def electrocute():
    """Electrocute: rigid, spread-eagled in the lightning, shaking hard all over (ElecPose: its first frame)."""
    d = 1.0
    keys = [(0, ELEC), (1.0, ELEC)]

    def extra(t, ch):
        f = 2 * np.pi * t
        k = 1.0
        add(ch, body__z=4 * k * np.sin(f * 13), head__y=6 * k * np.sin(f * 15 + 1), head__x=3 * np.sin(f * 11),
            arms__x=6 * np.sin(f * 14 + 2), hands__x=12 * np.sin(f * 17), wing=0.15 * np.sin(f * 12),
            jaw=6 * np.sin(f * 9), root__y=0.004 * np.sin(f * 16), tail__y=8 * np.sin(f * 10))
    return _frames(_ltrack(keys, d, extra))


# (review 4: lurching steps at the Goblin's drunk speed, about 0.014 a frame in its files, the feet lifted)
DRUNK = {"period": 1.4, "duty": 0.6, "speed": 0.68, "lift": 0.092}


def _drunk_track():
    """Drunk (the drunken walk): an unsteady stagger on the spot, uneven steps, the body lurching and
    swaying, head lolling, arms dangling, the wings hanging half open and lopsided."""
    period, duty = DRUNK["period"], DRUNK["duty"]
    stride = DRUNK["speed"] * duty * period / 2            # two steps a cycle
    TREADMILL_SPEED["staggering"] = stride / (duty * period / 2)
    tr = Track([(0, {"body.x": 4, "front.x": 8, "jaw": 12, "wingL": 0.5, "wingR": 0.2, "arms.x": 10,
                     "fores.x": 10, "arms.z": 12, "L.x": 0.05, "R.x": 0.05, "rootr.x": -14}),
                (period, {})], period, True)

    def extra(t, ch):
        ph, fwd = _walk_feet(t, ch, period / 2, stride, lift=DRUNK["lift"], duty=duty)
        w = 2 * np.pi * t / period
        add(ch, **{"rootr.z": -11 * np.sin(2 * np.pi * (t - 0.035) / (period / 2)), "rootr.y": 4 * np.sin(w + 1), "root.x": -.02 * np.sin(w),
                   "front.y": 18 * np.sin(w + 2), "head.z": 12 * np.sin(w + 2.5), "front.x": 6 * np.sin(2 * w),
                   "upperAL.x": 10 * fwd["L"] + 8 * np.sin(w), "upperAR.x": 10 * fwd["R"] - 8 * np.sin(w),
                   "tail.y": 16 * np.sin(w + 3), "root.y": -.008 + .01 * abs(np.sin(2 * w))})
    return tr, extra


def staggering():
    return _frames(_drunk_track())


def drunk_idle():
    """DrunkIdle: swaying round in a slow circle where the stagger left him (it starts and ends on the
    Drunk walk's first frame, the feet where that leaves them), head lolling, wings drooping; a hiccup
    jolts him."""
    d = 4.0
    d0 = _channels_at("staggering", 0.0)
    base = dict(d0, **{"jaw": 14, "wingL": 0.45, "wingR": 0.15, "front.x": 10, "arms.x": 12, "fores.x": 8})
    for c in ("L.y", "R.y", "L.p", "R.p"):
        base[c] = 0.0
    keys = [(0, d0), (0.6, base), (1.9, base), (2.02, dict(base, **{"root.y": .02, "front.x": -14, "jaw": 30,
                                                                   "wingL": .7, "wingR": .4})),
            (2.3, base), (3.4, base), (4.0, d0)]

    def extra(t, ch):
        w = 2 * np.pi * t / 4.0
        k = _env(t, 0.0, 0.6, 3.4, 4.0)
        add(ch, **{"rootr.z": 8 * k * np.sin(w), "rootr.x": 6 * k * np.sin(2 * w), "front.y": 16 * k * np.sin(w + 1),
                   "head.z": 12 * k * np.sin(w + 1.6), "tail.y": 12 * k * np.sin(w + 2)})
        _breath(t, ch, 2.0, 0.8 * k)
    return _frames(_ltrack(keys, d, extra))


BOTTLE_HAND = np.array([-.15, .15, .20])     # the right claws' grip (bind, on the handAR bone)
MOUTH = np.array([0, .47, .30])              # between the lips (bind, on the head bone)
DRINK_TIMES = {"show": (0.25, 0.6), "hide": (3.3, 3.6)}


def drinking():
    """Drink: a bottle appears in his right claw; he raises it high, tips its neck into his jaws and glugs
    with his head thrown back, lowers it, smacks his jaws and it is gone (the bottle is the Goblin's own
    Bottle1, carried by the build: gargoyle_build)."""
    d = 3.8
    # (the right arm out wide and up: the claw above and beside the head, the bottle bridging claw and jaws;
    # searched so nothing passes through his low-carried head)
    left = {"upperAL.x": -15, "upperAL.z": 15, "foreAL.x": -30}
    hold = dict(left, **{"upperAR.x": -40, "upperAR.z": -45, "foreAR.y": -30, "foreAR.x": -40, "jaw": 8,
                         "front.x": -8})
    glug = dict(left, **{"upperAR.x": -70, "upperAR.z": -70, "foreAR.y": -60, "foreAR.x": -30, "front.x": -25,
                         "head.x": -25, "jaw": 24, "front.y": 8})
    smack = dict(left, **{"upperAR.x": -35, "upperAR.z": -40, "foreAR.y": -20, "foreAR.x": -35,
                          "front.y": -10, "front.x": -16, "jaw": 4})
    keys = [(0, {}), (0.5, hold), (1.0, glug), (2.2, dict(glug, jaw=26)),
            (2.6, dict(hold, jaw=10)), (3.0, smack), (3.35, dict(smack, **{"front.y": 6})), (3.8, {})]

    def extra(t, ch):
        add(ch, head__x=4 * np.sin(2 * np.pi * t * 2.5) * _env(t, 1.1, 1.2, 2.1, 2.2))   # the glugs
        _breath(t, ch, 1.9, 0.6)
        if 3.0 <= t <= 3.4:                                                             # the smacks
            add(ch, jaw=8 * np.sin(2 * np.pi * (t - 3.0) / 0.2) ** 2)
    return _frames(_otrack(keys, d, extra))


CHICKEN_AT = np.array([0.0, 0.0, 0.40])      # the chicken on the floor in front of him (model units)
EAT_TIMES = {"tears": (1.0, 1.45, 1.9), "gone": 2.3}


def eating():
    """Eat: pounces on the chicken on the floor in front, pins it with both claws and tears at it with
    his jaws (three tearing jerks of the head, the chicken smaller after each), gulps the last of it and
    smacks his chops. (The chicken is the Goblin's, carried by the build: gargoyle_build.)"""
    d = 3.2
    down = {"rootr.x": 26, "root.z": .05, "root.y": -.03, "body.x": 14, "front.x": 14, "arms.x": -40,
            "arms.z": 16, "fores.x": -30, "hands.x": 20, "wing": 0.5}
    tear = dict(down, **{"front.x": -4, "head.x": -18, "jaw": 30, "front.y": 10})
    bite = dict(down, **{"front.x": 18, "jaw": 30})
    keys = [(0, {}), (0.5, down), (0.8, bite), (1.0, tear), (1.15, bite), (1.45, dict(tear, **{"front.y": -10})),
            (1.6, bite), (1.9, tear), (2.15, dict(down, **{"jaw": 2, "front.x": 0})),
            (2.35, {"jaw": 22, "front.x": -18, "head.x": -12}), (2.6, {"jaw": 4, "front.y": -14, "front.x": -6}),
            (3.2, {})]

    def extra(t, ch):
        for tt in EAT_TIMES["tears"]:                     # a shake of the head as each piece tears away
            _shake(t, ch, tt - 0.05, tt + 0.15, 1.0, 10, **{"head.y": 8})
        _breath(t, ch, 1.6, 0.5)
    return _frames(_otrack(keys, d, extra))


RESEARCH = {"jaw": 4, "front.x": 6, "rootr.x": 6, "body.x": 6, "arms.x": -20, "arms.z": 6, "fores.x": -20}


def _research_track():
    """Research: he is no scholar: he sniffs at the book (quick dips of the snout), cocks his head at it,
    rakes at the page with one claw then the other, and scratches his own head."""
    d = 4.0
    R = dict(RESEARCH)
    keys = [(0, R), (0.4, dict(R, **{"front.x": 18, "jaw": 8})), (0.55, dict(R, **{"front.x": 12})),
            (0.7, dict(R, **{"front.x": 20, "jaw": 8})), (0.9, dict(R, **{"front.x": 8, "front.y": 14, "head.z": -18})),
            (1.1, dict(R, **{"upperAR.x": -35, "upperAR.z": -15, "foreAR.x": -10, "front.y": 18})),
            (1.3, dict(R, **{"upperAR.x": -50, "upperAR.z": -15, "foreAR.x": 0, "front.y": 18})),
            (1.45, dict(R, **{"upperAR.x": -20, "upperAR.z": -15, "foreAR.x": -20, "front.y": 18})),
            (1.6, dict(R, **{"upperAR.x": -50, "upperAR.z": -15, "foreAR.x": 0, "front.y": 18})),
            (1.75, dict(R, **{"upperAR.x": -20, "upperAR.z": -15, "foreAR.x": -20, "front.y": 18})),
            (1.88, dict(R, **{"front.y": 18})),
            # (raking at the page low in front of him, the claws below his low-carried head, the head turned
            # away from the raking arm)
            (2.02, dict(R, **{"front.y": -18})),
            (2.15, dict(R, **{"upperAL.x": -50, "upperAL.z": 15, "foreAL.x": 0, "front.y": -18})),
            (2.28, dict(R, **{"upperAL.x": -20, "upperAL.z": 15, "foreAL.x": -20, "front.y": -18})),
            (2.4, dict(R, **{"upperAL.x": -50, "upperAL.z": 15, "foreAL.x": 0, "front.y": -18})),
            (2.52, dict(R, **{"upperAL.x": -20, "upperAL.z": 15, "foreAL.x": -20, "front.y": -18})),
            (2.8, dict(R, **{"upperAR.x": 40, "upperAR.z": -85, "foreAR.y": -70, "foreAR.x": 0, "front.x": -4,
                             "front.y": 10, "head.z": 10, "jaw": 10})),
            (3.3, dict(R, **{"upperAR.x": 35, "upperAR.z": -80, "foreAR.y": -75, "foreAR.x": 0, "front.x": -2,
                             "front.y": 10, "head.z": 10, "jaw": 10})), (4.0, R)]

    def extra(t, ch):
        _breath(t, ch, 2.0, 0.6)
        _shake(t, ch, 2.8, 3.4, 1.0, 6, **{"foreAR.y": 12})       # scratching his head
    return _ltrack(keys, d, extra)


def interacting():
    return _frames(_research_track())


BUILD = {"jaw": 10, "wing": 0.4}


def _pound(side, phase):
    """One fist's blow, driven from the body (review 4: the whole torso rears and slams, not just the
    claws): "up" reared back, the elbow drawn back and the fist cocked by his shoulder, the head turned
    away from it; "down"
    the body slammed forward and down, the fist on the work before him. (searched: nothing through
    anything; the other arm held out of the way)"""
    s = SIDE[side]
    o = "R" if side == "L" else "L"
    rest = {f"upperA{o}.x": -20, f"upperA{o}.z": -30 * s, f"foreA{o}.x": -30}
    if phase == "up":
        return dict(BUILD, **rest, **{"rootr.x": -6, "body.x": -6, "root.y": .015, "front.x": -30,
                                      "front.y": -20 * s, "jaw": 16, "wing": 0.7, f"upperA{side}.x": 20,
                                      f"upperA{side}.z": 40 * s, f"foreA{side}.x": -90})
    return dict(BUILD, **rest, **{"rootr.x": 22, "body.x": 10, "root.y": -.03, "front.x": -10,
                                  "front.y": -10 * s, "jaw": 6, "wing": 0.3, f"upperA{side}.x": -40,
                                  f"upperA{side}.z": 30 * s, f"foreA{side}.x": -15})


def _building_track():
    """Building: hammers at the work with his fists, left then right: rearing back with the fist cocked
    high, then slamming his whole weight down behind the blow, the wings jolting open with each hit."""
    d = 1.6
    coil = lambda side: dict(_pound(side, "up"), **{"rootr.x": -10, "root.y": .022, "wing": 0.8, "jaw": 22})
    keys = [(0, _pound("L", "up")), (0.14, coil("L")), (0.26, _pound("L", "down")),
            (0.36, dict(_pound("L", "down"), wing=0.8)), (0.8, _pound("R", "up")), (0.94, coil("R")),
            (1.06, _pound("R", "down")), (1.16, dict(_pound("R", "down"), wing=0.8)), (1.6, _pound("L", "up"))]

    def extra(t, ch):
        for tt in (0.26, 1.06):
            _shake(t, ch, tt, tt + 0.18, 1.0, 14, **{"body.z": 3, "head.y": 4, "wing": 0.08})
    return _ltrack(keys, d, extra)


def manufacturing():
    return _frames(_building_track())


# Pray: he walks PRAY_WALK forward to the temple's edge in PrayStart and prays there (the Goblin's PrayStart
# walks 0.35 of its height forward; this brings his average place in Pray to the Goblin's in Goblin-Pray).
# (review 5: the forearms a little lower at the top of the bow, clear of the upper fang tips)
PRAY_WALK = 0.72
PRAY = {"root.z": PRAY_WALK, "L.z": PRAY_WALK, "R.z": PRAY_WALK,
        "root.y": -.03, "rootr.x": 8, "body.x": 4, "front.x": -28, "arms.x": -60, "arms.z": 50, "fores.x": -10,
        "hands.x": 20, "wing": 1.5, "jaw": -2}


def _pray_track():
    """Pray: kneeling with the claws stretched up and out before him, wings raised in reverence, he bows
    slowly right down, the snout near the floor, and rises again, twice (review 4: bowing, not holding)."""
    d = 3.0
    P = dict(PRAY)
    deep = dict(P, **{"rootr.x": 30, "body.x": 24, "front.x": -10, "arms.x": -40, "arms.z": 50, "wing": 1.2,
                      "floor": 1.0})
    keys = [(0, P), (0.75, deep), (1.5, P), (2.25, deep), (3.0, P)]          # two slow bows a loop

    def extra(t, ch):
        _breath(t, ch, 1.5, 0.6)
    return _ltrack(keys, d, extra)


def praying_start():
    """PrayStart: from his crouch he scuttles PRAY_WALK forward in six short steps (a foot always down, the
    other lifted clear), then kneels into the prayer."""
    W, n, ts, tk = PRAY_WALK, 6, 0.2, 0.4
    T = n * ts
    s0, p0 = _stand_ch(), _channels_at("praying", 0.0)
    lead = W / (n - 1)
    pos = {"L": 0.0, "R": 0.0}
    windows = []                                     # (side, from, to, t0, t1)
    for i in range(n):
        side = "LR"[i % 2]
        other = pos["R" if side == "L" else "L"]
        to = min(other + lead, W) if i < n - 1 else W
        windows.append((side, pos[side], to, i * ts, (i + 1) * ts))
        pos[side] = to
    walkpose = dict(s0, **{"rootr.x": -6, "front.x": -6, "arms.x": -15, "arms.z": 30, "jaw": 8, "wing": 0.15})
    out = []
    for i in range(int(round((T + tk) * FPS)) + 1):
        t = i / FPS
        feet = {"L": 0.0, "R": 0.0}
        lift = {"L": 0.0, "R": 0.0}
        for side, z0, z1, t0, t1 in windows:
            if t >= t1:
                feet[side] = z1
            elif t > t0:
                k = (t - t0) / (t1 - t0)
                feet[side] = z0 + (z1 - z0) * _ease(k)
                lift[side] = np.sin(np.pi * k)
        if t <= T:
            e = _ease(t / 0.25)
            ch = {c: s0.get(c, 0.0) * (1 - e) + walkpose.get(c, 0.0) * e for c in set(s0) | set(walkpose)}
            ch["root.z"] = (feet["L"] + feet["R"]) / 2
            ch["root.y"] = ch.get("root.y", 0.0) + 0.012 * max(lift.values())
            for sd in "LR":
                ch[f"{sd}.z"] = feet[sd]
                ch[f"{sd}.y"] = PRAY_STEP_LIFT * lift[sd]
                ch[f"{sd}.x"] = 0.04 * lift[sd]                   # (out round the knee)
                ch[f"{sd}.p"] = 8 * lift[sd]
                ch[f"upperA{sd}.x"] = ch.get(f"upperA{sd}.x", 0.0) - 14 * (feet[sd] - ch["root.z"]) / lead
        else:
            e = _ease((t - T) / tk)
            a = dict(walkpose, **{"root.z": W, "L.z": W, "R.z": W})
            ch = {c: a.get(c, 0.0) * (1 - e) + p0.get(c, 0.0) * e for c in set(a) | set(p0)}
        out.append(build_pose(ch))
    return out, False


PRAY_STEP_LIFT = 0.1


def praying_end():
    """PrayEnd: he rises from the prayer into his crouch where he knelt (PRAY_WALK ahead of his start)."""
    b = dict(_stand_ch(), **{"root.z": PRAY_WALK, "L.z": PRAY_WALK, "R.z": PRAY_WALK})
    return bridge(_channels_at("praying", 0.0), b, 15), False


def praying():
    return _frames(_pray_track())


# asleep on his belly, curled small under the blanket: the legs folded under him, the claws tucked at his
# sides, chin down, the folded wings laid flat along his back, the tail curled round beside him (review 4:
# low enough for the Goblin's blanket; searched: nothing through anything, his ears clear of the wings)
SLEEP = {"root.z": .11, "rootr.x": 45, "front.x": 10, "jaw": 0, "wing": 0.0, "arms.x": -30, "arms.z": 50,
         "fores.x": -30, "hands.x": 10, "tail.y": 40, "tail.x": -10, "floor": 1.0,
         "wing1L.x": 30, "wing1R.x": 30, "wing1L.y": -60, "wing1R.y": 60, "wing1L.z": -20, "wing1R.z": 20}


def sleeping():
    """Sleep: curled up on his belly like a fallen statue, chin down, wings laid flat along his back, the
    tail curled beside him; a slow, deep stone breath (the build lays him in the Goblin's bed, under the
    blanket with his head out at the pillow end, as the Goblin sleeps)."""
    d = 4.0
    rest = body_grounded(SLEEP, 0.004)
    keys = [(0, rest), (4.0, rest)]

    def extra(t, ch):
        s_ = np.sin(2 * np.pi * t / 4.0)
        w = 2 * np.pi * t / 4.0       # (and a dreaming twitch of the tail and head: never quite stone-still)
        add(ch, body__x=-6 * s_, root__y=-.004 * s_, front__x=8 * s_, jaw=5 * (s_ + 1), hands__x=8 * s_,
            tail__y=6 * s_ + 5 * np.sin(3 * w + 1), tail3__x=6 * np.sin(2 * w + .5))
    return _frames(_ltrack(keys, d, extra))


# flat on his back: the wings flung out flat on the floor under him, the head laid back, the claws up over
# his chest, the tail curled along the floor (searched: his back down at the floor, nothing propping him
# up, nothing through anything, his big ears clear of his arms); knocked well back (review 4: his body
# lands 0.3 of the files' units behind where he stood, as the Goblin's does)
LIE = dict({"ik": 0.0, "root.z": -.22, "rootr.x": -110, "body.x": -10, "wing": 3.0, "arms.x": -40, "arms.z": 40, "fores.x": -20,
            "hands.x": -20, "front.x": -66, "jaw": 20, "tail.x": 80, "floor": 1.0},
           **fk_legs(thigh=(-20, 0, 25), shin=(30, 0, 0), foot=(-20, 0, 0)))


def falling():
    """Fallback: knocked flat on his back: he staggers back, his feet leave the floor, he lands on his
    back, bounces, and lies dazed (StunnedPose is its last frame)."""
    d = 1.3
    tip = {"rootr.x": -30, "root.z": -.07, "wing": 1.2, "arms.x": -30, "arms.z": 60, "front.x": -10,
           "jaw": 24, "tail.x": 40}
    air = dict(tip, **{"ik": 0.0, "rootr.x": -65, "root.z": -.17, "tail.x": 30},
               **fk_legs(thigh=(-30, 0, 10), shin=(20, 0, 0)))
    keys = [(0, {}), (0.18, tip), (0.4, grounded(air, 0.05)), (0.55, grounded(LIE)),
            (0.68, grounded(dict(LIE, **{"rootr.x": -100, "front.x": -65}), 0.015)), (0.85, grounded(LIE)),
            (1.3, grounded(LIE))]

    def extra(t, ch):
        _shake(t, ch, 0.9, 1.3, 1.0, 3, **{"front.y": 6})
    return _frames(_otrack(keys, d, extra))


# DraggedPose (review 4: like the Goblin's GOB-Dragged_Pose, on his back on the floor, not hanging): flat on
# his back, head and shoulders a little raised, the wings flat under him (searched: nothing through anything)
DRAGGED = dict(LIE, **{"root.z": 0.0, "rootr.x": -105, "front.x": -45})


def dragged_pose():
    return build_pose(grounded(DRAGGED))


STATIC_POSES = {"dragged": dragged_pose}        # poses that are no clip's frame (the build's POSES)

GETUP_AT = -0.5        # where he stands up: under where his body lay (his trunk stays put; review 4)


def waking_up():
    """GetUp: from flat on his back he rocks back once, then springs up with a beat of his wings, lands on
    his feet in the crouch where he lay (GETUP_AT behind where he was knocked down, as the Goblin's own
    GetUp stands up where it fell), shakes his head clear and snarls."""
    d = 2.2
    rock = dict(LIE, **{"rootr.x": -122, "front.x": -70},
                **fk_legs(thigh=(-32, 0, 25), shin=(30, 0, 0), foot=(-20, 0, 0)))
    at = {"root.z": GETUP_AT, "L.z": GETUP_AT, "R.z": GETUP_AT}         # he stands up where he lay
    spring = {"ik": 0.0, "root.y": .1, "rootr.x": 10, "body.x": 6, "wing": 2.4, "flap": 0.0, "arms.x": -50,
              "arms.z": 30, "front.x": -10, "jaw": 20, "L.y": .1, "R.y": .1, **at, **fk_legs(thigh=(10, 0, 0))}
    beat = dict(spring, **{"root.y": .08, "flap": 1.0, "ik": 1.0, "L.y": .07, "R.y": .07, "L.p": 10, "R.p": 10})
    land = {"root.y": -.035, "rootr.x": 8, "body.x": 8, "wing": 0.7, "jaw": 14, "arms.x": -30, "arms.z": 12, **at}
    keys = [(0, grounded(LIE)), (0.4, grounded(rock)), (0.65, grounded(LIE)), (0.9, spring), (1.05, beat),
            (1.25, land), (1.45, {"jaw": 6, "wing": 0.3, **at}), (1.75, {"jaw": 22, **at}), (2.2, dict(at))]

    def extra(t, ch):
        _shake(t, ch, 1.45, 1.8, 1.0, 5, **{"front.y": 18})
    return _frames(_otrack(keys, d, extra))


# the heap he dies in: flat on his belly, chin on the floor, claws flung out, the wings slumped open over the
# floor; his feet stay where he stood, the legs folded under him (searched: nothing through anything, the
# stand's own hip contacts aside)
HEAP = {"root.z": .11, "rootr.x": 40, "rootr.z": 4, "front.x": -15, "jaw": 14, "wing": 0.9, "flap": 0.6,
        "wingL": -0.02, "arms.x": -40, "arms.z": 90, "fores.x": -20, "hands.x": 10, "tail.x": -10, "floor": 1.0}


def body_grounded(ch, lift=0.0):
    """Like `grounded`, but the feet stay planted (IK) and the rest of him (chest, head, wings) comes to
    rest on the floor: for collapsing onto the ground where he stands."""
    ch = dict(ch)
    ch["root.y"] = 0.0
    sk = skinner()
    pts = sk(*build_pose(ch).world())
    low = min(q[:, 1].min() for q, p in zip(pts, sk.parts) if p.name != "legs")
    ch["root.y"] = GROUND - low + lift
    return ch


def die():
    """Die: struck through, he rears and shudders, his stone cracking, then his legs give way: he pitches
    forward and crashes down in a heap on his belly where he stood, legs folded under him, wings slumped
    open over the floor, a last tremor through the stone, and sets like a broken statue (DiePose: its last
    frame). His feet stay where they were."""
    d = 2.6
    rear = {"rootr.x": -14, "root.y": .02, "front.x": -24, "head.x": -20, "jaw": 40, "wing": 1.8, "arms.x": -45,
            "arms.z": 65, "fores.x": -15}
    sag = {"root.y": -.06, "rootr.x": 25, "body.x": 18, "front.x": 20, "jaw": 20, "wing": 0.7, "arms.x": -45,
           "arms.z": 45, "fores.x": -20}             # (bracing: arms out, or they go through the floor)
    pitch = dict(HEAP, **{"root.z": .06, "rootr.x": 40, "front.x": -6, "jaw": 24, "wing": 1.4, "flap": 0.2,
                          "arms.x": -25, "arms.z": 85})
    heap = body_grounded(HEAP)
    keys = [(0, {}), (0.25, rear), (0.75, dict(rear, jaw=34)), (1.15, dict(sag, floor=1.0)),
            (1.4, body_grounded(pitch, 0.06)), (1.6, heap),
            (1.7, dict(heap, **{"root.y": heap["root.y"] + .014, "front.x": -22})), (1.8, heap), (2.6, heap)]

    def extra(t, ch):
        _shake(t, ch, 0.3, 1.0, 1.0, 13, **{"body.z": 3, "head.y": 5, "hands.x": 10, "jaw": 4})
        _shake(t, ch, 1.85, 2.15, 0.6, 9, **{"body.z": 1.5, "wing": 0.04})
    return _frames(_otrack(keys, d, extra))


HELD = dict({"ik": 0.0, "root.y": .3, "rootr.x": 10, "front.x": -10, "arms.x": -30, "arms.z": 20, "fores.x": -20,
             "wing": 1.0, "jaw": 18, "tail.x": -20}, **fk_legs(thigh=(40, 0, 10), shin=(-20, 0, 0), foot=(30, 0, 0)))


def being_picked_up():
    """Swing (held by the Keeper's hand by the scruff): legs dangling and kicking, claws raking the air,
    wings beating angrily, the tail lashing, the jaws snapping (the build keeps the scruff on the hand)."""
    d = 2.0
    keys = [(0, HELD), (2.0, HELD)]

    def extra(t, ch):
        w = 2 * np.pi * t / 1.0
        add(ch, **{"thighL.x": 14 * np.sin(w), "thighR.x": -14 * np.sin(w), "shinL.x": -10 * np.sin(w + 1),
                   "shinR.x": 10 * np.sin(w + 1), "upperAL.x": -14 * np.sin(w + .5), "upperAR.x": 14 * np.sin(w + .5),
                   "flap": 0.55 + 0.2 * np.sin(2 * w),        # (beating low: the tips no higher than the
                   #                                            Goblin's above the hand; review 4) "tail.y": 25 * np.sin(w + 2), "jaw": 12 * max(np.sin(2 * w), 0),
                   "front.y": 15 * np.sin(w / 2 + 1), "rootr.z": 6 * np.sin(w / 2)})
    return _frames(_ltrack(keys, d, extra))


# upright and flat against the wheel: his hunch straightened, arms and wings spread flat out to the sides
# (as thin front to back as he goes; the wings swung forward into the plane of his back so they stay out
# of the wheel (review 4); searched: the claws, wings and his big ears clear of each other however he
# thrashes his head)
WHEEL = dict({"ik": 0.0, "root.y": .16, "rootr.x": -30, "arms.x": 10, "arms.z": 95, "fores.x": -5,
              "hands.x": -20, "wing": 2.4, "front.x": -10, "jaw": 20, "tail.x": -85,
              "wing1L.y": -20, "wing1R.y": 20, "wing1L.z": -30, "wing1R.z": 30},
             **fk_legs(thigh=(-10, 0, 25), shin=(25, 0, 0), foot=(-10, 0, 0)))


def torture():
    """Torture: strapped spread-eagled to the wheel, he struggles: wrenching at his bonds, thrashing his
    head and screaming, the wings twitching (the build puts him on the wheel, as the Goblin's own file)."""
    d = 3.0
    keys = [(0, WHEEL), (0.6, dict(WHEEL, **{"rootr.z": 8, "front.y": 25, "jaw": 34})),
            (1.2, dict(WHEEL, **{"rootr.z": -6, "front.y": -20, "jaw": 12})),
            (1.7, dict(WHEEL, **{"rootr.z": 4, "front.y": 10, "front.x": -16, "jaw": 44})),
            (2.3, dict(WHEEL, **{"rootr.z": -8, "front.y": -25, "jaw": 20})), (3.0, WHEEL)]

    def extra(t, ch):
        _shake(t, ch, 0.0, 3.0, 1.0, 7, **{"upperAL.x": 8, "upperAR.x": -8, "thighL.x": 6, "thighR.x": -6,
                                           "wing": 0.06, "body.z": 2})
    return _frames(_ltrack(keys, d, extra))


# (review 4: as fast as the Goblin's walkback, 0.0135 a frame in its files, with the feet clear of the floor)
WALKBACK = {"period": 0.7, "duty": 0.55, "speed": 0.68, "lift": 0.095}


def _walkback_track():
    """Walkback: backing away, wary: short steps backward, leaning back a little, the head low and turned
    to the threat, the claws raised before him."""
    period, duty = WALKBACK["period"], WALKBACK["duty"]
    stride = WALKBACK["speed"] * duty * period
    TREADMILL_SPEED["walkback"] = -stride / (duty * period)
    tr = Track([(0, {"rootr.x": -17, "front.x": 6, "arms.x": -40, "fores.x": -30, "jaw": 12}), (period, {})],
               period, True)

    def extra(t, ch):
        ph, fwd = _walk_feet(t, ch, period, -stride, lift=WALKBACK["lift"], duty=duty)
        w = 2 * np.pi * ph
        add(ch, **{"root__y": -0.008 + 0.012 * abs(np.sin(w)), "rootr__z": 3 * np.sin(w),
                   "rootr__y": 3 * np.sin(w), "front__y": 4 * np.sin(w + .8), "tail__y": 8 * np.sin(w + 1.2),
                   "wing": 0.15 + 0.05 * np.cos(2 * w)})
    return tr, extra


def walkback():
    return _frames(_walkback_track())


def walkback_start():
    """WalkbackStart: the left foot steps back while the right, planted, slides forward under him faster
    and faster until it moves at the walkback's own speed (as WalkStart sets off)."""
    return bridge(_stand_ch(), _channels_at("walkback", 0.0), 10,
                  steps=(("L", 0.0, 0.6, WALKBACK["lift"]), ("R", 0.0, 1.0, 0.0, "in")), splay=0.03), False


def walkback_end():
    """WalkbackEnd: the front (right) foot steps back beside the other while the left, planted, slides on
    forward and slows to a stop."""
    return bridge(_channels_at("walkback", 0.0), _stand_ch(), 12,
                  steps=(("R", 0.0, 0.6, WALKBACK["lift"]), ("L", 0.0, 1.0, 0.0, "out")), splay=0.06), False


def dejected():
    """Dejected (the DejectedPose is its first frame): slumped, head hung, wings drooping, arms dangling."""
    d = 3.0
    base = {"root.y": -.02, "rootr.x": 10, "body.x": 16, "front.x": 28, "head.x": 8, "jaw": -2, "wingL": 0.25,
            "wingR": 0.25, "arms.x": 10, "arms.z": -6, "fores.x": 15, "hands.x": 15, "tail.x": -15}
    keys = [(0, base), (1.5, dict(base, **{"front.x": 32, "front.y": 8})), (3.0, base)]

    def extra(t, ch):
        _breath(t, ch, 3.0, 1.2)
    return _frames(_ltrack(keys, d, extra))


def grounded(ch, lift=0.0):
    """The key pose `ch` with root.y set so its lowest point rests on the floor (+ lift): for lying and
    collapsing keys (computed once per key, so the in-betweens interpolate smoothly)."""
    ch = dict(ch)
    ch["root.y"] = 0.0
    low = min(q[:, 1].min() for q in skinner()(*build_pose(ch).world()))
    ch["root.y"] = GROUND - low + lift
    return ch


def _bridge_loop(loop, start, frames=12, steps=()):
    """A Start (start=True: the stand into the loop) or End clip (the loop back to the stand)."""
    a, b = (_stand_ch(), _channels_at(loop, 0.0)) if start else (_channels_at(loop, 0.0), _stand_ch())
    return bridge(a, b, frames, steps=steps or (("L", 0.0, 0.6), ("R", 0.4, 1.0))), False


_SAMPLERS = {"stand": lambda: _sampler(*_stand_track()), "walk": lambda: _sampler(*_walk_track()),
             "running": lambda: _sampler(*_run_track()), "walkback": lambda: _sampler(*_walkback_track()),
             "interacting": lambda: _sampler(*_research_track()),
             "manufacturing": lambda: _sampler(*_building_track()), "praying": lambda: _sampler(*_pray_track()),
             "staggering": lambda: _sampler(*_drunk_track())}

CLIPS = {"stand": stand, "statue": statue, "walk": walk, "walk_start": walk_start, "walk_end": walk_end,
         "running": running, "run_start": run_start, "run_end": run_end, "attack": attack, "angry": angry,
         "entrance": entrance,
         # the full set
         "melee2": melee2, "magic": magic, "happy": happy, "dance": dance, "getting_hit": getting_hit,
         "being_slapped": being_slapped, "electrocute": electrocute, "staggering": staggering,
         "drunk_idle": drunk_idle, "drinking": drinking, "eating": eating,
         "interacting": interacting, "interacting_start": lambda: _bridge_loop("interacting", True),
         "interacting_end": lambda: _bridge_loop("interacting", False),
         "manufacturing": manufacturing, "manufacturing_start": lambda: _bridge_loop("manufacturing", True),
         "manufacturing_end": lambda: _bridge_loop("manufacturing", False),
         "praying": praying, "praying_start": praying_start,
         "praying_end": praying_end,
         "sleeping": sleeping, "falling": falling, "waking_up": waking_up, "die": die,
         "being_picked_up": being_picked_up, "torture": torture,
         "walkback": walkback, "walkback_start": walkback_start, "walkback_end": walkback_end,
         "dejected": dejected}


def build_clips(names=None):
    """-> {name: (list of GargoylePoses, wrap)}."""
    return {n: CLIPS[n]() for n in (names or CLIPS)}
