"""Sejanus's clips, keyframed on the Dragon's machinery (dragon_anims.Track: keys over named channels,
monotone cubics, loops wrap; procedural layers on top).

Stage 2 (test animations for Soles): Idle1, Idle2 (the Hunt), Walk, Run (a horse's gallop on all fours,
the arms as forelegs: his signature), Melee (a rearing two-handed blow), Angry and Entrance (up out of the
Portal from below the floor, like the game's own). Clips that start or end at rest blend into Idle1's
first frame there (frames_of anchor), so switching clips does not twitch.

Channels:
  <bone>.x / .y / .z   local rotation in degrees (x pitch: + tips forward / down; y yaw; z roll) of a trunk,
                       head, jaw, ear, tail or arm bone (FK)
  neck.*, spine.*, tail.*, ears.*  spread over neck1..2, spine1 + chest, tail1..3, both ears
  jaw                  the jaw's opening in degrees
  root.x/y/z           pelvis offset (model units); rootr.x/y/z the pelvis rotation (the whole body)
  footL.x/y/z          the hoof's offset from where it stands (IK places it; it stays put otherwise);
                       footL.p cannon pitch (+ folds the hoof back), toeL.p pastern pitch
  armik                0 = the arms swing free (FK), 1 = they are forelegs (IK) with
  gripL.x/y/z          the middle fingertip's world position, gripL.p the hand's pitch (+ folds it back)
Every clip is built at 30 fps; a looping clip's last frame is a copy of its first.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_anims import Track, add
from scripts.sejanus_model import BONE_INDEX, FINGER_TIP, HEAD, TAIL
from scripts.sejanus_pose import Pose, blend, euler, leg_ik, two_bone_ik

FPS = 30
MACROS = {
    "neck": (("neck1", .5), ("neck2", .5)),
    "spine": (("spine1", .5), ("chest", .5)),
    "tail": (("tail1", .34), ("tail2", .33), ("tail3", .33)),
    "ears": (("earL", 1.0), ("earR", 1.0)),
}
FK = ["spine1", "chest", "neck1", "neck2", "head", "jaw", "earL", "earR", "tail1", "tail2", "tail3"] + \
     [b + s for s in "LR" for b in ("clav", "arm", "forearm", "hand")]
JAW_REST = 0.0          # a horse's mouth: closed (Soles, 13 Nov)
_M = np.array([-1.0, 1, 1])
HOOF = {s: TAIL[BONE_INDEX["toes" + s]] for s in "LR"}           # the sole (bind)
FET = {s: HEAD[BONE_INDEX["toes" + s]] for s in "LR"}
HOCK = {s: HEAD[BONE_INDEX["foot" + s]] for s in "LR"}
WRIST = {s: HEAD[BONE_INDEX["hand" + s]] for s in "LR"}
TIP = {"L": FINGER_TIP, "R": FINGER_TIP * _M}
SIGN = {"L": 1.0, "R": -1.0}


def _yaw(P):
    Rw, _ = P.world()
    f = Rw[BONE_INDEX["root"]] @ [0, 0, 1.0]
    return np.degrees(np.arctan2(f[0], f[2]))


def _softplus(x, k=8.0):
    """max(0, x) with the corner rounded over about +-k degrees (no kink, so no twitch as it starts)."""
    return k * np.log1p(np.exp(np.clip(x / k, -50, 50)))


def build_pose(ch):
    """Channel values -> Pose."""
    P = Pose()
    rot = {b: np.zeros(3) for b in FK}
    for k, v in ch.items():
        if "." not in k:
            continue
        name, ax = k.split(".")
        if ax not in ("x", "y", "z"):
            continue
        i = "xyz".index(ax)
        if name in MACROS:
            for b, w in MACROS[name]:
                rot[b][i] += v * w * (SIGN[b[-1]] if (b.startswith("ear") and i) else 1)
        elif name in rot:
            rot[name][i] += v
    rot["jaw"][0] += JAW_REST + ch.get("jaw", 0.0)
    # the shoulders shrug as an arm goes up (forward or out), so the armpit does not pinch
    for s in "LR":
        up = _softplus(-rot["arm" + s][0] - 50.0) + _softplus(abs(rot["arm" + s][2]) - 30.0)
        rot["clav" + s][2] += 0.4 * up * SIGN[s]
    for b, r in rot.items():
        P.local[BONE_INDEX[b]] = euler(*r)
    P.local[BONE_INDEX["root"]] = euler(ch.get("rootr.x", 0), ch.get("rootr.y", 0), ch.get("rootr.z", 0))
    P.root = np.array([ch.get("root.x", 0), ch.get("root.y", 0), ch.get("root.z", 0)])
    yaw = _yaw(P)
    # hind legs free (lying, falling, held up): legik 0 takes the hip / knee / hock / pastern channels
    lik = float(np.clip(ch.get("legik", 1.0), 0, 1))
    if lik < 1:
        Pfk = P.copy()
        for s in "LR":
            for b, nm in LEG_FK:
                Pfk.local[BONE_INDEX[b + s]] = euler(*[ch.get(f"{nm}{s}.{a}", 0.0) for a in "xyz"])
    # hind legs: the hoof goes where it is told, the cannon and pastern pitched, the stifle forward
    for s in ("LR" if lik > 0 else ""):
        off = np.array([ch.get(f"foot{s}.{a}", 0.0) for a in "xyz"])
        Rf = euler(ch.get(f"foot{s}.p", 0.0), yaw, 0)
        tp = ch.get(f"toe{s}.p", 0.0)
        Rt = Rf @ euler(tp, 0, 0)
        hoof = HOOF[s] + off
        hock = hoof - Rt @ (HOOF[s] - FET[s]) - Rf @ (FET[s] - HOCK[s])
        hint = euler(0, yaw, 0) @ np.array([ch.get("kneeout", 0.1) * SIGN[s], ch.get("kneeup", 0.0), 1.0])
        leg_ik(P, s, hock, hint, foot_world=Rf, toe_pitch=tp)
    # arms as forelegs: the middle fingertip on its mark, the elbow back like a horse's
    ik = float(np.clip(ch.get("armik", 0.0), 0, 1))
    if ik > 0:
        F = P.copy()
        for s in "LR":
            tip = np.array([ch.get(f"grip{s}.{a}", 0.0) for a in "xyz"])
            Rh = euler(ch.get(f"grip{s}.p", 0.0), yaw, 0)
            wrist = tip - Rh @ (TIP[s] - WRIST[s])
            # the shoulder shrugs as the arm reaches forward (as for FK arms above)
            Rw, Pw = P.world()
            sh = Pw[BONE_INDEX["arm" + s]]
            down = Rw[BONE_INDEX["chest"]] @ np.array([0, -1.0, 0])
            ang = np.degrees(np.arccos(np.clip(np.dot(down, (wrist - sh) / np.linalg.norm(wrist - sh)), -1, 1)))
            c = BONE_INDEX["clav" + s]
            P.local[c] = P.local[c] @ euler(0, 0, ik * 0.4 * _softplus(ang - 50.0) * SIGN[s])   # (blended in)
            hint = euler(0, yaw, 0) @ np.array([ch.get("elbowout", 0.25) * SIGN[s], ch.get("elbowup", 0.0),
                                                ch.get("elbowback", 1.0) * -1.0])
            two_bone_ik(P, "arm" + s, "forearm" + s, "hand" + s, wrist, hint, end_world=Rh)
        if ik < 1:
            mixed = blend(F, P, ik)
            for s in "LR":
                for b in ("arm", "forearm", "hand"):
                    P.local[BONE_INDEX[b + s]] = mixed.local[BONE_INDEX[b + s]]
    if lik < 1:
        mixed = blend(Pfk, P, lik)
        for s in "LR":
            for b, _ in LEG_FK:
                P.local[BONE_INDEX[b + s]] = mixed.local[BONE_INDEX[b + s]]
    g = float(ch.get("ground", 0.0))
    if g > 0:           # lying or falling: let the body down onto the floor (its lowest point at 0)
        P.root = P.root - np.array([0.0, g * _lowest(P), 0.0])
    fl = float(ch.get("floor", 0.0))
    if fl > 0:          # in the air while falling: only kept from going through the floor
        low = _lowest(P)
        if low < 0:
            P.root = P.root - np.array([0.0, fl * low, 0.0])
    return P


LEG_FK = (("thigh", "hip"), ("shin", "knee"), ("foot", "hock"), ("toes", "past"))
_SK = []


def _lowest(P):
    if not _SK:
        from scripts.sejanus_model import Skinner, build_sejanus
        _SK.append(Skinner(build_sejanus()))
    return float(min(v[:, 1].min() for v in _SK[0](*P.world())))


def frames_of(track, extra=None, anchor=False, end_offset=None):
    """Sample a Track at 30 fps -> list of Poses (a loop gets its first frame appended).

    anchor: blend the first and last ANCHOR seconds into Idle1's first frame (IDLE0), so that switching
    between this clip and the idle has no twitch: "start" or "end" for one end only, True for both.
    end_offset: channels added to IDLE0 at the end (a clip that finishes somewhere else, e.g. the Entrance's
    walk forward out of the Portal)."""
    n = int(round(track.duration * FPS))
    times = np.arange(n) / FPS if track.wrap else np.linspace(0, track.duration, n)
    d = track.duration
    out = []
    idle = idle0() if anchor else None
    for t in times:
        ch = track(t)
        if extra:
            extra(t, ch)
        if anchor:
            w0 = float(smoothstep(ANCHOR, 0.0, t)) if anchor in (True, "start") else 0.0
            w1 = float(smoothstep(d - ANCHOR, d, t)) if anchor in (True, "end") else 0.0
            for w, off in ((w0, None), (w1, end_offset)):
                if w <= 0:
                    continue
                target = dict(idle)
                for k, v in (off or {}).items():
                    target[k] = target.get(k, 0.0) + v
                for k in set(ch) | set(target):
                    dflt = DEFAULTS.get(k, 0.0)
                    ch[k] = ch.get(k, dflt) * (1 - w) + target.get(k, dflt) * w
        out.append(build_pose(ch))
    if track.wrap:
        out.append(out[0].copy())
    return out


ANCHOR = 0.3
DEFAULTS = {"kneeout": 0.1, "elbowout": 0.25, "elbowback": 1.0, "legik": 1.0}     # build_pose's values for unset channels


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _breath(t, ch, period=1.5, amount=1.0):
    """A hunter's quick, heavy breathing: the ribcage heaves, the shoulders lift, the head nods with it."""
    s = np.sin(2 * np.pi * t / period)
    add(ch, **{"chest__x": -1.6 * amount * s, "spine1__x": 0.8 * amount * s, "neck__x": 1.0 * amount * s,
               "head__x": -0.6 * amount * s, "clavL__z": 1.5 * amount * s, "clavR__z": -1.5 * amount * s,
               "root__y": 0.0025 * amount * s})


def _dangle(t, ch, period, amount=1.0, phase=0.0):
    """The long arms hanging loose, swinging a beat behind the body."""
    w = 2 * np.pi * t / period + phase
    for s, ph in (("L", 0.0), ("R", 1.3)):
        add(ch, **{f"arm{s}__x": -2.5 * amount * np.sin(w + ph), f"forearm{s}__x": -3 * amount * np.sin(w + ph - .6),
                   f"hand{s}__x": -4 * amount * np.sin(w + ph - 1.2)})


def _env(t, pts):
    """Piecewise-linear envelope through (time, value) points."""
    return float(np.interp(t, [p[0] for p in pts], [p[1] for p in pts]))


def _pulse(t, at, width):
    """A quick smooth bump (0..1..0) centred on `at`, `width` seconds wide."""
    x = (t - at) / (width / 2)
    return float(np.where(np.abs(x) < 1, 0.5 + 0.5 * np.cos(np.pi * x), 0.0))


# ------------------------------------------------------------------------------------------- idles

STAND_D = 3.0
STAND_KEYS = [(0, {"neck.y": 0, "head.y": 0, "head.x": 0, "earL.x": 0}),
              (0.7, {"neck.y": 8, "head.y": 6, "head.x": 4}),
              (1.1, {"head.x": -3}),                                   # a sniff
              (1.5, {"neck.y": 0, "head.y": 0, "head.x": 0}),
              (1.8, {"earL.x": 0}), (1.9, {"earL.x": -25}), (2.05, {"earL.x": 0}),     # an ear flicks back
              (2.2, {"neck.y": -8, "head.y": -6, "head.x": 4}),
              (2.6, {"head.x": -3}),
              (3.0, {"neck.y": 0, "head.y": 0, "head.x": 0})]


def _stand_extra(t, ch):
    _breath(t, ch)
    _dangle(t, ch, STAND_D)
    w = 2 * np.pi * t / STAND_D
    add(ch, **{"tail1__y": 6 * np.sin(2 * w), "tail2__y": 8 * np.sin(2 * w - .7),
               "tail3__y": 10 * np.sin(2 * w - 1.4), "rootr__z": 1.0 * np.sin(w)})


_IDLE0 = None


def idle0():
    """The channels of Idle1's first frame (every clip that starts or ends at rest starts or ends here)."""
    global _IDLE0
    if _IDLE0 is None:
        ch = Track(STAND_KEYS, STAND_D, True)(0.0)
        _stand_extra(0.0, ch)
        ch.update(DEFAULTS)
        _IDLE0 = ch
    return dict(_IDLE0)


def stand():
    """Idle1. Hunched and restless: breathing hard, scenting the air (the head swings low one way, then the
    other), an ear flicking, the arms hanging loose and swaying, the tail swishing."""
    return frames_of(Track(STAND_KEYS, STAND_D, True), _stand_extra), True


def hunt():
    """Idle2, the Hunt. He goes still, ears pricked forward; the head snaps round to stare at something off
    to his left; he holds the stare, breathing shallow, scenting it in three quick sniffs, an ear swivelling,
    the tail-tip twitching; a snort tosses the head; one more look, then he lets it go and relaxes."""
    d = 5.0
    keys = [(0, {}),
            (0.45, {"ears.x": 0, "neck.x": 0, "head.x": 0, "neck.y": 0, "head.y": 0, "spine.y": 0}),
            (0.6, {"ears.x": 42, "ears.z": -10, "neck.x": -6, "head.x": -6}),                  # still, ears prick
            (0.8, {"neck.y": 34, "head.y": 26, "spine.y": 6, "neck.x": -12, "head.x": -10,
                   "armL.x": -6, "armR.x": -4, "forearmL.x": -14, "forearmR.x": -10}),          # snaps round
            (0.9, {"neck.y": 31, "head.y": 23}),                                                 # settles
            (1.5, {"neck.y": 33, "head.y": 27, "head.x": -11}),                                 # leans into it
            (2.1, {"neck.y": 31, "head.y": 24, "head.x": -12}),
            (2.25, {"head.x": 6, "neck.x": -8, "chest.x": -5, "jaw": 4}),                       # snort: down
            (2.4, {"head.x": -18, "neck.x": -14, "chest.x": 2, "jaw": 0, "ears.x": 48}),        # ... and up
            (2.6, {"head.x": -10, "neck.x": -12, "chest.x": 0, "ears.x": 42}),
            (3.4, {"neck.y": 33, "head.y": 25}),
            (4.1, {"neck.y": 0, "head.y": 0, "spine.y": 0, "neck.x": 0, "head.x": 0, "ears.x": 0, "ears.z": 0,
                   "armL.x": 0, "armR.x": 0, "forearmL.x": 0, "forearmR.x": 0}),
            (5.0, {})]
    tr = Track(keys, d, True)

    def extra(t, ch):
        # shallow, held breathing while he listens (never still); heavy after the snort
        amt = _env(t, [(0, 1), (.45, 1), (.6, .35), (2.2, .35), (2.4, 1.6), (3.6, 1.2), (4.4, 1), (5, 1)])
        _breath(t, ch, period=1.25, amount=amt)
        _dangle(t, ch, 2.5, amount=_env(t, [(0, 1), (.5, 1), (.7, .25), (3.8, .25), (4.6, 1), (5, 1)]))
        # three sniffs: the muzzle dips, the lips part
        for at in (1.05, 1.4, 1.75):
            p = _pulse(t, at, 0.16)
            add(ch, **{"head__x": 5 * p, "neck__x": 1.5 * p, "jaw": 6 * p, "chest__x": -1.5 * p})
        # the right ear swivels back and returns; the tail-tip twitches
        add(ch, **{"earR__x": -30 * _pulse(t, 1.95, 0.3), "earR__y": 12 * _pulse(t, 1.95, 0.3),
                   "tail3__y": 14 * _pulse(t, 1.25, 0.25) - 10 * _pulse(t, 1.6, 0.25),
                   "tail2__y": 5 * _pulse(t, 1.3, 0.3)})
        sway = _env(t, [(0, 1), (.6, .15), (3.8, .15), (4.6, 1), (5, 1)])
        add(ch, **{"tail1__y": 4 * np.sin(2 * np.pi * t / 2.5) * sway,
                   "tail2__y": 6 * np.sin(2 * np.pi * t / 2.5 - .7) * sway})
    return frames_of(tr, extra, anchor=True), True


# ------------------------------------------------------------------------------------------- gaits

def _hoof_cycle(p, duty, stride, lift, fold, heel=25.0, out=0.0):
    """One hoof over a stride cycle at phase p (0..1): planted and sliding back from +stride/2 to -stride/2
    for `duty` of it (the game moves him forward over it; the slide is even), the cannon tipping forward
    (heel off, the hoof still flat) in the last fifth of the stance; then lifted and swung forward, folded,
    and set down. -> (x out, z, y, cannon pitch, pastern pitch)."""
    if p < duty:
        u = p / duty
        c = heel * float(smoothstep(0.78, 1.0, u))
        return 0.0, stride * (0.5 - u), 0.0, c, -c
    u = (p - duty) / (1 - duty)
    z = stride * (-0.5 + (1 - np.cos(np.pi * u)) / 2)
    # clears the floor at once; comes down eased (the last drop spread over the final frames, the fold
    # unwinding to a soft stop rather than snapping straight)
    late = u > 0.5
    y = lift * np.sin(np.pi * u) ** (1.3 if late else 0.5)
    fk = np.sin(np.pi * u) ** (2.2 if late else 1.2)
    c = heel * (1 - u) ** 2 + fold * fk
    tp = -heel * (1 - u) ** 2 + 0.4 * fold * fk
    # out and back in again by 80% of the swing: no sideways speed for the last frames before it lands
    return out * np.sin(np.pi * min(u / 0.8, 1.0)) ** 2, z, y, c, tp


WALK_PERIOD = 1.1
WALK_DUTY = 0.58          # double support about 2.5 frames at each change of feet
WALK_STRIDE = 0.56        # model units of slide per stance: 0.022 KMF units a frame, as the game's walks


def _walk_spec():
    """A long, loping, predatory walk on the hind legs: body low and pitched forward, the hips rolling over
    each planted hoof, the head nodding with every step like a horse's, the long arms swinging low and
    loose opposite to the legs, the tail swaying a beat behind the hips."""
    period = WALK_PERIOD
    tr = Track([(0, {})], period, True)

    def extra(t, ch):
        ph = (t / period) % 1.0
        for s, off in (("L", 0.0), ("R", 0.5 - 0.5 / 33)):    # both touch down on a whole frame (0 and 17)
            x, z, y, c, tp = _hoof_cycle((ph + off) % 1.0, WALK_DUTY, WALK_STRIDE, 0.085, 40, heel=22,
                                         out=0.012)
            add(ch, **{f"foot{s}__x": x * SIGN[s], f"foot{s}__z": z, f"foot{s}__y": y, f"foot{s}__p": c,
                       f"toe{s}__p": tp})
        w = 2 * np.pi * ph
        # the hips: lowest as weight comes onto a hoof, highest over it; rolling and swaying side to side
        add(ch, **{"root__y": -0.035 - 0.018 * np.cos(2 * w - 0.9), "root__x": 0.012 * np.sin(w - 0.3),
                   "root__z": 0.01 * np.cos(2 * w - 0.4), "rootr__x": 9.0 + 1.5 * np.cos(2 * w - 1.2),
                   "rootr__z": 3.5 * np.sin(w - 0.2), "rootr__y": 4.0 * np.sin(w),
                   "chest__y": -5.0 * np.sin(w - 0.2), "chest__z": -2.0 * np.sin(w - 0.4),
                   "neck__x": 6.0 + 5.0 * np.cos(2 * w - 1.4), "head__x": -4.0 * np.cos(2 * w - 0.8),
                   "neck__y": 3.0 * np.sin(w + .5), "head__y": -1.5 * np.sin(w + 1.0), "ears__x": 8,
                   "jaw": 1.0 + 1.0 * np.cos(2 * w)})
        # the tail sways a beat behind the hips and bounces with each step
        add(ch, **{"tail1__y": 8 * np.sin(w - 0.6), "tail2__y": 12 * np.sin(w - 1.2), "tail3__y": 16 * np.sin(w - 1.8),
                   "tail1__x": 3 * np.sin(2 * w - 1.0), "tail2__x": 4 * np.sin(2 * w - 1.6)})
        # arms: each swings back as its own hoof comes forward (opposite to the legs), held a little out
        # from the body so the stifle clears the hanging hand; loose at the elbow and wrist
        for s, sg in (("L", 1.0), ("R", -1.0)):
            a = sg * np.cos(w - 0.25)
            add(ch, **{f"arm{s}__x": 15 * a + 4, f"arm{s}__z": 4.5 * SIGN[s],
                       f"forearm{s}__x": 10 * sg * np.cos(w - 0.95) - 2,
                       f"hand{s}__x": 10 * sg * np.cos(w - 1.55) + 2})
    return tr, extra


def walk():
    tr, extra = _walk_spec()
    return frames_of(tr, extra), True


# the gallop: down on all fours, the arms for forelegs
RUN_PERIOD = 0.7          # 21 frames
RUN_DUTY = 0.27
SPINE0 = 4.0
RUN_STRIDE = 0.52         # 0.092 model units a frame: about 1.4 KMF units a cycle
RUN_BASE = {"rootr.x": 46.0, "root.y": -0.085, "root.z": -0.02, "neck.x": -66.0,
            "head.x": -20.0, "tail.x": 14.0, "armik": 1.0, "ears.x": -42.0, "kneeout": 0.1, "elbowout": -0.1, "elbowback": -0.4}
# the four limbs' touch-downs in a transverse gallop: hind, hind, fore, fore; the leading foreleg pushes off
# at frame 18 and he flies until the first hind lands at frame 1
RUN_PHASE = {"footL": 1 / 21, "footR": 3 / 21, "gripL": 8.5 / 21, "gripR": 12 / 21}   # (touch-downs on whole frames)


def _grip_rest(base, ahead=0.07, out_=0.035):
    """Where the fingertips go down for a pose on all fours: under and a little ahead of its shoulders."""
    P = build_pose({**base, "armik": 0.0})
    _, Pw = P.world()
    out = {}
    for s in "LR":
        sh = Pw[BONE_INDEX["arm" + s]]
        out[s] = np.array([sh[0] + out_ * SIGN[s], 0.0, sh[2] + ahead])
    return out


def _run_grip_rest():
    return _grip_rest(RUN_BASE, ahead=0.18, out_=-0.015)


def _running_spec():
    """Run: the fastest thing in the dungeon drops onto all fours and gallops like a horse: hind, hind, fore,
    fore, then a long moment in the air at the top of the arc. The back gathers (rounded, the hind hooves
    swinging up under the chest) and stretches out flat; the body rocks nose-up as the forelegs push off and
    nose-down as it lands on them; the neck pumps, the jaw works with the stride, the ears are pinned."""
    period = RUN_PERIOD
    tr = Track([(0, dict(RUN_BASE))], period, True)
    rest = _run_grip_rest()

    def extra(t, ch):
        ph = (t / period) % 1.0
        for limb, off in RUN_PHASE.items():
            p = (ph - off) % 1.0
            s = limb[-1]
            if limb.startswith("foot"):
                x, z, y, c, tp = _hoof_cycle(p, RUN_DUTY, RUN_STRIDE, 0.12, 60, heel=20, out=0.05)
                add(ch, **{f"foot{s}__x": x * SIGN[s], f"foot{s}__z": z - 0.06, f"foot{s}__y": y,
                           f"foot{s}__p": c, f"toe{s}__p": tp})
            else:
                x, z, y, c, tp = _hoof_cycle(p, RUN_DUTY, RUN_STRIDE, 0.2, 60, heel=15)
                g = rest[s]
                ch[f"grip{s}.x"], ch[f"grip{s}.y"], ch[f"grip{s}.z"] = g[0] - 0.045 * SIGN[s], y, g[2] + z
                add(ch, **{f"grip{s}__p": c})
        w = 2 * np.pi * ph
        add(ch, **{
            # the arc: highest in the air (frame 20), lowest under the stances
            "root__y": 0.045 * np.cos(2 * np.pi * (ph - 0.96)),
            # nose-down landing on the forelegs (frame 10), nose-up pushing off them (frame 20)
            "rootr__x": 13.0 * np.sin(2 * np.pi * (ph - 0.23)),
            # the back gathers (rounded) at frames 17-20 and stretches flat at 4-8
            "spine__x": SPINE0 + 14.0 * np.cos(2 * np.pi * (ph - 0.88)),
            "neck__x": -10.0 * np.sin(2 * np.pi * (ph - 0.23)) + 6.0 * np.sin(w + 0.9),
            "head__x": -4.0 * np.sin(w + 1.4),
            "jaw": 12.0 + 4.0 * np.sin(w - 0.5),
            "ears__x": 3.0 * np.sin(3 * w),
            "tail1__x": 8 * np.sin(w - 0.4), "tail2__x": 11 * np.sin(w - 1.0), "tail3__x": 14 * np.sin(w - 1.6),
            "tail2__y": 4 * np.sin(w), "rootr__z": 2.0 * np.sin(w)})
    return tr, extra


def running():
    tr, extra = _running_spec()
    return frames_of(tr, extra), True


# ------------------------------------------------------------------------------------------- fighting

# arms flung up and out: high overhead with the elbows wide, the claws raised to strike
HIGH = {"armL.x": -130, "armR.x": -126, "armL.y": 40, "armR.y": -40, "armL.z": -60, "armR.z": 60,
        "forearmL.x": 0, "forearmR.x": 0, "forearmL.z": -60, "forearmR.z": 60, "handL.x": 10, "handR.x": 10}


def attack():
    """Melee: a rearing two-handed blow. A small crouch, then he rears, ears pinned, mouth half open, both
    arms flung high overhead with the elbows wide; the hands come down first, driving his whole weight
    through them onto the enemy in front, the head following a beat later as the jaw snaps wide open;
    then he recovers, the mouth closing. (A turn-and-double-kick backwards would aim away from the target
    he must face.)"""
    d = 1.3
    rest = {"rootr.x": 0, "root.y": 0, "root.z": 0, "spine.x": 0, "neck.x": 0, "head.x": 0, "jaw": 0,
            "armL.x": 0, "armR.x": 0, "armL.y": 0, "armR.y": 0, "armL.z": 0, "armR.z": 0, "forearmL.x": 0,
            "forearmR.x": 0, "forearmL.z": 0, "forearmR.z": 0, "handL.x": 0, "handR.x": 0, "ears.x": 0, "tail.x": 0,
            "footR.y": 0, "footR.p": 0}
    keys = [(0, dict(rest)),
            (0.1, {"rootr.x": 6, "root.y": -.02, "neck.x": 6, "head.x": 4, "armL.x": 8, "armR.x": 8,
                   "ears.x": -20}),                                                                   # gathers
            (0.37, {"rootr.x": -28, "root.y": .035, "spine.x": -8, "neck.x": 12, "head.x": -4, "jaw": 12,
                    "ears.x": -45, "tail.x": -10, **HIGH}),                                           # rears
            (0.44, {"rootr.x": -32, "armL.x": -138, "armR.x": -134, "footR.y": .045, "footR.p": 25}),
            # the hands strike first (frame 17), the body and head land a beat behind (frames 18-20)
            (0.56, {"armL.x": -70, "armR.x": -64, "armL.y": 6, "armR.y": -6, "armL.z": 0, "armR.z": 0,
                    "forearmL.z": 0, "forearmR.z": 0, "forearmL.x": -6,
                    "forearmR.x": -4, "handL.x": -10, "handR.x": -10, "rootr.x": 12, "root.y": -.01,
                    "neck.x": 14, "head.x": 0}),
            (0.62, {"rootr.x": 26, "root.y": -.03, "root.z": .02, "spine.x": 10, "neck.x": 12, "head.x": 14,
                    "jaw": 44, "footR.y": 0, "footR.p": 0, "tail.x": 12, "armL.x": -66, "armR.x": -60}),  # SLAM
            (0.72, {"rootr.x": 22, "root.y": -.025, "jaw": 44, "armL.x": -64, "armR.x": -58}),
            (1.0, {"rootr.x": 6, "root.y": -.008, "spine.x": 2, "neck.x": 2, "head.x": 2, "jaw": 4,
                   "armL.x": -6, "armR.x": -6, "armL.y": 0, "armR.y": 0, "forearmL.x": 0, "forearmR.x": 0,
                   "handL.x": 0, "handR.x": 0, "ears.x": -10, "tail.x": 0, "root.z": 0}),
            (1.3, dict(rest))]
    tr = Track(keys, d, True)

    def extra(t, ch):
        _breath(t, ch, amount=_env(t, [(0, 1), (.2, 0), (1.0, 0), (1.3, 1)]))
    return frames_of(tr, extra, anchor=True), True


def angry():
    """Angry: a small crouch and a dip of the head, then he rears up tall on his hind legs, arms flung wide
    with the claws at shoulder height, head thrown back in a silent scream of a whinny, shaking it; then he
    stamps down hard and stands shaking with it."""
    d = 1.9
    rest = {"rootr.x": 0, "root.y": 0, "spine.x": 0, "neck.x": 0, "head.x": 0, "jaw": 0, "ears.x": 0,
            "armL.x": 0, "armR.x": 0, "armL.z": 0, "armR.z": 0, "forearmL.x": 0, "forearmR.x": 0,
            "forearmL.z": 0, "forearmR.z": 0, "handL.x": 0, "handR.x": 0, "tail.x": 0, "footR.y": 0, "footR.p": 0}
    wide = {"armL.z": 78, "armR.z": -78, "armL.x": -25, "armR.x": -25, "forearmL.z": 25, "forearmR.z": -25,
            "forearmL.x": -30, "forearmR.x": -30, "handL.x": -25, "handR.x": -25}
    keys = [(0, dict(rest)),
            (0.13, {"rootr.x": 8, "root.y": -.025, "neck.x": 10, "head.x": 8, "ears.x": -25,
                    "armL.x": 6, "armR.x": 6}),                                                       # anticipation
            (0.48, {"rootr.x": -36, "root.y": .05, "spine.x": -10, "neck.x": 4, "head.x": -32, "jaw": 46,
                    "ears.x": -45, "tail.x": -14, **wide}),                                            # rears
            (0.86, {"rootr.x": -38, "head.x": -34, "footR.y": .05, "footR.p": 30,
                    "armL.z": 84, "armR.z": -84}),
            (1.12, {"rootr.x": 14, "root.y": -.035, "spine.x": 6, "neck.x": 8, "head.x": 10, "jaw": 0,
                    "armL.x": -25, "armR.x": -25, "armL.z": 6, "armR.z": -6, "forearmL.x": -5, "forearmR.x": -5,
                    "forearmL.z": 0, "forearmR.z": 0, "handL.x": 0, "handR.x": 0, "footR.y": 0, "footR.p": 0,
                    "tail.x": 8}),                                                                     # STAMP
            (1.35, {"rootr.x": 6, "root.y": -.01, "spine.x": 2, "neck.x": 3, "head.x": 4, "armL.x": -6,
                    "armR.x": -6, "armL.z": 0, "armR.z": 0, "tail.x": 0, "ears.x": -15}),
            (1.9, dict(rest))]
    tr = Track(keys, d, False)

    def extra(t, ch):
        held = _env(t, [(0.4, 0), (0.5, 1), (0.95, 1), (1.05, 0)])     # the head shaking in the scream
        add(ch, **{"head__z": 9 * held * np.sin((t - 0.45) * 34), "neck__z": 4 * held * np.sin((t - 0.45) * 34 - 0.8),
                   "armL__x": -8 * held * np.sin((t - 0.45) * 17), "armR__x": 8 * held * np.sin((t - 0.45) * 17)})
        k = _env(t, [(1.12, 0), (1.16, 1), (1.6, 0)])          # shaking with rage after the stamp
        add(ch, **{"neck__z": 5 * k * np.sin(t * 50), "head__z": 6 * k * np.sin(t * 50 + 1)})
        _breath(t, ch, amount=_env(t, [(0, 1), (.3, 0), (1.2, 0), (1.5, 1.5), (1.9, 1)]))
    return frames_of(tr, extra, anchor=True), False


# ------------------------------------------------------------------------------------------- entrance

# Soles' pick (14 Nov): "Thrown out, lands hard". Times in seconds (30 fps); heights are the root's offset.
ENTER_D = 4.0             # 120 frames
D_LAND = 1.24             # model units forward where the root lands (every limb clear of the Portal's pit,
                          # which is the centre tile, +-0.5 KMF units: the rear of the legs is at 0.62+)
HOOF_BACK = 0.1           # the hooves strike this far behind their stance under him (a deep, sitting crouch)
ENTER_DIST = 1.41         # where he ends, after two steps forward (1.05 KMF units, like the game's own)
GRAV = 0.008 / 0.744 * 900     # 0.008 KMF units a frame per frame, in model units / s^2
LAND = {"rootr.x": 30, "root.y": -0.16, "spine.x": 10, "neck.x": -46, "head.x": -8}   # where the root strikes
Y_PEAK = LAND["root.y"] + 0.605                                     # 0.45 KMF units of fall to the floor
_FRAME = ENTER_D / (round(ENTER_D * 30) - 1)   # a non-looping clip's frames span the clip exactly
T_LAND = 27.6 * _FRAME                         # the root strikes just after frame 27 ...
T_PEAK = T_LAND - float(np.sqrt(2 * (Y_PEAK - LAND["root.y"]) / GRAV))
V0 = GRAV * T_PEAK                             # ... after one arc under constant gravity from the launch
Y_START = Y_PEAK - V0 * V0 / (2 * GRAV)        # (all of him below the floor, curled)
T_HAND = {"L": 27 * _FRAME, "R": 28 * _FRAME}  # the hands strike first, reaching down for the floor,
T_HOOF = {"L": 29 * _FRAME, "R": 30 * _FRAME}  # the hooves after the root, left before right
SQUASH = 0.16             # how far the root sinks after the strike (model units)
CROUCH = -0.07            # where it rests in the crouch (below the strike height)
T_STEP = {"L": (2.42, 2.64), "R": (2.66, 2.88)}   # the two steps forward as he rises


def _flight(t):
    """One ballistic arc under constant gravity, launch to strike (no easing at either end); an even
    forward speed all the way (the hang at the top is only vertical)."""
    tt = min(t, T_LAND)
    y = Y_START + V0 * tt - 0.5 * GRAV * tt * tt
    return y, D_LAND * min(t / T_LAND, 1.0)


def _impact(t):
    """The root after the strike, model units below the strike height: a decelerating squash over 4 frames,
    held 2 frames at the bottom, a small slow rebound (0.06), and down to the crouch."""
    u = (t - T_LAND) * 30
    if u <= 0:
        return 0.0
    if u <= 4:
        return -SQUASH * (1 - (1 - u / 4) ** 2.4)
    if u <= 6:
        return -SQUASH
    if u <= 10:
        return -SQUASH + 0.06 * float(smoothstep(6, 10, u))
    return -SQUASH + 0.06 + (CROUCH - (-SQUASH + 0.06)) * float(smoothstep(10, 18, u))


def _heavy_breaths(t, ch, t0=1.27, period=8 / 30, n=3):
    """Three heavy breaths after the landing: the chest and shoulders heave up, the head bobs, the jaw parts
    on each exhale."""
    u = (t - t0) / period
    if not 0 <= u <= n:
        return
    ph = 2 * np.pi * u
    inhale = 0.5 - 0.5 * np.cos(ph)               # 0 -> 1 -> 0 over a breath
    exhale = max(0.0, -np.sin(ph))
    add(ch, **{"root__y": 0.024 * inhale, "chest__x": -7 * inhale, "spine1__x": 3 * inhale, "clavL__z": 4 * inhale, "clavR__z": -4 * inhale,
               "neck__x": 4 * inhale - 2, "head__x": -3 * inhale, "jaw": 7 * exhale})


def entrance():
    """Entrance: thrown out of the Portal. Curled tight deep below the floor, he is spat up and forward
    through it in one fast arc under gravity, tumbling a little; he unfolds at the top and drops hard onto
    the floor clear of the pit: hands first, then the hooves, left before right; the body squashes deep
    through the knees and elbows, holds, comes up a little; the head snaps down and recoils, the spikes and
    tail whip; three heavy breaths, a beat crouched with eyes up; then he lifts his hands straight off the
    floor, rises with two steps forward, rears a little with a shaking, wide-jawed snarl, ears pinned, and
    settles into his idle."""
    d = ENTER_D
    D = ENTER_DIST
    tuck = {"spine.x": 30, "neck.x": 20, "head.x": 25, "ears.x": -45, "tail.x": 35, "jaw": 0,
            "armL.x": -35, "armR.x": -38, "armL.z": 10, "armR.z": -8, "forearmL.x": -95, "forearmR.x": -90,
            "handL.x": -40, "handR.x": -35, "armik": 0.0}
    reach = {"spine.x": 6, "neck.x": -14, "head.x": -6, "tail.x": -10, "ears.x": -35,
             "armL.x": -58, "armR.x": -50, "armL.z": 6, "armR.z": -8, "forearmL.x": -10, "forearmR.x": -16,
             "handL.x": -20, "handR.x": -14}
    crouch = {"spine.x": LAND["spine.x"], "neck.x": LAND["neck.x"], "head.x": LAND["head.x"], "ears.x": -40,
              "tail.x": 40, "armL.x": -40, "armR.x": -36, "armL.z": 0, "armR.z": 0, "forearmL.x": -20,
              "forearmR.x": -24, "handL.x": -10, "handR.x": -14}
    t_hold = T_LAND + 0.6
    keys = [(0.0, {**tuck, "rootr.x": 35, "kneeout": 0.5, "elbowout": 0.8}),
            (0.3, {**tuck, "rootr.x": 12}),
            (T_PEAK, {**reach, "rootr.x": -10}),                                         # unfolds at the top
            (T_LAND - 0.33, {"armik": 0.0}),
            (T_LAND - 0.15, {"rootr.x": 27, "kneeup": 0.6, "kneeout": 0.9}),   # (the stifles up and out, folding)
            (T_HAND["L"] - 0.08, {"armik": 1.0}),
            (T_LAND, {**crouch, "rootr.x": LAND["rootr.x"], "armik": 1.0, "kneeout": 0.7, "kneeup": 0.6}),  # STRIKES
            (T_LAND + 0.15, {"spine.x": 15, "rootr.x": 33}),
            (T_LAND + 0.4, {"spine.x": 11, "rootr.x": 30, "kneeout": 0.15, "kneeup": 0.35}),
            # a beat crouched, head low, eyes up at the camera
            (t_hold, {"spine.x": 12, "neck.x": -32, "head.x": -16, "rootr.x": 31, "ears.x": -30}),
            (2.0, {"neck.x": -34, "head.x": -18, "ears.x": -20}),
            (2.15, {"armik": 1.0, "kneeup": 0.35}),
            # rises, stepping forward twice
            (2.55, {"rootr.x": 6, "spine.x": 0, "neck.x": -6, "head.x": -6, "armik": 0.0, "kneeout": 0.3, "kneeup": 0.0,
                    "armL.x": -12,
                    "armR.x": -6, "forearmL.x": -8, "forearmR.x": -4, "handL.x": 0, "handR.x": 0, "ears.x": -10,
                    "tail.x": 22}),
            # rears a little: the snarl
            (3.05, {"rootr.x": -20, "kneeout": 0.1, "spine.x": -6, "neck.x": 8, "head.x": -14, "jaw": 44, "ears.x": -48,
                   "armL.x": -42, "armR.x": -32, "armL.z": 34, "armR.z": -20, "forearmL.x": -24,
                   "forearmR.x": -12, "handL.x": -25, "handR.x": -10, "tail.x": -8}),
            (3.25, {"jaw": 44, "head.x": -16}),
            (3.62, {"rootr.x": 2, "spine.x": 0, "neck.x": 0, "head.x": 0, "jaw": 0, "ears.x": 0, "armL.x": 0,
                   "armR.x": 0, "armL.z": 0, "armR.z": 0, "forearmL.x": 0, "forearmR.x": 0, "handL.x": 0,
                   "handR.x": 0, "tail.x": 0}),
            (d, {"rootr.x": 0})]
    tr = Track(keys, d, False)
    grip = _grip_rest({**LAND, "root.z": D_LAND}, ahead=0.15, out_=0.05)
    grip["L"] = grip["L"] + np.array([0.0, 0.0, 0.03])          # not a mirror: the left hand a little ahead,
    grip["R"] = grip["R"] + np.array([-0.025, 0.0, -0.01])      # the right one splayed a little wider
    crouch_y = LAND["root.y"] + CROUCH

    def extra(t, ch):
        y, z = _flight(t)
        rise = float(smoothstep(2.15, 2.65, t))
        snarl = _env(t, [(2.8, 0), (3.05, 0.03), (3.25, 0.03), (3.62, 0)])
        if t < T_LAND:
            ry = y
        else:
            ry = LAND["root.y"] + _impact(t)
        ch["root.y"] = ry * (1 - rise) + snarl
        ch["root.z"] = z + (D - D_LAND) * float(smoothstep(2.42, 2.88, t))
        # the hooves: drawn up in the air, reaching down to strike (left, then right), then two steps
        for s in "LR":
            th = T_HOOF[s]
            k = float(smoothstep(T_PEAK - 0.1, T_LAND, t))
            ry0 = y if t < T_LAND else LAND["root.y"] + _impact(t)
            w = float(smoothstep(T_PEAK - 0.15, th, t))
            off = 0.06 * (1 - w) - _impact(th) * w                         # drawn up; reaching the floor at th
            hy = (ry0 - LAND["root.y"] * k + off) if t < th else 0.0       # (under the floor at the start)
            # the last stretch: straight down onto the floor at an even 1.2 units/s, the leg reaching
            wl = float(smoothstep(th - 0.3, th - 0.12, t))
            hy = hy * (1 - wl) + 1.2 * max(th - t, 0.0) * wl
            land = float(smoothstep(th - 0.25, th, t))
            zx = (z - 0.02) * (1 - land) + (D_LAND - HOOF_BACK) * land
            ox = 0.03 * SIGN[s] * (1 - land)
            # the step forward (lifted clear, set down where he will stand)
            t0, t1 = T_STEP[s]
            u = float(np.clip((t - t0) / (t1 - t0), 0, 1))
            if u > 0:
                zx = D_LAND - HOOF_BACK + (D - D_LAND + HOOF_BACK) * float(smoothstep(0, 1, u))
                hy = 0.088 * np.sin(np.pi * u) ** 1.2
            ch[f"foot{s}.x"], ch[f"foot{s}.y"], ch[f"foot{s}.z"] = ox, hy, zx
            fold = 45 * (1 - land) + 20 * np.sin(np.pi * u)
            ch[f"foot{s}.p"] = fold
            ch[f"toe{s}.p"] = 0.4 * fold
        # the hands: reaching down onto their marks (left first) and slammed flat, held; later lifted
        # straight up off the floor before the arms swing free
        for s in "LR":
            down = 1 - float(np.clip((T_HAND[s] - t) / 0.2, 0, 1)) ** 1.6
            up_ = float(smoothstep(2.0, 2.15, t)) * 0.14
            fwd = float(smoothstep(2.1, 2.3, t)) * 0.06
            g = grip[s] + np.array([0.0, 0.2 * (1 - down) + up_, fwd])
            ch[f"grip{s}.x"], ch[f"grip{s}.y"], ch[f"grip{s}.z"] = g
            ch[f"grip{s}.p"] = (-25.0 if s == "L" else -12.0) - 30 * float(smoothstep(2.0, 2.3, t))
        # the impact: the head snaps down and recoils; spikes (the neck) and tail whip, damped
        u = (t - T_LAND) * 30
        if u > 0:
            whip = float(np.interp(u, [0, 1.5, 2.5, 5.0, 7.5, 11], [0, 11, 11, -7, 2, 0]))
            add(ch, **{"neck__x": whip, "head__x": 0.5 * whip})
            e = np.exp(-u / 6)
            add(ch, **{"neck__z": 4 * e * np.sin(u * 1.2), "tail1__x": -12 * e * np.sin(u * 0.8 + 3.1),
                       "tail2__x": -16 * e * np.sin(u * 0.8 + 2.5), "tail3__x": -20 * e * np.sin(u * 0.8 + 1.9),
                       "ears__z": 10 * e * np.sin(u * 1.3)})
        if t < T_LAND:      # the tumble in the air, gone by the strike
            tm = 1 - float(smoothstep(T_LAND - 0.25, T_LAND, t))
            add(ch, **{"rootr__z": 6 * tm * np.sin(t * 7), "tail__y": 10 * tm * np.sin(t * 9)})
        # breathing: three heavy breaths, then hard, trembling while he holds the crouch; the snarl shakes
        _heavy_breaths(t, ch)
        _breath(t, ch, period=1.1, amount=_env(t, [(0, .6), (T_LAND, .8), (1.25, 0), (2.07, 0), (2.2, 1.6), (3.0, 1.0), (d, 1.0)]))
        tr_ = _env(t, [(1.9, 0), (2.0, 1), (2.15, 1), (2.25, 0)])
        sh = _env(t, [(2.95, 0), (3.05, 1), (3.25, 1), (3.4, 0)])
        q = 2 * np.pi * t / (4 * _FRAME)              # the shake: a 4-frame cycle, even steps
        add(ch, **{"spine1__z": 1.2 * tr_ * np.sin(t * 41), "head__z": 1.6 * tr_ * np.sin(t * 37 + 1)
                   + 7 * sh * np.sin(q), "neck__z": 3 * sh * np.sin(q), "neck__y": 5 * sh * np.sin(q), "head__y": 2.2 * sh * np.sin(q),
                   "jaw": 3.0 * sh * np.cos(q), "earL__x": -18 * _pulse(t, 1.95, .25),
                   "tail3__y": 10 * _pulse(t, 2.05, .3)})
        add(ch, **{"neck__y": 7 * _pulse(t, 3.65, .5), "head__y": 5 * _pulse(t, 3.7, .5),
                   "earR__x": -20 * _pulse(t, 3.6, .2), "tail2__y": 9 * _pulse(t, 3.7, .45),
                   "tail3__y": 12 * _pulse(t, 3.75, .45)})
    return frames_of(tr, extra, anchor="end", end_offset={"root.z": D, "footL.z": D, "footR.z": D}), False


CLIPS = {"stand": stand, "hunt": hunt, "walk": walk, "running": running, "attack": attack, "angry": angry,
         "entrance": entrance}


def build_clips(names=None):
    """-> {name: (list of Poses, wrap)}."""
    if len(CLIPS) == len(APPROVED):
        from scripts.sejanus_more import MORE       # stage 3: the rest of the slots
        CLIPS.update(MORE)
    return {n: CLIPS[n]() for n in (names or CLIPS)}


APPROVED = tuple(CLIPS)
