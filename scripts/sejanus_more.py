"""Sejanus's full slot set (stage 3: every animation a DK2 creature plays, like the Dragon's 43 files), on the
same channels and machinery as sejanus_anims (whose 7 approved clips are untouched).

Character: the Hunter, a lean horse demon. He sleeps folded down on his knees like a horse in straw; eats by
tearing the chicken with hands and teeth; works (and trains) with fast, savage stamps; researches pacing
impatiently over the page; rears in a whinny-roar when happy; prances high when he dances; dies rearing and
collapsing onto his side like a horse.

Extra channels (sejanus_anims.build_pose): legik 0 frees the hind legs (hipL.x/y/z, kneeL.x, hockL.x,
pastL.x: local rotations in degrees), ground 1 lets the body down onto the floor (lowest point at 0).

Start / End files: the loop's own motion eased in from Idle1's first frame (Start, ending on the loop's
frame 0) or out to it (End, starting on the loop's frame 0); a gait's loop clock is slowed to a stop
across the transition, so the planted hooves slide with the creature's speed as it gathers or loses it.

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_anims import Track, add
from scripts import sejanus_anims as A
from scripts.sejanus_model import JAW_CHIN, head_point
from scripts.sejanus_anims import (DEFAULTS, FPS, SIGN, _breath, _dangle, _env, _hoof_cycle, _pulse, build_pose,
                                   frames_of, idle0, smoothstep)


_JAW_TIP = tuple(head_point(*JAW_CHIN))       # the chin (bind): the front of his mouth

# ------------------------------------------------------------------------------------------- helpers

def spec_ch(spec, t):
    tr, extra = spec
    ch = tr(t)
    if extra:
        extra(t, ch)
    return ch


def mix(a, b, w):
    out = {}
    for k in set(a) | set(b):
        d = DEFAULTS.get(k, 0.0)
        out[k] = a.get(k, d) * (1 - w) + b.get(k, d) * w
    return out


def _ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def _n(seconds):
    return int(round(seconds * FPS))


def transition(spec, seconds, end, slow=True, extra=None):
    """Idle1 frame 0 -> the loop `spec` (Start, ending on the loop's frame 0), or the loop -> Idle1 frame 0
    (End, starting on it). With `slow`, the loop's clock runs down to a stop toward the idle end (a gait
    gathering or losing speed); otherwise the loop plays on at its own pace while it is eased in or out."""
    n = _n(seconds) + 1
    d = (n - 1) / FPS
    idle = idle0()
    out = []
    for i in range(n):
        t = i / FPS
        x = t / d                                   # 0..1 through the transition
        e = _ease(1 - x) if end else _ease(x)       # the loop's weight
        if slow:
            # clock speed: 1 at the loop end, 0 at the idle end (integrated: the time on the loop's clock)
            tau = (d * (x - x * x / 2)) if end else (-d * (1 - x) ** 2 / 2)
        else:               # (one frame past the loop's frame 0: the motion runs on through the switch)
            tau = t + 1 / FPS if end else t - d - 1 / FPS
        loop = spec_ch(spec, tau)
        ch = mix(idle, loop, e)
        for k in loop:               # IK marks: the loop's own (their weight is armik's, mixed above)
            if k.startswith("grip") and k not in idle:
                ch[k] = loop[k]
        # the elbows turning from back (hanging) to forward (forelegs): out round the side, never through
        # a dead hint (the IK would flip the arm)
        if idle.get("elbowback", 1.0) * loop.get("elbowback", 1.0) < 0:
            ch["elbowout"] = ch.get("elbowout", 0.25) + 3.2 * e * (1 - e)
            for s_ in "LR":           # and the arms held out a little, clear of the thighs
                ch[f"arm{s_}.z"] = ch.get(f"arm{s_}.z", 0.0) + 60 * e * (1 - e) * SIGN[s_]
                ch[f"grip{s_}.x"] = ch.get(f"grip{s_}.x", 0.0) + 0.25 * e * (1 - e) * SIGN[s_]
        if extra:
            extra(t, ch, e)
        out.append(build_pose(ch))
    return out, False


def _anchored(spec, wrap=True, anchor=True):
    tr, extra = spec
    return frames_of(tr, extra, anchor=anchor), wrap


def _even(t, period_frames, phase=0.0):
    """A tremble sampled evenly on whole frames (a cycle of `period_frames` frames)."""
    return np.sin(2 * np.pi * t * FPS / period_frames + phase)


REST = {"rootr.x": 0, "rootr.y": 0, "rootr.z": 0, "root.x": 0, "root.y": 0, "root.z": 0, "spine.x": 0,
        "spine.y": 0, "neck.x": 0, "neck.y": 0, "neck.z": 0, "head.x": 0, "head.y": 0, "head.z": 0, "jaw": 0,
        "ears.x": 0, "ears.z": 0, "tail.x": 0, "tail.y": 0, "armL.x": 0, "armR.x": 0, "armL.y": 0, "armR.y": 0,
        "armL.z": 0, "armR.z": 0, "forearmL.x": 0, "forearmR.x": 0, "forearmL.z": 0, "forearmR.z": 0,
        "handL.x": 0, "handR.x": 0, "footL.y": 0, "footR.y": 0, "footL.z": 0, "footR.z": 0, "footL.p": 0,
        "footR.p": 0, "toeL.p": 0, "toeR.p": 0}


def sym(**kw):
    """{'arm.x': v} -> armL.x and armR.x; 'arm.z' / '.y' mirrored (R negated)."""
    out = {}
    for k, v in kw.items():
        name, ax = k.split("__")
        for s in "LR":
            out[f"{name}{s}.{ax}"] = v * (SIGN[s] if ax in ("y", "z") else 1.0)
    return out


# ------------------------------------------------------------------------------------------- gaits

# Gait Start / End: every limb either planted (sliding back with the ground, whose speed gathers from 0 at the
# idle end to the loop's own at the handoff) or lifted clear in a real step; from its last lift-off before
# the handoff each limb follows the loop itself. The Start ends one frame before the loop's frame 0 and the
# End begins one frame after it, so the motion runs straight on through the switch. An End is a Start
# solved on the loop played backwards, then reversed.

STEP_LIFT = 0.09          # model units (5.6 cm in the game's units): every repositioning step clears this


def _herm(p0, m0, p1, m1, u):
    u2, u3 = u * u, u * u * u
    return (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1


class _Limb:
    def __init__(self, name, grip=False):
        self.name, self.grip = name, grip
        s = name[-1]
        self.keys = ([f"grip{s}.x", f"grip{s}.y", f"grip{s}.z"] if grip else
                     [f"foot{s}.x", f"foot{s}.y", f"foot{s}.z"])
        self.pk = [f"grip{s}.p"] if grip else [f"foot{s}.p", f"toe{s}.p"]

    def get(self, ch):
        return np.array([ch.get(k, 0.0) for k in self.keys]), np.array([ch.get(k, 0.0) for k in self.pk])


def _solve(C, limbs, idle_state, T, k=1.0, lift=STEP_LIFT, fold=30.0, late=False):
    """C(tau): the loop's channels at loop time tau (handoff at tau 0, time t = T + tau). idle_state: limb
    name -> (position, pitches, planted?) at t = 0. -> function t -> {channel: value} for the limbs."""
    dt = 1 / 240
    # the ground: the speed of a planted limb in the loop (dz/dtau), and its direction
    v, d = None, None
    for L in limbs:
        p0, _ = L.get(C(0.0))
        p1, _ = L.get(C(dt))
        if abs(p0[1]) < 1e-6 and abs(p1[1]) < 1e-6 and abs(p1[2] - p0[2]) > 1e-6:
            v = abs(p1[2] - p0[2]) / dt
            d = np.sign(p1[2] - p0[2])
            break
    if v is None:                     # no limb planted at the handoff: look a little earlier
        for back in np.arange(dt, 0.5, dt):
            for L in limbs:
                p0, _ = L.get(C(-back))
                p1, _ = L.get(C(-back + dt))
                if abs(p0[1]) < 1e-6 and abs(p1[1]) < 1e-6 and abs(p1[2] - p0[2]) > 1e-6:
                    v, d = abs(p1[2] - p0[2]) / dt, np.sign(p1[2] - p0[2])
                    break
            if v is not None:
                break

    def S(a, b):                      # ground slid between times a and b (g = v (t/T)^k)
        return v * T / (k + 1) * ((b / T) ** (k + 1) - (a / T) ** (k + 1))

    def g(t):
        return v * (t / T) ** k

    def planted(tau):
        return all(abs(L.get(C(tau))[0][1]) < 1e-6 for L in [lim]) if False else None

    plans = {}
    for L in limbs:
        y_at = lambda tau: L.get(C(tau))[0][1]
        segs = []                   # (t0, t1, kind, data)
        pos0, pit0, on0 = idle_state[L.name]
        posT, pitT = L.get(C(0.0))
        if abs(posT[1]) < 1e-6:     # planted at the handoff: find where that stance began in the loop
            follow, t_off = T, None
            z_end = posT[2]
            t_stance_end = T
        else:                       # in the air: find its lift-off
            tau = 0.0
            while tau > -2.0 and abs(y_at(tau)) > 1e-6:
                tau -= dt
            t_off = T + tau
            if t_off < 0.18:        # the loop's swing is longer than we have: a swing of our own into it
                t_off = None
            follow = t_off if t_off is not None else T
            offp, offq = L.get(C(tau)) if t_off is not None else (None, None)
            z_end = offp[2] if t_off is not None else None
            t_stance_end = t_off
        if t_stance_end is not None:
            # planted up to t_stance_end at z_end; where would it have had to stand at the start?
            if on0:
                z0 = z_end - d * S(0.0, t_stance_end)
                err = pos0[2] - z0
                if abs(err) < 0.012:
                    segs.append((0.0, t_stance_end, "plant", (z_end, err, pos0[0])))
                else:               # a step to the right spot (first, or for an End last of all)
                    D = min(0.32, 0.6 * t_stance_end)
                    a, b = ((t_stance_end - 0.12 - D, t_stance_end - 0.12) if late and t_stance_end > D + 0.2
                            else (0.03, 0.03 + D))
                    z_land = z_end - d * S(b, t_stance_end)
                    z_lift = pos0[2] + d * S(0.0, a)
                    segs.append((0.0, a, "plant", (z_lift, 0.0, pos0[0], a)))
                    segs.append((a, b, "swing", (np.array([pos0[0], 0.0, z_lift]), pit0, d * g(a),
                                                  np.array([posT[0] if abs(posT[1]) < 1e-6 else offp[0], 0.0,
                                                            z_land]), None, d * g(b), lift)))
                    segs.append((b, t_stance_end, "plant", (z_end, 0.0, None)))
            else:                   # starts in the air (a hand): reach down onto its mark, then planted
                span = abs(z_end - (L.get(C(-0.0))[0][2] if False else z_end))
                # its landing mark: where the loop puts that limb down for this stance
                tau = (t_stance_end - T) - dt
                while tau > -2.0 and abs(y_at(tau)) < 1e-6:
                    tau -= dt
                land_p = L.get(C(tau + dt))[0]
                need = abs(land_p[2] - z_end)
                lo, hi = 0.0, t_stance_end
                for _ in range(40):                     # when it must land to slide that far
                    mid = (lo + hi) / 2
                    if S(mid, t_stance_end) > need:
                        lo = mid
                    else:
                        hi = mid
                t_land = max(lo, 0.3)          # (a reach of at least 9 frames)
                z_land = z_end - d * S(t_land, t_stance_end)
                segs.append((0.0, t_land, "swing", (pos0, pit0, 0.0, np.array([land_p[0], 0.0, z_land]),
                                                     None, d * g(t_land), lift * 1.6)))
                segs.append((t_land, t_stance_end, "plant", (z_end, 0.0, land_p[0])))
            if t_off is not None:
                segs.append((t_off, T, "follow", None))
        else:                       # a swing of our own, from planted, straight into the loop's mid-swing
            a = 0.08
            z_lift = pos0[2] + d * S(0.0, a)
            pT1, _ = L.get(C(dt))
            vel = (pT1 - posT) / dt
            segs.append((0.0, a, "plant", (z_lift, 0.0, pos0[0], a)))
            segs.append((a, T, "swing", (np.array([pos0[0], 0.0, z_lift]), pit0, d * g(a), posT, pitT,
                                          vel, lift)))
        plans[L.name] = segs

    def at(t):
        out = {}
        for L in limbs:
            segs = plans[L.name]
            seg = next((sg for sg in segs if sg[0] - 1e-9 <= t <= sg[1] + 1e-9), segs[-1])
            t0, t1, kind, data = seg
            if kind == "follow":
                p, q = L.get(C(t - T))
                if L.grip:              # a hand's lift-off spread over 3 frames (the gallop's own is sharper)
                    p = p.copy()
                    p[1] *= _ease((t - t0) / min(0.15, 0.7 * (T - t0)))
            elif kind == "plant":
                z_end, err = data[0], data[1]
                tend = data[3] if len(data) > 3 else t1
                z = z_end - d * S(t, tend) if len(data) <= 3 else z_end - d * S(t, tend)
                if err:
                    z += err * (S(t, tend) / max(S(0.0, tend), 1e-9))
                x = data[2] if data[2] is not None else L.get(C(0.0))[0][0]
                p = np.array([x, 0.0, z])
                base = np.zeros(len(L.pk)) + (-25.0 if L.grip else 0.0)
                pl, ql = L.get(C(t - T))           # near the handoff: the loop's own heel-lift
                w = _ease(1 - (T - t) / 0.2) if abs(pl[1]) < 1e-6 else 0.0
                q = base * (1 - w) + ql * w
            else:
                pa, qa, va, pb, qb, vb, h = data
                D = t1 - t0
                u = (t - t0) / D
                vb3 = vb if np.ndim(vb) else np.array([0.0, 0.0, vb])
                p = np.array([_herm(pa[0], 0.0, pb[0], vb3[0] * D, u),
                              _herm(pa[1], 0.0, pb[1], vb3[1] * D, u) + h * np.sin(np.pi * u) ** 2,
                              _herm(pa[2], va * D, pb[2], vb3[2] * D, u)])
                if L.grip:              # a hand goes up and forward clear of the legs, never across them
                    p[2] += 0.12 * np.sin(np.pi * u) ** 2
                qa_ = qa if qa is not None else np.zeros(len(L.pk))
                qb_ = qb if qb is not None else np.zeros(len(L.pk)) + (-25.0 if L.grip else 0.0)
                f_ = np.array([fold, 0.4 * fold][:len(L.pk)]) if not L.grip else np.array([fold])
                q = qa_ + (qb_ - qa_) * _ease(u) + f_ * np.sin(np.pi * u) ** 2
                if t1 >= T - 1e-9:      # a swing into the handoff: melts into the loop's own path at the end
                    w = _ease((t - (T - 0.12)) / 0.12)
                    if w > 0:
                        pl, ql = L.get(C(t - T))
                        p = p * (1 - w) + pl * w
                        q = q * (1 - w) + ql * w
            for kk, val in zip(L.keys, p):
                out[kk] = float(val)
            for kk, val in zip(L.pk, q):
                out[kk] = float(val)
        return out
    at.v, at.d, at.S, at.plans = v, d, S, plans
    return at


def gait_transition(spec, T, end, grips=False, k=1.0, body_span=1.0, extra=None, lift=STEP_LIFT):
    """Idle1 frame 0 -> the gait `spec` (Start) or back (End), with real steps (see above). T None: as long
    as the hoof planted at the handoff takes to slide from (or to) its idle spot as the ground gathers speed,
    so that hoof stays planted throughout."""
    h = 1 / FPS
    if end:
        C = lambda tau: spec_ch(spec, -tau + h)
    else:
        C = lambda tau: spec_ch(spec, tau - h)
    if T is None:
        c0, c1 = C(0.0), C(1 / 240)
        for sd in "LR":
            if abs(c0[f"foot{sd}.y"]) < 1e-6 and abs(c1[f"foot{sd}.y"]) < 1e-6:
                z = abs(c0[f"foot{sd}.z"])
                v = abs(c1[f"foot{sd}.z"] - c0[f"foot{sd}.z"]) * 240
                n = max(_n(2 * z / v), 8) + 1
                T = (n - 1) / FPS
                k = v * T / z - 1
                break
    n = _n(T) + 1
    T = (n - 1) / FPS
    idle = idle0()
    limbs = [_Limb("footL"), _Limb("footR")]
    idle_state = {"footL": (np.zeros(3), np.zeros(2), True), "footR": (np.zeros(3), np.zeros(2), True)}
    if grips:
        P0 = build_pose(idle)
        for sd in "LR":
            limbs.append(_Limb("grip" + sd, grip=True))
            tip = _tip_world(P0, sd)
            idle_state["grip" + sd] = (tip, np.array([-60.0]), False)
    sol = _solve(C, limbs, idle_state, T, k=k, lift=lift, late=end)
    frames = []
    land = {}
    if grips:
        for L in limbs[2:]:
            pl = [sg for sg in sol.plans[L.name] if sg[2] == "plant"]
            land[L.name] = pl[0][0] if pl else T
    for i in range(n):
        t = i / FPS
        tau_b = -sol.S(t, T) / sol.v                  # the body's clock runs with the ground
        e = _ease(min(1.0, t / (body_span * T)))
        loop = C(tau_b)
        ch = mix(idle, loop, e)
        ch.update(sol(t))
        if grips and end:           # pulling up, the hands go down a little further ahead, clear of the hooves
            te = T - t
            for sd in "LR":
                k_ = _ease(te / 0.25) * _ease((T - te) / 0.25)
                ch[f"grip{sd}.x"] -= 0.03 * SIGN[sd] * k_
                ch[f"grip{sd}.z"] += 0.09 * k_
        if grips:
            first = min(land.values())
            ch["armik"] = _ease(np.clip((t - 0.06) / max(first - 0.1, 0.05), 0, 1))
            if idle.get("elbowback", 1.0) * loop.get("elbowback", 1.0) < 0:
                # the elbow hint mixes from back to forward: carried out round the side, never through a
                # dead hint (the IK would flip the arm)
                w = ch["armik"]
                ch["elbowout"] = ch.get("elbowout", 0.25) + 3.2 * max(w * (1 - w), e * (1 - e))
        if extra:
            extra(T - t if end else t, ch, e)
        frames.append(ch)
    if end:
        frames = frames[::-1]
    return [build_pose(c) for c in frames], False


def _tip_world(P, sd):
    Rw, Pw = P.world()
    b = A.BONE_INDEX["hand" + sd]
    return Pw[b] + Rw[b] @ (A.TIP[sd] - A.HEAD[b])


def walk_start():
    return gait_transition(A._walk_spec(), None, False)


def walk_end():
    return gait_transition(A._walk_spec(), None, True)


def run_start():
    """Drops forward onto all fours: the hooves drive back, the hands reach down and land, push off, and he
    is away in the gallop."""
    return gait_transition(A._running_spec(), 0.9, False, grips=True, body_span=0.55)


def run_end():
    """Pulls up out of the gallop: the hands land and push him upright, the hooves skid to a stop."""
    return gait_transition(A._running_spec(), 0.9, True, grips=True, body_span=0.55)


WB_PERIOD = 1.3
WB_DUTY = 0.62
WB_STRIDE = -0.36          # sliding forward under him: he backs away


def _walkback_spec():
    """Backing away, facing the threat: low, wary, head down and ears pinned, the arms half raised."""
    tr = Track([(0, {"rootr.x": -4, "neck.x": 10, "head.x": 6, "ears.x": -30, "jaw": 6, "tail.x": 10,
                     "armL.x": -30, "armR.x": -26, "armL.z": 14, "armR.z": -14, "forearmL.x": -40,
                     "forearmR.x": -36, "handL.x": -10, "handR.x": -10})], WB_PERIOD, True)

    def extra(t, ch):
        ph = (t / WB_PERIOD) % 1.0
        for s, off in (("L", 0.0), ("R", 0.5)):
            x, z, y, c, tp = _hoof_cycle((ph + off) % 1.0, WB_DUTY, WB_STRIDE, 0.06, 30, heel=10, out=0.01)
            add(ch, **{f"foot{s}__x": x * SIGN[s], f"foot{s}__z": z, f"foot{s}__y": y, f"foot{s}__p": c,
                       f"toe{s}__p": tp})
        w = 2 * np.pi * ph
        add(ch, **{"root__y": -0.04 - 0.012 * np.cos(2 * w - 0.9), "rootr__z": 3 * np.sin(w),
                   "rootr__y": -3 * np.sin(w), "neck__y": 6 * np.sin(w + 1), "head__y": -4 * np.sin(w + 1.4),
                   "tail1__y": 6 * np.sin(w - .6), "tail2__y": 9 * np.sin(w - 1.2), "jaw": 4 * np.sin(2 * w)})
        _breath(t, ch, period=WB_PERIOD / 2, amount=0.8)
    return tr, extra


def walkback():
    return frames_of(*_walkback_spec()), True


def walkback_start():
    return gait_transition(_walkback_spec(), None, False)


def walkback_end():
    return gait_transition(_walkback_spec(), None, True)


# ------------------------------------------------------------------------------------------- moods

def magic():
    """The Hunter's Reveal: he rises tall, ears pricked, sweeps both open hands out and forward, the head
    raking slowly across the room in a wide stare (left to right), jaw parted; a hard snort; he settles."""
    d = 2.0
    cast = {"rootr.x": -12, "root.y": 0.02, "spine.x": -6, "neck.x": -10, "head.x": -6, "jaw": 14, "ears.x": 45,
            **sym(arm__x=-55, arm__z=62, forearm__x=-10, hand__x=35), "tail.x": 18}
    keys = [(0, {}), (0.15, {"rootr.x": 6, "root.y": -0.02, "neck.x": 8, "head.x": 6, "ears.x": 10,
                             **sym(arm__x=6, forearm__x=-20)}),
            (0.45, {**cast, "neck.y": 20, "head.y": 12}),
            (1.15, {"neck.y": -20, "head.y": -12}),
            (1.3, {"neck.y": -17, "head.x": 4, "jaw": 4}),                       # snort
            (1.42, {"head.x": -10, "jaw": 10}),
            (1.7, {**{k: 0 for k in cast}, "neck.y": 0, "head.y": 0}), (d, {})]
    tr = Track(keys, d, False)

    def extra(t, ch):
        g = _env(t, [(0.4, 0), (0.5, 1), (1.15, 1), (1.3, 0)])
        add(ch, **{"handL__x": 6 * g * _even(t, 7), "handR__x": 6 * g * _even(t, 7, 2),
                   "ears__z": 6 * g * _even(t, 9)})
        _breath(t, ch, amount=_env(t, [(0, 1), (0.3, .3), (1.3, .3), (1.6, 1.4), (d, 1)]))
    return frames_of(tr, extra, anchor=True), False


def happy():
    """A rearing whinny-roar: he gathers, rears high on the hind legs pawing the air with both arms out wide
    (clear of the thrown-back head), hopping on the left hoof as he whinnies, jaw wide, ears forward; comes
    down with a stamp, the arms swinging down, and a toss of the head."""
    d = 2.0
    rear = {"rootr.x": -32, "root.y": 0.05, "spine.x": -10, "neck.x": -6, "head.x": -18, "jaw": 46, "ears.x": 40,
            "tail.x": 30, **sym(arm__x=-32, arm__z=70, forearm__x=-22, hand__x=-10), "footR.y": 0.05,
            "footR.p": 30, "toeR.p": 12}
    keys = [(0, {}),
            (0.15, {"rootr.x": 8, "root.y": -0.03, "neck.x": 10, "head.x": 8, "ears.x": 10, **sym(arm__x=8)}),
            (0.5, rear), (1.05, {"rootr.x": -34, "head.x": -20}),
            (1.25, {"rootr.x": 10, "root.y": -0.03, "spine.x": 0, "neck.x": 6, "head.x": 10, "jaw": 8,
                    "tail.x": 12, "ears.x": 20, "footR.y": 0, "footR.p": 0, "toeR.p": 0,
                    **sym(arm__x=-40, arm__z=20, forearm__x=-40, hand__x=-10)}),                 # stamp down
            (1.4, {"head.x": -14, "neck.x": -6, "jaw": 20}),                                    # tosses the head
            (1.65, {"rootr.x": 0, "root.y": 0, "neck.x": 0, "head.x": 0, "jaw": 0, "tail.x": 0, "ears.x": 0,
                    **sym(arm__x=0, arm__z=0, forearm__x=0, hand__x=0)}),
            (d, {})]
    tr = Track(keys, d, False)

    def extra(t, ch):
        paw = _env(t, [(0.45, 0), (0.55, 1), (1.0, 1), (1.12, 0)])
        w = 2 * np.pi * (t - .5) / .4
        add(ch, **{"armL__x": 16 * paw * np.sin(w), "armR__x": -16 * paw * np.sin(w),
                   "forearmL__x": -12 * paw * (1 + np.sin(w - 1)), "forearmR__x": -12 * paw * (1 - np.sin(w - 1)),
                   "head__z": 5 * paw * _even(t, 4), "jaw": 4 * paw * _even(t, 4, 1)})
        # hops on the left hoof in the rear, the body bouncing with each whinny
        hop = _env(t, [(0.5, 0), (0.6, 1), (1.0, 1), (1.08, 0)])
        u = ((t - 0.55) / 0.2) % 1.0
        add(ch, **{"footL__y": 0.05 * hop * np.sin(np.pi * u) ** 2, "footL__p": 18 * hop * np.sin(np.pi * u) ** 2,
                   "root__y": 0.04 * hop * np.sin(np.pi * u) ** 2, "rootr__x": -4 * hop * np.sin(np.pi * u),
                   "tail1__y": 10 * hop * np.sin(2 * np.pi * u)})
        _breath(t, ch, amount=_env(t, [(0, 1), (.3, 0), (1.25, 0), (1.6, 1.5), (d, 1)]))
    return frames_of(tr, extra, anchor=True), False


DANCE_P = 2 / 3            # one prance per hoof each 20 frames


def dance():
    """High prancing (a piaffe): in place, each hind hoof lifted high and folded in turn, the body bouncing
    over them, the arms swinging high in time, the head tossing, the tail flagged up."""
    d = 4 * DANCE_P
    tr = Track([(0, {"rootr.x": -8, "neck.x": -8, "head.x": -4, "ears.x": 35, "tail.x": 30, "jaw": 6})], d, True)

    def extra(t, ch):
        ph = (t / DANCE_P) % 1.0
        for s, off in (("L", 0.0), ("R", 0.5)):
            p = (ph + off) % 1.0
            u = float(np.clip((p - 0.5) / 0.5, 0, 1))         # the swing: 10 frames of 20
            lift = np.sin(np.pi * u)
            add(ch, **{f"foot{s}__y": 0.16 * lift ** 1.2, f"foot{s}__p": 75 * lift, f"toe{s}__p": 30 * lift,
                       f"foot{s}__z": 0.03 * lift})
        w = 2 * np.pi * ph
        add(ch, **{"root__y": 0.03 * np.cos(2 * w), "rootr__z": 4 * np.sin(w), "spine__y": -5 * np.sin(w),
                   "neck__x": 6 * np.cos(2 * w + .6), "head__x": -8 * np.cos(2 * w + 1.2),
                   "neck__y": 10 * np.sin(w + .5), "tail1__y": 10 * np.sin(w - .5), "tail2__y": 14 * np.sin(w - 1),
                   "tail__x": 6 * np.cos(2 * w)})
        for s, sg in (("L", 1.0), ("R", -1.0)):
            a = sg * np.sin(w)
            add(ch, **{f"arm{s}__x": -60 + 40 * a, f"arm{s}__z": 20 * SIGN[s], f"forearm{s}__x": -50 - 30 * a,
                       f"hand{s}__x": -20 * a})
    return frames_of(tr, extra, anchor=True), True


def _recoil(sign):
    """Hit: from the front (sign 1) he is jolted back a step; from behind (-1) he is shoved forward a step.
    One hoof steps (lifted clear) with the shove and back again under him."""
    d = 0.8
    k = 1.0 if sign > 0 else 0.55          # thrown forward: less of the neck (the head stays up)
    shove = -0.07 * sign
    hit = {"rootr.x": -18 * sign, "root.z": shove, "spine.x": -9 * sign, "neck.x": -20 * sign * k,
           "head.x": -16 * sign * k - (8 if sign < 0 else 0), "jaw": 28, "ears.x": -45, "tail.x": 22 * sign,
           **sym(arm__x=-34 * sign, forearm__x=-34, arm__z=20)}
    keys = [(0, {}), (0.067, hit), (0.17, {"rootr.x": -13 * sign, "jaw": 30, "root.z": shove}),
            (0.42, {"root.z": shove}), (0.62, {k_: 0 for k_ in hit}), (d, {})]
    tr = Track(keys, d, False)
    sd = "R" if sign > 0 else "L"
    step = -0.16 * sign                    # 0.12 in the game's units

    def extra(t, ch):
        kk = _env(t, [(0.05, 0), (0.08, 1), (0.3, 0)])
        add(ch, **{"head__z": 7 * kk * _even(t, 4), "neck__z": 3 * kk * _even(t, 4, 1)})
        # the hoof steps with the shove (out at 0.05-0.25 s), and back under him (0.42-0.65 s)
        u1 = float(np.clip((t - 0.05) / 0.2, 0, 1))
        u2 = float(np.clip((t - 0.42) / 0.23, 0, 1))
        z = step * (_ease(u1) - _ease(u2))
        lift = STEP_LIFT * (1.5 * np.sin(np.pi * u1) ** 1.2 + np.sin(np.pi * u2) ** 1.2)
        fold = 25 * (np.sin(np.pi * u1) + np.sin(np.pi * u2))
        add(ch, **{f"foot{sd}__z": z, f"foot{sd}__y": lift, f"foot{sd}__p": fold, f"toe{sd}__p": 0.4 * fold})
    return frames_of(tr, extra, anchor=True), False


def recoil_hff():
    return _recoil(1.0)


def recoil_hfb():
    return _recoil(-1.0)


ELEC_D = 1.0


def _elec_ch(t):
    base = {"rootr.x": -14, "root.y": -0.03, "spine.x": -10, "neck.x": 6, "head.x": -30, "jaw": 40, "ears.x": -40,
            "tail.x": 35, **sym(arm__x=-40, arm__z=62, forearm__x=-8, hand__x=40), "kneeout": 0.4}
    ch = dict(base)
    k = 1.0
    add(ch, **{"rootr__z": 5 * k * _even(t, 3), "rootr__x": 4 * k * _even(t, 5, 1), "spine__y": 6 * _even(t, 4),
               "neck__z": 8 * _even(t, 3, 2), "head__z": 10 * _even(t, 3), "head__x": 6 * _even(t, 5),
               "jaw": 10 * _even(t, 2.5), "ears__z": 12 * _even(t, 3), "armL__x": 14 * _even(t, 3),
               "armR__x": 14 * _even(t, 3, 2), "armL__z": 10 * _even(t, 5), "armR__z": -10 * _even(t, 5, 2),
               "forearmL__x": -20 * _even(t, 3, 1), "forearmR__x": -20 * _even(t, 3, 3),
               "handL__x": 18 * _even(t, 2.5), "handR__x": 18 * _even(t, 2.5, 1), "tail1__y": 14 * _even(t, 3),
               "tail2__y": 18 * _even(t, 3, 1), "root__y": 0.015 * _even(t, 6),
               "footL__p": 6 * _even(t, 5), "footR__p": 6 * _even(t, 5, 2)})
    return ch


def electrocute():
    """Jolting in the current: rigid, arched back, arms flung out stiff with the fingers splayed, head thrown
    back, jaw locked open, every part juddering."""
    n = _n(ELEC_D)
    out = [build_pose(_elec_ch(i / FPS)) for i in range(n)]
    return out + [out[0].copy()], True


DRUNK_P = 1.6


def _drunk_spec():
    """Drunk: a weaving, lurching walk, the head lolling, arms swinging loose."""
    tr = Track([(0, {"neck.x": 14, "head.x": 16, "jaw": 12, "ears.x": -20, "tail.x": -6})], DRUNK_P, True)

    def extra(t, ch):
        ph = (t / DRUNK_P) % 1.0
        for s, off in (("L", 0.0), ("R", 0.5)):
            x, z, y, c, tp = _hoof_cycle((ph + off) % 1.0, 0.62, 0.32, 0.06, 30, heel=12, out=0.03)
            add(ch, **{f"foot{s}__x": x * SIGN[s], f"foot{s}__z": z, f"foot{s}__y": y, f"foot{s}__p": c,
                       f"toe{s}__p": tp})
        w = 2 * np.pi * ph
        add(ch, **{"rootr__z": 9 * np.sin(w), "rootr__x": 6 + 3 * np.sin(2 * w), "root__x": 0.025 * np.sin(w),
                   "root__y": -0.03 - 0.01 * np.cos(2 * w), "neck__z": -10 * np.sin(w - .6),
                   "head__z": -14 * np.sin(w - 1.1), "neck__y": 12 * np.sin(w + .3), "head__y": 8 * np.sin(w),
                   "tail__y": 14 * np.sin(w + 1), "jaw": 6 * np.sin(2 * w)})
        _dangle(t, ch, DRUNK_P / 2, amount=1.8)
        add(ch, **{"armL__z": 10, "armR__z": -10})
    return tr, extra


def drunk():
    return frames_of(*_drunk_spec()), True


def drunk_idle():
    """Standing drunk: swaying in a slow circle over planted hooves, head lolling, a hiccup."""
    d = 3.0
    tr = Track([(0, {"neck.x": 16, "head.x": 18, "jaw": 10, "ears.x": -25, "tail.x": -8, "rootr.x": 4})], d, True)

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"rootr__z": 7 * np.sin(w), "rootr__x": 4 * np.cos(w), "root__x": 0.03 * np.sin(w),
                   "root__z": 0.02 * np.cos(w), "spine__y": -4 * np.sin(w - .5), "neck__z": -12 * np.sin(w - .8),
                   "head__z": -14 * np.sin(w - 1.3), "neck__y": 10 * np.sin(w + .8), "jaw": 4 * np.sin(2 * w),
                   "tail__y": 10 * np.sin(w + 1.2), "kneeout": 0.2})
        h = _pulse(t, 1.9, 0.22)                    # hiccup
        add(ch, **{"chest__x": -6 * h, "neck__x": -8 * h, "head__x": -10 * h, "jaw": 12 * h, "root__y": 0.02 * h})
        _dangle(t, ch, d / 2, amount=1.6)
        add(ch, **{"armL__z": 9, "armR__z": -9})
        _breath(t, ch, period=1.5, amount=1.0)
    return frames_of(tr, extra), True


DRINK_D = 2.8
DRINK_TIMES = {"appear": (0.1, 0.35), "drink": (0.75, 1.95), "gone": (2.1, 2.4)}


def drink():
    """Drinking (the Tavern's bottle, sejanus_build): the bottle appears in his right hand; both hands reach
    it up and forward to his muzzle, the left steadying it; he tips his head back and the bottle up and
    swallows three times, the jaw opening on each, the bottle pushed higher each time; then down again and
    the bottle is gone."""
    d = DRINK_D
    hold = {**sym(arm__x=-30, arm__z=8, forearm__x=-70, hand__x=-10), "neck.x": 4, "head.x": 10}
    tip_back = {"head.x": -22, "neck.x": -14, "rootr.x": -8, "spine.x": -6}
    keys = [(0, {"armik": 0.0}), (0.4, {**hold, "armik": 0.0}), (0.75, {**tip_back, "armik": 1.0}),
            (1.15, {"head.x": -27, "neck.x": -17}), (1.55, {"head.x": -32, "neck.x": -20, "rootr.x": -10}),
            (1.9, {"armik": 1.0}),
            (2.15, {**hold, "armik": 0.0, "rootr.x": 0, "spine.x": 0}),
            (2.5, {k: 0 for k in hold} | {k: 0 for k in tip_back}), (d, {})]
    tr = Track(keys, d, False)
    jaw_tip = A.HEAD[A.BONE_INDEX["jaw"]] * 0 + np.array(_JAW_TIP)

    def extra(t, ch):
        gulp = 0.0
        for at in (0.95, 1.35, 1.75):
            g = _pulse(t, at, 0.22)
            gulp += g
            add(ch, **{"jaw": 18 * g, "neck__x": 3 * g, "chest__x": -3 * g})
        _breath(t, ch, amount=_env(t, [(0, 1), (.6, .3), (1.9, .3), (2.2, 1.3), (d, 1)]))
        # the bottle to the mouth: the right fingertips just ahead of and above the muzzle (the bottle's neck,
        # held there, at the lips), raised a little more each gulp; the left hand steadying it beneath
        P = build_pose({**ch, "armik": 0.0})
        Rw, Pw = P.world()
        b = A.BONE_INDEX["jaw"]
        mouth = Pw[b] + Rw[b] @ (jaw_tip - A.HEAD[b])
        lift = 0.012 * _env(t, [(0.9, 0), (1.1, 1), (1.5, 2), (1.9, 3), (2.0, 3)])
        r = mouth + np.array([-0.04, 0.09 + lift, 0.15])
        l_ = r + np.array([0.21, 0.0, 0.04])
        for sd, p in (("R", r), ("L", l_)):
            ch[f"grip{sd}.x"], ch[f"grip{sd}.y"], ch[f"grip{sd}.z"] = p
            ch[f"grip{sd}.p"] = -70.0 if sd == "R" else -40.0
        ch["elbowout"] = 0.6
        ch["elbowback"] = 0.6
    return frames_of(tr, extra, anchor=True), False


EAT_D = 3.6
EAT_GRAB = 0.5                            # the hands close on the chicken on the floor
EAT_BITES = (1.25, 1.9, 2.55)             # each bite: in, the jaws close, tear back and away


def eating():
    """Eating: he stoops and snatches the chicken up off the floor in both hands, lunges and bites into it,
    and tears the piece away, wrenching his head back and the hands down apart; chews; again; the last of
    it goes with the third bite (the chicken shrinks away with each, see sejanus_build); he straightens."""
    stoop = {"rootr.x": 42, "root.y": -0.09, "spine.x": 10, "neck.x": -14, "head.x": -6, "ears.x": -15,
             "kneeout": 0.3}
    hold = {**sym(arm__x=-30, arm__z=4, forearm__x=-60, hand__x=-10), "neck.x": 8, "head.x": 14, "spine.x": 8,
            "ears.x": -10, "rootr.x": 0, "root.y": 0, "kneeout": 0.1}
    keys = [(0, {"armik": 0.0}), (0.2, {"armik": 0.0}), (EAT_GRAB - 0.08, {**stoop, "armik": 1.0}),
            (EAT_GRAB + 0.08, {**stoop, "armik": 1.0}), (0.95, {**hold, "armik": 0.0})]
    for i, b in enumerate(EAT_BITES):
        keys += [(b - 0.12, {"neck.x": 18, "head.x": 26, "jaw": 34, **sym(arm__x=-34, forearm__x=-64)}),  # bite
                 (b, {"jaw": 4, "neck.x": 20, "head.x": 28}),
                 (b + 0.14, {"neck.x": -4, "head.x": -6, "neck.y": 14 * (-1) ** i, "jaw": 6,
                             **sym(arm__x=-22, forearm__x=-50, arm__z=14)}),           # tears it away
                 (b + 0.3, {"neck.x": 8, "head.x": 12, "neck.y": 0, **sym(arm__x=-30, forearm__x=-60, arm__z=4)})]
    keys += [(3.25, {k: 0 for k in hold} | {"neck.y": 0, "jaw": 0}), (EAT_D, {})]
    tr = Track(keys, EAT_D, False)
    grip = A._grip_rest({**stoop, "armik": 0.0}, ahead=0.22, out_=-0.06)
    spot = (grip["L"] + grip["R"]) / 2

    def extra(t, ch):
        # the hands down onto the chicken, and up with it (the IK marks rise as the arms go back to FK)
        rise = _ease(np.clip((t - EAT_GRAB - 0.08) / 0.35, 0, 1))
        ik = float(np.clip(ch.get("armik", 0.0), 0, 1))
        for sd in "LR":
            g = grip[sd]
            ch[f"grip{sd}.x"] = g[0] * 0.55 + spot[0] * 0.45
            ch[f"grip{sd}.y"] = 0.035 + 0.3 * rise + (0.25 * (1 - ik) ** 0.7 if t < EAT_GRAB else 0.0)
            ch[f"grip{sd}.z"] = g[2] - 0.05 * rise
            ch[f"grip{sd}.p"] = -30.0
        chew = sum(_env(t, [(b + 0.2, 0), (b + 0.3, 1), (b + 0.5, 1), (b + 0.6, 0)]) for b in EAT_BITES)
        add(ch, **{"jaw": 7 * chew * (0.5 + 0.5 * _even(t, 6)), "ears__x": 6 * chew * _even(t, 6, 1)})
        _breath(t, ch, amount=0.7)
    return frames_of(tr, extra, anchor=True), False


RES_D = 3.6


def _research_spec():
    """Research: an impatient, pacing read. Head down over the open book in his hands, he shifts from hoof to
    hoof, a hind hoof stamps three times, he flicks a page over with a snort, ears twitching."""
    read = {**sym(arm__x=-44, arm__z=4, forearm__x=-70, hand__x=-30), "neck.x": 16, "head.x": 22, "spine.x": 6,
            "ears.x": 8, "rootr.x": 4}
    keys = [(0, read), (1.2, {"neck.y": -10, "head.y": -6}), (2.0, {"neck.y": 8, "head.y": 6}),
            (2.3, {"handR.x": -30, "forearmR.x": -70}), (2.4, {"handR.x": 30, "forearmR.x": -60, "armR.z": 18}),
            (2.55, {"handR.x": -30, "forearmR.x": -70, "armR.z": -4}),           # flicks the page
            (3.0, {"neck.y": 0, "head.y": 0}), (RES_D, read)]
    tr = Track(keys, RES_D, True)

    def extra(t, ch):
        # shifting weight hoof to hoof: each hoof lifted and set back down where it stood
        for s, at in (("L", 0.3), ("R", 0.75), ("L", 1.25), ("R", 1.7)):
            u = float(np.clip((t - at) / 0.3, 0, 1))
            lift = np.sin(np.pi * u)
            add(ch, **{f"foot{s}__y": 0.05 * lift, f"foot{s}__p": 25 * lift, f"toe{s}__p": 10 * lift})
        sway = _env(t, [(0.3, 0), (0.45, 1), (0.75, -1), (1.25, 1), (1.7, -1), (2.05, 0)])
        add(ch, **{"root__x": -0.015 * sway, "rootr__z": 2.5 * sway})
        # the right hoof stamps impatiently: three quick taps
        for at in (2.65, 2.9, 3.15):
            u = float(np.clip((t - at) / 0.2, 0, 1))
            lift = np.sin(np.pi * u) ** 1.5
            add(ch, **{"footR__y": 0.045 * lift, "footR__p": 30 * lift, "toeR__p": 12 * lift})
        snort = _pulse(t, 2.45, 0.2)
        add(ch, **{"head__x": -8 * snort, "jaw": 8 * snort, "earL__x": -20 * _pulse(t, 1.0, .25),
                   "earR__x": -20 * _pulse(t, 3.3, .25), "tail3__y": 14 * _pulse(t, 2.9, 0.3)})
        _breath(t, ch, period=1.2, amount=1.4)
        w = 2 * np.pi * t / RES_D
        add(ch, **{"tail1__y": 6 * np.sin(w), "tail2__y": 9 * np.sin(w - .7), "ears__z": 4 * np.sin(2 * w)})
    return tr, extra


def research():
    return frames_of(*_research_spec()), True


def research_start():
    return transition(_research_spec(), 0.6, False, slow=False)


def research_end():
    return transition(_research_spec(), 0.6, True, slow=False)


BUILD_D = 1.2


def _building_spec():
    """Work (the Workshop; a Trainer's dummy): fast, savage blows: both fists come down hammering, then the
    right hind hoof is driven down onto it in a stamp, and again."""
    ready = {"rootr.x": 12, "root.y": -0.03, "spine.x": 8, "neck.x": 6, "head.x": 14, "ears.x": -40, "jaw": 8,
             **sym(arm__x=-40, arm__y=0, arm__z=10, forearm__x=-60, forearm__z=0), "tail.x": 10}
    up = {**A.HIGH, "rootr.x": -8, "spine.x": -6,
          "head.x": 0, "jaw": 20}
    slam = {**sym(arm__x=-42, arm__y=0, arm__z=8, forearm__x=-10, forearm__z=0, hand__x=-20), "rootr.x": 22, "spine.x": 12, "head.x": 20,
            "jaw": 36}
    keys = [(0, ready), (0.2, up), (0.36, slam), (0.48, ready),
            (0.6, {"rootr.x": 0, "head.x": 18}), (0.78, {"rootr.x": 18, "jaw": 30}), (0.95, ready), (BUILD_D, ready)]
    tr = Track(keys, BUILD_D, True)

    def extra(t, ch):
        # the stamp: the right hoof lifts high, is driven down ahead, held a beat, drawn back
        lift = _env(t, [(0.5, 0), (0.65, 0.17), (0.75, 0.0), (0.85, 0.0), (0.93, 0.05), (1.03, 0.03), (1.1, 0)])
        fwd = _env(t, [(0.5, 0), (0.65, 0.12), (0.75, 0.16), (0.88, 0.16), (1.06, 0)])
        fold = _env(t, [(0.5, 0), (0.62, 70), (0.75, 0), (0.85, 0), (0.95, 30), (1.1, 0)])
        add(ch, **{"footR__y": lift, "footR__z": fwd, "footR__p": fold, "toeR__p": 0.4 * fold})
        hit = _pulse(t, 0.4, 0.12) + _pulse(t, 0.78, 0.12)
        add(ch, **{"head__z": 5 * hit * _even(t, 3), "root__y": -0.015 * hit})
        _breath(t, ch, period=BUILD_D / 2, amount=1.4)
    return tr, extra


def building():
    return frames_of(*_building_spec()), True


def building_start():
    return transition(_building_spec(), 0.5, False, slow=False)


def building_end():
    return transition(_building_spec(), 0.5, True, slow=False)


def melee2():
    """A second blow (the Trainer's dummy, a second attack): savage stamping: he rears and drives the right
    hoof, then the left, down onto the enemy ahead."""
    d = 1.4
    rear = {"rootr.x": -24, "root.y": 0.03, "spine.x": -6, "neck.x": 10, "head.x": 4, "jaw": 30, "ears.x": -45,
            **sym(arm__x=-70, arm__z=26, forearm__x=-60)}
    keys = [(0, {}), (0.12, {"rootr.x": 6, "root.y": -0.02, "ears.x": -30}), (0.32, rear),
            (0.5, {"rootr.x": 14, "root.y": -0.02, "jaw": 40, "head.x": 14}), (0.62, {"rootr.x": -16, "jaw": 20}),
            (0.8, {"rootr.x": 16, "jaw": 44, "head.x": 16}),
            (1.1, {k: 0 for k in rear} | {"rootr.x": 0, "head.x": 0}), (d, {})]
    tr = Track(keys, d, False)

    def extra(t, ch):
        for s, t0 in (("R", 0.25), ("L", 0.55)):
            lift = _env(t, [(t0, 0), (t0 + 0.15, 0.18), (t0 + 0.24, 0), (t0 + 0.36, 0)])
            fwd = _env(t, [(t0, 0), (t0 + 0.15, 0.14), (t0 + 0.24, 0.17), (t0 + 0.38, 0.17), (t0 + 0.58, 0)])
            back = _env(t, [(t0 + 0.32, 0), (t0 + 0.42, 0.05), (t0 + 0.52, 0.03), (t0 + 0.6, 0)])
            fold = _env(t, [(t0, 0), (t0 + 0.12, 75), (t0 + 0.24, 0), (t0 + 0.36, 0), (t0 + 0.48, 30), (t0 + 0.6, 0)])
            add(ch, **{f"foot{s}__y": lift + back, f"foot{s}__z": fwd, f"foot{s}__p": fold, f"toe{s}__p": 0.4 * fold})
        _breath(t, ch, amount=_env(t, [(0, 1), (.2, 0), (1.1, 0), (d, 1)]))
    return frames_of(tr, extra, anchor=True), False


PRAY_D = 3.2


def _praying_spec():
    """Worship: arms flung up to the dark and the head thrown back, then a deep bow, the hands laid flat on
    the floor and the head down between them; up again."""
    up = {**A.HIGH, "rootr.x": -10, "spine.x": -8, "neck.x": -10,
          "head.x": -26, "jaw": 12, "ears.x": 20, "armik": 0.0, "tail.x": 20}
    bow = {"rootr.x": 40, "root.y": -0.07, "root.z": -0.03, "spine.x": 14, "neck.x": 6, "head.x": 24, "jaw": 0,
           "ears.x": -10, "armik": 1.0, "tail.x": 0, "kneeout": 0.25, **sym(arm__x=-70, arm__z=10, forearm__x=-20, hand__x=0)}
    keys = [(0, up), (0.5, up), (1.3, bow), (2.1, bow), (2.6, {**up, "root.y": 0, "root.z": 0, "kneeout": 0.1}),
            (PRAY_D, up)]
    tr = Track(keys, PRAY_D, True)
    grip = A._grip_rest({**bow, "armik": 0.0, "root.z": bow["root.z"] + PRAY_FWD}, ahead=0.2, out_=0.05)

    def extra(t, ch):
        # at the temple's edge: PRAY_FWD ahead of where he stood (PrayStart walks him there)
        add(ch, **{"root__z": PRAY_FWD, "footL__z": PRAY_FWD, "footR__z": PRAY_FWD})
        ik = float(np.clip(ch.get("armik", 0.0), 0, 1))
        for s in "LR":
            g = grip[s]
            ch[f"grip{s}.x"], ch[f"grip{s}.y"], ch[f"grip{s}.z"] = g[0], 0.25 * (1 - ik) ** 2, g[2]
            ch[f"grip{s}.p"] = -20.0
        _breath(t, ch, period=1.6, amount=1.2)
        bow = _env(t, [(1.2, 0), (1.35, 1), (2.05, 1), (2.2, 0)])
        add(ch, **{"head__x": 6 * bow * np.sin(2 * np.pi * (t - 1.3) / 0.4),
                   "neck__y": 6 * bow * np.sin(2 * np.pi * (t - 1.3) / 0.8)})
        add(ch, **{"jaw": 8 * _pulse(t, 0.25, 0.3) + 8 * _pulse(t, 2.95, 0.3)})
    return tr, extra


def praying():
    return frames_of(*_praying_spec()), True


PRAY_FWD = 0.279          # model units: 0.21 in the game's, the Bile Demon's place at the temple's edge


def praying_start():
    """From where he stands, two steps forward to the temple's edge (each hoof lifted clear), rising into the
    worship."""
    def extra(t, ch, e):
        for sd, (a, b) in (("L", (0.1, 0.45)), ("R", (0.42, 0.77))):
            u = float(np.clip((t - a) / (b - a), 0, 1))
            ch[f"foot{sd}.z"] = PRAY_FWD * _ease(u)
            ch[f"foot{sd}.y"] = STEP_LIFT * np.sin(np.pi * u) ** 1.2
            ch[f"foot{sd}.p"] = 25 * np.sin(np.pi * u)
            ch[f"toe{sd}.p"] = 10 * np.sin(np.pi * u)
        ch["root.z"] = ch.get("root.z", 0.0) - PRAY_FWD * e + PRAY_FWD * _ease(np.clip((t - 0.1) / 0.67, 0, 1))
    return transition(_praying_spec(), 1.2, False, slow=False, extra=extra)


def praying_end():
    """Back up out of the worship to stand, staying at the temple's edge."""
    def extra(t, ch, e):
        for k in ("root.z", "footL.z", "footR.z"):
            ch[k] = ch.get(k, 0.0) + PRAY_FWD * (1 - e)
    return transition(_praying_spec(), 1.2, True, slow=False, extra=extra)


# ------------------------------------------------------------------------------------------- lying down

# folded down on his knees like a horse in straw: the hind legs folded under, the arms folded under the
# chest, the neck curved round and the head down on the straw beside him
FOLD_LEGS = {"legik": 0.0, "ground": 1.0, **{f"hip{s}.x": -60 for s in "LR"}, **{f"knee{s}.x": 150 for s in "LR"},
             **{f"hock{s}.x": -100 for s in "LR"}, **{f"past{s}.x": -30 for s in "LR"},
             "hipL.z": 6, "hipR.z": -6}
# curled up asleep in the straw: legs folded under, the neck curled round to the left and the head down on
# the straw by his folded arms, the tail curled round the other way
SLEEP = {**FOLD_LEGS, "rootr.x": 45, "spine.x": 8, "neck.x": 6, "neck.y": 60, "neck.z": -20, "head.x": 0,
         "head.y": 45, "head.z": 18, "ears.x": -30, "ears.z": 14, "jaw": 0, "tail1.x": 22, "tail2.x": -10,
         "tail3.x": -8, "tail.y": -90, "tail.z": 70, **sym(arm__x=-40, arm__z=4, forearm__x=135, hand__x=40), "clavL.z": -4, "clavR.z": 4}


def sleeping():
    """Asleep, curled up on his folded knees in the straw, head down: slow deep breathing, an ear flicking,
    the tail-tip twitching, a dreaming twitch of a hand."""
    d = 4.0
    tr = Track([(0, SLEEP)], d, True)

    def extra(t, ch):
        s = np.sin(2 * np.pi * t / (4.0 / 3))
        add(ch, **{"chest__x": 7.0 * s, "spine1__x": -3.5 * s, "head__x": -3.0 * s, "neck__x": 3.0 * s,
                   "rootr__x": 1.2 * s,
                   "ears__z": 3 * np.sin(2 * np.pi * t / 4.0 + 1),
                   "tail1__y": 4 * np.sin(2 * np.pi * t / 4.0),
                   "jaw": 1.0 + 1.0 * s, "earR__x": -22 * _pulse(t, 1.1, 0.25), "earL__x": -12 * _pulse(t, 3.0, .2),
                   "tail3__y": 14 * _pulse(t, 2.2, 0.4), "handL__x": 14 * _pulse(t, 2.9, 0.3)})
    return frames_of(tr, extra), True


def _breath_lying(t, ch, period=0.9, amount=1.0):
    """Panting on the floor: the ribcage and neck only (the shoulders, on the floor, stay put)."""
    sn = np.sin(2 * np.pi * t / period)
    add(ch, **{"chest__x": -1.6 * amount * sn, "spine1__x": 0.8 * amount * sn, "neck__x": 1.0 * amount * sn,
               "head__x": -0.6 * amount * sn})


LIE_SIDE = {"legik": 0.0, "ground": 1.0, "rootr.z": -95, "rootr.x": 6, "hipL.x": -30, "kneeL.x": 30,
            "hockL.x": -20, "hipR.x": -50, "kneeR.x": 50, "hockR.x": -30, "hipR.z": -10, "hipL.z": -25,
            "armL.x": -75, "armL.z": 0, "forearmL.x": -15, "armR.x": -45, "armR.z": -15, "forearmR.x": -30,
            "neck.z": 10, "neck.x": -4, "head.z": 12, "tail.y": 20, "tail.x": 0}
FALL_BACK = -0.50            # model units: knocked 0.37 back (the game's own Fallbacks go 0.30-0.48)
FALL_D = 1.0


FALLEN_LEGS = {"hipL.x": -70, "kneeL.x": 80, "hipR.x": -85, "kneeR.x": 90, "hipL.z": -10, "hipR.z": -10}
LIE_Y = -0.286               # the root's drop that puts the lying body on the floor (its lowest point at 0)


def _falling_spec():
    keys = [(0, {"ground": 0.0, "floor": 1.0, "legik": 1.0, "root.z": 0.0, "root.y": 0.0}),
            (0.15, {"rootr.x": -14, "root.z": -0.12, "root.y": 0.02, "spine.x": -10, "neck.x": -16, "head.x": -20,
                    "jaw": 34, "ears.x": -45, **sym(arm__x=-45, arm__z=65, forearm__x=-20)}),
            # thrown off his hooves: they leave the floor with him, the legs go limp in the air, he drops
            (0.25, {"legik": 1.0, "footL.y": 0.08, "footR.y": 0.12, "footL.z": -0.2, "footR.z": -0.22,
                    "root.z": -0.24, "root.y": 0.0, "rootr.z": -20}),
            (0.42, {"legik": 0.0, "root.y": -0.16, "root.z": FALL_BACK, "rootr.z": -62, **FALLEN_LEGS}),   # legs up
            (0.55, {**LIE_SIDE, **FALLEN_LEGS, "root.z": FALL_BACK, "jaw": 30, "head.x": -10, "spine.x": -6,
                    "root.y": LIE_Y, "ground": 1.0}),                              # lands on his side
            (0.7, {"jaw": 20, "head.x": 0, "spine.x": 0, "ears.x": -30}),
            (0.82, {"neck.z": 22, "head.z": 18}),
            (FALL_D, {**LIE_SIDE, **FALLEN_LEGS, "root.z": FALL_BACK, "jaw": 14, "ears.x": -30})]
    tr = Track(keys, FALL_D, False)

    def extra(t, ch):
        _breath(t, ch, period=0.9, amount=_env(t, [(0, 1), (0.3, 0), (FALL_D, 0)]))
        _breath_lying(t, ch, period=0.9, amount=_env(t, [(0, 0), (0.6, 0), (0.75, 2.0), (FALL_D, 2.0)]))
    return tr, extra


def falling():
    """Knocked off his hooves: thrown back and sideways off his feet, arms flung up, crashing onto his side
    0.3 back from where he stood."""
    tr, extra = _falling_spec()
    return frames_of(tr, extra, anchor="start"), False


def _fallen_end():
    tr, extra = _falling_spec()
    return tr(FALL_D)


def waking_up():
    """Gets up: from where he fell, rolls from his side onto his folded knees, pushes up on his hands, gets
    the hooves under him and rises, stepping forward to stand where he was knocked from."""
    d = 2.3
    fallen = _fallen_end()
    belly = {**FOLD_LEGS, "rootr.z": 0, "rootr.x": 10, "neck.x": -10, "head.x": 10, "jaw": 6, "ears.x": -20,
             "neck.z": 0, "head.z": 0, "tail.y": 0, "tail.x": 0, **sym(arm__x=-20, arm__z=10, forearm__x=-30),
             "root.z": FALL_BACK + 0.04}
    under = FALL_BACK + 0.08
    keys = [(0, fallen), (0.15, {}),
            (0.6, belly),
            (0.85, {"armik": 1.0}),
            (1.15, {"legik": 1.0, "ground": 1.0, "rootr.x": 40, "root.y": -0.14, "spine.x": 10, "neck.x": -20,
                    "head.x": 6, "kneeout": 0.3, "root.z": under}),
            (1.45, {"armik": 0.0, "rootr.x": 6, "root.y": -0.02, "spine.x": 0, "neck.x": -4, "head.x": -6,
                    **sym(arm__x=-10, arm__z=16, forearm__x=-10), "kneeout": 0.1, "jaw": 10, "root.z": under}),
            (1.8, {**sym(arm__x=0, arm__z=0, forearm__x=0)}),
            (1.6, {"jaw": 0, "neck.x": 0, "head.x": 0, "rootr.x": 0, "root.y": 0}),
            (1.7, {"root.z": under * 0.5}), (1.95, {"root.z": 0.0}), (d, {})]
    tr = Track(keys, d, False)
    grip = A._grip_rest({"rootr.x": 40, "root.y": -0.14, "spine.x": 10, "root.z": under}, ahead=0.15, out_=0.05)

    def extra(t, ch):
        ik = float(np.clip(ch.get("armik", 0.0), 0, 1))
        for s in "LR":
            g = grip[s]
            ch[f"grip{s}.x"], ch[f"grip{s}.y"], ch[f"grip{s}.z"] = g[0], 0.25 * (1 - ik) ** 0.7, g[2]
            ch[f"grip{s}.p"] = -20.0
        # the hooves come down under him, then step forward (lifted clear) to where he stood
        for sd, (a, b) in (("L", (1.45, 1.7)), ("R", (1.68, 1.93))):
            u = float(np.clip((t - a) / (b - a), 0, 1))
            ch[f"foot{sd}.z"] = under * (1 - _ease(u))
            ch[f"foot{sd}.y"] = STEP_LIFT * np.sin(np.pi * u) ** 1.2
            ch[f"foot{sd}.p"] = 25 * np.sin(np.pi * u)
        add(ch, **{"head__z": 8 * _pulse(t, 1.6, 0.3) * _even(t, 4)})          # shakes it off
        # breathing on from the fall (the same rhythm as its last frame), then standing breath
        _breath_lying(t + FALL_D, ch, period=0.9, amount=_env(t, [(0, 2.0), (1.0, 2.0), (1.3, 0)]))
        _breath(t + FALL_D, ch, period=0.9, amount=_env(t, [(0, 0), (1.0, 0), (1.3, 1.6), (d, 1.6)]))
        add(ch, **{"tail__x": 28 * _env(t, [(0.1, 0), (0.3, 1), (0.6, 1), (0.8, 0)])})
    return frames_of(tr, extra, anchor="end"), False


DEAD = {**LIE_SIDE, "rootr.z": -95, "rootr.x": 0, "hipL.x": -45, "kneeL.x": 50, "hockL.x": -10, "hipR.x": -55,
        "kneeR.x": 60, "hockR.x": -12, "neck.x": -14, "neck.z": 14, "head.x": -10, "head.z": 16, "jaw": 30,
        "ears.x": -30, "tail.y": 26, "armL.x": -85, "armL.z": -5, "forearmL.x": -5, "armR.x": -60, "armR.z": -20,
        "forearmR.x": -10, "handL.x": 10, "handR.x": 10}


def die():
    """Death: he rears with a last scream, staggers, his legs give and he goes over onto his side like a
    horse, legs out stiff; a last spasm of the head, the jaw falling open."""
    d = 2.6
    keys = [(0, {"ground": 0.0, "floor": 1.0, "legik": 1.0, "root.y": 0.0}),
            (0.45, {"rootr.x": -36, "root.y": 0.04, "spine.x": -10, "neck.x": 0, "head.x": -36, "jaw": 46,
                    "ears.x": -45, **sym(arm__x=-45, arm__z=60, forearm__x=-30), "footR.y": 0.05, "footR.p": 30}),
            (0.9, {"rootr.x": -38, "head.x": -40, "jaw": 48}),
            (1.2, {"rootr.x": 8, "rootr.z": -10, "root.y": -0.06, "neck.x": 14, "head.x": 10, "jaw": 20,
                   "footR.y": 0.0, "footR.p": 0, **sym(arm__x=-20, arm__z=10, forearm__x=-20), "legik": 1.0,
                   "kneeout": 0.4}),
            (1.32, {"footL.y": 0.06, "footR.y": 0.1, "footL.p": 20, "footR.p": 30, "legik": 1.0,
                    "root.y": -0.06, "rootr.z": -28}),   # legs give
            (1.45, {"legik": 0.0, "root.y": -0.17, "rootr.z": -62, "hipL.x": -45, "kneeL.x": 50, "hipR.x": -55,
                    "kneeR.x": 60}),
            (1.62, {**DEAD, "jaw": 34, "ground": 1.0, "root.y": -0.29}),    # down: the legs as they will lie
            (2.1, {"neck.z": 22, "head.z": 24, "jaw": 40}),    # last spasm
            (2.6, DEAD)]
    tr = Track(keys, d, False)

    def extra(t, ch):
        _breath(t, ch, amount=_env(t, [(0, 1), (0.3, 0), (d, 0)]))
    return frames_of(tr, extra, anchor="start"), False


def dragged():
    """Dragged unconscious (the DraggedPose): limp on his side, legs trailing out behind, arms flung ahead,
    the head lolling forward on the floor."""
    pose = {**LIE_SIDE, "rootr.x": 0, "hipL.x": 10, "kneeL.x": 20, "hockL.x": -10, "hipR.x": 4, "kneeR.x": 24,
            "hockR.x": -10, "armL.x": -110, "armL.z": 0, "forearmL.x": -5, "armR.x": -95, "armR.z": -10,
            "forearmR.x": -10, "neck.x": 6, "neck.z": 10, "head.x": 14, "head.z": 14, "jaw": 18, "ears.x": -40,
            "tail.y": 6, "tail.x": 0}
    P = build_pose(pose)
    return [P, P.copy()], False


def being_picked_up():
    """Held up by the scruff in the Keeper's hand: hanging, the hind legs kicking, arms flailing, tail lashing,
    head twisting and snapping."""
    d = 2.0
    tr = Track([(0, {"legik": 0.0, "neck.x": -20, "head.x": -6, "jaw": 20, "ears.x": -40, "tail.x": -10,
                     **sym(arm__x=-30, arm__z=24, forearm__x=-40), "hipL.x": -20, "hipR.x": -20, "kneeL.x": 30,
                     "kneeR.x": 30, "hockL.x": -10, "hockR.x": -10, "pastL.x": 30, "pastR.x": 30})], d, True)

    def extra(t, ch):
        w = 2 * np.pi * t / d
        k = 2 * w
        add(ch, **{"rootr__z": 8 * np.sin(w), "rootr__x": 6 * np.sin(w + 1),
                   "hipL__x": -25 * np.sin(k), "hipR__x": 25 * np.sin(k), "kneeL__x": 20 * (1 + np.sin(k + 1)),
                   "kneeR__x": 20 * (1 - np.sin(k + 1)), "tail__y": 24 * np.sin(w + .5), "tail__x": 12 * np.sin(k),
                   "neck__y": 18 * np.sin(w - .7), "head__z": 12 * np.sin(w - 1.2), "jaw": 14 * np.sin(2 * k),
                   "armL__x": 30 * np.sin(k + .5), "armR__x": -30 * np.sin(k + .5),
                   "forearmL__x": -20 * np.sin(k), "forearmR__x": 20 * np.sin(k)})
    return frames_of(tr, extra), True


def torture():
    """On the wheel: spread-eagled, writhing, head thrashing and jaw screaming."""
    d = 2.4
    tr = Track([(0, {"legik": 0.0, **sym(arm__x=-10, arm__z=84, forearm__x=-6), "hipL.z": 24, "hipR.z": -24,
                     "hipL.x": 4, "hipR.x": 4, "kneeL.x": 10, "kneeR.x": 10, "rootr.x": -24, "spine.x": -14,
                     "neck.x": -34, "head.x": -26, "jaw": 34, "ears.x": -45, "tail.x": -20})], d, True)

    def extra(t, ch):
        w = 2 * np.pi * t / d
        add(ch, **{"spine__y": 10 * np.sin(w), "spine__z": 6 * np.sin(2 * w), "neck__y": -22 * np.sin(w + .5),
                   "neck__x": 10 * np.sin(2 * w + 1), "head__z": 16 * np.sin(w + 1.3), "jaw": 14 * np.sin(3 * w),
                   "tail__y": 25 * np.sin(w + 2), "hipL__x": 18 * np.sin(2 * w), "hipR__x": -18 * np.sin(2 * w),
                   "armL__x": 12 * np.sin(2 * w + .4), "armR__x": -12 * np.sin(2 * w + .4),
                   "handL__x": 20 * np.sin(3 * w), "handR__x": 20 * np.sin(3 * w + 1)})
    return frames_of(tr, extra), True


def dejected():
    """Dejected: slumped, head hanging, ears down, arms limp, tail low."""
    d = 3.0
    tr = Track([(0, {"spine.x": 22, "neck.x": 30, "head.x": 22, "ears.x": -15, "ears.z": 25, "tail.x": -25,
                     "rootr.x": 6, "root.y": -0.03, **sym(arm__x=4, arm__z=9, forearm__x=-6, hand__x=4)})], d, True)

    def extra(t, ch):
        _breath(t, ch, period=1.5, amount=1.6)
        w = 2 * np.pi * t / d
        add(ch, **{"head__y": 10 * np.sin(w), "neck__y": 5 * np.sin(w - .5), "tail2__y": 8 * np.sin(w + 1),
                   "earL__x": -14 * _pulse(t, 1.1, .3), "head__x": 6 * _pulse(t, 2.1, .6)})
        _dangle(t, ch, d / 2, amount=0.8)
    return frames_of(tr, extra), True


MORE = {
    "walk_start": walk_start, "walk_end": walk_end, "run_start": run_start, "run_end": run_end,
    "walkback": walkback, "walkback_start": walkback_start, "walkback_end": walkback_end,
    "magic": magic, "happy": happy, "dance": dance, "recoil_hff": recoil_hff, "recoil_hfb": recoil_hfb,
    "electrocute": electrocute, "drunk": drunk, "drunk_idle": drunk_idle, "drink": drink, "eating": eating,
    "research": research, "research_start": research_start, "research_end": research_end,
    "building": building, "building_start": building_start, "building_end": building_end, "melee2": melee2,
    "praying": praying, "praying_start": praying_start, "praying_end": praying_end,
    "sleeping": sleeping, "falling": falling, "waking_up": waking_up, "die": die,
    "being_picked_up": being_picked_up, "torture": torture, "dejected": dejected, "dragged": dragged,
}
