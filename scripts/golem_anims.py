"""The named Golem animations. LICENCE: GPL-3.0.

The MegaGlest Golem keeps its own stand / walk / attack / die / build_up (they are made for this body).
Everything else is new and has one of these origins (the "kind" field):
  "dk2"    a DK2 animation of the same meaning (mostly the Bile Demon), retargeted onto the Golem
  "golem"  made from the Golem's own MegaGlest walk/attack (run, turn, patrol, training, fighting ...)
  "hand"   no DK2 equivalent: hand-posed with the same skeleton (keypose_animation)
"""
import numpy as np

from scripts.donor_tracks import donor_tracks
from scripts.g3d import parse_g3d
from scripts.golem_retarget import BONES, _weights, keypose_animation, retarget
from scripts.make_golem_anims import FLIP, golem_rest

N = (0, 0, 0)
FLOOR = -0.23   # the Golem's lowest rest vertex
UP_BONES = ("torso", "armN", "armP", "foreN", "foreP")


def _arms(x, z=0.0, elbow=0.0):
    """Both arms swung forward by x degrees, closed in by z, forearms bent by elbow."""
    return {"armN": (x, 0, z), "armP": (x, 0, -z), "foreN": (elbow, 0, 0), "foreP": (elbow, 0, 0)}


def _resample(frames, n):
    t = np.linspace(0, len(frames) - 1, n)
    i = np.minimum(t.astype(int), len(frames) - 2)
    u = (t - i)[:, None, None]
    return frames[i] * (1 - u) + frames[i + 1] * u


def _loop_close(frames, k=6):
    """Blend the last k frames back to frame 0 so a looped play has no jump."""
    f = frames.copy()
    k = min(k, len(f) // 3)
    for i in range(k):
        u = (i + 1) / (k + 1)
        f[-k + i] = f[-k + i] * (1 - u) + f[0] * u
    return f


def _chain(*parts, k=6):
    """Join animations with a short cross-fade so the seams do not jump."""
    out = parts[0]
    for nxt in parts[1:]:
        u = (np.arange(k) + 1)[:, None, None] / (k + 1)
        out = np.concatenate([out[:-k], out[-k:] * (1 - u) + nxt[:k] * u, nxt[k:]])
    return out


def _rot_about(pts, pivot, axis, deg):
    """Rotate pts (any leading shape, last dim 3) about a pivot; deg may broadcast against pts' leading dims."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    d = pts - pivot
    if axis == "x":
        y, z = d[..., 1] * c - d[..., 2] * s, d[..., 1] * s + d[..., 2] * c
        return np.stack([d[..., 0], y, z], -1) + pivot
    if axis == "y":
        x, z = d[..., 0] * c + d[..., 2] * s, -d[..., 0] * s + d[..., 2] * c
        return np.stack([x, d[..., 1], z], -1) + pivot
    raise ValueError(axis)


def _lean(frames, deg, pivot=(0.0, 0.0, 0.3)):
    """Tip the whole body forward (+deg) about the feet."""
    out = np.array([_rot_about(f, np.array(pivot), "x", d) for f, d in zip(frames, np.broadcast_to(deg, len(frames)))])
    return _ground(out, FLOOR)


def _yaw_all(frames, deg, centre=(0.0, 0.0, 0.45)):
    return np.array([_rot_about(f, np.array(centre), "y", d) for f, d in zip(frames, np.broadcast_to(deg, len(frames)))])


def _yaw_upper(frames, w_upper, deg, pivot=(0.0, 0.0, 0.35)):
    """Twist the upper body (torso + arms, by skin weight) about the vertical axis; legs stay."""
    out = frames.copy()
    for i, f in enumerate(frames):
        rot = _rot_about(f, np.array(pivot), "y", np.broadcast_to(deg, len(frames))[i])
        out[i] = f + w_upper[:, None] * (rot - f)
    return out


def _ground(frames, floor):
    f = frames.copy()
    f[:, :, 1] += np.maximum(0, floor - f[:, :, 1].min(axis=1))[:, None]
    return f


def stock_frames(unit_dir, name):
    me = parse_g3d((unit_dir / f"models/golem_{name}.g3d").read_bytes()).meshes[0]
    return me.vertices.astype(float)


def build_all(meshes_dir, cache, unit_dir) -> dict:
    g0, idx = golem_rest(unit_dir)
    floor = g0[:, 1].min()
    W = _weights(g0, idx)
    w_up = W[:, [BONES.index(b) for b in UP_BONES]].sum(1)
    tracks = {}

    def rt(creature, anim, rest_anim=None, **kw):
        if creature not in tracks:
            tracks[creature] = donor_tracks(meshes_dir, cache, creature)
        T = tracks[creature]
        r = rest_anim or ("walk" if "walk" in T else "idle1")
        return retarget(T[anim][0] * FLIP, T[r][0][0] * FLIP, g0, idx, **kw)

    def kp(keys, n, loop, ground=True):
        return keypose_animation(g0, idx, keys, n, loop=loop, ground=ground)

    # the MegaGlest Golem's own gait and swing (15 frames each); its walk dips below the ground, so feet are planted
    walk = _ground(stock_frames(unit_dir, "walking"), floor)
    attack = _ground(stock_frames(unit_dir, "attacking"), floor)
    lib = {}

    def add(name, frames, seconds, wrap, kind, source):
        if len(frames) / seconds < 10:                  # keep playback smooth: too few frames for the time
            frames = _resample(np.asarray(frames), int(seconds * 14))
        lib[name] = {"frames": [np.asarray(f) for f in frames], "wrap": wrap,
                     "fps": round(len(frames) / seconds, 3), "kind": kind, "source": source}

    # ---- a DK2 animation of the same meaning
    add("fidgeting", _loop_close(rt("bile_demon", "idle2")), 6.0, True, "dk2", "Bile Demon idle2")
    add("eating", _loop_close(rt("bile_demon", "eat")), 7.0, True, "dk2", "Bile Demon eat")
    add("praying", _loop_close(rt("bile_demon", "pray")), 6.0, True, "dk2", "Bile Demon pray")
    add("happy", _loop_close(rt("bile_demon", "happy")), 8.0, True, "dk2", "Bile Demon happy")
    add("angry", _loop_close(rt("bile_demon", "angry")), 4.0, True, "dk2", "Bile Demon angry")
    dance = rt("bile_demon", "dance")
    add("celebrating", _loop_close(dance[:len(dance) // 2]), 5.0, True, "dk2", "Bile Demon dance, first half")
    add("victory_celebration", _chain(dance, rt("bile_demon", "happy")), 12.0, False, "dk2",
        "Bile Demon dance, then happy")
    add("manufacturing", _loop_close(rt("bile_demon", "building")), 4.0, True, "dk2", "Bile Demon building")
    add("getting_hit", rt("bile_demon", "recoil(hff)"), 1.6, False, "dk2", "Bile Demon recoil (hit from the front)")
    add("being_slapped", rt("bile_demon", "recoil(hfb)"), 1.6, False, "dk2", "Bile Demon recoil (hit from behind)")
    add("staggering", _loop_close(rt("bile_demon", "drunkidle")), 6.0, True, "dk2", "Bile Demon drunk idle (swaying)")
    add("waking_up", rt("bile_demon", "getup"), 4.5, False, "dk2", "Bile Demon getup")
    fall = rt("bile_demon", "fallback")
    add("knockback", _chain(fall, rt("bile_demon", "getup")), 6.5, False, "dk2", "Bile Demon fallback, then getup")
    add("guarding", _loop_close(_resample(rt("skeleton", "attention", amp={"armN": 1.4, "armP": 1.4, "torso": 1.4}), 40)), 4.0, True, "dk2",
        "Skeleton attention")
    add("interacting_with_objects", _loop_close(rt("dwarf", "claimwall", amp={"armN": 1.6, "armP": 1.6, "foreN": 1.6, "foreP": 1.6})),
        3.0, True, "dk2", "Dwarf claim wall (amplified)")
    # sleeping: collapse (Bile Demon fallback), then lie breathing
    slump = fall[-1]
    cen = slump.mean(0)
    breath = np.array([cen + (slump - cen) * (1 + 0.022 * np.sin(2 * np.pi * i / 48)) for i in range(48)])
    add("sleeping", _chain(fall, _ground(breath, floor)), 9.0, False, "dk2", "Bile Demon fallback, then breathing")

    # ---- from the Golem's own MegaGlest motion
    gait = _resample(walk, 30)
    add("running", _lean(_resample(walk, 30), 14 + 3 * np.sin(np.linspace(0, 4 * np.pi, 30, endpoint=False))), 1.7, True,
        "golem", "MegaGlest Golem walk, faster, leaning forward")
    turn = np.concatenate([gait, gait[:12]])
    add("turning", _yaw_all(turn, 100 * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, len(turn))))), 3.2, False, "golem",
        "MegaGlest Golem walk steps while the body turns 100 degrees")
    patrol = _resample(walk, 60)
    add("patrolling", _yaw_upper(patrol, w_up, 24 * np.sin(np.linspace(0, 2 * np.pi, 60, endpoint=False))), 8.0, True, "golem",
        "MegaGlest Golem walk with the upper body scanning left and right")
    swing = _resample(attack, 45)
    add("training", _chain(swing, swing, swing), 6.5, True, "golem", "MegaGlest Golem attack, three heavy swings")
    add("special_attack", _lean(_resample(attack, 44), 10 * np.sin(np.linspace(0, np.pi, 44))), 3.6, False, "golem",
        "MegaGlest Golem attack, slower, with a lunge")
    add("fighting_another_creature", _loop_close(_chain(_resample(attack, 36), rt("bile_demon", "recoil(hff)"), _resample(attack, 36),
                                            rt("bile_demon", "recoil(hfb)")), k=14), 8.0, True, "golem",
        "MegaGlest Golem attack, taking a hit, attacking again")

    # ---- no DK2 equivalent: hand-posed with the same skeleton
    look = [(0, {}, N), (0.2, {"torso": (2, 32, 0)}, N), (0.4, {"torso": (0, 32, 0)}, N), (0.6, {"torso": (2, -32, 0)}, N),
            (0.8, {"torso": (0, -32, 0)}, N)]
    add("looking_around", kp(look, 72, True), 6.0, True, "hand", "hand-posed torso turns left and right")
    shiver = [(i / 8, {"torso": (28, 3 if i % 2 else -3, 0), **_arms(-60, 25, -40), "legN": (-6, 0, 0), "legP": (-6, 0, 0)},
               (0, -0.1, 0)) for i in range(8)]
    add("frightened", kp(shiver, 32, True), 1.6, True, "hand", "hand-posed cowering and trembling")
    cross = _arms(-50, 40, -95)
    add("blocking", kp([(0, {}, N), (0.25, {"torso": (12, 0, 0), **cross}, (0, 0, -0.05)),
                        (0.55, {"torso": (12, 0, 0), **cross}, (0, 0, -0.05)),
                        (0.65, {"torso": (16, 4, 0), **cross}, (0, 0, -0.12)), (0.8, {"torso": (12, 0, 0), **cross}, (0, 0, -0.05)),
                        (1.0, {"torso": (12, 0, 0), **cross}, (0, 0, -0.05))], 36, False), 2.2, False, "hand",
        "hand-posed guard: forearms crossed in front, one impact shake")
    sad = {"armN": (10, 0, -4), "armP": (10, 0, 4), "foreN": (12, 0, 0), "foreP": (12, 0, 0)}
    add("unhappy", kp([(0, {"torso": (18, 0, 0), **sad}, (0, -0.05, 0)), (0.25, {"torso": (24, -10, 0), **sad}, (0, -0.07, 0)),
                       (0.5, {"torso": (18, 0, 0), **sad}, (0, -0.05, 0)), (0.75, {"torso": (24, 10, 0), **sad}, (0, -0.07, 0))],
                      60, True), 6.0, True, "hand", "hand-posed slump and slow head shake")
    hold = {"torso": (10, 0, 0), **_arms(-40, 12, -70)}
    add("picking_up", kp([(0, {}, N), (0.3, {"torso": (55, 0, 0), **_arms(-25, 0, -10)}, (0, -0.1, 0.2)),
                          (0.45, {"torso": (55, 0, 0), **_arms(-25, 8, -25)}, (0, -0.1, 0.2)), (0.8, hold, N),
                          (1.0, {"torso": (5, 0, 0), **_arms(-35, 12, -80)}, N)], 52, False), 3.2, False, "hand",
        "hand-posed bow, grab, lift to chest")
    add("dropping", kp([(0, {"torso": (5, 0, 0), **_arms(-35, 12, -80)}, N), (0.35, {"torso": (5, 0, 0), **_arms(-45, -30, -40)}, N),
                        (0.6, {"torso": (8, 0, 0), **_arms(-20, -20, -20)}, N), (1.0, {}, N)], 36, False), 2.2, False, "hand",
        "hand-posed open hands, let go")
    dangle_a = {"armN": (10, 0, -8), "armP": (-10, 0, 8), "legN": (-8, 0, 0), "legP": (8, 0, 0), "foreN": (6, 0, 0), "foreP": (6, 0, 0)}
    dangle_b = {"armN": (-10, 0, -8), "armP": (10, 0, 8), "legN": (8, 0, 0), "legP": (-8, 0, 0), "foreN": (6, 0, 0), "foreP": (6, 0, 0)}
    add("being_picked_up", kp([(0, dangle_a, (0, 1.3, 0)), (0.5, dangle_b, (0, 1.35, 0))], 48, True, ground=False), 4.0, True,
        "hand", "hand-posed dangling: lifted, limbs swaying")
    return lib
