"""The named Battlemage animations. LICENCE: GPL-3.0.

His own MegaGlest stand / walk / attack / die / morph are kept as exported. Everything else is new:
  "dk2"   a real DK2 Priest (hero) animation of the same meaning, retargeted onto the Battlemage
  "hand"  no DK2 equivalent: hand-posed with the same skeleton (picking things up, working a bench, ...)
  "mix"   DK2 motion combined with a hand-posed head scan (patrolling, guarding)
DK2 plays its creature animations at 30 frames per second (assumed), so a DK2 clip keeps its natural length.
"""
import numpy as np

from scripts.battlemage_rig import battlemage_rig
from scripts.donor_tracks import donor_tracks
from scripts.golem_anims import _chain, _ground, _loop_close, _resample, _rot_about
from scripts.humanoid_retarget import keypose, retarget

FLIP = np.array([-1.0, 1.0, 1.0])       # DK2 data is left-handed, MegaGlest is right-handed
DK2_FPS = 30.0
N = (0, 0, 0)


def _arms(x, z=0.0, elbow=0.0):
    """Both arms swung forward by x degrees, closed in by z, forearms bent by elbow."""
    return {"armN": (x, 0, z), "armP": (x, 0, -z), "foreN": (elbow, 0, 0), "foreP": (elbow, 0, 0)}


def _yaw_all(frames, deg, centre=(0.0, 0.0, 0.0)):
    return np.array([_rot_about(f, np.array(centre), "y", d) for f, d in zip(frames, np.broadcast_to(deg, len(frames)))])


def scan_head(deg, cycles=1.0, phase=0.0, torso=0.35):
    """extra() hook: the head (and a little of the torso) turn left and right while the donor motion plays."""
    def hook(f, n, r):
        a = deg * np.sin(2 * np.pi * (cycles * f / n + phase))
        r = dict(r)
        r["head"] = r["head"] + np.radians([0, a, 0])
        r["torso"] = r["torso"] + np.radians([0, a * torso, 0])
        return r
    return hook


def build_all(meshes_dir, cache, unit_dir) -> dict:
    rig = battlemage_rig(unit_dir)
    floor = rig.floor
    tracks = {}

    def rt(creature, anim, **kw):
        if creature not in tracks:
            tracks[creature] = donor_tracks(meshes_dir, cache, creature)
        T = tracks[creature]
        r = "idle1" if "idle1" in T else "walk"       # a neutral standing rest pose: a walk frame has the legs apart
        out = retarget(rig, T[anim][0] * FLIP, T[r][0][0] * FLIP, **kw)
        return out, len(T[anim][0]) / DK2_FPS          # frames, natural duration in seconds

    def kp(keys, n, loop, ground=True):
        return keypose(rig, keys, n, loop=loop, ground=ground)

    lib = {}

    def add(name, frames, seconds, wrap, kind, source):
        frames = np.asarray(frames)
        if len(frames) / seconds < 24:                      # keep playback smooth: too few frames for the time
            frames = _resample(frames, int(seconds * 30))
        if len(frames) / seconds > 36:                      # in-betweens added for slow limbs: keep it near 30 fps
            frames = _resample(frames, int(seconds * 30))
        lib[name] = {"frames": [np.asarray(f) for f in frames], "wrap": wrap, "fps": round(len(frames) / seconds, 3),
                     "kind": kind, "source": source}

    def dk2(name, creature, anim, wrap, source=None, loop=None, **kw):
        fr, secs = rt(creature, anim, **kw)
        add(name, _loop_close(fr) if (loop if loop is not None else wrap) else fr, secs, wrap, "dk2",
            source or f"{creature.title()} {anim}")
        return fr, secs

    # ---- a DK2 animation of the same meaning
    dk2("fidgeting", "pr1", "idle2", True)
    dk2("running", "pr1", "run", True)
    dk2("eating", "pr1", "eat", True, loop=True)
    dk2("praying", "pr1", "pray", True)
    dk2("researching", "pr1", "research", True)
    dk2("happy", "pr1", "happy", True, loop=True)
    dk2("angry", "pr1", "angry", True, loop=True)
    dance, dsecs = rt("pr1", "dance")
    add("celebrating", _loop_close(dance[: len(dance) // 2]), dsecs / 2, True, "dk2", "Priest dance, first half")
    happy, hsecs = rt("pr1", "happy")
    add("victory_celebration", _chain(dance, happy), dsecs + hsecs, False, "dk2", "Priest dance, then happy")
    dk2("casting_spell", "pr1", "magic", False, "Priest magic")
    dk2("special_attack", "dark_elf", "magic", False, "Dark Elf magic")
    dk2("getting_hit", "pr1", "recoil(hff)", False, "Priest recoil (hit from the front)")
    dk2("being_slapped", "pr1", "recoil(hfb)", False, "Priest recoil (hit from behind)")
    dk2("staggering", "pr1", "drunk", True, "Priest drunk (staggering)", loop=True)
    dk2("waking_up", "pr1", "getup", False)
    fall, fsecs = rt("pr1", "fallback")
    get, gsecs = rt("pr1", "getup")
    add("knockback", _chain(fall, get), fsecs + gsecs, False, "dk2", "Priest fallback, then getup")
    slump = fall[-1]
    cen = slump.mean(0)
    breath = np.array([cen + (slump - cen) * (1 + 0.02 * np.sin(2 * np.pi * i / 48)) for i in range(48)])
    add("sleeping", _chain(fall, _ground(breath, floor)), fsecs + 3.5, False, "dk2", "Priest fallback, then breathing")
    melee, msecs = rt("pr1", "melee")
    add("training", _chain(melee, melee, melee), 3 * msecs, True, "dk2", "Priest staff swing, three times")
    mel2, m2 = rt("dark_elf", "melee")
    add("fighting_another_creature",
        _loop_close(_chain(melee, rt("pr1", "recoil(hff)")[0], mel2, rt("pr1", "recoil(hfb)")[0]), k=10),
        msecs + m2 + 3.5, True, "dk2", "Priest and Dark Elf melee with Priest recoils")
    # turning: Priest walking steps while the body swings 100 degrees
    walk, wsecs = rt("pr1", "walk")
    turn = np.concatenate([walk, walk[:20]])
    add("turning", _yaw_all(turn, 100 * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, len(turn))))), wsecs * 1.3, False, "dk2",
        "Priest walk steps while the body turns 100 degrees")
    # donor motion plus a hand-posed head scan
    pat, psecs = rt("pr1", "walk", extra=scan_head(28, 1.0))
    add("patrolling", _loop_close(pat), psecs, True, "mix", "Priest walk with the head scanning left and right")
    gd, gds = rt("pr1", "idle1", extra=scan_head(35, 1.0))
    add("guarding", _loop_close(gd), gds, True, "mix", "Priest idle with a slow head scan")
    being, bsecs = rt("pr1", "swing", ground=False)
    lift = being.copy()
    lift[:, :, 1] += 0.9 - lift[:, :, 1].min()
    add("being_picked_up", _loop_close(lift), bsecs, True, "dk2", "Priest swing (the animation DK2 plays when a creature is picked up)")

    # ---- no DK2 equivalent: hand-posed with the same skeleton
    look = [(0, {}, N), (0.2, {"head": (0, 55, 0), "torso": (0, 12, 0)}, N), (0.4, {"head": (6, 55, 0), "torso": (0, 12, 0)}, N),
            (0.6, {"head": (0, -55, 0), "torso": (0, -12, 0)}, N), (0.8, {"head": (4, -55, 0), "torso": (0, -12, 0)}, N)]
    add("looking_around", kp(look, 96, True), 6.0, True, "hand", "hand-posed head and shoulder turns left and right")
    shiver = [(i / 8, {"torso": (25, 2 if i % 2 else -2, 0), "head": (-12, 0, 0), **_arms(-65, 25, -95),
                       "legN": (-8, 0, 0), "legP": (-8, 0, 0)}, (0, -0.1, 0)) for i in range(8)]
    add("frightened", kp(shiver, 36, True), 1.6, True, "hand", "hand-posed cowering with trembling")
    cross = _arms(-55, 45, -120)
    add("blocking", kp([(0, {}, N), (0.25, {"torso": (10, 0, 0), "head": (8, 0, 0), **cross}, (0, 0, -0.05)),
                        (0.55, {"torso": (10, 0, 0), "head": (8, 0, 0), **cross}, (0, 0, -0.05)),
                        (0.65, {"torso": (15, 4, 0), "head": (10, 0, 0), **cross}, (0, 0, -0.12)),
                        (0.8, {"torso": (10, 0, 0), "head": (8, 0, 0), **cross}, (0, 0, -0.05)),
                        (1.0, {"torso": (10, 0, 0), "head": (8, 0, 0), **cross}, (0, 0, -0.05))], 40, False), 2.2, False, "hand",
        "hand-posed guard: forearms crossed in front, one impact shake")
    sad = {"armN": (8, 0, -4), "armP": (8, 0, 4), "foreN": (10, 0, 0), "foreP": (10, 0, 0)}
    add("unhappy", kp([(0, {"torso": (15, 0, 0), "head": (32, 0, 0), **sad}, (0, -0.04, 0)),
                       (0.25, {"torso": (18, -6, 0), "head": (34, -22, 0), **sad}, (0, -0.05, 0)),
                       (0.5, {"torso": (15, 0, 0), "head": (32, 0, 0), **sad}, (0, -0.04, 0)),
                       (0.75, {"torso": (18, 6, 0), "head": (34, 22, 0), **sad}, (0, -0.05, 0))], 72, True), 6.0, True, "hand",
        "hand-posed slump, head down and a slow head shake")
    hold = {"torso": (8, 0, 0), "head": (14, 0, 0), **_arms(-45, 14, -85)}
    add("picking_up", kp([(0, {}, N), (0.3, {"torso": (50, 0, 0), "head": (-25, 0, 0), **_arms(-30, 0, -10), "legN": (-12, 0, 0), "legP": (-12, 0, 0)}, (0, -0.1, 0.2)),
                          (0.45, {"torso": (50, 0, 0), "head": (-25, 0, 0), **_arms(-30, 8, -30), "legN": (-12, 0, 0), "legP": (-12, 0, 0)}, (0, -0.1, 0.2)),
                          (0.8, hold, N), (1.0, {"torso": (4, 0, 0), **_arms(-40, 14, -90)}, N)], 56, False), 3.2, False, "hand",
        "hand-posed bow, grab, lift to the chest")
    add("dropping", kp([(0, {"torso": (4, 0, 0), **_arms(-40, 14, -90)}, N), (0.35, {"torso": (5, 0, 0), **_arms(-50, -30, -45)}, N),
                        (0.6, {"torso": (8, 0, 0), "head": (15, 0, 0), **_arms(-25, -20, -20)}, N), (1.0, {}, N)], 40, False),
        2.2, False, "hand", "hand-posed open hands, let go")
    work = [(i / 4, {"torso": (14, 0, 0), "head": (14, 0, 0), "armN": (-55, 0, 12), "armP": (-55, 0, -12),
                     "foreN": (-60 - 40 * (i % 2), 0, 0), "foreP": (-100 + 40 * (i % 2), 0, 0)}, (0, 0, 0)) for i in range(4)]
    add("manufacturing", kp(work, 48, True), 3.0, True, "hand", "hand-posed work at a bench: forearms hammering alternately")
    lever = [(0, {"torso": (8, 8, 0), "head": (6, 0, 0), "armP": (-80, 0, -5), "foreP": (-25, 0, 0), "armN": (-5, 0, 6)}, N),
             (0.4, {"torso": (14, 8, 0), "head": (8, 0, 0), "armP": (-85, 0, -5), "foreP": (-25, 0, 0), "armN": (-5, 0, 6)}, (0, 0, 0.05)),
             (0.75, {"torso": (4, 4, 0), "head": (4, 0, 0), "armP": (-40, 0, -5), "foreP": (-20, 0, 0), "armN": (-5, 0, 6)}, (0, 0, -0.05))]
    add("interacting_with_objects", kp(lever, 54, True), 3.0, True, "hand", "hand-posed reach and pull, like working a lever")
    return lib


def stretched_triangles(unit_dir, lib, min_len=0.45, ratio=3.0):
    """Indices (into the dying mesh's triangle list) of the few seam triangles that join the sleeve to the tunic side:
    hidden against the body at rest, but a raised arm pulls them into a wing of cloth between wrist and hip.
    A triangle counts when some clip stretches its longest edge beyond min_len and ratio x its rest length."""
    rig = battlemage_rig(unit_dir)
    tri = rig.indices.reshape(-1, 3)

    def longest(p):
        return np.max([np.linalg.norm(p[..., tri[:, a], :] - p[..., tri[:, b], :], axis=-1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)

    rest = longest(rig.rest)
    worst = np.zeros(len(tri))
    for a in lib.values():
        worst = np.maximum(worst, longest(np.asarray(a["frames"])).max(0))
    return np.where((worst > min_len) & (worst / np.maximum(rest, 0.05) > ratio))[0]
