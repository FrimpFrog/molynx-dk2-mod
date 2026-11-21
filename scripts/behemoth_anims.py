"""The named Behemoth animations. LICENCE: GPL-3.0.

His own MegaGlest stand / walk / attack / die are kept as exported (real speeds). Every other clip is a real DK2 animation
of the same meaning (Giant first; Bile Demon, Dwarf, Skeleton where the Giant has none) retargeted onto the Behemoth.
A clip with no DK2 equivalent (casting, researching, ...) is not made. Nothing here is hand-posed.
DK2 plays its creature animations at 30 frames per second (assumed), so a DK2 clip keeps its natural length.
"""
from pathlib import Path

import numpy as np

from scripts.behemoth_rig import behemoth_rest, behemoth_rig
from scripts.g3d import parse_g3d
from scripts.donor_tracks import donor_tracks
from scripts.golem_anims import _chain, _ground, _loop_close, _resample, _rot_about
from scripts.humanoid_retarget import retarget

FLIP = np.array([-1.0, 1.0, 1.0])       # DK2 data is left-handed, MegaGlest is right-handed
MIRROR = np.array([1.0, 1.0, 1.0])      # a donor mirrored left-right: the Giant carries its club in the arm that the Behemoth keeps free
DK2_FPS = 30.0
# His left arm is folded up holding the club on his shoulder. A donor arm hangs straight, so its elbow bend and full swing would drive the club
# through his chest: the forearm stays rigid with the upper arm, and the shoulder only follows part of the motion.
LEGS_NARROW = {"legN": 0.4, "legP": 0.4}      # the Bile Demon and Dwarf stand with their legs wide; he would be doing the splits
CLUB_ARM = {"armN": 0.45, "foreN": 0.0}
KNEEL_CAPS = {"torso": 120.0, "legN": 120.0, "legP": 120.0, "shinN": 150.0, "shinP": 150.0}      # a deep kneel and bow
CLUB_LOCKED = {"armN": 0.0, "foreN": 0.0}      # clips driven by the arms: the club stays on his shoulder, the other arm does the work


def kneel(f, n, r):
    """The Giant's pray clip is a kneel the whole way through (its legs hardly change): thighs upright, shins folded back along the floor,
    while the torso, head and arms follow the clip. The Giant's own leg numbers do not carry over to his body, so the kneel is set directly."""
    r = dict(r)
    for side in "NP":
        r["leg" + side] = np.radians([-8.0, 0, 0])
        r["shin" + side] = np.radians([105.0, 0, 0])
        r["foot" + side] = np.radians([80.0, 0, 0])       # toes point back along the floor
    return r


def build_all(meshes_dir, cache, unit_dir) -> dict:
    rig = behemoth_rig(unit_dir)
    floor = rig.floor
    tracks = {}

    def rt(creature, anim, mirror=False, **kw):
        if creature not in tracks:
            tracks[creature] = donor_tracks(meshes_dir, cache, creature)
        T = tracks[creature]
        r = "idle1" if "idle1" in T else "walk"       # a neutral standing rest pose
        kw = {**kw, "amp": {**CLUB_ARM, **kw.get("amp", {})}}
        flip = MIRROR if mirror else FLIP
        out = retarget(rig, T[anim][0] * flip, T[r][0][0] * flip, **kw)
        return out, len(T[anim][0]) / DK2_FPS

    lib = {}

    def add(name, frames, seconds, wrap, source):
        frames = np.asarray(frames)
        if len(frames) / seconds < 24 or len(frames) / seconds > 36:
            frames = _resample(frames, int(seconds * 30))
        lib[name] = {"frames": [np.asarray(f) for f in frames], "wrap": wrap, "fps": round(len(frames) / seconds, 3),
                     "kind": "dk2", "source": source}

    def dk2(name, creature, anim, wrap, source=None, loop=None, **kw):
        fr, secs = rt(creature, anim, **kw)
        add(name, _loop_close(fr) if (loop if loop is not None else wrap) else fr, secs, wrap, source or f"{creature.title()} {anim}")
        return fr, secs

    dk2("fidgeting", "giant", "idle2", True)
    dk2("running", "giant", "run", True)
    dk2("eating", "giant", "eat", True, loop=True, amp=CLUB_LOCKED, mirror=True)
    dk2("praying", "giant", "pray", True, amp=CLUB_LOCKED, mirror=True, tilt_body=False, caps=KNEEL_CAPS, extra=kneel)
    dk2("happy", "giant", "happy", True, loop=True, amp=CLUB_LOCKED, mirror=True)
    dk2("angry", "giant", "angry", True, loop=True)
    dance, dsecs = rt("giant", "dance", amp=CLUB_LOCKED, mirror=True)
    add("celebrating", _loop_close(dance[: len(dance) // 2]), dsecs / 2, True, "Giant dance, first half")
    happy, hsecs = rt("giant", "happy", amp=CLUB_LOCKED, mirror=True)
    add("victory_celebration", _chain(dance, happy), dsecs + hsecs, False, "Giant dance, then happy")
    dk2("getting_hit", "giant", "recoil(hff)", False, "Giant recoil (hit from the front)")
    dk2("being_slapped", "giant", "recoil(hfb)", False, "Giant recoil (hit from behind)")
    dk2("staggering", "giant", "drunk", True, "Giant drunk (staggering)", loop=True)
    dk2("waking_up", "giant", "getup", False)
    fall, fsecs = rt("giant", "fallback")
    get, gsecs = rt("giant", "getup")
    add("knockback", _chain(fall, get), fsecs + gsecs, False, "Giant fallback, then getup")
    # sleeping: he lies down the way DK2 shows a Giant going down (its die clip), then breathes
    die, dsecs = rt("giant", "die")
    slump = die[-1]
    cen = slump.mean(0)
    breath = np.array([cen + (slump - cen) * (1 + 0.02 * np.sin(2 * np.pi * i / 48)) for i in range(48)])
    add("sleeping", _chain(die, _ground(breath, floor)), dsecs + 3.5, False, "Giant die (lying down), then breathing")
    # his own club swing (MegaGlest attack, 24 frames at its real 18 fps), recentred like the rest pose, is his weapon clip: the DK2
    # melee clips swing an arm that is not holding a club, so they are not used for him
    off = behemoth_rest(unit_dir)[2]
    model = parse_g3d((Path(unit_dir) / "models/behemoth_attacking.g3d").read_bytes())
    swing = np.concatenate([m.vertices.astype(float) for m in model.meshes], axis=1) - off
    swing_secs = len(swing) / (len(swing) * 75 / 100.0)
    swing = _resample(swing, int(round(swing_secs * 30)))
    hit1, h1 = rt("giant", "recoil(hff)")
    hit2, h2 = rt("giant", "recoil(hfb)")
    add("training", _loop_close(_chain(swing, swing, swing)), 3 * swing_secs, True, "His own MegaGlest club swing, three times")
    add("fighting_another_creature", _loop_close(_chain(swing, hit1, swing, hit2), k=10), 2 * swing_secs + h1 + h2, True,
        "His own club swing twice with Giant recoils")
    pat, psecs = rt("giant", "walk")
    walk_cycle = _loop_close(pat)
    add("patrolling", walk_cycle, psecs, True, "Giant walk")
    gd, gds = rt("skeleton", "attention")
    add("guarding", _loop_close(gd), gds, True, "Skeleton attention")
    bui, bs = rt("bile_demon", "building", amp={**LEGS_NARROW, **CLUB_LOCKED}, mirror=True)
    add("manufacturing", _loop_close(bui), bs, True, "Bile Demon building")
    wall, wsecs = rt("dwarf", "claimwall", amp=LEGS_NARROW)
    add("interacting_with_objects", _loop_close(wall), wsecs, True, "Dwarf claim wall")
    being, bsecs = rt("giant", "swing", ground=False)
    lift = being.copy()
    lift[:, :, 1] += 0.9 * rig.height / 3.88 - lift[:, :, 1].min()
    add("being_picked_up", _loop_close(lift), bsecs, True, "Giant swing (the animation DK2 plays when a creature is picked up)")
    return lib
