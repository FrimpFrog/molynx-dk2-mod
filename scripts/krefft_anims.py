"""Krefft's full clip set (version B, v5), keyframed on his own rig (scripts.krefft_rig) with the Bile Demon
as the donor: every slot the Bile Demon's record fills (Creatures.kwd) gets a clip of the Bile Demon's own
length, and where his body motion fits the turtle (the walk's rock, the recoils, the falls, the rise out of
the Portal, the wheel, the bed, the prayer) it is measured from his file (krefft_rig.donor_body) and laid
under Krefft's own keys at the weight given per clip. His carried objects are the game's own: the Bile
Demon's chicken, bottle and lair bed, and the game's Coin mesh (the Coin Spill / Casino Jackpot effects'
coin).

Character: the Casino's loan shark. Slow, greedy, smug. A short waddling walk with the shell rocking; a
smug idle with a shell shrug; a turtle's snapping bite (the head shoots out of the shell); a coin spit with
the cheeks puffed; the head pulled into the shell when hit, stunned, asleep; a gloating claw rub; an angry
stomp; a coin flip; a fruit machine's lever.

A clip is (frames, wrap, channels(t), donor, props): `CLIPS[name]()` -> Clip. Every clip is at 30 fps (the
Sleep file plays at 15, like the Bile Demon's); a looping clip's last frame is a copy of its first.

LICENCE: GPL-3.0.
"""
from dataclasses import dataclass, field

import numpy as np

from scripts.dragon_anims import Track

FPS = 30


@dataclass
class Clip:
    name: str
    stem: str
    frames: int                    # file frames (a loop's last is a copy of its first)
    wrap: bool
    ch: object                     # t (s) -> channels
    donor: tuple = None            # (Bile Demon clip, rotation weight, offset weight, offset axes)
    props: list = field(default_factory=list)      # prop kinds this clip carries (krefft_anims_build)
    fps: int = FPS
    source: str = ""               # notes: the donor and what is his own

    @property
    def duration(self):
        return (self.frames - 1) / self.fps


DEFAULTS = {"ik": 1.0}


def K(dur, wrap, *keys, extra=None):
    """A Track over keys [(t, {channel: value})] plus an optional procedural layer extra(t, ch). Unlike the
    Dragon's Track, a channel a key leaves out is at its default there (0; ik 1), not held."""
    names = {c for _, d in keys for c in d}
    keys = [(t, {c: d.get(c, DEFAULTS.get(c, 0.0)) for c in names}) for t, d in keys]
    tr = Track(list(keys), dur, wrap)

    def f(t):
        ch = tr(t)
        if extra:
            extra(t, ch)
        return ch
    return f


def add(ch, **kw):
    for k, v in kw.items():
        k = k.replace("__", ".")
        ch[k] = ch.get(k, 0.0) + v
    return ch


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def pulse(t, t0, t1):
    """0 outside [t0, t1], a smooth bump to 1 in the middle."""
    if t <= t0 or t >= t1:
        return 0.0
    return float(np.sin(np.pi * (t - t0) / (t1 - t0)) ** 2)


def env(t, t0, t1, fade=.2):
    """1 inside [t0, t1], easing in and out over `fade`."""
    return ease((t - t0) / fade) * ease((t1 - t) / fade)


def breath(t, ch, period, amount=1.0):
    s = np.sin(2 * np.pi * t / period)
    add(ch, root__y=.0035 * s * amount, body__x=-.9 * s * amount, neck__x=.6 * s * amount,
        arms__z=1.2 * s * amount)


# the stance he stands in between moves: a little lean back (chest out), the chin up - the boss
STAND = {"body.x": -3.0, "head.x": -4.0}


def stand(**kw):
    d = dict(STAND)
    for k, v in kw.items():
        d[k.replace("__", ".")] = v
    return d


# ------------------------------------------------------------------------------------------- gaits

def gait(phase, ch, stride, lift, duty=.62, back=False, roll=6.0, bob=.012, sway=.016, yaw=4.0):
    """A waddle on the spot (the game carries him): each foot slides straight back on the floor for
    `duty` of the cycle (a planted foot moves at the walk's speed), then lifts and swings forward. The
    shell rolls over the planted foot and bobs down as each foot lands. `back`: the feet slide forward."""
    sgn = -1.0 if back else 1.0
    for side, off in (("L", 0.0), ("R", .5)):
        p = (phase + off) % 1.0
        if p < duty:                                  # planted: front -> back
            u = p / duty
            z = stride * (.5 - u)
            y = 0.0
            pitch = 0.0
        else:                                         # in the air: back -> front, lifted
            u = (p - duty) / (1 - duty)
            e = ease(u)
            z = stride * (-.5 + e)
            y = lift * np.sin(np.pi * u)
            pitch = 14 * np.sin(np.pi * u) * (1 if u < .5 else .4)
        ch[side + ".z"] = ch.get(side + ".z", 0.0) + sgn * z
        ch[side + ".y"] = ch.get(side + ".y", 0.0) + y
        ch[side + ".p"] = ch.get(side + ".p", 0.0) + pitch
    # the left foot is planted for phase in [0, duty): the shell leans over it
    c = np.cos(2 * np.pi * (phase - duty / 2))
    add(ch, body__z=-roll * c, root__x=sway * c, body__y=yaw * c * sgn)
    land = np.cos(4 * np.pi * phase)                  # two landings a cycle
    add(ch, root__y=-bob * (.5 + .5 * land), head__z=roll * .65 * c, head__y=-yaw * .8 * c * sgn)
    add(ch, upperFL__x=-8 * c * sgn, upperFR__x=8 * c * sgn)        # arms swing against the legs


WALK_STRIDE = .17          # model units a foot slides back per planted spell (Bile Demon walk: 40 frames)
RUN_STRIDE = .12


def walk_ch(t, period=40 / FPS):
    """The waddle, sized to read from DK2's high camera (final: the reviewers found the v5 waddle a glide
    from above): the shell rolls 16 degrees over each planted foot and swings across with it, rises as a
    foot swings through and drops as it lands, the feet lift higher, and the head lags a beat behind the
    shell (it nods down after each landing and rolls back against the shell's roll)."""
    ch = stand(body__x=3.0, head__x=-2.0)
    ph = (t / period) % 1.0
    gait(ph, ch, WALK_STRIDE, .115, roll=16.0, bob=.03, sway=.042, yaw=7.0)
    lag = 2 * np.pi * (ph - .12)
    add(ch, head__x=5.0 * np.cos(2 * lag), head__z=-6.0 * np.cos(lag - np.pi * .62), neck__y=4 * np.sin(lag),
        arms__z=6.0 + 3 * np.cos(2 * lag), jaw=1.0)
    return ch


def run_ch(t, period=24 / FPS):
    ch = {"body.x": 10.0, "head.x": -12.0, "ext": .25, "neck.x": -6}
    ph = (t / period) % 1.0
    gait(ph, ch, RUN_STRIDE, .075, duty=.5, roll=11.0, bob=.026, sway=.026, yaw=6)
    c = np.cos(2 * np.pi * ph)
    add(ch, upperFL__x=-30 * c, upperFR__x=30 * c, fores__x=-25, arms__z=10)
    return ch


def walkback_ch(t, period=26 / FPS):
    ch = stand(body__x=-6.0, head__x=4.0, ext=-.15)
    gait((t / period) % 1.0, ch, .07, .035, back=True, roll=5)
    add(ch, arms__x=-10, arms__z=12)
    return ch


def drunk_ch(t, period=60 / FPS):
    ph = (t / period) % 1.0
    ch = {"head.x": 6.0, "ext": -.1}
    gait(ph, ch, .07, .04, duty=.6, roll=11, bob=.015, sway=.035, yaw=8)
    s = np.sin(2 * np.pi * ph)
    add(ch, body__z=6 * np.sin(2 * np.pi * ph * 1 + .7), body__x=4 + 3 * s, head__z=-14 * s, head__y=10 * np.sin(4 * np.pi * ph),
        L__x=.01 * s, R__x=-.012 * s, arms__z=14 + 8 * s, jaw=6 + 4 * s)
    return ch


def transition(loop_ch, n, start, loop_period):
    """WalkStart / WalkEnd: Idle1's first frame eased into the loop (ending on its frame 0) or out of it
    (starting on its frame 0). The loop's clock slows to a stop toward the idle end, so a planted foot
    moves with the creature's speed as he gathers or loses it."""
    dur = (n - 1) / FPS
    idle0 = idle_ch(0.0)

    def f(t):
        x = t / dur
        if start:
            w = ease(x / .75)
            # the clock runs at the loop's pace at the end, slowed toward the start; ends on phase 0
            clock = loop_period - (dur - t) * (.35 + .65 * x)
            ph = clock
        else:
            w = ease((1 - x) / .75)
            ph = t * (1 - .5 * x)
        a, b = idle0, loop_ch(ph)
        out = {}
        for k in set(a) | set(b):
            out[k] = a.get(k, 0.0) * (1 - w) + b.get(k, 0.0) * w
        return out
    return f


# ------------------------------------------------------------------------------------------- idles

IDLE_N = 190                      # Bile Demon idle1


def idle_ch(t):
    dur = (IDLE_N - 1) / FPS
    return _IDLE(t % dur)


def _idle_extra(t, ch):
    breath(t, ch, (IDLE_N - 1) / FPS / 2)
    # three lazy claw taps on the plastron near the end (drumming his fingers: waiting for his money)
    for t0 in (5.0, 5.22, 5.44):
        add(ch, handFL__x=-22 * pulse(t, t0, t0 + .2))


_IDLE = K((IDLE_N - 1) / FPS, True,
          (0.0, stand()),
          (1.0, stand()),
          # the shell shrug: shoulders (the shell) up, the head sinking into it a moment, elbows out
          (1.35, stand(root__y=.022, body__x=-7, ext=-.3, arms__z=10, arms__x=-6, head__x=-8)),
          (1.75, stand(root__y=.018, body__x=-6, ext=-.25, arms__z=9, arms__x=-5, head__x=-8)),
          (2.1, stand(root__y=-.004, body__x=-2, ext=.03, arms__z=-2, head__x=-3)),
          (2.4, stand()),
          # a slow look round his dungeon: left, then right, then back to the front with a smirk
          (3.0, stand(head__y=26, neck__y=12, head__z=4)),
          (3.6, stand(head__y=26, neck__y=12, head__z=4)),
          (4.2, stand(head__y=-22, neck__y=-10, head__z=-3)),
          (4.6, stand(head__y=-22, neck__y=-10, head__z=-3)),
          (5.0, stand(head__y=0, neck__y=0, jaw=5, head__z=6)),
          (5.6, stand(jaw=0, head__z=3)),
          (6.3, stand()),
          extra=_idle_extra)


def idle():
    return Clip("idle", "Idle1", IDLE_N, True, idle_ch, donor=("idle1", 1.0, 1.0),
                source="Bile Demon idle1 (190 f): his body's breathing sway under Krefft's own shell shrug, "
                       "the look round and the claw drumming")


RUB = dict(aik=1.0, aL__x=-.14, aL__y=.09, aL__z=.075, aR__x=-.14, aR__y=.09, aR__z=.075, hands__x=-15)    # (clear of the plastron)


def rub_layer(t, c, e, period=.32):
    """The claws rubbed together (scaled by e): the wrists slide past each other, up and down."""
    r = np.sin(2 * np.pi * t / period)
    add(c, aL__y=.028 * r * e, aR__y=-.028 * r * e, aL__z=.014 * r * e, aR__z=-.014 * r * e,
        handFL__x=10 * r * e, handFR__x=-10 * r * e)


def idle2():
    """The coin flip (Bile Demon idle2, 101 frames): a coin appears in his right claw, he flips it high,
    follows it up, catches it, peers at it, smirks, and tucks it away behind the plastron's edge."""
    n = 101
    dur = (n - 1) / FPS
    up = dict(aik=1.0, aR__x=-.07, aR__y=.07, aR__z=.07, handFR__x=-25)
    ch = K(dur, False,
           (0.0, stand()),
           (0.45, stand(**up, head__x=8, head__y=-14)),
           (0.95, stand(**up, head__x=10, head__y=-16, jaw=3)),
           (1.1, stand(aik=1.0, aR__x=-.07, aR__y=.12, aR__z=.08, handFR__x=-45, head__x=-10, head__y=-10)),   # the flick
           (1.45, stand(**up, head__x=-34, neck__x=-10, head__y=-10, body__x=-7)),       # watching it climb
           (1.85, stand(**up, head__x=-2, head__y=-12)),                                 # caught
           (2.05, stand(aik=1.0, aR__x=-.1, aR__y=.16, aR__z=.06, handFR__x=-30, head__x=10, head__y=-22, ext=-.12)),
           (2.5, stand(aik=1.0, aR__x=-.1, aR__y=.16, aR__z=.06, handFR__x=-30, head__x=12, head__y=-24, head__z=8,
                       ext=-.1, jaw=6)),
           (2.85, stand(aik=1.0, aR__x=-.11, aR__y=.03, aR__z=-.02, handFR__x=10, head__y=-4, jaw=2)),   # tucked away
           (3.33, stand()),
           extra=lambda t, c: breath(t, c, 3.33, .7))
    return Clip("fidgeting", "Idle2", n, False, ch, donor=("idle2", .6, .6), props=["coin_flip"],
                source="Bile Demon idle2 (101 f) length and body sway; the coin flip is Krefft's own; the coin "
                       "is the game's Coin mesh")


def walk():
    return Clip("walk", "Walk", 41, True, walk_ch, donor=("walk", .35, 0.0),
                source="Bile Demon walk (40 + 1 f): his body's rock, at a third, under Krefft's waddle")


def walk_start():
    return Clip("walk_start", "WalkStart", 40, False, transition(walk_ch, 40, True, 40 / FPS),
                donor=("walkstart", .3, 0.0), source="Bile Demon walkstart (40 f)")


def walk_end():
    return Clip("walk_end", "WalkEnd", 40, False, transition(walk_ch, 40, False, 40 / FPS),
                donor=("walkend", .3, 0.0), source="Bile Demon walkend (40 f)")


def run():
    return Clip("running", "Run", 25, True, run_ch,
                source="the Bile Demon has no run (his slot is empty): Krefft's own hurried waddle, 24 + 1 f "
                       "(the Troll's run is 12)")


def walkback():
    return Clip("walkback", "Walkback", 27, True, walkback_ch, donor=("walkback", .35, 0.0),
                source="Bile Demon walkback (26 + 1 f)")


def walkback_start():
    return Clip("walkback_start", "WalkbackStart", 18, False, transition(walkback_ch, 18, True, 26 / FPS),
                donor=("walkbackstart", .3, 0.0), source="Bile Demon walkbackstart (18 f)")


def walkback_end():
    return Clip("walkback_end", "WalkbackEnd", 12, False, transition(walkback_ch, 12, False, 26 / FPS),
                donor=("walkbackend", .3, 0.0), source="Bile Demon walkbackend (12 f)")


def drunk():
    return Clip("staggering", "Drunk", 61, True, drunk_ch,
                source="the Bile Demon has no drunk walk: Krefft's own stagger, 60 + 1 f (the Troll's is 67)")


# ------------------------------------------------------------------------------------------- fighting

BITE_HIT = 0.6                   # seconds: the jaws close on the victim


def melee():
    """The bite: a clear wind-up (final: the reviewers wanted it readable from the high camera) - the head
    sinks deep into the shell, the shell drops and rears back on its haunches and the claws come up, a
    held, trembling coil - then the head shoots out on the neck (a snapping turtle's strike) with the jaws
    wide, snaps shut, holds the bite, tugs twice and slides back."""
    n = 46
    dur = (n - 1) / FPS
    coil = dict(ext=-1.55, body__x=-9, root__y=-.03, root__z=-.02, head__x=-12, arms__z=18, arms__x=-22,
                fores__x=-18, R__z=-.03)
    ch = K(dur, False,
           (0.0, stand()),
           (0.12, stand(ext=-.5, body__x=-3, root__y=-.01, head__x=-6, arms__z=8, arms__x=-8)),
           (0.3, stand(**coil, jaw=0)),
           (0.45, stand(**dict(coil, ext=-1.6, root__y=-.033, body__x=-10), jaw=4)),           # held, coiled
           (0.5, stand(ext=1.8, neck__x=2, head__x=-4, body__x=15, root__z=.035, root__y=-.02, jaw=40, arms__z=6,
                       arms__x=8, fores__x=0, R__z=.045, R__y=0)),
           (BITE_HIT, stand(ext=1.9, neck__x=4, head__x=0, body__x=16, root__z=.036, root__y=-.022, jaw=-3, arms__z=6,
                            arms__x=8, R__z=.045)),
           (0.7, stand(ext=1.88, neck__x=4, head__x=1, body__x=16, root__z=.036, root__y=-.022, jaw=-3, arms__z=6,
                       arms__x=8, R__z=.045)),                                             # the bite held
           (0.8, stand(ext=1.7, neck__x=4, head__x=0, head__y=12, body__x=13, root__z=.03, root__y=-.02, jaw=-3,
                       body__y=4, R__z=.04)),
           (0.92, stand(ext=1.65, neck__x=4, head__x=0, head__y=-12, body__x=13, root__z=.03, root__y=-.02, jaw=-3,
                        body__y=-4, R__z=.04)),
           (1.05, stand(ext=1.3, neck__x=3, head__x=0, head__y=4, body__x=10, root__z=.022, root__y=-.014, jaw=-2,
                        R__z=.035)),
           (1.28, stand(ext=.1, neck__x=2, head__x=-2, head__z=7, body__x=1, root__z=.004, jaw=4, R__z=.0)),
           (dur, stand()),
           extra=lambda t, c: add(c, head__z=2.5 * np.sin(t * 70) * env(t, .3, .47, .04),
                                  root__x=.004 * np.sin(t * 55) * env(t, .3, .47, .04),
                                  R__y=.03 * pulse(t, .44, .58)))
    return Clip("attack", "Melee", n, False, ch, donor=("melee", .2, 0.0),
                source="Bile Demon melee timing compressed (90 -> 46 f: Krefft is a biter, not a slammer; "
                       "his body's lunge at a fifth); the coiled wind-up and the strike are Krefft's own")


def melee2():
    """A dirty one: a backhand claw rake with the right paddle, then a shove with the shell's edge."""
    n = 40
    dur = (n - 1) / FPS
    ch = K(dur, False,
           (0.0, stand()),
           (0.3, stand(aik=1.0, aR__x=.06, aR__y=.22, aR__z=-.04, handFR__x=-30, body__y=-22, body__z=-4, head__y=-10,
                       ext=-.2, jaw=6)),
           (0.45, stand(aik=1.0, aR__x=-.13, aR__y=-.02, aR__z=.13, handFR__x=20, body__y=18, body__z=5, body__x=10,
                        root__z=.02, head__y=8, jaw=14)),
           (0.6, stand(aik=1.0, aR__x=-.1, aR__y=0, aR__z=.08, body__y=10, body__x=-4, ext=-.6, root__z=-.01, jaw=0)),
           (0.8, stand(body__x=16, body__y=-6, root__z=.04, root__y=-.01, ext=-.55, arms__z=16, L__z=.035)),
           (1.0, stand(body__x=4, ext=-.1, root__z=.01, head__z=6, jaw=4)),
           (dur, stand()))
    return Clip("attack2", "Melee2", n, False, ch,
                source="the Bile Demon has no second melee: Krefft's own claw rake and shell shove (40 f)")


SPIT_AT = 1.1                     # seconds: the gob of coins leaves his mouth (the shot is made)


def magic():
    """The coin spit: he rears back, the cheeks and throat swell with coins, a shake, then the head jerks
    forward out of the shell and he spits the gob; a wipe of the mouth with the back of his claw."""
    n = 65
    dur = (n - 1) / FPS
    wipe = dict(aik=1.0, aR__x=-.13, aR__y=.21, aR__z=.06, handFR__x=-20)
    ch = K(dur, False,
           (0.0, stand()),
           (0.25, stand(body__x=-6, head__x=-10, puff=.2, jaw=3)),
           (0.7, stand(body__x=-14, head__x=-6, root__y=.01, puff=1.0, jaw=-2, ext=.12, arms__z=14, arms__x=-12)),
           (0.95, stand(body__x=-15, head__x=-8, root__y=.012, puff=1.1, jaw=-2, ext=.12, arms__z=16, arms__x=-14)),
           (SPIT_AT - .05, stand(body__x=8, head__x=6, neck__x=6, ext=.55, puff=.9, jaw=10, arms__z=6, arms__x=6,
                                 root__z=.012)),
           (SPIT_AT + .05, stand(body__x=11, head__x=10, neck__x=8, ext=.7, puff=.15, jaw=34, arms__z=6, arms__x=8,
                                 root__z=.016)),
           (1.35, stand(body__x=6, head__x=6, ext=.35, puff=0, jaw=6, root__z=.008)),
           (1.6, stand(**wipe, head__y=-10, head__x=4)),
           (1.78, stand(**dict(wipe, aR__x=-.06), head__y=-2, head__x=2)),
           (dur, stand()),
           extra=lambda t, c: add(c, head__z=3 * np.sin(t * 60) * env(t, .6, 1.0, .1), head__y=2 * np.sin(t * 47) * env(t, .6, 1.0, .1)))
    return Clip("breathe_fire", "Magic", n, False, ch, donor=("magic", .5, 0.0), props=["coin_gob"],
                source="Bile Demon magic (65 f): his rear-back-and-release body at half, under Krefft's puff and "
                       "spit; the gob is the game's Coin mesh, the shot itself is the game's (see notes)")


def recoil_hff():
    n = 46
    dur = (n - 1) / FPS
    ch = K(dur, False,
           (0.0, stand()),
           (0.1, stand(ext=-1.0, body__x=-14, root__z=-.03, arms__x=-30, arms__z=18, fores__x=-25, head__x=4,
                       L__z=-.03)),
           (0.35, stand(ext=-1.0, body__x=-9, root__z=-.025, arms__x=-24, arms__z=16, fores__x=-20, head__x=4,
                        L__z=-.03)),
           (0.75, stand(ext=-.45, body__x=-5, root__z=-.012, arms__x=-8, arms__z=8, head__x=6, head__y=6, L__z=-.015)),
           (1.0, stand(ext=-.05, body__x=-2, head__x=8, jaw=3, head__y=-4)),
           (dur, stand()))
    return Clip("getting_hit", "RecoilHff", n, False, ch, donor=("recoil(hff)", .5, .5),
                source="Bile Demon recoil(hff) (46 f): his knock back at half, Krefft's head snapping into the shell")


def recoil_hfb():
    n = 46
    dur = (n - 1) / FPS
    ch = K(dur, False,
           (0.0, stand()),
           (0.1, stand(ext=-.8, body__x=14, root__z=.03, arms__x=12, arms__z=14, head__x=8, R__z=.03)),
           (0.4, stand(ext=-.6, body__x=9, root__z=.025, arms__x=6, arms__z=12, head__x=6, R__z=.03)),
           (0.75, stand(ext=.05, body__x=4, body__y=-10, neck__y=-24, head__y=-40, head__x=-4, jaw=6, R__z=.015)),
           (0.9, stand(ext=.1, body__x=3, body__y=-10, neck__y=-24, head__y=-42, head__x=-4, jaw=-2)),
           (1.05, stand(ext=.1, body__y=-8, neck__y=-20, head__y=-36, jaw=12)),
           (1.2, stand(ext=.05, body__y=-6, neck__y=-14, head__y=-26, jaw=0)),
           (dur, stand()))
    return Clip("being_slapped", "RecoilHfb", n, False, ch, donor=("recoil(hfb)", .5, .5),
                source="Bile Demon recoil(hfb) (46 f): the lurch forward; the glare back over the shell and the "
                       "snap are Krefft's")


def electrocute():
    n = 10
    dur = (n - 1) / FPS
    rng = np.random.default_rng(7)
    jit = rng.uniform(-1, 1, (n, 6))

    def f(t):
        i = min(int(round(t * FPS)), n - 1)
        on = 1.0 if 0 < i < n - 1 else 0.0
        j = jit[i] * on
        return stand(ext=-.6 * on, body__x=-6 * on + 5 * j[0], body__z=5 * j[1], aik=on,
                     aL__x=.05 + .02 * j[2], aL__y=.1 + .02 * j[3], aL__z=.02, aR__x=.05 + .02 * j[4], aR__y=.1 + .02 * j[5],
                     aR__z=.02, hands__x=-30 * on, jaw=18 * on + 6 * j[4], root__y=.012 * on,
                     L__y=.01 * abs(j[0]), R__y=.01 * abs(j[1]))
    return Clip("electrocute", "Electrocute", n, False, f, source="Bile Demon electrocute (10 f); Krefft's jolt")


def _on_back(**kw):
    """Lying on his shell, plastron up, head and legs pulled in, paddles in the air (a turtle on its back)."""
    # (the legs stick straight up from under the rim, soles to the sky - not folded over the plastron)
    d = {"body.x": -96.0, "ik": 0.0, "ground": 1.0, "ext": -.35, "thighL.x": -80, "thighR.x": -80, "shinL.x": 12,
         "shinR.x": 12, "footL.x": 30, "footR.x": 30, "thighL.z": 44, "thighR.z": 44, "arms.x": -35, "arms.z": 62,
         "tail.x": 90,                      # (the tail's point folded back under the rim, not standing on the plastron)
         "fores.x": -30, "head.x": 50, "head.y": 18, "neck.y": 6, "jaw": 8, "root.z": .12}
    # (the head tipped down onto his chest so his face, not his pale chin, looks up at the Keeper's camera;
    #  a little neck showing between it and the shell)     # (root.z: he lands near where he stood)
    for k, v in kw.items():
        d[k.replace("__", ".")] = v
    return d


def falling():
    n = 33
    dur = (n - 1) / FPS
    ch = K(dur, False,
           (0.0, stand(ground=1.0)),
           (0.2, stand(ground=1.0, body__x=-22, root__z=-.03, ext=-.4, arms__x=-50, arms__z=30, L__z=-.05, L__y=.01,
                       head__x=-10, jaw=14)),
           (0.45, _on_back(body__x=-80, ext=-.8, ik=.2, thighL__x=-40, thighR__x=-55, jaw=10, arms__x=-70, arms__z=45)),
           (0.62, _on_back(body__x=-100, thighL__x=-85, thighR__x=-60, shinL__x=60, arms__x=-55, arms__z=40)),
           (0.8, _on_back(body__x=-94, thighL__x=-60, thighR__x=-80)),
           (dur, _on_back()))
    return Clip("falling", "Fallback", n, False, ch, donor=("fallback", 0.0, .15, (0, 0, 1)),
                source="Bile Demon fallback (33 f): length and the drift back; the turtle's topple onto the shell "
                       "is Krefft's (the Bile Demon tips 135 degrees; a turtle stops on his back)")


def waking_up():
    """Righting himself (the paddles pushing on the floor out at his sides): rocks on the shell, a first heave that slips back, a harder heave rolling up onto his
    feet over two thirds of a second, a squash as the feet land, a head shake, a shuffle to where he stood."""
    n = 63
    dur = (n - 1) / FPS
    ch = K(dur, False,
           (0.0, _on_back()),
           (0.2, _on_back(body__z=-22, ext=-.2, thighL__x=-90, thighR__x=-72)),
           (0.42, _on_back(body__z=22, ext=-.1, thighL__x=-72, thighR__x=-90, head__y=-10)),
           (0.62, _on_back(aik=1.0, aL__x=.15, aR__x=.15, aL__y=-.02, aR__y=-.02, aL__z=-.1, aR__z=-.1, body__x=-72, body__z=-10, ext=.1, ik=.1, thighL__x=-55, thighR__x=-45, arms__x=0, arms__z=72,
                           head__x=20, jaw=10)),                         # a first heave...
           (0.8, _on_back(aik=1.0, aL__x=.15, aR__x=.15, aL__y=-.02, aR__y=-.02, aL__z=-.1, aR__z=-.1, body__x=-90, body__z=-4, ext=-.1, thighL__x=-75, thighR__x=-70, jaw=0)),    # ...slips back
           (1.0, _on_back(aik=1.0, aL__x=.15, aR__x=.15, aL__y=-.02, aR__y=-.02, aL__z=-.1, aR__z=-.1, body__x=-68, body__z=-12, ext=.15, ik=.15, thighL__x=-50, thighR__x=-40, arms__x=5,
                          arms__z=75, head__x=25, jaw=12)),             # the harder heave, paddles pushing
           (1.25, _on_back(aik=1.0, aL__x=.15, aR__x=.15, aL__y=-.02, aR__y=-.02, aL__z=-.1, aR__z=-.1, body__x=-38, body__z=-6, ext=.2, ik=.5, thighL__x=-30, thighR__x=-22, arms__x=15, arms__z=70,
                           head__x=20)),
           (1.5, stand(ground=1.0, body__x=14, root__y=-.035, ext=0, arms__x=10, arms__z=30, ik=1.0, head__x=10,
                       L__z=.02)),                                       # feet down: the squash
           (1.62, stand(ground=1.0, body__x=4, root__y=-.01, head__y=16, ext=.1, root__z=.02)),
           (1.74, stand(ground=1.0, head__y=-16, ext=.1, root__z=.04)),
           (1.86, stand(ground=1.0, head__y=10, ext=.05, root__z=.055)),
           (dur, stand(ground=1.0, root__z=.065)))     # (a shuffle forward: he ends where Idle1 stands)
    return Clip("waking_up", "GetUp", n, False, ch, donor=("getup", 0.0, .15, (0, 0, 1)),
                source="Bile Demon getup (63 f): length and the drift; rocking on the shell, a slipped heave and "
                       "rolling up onto his feet are Krefft's own")


def die():
    n = 79
    dur = (n - 1) / FPS
    ch = K(dur, False,
           (0.0, stand(ground=1.0)),
           (0.35, stand(ground=1.0, body__x=14, root__y=-.02, ext=.2, head__x=20, jaw=20, arms__x=-35, arms__z=-8,
                        fores__x=-50)),                     # clutching the plastron
           (0.7, stand(ground=1.0, body__x=-18, root__z=-.03, ext=.4, head__x=-30, jaw=30, arms__x=-60, arms__z=35,
                       L__z=-.04, L__y=.01)),
           (1.05, _on_back(body__x=-88, ext=0, ik=.15, jaw=26, head__x=-10, thighL__x=-50, thighR__x=-80)),
           (1.2, _on_back(body__x=-100, ext=.1, jaw=20, thighL__x=-95, thighR__x=-55)),
           (1.4, _on_back(body__x=-97, ext=.2, jaw=24, thighL__x=-55, thighR__x=-92)),
           (1.6, _on_back(body__x=-97, ext=.35, jaw=20, thighL__x=-85, thighR__x=-62, shinL__x=55)),
           (1.85, _on_back(body__x=-96, ext=.5, jaw=22, thighL__x=-70, thighR__x=-72, head__y=30, head__x=25)),
           (2.15, _on_back(body__x=-95, ext=-.1, jaw=26, thighL__x=-45, thighR__x=-48, thighL__z=62, thighR__z=60,
                           shinL__x=25, shinR__x=25, footL__x=0, footR__x=0, head__y=50, head__x=60, neck__y=20,
                           arms__x=-15, arms__z=85, fores__x=-5)),
           (dur, _on_back(body__x=-95, ext=-.1, jaw=30, thighL__x=-42, thighR__x=-45, thighL__z=66, thighR__z=64,
                          shinL__x=24, shinR__x=24, footL__x=0, footR__x=0, head__y=54, head__x=62, neck__y=22,
                          arms__x=-12, arms__z=90, fores__x=-4)))
    # (dead: limp, the limbs flopped out to the floor, the head lolled to one side, the jaw hanging)
    return Clip("die", "Die", n, False, ch, donor=("die", 0.0, .15, (0, 0, 1)),
                source="Bile Demon die (79 f): length and the drift; the fall onto the shell, the kicking and the "
                       "head flopping out are Krefft's")


# ------------------------------------------------------------------------------------------- moods

def happy():
    """Gloating: the claws rubbed together in front of the plastron, a wheezing chuckle, the head tipped
    back, the shell bouncing."""
    n = 130
    dur = (n - 1) / FPS

    def extra(t, c):
        rub_layer(t, c, env(t, .4, dur - .5, .3))
        ch_ = env(t, 1.0, 3.4, .2)
        add(c, jaw=7 * ch_ * (.5 + .5 * np.sin(2 * np.pi * t / .22)), root__y=.006 * ch_ * np.sin(2 * np.pi * t / .22),
            body__x=-1.5 * ch_ * np.sin(2 * np.pi * t / .22))
    ch = K(dur, False,
           (0.0, stand()),
           (0.4, stand(**RUB, head__x=-6)),
           (1.0, stand(**RUB, head__x=-18, body__x=-8, head__z=6)),
           (2.4, stand(**RUB, head__x=-20, body__x=-9, head__z=-6, head__y=10)),
           (3.4, stand(**RUB, head__x=-10, body__x=-5, head__z=4, head__y=-6)),
           (3.8, stand(**RUB, head__x=-6, body__x=-4)),
           (dur, stand()),
           extra=extra)
    return Clip("happy", "Happy", n, False, ch, donor=("happy", .5, .5),
                source="Bile Demon happy (130 f): his bounce at half; the claw rub and the chuckle are Krefft's")


def angry():
    n = 45
    dur = (n - 1) / FPS
    fist = dict(arms__x=-20, arms__z=-6, fores__x=-40, hands__x=-20)
    ch = K(dur, False,
           (0.0, stand()),
           (0.15, stand(**fist, L__y=.17, L__z=.04, L__p=18, body__z=-9, root__y=.012, ext=.3, head__x=8, jaw=20)),
           (0.25, stand(**fist, L__y=0, L__z=.03, body__z=5, root__y=-.03, ext=.4, head__x=12, jaw=-2, body__x=8)),
           (0.45, stand(**fist, L__z=.03, R__y=.17, R__z=.04, R__p=18, body__z=9, root__y=.012, ext=.35, jaw=24)),
           (0.55, stand(**fist, L__z=.03, R__y=0, R__z=.03, body__z=-5, root__y=-.03, ext=.45, jaw=-2, body__x=8)),
           (0.8, stand(**fist, L__z=.02, R__z=.02, ext=.5, jaw=28, neck__x=8, head__x=-6)),
           (1.0, stand(ext=.2, jaw=4)),
           (dur, stand()),
           extra=lambda t, c: add(c, hands__z=6 * np.sin(t * 70) * env(t, .1, .9, .1), body__y=3 * np.sin(t * 40) * env(t, .7, .95, .05)))
    return Clip("angry", "Angry", n, False, ch, donor=("angry", .3, .3),
                source="Bile Demon angry (45 f): his shudder at a third; the stamping and the snapping are Krefft's")


def celebrating():
    """The dance: a smug shell shimmy, stepping side to side, both claws up and pumping, the head bobbing."""
    n = 95
    dur = (n - 1) / FPS
    beat = dur / 6

    def f(t):
        e = env(t, 0, dur, .35)
        ph = t / beat
        s, c2 = np.sin(np.pi * ph), np.sin(2 * np.pi * ph)
        ch = stand(aik=e, aL__x=-.01, aR__x=-.01, aL__y=.2 + .04 * c2, aR__y=.2 - .04 * c2, aL__z=.05, aR__z=.05,
                   hands__x=-30 * e, body__z=10 * s * e, root__x=.025 * s * e, body__y=8 * s * e,
                   head__z=-12 * s * e, head__x=(-10 + 6 * c2) * e, jaw=6 * e * (c2 > 0),
                   root__y=-.012 * abs(c2) * e)
        k = int(ph) % 2
        u = ph % 1.0
        ch["L.y" if k == 0 else "R.y"] = .04 * np.sin(np.pi * u) * e
        ch["L.x" if k == 0 else "R.x"] = .015 * np.sin(np.pi * u) * e
        return ch
    return Clip("celebrating", "Dance", n, False, f, donor=("dance", .4, .4),
                source="Bile Demon dance (95 f): his sway at 40%; the shimmy, the steps and the claws are Krefft's")


def dejected():
    return Clip("dejected", "Dejected", 2, False, lambda t: {
        "body.x": 16.0, "root.y": -.02, "ext": -.55, "head.x": 26.0, "neck.x": 10.0, "arms.x": 12.0, "arms.z": -4.0,
        "fores.x": 8.0, "jaw": 2.0}, source="pose: Krefft's own (the Bile Demon has none)")


# ------------------------------------------------------------------------------------------- needs

def eating():
    """The chicken (the Bile Demon's own): held up in both claws, one savage turtle bite after another (the
    head darting), the last of it gulped down with the throat bulging, a pat on the plastron."""
    n = 80
    dur = (n - 1) / FPS
    hold = dict(aik=1.0, aL__x=-.13, aR__x=-.13, aL__y=.1, aR__y=.1, aL__z=.08, aR__z=.08, hands__x=-20)
    keys = [(0.0, stand()), (0.4, stand(**hold, head__x=14, ext=-.1))]
    t = .55
    for i in range(3):                                 # three bites
        keys += [(t, stand(**hold, head__x=14, ext=-.25, jaw=4)),
                 (t + .12, stand(**hold, head__x=24, ext=.35, neck__x=8, jaw=32)),
                 (t + .2, stand(**hold, head__x=26, ext=.4, neck__x=8, jaw=-2, head__y=10 * (-1) ** i)),
                 (t + .42, stand(**hold, head__x=-6, ext=0, jaw=-2, head__y=0))]
        t += .5
    keys += [(2.05, stand(head__x=-22, ext=.05, puff=.6, jaw=-2)),         # gulp
             (2.3, stand(head__x=-8, ext=0, puff=0, jaw=4, aik=1.0, aL__x=-.12, aL__y=.04, aL__z=0, handFL__x=-10)),
             (2.45, stand(aik=1.0, aL__x=-.12, aL__y=.03, aL__z=-.02, handFL__x=15, jaw=6)),
             (dur, stand())]
    return Clip("eating", "Eat", n, False, K(dur, False, *keys), donor=("eat", .3, .3), props=["chicken"],
                source="Bile Demon eat (80 f) and his chicken; the bites and the gulp are Krefft's")


def drinking():
    """The Bile Demon's bottle in his right claw: swigged with the head tipped right back, the throat
    pulsing, the bottle lowered, a satisfied wipe and a burp."""
    n = 150
    dur = (n - 1) / FPS
    lift = dict(aik=1.0, aR__x=-.08, aR__y=.1, aR__z=.06, handFR__x=-10)
    swig = dict(aik=1.0, aR__x=-.135, aR__y=.2, aR__z=.04, handFR__x=-55)
    wipe = dict(aik=1.0, aL__x=-.13, aL__y=.21, aL__z=.06, handFL__x=-20)

    def extra(t, c):
        add(c, puff=.35 * env(t, 1.4, 3.0, .15) * (.5 + .5 * np.sin(2 * np.pi * t / .45)))
        breath(t, c, dur / 2, .6)
    ch = K(dur, False,
           (0.0, stand(aik=1.0)),
           (0.6, stand(**lift, head__y=-10)),
           (1.2, stand(**swig, head__x=-12, neck__x=-6, body__x=-10, ext=.15, jaw=10)),
           (3.0, stand(**swig, head__x=-14, neck__x=-6, body__x=-11, ext=.15, jaw=10)),
           (3.5, stand(**lift, head__x=-4, jaw=0)),
           (3.9, stand(**wipe, head__y=8)),
           (4.2, stand(aik=1.0, aL__x=-.05, aL__y=.2, aL__z=.05, head__y=0)),
           (4.4, stand(aik=1.0, body__x=-8, head__x=-14, jaw=26, root__y=.008)),     # the burp
           (4.65, stand(aik=1.0, jaw=4)),
           (dur, stand(aik=1.0)),
           extra=extra)
    return Clip("drinking", "Drink", n, False, ch, donor=("drink", .3, .3), props=["bottle"],
                source="Bile Demon drink (150 f) and his bottle; the swig and the burp are Krefft's")


def drunk_idle():
    n = 120
    dur = (n - 1) / FPS

    def f(t):
        p = 2 * np.pi * t / dur
        h = pulse(t, 2.0, 2.35)                       # a hiccup
        return {"body.z": 8 * np.sin(p), "body.x": 5 + 3 * np.sin(2 * p), "root.x": .015 * np.sin(p),
                "head.z": -16 * np.sin(p + .5), "head.x": 10 + 6 * np.sin(2 * p) - 20 * h, "head.y": 12 * np.sin(p * 2 + 1),
                "ext": -.1 + .3 * h, "jaw": 8 + 6 * np.sin(3 * p) + 18 * h, "root.y": .015 * h, "puff": .4 * h,
                "arms.z": 16 + 6 * np.sin(p), "arms.x": 6 * np.sin(p + 1), "fores.x": -10,
                "L.x": .01 * np.sin(p), "R.x": -.01 * np.sin(p)}
    return Clip("drunk_idle", "DrunkIdle", n, False, f, donor=("drunkidle", .5, .5),
                source="Bile Demon drunkidle (120 f): his lurching at half; the hiccup is Krefft's")


PRAY = dict(body__x=24, root__y=-.03, ext=-.3, head__x=28, neck__x=10, aik=1.0, aL__x=-.15, aR__x=-.15, aL__y=.06,
            aR__y=.06, aL__z=.07, aR__z=.07, hands__x=-35, L__z=-.02, R__z=-.02)


def pray_ch(t, period=60 / FPS):
    s = np.sin(2 * np.pi * t / period)
    ch = stand(**PRAY)
    add(ch, body__x=5 * s, head__x=6 * s, root__y=-.006 * s, jaw=2 + 2 * s)
    return ch


def praying():
    return Clip("praying", "Pray", 61, True, pray_ch, donor=("pray", .25, 0.0),
                source="Bile Demon pray (60 + 1 f); the bow over the claws is Krefft's")


def _to_pose(target_ch, n, start):
    dur = (n - 1) / FPS
    idle0 = idle_ch(0.0)

    def f(t):
        x = t / dur
        w = ease(x) if start else ease(1 - x)
        a, b = idle0, target_ch(0.0)
        return {k: a.get(k, 0.0) * (1 - w) + b.get(k, 0.0) * w for k in set(a) | set(b)}
    return f


def praying_start():
    return Clip("praying_start", "PrayStart", 56, False, _to_pose(pray_ch, 56, True), donor=("praystart", .25, 0.0),
                source="Bile Demon praystart (56 f)")


def praying_end():
    return Clip("praying_end", "PrayEnd", 55, False, _to_pose(pray_ch, 55, False), donor=("prayend", .25, 0.0),
                source="Bile Demon prayend (55 f)")


SLEEP_FPS = 15


def sleeping():
    """Asleep sitting in the Bile Demon's lair bed, tucked into his shell: the head and arms pulled in, the
    legs folded in front, a slow breath that lifts the shell; halfway, one eye peeks out and he tucks back."""
    n = 120
    dur = (n - 1) / SLEEP_FPS
    sit = dict(ik=0.0, ground=1.0, body__x=-40, ext=-1.7, head__x=22, thighL__x=-82, thighR__x=-82, thighL__z=14,
               thighR__z=14, shinL__x=70, shinR__x=70, footL__x=-20, footR__x=-20, aik=1.0, aL__x=-.08, aR__x=-.08,
               aL__y=-.07, aR__y=-.07, aL__z=-.06, aR__z=-.06, hands__x=35)          # the claws folded on his belly

    def f(t):
        s = np.sin(2 * np.pi * t / (dur / 2))
        p = pulse(t, 3.6, 5.0)
        ch = {k.replace("__", "."): v for k, v in sit.items()}
        add(ch, body__x=-2.5 * s, root__y=.004 * s, ext=1.0 * p, head__y=-14 * p, head__x=-10 * p)
        return ch
    return Clip("sleeping", "Sleep", n, True, f, fps=SLEEP_FPS, donor=("bed(sleeping)", .3, 0.0), props=["bed"],
                source="Bile Demon bed(sleeping) (120 f at 15 fps) and his lair bed: he sleeps sitting up in it, "
                       "like the Bile Demon; tucked into the shell is Krefft's")


def being_picked_up():
    """In the Keeper's hand, held by the shell's front rim: legs and arms paddling, the head half pulled
    in, peering round."""
    n = 61
    dur = (n - 1) / FPS

    def f(t):
        p = 2 * np.pi * t / dur
        return {"ik": 0.0, "body.x": 8 + 4 * np.sin(p), "body.z": 6 * np.sin(p), "ext": -.45 + .15 * np.sin(2 * p),
                "head.y": 25 * np.sin(p), "head.x": 6, "jaw": 4 + 4 * np.sin(2 * p),
                "thighL.x": -25 + 30 * np.sin(2 * p), "thighR.x": -25 - 30 * np.sin(2 * p), "thighL.z": 8, "thighR.z": 8,
                "shinL.x": 25 + 15 * np.cos(2 * p), "shinR.x": 25 - 15 * np.cos(2 * p),
                "footL.x": 20, "footR.x": 20,
                "upperFL.x": -20 - 12 * np.sin(2 * p + 1), "upperFR.x": -20 + 12 * np.sin(2 * p + 1), "arms.z": 8,
                "fores.x": -20, "tail.x": 15 * np.sin(2 * p)}
    return Clip("being_picked_up", "Swing", n, True, f, donor=("swing", .4, 0.0),
                source="Bile Demon swing (60 + 1 f): his swing at 40%; the paddling is Krefft's")


def torture():
    """On the Torture Chamber's wheel where the Bile Demon hangs (his body's turn from his own file, his shell
    centred where the Bile Demon's body is): lying back spread-eagled to the straps, the head pulled into the
    shell at each jolt, then out again with a smug sneer and a snap. The debt collector is not bothered."""
    n = 100
    dur = (n - 1) / FPS

    def f(t):
        peek = env(t, .9, 2.4, .35)
        jolt = pulse(t, 2.5, 2.75) + pulse(t, .1, .35)
        return {"ik": 0.0, "body.x": -38 - 4 * jolt, "root.y": .12, "tail.x": 90, "aik": 1.0, "aL.x": .1, "aR.x": .1, "aL.y": .2,
                "aR.y": .2, "aL.z": -.04, "aR.z": -.04, "thighL.z": 50 + 8 * jolt, "thighR.z": 50 + 8 * jolt,
                "thighL.x": -10, "thighR.x": -10, "shinL.x": 10, "shinR.x": 10,
                "ext": -1.0 + .8 * peek, "head.y": 20 * peek * np.sin(2 * np.pi * (t - .9) / 1.5),
                "jaw": 6 * peek + 14 * pulse(t, 1.8, 2.1), "head.z": 8 * peek, "head.x": 18 + 12 * peek, "body.z": 4 * jolt,
                "hands.x": 25 * jolt,
                "root.x": .012 * jolt * np.sin(t * 70)}
    return Clip("torture", "Torture", n, False, f, donor=("torture", 1.0, 0.0),
                source="Bile Demon Bil-Torture (100 f): his body's turn on the wheel in full, Krefft's shell centred "
                       "where the Bile Demon's body is; strapped "
                       "out and riding it in his shell is Krefft's")


def entrance():
    """Out of the Portal: he rises up out of the floor where the Bile Demon does (his body's rise from his
    own file), tucked in his shell; the head pops out, looks left and right, a shell shrug, a smug claw rub."""
    n = 148
    dur = (n - 1) / FPS
    rub = RUB

    def extra(t, c):
        e = env(t, 3.7, 4.5, .15)
        rub_layer(t, c, e)
        add(c, jaw=5 * e * (np.sin(2 * np.pi * t / .32) > 0))
    ch = K(dur, False,
           (0.0, stand(ext=-1.0, arms__x=-30, arms__z=-10, fores__x=-60, feet=1.0)),
           (1.1, stand(ext=-1.0, arms__x=-30, arms__z=-10, fores__x=-60, feet=1.0)),
           (1.5, stand(ext=-.9, arms__x=-20, arms__z=-5, fores__x=-40, feet=1.0)),
           (1.8, stand(ext=.15, head__x=-6, feet=1.0)),            # pop
           (2.0, stand(ext=0, feet=1.0)),
           (2.4, stand(head__y=34, neck__y=14, feet=1.0)),
           (2.85, stand(head__y=-30, neck__y=-12, feet=1.0)),
           (3.25, stand(head__y=0, feet=1.0)),
           (3.45, stand(root__y=.02, body__x=-7, ext=-.3, arms__z=10, feet=1.0)),
           (3.75, stand(**rub, feet=1.0, head__x=-10)),
           (4.5, stand(**rub, feet=1.0, head__x=-12, head__z=6)),
           (dur, stand(feet=1.0)),
           extra=extra)
    return Clip("entrance", "Entrance", n, False, ch, donor=("entrance", 0.0, 1.0, (0, 1, 0)),
                source="Bile Demon bil-Entrance (148 f): his rise out of the floor in full; the tucked rise, the head popping out and the look round are Krefft's")


# The fruit machine of the Gamble preview: the game's Bar_Fruit_Machine (static; the Casino has no lever
# animation) turned 180 degrees to face him and set at GAMBLE_MACHINE, in the exported file's model space
# (x to his left, y up, z forward; krefft_v5_review draws it there and turns its lever by lever_angle()).
# Its lever, in the machine's own (unturned) file: the hub at KMF (-0.235, 0, -0.314), the knob at
# (-0.235, 0.30, -0.49); turned, the knob points back toward him from the hub, 30.4 degrees above level.
GAMBLE_MACHINE = (-0.09, 0.0, 0.40)
GAMBLE_YAW = 180.0
LEVER_X, LEVER_HUB_UP, LEVER_LEN, LEVER_REST = .235, .314, .3478, 30.4
LEVER_PULLED = -20.0              # degrees above level at the bottom of the pull
# file model space = PLAN_SCALE * rig space + PLAN_SHIFT (krefft_anims_build.plan(): scale 0.9254 and the
# offset of the Bile Demon's idle (tests/test_krefft_final checks it against plan())
PLAN_SCALE = 0.9253955828468413
PLAN_SHIFT = np.array([0.00032761, -0.00370158, -0.1751999])
GRIP = np.array([.0, .018, -.045])      # the wrist just behind and above the knob: the paddle closes over it
GRIP_PULLED = np.array([.0, -.004, -.03])   # at the bottom of the pull: lower, so the claws close over the knob


def grip(angle):
    """The wrist's offset from the knob, sliding from GRIP to GRIP_PULLED as the lever comes down."""
    u = float(np.clip((LEVER_REST - angle) / (LEVER_REST - LEVER_PULLED), 0, 1))
    return GRIP * (1 - u) + GRIP_PULLED * u


def lever_hub():
    return np.array([GAMBLE_MACHINE[0] - LEVER_X, LEVER_HUB_UP, GAMBLE_MACHINE[2]])


def lever_knob(angle):
    a = np.radians(angle)
    return lever_hub() + LEVER_LEN * np.array([0.0, np.sin(a), -np.cos(a)])


# (t, lever angle): reach (rest), the hitch up, the hard yank, held at the bottom, let go and it springs back
LEVER_KEYS = ((0.0, LEVER_REST), (0.5, LEVER_REST), (0.6, LEVER_REST + 6), (0.8, LEVER_PULLED),
              (0.95, LEVER_PULLED), (1.1, LEVER_REST + 9), (1.22, LEVER_REST - 3), (1.32, LEVER_REST))
GRIP_ON, GRIP_OFF = (.28, .46), (.95, 1.03)      # the claw closing on the knob, and letting go


def lever_angle(t):
    ts, vs = zip(*LEVER_KEYS)
    i = int(np.searchsorted(ts, t, side="right")) - 1
    if i >= len(ts) - 1:
        return vs[-1]
    u = ease((t - ts[i]) / (ts[i + 1] - ts[i]))
    return vs[i] + (vs[i + 1] - vs[i]) * u


def _lock_right_claw(c, target, w):
    """Point the right wrist's IK at `target` (file model space) with weight w: the offset the rig wants
    (krefft_rig.solve: target = root + R_body @ (bind wrist - bind root + offset), mirrored on the right)."""
    from scripts import krefft_rig as KR
    sk = KR.model()[1]
    H, r, ha = sk.head, sk.index["root"], sk.index["handFR"]
    Rb = KR.euler(c.get("body.x", 0.0), c.get("body.y", 0.0), c.get("body.z", 0.0))
    root = H[r] + np.array([c.get("root." + a, 0.0) for a in "xyz"])
    tgt = (np.asarray(target, float) - PLAN_SHIFT) / PLAN_SCALE
    off = Rb.T @ (tgt - root) - (H[ha] - H[r])
    want = {"aR.x": -off[0], "aR.y": off[1], "aR.z": off[2]}
    for k, v in want.items():
        c[k] = c.get(k, 0.0) * (1 - w) + v * w
    c["aik"] = max(c.get("aik", 0.0), w)


def gamble():
    """At a fruit machine (the game's Bar_Fruit_Machine, front right of him in the preview): he reaches out
    to the side for the lever's knob, hitches it up, yanks it right down with his whole shell dipping, holds
    it at the bottom, lets it spring back, watches the reels spin with his head following them, and on the
    win rubs his claws and chuckles."""
    n = 91
    dur = (n - 1) / FPS
    lean = dict(body__y=-14, body__z=-7, root__x=-.045, root__z=.045)      # turned and leaning to the machine

    def extra(t, c):
        breath(t, c, dur, .6)
        # the left foot shuffles a step after the shell as he leans to the machine, and back when he lets go
        st = ease((t - .08) / .22) * (1 - ease((t - 1.05) / .25))
        add(c, L__x=-.04 * st, L__z=.045 * st, L__y=.03 * (pulse(t, .08, .3) + pulse(t, 1.05, 1.3)))
        on = ease((t - GRIP_ON[0]) / (GRIP_ON[1] - GRIP_ON[0])) * (1 - ease((t - GRIP_OFF[0]) / (GRIP_OFF[1] - GRIP_OFF[0])))
        if on > 0:
            a = lever_angle(t)
            _lock_right_claw(c, lever_knob(a) + grip(a), on)
        add(c, handFR__x=-42 * on)                                           # the paddle curled over the knob
        reel = env(t, 1.15, 2.0, .1)
        # eyes on the spinning reels, leaning in to them, then a greedy jerk of the head as they stop (1.95)
        add(c, head__y=5 * np.sin(t * 34) * reel, head__x=3 * np.sin(t * 21) * reel + 10 * pulse(t, 1.85, 2.15),
            ext=.25 * pulse(t, 1.85, 2.15))
        e = env(t, 2.15, 2.75, .1)
        rub_layer(t, c, e, .3)
        # the chuckle: jaw and head bobbing together while the claws rub
        bob = (np.sin(2 * np.pi * t / .2) > 0)
        add(c, jaw=7 * e * bob, head__x=-5 * e * bob, root__y=.004 * e * bob)
    ch = K(dur, True,
           (0.0, stand()),
           (0.3, stand(**lean, head__y=-22, head__x=-6)),                    # eyes on the knob, reaching
           (0.5, stand(**lean, head__y=-20, head__x=-4, root__y=.004)),
           (0.6, stand(**dict(lean, root__y=.014), head__y=-16, head__x=-10, body__x=-4, ext=-.15)),   # the hitch up
           (0.8, stand(**dict(lean, root__y=-.038, root__z=.06), body__x=16, head__y=-12, head__x=10, ext=.2,
                       jaw=8)),                                              # the yank: the shell dips hard
           (0.95, stand(**dict(lean, root__y=-.034, root__z=.06), body__x=14, head__y=-12, head__x=8, ext=.2,
                        jaw=4)),                                             # held at the bottom
           (1.15, stand(body__y=-8, body__z=-2, root__x=-.01, head__y=-14, head__x=-8, ext=.3, body__x=4)),
           (2.0, stand(body__y=-8, body__z=-2, root__x=-.01, head__y=-14, head__x=-8, ext=.35, body__x=4)),
           (2.15, stand(**RUB, head__x=-14, body__x=-6)),
           (2.75, stand(**RUB, head__x=-16, body__x=-7, head__z=5)),
           (2.95, stand()),
           (dur, stand()),
           extra=extra)
    return Clip("gambling", "Gamble", n, True, ch,
                source="no Bile Demon (or any DK2) gambling clip exists: Krefft's own (the lever pull follows the "
                       "game's Bar_Fruit_Machine lever, set beside him in the preview)")


CLIPS = {c.__name__: c for c in (idle, idle2, walk, walk_start, walk_end, run, walkback, walkback_start, walkback_end,
                                 drunk, melee, melee2, magic, recoil_hff, recoil_hfb, electrocute, falling, waking_up,
                                 die, happy, angry, celebrating, dejected, eating, drinking, drunk_idle, praying,
                                 praying_start, praying_end, sleeping, being_picked_up, torture, entrance, gamble)}
