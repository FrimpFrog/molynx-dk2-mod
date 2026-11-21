"""Molynx's own sounds: one file for every sound of the Bile Demon's sound set (the set his record
still plays: Data/Sound/Sfx/Global/BileHD.sdt with creature_bile_demonSFX.map, sound category
CREATURE_BILE_DEMON), each made for the action it goes with.

Second version (2026-11-08). The first was built from the MegaGlest Behemoth's and Golem's unit
replies, which are spoken lines (the user heard "yes" in the angry sound). This one has no unit
replies at all:

- his voice is synthesised here (`voice`): a deep, rough source (a glottal pulse at 45-120 Hz, a
  sub-octave growl, breath and a crackling roughness) through the mouth shapes of a very large body
  (formants about 0.8 of a man's). It can only make grunts, growls, groans, roars and hums, never
  words, and each contour is written for its action and animation length (ANIMS below);
- under it, non-verbal MegaGlest sounds (CC BY-SA 3.0), pitched down: the Bull (snorts, bellows),
  the Pig (grunts, for chewing and laughing), the Daemon's attack and death roars, the Behemoth's
  attack and death roars, and the Golem's, Catapult's and Machine's stone and impact sounds;
- and synthesised stone: thuds, grinding, gravel, crumbling and splitting rock.

The sleep sound is the first version's (the user liked it).

`build_sounds(source_dirs, out_dir)` writes `<role>.wav` (22050 Hz, mono, 16-bit, the game's own
sample format) for every role in ROLES and returns {role: seconds}. ROLES maps each Bile Demon sample
(BileHD.sdt order) to the file that takes its place. Nothing here changes the game's files.

LICENCE: GPL-3.0 (code). Sounds made with MegaGlest data are CC BY-SA 3.0.
"""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal

RATE = 22050
RNG_SEED = 47

# animation lengths in seconds (30 fps; molynx_assets export, tests/molynx_files.ANIM_INFO)
ANIMS = {"attack": 41 / 30, "die": 31 / 30, "eating": 83 / 30, "drinking": 83 / 30, "angry": 23 / 30,
         "happy": 59 / 30, "getting_hit": 17 / 30, "being_slapped": 35 / 30, "falling": 15 / 30,
         "sleeping": 55 / 30, "torture": 100 / 30, "ground_slam": 61 / 30, "being_picked_up": 61 / 30,
         "praying": 121 / 30, "training": 121 / 30, "waking_up": 46 / 30}
SLAM_IMPACT = 34 / 30            # the mace hits the floor (molynx_record.GROUND_SLAM_IMPACT_FRAME)


def load(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0


def save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------- building blocks

def cut(x, a, b=None):
    return x[int(a * RATE):None if b is None else int(b * RATE)].copy()


def pitch(x, semitones):
    """Tape-style shift: lower is also slower, as a bigger body would sound."""
    if semitones == 0:
        return x
    n = max(1, int(round(len(x) / 2 ** (semitones / 12))))
    return signal.resample(x, n)


def filt(x, kind, freq, order=4):
    return signal.sosfilt(signal.butter(order, freq, btype=kind, fs=RATE, output="sos"), x)


def fade(x, fin=0.005, fout=0.04):
    x = x.copy()
    a, b = int(fin * RATE), min(int(fout * RATE), len(x))
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b) ** 2
    return x


def mix(*layers):
    """(signal, start seconds, gain dB) layers summed."""
    end = max(int(s * RATE) + len(x) for x, s, _g in layers)
    out = np.zeros(end)
    for x, s, g in layers:
        i = int(s * RATE)
        out[i:i + len(x)] += x * 10 ** (g / 20)
    return out


_ROOM = None


def room(x, wet=0.12):
    """A short stone-room tail (0.35 s), so the made and the recorded parts sit in one space."""
    global _ROOM
    if _ROOM is None:
        rng = np.random.default_rng(RNG_SEED + 999)
        t = np.arange(int(0.35 * RATE)) / RATE
        ir = filt(rng.standard_normal(len(t)), "bandpass", (150, 4000)) * np.exp(-t * 14)
        _ROOM = ir / np.sqrt(np.sum(ir ** 2))
    tail = signal.fftconvolve(x, _ROOM)
    out = np.zeros(len(tail))
    out[:len(x)] += x
    return out + wet * tail


def finish(x, rms, fout=0.04, wet=0.12):
    """Add the room (unless wet is 0), remove DC, fade the ends, bring to `rms`, and keep peaks under
    0.97 with a soft limiter."""
    if wet:
        x = room(x, wet)
        x = x[:len(x) - int(0.2 * RATE)] if len(x) > int(0.5 * RATE) else x
        fout = max(fout, 0.08)
    x = fade(filt(x, "highpass", 30, 2), 0.004, fout)
    x = x * (rms / (np.sqrt(np.mean(x ** 2)) or 1.0))
    over = np.abs(x) > 0.9
    x[over] = np.sign(x[over]) * (0.9 + 0.07 * np.tanh((np.abs(x[over]) - 0.9) / 0.07))
    return x


def curve(points, n):
    """Piecewise-linear curve over n samples from [(fraction 0..1, value), ...]."""
    xs, ys = zip(*points)
    return np.interp(np.linspace(0, 1, n), xs, ys)


def _lp_noise(rng, n, hz):
    return filt(rng.standard_normal(n), "lowpass", hz, 2) * np.sqrt(RATE / (2 * hz))


# mouth shapes for a very large body: (F1, F2, F3) of a man scaled by 0.8
VOWELS = {"u": (240, 600, 1900), "o": (360, 640, 2000), "a": (560, 900, 2000), "e": (400, 1450, 2000),
          "m": (200, 900, 1900)}
_BW = (70, 110, 180)


def _formants(x, vowel):
    fs = VOWELS[vowel]
    out = np.zeros_like(x)
    for f, bw, g in zip(fs, _BW, (1.0, 0.55, 0.25)):
        b, a = signal.iirpeak(f, f / bw, fs=RATE)
        out += g * signal.lfilter(b, a, x)
    if vowel == "m":                       # a closed-mouth hum: mostly the low resonance
        out = filt(out, "lowpass", 500, 2) * 1.6
    return out


def voice(dur, f0, vowels, amp, rough=0.4, sub=0.4, breath=0.15, seed=0):
    """A grunt, growl, groan or roar. f0 and amp: [(fraction, value)]; vowels: [(fraction, name)],
    crossfaded between neighbours. rough: crackle and growl; sub: the sub-octave; breath: air."""
    rng = np.random.default_rng(RNG_SEED + 500 + seed)
    n = int(dur * RATE)
    up = 4                                                     # build the pulse 4x oversampled
    f = curve(f0, n * up) * (1 + 0.012 * _lp_noise(rng, n * up, 6) + 0.012 * _lp_noise(rng, n * up, 40))
    ph = np.cumsum(f / (RATE * up))
    saw = 2 * (ph % 1) - 1
    sub_saw = 2 * ((ph / 2) % 1) - 1
    src = signal.resample_poly(saw + sub * sub_saw, 1, up)[:n]
    src = filt(src, "lowpass", 3500, 2)
    flutter = 1 + rough * 0.8 * np.clip(_lp_noise(rng, n, 35), -1.5, 1.5)
    src = src * flutter + breath * filt(rng.standard_normal(n), "bandpass", (300, 3000))
    # crackle: the vocal fry of a voice made of stone
    pops = (rng.random(n) < rough * 0.004) * rng.standard_normal(n)
    src += filt(pops, "bandpass", (150, 2500)) * rough * 2.5
    shaped = {v: _formants(src, v) for v in {v for _p, v in vowels}}
    pos = np.linspace(0, 1, n)
    ps = [p for p, _v in vowels]
    out = np.zeros(n)
    for i, (_p, v) in enumerate(vowels):       # piecewise-linear crossfade between the vowel points
        out += np.interp(pos, ps, np.eye(len(vowels))[i]) * shaped[v]
    env = curve(amp, n)
    return fade(out * env, 0.01, 0.03)


def thud(dur=0.5, f0=70, f1=38, noise=0.5, seed=0):
    """A heavy stone hitting the floor: a falling low sine plus a short low-passed noise crack."""
    rng = np.random.default_rng(RNG_SEED + seed)
    t = np.arange(int(dur * RATE)) / RATE
    freq = f1 + (f0 - f1) * np.exp(-t * 18)
    body = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * 7)
    crack = filt(rng.standard_normal(len(t)), "lowpass", 1800) * np.exp(-t * 40)
    grit = filt(rng.standard_normal(len(t)), "bandpass", (300, 3000)) * np.exp(-t * 14) * 0.4
    return body + noise * crack + noise * grit


def gravel(dur, density=60, bright=(800, 5000), decay=0.35, seed=0):
    """Loose stones and chips: many tiny clicks thinning out over `decay` seconds."""
    rng = np.random.default_rng(RNG_SEED + 300 + seed)
    n = int(dur * RATE)
    out = np.zeros(n)
    t = 0.0
    while True:
        t += rng.exponential(1 / density) * np.exp(t / decay * 0.6)
        if t >= dur:
            break
        g = int(rng.integers(40, 260))
        i = int(t * RATE)
        if i + g >= n:
            break
        grain = filt(rng.standard_normal(g), "bandpass", bright) * np.exp(-np.arange(g) / (g / 5))
        out[i:i + g] += grain * rng.uniform(0.2, 1) * np.exp(-t / decay)
    return out


def grind(dur, seed=0, low=90, high=1300):
    """Stone dragged over stone: low band-passed noise with an uneven, grainy loudness."""
    rng = np.random.default_rng(RNG_SEED + 100 + seed)
    n = int(dur * RATE)
    base = filt(rng.standard_normal(n), "bandpass", (low, high))
    grains = filt(np.abs(rng.standard_normal(n)) ** 3, "lowpass", 30, 2)
    env = np.minimum(1, np.arange(n) / (0.06 * RATE)) * np.minimum(1, (n - np.arange(n)) / (0.1 * RATE))
    return base * (0.35 + grains / (grains.max() or 1)) * env


def _scrape_v1(dur, seed=0):
    """The first version's stone scrape, kept exactly for the sleep sound."""
    rng = np.random.default_rng(RNG_SEED + 100 + seed)
    n = int(dur * RATE)
    base = filt(rng.standard_normal(n), "bandpass", (120, 1400))
    grains = filt(np.abs(rng.standard_normal(n)) ** 3, "lowpass", 35, 2)
    env = np.minimum(1, np.arange(n) / (0.08 * RATE)) * np.minimum(1, (n - np.arange(n)) / (0.12 * RATE))
    return base * (0.4 + grains / (grains.max() or 1)) * env


def crunch(dur, bites, seed=0):
    """Teeth on rock and bone: short bright cracks in quick bunches."""
    rng = np.random.default_rng(RNG_SEED + 200 + seed)
    n = int(dur * RATE)
    out = np.zeros(n)
    for k in range(bites):
        at = int((k + 0.15 + 0.5 * rng.random()) / bites * n * 0.9)
        for _ in range(rng.integers(3, 7)):
            g = int(rng.integers(150, 600))
            pos = min(n - g, at + int(rng.integers(0, 900)))
            grain = filt(rng.standard_normal(g), "bandpass", (900, 6000)) * np.exp(-np.arange(g) / (g / 4))
            out[pos:pos + g] += grain * rng.uniform(0.3, 1)
    return out


def gulp(seed=0):
    """One swallow: a low, quickly falling resonant 'glunk'."""
    rng = np.random.default_rng(RNG_SEED + 400 + seed)
    n = int(0.22 * RATE)
    t = np.arange(n) / RATE
    f = 170 * np.exp(-t * 9) + 65
    tone = np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t * 18)
    throat = filt(rng.standard_normal(n), "bandpass", (120, 700)) * np.exp(-t * 25) * 0.6
    click = filt(rng.standard_normal(n), "bandpass", (300, 1500)) * np.exp(-t * 90) * 0.4
    return tone + throat + click


def clank(seed=0):
    """Iron bars rattled: a few inharmonic metal partials."""
    rng = np.random.default_rng(RNG_SEED + 600 + seed)
    n = int(0.5 * RATE)
    t = np.arange(n) / RATE
    out = sum(np.sin(2 * np.pi * f * t + rng.random() * 6) * np.exp(-t * d)
              for f, d in ((420, 9), (1130, 12), (1870, 16), (2950, 22)))
    return out * (1 + 0.3 * rng.standard_normal(n) * np.exp(-t * 80))


# ---------------------------------------------------------------- the set

class Sources:
    """MegaGlest sound files by name, looked up under the given folders."""

    def __init__(self, dirs):
        self.dirs, self._cache, self._index = [Path(d) for d in dirs], {}, None

    def __call__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            if self._index is None:
                self._index = {}
                for d in self.dirs:
                    for p in d.rglob("*.wav"):
                        self._index.setdefault(p.stem, p)
            if name not in self._index:
                raise FileNotFoundError(f"{name}.wav is not under {', '.join(map(str, self.dirs))}")
            self._cache[name] = load(self._index[name])
        return self._cache[name]


def recipes(s: Sources) -> dict[str, np.ndarray]:
    S = lambda n, a=0.0, b=None, st=0: pitch(cut(s(n), a, b), st)
    out = {}
    V = voice

    # -- voice and moods ------------------------------------------------------------------------
    # hello: a deep, satisfied "hrmm" as he steps out or is clicked: closed mouth opening to "o"
    out["hello"] = finish(mix(
        (V(0.75, [(0, 70), (0.4, 78), (1, 62)], [(0, "m"), (0.45, "m"), (1, "o")],
           [(0, 0), (0.12, 1), (0.7, 0.9), (1, 0)], rough=0.35, sub=0.5), 0, 0),
        (filt(S("bull_select2", 0.0, 0.55, -7), "lowpass", 1200), 0.05, -12),
        (grind(0.6, 1), 0.05, -16)), 0.11, 0.12)
    # angry (23 frames): a sharp snarl, rising then cut off, nostrils flaring (bull snort first)
    out["angry"] = finish(mix(
        (filt(S("bull_select1", 0.0, 0.35, -3), "bandpass", (150, 3000)), 0, -2),
        (V(0.62, [(0, 85), (0.35, 120), (1, 70)], [(0, "e"), (0.3, "a"), (1, "a")],
           [(0, 0), (0.1, 1), (0.75, 0.9), (1, 0)], rough=0.9, sub=0.6, breath=0.3, seed=1), 0.18, 0),
        (filt(S("daemon_attack6", 0.0, 0.6, -6), "lowpass", 2500), 0.2, -10)), 0.14, 0.08)
    # happy (59 frames): a slow, rumbling three-beat chuckle, with a pig's grunt under each beat
    hah = lambda k: V(0.26, [(0, 92 - 6 * k), (1, 78 - 6 * k)], [(0, "a"), (1, "o")],
                      [(0, 0), (0.15, 1), (1, 0)], rough=0.45, sub=0.45, breath=0.2, seed=10 + k)
    out["happy_1"] = finish(mix(*[(hah(k), 0.05 + 0.38 * k, -1.5 * k) for k in range(4)],
                                (S("pig_ack1", 0.0, 0.7, -9), 0.0, -14)), 0.11, 0.15)
    out["happy_2"] = finish(mix(*[(hah(k), 0.05 + 0.32 * k, -2 * k) for k in range(2)],
                                (S("pig_select1", 0.0, 0.6, -9), 0.0, -14)), 0.11, 0.1)
    # sulk: a long sigh out through the nose, falling
    out["sulk"] = finish(mix(
        (V(0.95, [(0, 72), (1, 50)], [(0, "m"), (1, "m")], [(0, 0), (0.2, 0.8), (1, 0)],
           rough=0.2, sub=0.3, breath=0.6, seed=2), 0, 0),
        (filt(S("bull_ack1", 0.2, 1.4, -6), "lowpass", 900), 0.0, -14)), 0.07, 0.2)
    # unhappy: a groan that sinks
    out["unhappy_1"] = finish(V(1.2, [(0, 88), (0.3, 92), (1, 52)], [(0, "a"), (0.4, "o"), (1, "u")],
                                [(0, 0), (0.15, 1), (0.8, 0.6), (1, 0)], rough=0.5, sub=0.5, seed=3), 0.09, 0.2)
    out["unhappy_2"] = finish(V(0.7, [(0, 80), (1, 56)], [(0, "o"), (1, "u")], [(0, 0), (0.2, 1), (1, 0)],
                                rough=0.4, sub=0.5, seed=4), 0.08, 0.12)
    # praying (121 frames, loops): a low chant "ommm", steady, with the Temple's slow swell
    vib = lambda base, d: [(k / 8, base + (2 if k % 2 else -2)) for k in range(9)]
    out["pray_1"] = finish(mix(
        (V(2.4, vib(64, 2.4), [(0, "o"), (0.35, "m"), (1, "m")], [(0, 0), (0.15, 0.9), (0.8, 1), (1, 0)],
           rough=0.15, sub=0.35, breath=0.1, seed=5), 0, 0),
        (V(2.4, vib(96, 2.4), [(0, "o"), (0.35, "m"), (1, "m")], [(0, 0), (0.25, 0.4), (0.8, 0.4), (1, 0)],
           rough=0.1, sub=0.0, breath=0.05, seed=6), 0, -9)), 0.08, 0.4)
    out["pray_2"] = finish(V(1.4, vib(58, 1.4), [(0, "m"), (1, "m")], [(0, 0), (0.2, 1), (0.8, 0.9), (1, 0)],
                             rough=0.15, sub=0.4, breath=0.1, seed=7), 0.07, 0.3)
    # casino: a pleased grunt and a stone fist on the table / a disgusted bellow and a slam
    out["gamble_win"] = finish(mix((hah(0), 0, 0), (hah(1), 0.3, -1), (thud(0.4, 120, 60, 0.6, 11), 0.62, -6)), 0.12, 0.1)
    out["gamble_lose"] = finish(mix(
        (V(0.8, [(0, 95), (0.5, 80), (1, 55)], [(0, "a"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)],
           rough=0.7, sub=0.6, seed=8), 0, 0),
        (thud(0.6, 90, 40, 0.8, 12), 0.55, -3), (gravel(0.6, 50, seed=12), 0.58, -10)), 0.12, 0.15)
    # sleep: kept from the first version (the user liked it)
    snore = filt(pitch(cut(s("behemoth_ack6"), 0.15, 1.10), -9), "lowpass", 900)
    out["sleep"] = finish(mix((snore * np.hanning(len(snore)), 0, 0),
                              (filt(_scrape_v1(0.6, 9), "lowpass", 600), 0.75, -10)), 0.085, 0.3, wet=0)

    # -- fighting -------------------------------------------------------------------------------
    # charge: a bull's bellow, deeper, with his roar on top, building
    out["charge"] = finish(mix(
        (filt(S("bull_attack4", 0.0, 1.0, -5), "lowpass", 2500), 0, -2),
        (V(0.75, [(0, 70), (1, 105)], [(0, "o"), (1, "a")], [(0, 0), (0.2, 0.8), (0.85, 1), (1, 0)],
           rough=0.8, sub=0.6, breath=0.25, seed=20), 0.0, 0)), 0.15, 0.08)
    out["charge_roar_1"] = finish(mix(
        (V(1.45, [(0, 70), (0.3, 112), (0.8, 104), (1, 80)], [(0, "o"), (0.25, "a"), (1, "a")],
           [(0, 0), (0.12, 1), (0.85, 0.9), (1, 0)], rough=0.9, sub=0.7, breath=0.3, seed=21), 0, 0),
        (filt(S("daemon_attack3", 0.0, 0.98, -7), "lowpass", 2400), 0.1, -8),
        (filt(S("bull_attack1", 0.2, 1.5, -6), "lowpass", 1500), 0.05, -10)), 0.11, 0.2)
    out["charge_roar_2"] = finish(mix(
        (V(1.25, [(0, 60), (0.5, 98), (1, 72)], [(0, "u"), (0.4, "a"), (1, "o")],
           [(0, 0), (0.2, 1), (0.85, 0.85), (1, 0)], rough=0.85, sub=0.7, breath=0.3, seed=22), 0, 0),
        (filt(S("daemon_attack7", 0.0, 0.85, -6), "lowpass", 2400), 0.15, -9)), 0.11, 0.2)
    # swings (the attack clip is 41 frames; the grunt sits on the swing, a short hard exhale)
    for k in range(1, 5):
        out[f"melee_{k}"] = finish(mix(
            (V(0.32 + 0.04 * k, [(0, 100 - 5 * k), (1, 70)], [(0, "a" if k % 2 else "e"), (1, "u")],
               [(0, 0), (0.08, 1), (1, 0)], rough=0.75, sub=0.55, breath=0.45, seed=30 + k), 0, 0),
            (filt(S("behemoth_attack1" + "abcd"[k - 1], 0.25, 0.55, -6), "lowpass", 1800), 0.02, -12)), 0.13, 0.06)
    # hit (getting_hit is 17 frames): a pained "oof" and stone cracking off him
    out["hit_1"] = finish(mix(
        (V(0.38, [(0, 110), (1, 70)], [(0, "u"), (1, "o")], [(0, 0), (0.06, 1), (1, 0)],
           rough=0.8, sub=0.5, breath=0.35, seed=40), 0.02, 0),
        (filt(S("golem_hit1", 0.0, 0.35, -2), "lowpass", 5000), 0, -6), (gravel(0.4, 90, seed=40), 0.02, -9)), 0.14, 0.06)
    out["hit_2"] = finish(mix(
        (V(0.2, [(0, 105), (1, 80)], [(0, "u"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)],
           rough=0.7, sub=0.4, breath=0.4, seed=41), 0.01, 0),
        (filt(S("catapult_hit1", 0.0, 0.2, -3), "lowpass", 4000), 0, -6)), 0.18, 0.03)
    # ground slam (61 frames, the mace lands at 1.13 s): a rising roar, then a boom and the floor splits
    impact = SLAM_IMPACT
    out["ground_slam_roar"] = finish(mix(
        (V(impact + 0.05, [(0, 62), (0.7, 112), (1, 120)], [(0, "o"), (0.5, "a"), (1, "a")],
           [(0, 0), (0.25, 0.7), (0.95, 1), (1, 0.3)], rough=0.9, sub=0.7, breath=0.3, seed=50), 0, 0),
        (thud(1.0, 60, 28, 0.9, 50), impact, 2),
        (filt(S("ground_attack1", 0.0, 1.0, -4), "lowpass", 3500), impact, -4),
        (gravel(0.8, 140, decay=0.4, seed=50), impact + 0.02, -6),
        (V(0.5, [(0, 70), (1, 55)], [(0, "u"), (1, "u")], [(0, 0), (0.1, 0.6), (1, 0)],
           rough=0.6, sub=0.6, breath=0.5, seed=51), impact + 0.1, -10)), 0.12, 0.25)
    # training: two swings on the dummy, each a grunt and a dull wooden knock
    out["train"] = finish(mix(
        (out["melee_1"] / (np.abs(out["melee_1"]).max() or 1), 0, 0),
        (filt(S("catapult_hit2", 0.0, 0.25, -2), "lowpass", 3000), 0.12, -4)), 0.15, 0.06)
    # stunned: a dazed, wobbling groan
    out["stunned"] = finish(V(0.9, [(k / 10, 70 + (6 if k % 2 else -6)) for k in range(11)],
                              [(0, "o"), (0.5, "u"), (1, "o")], [(0, 0), (0.15, 0.9), (1, 0)],
                              rough=0.35, sub=0.6, breath=0.3, seed=52), 0.08, 0.2)
    # fleeing: alarmed huffs as he lumbers off
    huff = lambda k: V(0.22, [(0, 120), (1, 95)], [(0, "a"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)],
                       rough=0.6, sub=0.4, breath=0.6, seed=60 + k)
    out["fleeing_1"] = finish(mix(*[(huff(k), 0.25 * k, -k) for k in range(3)]), 0.12, 0.08)
    out["fleeing_2"] = finish(mix(*[(huff(k + 3), 0.22 * k, -k) for k in range(2)]), 0.12, 0.06)

    # -- in the Hand -------------------------------------------------------------------------------
    pick = lambda k, d: V(d, [(0, 70 + 4 * k), (0.6, 92 + 4 * k), (1, 84)], [(0, "m"), (0.5, "o"), (1, "a")],
                          [(0, 0), (0.15, 0.8), (0.9, 1), (1, 0)], rough=0.55, sub=0.5, breath=0.2, seed=70 + k)
    out["picked_up_1"] = finish(mix((pick(0, 0.45), 0.03, 0), (gravel(0.3, 50, seed=70), 0, -10)), 0.13)
    out["picked_up_2"] = finish(mix((pick(1, 0.7), 0.03, 0), (gravel(0.4, 50, seed=71), 0, -12)), 0.13)
    out["picked_up_3"] = finish(pick(2, 0.38), 0.12)
    # held too long: an annoyed, rumbling grumble, three beats
    grumble = lambda k: V(0.42, [(0, 66 + 3 * k), (0.5, 74), (1, 62)], [(0, "m"), (0.5, "o"), (1, "m")],
                          [(0, 0), (0.2, 1), (1, 0)], rough=0.6, sub=0.6, breath=0.15, seed=80 + k)
    out["in_hand"] = finish(mix(*[(grumble(k), 0.33 * k, -1 * k) for k in range(3)],
                                (S("pig_ack2", 0.0, 1.1, -10), 0.0, -14)), 0.12, 0.15)
    # slapped (35 frames): stone cracks, a sharp angry bark
    out["slapped"] = finish(mix(
        (filt(S("catapult_hit1", 0.0, 0.18, 0), "highpass", 300), 0, -2), (gravel(0.35, 120, seed=90), 0.0, -6),
        (V(0.45, [(0, 120), (0.3, 125), (1, 80)], [(0, "a"), (1, "u")], [(0, 0), (0.08, 1), (1, 0)],
           rough=0.85, sub=0.6, breath=0.35, seed=90), 0.05, 0)), 0.14, 0.08)
    # dropped: the floor takes his weight; a grunt as he lands
    for k in range(1, 5):
        out[f"dropped_{k}"] = finish(mix(
            (thud(0.7 + 0.05 * k, 70 + 6 * k, 32 + 2 * k, 0.75, 100 + k), 0, 0),
            (gravel(0.6, 70 + 10 * k, decay=0.3, seed=100 + k), 0.01, -9),
            (filt(S(["golem_die1", "golem_die3", "machine_die1", "golem_die2"][k - 1], *[(0.78, 1.10), (0.78, 1.0), (0.0, 0.4), (0.4, 0.62)][k - 1], -3), "lowpass", 3000), 0.0, -8),
            (V(0.3, [(0, 90), (1, 66)], [(0, "u"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)], rough=0.6, sub=0.5,
               breath=0.5, seed=100 + k), 0.05, -5 if k != 2 else -30)), 0.14, 0.15)
    # falling (15 frames): a startled grunt and air
    out["fall"] = finish(mix(
        (V(0.42, [(0, 95), (1, 125)], [(0, "o"), (1, "a")], [(0, 0), (0.15, 1), (1, 0.2)], rough=0.5, sub=0.4,
           breath=0.5, seed=110), 0, 0), (filt(np.random.default_rng(110).standard_normal(int(0.45 * RATE)),
                                               "bandpass", (300, 1500)) * np.linspace(0, 1, int(0.45 * RATE)), 0.05, -16)), 0.11, 0.06)
    out["land_1"] = finish(mix((thud(0.3, 90, 45, 0.8, 120), 0, 0), (gravel(0.25, 80, seed=120), 0, -10)), 0.17, 0.06)
    out["land_2"] = finish(mix((thud(0.45, 95, 40, 0.9, 121), 0, 0), (gravel(0.4, 140, seed=121), 0, -6)), 0.2, 0.08)
    # dragged: his stone body grinding over the floor, and a low complaint
    for k in range(1, 4):
        d = [0.5, 0.52, 0.42][k - 1]
        out[f"dragged_{k}"] = finish(mix(
            (grind(d, 130 + k), 0, 0), (gravel(d, 45, decay=1.5, bright=(500, 3000), seed=130 + k), 0, -8),
            (filt(S("machine_walk" + str(k), 0.0, d, -4), "lowpass", 2000), 0, -10)), 0.10, 0.08)

    # -- needs, prison and death ---------------------------------------------------------------
    # eating (83 frames): crunching through rock and bone, chewing grunts, a swallow
    chew = lambda k: V(0.2, [(0, 70), (1, 64)], [(0, "m"), (1, "m")], [(0, 0), (0.3, 1), (1, 0)],
                       rough=0.4, sub=0.4, breath=0.1, seed=140 + k)
    out["eat_1"] = finish(mix((crunch(0.9, 3, 1), 0, 0), *[(chew(k), 0.25 + 0.3 * k, -6) for k in range(2)],
                              (S("pig_select3", 0.0, 0.5, -8), 0.2, -14)), 0.09, 0.1)
    out["eat_2"] = finish(mix((crunch(1.2, 4, 2), 0, 0), *[(chew(k + 2), 0.2 + 0.28 * k, -7) for k in range(3)],
                              (gulp(2), 1.0, -4)), 0.08, 0.1)
    # drinking (83 frames): three slow gulps and a satisfied breath out
    out["drink"] = finish(mix(*[(gulp(k), 0.35 * k, -k) for k in range(3)],
                              (V(0.5, [(0, 70), (1, 58)], [(0, "a"), (1, "m")], [(0, 0), (0.2, 0.8), (1, 0)],
                                 rough=0.2, sub=0.3, breath=0.6, seed=150), 1.05, -4)), 0.07, 0.15)
    # imprisoned: furious growls, iron bars rattled
    out["imprisoned_1"] = finish(mix(
        (V(0.8, [(0, 80), (0.4, 100), (1, 70)], [(0, "a"), (1, "o")], [(0, 0), (0.15, 1), (1, 0)],
           rough=0.85, sub=0.6, breath=0.3, seed=160), 0, 0), (clank(160), 0.15, -6), (clank(161), 0.4, -10)), 0.12, 0.12)
    out["imprisoned_2"] = finish(mix((clank(162), 0, -2), (clank(163), 0.12, -5),
                                     (V(0.6, [(0, 90), (1, 66)], [(0, "e"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)],
                                        rough=0.8, sub=0.6, breath=0.35, seed=161), 0.1, 0)), 0.12, 0.1)
    out["imprisoned_3"] = finish(V(0.85, [(0, 62), (0.5, 70), (1, 50)], [(0, "u"), (1, "m")],
                                   [(0, 0), (0.2, 1), (1, 0)], rough=0.7, sub=0.7, breath=0.2, seed=162), 0.11, 0.15)
    # torture (100 frames): a pained roar, stone groaning under strain
    out["torture"] = finish(mix(
        (V(1.1, [(0, 95), (0.3, 130), (1, 80)], [(0, "a"), (0.5, "a"), (1, "o")], [(0, 0), (0.1, 1), (1, 0)],
           rough=0.9, sub=0.6, breath=0.35, seed=170), 0, 0),
        (filt(S("bull_die2", 0.2, 1.4, -6), "lowpass", 2000), 0.05, -10), (grind(0.9, 170, 60, 600), 0.1, -10)), 0.1, 0.2)
    # water torture: roaring through water (gurgling: the voice chopped by bubbles)
    rng = np.random.default_rng(RNG_SEED + 180)

    def gargle(d, seed):
        v = V(d, [(0, 88), (0.4, 110), (1, 70)], [(0, "o"), (0.5, "a"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)],
              rough=0.7, sub=0.6, breath=0.3, seed=seed)
        bub = 0.55 + 0.45 * np.sin(2 * np.pi * np.cumsum(9 + 6 * np.abs(_lp_noise(rng, len(v), 3))) / RATE)
        v = v * bub
        top = np.abs(v).max() or 1
        for j in range(5):
            g = gulp(seed + j)
            at = int(rng.integers(0, len(v) - len(g)))
            v[at:at + len(g)] += g * 0.25 * top
        return v

    out["torture_water_1"] = finish(gargle(1.4, 180), 0.11, 0.25)
    out["torture_water_2"] = finish(mix((gargle(1.5, 181), 0, 0), (gargle(1.0, 182), 1.3, -3)), 0.12, 0.3)
    # dying (31 frames): a last falling roar while he cracks apart and crumbles to rubble
    for k, (seed, roar) in enumerate(((190, "behemoth_die1"), (191, "daemon_die3")), 1):
        out[f"die_{k}"] = finish(mix(
            (V(1.0, [(0, 110), (0.25, 120), (1, 45)], [(0, "a"), (0.5, "o"), (1, "u")], [(0, 0), (0.08, 1), (0.7, 0.6), (1, 0)],
               rough=0.85, sub=0.7, breath=0.35, seed=seed), 0, 0),
            (filt(S(roar, 0.3 if k == 1 else 0.0, 1.3 if k == 1 else 1.0, -7), "lowpass", 2000), 0.05, -12),
            (filt(S("golem_die" + str(k), 0.0, 1.2, -2), "lowpass", 4500), 0.45, -5),
            (thud(0.9, 70, 30, 0.8, seed), 0.75, -2),
            (gravel(1.1, 160, decay=0.5, seed=seed), 0.8, -5)), 0.1, 0.3)
    return out


# ---------------------------------------------------------------- third version (2026-11-08)
# The user's review of the second version: hello, sulk, pray_1/2, sleep, charge, the melee and hit
# grunts and the drags are good; the rest sounded "8-bit". What the good ones share: the made voice
# stays low (under about 80 Hz) with a closed or dark mouth, or a real recording carries the sound
# and the made voice only sits under it. What the bad ones had: the made voice exposed and high
# (90-130 Hz, open "a"), and pure tones (sine thuds, gulps and bar clanks). So the rest are remade
# with real non-verbal recordings carrying them, the made voice only low and dark underneath, and
# every made impact or liquid built from shaped noise instead of tones. The good ones are untouched.

def crunch_dark(dur, bites, seed=0):
    """Teeth on rock: bunches of mid-range cracks, softer and darker than `crunch`."""
    rng = np.random.default_rng(RNG_SEED + 1200 + seed)
    n = int(dur * RATE)
    out = np.zeros(n)
    for k in range(bites):
        at = int((k + 0.15 + 0.5 * rng.random()) / bites * n * 0.9)
        for _ in range(rng.integers(4, 8)):
            g = int(rng.integers(300, 900))
            pos = min(n - g, at + int(rng.integers(0, 1500)))
            lo = rng.uniform(300, 800)
            grain = filt(rng.standard_normal(g), "bandpass", (lo, lo * 4)) * np.exp(-np.arange(g) / (g / 4))
            out[pos:pos + g] += grain * rng.uniform(0.3, 1)
    return out


def thump(dur=0.6, low=110, seed=0, crack=0.5):
    """A heavy body on stone: a burst of low-passed noise (no tone) with a short bright crack."""
    rng = np.random.default_rng(RNG_SEED + 700 + seed)
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    body = filt(rng.standard_normal(n), "lowpass", low, 4) * np.exp(-t * 9) * 3.0
    mid = filt(rng.standard_normal(n), "bandpass", (150, 900)) * np.exp(-t * 22)
    hi = filt(rng.standard_normal(n), "bandpass", (900, 4000)) * np.exp(-t * 60) * crack
    return body + mid + hi


def rubble(dur, density=40, decay=0.5, seed=0):
    """Falling stones: soft-edged knocks of different sizes (longer, darker grains than gravel)."""
    rng = np.random.default_rng(RNG_SEED + 800 + seed)
    n = int(dur * RATE)
    out = np.zeros(n)
    t = 0.0
    while True:
        t += rng.exponential(1 / density)
        if t >= dur:
            break
        g = int(rng.integers(300, 1600))
        i = int(t * RATE)
        if i + g >= n:
            continue
        lo = rng.uniform(150, 600)
        grain = filt(rng.standard_normal(g), "bandpass", (lo, lo * 5)) * np.hanning(g) ** 0.5 * np.exp(-np.arange(g) / (g / 3))
        out[i:i + g] += grain * rng.uniform(0.3, 1) * np.exp(-t / decay)
    return out


def bubbles(dur, rate=10, seed=0):
    """Water churned by roaring: short dark noise blips with falling resonance (no tones)."""
    rng = np.random.default_rng(RNG_SEED + 900 + seed)
    n = int(dur * RATE)
    out = np.zeros(n)
    for _k in range(int(dur * rate)):
        g = int(rng.integers(500, 1400))
        i = int(rng.integers(0, max(1, n - g)))
        f = rng.uniform(250, 700)
        b, a = signal.iirpeak(f, 6, fs=RATE)
        blip = signal.lfilter(b, a, rng.standard_normal(g) * np.exp(-np.arange(g) / (g / 5)))
        out[i:i + g] += blip * rng.uniform(0.3, 1)
    return out


def swallow(seed=0):
    """A gulp from a big throat: a dark noise thump that drops in pitch (a swept band, no tone)."""
    rng = np.random.default_rng(RNG_SEED + 950 + seed)
    n = int(0.3 * RATE)
    t = np.arange(n) / RATE
    x = rng.standard_normal(n) * np.exp(-t * 14)
    out = np.zeros(n)
    blk = 256
    for i in range(0, n, blk):               # a resonance sweeping 380 -> 140 Hz
        f = 140 + 240 * np.exp(-t[i] * 12)
        b, a = signal.iirpeak(f, 5, fs=RATE)
        out[i:i + blk] = signal.lfilter(b, a, x[max(0, i - 600):i + blk])[-len(x[i:i + blk]):]
    return out


def low_voice(dur, f0, vowels, amp, seed, rough=0.45, sub=0.6, breath=0.3):
    """The made voice kept low and dark, to sit under a recording."""
    return filt(voice(dur, f0, vowels, amp, rough=rough, sub=sub, breath=breath, seed=seed), "lowpass", 1100, 4)


def wobble(x, rate=5.0, depth=0.012):
    """A dazed pitch wobble, by reading the sound at a gently varying speed."""
    n = len(x)
    pos = np.cumsum(1 + depth * np.sin(2 * np.pi * rate * np.arange(n) / RATE))
    pos = pos[pos < n - 1]
    return np.interp(pos, np.arange(n), x)


def remake(s: "Sources") -> dict[str, np.ndarray]:
    S = lambda n, a=0.0, b=None, st=0: pitch(cut(s(n), a, b), st)
    L = lambda n, a, b, st, hz=2200: filt(S(n, a, b, st), "lowpass", hz)
    out = {}
    # angry: a snort, then a short snarl carried by the Daemon's roar, his growl under it
    out["angry"] = finish(mix(
        (filt(S("bull_select1", 0.0, 0.3, -3), "bandpass", (150, 3000)), 0, -3),
        (L("daemon_attack6", 0.0, 0.62, -6), 0.15, 0),
        (low_voice(0.6, [(0, 62), (0.4, 72), (1, 58)], [(0, "o"), (1, "u")], [(0, 0), (0.15, 1), (1, 0)], 1001), 0.17, -5)), 0.14, 0.08)
    # happy: a slow, rumbling chuckle "hm-hm-hm", made like hello (a low closed-mouth voice opening
    # a little, f0 60-72 Hz, some breath), with the Pig two octaves down as texture underneath
    chuck = lambda k, d=0.3: voice(d, [(0, 72 - 3 * k), (0.4, 76 - 3 * k), (1, 60 - 2 * k)], [(0, "m"), (0.4, "o"), (1, "m")],
                                   [(0, 0), (0.12, 1), (0.6, 0.7), (1, 0)], rough=0.35, sub=0.5, breath=0.3, seed=1010 + k)
    out["happy_1"] = finish(mix(*[(chuck(k), 0.05 + 0.36 * k, -1.5 * k) for k in range(4)],
                                (L("pig_ack2", 0.0, 1.17, -12, 1200), 0.0, -12)), 0.11, 0.15)
    out["happy_2"] = finish(mix(*[(chuck(k + 4), 0.05 + 0.33 * k, -2 * k) for k in range(2)],
                                (L("pig_ack1", 0.0, 0.77, -12, 1200), 0.0, -12)), 0.11, 0.12)
    # unhappy: a long groan made like sulk (low, breathy, sinking), the Bull an octave down beneath
    out["unhappy_1"] = finish(mix(
        (voice(1.15, [(0, 70), (0.3, 72), (1, 46)], [(0, "o"), (0.5, "u"), (1, "m")], [(0, 0), (0.15, 0.9), (0.7, 0.8), (1, 0)],
               rough=0.3, sub=0.4, breath=0.5, seed=1020), 0, 0),
        (L("bull_ack4", 0.0, 1.2, -12, 1000), 0.05, -12)), 0.08, 0.2)
    out["unhappy_2"] = finish(mix(
        (voice(0.75, [(0, 66), (1, 48)], [(0, "u"), (1, "m")], [(0, 0), (0.2, 0.9), (1, 0)],
               rough=0.3, sub=0.4, breath=0.5, seed=1021), 0, 0),
        (L("bull_ack2", 0.0, 0.8, -12, 1000), 0.0, -12)), 0.075, 0.15)
    # casino: the chuckle and a stone fist on the table / a disgusted groan and a slam
    out["gamble_win"] = finish(mix(*[(chuck(k + 6), 0.36 * k, -k) for k in range(2)], (thump(0.45, 120, 1), 0.66, -4)), 0.12, 0.1)
    out["gamble_lose"] = finish(mix(
        (low_voice(0.7, [(0, 70), (1, 50)], [(0, "o"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)], 1022, rough=0.4, sub=0.5, breath=0.45), 0, 0),
        (thump(0.6, 100, 2, 0.7), 0.5, -2), (rubble(0.5, 30, seed=2), 0.53, -10)), 0.12, 0.15)
    # battle roars: like the charge (which works): the Bull's bellow leads, his voice under it
    out["charge_roar_1"] = finish(mix((L("bull_attack1", 0.0, 1.6, -5, 2500), 0, 0),
                                      (L("daemon_attack3", 0.0, 0.98, -7), 0.15, -6),
                                      (low_voice(1.3, [(0, 60), (0.4, 78), (1, 62)], [(0, "o"), (1, "o")], [(0, 0), (0.15, 1), (1, 0)], 1030), 0.05, -5)), 0.11, 0.2)
    out["charge_roar_2"] = finish(mix((L("bull_attack3", 0.0, 1.3, -5, 2500), 0, 0),
                                      (L("daemon_attack7", 0.0, 0.85, -6), 0.1, -7),
                                      (low_voice(1.1, [(0, 58), (0.5, 74), (1, 60)], [(0, "u"), (1, "o")], [(0, 0), (0.2, 1), (1, 0)], 1031), 0.05, -5)), 0.11, 0.2)
    # ground slam: the roar builds (Bull and Daemon), the mace lands at 1.13 s, the floor booms
    roar = mix((L("bull_attack1", 0.0, 1.2, -5, 2500), 0, 0), (L("daemon_attack13", 0.0, 0.78, -6), 0.45, -4),
               (low_voice(SLAM_IMPACT, [(0, 56), (0.8, 80), (1, 84)], [(0, "o"), (1, "o")], [(0, 0), (0.3, 0.8), (1, 1)], 1040), 0, -5))
    roar = roar[:int((SLAM_IMPACT + 0.06) * RATE)]
    roar = fade(roar, 0.01, 0.06)
    out["ground_slam_roar"] = finish(mix((roar, 0, 0), (thump(1.1, 90, 3, 0.8), SLAM_IMPACT, 4),
                                         (L("ground_attack1", 0.0, 1.0, -4, 3500), SLAM_IMPACT, -3),
                                         (L("catapult_die1", 0.0, 1.2, -4, 4000), SLAM_IMPACT, -6),
                                         (rubble(0.9, 50, 0.5, 3), SLAM_IMPACT + 0.05, -6)), 0.12, 0.25)
    # training: a swing grunt (the melee that works) and a dull knock on the dummy
    out["train"] = finish(mix((L("behemoth_attack1b", 0.14, 0.64, -6, 1800), 0, 0),
                              (low_voice(0.35, [(0, 70), (1, 58)], [(0, "u"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)], 1050), 0.0, -4),
                              (thump(0.3, 160, 4, 0.3), 0.18, -6)), 0.15, 0.06)
    # stunned: a dazed moan, wobbling
    out["stunned"] = finish(wobble(mix((low_voice(0.95, [(0, 60), (0.5, 64), (1, 50)], [(0, "o"), (0.5, "u"), (1, "o")],
                                                  [(0, 0), (0.15, 1), (1, 0)], 1060, rough=0.3, breath=0.35), 0, 0),
                                       (L("behemoth_die4", 0.5, 0.86, -9, 1200), 0.1, -8)), 4.0, 0.025), 0.08, 0.2)
    # fleeing: heavy alarmed huffs as he lumbers off (the Behemoth's attack breaths)
    out["fleeing_1"] = finish(mix(*[(L("behemoth_attack1" + c, a, a + 0.3, -3, 2000), 0.25 * k, -k)
                                    for k, (c, a) in enumerate((("e", 0.26), ("g", 0.2), ("f", 0.2)))]), 0.12, 0.08)
    out["fleeing_2"] = finish(mix(*[(L("behemoth_attack1" + c, a, a + 0.28, -3, 2000), 0.22 * k, -k)
                                    for k, (c, a) in enumerate((("h", 0.24), ("c", 0.26)))]), 0.12, 0.06)
    # picked up / held: annoyed grumbles in the style of hello (which works)
    grum = lambda d, k: low_voice(d, [(0, 64 + 2 * k), (0.5, 74), (1, 62)], [(0, "m"), (0.5, "o"), (1, "m")],
                                  [(0, 0), (0.15, 1), (1, 0)], 1070 + k, rough=0.35, sub=0.5, breath=0.2)
    out["picked_up_1"] = finish(mix((grum(0.5, 0), 0, 0), (L("bull_select2", 0.0, 0.5, -7, 1200), 0.02, -10)), 0.12)
    out["picked_up_2"] = finish(mix((grum(0.75, 1), 0, 0), (L("bull_ack1", 0.0, 0.7, -7, 1200), 0.02, -9)), 0.12)
    out["picked_up_3"] = finish(grum(0.4, 2), 0.11)
    out["in_hand"] = finish(mix(*[(grum(0.45, 3 + k), 0.36 * k, -k) for k in range(3)],
                                (L("bull_ack4", 0.0, 1.4, -8, 1200), 0.0, -9)), 0.12, 0.15)
    # slapped: stone cracks, a sharp angry bark (the Daemon), his growl under it
    out["slapped"] = finish(mix((filt(S("catapult_hit1", 0.0, 0.18, 0), "highpass", 300), 0, -3),
                                (L("daemon_attack2", 0.0, 0.62, -5), 0.04, 0),
                                (low_voice(0.45, [(0, 70), (1, 58)], [(0, "o"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)], 1080), 0.04, -6)), 0.14, 0.08)
    # dropped / landing: his weight on the floor (noise thumps, real stone impacts), an "oof"
    drops = [("golem_die1", 0.78, 1.10), ("golem_die3", 0.78, 1.0), ("machine_die1", 0.0, 0.5), ("golem_die2", 0.4, 0.62)]
    for k, (src, a, b) in enumerate(drops, 1):
        out[f"dropped_{k}"] = finish(mix((thump(0.75, 100 + 10 * k, 10 + k), 0, 0),
                                         (L(src, a, b, -3, 3500), 0.0, -4), (rubble(0.6, 30, 0.35, 10 + k), 0.03, -10),
                                         (low_voice(0.3, [(0, 66), (1, 54)], [(0, "u"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)], 1090 + k), 0.05, -7)), 0.14, 0.15)
    out["fall"] = finish(mix((L("behemoth_attack1h", 0.2, 0.55, -2, 2000), 0, 0),
                             (filt(np.random.default_rng(1110).standard_normal(int(0.45 * RATE)), "bandpass", (300, 1500))
                              * np.linspace(0, 1, int(0.45 * RATE)), 0.05, -16)), 0.11, 0.06)
    out["land_1"] = finish(mix((thump(0.35, 120, 20, 0.6), 0, 0), (rubble(0.3, 30, 0.2, 20), 0, -10)), 0.17, 0.06)
    out["land_2"] = finish(mix((thump(0.5, 110, 21, 0.8), 0, 0), (L("golem_die3", 0.78, 1.0, -2, 3500), 0, -4),
                               (rubble(0.45, 45, 0.3, 21), 0.02, -8)), 0.2, 0.08)
    # eating: rock crunching, the Pig's chewing grunts far down, a swallow
    out["eat_1"] = finish(mix((crunch_dark(1.0, 3, 1), 0, -2), (L("pig_select3", 0.0, 0.51, -12, 1500), 0.25, 0),
                              (L("pig_ack3", 0.0, 0.97, -12, 1500), 0.4, -3)), 0.09, 0.1)
    out["eat_2"] = finish(mix((crunch_dark(1.3, 4, 2), 0, -2), (L("pig_ack3", 0.0, 0.97, -12, 1500), 0.2, 0),
                              (swallow(2), 1.1, 0)), 0.08, 0.1)
    # drinking: three slow gulps and a satisfied breath out (the breath as in sulk, which works)
    out["drink"] = finish(mix(*[(swallow(k), 0.4 * k, -k) for k in range(3)],
                              (low_voice(0.6, [(0, 62), (1, 52)], [(0, "m"), (1, "m")], [(0, 0), (0.2, 0.8), (1, 0)], 1100, rough=0.2, breath=0.6), 1.2, -3)), 0.07, 0.15)
    # imprisoned: furious snarls; a stone fist on the cell door
    out["imprisoned_1"] = finish(mix((L("daemon_attack5", 0.0, 0.87, -6), 0, 0), (thump(0.4, 140, 30, 0.6), 0.3, -6),
                                     (low_voice(0.8, [(0, 60), (1, 54)], [(0, "o"), (1, "u")], [(0, 0), (0.15, 1), (1, 0)], 1110), 0.0, -6)), 0.12, 0.12)
    out["imprisoned_2"] = finish(mix((thump(0.35, 140, 31, 0.6), 0, -3), (thump(0.35, 140, 32, 0.6), 0.18, -5),
                                     (L("daemon_attack9", 0.0, 0.71, -6), 0.1, 0)), 0.12, 0.1)
    out["imprisoned_3"] = finish(mix((L("bull_ack1", 0.0, 1.0, -8, 1500), 0, 0),
                                     (low_voice(0.85, [(0, 56), (1, 48)], [(0, "u"), (1, "m")], [(0, 0), (0.2, 1), (1, 0)], 1112), 0, -6)), 0.11, 0.15)
    # torture: a pained roar (the Bull's death bellow), stone groaning under strain
    out["torture"] = finish(mix((L("bull_die2", 0.1, 1.5, -5, 2200), 0, 0), (L("daemon_die1", 0.0, 1.0, -6), 0.2, -8),
                                (low_voice(1.2, [(0, 60), (1, 50)], [(0, "o"), (1, "u")], [(0, 0), (0.15, 1), (1, 0)], 1120), 0.05, -6),
                                (grind(1.0, 170, 60, 600), 0.1, -12)), 0.1, 0.2)
    # water torture: roaring into churning water
    churn = lambda x, seed: x * (0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(7 + 5 * np.abs(
        _lp_noise(np.random.default_rng(seed), len(x), 3))) / RATE))
    out["torture_water_1"] = finish(mix((churn(L("daemon_die1", 0.0, 1.2, -6, 1600), 1), 0, 0), (bubbles(1.4, 14, 1), 0, -3),
                                        (low_voice(1.2, [(0, 62), (1, 50)], [(0, "o"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)], 1140), 0.05, -7)), 0.11, 0.25)
    out["torture_water_2"] = finish(mix((churn(L("behemoth_die3", 0.5, 1.6, -6, 1600), 2), 0, 0), (bubbles(2.4, 12, 2), 0, -3),
                                        (churn(L("daemon_die4", 0.0, 1.0, -6, 1600), 3), 1.3, -3)), 0.12, 0.3)
    # dying: a last roar (Behemoth / Daemon death cries) while he cracks apart into rubble
    for k, (roar_src, a, b, bull) in enumerate((("behemoth_die1", 0.4, 1.4, "bull_die3"), ("daemon_die1", 0.0, 1.2, "bull_die2")), 1):
        out[f"die_{k}"] = finish(mix((L(roar_src, a, b, -6, 2200), 0, 0), (L(bull, 0.0, 1.0, -7, 1800), 0.05, -7),
                                     (low_voice(1.0, [(0, 64), (1, 42)], [(0, "o"), (1, "u")], [(0, 0), (0.1, 1), (1, 0)], 1130 + k), 0.0, -6),
                                     (L("golem_die" + str(k), 0.0, 1.2, -2, 4000), 0.5, -4),
                                     (thump(0.9, 90, 40 + k, 0.7), 0.75, 0), (rubble(1.1, 45, 0.6, 40 + k), 0.8, -5)), 0.1, 0.3)
    return out


# Bile Demon sample (BileHD.sdt, in file order, blanks left out) -> the Molynx file in its place
ROLES = {
    "bd_charge": "charge", "bd_fleeing": "fleeing_1", "bd_hello": "hello", "bd_hit16": "hit_1",
    "bd_imprison15": "imprisoned_1", "bd_inhand2": "in_hand", "bd_picked_up_1": "picked_up_1",
    "bd_picked_up_2": "picked_up_2", "bd_hit17": "hit_2", "bd_picked_up_3": "picked_up_3",
    "bd_pray17": "pray_1", "bd_slapped": "slapped", "bd_sleep": "sleep", "bd_stun10": "stunned",
    "bd_sulk12": "sulk", "bd_train6": "train", "bd_unhappy5": "unhappy_1", "bd_tort_gen": "torture",
    "bd_tort_water": "torture_water_1", "bd_fleeing2": "fleeing_2", "bd_gamble15": "gamble_win",
    "bd_imprison16": "imprisoned_2", "bd_imprison17": "imprisoned_3", "bd_pray18": "pray_2",
    "bd_unhappy6": "unhappy_2", "bd_angr01": "angry", "bd_chrg01": "charge_roar_1",
    "bd_chrg02": "charge_roar_2", "bd_die01": "die_1", "bd_die02": "die_2", "bd_drink01": "drink",
    "bd_drop01": "dropped_1", "bd_drop02": "dropped_2", "bd_drop03": "dropped_3", "bd_eat01": "eat_1",
    "bd_eat02": "eat_2", "bd_hap01": "happy_1", "bd_hap02": "happy_2", "bd_melee01": "melee_1",
    "bd_melee02": "melee_2", "bd_melee03": "melee_3", "bd_melee04": "melee_4", "bd_drop04": "dropped_4",
    "bd_gamble17": "gamble_lose", "bd_fall02": "fall", "bd_attksp02": "ground_slam_roar",
    "bd_land02": "land_1", "bd_land03": "land_2", "bd_tort_water2": "torture_water_2",
    "BILE DEMON DRAG 1": "dragged_1", "BILE DEMON DRAG 2": "dragged_2", "BILE DEMON DRAG 3": "dragged_3",
}


# the second version's sounds the user kept (review of 2026-11-08); the rest come from remake()
KEPT = {"hello", "sulk", "pray_1", "pray_2", "sleep", "charge", "melee_1", "melee_2", "melee_3", "melee_4",
        "hit_1", "hit_2", "dragged_1", "dragged_2", "dragged_3"}


def build_sounds(source_dirs, out_dir: Path) -> dict[str, float]:
    src = Sources(source_dirs)
    made = {k: v for k, v in recipes(src).items() if k in KEPT}
    made.update(remake(src))
    missing = sorted(set(ROLES.values()) - set(made))
    if missing:
        raise ValueError(f"roles without a sound: {missing}")
    out = {}
    for name, x in made.items():
        save(Path(out_dir) / f"{name}.wav", x)
        out[name] = len(x) / RATE
    return out


if __name__ == "__main__":
    *dirs, o = sys.argv[1:]
    for k, v in build_sounds([Path(d) for d in dirs], Path(o)).items():
        print(f"{k:20s} {v:5.2f}s")
