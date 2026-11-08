"""Molynx's own sounds: one file for every sound the Bile Demon's sound set has (the Bile Demon's
bank, Data/Sound/Sfx/Global/BileHD.sdt and creature_bile_demonSFX.map, is what his record still
plays, sound category CREATURE_BILE_DEMON).

He is a rock golem built on the MegaGlest Behemoth, so his voice is the Behemoth's (growls and
roars, pitched down a little for his size) and his body is the MegaGlest Golem's (grinding stone,
rock impacts), with a few synthesised stone sounds (thuds, scrapes, crunching) where neither has
one. The sources are MegaGlest data (CC BY-SA 3.0): megaglest_behemoth/sounds and the
megaglest-data repository's techs/megapack/commondata/sounds/golem_*.wav.

`build_sounds(behemoth_dir, golem_dir, out_dir)` writes `<role>.wav` (22050 Hz, mono, 16-bit: the
game's own sample format) for every role in ROLES, and returns {role: seconds}. ROLES maps each
Bile Demon sample (BileHD.sdt order) to the Molynx file that takes its place. Nothing here changes
the game's files: putting them into a sound bank is a later step.

LICENCE: GPL-3.0 (code). The sounds made from MegaGlest data are CC BY-SA 3.0.
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


def load(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float64) / 32768.0


def save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data)


# ---------------------------------------------------------------- building blocks

def cut(x: np.ndarray, a: float, b: float | None = None) -> np.ndarray:
    return x[int(a * RATE):None if b is None else int(b * RATE)].copy()


def pitch(x: np.ndarray, semitones: float) -> np.ndarray:
    """Tape-style shift: lower is also slower, as a bigger body would sound."""
    if semitones == 0:
        return x
    factor = 2 ** (semitones / 12)
    n = max(1, int(round(len(x) / factor)))
    return signal.resample(x, n)


def filt(x: np.ndarray, kind: str, freq, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, freq, btype=kind, fs=RATE, output="sos")
    return signal.sosfilt(sos, x)


def fade(x: np.ndarray, fin: float = 0.005, fout: float = 0.04) -> np.ndarray:
    x = x.copy()
    a, b = int(fin * RATE), int(fout * RATE)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        b = min(b, len(x))
        x[-b:] *= np.linspace(1, 0, b) ** 2
    return x


def mix(*layers: tuple[np.ndarray, float, float]) -> np.ndarray:
    """(signal, start seconds, gain dB) layers summed."""
    end = max(int(s * RATE) + len(x) for x, s, _g in layers)
    out = np.zeros(end)
    for x, s, g in layers:
        i = int(s * RATE)
        out[i:i + len(x)] += x * 10 ** (g / 20)
    return out


def finish(x: np.ndarray, rms: float, fout: float = 0.04) -> np.ndarray:
    """Remove DC, fade the ends, bring to `rms` and keep peaks under 0.97 with a soft limiter."""
    x = filt(x, "highpass", 30, 2)
    x = fade(x, 0.004, fout)
    loud = np.sqrt(np.mean(x ** 2)) or 1.0
    x = x * (rms / loud)
    peak = np.abs(x).max()
    if peak > 0.9:   # soft knee above 0.9
        over = np.abs(x) > 0.9
        x[over] = np.sign(x[over]) * (0.9 + 0.07 * np.tanh((np.abs(x[over]) - 0.9) / 0.07))
    return x


def thud(dur: float = 0.5, f0: float = 70, f1: float = 38, noise: float = 0.5, seed: int = 0) -> np.ndarray:
    """A heavy stone hitting the floor: a falling low sine plus a short low-passed noise crack."""
    rng = np.random.default_rng(RNG_SEED + seed)
    t = np.arange(int(dur * RATE)) / RATE
    freq = f1 + (f0 - f1) * np.exp(-t * 18)
    body = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * 7)
    crack = filt(rng.standard_normal(len(t)), "lowpass", 1800) * np.exp(-t * 40)
    grit = filt(rng.standard_normal(len(t)), "bandpass", (300, 3000)) * np.exp(-t * 14) * 0.4
    return body + noise * crack + noise * grit


def scrape(dur: float, seed: int = 0) -> np.ndarray:
    """Stone dragged over stone: band-passed noise with a grainy, uneven loudness."""
    rng = np.random.default_rng(RNG_SEED + 100 + seed)
    n = int(dur * RATE)
    base = filt(rng.standard_normal(n), "bandpass", (120, 1400))
    grains = filt(np.abs(rng.standard_normal(n)) ** 3, "lowpass", 35, 2)
    env = np.minimum(1, np.arange(n) / (0.08 * RATE)) * np.minimum(1, (n - np.arange(n)) / (0.12 * RATE))
    return base * (0.4 + grains / (grains.max() or 1)) * env


def crunch(dur: float, bites: int, seed: int = 0) -> np.ndarray:
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


# ---------------------------------------------------------------- the set

class Sources:
    def __init__(self, behemoth_dir: Path, golem_dir: Path):
        self.b, self.g, self._cache = Path(behemoth_dir), Path(golem_dir), {}

    def __call__(self, name: str) -> np.ndarray:
        if name not in self._cache:
            d = self.b if name.startswith("behemoth") else self.g
            self._cache[name] = load(d / f"{name}.wav")
        return self._cache[name]


def recipes(s: Sources) -> dict[str, np.ndarray]:
    B = lambda n, a, b, st=-3: pitch(cut(s(f"behemoth_{n}"), a, b), st)   # his voice, a little deeper
    G = lambda n, a, b, st=-2: pitch(cut(s(f"golem_{n}"), a, b), st)      # his stone body
    rock_hit = lambda st=-2: filt(G("hit1", 0.0, 0.45, st), "lowpass", 5000)
    out = {}

    # greetings and orders
    out["hello"] = finish(mix((B("select1", 0.10, 0.60), 0, 0), (G("select3", 0.04, 0.40), 0.02, -9)), 0.10)
    out["charge"] = finish(mix((B("select4", 0.08, 0.70, -2), 0, 0), (G("select4", 0.05, 0.40), 0.05, -8)), 0.15)
    out["charge_roar_1"] = finish(mix((B("select2", 0.10, 1.40, -2), 0, 0), (G("select8", 0.20, 0.95), 0.1, -10)), 0.10, 0.15)
    out["charge_roar_2"] = finish(mix((B("ack3", 0.0, 1.10, -2), 0, 0), (G("select9", 0.25, 1.0), 0.05, -10)), 0.10, 0.15)
    out["angry"] = finish(mix((B("ack5", 0.15, 1.90, -3), 0, 0), (G("select7", 0.25, 1.0), 0.3, -9)), 0.11, 0.2)
    out["happy_1"] = finish(mix((B("ack1", 0.04, 1.26, -1), 0, 0), (G("ack6", 0.0, 0.8, 0), 0.05, -12)), 0.10, 0.15)
    out["happy_2"] = finish(B("ack2", 0.05, 0.75, -1), 0.11)
    out["gamble_win"] = finish(mix((B("ack2", 0.05, 0.40, 0), 0, 0), (B("ack2", 0.48, 0.75, 0), 0.30, -2)), 0.13)
    out["gamble_lose"] = finish(B("ack4", 0.08, 0.95, -4), 0.10, 0.12)
    out["pray_1"] = finish(filt(mix((pitch(B("ack6", 0.12, 1.15, 0), -5), 0, 0),
                                    (G("select1", 0.04, 0.70, -6), 0.0, -8)), "lowpass", 2500), 0.10, 0.2)
    out["pray_2"] = finish(filt(B("ack4", 0.08, 0.90, -6), "lowpass", 2500), 0.08, 0.15)
    out["train"] = finish(mix((B("select3", 0.08, 0.45, -2), 0, 0), (rock_hit(), 0.12, -8)), 0.15)

    # being handled
    out["picked_up_1"] = finish(mix((B("ack2", 0.06, 0.38, -4), 0, 0), (G("ack3", 0.03, 0.30), 0, -9)), 0.13)
    out["picked_up_2"] = finish(B("select3", 0.08, 0.62, -4), 0.14)
    out["picked_up_3"] = finish(G("ack10", 0.05, 0.36, -3), 0.10)
    out["in_hand"] = finish(mix((B("ack4", 0.10, 0.85, -5), 0, 0), (G("ack4", 0.05, 0.75, -4), 0.05, -8)), 0.13, 0.15)
    out["slapped"] = finish(mix((rock_hit(0), 0, 0), (B("select4", 0.10, 0.40, -3), 0.03, -2)), 0.13)
    # dragged: his stone body grinding over the floor (the Golem's grinding, darker), a little grit
    grind = lambda n, a, b, st: filt(G(n, a, b, st), "lowpass", 2400)
    out["dragged_1"] = finish(mix((grind("select1", 0.05, 0.48, -4), 0, 0), (scrape(0.45, 1), 0, -14)), 0.10)
    out["dragged_2"] = finish(mix((grind("select2", 0.10, 0.55, -5), 0, 0), (scrape(0.48, 2), 0, -14)), 0.10)
    out["dragged_3"] = finish(mix((grind("select11", 0.06, 0.42, -3), 0, 0), (scrape(0.38, 3), 0, -14)), 0.10)
    out["dropped_1"] = finish(mix((thud(0.9, 80, 36, 0.6, 1), 0, 0), (G("die1", 0.78, 1.10, -3), 0.03, -6),
                                  (B("ack2", 0.06, 0.38, -4), 0.10, -8)), 0.14, 0.2)
    out["dropped_2"] = finish(mix((thud(0.65, 75, 40, 0.6, 2), 0, 0), (G("die3", 0.78, 1.0, -2), 0.02, -6)), 0.13, 0.15)
    out["dropped_3"] = finish(mix((thud(0.8, 70, 34, 0.7, 3), 0, 0), (rock_hit(-4), 0.0, -4)), 0.12, 0.2)
    out["dropped_4"] = finish(mix((thud(0.6, 85, 42, 0.5, 4), 0, 0), (G("die2", 0.40, 0.62, -2), 0.02, -6)), 0.14)
    out["fall"] = finish(mix((B("select4", 0.10, 0.50, -1), 0, 0), (scrape(0.25, 4), 0.25, -10)), 0.11)
    out["land_1"] = finish(thud(0.26, 90, 45, 0.8, 5), 0.16, 0.06)
    out["land_2"] = finish(mix((thud(0.26, 95, 48, 0.9, 6), 0, 0), (rock_hit(-1), 0, -8)), 0.2, 0.06)

    # fighting
    for k, (n, a, b) in enumerate([("select3", 0.08, 0.50), ("select4", 0.10, 0.55),
                                   ("ack2", 0.06, 0.34), ("select1", 0.12, 0.55)], 1):
        out[f"melee_{k}"] = finish(B(n, a, b, -2 - (k % 2)), 0.12)
    out["hit_1"] = finish(mix((rock_hit(0), 0, -2), (B("ack2", 0.06, 0.36, -2), 0.02, 0)), 0.14)
    out["hit_2"] = finish(mix((G("hit1", 0.0, 0.16, 1), 0, 0), (B("select3", 0.10, 0.20, -2), 0.0, -4)), 0.2, 0.03)
    out["ground_slam_roar"] = finish(mix((B("select2", 0.10, 1.40, -3), 0, 0),
                                         (thud(0.9, 75, 34, 0.8, 7), 0.80, -1),
                                         (G("attack1", 0.95, 1.40, -2), 0.82, -4)), 0.10, 0.2)
    out["stunned"] = finish(filt(B("ack4", 0.10, 0.60, -5), "lowpass", 2200), 0.08)
    out["fleeing_1"] = finish(B("select2", 0.48, 1.20, -1), 0.11)
    out["fleeing_2"] = finish(B("select4", 0.12, 0.45, 0), 0.10)
    out["die_1"] = finish(mix((B("die1", 0.40, 1.60, -3), 0, 0), (G("die1", 0.0, 1.20, -2), 0.25, -2),
                              (thud(0.8, 70, 32, 0.8, 8), 1.05, -3)), 0.09, 0.25)
    out["die_2"] = finish(mix((B("die4", 0.40, 1.70, -3), 0, 0), (G("die3", 0.0, 1.30, -2), 0.30, -2),
                              (thud(0.8, 65, 30, 0.8, 9), 1.15, -3)), 0.09, 0.25)

    # needs and moods
    snore = filt(pitch(cut(s("behemoth_ack6"), 0.15, 1.10), -9), "lowpass", 900)
    out["sleep"] = finish(mix((snore * np.hanning(len(snore)), 0, 0),
                              (filt(scrape(0.6, 9), "lowpass", 600), 0.75, -10)), 0.085, 0.3)
    out["eat_1"] = finish(mix((crunch(0.55, 3, 1), 0, 0), (B("ack2", 0.06, 0.30, -4), 0.32, -6)), 0.09)
    out["eat_2"] = finish(crunch(0.75, 4, 2), 0.06)
    gulp = filt(pitch(cut(s("behemoth_ack2"), 0.08, 0.30), -10), "lowpass", 700)
    out["drink"] = finish(mix((gulp, 0, 0), (gulp, 0.28, -2)), 0.05, 0.08)
    out["sulk"] = finish(B("ack4", 0.10, 0.70, -6), 0.08)
    out["unhappy_1"] = finish(B("ack5", 0.20, 1.10, -6), 0.08, 0.2)
    out["unhappy_2"] = finish(B("ack1", 0.76, 1.26, -6), 0.07)
    out["imprisoned_1"] = finish(mix((B("ack3", 0.0, 0.70, -3), 0, 0), (scrape(0.4, 5), 0.2, -12)), 0.12)
    out["imprisoned_2"] = finish(B("ack3", 0.60, 1.10, -4), 0.12)
    out["imprisoned_3"] = finish(B("ack5", 1.06, 1.70, -4), 0.12)
    out["torture"] = finish(mix((B("die2", 0.50, 1.10, -2), 0, 0), (rock_hit(-3), 0.05, -8)), 0.09)
    out["torture_water_1"] = finish(mix((B("die3", 0.50, 1.60, -3), 0, 0), (scrape(0.8, 6), 0.2, -14)), 0.11, 0.25)
    out["torture_water_2"] = finish(mix((B("die3", 0.95, 2.95, -3), 0, 0), (B("ack5", 0.20, 0.90, -5), 1.4, -4)), 0.12, 0.3)
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


def build_sounds(behemoth_dir: Path, golem_dir: Path, out_dir: Path) -> dict[str, float]:
    made = recipes(Sources(behemoth_dir, golem_dir))
    missing = sorted(set(ROLES.values()) - set(made))
    if missing:
        raise ValueError(f"roles without a sound: {missing}")
    out = {}
    for name, x in made.items():
        save(Path(out_dir) / f"{name}.wav", x)
        out[name] = len(x) / RATE
    return out


if __name__ == "__main__":
    b, g, o = map(Path, sys.argv[1:4])
    for k, v in build_sounds(b, g, o).items():
        print(f"{k:20s} {v:5.2f}s")
