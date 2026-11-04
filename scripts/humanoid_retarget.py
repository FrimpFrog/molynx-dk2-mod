"""Retarget DK2 humanoid animations (Warlock, Wizard, ...) onto a rigged humanoid mesh. LICENCE: GPL-3.0.

Same method as golem_retarget (rigid donor segments -> Kabsch rotations -> linear-blend skinning) but driven by a
Rig, so any humanoid MegaGlest unit can be given a rig. The Rig names its bones (parents, joints, labels, weights).
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripts.golem_retarget import GIANT_HEIGHT, giant_labels, soft_limit
from scripts.retarget import donor_motion, kabsch, pose_bones, unwrap_rotvecs

MAX_STEP_DEG = 14.0                  # a mage moves quicker than a golem
HUMANOID_BONES = ["root", "torso", "head", "armN", "armP", "legN", "legP", "foreN", "foreP", "shinN", "shinP", "cape"]
PARENTS = {"root": None, "torso": "root", "head": "torso", "armN": "torso", "armP": "torso", "legN": "root",
           "legP": "root", "foreN": "armN", "foreP": "armP", "shinN": "legN", "shinP": "legP", "cape": "torso"}
DONOR_SEGMENT = {"torso": 0, "head": 1, "armN": 2, "armP": 3, "legN": 4, "legP": 5, "foreN": 6, "foreP": 7, "shinN": 8, "shinP": 9}
SHIN_Y = 0.14        # donor units (Giant height 0.79): below this a leg is the shin
CAPS = {"root": 100.0, "torso": 85.0, "head": 70.0, "armN": 150.0, "armP": 150.0, "legN": 80.0, "legP": 80.0,
        "foreN": 130.0, "foreP": 130.0, "shinN": 140.0, "shinP": 140.0, "footN": 120.0, "footP": 120.0, "cape": 40.0}


@dataclass
class Rig:
    rest: np.ndarray                      # (M,3) rest pose
    indices: np.ndarray                   # triangle corner indices (for smoothing)
    joints: dict                          # bone -> pivot (3,)
    weights: np.ndarray                   # (M, len(bones))
    height: float                         # of the rest figure
    floor: float                          # lowest rest vertex y
    bones: list = field(default_factory=lambda: list(HUMANOID_BONES))
    parents: dict = field(default_factory=lambda: dict(PARENTS))


def retarget(rig: Rig, donor_frames, donor_rest, amp=None, extra: Callable | None = None, ground=True, max_step=MAX_STEP_DEG,
             caps=None, tilt_body=True):
    """donor_frames (F,N,3), donor_rest (N,3), both right-handed (x already flipped); NaN rows are ignored. -> (F', M, 3).
    tilt_body=False keeps the root upright (a kneeling bow: the hips and knees fold, the body does not fall over)."""
    k = GIANT_HEIGHT / (np.nanmax(donor_rest[:, 1]) - np.nanmin(donor_rest[:, 1]))
    donor_rest, donor_frames = donor_rest * k, donor_frames * k
    amp = {b: 1.0 for b in rig.bones} | (amp or {})
    caps = CAPS | (caps or {})
    keep = np.isfinite(donor_frames).all(axis=(0, 2)) & np.isfinite(donor_rest).all(axis=1)
    donor_frames, donor_rest = donor_frames[:, keep], donor_rest[keep]
    lab = giant_labels(donor_rest)
    shin = np.isin(lab, (4, 5)) & (donor_rest[:, 1] < SHIN_Y)
    lab_s = lab.copy()
    lab_s[shin] = lab[shin] + 4                                   # 8 shinN, 9 shinP
    mot = donor_motion(donor_frames, donor_rest, lab_s, 10)
    rv, tr = mot["rotvec"], mot["trans"]
    body = np.isin(lab, (4, 5))                                # the legs set the body's tilt: a bow is the torso bending over planted legs
    has = {n: (lab_s == s).sum() >= 4 for n, s in DONOR_SEGMENT.items()}
    F = len(donor_frames)
    pitch = np.zeros(F)
    for f in range(F):
        Rb, _ = kabsch(donor_rest[body], donor_frames[f][body])
        up = Rb @ np.array([0.0, 1.0, 0.0])
        pitch[f] = np.arctan2(up[2], up[1])
    pitch = np.unwrap(pitch) if tilt_body else np.zeros(F)
    mats = {b: np.zeros((F, 3, 3)) for b in rig.bones}
    for f in range(F):
        Rroot = Rot.from_euler("x", pitch[f]).as_matrix()
        rot = lambda name: Rot.from_rotvec(rv[f, DONOR_SEGMENT[name]]).as_matrix()
        world = {"torso": rot("torso"), "head": rot("head"), "armN": rot("armN"), "armP": rot("armP"),
                 "legN": rot("legN"), "legP": rot("legP"),
                 "foreN": rot("foreN") if has["foreN"] else rot("armN"),
                 "foreP": rot("foreP") if has["foreP"] else rot("armP"),
                 "shinN": rot("shinN") if has["shinN"] else rot("legN"),
                 "shinP": rot("shinP") if has["shinP"] else rot("legP")}
        parent_R = {"torso": Rroot, "head": Rroot @ world["torso"], "armN": Rroot @ world["torso"],
                    "armP": Rroot @ world["torso"], "legN": Rroot, "legP": Rroot,
                    "foreN": world["armN"], "foreP": world["armP"], "shinN": world["legN"], "shinP": world["legP"]}
        mats["root"][f] = Rroot
        for b in rig.bones[1:]:
            if b == "cape":
                continue
            mats[b][f] = parent_R[b].T @ world[b] if b in world else np.eye(3)       # a bone the donor cannot drive (a foot) stays still
    rvs = {b: unwrap_rotvecs(m) for b, m in mats.items() if b != "cape"}
    lim = {b: np.array([soft_limit(rvs[b][f] * amp[b], caps[b]) for f in range(F)]) for b in rig.bones if b not in ("cape", "root")}
    lim["root"] = rvs["root"]                    # the body's tilt is exact: a lying body must lie flat
    if "cape" in rig.bones:                       # the cape hangs plumb while the body leans, and flares back when the legs stride
        cape = np.zeros((F, 3))
        cape[:, 0] = -0.2 * (lim["root"][:, 0] + lim["torso"][:, 0]) + 0.15 * np.abs(lim["legN"][:, 0] - lim["legP"][:, 0])
        lim["cape"] = np.clip(cape, -np.radians(caps["cape"]), np.radians(caps["cape"]))
    scale = (rig.height / GIANT_HEIGHT)
    root_t = tr[:, DONOR_SEGMENT["torso"]] * scale * 0.9
    if F > 1:
        d = np.sum([np.degrees(np.linalg.norm(np.diff(v, axis=0), axis=1)) for v in lim.values()], axis=0) / 2
        reps = np.clip(np.ceil(d / max_step), 1, 6).astype(int)
        t_new = np.concatenate([i + np.arange(r) / r for i, r in enumerate(reps)] + [[F - 1.0]])
        t_old = np.arange(F)
        lim = {n: np.stack([np.interp(t_new, t_old, v[:, c]) for c in range(3)], 1) for n, v in lim.items()}
        root_t = np.stack([np.interp(t_new, t_old, root_t[:, c]) for c in range(3)], 1)
    n_out = len(root_t)
    out = []
    for f in range(n_out):
        r = {b: lim[b][f] for b in rig.bones}
        if extra:
            r = extra(f, n_out, r)
        out.append(pose_bones(rig.rest, rig.weights, rig.bones, rig.parents, rig.joints, r, root_t[f]))
    out = np.array(out)
    if ground:
        out = plant(rig, out)
    return out


def plant(rig: Rig, out):
    """Put the body's lowest point on the floor (lift or lower), ignoring the cape; the cape may rest on the floor but not sink in."""
    cape = np.zeros(len(rig.rest), bool)
    if "cape" in rig.bones:
        cape = rig.weights.argmax(1) == rig.bones.index("cape")
    body = out[:, ~cape, 1] if (~cape).any() else out[:, :, 1]
    out = out.copy()
    out[:, :, 1] += (rig.floor - body.min(axis=1))[:, None]
    out[:, cape, 1] = np.maximum(out[:, cape, 1], rig.floor)
    return out


def keypose(rig: Rig, keys, n_frames, loop=False, ground=True):
    """Hand-posed animation. keys = [(t, {bone: (rx, ry, rz) degrees}, (tx, ty, tz) root shift), ...], t in 0..1, cosine-eased."""
    ts = np.array([k[0] for k in keys], float)

    def at(t):
        ts_e, keys_e = (np.append(ts, ts[0] + 1.0), keys + [keys[0]]) if loop else (ts, keys)
        t = t % 1.0 if loop else t
        i = int(np.clip(np.searchsorted(ts_e, t, side="right") - 1, 0, len(keys_e) - 2))
        u = np.clip((t - ts_e[i]) / max(ts_e[i + 1] - ts_e[i], 1e-9), 0, 1)
        u = 0.5 - 0.5 * np.cos(np.pi * u)
        a, b = keys_e[i], keys_e[i + 1]
        rot = {n: np.radians((1 - u) * np.array(a[1].get(n, (0, 0, 0)), float) + u * np.array(b[1].get(n, (0, 0, 0)), float))
               for n in rig.bones}
        return rot, (1 - u) * np.array(a[2], float) + u * np.array(b[2], float)

    out = []
    for f in range(n_frames):
        rot, root = at(f / n_frames if loop else f / max(n_frames - 1, 1))
        out.append(pose_bones(rig.rest, rig.weights, rig.bones, rig.parents, rig.joints, rot, root))
    out = np.array(out)
    if ground:
        out = plant(rig, out)
    return out
