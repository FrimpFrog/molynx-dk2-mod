"""Golem (MegaGlest Magic faction) animations driven by real DK2 Giant motion. LICENCE: GPL-3.0."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripts.retarget import donor_motion, kabsch, pose_bones, smooth_weights, unwrap_rotvecs

BONES = ["root", "torso", "armN", "armP", "legN", "legP", "foreN", "foreP"]          # N = negative x, P = positive x
PARENTS = {"root": None, "torso": "root", "armN": "torso", "armP": "torso", "legN": "root", "legP": "root",
           "foreN": "armN", "foreP": "armP"}
JOINTS = {"root": np.array([0, 0.0, -0.3]), "torso": np.array([0, 0.55, 0.35]), "armN": np.array([-0.95, 1.3, 1.0]),
          "armP": np.array([0.95, 1.3, 1.0]), "legN": np.array([-0.55, 0.5, 0.3]),
          "legP": np.array([0.55, 0.5, 0.3]), "foreN": np.array([-0.65, 0.85, 1.05]), "foreP": np.array([0.65, 0.85, 1.05])}
GOLEM_HEIGHT, GIANT_HEIGHT = 2.43, 0.79


def golem_labels(rest: np.ndarray) -> np.ndarray:
    x, y, z = rest.T
    lab = np.zeros(len(rest), int)                              # torso
    # shoulders / upper arms sit out at the sides; forearms and hands hang forward below the elbow. The face and chest
    # (centre front, z >= 1) must stay with the torso, or they stretch whenever an arm moves.
    arm = ((np.abs(x) > 0.62) & (y > 0.85) & (z > 0.5)) | ((y <= 0.85) & (z >= 1.0))
    leg = (y < 0.6) & (z < 1.0)
    lab[arm] = np.where(x[arm] < 0, 1, 2)
    lab[leg & ~arm] = np.where(x[leg & ~arm] < 0, 3, 4)
    fore = arm & (y < 0.85)                                     # below the elbow
    lab[fore] = np.where(x[fore] < 0, 5, 6)
    return lab


def giant_labels(rest: np.ndarray) -> np.ndarray:
    """Donor rest pose (x flipped to right-handed): 0 torso, 1 head, 2 armN, 3 armP, 4 legN, 5 legP, 6 forearmN, 7 forearmP."""
    x, y, _ = rest.T
    arm_x = 0.65 * np.percentile(np.abs(x[y > 0.2]), 95) if (y > 0.2).any() else 0.17   # Giant: 0.17
    lab = np.zeros(len(rest), int)
    lab[(y > 0.64) & (np.abs(x) < arm_x)] = 1
    arm = (np.abs(x) > arm_x) & (y > 0.25)
    lab[arm] = np.where(x[arm] < 0, 2, 3)
    leg = y < 0.28
    lab[leg & ~arm] = np.where(x[leg & ~arm] < 0, 4, 5)
    for side, (upper, lower) in ((-1, (2, 6)), (1, (3, 7))):    # split each arm at its middle, measured from the shoulder
        m = lab == upper
        if m.sum() < 8:
            continue
        pts = rest[m]
        shoulder = pts[np.argmax(pts[:, 1] - 0.5 * np.abs(pts[:, 0]))]
        d = np.linalg.norm(pts - shoulder, axis=1)
        idx = np.where(m)[0]
        lab[idx[d > 0.5 * d.max()]] = lower
    return lab


MAX_STEP_DEG = 20.0
CAPS = {"root": 100.0, "torso": 60.0, "armN": 85.0, "armP": 85.0, "legN": 55.0, "legP": 55.0, "foreN": 110.0, "foreP": 110.0}   # degrees; a heavy golem does not whip its limbs


def _weights(golem_rest, golem_indices):
    """(M, len(BONES)) skin weights; the root bone owns no vertices. Labels 0..6 -> torso, armN, armP, legN, legP, foreN, foreP."""
    w7 = smooth_weights(golem_labels(golem_rest), golem_indices, 7, 2)
    x, y, z = golem_rest.T
    face = (np.abs(x) < 0.58) & (y > 0.8) & (y < 1.8) & (z > 0.5)      # face and chest: rigid with the torso
    w7[face] = 0
    w7[face, 0] = 1.0
    return np.hstack([np.zeros((len(w7), 1)), w7])


def soft_limit(rotvec: np.ndarray, cap_deg: float) -> np.ndarray:
    """Compress large rotation angles smoothly toward cap_deg (identity for small ones)."""
    ang = np.linalg.norm(rotvec)
    if ang < 1e-9:
        return rotvec
    cap = np.radians(cap_deg)
    return rotvec / ang * cap * np.tanh(ang / cap)


def retarget(giant_frames, giant_rest, golem_rest, golem_indices, amp=None, time_ground=True, extra=None):
    """giant_frames (F,N,3), giant_rest (N,3) (x flipped; NaN rows are ignored). -> (F, M, 3) golem frames.
    extra(f, F, rotvecs) -> rotvecs lets the caller add its own bone motion (e.g. a scanning torso).
    Any biped donor works: it is rescaled to the Giant's height first (limb thresholds are in Giant units)."""
    h = np.nanmax(giant_rest[:, 1]) - np.nanmin(giant_rest[:, 1])
    k = GIANT_HEIGHT / h
    giant_rest, giant_frames = giant_rest * k, giant_frames * k
    amp = {"root": 1.0, "torso": 0.8, "armN": 1.0, "armP": 1.0, "legN": 0.8, "legP": 0.8, "foreN": 1.0, "foreP": 1.0, **(amp or {})}
    keep = np.isfinite(giant_frames).all(axis=(0, 2)) & np.isfinite(giant_rest).all(axis=1)
    giant_frames, giant_rest = giant_frames[:, keep], giant_rest[keep]
    glab = giant_labels(giant_rest)
    mot = donor_motion(giant_frames, giant_rest, glab, 8)
    segs = mot["segments"]
    rv, tr = mot["rotvec"], mot["trans"]
    S = lambda name: segs.index(name)   # segment ids 0..5 are all present in the Giant
    w = _weights(golem_rest, golem_indices)
    scale = GOLEM_HEIGHT / GIANT_HEIGHT
    body = np.isin(glab, (0, 1, 4, 5))
    has_fore = [(glab == 6).sum() >= 4, (glab == 7).sum() >= 4]
    F = len(giant_frames)
    mats = {k: np.zeros((F, 3, 3)) for k in BONES}
    pitch = np.zeros(F)
    for f in range(F):
        Rb, _ = kabsch(giant_rest[body], giant_frames[f][body])
        up = Rb @ np.array([0.0, 1.0, 0.0])                    # where the body's up now points
        pitch[f] = np.arctan2(up[2], up[1])                     # tilt about x; carries lying-down poses (die, get up)
    pitch = np.unwrap(pitch)
    for f in range(F):
        Rroot = Rot.from_euler("x", pitch[f]).as_matrix()
        world = {k: Rot.from_rotvec(v).as_matrix() for k, v in
                 {"torso": 0.7 * rv[f, S(0)] + 0.3 * rv[f, S(1)], "armN": rv[f, S(2)], "armP": rv[f, S(3)],
                  "legN": rv[f, S(4)], "legP": rv[f, S(5)],
                  # a donor without a separable forearm: the forearm just follows the arm
                  "foreN": rv[f, S(6)] if has_fore[0] else rv[f, S(2)],
                  "foreP": rv[f, S(7)] if has_fore[1] else rv[f, S(3)]}.items()}
        parent_R = {"torso": Rroot, "armN": Rroot @ world["torso"], "armP": Rroot @ world["torso"],
                    "legN": Rroot, "legP": Rroot}
        parent_R["foreN"] = world["armN"]      # = the arm's composed world rotation
        parent_R["foreP"] = world["armP"]
        mats["root"][f] = Rroot
        for k in ("torso", "armN", "armP", "legN", "legP", "foreN", "foreP"):
            mats[k][f] = parent_R[k].T @ world[k]
    rvs = {k: unwrap_rotvecs(m) for k, m in mats.items()}      # continuous, so soft_limit never flips an axis
    lim = {k: np.array([soft_limit(rvs[k][f] * amp[k], CAPS[k]) for f in range(F)]) for k in BONES}
    root_t = tr[:, S(0)] * scale * 0.7
    # a heavy limb must not jump: if any bone turns more than MAX_STEP_DEG between donor frames, add in-between frames
    if F > 1:
        d = np.sum([np.degrees(np.linalg.norm(np.diff(v, axis=0), axis=1)) for v in lim.values()], axis=0) / 2   # chains add up, but not all bones peak together
        reps = np.clip(np.ceil(d / MAX_STEP_DEG), 1, 6).astype(int)      # in-between frames per donor interval
        t_new = np.concatenate([i + np.arange(r) / r for i, r in enumerate(reps)] + [[F - 1.0]])
        t_old = np.arange(F)
        lim = {n: np.stack([np.interp(t_new, t_old, v[:, c]) for c in range(3)], 1) for n, v in lim.items()}
        root_t = np.stack([np.interp(t_new, t_old, root_t[:, c]) for c in range(3)], 1)
    n_out = len(root_t)
    out = []
    for f in range(n_out):
        r = {n: lim[n][f] for n in BONES}
        if extra:
            r = extra(f, n_out, r)
        out.append(pose_bones(golem_rest, w, BONES, PARENTS, JOINTS, r, root_t[f]))
    out = np.array(out)
    if time_ground:                                             # never sink below the rest floor
        floor = golem_rest[:, 1].min()
        low = out[:, :, 1].min(axis=1)
        out[:, :, 1] += np.maximum(0, floor - low)[:, None]
    return out


def keypose_animation(golem_rest, golem_indices, keys, n_frames, loop=False, ground=True):
    """Hand-posed animation. keys = [(t, {bone: (rx, ry, rz) degrees}, (tx, ty, tz) root shift), ...] with t in 0..1;
    poses are eased (cosine) between keys. loop=True wraps t so the last key blends into the first."""
    w = _weights(golem_rest, golem_indices)
    ts = np.array([k[0] for k in keys])
    def at(t):
        if loop:
            ts_ext = np.append(ts, ts[0] + 1.0); keys_ext = keys + [keys[0]]
            t = t % 1.0
        else:
            ts_ext = ts; keys_ext = keys
        i = int(np.clip(np.searchsorted(ts_ext, t, side="right") - 1, 0, len(keys_ext) - 2))
        span = max(ts_ext[i + 1] - ts_ext[i], 1e-9)
        u = np.clip((t - ts_ext[i]) / span, 0, 1)
        u = 0.5 - 0.5 * np.cos(np.pi * u)
        a, b = keys_ext[i], keys_ext[i + 1]
        rot = {n: np.radians((1 - u) * np.array(a[1].get(n, (0, 0, 0)), float) + u * np.array(b[1].get(n, (0, 0, 0)), float))
               for n in BONES}
        root = (1 - u) * np.array(a[2], float) + u * np.array(b[2], float)
        return rot, root
    out = []
    for f in range(n_frames):
        t = f / n_frames if loop else f / max(n_frames - 1, 1)
        rot, root = at(t)
        out.append(pose_bones(golem_rest, w, BONES, PARENTS, JOINTS, rot, root))
    out = np.array(out)
    if ground:
        floor = golem_rest[:, 1].min()
        out[:, :, 1] += np.maximum(0, floor - out[:, :, 1].min(axis=1))[:, None]
    return out
