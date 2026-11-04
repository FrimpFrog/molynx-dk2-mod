"""Retarget DK2 (Giant) animations onto a hunched MegaGlest-style body by rigid segments.

LICENCE: GPL-3.0.
DK2 animation is per-vertex, so per-frame limb rotations are measured (Kabsch fit of each donor
segment against its rest pose), then applied to the target mesh with soft-weighted bones.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid (R, t) with Q ~= P @ R.T + t."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def smooth_weights(hard: np.ndarray, indices: np.ndarray, n_bones: int, iterations: int = 3) -> np.ndarray:
    """One-hot bone labels, blurred across mesh edges so joints bend smoothly."""
    n = len(hard)
    w = np.zeros((n, n_bones)); w[np.arange(n), hard] = 1
    nb = [set() for _ in range(n)]
    for a, b, c in np.asarray(indices).reshape(-1, 3):
        nb[a] |= {b, c}; nb[b] |= {a, c}; nb[c] |= {a, b}
    for _ in range(iterations):
        new = w.copy()
        for i in range(n):
            if nb[i]:
                new[i] = 0.5 * w[i] + 0.5 * w[list(nb[i])].mean(0)
        w = new
    return w / w.sum(1, keepdims=True)


def unwrap_rotvecs(mats: np.ndarray) -> np.ndarray:
    """Rotation matrices (F,3,3) -> rotation vectors (F,3) that change continuously from frame to frame
    (angles may pass pi instead of the axis flipping)."""
    q = Rot.from_matrix(mats).as_quat()                     # x, y, z, w
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    if q[0, 3] < 0:
        q = -q
    vn = np.linalg.norm(q[:, :3], axis=1)
    ang = 2 * np.arctan2(vn, q[:, 3])
    axis = np.divide(q[:, :3], vn[:, None], out=np.zeros_like(q[:, :3]), where=vn[:, None] > 1e-12)
    return axis * ang[:, None]


def donor_motion(frames: np.ndarray, rest: np.ndarray, labels: np.ndarray, n_segments: int | None = None) -> dict:
    """frames (F,N,3), rest (N,3), labels (N,) -> rotvec (F,S,3) world delta, trans (F,S,3) centroid shift."""
    segs = list(range(n_segments)) if n_segments else sorted(set(labels.tolist()))
    F = len(frames)
    rv = np.zeros((F, len(segs), 3)); tr = np.zeros((F, len(segs), 3))
    for si, s in enumerate(segs):
        m = labels == s
        if m.sum() < 4:                       # a limb the donor does not have: no motion
            continue
        c0 = rest[m].mean(0)
        Rs = np.zeros((F, 3, 3))
        for f in range(F):
            Rs[f], _ = kabsch(rest[m], frames[f][m])
            tr[f, si] = frames[f][m].mean(0) - c0
        rv[:, si] = unwrap_rotvecs(Rs)
    return {"rotvec": rv, "trans": tr, "segments": segs}


def pose_bones(rest, weights, names, parents, joints, rotvec, root_t):
    """Linear-blend skinning. rotvec[name] rotates that bone about joints[name] in world axes, inside its parent's frame."""
    mats = {}
    def world(name):
        if name in mats:
            return mats[name]
        R = Rot.from_rotvec(rotvec[name]).as_matrix()
        j = joints[name]
        local_R, local_t = R, j - R @ j
        p = parents[name]
        if p is None:
            M = (local_R, local_t + root_t)
        else:
            PR, Pt = world(p)
            M = (PR @ local_R, PR @ local_t + Pt)
        mats[name] = M
        return M
    out = np.zeros_like(rest)
    for bi, name in enumerate(names):
        R, t = world(name)
        out += weights[:, bi:bi + 1] * (rest @ R.T + t)
    return out
