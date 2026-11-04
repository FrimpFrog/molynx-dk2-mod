"""Rig of the MegaGlest Behemoth (Magic faction, spiked-club giant): rest pose, bone labels, joints, skin weights.
LICENCE: GPL-3.0 (model data: MegaGlest, CC BY-SA 3.0).

Rest pose = frame 0 of behemoth_standing.g3d (club on the shoulder), recentred on the feet. Vertex order = the standing model's
mesh order: first the 142-vertex club, then the 586-vertex body. The club is bound to the forearm that holds it.
All four MegaGlest files number their vertices identically, which lets their frames be used to check how rigid each part is.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from scripts.g3d import parse_g3d
from scripts.humanoid_retarget import Rig
from scripts.retarget import smooth_weights

REST_MODEL = "models/behemoth_standing.g3d"
BONES = ["root", "torso", "head", "armN", "armP", "legN", "legP", "foreN", "foreP", "shinN", "shinP", "footN", "footP"]
PARENTS = {"root": None, "torso": "root", "head": "torso", "armN": "torso", "armP": "torso", "legN": "root", "legP": "root",
           "foreN": "armN", "foreP": "armP", "shinN": "legN", "shinP": "legP", "footN": "shinN", "footP": "shinP"}
# label -> bone:  0 torso, 1 head, 2 armN, 3 armP, 4 legN, 5 legP, 6 foreN, 7 foreP, 8 shinN, 9 shinP, 10 club (follows foreN)
LABEL_BONE = ["torso", "head", "armN", "armP", "legN", "legP", "foreN", "foreP", "shinN", "shinP", "foreN"]
CLUB_VERTICES = 142
ANKLE_Y = 0.27         # below this a shin is the foot; the foot bone only matters in poses that fold it (kneeling)


def behemoth_rest(unit_dir):
    """-> (vertices (728,3) centred on the feet, triangle indices (T,3) into them, offset subtracted)."""
    model = parse_g3d((Path(unit_dir) / REST_MODEL).read_bytes())
    verts, tris, off = [], [], 0
    for me in model.meshes:
        verts.append(me.vertices[0].astype(float))
        tris.append(me.indices.reshape(-1, 3) + off)
        off += me.vertices.shape[1]
    v = np.concatenate(verts)
    feet = v[v[:, 1] < 0.25]
    o = np.array([feet[:, 0].mean(), 0.0, feet[:, 2].mean()])
    return v - o, np.concatenate(tris), o


def behemoth_labels(rest):
    """Hard labels 0..10 (see LABEL_BONE) from the body plan: legs below the hips, arms outside the chest, head above the neck."""
    x, y, z = rest.T
    n = len(rest)
    club = np.zeros(n, bool)
    club[:CLUB_VERTICES] = True
    cx = np.median(x[(y < 0.6) & ~club])
    lab = np.zeros(n, int)
    leg = (y < 1.45) & ~club
    lab[leg] = np.where(x[leg] < cx, 4, 5)
    shin = leg & (y < 0.62)
    lab[shin] = np.where(x[shin] < cx, 8, 9)
    head = (y > 3.3) & (np.abs(x - cx) < 0.5) & ~leg & ~club
    lab[head] = 1
    for side, sel in ((2, (x < cx - 0.55)), (3, (x > cx + 0.65))):
        arm = sel & (y > 1.2) & (y < 3.3) & ~club & ((y > 2.0) | (np.abs(x - cx) > 0.75))     # loincloth strips hang just inside the hand
        lab[arm] = side
        idx = np.where(arm)[0]
        pts = rest[idx]
        shoulder = pts[np.argmax(pts[:, 1] - 0.5 * np.abs(pts[:, 0] - cx))]
        d = np.linalg.norm(pts - shoulder, axis=1)
        lab[idx[d > 0.5 * d.max()]] = side + 4                    # the half away from the shoulder is the forearm
    lab[club] = 10
    return lab


def _joint(rest, a, b, k=10):
    """Pivot between two segments: the mean of the k closest vertex pairs' midpoints."""
    A, B = rest[a], rest[b]
    d = np.linalg.norm(A[:, None] - B[None], axis=2)
    flat = np.argsort(d, axis=None)[:k]
    i, j = np.unravel_index(flat, d.shape)
    return ((A[i] + B[j]) / 2).mean(0)


def behemoth_joints(rest, lab):
    m = lambda *ls: np.isin(lab, ls)
    j = {"root": np.array([rest[m(0), 0].mean(), 1.7, 0.0]),
         "torso": np.array([rest[m(0), 0].mean(), 1.7, 0.0])}
    j["head"] = _joint(rest, m(1), m(0))
    for s, (arm, fore, leg, shin) in {"N": (2, 6, 4, 8), "P": (3, 7, 5, 9)}.items():
        a = rest[m(arm)]
        j["arm" + s] = a[a[:, 1] > a[:, 1].max() - 0.25].mean(0)                 # the shoulder: the top of the arm mass
        j["fore" + s] = _joint(rest, m(arm), m(fore))
        thigh = rest[m(leg) & (rest[:, 1] > 0.7)]
        j["leg" + s] = np.array([thigh[:, 0].mean(), 1.45, thigh[:, 2].mean()])   # the hip: above the middle of the thigh
        j["shin" + s] = _joint(rest, m(leg), m(shin))
        f = rest[m(shin)]
        j["foot" + s] = np.array([f[:, 0].mean(), ANKLE_Y, f[:, 2].mean()])
    return j


def behemoth_weights(rest, indices, lab, joints=None):
    w = smooth_weights(lab, indices, 11, 2)
    limb = np.isin(lab, (2, 3, 4, 5, 6, 7, 8, 9, 10))
    onehot = np.eye(11)[lab]
    w[limb] = 0.8 * onehot[limb] + 0.2 * w[limb]
    # the loincloth hangs from the hips down to y~1.0 and swings far less than the thigh under it: half torso, half leg
    skirt = np.isin(lab, (4, 5)) & (rest[:, 1] > 0.9)
    w[skirt] = 0.5 * w[skirt] + 0.5 * np.eye(11)[0]
    if joints is not None:                       # the shoulder pad rolls with the chest as the arm lifts
        for lbl, bone in ((2, "armN"), (3, "armP")):
            near = (lab == lbl) & (np.linalg.norm(rest - joints[bone], axis=1) < 0.5)
            w[near] = 0.5 * w[near] + 0.5 * np.eye(11)[0]
    out = np.zeros((len(rest), len(BONES)))
    for i, bone in enumerate(LABEL_BONE):
        out[:, BONES.index(bone)] += w[:, i]
    for lbl, shin, foot in ((8, "shinN", "footN"), (9, "shinP", "footP")):      # the foot takes over from the shin toward the toes
        toe = np.where(lab == lbl, np.clip((ANKLE_Y + 0.08 - rest[:, 1]) / 0.16, 0, 1), 0.0)
        out[:, BONES.index(foot)] = out[:, BONES.index(shin)] * toe
        out[:, BONES.index(shin)] *= 1 - toe
    return out / out.sum(1, keepdims=True)


@lru_cache(maxsize=2)
def behemoth_rig(unit_dir) -> Rig:
    rest, idx, _ = behemoth_rest(unit_dir)
    lab = behemoth_labels(rest)
    joints = behemoth_joints(rest, lab)
    return Rig(rest=rest, indices=idx, joints=joints, weights=behemoth_weights(rest, idx, lab, joints),
               height=rest[CLUB_VERTICES:, 1].max() - rest[:, 1].min(),     # the body, not the club raised above the head
               floor=rest[:, 1].min(), bones=list(BONES), parents=dict(PARENTS))


def stretched_triangles(unit_dir, lib, min_len=0.7, ratio=3.0):
    """Triangle numbers (club's first, then the body's) that are flush against the body at rest (a hand on the hip against the loincloth
    and thigh, the shoulder pad against the chest) but stretch to long spikes in some clip as the limb swings away."""
    rig = behemoth_rig(unit_dir)
    tri = rig.indices.reshape(-1, 3)

    def longest(p):
        return np.max([np.linalg.norm(p[..., tri[:, a], :] - p[..., tri[:, b], :], axis=-1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)

    rest = longest(rig.rest)
    worst = np.zeros(len(tri))
    for a in lib.values():
        worst = np.maximum(worst, longest(np.asarray(a["frames"])).max(0))
    return np.where((worst > min_len) & (worst / np.maximum(rest, 0.05) > ratio))[0]
