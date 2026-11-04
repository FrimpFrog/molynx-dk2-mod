"""Rig of the MegaGlest Battlemage: rest pose, bone labels, joints, skin weights. LICENCE: GPL-3.0.

Rest pose = frame 0 of battlemage_dying.g3d (upright, arms hanging a little out), recentred on his feet.
Body parts are told apart with the model's texture layout (UV islands and the colour under each vertex), because
the mesh is low-poly and the cape, tunic hem and arms all sit close together in space.
"""
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from scripts.battlemage_texture import bake_battlemage
from scripts.g3d import parse_g3d
from scripts.humanoid_retarget import HUMANOID_BONES, Rig
from scripts.retarget import smooth_weights

REST_MODEL = "models/battlemage_dying.g3d"
TEXTURE = "models/texture_battle_mage.png"
# label -> bone. 0 torso, 1 head, 2/3 armN/P, 4/5 legN/P, 6/7 foreN/P, 8/9 shinN/P, 10 cape
LABEL_BONE = ["torso", "head", "armN", "armP", "legN", "legP", "foreN", "foreP", "shinN", "shinP", "cape"]
JOINTS = {"root": (0, 0, 0), "torso": (0, 0.95, 0), "head": (0, 1.82, 0.0),
          "armN": (-0.32, 1.68, 0), "armP": (0.32, 1.68, 0), "foreN": (-0.40, 1.38, 0), "foreP": (0.40, 1.38, 0),
          "legN": (-0.14, 0.86, 0), "legP": (0.14, 0.86, 0), "shinN": (-0.14, 0.45, 0), "shinP": (0.14, 0.45, 0),
          "cape": (0, 1.8, -0.12)}


def battlemage_rest(unit_dir):
    """-> (vertices (M,3) centred on the feet, triangle indices (T,3), offset subtracted)."""
    me = parse_g3d((unit_dir / REST_MODEL).read_bytes()).meshes[0]
    v = me.vertices[0].astype(float)
    feet = v[v[:, 1] < 0.25]
    off = np.array([feet[:, 0].mean(), 0.0, feet[:, 2].mean()])
    return v - off, me.indices.reshape(-1, 3), off


def _islands(n, idx):
    e = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]]])
    return connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)[1]


def _surroundings(unit_dir, uv):
    """RGB texture colours in a 3x3 pixel block around each vertex's UV: (M, 9, 3) of the baked texture."""
    tex = np.asarray(bake_battlemage(unit_dir / TEXTURE)).astype(int)
    h, w = tex.shape[:2]
    px = np.clip((uv[:, 0] * w).astype(int), 1, w - 2)
    py = np.clip(((1 - uv[:, 1]) * h).astype(int), 1, h - 2)
    return np.stack([tex[py + dy, px + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1), px, py


def battlemage_labels(rest, indices, uv, near):
    """near: (M,9,3) texture colours around each vertex. -> labels 0..10 (see LABEL_BONE)."""
    x, y, z = rest.T
    comp = _islands(len(rest), indices)
    sizes = np.bincount(comp)
    # the cape drape is the island whose texture sits in the top band of the atlas (v < 65 px of 256)
    v_px = (1 - uv[:, 1]) * 256
    top_band = np.array([np.mean(v_px[comp == c]) for c in range(len(sizes))]) < 65
    cape = top_band[comp]
    r, g, b = near[..., 0], near[..., 1], near[..., 2]
    yellow = ((r > 180) & (g > 130) & (b < 120)).mean(1) > 0.5                  # tunic / hat cloth (and red crab)
    redcrab = ((r > 150) & (g < 90) & (b < 90)).mean(1) > 0.3
    skin_or_band = ~(yellow | redcrab)                                         # skin, bracers, belt, boots
    lab = np.zeros(len(rest), int)
    in_arm_band = (v_px < 148) & (v_px > 60) & skin_or_band & ~cape            # the pink arm islands of the atlas
    legs = (v_px >= 148) & skin_or_band & ~cape
    lab[legs] = np.where(x[legs] < 0, 4, 5)
    arm = in_arm_band & (np.abs(x) > 0.285)                                     # the tunic's side edge (belt ends, side seam) reaches |x| 0.27: torso
    lab[arm] = np.where(x[arm] < 0, 2, 3)
    fore = arm & (y < 1.4)
    lab[fore] = np.where(x[fore] < 0, 6, 7)
    shin = np.isin(lab, (4, 5)) & (y < 0.45)
    lab[shin] += 4
    # the yellow sleeve cloth at the shoulder sits in the arm's UV island too
    sleeve = (lab == 0) & ~cape & (np.abs(x) > 0.26) & (y > 1.45) & (y < 1.82)
    lab[sleeve] = np.where(x[sleeve] < 0, 2, 3)
    lab[(y > 1.84) & (np.abs(x) < 0.3) & ~cape] = 1
    lab[cape] = 10
    return lab


def battlemage_weights(rest, indices, uv, near):
    n = len(HUMANOID_BONES)
    hard = battlemage_labels(rest, indices, uv, near)
    w = smooth_weights(hard, indices, 11, 2)
    limb = np.isin(hard, (2, 3, 4, 5, 6, 7, 8, 9))
    onehot = np.eye(11)[hard]
    w[limb] = 0.8 * onehot[limb] + 0.2 * w[limb]
    out = np.zeros((len(rest), n))
    for i, bone in enumerate(LABEL_BONE):
        out[:, HUMANOID_BONES.index(bone)] = w[:, i]
    x, y, z = rest.T
    torso = HUMANOID_BONES.index("torso")
    cape = hard == 10
    k = np.clip((1.72 - y) / 0.4, 0, 1)                                          # collar follows the shoulders, the drape swings
    for j in np.where(cape)[0]:
        out[j] = 0
        out[j, HUMANOID_BONES.index("cape")] = k[j]
        out[j, torso] = 1 - k[j]
    # the tunic hem (torso-labelled cloth below the waist) follows the thighs a little
    hem = (hard == 0) & (y < 1.0) & ~cape
    for j in np.where(hem)[0]:
        f = np.clip((1.0 - y[j]) / 0.3, 0, 0.7)
        leg = HUMANOID_BONES.index("legN" if x[j] < 0 else "legP")
        out[j] *= (1 - f)
        out[j, leg] += f
    # the tunic below the shoulder is the body's, whatever the smoothing leaked onto it from the arm next to it
    side = (hard == 0) & ~cape & (y < 1.55)
    for b in ("armN", "armP", "foreN", "foreP"):
        out[side, HUMANOID_BONES.index(b)] = 0
    out[side, torso] = np.maximum(out[side, torso], 1e-3)
    # tunic cloth that rides on the arm (the sleeve) follows the torso half way, so a raised arm does not tent the tunic up to the wrist
    cloth = ((near[..., 0] > 180) & (near[..., 1] > 130) & (near[..., 2] < 120)).mean(1) > 0.5
    sl = np.isin(hard, (2, 3, 6, 7)) & cloth
    out[sl] *= 0.5
    out[sl, torso] += 0.5
    out /= out.sum(1, keepdims=True)
    face = (np.abs(x) < 0.2) & (y > 1.55) & (y <= 1.84) & (z > 0.0) & ~cape                # neck / upper chest follow the torso
    out[face] = 0
    out[face, torso] = 1.0
    return out


def battlemage_rig(unit_dir) -> Rig:
    rest, idx, _ = battlemage_rest(unit_dir)
    me = parse_g3d((unit_dir / REST_MODEL).read_bytes()).meshes[0]
    near, _, _ = _surroundings(unit_dir, me.uv)
    return Rig(rest=rest, indices=idx, joints={k: np.array(v, float) for k, v in JOINTS.items()},
               weights=battlemage_weights(rest, idx, me.uv, near), height=rest[:, 1].max() - rest[:, 1].min(),
               floor=rest[:, 1].min())
