"""Build the custom Behemoth character (see character_parts.py) as viewer data: T-pose, the native idle / walk / attack clips.
LICENCE: GPL-3.0 (base model: MegaGlest, CC BY-SA 3.0).

Every clip is the base model's own animation with the new parts and the bigger head added per frame.
The T-pose is the base model posed with its skin weights (arms out, forearms straight), then dressed the same way.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripts.behemoth_rig import CLUB_VERTICES, behemoth_labels, behemoth_rest, behemoth_rig
from scripts.character_parts import Character
from scripts.character_paint import paint_character
from scripts.character_texture import build_atlas
from scripts.g3d import parse_g3d
from scripts.glest_export import SKILL_LABELS, parse_skills, playback_fps
from scripts.retarget import pose_bones

TEX_PREFIX = "custom_character"
TWIST = {"N": 90.0, "P": 90.0}                  # roll of each arm about its own axis in the T-pose, degrees


def _between(a, b):
    """Rotation vector that turns direction a onto direction b by the shortest way."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.zeros(3) if a @ b > 0 else np.array([0.0, 0.0, np.pi])
    return axis / s * np.arctan2(s, a @ b)


ARM_P = np.r_[198:246, 542:574]                      # the vertex blocks of the two arms in the original model (each arm is its own piece)
ARM_N = np.r_[382:430, 654:694]
AXIS_X = 0.05                                       # the body's centre line


def mirrored(p):
    p = np.array(p, float)
    p[..., 0] = 2 * AXIS_X - p[..., 0]
    return p


def tpose_frame(unit_dir) -> np.ndarray:
    """Base vertices (728,3) in a T-pose: the good (P) arm straight out sideways, forearm in line with the upper arm, club pointing up in
    the fist of the mirrored second arm (see tpose_mirror: the club arm of the original is barely modelled, so it is replaced by a mirror image)."""
    rig = behemoth_rig(Path(unit_dir))
    rest, lab, j = rig.rest, behemoth_labels(rig.rest), rig.joints
    rot = {b: np.zeros(3) for b in rig.bones}
    for side, arm, fore, out in (("N", "armN", "foreN", -1.0), ("P", "armP", "foreP", 1.0)):
        la, lf = (2, 6) if side == "N" else (3, 7)
        hand = rest[(lab == lf) & (np.arange(len(rest)) >= 142)]
        far = hand[np.argsort(np.linalg.norm(hand - j[fore], axis=1))[-6:]].mean(0)
        upper = j[fore] - j[arm]
        lower = far - j[fore]
        want = np.array([out, 0.0, 0.0])
        r1 = _between(upper, want)
        R1 = Rot.from_rotvec(r1).as_matrix()
        Rt = Rot.from_euler('x', np.radians(TWIST[side])).as_matrix()
        rot[arm] = Rot.from_matrix(Rt @ R1).as_rotvec()
        rot[fore] = _between(lower, R1.T @ want)               # world = R1 @ R_fore, so the forearm lines up with the arm
        if side == "P":
            hand_idx = np.where((lab == lf) & (np.arange(len(rest)) >= 142))[0]
            hand_idx = hand_idx[np.argsort(np.linalg.norm(rest[hand_idx] - j[fore], axis=1))[-6:]]
    # arms turn as solid pieces: the blend with the chest that the walk uses would fold the shoulder flat at 90 degrees
    W = rig.weights.copy()
    hand_claw = (lab == 1) & (rest[:, 2] > 0.8) & (np.arange(len(rest)) >= 142)       # the raised hand's claws are tagged 'head' by height
    lab2 = lab.copy()
    lab2[hand_claw] = 6
    for side, arm, fore, la, lf in (("N", "armN", "foreN", 2, 6), ("P", "armP", "foreP", 3, 7)):
        ia, ifo = rig.bones.index(arm), rig.bones.index(fore)
        v = j[fore] - j[arm]
        v = v / np.linalg.norm(v)
        t = (rest - j[fore]) @ v
        elbow = np.clip(t / 0.25 + 0.5, 0, 1)                                         # 0 on the shoulder side of the elbow, 1 on the hand side
        sel = np.zeros(len(rest), bool)
        if side == "N":                                                               # the club arm is replaced by the mirror image: its leftovers stay put
            W[ARM_N] = 0.0
            W[ARM_N, rig.bones.index("torso")] = 1.0
            continue
        sel[ARM_P] = True
        W[sel] = 0.0
        W[sel, ia] = 1 - elbow[sel]
        W[sel, ifo] = elbow[sel]
    pose = pose_bones(rig.rest, W, rig.bones, rig.parents, rig.joints, rot, np.zeros(3))
    pose = pose.copy()
    pose[ARM_P, 0] += 0.40 - pose[ARM_P, 0].min()                 # the arm's inner end tucks into the shoulder again (the swing leaves a gap)
    # the club stands upright in the fist (its handle end at the hand), instead of swinging round the elbow with the forearm
    club = rest[:142]
    axis0 = np.linalg.svd(club - club.mean(0), full_matrices=False)[2][0]
    far0 = rest[hand_idx].mean(0)                               # (the hand end of the club is the one nearer the fist)
    if axis0 @ (club.mean(0) - mirrored(rest[hand_idx].mean(0))) < 0:
        axis0 = -axis0                                          # handle -> head
    handle = club[np.argmin(club @ axis0)]
    R = Rot.from_rotvec(_between(axis0, np.array([0.0, 1.0, 0.0]))).as_matrix()
    hand = mirrored(pose[hand_idx].mean(0))
    pose = pose.copy()
    pose[:142] = (club - handle) @ R.T + hand + np.array([0.0, -0.15, 0.0])
    return pose


def tpose_mirror(unit_dir, pose, tris, uv):
    """The second arm of the T-pose: the posed P arm (its own triangles and UVs) mirrored across the body. -> (vertices, uv, triangles, drop)
    where drop is the mask of body triangles of the original club arm that this arm replaces."""
    rig = behemoth_rig(Path(unit_dir))
    rest, lab = rig.rest, behemoth_labels(rig.rest)
    arm = np.zeros(len(rest), bool)
    arm[ARM_P] = True
    old = np.zeros(len(rest), bool)
    old[ARM_N] = True
    keep = np.all(arm[tris], axis=1)
    remap = -np.ones(len(rest), int)
    remap[arm] = np.arange(arm.sum())
    new_tris = remap[tris[keep]][:, ::-1]                        # mirrored: the winding turns round
    limb = old & np.isin(lab, (2, 6))                           # the limb itself (not the chest surface that the raised arm happens to share)
    return mirrored(pose[arm]), uv[arm], new_tris, np.all(old[tris], axis=1) | np.any(limb[tris], axis=1)


def stretched(rest, tris, posed, min_len=0.55, ratio=2.6):
    """Triangle numbers whose longest edge is flush at rest but long and thin in `posed`."""
    def longest(p):
        return np.max([np.linalg.norm(p[tris[:, a]] - p[tris[:, b]], axis=-1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    r, q = longest(rest), longest(posed)
    return np.where((q > min_len) & (q / np.maximum(r, 0.05) > ratio))[0]


def native_clips(unit_dir) -> dict:
    """{label: (frames (F,728,3) recentred like the rig, fps, wraps)} for the unit's own stand / walk / attack clips."""
    unit_dir = Path(unit_dir)
    _, _, off = behemoth_rest(unit_dir)
    out = {}
    for sk in parse_skills(unit_dir / "behemoth.xml"):
        if sk["type"] not in ("stop", "move", "attack", "die") or SKILL_LABELS[sk["type"]] in out:
            continue
        model = parse_g3d((unit_dir / sk["paths"][0]).read_bytes())
        n = model.frame_count
        meshes = sorted(model.meshes, key=lambda m: m.vertices.shape[1] != CLUB_VERTICES)     # the club's mesh first, as in the standing file
        frames = np.stack([np.concatenate([m.vertices[f].astype(float) for m in meshes]) for f in range(n)]) - off
        looping = sk["type"] != "die"
        out[SKILL_LABELS[sk["type"]]] = (frames, playback_fps(n, sk["anim_speed"], looping), looping)
    return out


def build_character(unit_dir, clips=("stand", "walk"), extra=None, drop=None):
    """-> (data for write_dk2v, {texture name: image}, character).  Parts: the base body (club + body, one texture) and the new parts.
    extra: more clips {name: {"frames", "fps", "wrap"}} in the Behemoth's vertex layout (the retargeted DK2 clips); drop: triangle numbers of
    the body that are left out of every clip but the T-pose (flush against the body at rest, torn into spikes by some clip)."""
    unit_dir = Path(unit_dir)
    model = parse_g3d((unit_dir / "models/behemoth_standing.g3d").read_bytes())
    uv = np.concatenate([me.uv for me in model.meshes]).astype(float)
    uv[:, 1] = 1 - uv[:, 1]
    rest, idx, _ = behemoth_rest(unit_dir)
    ch = Character(unit_dir, uv, idx)
    body = build_atlas(unit_dir / "models/texture_daemon_giant.png").crop((0, 0, 256, 256))
    textures = {f"{TEX_PREFIX}/body": paint_character(unit_dir, body, uv, idx)}

    from scripts.character_mace import mace_texture
    from PIL import Image as _Image
    art = Path(__file__).resolve().parent.parent / "data" / "character_art"
    textures[f"{TEX_PREFIX}/mace"] = mace_texture()
    keep_all = ~np.all(idx < 142, axis=1)                         # the wooden club is gone: the Troll's mace replaces it
    if drop is not None:
        keep_all[np.asarray(drop, int)] = False
    body_idx = idx[keep_all]
    parts = [{"texture": f"{TEX_PREFIX}/body", "uv": uv.astype(np.float32), "indices": body_idx.reshape(-1).astype(np.uint16)}]
    for p in ch.parts:
        parts.append({"texture": f"{TEX_PREFIX}/{p.texture}", "uv": p.uv.astype(np.float32), "indices": p.t.reshape(-1).astype(np.uint16)})

    def dress(frames):
        out = []
        for f in frames:
            grown, extra = ch.apply(f)
            out.append([grown] + [v for _, _, v in extra])
        return out

    tp = tpose_frame(unit_dir)
    mv, muv, mt, old_arm = tpose_mirror(unit_dir, tp, idx, uv)
    gone = stretched(rest, idx, tp, min_len=0.4, ratio=2.0)    # triangles that were flush against the body and are torn into spikes by the pose
    keep = np.ones(len(idx), bool)
    keep[gone] = False
    keep &= ~np.all(idx < 142, axis=1)
    keep &= ~old_arm
    armP = np.zeros(len(rest), bool)
    armP[ARM_P] = True
    mixed = np.any(armP[idx], axis=1) & ~np.all(armP[idx], axis=1)
    keep &= ~mixed                                        # triangles that tie the posed arm to the chest would be torn by the pose
    raised = np.zeros(len(rest), bool)
    raised[ARM_N] = True
    raised &= rest[:, 2] > 0.7
    raised[[430, 689, 694, 695]] = True
    keep &= ~np.any(raised[idx], axis=1)                  # what is left of the club arm that stuck out in front of the chest
    hv = (behemoth_labels(rest) == 1) & (rest[:, 2] > 0.8) & (np.arange(len(rest)) >= 142)
    keep &= ~np.any(hv[idx], axis=1)                      # the club hand's claws belong to the arm that the mirror replaced
    tparts = [{**parts[0], "indices": idx[keep].reshape(-1).astype(np.uint16)}] + parts[1:]
    tparts.append({"texture": f"{TEX_PREFIX}/body", "uv": muv.astype(np.float32), "indices": mt.reshape(-1).astype(np.uint16)})
    tframe = [v for v in dress([tp])[0]] + [mv]
    data = {"tpose": {"frames": [tframe, tframe], "parts": tparts, "wrap": True, "fps": 1.0}}
    native = native_clips(unit_dir)
    for name in clips:
        frames, fps, wrap = native[name]
        data[name] = {"frames": dress(frames), "parts": parts, "wrap": wrap, "fps": fps}
    for name, c in (extra or {}).items():
        data[name] = {"frames": dress(c["frames"]), "parts": parts, "wrap": c["wrap"], "fps": c["fps"]}
    return data, textures, ch
