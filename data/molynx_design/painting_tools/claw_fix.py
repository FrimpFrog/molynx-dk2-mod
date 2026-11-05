"""Design fix: the free hand's 5-vertex claw (vertices 244, 571-574) is labelled torso/leg in the Behemoth rig, so the
retargeted DK2 clips leave it floating by the hip. In every frame it is moved rigidly with the 12 hand vertices it sits on
(Kabsch fit rest -> frame)."""
import numpy as np

CLAW = np.array([244, 574, 573, 572, 571])
HAND = np.array([231, 560, 226, 566, 238, 240, 236, 559, 564, 558, 556, 237])

def _kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca

def fix_claw(data, rest):
    out = {}
    for name, clip in data.items():
        frames = []
        for fr in clip['frames']:
            body = np.array(fr[0], float, copy=True)
            R, t = _kabsch(rest[HAND], body[HAND])
            body[CLAW] = rest[CLAW] @ R.T + t
            frames.append([body] + list(fr[1:]))
        out[name] = {**clip, 'frames': frames}
    return out


# The club hand (10 vertices tagged 'head' by the Behemoth rig) follows his head in the retargeted DK2 clips while the
# club (and the forearm holding it) follow the arm, so the wrist twists. In those clips the hand now moves rigidly with
# the club it grips. His own MegaGlest clips (stand, walk, attack, die) and the T-pose are left exactly as they are.
CLUB_HAND = np.array([422, 423, 425, 677, 680, 681, 682, 683, 684, 685])
CLUB = np.arange(142)
NATIVE = {'stand', 'walk', 'attack', 'die', 'tpose'}

def fix_club_hand(data, rest):
    out = {}
    for name, clip in data.items():
        if name in NATIVE:
            out[name] = clip
            continue
        frames = []
        for fr in clip['frames']:
            body = np.array(fr[0], float, copy=True)
            R, t = _kabsch(rest[CLUB], body[CLUB])
            body[CLUB_HAND] = rest[CLUB_HAND] @ R.T + t
            frames.append([body] + list(fr[1:]))
        out[name] = {**clip, 'frames': frames}
    return out


# The arm that holds the mace (its own piece of the model: vertices 382-429 and 654-693) is tagged mostly 'torso' and
# 'head' by the rig, so in the retargeted clips its forearm and hand bend away from the club (twisted backwards when he
# bows). In those clips the arm now moves with the club it holds, blending back to its retargeted position near the
# shoulder so it stays attached. Native clips and the T-pose are untouched.
ARM_N = np.r_[382:430, 654:694]

def fix_club_arm(data, rest, shoulder):
    d = np.linalg.norm(rest[ARM_N] - shoulder, axis=1)
    w = np.clip((d - 0.25) / 0.45, 0, 1)[:, None]          # 0 at the shoulder, 1 from the elbow on
    out = {}
    for name, clip in data.items():
        if name in NATIVE:
            out[name] = clip
            continue
        frames = []
        for fr in clip['frames']:
            body = np.array(fr[0], float, copy=True)
            R, t = _kabsch(rest[CLUB], body[CLUB])
            body[ARM_N] = (1 - w) * body[ARM_N] + w * (rest[ARM_N] @ R.T + t)
            frames.append([body] + list(fr[1:]))
        out[name] = {**clip, 'frames': frames}
    return out


# The clips made with the club arm locked (praying, happy, eating, the dances, manufacturing) are meant to keep the mace on
# his shoulder while the free arm does the work. The rig lets the forearm and hand drift (they are tagged 'torso'/'head'),
# so the arm swings down across his belly with the hand bent back. In those clips the whole mace arm, the club and the
# mace now keep exactly their idle-pose (stand frame 0) place on his upper chest, blending into the shoulder.
LOCKED = {'eating', 'praying', 'happy', 'celebrating', 'victory_celebration', 'manufacturing'}

def lock_club_arm(data, weights, bones, shoulder, locked=LOCKED):
    ref = data['stand']['frames'][0]
    ref_body, ref_mace = np.asarray(ref[0], float), np.asarray(ref[1], float)
    torso = np.nonzero((weights[:, bones.index('torso')] > 0.95) & (ref_body[:, 1] > 2.6))[0]
    torso = np.setdiff1d(torso, np.r_[ARM_N, CLUB])
    d = np.linalg.norm(ref_body[ARM_N] - shoulder, axis=1)
    w = np.clip((d - 0.2) / 0.4, 0, 1)[:, None]
    out = {}
    for name, clip in data.items():
        if name not in locked:
            out[name] = clip
            continue
        frames = []
        for fr in clip['frames']:
            body = np.array(fr[0], float, copy=True)
            R, t = _kabsch(ref_body[torso], body[torso])
            body[ARM_N] = (1 - w) * body[ARM_N] + w * (ref_body[ARM_N] @ R.T + t)
            body[CLUB] = ref_body[CLUB] @ R.T + t
            frames.append([body, ref_mace @ R.T + t] + list(fr[2:]))
        out[name] = {**clip, 'frames': frames}
    return out
