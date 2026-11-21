"""Posing the Gargoyle: the Dragon's pose machinery (scripts.dragon_pose: euler, blend, ease, keyed, the
spec format with "aim") on the gargoyle's own skeleton, and two-bone leg IK for his digitigrade legs.

A pose spec is a dict:
  bone name -> (rx, ry, rz)            local rotation, degrees, x then y then z (world-aligned bind frames)
  "aim": {bone: (direction, normal)}    the bone's world direction (bind head -> tail) and the normal of
                                        its plane (wing bones: the membrane's front face), resolved after
                                        its parents
  "root": (dx, dy, dz)                  root offset (model units)
Keys ending in L get mirrored to R by `sym` (x rotations kept, y and z negated; aims mirrored in x).

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_pose import Pose, _frame, blend, ease, euler, keyed, sym   # noqa: F401 (re-exported)
from scripts.gargoyle_model import BONE_INDEX, BONE_NAMES, HEAD, LEGS, NB_BONES, PARENT, TAIL, \
    gargoyle_forward

WING_BONE_STEMS = ("wing1", "wing2", "f1", "f2", "f3")


class GargoylePose(Pose):
    def __init__(self, local=None, root=(0, 0, 0)):
        self.local = np.tile(np.eye(3), (NB_BONES, 1, 1)) if local is None else np.array(local)
        self.root = np.asarray(root, float)

    def world(self):
        return gargoyle_forward(self.local, self.root)

    def rotate(self, bone, rx=0, ry=0, rz=0, pre=False):
        b = BONE_INDEX[bone]
        R = euler(rx, ry, rz)
        self.local[b] = R @ self.local[b] if pre else self.local[b] @ R
        return self

    def set_world(self, bone, Rworld):
        b = BONE_INDEX[bone]
        Rw, _ = self.world()
        p = PARENT[b]
        parent = np.eye(3) if p is None else Rw[p]
        self.local[b] = parent.T @ Rworld
        return self


def _bind_normal(b):
    name = BONE_NAMES[b]
    if name[:-1] in WING_BONE_STEMS:
        return np.array([0, 0, 1.0])      # the spread membrane faces forward
    d = TAIL[b] - HEAD[b]
    return np.array([0, 0, 1.0]) if abs(d[2]) < 0.8 * np.linalg.norm(d) else np.array([0, 1.0, 0])


def pose(spec):
    """Spec -> GargoylePose (see the module docstring)."""
    spec = sym(spec)
    P = GargoylePose(root=spec.get("root", (0, 0, 0)))
    for k, v in spec.items():
        if k in ("aim", "root"):
            continue
        P.local[BONE_INDEX[k]] = euler(*v)
    for k, (d, n) in sorted(spec.get("aim", {}).items(), key=lambda kv: BONE_INDEX[kv[0]]):
        b = BONE_INDEX[k]
        bind = _frame(TAIL[b] - HEAD[b], _bind_normal(b))
        P.set_world(k, _frame(d, n) @ bind.T)
    return P


def yaw_only(R):
    f = R @ np.array([0, 0, 1.0])
    return euler(0, np.degrees(np.arctan2(f[0], f[2])), 0)


def leg_ik(P, side, target, hint, foot_world=None, toe_pitch=0.0):
    """Bend thigh and shin of leg `side` so the ankle reaches `target` (world), the knee in the plane of
    `hint` and bending the way it does in the bind pose; the foot turned to `foot_world` (default: the
    bind orientation turned by the root's yaw), the toes pitched by `toe_pitch`."""
    a, b, c, d = (BONE_INDEX[n] for n in LEGS[side])
    Rw, Pw = P.world()
    hip = Pw[a]
    l1 = np.linalg.norm(HEAD[b] - HEAD[a])
    l2 = np.linalg.norm(HEAD[c] - HEAD[b])
    v = np.asarray(target, float) - hip
    dist = np.clip(np.linalg.norm(v), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dn = v / np.linalg.norm(v)
    h = np.asarray(hint, float) - np.dot(hint, dn) * dn
    h /= max(np.linalg.norm(h), 1e-9)
    along = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    knee = hip + dn * along + h * np.sqrt(max(l1 * l1 - along * along, 0))
    ankle = hip + dn * dist
    bind1, bind2 = HEAD[b] - HEAD[a], HEAD[c] - HEAD[b]
    bn = np.cross(bind1, bind2)
    nn = np.cross(knee - hip, ankle - knee)
    if np.dot(bn, nn) < 0:
        nn = -nn
    R1 = _frame(knee - hip, np.cross(nn, knee - hip)) @ _frame(bind1, np.cross(bn, bind1)).T
    R2 = _frame(ankle - knee, np.cross(nn, ankle - knee)) @ _frame(bind2, np.cross(bn, bind2)).T
    P.local[a] = Rw[PARENT[a]].T @ R1
    P.local[b] = R1.T @ R2
    if foot_world is None:
        foot_world = yaw_only(Rw[BONE_INDEX["root"]])
    P.local[c] = R2.T @ foot_world
    P.local[d] = euler(toe_pitch, 0, 0)
    return P


def ankle_rest(side):
    return HEAD[BONE_INDEX[LEGS[side][2]]].copy()
