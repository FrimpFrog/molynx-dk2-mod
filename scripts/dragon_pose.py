"""Posing the Dragon: poses from compact specs, blends, two-bone leg IK, and clip sampling.

A pose spec is a dict:
  bone name -> (rx, ry, rz)            local rotation, degrees, applied x then y then z (world-aligned
                                        bind frames: x pitches, y yaws, z rolls)
  "aim": {bone: (direction, normal)}    the bone's world direction (bind head -> tail) and the normal of
                                        its plane (wings: the membrane's front face), resolved after its
                                        parents
  "root": (dx, dy, dz)                  root offset (model units)
Keys ending in L get mirrored to R by `sym` (x rotations kept, y and z negated; aims mirrored in x).

LICENCE: GPL-3.0.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from scripts.dragon_model import BONE_INDEX, BONE_NAMES, BONES, HEAD, PARENT, TAIL, forward

NB = len(BONES)
MIRROR = np.diag([-1.0, 1, 1])


def _mir(n):
    return n[:-1] + "R" if n.endswith("L") else n


def sym(spec):
    """Copy every ...L entry to ...R, mirrored (an explicit ...R entry wins)."""
    out = dict(spec)
    for k, v in spec.items():
        if k in ("aim", "root"):
            continue
        if k.endswith("L") and _mir(k) not in spec:
            rx, ry, rz = v
            out[_mir(k)] = (rx, -ry, -rz)
    if "aim" in spec:
        aim = dict(spec["aim"])
        for k, (d, n) in spec["aim"].items():
            if k.endswith("L") and _mir(k) not in aim:
                aim[_mir(k)] = (np.asarray(d, float) * [-1, 1, 1], np.asarray(n, float) * [-1, 1, 1])
        out["aim"] = aim
    return out


def _frame(d, n):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    n = np.asarray(n, float)
    n = n - np.dot(n, d) * d
    n = n / np.linalg.norm(n)
    return np.stack([d, n, np.cross(d, n)], 1)


_BIND_NORMAL = {}


def _bind_normal(b):
    name = BONE_NAMES[b]
    if name[:-1] in ("wing1", "wing2", "wing3", "f1a", "f1b", "f2a", "f2b", "f3a", "f3b"):
        return np.array([0, 0, 1.0])      # the spread membrane faces forward
    d = TAIL[b] - HEAD[b]
    ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.8 * np.linalg.norm(d) else np.array([0, 1.0, 0])
    return ref


def euler(rx, ry, rz):
    return Rot.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()


class Pose:
    def __init__(self, local=None, root=(0, 0, 0)):
        self.local = np.tile(np.eye(3), (NB, 1, 1)) if local is None else np.array(local)
        self.root = np.asarray(root, float)

    def copy(self):
        return type(self)(self.local.copy(), self.root.copy())

    def world(self):
        return forward(self.local, self.root)

    def rotate(self, bone, rx=0, ry=0, rz=0, pre=False):
        """Compose an extra local rotation (after the existing one, in the bone's bind axes)."""
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


def pose(spec):
    """Spec -> Pose (see the module docstring)."""
    spec = sym(spec)
    P = Pose(root=spec.get("root", (0, 0, 0)))
    for k, v in spec.items():
        if k in ("aim", "root"):
            continue
        P.local[BONE_INDEX[k]] = euler(*v)
    for k, (d, n) in sorted(spec.get("aim", {}).items(), key=lambda kv: BONE_INDEX[kv[0]]):
        b = BONE_INDEX[k]
        bind = _frame(TAIL[b] - HEAD[b], _bind_normal(b))
        want = _frame(d, n)
        P.set_world(k, want @ bind.T)
    return P


def blend(a, b, t):
    """Slerp each bone's local rotation, lerp the root."""
    t = float(np.clip(t, 0, 1))
    if t <= 0:
        return a.copy()
    if t >= 1:
        return b.copy()
    ra, rb = Rot.from_matrix(a.local), Rot.from_matrix(b.local)
    qa, qb = ra.as_quat(), rb.as_quat()
    dot = np.sum(qa * qb, axis=1)
    qb = np.where(dot[:, None] < 0, -qb, qb)
    dot = np.abs(dot)
    out = np.zeros_like(qa)
    for i in range(len(qa)):
        if dot[i] > 0.9995:
            q = qa[i] * (1 - t) + qb[i] * t
        else:
            th = np.arccos(np.clip(dot[i], -1, 1))
            q = (np.sin((1 - t) * th) * qa[i] + np.sin(t * th) * qb[i]) / np.sin(th)
        out[i] = q / np.linalg.norm(q)
    return type(a)(Rot.from_quat(out).as_matrix(), a.root * (1 - t) + b.root * t)


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def keyed(keys, t, wrap=False):
    """keys: [(time, Pose)] sorted; smooth (eased) blend between neighbours at time t."""
    times = [k[0] for k in keys]
    if wrap:
        period = times[-1]
        t = t % period
    if t <= times[0]:
        return keys[0][1].copy()
    if t >= times[-1]:
        return keys[-1][1].copy()
    i = int(np.searchsorted(times, t, side="right")) - 1
    t0, t1 = times[i], times[i + 1]
    return blend(keys[i][1], keys[i + 1][1], ease((t - t0) / (t1 - t0)))


# ------------------------------------------------------------------------------------------- leg IK

def _min_rot(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return Rot.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rot.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def leg_ik(P, side, ankle_target, knee_hint=(0, 0, 1), foot_world=None, toe_pitch=0.0):
    """Bend the thigh and shin of leg `side` ("L"/"R") so the ankle reaches `ankle_target` (world),
    the knee pointing toward `knee_hint`; then turn the foot to `foot_world` (a world rotation, default
    level: the bind orientation turned by the root's yaw only) and pitch the toes by `toe_pitch`."""
    th, sh, ft, to = (BONE_INDEX[n + side] for n in ("thigh", "shin", "foot", "toes"))
    Rw, Pw = P.world()
    hip = Pw[th]
    l1 = np.linalg.norm(HEAD[sh] - HEAD[th])
    l2 = np.linalg.norm(HEAD[ft] - HEAD[sh])
    target = np.asarray(ankle_target, float)
    d = target - hip
    dist = np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dn = d / np.linalg.norm(d)
    # knee position: in the plane of d and the hint
    h = np.asarray(knee_hint, float)
    h = h - np.dot(h, dn) * dn
    h = h / max(np.linalg.norm(h), 1e-9)
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    r = np.sqrt(max(l1 * l1 - a * a, 0))
    knee = hip + dn * a + h * r
    ankle = hip + dn * dist
    # thigh: bind segment hip->knee to the new one, keeping the twist by aligning the bend plane
    bind_thigh = HEAD[sh] - HEAD[th]
    bind_shin = HEAD[ft] - HEAD[sh]
    bind_n = np.cross(bind_thigh, bind_shin)
    new_n = np.cross(knee - hip, ankle - knee)
    Fb = _frame(bind_thigh, np.cross(bind_n, bind_thigh))
    Fn = _frame(knee - hip, np.cross(new_n, knee - hip))
    Rth = Fn @ Fb.T
    Fb2 = _frame(bind_shin, np.cross(bind_n, bind_shin))
    Fn2 = _frame(ankle - knee, np.cross(new_n, ankle - knee))
    Rsh = Fn2 @ Fb2.T
    parent = Rw[PARENT[th]]
    P.local[th] = parent.T @ Rth
    P.local[sh] = Rth.T @ Rsh
    if foot_world is None:
        foot_world = _yaw_only(Rw[BONE_INDEX["root"]])
    P.local[ft] = Rsh.T @ foot_world
    P.local[to] = euler(toe_pitch, 0, 0)
    return P


def _yaw_only(R):
    f = R @ np.array([0, 0, 1.0])
    yaw = np.arctan2(f[0], f[2])
    return Rot.from_euler("y", yaw).as_matrix()


def ankle_rest(side):
    return HEAD[BONE_INDEX["foot" + side]].copy()
