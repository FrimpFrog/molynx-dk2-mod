"""Krefft B's rig for his clips: channels -> bone world transforms -> skinned points (scripts.krefft_standing's
v5 build, its own Skeleton), with the Bile Demon's body motion as a donor layer.

Model space: Y up, Z forward, X his LEFT (as krefft_model). Every bone's bind frame is world-aligned, so a
bone's local rotation is a rotation in its parent's (world-aligned) frame.

Channels (degrees for rotations, model units for offsets; unstated = 0):
  body.x/y/z            the shell's rotation about its centre (x + tips him forward, y + turns to his left,
                        z + rolls his left side up)
  root.x/y/z            the shell's offset
  neck.x/y/z            spread half and half over neck1 / neck2;  head.x/y/z the head;  jaw the jaw's opening
  ext                   the neck sliding along its own axis: + darts the head out of the shell (a turtle's
                        strike), - pulls it back into the shell (EXT_REACH model units at 1)
  puff                  cheeks / throat swelling (0..1): the jaw's skin widened and deepened
  tail.x/y              spread over tail1 / tail2
  <arm bone>.x/y/z      upperFL, foreFL, handFL ... (FK, local);  arms.* / fores.* / hands.* both sides at
                        once (y and z mirrored on the right: z + lifts both arms out sideways)
  aik                   0 (default): the arms are FK; 1: each wrist goes to its target (IK, the elbow keeping
                        its FK side), blended in between;  aL.x/y/z, aR.x/y/z the wrist's offset from its
                        place in the bind, in the shell's frame (x + outward); the hand's own channels still
                        turn the paddle on the forearm
  L.x/y/z, R.x/y/z      each foot's ankle offset from where it stands (x + outward); L.p the foot's pitch
                        (+ toes up), L.yaw its turn (+ toes outward)
  ik                    1 (default): the feet go where L.* / R.* put them (IK, knees follow); 0: the legs
                        are FK from the hip by thighL.x/y/z, shinL.x, footL.x (blended in between)
  feet                  0..1: the feet's targets ride along with the donor body's offset (rising out of
                        the portal with the Bile Demon)
  ground                0..1: the whole body moved up / down so its lowest point is on the floor
  donor / donor_w       (set by a clip, not keyed) the Bile Demon's body motion added to the shell's

LICENCE: GPL-3.0.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripts.dragon_model import Skinner, _unit

EXT_REACH = .11
_STATE = {}


def model():
    """-> (parts, Skeleton) of B (v5), built once."""
    if "parts" not in _STATE:
        from scripts.krefft_standing import build_standing
        parts, skel = build_standing(v5=True)
        _STATE.update(parts=parts, skel=skel, skinner=Skinner(parts, skel))
    return _STATE["parts"], _STATE["skel"]


def skinner():
    model()
    return _STATE["skinner"]


def rx(d):
    a = np.radians(d)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(d):
    a = np.radians(d)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(d):
    a = np.radians(d)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler(x, y, z):
    """yaw, then pitch, then roll (degrees)."""
    return ry(y) @ rx(x) @ rz(z)


def _c(ch, name, ax):
    return ch.get(f"{name}.{ax}", 0.0)


ARM = ("upperF", "foreF", "handF")
LEG = ("thigh", "shin", "foot")
SIGN = {"L": 1.0, "R": -1.0}


def _arm_local(ch, bone, side):
    """An arm bone's local rotation: its own channels plus the both-sides macro (mirrored on the right)."""
    macro = {"upperF": "arms", "foreF": "fores", "handF": "hands"}[bone]
    s = SIGN[side]
    x = _c(ch, bone + side, "x") + _c(ch, macro, "x")
    y = _c(ch, bone + side, "y") + _c(ch, macro, "y") * s
    z = _c(ch, bone + side, "z") + _c(ch, macro, "z") * s
    return euler(x, y, z)


def _frame(u, n):
    u = _unit(u)
    n = _unit(n - u * (n @ u))
    return np.stack([u, n, np.cross(u, n)], 1)


def solve(ch, donor=None):
    """Channels -> (world rotations (B,3,3) (the jaw's may carry the puff's scale), world heads (B,3)).
    `donor`: (R (3,3), t (3,)) the donor body's rotation and offset (model space) added to the shell's."""
    parts, sk = model()
    H, par, idx = sk.head, sk.parent, sk.index
    B = len(sk.names)
    Rw = np.zeros((B, 3, 3))
    Pw = np.zeros((B, 3))
    Rb = euler(_c(ch, "body", "x"), _c(ch, "body", "y"), _c(ch, "body", "z"))
    off = np.array([_c(ch, "root", a) for a in "xyz"])
    dt = np.zeros(3)
    if donor is not None:
        Rb = donor[0] @ Rb
        dt = np.asarray(donor[1], float)
        off = off + dt
    r = idx["root"]
    Rw[r] = Rb
    Pw[r] = H[r] + off
    local = {}
    for name in ("neck1", "neck2"):
        local[name] = euler(.5 * _c(ch, "neck", "x"), .5 * _c(ch, "neck", "y"), .5 * _c(ch, "neck", "z"))
    local["head"] = euler(_c(ch, "head", "x"), _c(ch, "head", "y"), _c(ch, "head", "z"))
    local["jaw"] = rx(ch.get("jaw", 0.0))
    for name, w in (("tail1", .5), ("tail2", .5)):
        local[name] = euler(w * _c(ch, "tail", "x"), w * _c(ch, "tail", "y"), 0)
    for side in "LR":
        for b in ARM:
            local[b + side] = _arm_local(ch, b, side)
    ext_axis = _unit(H[idx["neck2"]] - H[idx["neck1"]])
    puff = ch.get("puff", 0.0)
    for i, name in enumerate(sk.names):
        p = par[i]
        if p is None or name[:-1] in LEG:
            continue
        Rw[i] = Rw[p] @ local[name]
        t = H[i] - H[p]
        if name == "neck1":
            t = t + ext_axis * ch.get("ext", 0.0) * EXT_REACH
        Pw[i] = Pw[p] + Rw[p] @ t
        if name == "jaw" and puff:
            Rw[i] = Rw[i] @ np.diag([1 + .8 * puff, 1 + .9 * puff, 1 + .1 * puff])
    # the arms: FK above; IK toward the wrist targets by `aik`
    aik = float(np.clip(ch.get("aik", 0.0), 0, 1))
    if aik > 0:
        for side in "LR":
            s = SIGN[side]
            up, fo, ha = (idx[b + side] for b in ARM)
            shp, elb_fk, wr_fk = Pw[up], Pw[fo], Pw[ha]
            l1 = np.linalg.norm(H[fo] - H[up])
            l2 = np.linalg.norm(H[ha] - H[fo])
            off = np.array([s * ch.get("a" + side + ".x", 0.0), ch.get("a" + side + ".y", 0.0), ch.get("a" + side + ".z", 0.0)])
            tgt_ik = Pw[r] + Rb @ (H[ha] - H[r] + off)
            tgt = wr_fk * (1 - aik) + tgt_ik * aik
            d = tgt - shp
            dist = float(np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-4, (l1 + l2) * .999))
            dh = _unit(d)
            pole = elb_fk - (shp + dh * ((elb_fk - shp) @ dh))
            if np.linalg.norm(pole) < 1e-6:
                pole = Rb @ np.array([s, -.5, 0])
            pole = _unit(pole - dh * (pole @ dh))
            a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
            h = np.sqrt(max(l1 * l1 - a * a, 0.0))
            elb = shp + dh * a + pole * h
            wr = shp + dh * dist
            n0 = np.cross(H[fo] - H[up], H[ha] - H[fo])
            n1 = np.cross(elb - shp, wr - elb)
            if np.linalg.norm(n1) < 1e-9:
                n1 = Rb @ n0
            Rw[up] = _frame(elb - shp, n1) @ _frame(H[fo] - H[up], n0).T
            Rw[fo] = _frame(wr - elb, n1) @ _frame(H[ha] - H[fo], n0).T
            Rw[ha] = Rw[fo] @ local["handF" + side]
            Pw[fo], Pw[ha] = elb, wr
    # the legs: IK to the feet's targets, blended with FK from the hip
    ik = float(np.clip(ch.get("ik", 1.0), 0, 1))
    follow = ch.get("feet", 0.0)
    for side in "LR":
        s = SIGN[side]
        th, sh, ft = (idx[b + side] for b in LEG)
        hip = Pw[r] + Rb @ (H[th] - H[r])
        k0, a0 = H[sh], H[ft]
        l1, l2 = np.linalg.norm(k0 - H[th]), np.linalg.norm(a0 - k0)
        # IK target and the foot's world turn
        tgt = a0 + np.array([s * ch.get(side + ".x", 0.0), ch.get(side + ".y", 0.0), ch.get(side + ".z", 0.0)])
        tgt = tgt + follow * dt                 # (the donor's rise / drift only)
        Rf_ik = ry(s * ch.get(side + ".yaw", 0.0) + ch.get("feet.yaw", 0.0)) @ rx(-ch.get(side + ".p", 0.0))
        if ik < 1:
            R1 = Rb @ euler(_c(ch, "thigh" + side, "x"), s * _c(ch, "thigh" + side, "y"), s * _c(ch, "thigh" + side, "z"))
            knee = hip + R1 @ (k0 - H[th])
            R2 = R1 @ rx(_c(ch, "shin" + side, "x"))
            ank = knee + R2 @ (a0 - k0)
            R3 = R2 @ rx(_c(ch, "foot" + side, "x"))
            tgt = tgt * ik + ank * (1 - ik)
            Rf = Rot.from_matrix(np.stack([Rf_ik, R3])).as_rotvec()
            Rf = Rot.from_rotvec(Rf[0] * ik + Rf[1] * (1 - ik)).as_matrix() if ik > 0 else R3
        else:
            Rf = Rf_ik
        d = tgt - hip
        dist = float(np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-4, (l1 + l2) * .999))
        dh = _unit(d)
        pole0 = k0 - (H[th] + _unit(a0 - H[th]) * ((k0 - H[th]) @ _unit(a0 - H[th])))
        pole = Rb @ pole0
        pole = _unit(pole - dh * (pole @ dh))
        a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
        h = np.sqrt(max(l1 * l1 - a * a, 0.0))
        knee = hip + dh * a + pole * h
        ank = hip + dh * dist
        n0 = np.cross(k0 - H[th], a0 - k0)
        n1 = np.cross(knee - hip, ank - knee)
        if np.linalg.norm(n1) < 1e-9:
            n1 = Rb @ n0
        Rw[th] = _frame(knee - hip, n1) @ _frame(k0 - H[th], n0).T
        Rw[sh] = _frame(ank - knee, n1) @ _frame(a0 - k0, n0).T
        Rw[ft] = Rf
        Pw[th], Pw[sh], Pw[ft] = hip, knee, ank
    return Rw, Pw


def skin(ch, donor=None):
    """Channels -> posed points per part (model space), the floor applied when `ground` asks for it."""
    Rw, Pw = solve(ch, donor)
    pts = skinner()(Rw, Pw)
    g = ch.get("ground", 0.0)
    if g:
        low = min(p[:, 1].min() for p in pts)
        pts = [p - np.array([0, g * low, 0]) for p in pts]
        Pw = Pw - np.array([0, g * low, 0])
    return pts, (Rw, Pw)


def bone_point(RwPw, name, point):
    """A bind-space point carried by bone `name` (world)."""
    Rw, Pw = RwPw
    sk = model()[1]
    i = sk.index[name]
    return Pw[i] + Rw[i] @ (np.asarray(point, float) - sk.head[i])


# ------------------------------------------------------------------------------------------- the donor

TRACKS = {}
DONOR_SCALE = 1.1          # Krefft's model height (0.635) over the Bile Demon's file height (0.536), per KMF unit
_F = np.diag([-1.0, 1, 1])  # the donor's Y-up KMF frame (x to his right) -> model space (x to his left)


def donor_body(clip, meshes_dir=None):
    """The Bile Demon's body motion in one of his clips: per frame (R, t) in model space relative to
    his idle1's first frame (Kabsch over the 47 vertices of his central body, through donor_tracks'
    common vertex numbering), t in Krefft's model units. -> (list of (R, t), wrap)."""
    if clip in TRACKS:
        return TRACKS[clip]
    from pathlib import Path
    from scripts.donor_tracks import donor_tracks
    from scripts.dk2texture import TextureCache
    from scripts.retarget import kabsch
    if "_all" not in TRACKS:
        from scripts.krefft_render import GAME_TEXTURES
        cache = TextureCache(Path(GAME_TEXTURES + ".dir"), Path(GAME_TEXTURES + ".dat"))
        md = Path(meshes_dir or "/home/claude/dk2data/Meshes")
        TRACKS["_all"] = donor_tracks(md, cache, "bile_demon")
    T = TRACKS["_all"]
    rest = T["idle1"][0][0]
    ok = ~np.isnan(rest[:, 0])
    c = np.median(rest[ok], 0)
    body = np.where(ok)[0][np.linalg.norm(rest[ok] - c, axis=1) < .26]
    a, wrap = T[clip]
    vs = [v for v in body if not np.isnan(a[:, v, 0]).any()]
    out = []
    for f in range(len(a)):
        R, _ = kabsch(rest[vs], a[f, vs])
        t = a[f, vs].mean(0) - rest[vs].mean(0)
        out.append((_F @ R @ _F, (_F @ t) * DONOR_SCALE / .925))
    TRACKS[clip] = (out, wrap)
    return TRACKS[clip]


def scaled(Rt, w, rot=True, trans=True, axes=(1, 1, 1)):
    """A donor frame (R, t) at weight w (the rotation's angle and the offset scaled; `axes` scales the
    offset per axis)."""
    R, t = Rt
    rv = Rot.from_matrix(R).as_rotvec() * (w if rot else 0.0)
    return Rot.from_rotvec(rv).as_matrix(), np.asarray(t) * (w if trans else 0.0) * np.asarray(axes, float)
