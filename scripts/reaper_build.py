"""The Horned Reaper (DK1 style) as his own DK2 creature: Soles' brief (11 Nov 2026).

DK2's Horny is the starting point, but not a reskin: "the head is great; give the body a leaner,
muscly build, bigger shoulder spikes, chain around his waist instead of leather", after the DK1
Horned Reaper (render and CREATURE.JTY sprites 752-772).

What changes from Horny (REA3_* / Rea3-poseframe, Creatures.kwd record 27):
  * the body is reshaped on the reference pose (Rea3-poseframe): a V-shaped chest and back, a
    narrower waist, thicker upper arms, forearms and thighs, broader shoulders and big DK1 hooves
    (`reshape`); the head, horns and face are left exactly as they are;
  * DK1 shoulder plates: a big rounded bronze-olive dome on each shoulder with a long horn spike
    pointing out and up (`pauldrons`); Horny's small spikes are taken out (their triangles dropped);
  * a chain belt sagging round the hips with a short loose end hanging in front (`chain_belt`), in
    place of the leather belt and flaps, which are painted out;
  * his skins (`reaper_skin`): bare red muscled legs (no iron greaves, as DK1) on big dark
    olive hooves, the shoulder plates bronze, the muscles cut deeper; head and face untouched;
  * the scythe is Horny's, unchanged.

Vertex identity is necro_build's: every file is tied to the reference pose through texture and UV
("places"). The rig (`build_rig`) has humanoid_retarget's bones; N is the scythe side (-x).
Posing is done in a Y-up frame: (X, Y, Z) = (x, -z, y) of the KMF.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import scripts.necro_build as nb
from scripts.dk2kmf import Kmf, Mesh, Sprite, Vertex, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import anim_invariants, encode_anim
from scripts.humanoid_retarget import HUMANOID_BONES, Rig
from scripts.molynx_assets import GAME_MAX_KEYS, KEY_TOLERANCES
from scripts.necro_anims import from_yup, to_yup
from scripts.necro_build import Accessory, Clip, Reference, kabsch
from scripts.retarget import smooth_weights

PREFIX = "HReaper-"
REFERENCE = "Rea3-poseframe"
SKINS = {"reaperlegs": "HReaper_Legs", "reapertorso": "HReaper_Torso", "reaperback": "HReaper_Back",
         "reaperschythe": "HReaper_Scythe", "reaperfacesmirk": "HReaper_Face"}
SOURCE_TEX = {"reaperlegs": "ReaperLegsMM0", "reapertorso": "ReaperTorsoMM0", "reaperback": "ReaperBackMM0",
              "reaperschythe": "ReaperSchytheMM0", "reaperfacesmirk": "ReaperFaceSmirkMM0"}
GEAR = "HReaper_Gear"
BODY_TEX = ("reaperlegs", "reaperback", "reapertorso", "reaperfacesmirk", "reaperschythe")

BONES = [b for b in HUMANOID_BONES if b != "cape"]
PARENTS = {"root": None, "torso": "root", "head": "torso", "armN": "torso", "armP": "torso", "legN": "root",
           "legP": "root", "foreN": "armN", "foreP": "armP", "shinN": "legN", "shinP": "legP"}

# Joints in the Y-up frame of Rea3-poseframe (read off its places: shoulders at the top of the arm
# strips, elbows where the arm strip turns from upper arm to forearm, knees at the front kink)
JOINTS_YUP = {"root": (0.005, 0.37, -0.03), "torso": (0.005, 0.43, -0.02), "head": (0.005, 0.555, 0.06),
              "armN": (-0.132, 0.505, 0.0), "armP": (0.140, 0.512, 0.0),
              "foreN": (-0.112, 0.405, 0.005), "foreP": (0.130, 0.430, -0.012),
              "legN": (-0.050, 0.345, -0.03), "legP": (0.062, 0.345, -0.03),
              "shinN": (-0.062, 0.200, 0.0), "shinP": (0.072, 0.200, 0.0)}
HANDS_YUP = {"N": (-0.125, 0.290, 0.075), "P": (0.118, 0.310, 0.035)}
SHOULDER_OUT = -0.008       # reshape: the arms move out this much (in, if < 0; and up 0.004); so do their joints


def joints(reshaped: bool = True) -> dict:
    """The rig joints (Y-up), moved with the arms when the body is reshaped."""
    out = {k: np.array(v, float) for k, v in JOINTS_YUP.items()}
    if reshaped:
        for k in ("armN", "foreN", "armP", "foreP"):
            out[k] += [(-1 if k.endswith("N") else 1) * SHOULDER_OUT, 0.004, 0]
    return out


def hands(reshaped: bool = True) -> dict:
    out = {k: np.array(v, float) for k, v in HANDS_YUP.items()}
    if reshaped:
        for k in out:
            out[k] += [(-1 if k == "N" else 1) * SHOULDER_OUT, 0.004, 0]
    return out
HOOF_TOP = 0.112            # places below this are the hooves


# ------------------------------------------------------------------------------ loading

def load_clip(meshes_dir: Path, stem: str) -> Clip:
    """A Horny file: every sprite is body (the scythe too: it is posed with the hand)."""
    path = next(p for p in Path(meshes_dir).iterdir() if p.name.lower() == f"{stem.lower()}.kmf")
    model = parse_kmf(path.read_bytes())
    if model.anim is not None:
        frames = [evaluate_anim_frame(model.anim, f) for f in range(model.anim.frame_count)]
    else:
        g = np.array(model.mesh.geometry, float)
        frames = [[g[[v.geom for v in s.vertices]] for s in model.mesh.sprites]]
    sprites = (model.anim or model.mesh).sprites
    body = [i for i, s in enumerate(sprites) if model.materials[s.material].textures[0].lower() in BODY_TEX]
    return Clip(stem, model, frames, body)


def reference(meshes_dir: Path) -> Reference:
    nb.CX = 0.0
    return Reference(load_clip(meshes_dir, REFERENCE))


# ------------------------------------------------------------------------------ body parts

def _uv_sets(ref):
    """place -> [(texture, u, v)] over every vertex at the place."""
    c = ref.clip
    out = [[] for _ in range(len(ref.places))]
    for s in c.body:
        uv = c.uv(s)
        for i, pid in enumerate(c.place[s]):
            out[pid].append((c.texture(s), uv[i, 0], uv[i, 1]))
    return out


def regions(ref) -> dict:
    """Boolean masks over the reference places."""
    P = to_yup(ref.places)
    uvs = _uv_sets(ref)
    n = len(P)
    m = {k: np.zeros(n, bool) for k in ("scythe", "face", "horn", "arm_upper", "arm_fore", "plate", "old_spike",
                                        "legs_tex", "belt", "flap")}
    for i, lst in enumerate(uvs):
        for t, u, v in lst:
            if t == "reaperschythe":
                m["scythe"][i] = True
            elif t == "reaperfacesmirk":
                m["face"][i] = True
            elif t == "reaperlegs":
                if 0.46 <= u <= 0.60 and v < 0.50 and P[i, 1] > 0.6:
                    m["horn"][i] = True
                else:
                    m["legs_tex"][i] = True
                if v < 0.12 and P[i, 1] > 0.39:
                    m["belt"][i] = True
            elif t in ("reapertorso", "reaperback"):
                if t == "reapertorso" and u > 0.78 and v < 0.30:
                    m["old_spike"][i] = True
                elif v < 0.27:
                    (m["arm_upper"] if u < 0.29 else m["arm_fore"])[i] = True
                elif 0.30 < v < 0.66 and (u < 0.30 or u > 0.70):
                    m["plate"][i] = True
    # the arm strip also wraps the back of the neck at its root: those stay with the body
    neck = (np.abs(P[:, 0]) < 0.075) & (P[:, 1] > 0.5)
    m["arm_upper"] &= ~neck
    m["arm_fore"] &= ~neck
    m["flap"] = m["legs_tex"] & (np.abs(P[:, 0]) < 0.02) & (P[:, 1] > 0.3) & (P[:, 1] < 0.4)
    return m


def labels(ref) -> np.ndarray:
    """Bone index (into BONES) per reference place."""
    P = to_yup(ref.places)
    x, y = P[:, 0], P[:, 1]
    R = regions(ref)
    b = {k: i for i, k in enumerate(BONES)}
    side = np.where(x < 0, "N", "P")
    lab = np.full(len(P), b["torso"])
    legs = R["legs_tex"] & (y < 0.37)
    for i in np.flatnonzero(legs):
        lab[i] = b[("shin" if y[i] < JOINTS_YUP["shin" + side[i]][1] + 0.012 else "leg") + side[i]]
    lab[R["legs_tex"] & (y >= 0.37) & (y < 0.44)] = b["root"]
    lab[R["flap"]] = b["root"]
    for i in np.flatnonzero(R["arm_upper"] | R["arm_fore"]):
        lab[i] = b[("fore" if R["arm_fore"][i] else "arm") + side[i]]
    lab[(R["arm_upper"]) & (np.abs(x) < 0.11) & (y > 0.53)] = b["torso"]     # the trapezius, not the arm
    lab[R["face"] | R["horn"] | (y > 0.60)] = b["head"]
    lab[R["scythe"]] = b["foreN"]
    return lab


def triangles(ref) -> np.ndarray:
    c = ref.clip
    return np.array([[c.place[s][v] for v in t] for s in c.body for t in c.sprites()[s].triangles[0]])


def build_rig(ref, rest_kmf: np.ndarray | None = None) -> Rig:
    """The rig on the (reshaped) reference places."""
    tris = triangles(ref)
    hard = labels(ref)
    w = smooth_weights(hard, tris, len(BONES), 1)
    rigid = np.isin(hard, [BONES.index(n) for n in ("foreN", "foreP", "head", "shinN", "shinP")])
    w[rigid] = 0.85 * np.eye(len(BONES))[hard[rigid]] + 0.15 * w[rigid]
    R = regions(ref)
    for side in "NP":                    # shoulder plates: mostly arm, part chest
        P0 = to_yup(ref.places)
        sel = R["plate"] & (np.sign(P0[:, 0]) == (-1 if side == "N" else 1))
        w[sel] = 0
        w[sel, BONES.index("arm" + side)] = 0.6
        w[sel, BONES.index("torso")] = 0.4
    w[R["scythe"]] = np.eye(len(BONES))[BONES.index("foreN")]
    w[R["face"] | R["horn"]] = np.eye(len(BONES))[BONES.index("head")]
    w /= w.sum(1, keepdims=True)
    P = to_yup(ref.places if rest_kmf is None else rest_kmf)
    return Rig(rest=P, indices=tris, joints=joints(rest_kmf is not None), weights=w,
               height=float(P[:, 1].max() - P[:, 1].min()), floor=float(P[:, 1].min()), bones=list(BONES),
               parents=dict(PARENTS))


# ------------------------------------------------------------------------------ the DK1 build

def _radial(p, a, b):
    """Component of p away from the segment a-b (and the foot of the perpendicular)."""
    d = b - a
    t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
    foot = a + t[:, None] * d
    return p - foot, foot


# bone -> (radial scale, axis end) : thicker limbs; arms run elbow -> hand, legs knee -> hoof
# a bit leaner than DK2's Horny (Soles, 12 Nov 2026, with the DK1 reference render): thinner
# arms and legs, arms in closer, a slimmer chest
LIMB_SCALE = {"armN": 0.90, "armP": 0.90, "foreN": 0.92, "foreP": 0.92, "legN": 0.95, "legP": 0.95,
              "shinN": 0.97, "shinP": 0.97}


def reshape(ref) -> np.ndarray:
    """The reference places reshaped to the DK1 build (KMF frame). Head, horns, face and scythe
    do not move."""
    P = to_yup(ref.places).copy()
    R = regions(ref)
    rig = build_rig(ref)
    w = rig.weights
    J = {k: np.array(v) for k, v in JOINTS_YUP.items()}
    end = {"armN": J["foreN"], "armP": J["foreP"], "foreN": np.array(HANDS_YUP["N"]), "foreP": np.array(HANDS_YUP["P"]),
           "legN": J["shinN"], "legP": J["shinP"], "shinN": J["shinN"] * [1, 0, 1] + [0, 0.06, 0],
           "shinP": J["shinP"] * [1, 0, 1] + [0, 0.06, 0]}
    fixed = R["face"] | R["horn"] | R["scythe"]
    d = np.zeros_like(P)
    for bone, k in LIMB_SCALE.items():
        bi = BONES.index(bone)
        r, _ = _radial(P, J[bone], end[bone])
        d += (w[:, [bi]] * (k - 1) * r) * ~R["scythe"][:, None]
    # the chest and back: a V, broad at the chest, narrow at the waist (torso and root weights)
    tw = (w[:, BONES.index("torso")] + w[:, BONES.index("root")]) * ~R["scythe"]
    y = P[:, 1]
    xs = np.interp(y, [0.33, 0.38, 0.42, 0.46, 0.50, 0.56, 0.60], [1.0, 0.88, 0.84, 0.90, 0.94, 0.94, 1.0])
    zs = np.interp(y, [0.33, 0.40, 0.46, 0.52, 0.60], [1.0, 0.88, 0.92, 0.98, 1.0])
    cz = np.interp(y, [0.3, 0.45, 0.6], [-0.035, -0.02, 0.0])
    d[:, 0] += tw * (xs - 1) * (P[:, 0] - 0.005)
    d[:, 2] += tw * (zs - 1) * (P[:, 2] - cz)
    # broader shoulders: the whole arm and plate moves out a little
    for side, sx in (("N", -1), ("P", 1)):
        aw = w[:, BONES.index("arm" + side)] + w[:, BONES.index("fore" + side)]
        d[:, 0] += aw * sx * SHOULDER_OUT
        d[:, 1] += aw * 0.004
    # big DK1 hooves: wider and longer at the ground
    hoof = (P[:, 1] < HOOF_TOP) & R["legs_tex"]
    for side, sx in (("N", -1), ("P", 1)):
        sel = hoof & (np.sign(P[:, 0]) == sx)
        c = P[sel].mean(0)
        k = np.interp(P[sel, 1], [0.0, HOOF_TOP], [1.32, 1.08])
        d[sel, 0] += (k - 1) * (P[sel, 0] - c[0])
        d[sel, 2] += (k - 1) * (P[sel, 2] - c[2]) + 0.006
    d[fixed & ~R["scythe"]] = 0           # the scythe only moves out with the arm holding it
    out = P + d
    out = _no_collar(out, R)
    return from_yup(out)


def _no_collar(out, R) -> np.ndarray:
    """Horny's collar armour (the plates from his neck out over his shoulders) taken down to a
    plain sloping shoulder line: he wears only the two shoulder pads (Soles, 11 Nov, after the
    DK1 render). The plate places are lowered under the line from the neck to the shoulder,
    drawn in at the outer end and flattened front to back."""
    out = out.copy()
    sel = R["plate"] | R["old_spike"]
    x = np.abs(out[sel, 0] - 0.005)
    # (lower by the neck than before: Soles, 12 Nov: "going less up the neck")
    top = np.where(x < 0.16, 0.560 - 0.35 * np.clip(x - 0.05, 0, None), 0.5215 - 0.8 * (x - 0.16))
    out[sel, 1] = np.minimum(out[sel, 1], top)
    far = x > 0.175
    sx = np.sign(out[sel, 0] - 0.005)
    newx = np.where(far, 0.175 + (x - 0.175) * 0.45, x)
    out[sel, 0] = 0.005 + sx * newx
    zc = 0.008
    out[sel, 2] = zc + np.clip(out[sel, 2] - zc, -0.042, 0.046)
    return out


# ------------------------------------------------------------------------------ accessories

# UV boxes in HReaper_Gear (u0, v0, u1, v1)
UV_PLATE = (0.02, 0.02, 0.48, 0.48)
UV_HORN = (0.54, 0.02, 0.98, 0.48)
UV_STEEL = (0.02, 0.54, 0.48, 0.98)
UV_MEDAL = (0.55, 0.55, 0.97, 0.97)


def _box_uv(box, u, v):
    u0, v0, u1, v1 = box
    return np.stack([u0 + (u1 - u0) * np.asarray(u), v0 + (v1 - v0) * np.asarray(v)], -1)


def _frame(up):
    up = up / np.linalg.norm(up)
    a = np.cross(up, [0.0, 0.0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(up, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(up, a), up


def dome(centre, up, radii, seg=10, rings=(0.0, 0.42, 0.78, 1.0), lip=0.18):
    """A low-poly shoulder dome (Y-up): rings of `seg` points from the rim (0) to the top (1), a
    rolled lip under the rim, and the top point. -> points, triangles, UVs (UV_PLATE)."""
    a, b, n = _frame(np.asarray(up, float))
    rx, ry, rz = radii                     # along a (out), n (up), b (front)
    pts, uv = [], []
    ring_ids = []
    for k, r in enumerate((-lip,) + tuple(rings[:-1])):
        el = np.radians(90 * max(r, 0))
        drop = -lip * ry * 0.6 if r < 0 else 0.0
        ids = []
        for j in range(seg):
            th = 2 * np.pi * j / seg
            rad = np.cos(el) * (0.92 if r < 0 else 1.0)
            p = centre + rx * rad * np.cos(th) * a + rz * rad * np.sin(th) * b + (ry * np.sin(el) + drop) * n
            ids.append(len(pts))
            pts.append(p)
            uv.append((0.5 + 0.48 * rad * np.cos(th), 0.5 + 0.48 * rad * np.sin(th)))
        ring_ids.append(ids)
    top = len(pts)
    pts.append(centre + ry * n)
    uv.append((0.5, 0.5))
    tris = []
    for r0, r1 in zip(ring_ids[:-1], ring_ids[1:]):
        for j in range(seg):
            j2 = (j + 1) % seg
            tris += [(r0[j], r1[j], r0[j2]), (r0[j2], r1[j], r1[j2])]
    last = ring_ids[-1]
    for j in range(seg):
        tris.append((last[j], top, last[(j + 1) % seg]))
    u, v = np.array(uv).T
    return np.array(pts), np.array(tris), _box_uv(UV_PLATE, u, v)


def facet_cap(centre, up, radii, seg=8, elev=(-8.0, 30.0, 62.0)):
    """A faceted shoulder cap (Y-up), like the DK1 reference render's polygonal pauldrons: rings of
    `seg` points at the given elevations (degrees; the first a little under the rim) and a top point.
    -> points, triangles, UVs (UV_PLATE, projected down its axis). Flat-shaded later (unshared)."""
    a, b, n = _frame(np.asarray(up, float))
    rx, ry, rz = radii
    pts, ring_ids = [], []
    for k, e in enumerate(elev):
        el = np.radians(e)
        ids = []
        for j in range(seg):
            th = 2 * np.pi * (j + 0.5 * (k % 2)) / seg
            p = centre + np.cos(el) * (rx * np.cos(th) * a + rz * np.sin(th) * b) + ry * np.sin(el) * n
            ids.append(len(pts))
            pts.append(p)
        ring_ids.append(ids)
    top = len(pts)
    pts.append(centre + ry * n)
    tris = []
    for r0, r1 in zip(ring_ids[:-1], ring_ids[1:]):
        for j in range(seg):
            j2 = (j + 1) % seg
            tris += [(r0[j], r1[j], r0[j2]), (r0[j2], r1[j], r1[j2])]
    for j in range(seg):
        tris.append((ring_ids[-1][j], top, ring_ids[-1][(j + 1) % seg]))
    pts = np.array(pts)
    rel = pts - centre
    u = 0.5 + 0.48 * (rel @ a) / rx
    v = 0.5 + 0.48 * (rel @ b) / rz
    return pts, np.array(tris), _box_uv(UV_PLATE, np.clip(u, 0, 1), np.clip(v, 0, 1))


def unshare(pts, tris, uvs):
    """Every triangle its own corners (flat shading: the facets show)."""
    idx = np.asarray(tris).reshape(-1)
    return pts[idx], np.arange(len(idx)).reshape(-1, 3), uvs[idx]


def horn(base, axis, radius, length, sides=5, bend=None):
    """A horn spike: two rings and a tip, curving a little towards `bend`. -> points, tris, UVs (UV_HORN)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a, b, _ = _frame(axis)
    bend = np.zeros(3) if bend is None else np.asarray(bend, float)
    pts, uv = [], []
    for k, (t, r) in enumerate(((0.0, 1.0), (0.45, 0.62))):
        c = base + axis * length * t + bend * length * t * t
        for j in range(sides):
            th = 2 * np.pi * j / sides
            pts.append(c + radius * r * (np.cos(th) * a + np.sin(th) * b))
            uv.append((j / sides, 1.0 - t))
    pts.append(base + axis * length + bend * length)
    uv.append((0.5, 0.0))
    tris = []
    for j in range(sides):
        j2 = (j + 1) % sides
        tris += [(j, sides + j, j2), (j2, sides + j, sides + j2), (sides + j, 2 * sides, sides + j2)]
    u, v = np.array(uv).T
    return np.array(pts), np.array(tris), _box_uv(UV_HORN, u, v)


def _cat(parts):
    pts, tris, uvs, n = [], [], [], 0
    for p, t, u in parts:
        pts.append(p)
        tris.append(t + n)
        uvs.append(u)
        n += len(p)
    return np.vstack(pts), np.vstack(tris), np.vstack(uvs)


def _tri_flip(t):
    """KMF triangles wind the other way round from the Y-up construction (the frame change keeps
    handedness, but the game's front face is clockwise)."""
    return t[:, ::-1]


PAD_CENTRE = (0.140, 0.498, 0.0)       # Y-up, the +x cap (mirrored for -x): on the shoulder, out from the neck


def pad_centre(sx: int) -> np.ndarray:
    return np.array([sx * PAD_CENTRE[0], PAD_CENTRE[1], PAD_CENTRE[2]])


def _wrap(pts, centre, body, clearance, lip_ids, default_r):
    """Shrink-wrap dome points onto the body: each point moves along its ray from the centre to
    `clearance` outside the farthest body point within 24 degrees of the ray (the rim to just
    touching it), so the plate sits on the shoulder with no gap and nothing pokes through."""
    out = pts.copy()
    rel = body - centre
    rn = np.linalg.norm(rel, axis=1)
    for i, p in enumerate(pts):
        d = p - centre
        r0 = np.linalg.norm(d)
        if r0 < 1e-9:
            continue
        d /= r0
        cosang = (rel @ d) / np.maximum(rn, 1e-9)
        near = rn[(cosang > np.cos(np.radians(24))) & (rn < 0.14)]
        if not len(near):
            continue
        if i in lip_ids:                       # the rim: down onto the body
            r = np.clip(near.max() + 0.003, 0.55 * r0, 1.35 * r0)
        else:                                  # the dome: keeps its round shape, only pushed out
            r = np.clip(max(near.max() + clearance, 0.92 * r0), 0.55 * r0, 1.35 * r0)
        out[i] = centre + d * r
    return out


SPIKE_R, SPIKE_LEN = 0.017, 0.100     # the pads' horn spikes (were 0.022 / 0.135: 'a bit smaller')


PAD_RADII = (0.062, 0.046, 0.064)       # low and round (Soles: "flatten it down", "circular")
PAD_SEG, PAD_ELEV = 12, (-8.0, 22.0, 46.0, 70.0)
PAD_UP = 0.55                           # the cap's axis leans out this much (x per y): it faces up and out


def _wrap_cap(pts, centre, body, rim_ids):
    """The cap onto the shoulder: its rim points drawn in to just over the body (so it sits on him,
    no gap); the rest keep the round shape, only pushed out evenly where the shoulder would show
    through (one scale for the whole dome, so its facets stay regular)."""
    out = pts.copy()
    rel = body - centre
    rn = np.linalg.norm(rel, axis=1)
    grow = 1.0
    for i, p in enumerate(pts):
        d = p - centre
        r0 = np.linalg.norm(d)
        if r0 < 1e-9:
            continue
        d /= r0
        near = rn[((rel @ d) / np.maximum(rn, 1e-9) > np.cos(np.radians(20))) & (rn < 0.15)]
        if not len(near):
            continue
        if i in rim_ids:
            out[i] = centre + d * np.clip(near.max() + 0.002, 0.6 * r0, 1.2 * r0)
        else:
            grow = max(grow, min((near.max() + 0.006) / r0, 1.2))
    dome = [i for i in range(len(pts)) if i not in rim_ids]
    out[dome] = centre + (pts[dome] - centre) * grow
    return out


LIGHT = np.array([0.35, 0.80, 0.50]) / np.linalg.norm([0.35, 0.80, 0.50])


def _facet_shade(pts, tris, uvs):
    """Polished faceted silver: each facet one flat tone from how it faces the light (a spot on the
    gear texture's silver ramp), so the polygons read like the DK1 render's pauldrons."""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    cen = pts.mean(0)
    n *= np.sign(((a + b + c) / 3 - cen) @ n.T if False else np.sum(((a + b + c) / 3 - cen) * n, axis=1))[:, None]
    lit = np.clip(0.5 + 0.5 * (n @ LIGHT), 0, 1)
    uv = uvs.copy()
    for k, t in enumerate(tris):
        uv[t] = _box_uv(UV_PLATE, 0.05 + 0.90 * lit[k], 0.5)
    return pts, tris, uv


def _closest_on_tris(p, A, B, C):
    """Closest point to p on each triangle (A, B, C: (T, 3)) -> points (T, 3), barycentrics (T, 3)."""
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp, cp = p - B, p - C
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-18, 1e-18, va + vb + vc)
    v, w = vb / den, vc / den
    bary = np.stack([1 - v - w, v, w], 1)
    # outside the face: the nearest point on the nearest edge
    out = (bary < 0).any(1)
    if out.any():
        for e0, e1, i0, i1 in ((A, B, 0, 1), (B, C, 1, 2), (C, A, 2, 0)):
            pass
        best = np.full(len(A), np.inf)
        res = np.zeros((len(A), 3))
        for (P0, P1, i0, i1) in ((A, B, 0, 1), (B, C, 1, 2), (C, A, 2, 0)):
            d = P1 - P0
            t = np.clip(((p - P0) * d).sum(1) / np.maximum((d * d).sum(1), 1e-18), 0, 1)
            q = P0 + t[:, None] * d
            dist = np.linalg.norm(p - q, axis=1)
            better = out & (dist < best)
            best = np.where(better, dist, best)
            b = np.zeros((len(A), 3))
            b[:, i0], b[:, i1] = 1 - t, t
            res[better] = b[better]
        bary = np.where(out[:, None], res, bary)
    pts = bary[:, :1] * A + bary[:, 1:2] * B + bary[:, 2:] * C
    return pts, bary


SKIN_GAP = 0.0025        # how far a cap's rim stands off the skin it is bound to


def bind_rim(points, rim, rest, tris, near_centre, radius=0.13):
    """Bind the cap's rim points (KMF frame) onto the body's skin: each moves onto the nearest point
    of the body's surface (SKIN_GAP out along the surface's normal) and remembers that triangle and
    place, so in every frame it rides the skin there (the cap is part of him, with no gap under it).
    -> moved points, bindings [(point index, tri place ids, barycentrics, normal sign)]."""
    out = points.copy()
    A, B, C = rest[tris[:, 0]], rest[tris[:, 1]], rest[tris[:, 2]]
    cen = (A + B + C) / 3
    sel = np.flatnonzero(np.linalg.norm(cen - near_centre, axis=1) < radius)
    binds = []
    for i in rim:
        q, bary = _closest_on_tris(points[i], A[sel], B[sel], C[sel])
        k = int(np.argmin(np.linalg.norm(q - points[i], axis=1)))
        t = sel[k]
        n = np.cross(B[t] - A[t], C[t] - A[t])
        n /= max(np.linalg.norm(n), 1e-12)
        sign = 1.0 if n @ (points[i] - q[k]) >= 0 else -1.0
        out[i] = q[k] + sign * n * SKIN_GAP
        binds.append((i, tris[t].copy(), bary[k].copy(), sign))
    return out, binds


def acc_points(acc, rest_places, posed_places):
    """An accessory's points in one frame: carried rigidly by its places, its bound points (a cap's
    rim) riding the skin."""
    r, t = kabsch(rest_places[acc.places], posed_places[acc.places])
    pts = acc.points @ r.T + t
    for i, tp, bary, sign in getattr(acc, "binds", ()):
        A, B, C = posed_places[tp]
        n = np.cross(B - A, C - A)
        n /= max(np.linalg.norm(n), 1e-12)
        pts[i] = bary @ np.stack([A, B, C]) + sign * n * SKIN_GAP
    return pts


def shoulder_pads(ref, rest_kmf) -> list[Accessory]:
    """The DK1 reference render's shoulder caps (Soles, 12 Nov 2026: "a full polygonal circle over the
    shoulders with no more touching the neck ... part of the actual model ... a lot more silver"): a
    faceted silver cap over each shoulder, leaning out, its rim pulled down onto the shoulder so it
    sits on him with no gap, carried by the shoulder and upper arm. A horn spike out of each."""
    R = regions(ref)
    lab = labels(ref)
    P = to_yup(rest_kmf)
    keep = ~(R["face"] | R["horn"] | R["scythe"]) & ~np.isin(lab, [BONES.index(b) for b in
                                                                    ("head", "foreN", "foreP", "legN", "legP", "shinN", "shinP", "root")])
    out = []
    for side, sx in (("N", -1), ("P", 1)):
        arm = np.flatnonzero(lab == BONES.index("arm" + side))
        carriers = np.union1d(arm, np.flatnonzero(R["plate"] & (np.sign(to_yup(ref.places)[:, 0]) == sx)))
        centre = pad_centre(sx)
        up = np.array([sx * PAD_UP, 1.0, 0.0])
        dp, dt, du = facet_cap(centre, up, PAD_RADII, seg=PAD_SEG, elev=PAD_ELEV)
        body = P[keep & (np.sign(P[:, 0]) == sx) & (P[:, 1] > 0.40)]
        dp = _wrap_cap(dp, centre, body, set(range(PAD_SEG)))
        aim = np.array([sx * 1.0, 0.45, -0.10])
        aim /= np.linalg.norm(aim)
        rel = dp - centre
        k = int(np.argmax((rel / np.linalg.norm(rel, axis=1, keepdims=True)) @ aim))
        base = centre + (dp[k] - centre) * 0.85
        h = horn(base, aim, SPIKE_R, SPIKE_LEN + np.linalg.norm(dp[k] - base), bend=np.array([0, 0.30, -0.05]))
        # the rim onto his skin (KMF frame), bound there: the cap is part of him, no gap under it
        body_tris = triangles(ref)
        skin = ~(R["face"] | R["horn"] | R["scythe"])
        body_tris = body_tris[skin[body_tris].all(1)]
        rim = list(range(PAD_SEG))
        kp, binds = bind_rim(from_yup(dp), rim, rest_kmf, body_tris, from_yup(centre[None])[0])
        dp = to_yup(kp)
        idx = np.asarray(dt).reshape(-1)               # unshared: each rim point is several corners
        pts, tris, uvs = _facet_shade(*unshare(dp, dt, du))
        acc = Accessory("pad" + side, from_yup(pts), _tri_flip(tris), uvs, GEAR, carriers)
        acc.binds = [(int(j), tp, bary, sign) for (i, tp, bary, sign) in binds for j in np.flatnonzero(idx == i)]
        out.append(acc)
        # its spike: a piece of its own (a sprite holds at most 255 corners), carried the same way
        out.append(Accessory("spike" + side, from_yup(h[0]), _tri_flip(h[1]), h[2], GEAR, carriers))
    return out


def shoulder_spikes(ref, rest_kmf) -> list[Accessory]:
    """A horn spike out of each shoulder (the DK1 sprites' pad spikes). The pads themselves are
    painted on the body (Quuz's plates in the torso / back textures, in shiny silver; Soles 12 Nov
    2026: "scratch this whole 3d shoulder pad idea ... recolour the shoulder pads more shiny silver");
    the spike's root is the shoulder's surface along it."""
    R = regions(ref)
    lab = labels(ref)
    P = to_yup(rest_kmf)
    keep = ~(R["face"] | R["horn"] | R["scythe"]) & ~np.isin(lab, [BONES.index(b) for b in
                                                                    ("head", "foreN", "foreP", "legN", "legP", "shinN", "shinP", "root")])
    out = []
    for side, sx in (("N", -1), ("P", 1)):
        # Soles, 13 Nov 2026: the spike "floats a bit and should always be symmetrical with the other
        # side's one": rooted on (and carried only by) the torso skin at the shoulder, so the arms'
        # strumming and fretting never tilt it, and sunk deep enough that no gap opens
        torso = keep & (lab == BONES.index("torso")) & (np.sign(P[:, 0]) == sx)
        centre = np.array([sx * SPIKE_CENTRE[0], SPIKE_CENTRE[1], SPIKE_CENTRE[2]])
        body = P[torso & (P[:, 1] > 0.40)]
        aim = np.array([sx * 1.0, 0.40, -0.10])
        aim /= np.linalg.norm(aim)
        rel = body - centre
        along = rel @ aim
        off = np.linalg.norm(rel - along[:, None] * aim, axis=1)
        near = (off < 0.035) & (along > 0)
        reach = float(along[near].max()) if near.any() else 0.0
        base = centre + aim * (reach - SPIKE_SINK)
        pts, tris, uvs = horn(base, aim, SPIKE_R, SPIKE_LEN + SPIKE_SINK, bend=np.array([0, 0.30, -0.05]))
        d = np.linalg.norm(P - base, axis=1)
        carriers = np.flatnonzero(torso & (d < 0.09))
        out.append(Accessory("spike" + side, from_yup(pts), _tri_flip(tris), uvs, GEAR, carriers))
    return out


SPIKE_CENTRE = (0.118, 0.498, 0.002)     # Y-up, over the (leaner) shoulder joint
SPIKE_SINK = 0.022                        # how deep the spike's root sits in the shoulder


def _outward(pts, tris, centres):
    """Order each triangle so cross(b - a, c - a) points away from its centre point (the finished
    winding of the dome, which renders right)."""
    out = []
    for t, c in zip(tris, centres):
        a_, b_, c_ = pts[t[0]], pts[t[1]], pts[t[2]]
        n = np.cross(b_ - a_, c_ - a_)
        out.append(t if n @ ((a_ + b_ + c_) / 3 - c) >= 0 else (t[0], t[2], t[1]))
    return np.array(out)


def chain_link(centre, along, normal, length, width, wire, seg=6, sides=3):
    """One chunky chain link: a stadium-shaped ring of `seg` points (long axis `along`, the ring in
    the plane of `along` and `normal`), swept by a `sides`-sided wire. -> points, tris, UVs (UV_STEEL)."""
    along = along / np.linalg.norm(along)
    normal = normal - along * (normal @ along)
    normal /= np.linalg.norm(normal)
    third = np.cross(along, normal)
    ring = []
    for j in range(seg):
        th = 2 * np.pi * (j + 0.5) / seg
        ring.append(centre + along * (length / 2 - wire) * np.cos(th) + normal * (width / 2 - wire) * np.sin(th))
    pts, uv, tris, cents = [], [], [], []
    for j in range(seg):
        p = ring[j]
        tang = ring[(j + 1) % seg] - ring[j - 1]
        tang /= np.linalg.norm(tang)
        radial = p - centre
        radial -= tang * (radial @ tang)
        radial /= np.linalg.norm(radial)
        for k in range(sides):
            ph = 2 * np.pi * k / sides + np.pi / 2
            d = radial * np.cos(ph) + third * np.sin(ph)
            pts.append(p + wire * d)
            uv.append((0.5 + 0.45 * np.cos(ph), 0.5 + 0.45 * np.sin(ph)))     # bright on top, dark under
    for j in range(seg):
        j2 = (j + 1) % seg
        for k in range(sides):
            k2 = (k + 1) % sides
            a_, b_, c_, d_ = j * sides + k, j * sides + k2, j2 * sides + k, j2 * sides + k2
            tris += [(a_, b_, c_), (b_, d_, c_)]
            mid = (ring[j] + ring[j2]) / 2
            cents += [mid, mid]
    pts = np.array(pts)
    u, v = np.array(uv).T
    return pts, _outward(pts, tris, cents), _box_uv(UV_STEEL, u, v)


def chain_along(path, normal_fn, length, width, wire, pitch):
    """Links along a polyline `path` (Y-up), one every `pitch`, alternately turned a quarter turn
    about the chain (as real links hang). normal_fn(point) gives the out-of-body direction."""
    path = np.asarray(path, float)
    seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    n = max(2, int(cum[-1] / pitch))
    parts = []
    for i in range(n):
        s_ = (i + 0.5) * cum[-1] / n
        k = min(int(np.searchsorted(cum, s_) - 1), len(path) - 2)
        t = (s_ - cum[k]) / max(seglen[k], 1e-9)
        c = path[k] + (path[k + 1] - path[k]) * t
        along = path[k + 1] - path[k]
        out = normal_fn(c)
        side = np.cross(along, out)
        parts.append(chain_link(c, along, out if i % 2 == 0 else side, length, width, wire))
    return _cat(parts)


def section(ref, P, heights, keep):
    """Points where the body surface (triangles whose corners are all `keep` places) crosses the
    horizontal planes at `heights` (Y-up)."""
    tris = triangles(ref)
    tris = tris[keep[tris].all(1)]
    out = []
    for h in heights:
        for t in tris:
            q = P[t]
            for i in range(3):
                a_, b_ = q[i], q[(i + 1) % 3]
                if (a_[1] - h) * (b_[1] - h) < 0:
                    out.append(a_ + (b_ - a_) * (h - a_[1]) / (b_[1] - a_[1]))
    return np.array(out)


def chain_belt(ref, rest_kmf) -> list[Accessory]:
    """A heavy silver chain of big links round the hips, sagging a little at the front."""
    lab = labels(ref)
    P = to_yup(rest_kmf)
    H = section(ref, P, (0.385, 0.395, 0.405, 0.415),
                np.isin(lab, [BONES.index(n) for n in ("root", "torso", "legN", "legP")]))
    cx, cz = (H[:, 0].max() + H[:, 0].min()) / 2, (H[:, 2].max() + H[:, 2].min()) / 2
    ang = np.arctan2(H[:, 2] - cz, H[:, 0] - cx)
    rad = np.hypot(H[:, 2] - cz, H[:, 0] - cx)
    path = []
    for j in range(41):
        th = 2 * np.pi * j / 40
        dang = np.abs((ang - th + np.pi) % (2 * np.pi) - np.pi)
        sel = dang < np.radians(40)
        r = (rad[sel].max() if sel.any() else rad.mean()) + 0.014    # round the hips, just clear of them
        front = max(0.0, np.sin(th))                      # +z is the front
        path.append((cx + r * np.cos(th), 0.402 - 0.020 * front ** 2, cz + r * np.sin(th)))
    path = np.array(path)
    path[:-1, [0, 2]] = (np.roll(path[:-1, [0, 2]], 1, 0) + 2 * path[:-1, [0, 2]] + np.roll(path[:-1, [0, 2]], -1, 0)) / 4
    path[-1] = path[0]

    def out(c):
        v = np.array([c[0] - cx, 0.0, c[2] - cz])
        return v / np.linalg.norm(v)
    pts, tris, uvs = chain_along(path, out, 0.038, 0.029, 0.0068, 0.027)
    carriers = np.flatnonzero(lab == BONES.index("root"))
    return [Accessory("chain", from_yup(pts), tris, uvs, GEAR, carriers)]


def chest_chain(ref, rest_kmf) -> list[Accessory]:
    """DK1's heavy chain straight across the upper chest from one shoulder plate to the other,
    hanging a little in the middle, lying on the chest."""
    lab = labels(ref)
    P = to_yup(rest_kmf)
    chest = P[(lab == BONES.index("torso")) & (P[:, 1] > 0.44) & (P[:, 1] < 0.60) & (P[:, 2] > 0.0)]
    a = pad_centre(-1) + np.array([0.034, 0.004, 0.050])
    b = pad_centre(+1) + np.array([-0.034, 0.004, 0.050])
    path = []
    for t in np.linspace(0, 1, 13):
        p = a + (b - a) * t
        p[1] -= 0.022 * np.sin(np.pi * t)
        near = chest[np.argsort(np.hypot(chest[:, 0] - p[0], chest[:, 1] - p[1]))[:3]]
        surf = near[:, 2].mean() + 0.016
        p[2] = max(surf, p[2] - 0.02 * np.sin(np.pi * t))
        path.append(p)
    path = np.array(path)
    links = chain_along(path, lambda c: np.array([0.0, 0.15, 1.0]) / np.hypot(0.15, 1.0),
                        0.036, 0.027, 0.0062, 0.025)
    mid = path[len(path) // 2]
    pts, tris, uvs = _cat([links, medallion(mid + np.array([0.0, -0.004, 0.006]), np.array([0.0, 0.15, 1.0]), 0.024, 0.007)])
    carriers = np.flatnonzero(lab == BONES.index("torso"))
    return [Accessory("chestchain", from_yup(pts), tris, uvs, GEAR, carriers)]


def medallion(centre, facing, radius, thick, sides=12):
    """The round medallion in the middle of the chest chain (as the DK1 sprites): a thick disc, its
    face showing UV_MEDAL, its edge band silver. -> points, tris, UVs."""
    facing = facing / np.linalg.norm(facing)
    a_, b_, _ = _frame(facing)
    front, back = centre + facing * thick / 2, centre - facing * thick / 2
    pts, uv, tris, cents = [front], [(0.5, 0.5)], [], []
    for k in range(sides):
        th = 2 * np.pi * k / sides
        d = np.cos(th) * a_ + np.sin(th) * b_
        pts.append(front + radius * d)
        uv.append((0.5 + 0.5 * np.cos(th), 0.5 - 0.5 * np.sin(th)))
    for k in range(sides):
        tris.append((0, 1 + k, 1 + (k + 1) % sides))
        cents.append(centre)
    face = (np.array(pts), _outward(np.array(pts), tris, cents), _box_uv(UV_MEDAL, *np.array(uv).T))
    ep, eu, et, ec = [], [], [], []
    for k in range(sides):
        th = 2 * np.pi * k / sides
        d = np.cos(th) * a_ + np.sin(th) * b_
        ep += [front + radius * d, back + radius * d * 0.96]
        eu += [(0.5, 0.95), (0.5, 0.55)]
    for k in range(sides):
        k2 = (k + 1) % sides
        et += [(2 * k, 2 * k + 1, 2 * k2), (2 * k + 1, 2 * k2 + 1, 2 * k2)]
        ec += [centre, centre]
    ep = np.array(ep)
    edge = (ep, _outward(ep, et, ec), _box_uv(UV_STEEL, *np.array(eu).T))
    return _cat([face, edge])


def accessories(ref, rest_kmf) -> list[Accessory]:
    """Only the shoulder spikes: the pads are Quuz's painted plates (in shiny silver) and the chains
    his painted ones (reaper_skin). The 3D caps (shoulder_pads) were tried and dropped (Soles, 12 Nov
    2026: "scratch this whole 3d shoulder pad idea")."""
    return shoulder_spikes(ref, rest_kmf)


# ------------------------------------------------------------------------------ writing

def _normals(p, tris):
    return nb._normals(p, tris)


def reaper_kmf(ref: Reference, posed: np.ndarray, accs: list[Accessory], name: str, wrap: bool | None,
               lods: int = 16, extras: list | None = None, controls: bytes = b"") -> Kmf:
    """A Reaper file from per-frame place positions (KMF frame): Horny's sprites with his skins,
    the accessories carried rigidly by their places. One frame and wrap None: a mesh file.
    `extras` (hreaper_files): props of a game file that are not part of Horny (the gem he picks up),
    each (material, sprite, per-frame points), kept as they are; `controls` the source file's
    animation controls (kept, as necro_build keeps them)."""
    c = ref.clip
    model = c.model
    mats = []
    for m in model.materials:
        t = m.textures[0].lower()
        if t in SKINS:
            label = m.name[: len(m.name) - len(m.textures[0])] if m.name.endswith(m.textures[0]) else m.name
            mats.append(nb._material(m, label + SKINS[t], SKINS[t]))
        else:
            mats.append(m)
    base = model.materials[c.sprites()[0].material]
    gear_index = len(mats)
    mats.append(nb._material(base, "HReaper" + GEAR, GEAR))
    rows = sorted({c.row(s, i) for s in range(len(c.sprites())) for i in range(len(c.sprites()[s].vertices))})
    dense = {r: k for k, r in enumerate(rows)}
    anim = wrap is not None
    sprites = []
    for s in c.sprites():
        vs = []
        for v in s.vertices:
            row = dense[v.geom]                    # the reference is a mesh: its rows are geom indices
            vs.append(Vertex(v.uv, v.normal, row if not anim else 0, row if anim else 0))
        tris = s.triangles
        if model.materials[s.material].textures[0].lower() == "reapertorso":
            uv = np.array([v.uv for v in s.vertices], float) / 32768.0
            spike = (uv[:, 0] > 0.78) & (uv[:, 1] < 0.30)            # Horny's small shoulder spikes:
            tris = [[t for t in lod if not spike[list(t)].all()] for lod in tris]   # gone, under the new plates
        sprites.append(Sprite(s.material, tris, vs, s.mm_factor))
    frames = [[p[c.place[s]] for s in range(len(c.sprites()))] for p in posed]
    used = len(rows)
    for acc in accs:
        nrm = _normals(acc.points, acc.tris)
        verts = []
        for i in range(len(acc.points)):
            uv = tuple(int(round(np.clip(q, 0, 1) * 32767)) for q in acc.uv[i])
            verts.append(Vertex(uv, tuple(float(q) for q in nrm[i]), 0 if anim else used + i, used + i if anim else 0))
        used += len(acc.points)
        tris = [tuple(int(q) for q in t) for t in acc.tris]
        sprites.append(Sprite(gear_index, [list(tris) for _ in range(len(sprites[0].triangles))], verts, nb._mm(acc)))
        for f, p in enumerate(posed):
            frames[f].append(acc_points(acc, ref_rest_places, p))
    for mat, sp, pts in extras or []:
        mats.append(mat)
        verts = []
        for i, v in enumerate(sp.vertices):
            verts.append(Vertex(v.uv, v.normal, 0 if anim else used + i, used + i if anim else 0))
        used += len(sp.vertices)
        sprites.append(Sprite(len(mats) - 1, [list(t) for t in sp.triangles], verts, sp.mm_factor))
        for f in range(len(posed)):
            frames[f].append(np.asarray(pts[f], float))
    if anim:
        a = None
        for tol in KEY_TOLERANCES:
            a = encode_anim(name, sprites, frames, wrap, controls, tolerance=tol)
            if len(a.geom_xyz) - 1 <= GAME_MAX_KEYS:
                break
        bad = anim_invariants(a)
        if bad:
            raise ValueError(f"{name}: {bad}")
        return Kmf(model.version, "anim", mats, anim=a, groups=model.groups)
    g = np.zeros((used, 3))
    for s, p in zip(sprites, frames[0]):
        for v, q in zip(s.vertices, p):
            g[v.geom] = q
    centre = (g.min(0) + g.max(0)) / 2
    old = model.mesh
    mesh = Mesh(name, tuple(map(float, centre)), float(np.linalg.norm(g - centre, axis=1).max()), sprites,
                [tuple(map(float, q)) for q in g], old.controls)
    return Kmf(model.version, "mesh", mats, mesh=mesh, groups=model.groups)


ref_rest_places = None      # set by `prepare`: the reshaped rest places the accessories were built on


@dataclass
class Build:
    ref: Reference
    rest: np.ndarray            # reshaped places (KMF)
    rig: Rig
    accs: list


def prepare(meshes_dir: Path) -> Build:
    global ref_rest_places
    ref = reference(meshes_dir)
    rest = reshape(ref)
    ref_rest_places = rest
    rig = build_rig(ref, rest)
    return Build(ref, rest, rig, accessories(ref, rest))
