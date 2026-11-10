"""The Orc (from Dungeon Keeper 1) as a DK2 creature, made the way DK1 made him: the Troll, recoloured,
with hair added and his own weapons.

Every Orc file is the DK2 Troll's file of the same slot, re-encoded with
  * the Troll's two skins recoloured (`orc_texture`), shipped under new names so the Troll keeps his,
  * extra sprites carried rigidly with a body part: hair on the head, a buckler on the free forearm.

Body parts are found by texture position (UV), since the Troll's files each order their vertices
differently but all map the same two skins the same way:
  head      the face block of trollfrnt and the back-of-head block of trollback
  forearm   the hand end of the arm strips (both arms share these UVs; the free arm is the one on the
            other side of the body from the mallet in the file's first frame)
  mallet    the grey head and wooden handle blocks of trollfrnt
Each part's pose in a frame is the rigid fit (Kabsch) of its vertices to their place in the reference
frame (Troll-Idle1 frame 0), and an accessory follows that fit.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.dk2kmf import Kmf, Material, Mesh, Sprite, Vertex, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import encode_anim, write_kmf
from scripts.molynx_assets import GAME_MAX_KEYS, KEY_TOLERANCES

PREFIX = "Orc-"
REFERENCE = ("Troll-Idle1", 0)
# Troll record slot -> Troll file (Creatures.kwd record 10); the Orc's files keep the slot, not the name
TROLL_FILES = {
    "walk": "Troll_Walk", "run": "Troll_Run", "draggedPose": "TRO-Dragged_Pose", "recoilHff": "Troll_Recoil(hff)",
    "melee1": "Troll_Melee", "die": "Troll-Die", "happy": "Troll_Happy", "angry": "Troll_Angry",
    "stunnedPose": "Troll_Stunned", "swing": "Troll-Swing", "sleep": "Troll-bed(sleeping)", "eat": "Troll-Eat",
    "torture": "Tro-Torture", "drink": "Troll-Drink", "idle1": "Troll-Idle1", "recoilHfb": "Troll_Recoil(hfb)",
    "pray": "Troll-Pray", "fallback": "Troll-Fallback", "elec": "tro-elec", "getUp": "Troll-Getup",
    "dance": "Tro-Dance", "entrance": "tro-Entrance", "idle2": "Troll-Idle2", "drunk2": "Tro-Drunk",
    "drunkIdle": "TRO-Drunkidle", "walkback": "TRO_WalkBack", "poseFrame": "Tro-PoseFrame",
    "diePose": "Troll-DiePose",
}
SKINS = {"trollfrnt": "Orc_Front", "trollback": "Orc_Back"}

# UV boxes (u0, v0, u1, v1) in 0..1 of the named texture
HEAD_UV = {"trollfrnt": [(0.38, 0.26, 0.90, 0.66)], "trollback": [(0.0, 0.56, 0.40, 1.0)]}
HEAD_MIN_HEIGHT = 0.38      # of the reference frame (the Troll's top is 0.498): keys seen lower are not head
MALLET_UV = {"trollfrnt": [(0.0, 0.66, 0.45, 1.0), (0.45, 0.66, 0.90, 0.86)]}


def _in_boxes(uv: np.ndarray, boxes) -> np.ndarray:
    m = np.zeros(len(uv), bool)
    for u0, v0, u1, v1 in boxes:
        m |= (uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
    return m


@dataclass
class Clip:
    stem: str
    model: Kmf
    frames: list            # frames[f] = list of (n_s, 3) per sprite (all sprites, props included)
    troll: list             # indices of the sprites drawn with the Troll's skins

    @property
    def is_anim(self) -> bool:
        return self.model.anim is not None

    def sprites(self):
        return (self.model.anim or self.model.mesh).sprites

    def texture(self, s: int) -> str:
        return self.model.materials[self.sprites()[s].material].textures[0].lower()

    def uv(self, s: int) -> np.ndarray:
        return np.array([v.uv for v in self.sprites()[s].vertices], float) / 32768.0

    def part(self, boxes: dict) -> list[tuple[int, np.ndarray]]:
        """(sprite, vertex mask) for every Troll sprite with vertices in the part's UV boxes."""
        out = []
        for s in self.troll:
            t = self.texture(s)
            if t in boxes:
                m = _in_boxes(self.uv(s), boxes[t])
                if m.any():
                    out.append((s, m))
        return out

    def points(self, f: int, part) -> np.ndarray:
        return np.concatenate([self.frames[f][s][m] for s, m in part])

    def keys(self, part) -> list:
        """UV keys of a part's vertices (to pair them with the reference's)."""
        return [(self.texture(s), tuple(np.round(u, 4))) for s, m in part for u in self.uv(s)[m]]


def load_clip(meshes_dir: Path, stem: str) -> Clip:
    model = parse_kmf((Path(meshes_dir) / f"{stem}.kmf").read_bytes())
    if model.anim is not None:
        frames = [evaluate_anim_frame(model.anim, f) for f in range(model.anim.frame_count)]
    else:
        g = np.array(model.mesh.geometry, float)
        frames = [[g[[v.geom for v in s.vertices]] for s in model.mesh.sprites]]
    sprites = (model.anim or model.mesh).sprites
    troll = [i for i, s in enumerate(sprites) if model.materials[s.material].textures[0].lower() in SKINS]
    return Clip(stem, model, frames, troll)


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with R @ src_i + t ~ dst_i (least squares, proper rotation)."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cd - r @ cs


def _restrict(part, mask: np.ndarray):
    out, at = [], 0
    for s, m in part:
        n = int(m.sum())
        sub = m.copy()
        sub[np.flatnonzero(m)] = mask[at:at + n]
        at += n
        if sub.any():
            out.append((s, sub))
    return out


class PartTracker:
    """A body part's rigid pose in each frame of a clip, relative to the reference frame."""

    def __init__(self, ref: Clip, ref_frame: int, boxes: dict, min_height: float | None = None):
        self.boxes = boxes
        part = ref.part(boxes)
        pts = ref.points(ref_frame, part)
        self.ref = {}
        low = set()
        for k, p in zip(ref.keys(part), pts):
            if min_height is not None and -p[2] < min_height:
                low.add(k)
            self.ref.setdefault(k, []).append(p)
        for k in low:                      # a key also used below the part (a seam) is not the part's
            self.ref.pop(k, None)

    def poses(self, clip: Clip, select=None):
        """Per frame (R, t) taking the reference part onto the clip's part.

        Within a clip every vertex keeps its identity, so frame f is fitted to the clip's own frame 0
        exactly; frame 0 is tied to the reference once, through the UV keys (a key the reference uses
        for several vertices, e.g. a mirrored half, is paired with the nearest after a first fit on the
        unambiguous ones, and the worst fifth of the pairs is dropped)."""
        part = clip.part(self.boxes)
        if select is not None:
            part = select(clip, part)
        keys = clip.keys(part)
        mine = np.array([k in self.ref for k in keys])
        # keep only the vertices whose key is the part's (the UV box also catches seam vertices)
        part = _restrict(part, mine)
        keys = [k for k, m in zip(keys, mine) if m]
        if len(keys) < 8:
            return self._poses_by_shape(clip)
        p0 = clip.points(0, part)
        use = np.ones(len(keys), bool)
        uniq = [i for i, k in enumerate(keys) if use[i] and len(self.ref[k]) == 1]
        if len(uniq) >= 4:
            r, t = kabsch(np.array([self.ref[keys[i]][0] for i in uniq]), p0[uniq])
        else:
            r, t = np.eye(3), p0[use].mean(0) - np.concatenate([np.array(v) for v in self.ref.values()]).mean(0)
        for _ in range(3):
            idx = np.flatnonzero(use)
            src = np.array([np.array(self.ref[keys[i]])[np.argmin(np.linalg.norm(
                np.array(self.ref[keys[i]]) @ r.T + t - p0[i], axis=1))] for i in idx])
            err = np.linalg.norm(src @ r.T + t - p0[idx], axis=1)
            keep = err <= np.quantile(err, 0.8) + 1e-9
            r, t = kabsch(src[keep], p0[idx][keep])
        tie = (r, t)
        self.residual = float(np.quantile(err, 0.8))
        out = []
        for f in range(len(clip.frames)):
            rf, tf = kabsch(p0, clip.points(f, part))
            out.append((rf @ tie[0], rf @ tie[1] + tf))
        return out

    def _poses_by_shape(self, clip: Clip):
        """For a file mapped differently from the others (the game's TRO_WalkBack uses an older UV
        layout): the part is the top of the body in frame 0 (within the reference part's height span
        of the top), tied to the reference by nearest-point ICP."""
        refp = np.concatenate([np.array(v) for v in self.ref.values()])
        span = float(refp[:, 2].max() - refp[:, 2].min())
        part = []
        allz = np.concatenate([clip.frames[0][s][:, 2] for s in clip.troll])
        top = float(allz.min())
        for s in clip.troll:
            m = clip.frames[0][s][:, 2] <= top + span
            if m.any():
                part.append((s, m))
        p0 = clip.points(0, part)
        r, t = np.eye(3), p0.mean(0) - refp.mean(0)
        for _ in range(30):
            moved = refp @ r.T + t
            near = p0[np.argmin(np.linalg.norm(moved[:, None] - p0[None], axis=2), axis=1)]
            r, t = kabsch(refp, near)
        self.residual = float(np.quantile(np.linalg.norm(refp @ r.T + t - near, axis=1), 0.8))
        out = []
        for f in range(len(clip.frames)):
            rf, tf = kabsch(p0, clip.points(f, part))
            out.append((rf @ r, rf @ t + tf))
        return out


# ------------------------------------------------------------------------------------- writing


def _normals(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.zeros_like(p)
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    f = -np.cross(b - a, c - a)               # the game's outward normal
    for k in range(3):
        np.add.at(n, tris[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


@dataclass
class Accessory:
    """A rigid prop: points in reference-frame KMF space, triangles, UVs (0..1), its texture."""
    name: str
    points: np.ndarray
    tris: np.ndarray
    uv: np.ndarray
    texture: str
    tracker: str            # which PartTracker carries it


def _material(base: Material, name: str, texture: str) -> Material:
    return Material(name, [texture], base.flags, base.brightness, base.shininess, base.environment_map)


def orc_kmf(clip: Clip, accessories: list[Accessory], poses: dict, lods: int = 16) -> Kmf:
    """The clip with Orc skins and the accessories, re-encoded."""
    model = clip.model
    mats = []
    for m in model.materials:
        t = m.textures[0].lower()
        if t in SKINS:
            label = m.name[: len(m.name) - len(m.textures[0])] if m.name.endswith(m.textures[0]) else m.name
            mats.append(_material(m, label + SKINS[t], SKINS[t]))
        else:
            mats.append(m)
    base = model.materials[clip.sprites()[clip.troll[0]].material]
    sprites = [Sprite(s.material, s.triangles, list(s.vertices), s.mm_factor) for s in clip.sprites()]
    frames = [list(f) for f in clip.frames]
    rows_used = max(v.itab_index if clip.is_anim else v.geom for s in sprites for v in s.vertices) + 1
    for acc in accessories:
        tex_index = next((i for i, m in enumerate(mats) if m.textures == [acc.texture]), None)
        if tex_index is None:
            mats.append(_material(base, acc.name + acc.texture, acc.texture))
            tex_index = len(mats) - 1
        nrm = _normals(acc.points, acc.tris)
        verts = []
        for i in range(len(acc.points)):
            uv = tuple(int(round(np.clip(c, 0, 1) * 32768)) for c in acc.uv[i])
            v = Vertex(uv, tuple(float(c) for c in nrm[i]))
            if clip.is_anim:
                v.itab_index = rows_used + i
            else:
                v.geom = rows_used + i
            verts.append(v)
        rows_used += len(acc.points)
        tris = [tuple(int(c) for c in t) for t in acc.tris]
        sprites.append(Sprite(tex_index, [list(tris) for _ in range(lods)], verts, _mm(acc)))
        for f, (r, t) in enumerate(poses[acc.tracker]):
            frames[f].append(acc.points @ r.T + t)
    if clip.is_anim:
        a = model.anim
        anim = None
        for tol in KEY_TOLERANCES:
            anim = encode_anim(PREFIX + clip.stem, sprites, frames, a.wrap, a.controls, tolerance=tol)
            if len(anim.geom_xyz) - 1 <= GAME_MAX_KEYS:
                break
        return Kmf(model.version, "anim", mats, anim=anim, groups=model.groups)
    g = np.zeros((rows_used, 3))
    for s, p in zip(sprites, frames[0]):
        for v, q in zip(s.vertices, p):
            g[v.geom] = q
    old = model.mesh
    centre = (g.min(0) + g.max(0)) / 2
    mesh = Mesh(PREFIX + clip.stem, tuple(map(float, centre)), float(np.linalg.norm(g - centre, axis=1).max()),
                sprites, [tuple(map(float, q)) for q in g], old.controls)
    return Kmf(model.version, "mesh", mats, mesh=mesh, groups=model.groups)


def _mm(acc: Accessory) -> float:
    """mmFactor like molynx_assets: texels per model unit of the prop."""
    from scripts.molynx_assets import _mm_factor
    return _mm_factor(acc.points, acc.uv * 32768.0, acc.tris)


def write(model: Kmf, path: Path) -> None:
    Path(path).write_bytes(write_kmf(model))
