"""The Necromancer: Soles' reworked Black Knight, as his own DK2 creature (not a reskin).

What changes from the Black Knight (DRKnight-* files, Creatures.kwd record 22):
  * his own skins, Necro_Front and Necro_Back (necro_skin: steel, blue electric arms and belly,
    gold horns and hilt, glowing blue eyes, a skull on the belt, a clean blade);
  * one horn knocked off: the helmet horn on his shield side (+x, away from the sword) is cut down
    to a jagged stub in every file;
  * three steel spikes on each shoulder plate, carried rigidly with the plate;
  * his lightning (necro_anims): a storm idle of his own (Idle2) and a research clip (the Black
    Knight never researches), and crackling lightning added to Happy, Angry and Entrance;
  * every other slot keeps the Black Knight's own motion, without lightning.

Vertex identity. The Black Knight's files number their vertices differently, so every file is tied
to the reference pose (DRKnight-Idle1 frame 0) once: a vertex's key is its texture and UV; where a
key is used by several reference vertices (the two mirrored halves) the candidate whose distances
to the vertex's already-matched triangle neighbours best match the clip's is taken. All edits are
made on reference "places" (reference vertices at one position), so every vertex at a place, in
any sprite, moves together, as the KMF rows require.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.dk2kmf import Kmf, Material, Mesh, Sprite, Vertex, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import anim_invariants, encode_anim, write_kmf
from scripts.molynx_assets import GAME_MAX_KEYS, KEY_TOLERANCES

PREFIX = "Necro-"
REFERENCE = "DRKnight-Idle1"
SKINS = {"dknightfrnt": "Necro_Front", "dknightbak": "Necro_Back"}
SOURCE_TEX = {"dknightfrnt": "DKnightfrntMM0", "dknightbak": "DKnightBakMM0"}

# Black Knight record slot -> (file stem, has Start file, has End file). Read from Creatures.kwd
# record 22 ("Black Knight"); the slot names are dk2kwd_creatures'.
BK_SLOTS = {
    "animWalkResource": ("DRKnight-Walk", True, True),
    "animRunResource": ("DRKnight-Run", True, False),
    "animDraggedPoseResource": ("DRK-Dragged_Pose", False, False),
    "animRecoilHffResource": ("DRKnight-Recoil(hff)", False, False),
    "animMelee1Resource": ("DRKnight-Melee", False, False),
    "animDieResource": ("DRKnight-Die", False, False),
    "animHappyResource": ("DRKnight-Happy", False, False),
    "animAngryResource": ("DRKnight-Angry", False, False),
    "animStunnedPoseResource": ("DRKnight-Stunned_Pose", False, False),
    "animSwingResource": ("DRKnight-Swing", False, False),
    "animSleepResource": ("DRKnight-Sleep_Pose", False, False),
    "animEatResource": ("DarkKnight-Eat", False, False),
    "animTortureResource": ("DRKni-Torture", False, False),
    "animDrinkResource": ("DRKnight-Drink", False, False),
    "animIdle1Resource": ("DRKnight-Idle1", False, False),
    "animRecoilHfbResource": ("DRKnight-Recoil(hfb)", False, False),
    "animPrayResource": ("DRKnight-Pray", True, True),
    "animFallbackResource": ("DRKnight-Fallback", False, False),
    "animElecResource": ("DKN-Elec", False, False),
    "animGetUpResource": ("DRKnight-Getup", False, False),
    "animDanceResource": ("Dkni-Dance", False, False),
    "animEntranceResource": ("DKNI-Entrance", False, False),
    "animIdle2Resource": ("DRKnight-Idle2", False, False),
    "animDrunk2Resource": ("DKNI-Drunk", False, False),
    "drunkIdle": ("DRK-Drunkidle", False, False),
    "animMelee2Resource": ("DRKnight-Melee2", True, True),
    "animWalkbackResource": ("DRKNI-Walkback", True, True),
    "animPoseFrameResource": ("DKni-PoseFrame", False, False),
    "animDiePoseResource": ("DRKnight-DiePose", False, False),
}
# The Necromancer's file stem per slot (Necro-<stem>, plus Start / End where the slot has them).
SLOT_STEM = {
    "animWalkResource": "Walk", "animRunResource": "Run", "animDraggedPoseResource": "DraggedPose",
    "animRecoilHffResource": "RecoilHff", "animMelee1Resource": "Melee", "animDieResource": "Die",
    "animHappyResource": "Happy", "animAngryResource": "Angry", "animStunnedPoseResource": "StunnedPose",
    "animSwingResource": "Swing", "animSleepResource": "SleepPose", "animEatResource": "Eat",
    "animTortureResource": "Torture", "animDrinkResource": "Drink", "animIdle1Resource": "Idle1",
    "animRecoilHfbResource": "RecoilHfb", "animPrayResource": "Pray", "animFallbackResource": "Fallback",
    "animElecResource": "ElecPose", "animGetUpResource": "GetUp", "animDanceResource": "Dance",
    "animEntranceResource": "Entrance", "animIdle2Resource": "Idle2", "animDrunk2Resource": "Drunk",
    "drunkIdle": "DrunkIdle", "animMelee2Resource": "Melee2", "animWalkbackResource": "Walkback",
    "animPoseFrameResource": "PoseFrame", "animDiePoseResource": "DiePose",
    # his own
    "animResearchResource": "Research", "animMagicResource": "Magic",
}

CX = 0.011                  # the Black Knight's centre line (x)
SWORD_SIDE = -1             # the sword hand is on -x
BROKEN_SIDE = +1            # so the knocked-off horn is the +x one (the preview: the full horn is on the sword side)
HORN_TIP_HEIGHT = 0.74      # reference vertices above this are the two horn tips
STUB = 0.20                 # the broken horn keeps this much of its length


# ------------------------------------------------------------------------------ clips

@dataclass
class Clip:
    stem: str
    model: Kmf
    frames: list                     # frames[f][s] = (n_s, 3)
    body: list                       # sprite indices drawn with the Black Knight's skins
    place: list = field(default_factory=list)   # per sprite: reference place id per vertex (-1 none)

    @property
    def is_anim(self) -> bool:
        return self.model.anim is not None

    def sprites(self):
        return (self.model.anim or self.model.mesh).sprites

    def texture(self, s: int) -> str:
        return self.model.materials[self.sprites()[s].material].textures[0].lower()

    def uv(self, s: int) -> np.ndarray:
        return np.array([v.uv for v in self.sprites()[s].vertices], float) / 32768.0

    def row(self, s: int, i: int) -> int:
        v = self.sprites()[s].vertices[i]
        return v.itab_index if self.is_anim else v.geom


def load_clip(meshes_dir: Path, stem: str) -> Clip:
    path = next(p for p in Path(meshes_dir).iterdir() if p.name.lower() == f"{stem.lower()}.kmf")
    model = parse_kmf(path.read_bytes())
    if model.anim is not None:
        frames = [evaluate_anim_frame(model.anim, f) for f in range(model.anim.frame_count)]
    else:
        g = np.array(model.mesh.geometry, float)
        frames = [[g[[v.geom for v in s.vertices]] for s in model.mesh.sprites]]
    sprites = (model.anim or model.mesh).sprites
    body = [i for i, s in enumerate(sprites) if model.materials[s.material].textures[0].lower() in SKINS]
    return Clip(stem, model, frames, body)


def kabsch(src: np.ndarray, dst: np.ndarray):
    """R, t with R @ src_i + t ~ dst_i (least squares, proper rotation)."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cd - r @ cs


# ------------------------------------------------------------------------------ the reference

class Reference:
    """The reference pose: its vertices grouped into places (one position each)."""

    def __init__(self, clip: Clip, frame: int = 0):
        self.clip = clip
        self.items = [(s, i) for s in clip.body for i in range(len(clip.sprites()[s].vertices))]
        pts = np.array([clip.frames[frame][s][i] for s, i in self.items])
        # places: identical positions
        key = [tuple(np.round(p, 5)) for p in pts]
        ids: dict = {}
        self.place_of = np.array([ids.setdefault(k, len(ids)) for k in key])
        self.places = np.zeros((len(ids), 3))
        for k, pid in ids.items():
            self.places[pid] = k
        self.by_key: dict = {}
        for (s, i), pid in zip(self.items, self.place_of):
            k = (clip.texture(s), tuple(np.round(clip.uv(s)[i], 4)))
            if pid not in self.by_key.setdefault(k, []):
                self.by_key[k].append(pid)
        clip.place = [np.full(len(clip.sprites()[s].vertices), -1) for s in range(len(clip.sprites()))]
        for (s, i), pid in zip(self.items, self.place_of):
            clip.place[s][i] = pid
        x, h = self.places[:, 0] - CX, -self.places[:, 2]
        self.height = h
        self.x = x

    # ---- parts (sets of place ids)
    def horn_tips(self) -> dict:
        """{+1: place, -1: place}: the topmost place on each side."""
        out = {}
        for side in (+1, -1):
            m = (self.height > HORN_TIP_HEIGHT) & (np.sign(self.x) == side)
            out[side] = int(np.flatnonzero(m)[np.argmax(self.height[m])])
        return out

    def horn_places(self, side: int) -> list[int]:
        """The places of one horn: the tip and the corners of every triangle that uses it."""
        tip = self.horn_tips()[side]
        out = {tip}
        c = self.clip
        for s in c.body:
            for t in c.sprites()[s].triangles[0]:
                pids = [c.place[s][v] for v in t]
                if tip in pids:
                    out |= set(int(p) for p in pids)
        return sorted(out)

    def head_places(self) -> np.ndarray:
        return np.flatnonzero((self.height > 0.60) & (np.abs(self.x) < 0.1))

    def sword_places(self) -> set:
        """Places of the blade (the 12-vertex sprite of the front skin)."""
        c = self.clip
        s = next(s for s in c.body if len(c.sprites()[s].vertices) == 12 and c.texture(s) == "dknightfrnt")
        return set(int(p) for p in c.place[s])

    def pauldron_places(self, side: int) -> np.ndarray:
        """The shoulder plate: the top of the arm's texture islands (it rides with the arm, so the
        spikes follow it when he raises his arms, as in Torture)."""
        from scripts.necro_anims import _arm_places
        m = ((self.height > 0.50) & (self.height < 0.64) & (np.abs(self.x) > 0.085) & (np.sign(self.x) == side)
             & (self.places[:, 1] < 0.1) & _arm_places(self))
        m[list(self.sword_places())] = False
        return np.flatnonzero(m)


def match(ref: Reference, clip: Clip) -> None:
    """Fill clip.place: each body vertex's reference place (see the module notes)."""
    if clip is ref.clip:
        return
    sprites = clip.sprites()
    clip.place = [np.full(len(sp.vertices), -1) for sp in sprites]
    p0 = clip.frames[0]
    amb = []
    for s in clip.body:
        uv = clip.uv(s)
        for i in range(len(uv)):
            cands = ref.by_key.get((clip.texture(s), tuple(np.round(uv[i], 4))), [])
            if len(cands) == 1:
                clip.place[s][i] = cands[0]
            elif cands:
                amb.append((s, i, cands))
    # vertices at one position share a row: give every vertex of a row the place any of them got
    rows: dict = {}
    for s in clip.body:
        for i in range(len(sprites[s].vertices)):
            rows.setdefault(clip.row(s, i), []).append((s, i))
    # neighbours through triangles
    nbr: dict = {}
    for s in clip.body:
        for t in sprites[s].triangles[0]:
            for a in t:
                for b in t:
                    if a != b:
                        nbr.setdefault((s, a), set()).add((s, b))
    for _ in range(4):
        left = []
        for s, i, cands in amb:
            if clip.place[s][i] >= 0:
                continue
            known = [(q, clip.place[q[0]][q[1]]) for q in nbr.get((s, i), ()) if clip.place[q[0]][q[1]] >= 0]
            if not known:
                left.append((s, i, cands))
                continue
            here = p0[s][i]
            best, score = None, np.inf
            for c in cands:
                e = sum(abs(np.linalg.norm(here - p0[q[0]][q[1]]) - np.linalg.norm(ref.places[c] - ref.places[pid]))
                        for q, pid in known)
                if e < score:
                    best, score = c, e
            clip.place[s][i] = best
        # share along rows
        for members in rows.values():
            got = [clip.place[s][i] for s, i in members if clip.place[s][i] >= 0]
            if got:
                for s, i in members:
                    if clip.place[s][i] < 0:
                        clip.place[s][i] = got[0]
        amb = left
        if not amb:
            break
    # anything still unknown: nearest after a global rigid fit
    known = [(s, i) for s in clip.body for i in range(len(sprites[s].vertices)) if clip.place[s][i] >= 0]
    if amb and len(known) >= 4:
        r, t = kabsch(np.array([p0[s][i] for s, i in known]), np.array([ref.places[clip.place[s][i]] for s, i in known]))
        for s, i, cands in amb:
            q = r @ p0[s][i] + t
            clip.place[s][i] = min(cands, key=lambda c: np.linalg.norm(ref.places[c] - q))


def part_pose(ref: Reference, clip: Clip, places, frame: int):
    """(R, t) taking the reference places onto where the clip has them in `frame` (None if the clip
    has fewer than 3 of them)."""
    want = set(int(p) for p in places)
    src, dst = [], []
    seen = set()
    for s in clip.body:
        for i, pid in enumerate(clip.place[s]):
            if pid in want and pid not in seen:
                seen.add(pid)
                src.append(ref.places[pid])
                dst.append(clip.frames[frame][s][i])
    if len(src) < 3:
        return None
    src, dst = np.array(src), np.array(dst)
    r, t = kabsch(src, dst)
    err = np.linalg.norm(src @ r.T + t - dst, axis=1)
    keep = err <= max(2.0 * np.median(err), 1e-4)        # a stray match must not swing the part
    if keep.sum() >= 3 and not keep.all():
        r, t = kabsch(src[keep], dst[keep])
    return r, t


# ------------------------------------------------------------------------------ the broken horn

def break_horn(ref: Reference, clip: Clip) -> None:
    """Cut the BROKEN_SIDE horn down to a jagged stub, in every frame (in place)."""
    tip = ref.horn_tips()[BROKEN_SIDE]
    horn = ref.horn_places(BROKEN_SIDE)
    base = ref.places[[p for p in horn if p != tip]].mean(0)
    v = ref.places[tip] - base
    side = np.array([1.0, 0.0, 0.0]) * BROKEN_SIDE
    stub = base + STUB * v + 0.006 * side            # a little off-axis: broken, not filed down
    head = [p for p in ref.head_places() if p not in set(ref.horn_tips().values())]
    tips = list(ref.horn_tips().values())
    # which of the clip's tip vertices is the broken one: the two horns share UVs, so the match may
    # swap them; decide by where each sits on the helmet (head pose of frame 0)
    pose0 = part_pose(ref, clip, head, 0)
    if pose0 is None:
        return
    r0, t0 = pose0
    other = ref.horn_tips()[-BROKEN_SIDE]
    sel = []
    for s in clip.body:
        m = np.isin(clip.place[s], tips)
        if not m.any():
            continue
        local = (clip.frames[0][s][m] - t0) @ r0          # back into the reference pose
        near = (np.linalg.norm(local - ref.places[tip], axis=1) < np.linalg.norm(local - ref.places[other], axis=1))
        idx = np.flatnonzero(m)[near]
        if len(idx):
            sel.append((s, idx))
    for f in range(len(clip.frames)):
        pose = part_pose(ref, clip, head, f)
        if pose is None:
            return
        r, t = pose
        new = r @ stub + t
        for s, idx in sel:
            clip.frames[f][s][idx] = new


# ------------------------------------------------------------------------------ shoulder spikes

@dataclass
class Accessory:
    name: str
    points: np.ndarray            # reference-pose KMF points
    tris: np.ndarray
    uv: np.ndarray                # 0..1 in `texture`
    texture: str
    places: object                # reference places that carry it


SPIKE_UV = ((0.055, 0.12), (0.095, 0.12), (0.075, 0.42))    # a steel strip of the blade in Necro_Front


def _cone(base: np.ndarray, axis: np.ndarray, radius: float, length: float, sides: int = 4):
    axis = axis / np.linalg.norm(axis)
    a = np.cross(axis, [0.0, 1.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(axis, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    ring = [base + radius * (np.cos(k * 2 * np.pi / sides) * a + np.sin(k * 2 * np.pi / sides) * b) for k in range(sides)]
    pts = np.array(ring + [base + axis * length])
    tip = sides
    tris = [(k, tip, (k + 1) % sides) for k in range(sides)]
    uv = [SPIKE_UV[0] if k % 2 == 0 else SPIKE_UV[1] for k in range(sides)] + [SPIKE_UV[2]]
    return pts, np.array(tris), np.array(uv, float)


def shoulder_spikes(ref: Reference) -> list[Accessory]:
    """Three spikes along the top of each shoulder plate, pointing up and out."""
    out = []
    for side in (+1, -1):
        pl = ref.pauldron_places(side)
        P = ref.places[pl]
        h = -P[:, 2]
        top = P[h > np.quantile(h, 0.55)]
        outer = top[np.argsort(-np.abs(top[:, 0] - CX))][: max(3, len(top) // 2)]
        ys = np.linspace(outer[:, 1].min(), outer[:, 1].max(), 5)[1:4]     # front-to-back along the plate
        pts, tris, uvs = [], [], []
        for k, y in enumerate(ys):
            near = outer[np.argsort(np.abs(outer[:, 1] - y))[:2]].mean(0)
            near[1] = y
            base = near + np.array([0.0, 0.0, 0.004])                     # sink a little into the plate
            axis = np.array([side * (0.55 + 0.1 * k), (k - 1) * 0.25, -1.0])   # up (-z) and out
            length = (0.040, 0.050, 0.038)[k]
            p, t, u = _cone(base, axis, 0.0085, length)
            tris.append(t + sum(len(q) for q in pts))
            pts.append(p)
            uvs.append(u)
        # carried by the plate together with the upper arm: the plate alone bends in big arm moves
        from scripts.necro_anims import BONES, labels
        arm = np.flatnonzero(labels(ref) == BONES.index("armP" if side > 0 else "armN"))
        out.append(Accessory(f"spikes{'L' if side > 0 else 'R'}", np.vstack(pts), np.vstack(tris), np.vstack(uvs),
                             "Necro_Front", np.union1d(pl, arm)))
    return out


# ------------------------------------------------------------------------------ writing

def _normals(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.zeros_like(p)
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    f = -np.cross(b - a, c - a)               # the game's outward normal (as orc_build)
    for k in range(3):
        np.add.at(n, tris[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _material(base: Material, name: str, texture: str) -> Material:
    return Material(name, [texture], base.flags, base.brightness, base.shininess, base.environment_map)


def _mm(acc: Accessory) -> float:
    from scripts.molynx_assets import _mm_factor
    return _mm_factor(acc.points, acc.uv * 32768.0, acc.tris)


@dataclass
class Moving:
    """A prop whose points are given per frame (the tome, the cast's crackle)."""
    name: str
    texture: str
    tris: np.ndarray
    uv: np.ndarray
    frames: list                  # per frame (n, 3) KMF points


def clip_from_places(ref: Reference, posed: np.ndarray, stem: str) -> Clip:
    """A new clip on the reference's sprites: posed[f] gives every reference place's position."""
    c = ref.clip
    frames = []
    for f in range(len(posed)):
        fr = []
        for s in range(len(c.sprites())):
            if s in c.body:
                fr.append(posed[f][c.place[s]])
            else:
                fr.append(c.frames[0][s].copy())
        frames.append(fr)
    new = Clip(stem, c.model, frames, list(c.body))
    new.place = [p.copy() for p in c.place]
    return new


def necro_kmf(ref: Reference, clip: Clip, accessories: list[Accessory], name: str, lods: int = 16,
              wrap: bool | None = None, moving: list | None = None) -> Kmf:
    """The clip with the Necromancer's skins, broken horn and spikes (and any moving props), re-encoded."""
    match(ref, clip)
    break_horn(ref, clip)
    model = clip.model
    mats = []
    for m in model.materials:
        t = m.textures[0].lower()
        if t in SKINS:
            label = m.name[: len(m.name) - len(m.textures[0])] if m.name.endswith(m.textures[0]) else m.name
            mats.append(_material(m, label + SKINS[t], SKINS[t]))
        else:
            mats.append(m)
    base = model.materials[clip.sprites()[clip.body[0]].material]
    # rows renumbered densely (some game files carry rows no vertex uses; the encoder wants none)
    old_rows = sorted({clip.row(s, i) for s in range(len(clip.sprites())) for i in range(len(clip.sprites()[s].vertices))})
    dense = {r: k for k, r in enumerate(old_rows)}
    sprites = []
    for s in clip.sprites():
        vs = []
        for v in s.vertices:
            if clip.is_anim:
                vs.append(Vertex(v.uv, v.normal, v.geom, dense[v.itab_index]))
            else:
                vs.append(Vertex(v.uv, v.normal, dense[v.geom], v.itab_index))
        sprites.append(Sprite(s.material, s.triangles, vs, s.mm_factor))
    frames = [list(f) for f in clip.frames]
    rows_used = len(old_rows)
    for acc in accessories:
        poses = [part_pose(ref, clip, acc.places, f) for f in range(len(frames))]
        if any(p is None for p in poses):
            continue
        tex_index = next((i for i, m in enumerate(mats) if m.textures == [acc.texture]), None)
        if tex_index is None:
            mats.append(_material(base, acc.name + acc.texture, acc.texture))
            tex_index = len(mats) - 1
        nrm = _normals(acc.points, acc.tris)
        verts = []
        for i in range(len(acc.points)):
            uv = tuple(int(round(np.clip(c, 0, 1) * 32767)) for c in acc.uv[i])
            v = Vertex(uv, tuple(float(c) for c in nrm[i]))
            if clip.is_anim:
                v.itab_index = rows_used + i
            else:
                v.geom = rows_used + i
            verts.append(v)
        rows_used += len(acc.points)
        tris = [tuple(int(c) for c in t) for t in acc.tris]
        sprites.append(Sprite(tex_index, [list(tris) for _ in range(lods)], verts, _mm(acc)))
        for f, (r, t) in enumerate(poses):
            frames[f].append(acc.points @ r.T + t)
    for mv in moving or []:
        tex_index = next((i for i, m in enumerate(mats) if m.textures == [mv.texture]), None)
        if tex_index is None:
            mats.append(_material(base, mv.name + mv.texture, mv.texture))
            tex_index = len(mats) - 1
        n0 = np.asarray(mv.frames[0])
        nrm = _normals(n0 + 1e-6, mv.tris) if np.ptp(n0) > 0 else np.tile([0.0, 0.0, -1.0], (len(n0), 1))
        verts = []
        for i in range(len(n0)):
            v = Vertex(tuple(int(round(np.clip(c, 0, 1) * 32767)) for c in mv.uv[i]), tuple(float(c) for c in nrm[i]))
            v.itab_index = rows_used + i
            verts.append(v)
        rows_used += len(n0)
        sprites.append(Sprite(tex_index, [[tuple(int(c) for c in t) for t in mv.tris] for _ in range(lods)], verts,
                              _mm(Accessory(mv.name, n0 if np.ptp(n0) > 0 else n0 + np.random.default_rng(0).random(n0.shape) * 1e-3,
                                            mv.tris, mv.uv, mv.texture, []))))
        for f in range(len(frames)):
            frames[f].append(np.asarray(mv.frames[f], float))
    if clip.is_anim:
        a = model.anim
        anim = None
        for tol in KEY_TOLERANCES:
            anim = encode_anim(name, sprites, frames, a.wrap if wrap is None else wrap, a.controls, tolerance=tol)
            if len(anim.geom_xyz) - 1 <= GAME_MAX_KEYS:
                break
        bad = anim_invariants(anim)
        if bad:
            raise ValueError(f"{name}: {bad}")
        return Kmf(model.version, "anim", mats, anim=anim, groups=model.groups)
    g = np.zeros((rows_used, 3))
    for s, p in zip(sprites, frames[0]):
        for v, q in zip(s.vertices, p):
            g[v.geom] = q
    old = model.mesh
    centre = (g.min(0) + g.max(0)) / 2
    mesh = Mesh(name, tuple(map(float, centre)), float(np.linalg.norm(g - centre, axis=1).max()),
                sprites, [tuple(map(float, q)) for q in g], old.controls)
    return Kmf(model.version, "mesh", mats, mesh=mesh, groups=model.groups)


# ------------------------------------------------------------------------------ skins

def skins(meshes_dir: Path, cache) -> dict:
    """{'Necro_Front': 128 x 128 image, 'Necro_Back': ...} (+ the 512 paintings under '<name>@512')."""
    from scripts.necro_skin import UP, SIZE, body_label, finish, paint, texel_maps
    ref_clip = load_clip(meshes_dir, REFERENCE)
    ref = Reference(ref_clip)
    sprites = ref_clip.sprites()
    textures = [ref_clip.texture(s) for s in range(len(sprites))]
    points = [p.copy() for p in ref_clip.frames[0]]
    sword = next(s for s in range(len(sprites)) if len(sprites[s].vertices) == 12 and textures[s] == "dknightfrnt")
    # horn texels: the triangles that use either horn tip
    tips = set(ref.horn_tips().values())
    out = {}
    for src, dst in SKINS.items():
        res = SIZE * UP
        pos, spr = texel_maps(sprites, points, textures, src, res)
        horn = np.zeros((res, res), bool)
        from scripts.necro_skin import texel_maps as _tm
        horn_sprites = []
        for s in range(len(sprites)):
            if textures[s] != src:
                continue
            tri = [t for t in sprites[s].triangles[0] if any(ref_clip.place[s][v] in tips for v in t)]
            if tri:
                fake = Sprite(sprites[s].material, [tri], sprites[s].vertices)
                hp, _ = _tm([fake], [points[s]], [src], src, res)
                horn |= ~np.isnan(hp[..., 0])
        lab = body_label(pos, spr, sword, horn)
        big = paint(src, cache.get(SOURCE_TEX[src]), pos, spr, lab, seed=7 if src == "dknightfrnt" else 11)
        big = _skull(big, pos, src)
        out[dst + "@512"] = big
        out[dst] = finish(big)
    return out


def _skull(img, pos, src):
    """The belt skull: at the front of the belt, centred, just under the waist (front skin only)."""
    from scripts.necro_skin import paint_skull, UP
    if src != "dknightfrnt":
        return img
    a = np.asarray(img, float) / 255
    x, y, h = pos[..., 0] - CX, pos[..., 1], -pos[..., 2]
    target = np.array([0.0, 0.10, 0.345])
    d = np.sqrt(x ** 2 + (h - target[2]) ** 2) - 0.5 * np.clip(y, -1, 0.2)
    d[np.isnan(d)] = np.inf
    d[y < 0.03] = np.inf                                # the front of the body only
    r, c = np.unravel_index(np.argmin(d), d.shape)
    # which way is up on the texture here: the direction (row, col) in which height grows
    hh = np.nan_to_num(h, nan=np.nanmean(h))
    gr = hh[min(r + 4, hh.shape[0] - 1), c] - hh[max(r - 4, 0), c]
    gc = hh[r, min(c + 4, hh.shape[1] - 1)] - hh[r, max(c - 4, 0)]
    up = np.array([gr, gc])
    if np.linalg.norm(up) < 1e-9:
        up = np.array([-1.0, 0.0])
    a = paint_skull(a, (r, c), UP, up)
    from PIL import Image
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


class TextureOverlay:
    """A TextureCache with some textures replaced or added (for previews)."""

    def __init__(self, base, extra: dict):
        self.base = base
        self.extra = {f"{k}MM0": v for k, v in extra.items()}
        self.names = list(base.names) + list(self.extra)

    def get(self, name):
        return self.extra[name] if name in self.extra else self.base.get(name)


# ------------------------------------------------------------------------------ everything

# Slots that play his own clip instead of the Black Knight's (necro_anims.own_clips). An own clip
# has no Start / End files: the record clears those flags for its slot.
OWN_SLOTS = {"animWalkResource": "Walk", "animRunResource": "Run", "animIdle2Resource": "Storm",
             "animResearchResource": "Research", "animMagicResource": "Magic"}


def file_plan() -> list[tuple[str, str, str]]:
    """(slot, Necromancer file stem, source) for every file: the Black Knight's (with their Start /
    End files) and his own ("own:Walk", "own:Run", "own:Storm", "own:Research")."""
    out = []
    for slot, (stem, has_start, has_end) in BK_SLOTS.items():
        mine = PREFIX + SLOT_STEM[slot]
        if slot in OWN_SLOTS:
            out.append((slot, mine, "own:" + OWN_SLOTS[slot]))
            continue
        out.append((slot, mine, stem))
        if has_start:
            out.append((slot, mine + "Start", stem + "Start"))
        if has_end:
            out.append((slot, mine + "End", stem + "End"))
    out.append(("animResearchResource", PREFIX + "Research", "own:Research"))
    out.append(("animMagicResource", PREFIX + "Magic", "own:Magic"))   # his Raise Dead cast
    return out


def export(meshes_dir: Path, cache, out_dir: Path) -> dict:
    """Write every Necromancer file (<name>.kmf) and his three skins (PNG) into out_dir. Returns per
    file: source, frames, wrap, keys, triangles of the first LOD."""
    from scripts.necro_anims import gear_texture, lightning_for, own_clips
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = Reference(load_clip(meshes_dir, REFERENCE))
    spikes = shoulder_spikes(ref)
    own = own_clips(ref)
    info = {}
    for slot, name, src in file_plan():
        if src.startswith("own:"):
            clip, moving, wrap = own[src[4:]]
            k = necro_kmf(ref, clip, spikes, name, wrap=wrap, moving=moving)
        else:
            clip = load_clip(meshes_dir, src)
            k = necro_kmf(ref, clip, spikes, name, moving=lightning_for(ref, clip, name))
        (out_dir / f"{name}.kmf").write_bytes(write_kmf(k))
        x = k.anim or k.mesh
        info[name] = {"slot": slot, "from": src, "kind": k.kind,
                      "frames": k.anim.frame_count if k.anim else 1,
                      "wrap": bool(k.anim.wrap) if k.anim else None,
                      "keys": len(k.anim.geom_xyz) - 1 if k.anim else 0,
                      "tris": sum(len(s.triangles[0]) for s in x.sprites)}
    sk = skins(meshes_dir, cache)
    for n in SKINS.values():
        sk[n].save(out_dir / f"{n}.png")
    gear_texture().save(out_dir / "Necro_Gear.png")
    return info
