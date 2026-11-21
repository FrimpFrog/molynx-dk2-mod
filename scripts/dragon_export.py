"""The Dragon's game files: every clip as a DK2 .kmf animation, the static poses, the three skins.

The conventions are molynx_assets' (measured on the game's own files, see its docstring); this module
reuses its row, normal, sprite and mmFactor code on a plan of the same shape:

* KMF axes: x sideways, y forward, z DOWN. A model point (x left, y up, z forward) goes to KMF as
  (-x, z, -y) times SCALE, then OFFSET puts the feet of `stand` frame 0 at z = 0 and the mean of its
  body vertices (trunk, head, legs) at x = y = 0. The mirror in x turns the model's counter-clockwise
  outward triangles into the game's winding (outward normal -(b - a) x (c - a)).
* Height: `stand` frame 0 is HEIGHT_RATIO x the Bile Demon's idle1 height (0.536), i.e. as tall as the
  Horned Reaper's files (0.739); the creature record's scale then sets its size in the dungeon.
* Sprites of at most 255 vertices and 255 triangles, 16 identical LOD lists, one material per skin
  (named <Part><texture> like the game's, flags / brightness / shininess / environment map copied from
  the Bile Demon's body material), mmFactor from the bind-pose UVs.
* Control points (CTRL): the Bile Demon's 20 ids, each moved to the nearest dragon vertex of its body
  (not wings or spikes) after both are normalised to their bounding boxes, as Molynx's are.
* Animations: 30 fps, reduced keyframes; a looping clip ends on a copy of its first frame.
* Normals (v2, `DragonPlan.vertex_normals`): each vertex's own faces only. molynx_assets shares a normal
  between all of a part's vertices at one position, which sums a double-sided membrane's or fin's two
  faces to nothing (the game then lit the dragon's wings, spikes and fangs as if they faced straight
  up: flat). With per-vertex normals each side keeps its own, and the model's crease splits
  (dragon_model.crease_split) give hard edges where the game's own creatures have them.

Nothing here writes into a mod folder: `export_dragon(out_dir)` writes to the folder it is given.

LICENCE: GPL-3.0.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.dk2kmf import Kmf, Material, Mesh, Sprite, Vertex, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import encode_anim, write_kmf
from scripts.molynx_assets import (GAME_MAX_KEYS, KEY_TOLERANCES, LODS, VERSION, SpriteLayout, _mm_factor,
                                   _rows, _split, _sprites as _shared_sprites, _check_rows)

PREFIX = "Dragon-"
BILE_IDLE_HEIGHT = 0.536
HEIGHT_RATIO = 1.38             # 0.74 KMF units, the Horned Reaper's file height
GAME_FPS = 30
PART_LABEL = {"Dragon_Body": "Body", "Dragon_Head": "Head", "Dragon_Wing": "Wing"}
BODY_PARTS = ("trunk", "head", "jaw", "legs")


@dataclass
class DragonPlan:
    parts: list                      # [{"name", "uv", "texture"}] (molynx_assets._sprites reads "uv")
    sprites: list                    # SpriteLayout
    materials: list
    mm_factors: list
    scale: float = 1.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    controls: list = field(default_factory=list)
    prefix: str = PREFIX             # file and animation name prefix
    vertex_normals: bool = False     # v2: each vertex's own faces (see the module docstring)

    def to_kmf(self, model_parts):
        out = []
        for p in model_parts:
            p = np.asarray(p, float)
            q = np.stack([-p[:, 0], p[:, 2], -p[:, 1]], 1)
            out.append(q * self.scale + self.offset)
        return out

    def to_sprites(self, kmf_parts):
        return [np.asarray(kmf_parts[s.part])[s.vertices] for s in self.sprites]

    def frame(self, model_parts):
        return self.to_sprites(self.to_kmf(model_parts))


def _vertex_normals(plan, frame):
    """Unit normals per sprite vertex from the faces that use that very vertex of its part (a part split
    over several sprites adds up across them): -(b - a) x (c - a), area-weighted, game winding."""
    acc = {}
    for lay, pts in zip(plan.sprites, frame):
        t = np.asarray(lay.triangles)
        face = -np.cross(pts[t[:, 1]] - pts[t[:, 0]], pts[t[:, 2]] - pts[t[:, 0]])
        for k in range(3):
            for tri, f in zip(t[:, k], face):
                key = (lay.part, int(lay.vertices[tri]))
                acc[key] = acc.get(key, 0) + f
    out = []
    for lay in plan.sprites:
        n = np.array([acc[(lay.part, int(v))] for v in lay.vertices], float)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        out.append(np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), [0.0, 0.0, -1.0]))
    return out


def _sprites(plan, frame, rows, mesh):
    if not plan.vertex_normals:
        return _shared_sprites(plan, frame, rows, mesh)
    normals = _vertex_normals(plan, frame)
    starts = np.cumsum([0] + [len(p) for p in frame])
    material = {m.textures[0]: i for i, m in enumerate(plan.materials)}
    out = []
    for s, lay in enumerate(plan.sprites):
        uv = np.asarray(plan.parts[lay.part]["uv"], float)[lay.vertices]
        q = np.clip(np.rint(uv * 32768), 0, 65535).astype(int)
        verts = []
        for i in range(len(lay.vertices)):
            normal = tuple(float(np.float32(c)) for c in normals[s][i])
            row = int(rows[starts[s] + i])
            verts.append(Vertex((int(q[i, 0]), int(q[i, 1])), normal,
                                **({"geom": row} if mesh else {"itab_index": row})))
        out.append(Sprite(material[lay.texture], [list(lay.triangles) for _ in range(LODS)], verts,
                          plan.mm_factors[s]))
    return out


def make_plan(parts, rest_points, bile_idle: Kmf, height_ratio=HEIGHT_RATIO, *, uv_of=None, part_label=None,
              body_parts=None, prefix=PREFIX, vertex_normals=False, fixed_scale=None, fixed_xy=None):
    """parts: dragon_model Parts; rest_points: the stand frame 0 points per part (model space).
    Another creature passes its own `uv_of(part)` (atlas UVs), `part_label`, `body_parts` and `prefix`."""
    if uv_of is None:
        from scripts.dragon_texture import atlas_uv as uv_of
    part_label = PART_LABEL if part_label is None else part_label
    body_parts = BODY_PARTS if body_parts is None else body_parts
    pl = [{"name": p.name, "uv": uv_of(p), "texture": p.texture} for p in parts]
    sprites = [lay for i, p in enumerate(parts) for lay in _split(i, np.array(p.tris).ravel(), p.texture)]
    textures = list(dict.fromkeys(s.texture for s in sprites))
    donor = bile_idle.materials[bile_idle.anim.sprites[0].material]
    materials = [Material(part_label[t] + t, [t], donor.flags, donor.brightness, donor.shininess,
                          donor.environment_map) for t in textures]
    plan = DragonPlan(pl, sprites, materials, [], prefix=prefix, vertex_normals=vertex_normals)
    raw = plan.to_kmf(rest_points)
    allp = np.concatenate(raw)
    target = BILE_IDLE_HEIGHT * height_ratio
    plan.scale = target / float(allp[:, 2].max() - allp[:, 2].min())
    if fixed_scale is not None:            # Dragon v2: v1's size, whatever the new mesh's extremes
        plan.scale = fixed_scale
    body = np.concatenate([r for r, p in zip(raw, parts) if p.name in body_parts]) * plan.scale
    plan.offset = np.array([-body[:, 0].mean(), -body[:, 1].mean(), -allp[:, 2].max() * plan.scale])
    if fixed_xy is not None:               # Dragon v2: v1's place over the file origin (not the new mean)
        plan.offset[:2] = fixed_xy
    rest = plan.frame(rest_points)
    plan.mm_factors = [_mm_factor(p, np.asarray(pl[s.part]["uv"], float)[s.vertices], s.triangles)
                       for p, s in zip(rest, sprites)]
    plan.controls = _controls(plan, parts, bile_idle, rest, body_parts)
    return plan


def _unit_box(p):
    lo, hi = p.min(0), p.max(0)
    return (p - lo) / np.maximum(hi - lo, 1e-12)


def _controls(plan, parts, bile_idle, rest, body_parts=BODY_PARTS):
    theirs = evaluate_anim_frame(bile_idle.anim, 0)
    pairs = np.frombuffer(bile_idle.anim.controls, "<u4").reshape(-1, 2)
    flat = _unit_box(np.concatenate(theirs))
    starts = np.cumsum([0] + [len(p) for p in theirs])
    ours = _unit_box(np.concatenate(rest))
    our_starts = np.cumsum([0] + [len(p) for p in rest])
    body = [s for s, lay in enumerate(plan.sprites) if parts[lay.part].name in body_parts]
    cand = np.concatenate([np.arange(our_starts[s], our_starts[s + 1]) for s in body])
    out = []
    for ref, ident in pairs:
        sprite, vertex = int(ref) & 0xFFFF, int(ref) >> 16
        point = flat[starts[sprite] + vertex]
        best = int(cand[np.argmin(np.linalg.norm(ours[cand] - point, axis=1))])
        s = int(np.searchsorted(our_starts, best, side="right") - 1)
        out.append((s, best - int(our_starts[s]), int(ident)))
    return out


def anim_kmf(plan, stem, frames, wrap, reduce=True, extra=None):
    """frames: list of per-sprite KMF arrays. `extra`: (materials, sprites, frames) of a carried prop."""
    materials = list(plan.materials)
    if extra is not None:
        pm, psprites, pframes = extra
        frames = [list(f) + list(p) for f, p in zip(frames, pframes)]
    rows = _rows(frames)
    sprites = _sprites(plan, frames[0][:len(plan.sprites)], rows, mesh=False)
    if extra is not None:
        start = sum(len(p) for p in frames[0][:len(plan.sprites)])
        for k, sp in enumerate(psprites):
            verts = [Vertex(v.uv, v.normal, itab_index=int(rows[start + i])) for i, v in enumerate(sp.vertices)]
            index = len(materials)
            materials.append(pm[k])
            sprites.append(Sprite(index, [list(t) for t in sp.triangles], verts, sp.mm_factor))
            start += len(sp.vertices)
    controls = b"".join(np.array([(v << 16) | s, ident], "<u4").tobytes() for s, v, ident in plan.controls)
    anim = None
    for tol in KEY_TOLERANCES if reduce else (1.0,):
        anim = encode_anim(plan.prefix + stem, sprites, frames, wrap, controls, reduce=reduce, tolerance=tol)
        if len(anim.geom_xyz) - 1 <= GAME_MAX_KEYS:
            break
    return Kmf(VERSION, "anim", materials, anim=anim)


def pose_kmf(plan, stem, frame):
    rows = _rows([frame])
    _check_rows(rows, frame)
    pts = np.concatenate(frame)
    geometry = np.zeros((int(rows.max()) + 1, 3))
    geometry[rows] = pts
    lo, hi = geometry.min(0), geometry.max(0)
    centre = (lo + hi) / 2
    starts = np.cumsum([0] + [len(p) for p in frame])
    controls = b"".join(np.array([int(rows[starts[s] + v]), ident], "<u4").tobytes()
                        for s, v, ident in plan.controls)
    mesh = Mesh(plan.prefix + stem, tuple(float(c) for c in centre),
                float(np.linalg.norm(geometry - centre, axis=1).max()),
                _sprites(plan, frame, rows, mesh=True), [tuple(map(float, g)) for g in geometry], controls)
    return Kmf(VERSION, "mesh", list(plan.materials), mesh=mesh)
