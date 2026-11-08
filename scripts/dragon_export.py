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

Nothing here writes into a mod folder: `export_dragon(out_dir)` writes to the folder it is given.

LICENCE: GPL-3.0.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.dk2kmf import Kmf, Material, Mesh, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import encode_anim, write_kmf
from scripts.molynx_assets import (GAME_MAX_KEYS, KEY_TOLERANCES, VERSION, SpriteLayout, _mm_factor, _rows,
                                   _split, _sprites, _check_rows)

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


def make_plan(parts, rest_points, bile_idle: Kmf, height_ratio=HEIGHT_RATIO):
    """parts: dragon_model Parts; rest_points: the stand frame 0 points per part (model space)."""
    from scripts.dragon_texture import atlas_uv
    pl = [{"name": p.name, "uv": atlas_uv(p), "texture": p.texture} for p in parts]
    sprites = [lay for i, p in enumerate(parts) for lay in _split(i, np.array(p.tris).ravel(), p.texture)]
    textures = list(dict.fromkeys(s.texture for s in sprites))
    donor = bile_idle.materials[bile_idle.anim.sprites[0].material]
    materials = [Material(PART_LABEL[t] + t, [t], donor.flags, donor.brightness, donor.shininess,
                          donor.environment_map) for t in textures]
    plan = DragonPlan(pl, sprites, materials, [])
    raw = plan.to_kmf(rest_points)
    allp = np.concatenate(raw)
    target = BILE_IDLE_HEIGHT * height_ratio
    plan.scale = target / float(allp[:, 2].max() - allp[:, 2].min())
    body = np.concatenate([r for r, p in zip(raw, parts) if p.name in BODY_PARTS]) * plan.scale
    plan.offset = np.array([-body[:, 0].mean(), -body[:, 1].mean(), -allp[:, 2].max() * plan.scale])
    rest = plan.frame(rest_points)
    plan.mm_factors = [_mm_factor(p, np.asarray(pl[s.part]["uv"], float)[s.vertices], s.triangles)
                       for p, s in zip(rest, sprites)]
    plan.controls = _controls(plan, parts, bile_idle, rest)
    return plan


def _unit_box(p):
    lo, hi = p.min(0), p.max(0)
    return (p - lo) / np.maximum(hi - lo, 1e-12)


def _controls(plan, parts, bile_idle, rest):
    theirs = evaluate_anim_frame(bile_idle.anim, 0)
    pairs = np.frombuffer(bile_idle.anim.controls, "<u4").reshape(-1, 2)
    flat = _unit_box(np.concatenate(theirs))
    starts = np.cumsum([0] + [len(p) for p in theirs])
    ours = _unit_box(np.concatenate(rest))
    our_starts = np.cumsum([0] + [len(p) for p in rest])
    body = [s for s, lay in enumerate(plan.sprites) if parts[lay.part].name in BODY_PARTS]
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
    from scripts.dk2kmf import Sprite, Vertex
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
        anim = encode_anim(PREFIX + stem, sprites, frames, wrap, controls, reduce=reduce, tolerance=tol)
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
    mesh = Mesh(PREFIX + stem, tuple(float(c) for c in centre),
                float(np.linalg.norm(geometry - centre, axis=1).max()),
                _sprites(plan, frame, rows, mesh=True), [tuple(map(float, g)) for g in geometry], controls)
    return Kmf(VERSION, "mesh", list(plan.materials), mesh=mesh)
