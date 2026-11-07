"""Molynx's game files: his model and clips as DK2 .kmf files, his two textures and two portraits.

`export_molynx_assets(out_dir)` rebuilds the finished character from the repo's code (the MegaGlest
Behemoth dressed by custom_character.build_character, with the 21 DK2 clips retargeted by
behemoth_anims.build_all), turns every clip into a game animation and writes

  Molynx-<clip>.kmf      the 25 clips of molynx_record.CLIPS (animation files)
  Molynx-<clip>_once.kmf / _loop.kmf  variants the slots need (Bile Demon: walk_once)
  Molynx-PoseFrame.kmf   `stand` frame 0          (static mesh files, like the Bile Demon's poses)
  Molynx-DiePose.kmf     `die` last frame
  Molynx-DraggedPose.kmf `being_picked_up` frame 0
  Molynx-StunnedPose.kmf `staggering` frame 48 of 96 (the middle frame)
  Molynx-ElecPose.kmf    `getting_hit` frame 5 of 17 (the peak: the largest mean vertex distance
                         from frame 0; the largest single-vertex distance is at the last frame, 16,
                         where one arm drifts, not the flinch)
  Molynx_Body.png, Molynx_Mace.png      the textures the materials name
  molynx-bg.png (64x128), molynx.png (64x64), portraits_preview.png

Every convention below was measured on the 2015 game files (Meshes.WAD, EngineTextures, the editor
Creatures.kwd) or on the user's own Flame texture mods; the numbers are quoted where they decide.

Space, scale and placement. KMF axes: x sideways, y forward, z DOWN (kmf_textured._yup shows a
KMF point as Y-up (x, -z, y)). The character data is Y-up and right-handed (MegaGlest), DK2 data
left-handed: behemoth_anims retargets with `T * FLIP` (FLIP = [-1, 1, 1]), so a character point p
goes to DK2 Y-up as p * FLIP and to KMF as (X, Z, -Y). One uniform factor makes the height of
`stand` frame 0 (max - min of z over all sprites) the height of the Bile Demon's idle1 frame 0 (the
file its record names, BileDemon-Idle1: 0.536) times `height_ratio`. Placement follows the game's
creatures (idle1 frame 0 of 30 creatures: the lowest point, largest z, is 0 within 0.004; the
vertex mean is 0 in x within 0.04 and near 0 in y): feet at z = 0, the mean of the body sprites'
vertices at x = y = 0, all from `stand` frame 0. He faces +y like the Bile Demon (its control
points 2000 / 2001, at head height, sit at y = +0.12; Molynx's head is forward of his body).

Winding and normals. In the game's files the stored (outward, unit) normal points along
-(b - a) x (c - a) of the triangles (Bile Demon PoseFrame: the area-weighted sum disagrees in sign
with the stored normal at about 1 % of vertices; the cross product points inward for 78 % of
triangles). The MegaGlest mesh is wound the other way round in a
right-handed space; the mirror and the proper rotation to KMF axes turn it into exactly the game's
winding, so triangle corners are kept in their order (reversing them would break it). Normals:
area-weighted sum of -(b - a) x (c - a) over the triangles of a part, shared by all vertices of the
part at the same position (UV seams stay smooth), from frame 0 of each file.

Sprites, LODs and mmFactor. Triangle corners are bytes, and no game sprite has more than 255
vertices or more than 255 triangles (6551 sprites in 2007 files; 2 have exactly 255 triangles), so a
part is cut into sprites of at most 255 vertices and 255 triangles (triangles in order, greedily);
the body becomes 3 sprites (255 / 222 / 156 vertices, 254 / 255 / 154 triangles), the mace 1 (246,
86), all body sprites sharing one material (132 game
files reuse a material across sprites, e.g. DRKnight-Melee). Every sprite carries 16 LOD lists; the
game's own lists shrink with distance (Bile Demon 120 -> 7 triangles) from its artists' tool. We have
no reduction, so every LOD is the full list, the conservative valid choice (22 game sprites do the
same): far away he costs a few more triangles, never a hole. mmFactor (SPHD float, per sprite; the
Bile Demon's poses copy idle1's values) tracks texture-space size per model size: over a sample of
1173 game sprites (every 5th file) it is about 2.33 x sum(uv edge lengths) / sum(model edge
lengths) (median fit, log IQR 0.35, so not the game's exact formula); it is set that way from
`stand` frame 0. Its role is not known; it looks like a mip-map hint.

Animations. Keyframes are reduced (encode_anim reduce=True: a vertex keeps only the keyframes it
needs within one quantisation step), as in 661 game files and 4 of the 21 Bile Demon files; fully
keyed files (reduce=False, like the other 17) would be far larger than any game animation (see
"Size").
A KMF has no rate: the creature record gives each slot (frames, fps) and the Bile Demon's slots all
say 30 fps (sleep 15). Every clip is resampled to 30 fps by linear interpolation of the vertices,
over the clip's own duration ((n - 1) / fps; a looping clip n / fps, or (n - 1) / fps when its last
frame already closes the loop), stretched by at most half a frame so the first and last source
frames are kept exactly. Clips at 30 fps are unchanged. A looping (`wrap`) clip ends on a copy of
frame 0, as all 209 looping game files do. A loop "already closes" (`loop_closes`) when its last
frame is within LOOP_GAP = 0.3 x the median frame-to-frame step of frame 0: that last frame is
REPLACED by the copy (frame count unchanged; the 8 retargeted loops that close exactly and
fidgeting, staggering, interacting_with_objects at 0.16 / 0.22 / 0.21 of a step), so the loop has
no near-duplicate frame; any other loop gets the copy appended. `plan.info` gives each clip's own
frame count, fps and wrap.

Loop flags per slot. The engine's idle / job logic keys on a file looping or ending, so a file's
wrap flag is the one its slots need (molynx_record.file_plan, the single source of truth shared
with the record; `required_wraps(bile)` here reads the Bile Demon's own files): `export_molynx_
assets(wraps=...)` (default required_wraps of the editor's Bile Demon) writes each listed file
with that flag, including the `<clip>_once` / `<clip>_loop` variants; clips not listed keep their
own flag. `file_clip` changes a clip's flag:
  - opening a loop keeps its frames (same count; it still ends on its copy of frame 0) and clears
    wrap. Evidence: 182 of the game's 936 non-looping files end on a copy of frame 0, among them 52
    of its 83 idle files and the Bile Demon's own idle1, idle2 and happy (the very slots that open
    stand, fidgeting and happy), so an idle that ends where it began is the game's own practice and
    lets idle follow idle without a jump; the record counts that last frame (non-looping: frames =
    file frames);
  - closing a clip that does not loop resamples its source as a loop (the rule above: the copy of
    frame 0 replaces a nearly closing last frame, else is appended). The step from the last source
    frame back to frame 0 is then played as one frame, so only a clip that ends near where it began
    is ever closed: its `loop_gap` (last-to-first gap / median step) must be at most
    MAX_CLOSE_GAP = 10 (getting_hit 5.8 could be closed; no slot of the Bile Demon needs it now). The one-way clips above it
    (molynx_record.ONE_WAY_GAPS: sleeping 37.1, waking_up 25.3) are never closed; plan_molynx
    checks that list against the source, and a slot that wanted one looped is reported in
    `wrap_exceptions` (the real Bile Demon: sleeping in the sleep slot).
The returned `info` gives the FINAL frames, fps and wrap of every animation file written; the
stage-2 record turns it into each slot's frames (file frames - 1 for looping files, as the Bile
Demon's record does: walk 41 -> 40; = file frames otherwise: idle1 190).

Size. Fully keyed (reduce=False), the biggest GEOM would be 136,344 keyframe positions
(Molynx-victory_celebration, 552 rows x 247) against 34,816 in the largest game animation
(Mis-Idle1), with five shipped clips above 34,816 and two above 65,535; reduced, every shipped
animation stays below 65,535 and only `staggering` is above 34,816 (an accepted exception; the
test pins the numbers). DraggedPose has 552 GEOM positions against 311 in the largest game mesh.

Poses. The Bile Demon's PoseFrame / DiePose / Dragged_Pose / Stunned_Pose / Elec files are static
meshes (kind 1, three sprites, 20 CTRL entries, record type byte 5 = mesh), so Molynx's five are
static meshes too:
GEOM holds each distinct position once, `pos` is its bounding-box centre and `scale` the largest
distance from it (862 of 862 game meshes).

Control points (CTRL). Not named by the community template ("uk"); decoded here from the Bile Demon:
in an animation an entry is (vertex << 16 | sprite, id) and in a static mesh (GEOM index, id) - the
idle1 and PoseFrame entries name the same positions this way. Ids 1000..1014 (1006 twice), 2000,
2001, 3000 and 5000 (the Bile Demon's 20) are kept in order; their meaning is not known (1000..
spread over the body, 2000 / 2001 at the front of the head, 3000 low at the back, 5000 = 1000 on top
of the head). Each is moved to Molynx by one rule: the Bile Demon's point in idle1 frame 0 and
Molynx's `stand` frame 0 are both normalised to their own bounding box (each axis to 0..1), and the
nearest Molynx vertex of his body, head or neck (every sprite but the mace: every Bile Demon point
is on its body; since the redesign his face is the Golem head's own sprites, so the points at the
front of the head land on it) takes it, the first on a tie. The same vertex is used in every file.

Materials and textures. Game materials are named <part><texture> with one texture each
(FrontBDemonFrnt / BDemonFrnt, SpikesBDEMONback / BDEMONback); Molynx's are BodyMolynx_Body and
MaceMolynx_Mace, and copy the rest (flags 0, brightness 0, shininess 0, environment map
DefaultEnvMap) from the Bile Demon's body material FrontBDemonFrnt. A material names its texture
without the cache's mip suffix (the cache holds BDemonFrntMM0..MM3), and the user's Flame texture
mods deliver a texture as Data/EngineTextures/<that name>.png: Quuz_UniqueBatTex ships
Unique_Bat.png, Quuz_HD BDemonFrnt-style names such as DarkElfFrnt.png, and their EngineTextures.dir
renames ("rubs out") exactly the MM0..MM3 entries of every name they ship, so the engine finds the
loose file. Hence Molynx_Body.png and Molynx_Mace.png, named as planned. Size: no texture in the
game's cache is larger than 128 px (0 of 1911 MM0 entries; creature skins are 128x128, RGB) and the
Quuz HD pack says the engine does not allow a higher resolution, so the body is resampled from
512 to MAX_TEXTURE = 128 (not the planned 256, which the evidence does not support; `body_size`
overrides it), Lanczos. Creature skins and the opaque HD-pack PNGs are RGB without alpha, so both
textures are RGB.

Portraits. The Bile Demon's gui\\creatures\\evil\\bile_demon-bg (64x128) and bile_demon (64x64)
are RGB with a 7-pixel grey riveted metal frame (black inner line at pixel 6) round the picture;
-bg shows the creature on a dark blue background, the icon a white head emblem on a dark gradient.
The user's image (1024x1536) has its own frame round a blue picture. Its frame is cut away (the
inner picture is where the pixels are coloured), the picture is cut to the game window's aspect
(50x114) about its centre (the brief's 1:2 crop, adjusted for the frame) and set into the Bile
Demon's own frame, so the panel looks like the game's. The icon is the face (with the horns) cut
square from the picture into the Bile Demon icon's frame: the game's icons are emblems, which the
user's picture does not have, so the preview shows both side by side for approval.

LICENCE: GPL-3.0 (uses scripts.dk2texture; base model: MegaGlest, CC BY-SA 3.0).
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.behemoth_anims import FLIP  # DK2 is left-handed, the character right-handed
from scripts.dk2kmf import Kmf, Material, Mesh, Sprite, Vertex, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import encode_anim, write_kmf
from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_creatures import ART_TYPE_NONE, Creature, read_creatures
from scripts.dk2paths import BEHEMOTH_DIR, EDITOR_DIR, GAME_DIR, MESHES_DIR, PORTRAIT_SRC
from scripts.molynx_record import (CLIPS, GAME_FPS, MAX_CLOSE_GAP, ONE_WAY_GAPS, POSE_STEMS, PREFIX,
                                   SOURCE_WRAP, FilePlan, clip_of, file_plan, slot_assignment)

VERSION = 17
LODS = 16
MAX_SPRITE_VERTICES = 255
MAX_SPRITE_TRIANGLES = 255
LOOP_GAP = 0.3  # a loop whose last frame is this close to frame 0 (x the median step) is closed
MAX_TEXTURE = 128
MM_PER_RATIO = 2.33
# The game's largest animation (Mis-Idle1) has 34,816 keyframe positions. An animation that would
# have more at the default tolerance (one quantisation step) is reduced with the smallest of
# KEY_TOLERANCES (in steps) that keeps it within that; one that none brings within it keeps one
# step (only victory_celebration, which no slot ships). 2.5 steps of the redesign's celebrating
# (the Troll dance, the most) is about 0.004 model units, half a percent of his height.
GAME_MAX_KEYS = 34_816
KEY_TOLERANCES = (1.0, 1.5, 2.0, 2.5, 3.0)
BODY_TEXTURE, MACE_TEXTURE = "Molynx_Body", "Molynx_Mace"
HEAD_TEXTURE, NECK_TEXTURE = "Molynx_Head", "Molynx_Neck"          # the redesign's Golem head and its band
TEXTURE_NAMES = {"custom_character/body": BODY_TEXTURE, "custom_character/mace": MACE_TEXTURE,
                 "custom_character/golem_head": HEAD_TEXTURE, "custom_character/head_neck": NECK_TEXTURE}
PART_NAMES = {BODY_TEXTURE: "Body", MACE_TEXTURE: "Mace", HEAD_TEXTURE: "Head", NECK_TEXTURE: "Neck"}
# (width, height) of each game texture: powers of two up to MAX_TEXTURE, all sizes the game's own
# cache uses (128x128: 629 textures, 128x32: 10). The neck band is a long strip (685 x 103 px).
TEXTURE_SIZES = {BODY_TEXTURE: (128, 128), MACE_TEXTURE: (128, 128), HEAD_TEXTURE: (128, 128),
                 NECK_TEXTURE: (128, 32)}
# Textures cut to the part of the image their part's UVs use (with UV_MARGIN round it) before they
# are scaled to the game size: the Golem face uses only a third by a half of its 512 px image.
CROP_TO_UV = (HEAD_TEXTURE,)
UV_MARGIN = 0.02
# The redesign is as tall as the Horned Reaper (the top of his head level with the Reaper's skull,
# as approved in the design scenes): scale 0.18 = 1.38 x the Bile Demon's idle1 height over his
# stand frame 0 height (the top of the mace, the same as before the redesign).
DESIGN_HEIGHT_RATIO = 1.38
POSE_SOURCES = {"PoseFrame": ("stand", 0), "DiePose": ("die", -1), "DraggedPose": ("being_picked_up", 0),
                "StunnedPose": ("falling", -1), "ElecPose": ("getting_hit", 5)}  # see the docstring
FRAME = 7                       # the game portraits' frame, pixels on every side
# The redesign's portrait pictures: renders of the design (stand frame 0, turned 15 degrees, on the
# blue of the user's picture, 10/23/89), made by data/molynx_design/painting_tools/portrait_render.py.
_DESIGN_DIR = Path(__file__).resolve().parent.parent / "data" / "molynx_design"
DESIGN_PORTRAIT = _DESIGN_DIR / "portrait_bg_A2.png"      # upper body, head centred: the -bg panel
DESIGN_ICON = _DESIGN_DIR / "portrait_icon_A2.png"        # the face with the horns: the icon
FACE_BOX = (0.503, 0.355, 0.508)  # the face square in the user's image: centre x, y and side, as
                                  # fractions of the image width, height and width
PREVIEW_SCALE = 4


# ---------------------------------------------------------------- game data

def bile_demon_record(editor_dir: Path = EDITOR_DIR) -> Creature:
    """The Bile Demon's record from the editor's Creatures.kwd."""
    blocks = read_blocks((Path(editor_dir) / "Creatures.kwd").read_bytes())
    for block in blocks:
        if block.signature == 0xAA:
            for creature in read_creatures(block):
                if creature.get("name") == "Bile Demon":
                    return creature
    raise LookupError("no Bile Demon in Creatures.kwd")


def game_mesh_path(name: str, meshes_dir: Path = MESHES_DIR) -> Path:
    """The extracted game file a resource name refers to (file names are matched ignoring case)."""
    want = f"{name}.kmf".lower()
    for path in Path(meshes_dir).iterdir():
        if path.name.lower() == want:
            return path
    raise FileNotFoundError(f"{name}.kmf is not in {meshes_dir}")


def bile_file_wraps(biles, meshes_dir: Path = MESHES_DIR) -> dict:
    """{file name, lower case: its wrap flag, None for a static mesh} of every file the Molynx
    slots of these Bile Demon records replace (a slot of type NONE is not loaded and is skipped)."""
    out = {}
    for bile in biles:
        for slot in slot_assignment(bile):
            name = bile.resource_name(slot)
            if bile.art_tail(slot).type == ART_TYPE_NONE or name.lower() in out:
                continue
            model = parse_kmf(game_mesh_path(name, meshes_dir).read_bytes())
            out[name.lower()] = bool(model.anim.wrap) if model.kind == "anim" else None
    return out


def molynx_files(bile: Creature, meshes_dir: Path = MESHES_DIR) -> FilePlan:
    """molynx_record.file_plan on the Bile Demon's own files (slots, wraps, wrap_exceptions)."""
    return file_plan(bile, bile_file_wraps([bile], meshes_dir))


def required_wraps(bile: Creature, meshes_dir: Path = MESHES_DIR) -> dict:
    """{file stem: wrap flag} of every Molynx animation file the slots of `bile` name, variants
    included. One-way clips are listed non-looping (see molynx_files(...).wrap_exceptions)."""
    return dict(molynx_files(bile, meshes_dir).wraps)


def texture_cache(game_dir: Path = GAME_DIR):
    from scripts.dk2texture import TextureCache
    return TextureCache(Path(game_dir) / "EngineTextures.dir", Path(game_dir) / "EngineTextures.dat")


def _cache_image(cache, name: str) -> Image.Image:
    lowered = {n.lower(): n for n in cache.names}
    return cache.get(lowered[f"{name}MM0".lower()]).convert("RGB")


def build_source(unit_dir: Path = BEHEMOTH_DIR, meshes_dir: Path = MESHES_DIR, cache=None,
                 design: bool = True):
    """(data, textures) of the finished character: the MegaGlest stand / walk / attack / die plus
    the 21 retargeted clips, with the body triangles that some clip tears into spikes left out
    (behemoth_rig.stretched_triangles, measured on the original clip library). design=True (the
    default) gives the approved redesign of scripts.molynx_design (Troll fidgeting and
    celebrating, claw and mace-arm fixes, the Golem head, the red rock skin); design=False the
    character as first approved (equal to the reference pickle CHARACTER_PICKLE)."""
    from scripts.behemoth_anims import build_all
    from scripts.behemoth_rig import stretched_triangles
    from scripts.custom_character import build_character
    cache = texture_cache() if cache is None else cache
    with warnings.catch_warnings():
        # the donor tracks hold NaN for vertices a game file lacks; retarget's means of them warn
        warnings.simplefilter("ignore", RuntimeWarning)
        lib = build_all(Path(meshes_dir), cache, Path(unit_dir))
    drop = stretched_triangles(Path(unit_dir), lib)
    if design:
        from scripts.molynx_design import design_clips, review_clips
        lib = {**lib, **design_clips(Path(meshes_dir), cache, Path(unit_dir))}
        lib = {**lib, **review_clips(Path(meshes_dir), cache, Path(unit_dir), lib)}
    data, textures, _ = build_character(Path(unit_dir), clips=("stand", "walk", "attack", "die"),
                                        extra=lib, drop=drop)
    if design:
        from scripts.molynx_design import apply_design
        data, textures = apply_design(data, textures, Path(unit_dir))
    return data, textures


# ---------------------------------------------------------------- plan

@dataclass
class SpriteLayout:
    part: int                # index into the character's parts
    vertices: np.ndarray     # the part's vertex numbers, in sprite order
    triangles: list          # (a, b, c) in sprite vertex numbers
    texture: str


@dataclass
class Clip:
    frames: list             # frames[f][sprite] = (n, 3) KMF positions
    wrap: bool


@dataclass
class MolynxPlan:
    source: tuple
    height_ratio: float
    parts: list
    sprites: list
    scale: float
    offset: np.ndarray
    target_height: float
    clips: dict = field(default_factory=dict)
    poses: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    controls: list = field(default_factory=list)   # (sprite, vertex, id)
    materials: list = field(default_factory=list)
    mm_factors: list = field(default_factory=list)
    crops: dict = field(default_factory=dict)      # game texture -> (u0, v0, u1, v1) of its image used
    props: dict = field(default_factory=dict)      # clip -> Prop (game objects the file carries)

    def to_kmf(self, parts) -> list:
        """Character Y-up points (one array per part) -> placed KMF points."""
        return [_kmf_axes(np.asarray(p, float)) * self.scale + self.offset for p in parts]

    def to_sprites(self, parts) -> list:
        """Per-part arrays -> per-sprite arrays."""
        return [np.asarray(parts[s.part])[s.vertices] for s in self.sprites]


def _kmf_axes(p: np.ndarray) -> np.ndarray:
    q = p * FLIP
    return np.stack([q[..., 0], q[..., 2], -q[..., 1]], axis=-1)


def _split(part: int, indices, texture: str) -> list[SpriteLayout]:
    """Cut a part's triangles, in order, into sprites of at most MAX_SPRITE_VERTICES vertices and
    MAX_SPRITE_TRIANGLES triangles."""
    out, tris, used = [], [], {}

    def flush():
        if tris:
            order = np.array(sorted(used, key=used.get))
            out.append(SpriteLayout(part, order, [tuple(used[i] for i in t) for t in tris], texture))
    for tri in np.asarray(indices, int).reshape(-1, 3):
        new = [i for i in dict.fromkeys(tri.tolist()) if i not in used]
        if len(used) + len(new) > MAX_SPRITE_VERTICES or len(tris) == MAX_SPRITE_TRIANGLES:
            flush()
            tris, used = [], {}
            new = list(dict.fromkeys(tri.tolist()))
        for i in new:
            used[i] = len(used)
        tris.append(tuple(int(i) for i in tri))
    flush()
    return out


def loop_gap(frames: list) -> float:
    """The last frame's largest vertex distance from frame 0, in median steps between consecutive
    frames (each step its largest vertex move)."""
    track = np.stack([np.concatenate(f) for f in frames])
    steps = np.linalg.norm(np.diff(track, axis=0), axis=2).max(axis=1)
    gap = np.linalg.norm(track[-1] - track[0], axis=1).max()
    return float(gap / np.median(steps))


def loop_closes(frames: list) -> bool:
    """True when the last frame already is (or nearly is) frame 0: loop_gap <= LOOP_GAP."""
    return loop_gap(frames) <= LOOP_GAP


def _resample(frames: list, fps: float, wrap: bool) -> list:
    """Frames (lists of per-part arrays) at `fps` -> frames at GAME_FPS (see the module docstring)."""
    arr = [np.stack([f[p] for f in frames]) for p in range(len(frames[0]))]
    closed = wrap and loop_closes(frames)
    n = len(frames)
    if wrap:
        cycle = n - 1 if closed else n
        m = max(1, int(round(cycle / fps * GAME_FPS)))
        times = np.arange(m) * cycle / m
    else:
        m = max(1, int(round((n - 1) / fps * GAME_FPS)))
        times = np.arange(m + 1) * (n - 1) / m
    out = []
    for t in times:
        i = int(np.floor(t + 1e-9))
        u = t - i
        if u < 1e-9:
            out.append([a[i % n] for a in arr])
        else:
            j = (i + 1) % cycle if wrap else min(i + 1, n - 1)
            out.append([a[i] * (1 - u) + a[j] * u for a in arr])
    if wrap:
        out.append([np.array(p) for p in out[0]])
    return out


def _bbox_unit(p: np.ndarray) -> np.ndarray:
    lo, hi = p.min(0), p.max(0)
    return (p - lo) / np.maximum(hi - lo, 1e-12)


def _remap_controls(plan: MolynxPlan, bile_idle: Kmf, rest: list) -> list:
    theirs = evaluate_anim_frame(bile_idle.anim, 0)
    pairs = np.frombuffer(bile_idle.anim.controls, "<u4").reshape(-1, 2)
    flat_theirs = _bbox_unit(np.concatenate(theirs))
    starts = np.cumsum([0] + [len(p) for p in theirs])
    ours = _bbox_unit(np.concatenate(rest))
    our_starts = np.cumsum([0] + [len(p) for p in rest])
    body = [s for s, lay in enumerate(plan.sprites) if lay.texture != MACE_TEXTURE]
    cand = np.concatenate([np.arange(our_starts[s], our_starts[s + 1]) for s in body])
    out = []
    for ref, ident in pairs:
        sprite, vertex = int(ref) & 0xFFFF, int(ref) >> 16
        point = flat_theirs[starts[sprite] + vertex]
        best = int(cand[np.argmin(np.linalg.norm(ours[cand] - point, axis=1))])
        s = int(np.searchsorted(our_starts, best, side="right") - 1)
        out.append((s, best - int(our_starts[s]), int(ident)))
    return out


def _crop_uvs(parts: list, images: dict) -> tuple[list, dict]:
    """The parts with the UVs of each CROP_TO_UV texture's part moved into the box of its image they
    use (their used UV range, UV_MARGIN round it, out to whole pixels of the image, within 0..1),
    and {texture: that box}."""
    out, crops = [], {}
    for p in parts:
        name = TEXTURE_NAMES[p["texture"]]
        if name in CROP_TO_UV:
            uv = np.asarray(p["uv"], float)
            used = uv[np.unique(np.asarray(p["indices"], int))]
            size = np.array(images[p["texture"]].size, float)
            lo = np.clip(np.floor((used.min(0) - UV_MARGIN) * size) / size, 0, 1)
            hi = np.clip(np.ceil((used.max(0) + UV_MARGIN) * size) / size, 0, 1)
            crops[name] = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
            p = {**p, "uv": ((uv - lo) / (hi - lo)).astype(np.float32)}
        out.append(p)
    return out, crops


def _mm_factor(points: np.ndarray, uv: np.ndarray, tris) -> float:
    t = np.asarray(tris)
    model = sum(np.linalg.norm(points[t[:, a]] - points[t[:, b]], axis=1).sum()
                for a, b in ((0, 1), (1, 2), (2, 0)))
    tex = sum(np.linalg.norm(uv[t[:, a]] - uv[t[:, b]], axis=1).sum()
              for a, b in ((0, 1), (1, 2), (2, 0)))
    return float(np.float32(MM_PER_RATIO * tex / model))


# ---------------------------------------------------------------- props and anchors (review round, 2026-11-07)
# Some game animation files carry an object with the creature: every eat file the chicken it eats
# (Bile_Demon-Eat: materials chickenchicken, Feet+Beakchicken), every bed(sleeping) file the
# creature's bed (the game hides the lair while the creature sleeps: Molynx's sleep file had no bed,
# so his bed vanished). Torture and bed files also place the creature where the room's object is
# (the wheel, the bed). PROPS: clip -> (donor file, texture names of the donor sprites to carry,
# how the prop moves): "hand" follows his free hand (molynx_design.HAND) the way it follows the
# donor's hand; "fixed" keeps the donor's own positions. ANCHORS: clip -> (donor file, how he is
# placed): "centre" puts his torso's centre on the donor creature's; "bed" centres his whole body
# over the bed sprite and lays his lowest point on the bed's top, BED_SINK of the bed's height in.
PROPS = {"eating": ("Knight-Eat", ("chicken",), "hand"),
         "sleeping": ("BileDemon-bed(sleeping)", ("Creature Bed_Bile Demon1",), "fixed")}
# ANCHORS: clips placed on the Bile Demon's own file of that slot. "centre": his torso on the Bile
# Demon's body, frame by frame (the torture wheel); "bed": in the bed (see BED_SINK); "average": the
# whole clip moved across the floor once, so his body's average place is the Bile Demon's (praying:
# the game stands a creature at the temple's edge and the file says how far in it bows; his own bow
# reached twice as far in, into the temple).
ANCHORS = {"torture": ("Bil-Torture", "centre"), "sleeping": ("BileDemon-bed(sleeping)", "bed"),
           "praying": ("Bile_Demon-Pray", "average")}
BED_SINK = 0.85          # his lowest point, as a fraction of the way from the bed's top to the floor: the Bile Demon sits in its bed with its bottom at the floor (z -0.03 of -0.24)
HAND_GRAB_DISTANCE = 0.06   # the donor's chicken counts as held once this close to the donor's hand
# MIRRORED_ARM: clips where both his arms are spread the same way (strapped to the torture wheel).
# His two arms are not built alike (the mace arm is bent up to hold the mace), so the mace arm is
# folded away (its vertices onto the middle of the ring where it meets his shoulder) and the file
# carries a mirror copy of his free arm instead: extra Molynx_Body sprites, mirrored across his
# middle in his torso's own frame each frame. ARM_WEIGHT: a vertex is the arm's when its arm and
# forearm skin weights add up to more than this.
MIRRORED_ARM = {"torture"}
ARM_WEIGHT = 0.5
FOLD_INWARD = 0.6


@dataclass
class Prop:
    materials: list          # donor Materials, or texture names of his own materials
    sprites: list            # donor Sprites (uv, normals, triangles)
    frames: list             # frames[f][k] = (n, 3) KMF positions of prop sprite k


def _donor_anim(name: str, meshes_dir: Path = MESHES_DIR):
    model = parse_kmf(game_mesh_path(name, meshes_dir).read_bytes())
    frames = [evaluate_anim_frame(model.anim, f) for f in range(model.anim.frame_count)]
    return model, frames


def _carried(model, names) -> list[int]:
    want = {n.lower() for n in names}
    return [i for i, s in enumerate(model.anim.sprites)
            if any(t.lower() in want for t in model.materials[s.material].textures)]


def _part_vertices(plan: "MolynxPlan", part: int, ids) -> list[tuple[int, int]]:
    """(sprite, index) in plan.sprites of each vertex `ids` of part `part`."""
    where = {}
    for s, lay in enumerate(plan.sprites):
        if lay.part == part:
            for i, v in enumerate(lay.vertices):
                where.setdefault(int(v), (s, i))
    return [where[int(v)] for v in ids if int(v) in where]


def _gather(frame: list, at: list) -> np.ndarray:
    return np.array([frame[s][i] for s, i in at], float)


def _kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, _S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T, ca, cb


def _torso_ids(rest: np.ndarray, unit_dir: Path = BEHEMOTH_DIR) -> np.ndarray:
    """His torso's vertices: labelled torso, not the mace and not skinned mostly to an arm (the
    shoulder edges move with the arms; the mace arm is folded away on the torture wheel)."""
    from scripts.behemoth_rig import CLUB_VERTICES, behemoth_labels
    lab = behemoth_labels(rest)
    ids = np.nonzero(lab == 0)[0]
    arms = np.r_[_arm_ids(unit_dir, "N"), _arm_ids(unit_dir, "P")]
    return np.setdiff1d(ids[ids >= CLUB_VERTICES], arms)


def _donor_index(i: int, n: int, nd: int) -> int:
    return int(round(i * (nd - 1) / max(n - 1, 1)))


def _anchor(plan: "MolynxPlan", clip: "Clip", donor: str, how: str, meshes_dir: Path) -> "Clip":
    model, dframes = _donor_anim(donor, meshes_dir)
    bed = set(_carried(model, ("Creature Bed_Bile Demon1",)))
    rest = np.asarray(plan.source[0]["stand"]["frames"][0][0], float)
    at = _part_vertices(plan, 0, _torso_ids(rest))
    out = []
    if how == "average":    # across the floor only (x, y); z is up and down in KMF space
        theirs = np.mean([np.concatenate(f).mean(0) for f in dframes], axis=0)
        mine = np.mean([np.concatenate(f).mean(0) for f in clip.frames], axis=0)
        shift = np.array([theirs[0] - mine[0], theirs[1] - mine[1], 0.0])
        return Clip([[p + shift for p in frame] for frame in clip.frames], clip.wrap)
    for i, frame in enumerate(clip.frames):
        dfr = dframes[_donor_index(i, len(clip.frames), len(dframes))]
        if how == "centre":
            d = np.concatenate([p for k, p in enumerate(dfr) if k not in bed])
            shift = d.mean(0) - _gather(frame, at).mean(0)
        else:   # bed: z is down in KMF space (the lowest point has the largest z)
            b = np.concatenate([dfr[k] for k in bed])
            mine = np.concatenate(frame)
            top, floor = b[:, 2].min(), b[:, 2].max()
            centre = (b[:, :2].min(0) + b[:, :2].max(0)) / 2
            lay = top + BED_SINK * (floor - top)
            shift = np.array([centre[0] - (mine[:, 0].min() + mine[:, 0].max()) / 2,
                              centre[1] - (mine[:, 1].min() + mine[:, 1].max()) / 2,
                              lay - mine[:, 2].max()])
        out.append([p + shift for p in frame])
    if clip.wrap:   # a loop still ends on a copy of its first frame
        out[-1] = [p.copy() for p in out[0]]
    return Clip(out, clip.wrap)


def _prop(plan: "MolynxPlan", clip: "Clip", donor: str, names, how: str, meshes_dir: Path) -> Prop:
    from scripts.molynx_design import HAND
    model, dframes = _donor_anim(donor, meshes_dir)
    keep = _carried(model, names)
    if not keep:
        raise ValueError(f"{donor} has no sprite with textures {names}")
    n, nd = len(clip.frames), len(dframes)
    props = [[dframes[_donor_index(i, n, nd)][k] for k in keep] for i in range(n)]
    if how == "hand":
        body = [k for k in range(len(model.anim.sprites)) if k not in keep]
        cen = [np.concatenate(p).mean(0) for p in props]
        # the donor's holding hand: its body vertices nearest the chicken where the chicken is highest
        top = int(np.argmin([c[2] for c in cen]))
        dtop = dframes[_donor_index(top, n, nd)]
        allb = np.concatenate([dtop[k] for k in body])
        which = np.argsort(np.linalg.norm(allb - cen[top], axis=1))[:10]
        starts = np.cumsum([0] + [len(dtop[k]) for k in body])
        dat = [(body[int(np.searchsorted(starts, w, side="right") - 1)],
                int(w - starts[int(np.searchsorted(starts, w, side="right") - 1)])) for w in which]
        mat = _part_vertices(plan, 0, HAND)
        held = [np.linalg.norm(cen[i] - _gather(dframes[_donor_index(i, n, nd)], dat).mean(0)) < HAND_GRAB_DISTANCE
                for i in range(n)]
        grab = held.index(True) if True in held else n
        ref_d = _gather(dframes[_donor_index(grab if grab < n else top, n, nd)], dat)
        ref_m = _gather(clip.frames[grab if grab < n else top], mat)
        out = []
        for i in range(n):
            dh = _gather(dframes[_donor_index(i, n, nd)], dat)
            mh = _gather(clip.frames[i], mat)
            if i >= grab:
                Rd, cd, _ = _kabsch(ref_d, dh)
                Rm, cm, _ = _kabsch(ref_m, mh)
                # the prop in the donor hand's frame, carried by his hand
                out.append([(p - dh.mean(0)) @ Rd @ Rm.T + mh.mean(0) for p in props[i]])
            else:   # on the floor: moved under the place where his hand will take it
                shift = ref_m.mean(0) - ref_d.mean(0)
                shift[2] = 0.0
                out.append([p + shift for p in props[i]])
        props = out
    return Prop([model.materials[model.anim.sprites[k].material] for k in keep],
                [model.anim.sprites[k] for k in keep], props)


def _arm_ids(unit_dir: Path, side: str) -> np.ndarray:
    """The vertices of one arm (N the mace arm, P the free arm): skinned mostly to it."""
    from scripts.behemoth_rig import CLUB_VERTICES, behemoth_rig
    rig = behemoth_rig(Path(unit_dir))
    w = np.asarray(rig.weights)
    b = list(rig.bones)
    ids = np.nonzero(w[:, b.index("arm" + side)] + w[:, b.index("fore" + side)] > ARM_WEIGHT)[0]
    return ids[ids >= CLUB_VERTICES]


def _mirror_arm(plan: "MolynxPlan", clip: "Clip", unit_dir: Path) -> tuple["Clip", Prop]:
    """(the clip with his mace arm folded away, the mirror copy of his free arm), see MIRRORED_ARM."""
    rest_c = np.asarray(plan.source[0]["stand"]["frames"][0][0], float)
    torso = _part_vertices(plan, 0, _torso_ids(rest_c))
    tris = np.asarray(plan.parts[0]["indices"], int).reshape(-1, 3)
    free, mace = set(_arm_ids(unit_dir, "P").tolist()), set(_arm_ids(unit_dir, "N").tolist())
    arm_tris = [t for t in tris if all(int(v) in free for v in t)]
    used = sorted({int(v) for t in arm_tris for v in t})
    layouts = _split(0, [(a, c, b) for a, b, c in arm_tris], BODY_TEXTURE)   # a mirror turns the winding
    def ring(arm):   # the arm's vertices on its edge with his shoulder
        return sorted({int(v) for t in tris if any(int(v) in arm for v in t)
                       and not all(int(v) in arm for v in t) for v in t if int(v) in arm})
    fold = _part_vertices(plan, 0, sorted(mace))
    at_ring, at_used = _part_vertices(plan, 0, ring(mace)), _part_vertices(plan, 0, used)
    if len(at_used) != len(used):
        raise ValueError("a vertex of his free arm's triangles is in no sprite")
    rest_k = plan.poses["PoseFrame"]
    rest_t = _gather(rest_k, torso)
    frames, props = [], []
    for frame in clip.frames:
        R, ca, cb = _kabsch(rest_t, _gather(frame, torso))     # rest -> this frame
        # the mirror is upright (z is down in KMF space): through his torso's centre, across the
        # way his torso faces, so the copy is level with the free arm as seen from the front
        normal = R @ np.array([1.0, 0.0, 0.0])
        normal[2] = 0.0
        normal /= np.linalg.norm(normal)
        arm = _gather(frame, at_used)
        mirrored = dict(zip(used, arm - 2 * np.outer((arm - cb) @ normal, normal)))
        props.append([np.array([mirrored[int(v)] for v in lay.vertices]) for lay in layouts])
        # folded inside his shoulder, part of the way from the edge's middle to his torso's centre,
        # so the triangles that join the edge to it stay inside him
        centre = _gather(frame, at_ring).mean(0)
        centre = centre + FOLD_INWARD * (_gather(frame, torso).mean(0) - centre)
        out = [p.copy() for p in frame]
        for s_, i in fold:
            out[s_][i] = centre
        frames.append(out)
    uv = np.asarray(plan.parts[0]["uv"], float)
    acc = {}
    for lay, p in zip(layouts, props[0]):
        t = np.asarray(lay.triangles)
        face = -np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        for k in range(3):
            for row, v in zip(t[:, k], face):
                key = int(lay.vertices[row])
                acc[key] = acc.get(key, 0) + v
    sprites = []
    for lay in layouts:
        q = np.clip(np.rint(uv[lay.vertices] * 32768), 0, 65535).astype(int)
        verts = []
        for i, v in enumerate(lay.vertices):
            n = np.asarray(acc.get(int(v), np.zeros(3)), float)
            n = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-12 else np.array([0.0, 0.0, -1.0])
            verts.append(Vertex((int(q[i, 0]), int(q[i, 1])), tuple(float(np.float32(c)) for c in n)))
        mm = _mm_factor(_gather(rest_k, _part_vertices(plan, 0, lay.vertices)), uv[lay.vertices],
                        np.asarray(lay.triangles))
        sprites.append(Sprite(0, [list(lay.triangles) for _ in range(LODS)], verts, mm))
    return Clip(frames, clip.wrap), Prop([BODY_TEXTURE] * len(sprites), sprites, props)


def plan_molynx(height_ratio: float = DESIGN_HEIGHT_RATIO, source=None, *, unit_dir: Path = BEHEMOTH_DIR,
                meshes_dir: Path = MESHES_DIR, editor_dir: Path = EDITOR_DIR,
                game_dir: Path = GAME_DIR) -> MolynxPlan:
    """Everything the export writes, in KMF space, before any file is written."""
    if source is None:
        source = build_source(unit_dir, meshes_dir, texture_cache(game_dir))
    data, _ = source
    parts = data["stand"]["parts"]
    for name in CLIPS:
        if [p["texture"] for p in data[name]["parts"]] != [p["texture"] for p in parts] or any(
                not np.array_equal(a["indices"], b["indices"]) for a, b in zip(data[name]["parts"], parts)):
            raise ValueError(f"clip {name} does not share the stand clip's parts")
    parts, crops = _crop_uvs(parts, source[1])
    sprites = [lay for i, p in enumerate(parts) for lay in _split(i, p["indices"], TEXTURE_NAMES[p["texture"]])]

    bile = bile_demon_record(editor_dir)
    idle = parse_kmf(game_mesh_path(bile.anim_name("idle1"), meshes_dir).read_bytes())
    their = np.concatenate(evaluate_anim_frame(idle.anim, 0))
    target = float(their[:, 2].max() - their[:, 2].min()) * height_ratio

    plan = MolynxPlan(source, height_ratio, parts, sprites, 1.0, np.zeros(3), target, crops=crops)
    raw = plan.to_sprites(plan.to_kmf(data["stand"]["frames"][0]))
    allp = np.concatenate(raw)
    plan.scale = target / float(allp[:, 2].max() - allp[:, 2].min())
    body = np.concatenate([p for p, s in zip(raw, sprites) if s.texture == BODY_TEXTURE]) * plan.scale
    plan.offset = np.array([-body[:, 0].mean(), -body[:, 1].mean(), -allp[:, 2].max() * plan.scale])

    if {name: bool(data[name]["wrap"]) for name in CLIPS} != SOURCE_WRAP:
        raise ValueError("the clip library's loop flags are not molynx_record.SOURCE_WRAP")
    far = {name: loop_gap(data[name]["frames"]) for name in CLIPS if not data[name]["wrap"]}
    far = {name: gap for name, gap in far.items() if gap > MAX_CLOSE_GAP}
    if set(far) != set(ONE_WAY_GAPS) or any(abs(far[n] - ONE_WAY_GAPS[n]) > 0.01 for n in far):
        raise ValueError(f"the one-way clips measure {far}, not molynx_record.ONE_WAY_GAPS")
    for name in CLIPS:
        clip = data[name]
        frames = _resample(clip["frames"], clip["fps"], clip["wrap"])
        plan.clips[name] = Clip([plan.to_sprites(plan.to_kmf(f)) for f in frames], clip["wrap"])
        plan.info[name] = {"frames": len(frames), "fps": GAME_FPS, "wrap": clip["wrap"]}
    for name, (donor, how) in ANCHORS.items():
        plan.clips[name] = _anchor(plan, plan.clips[name], donor, how, Path(meshes_dir))
    for name, (donor, names, how) in PROPS.items():
        plan.props[name] = _prop(plan, plan.clips[name], donor, names, how, Path(meshes_dir))
    for stem, (name, index) in POSE_SOURCES.items():
        plan.poses[stem] = plan.to_sprites(plan.to_kmf(data[name]["frames"][index]))
    for name in MIRRORED_ARM:
        plan.clips[name], plan.props[name] = _mirror_arm(plan, plan.clips[name], Path(unit_dir))

    rest = plan.poses["PoseFrame"]
    plan.controls = _remap_controls(plan, idle, rest)
    donor = idle.materials[idle.anim.sprites[0].material]  # FrontBDemonFrnt, the body's front
    textures = list(dict.fromkeys(s.texture for s in sprites))
    plan.materials = [Material(PART_NAMES[t] + t, [t], donor.flags, donor.brightness,
                               donor.shininess, donor.environment_map) for t in textures]
    plan.mm_factors = [_mm_factor(p, np.asarray(parts[s.part]["uv"], float)[s.vertices], s.triangles)
                       for p, s in zip(rest, sprites)]
    return plan


# ---------------------------------------------------------------- KMF files

def _rows(frames: list) -> np.ndarray:
    """Row (shared position track) of every sprite vertex, numbered in order of first appearance."""
    tracks = np.stack([np.concatenate(f) for f in frames], axis=1)  # (vertices, frames, 3)
    flat = np.ascontiguousarray(tracks.reshape(len(tracks), -1))
    _, first, inverse = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse.reshape(-1)]


def _normals(plan: MolynxPlan, frame: list, rows: np.ndarray) -> list:
    """Unit normals per sprite vertex: area-weighted -(b-a)x(c-a), shared by a part's vertices at
    one position (one row)."""
    starts = np.cumsum([0] + [len(p) for p in frame])
    acc = {}
    for s, (lay, pts) in enumerate(zip(plan.sprites, frame)):
        t = np.asarray(lay.triangles)
        face = -np.cross(pts[t[:, 1]] - pts[t[:, 0]], pts[t[:, 2]] - pts[t[:, 0]])
        vert = np.zeros_like(pts)
        for k in range(3):
            np.add.at(vert, t[:, k], face)
        for i in range(len(pts)):
            key = (lay.part, int(rows[starts[s] + i]))  # each triangle is in one sprite only
            acc[key] = acc.get(key, 0) + vert[i]
    out = []
    for s, (lay, pts) in enumerate(zip(plan.sprites, frame)):
        n = np.array([acc[(lay.part, int(rows[starts[s] + i]))] for i in range(len(pts))])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(length > 1e-12, n / np.maximum(length, 1e-12), [0.0, 0.0, -1.0])
        out.append(n)
    return out


def _sprites(plan: MolynxPlan, frame: list, rows: np.ndarray, mesh: bool) -> list[Sprite]:
    normals = _normals(plan, frame, rows)
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


def _check_rows(rows: np.ndarray, frame: list) -> None:
    pts = np.concatenate(frame)
    for r in np.unique(rows):
        same = pts[rows == r]
        if np.abs(same - same[0]).max() > 0:
            raise AssertionError("vertices sharing a row differ")


def file_clip(plan: MolynxPlan, stem: str, wrap: bool) -> Clip:
    """The frames of file `stem` (a clip or its _once / _loop variant) with loop flag `wrap`:
    the clip itself, its loop opened (same frames) or its source resampled as a loop (see the
    module docstring)."""
    clip, forced = clip_of(stem)
    if clip not in plan.clips:
        raise ValueError(f"{stem} is not a clip or a clip variant")
    if forced is not None and forced != wrap:
        raise ValueError(f"{stem} must {'' if forced else 'not '}loop")
    base = plan.clips[clip]
    if base.wrap == wrap:
        return base
    if not wrap:
        return Clip(base.frames, False)
    data, _ = plan.source
    frames = _resample(data[clip]["frames"], data[clip]["fps"], True)
    return Clip([plan.to_sprites(plan.to_kmf(f)) for f in frames], True)


def _file_wraps(plan: MolynxPlan, wraps: dict) -> dict:
    """{file stem: wrap} of every animation file to write: each clip (its own flag unless `wraps`
    names it) and the variants `wraps` names. ValueError for anything else."""
    for stem, wrap in wraps.items():
        clip, forced = clip_of(stem)
        if clip not in CLIPS:
            raise ValueError(f"wraps names {stem!r}, which is not a clip or a clip variant")
        if not isinstance(wrap, bool) or (forced is not None and forced != wrap):
            raise ValueError(f"wraps gives {stem!r} the loop flag {wrap!r}")
        if wrap and clip in ONE_WAY_GAPS:
            raise ValueError(f"{clip} is one-way (gap {ONE_WAY_GAPS[clip]} median steps): "
                             f"it is never closed into a loop")
    files = {clip: plan.clips[clip].wrap for clip in CLIPS}
    files.update(wraps)
    return files


def clip_kmf(plan: MolynxPlan, stem: str, reduce: bool = True, wrap: bool | None = None) -> Kmf:
    """The animation file `stem` (a clip, or a variant with `wrap`); `wrap` None = the clip's own."""
    clip = plan.clips[stem] if wrap is None else file_clip(plan, stem, wrap)
    base, _forced = clip_of(stem)
    prop = plan.props.get(base)
    materials = list(plan.materials)
    if prop is not None:
        if len(prop.frames) != len(clip.frames):
            raise ValueError(f"{stem}: its prop has {len(prop.frames)} frames, the clip {len(clip.frames)}")
        pframes = list(prop.frames)
        if clip.wrap:   # a loop ends on a copy of its first frame, the prop's too
            pframes[-1] = [p.copy() for p in pframes[0]]
        clip = Clip([list(f) + list(p) for f, p in zip(clip.frames, pframes)], clip.wrap)
    rows = _rows(clip.frames)
    sprites = _sprites(plan, clip.frames[0][:len(plan.sprites)], rows, mesh=False)
    if prop is not None:
        start = sum(len(p) for p in clip.frames[0][:len(plan.sprites)])
        for k, sp in enumerate(prop.sprites):
            verts = [Vertex(v.uv, v.normal, itab_index=int(rows[start + i])) for i, v in enumerate(sp.vertices)]
            mat = prop.materials[k]
            if isinstance(mat, str):     # one of his own textures: his own material
                index = next(i for i, m in enumerate(plan.materials) if m.textures[0] == mat)
            else:
                index = len(materials)
                materials.append(mat)
            sprites.append(Sprite(index, [list(t) for t in sp.triangles], verts, sp.mm_factor))
            start += len(sp.vertices)
    controls = b"".join(np.array([(v << 16) | s, ident], "<u4").tobytes()
                        for s, v, ident in plan.controls)
    anim = None
    for tolerance in KEY_TOLERANCES if reduce else (1.0,):
        anim = encode_anim(PREFIX + stem, sprites, clip.frames, clip.wrap, controls, reduce=reduce,
                           tolerance=tolerance)
        if len(anim.geom_xyz) - 1 <= GAME_MAX_KEYS:
            break
    else:
        if reduce:
            anim = encode_anim(PREFIX + stem, sprites, clip.frames, clip.wrap, controls, reduce=True)
    return Kmf(VERSION, "anim", materials, anim=anim)


def pose_kmf(plan: MolynxPlan, stem: str) -> Kmf:
    frame = plan.poses[stem]
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
                _sprites(plan, frame, rows, mesh=True), [tuple(map(float, g)) for g in geometry],
                controls)
    return Kmf(VERSION, "mesh", list(plan.materials), mesh=mesh)


# ---------------------------------------------------------------- textures and portraits

def _texture(image: Image.Image, size, crop=None) -> Image.Image:
    """`image` (cut to `crop`, fractions (u0, v0, u1, v1), when given) at `size` (one side, or
    (width, height)), Lanczos, RGB."""
    w, h = (size, size) if isinstance(size, int) else size
    for n in (w, h):
        if n & (n - 1) or not 0 < n <= 256:
            raise ValueError(f"texture size {n} is not a power of two up to 256")
    image = image.convert("RGB")
    if crop is not None:
        W, H = image.size
        u0, v0, u1, v1 = crop
        box = (round(u0 * W), round(v0 * H), round(u1 * W), round(v1 * H))
        if any(abs(b - c) > 1e-6 for b, c in zip(box, (u0 * W, v0 * H, u1 * W, v1 * H))):
            raise ValueError(f"crop {crop} is not on the {W}x{H} image's pixel grid")
        image = image.crop(box)
    return image if image.size == (w, h) else image.resize((w, h), Image.LANCZOS)


def _inner_picture(image: Image.Image) -> Image.Image:
    """The user's picture without its metal frame: the rows and columns where most pixels are
    coloured (the frame is grey and black)."""
    a = np.asarray(image.convert("RGB")).astype(int)
    coloured = (a.max(2) - a.min(2)) > 30
    h, w = coloured.shape
    cols = np.flatnonzero(coloured[h // 5: 4 * h // 5].mean(0) > 0.5)
    rows = np.flatnonzero(coloured[:, w // 5: 4 * w // 5].mean(1) > 0.5)
    return image.convert("RGB").crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def _crop_aspect(image: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = image.size
    if iw * h > ih * w:
        cw = round(ih * w / h)
        x = (iw - cw) // 2
        return image.crop((x, 0, x + cw, ih))
    ch = round(iw * h / w)
    y = (ih - ch) // 2
    return image.crop((0, y, iw, y + ch))


def _framed(frame: Image.Image, picture: Image.Image) -> Image.Image:
    out = frame.convert("RGB").copy()
    w, h = out.size
    out.paste(picture.resize((w - 2 * FRAME, h - 2 * FRAME), Image.LANCZOS), (FRAME, FRAME))
    return out


def make_portraits(src: Path, bile: Creature, cache, icon_src: Path | None = None
                   ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """(molynx-bg, molynx, bile_demon-bg, bile_demon) images. With `icon_src` both pictures are
    unframed renders (the redesign): the panel is `src` cut to the game window's aspect about its
    centre, the icon is `icon_src`; without it `src` is the user's framed picture (see the module
    docstring)."""
    bg_frame = _cache_image(cache, bile.resource_name("portraitResource"))
    icon_frame = _cache_image(cache, bile.resource_name("icon1Resource"))
    if icon_src is not None:
        picture = Image.open(src).convert("RGB")
        bg = _framed(bg_frame, _crop_aspect(picture, bg_frame.width - 2 * FRAME,
                                            bg_frame.height - 2 * FRAME))
        icon = _framed(icon_frame, _crop_aspect(Image.open(icon_src).convert("RGB"),
                                                icon_frame.width - 2 * FRAME,
                                                icon_frame.height - 2 * FRAME))
        return bg, icon, bg_frame, icon_frame
    user = Image.open(src).convert("RGB")
    inner = _inner_picture(user)
    bg = _framed(bg_frame, _crop_aspect(inner, bg_frame.width - 2 * FRAME,
                                        bg_frame.height - 2 * FRAME))
    cx, cy, side = FACE_BOX
    W, H = user.size
    s = side * W
    face = user.crop((round(cx * W - s / 2), round(cy * H - s / 2),
                      round(cx * W + s / 2), round(cy * H + s / 2)))
    icon = _framed(icon_frame, face)
    return bg, icon, bg_frame, icon_frame


def _preview(images: list[tuple[str, Image.Image]]) -> Image.Image:
    k, pad, label = PREVIEW_SCALE, 12, 18
    tiles = [(t, im.resize((im.width * k, im.height * k), Image.NEAREST)) for t, im in images]
    w = sum(im.width for _, im in tiles) + pad * (len(tiles) + 1)
    h = max(im.height for _, im in tiles) + 2 * pad + label
    sheet = Image.new("RGB", (w, h), (32, 32, 36))
    draw = ImageDraw.Draw(sheet)
    x = pad
    for title, im in tiles:
        draw.text((x, pad // 2), title, fill=(230, 230, 230))
        sheet.paste(im, (x, pad + label))
        x += im.width + pad
    return sheet


# ---------------------------------------------------------------- export

def export_molynx_assets(out_dir: Path, height_ratio: float = DESIGN_HEIGHT_RATIO,
                         portrait_src: Path | None = DESIGN_PORTRAIT, *, portrait_icon: Path | None = DESIGN_ICON,
                         plan: MolynxPlan | None = None,
                         body_size: int = MAX_TEXTURE, reduce: bool = True,
                         editor_dir: Path = EDITOR_DIR, game_dir: Path = GAME_DIR,
                         meshes_dir: Path = MESHES_DIR, wraps: dict | None = None) -> dict:
    """Write Molynx's .kmf files (the 25 clips, the variants `wraps` names and the 5 poses), 2
    textures and (with `portrait_src`) 2 portraits plus a preview into `out_dir`, which must not
    exist or be empty (nothing is ever deleted). `out_dir`, the portrait source, the plan's height
    ratio and `wraps` are checked before the first file is written; an encoding error can still
    stop the export part-way. `wraps` ({file stem: loop flag}, default `required_wraps` of the
    editor's Bile Demon) sets the loop flag of the files it names (see the module docstring).
    `reduce=True` (the default) keeps only the keyframes each vertex needs; `reduce=False` keys
    every frame.
    -> {"meshes": {stem: path}, "textures": {name: path}, "portraits": {name: path},
        "preview": path | None, "info": {file stem: {frames, fps, wrap}} (final files),
        "wrap_exceptions": [molynx_record.WrapException] (default wraps only, else []),
        "scale": float, "target_height": float}"""
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"{out_dir} exists and is not an empty folder")
    for src in (portrait_src, portrait_icon if portrait_src is not None else None):
        if src is not None and not Path(src).is_file():
            raise FileNotFoundError(f"portrait source {src} not found")
    if plan is None:
        plan = plan_molynx(height_ratio, meshes_dir=meshes_dir, editor_dir=editor_dir,
                           game_dir=game_dir)
    elif plan.height_ratio != height_ratio:
        raise ValueError(f"the plan was made for height_ratio {plan.height_ratio}, "
                         f"not {height_ratio}")
    exceptions = []
    if wraps is None:
        files_plan = molynx_files(bile_demon_record(editor_dir), meshes_dir)
        wraps, exceptions = files_plan.wraps, list(files_plan.wrap_exceptions)
    files = _file_wraps(plan, wraps)
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes, info = {}, {}
    for stem, wrap in files.items():
        model = clip_kmf(plan, stem, reduce, wrap)
        meshes[stem] = out_dir / f"{PREFIX}{stem}.kmf"
        meshes[stem].write_bytes(write_kmf(model))
        info[stem] = {"frames": model.anim.frame_count, "fps": GAME_FPS, "wrap": wrap}
    for stem in POSE_STEMS:
        meshes[stem] = out_dir / f"{PREFIX}{stem}.kmf"
        meshes[stem].write_bytes(write_kmf(pose_kmf(plan, stem)))
    _, images = plan.source
    sizes = {**TEXTURE_SIZES, BODY_TEXTURE: (body_size, body_size)}
    textures = {}
    used = {s.texture for s in plan.sprites}
    for key, name in TEXTURE_NAMES.items():
        if name not in used:
            continue
        textures[name] = out_dir / f"{name}.png"
        _texture(images[key], sizes[name], plan.crops.get(name)).save(textures[name])
    portraits, preview = {}, None
    if portrait_src is not None:
        cache = texture_cache(game_dir)
        bile = bile_demon_record(editor_dir)
        bg, icon, bile_bg, bile_icon = make_portraits(
            Path(portrait_src), bile, cache, None if portrait_icon is None else Path(portrait_icon))
        for name, image in (("molynx-bg", bg), ("molynx", icon)):
            portraits[name] = out_dir / f"{name}.png"
            image.save(portraits[name])
        preview = out_dir / "portraits_preview.png"
        _preview([("Molynx -bg", bg), ("Molynx", icon), ("Bile Demon -bg", bile_bg),
                  ("Bile Demon", bile_icon)]).save(preview)
    return {"meshes": meshes, "textures": textures, "portraits": portraits, "preview": preview,
            "info": info, "wrap_exceptions": exceptions, "scale": plan.scale, "target_height": plan.target_height}
