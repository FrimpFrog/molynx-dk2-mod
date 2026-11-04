"""Molynx's game files: his model and clips as DK2 .kmf files, his two textures and two portraits.

`export_molynx_assets(out_dir)` rebuilds the finished character from the repo's code (the MegaGlest
Behemoth dressed by custom_character.build_character, with the 21 DK2 clips retargeted by
behemoth_anims.build_all), turns every clip into a game animation and writes

  Molynx-<clip>.kmf      the 25 clips of molynx_record.CLIPS (animation files)
  Molynx-<clip>_once.kmf / _loop.kmf  variants the slots need (Bile Demon: walk_once, getting_hit_loop)
  Molynx-PoseFrame.kmf   `stand` frame 0          (static mesh files, like the Bile Demon's poses)
  Molynx-DiePose.kmf     `die` last frame
  Molynx-DraggedPose.kmf `being_picked_up` frame 0
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

Animations. Every frame is keyed (encode_anim reduce=False), as in 17 of the 21 Bile Demon files.
A KMF has no rate: the creature record gives each slot (frames, fps) and the Bile Demon's slots all
say 30 fps (sleep 15). Every clip is resampled to 30 fps by linear interpolation of the vertices,
over the clip's own duration ((n - 1) / fps; a looping clip n / fps, or (n - 1) / fps when its last
frame already closes the loop), stretched by at most half a frame so the first and last source
frames are kept exactly. Clips at 30 fps are unchanged. A looping (`wrap`) clip ends on a copy of
frame 0, as all 209 looping game files do. A loop "already closes" (`loop_closes`) when its last
frame is within LOOP_GAP = 0.3 x the median frame-to-frame step of frame 0: that last frame is
REPLACED by the copy (frame count unchanged; the 9 retargeted loops that close exactly and
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
    MAX_CLOSE_GAP = 10 (getting_hit 5.8 -> getting_hit_loop). The one-way clips above it
    (molynx_record.ONE_WAY_GAPS: sleeping 37.1, waking_up 25.3) are never closed; plan_molynx
    checks that list against the source, and a slot that wanted one looped is reported in
    `wrap_exceptions` (the real Bile Demon: sleeping in the sleep slot).
The returned `info` gives the FINAL frames, fps and wrap of every animation file written; the
stage-2 record turns it into each slot's frames (file frames - 1 for looping files, as the Bile
Demon's record does: walk 41 -> 40; = file frames otherwise: idle1 190).

Size (a known risk, kept on purpose). Fully keyed files are large: the biggest GEOM is 136,344
keyframe positions (Molynx-victory_celebration: 552 rows x 247, its 246 frames plus the block
seam), against 34,816 in the largest game animation (Mis-Idle1), and DraggedPose has 552 GEOM
positions against 311 in the largest game mesh. If the game rejects or mis-handles them, `export_molynx_assets(reduce=True)`
writes the reduced family instead (one quantisation step of tolerance), like 661 game files.

Poses. The Bile Demon's PoseFrame / DiePose / Dragged_Pose files are static meshes (kind 1, three
sprites, 20 CTRL entries, record type byte 5 = mesh), so Molynx's three are static meshes too:
GEOM holds each distinct position once, `pos` is its bounding-box centre and `scale` the largest
distance from it (862 of 862 game meshes).

Control points (CTRL). Not named by the community template ("uk"); decoded here from the Bile Demon:
in an animation an entry is (vertex << 16 | sprite, id) and in a static mesh (GEOM index, id) - the
idle1 and PoseFrame entries name the same positions this way. Ids 1000..1014 (1006 twice), 2000,
2001, 3000 and 5000 (the Bile Demon's 20) are kept in order; their meaning is not known (1000..
spread over the body, 2000 / 2001 at the front of the head, 3000 low at the back, 5000 = 1000 on top
of the head). Each is moved to Molynx by one rule: the Bile Demon's point in idle1 frame 0 and
Molynx's `stand` frame 0 are both normalised to their own bounding box (each axis to 0..1), and the
nearest Molynx body vertex (body sprites only: every Bile Demon point is on its body) takes it, the
first on a tie. The same vertex is used in every file.

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
from scripts.dk2paths import BEHEMOTH_DIR, EDITOR_DIR, GAME_DIR, MESHES_DIR
from scripts.molynx_record import (CLIPS, MAX_CLOSE_GAP, ONE_WAY_GAPS, POSE_STEMS, PREFIX,
                                   SOURCE_WRAP, FilePlan, clip_of, file_plan, slot_assignment)

PORTRAIT_SRC = Path("/mnt/user-data/attachments/76eb4112-412f-58e4-a30a-b275f992be89/"
                    "8582356e-image.png")
GAME_FPS = 30
VERSION = 17
LODS = 16
MAX_SPRITE_VERTICES = 255
MAX_SPRITE_TRIANGLES = 255
LOOP_GAP = 0.3  # a loop whose last frame is this close to frame 0 (x the median step) is closed
MAX_TEXTURE = 128
MM_PER_RATIO = 2.33
BODY_TEXTURE, MACE_TEXTURE = "Molynx_Body", "Molynx_Mace"
TEXTURE_NAMES = {"custom_character/body": BODY_TEXTURE, "custom_character/mace": MACE_TEXTURE}
PART_NAMES = {BODY_TEXTURE: "Body", MACE_TEXTURE: "Mace"}
POSE_SOURCES = {"PoseFrame": ("stand", 0), "DiePose": ("die", -1), "DraggedPose": ("being_picked_up", 0)}
FRAME = 7                       # the game portraits' frame, pixels on every side
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


def build_source(unit_dir: Path = BEHEMOTH_DIR, meshes_dir: Path = MESHES_DIR, cache=None):
    """(data, textures) of the finished character, built exactly as the approved build was: the
    MegaGlest stand / walk / attack / die plus the 21 retargeted clips, with the body triangles that
    some clip tears into spikes left out (behemoth_rig.stretched_triangles)."""
    from scripts.behemoth_anims import build_all
    from scripts.behemoth_rig import stretched_triangles
    from scripts.custom_character import build_character
    cache = texture_cache() if cache is None else cache
    with warnings.catch_warnings():
        # the donor tracks hold NaN for vertices a game file lacks; retarget's means of them warn
        warnings.simplefilter("ignore", RuntimeWarning)
        lib = build_all(Path(meshes_dir), cache, Path(unit_dir))
    data, textures, _ = build_character(Path(unit_dir), clips=("stand", "walk", "attack", "die"),
                                        extra=lib, drop=stretched_triangles(Path(unit_dir), lib))
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
    body = [s for s, lay in enumerate(plan.sprites) if lay.texture == BODY_TEXTURE]
    cand = np.concatenate([np.arange(our_starts[s], our_starts[s + 1]) for s in body])
    out = []
    for ref, ident in pairs:
        sprite, vertex = int(ref) & 0xFFFF, int(ref) >> 16
        point = flat_theirs[starts[sprite] + vertex]
        best = int(cand[np.argmin(np.linalg.norm(ours[cand] - point, axis=1))])
        s = int(np.searchsorted(our_starts, best, side="right") - 1)
        out.append((s, best - int(our_starts[s]), int(ident)))
    return out


def _mm_factor(points: np.ndarray, uv: np.ndarray, tris) -> float:
    t = np.asarray(tris)
    model = sum(np.linalg.norm(points[t[:, a]] - points[t[:, b]], axis=1).sum()
                for a, b in ((0, 1), (1, 2), (2, 0)))
    tex = sum(np.linalg.norm(uv[t[:, a]] - uv[t[:, b]], axis=1).sum()
              for a, b in ((0, 1), (1, 2), (2, 0)))
    return float(np.float32(MM_PER_RATIO * tex / model))


def plan_molynx(height_ratio: float = 1.15, source=None, *, unit_dir: Path = BEHEMOTH_DIR,
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
    sprites = [lay for i, p in enumerate(parts) for lay in _split(i, p["indices"], TEXTURE_NAMES[p["texture"]])]

    bile = bile_demon_record(editor_dir)
    idle = parse_kmf(game_mesh_path(bile.anim_name("idle1"), meshes_dir).read_bytes())
    their = np.concatenate(evaluate_anim_frame(idle.anim, 0))
    target = float(their[:, 2].max() - their[:, 2].min()) * height_ratio

    plan = MolynxPlan(source, height_ratio, parts, sprites, 1.0, np.zeros(3), target)
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
    for stem, (name, index) in POSE_SOURCES.items():
        plan.poses[stem] = plan.to_sprites(plan.to_kmf(data[name]["frames"][index]))

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


def clip_kmf(plan: MolynxPlan, stem: str, reduce: bool = False, wrap: bool | None = None) -> Kmf:
    """The animation file `stem` (a clip, or a variant with `wrap`); `wrap` None = the clip's own."""
    clip = plan.clips[stem] if wrap is None else file_clip(plan, stem, wrap)
    rows = _rows(clip.frames)
    sprites = _sprites(plan, clip.frames[0], rows, mesh=False)
    controls = b"".join(np.array([(v << 16) | s, ident], "<u4").tobytes()
                        for s, v, ident in plan.controls)
    anim = encode_anim(PREFIX + stem, sprites, clip.frames, clip.wrap, controls, reduce=reduce)
    return Kmf(VERSION, "anim", list(plan.materials), anim=anim)


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

def _texture(image: Image.Image, size: int) -> Image.Image:
    if size & (size - 1) or not 0 < size <= 256:
        raise ValueError(f"texture size {size} is not a power of two up to 256")
    image = image.convert("RGB")
    return image if image.size == (size, size) else image.resize((size, size), Image.LANCZOS)


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


def make_portraits(src: Path, bile: Creature, cache) -> tuple[Image.Image, Image.Image, Image.Image,
                                                              Image.Image]:
    """(molynx-bg, molynx, bile_demon-bg, bile_demon) images."""
    bg_frame = _cache_image(cache, bile.resource_name("portraitResource"))
    icon_frame = _cache_image(cache, bile.resource_name("icon1Resource"))
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

def export_molynx_assets(out_dir: Path, height_ratio: float = 1.15,
                         portrait_src: Path | None = PORTRAIT_SRC, *, plan: MolynxPlan | None = None,
                         body_size: int = MAX_TEXTURE, reduce: bool = False,
                         editor_dir: Path = EDITOR_DIR, game_dir: Path = GAME_DIR,
                         meshes_dir: Path = MESHES_DIR, wraps: dict | None = None) -> dict:
    """Write Molynx's .kmf files (the 25 clips, the variants `wraps` names and the 3 poses), 2
    textures and (with `portrait_src`) 2 portraits plus a preview into `out_dir`, which must not
    exist or be empty (nothing is ever deleted). Every input is checked before the first file is
    written. `wraps` ({file stem: loop flag}, default `required_wraps` of the editor's Bile Demon)
    sets the loop flag of the files it names (see the module docstring). `reduce=True` keeps only
    the keyframes each vertex needs (the fallback if the fully keyed files prove too big).
    -> {"meshes": {stem: path}, "textures": {name: path}, "portraits": {name: path},
        "preview": path | None, "info": {file stem: {frames, fps, wrap}} (final files),
        "wrap_exceptions": [molynx_record.WrapException] (default wraps only, else []),
        "scale": float, "target_height": float}"""
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"{out_dir} exists and is not an empty folder")
    if portrait_src is not None and not Path(portrait_src).is_file():
        raise FileNotFoundError(f"portrait source {portrait_src} not found")
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
    sizes = {BODY_TEXTURE: body_size, MACE_TEXTURE: MAX_TEXTURE}
    textures = {}
    for key, name in TEXTURE_NAMES.items():
        textures[name] = out_dir / f"{name}.png"
        _texture(images[key], min(sizes[name], max(images[key].size))).save(textures[name])
    portraits, preview = {}, None
    if portrait_src is not None:
        cache = texture_cache(game_dir)
        bile = bile_demon_record(editor_dir)
        bg, icon, bile_bg, bile_icon = make_portraits(Path(portrait_src), bile, cache)
        for name, image in (("molynx-bg", bg), ("molynx", icon)):
            portraits[name] = out_dir / f"{name}.png"
            image.save(portraits[name])
        preview = out_dir / "portraits_preview.png"
        _preview([("Molynx -bg", bg), ("Molynx", icon), ("Bile Demon -bg", bile_bg),
                  ("Bile Demon", bile_icon)]).save(preview)
    return {"meshes": meshes, "textures": textures, "portraits": portraits, "preview": preview,
            "info": info, "wrap_exceptions": exceptions, "scale": plan.scale, "target_height": plan.target_height}
