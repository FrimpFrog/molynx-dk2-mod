"""Quuz's Horned Reaper (creature 50) as game files: every file DK2's Horny plays, re-encoded onto
the Reaper's body (Soles, 13 Nov 2026: "he goes in the game with vanilla Horny's animations, no
custom set").

What his files are. Horny's record (Creatures.kwd record 27) fills 12 animation slots with 11
REA3_* clips and the Rea3-poseframe pose; Walk and Melee1 also ask for their Start / End files
(flags HAS_START / HAS_END). HORNY_SLOTS lists them, read from the record; `file_plan` names the
Reaper's file for each (HReaper-<the clip's name without REA3_>, so a clip Horny uses in two slots,
REA3_Roar in Idle1 and Loot, is one file). Every file keeps the frame count, loop flag and
animation controls of Horny's, so Horny's slot tails (frames, fps, flags) stay right for his record.

How a frame is made. The Reaper is reaper_build's: Horny's reference pose (Rea3-poseframe)
reshaped (`reaper_build.reshape`: leaner, the DK1 build), Quuz's painted skins, the shoulder spikes
rooted on the torso. For each frame of a Horny clip every reference place (necro_build.match: by
texture and UV) takes Horny's own position in that frame, plus the reshape's offset at that place
turned with the bones that carry it: each bone's rotation is the rigid fit (part_pose) of its
places from the reference pose to the frame, and the offset is turned by the place's skin weights
(reaper_build.build_rig). So his hands, hooves and scythe follow Horny's motion exactly, and his
body keeps its leaner shape in every pose. A UV key both legs (or both faces of the scythe blade)
share can pair a clip vertex with the wrong place; such a vertex is told by how far it runs from
where its bones carry the place, and is not used (`posed_places`: without this a leg vertex shot
out as a long spike in Melee3 and GemIntro). A place a clip does not draw (Melee1 and Melee2 draw a
different face texture first, ReaperFace, on the same layout: read as the smirk, `load_clip`)
follows its bones from the reference pose. Props of a clip that
are not Horny (the gem of REA3_PickUpGem4) are kept as they are.

The scythe stays in his hand, as in Horny's clips (he swings it in Melee).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

import scripts.necro_build as nb
import scripts.reaper_build as rb
from scripts.dk2kmf_write import write_kmf

PREFIX = rb.PREFIX                       # "HReaper-"
HORNY_ID = 27
# Horny's record (27) slot -> (file, has Start, has End), from the editor Creatures.kwd
HORNY_SLOTS = {
    "animWalkResource": ("REA3_Walk", True, True),
    "animDraggedPoseResource": ("REA3_GemIntro", False, False),
    "animMelee1Resource": ("REA3_Melee3", True, True),
    "animMagicResource": ("REA3_Footstamp", False, False),
    "animEatResource": ("REA3_Melee2", False, False),
    "animResearchResource": ("REA3_Melee1", False, False),
    "animIdle1Resource": ("REA3_Roar", False, False),
    "animPrayResource": ("REA3_PickUpGem4", False, False),
    "animElecResource": ("REA3_StampOnChicken", False, False),
    "animDanceResource": ("REA3_Dance", False, False),
    "animLootResource": ("REA3_Roar", False, False),
    "animPoseFrameResource": ("Rea3-poseframe", False, False),
}


def reaper_stem(horny_stem: str) -> str:
    """HReaper-<name>: REA3_Walk -> HReaper-Walk, Rea3-poseframe -> HReaper-PoseFrame."""
    if horny_stem.lower() == "rea3-poseframe":
        return PREFIX + "PoseFrame"
    if not horny_stem.startswith("REA3_"):
        raise ValueError(f"{horny_stem} is not one of Horny's files")
    return PREFIX + horny_stem[len("REA3_"):]


def file_plan() -> list[tuple[str, str, str]]:
    """(slot, Reaper file, Horny file) for every file his record names, Start / End files included;
    a file two slots share is listed once (under its first slot)."""
    out, seen = [], set()
    for slot, (stem, has_start, has_end) in HORNY_SLOTS.items():
        for suffix, wanted in (("", True), ("Start", has_start), ("End", has_end)):
            if wanted and stem + suffix not in seen:
                seen.add(stem + suffix)
                out.append((slot, reaper_stem(stem) + suffix, stem + suffix))
    return out


def slot_files() -> dict[str, str]:
    """Slot -> the Reaper file its name points at (Start / End follow by name)."""
    return {slot: reaper_stem(stem) for slot, (stem, _s, _e) in HORNY_SLOTS.items()}


# ------------------------------------------------------------------------------ posing

PICK_SLACK = 0.006      # KMF units (he stands 0.74): vertices this close to the best one count too
STRAY = 0.08            # a place whose best vertex strays this far on average follows its bones


def _bone_places(weights: np.ndarray, bone: int) -> np.ndarray:
    return np.flatnonzero(weights[:, bone] >= 0.5)


def posed_places(build: rb.Build, clip: nb.Clip) -> tuple[list[np.ndarray], float]:
    """Per frame, the Reaper's place positions (KMF frame) for a Horny clip, and the share of the
    reference places the clip draws (1.0 for most files)."""
    ref = build.ref
    nb.match(ref, clip)
    n = len(ref.places)
    w = build.rig.weights
    delta = build.rest - ref.places
    bones = build.rig.bones
    parents = build.rig.parents
    F = len(clip.frames)
    turned = np.zeros((F, n, 3))
    moved = np.zeros((F, n, 3))
    for f in range(F):
        poses = {}
        for b, name in enumerate(bones):
            poses[name] = nb.part_pose(ref, clip, _bone_places(w, b), f)
        for name in bones:                       # a bone without a fit takes its parent's
            k = name
            while poses[k] is None and parents.get(k):
                k = parents[k]
            poses[name] = poses[k] if poses[k] is not None else (np.eye(3), np.zeros(3))
        for b, name in enumerate(bones):
            r, t = poses[name]
            turned[f] += w[:, [b]] * (delta @ r.T)
            moved[f] += w[:, [b]] * (ref.places @ r.T + t)
    # Each place's vertices in the clip. A UV key both sides share (the legs, the scythe's two
    # faces) can pair a clip vertex with the wrong place (necro_build.match); such a vertex runs far
    # from where its bones carry the place. So per place only the clip vertices that stay with
    # their bones are used (mean distance over the clip within PICK_SLACK of the best), and a place
    # none of whose vertices does (STRAY) follows its bones.
    cands: dict[int, list] = {}
    for s in clip.body:
        for i, p in enumerate(clip.place[s]):
            if p >= 0:
                cands.setdefault(int(p), []).append(np.array([clip.frames[f][s][i] for f in range(F)]))
    here = moved.copy()
    seen = np.zeros(n, bool)
    for p, tracks in cands.items():
        tr = np.stack(tracks)                                   # (k, F, 3)
        dist = np.linalg.norm(tr - moved[None, :, p], axis=2).mean(1)
        best = dist.min()
        if best > STRAY:
            continue
        keep = dist <= best * 1.5 + PICK_SLACK
        here[:, p] = tr[keep].mean(0)
        seen[p] = True
    return [here[f] + turned[f] for f in range(F)], float(seen.mean())


def _extras(clip: nb.Clip) -> list:
    """The clip's sprites that are not Horny's body (props), with their per-frame points."""
    sprites = clip.sprites()
    out = []
    for s in range(len(sprites)):
        if s in clip.body:
            continue
        mat = clip.model.materials[sprites[s].material]
        out.append((mat, sprites[s], [clip.frames[f][s] for f in range(len(clip.frames))]))
    return out


# Horny's face material lists three expression textures on one UV layout (ReaperFaceSmirk,
# ReaperFace, ReaperFaceSmile); Melee1 and Melee2 draw it with ReaperFace first. Read as the smirk
# (the texture his face is matched and drawn with), so those clips' faces are his, not a second face.
FACE = "ReaperFaceSmirk"


def load_clip(meshes_dir: Path, stem: str) -> nb.Clip:
    """A Horny file as reaper_build reads it, every face material read as the smirk."""
    clip = rb.load_clip(meshes_dir, stem)
    mats = []
    for m in clip.model.materials:
        names = [t.lower() for t in m.textures]
        if FACE.lower() in names and names[0] != FACE.lower():
            m = dataclasses.replace(m, textures=[FACE] + [t for t in m.textures if t.lower() != FACE.lower()])
        mats.append(m)
    clip.model = dataclasses.replace(clip.model, materials=mats)
    sprites = clip.sprites()
    clip.body = [i for i, sp in enumerate(sprites) if mats[sp.material].textures[0].lower() in rb.BODY_TEX]
    return clip


def reaper_file(build: rb.Build, meshes_dir: Path, horny_stem: str, name: str):
    """(Kmf, info) of one Reaper file from one Horny file."""
    clip = load_clip(meshes_dir, horny_stem)
    posed, share = posed_places(build, clip)
    if clip.is_anim:
        a = clip.model.anim
        k = rb.reaper_kmf(build.ref, posed, build.accs, name, bool(a.wrap), extras=_extras(clip),
                          controls=a.controls)
    else:
        k = rb.reaper_kmf(build.ref, posed, build.accs, name, None, extras=_extras(clip))
    return k, {"from": horny_stem, "kind": k.kind, "frames": k.anim.frame_count if k.anim else 1,
               "source_frames": len(clip.frames),
               "wrap": bool(k.anim.wrap) if k.anim else None,
               "keys": len(k.anim.geom_xyz) - 1 if k.anim else 0, "places_drawn": round(share, 4)}


def export(meshes_dir: Path, cache, out_dir: Path) -> dict:
    """Write every Reaper file (<name>.kmf) and his skins (PNG) into out_dir; returns per file:
    source, kind, frames, wrap, keys."""
    from scripts.reaper_skin import all_skins
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build = rb.prepare(meshes_dir)
    info = {}
    for slot, name, stem in file_plan():
        k, i = reaper_file(build, meshes_dir, stem, name)
        (out_dir / f"{name}.kmf").write_bytes(write_kmf(k))
        info[name] = {"slot": slot, **i}
    for skin, img in all_skins(cache, build.ref).items():
        img.save(out_dir / f"{skin}.png")
    return info
