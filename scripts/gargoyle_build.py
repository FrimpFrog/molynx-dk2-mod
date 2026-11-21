"""Build the Gargoyle's game files: model, skins, his clips and poses as KMF, through the Dragon's export
(scripts.dragon_export.make_plan / scripts.dragon_build.export: the same Bile Demon references for
materials and control points, the same file conventions).

Size: Soles wants him small, "a cheap fighter like the goblin": `stand` frame 0 (his crouch, the folded
wings' wrist knobs the highest point) is HEIGHT_RATIO x the Bile Demon's idle1 height (0.536) = 0.50 KMF
units; the top of his head is then about 0.43, between the Imp's own files (0.36) and the Goblin's
(0.61).

The full set (after Soles approved the test clips): every slot a DK2 creature plays, named like the game's
own creature files (STEMS; SLOTS: which Creatures.kwd slot each fills, with the Start / End files the slot's
flags ask for), and the static poses (POSES).

Placement and carried objects follow the Goblin's own files (measured from them, the way dragon_build
follows the Bile Demon's):
  Swing    hangs from the Keeper's hand: Goblin-Swing has the hand at the file origin and the body below,
           so the scruff of his neck (HOLD, on the chest bone between the wing roots) is on the origin.
  Sleep    the file carries the Bile Demon's nest (lair object 18, Soles 15 Nov: the bed the Bile Demon and
           the Dragon use), as BileDemon-bed(sleeping) carries it at frame 0, the way dragon_build puts it
           in the Dragon's Sleep; he lies prone in it (turned as he lay in the Goblin's bed), centred on
           it and lifted just enough that no part of him in any frame is under the nest's top surface
           (_nest_lift: his lowest point then touches the straw).
  Eat      the file carries the Goblin's chicken (Goblin-Eat's two chicken sprites, real size) on the floor
           in front of him (CHICKEN_AT); each tear (EAT_TIMES) takes a piece and it is gone after the gulp,
           shrunk to a point as the game's own eat files do.
  Drink    the file carries the Goblin's bottle (Goblin-Drink's two Bottle1 sprites, real size) in his
           right claw, its neck towards his jaws; it grows out of nothing and shrinks away again
           (DRINK_TIMES), as the Goblin's does.
  Torture  his middle (the spine bone) is kept where the Goblin's body is on average in Gob-Torture.
  Pray     he walks PRAY_WALK forward in PrayStart (as the Goblin's PrayStart walks forward) and prays
           there: his average place in Pray is then about the Goblin's in Goblin-Pray.
  DraggedPose  on his back on the floor, his middle where the Goblin's is in GOB-Dragged_Pose.

LICENCE: GPL-3.0.
"""
from pathlib import Path

import numpy as np

from scripts.dk2kmf import evaluate_anim_frame, mesh_points
from scripts.dragon_build import Built, _donor, _donor_sprites, export as _export, to_model
from scripts.dragon_export import make_plan
from scripts.gargoyle_anims import BOTTLE_HAND, CHICKEN_AT, DRINK_TIMES, EAT_TIMES, MOUTH, STATIC_POSES, build_clips
from scripts.gargoyle_model import BONE_INDEX, HEAD, build_gargoyle, gargoyle_skinner
from scripts.gargoyle_texture import BODY, HEAD as HEAD_TEX, WING, atlas_uv, bake

PREFIX = "Gargoyle-"
HEIGHT_RATIO = 0.93            # x the Bile Demon's idle1 height (0.536): 0.50 KMF units
PART_LABEL = {BODY: "Body", HEAD_TEX: "Head", WING: "Wing"}
BODY_PARTS = ("trunk", "head", "jaw", "legs")
# File stem of each clip, named like the game's own creature files (DK2 finds a slot's Start / End files by
# adding "Start" / "End" to the slot's file name)
STEMS = {
    "stand": "Idle1", "statue": "Idle2", "walk": "Walk", "walk_start": "WalkStart", "walk_end": "WalkEnd",
    "running": "Run", "run_start": "RunStart", "run_end": "RunEnd", "attack": "Melee", "melee2": "Melee2",
    "magic": "Magic", "angry": "Angry", "entrance": "Entrance", "happy": "Happy", "dance": "Dance",
    "getting_hit": "RecoilHff", "being_slapped": "RecoilHfb", "electrocute": "Electrocute",
    "staggering": "Drunk", "drunk_idle": "DrunkIdle", "drinking": "Drink", "eating": "Eat",
    "interacting": "Research", "interacting_start": "ResearchStart", "interacting_end": "ResearchEnd",
    "manufacturing": "Building", "manufacturing_start": "BuildingStart", "manufacturing_end": "BuildingEnd",
    "praying": "Pray", "praying_start": "PrayStart", "praying_end": "PrayEnd", "sleeping": "Sleep",
    "falling": "Fallback", "waking_up": "GetUp", "die": "Die", "being_picked_up": "Swing", "torture": "Torture",
    "walkback": "Walkback", "walkback_start": "WalkbackStart", "walkback_end": "WalkbackEnd",
    "dejected": "Dejected",
}
# Creatures.kwd slot -> (file stem, has a Start file, has an End file): DK2 loads <stem>Start / <stem>End
# when the slot's art resource carries the start / end flags (scripts.dk2kwd_creatures), as the Goblin's
# Walk, Pray and Walkback do. RunStart unfurls the wings, RunEnd folds them. Dejected has no slot of its own
# (its first frame is the DejectedPose).
SLOTS = {
    "animIdle1Resource": ("Idle1", False, False), "animIdle2Resource": ("Idle2", False, False),
    "animWalkResource": ("Walk", True, True), "animRunResource": ("Run", True, True),
    "animWalkbackResource": ("Walkback", True, True),
    "animMelee1Resource": ("Melee", False, False), "animMelee2Resource": ("Melee2", False, False),
    "animMagicResource": ("Magic", False, False), "animAngryResource": ("Angry", False, False),
    "animHappyResource": ("Happy", False, False), "animDanceResource": ("Dance", False, False),
    "animEntranceResource": ("Entrance", False, False),
    "animRecoilHffResource": ("RecoilHff", False, False), "animRecoilHfbResource": ("RecoilHfb", False, False),
    "animElectrocuteResource": ("Electrocute", False, False), "animElecResource": ("ElecPose", False, False),
    "animDrunk2Resource": ("Drunk", False, False), "drunkIdle": ("DrunkIdle", False, False),
    "animDrinkResource": ("Drink", False, False), "animEatResource": ("Eat", False, False),
    "animResearchResource": ("Research", True, True), "animBuildingResource": ("Building", True, True),
    "animPrayResource": ("Pray", True, True), "animSleepResource": ("Sleep", False, False),
    "animFallbackResource": ("Fallback", False, False), "animGetUpResource": ("GetUp", False, False),
    "animDieResource": ("Die", False, False), "animSwingResource": ("Swing", False, False),
    "animTortureResource": ("Torture", False, False),
    "animPoseFrameResource": ("PoseFrame", False, False), "animDiePoseResource": ("DiePose", False, False),
    "animDraggedPoseResource": ("DraggedPose", False, False),
    "animStunnedPoseResource": ("StunnedPose", False, False),
    "animDejectedPoseResource": ("DejectedPose", False, False),
}
# pose -> (clip, frame), or (one of gargoyle_anims.STATIC_POSES, 0)
POSES = {"PoseFrame": ("stand", 0), "DiePose": ("die", -1), "DraggedPose": ("dragged", 0),
         "StunnedPose": ("falling", -1), "ElecPose": ("electrocute", 0), "DejectedPose": ("dejected", 0)}
BED_FLOOR = -0.004                      # his lowest point in the bed (KMF z, down): just off the floor
HOLD = np.array([0, .47, -.02])          # the scruff: between the wing roots (model space, bind, chest bone)
FPS = 30.0


def meshes_dir():
    """The game's extracted meshes: dk2paths.MESHES_DIR, else the 2015 copy in /home/claude/dk2data."""
    from scripts.dk2paths import MESHES_DIR, have
    return MESHES_DIR if have(MESHES_DIR) else Path("/home/claude/dk2data/Meshes")


def build(meshes_dir, names=None, height_ratio=HEIGHT_RATIO):
    parts = build_gargoyle()
    sk = gargoyle_skinner(parts)
    clips = build_clips(names)
    stand = clips["stand"][0] if "stand" in clips else build_clips(["stand"])["stand"][0]
    rest = sk(*stand[0].world())
    plan = make_plan(parts, rest, _donor(meshes_dir, "BileDemon-Idle1"), height_ratio, uv_of=atlas_uv,
                     part_label=PART_LABEL, body_parts=BODY_PARTS, prefix=PREFIX)
    frames, wraps, props = {}, {}, {}
    for name, (poses, wrap) in clips.items():
        kmf = [plan.frame(sk(*P.world())) for P in poses]
        stem = STEMS.get(name, name)
        if name == "being_picked_up":
            kmf = [[q - plan.to_kmf([_bone_world(P, "chest", HOLD)[None]])[0][0] for q in f]
                   for f, P in zip(kmf, poses)]
        elif name == "torture":
            target = _donor_mean(meshes_dir, "Gob-Torture")
            kmf = [[q - plan.to_kmf([_bone_world(P, "spine", HEAD[BONE_INDEX["spine"]])[None]])[0][0] + target
                    for q in f] for f, P in zip(kmf, poses)]
        elif name == "sleeping":
            kmf, props[stem] = _bed(plan, kmf, meshes_dir)
        elif name == "eating":
            props[stem] = _chicken(plan, len(kmf), meshes_dir)
        elif name == "drinking":
            props[stem] = _bottle(plan, poses, meshes_dir)
        frames[stem] = kmf
        wraps[stem] = wrap
    poses = {stem: frames[STEMS.get(clip, clip)][idx] for stem, (clip, idx) in POSES.items()
             if STEMS.get(clip, clip) in frames}
    for stem, (clip, idx) in POSES.items():
        if clip in STATIC_POSES and (names is None or clip in names):
            f = plan.frame(sk(*STATIC_POSES[clip]().world()))
            if clip == "dragged":
                theirs = np.concatenate(list(mesh_points(_donor(meshes_dir, "GOB-Dragged_Pose")))).mean(0)
                ours = np.concatenate(f).mean(0)
                f = [q + np.array([theirs[0] - ours[0], theirs[1] - ours[1], 0.0]) for q in f]
            poses[stem] = f
    return Built(plan, parts, frames, wraps, props, poses)


def _bone_world(P, name, point):
    Rw, Pw = P.world()
    b = BONE_INDEX[name]
    return Pw[b] + Rw[b] @ (np.asarray(point, float) - HEAD[b])


def _donor_mean(meshes_dir, name):
    donor = _donor(meshes_dir, name)
    return np.mean([np.concatenate(evaluate_anim_frame(donor.anim, j)).mean(0)
                    for j in range(donor.anim.frame_count)], 0)


def _bed(plan, kmf, meshes_dir):
    """Into the Bile Demon's nest as BileDemon-bed(sleeping) carries it (frame 0, as the Dragon's Sleep
    has it): turned round as he was in the Goblin's bed (a half turn about the vertical), centred on the
    nest, on the floor (BED_FLOOR), then lifted by _nest_lift."""
    donor = _donor(meshes_dir, "BileDemon-bed(sleeping)")
    ids = _donor_sprites(donor, "bed")
    f0 = evaluate_anim_frame(donor.anim, 0)
    nest = np.concatenate([f0[i] for i in ids])
    turn = np.array([-1.0, -1.0, 1.0])
    kmf = [[q * turn for q in f] for f in kmf]
    allp = np.concatenate([np.concatenate(f) for f in kmf])
    lo, hi = allp.min(0), allp.max(0)
    mid = (nest.min(0) + nest.max(0)) / 2
    shift = np.array([mid[0] - (lo[0] + hi[0]) / 2, mid[1] - (lo[1] + hi[1]) / 2, BED_FLOOR - hi[2]])  # (z down)
    out = [[q + shift for q in f] for f in kmf]
    lift = _nest_lift(plan, out, [f0[i] for i in ids], [donor.anim.sprites[i] for i in ids])
    out = [[q - np.array([0.0, 0.0, lift]) for q in f] for f in out]
    pframes = [[f0[i] for i in ids]] * len(kmf)
    psprites = [donor.anim.sprites[i] for i in ids]
    return out, ([donor.materials[sp.material] for sp in psprites], psprites, pframes)


NEST_STEP = 0.004            # the nest's surface is read on a grid this fine; his triangles sampled as finely
NEST_CLEAR = 0.001           # and he lies this far above it at his lowest contact


def nest_surface(points, tris, step=NEST_STEP):
    """The top surface (height up = -z) of the nest over a grid in KMF x, y: (xs, ys, H), -inf off it."""
    lo, hi = points[:, :2].min(0) - step, points[:, :2].max(0) + step
    xs, ys = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
    H = np.full((len(ys), len(xs)), -np.inf)
    up = -points[:, 2]
    for a, b, c in tris:
        A, B, C = points[a, :2], points[b, :2], points[c, :2]
        i0, i1 = np.searchsorted(xs, min(A[0], B[0], C[0])), np.searchsorted(xs, max(A[0], B[0], C[0]), "right")
        j0, j1 = np.searchsorted(ys, min(A[1], B[1], C[1])), np.searchsorted(ys, max(A[1], B[1], C[1]), "right")
        den = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        if i1 <= i0 or j1 <= j0 or abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(xs[i0:i1], ys[j0:j1])
        w0 = ((B[1] - C[1]) * (gx - C[0]) + (C[0] - B[0]) * (gy - C[1])) / den
        w1 = ((C[1] - A[1]) * (gx - C[0]) + (A[0] - C[0]) * (gy - C[1])) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        sub = H[j0:j1, i0:i1]
        h = w0 * up[a] + w1 * up[b] + w2 * up[c]
        sub[inside] = np.maximum(sub[inside], h[inside])
    return xs, ys, H


def surface_under(surf, P):
    """The nest's top (or the floor, 0) under each point of P (KMF)."""
    xs, ys, H = surf
    i = np.clip(np.round((P[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1)
    j = np.clip(np.round((P[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
    h = H[j, i]
    off = (P[:, 0] < xs[0]) | (P[:, 0] > xs[-1]) | (P[:, 1] < ys[0]) | (P[:, 1] > ys[-1]) | ~np.isfinite(h)
    return np.maximum(np.where(off, 0.0, h), 0.0)


def body_samples(points, tris, step=NEST_STEP):
    """His vertices and points across each triangle about `step` apart."""
    out = [points]
    for a, b, c in tris:
        A, B, C = points[a], points[b], points[c]
        n = int(max(np.linalg.norm(B - A), np.linalg.norm(C - A), np.linalg.norm(C - B)) / step)
        if n >= 2:
            k = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
            out.append(A + np.outer(k[:, 0], B - A) + np.outer(k[:, 1], C - A))
    return np.concatenate(out)


def _nest_lift(plan, frames, nest_parts, nest_sprites):
    """How far up he must go so that, in every frame, no part of him is under the nest's top surface."""
    pts = np.concatenate(nest_parts)
    base = np.cumsum([0] + [len(p) for p in nest_parts])
    tris = np.concatenate([np.asarray(sp.triangles[0]) + base[k] for k, sp in enumerate(nest_sprites)])
    surf = nest_surface(pts, tris)
    need = 0.0
    for f in frames:
        for q, sp in zip(f, plan.sprites):
            S = body_samples(q, np.asarray(sp.triangles))
            need = max(need, float((surface_under(surf, S) - (-S[:, 2])).max()))
    return max(need, 0.0) + NEST_CLEAR if need > 0 else 0.0


def _ramp(t, t0, t1):
    return float(np.clip((t - t0) / (t1 - t0), 0, 1))


def chicken_left(t):
    """How much of the chicken is left at time t: a piece goes with each tear, the rest with the gulp."""
    tears, gone = EAT_TIMES["tears"], EAT_TIMES["gone"]
    s = 1.0
    for k, tt in enumerate(tears):
        s -= 0.22 * _ramp(t, tt - 0.04, tt + 0.08)
    return s * (1 - _ramp(t, gone - 0.25, gone))


def _chicken(plan, n, meshes_dir):
    """The Goblin's chicken, as it lies in Goblin-Eat's first frame, on the floor at CHICKEN_AT; it shrinks
    towards its middle on the floor as he tears it up."""
    donor = _donor(meshes_dir, "Goblin-Eat")
    ids = _donor_sprites(donor, "chick")
    shape = [evaluate_anim_frame(donor.anim, 0)[i] for i in ids]
    allc = np.concatenate(shape)
    base = np.array([(allc[:, 0].min() + allc[:, 0].max()) / 2, (allc[:, 1].min() + allc[:, 1].max()) / 2,
                     allc[:, 2].max()])                            # its middle on the floor (z down)
    at = plan.to_kmf([np.array([[CHICKEN_AT[0], 0.0, CHICKEN_AT[2]]])])[0][0]
    at[2] = 0.0
    out = []
    for i in range(n):
        s = max(chicken_left(i / FPS), 0.0)
        out.append([at + s * (q - base) for q in shape])
    psprites = [donor.anim.sprites[i] for i in ids]
    return [donor.materials[sp.material] for sp in psprites], psprites, out


def _rotation(a, b):
    """The smallest rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:
        p = np.cross(a, [1, 0, 0]) if abs(a[0]) < .9 else np.cross(a, [0, 1, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + V + V @ V / (1 + c)


BOTTLE_GRIP = 0.8      # of the bottle's length from its neck: where the claws hold it


def _bottle(plan, poses, meshes_dir):
    """The Goblin's bottle (Goblin-Drink, as it is at his first glug), held in the right claws with its neck
    towards his jaws; it grows out of nothing in his claw and shrinks away (DRINK_TIMES)."""
    donor = _donor(meshes_dir, "Goblin-Drink")
    ids = _donor_sprites(donor, "bottle")
    shape = [to_model(plan, evaluate_anim_frame(donor.anim, 60)[i]) for i in ids]
    allb = np.concatenate(shape)
    c = allb.mean(0)
    axis = np.linalg.svd(allb - c)[2][0]
    if np.dot(shape[-1].mean(0) - c, axis) < 0:          # the small second sprite is the neck end
        axis = -axis
    proj = (allb - c) @ axis
    neck = c + axis * proj.max()
    length = proj.max() - proj.min()
    radius = np.percentile(np.linalg.norm((allb - c) - np.outer(proj, axis), axis=1), 90)
    local = [q - neck for q in shape]                      # neck at the origin, the bottle along -axis
    (s0, s1), (h0, h1) = DRINK_TIMES["show"], DRINK_TIMES["hide"]
    out = []
    for i, P in enumerate(poses):
        t = i / FPS
        palm = _bone_world(P, "handAR", BOTTLE_HAND)
        wrist = _bone_world(P, "handAR", HEAD[BONE_INDEX["handAR"]])
        mouth = _bone_world(P, "head", MOUTH)
        u = mouth - palm
        u /= np.linalg.norm(u)
        out_ = palm - wrist                                 # the claws close round it: the bottle's side
        out_ -= u * np.dot(out_, u)                         # lies against the palm, not through the hand
        grip = palm + out_ / (np.linalg.norm(out_) + 1e-9) * radius
        u = mouth - grip
        u /= np.linalg.norm(u)
        R = _rotation(axis, u)
        neck_at = grip + u * BOTTLE_GRIP * length
        s = _ramp(t, s0, s1) * (1 - _ramp(t, h0, h1))
        out.append([plan.to_kmf([grip + s * (neck_at + q @ R.T - grip)])[0] for q in local])
    psprites = [donor.anim.sprites[i] for i in ids]
    return [donor.materials[sp.material] for sp in psprites], psprites, out


def export(built, out_dir, textures=None):
    """Every clip, pose and skin of `built` into `out_dir` (Gargoyle-<slot>.kmf, Gargoyle_*.png)."""
    if textures is None:
        textures = bake(built.parts)
    return _export(built, out_dir, textures=textures)


if __name__ == "__main__":
    import json
    import sys
    # python3 -m scripts.gargoyle_build <out dir> [<the game's Meshes dir>]
    b = build(Path(sys.argv[2]) if len(sys.argv) > 2 else meshes_dir())
    print(json.dumps(export(b, Path(sys.argv[1])), indent=1))
