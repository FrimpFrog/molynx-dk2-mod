"""Build the Demon Spawn's game files end to end: model, skins, every clip and pose as KMF, placed the
way the Dragon's are (scripts.dragon_build: the same Bile Demon references and the same export).

Size: DK1's sprites put the spawn at two thirds of the Troll's height (31 against 47 pixels standing; its
thing size 200 against the Troll's 350 and the Dragon's 900), so `stand` frame 0 is 0.30 KMF units tall
(HEIGHT_RATIO x the Bile Demon's idle1 height) against the Troll's own files' 0.50 and the Dragon's 0.74.

Placement and props, as the Dragon's:
  being_picked_up  the spawn hangs by the scruff from the Keeper's hand: HOLD (on its back between the
                   shoulders) stays on the file origin in every frame.
  sleeping         carries the Bile Demon's lair bed; the spawn lies sprawled in it, centred, its lowest
                   point a little into the straw.
  eating           carries the Bile Demon's chicken (real size), standing on the floor in front of the
                   spawn: the tongue (a part of the mesh, rebuilt along its flight in every frame) shoots out
                   to it, reels it into the maw, and it is swallowed (shrunk to a point, as the game's own eat
                   files do); it is back on the floor before the loop closes.
  torture          the spawn's belly is kept where the Bile Demon's body is on average in Bil-Torture.
  praying          moved so its average place is the Bile Demon's in Bile_Demon-Pray.
The lolling tongue lies on the floor rather than through it (TONGUE_FLOOR) whenever the spawn stands.
Poses: PoseFrame = stand 0, DiePose = die last, DraggedPose = being_picked_up 0, StunnedPose = falling
last, ElecPose = getting_hit's peak.

Nothing here writes into a mod folder or a creature record: `export` writes to the folder it is given.

LICENCE: GPL-3.0.
"""
from pathlib import Path

import numpy as np

from scripts.dk2kmf import evaluate_anim_frame
from scripts.dragon_build import POSES, Built, _bed, _donor, _donor_sprites, export as _export, to_model
from scripts.dragon_export import make_plan
from scripts.spawn_anims import EAT_TIMES, FPS, build_clips
from scripts.spawn_model import BONE_INDEX, HEAD, build_spawn, build_tongue, spawn_skinner, tongue_path_rest
from scripts.spawn_texture import BODY, HEAD as HEAD_TEX, atlas_uv, bake

PREFIX = "DemonSpawn-"
HEIGHT_RATIO = 0.56            # x the Bile Demon's idle1 height (0.536): 0.30 KMF units
PART_LABEL = {BODY: "Body", HEAD_TEX: "Head"}
BODY_PARTS = ("trunk", "head", "jaw", "legs")
HOLD = np.array([0, .43, .12])           # on the back between the shoulders (model space, bind; chest bone)
MOUTH = np.array([0, .225, .37])           # inside the maw (head bone, bind)
CHICKEN_Z = 0.74                         # the chicken stands this far forward (model units) of the spawn
TONGUE_FLOOR = 0.004
FREE_TONGUE = ("being_picked_up", "torture")


def _bone_world(P, name, point):
    Rw, Pw = P.world()
    b = BONE_INDEX[name]
    return Pw[b] + Rw[b] @ (np.asarray(point, float) - HEAD[b])


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def build(meshes_dir, names=None, height_ratio=HEIGHT_RATIO):
    parts = build_spawn()
    sk = spawn_skinner(parts)
    tongue_i = next(i for i, p in enumerate(parts) if p.name == "tongue")
    clips = build_clips(names)
    bile_idle = _donor(meshes_dir, "BileDemon-Idle1")
    stand = clips["stand"][0] if "stand" in clips else build_clips(["stand"])["stand"][0]
    rest = sk(*stand[0].world())
    plan = make_plan(parts, rest, bile_idle, height_ratio, uv_of=atlas_uv, part_label=PART_LABEL,
                     body_parts=BODY_PARTS, prefix=PREFIX)
    frames, wraps, props = {}, {}, {}
    for name, (poses, wrap) in clips.items():
        model = [sk(*P.world()) for P in poses]
        if name == "eating":
            model, chicken = _eat(plan, poses, model, tongue_i, meshes_dir)
            props[name] = chicken
        if name not in FREE_TONGUE:
            for m in model:
                m[tongue_i][:, 1] = np.maximum(m[tongue_i][:, 1], TONGUE_FLOOR)
        kmf = [plan.frame(m) for m in model]
        if name == "being_picked_up":
            kmf = [[s - plan.to_kmf([_bone_world(P, "chest", HOLD)[None]])[0][0] for s in f]
                   for f, P in zip(kmf, poses)]
        elif name == "torture":
            donor = _donor(meshes_dir, "Bil-Torture")
            target = np.mean([evaluate_anim_frame(donor.anim, j)[0].mean(0)
                              for j in range(donor.anim.frame_count)], 0)
            kmf = [[s - plan.to_kmf([_bone_world(P, "spine", HEAD[BONE_INDEX["spine"]])[None]])[0][0] + target
                    for s in f] for f, P in zip(kmf, poses)]
        elif name == "praying":
            donor = _donor(meshes_dir, "Bile_Demon-Pray")
            theirs = np.mean([evaluate_anim_frame(donor.anim, j)[0].mean(0)
                              for j in range(donor.anim.frame_count)], 0)
            ours = np.mean([np.concatenate(f[:3]).mean(0) for f in kmf], 0)
            shift = np.array([theirs[0] - ours[0], theirs[1] - ours[1], 0.0])
            kmf = [[s + shift for s in f] for f in kmf]
        elif name == "sleeping":
            kmf, props[name] = _bed(plan, kmf, meshes_dir)
        frames[name] = kmf
        wraps[name] = wrap
    poses = {}
    for stem, (clip, idx) in POSES.items():
        if clip not in frames:
            continue
        f = frames[clip]
        if idx == "peak":
            base = np.concatenate(f[0])
            idx = int(np.argmax([np.linalg.norm(np.concatenate(x) - base, axis=1).mean() for x in f]))
        poses[stem] = f[idx]
    return Built(plan, parts, frames, wraps, props, poses)


def _eat(plan, poses, model, tongue_i, meshes_dir):
    """The tongue's flight and the chicken. -> (model points with the tongue rebuilt, chicken prop)."""
    donor = _donor(meshes_dir, "Bile_Demon-Eat")
    ids = _donor_sprites(donor, "chick")
    shape = [evaluate_anim_frame(donor.anim, 0)[i] for i in ids]
    allc = np.concatenate(shape)
    centre = (allc.min(0) + allc.max(0)) / 2
    local = [to_model(plan, s) - to_model(plan, centre[None])[0] for s in shape]     # model units, real size
    lo = np.concatenate(local).min(0)
    floor_c = np.array([0, -lo[1], CHICKEN_Z])                 # its centre, standing on the floor
    reach = floor_c + np.array([0, -.2 * lo[1], lo[2] * .6])   # where the tongue strikes it
    T = EAT_TIMES
    rest = tongue_path_rest()
    out_model, out_chicken = [], []
    for i, (P, m) in enumerate(zip(poses, model)):
        t = i / FPS
        Rw, Pw = P.world()
        jaw = BONE_INDEX["jaw"]
        posed = np.array([Pw[jaw] + Rw[jaw] @ (q - HEAD[jaw]) for q in rest])
        if t < T["shoot"]:
            e = 0.0
        elif t < T["hit"]:
            e = _smooth((t - T["shoot"]) / (T["hit"] - T["shoot"]))
        elif t < T["reel"]:
            e = 1 - _smooth((t - T["hit"]) / (T["reel"] - T["hit"]))
        else:
            e = 0.0
        tip = posed[2] * (1 - e) + reach * e
        mid = posed[0] + (tip - posed[0]) * .5 + np.array([0, .03 * np.sin(np.pi * e), 0])
        path = posed if e <= 0 else np.array([posed[0], posed[1] * (1 - e) + mid * e, tip])
        m = list(m)
        m[tongue_i] = np.array(build_tongue([], path).points)
        out_model.append(m)
        # the chicken
        mouth = _bone_world(P, "head", MOUTH)
        R = Rw[BONE_INDEX["head"]]
        if t < T["hit"] or t >= T["back"]:
            pos, Rc, s = floor_c, np.eye(3), 1.0
            if t >= T["back"]:
                s = float(_smooth((t - T["back"]) / 0.2))
        elif t < T["reel"]:                    # stuck on the tongue's tip, reeled in, tumbling a little
            u = (t - T["hit"]) / (T["reel"] - T["hit"])
            pos = tip + (floor_c - reach)
            pos[1] = max(pos[1], floor_c[1] * 1.15)       # never through the floor
            a = .6 * _smooth(u)
            Rc = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            s = 1.0
        else:                                  # in the maw, swallowed
            u = np.clip((t - T["reel"]) / (T["gone"] - T["reel"]), 0, 1)
            k = _smooth(u * 2)
            start = tip + (floor_c - reach)
            start[1] = max(start[1], floor_c[1] * 1.15)
            pos = start * (1 - k) + mouth * k
            Rc = R
            s = float(1 - _smooth(u))
        out_chicken.append([plan.to_kmf([pos + s * (q @ Rc.T)])[0] for q in local])
    psprites = [donor.anim.sprites[i] for i in ids]
    pmats = [donor.materials[s.material] for s in psprites]
    return out_model, (pmats, psprites, out_chicken)


def export(built, out_dir, textures=None):
    """Every clip, pose and skin of `built` into `out_dir` (DemonSpawn-<slot>.kmf, DemonSpawn_*.png)."""
    if textures is None:
        textures = bake(built.parts)
    return _export(built, out_dir, textures=textures)


if __name__ == "__main__":
    import json
    import sys
    from scripts.dk2paths import MESHES_DIR
    b = build(MESHES_DIR)
    print(json.dumps(export(b, Path(sys.argv[1])), indent=1))
