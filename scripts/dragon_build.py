"""Build the Dragon's game files end to end: model, skins, every clip and pose as KMF, placed the way the
game's own files place a creature in the slots that need it.

Placement and carried objects follow what the Bile Demon's files do (measured here; molynx_assets found
the same in the review rounds of Molynx):
  being_picked_up  the creature hangs from the Keeper's hand: Bile_Demon-Swing has the body below the
                   file origin (z -0.12 .. +0.42) with the hand at the origin, so the dragon's shoulders
                   (HOLD, on its back between the wings) are kept on the origin in every frame.
  sleeping         the file carries the lair's bed (Bile Demon's: the game hides the lair object while a
                   creature sleeps in it), and the dragon lies in it: centred over the bed, its lowest
                   point on the bed's inner floor.
  eating           the file carries the chicken (Bile_Demon-Eat's two chicken sprites, real size): held in
                   the jaws, tossed up, caught and swallowed; the game's own eat files shrink the chicken to
                   a point once it is eaten, and so does this one.
  torture          the dragon's belly is kept where the Bile Demon's body is on average in Bil-Torture
                   (on the wheel).
  praying          the clip is moved so the dragon's average place is the Bile Demon's in Bile_Demon-Pray
                   (the game stands a creature at the temple's edge; the file says how far in it bows).
Poses (static meshes, like the Bile Demon's): PoseFrame = stand 0, DiePose = die last, DraggedPose =
being_picked_up 0, StunnedPose = falling last, ElecPose = getting_hit's peak.

LICENCE: GPL-3.0.
"""
from pathlib import Path

import numpy as np

from scripts.dk2kmf import Sprite, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import anim_invariants, write_kmf
from scripts.dragon_anims import CLIPS, build_clips
from scripts.dragon_export import PREFIX, anim_kmf, make_plan, pose_kmf
from scripts.dragon_model import BONE_INDEX, HEAD, Skinner, build_dragon
from scripts.dragon_texture import bake, vertex_occlusion

HOLD = np.array([0, .74, -.10])          # on the back between the shoulders (model space, bind)
POSES = {"PoseFrame": ("stand", 0), "DiePose": ("die", -1), "DraggedPose": ("being_picked_up", 0),
         "StunnedPose": ("falling", -1), "ElecPose": ("getting_hit", "peak")}
BED_SINK = 0.25           # of the bed's rim height above its inner floor: how far into the straw he lies


def to_model(plan, k):
    """KMF points -> model space (inverse of DragonPlan.to_kmf)."""
    q = (np.asarray(k, float) - plan.offset) / plan.scale
    return np.stack([-q[:, 0], -q[:, 2], q[:, 1]], 1)


def _bone_world(P, name, point):
    Rw, Pw = P.world()
    b = BONE_INDEX[name]
    return Pw[b] + Rw[b] @ (np.asarray(point, float) - HEAD[b])


def _donor(meshes_dir, name):
    return parse_kmf((Path(meshes_dir) / f"{name}.kmf").read_bytes())


def _donor_sprites(model, word):
    return [i for i, s in enumerate(model.anim.sprites)
            if word in model.materials[s.material].textures[0].lower()]


class Built:
    def __init__(self, plan, parts, frames, wraps, props, poses):
        self.plan, self.parts, self.frames, self.wraps, self.props, self.poses = plan, parts, frames, wraps, props, poses


def build(meshes_dir, names=None, height_ratio=None):
    parts = build_dragon()
    sk = Skinner(parts)
    clips = build_clips(names)
    bile_idle = _donor(meshes_dir, "BileDemon-Idle1")
    stand = clips["stand"][0] if "stand" in clips else build_clips(["stand"])["stand"][0]
    rest = sk(*stand[0].world())
    plan = make_plan(parts, rest, bile_idle, **({} if height_ratio is None else {"height_ratio": height_ratio}))
    frames, wraps, props = {}, {}, {}
    for name, (poses, wrap) in clips.items():
        model = [sk(*P.world()) for P in poses]
        kmf = [plan.frame(m) for m in model]
        shift = None
        if name == "being_picked_up":
            kmf = [[s - plan.to_kmf([_bone_world(P, "chest", HOLD)[None]])[0][0] for s in f]
                   for f, P in zip(kmf, poses)]
        elif name == "torture":
            donor = _donor(meshes_dir, "Bil-Torture")
            target = np.mean([evaluate_anim_frame(donor.anim, j)[0].mean(0)
                              for j in range(donor.anim.frame_count)], 0)
            out = []
            for i, (f, P) in enumerate(zip(kmf, poses)):
                belly = plan.to_kmf([_bone_world(P, "spine1", HEAD[BONE_INDEX["spine1"]])[None]])[0][0]
                out.append([s - belly + target for s in f])
            kmf = out
        elif name == "praying":
            donor = _donor(meshes_dir, "Bile_Demon-Pray")
            theirs = np.mean([evaluate_anim_frame(donor.anim, j)[0].mean(0)
                              for j in range(donor.anim.frame_count)], 0)
            ours = np.mean([np.concatenate(f[:3]).mean(0) for f in kmf], 0)
            shift = np.array([theirs[0] - ours[0], theirs[1] - ours[1], 0.0])
            kmf = [[s + shift for s in f] for f in kmf]
        elif name == "sleeping":
            kmf, props[name] = _bed(plan, kmf, meshes_dir)
        elif name == "eating":
            props[name] = _chicken(plan, poses, meshes_dir)
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


def _bed(plan, kmf, meshes_dir):
    donor = _donor(meshes_dir, "BileDemon-bed(sleeping)")
    ids = _donor_sprites(donor, "bed")
    bed0 = np.concatenate([evaluate_anim_frame(donor.anim, 0)[i] for i in ids])
    centre = (bed0.min(0) + bed0.max(0)) / 2
    r = np.linalg.norm(bed0[:, :2] - centre[:2], axis=1)
    inner = bed0[r < 0.35 * r.max()]
    floor = inner[:, 2].min() if len(inner) else bed0[:, 2].max()     # z down: the inner floor's top
    rim = bed0[:, 2].min()
    lie_z = floor + (rim - floor) * BED_SINK                            # a little into the straw
    allp = np.concatenate([np.concatenate(f) for f in kmf])
    lo, hi = allp.min(0), allp.max(0)
    body = (lo + hi) / 2
    shift = np.array([centre[0] - body[0], centre[1] - body[1], lie_z - hi[2]])
    out = [[s + shift for s in f] for f in kmf]
    pframes = [[evaluate_anim_frame(donor.anim, 0)[i] for i in ids]] * len(kmf)
    psprites = [donor.anim.sprites[i] for i in ids]
    pmats = [donor.materials[s.material] for s in psprites]
    return out, (pmats, psprites, pframes)


# the chicken in the jaws: its centre this far along the snout (model space, bind) and the timeline (s)
MOUTH = np.array([0, .885, .20])
CHICKEN_TIMES = {"release": 0.7, "catch": 1.05, "gone": 1.4, "back": 2.2}


def _chicken(plan, poses, meshes_dir):
    donor = _donor(meshes_dir, "Bile_Demon-Eat")
    ids = _donor_sprites(donor, "chick")
    shape = [evaluate_anim_frame(donor.anim, 0)[i] for i in ids]
    allc = np.concatenate(shape)
    centre = (allc.min(0) + allc.max(0)) / 2
    local = [to_model(plan, s) - to_model(plan, centre[None])[0] for s in shape]   # model units, real size
    n = len(poses)
    fps = 30.0
    mouths, rots = [], []
    for P in poses:
        Rw, Pw = P.world()
        mouths.append(_bone_world(P, "head", MOUTH))
        rots.append(Rw[BONE_INDEX["head"]])
    t_rel, t_cat, t_gone, t_back = (CHICKEN_TIMES[k] for k in ("release", "catch", "gone", "back"))
    i_rel, i_cat = int(round(t_rel * fps)), int(round(t_cat * fps))
    p_rel, p_cat = mouths[i_rel], mouths[min(i_cat, n - 1)]
    out = []
    for i in range(n):
        t = i / fps
        if t <= t_rel or t >= t_back:
            pos, R, s = mouths[i], rots[i], 1.0
            if t >= t_back:
                s = min(1.0, (t - t_back) / 0.15)
        elif t < t_cat:     # in the air: a parabola from the release to the catch, tumbling
            u = (t - t_rel) / (t_cat - t_rel)
            pos = p_rel * (1 - u) + p_cat * u + np.array([0, 0.22 * 4 * u * (1 - u), 0])
            ang = u * 2 * np.pi
            c, si = np.cos(ang), np.sin(ang)
            R = rots[i_rel] @ np.array([[1, 0, 0], [0, c, -si], [0, si, c]])
            s = 1.0
        else:              # caught, then swallowed: it shrinks into the mouth
            pos, R = mouths[i], rots[i]
            s = float(np.clip(1 - (t - t_cat) / (t_gone - t_cat), 0, 1))
        if s <= 0:
            s = 0.0
        pts = [plan.to_kmf([(pos + s * (l @ R.T))])[0] for l in local]
        out.append(pts)
    if n > 1 and poses[0] is not None:
        pass
    psprites = [donor.anim.sprites[i] for i in ids]
    pmats = [donor.materials[s.material] for s in psprites]
    return pmats, psprites, out


def export(built, out_dir, textures=None, reduce=True):
    """Write every clip, pose and skin of `built` into `out_dir` (a new or empty folder)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, f in built.frames.items():
        k = anim_kmf(built.plan, name, f, built.wraps[name], reduce=reduce, extra=built.props.get(name))
        bad = anim_invariants(k.anim)
        if bad:
            raise ValueError(f"{name}: {bad}")
        (out_dir / f"{built.plan.prefix}{name}.kmf").write_bytes(write_kmf(k))
        info[name] = {"frames": k.anim.frame_count, "fps": 30, "wrap": built.wraps[name],
                      "keys": len(k.anim.geom_xyz) - 1}
    for stem, f in built.poses.items():
        (out_dir / f"{built.plan.prefix}{stem}.kmf").write_bytes(write_kmf(pose_kmf(built.plan, stem, f)))
    if textures is None:
        textures = bake(built.parts, occlusion=vertex_occlusion(built.parts))
    for name, img in textures.items():
        img.save(out_dir / f"{name}.png")
    return info
