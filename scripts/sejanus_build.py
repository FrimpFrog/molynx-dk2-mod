"""Build Sejanus's game files: model, skins, clips and poses as DK2 KMF files named like the game's own
(Sejanus-Idle1, Sejanus-PoseFrame, ...), through the Dragon's exporter (dragon_export: KMF axes, sprites,
materials, control points and the height of the Horned Reaper's files, 0.74).

Stage 2 (test animations): Idle1, Idle2 (the Hunt), Walk, Run (the gallop), Melee, Angry, Entrance, and the
PoseFrame (Idle1's first frame, a static mesh like the game's own PoseFrames). Stage 3: every other slot a
DK2 creature plays (SLOTS), placed like the Dragon's (dragon_build: held from the hand, on the wheel, at the
temple's edge, in the lair's bed, with the Hatchery's chicken). Nothing here writes into the game or a mod
folder: `export` writes to the folder it is given.

LICENCE: GPL-3.0.
"""
from pathlib import Path

import numpy as np

from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import anim_invariants, write_kmf
from scripts.dragon_export import anim_kmf, make_plan, pose_kmf
from scripts.dragon_build import _bed, _donor, _donor_sprites, to_model
from scripts.sejanus_anims import FINGER_TIP, TIP, build_clips
from scripts.sejanus_model import BONE_INDEX, HEAD, Skinner, build_sejanus, head_point, JAW_CHIN
from scripts.sejanus_texture import atlas_uv, bake, vertex_occlusion

PREFIX = "Sejanus-"
PART_LABEL = {"Sejanus_Body": "Body", "Sejanus_Limbs": "Limbs", "Sejanus_Head": "Head"}
BODY_PARTS = ("trunk", "head", "jaw", "legs", "arms")
STEMS = {"stand": "Idle1", "hunt": "Idle2", "walk": "Walk", "running": "Run", "attack": "Melee",
         "angry": "Angry", "entrance": "Entrance",
         "walk_start": "WalkStart", "walk_end": "WalkEnd", "run_start": "RunStart", "run_end": "RunEnd",
         "walkback": "Walkback", "walkback_start": "WalkbackStart", "walkback_end": "WalkbackEnd",
         "magic": "Magic", "happy": "Happy", "dance": "Dance", "recoil_hff": "RecoilHff", "recoil_hfb": "RecoilHfb",
         "electrocute": "Electrocute", "drunk": "Drunk", "drunk_idle": "DrunkIdle", "drink": "Drink",
         "eating": "Eat", "research": "Research", "research_start": "ResearchStart", "research_end": "ResearchEnd",
         "building": "Building", "building_start": "BuildingStart", "building_end": "BuildingEnd",
         "melee2": "Melee2", "praying": "Pray", "praying_start": "PrayStart", "praying_end": "PrayEnd",
         "sleeping": "Sleep", "falling": "Fallback", "waking_up": "GetUp", "die": "Die", "being_picked_up": "Swing",
         "torture": "Torture", "dejected": "Dejected", "dragged": "Dragged"}
# poses (static meshes, like the game's own): clip and frame ("peak": the frame furthest from the first)
POSES = {"PoseFrame": ("stand", 0), "DiePose": ("die", -1), "DraggedPose": ("dragged", 0),
         "StunnedPose": ("falling", -1), "ElecPose": ("electrocute", "peak"), "DejectedPose": ("dejected", 0)}
# Creatures.kwd slot -> (file stem, has a Start file, has an End file). DK2 finds a slot's Start / End files
# by adding "Start" / "End" to its file name.
SLOTS = {
    "animWalkResource": ("Walk", True, True), "animRunResource": ("Run", True, True),
    "animDraggedPoseResource": ("DraggedPose", False, False), "animRecoilHffResource": ("RecoilHff", False, False),
    "animMelee1Resource": ("Melee", False, False), "animMelee2Resource": ("Melee2", False, False),
    "animMagicResource": ("Magic", False, False), "animDieResource": ("Die", False, False),
    "animHappyResource": ("Happy", False, False), "animAngryResource": ("Angry", False, False),
    "animStunnedPoseResource": ("StunnedPose", False, False), "animSwingResource": ("Swing", False, False),
    "animSleepResource": ("Sleep", False, False), "animEatResource": ("Eat", False, False),
    "animResearchResource": ("Research", True, True), "animDejectedPoseResource": ("DejectedPose", False, False),
    "animTortureResource": ("Torture", False, False), "animDrinkResource": ("Drink", False, False),
    "animIdle1Resource": ("Idle1", False, False), "animRecoilHfbResource": ("RecoilHfb", False, False),
    "animBuildingResource": ("Building", True, True), "animPrayResource": ("Pray", True, True),
    "animFallbackResource": ("Fallback", False, False), "animElecResource": ("ElecPose", False, False),
    "animElectrocuteResource": ("Electrocute", False, False), "animGetUpResource": ("GetUp", False, False),
    "animDanceResource": ("Dance", False, False), "animEntranceResource": ("Entrance", False, False),
    "animIdle2Resource": ("Idle2", False, False), "animDrunk2Resource": ("Drunk", False, False),
    "drunkIdle": ("DrunkIdle", False, False), "animWalkbackResource": ("Walkback", True, True),
    "animPoseFrameResource": ("PoseFrame", False, False), "animDiePoseResource": ("DiePose", False, False),
}
POSE_ONLY = ("dragged",)               # clips that only feed a pose
HOLD = np.array([0.0, 0.05, -0.04])
JAW_TIP = np.array(head_point(*JAW_CHIN))        # the chin (bind): the mouth's front     # the scruff: above and behind the head of neck1 (bind offset)


class Built:
    def __init__(self, plan, parts, frames, wraps, poses):
        self.plan, self.parts, self.frames, self.wraps, self.poses = plan, parts, frames, wraps, poses
        self.props = {}


def _bone_world(P, name, point):
    Rw, Pw = P.world()
    b = BONE_INDEX[name]
    return Pw[b] + Rw[b] @ (np.asarray(point, float) - HEAD[b])


def build(meshes_dir, names=None):
    parts = build_sejanus()
    sk = Skinner(parts)
    clips = build_clips(names)
    bile_idle = parse_kmf((Path(meshes_dir) / "BileDemon-Idle1.kmf").read_bytes())
    stand = clips["stand"][0] if "stand" in clips else build_clips(["stand"])["stand"][0]
    rest = sk(*stand[0].world())
    plan = make_plan(parts, rest, bile_idle, uv_of=atlas_uv, part_label=PART_LABEL, body_parts=BODY_PARTS,
                     prefix=PREFIX)
    frames, wraps, props = {}, {}, {}
    for name, (poses, wrap) in clips.items():
        kmf = [plan.frame(sk(*P.world())) for P in poses]
        if name == "being_picked_up":       # hanging from the Keeper's hand, the scruff at the file origin
            nk = HEAD[BONE_INDEX["neck1"]]
            kmf = [[s - plan.to_kmf([_bone_world(P, "neck1", nk + HOLD)[None]])[0][0] for s in f]
                   for f, P in zip(kmf, poses)]
        elif name == "torture":             # the belly where the Bile Demon's body is on the wheel
            donor = _donor(meshes_dir, "Bil-Torture")
            target = np.mean([evaluate_anim_frame(donor.anim, j)[0].mean(0)
                              for j in range(donor.anim.frame_count)], 0)
            sp = HEAD[BONE_INDEX["spine1"]]
            kmf = [[s - plan.to_kmf([_bone_world(P, "spine1", sp)[None]])[0][0] + target for s in f]
                   for f, P in zip(kmf, poses)]
        # (praying: at the temple's edge, as the Bile Demon: walked there in the clips, sejanus_more.PRAY_FWD)
        elif name == "dragged":             # lying limp from the file origin forward, as the game's own
            allp = np.concatenate(kmf[0])
            shift = np.array([-(allp[:, 0].min() + allp[:, 0].max()) / 2, -allp[:, 1].min() + 0.01, 0.0])
            kmf = [[s + shift for s in f] for f in kmf]
        elif name == "sleeping":            # down in the straw, his lowest point on the bed's inner floor
            import scripts.dragon_build as DB
            keep, DB.BED_SINK = DB.BED_SINK, 0.0
            try:
                kmf, props[name] = _bed(plan, kmf, meshes_dir)
            finally:
                DB.BED_SINK = keep
        elif name == "eating":
            props[name] = _chicken(plan, poses, meshes_dir)
        elif name == "drink":
            props[name] = _bottle(plan, poses, meshes_dir)
        frames[name] = kmf
        wraps[name] = wrap
    out = {}
    for stem, (clip, idx) in POSES.items():
        if clip not in frames:
            continue
        f = frames[clip]
        if idx == "peak":
            base = np.concatenate(f[0])
            idx = int(np.argmax([np.linalg.norm(np.concatenate(x) - base, axis=1).mean() for x in f]))
        out[stem] = f[idx]
    b = Built(plan, parts, frames, wraps, out)
    b.props = props
    return b


# the chicken's share left after each bite (sejanus_more.EAT_BITES): gone after the third, as in the game's
# own Eat files (played once)
CHICKEN_LEFT = (1.0, 0.72, 0.42, 0.0)


def _prop_local(plan, meshes_dir, donor_name, word, frame=0):
    donor = _donor(meshes_dir, donor_name)
    ids = _donor_sprites(donor, word)
    shape = [evaluate_anim_frame(donor.anim, frame)[i] for i in ids]
    allc = np.concatenate(shape)
    centre = (allc.min(0) + allc.max(0)) / 2
    local = [to_model(plan, s) - to_model(plan, centre[None])[0] for s in shape]
    psprites = [donor.anim.sprites[i] for i in ids]
    pmats = [donor.materials[sp.material] for sp in psprites]
    return local, pmats, psprites


def _chicken(plan, poses, meshes_dir):
    """The Hatchery's chicken (Bile_Demon-Eat's chicken sprites, real size): on the floor in front of him
    until his hands close on it, then held between his fingertips; each bite tears a share of it away."""
    from scripts.sejanus_more import EAT_BITES, EAT_GRAB
    local, pmats, psprites = _prop_local(plan, meshes_dir, "Bile_Demon-Eat", "chick")
    half = -min(l[:, 1].min() for l in local)                 # its centre's height when it stands on the floor
    mids = []
    for P in poses:
        tips = [_bone_world(P, "hand" + s, TIP[s]) for s in "LR"]
        mids.append((tips[0] + tips[1]) / 2)
    ig = int(round(EAT_GRAB * 30))
    floor = np.array([mids[ig][0], half, mids[ig][2]])
    out = []
    for i, P in enumerate(poses):
        t = i / 30.0
        held = mids[i] + np.array([0.0, 0.01, 0.0])
        w = float(np.clip((t - (EAT_GRAB - 0.05)) / 0.12, 0, 1))
        pos = floor * (1 - w) + held * w
        pos[1] = max(pos[1], half)                     # (never into the floor as the hands close on it)
        left = CHICKEN_LEFT[0]
        for k, b in enumerate(EAT_BITES):
            left += (CHICKEN_LEFT[k + 1] - CHICKEN_LEFT[k]) * float(np.clip((t - b) / 0.12, 0, 1))
        s_ = max(left, 0.0) ** (1 / 3)
        out.append([plan.to_kmf([pos + s_ * l])[0] for l in local])
    return pmats, psprites, out


def _rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def _bottle(plan, poses, meshes_dir):
    """The Tavern's bottle (Troll-Drink's bottle sprites, real size, taken where it is whole): it appears in
    his right hand, upright; tipped with its neck to his mouth while he drinks; then shrinks away, as the
    game's own Drink files do."""
    from scripts.sejanus_more import DRINK_TIMES
    local, pmats, psprites = _prop_local(plan, meshes_dir, "Troll-Drink", "ottle", frame=60)
    pts = np.concatenate(local)
    # its long axis, pointing to the neck (the narrower end)
    _, _, vt = np.linalg.svd(pts - pts.mean(0))
    ax = vt[0]
    proj = pts @ ax
    hi, lo = pts[proj > np.percentile(proj, 80)], pts[proj < np.percentile(proj, 20)]
    if np.ptp(hi - hi.mean(0), 0).sum() > np.ptp(lo - lo.mean(0), 0).sum():
        ax = -ax
    a0, a1 = DRINK_TIMES["appear"]
    d0, d1 = DRINK_TIMES["drink"]
    g0, g1 = DRINK_TIMES["gone"]
    out = []
    for i, P in enumerate(poses):
        t = i / 30.0
        hand = _bone_world(P, "handR", TIP["R"])
        mouth = _bone_world(P, "jaw", JAW_TIP)
        tip = float(np.clip(min((t - d0 + 0.3) / 0.3, (d1 + 0.15 - t) / 0.3), 0, 1))
        want = np.array([0.0, 1.0, 0.0]) * (1 - tip) + (mouth - hand) / np.linalg.norm(mouth - hand) * tip
        R = _rot_between(ax, want)
        s_ = float(np.clip((t - a0) / (a1 - a0), 0, 1)) * float(np.clip((g1 - t) / (g1 - g0), 0, 1))
        s_ = max(s_, 0.02)
        centre = hand + R @ ax * 0.05 * tip - R @ ax * 0.02 * (1 - tip)   # its neck reaches past the fingers
        out.append([plan.to_kmf([centre + s_ * (l @ R.T)])[0] for l in local])
    return pmats, psprites, out


def skins(parts):
    return bake(parts, occlusion=vertex_occlusion(parts))


def export(built, out_dir, textures=None, reduce=True):
    """Write every clip, pose and skin of `built` into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, f in built.frames.items():
        if name in POSE_ONLY:
            continue
        stem = STEMS.get(name, name)
        k = anim_kmf(built.plan, stem, f, built.wraps[name], reduce=reduce, extra=built.props.get(name))
        bad = anim_invariants(k.anim)
        if bad:
            raise ValueError(f"{name}: {bad}")
        (out_dir / f"{PREFIX}{stem}.kmf").write_bytes(write_kmf(k))
        info[name] = {"frames": k.anim.frame_count, "fps": 30, "wrap": built.wraps[name],
                      "keys": len(k.anim.geom_xyz) - 1}
    for stem, f in built.poses.items():
        (out_dir / f"{PREFIX}{stem}.kmf").write_bytes(write_kmf(pose_kmf(built.plan, stem, f)))
    if textures is None:
        textures = skins(built.parts)
    for name, img in textures.items():
        img.save(out_dir / f"{name}.png")
    return info


if __name__ == "__main__":
    import sys
    meshes = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/dk2data/Meshes"
    print(export(build(meshes), sys.argv[1]))
