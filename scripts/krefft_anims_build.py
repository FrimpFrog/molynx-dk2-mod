"""Krefft's clips (scripts.krefft_anims) as the game's files, and review renders of them from those files.

Every clip is sampled, skinned (scripts.krefft_rig) and written as a DK2 animation KMF through the same
export plan as his PoseFrame (scripts.krefft_build.export_standing --v5: the same scale, offset, sprites,
materials and control points), with the game's own carried objects where his clips need them:
  Eat     the Bile Demon's chicken (Bile_Demon-Eat's two chicken sprites, real size), held in his claws and
          shrinking bite by bite (the game's own eat files shrink it to a point when it is eaten)
  Drink   the Bile Demon's bottle (Bile_Demon-Drink's two bottle sprites) in his right claw
  Sleep   the Bile Demon's lair bed (BileDemon-bed(sleeping)'s bed sprite: the game hides the lair object
          while a creature sleeps in it); Krefft sits in it, his lowest point a little into the straw
  Idle2   the game's Coin mesh (Coin.kmf: the Coin Spill / Casino Jackpot effects' coin) for the coin flip
  Magic   four Coin meshes: the gob of coins between his lips, gone at the spit (the shot takes over)
Placement as the Bile Demon's files: Swing hangs from the Keeper's hand at the file origin (held by the
front of the shell's rim); Pray is moved so his average place is the Bile Demon's in Bile_Demon-Pray;
Torture and Entrance follow the Bile Demon's own body place (krefft_rig.donor_body, in the clips).
Poses (static meshes): PoseFrame = the bind (unchanged from the model export), DiePose = Die's last frame,
DraggedPose = Swing's first, StunnedPose = Fallback's last, ElecPose = Electrocute's peak, DejectedPose.

  python3 -m scripts.krefft_anims_build <export dir> [<gif dir>] [clip,clip,...]

Nothing is written anywhere but the folders given.

LICENCE: GPL-3.0.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts import krefft_rig as KR
from scripts.krefft_anims import CLIPS, SPIT_AT
from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf
from scripts.dk2kmf_write import anim_invariants, write_kmf
from scripts.dragon_export import BILE_IDLE_HEIGHT, anim_kmf, make_plan, pose_kmf

MESHES = Path("/home/claude/dk2data/Meshes")
PREFIX = "Krefft-"
POSES = {"PoseFrame": None, "DiePose": ("die", -1), "DraggedPose": ("being_picked_up", 0),
         "StunnedPose": ("falling", -1), "ElecPose": ("electrocute", "peak"), "DejectedPose": ("dejected", 0)}
MOUTH = np.array([0, .563, .405])            # between the lips (B's bind, carried by the head)
HAND = {"L": np.array([.215, .29, .33]), "R": np.array([-.215, .29, .33])}     # on the paddle (handF bones)
HOLD = np.array([0, .6, .07])                # the shell's front rim, over the neck (root)
BED_SINK = .3


def _kmf(name):
    return parse_kmf((MESHES / f"{name}.kmf").read_bytes())


def plan():
    from scripts.dragon_build import _donor
    from scripts.krefft_build import BODY_PARTS, PART_LABEL, STANDING_HEIGHT
    from scripts.krefft_texture import atlas_uv
    parts, _ = KR.model()
    rest = [np.array(p.points) for p in parts]
    return make_plan(parts, rest, _donor(MESHES, "BileDemon-Idle1"), STANDING_HEIGHT / BILE_IDLE_HEIGHT,
                     uv_of=atlas_uv, part_label=PART_LABEL, body_parts=BODY_PARTS, prefix=PREFIX)


def to_model(pl, k):
    q = (np.asarray(k, float) - pl.offset) / pl.scale
    return np.stack([-q[:, 0], -q[:, 2], q[:, 1]], 1)


def sample(clip):
    """-> (points per part per frame (model space), bone transforms per frame, times)."""
    n = clip.frames
    times = np.arange(n - 1 if clip.wrap else n) / clip.fps
    dn = None
    if clip.donor:
        dn = KR.donor_body(clip.donor[0])[0]
        if clip.name == "entrance":
            # the Bile Demon ends his entrance standing taller than his idle: his rise is kept, measured to
            # his last frame, so Krefft lands on the floor where the clip ends (Idle1 follows)
            # never above the floor: the Bile Demon's lift after the rise and his hop are dropped (the hop read
            # as a pop, the lift left the feet hovering)
            end = dn[-1][1]

            def flat(t):
                t = t - end
                return np.array([t[0], min(t[1], 0.0), t[2]])     # (no float, no hop: he stands once up)
            dn = [(R, flat(t)) for R, t in dn]
    out, bones = [], []
    for i, t in enumerate(times):
        ch = clip.ch(float(t))
        donor = None
        if dn is not None:
            m = len(dn)
            j = int(round(i * (m - 1) / max(n - 1, 1)))
            w_r, w_t = clip.donor[1], clip.donor[2]
            axes = clip.donor[3] if len(clip.donor) > 3 else (1, 1, 1)
            R, _ = KR.scaled(dn[j], w_r)
            _, tt = KR.scaled(dn[j], w_t, axes=axes)
            # the rest-relative turn of the donor's own first frame is kept only where he starts posed
            donor = (R, tt)
        pts, RP = KR.skin(ch, donor)
        out.append(pts)
        bones.append(RP)
    if clip.wrap:
        out.append(out[0])
        bones.append(bones[0])
        times = np.r_[times, times[-1] + 1 / clip.fps]
    return out, bones, times


# ------------------------------------------------------------------------------------------- props

def _shape(model, ids, frame=None):
    """A carried object's sprites (KMF), each centred on the object's centre -> (materials, sprites, local
    shapes (KMF units, KMF axes))."""
    if model.kind == "mesh":
        pts = list(mesh_points(model))
        sprites = model.mesh.sprites
    else:                     # the frame where the object is biggest (the game's files grow it from a point)
        if frame is None:
            frame = int(np.argmax([np.ptp(np.concatenate([evaluate_anim_frame(model.anim, f)[i] for i in ids]), 0).sum()
                                   for f in range(model.anim.frame_count)]))
        pts = evaluate_anim_frame(model.anim, frame)
        sprites = model.anim.sprites
    sel = [pts[i] for i in ids]
    allp = np.concatenate(sel)
    c = (allp.min(0) + allp.max(0)) / 2
    return [model.materials[sprites[i].material] for i in ids], [sprites[i] for i in ids], [s - c for s in sel]


def _sprites_with(model, word):
    sp = model.mesh.sprites if model.kind == "mesh" else model.anim.sprites
    return [i for i, s in enumerate(sp) if word in model.materials[s.material].textures[0].lower()]


def _place(pl, local, pos, R, s):
    """A local KMF shape at model point `pos`, turned by model rotation R, scaled by s -> KMF points."""
    F = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0.0]])       # KMF axes -> model axes (rows)
    m = (np.asarray(local) @ F.T) * s / 1.0                    # (model axes, KMF units)
    p = pos * pl.scale                                         # model point in KMF units, model axes
    w = (m @ R.T) + p
    return np.stack([-w[:, 0], w[:, 2], -w[:, 1]], 1) + pl.offset


def coin_shape():
    k = _kmf("Coin")
    return _shape(k, [0])


def _spin(a, axis=0):
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def prop_coin_flip(pl, bones, times):
    mats, sps, loc = coin_shape()
    frames = []
    t_rel, t_cat = 1.1, 1.85
    hand = [KR.bone_point(b, "handFR", HAND["R"] + np.array([0, .012, 0])) for b in bones]
    rot = [b[0][KR.model()[1].index["handFR"]] for b in bones]
    i_rel = int(np.argmin(abs(times - t_rel)))
    i_cat = int(np.argmin(abs(times - t_cat)))
    for i, t in enumerate(times):
        s = .42 * float(np.clip((t - .3) / .15, 0, 1)) * float(np.clip((2.95 - t) / .12, 0, 1))
        flat = rot[i] @ _spin(np.pi / 2 * 0)
        if t_rel < t < t_cat:
            u = (t - t_rel) / (t_cat - t_rel)
            pos = hand[i_rel] * (1 - u) + hand[i_cat] * u + np.array([0, .32 * 4 * u * (1 - u), 0])
            R = _spin(u * 6 * np.pi) @ rot[i_rel]
        else:
            pos, R = hand[i], flat
        frames.append([_place(pl, l, pos, R, max(s, 1e-4)) for l in loc])
    return mats, sps, frames


def prop_coin_gob(pl, bones, times, fly=False):
    """The gob of coins between his lips (0.9 s -> the spit). `fly` (the preview only, never written): the
    shot flying on from the mouth, and the game's Coin Spill burst where it lands."""
    mats, sps, loc = coin_shape()
    offs = [np.array(o) for o in ((0, 0, 0), (.018, .012, -.006), (-.016, .008, -.004), (.004, -.012, .006))]
    turns = [_spin(.3 * k, 1) @ _spin(.9 + .5 * k) for k in range(4)]
    head = KR.model()[1].index["head"]
    mouths = [KR.bone_point(b, "head", MOUTH) for b in bones]
    fwd = [b[0][head] @ np.array([0, -.15, 1.0]) for b in bones]
    i0 = int(np.argmin(abs(times - SPIT_AT)))
    rng = np.random.default_rng(3)
    burst = [(rng.uniform(-1, 1, 3) * np.array([.25, 0, .25]) + np.array([0, .35, 0]), rng.uniform(0, 6)) for _ in range(4)]
    frames = []
    for i, t in enumerate(times):
        out = []
        for k, (o, R0) in enumerate(zip(offs, turns)):
            if t < SPIT_AT:
                s = .6 * float(np.clip((t - .88) / .12, 0, 1))
                pos, R = mouths[i] + (bones[i][0][head] @ o), bones[i][0][head] @ R0
            elif fly:
                # the shot as the game data makes it (krefft_coin_spit): ONE Coin mesh at its full size (the
                # shot's art scale 1.0) flying nearly level at the Salamander spit's pace, with the Gold Sparkle
                # trail (krefft_v5_review draws the game's GSparkle sprites behind it); then the Coin Spill
                u = t - SPIT_AT
                land = .55
                d = fwd[i0] / np.linalg.norm(fwd[i0])
                d = d - np.array([0, d[1], 0]) * .7
                d = d / np.linalg.norm(d)
                if u < land:
                    pos = mouths[i0] + d * 1.8 * u + np.array([0, .04 * u - .2 * u * u, 0])
                    R = _spin(u * 20 + k) @ R0
                    s = 1.0 if k == 0 else 0.0
                else:                                   # the Coin Spill: the coins burst out and bounce
                    v = u - land
                    base = mouths[i0] + d * 1.8 * land + np.array([0, .04 * land - .2 * land * land, 0])
                    vel, a = burst[k]
                    pos = base + np.array([vel[0], 0, vel[2]]) * v + np.array([0, max(vel[1] * v - 2.5 * v * v, -base[1]), 0])
                    R = _spin(a + v * 9) @ R0
                    s = .6 * float(np.clip((1.0 - v) / .3, 0, 1))
            else:
                pos, R, s = mouths[i], bones[i][0][head], 0.0
            out.append(_place(pl, loc[0], pos, R, max(s, 1e-4)))
        frames.append(out)
    return mats * 4, sps * 4, frames


def prop_chicken(pl, bones, times):
    """The game's own chicken (Chicken-Idle: standing, upright), real size, between his claws; smaller after
    each bite, gone at the gulp (the game's eat files shrink the chicken to a point)."""
    k = _kmf("Chicken-Idle")
    mats, sps, loc = _shape(k, [0, 1])
    frames = []
    for i, t in enumerate(times):
        a = KR.bone_point(bones[i], "handFL", HAND["L"])
        b = KR.bone_point(bones[i], "handFR", HAND["R"])
        pos = (a + b) / 2 + np.array([0, .045, .0])
        bites = sum(t > tb for tb in (.75, 1.25, 1.75))
        s = [1.0, .8, .6, .42][bites] * float(np.clip((2.05 - t) / .2, 0, 1)) * float(np.clip((t - .2) / .2, 0, 1))
        R = _spin(-.5)                              # its head tipped toward his mouth
        frames.append([_place(pl, l, pos, R, max(s, 1e-4)) for l in loc])
    return mats, sps, frames


def _upright(loc):
    """Turn a bottle's sprites (KMF) so its long axis stands up, the narrow end (the neck) on top."""
    P = np.concatenate(loc)
    _, _, vt = np.linalg.svd(P - P.mean(0))
    a = vt[0]
    t = P @ a
    lo = P[t < t.min() + .25 * np.ptp(t)]
    hi = P[t > t.max() - .25 * np.ptp(t)]
    spread = lambda Q: np.ptp(Q - np.outer(Q @ a, a), 0).sum()
    if spread(hi) > spread(lo):
        a = -a
    up = np.array([0, 0, -1.0])                      # KMF z is down
    v = np.cross(a, up)
    c = float(a @ up)
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
    return [l @ R.T for l in loc]


def prop_bottle(pl, bones, times):
    k = _kmf("Bile_Demon-Drink")
    mats, sps, loc = _shape(k, _sprites_with(k, "bottle"))
    loc = _upright(loc)
    idx = KR.model()[1].index["handFR"]
    frames = []
    for i, t in enumerate(times):
        pos = KR.bone_point(bones[i], "handFR", HAND["R"] + np.array([0, .03, -.01]))
        tip = float(np.clip((pos[1] - .42) / .07, 0, 1))      # raised to the mouth: tipped up, the neck in
        R = bones[i][0][idx] @ _spin(-.5 - 1.3 * tip)    # upright in the claw at rest
        frames.append([_place(pl, l, pos, R, .55) for l in loc])
    return mats, sps, frames


def bed(pl, kmf):
    k = _kmf("BileDemon-bed(sleeping)")
    ids = _sprites_with(k, "bed")
    bed0 = np.concatenate([evaluate_anim_frame(k.anim, 0)[i] for i in ids])
    centre = (bed0.min(0) + bed0.max(0)) / 2
    r = np.linalg.norm(bed0[:, :2] - centre[:2], axis=1)
    inner = bed0[r < .35 * r.max()]
    floor = inner[:, 2].min() if len(inner) else bed0[:, 2].max()
    rim = bed0[:, 2].min()
    lie_z = floor + (rim - floor) * BED_SINK
    allp = np.concatenate([np.concatenate(f) for f in kmf])
    lo, hi = allp.min(0), allp.max(0)
    body = (lo + hi) / 2
    shift = np.array([centre[0] - body[0], centre[1] - body[1], lie_z - hi[2]])
    out = [[s + shift for s in f] for f in kmf]
    pframes = [[evaluate_anim_frame(k.anim, 0)[i] for i in ids]] * len(kmf)
    return out, ([k.materials[k.anim.sprites[i].material] for i in ids], [k.anim.sprites[i] for i in ids], pframes)


PROPS = {"coin_flip": prop_coin_flip, "coin_gob": prop_coin_gob, "chicken": prop_chicken, "bottle": prop_bottle}


# ------------------------------------------------------------------------------------------- build

class Built:
    def __init__(self):
        self.plan = plan()
        self.frames, self.wraps, self.props, self.clips, self.bones, self.times = {}, {}, {}, {}, {}, {}


def build(names=None):
    B = Built()
    pl = B.plan
    for name, make in CLIPS.items():
        clip = make()
        if names and clip.name not in names and name not in names:
            continue
        pts, bones, times = sample(clip)
        kmf = [pl.frame(p) for p in pts]
        extra = None
        if clip.name == "being_picked_up":
            kmf = [[s - pl.to_kmf([KR.bone_point(b, "root", HOLD)[None]])[0][0] for s in f] for f, b in zip(kmf, bones)]
        elif clip.name in ("praying", "praying_start", "praying_end"):
            d = _kmf("Bile_Demon-Pray")
            theirs = np.mean([evaluate_anim_frame(d.anim, j)[0].mean(0) for j in range(d.anim.frame_count)], 0)
            ref = CLIPS["praying"]()
            ours = np.mean([np.concatenate(pl.frame(p)[:3]).mean(0) for p in sample(ref)[0]], 0)
            shift = np.array([theirs[0] - ours[0], theirs[1] - ours[1], 0.0])
            kmf = [[s + shift for s in f] for f in kmf]
        elif clip.name == "torture":
            # his shell's centre, on average, where the Bile Demon's body is on average in Bil-Torture (the
            # room's wheel is a room object: the creature's file only places him on it), as the Dragon's
            d = _kmf("Bil-Torture")
            target = np.mean([evaluate_anim_frame(d.anim, j)[0].mean(0) for j in range(d.anim.frame_count)], 0)
            ours = np.mean([pl.to_kmf([b[1][0][None]])[0][0] for b in bones], 0)
            kmf = [[sp + (target - ours) for sp in f] for f in kmf]
        elif clip.name == "sleeping":
            kmf, extra = bed(pl, kmf)
        for p in clip.props:
            if p in PROPS:
                extra = PROPS[p](pl, bones, times)
        B.frames[clip.name], B.wraps[clip.name], B.clips[clip.name] = kmf, clip.wrap, clip
        B.bones[clip.name], B.times[clip.name] = bones, times
        if extra is not None:
            B.props[clip.name] = extra
    return B


def poses(B):
    out = {}
    for stem, spec in POSES.items():
        if spec is None:
            parts, _ = KR.model()
            out[stem] = B.plan.frame([np.array(p.points) for p in parts])
            continue
        clip, idx = spec
        if clip not in B.frames:
            continue
        f = B.frames[clip]
        if idx == "peak":
            base = np.concatenate(f[0])
            idx = int(np.argmax([np.linalg.norm(np.concatenate(x) - base, axis=1).mean() for x in f]))
        out[stem] = f[idx]
    return out


def export(B, out_dir, skins_from):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, f in B.frames.items():
        clip = B.clips[name]
        if clip.stem == "Dejected":
            continue                           # a pose only
        k = anim_kmf(B.plan, clip.stem, f, clip.wrap, extra=B.props.get(name))
        bad = anim_invariants(k.anim)
        if bad:
            raise ValueError(f"{name}: {bad}")
        (out_dir / f"{PREFIX}{clip.stem}.kmf").write_bytes(write_kmf(k))
        info[name] = {"stem": clip.stem, "frames": k.anim.frame_count, "fps": clip.fps, "wrap": clip.wrap,
                      "source": clip.source}
    for stem, f in poses(B).items():
        (out_dir / f"{PREFIX}{stem}.kmf").write_bytes(write_kmf(pose_kmf(B.plan, stem, f)))
    for n in ("Krefft_Body", "Krefft_Head", "Krefft_Shell"):
        shutil.copy(Path(skins_from) / f"{n}.png", out_dir / f"{n}.png")
    return info


# ------------------------------------------------------------------------------------------- renders

INK = (235, 220, 140)
CAM = {"Walk": 62, "WalkStart": 62, "WalkEnd": 62, "Run": 62, "Walkback": 62, "WalkbackStart": 62,
       "WalkbackEnd": 62, "Drunk": 62, "Melee": 55, "Melee2": 40, "Magic": 60, "Fallback": 70, "GetUp": 70,
       "Die": 70, "Torture": 25, "Swing": 30, "Gamble": 20}
EL = 30                     # a mid-high camera (DK2's own sits at about 45-50 above the floor)


def _frame_box(k, az, el, frames):
    from scripts.dragon_render import view
    R = view(az, el)
    pts = np.concatenate([np.concatenate(evaluate_anim_frame(k.anim, f)) for f in frames])
    q = np.stack([-pts[:, 0], -pts[:, 2], pts[:, 1]], 1) @ R.T
    q = np.vstack([q, np.zeros((1, 3))])
    lo, hi = q.min(0), q.max(0)
    return (lo + hi) / 2, float(max(hi[:2] - lo[:2]))


NO_FLOOR = {"Swing"}            # hangs below the Keeper's hand at the file's origin: no floor drawn
FRAME_FROM = {"Entrance": .35}   # framed on the end of the clip (the start is below the floor)


def clip_frames(k, tex, stem, S=360, step=1, az=None, el=EL, backdrop=None, label=True, span=None, centre=None):
    """Render every `step`-th frame of an animation KMF (what the game loads), framed once round the clip.
    `backdrop`: (Kmf, (dx, dy, dz) KMF offset) a game mesh drawn beside him (the preview only)."""
    from scripts.kmf_render import render
    n = k.anim.frame_count
    az = CAM.get(stem, 35) if az is None else az
    c, sp = _frame_box(k, az, el, range(int(n * FRAME_FROM.get(stem, 0)), n, 2))
    span = span or max(sp * 1.25, .7)
    centre = centre if centre is not None else (c[0], c[1], 0)
    last = n - 1 if k.anim.wrap else n
    frames = []
    for f in range(0, last, step):
        im = render(k, tex, f, az, el, (S, S), centre=centre, span=span, floor=stem not in NO_FLOOR)
        if label:
            ImageDraw.Draw(im).text((8, 6), f"Krefft-{stem}  {f}/{n - 1}", fill=INK)
        frames.append(im)
    return frames


def save_gif(frames, out, ms):
    from scripts.krefft_render import save_gif as sg
    sg(frames, out, ms)


def magic_preview_kmf(B):
    """The Magic clip with the shot flying on and the Coin Spill burst (the preview only; the game file
    carries just the gob between his lips, the game makes the shot)."""
    clip = B.clips["breathe_fire"]
    extra = prop_coin_gob(B.plan, B.bones["breathe_fire"], B.times["breathe_fire"], fly=True)
    return anim_kmf(B.plan, "MagicPreview", B.frames["breathe_fire"], clip.wrap, extra=extra)


def main(out_dir, gif_dir=None, names=None, skins_from=None):
    B = build(names)
    info = export(B, out_dir, skins_from)
    if gif_dir:
        from scripts.krefft_render import textures
        tex = textures(out_dir)
        gif_dir = Path(gif_dir)
        gif_dir.mkdir(parents=True, exist_ok=True)
        for name, d in info.items():
            k = parse_kmf((Path(out_dir) / f"{PREFIX}{d['stem']}.kmf").read_bytes())
            fr = clip_frames(k, tex, d["stem"])
            save_gif(fr, gif_dir / f"Krefft-{d['stem']}.gif", int(round(1000 / d["fps"])))
    return info


if __name__ == "__main__":
    import json
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skins = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--skins=")), args[0])
    print(json.dumps(main(args[0], args[1] if len(args) > 1 else None,
                          args[2].split(",") if len(args) > 2 else None, skins), indent=1))
