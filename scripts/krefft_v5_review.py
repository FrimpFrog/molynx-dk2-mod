"""Review renders of Krefft v5 (standing, no chain, joints closed, full clip set) from his exported game
files, with the Dragon's software renderer (scripts.dragon_render.Canvas: orthographic, z-buffered, back
faces culled the way the game culls them, so a wrongly wound triangle shows as a hole), in a DK2-like light:
a claimed-floor tile (the game's T_Claimed_Top2) under him, a warm torch-lit key light and a dark dungeon
background, a mid-high camera.

  python3 -m scripts.krefft_v5_review <export dir> <v4 export dir> <out dir>
writes, into <out dir>:
  Krefft-<Stem>.gif            one per clip, at the clip's own speed (Sleep at 15 fps)
  Krefft-MagicPreview.gif      the coin spit with the shot flying on (the game's Coin mesh) and the Coin Spill
  Krefft-Gamble.gif            at the game's Bar_Fruit_Machine (the machine is the preview's only)
  krefft_v5_poses.png          the six static poses
  krefft_v5_all_clips.png      the contact sheet: every clip, eight frames each
  krefft_v5_turntable.gif      the PoseFrame turning, dipping to DK2's high camera
  krefft_v4_vs_v5.png          v4 above, v5 below, the same angles
  joints_before_after.png      each joint close up, v4 bind / v5 bind / v5 in its hardest animation frame

LICENCE: GPL-3.0.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf
from scripts.dragon_render import Canvas, view
from scripts.kmf_render import kmf_parts

BG = (14, 12, 16)
BG_CHECK = (70, 70, 80)
INK = (235, 220, 140)
WARM = np.array([1.0, .9, .74])            # torchlight
LIGHT = (0.35, 0.8, 0.5)
FLOOR_TEX = "T_Claimed_Top2"
TILE = 1.0                                  # a DK2 tile is one KMF unit across
MESHES = Path("/home/claude/dk2data/Meshes")
EL = 30


def textures(export_dir):
    from scripts.krefft_render import textures as t
    return t(export_dir)


def load(d, stem):
    return parse_kmf((Path(d) / f"{stem}.kmf").read_bytes())


def scene(models, tex, az, el, size, centre, span, floor=True, tiles=1, bright=False):
    """models: [(Kmf, frame or None, (dx, dy, dz) model-space offset)]."""
    c = Canvas(size, centre=centre, span=span, bg=BG)
    R = view(az, el)
    if floor:
        ft = tex(FLOOR_TEX)
        g = ft.mean(2, keepdims=True)
        ft = (ft * .55 + g * .45) * .5         # in the shade of the room: the creature reads over it
        for i in range(-tiles, tiles + 1):
            for j in range(-tiles, tiles + 1):
                x0, z0 = (i - .5) * TILE, (j - .5) * TILE
                P = np.array([(x0, 0, z0), (x0 + TILE, 0, z0), (x0 + TILE, 0, z0 + TILE), (x0, 0, z0 + TILE)], float)
                T = np.array([(0, 2, 1), (0, 3, 2)])
                uv = np.array([(0, 0), (1, 0), (1, 1), (0, 1.0)])
                c.draw(P, T, R, uv=uv, tex=ft, cull=False, light=LIGHT, ambient=.55)
    for m in models:
        k, frame, off = m[:3]
        Ry = view(m[3], 0) if len(m) > 3 else np.eye(3)        # a prop's own turn about the vertical
        parts = k if isinstance(k, list) else kmf_parts(k, frame)     # (or parts already posed: the lever)
        for P, T, uv, name in parts:
            c.draw(P @ Ry.T + np.asarray(off, float), T, R, uv=uv, tex=tex(name), light=LIGHT, ambient=.55 if bright else .5)
    img = np.asarray(c.image().convert("RGB"), float)
    if bright:                    # close-ups for checking the mesh: no torch tint or vignette
        return Image.fromarray(np.clip(img * 1.1, 0, 255).astype(np.uint8))
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - .3 * (((xx - w / 2) / w) ** 2 + ((yy - h * .55) / h) ** 2) * 2.2
    img = img * WARM[None, None] * np.clip(vig, .45, 1)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def frame_box(k, az, el, frames, extra=(), sprites=None):
    R = view(az, el)
    pts = [np.concatenate(evaluate_anim_frame(k.anim, f)[:sprites]) for f in frames]
    pts = np.concatenate(pts + [np.asarray(e) for e in extra])
    q = np.stack([-pts[:, 0], -pts[:, 2], pts[:, 1]], 1) @ R.T
    lo, hi = q.min(0), q.max(0)
    return (lo + hi) / 2, float(max(hi[:2] - lo[:2]))


CAM = {"Walk": 62, "WalkStart": 62, "WalkEnd": 62, "Run": 62, "Walkback": 62, "WalkbackStart": 62,
       "WalkbackEnd": 62, "Drunk": 50, "Melee": 50, "Melee2": 40, "Magic": 30, "MagicPreview": 30, "Fallback": 70,
       "GetUp": 70, "Die": 70, "Torture": 20, "Swing": 30, "Drink": 45, "Eat": 25, "Idle2": 25,
       "Happy": 25, "Gamble": 80, "Pray": 50, "PrayStart": 50, "PrayEnd": 50}
NO_FLOOR = {"Swing"}                     # hangs below the Keeper's hand at the file's origin
FRAME_FROM = {"Entrance": .3}            # framed on the end (the start is below the floor)
FPS = {"Sleep": 15}
BODY_SPRITES = {}                        # (the coin spit preview is framed with its flying coin)


SPARKLE_SIZE = .11                       # the Gold Sparkle element's scale 819 / 4096 of a tile, about half
SPARKLE_LIFE = 16                        # its 16 frames (GSparkle0..15) and 16-tick life


def _sparkles():
    """The game's Gold Sparkle sprite frames (EngineTextures GSparkle0..15), as float RGB."""
    from scripts.dk2texture import TextureCache
    from scripts.krefft_render import GAME_TEXTURES
    if "gs" not in _SPARK:
        c = TextureCache(Path(GAME_TEXTURES + ".dir"), Path(GAME_TEXTURES + ".dat"))
        _SPARK["gs"] = [np.asarray(c.get(f"GSparkle{i}MM0").convert("RGB"), np.float32) for i in range(16)]
    return _SPARK["gs"]


_SPARK = {}


def sparkle_trail(img, k, f, az, el, centre, span, first=33, last=49, body=7):
    """The Coin Spit preview's Gold Sparkle trail (effect 34, the shot's generalEffectId): one sparkle left
    where the flying coin (sprite `body`, the first after his own) was on each earlier frame, playing the
    game's 16 GSparkle frames, added onto the picture (the game draws them additive)."""
    if f <= first:
        return img
    frames = _sparkles()
    R = view(az, el)
    W = img.size[0]
    a = np.asarray(img, np.float32).copy()
    rng = np.random.default_rng(7)
    for j in range(max(first + 1, f - SPARKLE_LIFE + 1), min(f, last) + 1):
        pts = evaluate_anim_frame(k.anim, j)[body]
        P = np.asarray(pts, float).mean(0)
        if np.ptp(np.asarray(pts, float), axis=0).max() < .05:      # hidden (scaled to a point): no shot
            continue
        jit = rng.uniform(-.03, .03, 3)
        q = R @ (np.array([-P[0], -P[2], P[1]]) + jit)
        x = W / 2 + (q[0] - centre[0]) / span * W
        y = W / 2 - (q[1] - centre[1]) / span * W
        age = f - j
        spr = frames[min(age, 15)]
        n = max(4, int(SPARKLE_SIZE / span * W))
        tile = np.asarray(Image.fromarray(spr.astype(np.uint8)).resize((n, n), Image.BILINEAR), np.float32)
        x0, y0 = int(x - n / 2), int(y - n / 2)
        xs, ys = slice(max(x0, 0), min(x0 + n, W)), slice(max(y0, 0), min(y0 + n, W))
        if xs.start >= xs.stop or ys.start >= ys.stop:
            continue
        a[ys, xs] += tile[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] * (1 - age / 20)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def clip_frames(k, tex, stem, S=360, step=1, az=None, el=EL, backdrop=None, label=True):
    n = k.anim.frame_count
    az = CAM.get(stem, 35) if az is None else az
    ext = []
    if backdrop is not None:              # frame the machine with him (its KMF points, moved like the scene)
        o = np.asarray(backdrop[1], float)
        ext = [np.concatenate(list(mesh_points(backdrop[0]))) + np.array([-o[0], o[2], -o[1]])]
    c, sp = frame_box(k, az, el, range(int(n * FRAME_FROM.get(stem, 0)), n, 2), ext, BODY_SPRITES.get(stem))
    span = max(sp * 1.15, .75)
    last = n - 1 if k.anim.wrap else n
    out = []
    for f in range(0, last, step):
        models = [(k, f, (0, 0, 0))]
        if backdrop is not None:
            models.append(machine_model(f))
        im = scene(models, tex, az, el, (S, S), (c[0], c[1], 0), span, floor=stem not in NO_FLOOR)
        if stem == "MagicPreview":
            im = sparkle_trail(im, k, f, az, el, (c[0], c[1], 0), span)
        if label:
            ImageDraw.Draw(im).text((8, 6), f"Krefft-{stem}  {f}/{n - 1}", fill=INK)
        out.append(im)
    return out


def save_gif(frames, out, ms):
    from scripts.krefft_render import save_gif as sg
    sg(frames, out, ms)


def fruit_machine():
    return load(MESHES, "Bar_Fruit_Machine")


from scripts.krefft_anims import GAMBLE_MACHINE, GAMBLE_YAW, LEVER_HUB_UP, LEVER_REST, lever_angle   # noqa: E402


def machine_parts(t):
    """The fruit machine's parts with its lever turned to lever_angle(t) (the preview's only: the game's
    machine is a static mesh). In the machine's own model space (before its turn) the lever is the
    geometry outside its side panel (x > 0.207) more than 0.07 from the hub's axis (the hub disc stays);
    it turns about the x axis through the hub (y 0.314, z 0), the knob 30.4 degrees above level."""
    d = np.radians(lever_angle(t) - LEVER_REST)
    c, s_ = np.cos(d), np.sin(d)
    out = []
    for P, T, uv, name in kmf_parts(fruit_machine()):
        P = P.copy()
        y, z = P[:, 1] - LEVER_HUB_UP, P[:, 2]
        sel = (P[:, 0] > .207) & (np.hypot(y, z) > .07)
        # (z, up) turned by d: up = r sin(a), z = r cos(a)
        z2, y2 = z * c - y * s_, z * s_ + y * c
        P[sel, 1] = y2[sel] + LEVER_HUB_UP
        P[sel, 2] = z2[sel]
        out.append((P, T, uv, name))
    return out


def machine_model(frame, fps=30):
    return (machine_parts(frame / fps), None, GAMBLE_MACHINE, GAMBLE_YAW)


def all_gifs(export_dir, out_dir, tex, extra_kmfs=()):
    out_dir = Path(out_dir)
    stems = sorted(p.stem.replace("Krefft-", "") for p in Path(export_dir).glob("Krefft-*.kmf")
                   if load(export_dir, p.stem).kind == "anim")
    made = []
    for stem in stems:
        k = load(export_dir, f"Krefft-{stem}")
        bd = (fruit_machine(), GAMBLE_MACHINE) if stem == "Gamble" else None
        fr = clip_frames(k, tex, stem, backdrop=bd)
        save_gif(fr, out_dir / f"Krefft-{stem}.gif", int(round(1000 / FPS.get(stem, 30))))
        made.append(stem)
    for stem, k in extra_kmfs:
        fr = clip_frames(k, tex, stem)
        save_gif(fr, out_dir / f"Krefft-{stem}.gif", 33)
        made.append(stem)
    return made


def contact_sheet(export_dir, out_png, tex, extra_kmfs=(), n=8, S=150):
    stems = sorted((p.stem.replace("Krefft-", "") for p in Path(export_dir).glob("Krefft-*.kmf")
                    if load(export_dir, p.stem).kind == "anim"))
    rows = [(s, load(export_dir, f"Krefft-{s}")) for s in stems] + list(extra_kmfs)
    LW = 120
    img = Image.new("RGB", (LW + S * n, S * len(rows)), BG)
    d = ImageDraw.Draw(img)
    for r, (stem, k) in enumerate(rows):
        F = k.anim.frame_count
        fs = np.linspace(0, F - (2 if k.anim.wrap else 1), n).round().astype(int)
        az = CAM.get(stem, 35)
        c, sp = frame_box(k, az, EL, range(int(F * FRAME_FROM.get(stem, 0)), F, 2), sprites=BODY_SPRITES.get(stem))
        span = max(sp * 1.3, .75)
        bd = stem == "Gamble"
        if bd:
            span = max(span, 1.25)
        for i, f in enumerate(fs):
            im = scene([(k, int(f), (0, 0, 0))] + ([machine_model(int(f))] if bd else []), tex, az, EL, (S, S), (c[0], c[1], 0), span,
                       floor=stem not in NO_FLOOR)
            img.paste(im, (LW + i * S, r * S))
        d.text((6, r * S + 6), stem, fill=INK)
        d.text((6, r * S + 22), f"{F} frames", fill=(170, 160, 120))
        d.text((6, r * S + 36), "loop" if k.anim.wrap else "once", fill=(170, 160, 120))
    img.save(out_png)


def poses_png(export_dir, out_png, tex, S=300):
    stems = ["PoseFrame", "DiePose", "StunnedPose", "DraggedPose", "ElecPose", "DejectedPose"]
    img = Image.new("RGB", (S * len(stems), S + 22), BG)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(stems):
        k = load(export_dir, f"Krefft-{s}")
        pts = np.concatenate(list(mesh_points(k)))
        R = view(35, EL)
        q = np.stack([-pts[:, 0], -pts[:, 2], pts[:, 1]], 1) @ R.T
        c = (q.min(0) + q.max(0)) / 2
        span = max(float(max(q.max(0)[:2] - q.min(0)[:2])) * 1.3, .75)
        img.paste(scene([(k, None, (0, 0, 0))], tex, 35, EL, (S, S), (c[0], c[1], 0), span,
                        floor=s != "DraggedPose"), (i * S, 22))
        d.text((i * S + 6, 5), s, fill=INK)
    img.save(out_png)


def turntable(export_dir, out_gif, tex, S=420, label="Krefft v5 (no chain)"):
    k = load(export_dir, "Krefft-PoseFrame")
    frames = []
    for i in range(36):
        az = i * 10
        el = 14 + 31 * (0.5 - 0.5 * np.cos(2 * np.pi * i / 36))
        im = scene([(k, None, (0, 0, 0))], tex, az, el, (S, S), (0, .27 * (1 - el / 90) - .02, 0), 1.0)
        ImageDraw.Draw(im).text((8, 6), label, fill=INK)
        frames.append(im)
    save_gif(frames, out_gif, 100)


def compare(v4_dir, v5_dir, out_png, S=340):
    tiles = [(0, 10, "front"), (35, 18, "3/4 front"), (90, 8, "side"), (210, 22, "3/4 back"), (30, 50, "DK2's high camera")]
    img = Image.new("RGB", (S * len(tiles), (S + 24) * 2), BG)
    d = ImageDraw.Draw(img)
    for r, (folder, lab) in enumerate(((v4_dir, "v4 (approved)"), (v5_dir, "v5: no chain or medallion, joints closed"))):
        tex = textures(folder)
        k = load(folder, "Krefft-PoseFrame")
        for i, (az, el, t) in enumerate(tiles):
            img.paste(scene([(k, None, (0, 0, 0))], tex, az, el, (S, S), (0, .27 * (1 - el / 90) - .02, 0), 1.0),
                      (i * S, r * (S + 24) + 24))
            d.text((i * S + 6, r * (S + 24) + 6), f"{lab} - {t}" if i == 0 else t, fill=INK)
    img.save(out_png)


# the joints: (name, bone for the centre, a v5 clip and frame where it is bent hardest, camera az, el)
JOINTS = [("wrist L (Soles' circle)", "handFL", "Happy", 60, 95, -35), ("wrist R (Soles' circle)", "handFR", "Drink", 30, -95, -30),
          ("wrist L in the hand (Swing)", "handFL", "Swing", 40, 70, 0), ("wrist R on his back (Die)", "handFR", "Die", 78, -60, 40),
          ("elbow / shoulder L", "foreFL", "Dance", 50, 50, 10), ("shoulder R into the shell", "upperFR", "Melee2", 13, -60, 15),
          ("hip L into the shell", "thighL", "Walk", 16, 60, 5), ("knee R", "shinR", "Angry", 13, -45, 12),
          ("ankle L into the foot", "footL", "Walk", 30, 80, 8), ("hip / knee on his back (Fallback)", "thighR", "Fallback", 32, -50, 40),
          ("legs on the wheel (Torture)", "shinL", "Torture", 45, 40, 20), ("neck into the shell", "neck1", "Melee", 16, 75, 15),
          ("head / jaw", "jaw", "Magic", 35, 35, 10), ("tail", "tail1", "Walk", 20, 160, 10),
          ("plastron edge / bridge", "root", "Fallback", 32, 30, 60), ("hands from below", "handFL", "Dance", 40, 30, -35)]


def joints(v4_dir, v5_dir, out_png, S=240):
    from scripts import krefft_rig as KR
    from scripts.krefft_anims_build import plan
    from scripts.krefft_anims import CLIPS
    from scripts.krefft_anims_build import sample
    pl = plan()
    sk = KR.model()[1]
    tex4, tex5 = textures(v4_dir), textures(v5_dir)
    k4, k5 = load(v4_dir, "Krefft-PoseFrame"), load(v5_dir, "Krefft-PoseFrame")
    stem_clip = {mk().stem: mk for mk in CLIPS.values()}
    img = Image.new("RGB", (S * 3 + 200, (S + 4) * len(JOINTS) + 24), BG)
    d = ImageDraw.Draw(img)
    for c, t in enumerate(("v4 (before)", "v5 bind (after)", "v5 bent hardest (after)")):
        d.text((200 + c * S + 6, 5), t, fill=INK)
    for r, (lab, bone, stem, f, az, el) in enumerate(JOINTS):
        y = 24 + r * (S + 4)
        i = sk.index[bone]
        p = sk.head[i]
        kp = pl.to_kmf([p[None]])[0][0]
        cb = np.array([-kp[0], -kp[2], kp[1]])
        ka = load(v5_dir, f"Krefft-{stem}")
        clip = stem_clip[stem]()
        spts, bones, _ = sample(clip)
        ff = min(f, len(bones) - 1)
        bp = bones[ff][1][i]
        kq = pl.to_kmf([bp[None]])[0][0]
        # the file's own placement (the hand's hold, the bed, the prayer's place, the wheel): the rig's frame
        # against the file's frame, so the crop follows what the game loads
        ours = np.concatenate(pl.frame(spts[ff])).mean(0)
        theirs = np.concatenate(evaluate_anim_frame(ka.anim, ff)[:len(pl.sprites)]).mean(0)
        kq = kq + (theirs - ours)
        ca = np.array([-kq[0], -kq[2], kq[1]])
        R = view(az, el)
        for col, (k, tex, fr, cen) in enumerate(((k4, tex4, None, cb), (k5, tex5, None, cb), (ka, tex5, f, ca))):
            cc = R @ cen
            im = scene([(k, fr, (0, 0, 0))], tex, az, el, (S, S), (cc[0], cc[1], 0), .42, floor=False, bright=True)
            img.paste(im, (200 + col * S, y))
        d.text((6, y + 6), lab, fill=INK)
        d.text((6, y + 24), f"anim: {stem} {f}", fill=(170, 160, 120))
        d.text((6, y + 40), f"view az {az} el {el}", fill=(120, 115, 95))
    img.save(out_png)


STRIPS = [("bite", "Melee", [0, 4, 9, 13, 15, 16, 18, 21, 24, 30, 38, 45], 55, 25),
          ("bite_high_camera", "Melee", [0, 9, 13, 15, 18, 21, 30], 30, 50),
          ("coin_spit", "MagicPreview", [0, 12, 21, 28, 31, 33, 35, 40, 46, 52, 64], 30, 25),
          ("walk_side", "Walk", list(range(0, 40, 4)), 90, 10),
          ("walk_high_camera", "Walk", list(range(0, 40, 5)), 30, 50),
          ("fall_and_get_up", "Fallback", [0, 6, 12, 16, 20, 26, 32], 70, 30),
          ("get_up", "GetUp", [0, 8, 16, 26, 34, 40, 44, 50, 56, 62], 70, 30),
          ("die", "Die", [0, 10, 20, 30, 36, 42, 50, 60, 70, 78], 70, 30),
          ("claw_rub", "Happy", [0, 12, 24, 40, 60, 80, 100, 110, 120, 129], 20, 20),
          ("angry_stomp", "Angry", [0, 4, 8, 12, 16, 20, 24, 30, 36, 44], 35, 25),
          ("sleep", "Sleep", [0, 30, 55, 60, 66, 75, 119], 35, 30),
          ("torture", "Torture", [0, 15, 30, 45, 60, 75, 90, 99], 20, 25),
          ("entrance", "Entrance", [0, 15, 30, 45, 55, 70, 85, 100, 115, 125, 147], 35, 30),
          ("coin_flip", "Idle2", [0, 15, 30, 33, 40, 48, 56, 66, 80, 100], 25, 20),
          ("gamble", "Gamble", [0, 8, 14, 18, 22, 25, 28, 34, 45, 70, 80, 90], 80, 30),
          ("idle_shrug", "Idle1", [0, 30, 40, 50, 60, 75, 95, 110, 130, 150, 160, 189], 35, 25),
          ("swing", "Swing", [0, 10, 20, 30, 40, 50], 30, 20),
          ("eat_drink", "Eat", [0, 12, 21, 25, 36, 52, 62, 70, 79], 25, 20),
          ("drink", "Drink", [0, 18, 36, 60, 100, 112, 120, 132, 149], 45, 20),
          ("pray", "PrayStart", [0, 15, 30, 45, 55], 50, 25),
          ("dance", "Dance", [0, 10, 20, 30, 40, 50, 60, 70, 80, 94], 25, 25)]


def strips(export_dir, out_dir, tex, extra=(), S=220):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = dict(extra)
    for name, stem, fs, az, el in STRIPS:
        k = extra.get(stem) or load(export_dir, f"Krefft-{stem}")
        F = k.anim.frame_count
        fs = [min(f, F - 1) for f in fs]
        c, sp = frame_box(k, az, el, range(int(F * FRAME_FROM.get(stem, 0)), F, 2), sprites=BODY_SPRITES.get(stem))
        span = max(sp * 1.2, .75)
        bd = stem == "Gamble"
        if bd:
            o = np.asarray(GAMBLE_MACHINE, float)
            ext = [np.concatenate(list(mesh_points(fruit_machine()))) + np.array([-o[0], o[2], -o[1]])]
            c, sp = frame_box(k, az, el, range(0, F, 2), ext)
            span = sp * 1.1
        img = Image.new("RGB", (S * len(fs), S), BG)
        for i, f in enumerate(fs):
            im = scene([(k, f, (0, 0, 0))] + ([machine_model(f)] if bd else []), tex, az, el, (S, S), (c[0], c[1], 0), span, floor=stem not in NO_FLOOR)
            if stem == "MagicPreview":
                im = sparkle_trail(im, k, f, az, el, (c[0], c[1], 0), span)
            ImageDraw.Draw(im).text((6, 4), f"{stem} {f}", fill=INK)
            img.paste(im, (i * S, 0))
        img.save(out_dir / f"{name}.png")


def torture_vs_donor(export_dir, out_png, tex, S=260):
    """The Bile Demon's own Bil-Torture and Krefft's Torture, same camera and framing: he hangs where the
    Bile Demon hangs (the room's wheel is a room object, not in either file)."""
    from scripts.kmf_render import Textures
    bd = load(MESHES, "Bil-Torture")
    kt = load(export_dir, "Krefft-Torture")
    fs = [0, 30, 60, 90]
    c, sp = frame_box(bd, 20, 25, range(0, bd.anim.frame_count, 4))
    img = Image.new("RGB", (S * len(fs), S * 2 + 20), BG)
    d = ImageDraw.Draw(img)
    for r, (k, lab) in enumerate(((bd, "Bile Demon Bil-Torture (the game's)"), (kt, "Krefft-Torture"))):
        for i, f in enumerate(fs):
            im = scene([(k, f, (0, 0, 0))], tex, 20, 25, (S, S), (c[0], c[1], 0), sp * 1.3)
            ImageDraw.Draw(im).text((6, 4), f"{lab} {f}", fill=INK)
            img.paste(im, (i * S, 20 + r * S))
    img.save(out_png)


def main(export_dir, v4_dir, out_dir, what=("gifs", "sheet", "poses", "turntable", "compare", "joints", "strips")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = textures(export_dir)
    extra = []
    if "gifs" in what or "sheet" in what:
        from scripts.krefft_anims_build import build, magic_preview_kmf
        B = build(["breathe_fire"])
        extra = [("MagicPreview", magic_preview_kmf(B))]
    if "gifs" in what:
        all_gifs(export_dir, out_dir, tex, extra)
    if "sheet" in what:
        contact_sheet(export_dir, out_dir / "krefft_v5_all_clips.png", tex, extra)
    if "poses" in what:
        poses_png(export_dir, out_dir / "krefft_v5_poses.png", tex)
    if "turntable" in what:
        turntable(export_dir, out_dir / "krefft_v5_turntable.gif", tex)
    if "compare" in what:
        compare(v4_dir, export_dir, out_dir / "krefft_v4_vs_v5.png")
    if "joints" in what:
        joints(v4_dir, export_dir, out_dir / "joints_before_after.png")
    if "strips" in what:
        if not extra:
            from scripts.krefft_anims_build import build, magic_preview_kmf
            extra = [("MagicPreview", magic_preview_kmf(build(["breathe_fire"])))]
        strips(export_dir, out_dir / "strips", tex, extra)
        torture_vs_donor(export_dir, out_dir / "strips" / "torture_vs_bile_demon.png", tex)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    only = next((x.split("=", 1)[1].split(",") for x in sys.argv if x.startswith("--only=")), None)
    main(a[0], a[1], a[2], *( [only] if only else []))
