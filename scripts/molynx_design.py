"""Molynx's approved redesign (2026-11-06), applied to the finished character before it becomes game files.

The user approved the design in review renders made by the design tools (kept outside the repo in
/home/claude/molynx_design, README_DESIGN.md; snapshot in data/molynx_design/painting_tools/). This
module is the same design as code, so molynx_assets.build_source() and everything built from it
carry it. The changes, in the order they are applied:

1. Clips. fidgeting = the Troll's idle2 and celebrating = the Troll's dance (club arm locked,
   mirrored like the Giant's dance), retargeted exactly as behemoth_anims.build_all retargets every
   DK2 clip (same rig, rest pose, amplitudes, 30 fps rule and loop closing). They replace the Giant
   idle2 and the first half of the Giant dance in the clip library, keeping the old body triangle
   set (the dropped stretched triangles are measured on the original library, as approved).
2. Free-hand claw. The claw (vertices 244, 571-574, labelled torso/leg by the rig) moves rigidly
   with the 12 hand vertices it sits on, in every frame (it floated by the hip in the DK2 clips).
3. Mace arm. In the retargeted clips the mace arm (vertices 382-429 and 654-693) follows the club it
   holds, blending back to the retargeted arm near the shoulder; in the clips made with the club
   arm locked (LOCKED) the arm, the club and the mace keep their stand frame 0 place on his upper
   chest, carried by the upper torso (the arm twisted backwards in praying and happy).
4. Head. The MegaGlest Golem's face and dome (rounder top, jutting jaw) replace his face as the part
   `custom_character/golem_head`; a closing band of his own skin (`custom_character/head_neck`)
   runs from the face's edge back into his skull and neck. His horns, the horn roots and the back
   of his skull stay. In every frame both parts move rigidly with his upper skull (Kabsch fit).
5. Skin. The red rock textures of design A2: body (Battlemage belt, no harness, rock cracks,
   graded to the brightness of the real DK2 skins with a gloss highlight, wrist smear repainted),
   head (angry face, bottom teeth over the lip) and neck strip, as PNGs in data/molynx_design/.
   The mace keeps its texture.

Size is not part of the character data: molynx_assets.DESIGN_HEIGHT_RATIO places him in the game
as tall as the Horned Reaper (the top of his head level with the Reaper's skull).

LICENCE: GPL-3.0 (base models: MegaGlest Behemoth and Golem, CC BY-SA 3.0).
"""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "molynx_design"
BODY_TEX, MACE_TEX = "custom_character/body", "custom_character/mace"
HEAD_TEX, NECK_TEX = "custom_character/golem_head", "custom_character/head_neck"
TEXTURE_FILES = {BODY_TEX: "body_A2.png", HEAD_TEX: "head_A2.png", NECK_TEX: "neck_A2.png"}

# ---------------------------------------------------------------- 1. clips
NEW_CLIPS = {  # clip -> (donor creature, donor animation, retarget options), see build_all
    "fidgeting": ("troll", "idle2", {}),
    "celebrating": ("troll", "dance", {"locked": True, "mirror": True}),
}
NEW_CLIP_SOURCES = {"fidgeting": "Troll idle2", "celebrating": "Troll dance"}


def design_clips(meshes_dir, cache, unit_dir) -> dict:
    """{clip: library entry} of the clips the design replaces, made the way build_all makes its own."""
    import warnings
    from scripts.behemoth_anims import CLUB_ARM, CLUB_LOCKED, DK2_FPS, FLIP, MIRROR
    from scripts.behemoth_rig import behemoth_rig
    from scripts.donor_tracks import donor_tracks
    from scripts.golem_anims import _loop_close, _resample
    from scripts.humanoid_retarget import retarget
    rig = behemoth_rig(unit_dir)
    tracks, out = {}, {}
    for name, (creature, anim, opt) in NEW_CLIPS.items():
        if creature not in tracks:
            tracks[creature] = donor_tracks(meshes_dir, cache, creature)
        T = tracks[creature]
        rest = "idle1" if "idle1" in T else "walk"
        flip = MIRROR if opt.get("mirror") else FLIP
        amp = {**CLUB_ARM, **(CLUB_LOCKED if opt.get("locked") else {})}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frames = retarget(rig, T[anim][0] * flip, T[rest][0][0] * flip, amp=amp)
        seconds = len(T[anim][0]) / DK2_FPS
        frames = np.asarray(frames)
        if len(frames) / seconds < 24 or len(frames) / seconds > 36:
            frames = _resample(frames, int(seconds * 30))
        frames = _loop_close(frames)
        out[name] = {"frames": [np.asarray(f) for f in frames], "wrap": True,
                     "fps": round(len(frames) / seconds, 3), "kind": "dk2", "source": NEW_CLIP_SOURCES[name]}
    return out


# ---------------------------------------------------------------- 2-3. claw and mace arm
CLAW = np.array([244, 574, 573, 572, 571])
HAND = np.array([231, 560, 226, 566, 238, 240, 236, 559, 564, 558, 556, 237])
CLUB = np.arange(142)                      # behemoth_rig.CLUB_VERTICES
ARM_N = np.r_[382:430, 654:694]
NATIVE = {"stand", "walk", "attack", "die", "tpose"}
LOCKED = {"eating", "praying", "happy", "celebrating", "victory_celebration", "manufacturing"}


def kabsch(a, b):
    """(R, t) of the rigid motion that best maps the points a onto b (b ~ a @ R.T + t)."""
    ca, cb = a.mean(0), b.mean(0)
    U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def fix_claw(data, rest):
    out = {}
    for name, clip in data.items():
        frames = []
        for fr in clip["frames"]:
            body = np.array(fr[0], float, copy=True)
            R, t = kabsch(rest[HAND], body[HAND])
            body[CLAW] = rest[CLAW] @ R.T + t
            frames.append([body] + list(fr[1:]))
        out[name] = {**clip, "frames": frames}
    return out


def fix_club_arm(data, rest, shoulder):
    d = np.linalg.norm(rest[ARM_N] - shoulder, axis=1)
    w = np.clip((d - 0.25) / 0.45, 0, 1)[:, None]          # 0 at the shoulder, 1 from the elbow on
    out = {}
    for name, clip in data.items():
        if name in NATIVE:
            out[name] = clip
            continue
        frames = []
        for fr in clip["frames"]:
            body = np.array(fr[0], float, copy=True)
            R, t = kabsch(rest[CLUB], body[CLUB])
            body[ARM_N] = (1 - w) * body[ARM_N] + w * (rest[ARM_N] @ R.T + t)
            frames.append([body] + list(fr[1:]))
        out[name] = {**clip, "frames": frames}
    return out


def lock_club_arm(data, weights, bones, shoulder, locked=LOCKED):
    ref = data["stand"]["frames"][0]
    ref_body, ref_mace = np.asarray(ref[0], float), np.asarray(ref[1], float)
    torso = np.nonzero((weights[:, bones.index("torso")] > 0.95) & (ref_body[:, 1] > 2.6))[0]
    torso = np.setdiff1d(torso, np.r_[ARM_N, CLUB])
    d = np.linalg.norm(ref_body[ARM_N] - shoulder, axis=1)
    w = np.clip((d - 0.2) / 0.4, 0, 1)[:, None]
    out = {}
    for name, clip in data.items():
        if name not in locked:
            out[name] = clip
            continue
        frames = []
        for fr in clip["frames"]:
            body = np.array(fr[0], float, copy=True)
            R, t = kabsch(ref_body[torso], body[torso])
            body[ARM_N] = (1 - w) * body[ARM_N] + w * (ref_body[ARM_N] @ R.T + t)
            body[CLUB] = ref_body[CLUB] @ R.T + t
            frames.append([body, ref_mace @ R.T + t] + list(fr[2:]))
        out[name] = {**clip, "frames": frames}
    return out


# ---------------------------------------------------------------- 4. the Golem head
HEAD = dict(scale=0.44, centre=(0.05, 3.62, 0.50), face_cut=0.30, skirt=0.44, taper=1.0)
JAW_TAPER = 0.30


def golem_dir(unit_dir) -> Path:
    return Path(unit_dir).parent / "megaglest_golem"


def golem(unit_dir):
    from scripts.g3d import parse_g3d
    me = parse_g3d((golem_dir(unit_dir) / "models" / "golem_standing.g3d").read_bytes()).meshes[0]
    v = np.asarray(me.vertices[0], float)
    uv = np.asarray(me.uv, float).copy()
    uv[:, 1] = 1 - uv[:, 1]
    return v, np.asarray(me.indices, int).reshape(-1, 3), uv


def molynx_masks(unit_dir):
    """(head, horn) vertex masks of the Behemoth body (rest pose)."""
    from scripts.behemoth_rig import CLUB_VERTICES, behemoth_labels, behemoth_rest
    from scripts.character_paint import HORNS
    rest, _, _ = behemoth_rest(unit_dir)
    lab = behemoth_labels(rest)
    horn = np.zeros(len(rest), bool)
    for a, c in HORNS:
        horn[a:c + 1] = True
    hand = (lab == 1) & (rest[:, 2] > 0.8)
    head = (lab == 1) & (np.arange(len(rest)) >= CLUB_VERTICES) & ~hand & ~horn
    return head, horn


def reshape_head(gv):
    """A rounder, less pointed top and a heavier jaw that juts forward (Golem units, before scaling)."""
    gv = gv.copy()
    y, x = gv[:, 1], np.abs(gv[:, 0])
    for i in np.nonzero(y > 2.05)[0]:
        xi, front = x[i], gv[i, 2] > 0.6
        if xi < 0.1:
            gv[i, 1] = 2.27 if front else 2.33
        elif xi < 0.3:
            gv[i, 1] = 2.24
        else:
            gv[i, 1] = 2.15 if gv[i, 2] < 0.6 else 2.14
    jaw = y < 1.235
    w = np.clip((1.235 - y) / 0.30, 0, 1)
    gv[jaw, 2] += 0.13 * w[jaw] * np.clip(1.2 - x[jaw] / 0.6, 0.2, 1)
    gv[jaw, 1] -= 0.05 * w[jaw]
    return gv


def golem_head(unit_dir):
    """The Golem's face and dome: (vertices, triangles, uv), reshaped."""
    v, tri, uv = golem(unit_dir)
    sel = np.all((v[tri][:, :, 2] > 0.35) & (v[tri][:, :, 1] > 0.9) & (np.abs(v[tri][:, :, 0]) < 0.62), axis=1)
    t = tri[sel]
    keep = np.unique(t)
    remap = -np.ones(len(v), int)
    remap[keep] = np.arange(len(keep))
    return reshape_head(v[keep]), remap[t], uv[keep]


def boundary_edges(tri):
    count = Counter()
    for a, b, c in tri:
        for e in ((a, b), (b, c), (c, a)):
            count[tuple(sorted(e))] += 1
    return [e for a, b, c in tri for e in ((a, b), (b, c), (c, a)) if count[tuple(sorted(e))] == 1]


def rim_loops(tri):
    """Open edges chained into closed loops, following the triangles' winding."""
    nxt = {a: b for a, b in boundary_edges(tri)}
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            seen.add(cur)
            loop.append(cur)
            cur = nxt[cur]
        loops.append(loop)
    return loops


def skirt_geometry(v, tri, depth, taper, scale):
    """The band that closes the face into a head, as its own vertices and triangles: every rim loop
    extruded back (-z) by `depth`, narrowing towards the head's centre, turning in under the jaw and
    sloping down at the back. UV: u = distance round the rim (0..1 over all loops), v = 0 at the
    face's edge, 1 at the back. -> (points, triangles, uv)"""
    cx, cy = v[:, 0].mean(), v[:, 1].mean()
    pts, tris, uvs, total = [], [], [], 0.0
    for loop in rim_loops(tri):
        ring = loop + [loop[0]]                      # the seam vertex twice (u = 0 and u = loop length)
        front = v[ring].copy()
        back = front.copy()
        jawk = np.clip((1.30 - front[:, 1]) / 0.30, 0, 1)
        tp = taper - JAW_TAPER * jawk
        back[:, 2] -= depth * (1 - 0.35 * jawk)
        back[:, 0] = cx + (back[:, 0] - cx) * tp
        back[:, 1] = cy + (back[:, 1] - cy) * (0.5 + 0.5 * tp) + 0.10 * jawk
        back[:, 1] -= 0.30 * np.clip((front[:, 1] - 1.85) / 0.35, 0, 1)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(front, axis=0), axis=1))] * scale
        base = sum(len(p) for p in pts)
        n = len(ring)
        pts += [front, back]
        uvs += [np.stack([total + seg, np.zeros(n)], 1), np.stack([total + seg, np.ones(n)], 1)]
        for k in range(n - 1):
            f0, f1, b0, b1 = base + k, base + k + 1, base + n + k, base + n + k + 1
            tris += [(f1, f0, b0), (f1, b0, b1)]
        total += seg[-1] + 0.05
    uv = np.vstack(uvs)
    return np.vstack(pts), np.array(tris), np.stack([uv[:, 0] / total, uv[:, 1]], 1)


def add_head(data, unit_dir, p=HEAD):
    """Every clip with his face taken off and the Golem head and its neck band added as two parts
    that follow his upper skull rigidly."""
    head, horn = molynx_masks(unit_dir)
    gv, gt, guv = golem_head(unit_dir)
    sv, st, suv = skirt_geometry(gv, gt, p["skirt"] / p["scale"], p["taper"], p["scale"])
    c = (gv.min(0) + gv.max(0)) / 2
    face_pl = (gv - c) * p["scale"] + np.array(p["centre"])
    skirt_pl = (sv - c) * p["scale"] + np.array(p["centre"])
    ref = np.asarray(data["stand"]["frames"][0][0])
    face = head & ~(ref[:, 2] < p["face_cut"])       # his face; the back of his skull stays
    fit = head & (ref[:, 1] > 3.6)                    # the upper skull moves rigidly (the neck bends)
    out = {}
    for name, clip in data.items():
        parts = []
        for k, part in enumerate(clip["parts"]):
            tri = np.asarray(part["indices"], int).reshape(-1, 3)
            if part["texture"] == BODY_TEX and k == 0:
                tri = tri[~np.any(face[tri], axis=1) | np.any(horn[tri], axis=1)]   # horn roots stay
            parts.append({**part, "indices": tri.reshape(-1).astype(np.uint16)})
        parts.append({"texture": HEAD_TEX, "uv": guv.astype(np.float32), "indices": gt.reshape(-1).astype(np.uint16)})
        parts.append({"texture": NECK_TEX, "uv": suv.astype(np.float32), "indices": st.reshape(-1).astype(np.uint16)})
        frames = []
        for fr in clip["frames"]:
            R, t = kabsch(ref[fit], np.asarray(fr[0])[fit])
            frames.append(list(fr) + [face_pl @ R.T + t, skirt_pl @ R.T + t])
        out[name] = {**clip, "parts": parts, "frames": frames}
    return out


# ---------------------------------------------------------------- 5. skin, and all together
def design_textures(textures: dict) -> dict:
    """His textures with the design's body, head and neck skins (the mace keeps its own)."""
    out = dict(textures)
    for key, name in TEXTURE_FILES.items():
        out[key] = Image.open(DATA_DIR / name).convert("RGB")
    return out


def apply_design(data: dict, textures: dict, unit_dir) -> tuple[dict, dict]:
    """Steps 2-5 on a built character (step 1 happens in the clip library, see build_source)."""
    from scripts.behemoth_rig import behemoth_rest, behemoth_rig
    rest, _, _ = behemoth_rest(unit_dir)
    rig = behemoth_rig(unit_dir)
    shoulder = np.asarray(rig.joints["armN"], float)
    data = fix_club_arm(fix_claw(data, rest), rest, shoulder)
    data = lock_club_arm(data, np.asarray(rig.weights), list(rig.bones), shoulder)
    return add_head(data, unit_dir), design_textures(textures)
