"""Build Krefft's test files: the model as a KMF pose (Krefft-PoseFrame.kmf, his bind pose: standing, head
up a little) and the three skins, through the Dragon's export (scripts.dragon_export.make_plan /
pose_kmf: the Bile Demon's material flags and control points, the same file conventions).

This round is the model and skin only (no clips, no install): nothing here writes anywhere but the folder
it is given.

Scale: the model is made in KMF units (krefft_model), so HEIGHT_RATIO is chosen to export at scale 1:
about 0.28 tall at the top of the shell, 0.88 from snout to hind claws, 0.6 across the feet (the
Salamander's files: 0.41 tall, 0.91 long; the Bile Demon's: 0.54 tall, 0.85 across).

  python3 -m scripts.krefft_build <out dir> [<the game's Meshes dir>]

LICENCE: GPL-3.0.
"""
import json
import sys
from pathlib import Path

import numpy as np

from scripts.dk2kmf_write import write_kmf
from scripts.dragon_build import _donor
from scripts.dragon_export import BILE_IDLE_HEIGHT, make_plan, pose_kmf
from scripts.krefft_model import BODY, HEADT, SHELL, build_krefft, counts
from scripts.dragon_texture import vertex_occlusion
from scripts.krefft_texture import atlas_uv, bake, detail_metric, no_props

PREFIX = "Krefft-"
PART_LABEL = {BODY: "Body", HEADT: "Head", SHELL: "Shell"}
BODY_PARTS = ("shell", "neck", "head", "jaw", "legs")


def meshes_dir():
    from scripts.dk2paths import MESHES_DIR, have
    return MESHES_DIR if have(MESHES_DIR) else Path("/home/claude/dk2data/Meshes")


def build(meshes_dir):
    parts = build_krefft()
    rest = [np.array(p.points) for p in parts]
    allp = np.concatenate(rest)
    ratio = float(allp[:, 1].max() - allp[:, 1].min()) / BILE_IDLE_HEIGHT       # -> scale 1
    plan = make_plan(parts, rest, _donor(meshes_dir, "BileDemon-Idle1"), ratio, uv_of=atlas_uv,
                     part_label=PART_LABEL, body_parts=BODY_PARTS, prefix=PREFIX)
    return plan, parts, rest


def export(out_dir, meshes_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan, parts, rest = build(meshes_dir)
    k = pose_kmf(plan, "PoseFrame", plan.frame(rest))
    (out_dir / f"{PREFIX}PoseFrame.kmf").write_bytes(write_kmf(k))
    skins = bake(parts)
    for name, img in skins.items():
        img.save(out_dir / f"{name}.png")
    c = counts(parts)
    pts = np.concatenate(plan.frame(rest))
    lo, hi = pts.min(0), pts.max(0)
    return {"scale": plan.scale, "triangles": sum(t for _, t in c.values()),
            "parts": {k_: {"vertices": v, "triangles": t} for k_, (v, t) in c.items()},
            "sprites": len(plan.sprites), "materials": [m.name for m in plan.materials],
            "extent_kmf": {"x": float(hi[0] - lo[0]), "y": float(hi[1] - lo[1]), "z": float(hi[2] - lo[2])},
            "skin_detail": {n: round(detail_metric(img), 2) for n, img in skins.items()},
            "skin_detail_used_texels": {n: round(detail_metric(img, parts, n), 2) for n, img in skins.items()}}


STANDING_HEIGHT = 0.58       # version B: about the Bile Demon's idle height (0.54) in KMF units


def export_standing(out_dir, meshes_dir, skins_from, v4=False, v5=False):
    """Version B (scripts.krefft_standing) as Krefft-PoseFrame.kmf in `out_dir`, with A's skins (copied
    from `skins_from`: B uses A's islands unchanged). `v4`: Soles' v4 changes; the head and shell skins
    are then baked from A's own parts as before (so every approved texel stays), with the v4 coin, its
    crack and B's jewellery chain. `v5`: the v4 build with no chain or medallion at all (the head skin's
    chain strip and the shell skin's medallion island are no longer painted) and B's limb joints closed."""
    import copy
    import shutil
    from scripts.krefft_standing import build_standing
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts, _ = build_standing(v4, v5=v5)
    v4 = v4 or v5
    rest = [np.array(p.points) for p in parts]
    allp = np.concatenate(rest)
    ratio = STANDING_HEIGHT / BILE_IDLE_HEIGHT
    if v4:            # the v4 chain lies over the nape a hair above the neck's top: keep v3's scale
        bare = np.concatenate([r for r, p in zip(rest, parts) if p.name != "chain"])
        ratio *= np.ptp(allp[:, 1]) / np.ptp(bare[:, 1])
    plan = make_plan(parts, rest, _donor(meshes_dir, "BileDemon-Idle1"), ratio, uv_of=atlas_uv,
                     part_label=PART_LABEL, body_parts=BODY_PARTS, prefix=PREFIX)
    (out_dir / f"{PREFIX}PoseFrame.kmf").write_bytes(write_kmf(pose_kmf(plan, "PoseFrame", plan.frame(rest))))
    # the body skin baked on B itself: the limbs' painted light follows B's upright arms and legs
    occ = vertex_occlusion(no_props(parts), skip=("coins", "chain"), reach=0.05)     # (shorter reach: under the shell)
    occ = {k: np.nan_to_num(v, nan=1.0) for k, v in occ.items()}
    import scripts.krefft_texture as kt
    keep = kt.LIMB_LIGHT
    if v5:
        kt.LIMB_LIGHT = kt.LIMB_LIGHT_V5
    bake(parts, occlusion={k: 0.75 + 0.25 * v for k, v in occ.items()})[BODY].save(out_dir / f"{BODY}.png")
    kt.LIMB_LIGHT = keep
    if v5:                                             # (after B's body: build_krefft resets the globals)
        baked = bake([p for p in build_krefft(v4=True) if p.name != "chain"])
        for name in (HEADT, SHELL):
            baked[name].save(out_dir / f"{name}.png")
    elif v4:                                           # (after B's body: build_krefft resets the globals)
        chain = next(p for p in parts if p.name == "chain")
        mixed = [chain if p.name == "chain" else p for p in build_krefft(v4=True)]
        medal = copy.copy(next(p for p in parts if p.name == "coins"))
        medal.name = "medal"
        medal.tris = [t for t in medal.tris if medal.island[t[0]] == "medal"]
        baked = bake(mixed, extra=[medal])
        for name in (HEADT, SHELL):
            baked[name].save(out_dir / f"{name}.png")
    else:
        for name in (HEADT, SHELL):                        # the head and the shell: A's skins unchanged
            shutil.copy(Path(skins_from) / f"{name}.png", out_dir / f"{name}.png")
    c = counts(parts)
    pts = np.concatenate(plan.frame(rest))
    lo, hi = pts.min(0), pts.max(0)
    return {"scale": plan.scale, "triangles": sum(t for _, t in c.values()),
            "parts": {k_: t for k_, (v, t) in c.items()},
            "extent_kmf": {"x": float(hi[0] - lo[0]), "y": float(hi[1] - lo[1]), "z": float(hi[2] - lo[2])}}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--standing":
        # python3 -m scripts.krefft_build --standing <out dir> <A's export dir (skins)> [--v4 | --v5]
        print(json.dumps(export_standing(sys.argv[2], meshes_dir(), sys.argv[3], "--v4" in sys.argv,
                                         v5="--v5" in sys.argv), indent=1))
    else:
        info = export(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else meshes_dir())
        print(json.dumps(info, indent=1))
