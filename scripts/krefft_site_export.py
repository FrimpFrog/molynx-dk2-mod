"""Krefft for the mod site's three.js pack: his game files exported the way the other creatures were
(/home/claude/video2/cgi/tools/export_kmf.py: per clip <name>.json + _pos.bin (float32, frames x vertices x
3, y up, the last frame of each file dropped as for every clip there) + _index.bin (uint32) + _uv.bin
(float32), textures as tex_<name>.png; a walk's ground speed measured from its planted feet), plus his
portrait. Nothing is written but --out (the site agent packs it: tools/pack.py's CLIPS list).

  python3 -m scripts.krefft_site_export <Krefft's built files> <panel dir> <out dir>

Clips (krefft_<clip>): idle1, idle2 (the coin flip), walk, run, melee (the bite), magic (the coin spit as the
game file has it: the gob of coins between his lips, gone at the spit), magic_shot (the same with the coins
flying on and the Coin Spill where they land: the site has no shot system, so this one shows the attack),
gamble, gamble_machine (gamble with the game's Bar_Fruit_Machine in front of him and its lever pulled with
his claw, one mesh), happy (the claw rub), sleep (in the Bile Demon's lair bed), plus pose (the bind,
one frame).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

EXPORTER = Path("/home/claude/video2/cgi/tools/export_kmf.py")
CLIPS = {"idle1": "Idle1", "idle2": "Idle2", "walk": "Walk", "run": "Run", "melee": "Melee", "magic": "Magic",
         "gamble": "Gamble", "happy": "Happy", "sleep": "Sleep", "pose": "PoseFrame"}


def _exporter(out: Path):
    spec = importlib.util.spec_from_file_location("krefft_export_kmf", EXPORTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.W = out                     # write into our folder, not the teaser's data folder
    return mod


def gamble_machine(E, files: Path, out: Path):
    """krefft_gamble + the fruit machine as extra groups, its lever turned per frame (krefft_v5_review)."""
    import scripts.krefft_v5_review as R
    from scripts.krefft_anims import GAMBLE_MACHINE, GAMBLE_YAW
    from scripts.dragon_render import view
    name = "krefft_gamble_machine"
    E.export(str(files / "Krefft-Gamble.kmf"), name, [str(files)])
    info = json.loads((out / f"{name}.json").read_text())
    V, F = info["vertices"], info["frames"]
    pos = np.fromfile(out / f"{name}_pos.bin", "<f4").reshape(F, V, 3)
    idx = np.fromfile(out / f"{name}_index.bin", "<u4")
    uv = np.fromfile(out / f"{name}_uv.bin", "<f4").reshape(-1, 2)
    Ry = view(GAMBLE_YAW, 0)
    off = np.asarray(GAMBLE_MACHINE, float)
    parts0 = R.machine_parts(0.0)
    extra_pos, extra_idx, extra_uv, groups = [], [], [], list(info["groups"])
    start_v, start_i = V, len(idx)
    for P, T, uvp, tex in parts0:
        fn = E.texture(tex, [])
        n = len(P)
        tri = np.asarray(T, int).reshape(-1, 3) + start_v
        extra_idx.append(tri.ravel())
        extra_uv.append(np.asarray(uvp, float))
        if fn:
            groups.append({"tex": fn, "start": start_i, "count": int(tri.size), "alpha": False})
        start_v += n
        start_i += tri.size
    frames = []
    for f in range(F):
        pts = []
        for P, _T, _uv, _tex in R.machine_parts(f / 30.0):
            q = P @ Ry.T + off                  # the review scene's model space (x to his left)
            pts.append(np.stack([-q[:, 0], q[:, 1], q[:, 2]], 1))    # the site's (KMF x, y up, KMF y)
        frames.append(np.concatenate(pts))
    allpos = np.concatenate([pos, np.stack(frames).astype(np.float32)], axis=1)
    (out / f"{name}_pos.bin").write_bytes(allpos.astype("<f4").tobytes())
    (out / f"{name}_index.bin").write_bytes(np.concatenate([idx] + extra_idx).astype("<u4").tobytes())
    (out / f"{name}_uv.bin").write_bytes(np.concatenate([uv] + extra_uv).astype("<f4").tobytes())
    info.update(vertices=int(start_v), groups=groups, min=allpos.reshape(-1, 3).min(0).round(3).tolist(),
                max=allpos.reshape(-1, 3).max(0).round(3).tolist())
    (out / f"{name}.json").write_text(json.dumps(info))
    return info


def main(files, panel, out):
    files, panel, out = Path(files), Path(panel), Path(out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    E = _exporter(data)
    for short, stem in CLIPS.items():
        E.export(str(files / f"Krefft-{stem}.kmf"), f"krefft_{short}", [str(files)])
    # the coin spit with the shot (the review's preview file)
    from scripts.dk2kmf_write import write_kmf
    from scripts.krefft_anims_build import build, magic_preview_kmf
    tmp = out / "_MagicPreview.kmf"
    tmp.write_bytes(write_kmf(magic_preview_kmf(build(["breathe_fire"]))))
    E.export(str(tmp), "krefft_magic_shot", [str(files)])
    tmp.unlink()
    gamble_machine(E, files, data)
    (out / "img").mkdir(exist_ok=True)
    shutil.copy(panel / "krefft-bg.png", out / "img" / "p_krefft.png")
    shutil.copy(panel / "krefft.png", out / "img" / "i_krefft.png")
    skins = out / "skins"
    skins.mkdir(exist_ok=True)
    for n in ("Krefft_Body", "Krefft_Head", "Krefft_Shell"):
        shutil.copy(files / f"{n}.png", skins / f"{n}.png")
    return {p.stem: json.loads(p.read_text()) for p in sorted(data.glob("krefft_*.json"))}


if __name__ == "__main__":
    r = main(*sys.argv[1:4])
    for k, v in r.items():
        print(k, v["frames"], v["vertices"], v["speed"], [g["tex"] for g in v["groups"]][-2:])
