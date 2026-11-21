"""Build Krefft (creature 54) as an install folder for the Soles Creatures Mod, the way the cast went in
(cast_install), WITHOUT installing anything: every file is written under --work only, for Soles to approve
and copy himself.

  python3 -m scripts.krefft_install --game "<staged DK2 folder>" --base <MoreCreatures as on the PC now> \\
      --files <Krefft's built files> --work <folder>

<files>                            his .kmf files and three skins (krefft_anims_build; Krefft-*.kmf, Krefft_*.png)
<work>/panel/krefft-bg.png, krefft.png      his panel portrait (64 x 128) and icon (64 x 64), cast_portraits
<work>/install/MoreCreatures/...            the files to copy into flame\\resources\\MoreCreatures (only
                                            those that change or are new), with install_list.txt,
                                            backup_list.txt and report.json beside it
<work>/check.txt                            cast_check over the folder as it would be, with Krefft

What changes (cast_install.install with Krefft alone, then the extra steps here):
  * his record (krefft_record.KREFFT, built from each table's own Bile Demon) spliced in as creature 54
    after the Gargoyle (53) in Data/editor/Creatures.kwd and every map's Globals.kld;
  * Coin Spit (krefft_coin_spit): CreatureSpells 35 and Shots 48 rewritten in every Globals.kld and in
    Data/editor/CreatureSpells.kwd and Shots.kwd (the folder's own copies);
  * his four texts appended to Text.str; his files, skins and panel pictures;
  * Keeper 1's creature pool and availability on Dead Walking, Furnace and My Pet Dungeon 1 (mpd1) gain
    him, and he stands by Keeper 1's heart on all three (krefft_record.PLACEMENTS, MPD1_TILE). No other
    map's portal offers him until Soles says so.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", type=Path, required=True, help="the staged Dungeon Keeper 2 folder (read only)")
    p.add_argument("--base", type=Path, required=True, help="the MoreCreatures folder as it is on the PC now")
    p.add_argument("--files", type=Path, required=True, help="Krefft's built files")
    p.add_argument("--meshes", type=Path, default=Path("/home/claude/dk2data/Meshes"), help="the game's meshes")
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args(argv)
    import scripts.cast_check as CC
    from scripts.cast_install import (MAPS, MPD1, PLACEMENTS, install, loaded_copy, map_install, write_lists)
    from scripts.cast_portraits import make
    from scripts.cast_records import CAST
    from scripts.dk2texture import TextureCache
    from scripts.krefft_coin_spit import EDITOR_FILES, apply_coin_spit, coin_art
    from scripts.krefft_record import KREFFT, KREFFT_ID, MPD1_TILE, PLACEMENTS as MINE

    game, base, work = a.game, a.base, a.work
    for d in (game, base):
        if work.resolve().is_relative_to(d.resolve()):
            raise ValueError(f"--work is inside {d}: nothing may be written there")
    data = game / "Data"
    codepage = data / "Text/Default/MBToUni.dat"
    cache = TextureCache(data / "EngineTextures.dir", data / "EngineTextures.dat")
    things, _order = loaded_copy(game, f"{MAPS}/FurnaceThings.kld")
    furnace = things.parent
    out = work / "install" / "MoreCreatures"
    if out.exists():
        shutil.rmtree(work / "install")
    # his panel pictures
    make(KREFFT.icon, a.files, cache, work / "panel")
    # the record, texts, files, pools and placements (Dead Walking, Furnace)
    places = {m: {**PLACEMENTS[m], KREFFT_ID: MINE[m]} for m in ("Dead Walking", "Furnace")}
    rep = install(base, furnace, {KREFFT_ID: a.files}, work / "panel", out, codepage, {}, (KREFFT,), places)
    # My Pet Dungeon 1: the folder ships its Things and Variables
    tmp = work / "_mpd1"
    if tmp.exists():
        shutil.rmtree(tmp)
    src, _ = loaded_copy(game, f"{MAPS}/{MPD1}Map.kld")
    r2 = map_install(base, src.parent, MPD1, {KREFFT_ID: MPD1_TILE}, tmp, (KREFFT,))
    for rel in r2["files"]:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / rel), dst)
        rep["files"].append(rel)
        rep["replaces" if (base / rel).exists() else "new"].append(rel)
    rep["placed"].update(r2["placed"])
    rep["variables"] += r2["variables"]
    shutil.rmtree(tmp)
    # Coin Spit: every Globals (all now in out, with his record) and the folder's editor tables
    art = coin_art(base / "Data/editor" if (base / "Data/editor/EffectElements.kwd").exists() else data / "editor")
    spit = []
    for g in sorted((out / MAPS).glob("*Globals.kld")):
        g.write_bytes(apply_coin_spit(g.read_bytes(), art))
        spit.append(g.name)
    for name in EDITOR_FILES:
        rel = f"Data/editor/{name}"
        srcf = base / rel if (base / rel).exists() else data / "editor" / name
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(apply_coin_spit(srcf.read_bytes(), art))
        rep["files"].append(rel)
        rep["replaces" if (base / rel).exists() else "new"].append(rel)
        spit.append(name)
    rep["coin_spit"] = spit
    missing = sorted(g.name for g in (base / MAPS).glob("*Globals.kld") if not (out / MAPS / g.name).exists())
    if missing:
        raise ValueError(f"Globals without his record: {missing}")
    write_lists(rep, work / "install")
    # the check, with him as one of the cast
    CC.configure(CAST + (KREFFT,))
    problems = CC.check(out, base, furnace, a.meshes, codepage, cache,
                        map_dirs={MPD1: src.parent})
    (work / "check.txt").write_text("\n".join(problems) + ("\n" if problems else "OK\n"))
    for line in problems:
        print("PROBLEM", line)
    print(json.dumps({"files": len(rep["files"]), "text_ids": rep.get("text_ids"), "placed": rep["placed"],
                      "variables": rep["variables"], "coin_spit_files": len(spit)}, default=str))
    print(f"check: {'OK' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
