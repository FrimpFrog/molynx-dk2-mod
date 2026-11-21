"""Build the cast install end to end: every creature's files, panel pictures, the install folder
with its lists, the check, the map pictures and the stats table.

  python -m scripts.cast_build --game "<staged DK2 folder>" --meshes <extracted meshes> \\
      --dragon-pkg <approved Dragon package Data folder> --necro-info <files.json> --work <folder>

<work>/files/{orc,hreaper,dragon,sejanus,gargoyle}   his .kmf files and skins (the Orc and the Reaper are rebuilt
                                    from the game's Troll and Horny files; the Dragon, Sejanus and the Gargoyle are their
                                    approved builds)
<work>/panel                        <icon>.png, <icon>-bg.png of the creatures this round installs
<work>/install/MoreCreatures/...    the files to copy (only those that change or are new)
<work>/install/install_list.txt, backup_list.txt, report.json
<work>/maps/*.png                   Dead Walking and Furnace with the heart and the cast
<work>/stats_table.md               each creature next to his donor

The FurnaceThings / FurnaceMap it builds on are the ones the game loads today, found from the
alt-resources line of <game>/flame/config.toml (cast_install.loaded_copy: Quuz_LevelFixes' on Soles' PC);
`--furnace` names another folder. Nothing is written to the game folder.

A later creature goes in on top of what is installed: `--base` is the PC's MoreCreatures as it is now
(the staged folder with the earlier install laid over it), `--only 52` installs Sejanus alone and
`--out-name install_sejanus` keeps the earlier install folder; the check runs over the combined state.
`--update 50` rebuilds installed creatures in place instead (cast_install.update_install: the Reaper made a
regular creature): his record in every table, his texts at their ids, no new pictures or tiles.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _now(out: Path, base: Path, name: str) -> bytes:
    """A file as it will be: the new one if the install writes it, else the base folder's."""
    return (out / name if (out / name).exists() else base / name).read_bytes()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", type=Path, required=True, help="the staged Dungeon Keeper 2 folder")
    p.add_argument("--meshes", type=Path, required=True)
    p.add_argument("--dragon-pkg", type=Path, required=True, help="…/MoreCreatures/Data of the Dragon package")
    p.add_argument("--sejanus-build", type=Path, required=True, help="Sejanus' built files (sejanus_build)")
    p.add_argument("--gargoyle-build", type=Path, required=True, help="the Gargoyle's built files (gargoyle_build)")
    p.add_argument("--base", type=Path, help="the MoreCreatures folder as it is on the PC now (default: the staged one)")
    p.add_argument("--only", type=int, nargs="*", help="install only these creature ids (the others are in --base)")
    p.add_argument("--out-name", default="install", help="the install folder's name under --work")
    p.add_argument("--update", type=int, nargs="*", help="rebuild these installed creature ids in place")
    p.add_argument("--necro-info", type=Path, required=True)
    p.add_argument("--furnace", type=Path, help="default: <game>/Data/editor/maps")
    p.add_argument("--work", type=Path, required=True)
    a = p.parse_args(argv)
    from scripts.cast_check import check, map_image
    from scripts.cast_install import RESERVED_TILES, install, update_install, write_lists
    from scripts.cast_portraits import make
    from scripts.cast_records import stats_table
    from scripts.dk2kld_splice import _creatures_index, _records
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.dk2texture import TextureCache
    import scripts.hreaper_files as hreaper
    import scripts.orc_build as orc

    data = a.game / "Data"
    base = a.base or a.game / "flame/resources/MoreCreatures"
    if a.furnace:
        furnace = a.furnace
    else:                                   # the folder the game loads Furnace's Things and Map from today
        from scripts.cast_install import loaded_copy
        things, order = loaded_copy(a.game, "Data/editor/maps/FurnaceThings.kld")
        fmap, _ = loaded_copy(a.game, "Data/editor/maps/FurnaceMap.kld")
        if things.parent != fmap.parent:
            raise ValueError(f"Furnace's Things ({things}) and Map ({fmap}) load from different folders")
        furnace = things.parent
        print(f"alt-resources: {';'.join(order)}; Furnace Things and Map load from {furnace}")
    codepage = data / "Text/Default/MBToUni.dat"
    cache = TextureCache(data / "EngineTextures.dir", data / "EngineTextures.dat")
    files = a.work / "files"
    orc.export(a.meshes, cache, files / "orc")
    hreaper.export(a.meshes, cache, files / "hreaper")
    (files / "dragon").mkdir(parents=True, exist_ok=True)
    for src in [*(a.dragon_pkg / "Meshes").glob("Dragon-*.kmf"), *(a.dragon_pkg / "EngineTextures").glob("Dragon_*.png")]:
        shutil.copy2(src, files / "dragon" / src.name)
    (files / "sejanus").mkdir(parents=True, exist_ok=True)
    for src in [*a.sejanus_build.glob("Sejanus-*.kmf"), *a.sejanus_build.glob("Sejanus_*.png")]:
        shutil.copy2(src, files / "sejanus" / src.name)
    (files / "gargoyle").mkdir(parents=True, exist_ok=True)
    for src in [*a.gargoyle_build.glob("Gargoyle-*.kmf"), *a.gargoyle_build.glob("Gargoyle_*.png")]:
        shutil.copy2(src, files / "gargoyle" / src.name)
    folders = {49: "orc", 50: "hreaper", 51: "dragon", 52: "sejanus", 53: "gargoyle"}
    from scripts.cast_records import CAST
    out = a.work / a.out_name
    if a.update:
        members = tuple(m for m in CAST if m.creature_id in a.update)
        rep = update_install(base, {cid: files / f for cid, f in folders.items()}, out / "MoreCreatures", codepage,
                             members)
    else:
        members = tuple(m for m in CAST if not a.only or m.creature_id in a.only)
        for m in members:                  # only the pictures this round installs (the others stay as they are)
            make(m.icon, files / folders[m.creature_id], cache, a.work / "panel")
        rep = install(base, furnace, {cid: files / f for cid, f in folders.items()}, a.work / "panel",
                      out / "MoreCreatures", codepage, json.loads(a.necro_info.read_text()), members)
    write_lists(rep, out)
    problems = check(out / "MoreCreatures", base, furnace, a.meshes, codepage, cache)
    maps = a.work / "maps"
    maps.mkdir(exist_ok=True)
    MAPS_DIR = "Data/editor/maps"
    m = out / "MoreCreatures" / MAPS_DIR
    map_image((base / "Data/editor/maps/Dead WalkingMap.kld").read_bytes(), _now(m, base / MAPS_DIR, "Dead WalkingThings.kld"),
              "Dead Walking: Keeper 1's heart and the cast", RESERVED_TILES["Dead Walking"], 8, (0, 31, 12, 43)
              ).save(maps / "DeadWalking_cast.png")
    fmap = base / "Data/editor/maps/FurnaceMap.kld"
    fmap = fmap if fmap.exists() else furnace / "FurnaceMap.kld"
    map_image(fmap.read_bytes(), _now(m, base / MAPS_DIR, "FurnaceThings.kld"), "Furnace: Keeper 1's heart and the cast",
              RESERVED_TILES["Furnace"], 8, (3, 49, 13, 59)).save(maps / "Furnace_cast.png")
    b = read_blocks(_now(out / "MoreCreatures/Data/editor", base / "Data/editor", "Creatures.kwd"))
    (a.work / "stats_table.md").write_text(stats_table(_records(b[_creatures_index(b)])))
    for line in problems:
        print("PROBLEM", line)
    print(f"{len(rep['files'])} files; check: {'OK' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
