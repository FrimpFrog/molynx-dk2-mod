"""Build the Molynx mod as a Flame resource folder: `build_mod(stage, out_dir, name, maps, text_ids)`.

The folder is what the user copies into `<DK2>\\flame\\resources\\`. It holds, in the layout of the
real Flame mods on the user's machine (Quuz_LevelFixes, Quuz_AudioFix, ...):

  <name>/resource.toml                   [resource] name, version, authors, description (CRLF lines)
  <name>/Data/editor/maps/<map><suffix>  the seven files of each test map (`MAP_SUFFIXES`)

and nothing else at stage 1 ("slot only": an exact Bile Demon copy named Molynx at creatureId 47).
From each test map only two files differ from their source: `<map>Globals.kld` (a 47th record in its
Creatures block, see dk2kld_splice) and `<map>Things.kld` (the creature id byte of the first Bile
Demon that a keeper owns is changed from 3 to 47, so Molynx stands on the map from the start). The
other five files are byte-copies. Stages 2 and 3 add the Molynx record fields of molynx_record
(`Data/Meshes/*.kmf` and `Data/EngineTextures/*.png` are delivered by the later asset steps).

Why a map copy at all. The game keeps its creature table in each map's Globals file, so a 47th creature
exists only in a map that carries a 47-record Creatures block. Flame looks a file up in its list of
alternative resources (`alt-resources` in `flame\\config.toml`) from the first entry to the last and
takes the first folder that holds the file (src/patches/alternative_resources.cpp, `tryUse`, in the
Flame build of the user's install). The user's Quuz_LevelFixes folder holds its own Siege files, so
this folder has to be listed BEFORE Quuz_LevelFixes: the guide tells the user to put it first.

Test maps (`test_maps`): the smallest suitable map, "Siege", and a second one that certainly shows in
the Skirmish list, "13coop" (see dk2kld_splice, `pick_skirmish_test_map`).

Nothing is ever deleted or overwritten: `build_mod` refuses an existing `out_dir/<name>`, builds all
file contents in memory before it creates the folder, and so leaves no half-built folder behind when
a map or a record is refused.

The independent check of a built folder is in audit_molynx_mod.py.

LICENCE: GPL-3.0.
"""
import argparse
import re
import sys
from pathlib import Path

from scripts.dk2kld_splice import (MAP_SUFFIXES, first_player_bile_thing, map_source_dir,
                                   molynx_for_map, pick_skirmish_test_map, pick_test_map,
                                   retarget_creature_thing, splice_creatures)
from scripts.molynx_record import CREATURE_ID

# Flame lists mods by folder name: names are split at ';' in alt-resources, the launcher skips a
# folder whose name has a '.', and a name is also a path component. So: letters, digits, _ and -.
_FOLDER_NAME = re.compile(r"[A-Za-z0-9_-]+")
MAPS_PATH = "Data/editor/maps"
STAGE1_FOLDER = "Molynx_slot_only"

_DESCRIPTIONS = {
    1: ("Molynx test 1 (slot only). Adds a 47th creature named Molynx that is an exact copy of the "
        "Bile Demon, and puts one Molynx on the maps {maps}. Only the map files of those maps are "
        "included; nothing else in the game is changed. Enable only one Molynx folder at a time."),
    2: ("Molynx test 2 (looks). The 47th creature Molynx with his own model and animations, on the "
        "maps {maps}. Enable only one Molynx folder at a time."),
    3: ("Molynx, the 47th creature, with his own looks, stats, icon and texts, on the maps {maps}. "
        "Enable only one Molynx folder at a time."),
}


def test_maps() -> list[str]:
    """The maps the mod is built for: the smallest suitable map and one that is a skirmish map."""
    return [pick_test_map(), pick_skirmish_test_map()]


test_maps.__test__ = False  # a name starting with "test" is not a pytest test, even when imported


def _check_stage(stage) -> None:
    if isinstance(stage, bool) or not isinstance(stage, int) or stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2 or 3, not {stage!r}")


def _check_name(name) -> None:
    if not isinstance(name, str) or not _FOLDER_NAME.fullmatch(name):
        raise ValueError(f"name {name!r} cannot be a Flame resource folder: use only letters, digits, "
                         "'_' and '-' (alt-resources is split at ';' and a '.' hides the folder)")


def _check_maps(maps: list[str]) -> None:
    if not maps or len(set(maps)) != len(maps):
        raise ValueError(f"maps must be a non-empty list without repeats, not {maps!r}")
    for name in maps:
        if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"maps: {name!r} is not a plain map name")


def _resource_toml(name: str, stage: int, maps: list[str]) -> bytes:
    """resource.toml in the format of the real mods: CRLF lines, the keys name, version, authors,
    description (a multi-line string). The description is ASCII without backslashes or quotes."""
    description = _DESCRIPTIONS[stage].format(maps=", ".join(maps))
    lines = ["[resource]", f'name = "{name}"', 'version = ""', 'authors = ""',
             'description = """', description, '"""']
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def _map_files(name: str, stage: int, text_ids: dict[str, int] | None) -> dict[str, bytes]:
    """The seven files of map `name` as they go into the folder (relative path -> bytes)."""
    source = map_source_dir(name)
    files = {f"{MAPS_PATH}/{name}{suffix}": (source / (name + suffix)).read_bytes()
             for suffix in MAP_SUFFIXES}
    globals_path, things_path = f"{MAPS_PATH}/{name}Globals.kld", f"{MAPS_PATH}/{name}Things.kld"
    globals_kld = files[globals_path]
    files[globals_path] = splice_creatures(globals_kld, [molynx_for_map(globals_kld, stage, text_ids)])
    things_kld = files[things_path]
    files[things_path] = retarget_creature_thing(things_kld, first_player_bile_thing(things_kld),
                                                 CREATURE_ID)
    return files


def build_mod(stage: int, out_dir: Path, name: str = "Molynx", maps: list[str] | None = None,
              text_ids: dict[str, int] | None = None) -> Path:
    """Create `out_dir/<name>/` (see the module docstring) and return its path.

    `stage` is 1, 2 or 3 (molynx_record); stage 3 needs `text_ids`. `maps` defaults to `test_maps()`.
    ValueError for a bad stage, folder name, map list or record; FileExistsError when
    `out_dir/<name>` already exists (nothing is deleted or overwritten); FileNotFoundError for a map
    that has no complete set of seven files. Nothing is created unless the whole build succeeded.
    """
    _check_stage(stage)
    _check_name(name)
    if stage == 3 and text_ids is None:
        raise ValueError("stage 3 needs text_ids (keys name, tooltip, general, strength, weakness)")
    target = Path(out_dir) / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; it is never overwritten or deleted")
    maps = test_maps() if maps is None else list(maps)
    _check_maps(maps)

    files = {"resource.toml": _resource_toml(name, stage, maps)}
    for map_name in maps:
        files.update(_map_files(map_name, stage, text_ids))

    target.mkdir(parents=True)  # not exist_ok: never reuse a folder
    for relative, data in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Molynx Flame mod folder (stage 1 or 2).")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--out", type=Path, required=True, help="folder that will hold the mod folder")
    parser.add_argument("--name", help=f"mod folder name (default {STAGE1_FOLDER} at stage 1, else Molynx)")
    parser.add_argument("--maps", nargs="+", help="map names (default: the test maps)")
    args = parser.parse_args(argv)
    name = args.name or (STAGE1_FOLDER if args.stage == 1 else "Molynx")
    print(build_mod(args.stage, args.out, name, args.maps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
