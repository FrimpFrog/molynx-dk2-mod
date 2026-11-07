"""Build the Molynx mod as a Flame resource folder: `build_mod(stage, out_dir, name, maps, text_ids,
assets_dir=...)`.

The folder is what the user copies into `<DK2>\\flame\\resources\\`. It holds, in the layout of the
real Flame mods on the user's machine (Quuz_LevelFixes, Quuz_AudioFix, ...):

  <name>/resource.toml                   [resource] name, version, authors, description (CRLF lines)
  <name>/Data/editor/maps/<map><suffix>  the seven files of each test map (`MAP_SUFFIXES`)

and nothing else at stage 1 ("slot only": an exact Bile Demon copy named Molynx at creatureId 47).
From each test map only two files differ from their source: `<map>Globals.kld` (a 47th record in its
Creatures block, see dk2kld_splice) and `<map>Things.kld` (the creature id byte of the first Bile
Demon that a keeper owns is changed from 3 to 47, so Molynx stands on the map from the start). The
other five files are byte-copies. Stages 2 and 3 add the Molynx record fields of molynx_record and
his files, copied byte for byte from an export folder (`assets_dir`, written by
molynx_assets.export_molynx_assets; the builder never runs that export itself):

  <name>/Data/Meshes/Molynx-<stem>.kmf   exactly the files the record names (the file plan of each
                                         map's Bile Demon, `stems_used`: for the test
                                         maps 18 clips, the variant walk_once and the 5
                                         static poses; the clips no slot uses stay out)
  <name>/Data/EngineTextures/<t>.png     the textures those files' materials name (Molynx_Body,
                                         Molynx_Head, Molynx_Mace, Molynx_Neck), where the user's
                                         Flame texture mods put theirs

What the record says about each file (frame count, loop flag) is not taken from a separate list: it
is read from the exported files themselves, and the build stops (ValueError, nothing written) when a
file the record names is missing, is named differently (also in case, or inside the file), does not
parse to its end, or loops differently from what the file plan (molynx_record.file_plan, on the Bile
Demon's own files) needs.

Why a map copy at all. The game keeps its creature table in each map's Globals file, so a 47th creature
exists only in a map that carries a 47-record Creatures block. Flame looks a file up in its list of
alternative resources (`alt-resources` in `flame\\config.toml`) from the first entry to the last and
takes the first folder that holds the file (src/patches/alternative_resources.cpp, `tryUse`, in the
Flame build of the user's install). The user's Quuz_LevelFixes folder holds its own Siege files, so
this folder has to be listed BEFORE Quuz_LevelFixes: the guide tells the user to put it first.

Test maps (`test_maps`): the smallest suitable map, "Siege", and a second one that certainly shows in
the Skirmish list and puts Molynx in Keeper 1's dungeon, "Dead Walking" (see dk2kld_splice,
`pick_skirmish_test_map`). On both maps the Molynx thing is owned by player 3, "Keeper 1". These
are "copied" maps: the map's own files with the two changes above.

Flat maps (`FLAT_MAPS`, the flat test map of Test A, map 2): the flat test map of dk2_testmap (Keeper 1 on a big claimed floor
with a portal, gems, gold, six Imps and Molynx beside the heart; Keeper 2 sealed in Impenetrable
Rock) under three names: "Molynx Test", "Molynx Test Portal" (Molynx also in the portal pool) and
"Dead Walking" (the same map as "Molynx Test" under the shipped name, so the game finds it even if
it lists no new map name). The six files come from `testmap_files` (checked by `validate_testmap`
before anything is written); the level file shows "Molynx Test" for both "Molynx Test" and "Dead
Walking", "Molynx Test Portal" for the variant, and its path records name the stem's own files
(for "Dead Walking" the shipped paths). The level file names a Globals file too and the game opens
it, so `<stem>Globals.kld` is FLAT_GLOBALS' (Siege's) Globals spliced with this folder's record 47,
exactly as for a copied map. Not Dead Walking's own Globals: that map changes its creature table (its
Imps cannot be picked up, hit for 200 and more; most creatures lose a flag), which the first in-game
test showed as Imps that could only be slapped. The default build (`maps` and `flat_maps` both left out)
is Siege copied plus the three flat maps.

Nothing is ever deleted or overwritten: `build_mod` refuses an existing `out_dir/<name>`, builds all
file contents in memory before it creates the folder, and so leaves no half-built folder behind when
a map or a record is refused.

The independent check of a built folder is in audit_molynx_mod.py.

LICENCE: GPL-3.0.
"""
import argparse
import os
import re
import struct
import sys
from collections.abc import Mapping
from pathlib import Path

from scripts.dk2_testmap import BASE as FLAT_BASE, testmap_files, validate_testmap
from scripts.dk2kld_splice import (BILE_DEMON_ID, BILE_DEMON_NAME, MAP_SUFFIXES, first_player_bile_thing,
                                   map_source_dir, molynx_for_map, pick_skirmish_test_map,
                                   pick_test_map, retarget_creature_thing, splice_creatures)
from scripts.dk2kmf import parse_kmf
from scripts.dk2paths import EDITOR_DIR
from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_creatures import Creature, read_creatures
from scripts.molynx_assets import MAX_TEXTURE, bile_file_wraps
from scripts.molynx_record import CREATURE_ID, GAME_FPS, POSE_STEMS, PREFIX, file_plan

# Flame lists mods by folder name: names are split at ';' in alt-resources, the launcher skips a
# folder whose name has a '.', and a name is also a path component. So: letters, digits, _ and -.
_FOLDER_NAME = re.compile(r"[A-Za-z0-9_-]+")
MAPS_PATH = "Data/editor/maps"
MESHES_PATH = "Data/Meshes"
TEXTURES_PATH = "Data/EngineTextures"
EDITOR_CREATURES_PATH = "Data/editor/Creatures.kwd"  # the base creature table with record 47
STAGE1_FOLDER = "Molynx_slot_only"
# The flat maps of the handed-over folders: stem -> Molynx also in the portal pool.
FLAT_MAPS = {"Molynx Test": False, "Molynx Test Portal": True, FLAT_BASE: False}
# The name the level file shows in the Skirmish list (default: the stem).
FLAT_DISPLAY_NAMES = {FLAT_BASE: "Molynx Test"}
FLAT_GLOBALS = "Siege"   # the flat maps' creature table (see the module notes: not Dead Walking's)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_DESCRIPTIONS = {
    1: ("Molynx test A fallback (slot only). Use this only if the folder Molynx fails. Adds a "
        "47th creature that is an exact copy of the Bile Demon, same look and same name, and puts "
        "him on the maps {maps}. Only the map files of those maps are included; nothing else in "
        "the game is changed. Enable only one Molynx folder at a time."),
    2: ("Molynx test A (his own look). Use this first. Adds Molynx, a 47th creature with his own "
        "model and animations, and puts him on the maps {maps}. Enable only one Molynx folder at "
        "a time."),
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


def _check_maps(maps: list[str], flat: Mapping[str, bool]) -> None:
    if not isinstance(flat, Mapping):
        raise ValueError(f"flat_maps must be a mapping of map name -> bool, not {flat!r}")
    if not maps and not flat or len(set(maps)) != len(maps):
        raise ValueError(f"maps must be a list without repeats, and maps and flat_maps together "
                         f"must name at least one map, not {maps!r} and {dict(flat)!r}")
    both = sorted(set(maps) & set(flat))
    if both:
        raise ValueError(f"{', '.join(both)}: a map is either copied (maps) or flat (flat_maps), "
                         "not both")
    for name in [*maps, *flat]:
        if not isinstance(name, str) or name in ("", ".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"maps: {name!r} is not a plain map name")


def _resource_toml(name: str, stage: int, maps: list[str], flat: list[str] = ()) -> bytes:
    """resource.toml in the format of the real mods: CRLF lines, the keys name, version, authors,
    description (a multi-line string). The description is ASCII without backslashes or quotes."""
    description = _DESCRIPTIONS[stage].format(maps=", ".join([*maps, *flat]))
    if flat:
        description += (" The maps " + ", ".join(flat) + " are the flat test map: Keeper 2 is "
                        "sealed in, Molynx stands next to your heart.")
    lines = ["[resource]", f'name = "{name}"', 'version = ""', 'authors = ""',
             'description = """', description, '"""']
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


class _Assets:
    """The export folder: each `Molynx-<stem>.kmf` read and parsed once, checked as it is read."""

    def __init__(self, folder) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise ValueError(f"assets_dir {self.folder} is not a folder")
        self.names = set(os.listdir(self.folder))
        self._models: dict[str, tuple] = {}

    def _bytes(self, file_name: str) -> bytes:
        if file_name not in self.names:
            other = [n for n in self.names if n.lower() == file_name.lower()]
            hint = f" (only {other[0]!r}, other case, exists)" if other else ""
            raise ValueError(f"assets_dir has no {file_name}{hint}")
        return (self.folder / file_name).read_bytes()

    def model(self, stem: str):
        """(file bytes, parsed model) of `Molynx-<stem>.kmf`."""
        if stem not in self._models:
            file_name = f"{PREFIX}{stem}.kmf"
            data = self._bytes(file_name)
            try:
                model = parse_kmf(data)
            except (ValueError, IndexError, struct.error) as err:
                raise ValueError(f"{file_name} in assets_dir does not parse: {err}") from None
            if model.end != len(data):
                raise ValueError(f"{file_name} in assets_dir has {len(data) - model.end} bytes after "
                                 "its model")
            body = model.anim if model.kind == "anim" else model.mesh
            if body is None or body.name != PREFIX + stem:
                raise ValueError(f"{file_name} in assets_dir calls itself "
                                 f"{getattr(body, 'name', None)!r} inside, not {PREFIX + stem!r}")
            self._models[stem] = (data, model)
        return self._models[stem]

    def texture(self, name: str) -> bytes:
        data = self._bytes(name + ".png")
        if data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
            raise ValueError(f"{name}.png in assets_dir is not a PNG file")
        width, height = struct.unpack_from(">II", data, 16)
        if not all(0 < n <= MAX_TEXTURE and not n & (n - 1) for n in (width, height)):
            raise ValueError(f"{name}.png in assets_dir is {width}x{height}: the game takes powers "
                             f"of two up to {MAX_TEXTURE} px")
        return data


def _map_bile(globals_kld: bytes) -> Creature:
    (block,) = [b for b in read_blocks(globals_kld) if b.signature == 0xAA]
    biles = [c for c in read_creatures(block)
             if c.get("creatureId") == BILE_DEMON_ID and c.get("name") == BILE_DEMON_NAME]
    if len(biles) != 1:
        raise ValueError(f"expected one {BILE_DEMON_NAME!r} with creatureId {BILE_DEMON_ID}, "
                         f"found {len(biles)}")
    return biles[0]


def _files_for(bile: Creature, assets: _Assets, anim_info: dict | None,
               file_wraps: dict | None) -> tuple[dict, dict, set[str]]:
    """(anim_info, file_wraps, file stems) for one map's Bile Demon, read from the exported files
    and checked against its file plan; a given `anim_info` / `file_wraps` must agree with them."""
    wraps = bile_file_wraps([bile])
    if file_wraps is not None:
        if not isinstance(file_wraps, Mapping):
            raise ValueError(f"file_wraps must be a mapping, not {type(file_wraps).__name__}")
        given = {k.lower(): v for k, v in file_wraps.items()}
        wrong = sorted(k for k, v in wraps.items() if k not in given or given[k] != v)
        if wrong:
            raise ValueError(f"file_wraps disagrees with the Bile Demon's own files on {wrong}")
    plan = file_plan(bile, wraps)
    info = {}
    for stem in sorted(set(plan.slots.values())):
        _data, model = assets.model(stem)
        if stem in POSE_STEMS:
            if model.kind != "mesh":
                raise ValueError(f"{PREFIX}{stem}.kmf must be a static mesh (a pose), it is a {model.kind}")
            continue
        if model.kind != "anim":
            raise ValueError(f"{PREFIX}{stem}.kmf must be an animation, it is a {model.kind}")
        if bool(model.anim.wrap) != plan.wraps[stem]:
            raise ValueError(f"{PREFIX}{stem}.kmf {'loops' if model.anim.wrap else 'does not loop'}, "
                             f"the file plan needs the opposite")
        info[stem] = {"frames": model.anim.frame_count, "fps": GAME_FPS, "wrap": bool(model.anim.wrap)}
    if anim_info is not None:
        if not isinstance(anim_info, Mapping):
            raise ValueError(f"anim_info must be a mapping, not {type(anim_info).__name__}")
        wrong = sorted(stem for stem in info if anim_info.get(stem) != info[stem])
        if wrong:
            raise ValueError(f"anim_info disagrees with the exported files on {wrong}")
    return info, wraps, set(plan.slots.values())


def _map_files(name: str, stage: int, text_ids: dict[str, int] | None, assets: _Assets | None,
               anim_info: dict | None, file_wraps: dict | None) -> tuple[dict[str, bytes], set[str]]:
    """The seven files of map `name` as they go into the folder (relative path -> bytes), and the
    Molynx file stems its record names (stage >= 2)."""
    source = map_source_dir(name)
    files = {f"{MAPS_PATH}/{name}{suffix}": (source / (name + suffix)).read_bytes()
             for suffix in MAP_SUFFIXES}
    globals_path, things_path = f"{MAPS_PATH}/{name}Globals.kld", f"{MAPS_PATH}/{name}Things.kld"
    globals_kld = files[globals_path]
    stems: set[str] = set()
    info = wraps = None
    if stage >= 2:
        info, wraps, stems = _files_for(_map_bile(globals_kld), assets, anim_info, file_wraps)
    files[globals_path] = splice_creatures(globals_kld, [molynx_for_map(
        globals_kld, stage, text_ids, anim_info=info, file_wraps=wraps)])
    things_kld = files[things_path]
    files[things_path] = retarget_creature_thing(things_kld, first_player_bile_thing(things_kld),
                                                 CREATURE_ID)
    return files, stems


def _flat_map_files(stem: str, in_pool: bool, stage: int, text_ids: dict[str, int] | None,
                    assets: _Assets | None, anim_info: dict | None,
                    file_wraps: dict | None) -> tuple[dict[str, bytes], set[str]]:
    """The seven files of the flat map `stem` (relative path -> bytes) and the Molynx file stems its
    record names: the generator's six files and the base map's Globals with record 47."""
    files = testmap_files(stem, molynx_in_pool=in_pool,
                          display_name=FLAT_DISPLAY_NAMES.get(stem, stem))
    errors = validate_testmap(files, stem, molynx_in_pool=in_pool)
    if errors:
        raise ValueError(f"the flat map {stem!r} breaks its layout rules: " + "; ".join(errors))
    globals_kld = (map_source_dir(FLAT_GLOBALS) / f"{FLAT_GLOBALS}Globals.kld").read_bytes()
    stems: set[str] = set()
    info = wraps = None
    if stage >= 2:
        info, wraps, stems = _files_for(_map_bile(globals_kld), assets, anim_info, file_wraps)
    out = {f"{MAPS_PATH}/{name}": data for name, data in files.items()}
    out[f"{MAPS_PATH}/{stem}Globals.kld"] = splice_creatures(globals_kld, [molynx_for_map(
        globals_kld, stage, text_ids, anim_info=info, file_wraps=wraps)])
    return out, stems


def _editor_creatures_file(stage: int, text_ids: dict[str, int] | None, assets: _Assets | None,
                           anim_info: dict | None, file_wraps: dict | None,
                           editor_dir: Path | None = None) -> tuple[bytes, set[str]]:
    """`Data/editor/Creatures.kwd`: the editor's own file with record 47 spliced in (its Bile Demon,
    in its 5449-byte record size), and the Molynx file stems that record names (stage >= 2).

    Why. The game loads the editor's Creatures.kwd as its base creature table (46 records) and then
    applies a map's Globals.kld as an update: with the update flag set (CWorld field 0xA3C7, the
    global 0x6FCDCF, set by CFrontEndComponent::loadLevel_data 0x541AA0 when a map is picked in the
    Skirmish lobby), CWorld_loadCreature (0x51F240) looks each record up by creatureId
    (CWorld::fun_50D390) and updates it in place (MyCreatureDataObj::loadCreatureType 0x492320). A
    creatureId missing from the base table gives a NULL object: the Test B crash (2026-11-07, Flame
    crash report: eip 0x492380, `mov [ebx+0x770], al` with ebx = 0, creatureId eax = 0x2F). Flame's
    alt-resources serve Data/editor files from the mod folder, so this copy is the one loaded."""
    source = ((editor_dir or EDITOR_DIR) / "Creatures.kwd").read_bytes()
    stems: set[str] = set()
    info = wraps = None
    if stage >= 2:
        info, wraps, stems = _files_for(_map_bile(source), assets, anim_info, file_wraps)
    data = splice_creatures(source, [molynx_for_map(source, stage, text_ids, anim_info=info,
                                                    file_wraps=wraps)])
    return data, stems


def _asset_files(assets: _Assets, stems: set[str]) -> dict[str, bytes]:
    """Data/Meshes/<file> for every stem and Data/EngineTextures/<t>.png for every texture they name."""
    files, textures = {}, set()
    for stem in sorted(stems):
        data, model = assets.model(stem)
        files[f"{MESHES_PATH}/{PREFIX}{stem}.kmf"] = data
        textures |= {t for material in model.materials for t in material.textures}
    game = _game_textures()
    for texture in sorted(textures):
        if texture.lower() in game:     # a prop's own game texture (the chicken, the bed): the game has it
            continue
        files[f"{TEXTURES_PATH}/{texture}.png"] = assets.texture(texture)
    return files


def _game_textures() -> set[str]:
    """Lower-case names of the game's own textures (EngineTextures cache, without the MMn suffix)."""
    from scripts.molynx_assets import texture_cache
    return {n[:-3].lower() for n in texture_cache().names if n[-3:-1].upper() == "MM"}


def build_mod(stage: int, out_dir: Path, name: str = "Molynx", maps: list[str] | None = None,
              text_ids: dict[str, int] | None = None, *, anim_info: dict | None = None,
              file_wraps: dict | None = None, assets_dir: Path | None = None,
              flat_maps: Mapping[str, bool] | None = None, editor_creatures: bool = True) -> Path:
    """Create `out_dir/<name>/` (see the module docstring) and return its path.

    `stage` is 1, 2 or 3 (molynx_record); stage 3 needs `text_ids`; stages 2 and 3 need
    `assets_dir`, an export folder of molynx_assets.export_molynx_assets: the frame counts and loop
    flags the record gives each slot are read from its files, and the Bile Demon files' loop flags
    from the game's meshes. `anim_info` / `file_wraps`, when given, must agree with what is read
    (ValueError otherwise). `maps` are the copied maps, `flat_maps` the flat ones (stem -> Molynx
    also in the portal pool); when both are left out the folder holds Siege copied and the three
    `FLAT_MAPS`, when one is given the other defaults to none. ValueError for a bad stage, folder
    name, map list (a name in both, or no map at all), record or asset; FileExistsError when `out_dir/<name>` already exists (nothing is
    deleted or overwritten); FileNotFoundError for a map that has no complete set of seven files.
    Nothing is created unless the whole build succeeded. `editor_creatures` (default True) also
    writes `Data/editor/Creatures.kwd` with record 47 (see `_editor_creatures_file`).
    """
    _check_stage(stage)
    _check_name(name)
    if stage == 3 and text_ids is None:
        raise ValueError("stage 3 needs text_ids (keys name, tooltip, general, strength, weakness)")
    if stage >= 2 and assets_dir is None:
        raise ValueError(f"stage {stage} needs assets_dir (an export folder of "
                         "molynx_assets.export_molynx_assets); anim_info and file_wraps are read "
                         "from it")
    if stage == 1 and assets_dir is not None:
        raise ValueError("stage 1 delivers no Molynx files, so it takes no assets_dir")
    target = Path(out_dir) / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; it is never overwritten or deleted")
    if maps is None and flat_maps is None:
        flat_maps = dict(FLAT_MAPS)
        maps = [m for m in test_maps() if m not in flat_maps]
    maps = [] if maps is None else list(maps)
    flat_maps = {} if flat_maps is None else flat_maps
    _check_maps(maps, flat_maps)
    flat_maps = dict(flat_maps)
    assets = _Assets(assets_dir) if stage >= 2 else None

    files = {"resource.toml": _resource_toml(name, stage, maps, list(flat_maps))}
    stems: set[str] = set()
    for map_name in maps:
        map_files, used = _map_files(map_name, stage, text_ids, assets, anim_info, file_wraps)
        files.update(map_files)
        stems |= used
    for stem, in_pool in flat_maps.items():
        map_files, used = _flat_map_files(stem, bool(in_pool), stage, text_ids, assets, anim_info,
                                          file_wraps)
        files.update(map_files)
        stems |= used
    if editor_creatures:
        data, used = _editor_creatures_file(stage, text_ids, assets, anim_info, file_wraps)
        files[EDITOR_CREATURES_PATH] = data
        stems |= used
    if assets is not None:
        files.update(_asset_files(assets, stems))

    target.mkdir(parents=True)  # not exist_ok: never reuse a folder
    for relative, data in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Molynx Flame mod folder. Stage 1: the 47th creature as an exact Bile "
                    "Demon copy (folder Molynx_slot_only). Stage 2: with his own model and "
                    "animations, copied from an export folder (--assets).")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1, help="1 (default) or 2")
    parser.add_argument("--out", type=Path, required=True, help="folder that will hold the mod folder")
    parser.add_argument("--assets", type=Path,
                        help="stage 2 only: the folder molynx_assets.export_molynx_assets wrote")
    parser.add_argument("--name", help=f"mod folder name (default {STAGE1_FOLDER} at stage 1, else Molynx)")
    parser.add_argument("--maps", nargs="+",
                        help="copied map names only, no flat map (default: Siege copied and the "
                             "flat maps " + ", ".join(FLAT_MAPS) + ")")
    args = parser.parse_args(argv)
    if args.stage == 2 and args.assets is None:
        parser.error("--stage 2 needs --assets (the export folder with Molynx's files)")
    if args.stage == 1 and args.assets is not None:
        parser.error("--assets is only for --stage 2")
    name = args.name or (STAGE1_FOLDER if args.stage == 1 else "Molynx")
    print(build_mod(args.stage, args.out, name, args.maps, assets_dir=args.assets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
