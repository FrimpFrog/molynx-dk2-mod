"""The Molynx release folder: Molynx on every Skirmish and My Pet Dungeon map, never on the campaign.

`build_release(out_dir, map_folders, assets_dir, name="Molynx")` writes `out_dir/<name>/`, a Flame
resource folder like build_molynx_mod's, holding:

  Data/editor/maps/<map>Globals.kld     for every Skirmish and My Pet Dungeon map: the map's own
                                        creature table with record 47 (Molynx, made from that map's
                                        Bile Demon by molynx_for_map, stage 3) and Ground Slam
  Data/editor/maps/<map>Variables.kld   the map's own variables with Molynx in the portal pool and
                                        available to every keeper that has a creature pool (not
                                        shipped for a map without any creature pool)
  Data/editor/Creatures.kwd, CreatureSpells.kwd, Shots.kwd, Effects.kwd, EffectElements.kwd,
  Data/Meshes, Data/EngineTextures, Data/Text/Default/Text.str
                                        exactly what build_molynx_mod ships at stage 3
  Data/Sound/Sfx/Global/creature_molynxBANK.map, creature_molynxSFX.map, MolynxHD.sdt, MolynxHW.sdt
                                        his own sound bank (molynx_soundbank), loaded by his
                                        record's sound category CREATURE_MOLYNX

Nothing else of a map is shipped: its level file, Map, Things, Players and Triggers are not changed,
so Flame keeps loading them from the next folder in alt-resources (Quuz_LevelFixes, then the game).
No creature is placed on any map and nothing is retargeted: Molynx only comes through the portal.
There are no records 48 and up: the creature table has 47 types, the slots after him stay empty.

Which maps. A map is the set of files `<map>.kwd`, `<map>Globals.kld`, `<map>Variables.kld` in
`map_folders`, searched in the order given (pass the user's Quuz_LevelFixes maps first, then the
game's, the order Flame loads them in); each file comes from the first folder that has it, matching
names without regard to case as Windows does. A map is in when
  - its name is a My Pet Dungeon level (`mpd<n>`), or
  - any copy of its level file has IS_SKIRMISH_LEVEL or IS_MULTIPLAYER_LEVEL (the Skirmish maps
    are the multiplayer maps: Quuz_LevelFixes' copies of the original maps, the ones the game loads,
    carry only IS_MULTIPLAYER_LEVEL; the game's own newer copies also IS_SKIRMISH_LEVEL),
and it is never in when its name is a campaign level (`level<n>[a-c]`, `Secret<n>`), whatever its
flags say. The level flags' IS_MY_PET_DUNGEON_LEVEL bit is not used: the game sets it on mpd7 only.

The creature pool. A map whose Variables name no creature pool and no creature availability (a
few newer custom maps) is left to the game's default and gets no Variables file. Otherwise, for
every player that has creature-pool entries, Molynx is added with the count
the map gives that player's Bile Demon (his base creature), or the player's largest count when the
map has no Bile Demon in that player's pool. For every player that has creature-availability
entries, Molynx is added as ENABLE. Each new entry goes right after the last entry of the same kind
of that player, the place the game's editor writes it.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

from scripts.build_molynx_mod import (EDITOR_CREATURES_PATH, MAPS_PATH, TEXT_PATH, _Assets,
                                      _asset_files, _check_name, _editor_creatures_file, _files_for,
                                      _ground_slam_files, _map_bile, _panel_files, molynx_text_file)
from scripts.dk2kld_splice import (BILE_DEMON_ID, IS_MULTIPLAYER_LEVEL, IS_SKIRMISH_LEVEL, level_flags,
                                   molynx_for_map, splice_creatures)
from scripts.dk2kwd_blocks import HEADER_SIZE, Block, read_blocks, rebuilt_block, write_blocks
from scripts.molynx_ground_slam import apply_ground_slam
from scripts.molynx_record import CREATURE_ID
from scripts.molynx_soundbank import sound_files

STAGE = 3
MPD_NAME = re.compile(r"mpd\d+", re.IGNORECASE)
CAMPAIGN_NAME = re.compile(r"(level\d+[a-c]?|secret\d+)", re.IGNORECASE)
CREATURE_POOL, AVAILABILITY, AV_CREATURE, ENABLE = 1, 2, 2, 3
ENTRY = struct.Struct("<iiii")
SHIPPED = ("Globals.kld", "Variables.kld")
# The game's Skirmish list is its multiplayer list: the original maps' level files in Quuz_LevelFixes
# carry IS_MULTIPLAYER_LEVEL only, the game's newer copies of the same maps also IS_SKIRMISH_LEVEL.
SKIRMISH_FLAGS = IS_MULTIPLAYER_LEVEL | IS_SKIRMISH_LEVEL


# ---- which maps ----------------------------------------------------------------------------------

def map_kind(name: str, kwds: list[bytes]) -> str | None:
    """"mpd", "skirmish" or None (not a map Molynx goes on), from the map's name and every copy of
    its level file (see the module docstring)."""
    if CAMPAIGN_NAME.fullmatch(name):
        return None
    if MPD_NAME.fullmatch(name):
        return "mpd"
    flags = 0
    for kwd in kwds:
        flags |= level_flags(kwd)
    return "skirmish" if flags & SKIRMISH_FLAGS else None


def _index(folders: list[Path]) -> dict[str, list[Path]]:
    """lower-case file name -> its paths, in the order of the folders."""
    found: dict[str, list[Path]] = {}
    for folder in folders:
        for path in sorted(Path(folder).iterdir()):
            if path.is_file():
                found.setdefault(path.name.lower(), []).append(path)
    return found


def find_maps(map_folders: list[Path]) -> dict[str, dict]:
    """{map name: {".kwd" / "Globals.kld" / "Variables.kld": source path, "kind": map_kind}} of every map Molynx goes
    on, the name spelled as in its level file's path. A map without its Globals or Variables file in
    any folder is refused (ValueError): its creature table would be the editor's, which this folder
    does not change per map."""
    files = _index([Path(f) for f in map_folders])
    maps: dict[str, dict[str, Path]] = {}
    for key, kwds in files.items():
        if not key.endswith(".kwd"):
            continue
        name = kwds[0].name[:-4]
        kind = map_kind(name, [k.read_bytes() for k in kwds])
        if kind is None:
            continue
        sources = {".kwd": kwds[0], "kind": kind}
        for suffix in SHIPPED:
            paths = files.get((name + suffix).lower())
            if not paths:
                raise ValueError(f"map {name!r} has no {suffix} in {', '.join(map(str, map_folders))}")
            sources[suffix] = paths[0]
        maps[name] = sources
    return dict(sorted(maps.items(), key=lambda item: item[0].lower()))


# ---- the two files of a map -----------------------------------------------------------------------

def _entries(data: bytes):
    """(the Variables block, its entries). A block whose header claims more entries than the file
    holds (the Quuz copy of Spiral's: 310 claimed, 309 there) is read as the entries that are
    there; the rebuilt block then states its true size and count."""
    try:
        (block,) = read_blocks(data)
    except ValueError:
        if len(data) < HEADER_SIZE or (len(data) - HEADER_SIZE) % ENTRY.size:
            raise
        block = Block(struct.unpack_from("<I", data, 0)[0], data[:HEADER_SIZE], data[HEADER_SIZE:])
    if len(block.content) % ENTRY.size:
        raise ValueError("a Variables block whose content is not 16-byte entries")
    return block, [ENTRY.unpack_from(block.content, i) for i in range(0, len(block.content), ENTRY.size)]


def _pool_player(e) -> int | None:
    return e[3] if e[0] == CREATURE_POOL else None


def _availability_player(e) -> int | None:
    return e[1] >> 16 if e[0] == AVAILABILITY and (e[1] & 0xFFFF) == AV_CREATURE else None


def molynx_variables(variables_kld: bytes) -> bytes:
    """The map's Variables with Molynx in every creature pool and every creature availability list
    (see the module docstring). A map that names no creature pool and no creature availability at
    all (some newer custom maps) is returned unchanged: it leaves the creatures to the game's
    default, and a pool of Molynx alone would shut every other creature out. ValueError when the map
    already names creature 47, or lists creature availability without a pool."""
    block, ents = _entries(variables_kld)
    if any((e[0] == CREATURE_POOL and e[1] == CREATURE_ID)
           or (_availability_player(e) is not None and e[2] == CREATURE_ID) for e in ents):
        raise ValueError(f"the map already has Variables entries for creature {CREATURE_ID}")
    pool_players = sorted({p for e in ents if (p := _pool_player(e)) is not None})
    if not pool_players:
        if any(_availability_player(e) is not None for e in ents):
            raise ValueError("the map lists creature availability but has no creature pool")
        return variables_kld   # no pool and no creature list at all: the game's own default applies
    after: dict[int, list[tuple]] = {}   # entry index -> the entries to put after it
    for player in pool_players:
        own = [(i, e) for i, e in enumerate(ents) if _pool_player(e) == player]
        biles = [e[2] for _i, e in own if e[1] == BILE_DEMON_ID and e[2] > 0]
        count = biles[0] if biles else max(e[2] for _i, e in own)
        after.setdefault(own[-1][0], []).append((CREATURE_POOL, CREATURE_ID, count, player))
    for player in sorted({p for e in ents if (p := _availability_player(e)) is not None}):
        last = max(i for i, e in enumerate(ents) if _availability_player(e) == player)
        after.setdefault(last, []).append((AVAILABILITY, (player << 16) | AV_CREATURE, CREATURE_ID, ENABLE))
    out = []
    for i, e in enumerate(ents):
        out.append(e)
        out += after.get(i, [])
    return write_blocks([rebuilt_block(block, [ENTRY.pack(*e) for e in out])])


def molynx_globals(globals_kld: bytes, text_ids: dict[str, int], assets: _Assets) -> tuple[bytes, set[str]]:
    """The map's Globals with record 47 (stage 3, from its own Bile Demon) and Ground Slam, and the
    Molynx file stems that record names."""
    info, wraps, stems = _files_for(_map_bile(globals_kld), assets, None, None)
    data = splice_creatures(globals_kld, [molynx_for_map(globals_kld, STAGE, text_ids, anim_info=info,
                                                         file_wraps=wraps)])
    return apply_ground_slam(data), stems


# ---- the folder ----------------------------------------------------------------------------------

def _resource_toml(name: str, maps: dict[str, dict[str, Path]]) -> bytes:
    skirmish = sum(1 for m in maps.values() if m["kind"] == "skirmish")
    mpd = len(maps) - skirmish
    description = (f"Molynx, the 47th creature, with his own looks, stats, icon, texts and Ground Slam. "
                   f"He can come through the portal on every Skirmish map ({skirmish}) and every My "
                   f"Pet Dungeon level ({mpd}); the campaign is not changed. Attract him with a Guard "
                   f"Room and a Training Room of 25 tiles. List this folder before Quuz_LevelFixes in "
                   f"alt-resources.")
    lines = ["[resource]", f'name = "{name}"', 'version = ""', 'authors = "Soles"',
             'description = """', description, '"""']
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def release_files(map_folders: list[Path], assets_dir: Path, name: str = "Molynx",
                  sounds: tuple[Path, Path] | None = None) -> dict[str, bytes]:
    """Every file of the release folder (relative path -> bytes), built in memory. `sounds` is
    (the game's Data/Sound/Sfx/Global, the folder of Molynx's WAVs): his own sound bank
    (molynx_soundbank), which his record's sound category names; without it the folder has no
    sound files and he is silent."""
    _check_name(name)
    maps = find_maps(map_folders)
    if not maps:
        raise ValueError("no Skirmish or My Pet Dungeon map in the map folders")
    text_file, text_ids = molynx_text_file()
    assets = _Assets(assets_dir)
    files = {"resource.toml": _resource_toml(name, maps)}
    stems: set[str] = set()
    for map_name, sources in maps.items():
        try:
            data, used = molynx_globals(sources["Globals.kld"].read_bytes(), text_ids, assets)
            files[f"{MAPS_PATH}/{map_name}Globals.kld"] = data
            source = sources["Variables.kld"].read_bytes()
            variables = molynx_variables(source)
            if variables != source:   # unchanged: Flame keeps loading the map's own
                files[f"{MAPS_PATH}/{map_name}Variables.kld"] = variables
        except ValueError as err:
            raise ValueError(f"{map_name}: {err}") from err
        stems |= used
    data, used = _editor_creatures_file(STAGE, text_ids, assets, None, None)
    files[EDITOR_CREATURES_PATH] = data
    stems |= used
    files.update(_asset_files(assets, stems))
    files.update(_panel_files(assets))
    files[TEXT_PATH] = text_file
    files.update(_ground_slam_files(assets))
    if sounds is not None:
        files.update(sound_files(*sounds))
    return files


def build_release(out_dir: Path, map_folders: list[Path], assets_dir: Path, name: str = "Molynx",
                  sounds: tuple[Path, Path] | None = None) -> Path:
    """Write `out_dir/<name>/` (never over an existing folder) and return its path."""
    target = Path(out_dir) / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; it is never overwritten or deleted")
    files = release_files(map_folders, assets_dir, name, sounds)
    target.mkdir(parents=True)
    for relative, data in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Molynx release folder: Molynx on every "
                                                 "Skirmish and My Pet Dungeon map, not the campaign.")
    parser.add_argument("--out", type=Path, required=True, help="folder that will hold the mod folder")
    parser.add_argument("--assets", type=Path, required=True, help="molynx_assets export folder")
    parser.add_argument("--name", default="Molynx")
    parser.add_argument("--game-sounds", type=Path, required=True,
                        help="the game's Data/Sound/Sfx/Global (the Bile Demon's bank files)")
    parser.add_argument("--wavs", type=Path, required=True, help="Molynx's WAVs (molynx_sounds)")
    parser.add_argument("map_folders", type=Path, nargs="+",
                        help="map folders in load order (Quuz_LevelFixes maps first, then the game's)")
    args = parser.parse_args(argv)
    print(build_release(args.out, args.map_folders, args.assets, args.name, (args.game_sounds, args.wavs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
