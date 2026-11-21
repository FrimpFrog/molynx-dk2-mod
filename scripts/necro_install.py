"""Install the Necromancer (creature 48) into an existing MoreCreatures folder (the Molynx release),
the same way Molynx went in, without touching Molynx:

  Data/editor/maps/<map>Globals.kld    every map the folder already ships: record 48 spliced in,
                                       built from that map's own Black Knight (necro_record)
  Data/editor/maps/<map>Variables.kld  every Variables file the folder ships: the Necromancer in each
                                       player's creature pool (the count that player's Black Knight
                                       has, else the player's largest) and ENABLE in each creature
                                       availability list (build_molynx_release's rules, for 48)
  Data/editor/Creatures.kwd            the editor table the game loads as its base: record 48 too
                                       (a creatureId missing from the base table crashes the game)
  Data/Text/Default/Text.str           his four texts appended
  Data/Meshes/Necro-*.kmf, Data/EngineTextures/Necro_*.png,
  Data/EngineTextures/GUI/Creatures/Evil/necromancer.png, necromancer-bg.png
  Dead Walking (the Molynx test map):  Things gets a Necromancer (level 5, Keeper 1) on the tile next
                                       to Molynx by the Dungeon Heart

`install(base_dir, files_dir, panel_dir, out_dir)` reads the folder in base_dir and writes only the
files that change (or are new) into out_dir, in the same layout; nothing in base_dir is modified.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from scripts.build_molynx_release import (AVAILABILITY, AV_CREATURE, CREATURE_POOL, ENABLE, ENTRY,
                                          _availability_player, _entries, _pool_player)
from scripts.dk2kld_splice import (_creatures_index, _records, read_things, splice_creatures,
                                   write_things, Thing, CREATURE_THING)
from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2str import StrFile
from scripts.necro_record import BLACK_KNIGHT_ID, CREATURE_ID, TEXTS, build_necromancer

MAPS = "Data/editor/maps"
SHARED_TOOLTIP_ID = 2408                   # the game's shared creature tooltip (as Molynx)
MOLYNX_ID = 47
TEST_MAP = "Dead Walking"
LEVEL = 5                                  # as the Molynx by the heart


def necro_for(creatures_file: bytes, files_info: dict, text_ids: dict) -> bytes:
    """The Globals (or editor Creatures.kwd) with record 48 from its own Black Knight."""
    blocks = read_blocks(creatures_file)
    recs = _records(blocks[_creatures_index(blocks)])
    if any(c.get("creatureId") == CREATURE_ID for c in recs):
        raise ValueError("already has creature 48")
    bks = [c for c in recs if c.get("creatureId") == BLACK_KNIGHT_ID]
    if len(bks) != 1:
        raise ValueError(f"expected one Black Knight, found {len(bks)}")
    order = max(c.get("orderInEditor") for c in recs) + 1
    return splice_creatures(creatures_file, [build_necromancer(bks[0], order, files_info, text_ids)])


def necro_variables(variables_kld: bytes) -> bytes:
    """The map's Variables with the Necromancer beside Molynx in every pool and availability list.
    Unchanged for a map with neither (the game's default list applies there, as for Molynx)."""
    block, ents = _entries(variables_kld)
    if any((e[0] == CREATURE_POOL and e[1] == CREATURE_ID)
           or (_availability_player(e) is not None and e[2] == CREATURE_ID) for e in ents):
        raise ValueError("the map already has Variables entries for creature 48")
    pool_players = sorted({p for e in ents if (p := _pool_player(e)) is not None})
    if not pool_players:
        return variables_kld
    after: dict[int, list[tuple]] = {}
    for player in pool_players:
        own = [(i, e) for i, e in enumerate(ents) if _pool_player(e) == player]
        bks = [e[2] for _i, e in own if e[1] == BLACK_KNIGHT_ID and e[2] > 0]
        count = bks[0] if bks else max(e[2] for _i, e in own)
        after.setdefault(own[-1][0], []).append((CREATURE_POOL, CREATURE_ID, count, player))
    for player in sorted({p for e in ents if (p := _availability_player(e)) is not None}):
        last = max(i for i, e in enumerate(ents) if _availability_player(e) == player)
        after.setdefault(last, []).append((AVAILABILITY, (player << 16) | AV_CREATURE, CREATURE_ID, ENABLE))
    out = []
    for i, e in enumerate(ents):
        out.append(e)
        out += after.get(i, [])
    return write_blocks([rebuilt_block(block, [ENTRY.pack(*e) for e in out])])


# A creature thing (type 200): x, y (i32) ... level (u8 at 14) ... creature id (u8 at 26), owner (27)
_LEVEL_AT, _ID_AT = 14, 26


def place_beside_molynx(things_kld: bytes, map_kld: bytes) -> tuple[bytes, tuple[int, int]]:
    """Things with a Necromancer (Molynx's thing copied: same owner, flags, health) on the nearest
    free tile around Molynx whose terrain is the same as Molynx's (his claimed floor)."""
    from scripts.dk2_testmap import read_map
    block, things = read_things(things_kld)
    mol = [t for t in things if t.type == CREATURE_THING and t.data[_ID_AT] == MOLYNX_ID]
    if len(mol) != 1:
        raise ValueError(f"expected one Molynx on the map, found {len(mol)}")
    mx, my = struct.unpack_from("<ii", mol[0].data, 0)
    m = read_map(map_kld)
    taken = {struct.unpack_from("<ii", t.data, 0) for t in things if t.type == CREATURE_THING}
    floor = m.at(mx, my)[0]
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 2), (2, 0)):
        x, y = mx + dx, my + dy
        if (x, y) not in taken and 0 <= x < m.width and 0 <= y < m.height and m.at(x, y)[0] == floor:
            data = bytearray(mol[0].data)
            struct.pack_into("<ii", data, 0, x, y)
            data[_ID_AT] = CREATURE_ID
            data[_LEVEL_AT] = LEVEL
            at = things.index(mol[0]) + 1
            return write_things(block, things[:at] + [Thing(CREATURE_THING, bytes(data))] + things[at:]), (x, y)
    raise ValueError("no free floor tile next to Molynx")


def install(base_dir: Path, files_dir: Path, panel_dir: Path, out_dir: Path, codepage: Path) -> dict:
    """Write the changed and new files into out_dir (see the module docstring). Returns a report."""
    base_dir, files_dir, panel_dir, out_dir = map(Path, (base_dir, files_dir, panel_dir, out_dir))
    info = json.loads((files_dir.parent / "files.json").read_text())
    report = {"globals": [], "variables": [], "variables_unchanged": [], "files": 0}

    def put(rel: str, data: bytes):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        report["files"] += 1

    # texts
    text = StrFile.read(base_dir / "Data/Text/Default/Text.str", codepage)
    keys = list(TEXTS)
    text_ids = dict(zip(keys, text.append([TEXTS[k] for k in keys])))
    text_ids["tooltip"] = SHARED_TOOLTIP_ID
    put("Data/Text/Default/Text.str", text.to_bytes())
    report["text_ids"] = text_ids
    # editor base table
    put("Data/editor/Creatures.kwd", necro_for((base_dir / "Data/editor/Creatures.kwd").read_bytes(), info, text_ids))
    # every map
    for g in sorted((base_dir / MAPS).glob("*Globals.kld")):
        put(f"{MAPS}/{g.name}", necro_for(g.read_bytes(), info, text_ids))
        report["globals"].append(g.name[:-len("Globals.kld")])
    for v in sorted((base_dir / MAPS).glob("*Variables.kld")):
        old = v.read_bytes()
        new = necro_variables(old)
        if new == old:
            report["variables_unchanged"].append(v.name)
            continue
        put(f"{MAPS}/{v.name}", new)
        report["variables"].append(v.name[:-len("Variables.kld")])
    # the test map: him beside Molynx
    things, at = place_beside_molynx((base_dir / MAPS / f"{TEST_MAP}Things.kld").read_bytes(),
                                     (base_dir / MAPS / f"{TEST_MAP}Map.kld").read_bytes())
    put(f"{MAPS}/{TEST_MAP}Things.kld", things)
    report["test_map_tile"] = at
    # his files
    for p in sorted(files_dir.glob("Necro-*.kmf")):
        put(f"Data/Meshes/{p.name}", p.read_bytes())
    for p in sorted(files_dir.glob("Necro_*.png")):
        put(f"Data/EngineTextures/{p.name}", p.read_bytes())
    for name in ("necromancer.png", "necromancer-bg.png"):
        put(f"Data/EngineTextures/GUI/Creatures/Evil/{name}", (panel_dir / name).read_bytes())
    return report
