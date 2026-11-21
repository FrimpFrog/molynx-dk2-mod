"""Install the cast (the Orc 49, Quuz's Horned Reaper 50, the Dragon 51, Sejanus 52, the Gargoyle 53) into the MoreCreatures
folder as it is on Soles' PC, and put the whole cast by Keeper 1's Dungeon Heart on the test map
"Dead Walking" and the skirmish map "Furnace". Like necro_install, it reads the folder and writes
only the files that change or are new, in the same layout, into out_dir; nothing is modified in
place. Sejanus (52) and the Gargoyle (53) plug in later by one entry each (see "Adding a creature").

What it writes (paths under MoreCreatures/):
  Data/editor/maps/<map>Globals.kld   every map the folder ships: records 49..51 spliced in, each
                                      from that map's own donor (cast_records). A map still without
                                      the Necromancer (the staged copies of 8 Whynots_ maps were
                                      left at the pre-Necromancer version) first gets record 48,
                                      built exactly as necro_install builds it (checked byte for
                                      byte against the 116 tables that have him), so every table
                                      holds the creatures 1..51 with no gap.
  Data/editor/Creatures.kwd           the editor table the game loads as its base: 49..51 too
  Data/editor/maps/<map>Variables.kld a map still without the Necromancer's pool and availability
                                      gets them (necro_install.necro_variables); Dead Walking and
                                      Furnace also get the cast in Keeper 1's creature pool and
                                      availability (ENABLE). No other map's pool changes: the
                                      new creatures are not offered by other maps' portals until
                                      Soles says so.
  Data/editor/maps/Dead WalkingThings.kld   the cast placed by Keeper 1's heart (PLACEMENTS)
  Data/editor/maps/FurnaceThings.kld  new in the folder: the Furnace Things the game would load
                                      otherwise (the source folder given; see Flame below) with the
                                      whole cast placed by Keeper 1's heart
  Data/Text/Default/Text.str          his four texts per creature appended (ids after the last)
  Data/Meshes/<file>.kmf, Data/EngineTextures/<skin>.png, Data/EngineTextures/GUI/Creatures/Evil/
  <icon>.png and <icon>-bg.png        his files and panel pictures

Flame. Flame looks a file up in the alt-resources folders of flame/config.toml in order and uses the
first that has it, file by file (a resource.toml holds no priority; on Soles' PC the line is
"MoreCreatures;Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"). The
folder already ships only FurnaceGlobals and FurnaceVariables, and the game takes the rest of Furnace
from the next folder that has it: Quuz_LevelFixes, which ships the whole Furnace set (re-saved by
the editor: the same 65 things and the same terrain round Keeper 1's heart as the game's, but other
bytes and three Gold tiles below the heart turned to Rock). So a FurnaceThings.kld in MoreCreatures
replaces Furnace's Things alone, and it is built from that copy (`loaded_copy` finds it), with his
heart and floor read from the FurnaceMap that loads with it.

Placement (PLACEMENTS). Keeper 1 is player 3. His heart is the 5 x 5 block of Dungeon Heart terrain
he owns, with the heart's room thing (type 204) at its centre. A creature goes on a free tile of his
own Claimed Path touching the heart, or, where the map gives him none (Furnace: his heart is walled
in by his own Reinforced Wall, and the game's own Furnace starts his four Imps on the heart's outer
ring), on the heart's outer ring, never on its inner 3 x 3. Each placed creature is a copy of a
creature thing of Keeper 1 on that map (Molynx's on Dead Walking, his first Imp's on Furnace) with
its tile, creature id and level (5) set.

Adding a creature (Sejanus 52 and the Gargoyle 53 went in this way). Add his Member to
cast_records.CAST (donor, slots, stats, texts, icon), put his files in `files[<id>]`, move his tile
from RESERVED_TILES to PLACEMENTS, and run the installer on the folder as it then is (the installed
files laid over the staged folder) with `members` = his Member alone: every table, Variables and
Things already holding the others is left as it is, and only what changes is written.

Changing an installed creature (update_install; the Reaper made a regular creature, 15 Nov): his record
is rebuilt in place in every table that has him (same orderInEditor and text ids, every other record
and block untouched), his changed texts replaced at their ids in Text.str, and only files the folder
lacks or holds otherwise are written; his tiles, pool and availability are checked, not changed.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from scripts.build_molynx_release import AVAILABILITY, AV_CREATURE, CREATURE_POOL, ENABLE, ENTRY, _entries
from scripts.cast_records import CAST, RESERVED, all_files, build
from scripts.dk2kld_splice import CREATURE_THING, Thing, _creatures_index, _records, read_things, splice_creatures, write_things
from scripts.dk2kmf import parse_kmf
from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2str import StrFile
from scripts.necro_install import necro_variables
from scripts.necro_record import BLACK_KNIGHT_ID, build_necromancer

MAPS = "Data/editor/maps"
EDITOR_TABLE = "Data/editor/Creatures.kwd"
TEXT = "Data/Text/Default/Text.str"
PANEL = "Data/EngineTextures/GUI/Creatures/Evil"
TARGET_ROOT = "flame\\resources\\MoreCreatures"
KEEPER1 = 3
LEVEL = 5
HEART, CLAIMED = 14, 8                       # Terrain.kwd ids
MOLYNX_ID, NECRO_ID = 47, 48
ROOM_THING = 204
# A creature thing (type 200) data: x, y (i32) at 0, 4; level u8 at 14; creature id u8 at 26; owner at 27.
X_AT, LEVEL_AT, ID_AT, OWNER_AT = 0, 14, 26, 27

# map -> creature id -> tile (x, y). Molynx and the Necromancer keep their tiles on Dead Walking.
PLACEMENTS = {
    "Dead Walking": {49: (7, 36), 50: (8, 37), 51: (8, 38), 52: (8, 36), 53: (8, 39)},
    "Furnace": {47: (7, 52), 48: (8, 52), 49: (9, 52), 50: (6, 53), 51: (10, 53), 52: (6, 54), 53: (10, 54)},
}
RESERVED_TILES: dict = {"Dead Walking": {}, "Furnace": {}}
MAP_FLOOR = {"Dead Walking": "claimed", "Furnace": "heart ring", "mpd1": "claimed near"}

# My Pet Dungeon level 1 (Soles, 15 Nov; map_install). Keeper 1's heart (28..32, 28..32) is walled in on
# three sides; its one way out is the Claimed Path below it, a T: the corridor x = 30 from the heart's door
# (30, 33) down to the Gold at (30, 39), with a claimed room each side, joined to it on row 36. The cast
# stands in the two rooms on their top rows (the 7th one row lower), off that corridor and its two joins.
MPD1 = "mpd1"
MPD1_PLACEMENTS = {47: (28, 34), 48: (27, 34), 49: (26, 34), 50: (32, 34), 51: (33, 34), 52: (34, 34), 53: (28, 35)}
KEEP_CLEAR = {MPD1: frozenset({(30, y) for y in range(33, 40)} | {(29, 36), (31, 36)})}
NEAR = 3                                     # "claimed near": within this many tiles of the heart


# ------------------------------------------------------------------------------ which copy the game loads

def alt_resources(game_dir: Path) -> list[str]:
    """The alt-resources folders of flame/config.toml, in the order Flame searches them."""
    import tomllib
    cfg = tomllib.loads((Path(game_dir) / "flame/config.toml").read_text(encoding="utf-8", errors="replace"))
    return [x for x in cfg["flame"]["alt-resources"].split(";") if x]


def loaded_copy(game_dir: Path, rel: str, before: str = "MoreCreatures") -> tuple[Path, list[str]]:
    """The file the game loads for `rel` when `before` does not ship it: the first alt-resources
    folder after `before` that has it (Flame takes the first listed folder holding a file), else the
    game's own. Returns (path, the order searched). ValueError when `before` is not listed first
    of the folders that ship `rel` (its copy would then not win)."""
    game_dir = Path(game_dir)
    order = alt_resources(game_dir)
    if before not in order:
        raise ValueError(f"{before} is not in alt-resources: {order}")
    for name in order[order.index(before) + 1:]:
        p = game_dir / "flame/resources" / name / rel
        if p.exists():
            return p, order
    for name in order[:order.index(before)]:
        if (game_dir / "flame/resources" / name / rel).exists():
            raise ValueError(f"{name} comes before {before} in alt-resources and ships {rel}")
    return game_dir / rel, order


# ------------------------------------------------------------------------------ files

def files_info(folder: Path, names: list[str]) -> dict:
    """{name: kind, frames, wrap} read from his .kmf files (a missing file is an error)."""
    out = {}
    for n in names:
        k = parse_kmf((Path(folder) / f"{n}.kmf").read_bytes())
        out[n] = {"kind": k.kind, "frames": k.anim.frame_count if k.anim else 1,
                  "wrap": bool(k.anim.wrap) if k.anim else None}
    return out


# ------------------------------------------------------------------------------ tables

def cast_table(table_file: bytes, infos: dict, text_ids: dict, necro: tuple[dict, dict] | None,
               members=CAST) -> tuple[bytes, list[int]]:
    """A Globals (or Creatures.kwd) with the members it lacks (and the Necromancer first, when it
    lacks him and `necro` = (his files info, his text ids) is given). Returns (bytes, ids added)."""
    blocks = read_blocks(table_file)
    recs = _records(blocks[_creatures_index(blocks)])
    ids = {c.get("creatureId") for c in recs}
    order = max(c.get("orderInEditor") for c in recs)
    new = []
    if NECRO_ID not in ids:
        if necro is None:
            raise ValueError("the table has no Necromancer (48)")
        bks = [c for c in recs if c.get("creatureId") == BLACK_KNIGHT_ID]
        if len(bks) != 1:
            raise ValueError(f"expected one Black Knight, found {len(bks)}")
        order += 1
        new.append(build_necromancer(bks[0], order, necro[0], necro[1]))
    for m in members:
        if m.creature_id in ids:
            continue
        order += 1
        new.append(build(m, recs, order, infos[m.creature_id], text_ids[m.creature_id]))
    if not new:
        return table_file, []
    want = sorted(ids | {c.get("creatureId") for c in new})
    if want != list(range(1, len(want) + 1)):
        raise ValueError(f"the creature ids would not be dense 1..N: {want[-8:]}")
    return splice_creatures(table_file, new), [c.get("creatureId") for c in new]


def keeper_variables(variables_kld: bytes, creature_ids: list[int], player: int = KEEPER1,
                     count_from: dict[int, int] | None = None) -> bytes:
    """Variables with each creature in `player`'s creature pool and availability (ENABLE), after
    that player's last entry of each kind; one already there is left alone. The pool count is the
    count of the creature's donor in that pool (count_from), else the player's largest."""
    block, ents = _entries(variables_kld)
    own_pool = [(i, e) for i, e in enumerate(ents) if e[0] == CREATURE_POOL and e[3] == player]
    own_av = [(i, e) for i, e in enumerate(ents)
              if e[0] == AVAILABILITY and (e[1] & 0xFFFF) == AV_CREATURE and e[1] >> 16 == player]
    if not own_pool or not own_av:
        raise ValueError(f"player {player} has no creature pool or no creature availability")
    in_pool = {e[1] for _i, e in own_pool}
    in_av = {e[2] for _i, e in own_av}
    counts = {e[1]: e[2] for _i, e in own_pool}
    after: dict[int, list] = {}
    for cid in creature_ids:
        if cid not in in_pool:
            donor = (count_from or {}).get(cid)
            count = counts.get(donor) if counts.get(donor, 0) > 0 else max(counts.values())
            after.setdefault(own_pool[-1][0], []).append((CREATURE_POOL, cid, count, player))
        if cid not in in_av:
            after.setdefault(own_av[-1][0], []).append((AVAILABILITY, (player << 16) | AV_CREATURE, cid, ENABLE))
    if not after:
        return variables_kld
    out = []
    for i, e in enumerate(ents):
        out.append(e)
        out += after.get(i, [])
    return write_blocks([rebuilt_block(block, [ENTRY.pack(*e) for e in out])])


def drop_creatures(variables_kld: bytes, creature_ids) -> bytes:
    """Variables without any player's creature pool or creature availability entry for these creatures
    (a creature that never comes through the Portal, as the game's maps leave out the Skeleton, the
    Vampire and Horny); every other entry kept in order."""
    block, ents = _entries(variables_kld)
    ids = set(creature_ids)
    keep = [e for e in ents if not (e[0] == CREATURE_POOL and e[1] in ids)
            and not (e[0] == AVAILABILITY and (e[1] & 0xFFFF) == AV_CREATURE and e[2] in ids)]
    if len(keep) == len(ents):
        return variables_kld
    return write_blocks([rebuilt_block(block, [ENTRY.pack(*e) for e in keep])])


# ------------------------------------------------------------------------------ the heart and its tiles

def read_map_tiles(map_kld: bytes):
    from scripts.dk2_testmap import read_map
    return read_map(map_kld)


def keeper_heart(m, things: list[Thing], player: int = KEEPER1) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the player's 5 x 5 Dungeon Heart; ValueError unless it is one 5 x 5
    block with the heart's room thing at its centre."""
    cells = [(i % m.width, i // m.width) for i, t in enumerate(m.tiles) if t[0] == HEART and t[1] == player]
    if len(cells) != 25:
        raise ValueError(f"player {player} has {len(cells)} Dungeon Heart tiles, not 25")
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    box = (min(xs), min(ys), max(xs), max(ys))
    if box[2] - box[0] != 4 or box[3] - box[1] != 4:
        raise ValueError(f"player {player}'s heart is not a 5 x 5 block: {box}")
    centre = (box[0] + 2, box[1] + 2)
    rooms = [t for t in things if t.type == ROOM_THING and struct.unpack_from("<ii", t.data)[:2] == centre
             and t.data[-1] == player]
    if len(rooms) != 1:
        raise ValueError(f"no heart room thing of player {player} at {centre}")
    return box


def tile_ok(m, things: list[Thing], heart: tuple, xy: tuple[int, int], floor: str,
            player: int = KEEPER1, ignore: Thing | None = None, keep_clear=frozenset()) -> str | None:
    """None when (x, y) is a free tile for a creature by the heart (see the module docstring), else
    the reason it is not."""
    x, y = xy
    if not (0 <= x < m.width and 0 <= y < m.height):
        return "off the map"
    terrain, owner = m.at(x, y)[0], m.at(x, y)[1]
    x0, y0, x1, y1 = heart
    inside = x0 <= x <= x1 and y0 <= y <= y1
    ring = inside and (x in (x0, x1) or y in (y0, y1))
    if xy in keep_clear:
        return "on the way in and out of his dungeon (kept clear)"
    if floor in ("claimed", "claimed near"):
        if not (terrain == CLAIMED and owner == player):
            return f"terrain {terrain} owner {owner} is not player {player}'s Claimed Path"
        if inside or max(x0 - x, x - x1, y0 - y, y - y1) > (2 if floor == "claimed" else NEAR):
            return f"not within {2 if floor == 'claimed' else NEAR} tiles of the heart"
    elif floor == "heart ring":
        if not (terrain == HEART and owner == player and ring):
            return "not on the outer ring of player's heart"
    else:
        raise ValueError(floor)
    for t in things:
        if t is not ignore and t.type == CREATURE_THING and struct.unpack_from("<ii", t.data)[:2] == (x, y):
            return f"taken by creature {t.data[ID_AT]}"
    return None


def place_cast(things_kld: bytes, map_kld: bytes, placements: dict[int, tuple], floor: str,
               template_id: int | None, keep_clear=frozenset()) -> tuple[bytes, dict[int, tuple]]:
    """Things with a creature of each id in `placements` on its tile, level LEVEL, owned by Keeper 1,
    copied from Keeper 1's creature thing of `template_id` (None: his first creature thing); one
    already on the map is left where it is. Each new thing goes after Keeper 1's last creature."""
    block, things = read_things(things_kld)
    m = read_map_tiles(map_kld)
    heart = keeper_heart(m, things)
    k1 = [t for t in things if t.type == CREATURE_THING and t.data[OWNER_AT] == KEEPER1]
    tpl = [t for t in k1 if template_id is None or t.data[ID_AT] == template_id]
    if not tpl:
        raise ValueError(f"no Keeper 1 creature thing of id {template_id} to copy")
    have = {t.data[ID_AT] for t in k1}
    out, placed = list(things), {}
    at = out.index(k1[-1]) + 1
    for cid, xy in placements.items():
        if cid in have:
            continue
        why = tile_ok(m, out, heart, xy, floor, keep_clear=keep_clear)
        if why:
            raise ValueError(f"creature {cid} at {xy}: {why}")
        data = bytearray(tpl[0].data)
        struct.pack_into("<ii", data, X_AT, *xy)
        data[ID_AT], data[LEVEL_AT] = cid, LEVEL
        out.insert(at, Thing(CREATURE_THING, bytes(data)))
        at += 1
        placed[cid] = xy
    return write_things(block, out), placed


# ------------------------------------------------------------------------------ everything

def install(base_dir: Path, furnace_dir: Path, files: dict[int, Path], panel_dir: Path, out_dir: Path,
            codepage: Path, necro_info: dict, members=CAST, placements: dict | None = None) -> dict:
    """Write the changed and new files into out_dir (the MoreCreatures layout). Returns a report:
    every file written with its target path, the files it replaces, the text ids, the placements.
    `placements`: map -> creature id -> tile, default PLACEMENTS (a creature after the cast, Krefft, brings
    his own: krefft_install)."""
    PLACEMENTS_ = placements if placements is not None else PLACEMENTS
    base_dir, furnace_dir, panel_dir, out_dir = map(Path, (base_dir, furnace_dir, panel_dir, out_dir))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    report = {"files": [], "replaces": [], "new": [], "added": {}, "placed": {}, "variables": [],
              "necromancer_completed": []}

    def put(rel: str, data: bytes):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        report["files"].append(rel)
        (report["replaces"] if (base_dir / rel).exists() else report["new"]).append(rel)

    infos = {m.creature_id: files_info(files[m.creature_id], all_files(m)) for m in members}
    # texts
    text = StrFile.read(base_dir / TEXT, codepage)
    text_ids = {}
    for m in members:
        keys = ["name", "general", "strength", "weakness"]
        text_ids[m.creature_id] = dict(zip(keys, text.append([m.texts[k] for k in keys])))
    put(TEXT, text.to_bytes())
    report["text_ids"] = text_ids
    # the Necromancer's own text ids, from the installed editor table
    ed = read_blocks((base_dir / EDITOR_TABLE).read_bytes())
    necro_rec = [c for c in _records(ed[_creatures_index(ed)]) if c.get("creatureId") == NECRO_ID][0]
    necro_texts = {"name": necro_rec.get("nameStringId"), "general": necro_rec.get("generalDescriptionStringId"),
                   "strength": necro_rec.get("strengthStringId"), "weakness": necro_rec.get("weaknessStringId"),
                   "tooltip": necro_rec.get("tooltipStringId")}
    necro = (necro_info, necro_texts)
    # tables
    data, added = cast_table((base_dir / EDITOR_TABLE).read_bytes(), infos, text_ids, necro, members)
    put(EDITOR_TABLE, data)
    report["added"]["Creatures.kwd"] = added
    for g in sorted((base_dir / MAPS).glob("*Globals.kld")):
        data, added = cast_table(g.read_bytes(), infos, text_ids, necro, members)
        if added:
            put(f"{MAPS}/{g.name}", data)
            report["added"][g.name] = added
            if NECRO_ID in added:
                report["necromancer_completed"].append(g.name)
    # variables: the Necromancer where a map still lacks him; the cast on the two maps
    donors = {m.creature_id: m.donor_id for m in members}
    donors.update({MOLYNX_ID: 3, NECRO_ID: BLACK_KNIGHT_ID})
    for v in sorted((base_dir / MAPS).glob("*Variables.kld")):
        old = v.read_bytes()
        try:
            new = necro_variables(old)
        except ValueError:
            new = old                                   # already has 48
        name = v.name[:-len("Variables.kld")]
        if name in PLACEMENTS_:
            placed_only = {m.creature_id for m in CAST if m.placed_only}
            ids = sorted({MOLYNX_ID, NECRO_ID, *PLACEMENTS_[name], *(m.creature_id for m in members)} - placed_only)
            new = keeper_variables(new, ids, count_from=donors)
        if new != old:
            put(f"{MAPS}/{v.name}", new)
            report["variables"].append(v.name)
    if not (base_dir / MAPS / "FurnaceVariables.kld").exists():
        raise FileNotFoundError("the folder has no FurnaceVariables.kld")
    # things
    dw = PLACEMENTS_["Dead Walking"]
    data, placed = place_cast((base_dir / MAPS / "Dead WalkingThings.kld").read_bytes(),
                              (base_dir / MAPS / "Dead WalkingMap.kld").read_bytes(),
                              {k: v for k, v in dw.items() if k in {m.creature_id for m in members}},
                              MAP_FLOOR["Dead Walking"], MOLYNX_ID)
    put(f"{MAPS}/Dead WalkingThings.kld", data)
    report["placed"]["Dead Walking"] = placed
    fu = PLACEMENTS_["Furnace"]
    things_src = base_dir / MAPS / "FurnaceThings.kld"
    if not things_src.exists():
        things_src = furnace_dir / "FurnaceThings.kld"
    map_src = base_dir / MAPS / "FurnaceMap.kld"
    if not map_src.exists():
        map_src = furnace_dir / "FurnaceMap.kld"
    data, placed = place_cast(things_src.read_bytes(), map_src.read_bytes(),
                              {k: v for k, v in fu.items() if k in {m.creature_id for m in members} | {47, 48}},
                              MAP_FLOOR["Furnace"], None)
    put(f"{MAPS}/FurnaceThings.kld", data)
    report["placed"]["Furnace"] = placed
    report["furnace_sources"] = {"things": str(things_src), "map": str(map_src)}
    # his files
    for m in members:
        for n in all_files(m):
            put(f"Data/Meshes/{n}.kmf", (files[m.creature_id] / f"{n}.kmf").read_bytes())
        for t in m.textures:
            put(f"Data/EngineTextures/{t}.png", (files[m.creature_id] / f"{t}.png").read_bytes())
        for suffix in ("", "-bg"):
            put(f"{PANEL}/{m.icon}{suffix}.png", (panel_dir / f"{m.icon}{suffix}.png").read_bytes())
    report["reserved"] = RESERVED
    return report


TEXT_KEYS = {"name": "nameStringId", "general": "generalDescriptionStringId", "strength": "strengthStringId",
             "weakness": "weaknessStringId"}


def replace_records(table_file: bytes, infos: dict, members) -> tuple[bytes, list[int]]:
    """A Globals (or Creatures.kwd) with the records of `members` rebuilt in place: each from this
    table's own donor (as cast_table builds him), keeping his orderInEditor and his text ids. Every
    other record and block is left as it is. Returns (bytes, ids whose record changed)."""
    from scripts.dk2kwd_creatures import creatures_block
    blocks = read_blocks(table_file)
    at = _creatures_index(blocks)
    recs = _records(blocks[at])
    by_id = {c.get("creatureId"): i for i, c in enumerate(recs)}
    changed = []
    for m in members:
        if m.creature_id not in by_id:
            raise ValueError(f"the table has no record {m.creature_id} to replace")
        old = recs[by_id[m.creature_id]]
        ids = {k: old.get(f) for k, f in TEXT_KEYS.items()}
        new = build(m, recs, old.get("orderInEditor"), infos[m.creature_id], ids)
        if new.raw != old.raw:
            recs[by_id[m.creature_id]] = new
            changed.append(m.creature_id)
    if not changed:
        return table_file, []
    blocks[at] = creatures_block(recs, blocks[at])
    return write_blocks(blocks), changed


def update_install(base_dir: Path, files: dict[int, Path], out_dir: Path, codepage: Path, members) -> dict:
    """For a creature already installed (Soles, 15 Nov: the Reaper made a regular creature): his record
    rebuilt in place in every table that has him, his texts replaced at their ids in Text.str, and any of
    his files the folder lacks or holds otherwise. His tiles, pools and availability stay as installed
    (checked present). Writes into out_dir only what changes; returns a report like install's."""
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    report = {"files": [], "replaces": [], "new": [], "replaced_records": {}, "texts": {}}

    def put(rel: str, data: bytes):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        report["files"].append(rel)
        (report["replaces"] if (base_dir / rel).exists() else report["new"]).append(rel)

    infos = {m.creature_id: files_info(files[m.creature_id], all_files(m)) for m in members}
    # texts, at the ids his installed record names (the same in every table: checked below)
    ed = read_blocks((base_dir / EDITOR_TABLE).read_bytes())
    installed = {c.get("creatureId"): c for c in _records(ed[_creatures_index(ed)])}
    text = StrFile.read(base_dir / TEXT, codepage)
    before = list(text.entries)
    for m in members:
        rec = installed[m.creature_id]
        for k, f in TEXT_KEYS.items():
            i = rec.get(f)
            if text.text(i) != m.texts[k]:
                report["texts"][i] = {"was": text.text(i), "now": m.texts[k]}
                text.entries[i] = text.encode(m.texts[k])
    if text.entries != before:
        put(TEXT, text.to_bytes())
    # tables
    tables = [base_dir / EDITOR_TABLE, *sorted((base_dir / MAPS).glob("*Globals.kld"))]
    for t in tables:
        tb = read_blocks(t.read_bytes())
        recs = {c.get("creatureId"): c for c in _records(tb[_creatures_index(tb)])}
        for m in members:
            for k, f in TEXT_KEYS.items():
                if recs[m.creature_id].get(f) != installed[m.creature_id].get(f):
                    raise ValueError(f"{t.name}: record {m.creature_id}'s {f} differs from Creatures.kwd's")
        data, changed = replace_records(t.read_bytes(), infos, members)
        if changed:
            rel = EDITOR_TABLE if t.name == "Creatures.kwd" else f"{MAPS}/{t.name}"
            put(rel, data)
            report["replaced_records"][t.name] = changed
    # Keeper 1's pool and availability: a placed_only creature on none (he never comes through the
    # Portal), the others as installed; his tiles stay as installed
    for v in sorted((base_dir / MAPS).glob("*Variables.kld")):
        old = v.read_bytes()
        new = drop_creatures(old, [m.creature_id for m in members if m.placed_only])
        if new != old:
            put(f"{MAPS}/{v.name}", new)
            report.setdefault("variables", []).append(v.name)
    for name, tiles in PLACEMENTS.items():
        block, ents = _entries((base_dir / MAPS / f"{name}Variables.kld").read_bytes())
        for m in members:
            if m.placed_only:
                continue
            if not any(e[0] == CREATURE_POOL and e[1] == m.creature_id and e[3] == KEEPER1 for e in ents):
                raise ValueError(f"{name}: creature {m.creature_id} is not in Keeper 1's pool")
            if not any(e[0] == AVAILABILITY and e[2] == m.creature_id and e[1] >> 16 == KEEPER1 for e in ents):
                raise ValueError(f"{name}: creature {m.creature_id} is not available to Keeper 1")
        _blk, things = read_things((base_dir / MAPS / f"{name}Things.kld").read_bytes())
        for m in members:
            xy = tiles[m.creature_id]
            if not any(t.type == CREATURE_THING and t.data[ID_AT] == m.creature_id and t.data[OWNER_AT] == KEEPER1
                       and struct.unpack_from("<ii", t.data)[:2] == xy for t in things):
                raise ValueError(f"{name}: creature {m.creature_id} is not on {xy}")
    # his files: only those the folder lacks or holds otherwise
    for m in members:
        for n in all_files(m):
            rel, data = f"Data/Meshes/{n}.kmf", (files[m.creature_id] / f"{n}.kmf").read_bytes()
            if not (base_dir / rel).exists() or (base_dir / rel).read_bytes() != data:
                put(rel, data)
        for tx in m.textures:
            rel, data = f"Data/EngineTextures/{tx}.png", (files[m.creature_id] / f"{tx}.png").read_bytes()
            if not (base_dir / rel).exists() or (base_dir / rel).read_bytes() != data:
                put(rel, data)
    return report


def map_install(base_dir: Path, source_dir: Path, stem: str, placements: dict[int, tuple], out_dir: Path,
                members=CAST) -> dict:
    """Put the cast by Keeper 1's heart on one more map (My Pet Dungeon level 1): its Things built from
    the copy the game loads (`source_dir`, from loaded_copy, unless the folder already ships one), each
    creature copied from Keeper 1's first creature thing (an Imp, as on Furnace) at level LEVEL; and every
    member but the placed_only ones in Keeper 1's creature pool and availability in the folder's Variables
    (the pool count of his donor there, else Keeper 1's largest). Nothing else changes."""
    base_dir, source_dir, out_dir = Path(base_dir), Path(source_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    report = {"files": [], "replaces": [], "new": [], "placed": {}, "variables": []}

    def put(rel: str, data: bytes):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        report["files"].append(rel)
        (report["replaces"] if (base_dir / rel).exists() else report["new"]).append(rel)

    def src(name: str) -> Path:
        return base_dir / MAPS / name if (base_dir / MAPS / name).exists() else source_dir / name

    things, mapf = src(f"{stem}Things.kld"), src(f"{stem}Map.kld")
    data, placed = place_cast(things.read_bytes(), mapf.read_bytes(), placements, MAP_FLOOR[stem], None,
                              KEEP_CLEAR.get(stem, frozenset()))
    put(f"{MAPS}/{stem}Things.kld", data)
    report["placed"][stem] = placed
    report["sources"] = {"things": str(things), "map": str(mapf)}
    v = base_dir / MAPS / f"{stem}Variables.kld"
    placed_only = {m.creature_id for m in CAST if m.placed_only}
    donors = {m.creature_id: m.donor_id for m in members}
    donors.update({MOLYNX_ID: 3, NECRO_ID: BLACK_KNIGHT_ID})
    old = v.read_bytes()
    new = drop_creatures(keeper_variables(old, sorted(set(placements) - placed_only), count_from=donors),
                         placed_only)
    if new != old:
        put(f"{MAPS}/{v.name}", new)
        report["variables"].append(v.name)
    return report


def files_install(base_dir: Path, out_dir: Path, files: dict[str, Path]) -> dict:
    """Plain file swaps (Sejanus' skins, recolour B, 15 Nov): each `rel: source` is written into out_dir
    when the folder lacks it or holds other bytes; nothing else. Returns a report like install's."""
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    report = {"files": [], "replaces": [], "new": [], "unchanged": []}
    for rel, src in files.items():
        data = Path(src).read_bytes()
        if (base_dir / rel).exists() and (base_dir / rel).read_bytes() == data:
            report["unchanged"].append(rel)
            continue
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        report["files"].append(rel)
        (report["replaces"] if (base_dir / rel).exists() else report["new"]).append(rel)
    return report


def write_lists(report: dict, out_dir: Path) -> None:
    """install_list.txt (every file, its target path on the PC, new or replacing) and
    backup_list.txt (the current files it replaces, to copy aside first)."""
    out_dir = Path(out_dir)
    lines = [f"{'REPLACES' if r in report['replaces'] else 'NEW     '}  {r}  ->  {TARGET_ROOT}\\{r.replace('/', chr(92))}"
             for r in report["files"]]
    (out_dir / "install_list.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "backup_list.txt").write_text(
        "\n".join(f"{TARGET_ROOT}\\{r.replace('/', chr(92))}" for r in report["replaces"]) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Install the Orc, Quuz's Horned Reaper, the Dragon and Sejanus (49-52) "
                                            "into a MoreCreatures folder, writing only changed and new files.")
    p.add_argument("--base", type=Path, required=True, help="the MoreCreatures folder as installed")
    p.add_argument("--furnace", type=Path, required=True,
                   help="folder with the FurnaceThings/FurnaceMap the game loads today (Quuz_LevelFixes' or the game's)")
    p.add_argument("--orc", type=Path, required=True)
    p.add_argument("--reaper", type=Path, required=True)
    p.add_argument("--dragon", type=Path, required=True)
    p.add_argument("--sejanus", type=Path, required=True)
    p.add_argument("--gargoyle", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--necro-info", type=Path, required=True, help="the Necromancer's files.json (v5)")
    p.add_argument("--codepage", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="empty folder; the files go in <out>/MoreCreatures")
    a = p.parse_args(argv)
    rep = install(a.base, a.furnace, {49: a.orc, 50: a.reaper, 51: a.dragon, 52: a.sejanus, 53: a.gargoyle}, a.panel, a.out / "MoreCreatures",
                  a.codepage, json.loads(a.necro_info.read_text()))
    write_lists(rep, a.out)
    print(json.dumps({k: rep[k] for k in ("placed", "text_ids", "necromancer_completed")}, indent=1))
    print(f"{len(rep['files'])} files ({len(rep['new'])} new, {len(rep['replaces'])} replacing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
