"""Generate and check the flat Molynx test map, built from the shipped skirmish map "Dead Walking".

`testmap_files(stem, molynx_in_pool=...)` returns the six map files of a 69 x 69 map in which
Keeper 1 (player 3, the human) owns a large pre-claimed open area with its Dungeon Heart, a Portal,
5 Gems and 2 Gold tiles, six Imps and Molynx (creature id 47) next to the heart, while Keeper 2
(player 4) sits in a small chamber sealed by Impenetrable Rock with four Imps. The Globals file is
not made here: the mod builder splices it. `validate_testmap` re-reads the bytes and checks every
layout rule; `preview_text` / `preview_png` draw the map. The reconnaissance behind every value is
in `.superpowers/sdd/2026-11-05-molynx-testmap/task-1-recon.md`; in short:

Containers. Every file is one block: u32 signature, u32 4, u32 file size, u32 signature + 1,
u32 headerEndOffset; the header is 28 + headerEndOffset bytes (56 for Things, Players, Variables and
the level file, 36 for Map, 60 for Triggers) and ends with u32 checkTwo (signature + 2, + 3 in the
level file) and u32 content size. Headers are copied from the base files (timestamps included), and
only file size, item count and content size are recomputed, so the output is deterministic.

Map (0x64). width, height at bytes 20, 24; then width * height tiles of 4 bytes (terrain id,
1-based in Terrain.kwd; player id; flag; unknown). All tiles are copies of base tiles: Dead Walking's
two 5x5 Dungeon Hearts, its Rock / Impenetrable / Dirt / Claimed / Gold / Gems tiles and its outer
ring of Edge of Map, and Siege's 3x3 Keeper 1 Portal (Dead Walking has no Portal).

Things (0xBE, see dk2kld_splice). A working heart in Dead Walking is its 25 map tiles plus one
type-204 "room" thing at the centre tile (x, y, 0, 0, dir 0, 0, initialHealth 100, room type 14,
owner); it has no objects. Siege's Keeper 1 Portal has no thing at all. Creatures are copies of
Dead Walking's own records with x, y (and for Molynx the creature id) changed: its first Keeper 1 /
Keeper 2 Imps (level 3, flags 9) and its Keeper 1 Bile Demon (level 2, flags 41, the record Test A
turned into Molynx). Nothing else is kept: the neutral creatures, doors, traps, objects, hero gate,
hero parties and action points of Dead Walking serve its own dungeon and script only.

Players (0xB4). 205-byte records; Keeper 1 (record 2) gets startingGold 100000 (i32 at +0); its
isAi (u32 at +4) is already 0. Keeper 1's heart keeps its base position, so its camera (4, 37)
still looks at it. Keeper 2's record is unchanged.

Variables (0xE6). 16-byte entries (i32 variable id, then 12 bytes). Base entries keep their order;
for player 3 every room, door, trap and keeper-spell availability (id 2: u16 type, u16 player,
i32 type id, i32 value) is set to ENABLE (3), the missing ones are appended; the creature pool
(id 1: creature id, value, player) and creature availability of Siege's player 3 are appended, and
creature 47 (pool 50, ENABLE) only for the portal variant. The type ids: rooms are the BUILDABLE
rooms of Rooms.kwd (room ids 1, 2, 4, 6..17) plus room ids the base already lists for player 3
(the Portal, 3); doors 1..6, traps 1..11 and 16, spells 1..11 and 13: the ids Doors.kwd, Traps.kwd
and KeeperSpells.kwd define for keepers and that shipped maps give to keepers (the hero doors and
traps, the trap-table doors 12/13 and the OneShotReaper spell 14 are left out).

Triggers (0xD2). The empty 60-byte file of the editor (`AutoSaveTriggers.kld`, the same layout as
the empty Triggers of 14 shipped skirmish maps such as Small, U Turn and Warrens), with the base
file's timestamps.

Level file (.kwd, 0xDC). The base file with the name (wchar[64] at byte 56) and description
(wchar[1024] at byte 184) replaced and its path list (after the content: u32 222, u32 size, then
72-byte records of u32 type, u32 0, char path[64]) pointing at `<stem>` files: the game opens the
files these paths name (CWorld_loadLevel_unk2 at 0x51EA40), so a new stem must name its own files.
Flags, the creature id list and every other byte stay; the level file stores no width or height.

LICENCE: GPL-3.0.
"""
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from scripts.dk2kld_splice import Thing, map_source_dir, read_things, write_things
from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2paths import EDITOR_DIR

BASE = "Dead Walking"
PORTAL_SOURCE = "Siege"  # also the source of the creature pool
SUFFIXES = (".kwd", "Map.kld", "Things.kld", "Players.kld", "Triggers.kld", "Variables.kld")

# Terrain ids (1-based index into Terrain.kwd).
IMPENETRABLE, ROCK, DIRT, GOLD, GEMS, CLAIMED, PORTAL, HEART, EDGE = 1, 2, 3, 6, 7, 8, 12, 14, 30
REINFORCED_WALL = 9
SOLID = frozenset({IMPENETRABLE, ROCK, GOLD, GEMS, REINFORCED_WALL, EDGE})  # everything else is passable
FLOOR = frozenset({DIRT, CLAIMED})
NEUTRAL, KEEPER1, KEEPER2 = 2, 3, 4

MOLYNX_ID, IMP_ID, BILE_DEMON_ID = 47, 1, 3
STARTING_GOLD = 100000
POOL_VALUE = 50
ENABLE = 3
CREATURE_POOL, AVAILABILITY = 1, 2
AV_ROOM, AV_CREATURE, AV_TRAP, AV_DOOR, AV_SPELL = 1, 2, 3, 4, 5
DOOR_IDS = tuple(range(1, 7))
TRAP_IDS = (*range(1, 12), 16)
SPELL_IDS = (*range(1, 12), 13)
DESCRIPTION = "Flat test map: Keeper 2 is sealed in, no enemies, no script."

# ---- layout (tile coordinates, inclusive) -------------------------------------------------------
# Keeper 1's open area; its heart stays where the base has it (x 2..6, y 35..39).
AREA = (1, 21, 33, 53)  # x0, y0, x1, y1: 33 x 33 tiles
ROCK_BAND = (0, 19, 35, 55)  # Rock around the area (2 tiles thick on the open sides)
PORTAL_AT = (20, 36)  # top-left tile of the 3x3 portal
GEMS_AT = ((34, 25), (34, 29), (34, 33), (34, 41), (34, 45))
GOLD_AT = ((10, 20), (24, 20))
K1_IMPS_AT = ((7, 35), (7, 39), (3, 34), (5, 34), (3, 40), (5, 40))
MOLYNX_AT = (7, 37)
K2_HEART_AT = (56, 56)  # top-left tile of Keeper 2's moved heart
K2_RING = 1  # chamber floor around Keeper 2's heart (Claimed Path)
K2_IMPS_AT = ((55, 55), (61, 55), (55, 61), (61, 61))

PLAYER_SIZE = 205
VARIABLE_SIZE = 16
LEVEL_NAME_AT, LEVEL_NAME_CHARS = 56, 64
LEVEL_DESC_AT, LEVEL_DESC_CHARS = 56 + 128, 1024
LEVEL_FLAGS_AT = 56 + 25484
PATH_RECORD, PATH_CHARS = 72, 64
MAPS_PREFIX = "Data\\Editor\\Maps\\"


# ---- containers ---------------------------------------------------------------------------------

def header_size(data: bytes) -> int:
    """28 + headerEndOffset: 56 for most files, 36 for Map.kld, 60 for Triggers.kld."""
    return 28 + struct.unpack_from("<I", data, 16)[0]


def _set_sizes(header: bytes, total: int, count: int | None = None) -> bytes:
    """`header` with file size and content size (its last u32) set; item count (u32 at 20) if given."""
    out = bytearray(header)
    struct.pack_into("<I", out, 8, total)
    struct.pack_into("<I", out, len(out) - 4, total - len(out))
    if count is not None:
        struct.pack_into("<I", out, 20, count)
    return bytes(out)


@dataclass(frozen=True)
class MapFile:
    header: bytes
    width: int
    height: int
    tiles: tuple[bytes, ...]

    def at(self, x: int, y: int) -> bytes:
        return self.tiles[y * self.width + x]


def read_map(data: bytes) -> MapFile:
    """ValueError unless `data` is one Map block (0x64) holding width * height 4-byte tiles."""
    if struct.unpack_from("<I", data, 0)[0] != 0x64:
        raise ValueError("not a Map.kld (signature 0x64)")
    hsz = header_size(data)
    width, height = struct.unpack_from("<II", data, 20)
    if len(data) != hsz + 4 * width * height:
        raise ValueError(f"Map.kld of {len(data)} bytes does not hold {width} x {height} tiles")
    return MapFile(data[:hsz], width, height,
                   tuple(data[hsz + 4 * i:hsz + 4 * i + 4] for i in range(width * height)))


def write_map(m: MapFile) -> bytes:
    return m.header + b"".join(m.tiles)


def _records(data: bytes, size: int, signature: int) -> tuple:
    blocks = read_blocks(data)
    if len(blocks) != 1 or blocks[0].signature != signature:
        raise ValueError(f"expected one block 0x{signature:X}")
    (block,) = blocks
    if len(block.content) % size or len(block.content) // size != block.item_count:
        raise ValueError(f"block 0x{signature:X}: {len(block.content)} bytes do not hold "
                         f"{block.item_count} records of {size}")
    return block, [block.content[i:i + size] for i in range(0, len(block.content), size)]


def read_players(data: bytes):
    return _records(data, PLAYER_SIZE, 0xB4)


def read_variables(data: bytes):
    return _records(data, VARIABLE_SIZE, 0xE6)


def write_records(block, records: list[bytes]) -> bytes:
    return write_blocks([rebuilt_block(block, records)])


@dataclass(frozen=True)
class LevelFile:
    header: bytes  # 56 bytes
    content: bytes  # LevelData (name, description, flags, ...)
    paths_header: bytes  # u32 222, u32 size
    paths: tuple[bytes, ...]  # 72-byte records
    rest: bytes  # the creature id list


def read_level(data: bytes) -> LevelFile:
    if struct.unpack_from("<I", data, 0)[0] != 0xDC:
        raise ValueError("not a level .kwd (signature 0xDC)")
    hsz = header_size(data)
    (content_size,) = struct.unpack_from("<I", data, hsz - 4)
    (n_paths,) = struct.unpack_from("<H", data, 20)
    end = hsz + content_size
    if struct.unpack_from("<I", data, end)[0] != 222:
        raise ValueError("level .kwd: no path list (222) after the content")
    p = end + 8
    paths = tuple(data[p + PATH_RECORD * i:p + PATH_RECORD * (i + 1)] for i in range(n_paths))
    return LevelFile(data[:hsz], data[hsz:end], data[end:end + 8], paths,
                     data[p + PATH_RECORD * n_paths:])


def write_level(lv: LevelFile) -> bytes:
    return lv.header + lv.content + lv.paths_header + b"".join(lv.paths) + lv.rest


def _wstr(data: bytes, at: int, chars: int) -> str:
    return data[at:at + 2 * chars].decode("utf-16-le").split("\0")[0]


def _put_wstr(buf: bytearray, at: int, chars: int, text: str) -> None:
    raw = text.encode("utf-16-le")
    if len(raw) > 2 * (chars - 1):
        raise ValueError(f"{text!r} does not fit {chars - 1} characters")
    buf[at:at + 2 * chars] = raw.ljust(2 * chars, b"\0")


def level_name(kwd: bytes) -> str:
    return _wstr(kwd, LEVEL_NAME_AT, LEVEL_NAME_CHARS)


def level_paths(kwd: bytes) -> list[str]:
    return [p[8:].split(b"\0")[0].decode("latin-1") for p in read_level(kwd).paths]


def _repath(record: bytes, base: str, stem: str) -> bytes:
    path = record[8:].split(b"\0")[0].decode("latin-1")
    prefix = MAPS_PREFIX + base
    if not path.startswith(prefix):
        return record
    new = (MAPS_PREFIX + stem + path[len(prefix):]).encode("latin-1")
    if len(new) >= PATH_CHARS:
        raise ValueError(f"path {new!r} does not fit {PATH_CHARS - 1} characters")
    return record[:8] + new.ljust(PATH_CHARS, b"\0")


def rewrite_unchanged(suffix: str, data: bytes) -> bytes:
    """`data` (a map file with this suffix) parsed and written back with the code paths the
    generator uses; equals `data` for a well-formed file."""
    if suffix == ".kwd":
        return write_level(read_level(data))
    if suffix == "Map.kld":
        return write_map(read_map(data))
    if suffix == "Things.kld":
        return write_things(*read_things(data))
    if suffix == "Players.kld":
        return write_records(*read_players(data))
    if suffix == "Variables.kld":
        return write_records(*read_variables(data))
    if suffix == "Triggers.kld":
        # Only the file size is recomputed: in a non-empty Triggers file the header's last u32 is
        # not the content size (Dead Walking: 272 in a 1668-byte file), so it is kept as read.
        hsz = header_size(data)
        header = bytearray(data[:hsz])
        struct.pack_into("<I", header, 8, len(data))
        return bytes(header) + data[hsz:]
    raise ValueError(f"unknown map file suffix {suffix!r}")


# ---- reading the inputs -------------------------------------------------------------------------

def _load(name: str) -> dict[str, bytes]:
    folder = map_source_dir(name)
    return {s: (folder / (name + s)).read_bytes() for s in SUFFIXES}


def _cells(m: MapFile, pred) -> list[tuple[int, int]]:
    return [(i % m.width, i // m.width) for i, t in enumerate(m.tiles) if pred(t)]


def _bbox(cells) -> tuple[int, int, int, int]:
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class Structure:
    """A room copied from a map: its tiles relative to the top-left corner and the things on it."""
    origin: tuple[int, int]
    tiles: tuple[tuple[int, int, bytes], ...]  # (dx, dy, tile)
    things: tuple[Thing, ...]  # positioned at the source map's coordinates


def _structure(m: MapFile, things: list[Thing], terrain: int, owner: int) -> Structure:
    cells = _cells(m, lambda t: t[0] == terrain and t[1] == owner)
    if not cells:
        raise ValueError(f"no terrain {terrain} tiles of player {owner}")
    x0, y0, x1, y1 = _bbox(cells)
    on = tuple(t for t in things if t.type in (194, 204) and x0 <= _xy(t)[0] <= x1
               and y0 <= _xy(t)[1] <= y1)
    return Structure((x0, y0), tuple((x - x0, y - y0, m.at(x, y)) for x, y in cells), on)


def _xy(thing: Thing) -> tuple[int, int]:
    return struct.unpack_from("<ii", thing.data)


def _moved(thing: Thing, x: int, y: int) -> Thing:
    return Thing(thing.type, struct.pack("<ii", x, y) + thing.data[8:])


def _creature(thing: Thing, x: int, y: int, creature_id: int | None = None) -> Thing:
    data = bytearray(_moved(thing, x, y).data)
    if creature_id is not None:
        data[26] = creature_id
    return Thing(thing.type, bytes(data))


def _first_creature(things: list[Thing], creature_id: int, owner: int) -> Thing:
    for t in things:
        if t.type == 200 and t.data[26] == creature_id and t.data[27] == owner:
            return t
    raise ValueError(f"no creature {creature_id} of player {owner} in the base Things")


def _plain_tile(m: MapFile, terrain: int, owner: int) -> bytes:
    for t in m.tiles:
        if t[0] == terrain and t[1] == owner:
            return t
    raise ValueError(f"the base map has no tile of terrain {terrain}, player {owner}")


def _buildable_rooms() -> list[int]:
    """Room ids of the Rooms.kwd records flagged BUILDABLE (u32 flags at +880 bit 7, id at +914)."""
    (block,) = read_blocks((EDITOR_DIR / "Rooms.kwd").read_bytes())
    size = len(block.content) // block.item_count
    ids = []
    for i in range(block.item_count):
        rec = block.content[i * size:(i + 1) * size]
        if struct.unpack_from("<I", rec, 880)[0] & 0x80:
            ids.append(rec[914])
    return sorted(ids)


def _var(entry: bytes) -> tuple[int, int, int, int]:
    return struct.unpack("<iiii", entry)


def _av_key(e) -> tuple[int, int, int]:
    """(type, player, type id) of an availability entry."""
    return e[1] & 0xFFFF, e[1] >> 16, e[2]


def _availability(av_type: int, player: int, type_id: int, value: int) -> bytes:
    return struct.pack("<iHHii", AVAILABILITY, av_type, player, type_id, value)


def _keeper_types(base_entries) -> list[tuple[int, int]]:
    """(availability type, type id) pairs Keeper 1 must have ENABLE: see the module docstring."""
    want = {(AV_ROOM, i) for i in _buildable_rooms()}
    want |= {(AV_DOOR, i) for i in DOOR_IDS} | {(AV_TRAP, i) for i in TRAP_IDS}
    want |= {(AV_SPELL, i) for i in SPELL_IDS}
    want |= {(t, i) for e in base_entries if e[0] == AVAILABILITY
             for t, p, i in [_av_key(e)] if p == KEEPER1 and t != AV_CREATURE}
    return sorted(want)


def _siege_creatures() -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Siege's player-3 pool entries and creature availability entries, in file order."""
    _block, recs = read_variables(_load(PORTAL_SOURCE)["Variables.kld"])
    ents = [_var(r) for r in recs]
    pool = [e for e in ents if e[0] == CREATURE_POOL and e[3] == KEEPER1]
    av = [e for e in ents if e[0] == AVAILABILITY and _av_key(e)[:2] == (AV_CREATURE, KEEPER1)]
    return pool, av


# ---- the generator ------------------------------------------------------------------------------

def _rect(r):
    x0, y0, x1, y1 = r
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def _build_map(base: MapFile, k1: Structure, k2: Structure, portal: Structure) -> MapFile:
    w, h = base.width, base.height
    impenetrable = _plain_tile(base, IMPENETRABLE, NEUTRAL)
    rock = _plain_tile(base, ROCK, NEUTRAL)
    gems = _plain_tile(base, GEMS, NEUTRAL)
    gold = _plain_tile(base, GOLD, NEUTRAL)
    claimed1 = _plain_tile(base, CLAIMED, KEEPER1)
    claimed2 = _plain_tile(base, CLAIMED, KEEPER2)
    grid = [impenetrable] * (w * h)

    def put(x, y, tile):
        grid[y * w + x] = tile

    for x, y in _rect(ROCK_BAND):
        put(x, y, rock)
    for x, y in _rect(AREA):
        put(x, y, claimed1)
    for x, y in GEMS_AT:
        put(x, y, gems)
    for x, y in GOLD_AT:
        put(x, y, gold)
    for dx, dy, tile in k1.tiles:  # Keeper 1's heart stays at its base position
        put(k1.origin[0] + dx, k1.origin[1] + dy, tile)
    for dx, dy, tile in portal.tiles:
        put(PORTAL_AT[0] + dx, PORTAL_AT[1] + dy, tile)
    kx1, ky1 = _bbox([(dx, dy) for dx, dy, _ in k2.tiles])[2:]
    for x, y in _rect((K2_HEART_AT[0] - K2_RING, K2_HEART_AT[1] - K2_RING,
                       K2_HEART_AT[0] + kx1 + K2_RING, K2_HEART_AT[1] + ky1 + K2_RING)):
        put(x, y, claimed2)
    for dx, dy, tile in k2.tiles:
        put(K2_HEART_AT[0] + dx, K2_HEART_AT[1] + dy, tile)
    for i in range(w * h):  # the outer ring is the base's, byte for byte
        x, y = i % w, i // w
        if x in (0, w - 1) or y in (0, h - 1):
            grid[i] = base.tiles[i]
    return MapFile(base.header, w, h, tuple(grid))


def _shifted(structure: Structure, to: tuple[int, int]) -> list[Thing]:
    dx, dy = to[0] - structure.origin[0], to[1] - structure.origin[1]
    return [_moved(t, _xy(t)[0] + dx, _xy(t)[1] + dy) for t in structure.things]


def _variables(base_vars: bytes, molynx_in_pool: bool) -> bytes:
    block, recs = read_variables(base_vars)
    ents = [_var(r) for r in recs]
    out = list(recs)
    have = {}
    for i, e in enumerate(ents):
        if e[0] == AVAILABILITY:
            have[_av_key(e)] = i
    for (t, type_id) in _keeper_types(ents):
        key = (t, KEEPER1, type_id)
        if key in have:
            out[have[key]] = _availability(t, KEEPER1, type_id, ENABLE)
        else:
            out.append(_availability(t, KEEPER1, type_id, ENABLE))
            have[key] = len(out) - 1
    pool, av = _siege_creatures()
    pool_have = {(e[1], e[3]) for e in ents if e[0] == CREATURE_POOL}
    for e in pool:
        if (e[1], e[3]) not in pool_have:
            out.append(struct.pack("<iiii", *e))
    if molynx_in_pool and (MOLYNX_ID, KEEPER1) not in pool_have:
        out.append(struct.pack("<iiii", CREATURE_POOL, MOLYNX_ID, POOL_VALUE, KEEPER1))
    creature_ids = [e[2] for e in av] + ([MOLYNX_ID] if molynx_in_pool else [])
    for type_id in creature_ids:
        key = (AV_CREATURE, KEEPER1, type_id)
        if key in have:
            out[have[key]] = _availability(AV_CREATURE, KEEPER1, type_id, ENABLE)
        else:
            out.append(_availability(AV_CREATURE, KEEPER1, type_id, ENABLE))
            have[key] = len(out) - 1
    return write_records(block, out)


def _empty_triggers(base_triggers: bytes) -> bytes:
    """The editor's empty Triggers (counts 0, unknown 1, no content) with the base's timestamps."""
    hsz = header_size(base_triggers)
    out = bytearray(base_triggers[:hsz])
    struct.pack_into("<III", out, 20, 0, 0, 1)
    return _set_sizes(bytes(out), hsz)


def _level(base_kwd: bytes, stem: str, display_name: str) -> bytes:
    lv = read_level(base_kwd)
    buf = bytearray(base_kwd)
    _put_wstr(buf, LEVEL_NAME_AT, LEVEL_NAME_CHARS, display_name)
    _put_wstr(buf, LEVEL_DESC_AT, LEVEL_DESC_CHARS, DESCRIPTION)
    hsz = len(lv.header)
    lv = LevelFile(lv.header, bytes(buf[hsz:hsz + len(lv.content)]), lv.paths_header,
                   tuple(_repath(p, BASE, stem) for p in lv.paths), lv.rest)
    return write_level(lv)


def testmap_files(stem: str, *, molynx_in_pool: bool, display_name: str = "Molynx Test") -> dict[str, bytes]:
    """The six files of the flat test map, named `<stem>` + suffix (see the module docstring)."""
    base, siege = _load(BASE), _load(PORTAL_SOURCE)
    bmap, smap = read_map(base["Map.kld"]), read_map(siege["Map.kld"])
    things_block, bthings = read_things(base["Things.kld"])
    _sblock, sthings = read_things(siege["Things.kld"])
    k1 = _structure(bmap, bthings, HEART, KEEPER1)
    k2 = _structure(bmap, bthings, HEART, KEEPER2)
    portal = _structure(smap, sthings, PORTAL, KEEPER1)

    imp1 = _first_creature(bthings, IMP_ID, KEEPER1)
    imp2 = _first_creature(bthings, IMP_ID, KEEPER2)
    bile = _first_creature(bthings, BILE_DEMON_ID, KEEPER1)
    things = [_creature(imp1, x, y) for x, y in K1_IMPS_AT]
    things.append(_creature(bile, *MOLYNX_AT, creature_id=MOLYNX_ID))
    things += [_creature(imp2, x, y) for x, y in K2_IMPS_AT]
    things += _shifted(k1, k1.origin) + _shifted(portal, PORTAL_AT) + _shifted(k2, K2_HEART_AT)

    _pblock, players = read_players(base["Players.kld"])
    players = list(players)
    k1_rec = next(i for i, r in enumerate(players) if r[-37] == KEEPER1)
    players[k1_rec] = struct.pack("<iI", STARTING_GOLD, 0) + players[k1_rec][8:]

    return {
        stem + ".kwd": _level(base[".kwd"], stem, display_name),
        stem + "Map.kld": write_map(_build_map(bmap, k1, k2, portal)),
        stem + "Things.kld": write_things(things_block, things),
        stem + "Players.kld": write_records(_pblock, players),
        stem + "Triggers.kld": _empty_triggers(base["Triggers.kld"]),
        stem + "Variables.kld": _variables(base["Variables.kld"], molynx_in_pool),
    }


# ---- the validator ------------------------------------------------------------------------------

def _flood(m: MapFile, starts, ok) -> set[tuple[int, int]]:
    seen, todo = set(starts), list(starts)
    while todo:
        x, y = todo.pop()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < m.width and 0 <= ny < m.height and (nx, ny) not in seen and ok(m.at(nx, ny)):
                seen.add((nx, ny))
                todo.append((nx, ny))
    return seen


def _k1_area(m: MapFile) -> set[tuple[int, int]]:
    """Tiles 4-connected to Keeper 1's heart through floor, portal and heart tiles of Keeper 1
    (Dirt Path may be neutral)."""
    starts = _cells(m, lambda t: t[0] == HEART and t[1] == KEEPER1)
    ok = lambda t: (t[0] in (CLAIMED, PORTAL, HEART) and t[1] == KEEPER1) or t[0] == DIRT
    return _flood(m, starts, ok) if starts else set()


def _check_structure(errors, rule, what, m, things, src: Structure, terrain, owner):
    cells = _cells(m, lambda t: t[0] == terrain and t[1] == owner)
    if not cells:
        errors.append(f"rule {rule}: no {what}")
        return None
    origin = _bbox(cells)[:2]
    got = sorted((x - origin[0], y - origin[1], m.at(x, y)) for x, y in cells)
    if got != sorted(src.tiles):
        errors.append(f"rule {rule}: {what} tiles differ from the source shape")
    want = sorted(t.raw for t in _shifted(src, origin))
    on = sorted(t.raw for t in things if t.type in (194, 204) and _in_box(t.raw, origin, src))
    if on != want:
        errors.append(f"rule {rule}: {what} things are not the source's moved with it")
    return origin


def _in_box(raw: bytes, origin, src: Structure) -> bool:
    x, y = struct.unpack_from("<ii", raw, 8)
    x1, y1 = _bbox([(dx, dy) for dx, dy, _ in src.tiles])[2:]
    return origin[0] <= x <= origin[0] + x1 and origin[1] <= y <= origin[1] + y1


def _container_errors(errors, name, data, like) -> None:
    try:
        read_blocks(data)
    except ValueError as exc:
        errors.append(f"rule 11: {name}: read_blocks refuses it ({exc})")
        return
    if len(data) < 36:
        errors.append(f"rule 11: {name}: too short")
        return
    hsz = header_size(data)
    sig = struct.unpack_from("<I", data, 0)[0]
    if hsz != header_size(like) or sig != struct.unpack_from("<I", like, 0)[0]:
        errors.append(f"rule 11: {name}: signature or header size differs from the base")
        return
    if struct.unpack_from("<I", data, 12)[0] != sig + 1:
        errors.append(f"rule 11: {name}: check word 1 is not signature + 1")
    if struct.unpack_from("<I", data, hsz - 8) != struct.unpack_from("<I", like, hsz - 8):
        errors.append(f"rule 11: {name}: check word 2 differs from the base")
    if struct.unpack_from("<I", data, 8)[0] != len(data):
        errors.append(f"rule 11: {name}: file size field is not the file's size")
    if sig != 0xDC and struct.unpack_from("<I", data, hsz - 4)[0] != len(data) - hsz:
        errors.append(f"rule 11: {name}: content size field is wrong")
    stamps = slice(hsz - 28, hsz - 8)
    if data[stamps] != like[stamps] and sig not in (0x64,):
        errors.append(f"rule 11: {name}: timestamps differ from the base")


def validate_testmap(files: dict[str, bytes], stem: str, *, molynx_in_pool: bool) -> list[str]:
    """One line per broken layout rule (see the task brief / module docstring); [] if all hold."""
    errors: list[str] = []
    missing = [stem + s for s in SUFFIXES if stem + s not in files]
    if missing:
        return [f"rule 11: missing files {missing}"]
    f = {s: files[stem + s] for s in SUFFIXES}
    base, siege = _load(BASE), _load(PORTAL_SOURCE)
    for s in SUFFIXES:
        _container_errors(errors, stem + s, f[s], base[s])
    try:
        m = read_map(f["Map.kld"])
        _tblock, things = read_things(f["Things.kld"])
        _pb, players = read_players(f["Players.kld"])
        _vb, var_recs = read_variables(f["Variables.kld"])
        read_level(f[".kwd"])
    except (ValueError, struct.error) as exc:
        return errors + [f"rule 11: a file does not parse ({exc})"]
    bmap = read_map(base["Map.kld"])
    _b, bthings = read_things(base["Things.kld"])

    # rule 1
    if (m.width, m.height) != (bmap.width, bmap.height):
        errors.append("rule 1: width/height differ from the base")
        return errors
    w, h = m.width, m.height
    ring = [i for i in range(w * h) if i % w in (0, w - 1) or i // w in (0, h - 1)]
    if any(m.tiles[i] != bmap.tiles[i] for i in ring):
        errors.append("rule 1: the outer ring differs from the base")
    if m.header != bmap.header:
        errors.append("rule 1: the Map header differs from the base")

    # rule 2
    hearts = Counter(t[1] for t in m.tiles if t[0] == HEART)
    if set(hearts) != {KEEPER1, KEEPER2}:
        errors.append(f"rule 2: heart tiles belong to players {sorted(hearts)}, not 3 and 4")
    for owner in (KEEPER1, KEEPER2):
        src = _structure(bmap, bthings, HEART, owner)
        _check_structure(errors, 2, f"heart of player {owner}", m, things, src, HEART, owner)
    heart_rooms = [t for t in things if t.type == 204]
    if sorted(t.data[19] for t in heart_rooms if t.data[18] == HEART) != [KEEPER1, KEEPER2]:
        errors.append("rule 2: not exactly one heart room thing per keeper")

    # rule 3
    _sb, sthings = read_things(siege["Things.kld"])
    portal_src = _structure(read_map(siege["Map.kld"]), sthings, PORTAL, KEEPER1)
    portal_cells = _cells(m, lambda t: t[0] == PORTAL)
    if len(portal_cells) != len(portal_src.tiles) or any(m.at(x, y)[1] != KEEPER1 for x, y in portal_cells):
        errors.append(f"rule 3: {len(portal_cells)} portal tiles, want one portal of "
                      f"{len(portal_src.tiles)} tiles of player 3")
    _check_structure(errors, 3, "portal", m, things, portal_src, PORTAL, KEEPER1)

    # rule 4
    area = _k1_area(m)
    floor = [c for c in area if m.at(*c)[0] in FLOOR]
    claimed = [c for c in area if m.at(*c)[0] == CLAIMED]
    if len(floor) < 900:
        errors.append(f"rule 4: Keeper 1's open area has {len(floor)} floor tiles (< 900)")
    if len(claimed) < 300:
        errors.append(f"rule 4: Keeper 1's open area has {len(claimed)} Claimed Path tiles (< 300)")
    if portal_cells and not set(portal_cells) <= area:
        errors.append("rule 3: the portal is not connected by floor to Keeper 1's heart")
    outside = {(x, y) for y in range(h) for x in range(w)} - area
    reach = _flood_set(outside, (0, 0))
    if reach != outside:
        errors.append(f"rule 4: {len(outside - reach)} tiles are enclosed inside Keeper 1's area")
    gems = _cells(m, lambda t: t[0] == GEMS)
    gold = _cells(m, lambda t: t[0] == GOLD)
    if not 4 <= len(gems) <= 6:
        errors.append(f"rule 4: {len(gems)} Gems tiles (want 4..6)")
    if len(gold) != 2:
        errors.append(f"rule 4: {len(gold)} Gold tiles (want 2)")
    for x, y in gems + gold:
        near = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if not any(c in area for c in near):
            errors.append(f"rule 4: Gems/Gold at {x},{y} does not touch Keeper 1's open area")
        if not all(c in area or m.at(*c)[0] in (ROCK, GEMS, GOLD) for c in near):
            errors.append(f"rule 4: Gems/Gold at {x},{y} is not embedded in Rock")
    other = sorted({t[0] for t in m.tiles} - {IMPENETRABLE, ROCK, DIRT, GOLD, GEMS, CLAIMED, PORTAL, HEART, EDGE})
    if other:
        errors.append(f"rule 4: other terrain on the map: {other}")

    # rule 5
    k2 = _cells(m, lambda t: t[0] == HEART and t[1] == KEEPER2)
    chamber = _flood(m, k2, lambda t: t[0] not in SOLID) if k2 else set()
    if chamber & area or any(m.at(*c)[1] == KEEPER1 for c in chamber):
        errors.append("rule 5: Keeper 2's chamber reaches Keeper 1's tiles")
    wall_bad = set()
    for x, y in chamber:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                c = (x + dx, y + dy)
                if c in chamber:
                    continue
                if not (0 <= c[0] < w and 0 <= c[1] < h) or m.at(*c)[0] != IMPENETRABLE:
                    wall_bad.add(c)
    if wall_bad:
        errors.append(f"rule 5: {len(wall_bad)} tiles within 2 of Keeper 2's chamber are not "
                      f"Impenetrable Rock, e.g. {sorted(wall_bad)[0]}")
    if any(m.at(*c)[0] in (GEMS, GOLD) for c in chamber):
        errors.append("rule 5: Gems or Gold in Keeper 2's chamber")

    # rule 6
    allowed = {t.raw for t in _shifted(portal_src, _bbox(portal_cells)[:2])} if portal_cells else set()
    for owner in (KEEPER1, KEEPER2):
        cells = _cells(m, lambda t, o=owner: t[0] == HEART and t[1] == o)
        if cells:
            allowed |= {t.raw for t in _shifted(_structure(bmap, bthings, HEART, owner), _bbox(cells)[:2])}
    creatures = Counter()
    for t in things:
        if t.type == 200:
            cid, owner = t.data[26], t.data[27]
            creatures[(cid, owner)] += 1
            x, y = _xy(t)
            region = area if owner == KEEPER1 else chamber if owner == KEEPER2 else set()
            if (x, y) not in region or m.at(x, y)[0] in SOLID:
                errors.append(f"rule 6: creature {cid} of player {owner} at {x},{y} is not on "
                              f"its keeper's floor")
            if cid == MOLYNX_ID:
                tile = m.at(x, y)
                beside = any(m.at(x + dx, y + dy)[:2] == bytes([HEART, KEEPER1])
                             for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                             if 0 <= x + dx < w and 0 <= y + dy < h)
                if tile[0] not in FLOOR or not beside:
                    errors.append(f"rule 6: Molynx at {x},{y} is not on floor next to the heart")
        elif t.raw not in allowed:
            errors.append(f"rule 6: unexpected thing of type {t.type}")
    want = Counter({(IMP_ID, KEEPER1): 6, (MOLYNX_ID, KEEPER1): 1, (IMP_ID, KEEPER2): 4})
    if creatures != want:
        errors.append(f"rule 6: creatures {dict(creatures)}, want {dict(want)}")

    # rule 7
    _bp, bplayers = read_players(base["Players.kld"])
    if len(players) != len(bplayers):
        errors.append("rule 7: player count differs from the base")
    else:
        for new, old in zip(players, bplayers):
            if new[-37] == KEEPER1:
                gold_, is_ai = struct.unpack_from("<iI", new)
                if gold_ != STARTING_GOLD or is_ai != 0 or new[8:] != old[8:]:
                    errors.append("rule 7: Keeper 1 is not the base's with 100000 gold and isAi NO")
            elif new != old:
                errors.append(f"rule 7: player {new[-37]} differs from the base")

    # rule 8
    errors += _variable_errors(f["Variables.kld"], base["Variables.kld"], molynx_in_pool)

    # rule 9
    # The editor's empty file byte for byte, except its timestamps, which are the base file's.
    tr, auto, btr = f["Triggers.kld"], (EDITOR_DIR / "AutoSaveTriggers.kld").read_bytes(), base["Triggers.kld"]
    stamps = slice(header_size(auto) - 28, header_size(auto) - 8)
    if len(tr) != len(auto) or tr[:stamps.start] != auto[:stamps.start] or tr[stamps.stop:] != auto[stamps.stop:]:
        errors.append(f"rule 9: Triggers is not the editor's empty file ({len(tr)} bytes, "
                      f"editor's {len(auto)})")
    elif tr[stamps] != btr[stamps]:
        errors.append("rule 9: Triggers timestamps are not the base file's")

    # rule 10
    errors += _level_errors(f[".kwd"], base[".kwd"], stem)
    return errors


def _flood_set(cells: set, start) -> set:
    if start not in cells:
        return set()
    seen, todo = {start}, [start]
    while todo:
        x, y = todo.pop()
        for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if c in cells and c not in seen:
                seen.add(c)
                todo.append(c)
    return seen


def _variable_errors(new: bytes, base: bytes, molynx_in_pool: bool) -> list[str]:
    errors = []
    try:
        _nb, recs = read_variables(new)
    except ValueError as exc:
        return [f"rule 8: {exc}"]
    _bb, brecs = read_variables(base)
    ents, bents = [_var(r) for r in recs], [_var(r) for r in brecs]
    if len(ents) < len(bents):
        return ["rule 8: fewer entries than the base"]
    for old, cur in zip(bents, ents):
        if old[0] == AVAILABILITY and _av_key(old)[1] == KEEPER1:
            if cur[:3] != old[:3]:
                errors.append("rule 8: a base entry moved or changed")
                break
        elif cur != old:
            errors.append("rule 8: a base entry moved or changed")
            break
    keys = [_av_key(e) for e in ents if e[0] == AVAILABILITY]
    dup = sorted(k for k, n in Counter(keys).items() if n > 1)
    if dup:
        errors.append(f"rule 8: duplicate availability entries {dup}")
    pool_keys = [(e[1], e[3]) for e in ents if e[0] == CREATURE_POOL]
    pdup = sorted(k for k, n in Counter(pool_keys).items() if n > 1)
    if pdup:
        errors.append(f"rule 8: duplicate creature pool entries {pdup}")
    k1 = {(_av_key(e)[0], e[2]): e[3] for e in ents if e[0] == AVAILABILITY and _av_key(e)[1] == KEEPER1}
    for key in _keeper_types(bents):
        if k1.get(key) != ENABLE:
            errors.append(f"rule 8: player 3 availability {key} is {k1.get(key)}, not ENABLE")
    not_enabled = sorted(k for k, v in k1.items() if v != ENABLE)
    if not_enabled:
        errors.append(f"rule 8: player 3 availability not ENABLE: {not_enabled}")
    pool, av = _siege_creatures()
    want_pool = {(e[1], e[2]) for e in pool} | ({(MOLYNX_ID, POOL_VALUE)} if molynx_in_pool else set())
    got_pool = {(e[1], e[2]) for e in ents if e[0] == CREATURE_POOL and e[3] == KEEPER1}
    if got_pool != want_pool:
        errors.append(f"rule 8: player 3 pool {sorted(got_pool)}, want {sorted(want_pool)}")
    want_av = {e[2] for e in av} | ({MOLYNX_ID} if molynx_in_pool else set())
    got_av = {i for (t, i) in k1 if t == AV_CREATURE}
    if got_av != want_av:
        errors.append(f"rule 8: player 3 creature availability {sorted(got_av)}, want {sorted(want_av)}")
    return errors


def _level_errors(new: bytes, base: bytes, stem: str) -> list[str]:
    errors = []
    if len(new) != len(base):
        return [f"rule 10: level file is {len(new)} bytes, base {len(base)}"]
    if new[:56] != base[:56]:
        errors.append("rule 10: level header differs from the base")
    if struct.unpack_from("<H", new, LEVEL_FLAGS_AT) != struct.unpack_from("<H", base, LEVEL_FLAGS_AT):
        errors.append("rule 10: level flags differ from the base")
    if not level_name(new):
        errors.append("rule 10: empty display name")
    if _wstr(new, LEVEL_DESC_AT, LEVEL_DESC_CHARS) != DESCRIPTION:
        errors.append(f"rule 10: the description is not {DESCRIPTION!r}")
    lv, blv = read_level(new), read_level(base)
    want_paths = [_repath(p, BASE, stem) for p in blv.paths]
    if list(lv.paths) != want_paths:
        errors.append("rule 10: the path list does not name the stem's files")
    keep = [(0, LEVEL_NAME_AT), (LEVEL_DESC_AT + 2 * LEVEL_DESC_CHARS, 56 + len(blv.content))]
    if any(new[a:b] != base[a:b] for a, b in keep) or lv.rest != blv.rest or lv.paths_header != blv.paths_header:
        errors.append("rule 10: level bytes other than name, description and paths changed")
    return errors


# ---- previews -----------------------------------------------------------------------------------

_CHARS = {IMPENETRABLE: "#", ROCK: "r", DIRT: ".", GOLD: "$", GEMS: "*", PORTAL: "P", EDGE: "E"}
_LEGEND = ("Legend: E Edge of Map, # Impenetrable Rock, r Rock, . Dirt Path, , Claimed Path "
           "(Keeper 1), 2 Claimed Path (Keeper 2), $ Gold, * Gems, P Portal, H Dungeon Heart "
           "(Keeper 1), K Dungeon Heart (Keeper 2), i Imp (Keeper 1), j Imp (Keeper 2), M Molynx, "
           "? other terrain.")


def _creature_marks(things: list[Thing]) -> dict[tuple[int, int], str]:
    marks = {}
    for t in things:
        if t.type == 200:
            cid, owner = t.data[26], t.data[27]
            marks[_xy(t)] = "M" if cid == MOLYNX_ID else ("i" if owner == KEEPER1 else "j")
    return marks


def _tile_char(tile: bytes) -> str:
    if tile[0] == CLAIMED:
        return "," if tile[1] == KEEPER1 else "2"
    if tile[0] == HEART:
        return "H" if tile[1] == KEEPER1 else "K"
    return _CHARS.get(tile[0], "?")


def preview_text(files: dict[str, bytes], stem: str) -> str:
    """One character per tile (rows top to bottom), creatures drawn over their tiles, then a legend."""
    m = read_map(files[stem + "Map.kld"])
    _b, things = read_things(files[stem + "Things.kld"])
    marks = _creature_marks(things)
    rows = ["".join(marks.get((x, y), _tile_char(m.at(x, y))) for x in range(m.width))
            for y in range(m.height)]
    return "\n".join(rows) + "\n" + _LEGEND + "\n"


_COLOURS = {IMPENETRABLE: (40, 40, 44), ROCK: (110, 90, 70), DIRT: (170, 140, 100),
            GOLD: (240, 200, 40), GEMS: (60, 220, 230), PORTAL: (150, 60, 200), EDGE: (0, 0, 0)}


def preview_png(files: dict[str, bytes], stem: str, path: Path, scale: int = 8) -> None:
    """A picture of the map, `scale` pixels per tile: terrain colours, creatures as dots."""
    from PIL import Image, ImageDraw
    m = read_map(files[stem + "Map.kld"])
    _b, things = read_things(files[stem + "Things.kld"])
    img = Image.new("RGB", (m.width * scale, m.height * scale))
    draw = ImageDraw.Draw(img)
    for y in range(m.height):
        for x in range(m.width):
            t = m.at(x, y)
            if t[0] == CLAIMED:
                colour = (200, 70, 60) if t[1] == KEEPER1 else (60, 90, 200)
            elif t[0] == HEART:
                colour = (255, 30, 30) if t[1] == KEEPER1 else (30, 60, 255)
            else:
                colour = _COLOURS.get(t[0], (255, 0, 255))
            draw.rectangle([x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1], fill=colour)
    for (x, y), mark in _creature_marks(things).items():
        colour = {"M": (255, 255, 255), "i": (255, 230, 120), "j": (160, 255, 160)}[mark]
        r = scale // 2 - 1 if mark != "M" else scale // 2
        cx, cy = x * scale + scale // 2, y * scale + scale // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour, outline=(0, 0, 0))
    img.save(path)
