"""Splice creature records into the Creatures block of a Dungeon Keeper 2 `<map>Globals.kld`.

A Globals file is a plain sequence of blocks (see dk2kwd_blocks) and holds exactly one Creatures
block (signature 0xAA). The game builds its creature table from that block's item count, so a map
gets a 47th creature by appending a record there and updating the block's item count, file size and
content size. `splice_creatures` does exactly that through `read_blocks`, `rebuilt_block` and
`write_blocks`; since `read_blocks` tiles the whole file, every other block, and every other byte of
the Creatures block header (signature, check words, timestamps), comes out unchanged. The input is
never modified.

`molynx_for_map` builds the Molynx record for one map: that map's own Bile Demon (found by name
"Bile Demon" and creatureId 3, so a map whose block is shuffled or renamed is refused, not guessed)
copied in the map's record size (5449 or 5537 bytes) and run through `build_molynx`.

orderInEditor. Molynx's orderInEditor is the largest orderInEditor of the map's records plus one.
Checked on all 241 maps of the editor folder and on the editor Creatures.kwd: the values are a
permutation of 1..N, where N is the number of records (46 in the editor file and 217 maps, 44 in
the 24 maps without Maiden Of The Nest and Jack), so "max + 1" is N + 1, the slot after the last
creature: 47 in a 46-creature map, 45 in a 44-creature map. (The editor's own order is not the
record order: the Imp is 1, Prince Tristran 29, Bile Demon 11, while records 31..46 are 31..46.)
A map whose largest value is 255 cannot take Molynx (the field is one byte): `build_molynx` raises.

creatureId. Molynx is creatureId 47 in every map. In the 46-creature maps that is the next free
ID after the last record. The 24 maps that hold 44 creatures lack IDs 45 and 46, so there the new
record has ID 47 at index 44. The loader inserts records into a table sorted by creature ID
(spec section "The loader is count-driven", a static reading not yet run), so the ID gap is
expected to be harmless, but it is untested in the game.

Things. A `<map>Things.kld` lists the objects placed on the map. Layout taken from the community
template ufdada/dk2-tools Formats/Level/things_struct.bt (with kwd_struct.bt) and checked on all 345
`*Things.kld` files of MAPS_DIR (283) and QUUZ_MAPS_DIR (62):
- The file is the same block container that `read_blocks` handles, holding exactly one block, the
  Things block (signature 0xBE, check words 0xBF and 0xC0 in every file). Its item count is the
  number of things.
- The content is `item count` variable-size records, back to back: u32 thing type, u32 data size,
  then `data size` bytes. In every file the records tile the content exactly and their number is
  the item count. Every type has one data size in all files: 194 Object 24, 195 Trap 16, 196 Door
  20, 197 Action point 56, 198 Neutral creature 24, 199 Good (hero) creature 32, 200 Creature 28,
  201 Hero party 555, 202 Dead body 16, 203 Effect generator 36, 204 Room 20 (205 Camera unused).
- Creature things are the records of type 200 (a keeper's creature) and only those: hero creatures
  (199), hero-party members (inside 201 records), neutral creatures (198) and dead bodies (202) are
  other types and are never returned. A type-200 record is 36 bytes; from its first byte: +0 u32
  type 200, +4 u32 28, +8 i32 x, +12 i32 y, +16 i32 z, +20 u16 gold held, +22 u8 level, +23 u8
  behaviour flags, +24 i32 initial health, +28 i32 objective target player, +32 u16 trigger id,
  +34 u8 creatureId, +35 u8 owning player. `ThingRef.offset` is the byte offset in the file of
  the record's first byte (its type word), so the creature id is the one byte at offset + 34 and
  the owner the one byte at offset + 35.
- Checked: all 3282 type-200 records have data size 28, creatureId in 1..45 (so within 1..46) and
  owner in 3..7. Player ids come from the maps' `*Players.kld`: in all 345 of them player 1 is named
  "Good Player", 2 "Neutral Player", and 3..7 are the keepers ("Keeper 1".."Keeper 5" or custom
  names). A "real player" owner is therefore a keeper slot 3..7 (`KEEPER_PLAYERS`); 0, 1 (Good), 2
  (Neutral) and anything above 7 are not. The Bile Demon is creatureId 3 (the Globals record named
  "Bile Demon", see `molynx_for_map`), and type-200 Bile Demons exist in the maps the test map is
  chosen from.

The test map (`pick_test_map`): the smallest map, by the total bytes of its seven files (`<name>.kwd`
and `<name>` + Map, Things, Players, Triggers, Variables, Globals `.kld`) in `map_source_dir(name)`,
ties broken by name, whose Globals holds exactly 46 records of 5537 bytes (so Molynx becomes record
index 46 with creatureId 47 == index + 1) and whose Things holds a type-200 Bile Demon owned by a
keeper. Maps whose name (`<name>.kwd`) is in both folders are preferred when any qualifies. The
user's game loads QUUZ_MAPS_DIR first, so a map's files are read from there when all seven are
present, else from MAPS_DIR. File names are matched exactly (case-sensitive): the one mixed-case
map of MAPS_DIR (level4.kwd with Level4Globals.kld) is only seen through its Quuz copy "Level4".
43 `.kwd` files of MAPS_DIR have no Globals file of the same spelling (level4 among them); 36 of
those maps have one in QUUZ_MAPS_DIR. The result is "Siege" (in QUUZ_MAPS_DIR), a multiplayer map: the
level flags of the Quuz Siege.kwd (u16 at byte 56 + 25484) set IS_MULTIPLAYER_LEVEL but not
IS_SKIRMISH_LEVEL, while the editor-folder Siege.kwd sets both.

The second test map (`pick_skirmish_test_map`). A map the Skirmish list may not offer is a wasted test
round, so a second map is shipped whose source `.kwd` has IS_SKIRMISH_LEVEL set: the best map by the
same ordering and criteria as `pick_test_map`, other than that one, and in which the Bile Demon that
`first_player_bile_thing` picks (the one Molynx replaces) is owned by player 3, "Keeper 1" (`HUMAN_PLAYER`),
so Molynx stands in the human's own dungeon. Of the 41 candidates, 30 have the skirmish flag; the
smallest, 13coop, would give Molynx to Keeper 4 (player 6, usually an AI side), and the only one whose
first keeper Bile Demon is Keeper 1's is "Dead Walking" (editor folder, 745,684 B, flags 0xC04). The
`pick_test_map` choice (Siege, Keeper 1's Bile Demon too) does not use this rule. The level flags are the bit
struct LevelFlag of the template ufdada/dk2-tools Formats/Level/level_struct.bt, read little-endian
from the least significant bit: bit 2 UNK_1 (always on), 3 ALWAYS_IMPRISON_ENEMIES, 4 ONE_SHOT_HORNY,
5 IS_SECRET_LEVEL, 6 IS_SPECIAL_LEVEL, 7 SHOW_HERO_KILLS, 8 AUTO_OBJECTIVE_BOX, 9 HEART_MAKES_GEM,
10 IS_MULTIPLAYER_LEVEL, 11 IS_SKIRMISH_LEVEL, 12 FREEZE_OPTIONS, 13 IS_MY_PET_DUNGEON_LEVEL. This
reading matches the data: the Quuz Siege.kwd holds 0x404 (bits 10 and 2), the editor Siege.kwd and
13coop.kwd 0xC04 (bits 11, 10 and 2); over the 283 `.kwd` files of MAPS_DIR bit 11 is set in 109.

LICENCE: GPL-3.0.
"""
import struct
from dataclasses import dataclass
from pathlib import Path

from scripts.dk2kwd_blocks import HEADER_SIZE, Block, read_blocks, rebuilt_block, write_blocks
from scripts.dk2kwd_creatures import Creature, creatures_block, read_creatures
from scripts.dk2paths import MAPS_DIR, QUUZ_MAPS_DIR
from scripts.molynx_record import build_molynx

CREATURES = 0xAA
BILE_DEMON_NAME = "Bile Demon"
BILE_DEMON_ID = 3


def _creatures_index(blocks: list[Block]) -> int:
    """Index of the one Creatures block; ValueError if there is none or several."""
    found = [i for i, block in enumerate(blocks) if block.signature == CREATURES]
    if len(found) != 1:
        raise ValueError(f"expected one Creatures block (0x{CREATURES:X}), found {len(found)}")
    return found[0]


def _records(block: Block) -> list[Creature]:
    if block.item_count == 0:
        raise ValueError("the Creatures block has no records, so its record size is unknown")
    return read_creatures(block)


def splice_creatures(globals_kld: bytes, extra: list[Creature]) -> bytes:
    """`globals_kld` with the records of `extra` added to its Creatures block, ordered by creatureId.

    The map's own records keep their order and bytes, and every other block is copied unchanged.
    ValueError for an empty `extra`; a record size that differs from the map's; a creatureId that is
    already in the map or repeated in `extra`; records of the map that are not in ascending
    creatureId order (they would have to be reordered); or a file without exactly one Creatures block.
    """
    if not extra:
        raise ValueError("no creature records to splice")
    blocks = read_blocks(globals_kld)
    at = _creatures_index(blocks)
    existing = _records(blocks[at])

    size = len(existing[0].raw)
    for creature in extra:
        if len(creature.raw) != size:
            raise ValueError(f"a creature record of {len(creature.raw)} bytes cannot go into a "
                             f"map whose records are {size} bytes")
    old_ids = [c.get("creatureId") for c in existing]
    if old_ids != sorted(set(old_ids)):
        raise ValueError("the map's creature records are not in ascending creatureId order")
    new_ids = [c.get("creatureId") for c in extra]
    taken = sorted(set(new_ids) & set(old_ids))
    if taken:
        raise ValueError(f"creatureId {taken} is already in the map")
    if len(set(new_ids)) != len(new_ids):
        repeated = sorted({i for i in new_ids if new_ids.count(i) > 1})
        raise ValueError(f"creatureId {repeated} appears more than once in the records to splice")

    merged = sorted([*existing, *extra], key=lambda c: c.get("creatureId"))
    blocks[at] = creatures_block(merged, blocks[at])
    return write_blocks(blocks)


def molynx_for_map(globals_kld: bytes, stage: int, text_ids: dict[str, int] | None = None, *,
                   anim_info: dict | None = None, file_wraps: dict | None = None) -> Creature:
    """The Molynx record for this map: its Bile Demon, in the map's record size, through `build_molynx`.

    `stage` (1, 2 or 3), `text_ids` (used at stage 3 only), `anim_info` and `file_wraps` (both
    needed from stage 2: the exported files' info and the Bile Demon files' loop flags) are
    `build_molynx`'s; orderInEditor is the map's largest plus one (see the module docstring). ValueError if the map has no Creatures
    block (or several), no creature named "Bile Demon" with creatureId 3 (or several), or bad input.
    """
    blocks = read_blocks(globals_kld)
    creatures = _records(blocks[_creatures_index(blocks)])
    biles = [c for c in creatures
             if c.get("creatureId") == BILE_DEMON_ID and c.get("name") == BILE_DEMON_NAME]
    if len(biles) != 1:
        raise ValueError(f"expected one {BILE_DEMON_NAME!r} with creatureId {BILE_DEMON_ID} in the "
                         f"map, found {len(biles)}")
    order = max(c.get("orderInEditor") for c in creatures) + 1
    return build_molynx(biles[0], stage, order, text_ids, anim_info, file_wraps=file_wraps)


# ---- Things -------------------------------------------------------------------------------------

THINGS = 0xBE
CREATURE_THING = 200
_CREATURE_DATA_SIZE = 28
_CREATURE_ID_AT, _PLAYER_AT = 34, 35  # from the first byte of a type-200 record
KEEPER_PLAYERS = range(3, 8)
HUMAN_PLAYER = 3  # "Keeper 1": the first keeper slot, the one the human plays
MAP_SUFFIXES = (".kwd", "Map.kld", "Things.kld", "Players.kld", "Triggers.kld", "Variables.kld",
                "Globals.kld")


@dataclass(frozen=True)
class Thing:
    type: int
    data: bytes

    @property
    def raw(self) -> bytes:
        """The record as stored: u32 type, u32 data size, data."""
        return struct.pack("<II", self.type, len(self.data)) + self.data


@dataclass(frozen=True)
class ThingRef:
    """A creature thing: `offset` is the file offset of its record's first byte (see module doc)."""
    offset: int
    creature_id: int
    player: int


def read_things(things_kld: bytes) -> tuple[Block, list[Thing]]:
    """The Things block (header kept as is) and its records; ValueError if the file is not exactly
    one Things block whose records tile its content and number its item count."""
    blocks = read_blocks(things_kld)
    if len(blocks) != 1 or blocks[0].signature != THINGS:
        raise ValueError(f"expected one Things block (0x{THINGS:X}), found signatures "
                         f"{[hex(b.signature) for b in blocks]}")
    (block,) = blocks
    content, pos, things = block.content, 0, []
    while pos < len(content):
        if len(content) - pos < 8:
            raise ValueError(f"truncated thing record at content byte {pos}")
        kind, size = struct.unpack_from("<II", content, pos)
        if pos + 8 + size > len(content):
            raise ValueError(f"thing record of type {kind} at content byte {pos} overruns the block")
        things.append(Thing(kind, content[pos + 8:pos + 8 + size]))
        pos += 8 + size
    if len(things) != block.item_count:
        raise ValueError(f"the Things block says {block.item_count} things but holds {len(things)}")
    return block, things


def write_things(like: Block, things: list[Thing]) -> bytes:
    """A Things file holding `things`; header sizes and count are recomputed, the rest copied from `like`."""
    return write_blocks([rebuilt_block(like, [t.raw for t in things])])


def find_creature_things(things_kld: bytes) -> list[ThingRef]:
    """The creature things (type 200) of a Things file, in file order; ValueError if one is not 28 bytes."""
    _block, things = read_things(things_kld)
    refs, offset = [], HEADER_SIZE  # one block, so the records start right after its header
    for thing in things:
        if thing.type == CREATURE_THING:
            if len(thing.data) != _CREATURE_DATA_SIZE:
                raise ValueError(f"creature thing at byte {offset} has {len(thing.data)} data bytes, "
                                 f"not {_CREATURE_DATA_SIZE}")
            refs.append(ThingRef(offset, things_kld[offset + _CREATURE_ID_AT],
                                 things_kld[offset + _PLAYER_AT]))
        offset += len(thing.raw)
    return refs


def retarget_creature_thing(things_kld: bytes, index: int, new_creature_id: int) -> bytes:
    """`things_kld` with the creature id byte of its `index`-th creature thing set to `new_creature_id`.

    No other byte changes (the record keeps its size). ValueError if `index` is not in
    0..len(find_creature_things) - 1 or `new_creature_id` is not in 1..255 (0 is no creature).
    """
    refs = find_creature_things(things_kld)
    if not 0 <= index < len(refs):
        raise ValueError(f"creature thing index {index} is not in 0..{len(refs) - 1}")
    if not 1 <= new_creature_id <= 255:
        raise ValueError(f"creatureId {new_creature_id} does not fit the one-byte field (1..255)")
    out = bytearray(things_kld)
    out[refs[index].offset + _CREATURE_ID_AT] = new_creature_id
    return bytes(out)


def first_player_bile_thing(things_kld: bytes) -> int:
    """Index into `find_creature_things` of the first Bile Demon owned by a keeper; ValueError if none."""
    for i, ref in enumerate(find_creature_things(things_kld)):
        if ref.creature_id == BILE_DEMON_ID and ref.player in KEEPER_PLAYERS:
            return i
    raise ValueError("no Bile Demon creature thing owned by a keeper (player 3..7)")


# ---- the test map -------------------------------------------------------------------------------

def _complete(folder: Path, name: str) -> bool:
    return all((folder / (name + suffix)).is_file() for suffix in MAP_SUFFIXES)


def map_source_dir(name: str) -> Path:
    """QUUZ_MAPS_DIR if all seven files of map `name` are there (the game loads it first), else
    MAPS_DIR if they are there; FileNotFoundError otherwise."""
    for folder in (QUUZ_MAPS_DIR, MAPS_DIR):
        if _complete(folder, name):
            return folder
    raise FileNotFoundError(f"map {name!r} does not have all seven files in {QUUZ_MAPS_DIR} or {MAPS_DIR}")


def _globals_shape(globals_kld: bytes) -> tuple[int, int]:
    """(record count, record size) of a Globals file's Creatures block."""
    blocks = read_blocks(globals_kld)
    block = blocks[_creatures_index(blocks)]
    return block.item_count, len(_records(block)[0].raw)


# The level flags: a u16 inside the .kwd's level header (see the module docstring for the bits).
LEVEL_FLAGS_AT = 56 + 25484
IS_MULTIPLAYER_LEVEL = 1 << 10
IS_SKIRMISH_LEVEL = 1 << 11


def level_flags(kwd: bytes) -> int:
    """The level flags (u16) of a map's `.kwd` file; ValueError if the file is too short."""
    if len(kwd) < LEVEL_FLAGS_AT + 2:
        raise ValueError(f"a .kwd of {len(kwd)} bytes is too short for the level flags "
                         f"at byte {LEVEL_FLAGS_AT}")
    (flags,) = struct.unpack_from("<H", kwd, LEVEL_FLAGS_AT)
    return flags


def _test_map_candidates() -> list[tuple[bool, int, str]]:
    """(not in both folders, total bytes of the seven files, name) of every map that meets the
    test-map criteria in the module docstring; the smallest tuple is the preferred map."""
    names = {p.stem for folder in (QUUZ_MAPS_DIR, MAPS_DIR) for p in folder.glob("*.kwd")
             if _complete(folder, p.stem)}
    candidates = []
    for name in sorted(names):
        folder = map_source_dir(name)
        if _globals_shape((folder / (name + "Globals.kld")).read_bytes()) != (46, 5537):
            continue
        try:
            first_player_bile_thing((folder / (name + "Things.kld")).read_bytes())
        except ValueError:
            continue
        size = sum((folder / (name + suffix)).stat().st_size for suffix in MAP_SUFFIXES)
        in_both = all((d / (name + ".kwd")).is_file() for d in (QUUZ_MAPS_DIR, MAPS_DIR))
        candidates.append((not in_both, size, name))
    return candidates


def pick_test_map() -> str:
    """The map Molynx is tested on (see the module docstring for the criteria); ValueError if none."""
    candidates = _test_map_candidates()
    if not candidates:
        raise ValueError("no map meets the test-map criteria")
    return min(candidates)[2]


def _molynx_owner(name: str) -> int:
    """Owner (player id) of the creature thing that Molynx replaces in map `name`."""
    things = (map_source_dir(name) / (name + "Things.kld")).read_bytes()
    return find_creature_things(things)[first_player_bile_thing(things)].player


def pick_skirmish_test_map() -> str:
    """The second test map: the best candidate other than `pick_test_map()` whose source `.kwd`
    has IS_SKIRMISH_LEVEL set and whose Molynx thing is owned by `HUMAN_PLAYER` (see the module
    docstring); ValueError if none."""
    candidates = _test_map_candidates()
    if not candidates:
        raise ValueError("no map meets the test-map criteria")
    first = min(candidates)[2]
    second = [c for c in candidates
              if c[2] != first
              and level_flags((map_source_dir(c[2]) / (c[2] + ".kwd")).read_bytes()) & IS_SKIRMISH_LEVEL
              and _molynx_owner(c[2]) == HUMAN_PLAYER]
    if not second:
        raise ValueError("no second map meets the test-map criteria, has the skirmish flag and "
                         f"gives Molynx to player {HUMAN_PLAYER}")
    return min(second)[2]
