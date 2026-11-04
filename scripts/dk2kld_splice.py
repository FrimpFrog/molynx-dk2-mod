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

LICENCE: GPL-3.0.
"""
from scripts.dk2kwd_blocks import Block, read_blocks, write_blocks
from scripts.dk2kwd_creatures import Creature, creatures_block, read_creatures
from scripts.molynx_record import build_molynx

CREATURES = 0xAA
BILE_DEMON_NAME = "Bile Demon"
BILE_DEMON_ID = 3


def _creatures_block(blocks: list[Block]) -> int:
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
    at = _creatures_block(blocks)
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


def molynx_for_map(globals_kld: bytes, stage: int, text_ids: dict[str, int] | None = None) -> Creature:
    """The Molynx record for this map: its Bile Demon, in the map's record size, through `build_molynx`.

    `stage` (1, 2 or 3) and `text_ids` (used at stage 3 only) are `build_molynx`'s; orderInEditor is
    the map's largest plus one (see the module docstring). ValueError if the map has no Creatures
    block (or several), no creature named "Bile Demon" with creatureId 3 (or several), or bad input.
    """
    blocks = read_blocks(globals_kld)
    creatures = _records(blocks[_creatures_block(blocks)])
    biles = [c for c in creatures
             if c.get("creatureId") == BILE_DEMON_ID and c.get("name") == BILE_DEMON_NAME]
    if len(biles) != 1:
        raise ValueError(f"expected one {BILE_DEMON_NAME!r} with creatureId {BILE_DEMON_ID} in the "
                         f"map, found {len(biles)}")
    order = max(c.get("orderInEditor") for c in creatures) + 1
    return build_molynx(biles[0], stage, order, text_ids)
