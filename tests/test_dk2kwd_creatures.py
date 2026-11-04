import struct

import pytest

from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_creatures import (
    ANIM_SLOTS, FIELDS, RESOURCE_FIELDS, Creature, creatures_block, read_creatures,
)
from scripts.dk2paths import EDITOR_DIR, MAPS_DIR, have

needs_maps = pytest.mark.skipif(not have(MAPS_DIR), reason="DK2 editor maps not present")
needs_editor = pytest.mark.skipif(not have(EDITOR_DIR), reason="DK2 editor data not present")

CREATURES = 0xAA
MAP_5537 = "01coopGlobals.kld"
NAMES = [
    "Imp", "Prince Tristran", "Bile Demon", "Mistress", "Warlock", "Dark Elf", "Goblin", "Vampire",
    "Skeleton", "Troll", "Salamander", "Firefly", "Knight", "Dwarf", "Giant", "Wizard", "Elven Archer",
    "Thief", "Monk", "Fairy", "King Reginald", "Black Knight", "Dark Angel ", "Rogue", "Guard",
    "Prince Felix", "Horny", "Stone Knight", "Lord Of The Land", "Royal Guard", "Prince Balder",
    "Unique Firefly", "Unique Goblin", "Unique Warlock", "Unique Troll", "Unique Dark Elf",
    "Unique Skeleton", "Unique Mistress", "Unique Salamander", "Unique Rogue", "Unique Bile Demon",
    "Unique Vampire", "Unique Black Knight", "Unique Dark Angel", "Maiden Of The Nest", "Jack",
]
FIXED = 4096  # template float32 = int32 / 4096


def _creatures_block(path):
    return next(b for b in read_blocks(path.read_bytes()) if b.signature == CREATURES)


def _sources():
    return [(EDITOR_DIR / "Creatures.kwd", 5449), (MAPS_DIR / MAP_5537, 5537)]


@pytest.fixture(scope="module", params=_sources(), ids=["editor-5449", "map-5537"])
def source(request):
    path, size = request.param
    if not have(path):
        pytest.skip(f"{path.name} not present")
    block = _creatures_block(path)
    return block, read_creatures(block), size


def test_record_size(source):
    block, creatures, size = source
    assert len(block.content) // block.item_count == size
    assert [len(c.raw) for c in creatures] == [size] * 46


def test_names_in_order(source):
    _, creatures, _ = source
    assert [c.get("name") for c in creatures] == NAMES


def test_creature_ids_follow_record_index(source):
    _, creatures, _ = source
    assert [c.get("creatureId") for c in creatures] == list(range(1, 47))


def test_imp_resources_land_on_the_data(source):
    imp = source[1][0]
    assert imp.anim_name("walk") == "Imp_Run"
    assert imp.resource_name("portraitResource") == r"gui\creatures\evil\imp-bg"
    assert imp.resource_name("icon1Resource") == r"gui\creatures\evil\imp"
    assert imp.resource_name("icon2Resource") == "CreatureIcons#0"
    assert imp.resource_name("firstPersonMeleeResource") == r"gui\icons\1st-person\m-imp"
    assert imp.get("soundGategory") == "CREATURE_IMP"
    assert imp.anim_name("melee2") == "IMP-Idle4"
    assert imp.anim_name("extra47") == "IMP-Dig2"


def test_numbers_are_plausible(source):
    _, creatures, _ = source
    # The brief expected pay >= 1 for all; the data has these unpaid creatures (an Imp is never paid).
    unpaid = {"Imp", "Skeleton", "Dwarf", "Horny", "Stone Knight", "Unique Skeleton"}
    assert {c.get("name") for c in creatures if c.get("pay") == 0} == unpaid
    for c in creatures:
        assert 1 <= c.get("hp") <= 65535, c.get("name")
        # float32 fields sit where whole numbers in fixed point would: the template's "1..5" mass
        assert 1 * FIXED <= c.get("mass") <= 5 * FIXED, c.get("name")
        assert 0 < c.get("speed") <= c.get("runSpeed") <= 10 * FIXED, c.get("name")
    # angerSlap / angerInHand are signed (the template says ushort): a Mistress enjoys a slap.
    assert creatures[NAMES.index("Mistress")].get("angerSlap") == -5000
    assert creatures[NAMES.index("Bile Demon")].get("angerInHand") == -300


def test_set_of_get_leaves_raw_unchanged(source):
    _, creatures, size = source
    for c in creatures:
        before = bytes(c.raw)
        for name, field in FIELDS.items():
            if field.end <= size:
                c.set(name, c.get(name))
        # unk_1 (slot 0, never loaded by the game) holds junk after its NUL, so its name alone does
        # not round-trip; every other resource name is NUL-padded and does.
        for name in ANIM_SLOTS[1:] + RESOURCE_FIELDS:
            c.set_resource_name(name, c.resource_name(name))
        assert bytes(c.raw) == before, c.get("name")


def test_block_roundtrip(source):
    block, creatures, _ = source
    assert creatures_block(creatures, block) == block


@needs_maps
def test_every_map_roundtrips():
    for path in sorted(MAPS_DIR.glob("*Globals.kld")):
        block = _creatures_block(path)
        assert creatures_block(read_creatures(block), block) == block, path.name


def test_fields_tile_the_record_without_gaps():
    pos = 0
    for name, field in sorted(FIELDS.items(), key=lambda kv: kv[1].offset):
        assert field.offset == pos, name
        pos = field.end
    assert pos == 5537
    assert FIELDS["flags2"].end == 5449 and FIELDS["unk_40"].offset == 5449  # base record, then tail


def test_anim_slots():
    assert len(ANIM_SLOTS) == 48 and len(set(ANIM_SLOTS)) == 48
    assert all(struct.calcsize(FIELDS[s].fmt) == 84 for s in ANIM_SLOTS + RESOURCE_FIELDS)
    assert ANIM_SLOTS[:2] == ["unk_1", "animWalkResource"]
    assert {"drunkIdle", "animWalkbackResource", "animExtra47Resource"} <= set(ANIM_SLOTS)


def _blank(size=5449):
    return Creature(bytearray(size))


@pytest.mark.parametrize("short, field", [
    ("walk", "animWalkResource"), ("Walk", "animWalkResource"), ("poseFrame", "animPoseFrameResource"),
    ("drunkIdle", "drunkIdle"), ("walk2", "animWalk2Resource"), ("diePose", "animDiePoseResource"),
    ("walkback", "animWalkbackResource"), ("melee2", "animMelee2Resource"),
    ("extra43", "animExtra43Resource"), ("unk_1", "unk_1"), ("unused15", "animUnused15Resource"),
    ("unused18", "animUnused18Resource"), ("unused36", "animUnused36Resource"),
    ("animRunResource", "animRunResource"),
])
def test_set_anim_name_by_short_name(short, field):
    c = _blank()
    c.set_anim_name(short, "Molynx-walk")
    off = FIELDS[field].offset
    assert c.raw[off:off + 12] == b"Molynx-walk\0"
    assert c.resource_name(field) == "Molynx-walk"
    assert sum(c.raw) == sum(b"Molynx-walk")  # nothing else written


def test_set_name_pads_and_rejects_too_long():
    c = _blank()
    c.raw[0:32] = b"x" * 32
    c.set("name", "Molynx")
    assert c.raw[0:32] == b"Molynx" + bytes(26)
    with pytest.raises(ValueError):
        c.set("name", "y" * 32)  # needs a terminating NUL
    with pytest.raises(ValueError):
        c.set_resource_name("portraitResource", "z" * 64)
    with pytest.raises(KeyError):
        c.anim_name("fly")


def test_set_numbers():
    c = _blank(5537)
    c.set("hp", 900)
    c.set("cloneCreatureId", 7)
    c.set("light.mKPos", (1, 2, 3))
    c.set("unk_42", -2)
    assert c.get("hp") == 900 and c.get("cloneCreatureId") == 7
    assert c.get("light.mKPos") == (1, 2, 3) and c.get("unk_42") == -2
    assert struct.unpack_from("<H", c.raw, FIELDS["hp"].offset) == (900,)


def test_record_sizes_checked():
    with pytest.raises(ValueError):
        Creature(bytearray(5448))
    with pytest.raises(KeyError):
        _blank(5449).get("unk_40")  # only the 5537-byte records have the tail
