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


# Exact values from the editor Creatures.kwd for every field the Molynx record (Task 3) reads or
# writes, plus their same-width neighbours, so a swap of neighbours or an off-by-one in a byte run
# cannot pass.
_COMMON_IDS = dict(tooltipStringId=2408, strengthStringId=2410, weaknessStringId=2410)
PINS = {
    "Imp": dict(
        scoreRating=10, roomCapacityNeeded=1, hungerFill=0, unhappyThreshold=50, slapDamage=100,
        creatureId=1, orderInEditor=1, cloneCreatureId=0, hp=500, hpFromChicken=0, meleeDamage=30,
        fear=5, threat=2, mass=1 * FIXED, pay=0, lairObjectId=0, unk_26=188,
        soundGategory="CREATURE_IMP", nameStringId=2771, generalDescriptionStringId=2347,
        uniqueNameTextId=0, **_COMMON_IDS),
    "Bile Demon": dict(
        scoreRating=90, roomCapacityNeeded=1, hungerFill=5, unhappyThreshold=50, slapDamage=182,
        creatureId=3, orderInEditor=11, cloneCreatureId=0, hp=6000, hpFromChicken=200,
        meleeDamage=525, fear=300, threat=120, mass=5 * FIXED, pay=600, lairObjectId=18, unk_26=188,
        soundGategory="CREATURE_BILE_DEMON", nameStringId=2356, generalDescriptionStringId=2357,
        uniqueNameTextId=0, **_COMMON_IDS),
    "Unique Bile Demon": dict(
        scoreRating=90, roomCapacityNeeded=1, hungerFill=5, unhappyThreshold=50, slapDamage=182,
        creatureId=41, orderInEditor=41, cloneCreatureId=3, hp=6000, hpFromChicken=200,
        meleeDamage=525, fear=400, threat=200, mass=5 * FIXED, pay=900, lairObjectId=18, unk_26=188,
        soundGategory="CREATURE_BILE_DEMON", nameStringId=2356, generalDescriptionStringId=2357,
        uniqueNameTextId=1774, **_COMMON_IDS),
    "Unique Firefly": dict(
        scoreRating=20, roomCapacityNeeded=1, hungerFill=0, unhappyThreshold=50, slapDamage=100,
        creatureId=32, orderInEditor=32, cloneCreatureId=12, hp=2000, hpFromChicken=0,
        meleeDamage=130, fear=19, threat=15, mass=2 * FIXED, pay=450, lairObjectId=21, unk_26=200,
        soundGategory="CREATURE_FIREFLY", nameStringId=2348, generalDescriptionStringId=2349,
        uniqueNameTextId=1765, **_COMMON_IDS),
}
PIN_RESOURCES = {
    "Imp": dict(icon1Resource=r"gui\creatures\evil\imp", icon2Resource="CreatureIcons#0",
                portraitResource=r"gui\creatures\evil\imp-bg"),
    "Bile Demon": dict(icon1Resource=r"gui\creatures\evil\bile_demon", icon2Resource="CreatureIcons#2",
                       portraitResource=r"gui\creatures\evil\bile_demon-bg"),
    "Unique Bile Demon": dict(icon1Resource=r"gui\creatures\evil\bile_demon",
                              icon2Resource="CreatureIcons#2", portraitResource=""),
}
PIN_ANIMS = {
    "Imp": dict(walk="Imp_Run", run="", melee1="Imp_Dig", die="Imp_Die1", idle1="Imp_Idle1",
                poseFrame="Imp-PoseFrame", drunkIdle=""),
    "Bile Demon": dict(walk="Bile_Demon-Walk", run="", melee1="Bile_Demon-Melee", die="Bile_Demon-Die",
                       idle1="BileDemon-Idle1", poseFrame="BD-PoseFrame", drunkIdle="BIL-Drunkidle"),
    "Unique Bile Demon": dict(walk="Bile_Demon-Walk", melee1="Bile_Demon-Melee", die="Bile_Demon-Die",
                              idle1="BileDemon-Idle1", drunkIdle=""),
}


@pytest.fixture(scope="module")
def editor_creatures():
    path = EDITOR_DIR / "Creatures.kwd"
    if not have(path):
        pytest.skip("editor Creatures.kwd not present")
    return {c.get("name"): c for c in read_creatures(_creatures_block(path))}


@pytest.mark.parametrize("creature", sorted(PINS))
def test_pinned_values(editor_creatures, creature):
    c = editor_creatures[creature]
    assert {f: c.get(f) for f in PINS[creature]} == PINS[creature]
    want = PIN_RESOURCES.get(creature, {})
    assert {f: c.resource_name(f) for f in want} == want
    want = PIN_ANIMS.get(creature, {})
    assert {s: c.anim_name(s) for s in want} == want


def test_offsets_match_the_template_comments():
    # Absolute offsets written as comments in creatures_struct.bt (hex), and for nameStringId /
    # tooltipStringId (f24, f26: between decomposeValue f22 and angerNoLair f28) and
    # cloneCreatureId (d3b: between shuffleSpeed d37 and firstPersonGammaEffect d3c) by position.
    # The template's "f3c" on generalDescriptionStringId is stale: f38 is per-creature (name id + 1)
    # while f3a and f3c hold 2410 for every creature but Jack, as the other two string ids do.
    template = dict(
        icon1Resource=0xC44, swipeSpeed=0xCEC, orderInEditor=0xCF6, angerStringIdGeneral=0xCF7,
        cloneCreatureId=0xD3B, firstPersonGammaEffect=0xD3C, portraitResource=0xD40, mass=0xEA8,
        speed=0xEB4, flags=0xEFC, hp=0xF0C, hpFromChicken=0xF0E, fear=0xF10, threat=0xF12,
        meleeDamage=0xF14, pay=0xF1C, decomposeValue=0xF22, nameStringId=0xF24, tooltipStringId=0xF26,
        angerNoLair=0xF28, entranceEffectId=0xF36, generalDescriptionStringId=0xF38,
        strengthStringId=0xF3A, weaknessStringId=0xF3C, slapEffectId=0xF3E, creatureId=0xF4B,
        unhappyThreshold=0xF4F, lairObjectId=0xF52, unk_26=0xF53, soundGategory=0xF56,
        drunkIdle=0xFD1, firstPersonMeleeResource=0x1027, animMelee2Resource=0x1083,
        animWalkbackResource=0x12CF, animPoseFrameResource=0x13E3, uniqueResource=0x14F1,
        flags2=0x1545)
    assert {f: FIELDS[f].offset for f in template} == template


def _base(name: str) -> str:
    return name.removeprefix("Unique ")


def test_unique_creatures_follow_their_base(source):
    _, creatures, _ = source
    by_name = {c.get("name").strip(): c for c in creatures}
    uniques = [c for c in creatures if c.get("name").startswith("Unique ")]
    assert len(uniques) == 13
    for u in uniques:
        base = by_name[_base(u.get("name"))]
        assert u.get("cloneCreatureId") == base.get("creatureId"), u.get("name")
        assert u.get("hp") >= base.get("hp"), u.get("name")
        for f in ("fear", "pay", "meleeDamage"):
            assert u.get(f) >= base.get(f), (u.get("name"), f)
        for f in ("nameStringId", "generalDescriptionStringId", "lairObjectId", "soundGategory", "mass"):
            assert u.get(f) == base.get(f), (u.get("name"), f)
        assert u.get("uniqueNameTextId") != 0 and base.get("uniqueNameTextId") == 0
    # Only unique creatures are clones.
    assert {c.get("name") for c in creatures if c.get("cloneCreatureId")} == {u.get("name") for u in uniques}


def test_lair_objects_are_creature_beds(source):
    # lairObjectId is an Objects.kwd id (record id - 1, its own id at record offset 858): each creature
    # with a lair sleeps in its own "<creature> Creature Bed" or the "Hero Creature Bed". unk_26 (the
    # next byte) holds 188 / 200, beyond the 145 objects, so a shift onto it cannot pass.
    path = EDITOR_DIR / "Objects.kwd"
    if not have(path):
        pytest.skip("Objects.kwd not present")
    objects = next(b for b in read_blocks(path.read_bytes()) if b.signature == 0xF0)
    size = len(objects.content) // objects.item_count
    names = [objects.content[i * size:i * size + 32].split(b"\0")[0].decode() for i in range(objects.item_count)]
    _, creatures, _ = source
    beds = {c.get("name"): names[c.get("lairObjectId") - 1] for c in creatures if c.get("lairObjectId")}
    assert len(beds) == 36
    for creature, bed in beds.items():
        if bed == "Hero Creature Bed" or creature == "Fairy":  # a Fairy uses the Dark Elf bed
            continue
        assert bed.endswith(" Creature Bed") and bed.removesuffix(" Creature Bed") in creature, creature
    assert beds["Fairy"] == "Dark Elf Creature Bed" and beds["Maiden Of The Nest"] == "Maiden Creature Bed"


def test_string_ids_relate(source):
    _, creatures, _ = source
    plain = [c for c in creatures if not c.get("name").startswith("Unique ") and c.get("name") != "Jack"]
    # Every non-unique creature has its own name string; the description string follows it except
    # for the Imp and the three princes (who share description 2401).
    assert len({c.get("nameStringId") for c in plain}) == len(plain) == 32
    off_by_one = {c.get("name") for c in plain if c.get("generalDescriptionStringId") != c.get("nameStringId") + 1}
    assert off_by_one == {"Imp", "Prince Tristran", "Prince Felix", "Prince Balder"}
    # tooltipStringId is the shared tooltip 2408 (Horny 2374, Jack none); strength and weakness are
    # 2410 for all but Jack, so the data cannot tell those two apart: their order is the template's.
    assert {c.get("name"): c.get("tooltipStringId") for c in creatures if c.get("tooltipStringId") != 2408} \
        == {"Horny": 2374, "Jack": 0}
    for f in ("strengthStringId", "weaknessStringId"):
        assert {c.get("name") for c in creatures if c.get(f) != 2410} == {"Jack"}
    # A fearsome creature always scares more than it threatens: fear and threat cannot be swapped.
    assert all(c.get("fear") > c.get("threat") for c in creatures)
    assert sorted(c.get("orderInEditor") for c in creatures) == list(range(1, 47))


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
