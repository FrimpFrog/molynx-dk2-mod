import struct

import pytest

from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_creatures import (
    ANIM_SLOTS, FIELDS, RECORD_SIZE, RESOURCE_NAME_SIZE, Creature, anim_slot, read_creatures,
)
from scripts.dk2paths import EDITOR_DIR, MAPS_DIR, have
from scripts.molynx_record import (
    CLIPS, MULTIPLIERS, POSE_STEMS, SLOT_CLIPS, build_molynx, review_list, slot_assignment,
    stems_used,
)

CREATURES = 0xAA
MAP_5537 = "01coopGlobals.kld"
ORDER = 47  # any order_in_editor; the real one is max(orderInEditor of the map) + 1
TEXT_IDS = {"name": 3001, "tooltip": 3002, "general": 3003, "strength": 3004, "weakness": 3005}
TEXT_FIELDS = {
    "name": "nameStringId", "tooltip": "tooltipStringId", "general": "generalDescriptionStringId",
    "strength": "strengthStringId", "weakness": "weaknessStringId",
}

STAGE1_FIELDS = {"name", "creatureId", "orderInEditor"}
STAT_FIELDS = {"hp", "meleeDamage", "fear", "threat", "mass", "pay"}
PORTRAIT_FIELDS = {"icon1Resource", "portraitResource"}

CLIP_LIBRARY = {
    "stand", "walk", "attack", "die", "fidgeting", "running", "eating", "praying", "happy", "angry",
    "celebrating", "victory_celebration", "getting_hit", "being_slapped", "staggering", "waking_up",
    "knockback", "sleeping", "training", "fighting_another_creature", "patrolling", "guarding",
    "manufacturing", "interacting_with_objects", "being_picked_up",
}
NEAREST_ONLY = [
    "magic", "torture", "elec", "electrocute", "research", "loot", "special", "entrance", "walkback",
    "melee2", "drink", "eatCorpse", "stunnedPose", "dejectedPose", "drunkIdle",
]
# The Bile Demon record (editor and every map) leaves exactly these animation slots empty.
BILE_EMPTY_SLOTS = {
    "animRunResource", "animResearchResource", "animUnused15Resource", "animDejectedPoseResource",
    "animUnused18Resource", "animDrunkResource", "animSpecialResource", "animEatCorpseResource",
    "animDrunk2Resource", "animLootResource", "animUnused36Resource", "animMelee2Resource",
    "animExtra43Resource", "animExtra44Resource", "animExtra45Resource", "animExtra46Resource",
    "animExtra47Resource", "animWalk2Resource",
}

needs_data = pytest.mark.skipif(not have(EDITOR_DIR) or not have(MAPS_DIR),
                                reason="DK2 editor data not present")


def _all_creatures(path):
    block = next(b for b in read_blocks(path.read_bytes()) if b.signature == CREATURES)
    return read_creatures(block)


@pytest.fixture(scope="module", params=[(EDITOR_DIR / "Creatures.kwd", 5449), (MAPS_DIR / MAP_5537, 5537)],
                ids=["editor-5449", "map-5537"])
def creatures(request):
    path, size = request.param
    if not have(path):
        pytest.skip(f"{path.name} not present")
    found = _all_creatures(path)
    assert len(found[0].raw) == size
    return found


@pytest.fixture
def bile(creatures):
    """A fresh copy of the Bile Demon (tests may edit it)."""
    original = creatures[2]
    assert original.get("name") == "Bile Demon"
    return Creature(original.raw)


# ---- helpers: which fields do two records differ at? -------------------------------------------

_FIELD_AT: dict[int, str] = {}
for _name, _field in FIELDS.items():
    for _offset in range(_field.offset, _field.end):
        assert _offset not in _FIELD_AT  # the fields tile the record
        _FIELD_AT[_offset] = _name


def changed_offsets(a: Creature, b: Creature) -> list[int]:
    assert len(a.raw) == len(b.raw)
    return [i for i, (x, y) in enumerate(zip(a.raw, b.raw)) if x != y]


def changed_fields(a: Creature, b: Creature) -> set[str]:
    return {_FIELD_AT[i] for i in changed_offsets(a, b)}


def expected_slot_names(bile: Creature) -> dict[str, str]:
    """Slot field -> expected name, derived afresh from the table and RULING F."""
    expected = {}
    in_table = {anim_slot(short): stem for short, stem in SLOT_CLIPS.items()}
    for field in ANIM_SLOTS:
        if field == "unk_1" or not bile.resource_name(field):
            continue
        expected[field] = "Molynx-" + in_table.get(field, "stand")
    return expected


# ---- the tables --------------------------------------------------------------------------------

def test_multipliers_are_the_specified_ones():
    assert MULTIPLIERS == {"hp": 1.6, "meleeDamage": 1.4, "fear": 1.2, "threat": 1.4, "mass": 1.2, "pay": 1.3}


def test_clip_library_is_the_25_clips():
    assert len(CLIPS) == 25 and set(CLIPS) == CLIP_LIBRARY
    assert set(POSE_STEMS) == {"PoseFrame", "DiePose", "DraggedPose"}


def test_every_slot_name_resolves_to_a_distinct_animation_slot():
    fields = [anim_slot(short) for short in SLOT_CLIPS]
    assert len(set(fields)) == len(fields) == 39
    assert "unk_1" not in fields


def test_every_clip_key_is_a_clip_or_a_pose_stem():
    assert set(SLOT_CLIPS.values()) <= CLIP_LIBRARY | set(POSE_STEMS)


def test_every_resource_name_fits_the_64_byte_field():
    for stem in set(SLOT_CLIPS.values()) | {"stand"}:
        name = "Molynx-" + stem
        assert len(name.encode("ascii")) < RESOURCE_NAME_SIZE, name  # < 64 leaves room for the NUL


def test_table_entries_from_the_brief():
    expected = {
        "walk": "walk", "run": "running", "draggedPose": "DraggedPose", "recoilHff": "getting_hit",
        "recoilHfb": "being_slapped", "melee1": "attack", "melee2": "attack", "special": "attack",
        "die": "die", "diePose": "DiePose", "happy": "happy", "angry": "angry",
        "swing": "being_picked_up", "sleep": "sleeping", "eat": "eating", "eatCorpse": "eating",
        "drink": "eating", "building": "manufacturing", "pray": "praying", "fallback": "knockback",
        "getUp": "waking_up", "dance": "celebrating", "drunk": "staggering", "drunk2": "staggering",
        "stunnedPose": "staggering", "dejectedPose": "staggering", "idle1": "stand",
        "idle2": "fidgeting", "drunkIdle": "fidgeting", "poseFrame": "PoseFrame",
        "walk2": "patrolling", "entrance": "walk", "walkback": "walk",
        "research": "interacting_with_objects", "loot": "interacting_with_objects",
        "magic": "getting_hit", "torture": "getting_hit", "elec": "getting_hit",
        "electrocute": "getting_hit",
    }
    assert SLOT_CLIPS == expected


def test_review_list_names_exactly_the_nearest_only_slots():
    rows = review_list()
    assert sorted(slot for slot, _, _ in rows) == sorted(NEAREST_ONLY)
    for slot, clip, reason in rows:
        assert clip == SLOT_CLIPS[slot]
        assert isinstance(reason, str) and len(reason) > 10


# ---- build_molynx ------------------------------------------------------------------------------

@needs_data
def test_bile_demon_slots_this_module_relies_on(bile):
    empty = {slot for slot in ANIM_SLOTS if slot != "unk_1" and not bile.resource_name(slot)}
    assert empty == BILE_EMPTY_SLOTS
    assert bile.get("creatureId") == 3
    assert bile.get("cloneCreatureId") == 0


@needs_data
@pytest.mark.parametrize("stage", [0, 4, -1, 5])
def test_stage_must_be_1_2_or_3(bile, stage):
    with pytest.raises(ValueError):
        build_molynx(bile, stage, ORDER)


@needs_data
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_build_does_not_mutate_the_bile_demon_and_keeps_the_size(bile, stage):
    before = bytes(bile.raw)
    molynx = build_molynx(bile, stage, ORDER, TEXT_IDS)
    assert bytes(bile.raw) == before
    assert molynx is not bile and molynx.raw is not bile.raw
    assert len(molynx.raw) == len(bile.raw)


@needs_data
def test_stage1_differs_only_at_name_creature_id_and_order(bile):
    molynx = build_molynx(bile, 1, ORDER)
    assert changed_fields(bile, molynx) == STAGE1_FIELDS
    assert molynx.get("name") == "Molynx"
    assert molynx.get("creatureId") == 47
    assert molynx.get("orderInEditor") == ORDER
    # Everything else, the tail of a 5537-byte record included, is the Bile Demon's bytes.
    assert molynx.get("cloneCreatureId") == 0
    assert bytes(molynx.raw[RECORD_SIZE:]) == bytes(bile.raw[RECORD_SIZE:])
    assert molynx.get("soundGategory") == bile.get("soundGategory") == "CREATURE_BILE_DEMON"
    assert molynx.get("lairObjectId") == bile.get("lairObjectId") == 18


@needs_data
def test_stage1_ignores_text_ids(bile):
    assert bytes(build_molynx(bile, 1, ORDER, TEXT_IDS).raw) == bytes(build_molynx(bile, 1, ORDER).raw)


@needs_data
def test_stage2_adds_only_animation_slot_names(bile):
    stage1, stage2 = build_molynx(bile, 1, ORDER), build_molynx(bile, 2, ORDER)
    expected = expected_slot_names(bile)
    assert len(expected) == 29  # 48 slots - unk_1 - the 18 the Bile Demon leaves empty
    assert changed_fields(stage1, stage2) == set(expected)
    assert changed_fields(bile, stage2) == STAGE1_FIELDS | set(expected)
    for field, name in expected.items():
        assert stage2.resource_name(field) == name
        assert len(name.encode("ascii")) <= RESOURCE_NAME_SIZE
    # Only the 64-byte name area changes; the 20 bytes after it are the Bile Demon's.
    for offset in changed_offsets(stage1, stage2):
        field = FIELDS[_FIELD_AT[offset]]
        assert offset - field.offset < RESOURCE_NAME_SIZE
    assert bytes(stage2.raw[RECORD_SIZE:]) == bytes(bile.raw[RECORD_SIZE:])


@needs_data
def test_stage2_slot_names_for_the_real_bile_demon(bile):
    stage2 = build_molynx(bile, 2, ORDER)
    got = {slot: stage2.anim_name(slot) for slot in
           ("walk", "draggedPose", "poseFrame", "diePose", "swing", "idle1", "idle2", "drunkIdle",
            "entrance", "walkback", "magic", "building", "stunnedPose", "recoilHfb")}
    assert got == {
        "walk": "Molynx-walk", "draggedPose": "Molynx-DraggedPose", "poseFrame": "Molynx-PoseFrame",
        "diePose": "Molynx-DiePose", "swing": "Molynx-being_picked_up", "idle1": "Molynx-stand",
        "idle2": "Molynx-fidgeting", "drunkIdle": "Molynx-fidgeting", "entrance": "Molynx-walk",
        "walkback": "Molynx-walk", "magic": "Molynx-getting_hit", "building": "Molynx-manufacturing",
        "stunnedPose": "Molynx-staggering", "recoilHfb": "Molynx-being_slapped",
    }


@needs_data
def test_empty_bile_slots_stay_empty_and_unk_1_is_untouched(bile):
    for stage in (2, 3):
        molynx = build_molynx(bile, stage, ORDER, TEXT_IDS)
        for field in BILE_EMPTY_SLOTS:
            assert molynx.resource_name(field) == ""
        unk = FIELDS["unk_1"]
        assert bytes(molynx.raw[unk.offset:unk.end]) == bytes(bile.raw[unk.offset:unk.end])


@needs_data
def test_slots_not_in_the_table_take_stand_when_non_empty(bile):
    # The real Bile Demon uses no such slot, so give it two.
    bile.set_anim_name("unused15", "Bile-Odd1")
    bile.set_anim_name("extra43", "Bile-Odd2")
    bile.set_anim_name("run", "Bile-Run")  # empty in the real record, in the table as running
    stage2 = build_molynx(bile, 2, ORDER)
    assert stage2.anim_name("unused15") == "Molynx-stand"
    assert stage2.anim_name("extra43") == "Molynx-stand"
    assert stage2.anim_name("run") == "Molynx-running"
    assert stage2.anim_name("extra44") == ""
    assert slot_assignment(bile)["animUnused15Resource"] == "stand"


@needs_data
def test_slot_assignment_and_stems_used(bile):
    assignment = slot_assignment(bile)
    assert set(assignment) == set(ANIM_SLOTS) - BILE_EMPTY_SLOTS - {"unk_1"}
    assert {slot: "Molynx-" + stem for slot, stem in assignment.items()} == expected_slot_names(bile)
    assert stems_used(bile) == set(assignment.values())
    assert stems_used(bile) <= CLIP_LIBRARY | set(POSE_STEMS)
    # What Task 8 has to deliver for the real Bile Demon: 18 of the 25 clips and the 3 poses.
    assert stems_used(bile) == {
        "walk", "DraggedPose", "getting_hit", "being_slapped", "attack", "die", "DiePose", "happy",
        "angry", "being_picked_up", "sleeping", "eating", "manufacturing", "praying", "knockback",
        "waking_up", "celebrating", "staggering", "stand", "fidgeting", "PoseFrame",
    }
    assert set(CLIPS) - stems_used(bile) == {
        "running", "victory_celebration", "training", "fighting_another_creature", "patrolling",
        "guarding", "interacting_with_objects",
    }


@needs_data
def test_stage3_adds_only_stats_portrait_resources_and_text_ids(bile):
    stage2, stage3 = build_molynx(bile, 2, ORDER), build_molynx(bile, 3, ORDER, TEXT_IDS)
    changed = changed_fields(stage2, stage3)
    assert changed == STAT_FIELDS | PORTRAIT_FIELDS | set(TEXT_FIELDS.values())
    # The change from the Bile Demon is stage 1 + the slot names + that.
    assert changed_fields(bile, stage3) == (
        STAGE1_FIELDS | set(expected_slot_names(bile)) | STAT_FIELDS | PORTRAIT_FIELDS
        | set(TEXT_FIELDS.values()))
    assert bytes(stage3.raw[RECORD_SIZE:]) == bytes(bile.raw[RECORD_SIZE:])
    for offset in changed_offsets(stage2, stage3):
        field = _FIELD_AT[offset]
        if field in PORTRAIT_FIELDS:
            assert offset - FIELDS[field].offset < RESOURCE_NAME_SIZE


@needs_data
def test_stage3_portrait_resources(bile):
    stage3 = build_molynx(bile, 3, ORDER, TEXT_IDS)
    assert stage3.resource_name("icon1Resource") == r"gui\creatures\evil\molynx"
    assert stage3.resource_name("portraitResource") == r"gui\creatures\evil\molynx-bg"
    # RULING A: the second icon (CreatureIcons#N) is the Bile Demon's.
    assert stage3.resource_name("icon2Resource") == bile.resource_name("icon2Resource") == "CreatureIcons#2"
    assert bile.resource_name("icon1Resource") == r"gui\creatures\evil\bile_demon"
    assert bile.resource_name("portraitResource") == r"gui\creatures\evil\bile_demon-bg"


@needs_data
def test_stage3_stats_are_the_scaled_bile_values_capped_at_the_field_maximum(bile):
    stage3 = build_molynx(bile, 3, ORDER, TEXT_IDS)
    for name, multiplier in MULTIPLIERS.items():
        fmt = FIELDS[name].fmt
        bits = 8 * struct.calcsize(fmt)
        cap = 2 ** (bits - 1) - 1 if fmt[-1].islower() else 2 ** bits - 1  # u16 -> 65535, int32 -> 2**31-1
        assert stage3.get(name) == min(cap, round(bile.get(name) * multiplier)), name
        assert stage3.get(name) > bile.get(name), name
    # Pinned for the real Bile Demon (hp 6000, melee 525, fear 300, threat 120, mass 5.0, pay 600).
    assert {n: stage3.get(n) for n in MULTIPLIERS} == {
        "hp": 9600, "meleeDamage": 735, "fear": 360, "threat": 168, "mass": 24576, "pay": 780}


@needs_data
def test_stat_cap_is_the_field_maximum_not_a_fixed_65535(bile):
    bile.set("hp", 60000)       # u16: 60000 * 1.6 = 96000 -> capped at 65535
    bile.set("mass", 2 ** 31 - 10)  # int32: capped at 2**31 - 1, not 65535
    stage3 = build_molynx(bile, 3, ORDER, TEXT_IDS)
    assert stage3.get("hp") == 65535
    assert stage3.get("mass") == 2 ** 31 - 1


@needs_data
def test_stage3_text_ids(bile):
    stage3 = build_molynx(bile, 3, ORDER, TEXT_IDS)
    for key, field in TEXT_FIELDS.items():
        assert stage3.get(field) == TEXT_IDS[key]
        assert bile.get(field) != TEXT_IDS[key]


@needs_data
def test_stage3_without_text_ids_keeps_the_bile_demons(bile):
    stage2, stage3 = build_molynx(bile, 2, ORDER), build_molynx(bile, 3, ORDER)
    assert changed_fields(stage2, stage3) == STAT_FIELDS | PORTRAIT_FIELDS
    for field in TEXT_FIELDS.values():
        assert stage3.get(field) == bile.get(field)


@needs_data
@pytest.mark.parametrize("bad", [
    {"name": 1, "tooltip": 2, "general": 3, "strength": 4},                       # weakness missing
    {**TEXT_IDS, "flavour": 9},                                                    # unknown key
    {**TEXT_IDS, "name": 65536},                                                   # not a ushort
])
def test_bad_text_ids_are_rejected(bile, bad):
    with pytest.raises(ValueError):
        build_molynx(bile, 3, ORDER, bad)


@needs_data
def test_order_in_editor_must_fit_a_byte(bile):
    assert build_molynx(bile, 1, 255).get("orderInEditor") == 255
    with pytest.raises(ValueError):
        build_molynx(bile, 1, 256)
    with pytest.raises(ValueError):
        build_molynx(bile, 1, -1)


@needs_data
def test_clone_creature_id_follows_the_non_unique_pattern(creatures):
    # Evidence for the decision recorded in the module docstring: the 33 creatures that are not
    # "Unique ..." copies all have cloneCreatureId 0 (never their own ID); only the 13 uniques
    # point at the creature they clone.
    plain = [c for c in creatures if not c.get("name").startswith("Unique")]
    assert len(plain) == 33
    assert {c.get("cloneCreatureId") for c in plain} == {0}
    assert not any(c.get("cloneCreatureId") == c.get("creatureId") for c in creatures)
    assert build_molynx(creatures[2], 1, ORDER).get("cloneCreatureId") == 0


@needs_data
def test_molynx_is_never_a_clone_even_of_a_unique_creature(bile):
    bile.set("cloneCreatureId", 3)
    assert build_molynx(bile, 1, ORDER).get("cloneCreatureId") == 0
