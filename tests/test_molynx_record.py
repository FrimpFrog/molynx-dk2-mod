import struct

import pytest

from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_creatures import (
    ANIM_SLOTS, FIELDS, RECORD_SIZE, RESOURCE_NAME_SIZE, Creature, anim_slot, read_creatures,
)
from scripts.dk2paths import EDITOR_DIR, MAPS_DIR, have
from molynx_files import ANIM_INFO, FILE_WRAPS
from scripts.molynx_record import (
    CLIPS, MAX_CLOSE_GAP, MULTIPLIERS, ONE_WAY_GAPS, POSE_STEMS, SLOT_CLIPS, SOURCE_WRAP, STATIC_POSES,
    WrapException, build_molynx, file_plan, review_list, slot_assignment, stems_used,
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
# Stage 2 also gives him the Horned Reaper's size (the editor's "Horny" record).
SIZE = {"scale": 5734, "height": 4096, "eyeHeight": 2744}
STAT_FIELDS = {"hp", "meleeDamage", "fear", "threat", "mass", "pay"}
PORTRAIT_FIELDS = {"icon1Resource", "portraitResource"}

CLIP_LIBRARY = {
    "stand", "walk", "attack", "die", "fidgeting", "running", "eating", "praying", "happy", "angry",
    "celebrating", "victory_celebration", "getting_hit", "being_slapped", "staggering", "waking_up",
    "knockback", "sleeping", "training", "fighting_another_creature", "patrolling", "guarding",
    "manufacturing", "interacting_with_objects", "being_picked_up",
    "falling", "torture", "drinking",   # the in-game review round (2026-11-07)
}
NEAREST_ONLY = [
    "magic", "elec", "electrocute", "research", "loot", "special", "entrance", "walkback",
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

# The real Bile Demon: slot -> (file stem, its 20-byte tail before, after). Before: the editor
# record. After, by the stage-2 rule: flags lose 0x8 / 0x10; frames = file frames (- 1 when the file
# loops); fps 30; startDist / endDist / sometimesOne and type 6 kept; startAf = endAf = 0. Pose slots
# (static meshes, type 5: the three poses, stunnedPose and elec) keep their tail.
TAILS = {
    "animWalkResource": ("walk", "19000000280000001e0000006e08560606282800",
                         "010000002e0000001e0000006e08560606000000"),
    "animDraggedPoseResource": ("DraggedPose", "00000000001000000100000068049e0005000000",
                                "00000000001000000100000068049e0005000000"),
    "animRecoilHffResource": ("getting_hit", "000000002e0000001e0000000000000006000000",
                              "00000000110000001e0000000000000006000000"),
    "animMelee1Resource": ("attack", "000000005a0000001e0000000000000006000000",
                           "00000000290000001e0000000000000006000000"),
    "animMagicResource": ("getting_hit", "00000000410000001e0000000000000006000000",
                          "00000000110000001e0000000000000006000000"),
    "animDieResource": ("die", "010000004f0000001e0000000000000006000000",
                        "010000001f0000001e0000000000000006000000"),
    "animHappyResource": ("happy", "00000000820000001e0000000000000006000000",
                          "000000003b0000001e0000000000000006000000"),
    "animAngryResource": ("angry", "000000002d0000001e0000000000000006000000",
                          "00000000170000001e0000000000000006000000"),
    "animStunnedPoseResource": ("StunnedPose", "00000000001000000100000068049e0005000000",
                                "00000000001000000100000068049e0005000000"),
    "animSwingResource": ("being_picked_up", "000000003c0000001e0000000000000006000000",
                          "000000003c0000001e0000000000000006000000"),
    "animSleepResource": ("sleeping", "00000000780000000f0000000000000006000000",
                          "00000000360000001e0000000000000006000000"),  # 55, looping (review round)
    "animEatResource": ("eating", "01000000500000001e0000000000000006000000",
                        "01000000530000001e0000000000000006000000"),
    "animTortureResource": ("torture", "01000000640000001e0000000000000006000000",
                            "01000000640000001e0000000000000006000000"),  # 100 (Giant torture)
    "animDrinkResource": ("drinking", "00000000960000001e0000000000000006000000",
                          "00000000530000001e0000000000000006000000"),
    "animIdle1Resource": ("stand", "00000000be0000001e0000000000000006000000",
                          "000000003d0000001e0000000000000006000000"),
    "animRecoilHfbResource": ("being_slapped", "000000002e0000001e0000000000000006000000",
                              "00000000230000001e0000000000000006000000"),
    "animBuildingResource": ("manufacturing", "18000000210000001e00000000000000062d2000",
                             "00000000440000001e0000000000000006000000"),  # 69, looping (the mace)
    "animPrayResource": ("praying", "180000003c0000001e0000000000000006383700",
                         "00000000780000001e0000000000000006000000"),
    "animFallbackResource": ("falling", "01000000210000001e0000000000000006000000",
                             "010000000f0000001e0000000000000006000000"),  # 15 (Giant fallback)
    "animElecResource": ("ElecPose", "0000000000100000010000006804c80005000000",
                         "0000000000100000010000006804c80005000000"),
    "animElectrocuteResource": ("getting_hit", "010000000a0000001e0000000000000006000000",
                                "01000000110000001e0000000000000006000000"),
    "animGetUpResource": ("waking_up", "010000003f0000001e0000000000000006000000",
                          "010000002e0000001e0000000000000006000000"),
    "animDanceResource": ("celebrating", "010000005f0000001e0000000000000006000000",
                          "01000000610000001e0000000000000006000000"),  # 97 (Troll dance)
    "animEntranceResource": ("walk_once", "00000000930000001e0000000000000006000000",
                             "000000002f0000001e0000000000000006000000"),
    "animIdle2Resource": ("fidgeting", "00000000650000001e0000000000000006000000",
                          "000000005f0000001e0000000000000006000000"),  # 95 (Troll idle2)
    "drunkIdle": ("fidgeting", "01000000780000001e0000000000000006000001",
                  "010000005f0000001e0000000000000006000001"),
    "animWalkbackResource": ("walk", "190000001a0000001e000000a103000006120c00",
                             "010000002e0000001e000000a103000006000000"),
    "animPoseFrameResource": ("PoseFrame", "0100000000100000000000006804f40005c90601",
                              "0100000000100000000000006804f40005c90601"),
    "animDiePoseResource": ("DiePose", "0100000000100000000000006804ec0005c9fe00",
                            "0100000000100000000000006804ec0005c9fe00"),
}
# resource.bt: the tail's sub-fields by offset after the 64-byte name.
TAIL_FIELDS = {0: "flags", 1: "flags", 2: "flags", 3: "flags", 4: "frames", 5: "frames",
               6: "frames", 7: "frames", 8: "fps", 9: "fps", 10: "fps", 11: "fps",
               12: "startDist", 13: "startDist", 14: "endDist", 15: "endDist", 16: "type",
               17: "startAf", 18: "endAf", 19: "sometimesOne"}


def build(bile, stage, order=ORDER, text_ids=None, **kw):
    """build_molynx with the real files' anim_info and wrap flags."""
    kw.setdefault("file_wraps", FILE_WRAPS)
    return build_molynx(bile, stage, order, text_ids, kw.pop("anim_info", ANIM_INFO), **kw)


def tail_hex(creature, field):
    offset = FIELDS[field].offset + RESOURCE_NAME_SIZE
    return bytes(creature.raw[offset:offset + 20]).hex()


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


def test_clip_library_is_the_28_clips():
    assert len(CLIPS) == 28 and set(CLIPS) == CLIP_LIBRARY
    assert set(POSE_STEMS) == {"PoseFrame", "DiePose", "DraggedPose", "StunnedPose", "ElecPose"}
    assert STATIC_POSES == {"staggering": "StunnedPose"}


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
        "drink": "drinking", "building": "manufacturing", "pray": "praying", "fallback": "falling",
        "getUp": "waking_up", "dance": "celebrating", "drunk": "staggering", "drunk2": "staggering",
        "stunnedPose": "StunnedPose", "dejectedPose": "staggering", "idle1": "stand",
        "idle2": "fidgeting", "drunkIdle": "fidgeting", "poseFrame": "PoseFrame",
        "walk2": "patrolling", "entrance": "walk", "walkback": "walk",
        "research": "interacting_with_objects", "loot": "interacting_with_objects",
        "magic": "getting_hit", "torture": "torture", "elec": "ElecPose",
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
    molynx = build(bile, stage, text_ids=TEXT_IDS)
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


def _changed_tail_fields(a: Creature, b: Creature) -> set[tuple[str, str]]:
    """(resource field, part) of every differing byte: part is "name" or a tail sub-field."""
    out = set()
    for offset in changed_offsets(a, b):
        field = _FIELD_AT[offset]
        at = offset - FIELDS[field].offset
        out.add((field, "name" if at < RESOURCE_NAME_SIZE else TAIL_FIELDS[at - RESOURCE_NAME_SIZE]))
    return out


@needs_data
def test_stage2_differs_only_in_slot_names_and_the_tails_of_animation_slots(bile):
    stage1, stage2 = build_molynx(bile, 1, ORDER), build(bile, 2)
    expected = expected_slot_names(bile)
    assert len(expected) == 29  # 48 slots - unk_1 - the 18 the Bile Demon leaves empty
    assert set(TAILS) == set(expected)
    assert changed_fields(stage1, stage2) == set(expected) | set(SIZE)
    assert changed_fields(bile, stage2) == STAGE1_FIELDS | set(expected) | set(SIZE)
    assert {k: stage2.get(k) for k in SIZE} == SIZE
    changed = {(f, part) for f, part in _changed_tail_fields(stage1, stage2) if f not in SIZE}
    assert {field for field, part in changed if part == "name"} == set(expected)
    tails = {(field, part) for field, part in changed if part != "name"}
    poses = {"animPoseFrameResource", "animDiePoseResource", "animDraggedPoseResource",
             "animStunnedPoseResource", "animElecResource"}
    assert not {field for field, _ in tails} & poses  # pose tails stay the Bile Demon's
    assert {part for _, part in tails} <= set(TAIL_FIELDS.values())
    # Pinned per slot: walk loses Start/End and its start/end frames, its frames become 46 ...
    assert ("animWalkResource", "flags") in tails and ("animWalkResource", "startAf") in tails
    assert ("animWalkResource", "startDist") not in tails  # kept: walk was an animation
    assert not any(part in ("type", "startDist", "endDist", "sometimesOne") for _, part in tails)
    assert bytes(stage2.raw[RECORD_SIZE:]) == bytes(bile.raw[RECORD_SIZE:])


@needs_data
def test_stage2_tails_for_the_real_bile_demon(bile):
    stage2 = build(bile, 2)
    for field, (stem, before, after) in TAILS.items():
        assert tail_hex(bile, field) == before, field
        assert tail_hex(stage2, field) == after, field
        assert stage2.resource_name(field) == "Molynx-" + stem, field


@needs_data
def test_stage2_tail_rule_written_out(bile):
    """The same rule as TAILS, spelled out field by field against the record's own values."""
    stage2 = build(bile, 2)
    for field, (stem, _, _) in TAILS.items():
        old, new = bile.art_tail(field), stage2.art_tail(field)
        if stem in POSE_STEMS:
            assert new == old and old.type == 5, field
            continue
        info = ANIM_INFO[stem]
        assert new.flags == old.flags & ~0x18, field
        assert new.frames == info["frames"] - (1 if info["wrap"] else 0), field
        assert (new.fps, new.type, new.start_af, new.end_af) == (30, 6, 0, 0), field
        assert old.type == 6, field  # every clip goes into an animation slot
        assert (new.start_dist, new.end_dist, new.sometimes_one) == (
            old.start_dist, old.end_dist, old.sometimes_one), field


@needs_data
def test_stage2_slot_names_for_the_real_bile_demon(bile):
    stage2 = build(bile, 2)
    got = {slot: stage2.anim_name(slot) for slot in
           ("walk", "draggedPose", "poseFrame", "diePose", "swing", "idle1", "idle2", "drunkIdle",
            "entrance", "walkback", "magic", "building", "stunnedPose", "recoilHfb", "elec",
            "recoilHff")}
    assert got == {
        "walk": "Molynx-walk", "draggedPose": "Molynx-DraggedPose", "poseFrame": "Molynx-PoseFrame",
        "diePose": "Molynx-DiePose", "swing": "Molynx-being_picked_up", "idle1": "Molynx-stand",
        "idle2": "Molynx-fidgeting", "drunkIdle": "Molynx-fidgeting", "entrance": "Molynx-walk_once",
        "walkback": "Molynx-walk", "magic": "Molynx-getting_hit", "building": "Molynx-manufacturing",
        "stunnedPose": "Molynx-StunnedPose", "recoilHfb": "Molynx-being_slapped",
        "elec": "Molynx-ElecPose", "recoilHff": "Molynx-getting_hit",
    }


# ---- the file plan: which file each slot names and whether it loops ------------------------------

SOURCE_WRAPS = {  # the clip library's own flags (test_molynx_assets checks them on the source)
    "die": False, "victory_celebration": False, "getting_hit": False, "being_slapped": False,
    "waking_up": False, "knockback": False, "falling": False, "torture": False,
}


def test_source_wrap_is_the_clip_librarys():
    assert SOURCE_WRAP == {clip: SOURCE_WRAPS.get(clip, True) for clip in CLIP_LIBRARY}


@needs_data
def test_file_plan_for_the_real_bile_demon(bile):
    plan = file_plan(bile, FILE_WRAPS)
    assert plan.slots == {field: stem for field, (stem, _, _) in TAILS.items()}
    # walk loops in walk / walkback and is opened for entrance; every other stem is needed one way
    # only (stunnedPose and elec take static poses, which have no loop flag).
    assert plan.wraps == {
        "walk": True, "walk_once": False, "getting_hit": False,
        "stand": False, "fidgeting": False, "eating": False, "happy": False, "attack": False,
        "angry": False, "celebrating": False, "die": False, "being_slapped": False,
        "falling": False, "waking_up": False, "sleeping": True, "being_picked_up": True,
        "manufacturing": True, "praying": True, "torture": False, "drinking": False,
    }
    assert "staggering" not in stems_used(bile, plan)  # the editor's Bile Demon has no staggering slot
    assert stems_used(bile, plan) == set(plan.slots.values()) == set(plan.wraps) | set(POSE_STEMS)
    for stem in stems_used(bile, plan):
        assert len(("Molynx-" + stem).encode("ascii")) < RESOURCE_NAME_SIZE
    assert all(ANIM_INFO[stem]["wrap"] == wrap for stem, wrap in plan.wraps.items())


@needs_data
def test_file_plan_case_of_file_names_does_not_matter(bile):
    upper = {name.upper(): wrap for name, wrap in FILE_WRAPS.items()}
    assert file_plan(bile, upper) == file_plan(bile, FILE_WRAPS)


@needs_data
def test_a_stem_needed_both_ways_keeps_its_source_flag_and_the_other_slots_take_a_variant(bile):
    # fidgeting loops in the source; make drunkIdle's file loop while idle2's does not.
    plan = file_plan(bile, {**FILE_WRAPS, "bil-drunkidle": True})
    assert plan.slots["drunkIdle"] == "fidgeting" and plan.wraps["fidgeting"] is True
    assert plan.slots["animIdle2Resource"] == "fidgeting_once"
    assert plan.wraps["fidgeting_once"] is False


@needs_data
def test_a_stem_needed_one_way_has_no_variant(bile):
    plan = file_plan(bile, FILE_WRAPS)
    assert SOURCE_WRAP["stand"] is True and plan.wraps["stand"] is False  # opened, same name
    assert not any(s.startswith(("stand_", "sleeping_")) for s in plan.wraps)


def test_one_way_clips_are_the_measured_ones():
    # last-to-first gap in median frame steps of the source clips that do not loop (pinned; the
    # asset tests measure them on the source): sleeping lies down, waking_up gets up.
    assert MAX_CLOSE_GAP == 10
    assert set(ONE_WAY_GAPS) == {"waking_up"}   # sleeping loops since the review round
    assert ONE_WAY_GAPS["waking_up"] == pytest.approx(25.25, abs=0.01)
    assert all(not SOURCE_WRAP[clip] for clip in ONE_WAY_GAPS)


@needs_data
def test_a_one_way_clip_is_not_looped_even_where_the_bile_demons_file_loops(bile):
    plan = file_plan(bile, FILE_WRAPS)
    # since the review round sleeping is a breathing loop, like the Bile Demon's bed file: no exception
    assert FILE_WRAPS["biledemon-bed(sleeping)"] is True
    assert plan.wraps["sleeping"] is True and plan.slots["animSleepResource"] == "sleeping"
    assert plan.wrap_exceptions == ()
    # the tail of a looping file: frames = file frames - 1 (55 - 1), fps 30
    assert build(bile, 2).art_tail("animSleepResource").frames == 54
    # a one-way clip still refuses a loop: waking_up in a slot whose Bile file loops
    looped = file_plan(bile, {**FILE_WRAPS, "bile_demon-getup": True})
    assert looped.wraps["waking_up"] is False
    assert [e.stem for e in looped.wrap_exceptions] == ["waking_up"]


@needs_data
def test_a_one_way_clip_needed_both_ways_gets_no_loop_variant(bile):
    plan = file_plan(bile, FILE_WRAPS)
    assert not any(e.stem == "waking_up" for e in plan.wrap_exceptions)  # getUp ends anyway
    # Were getUp's file a loop, waking_up would still not be looped (no waking_up_loop either).
    looped = file_plan(bile, {**FILE_WRAPS, "bile_demon-getup": True})
    assert looped.wraps["waking_up"] is False and "waking_up_loop" not in looped.wraps
    assert looped.slots["animGetUpResource"] == "waking_up"
    assert WrapException("waking_up", ("animGetUpResource",), True,
                         ONE_WAY_GAPS["waking_up"]) in looped.wrap_exceptions


@needs_data
def test_file_plan_refuses_unknown_files_and_mismatched_kinds(bile):
    missing = dict(FILE_WRAPS)
    del missing["bile_demon-walk"]
    with pytest.raises(ValueError, match="Bile_Demon-Walk"):
        file_plan(bile, missing)
    with pytest.raises(ValueError, match="animWalkResource"):
        file_plan(bile, {**FILE_WRAPS, "bile_demon-walk": None})  # type 6 slot, mesh file
    idle1 = bile.art_tail("animIdle1Resource")
    bile.set_art_tail("animIdle1Resource", idle1._replace(type=5))
    with pytest.raises(ValueError, match="animIdle1Resource"):
        file_plan(bile, FILE_WRAPS)  # a static mesh slot that would take a clip (stand)


@needs_data
def test_a_pose_slot_that_is_not_a_mesh_is_refused(bile):
    tail = bile.art_tail("animDraggedPoseResource")
    bile.set_art_tail("animDraggedPoseResource", tail._replace(type=6))
    with pytest.raises(ValueError, match="animDraggedPoseResource"):
        file_plan(bile, FILE_WRAPS)


@needs_data
def test_a_named_slot_of_type_none_is_renamed_but_keeps_its_tail(bile):
    # 63 shipped maps name Bile_Demon-Dejected_Pose in a slot whose tail is all zero (type NONE).
    bile.set_anim_name("dejectedPose", "Bile_Demon-Dejected_Pose")
    plan = file_plan(bile, FILE_WRAPS)
    assert plan.slots["animDejectedPoseResource"] == "staggering"
    stage2 = build(bile, 2)
    assert stage2.anim_name("dejectedPose") == "Molynx-staggering"
    assert tail_hex(stage2, "animDejectedPoseResource") == "00" * 20


@needs_data
def test_a_static_mesh_dejected_pose_slot_takes_the_stunned_pose(bile):
    # 24 shipped maps name Bile_Demon-Dejected_Pose in a static mesh slot (type 5).
    bile.set_anim_name("dejectedPose", "Bile_Demon-Dejected_Pose")
    stunned = bile.art_tail("animStunnedPoseResource")
    bile.set_art_tail("animDejectedPoseResource", stunned)
    assert file_plan(bile, FILE_WRAPS).slots["animDejectedPoseResource"] == "StunnedPose"
    stage2 = build(bile, 2)
    assert stage2.anim_name("dejectedPose") == "Molynx-StunnedPose"
    assert stage2.art_tail("animDejectedPoseResource") == stunned


# ---- anim_info and file_wraps are required from stage 2 -----------------------------------------

@needs_data
@pytest.mark.parametrize("stage", [2, 3])
def test_stage_2_and_3_need_anim_info_and_file_wraps(bile, stage):
    with pytest.raises(ValueError, match="anim_info"):
        build_molynx(bile, stage, ORDER, TEXT_IDS, file_wraps=FILE_WRAPS)
    with pytest.raises(ValueError, match="file_wraps"):
        build_molynx(bile, stage, ORDER, TEXT_IDS, ANIM_INFO)


@needs_data
def test_stage1_ignores_anim_info_and_file_wraps(bile):
    plain = bytes(build_molynx(bile, 1, ORDER).raw)
    assert bytes(build_molynx(bile, 1, ORDER, None, {"x": 1}, file_wraps={}).raw) == plain


@needs_data
def test_anim_info_must_cover_every_used_file_and_extra_entries_are_fine(bile):
    for stem in ("walk_once", "getting_hit", "stand"):
        short = {k: v for k, v in ANIM_INFO.items() if k != stem}
        with pytest.raises(ValueError, match=stem):
            build(bile, 2, anim_info=short)
    extra = {**ANIM_INFO, "dancing_extra": {"frames": 5, "fps": 30, "wrap": True}}
    assert build(bile, 2, anim_info=extra).raw == build(bile, 2).raw


@needs_data
@pytest.mark.parametrize("change", [
    {"wrap": True},          # stand must not loop: its slot (idle1) ends
    {"fps": 15},             # every file is 30 fps
    {"frames": 1},           # the game has no 1-frame animation
    {"frames": 2 ** 32},     # does not fit the u32
])
def test_anim_info_that_contradicts_the_plan_is_refused(bile, change):
    bad = {**ANIM_INFO, "stand": {**ANIM_INFO["stand"], **change}}
    with pytest.raises(ValueError, match="stand"):
        build(bile, 2, anim_info=bad)


@needs_data
@pytest.mark.parametrize("bad", [
    [("stand", 61)],                                       # not a mapping at all
    {**ANIM_INFO, "stand": (61, 30, False)},               # an entry that is not a mapping
    {**ANIM_INFO, "stand": None},
])
def test_anim_info_that_is_not_a_mapping_is_a_value_error(bile, bad):
    with pytest.raises(ValueError, match="anim_info"):
        build(bile, 2, anim_info=bad)


@needs_data
def test_frames_follow_the_files_loop_flag(bile):
    info = {**ANIM_INFO, "walk": {"frames": 100, "fps": 30, "wrap": True},
            "stand": {"frames": 100, "fps": 30, "wrap": False}}
    stage2 = build(bile, 2, anim_info=info)
    assert stage2.art_tail("animWalkResource").frames == 99
    assert stage2.art_tail("animWalkbackResource").frames == 99
    assert stage2.art_tail("animIdle1Resource").frames == 100


@needs_data
def test_empty_bile_slots_stay_empty_and_unk_1_is_untouched(bile):
    for stage in (2, 3):
        molynx = build(bile, stage, text_ids=TEXT_IDS)
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
    stage2 = build(bile, 2)
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
    # What Task 8 has to deliver for the editor's Bile Demon: 17 of the 25 clips and the 5 poses
    # (staggering only where a map's Bile Demon names dejectedPose in a type NONE slot, e.g. Siege).
    assert stems_used(bile) == {
        "walk", "DraggedPose", "getting_hit", "being_slapped", "attack", "die", "DiePose", "happy",
        "angry", "being_picked_up", "sleeping", "eating", "manufacturing", "praying", "falling",
        "torture", "drinking",
        "waking_up", "celebrating", "StunnedPose", "ElecPose", "stand", "fidgeting", "PoseFrame",
    }
    assert set(CLIPS) - stems_used(bile) == {
        "running", "victory_celebration", "training", "fighting_another_creature", "patrolling",
        "guarding", "interacting_with_objects", "staggering", "knockback",
    }


@needs_data
def test_stage3_adds_only_stats_portrait_resources_and_text_ids(bile):
    stage2, stage3 = build(bile, 2), build(bile, 3, text_ids=TEXT_IDS)
    changed = changed_fields(stage2, stage3)
    assert changed == STAT_FIELDS | PORTRAIT_FIELDS | set(TEXT_FIELDS.values())
    # The change from the Bile Demon is stage 1 + the slot names and tails + that.
    assert changed_fields(bile, stage3) == (
        STAGE1_FIELDS | set(expected_slot_names(bile)) | set(SIZE) | STAT_FIELDS | PORTRAIT_FIELDS
        | set(TEXT_FIELDS.values()))
    assert bytes(stage3.raw[RECORD_SIZE:]) == bytes(bile.raw[RECORD_SIZE:])
    for offset in changed_offsets(stage2, stage3):
        field = _FIELD_AT[offset]
        if field in PORTRAIT_FIELDS:
            assert offset - FIELDS[field].offset < RESOURCE_NAME_SIZE


@needs_data
def test_stage3_portrait_resources(bile):
    stage3 = build(bile, 3, text_ids=TEXT_IDS)
    assert stage3.resource_name("icon1Resource") == r"gui\creatures\evil\molynx"
    assert stage3.resource_name("portraitResource") == r"gui\creatures\evil\molynx-bg"
    # RULING A: the second icon (CreatureIcons#N) is the Bile Demon's.
    assert stage3.resource_name("icon2Resource") == bile.resource_name("icon2Resource") == "CreatureIcons#2"
    assert bile.resource_name("icon1Resource") == r"gui\creatures\evil\bile_demon"
    assert bile.resource_name("portraitResource") == r"gui\creatures\evil\bile_demon-bg"


@needs_data
def test_stage3_stats_are_the_scaled_bile_values_capped_at_the_field_maximum(bile):
    stage3 = build(bile, 3, text_ids=TEXT_IDS)
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
    stage3 = build(bile, 3, text_ids=TEXT_IDS)
    assert stage3.get("hp") == 65535
    assert stage3.get("mass") == 2 ** 31 - 1


@needs_data
def test_stage3_text_ids(bile):
    stage3 = build(bile, 3, text_ids=TEXT_IDS)
    for key, field in TEXT_FIELDS.items():
        assert stage3.get(field) == TEXT_IDS[key]
        assert bile.get(field) != TEXT_IDS[key]


@needs_data
def test_stage3_without_text_ids_keeps_the_bile_demons(bile):
    stage2, stage3 = build(bile, 2), build(bile, 3)
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
        build(bile, 3, text_ids=bad)


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


@pytest.mark.skipif(not have(EDITOR_DIR), reason="editor data not present")
def test_size_fields_are_the_horned_reapers():
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.dk2kwd_creatures import read_creatures
    from scripts.molynx_record import SIZE_FIELDS
    (block,) = [b for b in read_blocks((EDITOR_DIR / "Creatures.kwd").read_bytes()) if b.signature == 0xAA]
    (horny,) = [c for c in read_creatures(block) if c.get("name") == "Horny"]
    assert SIZE_FIELDS == SIZE == {k: horny.get(k) for k in SIZE}
