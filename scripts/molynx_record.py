"""The Molynx creature record: a modified copy of the Bile Demon's Creatures-block record.

`build_molynx(bile, stage, order_in_editor, text_ids)` copies the Bile Demon record and changes only
what makes it Molynx, in three cumulative stages (the splice and the test bundle build all three):

  1  name "Molynx", creatureId 47, orderInEditor (the caller's; max of the map's + 1).
  2  + the animation slot names (`Molynx-<stem>`, see SLOT_CLIPS and RULING F below).
  3  + the six stats of MULTIPLIERS, the icon1 / portrait resource names, and the five text IDs.

Everything else (sound category, lair, spells, attractions, jobs, anger values, flags, the
88-byte tail of a 5537-byte record, ...) stays the Bile Demon's. The Bile Demon is not modified.

cloneCreatureId is 0. The 33 creatures that are not "Unique ..." copies all have cloneCreatureId 0
(never their own ID; the Bile Demon's is 0 too), and only the 13 "Unique ..." creatures point at the
creature they clone (Unique Bile Demon -> 3); see creature_table.txt in the scratchpad and
`test_clone_creature_id_follows_the_non_unique_pattern`. Molynx is not a unique copy, so it gets 0.

Stats: float32 / float16 fields are fixed-point ints (value / 4096); MULTIPLIERS apply to the raw
integer `Creature.get` returns (Bile Demon mass 20480 raw = 5.0). The cap of
`min(cap, round(bile * multiplier))` is the field's own maximum from its type: 65535 for the ushort
fields (hp, meleeDamage, fear, threat, pay), 2**31 - 1 for the int32 mass.

Animation slots. SLOT_CLIPS maps an animation slot (short name, as `anim_slot` accepts) to the key
of the Molynx clip that drives it: a clip name as in the clip library (lowercase, underscores,
e.g. `being_picked_up`) or one of the pose stems PoseFrame, DiePose, DraggedPose. The resource name
written to the slot is "Molynx-" + that stem (walk -> `Molynx-walk`, draggedPose ->
`Molynx-DraggedPose`). RULING F: a slot that is empty in the Bile Demon record stays empty (the game
leaves it unused: the real record empties run, research, special, eatCorpse, loot, melee2, walk2,
dejectedPose, drunk, drunk2 and unused / extra slots), `unk_1` is never touched, and every other
non-empty slot not in SLOT_CLIPS takes `stand`. `slot_assignment(bile)` is the resulting
slot -> stem map and `stems_used(bile)` its set of stems (what the asset builder has to deliver).

LICENCE: GPL-3.0.
"""
import struct

from scripts.dk2kwd_creatures import ANIM_SLOTS, FIELDS, Creature, anim_slot

NAME = "Molynx"
CREATURE_ID = 47
CLONE_CREATURE_ID = 0  # see the module docstring
ICON_RESOURCE = r"gui\creatures\evil\molynx"
PORTRAIT_RESOURCE = r"gui\creatures\evil\molynx-bg"
PREFIX = "Molynx-"
DEFAULT_STEM = "stand"

MULTIPLIERS = {"hp": 1.6, "meleeDamage": 1.4, "fear": 1.2, "threat": 1.4, "mass": 1.2, "pay": 1.3}

# text_ids key -> creature field
TEXT_ID_FIELDS = {
    "name": "nameStringId", "tooltip": "tooltipStringId", "general": "generalDescriptionStringId",
    "strength": "strengthStringId", "weakness": "weaknessStringId",
}

# The 25 clips of the clip library: Molynx's own four (MegaGlest) and 21 retargeted ones.
CLIPS = (
    "stand", "walk", "attack", "die",
    "fidgeting", "running", "eating", "praying", "happy", "angry", "celebrating",
    "victory_celebration", "getting_hit", "being_slapped", "staggering", "waking_up", "knockback",
    "sleeping", "training", "fighting_another_creature", "patrolling", "guarding", "manufacturing",
    "interacting_with_objects", "being_picked_up",
)
POSE_STEMS = ("PoseFrame", "DiePose", "DraggedPose")

SLOT_CLIPS: dict[str, str] = {
    "walk": "walk", "run": "running", "draggedPose": "DraggedPose",
    "recoilHff": "getting_hit", "recoilHfb": "being_slapped",
    "melee1": "attack", "melee2": "attack", "special": "attack",
    "die": "die", "diePose": "DiePose", "happy": "happy", "angry": "angry",
    "swing": "being_picked_up", "sleep": "sleeping",
    "eat": "eating", "eatCorpse": "eating", "drink": "eating",
    "building": "manufacturing", "pray": "praying", "fallback": "knockback",
    "getUp": "waking_up", "dance": "celebrating",
    "drunk": "staggering", "drunk2": "staggering", "stunnedPose": "staggering",
    "dejectedPose": "staggering",
    "idle1": "stand", "idle2": "fidgeting", "drunkIdle": "fidgeting",
    "poseFrame": "PoseFrame", "walk2": "patrolling",
    "entrance": "walk", "walkback": "walk",
    "research": "interacting_with_objects", "loot": "interacting_with_objects",
    "magic": "getting_hit", "torture": "getting_hit", "elec": "getting_hit",
    "electrocute": "getting_hit",
}

# Slots that use a clip only as the nearest available, with why (for the in-game review).
_NEAREST_ONLY = {
    "magic": "No spell-casting clip in the library; a hit reaction is the nearest.",
    "torture": "No torture clip in the library; flinching (getting_hit) is the nearest.",
    "elec": "No electrocution clip in the library; flinching (getting_hit) is the nearest.",
    "electrocute": "No electrocution clip in the library; flinching (getting_hit) is the nearest.",
    "research": "No studying clip in the library; interacting_with_objects is the nearest.",
    "loot": "No looting clip in the library; interacting_with_objects is the nearest.",
    "special": "No special-attack clip in the library; the ordinary attack is reused.",
    "entrance": "No entrance clip in the library; the walk is the nearest.",
    "walkback": "No walking-backwards clip in the library; the forward walk is reused.",
    "melee2": "No second melee clip in the library; the ordinary attack is reused.",
    "drink": "No drinking clip in the library; eating is the nearest.",
    "eatCorpse": "No corpse-eating clip in the library; ordinary eating is reused.",
    "stunnedPose": "No stunned pose in the library; staggering is the nearest.",
    "dejectedPose": "No dejected pose in the library; staggering is the nearest.",
    "drunkIdle": "No drunken idle in the library; fidgeting is the nearest.",
}


def review_list() -> list[tuple[str, str, str]]:
    """(slot, clip used, reason it is only the nearest) for every slot that reuses a clip only as
    the nearest available one. A Bile Demon-empty slot (see RULING F) is listed but has no effect."""
    return [(slot, SLOT_CLIPS[slot], reason) for slot, reason in _NEAREST_ONLY.items()]


def slot_assignment(bile: Creature) -> dict[str, str]:
    """Animation slot field name -> stem, for every slot that is non-empty in `bile` (RULING F).
    `unk_1` is excluded; a slot not in SLOT_CLIPS takes `stand`."""
    table = {anim_slot(short): stem for short, stem in SLOT_CLIPS.items()}
    return {slot: table.get(slot, DEFAULT_STEM) for slot in ANIM_SLOTS
            if slot != "unk_1" and bile.resource_name(slot)}


def stems_used(bile: Creature) -> set[str]:
    """The clip keys and pose stems that the Molynx slots for `bile` point at."""
    return set(slot_assignment(bile).values())


def _field_max(name: str) -> int:
    fmt = FIELDS[name].fmt
    bits = 8 * struct.calcsize(fmt)
    return 2 ** (bits - 1) - 1 if fmt[-1].islower() else 2 ** bits - 1  # lowercase = signed


def _check_text_ids(text_ids: dict[str, int]) -> None:
    if set(text_ids) != set(TEXT_ID_FIELDS):
        raise ValueError(f"text_ids needs exactly the keys {sorted(TEXT_ID_FIELDS)}, got {sorted(text_ids)}")
    for key, value in text_ids.items():
        if not 0 <= value <= _field_max(TEXT_ID_FIELDS[key]):
            raise ValueError(f"text id {key}={value} does not fit a ushort")


def build_molynx(bile: Creature, stage: int, order_in_editor: int,
                 text_ids: dict[str, int] | None = None) -> Creature:
    """A new Molynx record built from `bile` (not modified), at stage 1, 2 or 3 (cumulative; see
    the module docstring). `text_ids` (keys name, tooltip, general, strength, weakness) is used at
    stage 3 only; without it stage 3 keeps the Bile Demon's text IDs."""
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2 or 3, not {stage!r}")
    if not 0 <= order_in_editor <= _field_max("orderInEditor"):
        raise ValueError(f"order_in_editor {order_in_editor} does not fit a byte")
    if stage == 3 and text_ids is not None:
        _check_text_ids(text_ids)

    molynx = Creature(bile.raw)  # copies the bytes
    molynx.set("name", NAME)
    molynx.set("creatureId", CREATURE_ID)
    molynx.set("orderInEditor", order_in_editor)
    molynx.set("cloneCreatureId", CLONE_CREATURE_ID)
    if stage >= 2:
        for slot, stem in slot_assignment(bile).items():
            molynx.set_resource_name(slot, PREFIX + stem)
    if stage >= 3:
        for name, multiplier in MULTIPLIERS.items():
            molynx.set(name, min(_field_max(name), round(bile.get(name) * multiplier)))
        molynx.set_resource_name("icon1Resource", ICON_RESOURCE)
        molynx.set_resource_name("portraitResource", PORTRAIT_RESOURCE)
        if text_ids is not None:
            for key, field in TEXT_ID_FIELDS.items():
                molynx.set(field, text_ids[key])
    return molynx
