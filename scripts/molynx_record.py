"""The Molynx creature record: a modified copy of the Bile Demon's Creatures-block record.

`build_molynx(bile, stage, order_in_editor, text_ids, anim_info, file_wraps=...)` copies the Bile
Demon record and changes only what makes it Molynx, in three cumulative stages (the splice and the
test bundle build all three):

  1  name "Molynx", creatureId 47, orderInEditor (the caller's; max of the map's + 1).
  2  + the animation slots: each assigned slot's name (`Molynx-<file stem>`, see SLOT_CLIPS,
     RULING F and the file plan below) and, for a slot that plays an animation, the 20 bytes after
     the name that describe the file (frames, fps, flags, type; see "Slot tails"); and his size,
     SIZE_FIELDS (the redesign, 2026-11-06); and the icon1 / portrait resource names (his own
     creature-panel pictures, second review round 2026-11-07).
  3  + his own character (2026-11-07, the user's brief: a tankier, stronger, slightly faster Bile
     Demon, the Horned Reaper's bodyguard, eating less, made his own; 2026-11-10: the Bile Demon's
     health, drawn by a 5 x 5 Hatchery and a 10 x 10 Combat Pit): STATS, PERSONALITY and the
     five text IDs (his texts are TEXTS, added to the game's Text.str by the builder).

Everything else (sound category, lair, spells, attractions, jobs, anger values, flags, the
88-byte tail of a 5537-byte record, ...) stays the Bile Demon's. The Bile Demon is not modified.

cloneCreatureId is 0. The 33 creatures that are not "Unique ..." copies all have cloneCreatureId 0
(never their own ID; the Bile Demon's is 0 too), and only the 13 "Unique ..." creatures point at the
creature they clone (Unique Bile Demon -> 3); see creature_table.txt in the scratchpad and
`test_clone_creature_id_follows_the_non_unique_pattern`. Molynx is not a unique copy, so it gets 0.

Stats: float32 / float16 fields are fixed-point ints (value / 4096): STATS holds the raw integers
`Creature.set` takes (mass 24576 raw = 6.0).

Animation slots. SLOT_CLIPS maps an animation slot (short name, as `anim_slot` accepts) to the key
of the Molynx clip that drives it: a clip name as in the clip library (lowercase, underscores,
e.g. `being_picked_up`) or one of the pose stems (POSE_STEMS: PoseFrame, DiePose, DraggedPose,
StunnedPose, ElecPose; static meshes, one frame of a clip each). The resource name
written to the slot is "Molynx-" + that stem (walk -> `Molynx-walk`, draggedPose ->
`Molynx-DraggedPose`). RULING F: a slot that is empty in the Bile Demon record stays empty (the game
leaves it unused: the real record empties run, research, special, eatCorpse, loot, melee2, walk2,
dejectedPose, drunk, drunk2 and unused / extra slots), `unk_1` is never touched, and every other
non-empty slot not in SLOT_CLIPS takes `stand`. Static mesh slots get static poses: the game puts a
static mesh (type 5) in animStunnedPoseResource for 44 of 44 creatures and in animElecResource for
38 of 39, and the Bile Demon's are type 5, so stunnedPose takes `StunnedPose` and elec `ElecPose`.
A static mesh slot whose clip has a static pose of its own (STATIC_POSES: staggering ->
StunnedPose; dejectedPose is type 5 in 24 shipped maps) takes that pose. `slot_assignment(bile)` is
the resulting slot -> stem map and `stems_used(bile)` its set of stems.

File plan (`file_plan(bile, file_wraps)`, the one source of truth for the record and the exporter).
The engine's idle / job logic keys on a file looping or ending, so each slot's Molynx file must loop
exactly when the Bile Demon's file in that slot does (`file_wraps`: the Bile Demon's file name,
any case -> its KMF wrap flag, None for a static mesh; molynx_assets.bile_file_wraps reads them). A
static mesh slot (type 5) takes a pose and needs no flag. A slot that names a file but whose type
is NONE (0; 63 shipped maps name Bile_Demon-Dejected_Pose so) is not loaded by the game: it is renamed but needs nothing and keeps
its tail. One-way clips are never looped: a clip whose source does not loop and whose last frame
is more than MAX_CLOSE_GAP = 10 median frame steps from frame 0 (ONE_WAY_GAPS, measured on the
source and checked by molynx_assets.plan_molynx: sleeping 37.09, which lies down and stays down,
and waking_up 25.25) stays non-looping even in a slot that needs a loop; closing it would snap
back across the whole motion every cycle. Such a slot is listed in `FilePlan.wrap_exceptions`
(stem, slots, the Bile Demon file's flag, the gap); the engine then most likely plays the file once
and holds its last frame (for sleeping, the lying pose). The real Bile Demon has one exception:
sleeping in the sleep slot, whose Bile file loops. A stem needed one way only is that one file
with that flag (stand, fidgeting, eating, happy, attack, angry, celebrating are opened). A stem
needed BOTH ways keeps
its plain file for the flag that equals its source flag (SOURCE_WRAP) and the slots that need the
other flag name a variant, `<stem>_once` (not looping) or `<stem>_loop` (looping): for the real
Bile Demon `entrance` -> `walk_once` (walk / walkback loop, the entrance file ends).
`FilePlan.slots` is slot -> file stem, `FilePlan.wraps` file stem -> wrap for the animation files; the exporter writes exactly those flags (molynx_assets.required_wraps).

Slot tails (dk2kwd_creatures.ArtTail; resource.bt ArtResource after the 64-byte name): flags u32
@+0, frames u32 @+4, fps u32 @+8, startDist u16 @+12, endDist u16 @+14, type u8 @+16 (5 mesh,
6 animated mesh), startAf @+17, endAf @+18, sometimesOne @+19. Evidence, over the 1220 non-empty
slots of the 46 editor creatures checked against their files (Task 8a report, item C): flag 0x8
(HAS_START_ANIMATION) is set in 203 slots and a <name>Start.kmf exists for all 203, 0x10
(HAS_END_ANIMATION) in 176 with <name>End.kmf for all 176, and startAf / endAf are those files'
frame counts; frames = the file's frame count in 649 of 753 non-looping slots and = frame count - 1
in 222 of 243 looping ones (Bile Demon walk 41 -> 40, idle1 190 -> 190); fps is 30 in 969 of 996
(Bile Demon sleep 15); type always matches the file kind (996 / 996 type 6 animations, 224 / 224
type 5 meshes); startDist / endDist are non-zero almost only on movement slots. Stage 2 therefore
writes, for every slot that plays one of Molynx's animation files: flags without 0x8 and 0x10 (he
has no Start / End files; every other bit stays the Bile Demon's), frames = the file's frames (- 1
when it loops), fps 30 (every file is resampled to 30), type 6, startAf = endAf = 0, and keeps
startDist / endDist / sometimesOne. Pose slots (type 5 static meshes: poseFrame, diePose,
draggedPose, stunnedPose, elec) keep their tail exactly (a pose in a slot of another type, or a
clip in a type-5 slot, is refused). `anim_info` ({file stem: {frames, fps, wrap}}, as
molynx_assets.export_molynx_assets returns it) must cover every animation file the plan uses and
agree with its flags; extra entries are ignored.

LICENCE: GPL-3.0.
"""
import struct

from dataclasses import dataclass
from typing import Mapping, NamedTuple

from scripts.dk2kwd_creatures import (
    ANIM_SLOTS, ART_FLAG_HAS_END_ANIMATION, ART_FLAG_HAS_START_ANIMATION, ART_TYPE_ANIMATING_MESH,
    ART_TYPE_MESH, ART_TYPE_NONE, FIELDS, Creature, anim_slot,
)

NAME = "Molynx"
CREATURE_ID = 47
CLONE_CREATURE_ID = 0  # see the module docstring
ICON_RESOURCE = r"gui\creatures\evil\molynx"
PORTRAIT_RESOURCE = r"gui\creatures\evil\molynx-bg"
PREFIX = "Molynx-"
DEFAULT_STEM = "stand"

# His size (stage 2): the Horned Reaper's. The game draws a creature's model at its record's
# `scale` (fixed point, 4096 = 1): 8 creatures use it (Horny 5734 = 1.4, King 1.25, Prince and Lord
# 1.2), and a record's `height` follows file height x scale (Horny 0.739 x 1.4 = 1.035, height
# 1.000). Molynx's model files are as tall as Horny's (0.740 vs 0.739), so with Horny's scale,
# height and eyeHeight he stands exactly as tall as the Horned Reaper in the game. `width` stays the
# Bile Demon's (0.425; Horny 0.348): Molynx is as bulky as the Bile Demon.
SIZE_FIELDS = {"scale": 5734, "height": 4096, "eyeHeight": 2744}
FX = 4096   # fixed point: float32 fields hold value x 4096

# Stage 3: his own numbers (raw values; the Bile Demon's in the comments). A tank and a bodyguard:
# the Bile Demon's health (the user, 2026-11-10: "Health should be the same as bile demons"; not in
# STATS, so each map keeps its own Bile Demon's hp, which differs between maps: 6000 in most, 1813
# or 8000 in a few) and harder hits than the Bile Demon (still below the Horned Reaper's 700), a
# little faster, quicker out of a stun, heavier, more feared, paid more, loyal under torture. He is
# rock, so he hardly ever eats: the Horned Reaper's hungerRate (32767 s between meals) and no anger
# when unfed (angerNoFood 1, the Reaper's). A meal stays one chicken (hungerFill 1, a chicken's 200
# health): with the Reaper's hungerFill 0 and no health from a chicken, the first in-game test had him
# standing in the Hatchery while chickens and eggs poured out of it without end.
STATS = {
    "meleeDamage": 600,                      # 525 (Horned Reaper 700)
    "speed": round(0.45 * FX),               # 0.40
    "runSpeed": round(0.65 * FX),            # 0.56
    "meleeRecharge": round(3.0 * FX),        # 3.5 s
    "meleeRange": round(0.7 * FX),           # 0.8 (his gas reached further than the mace)
    "stunDuration": round(3.0 * FX),         # 5 s
    "mass": round(6.0 * FX),                 # 5.0
    "fear": 600,                             # 300
    "threat": 250,                           # 120
    "pay": 700,                              # 600
    "maxGoldHeld": 7000,                     # 6000
    "hungerRate": 134213632,                 # 220 s; the Horned Reaper's: never hungry
    "hungerFill": 1,                         # 5 chickens a meal
    "angerNoFood": 1,                        # 150
    "timeAwake": 330,                        # 270 s
    "tortureTimeToConvert": round(240 * FX), # 200 s
    "possessionManaCost": 400,               # 300
    "scoreRating": 115,                      # 90
    "manufacturePerSecond": 80,              # 50
    "expPerSecondTraining": 25,              # 20
    "manaGenPrayer": 70,                     # 50
    "angerSlap": 4500,                       # 3000: proud, slaps anger him more
}
# Room and job IDs (OpenKeeper's tables; the room order is Rooms.kwd's): 4 Hatchery, 7 Training
# Room, 9 Guard Room, 10 Workshop, 16 Combat Pit; jobs 3 research, 4 train, 5 manufacture, 6 guard.
# An attraction's roomSize is the room's tile count (the Bile Demon's Hatchery 25 = 5 x 5).
ROOM_HATCHERY, ROOM_TRAINING, ROOM_GUARD, ROOM_COMBAT_PIT = 4, 7, 9, 16
HATCHERY_TILES, COMBAT_PIT_TILES = 5 * 5, 10 * 10
JOB_RESEARCH, JOB_TRAIN, JOB_GUARD = 3, 4, 6
SPELL_CAST_ARMOUR = 15   # creature spell IDs are 1 + the entry's place in the CreatureSpells block
# Ground Slam: creature spell 23 rebuilt by molynx_ground_slam (the game's unused Chicken Arrow
# spell). It plays his magic clip (ground_slam, molynx_design.slam_frames), whose mace reaches the
# floor at frame GROUND_SLAM_IMPACT_FRAME of 30 fps: the spell's shot leaves then (shotDelay, in
# 1/4096 s since the clip began; the Bile Demon's Gas Cloud leaves at 2744, frame 20 of his 65).
# The shot starts where the mace lands: the spell's shotOffset is (x, y, z) from his feet turned
# to his facing, -y ahead (every creature casting ahead of itself has a negative y: Warlock -1761,
# Dark Elf -2048, Salamander -2457). The mace head lands 0.545 of his height ahead (slam_frames);
# his height in the game is the KMF's 0.74 times his scale 1.4: about one tile, 0.56 tile ahead.
SPELL_GROUND_SLAM = 23
GROUND_SLAM_IMPACT_FRAME = 34
GROUND_SLAM_DELAY = round(GROUND_SLAM_IMPACT_FRAME / 30 * 4096)      # 4642
GROUND_SLAM_OFFSET = (0, -2300, 205)
CAST_ARMOUR_OFFSET = (0, -1433, 1638)       # the Monk's (his Cast Armour)
# Stage 3: what makes him unlike the Bile Demon. Drawn by a 5 x 5 Hatchery and a 10 x 10 Combat Pit
# (the user, 2026-11-10; the Bile Demon: any Workshop and a 5 x 5 Hatchery), happiest guarding and training (the Bile Demon: training and
# manufacturing), hates research, and no gas: in place of the Bile Demon's Gas Cloud (level 4) and
# Gas Missile (level 8) he has two close-range abilities: Ground Slam from level 2 (his mace into
# the floor, hurting every enemy within two tiles) and Stone Skin from level 4 (the game's Cast
# Armour creature spell, CreatureSpells entry 15, the Monk's and the Fairy's at level 8; the game
# casts it on his allies). Both play his magic clip and leave at its impact frame.
SOUND_CATEGORY = "CREATURE_MOLYNX"
PERSONALITY = {
    "attractions[0].present": 1, "attractions[0].roomId": ROOM_HATCHERY,
    "attractions[0].roomSize": HATCHERY_TILES,
    "attractions[1].present": 1, "attractions[1].roomId": ROOM_COMBAT_PIT,
    "attractions[1].roomSize": COMBAT_PIT_TILES,
    "happyJobs[0].jobType": JOB_GUARD, "happyJobs[0].chance": 50,
    "happyJobs[1].jobType": JOB_TRAIN, "happyJobs[1].chance": 50,
    "hateJobs[0]": JOB_RESEARCH, "hateJobs[1]": 0,
    "spells[0].creatureSpellId": SPELL_GROUND_SLAM, "spells[0].levelAvailable": 2,
    "spells[0].playAnimation": 1, "spells[0].shotDelay": GROUND_SLAM_DELAY,
    "spells[0].shotOffset": GROUND_SLAM_OFFSET,
    "spells[1].creatureSpellId": SPELL_CAST_ARMOUR, "spells[1].levelAvailable": 4,
    "spells[1].playAnimation": 1, "spells[1].shotDelay": GROUND_SLAM_DELAY,
    "spells[1].shotOffset": CAST_ARMOUR_OFFSET,
    # his own sounds: the game loads Data/Sound/Sfx/Global/<category>BANK.map and SFX.map by this
    # name (molynx_soundbank), so the folder's creature_molynx bank plays in place of the Bile Demon's
    "soundGategory": SOUND_CATEGORY,
}
# Stage 3: his texts, by text_ids key ({n}: a value the game fills in, as in the Bile Demon's own
# description "Bile Demon.  {26}.  Fighting Style: {27}."). The tooltip stays the game's shared one
# ("{28} the {29} ...", text 2408), so TEXTS has no "tooltip".
TEXTS = {
    "name": "Molynx",
    "general": "Molynx.  Living stone, sworn to guard the Horned Reaper.  {26}.  Fighting Style: {27}.",
    "strength": "Rock-hard hide and a crushing mace. Made of stone, he hardly ever eats, shrugs off "
                "blows, slams the ground to shake every foe around him and hardens into Stone Skin "
                "when the fighting starts.",
    "weakness": "Proud and slow-witted. He won't study in the Library, and slapping him makes him "
                "furious.",
}
SHARED_TOOLTIP_ID = 2408

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
    "falling", "torture", "drinking",   # the in-game review round (molynx_design.REVIEW_CLIPS)
    "ground_slam",                      # his spell-casting clip (Ground Slam, molynx_design.slam_frames)
)
POSE_STEMS = ("PoseFrame", "DiePose", "DraggedPose", "StunnedPose", "ElecPose")
STATIC_POSES = {"staggering": "StunnedPose"}  # a type-5 slot assigned this clip takes this pose
# Whether each clip loops in the clip library itself (molynx_assets checks it against the source).
SOURCE_WRAP: dict[str, bool] = {clip: clip not in (
    "die", "victory_celebration", "getting_hit", "being_slapped", "waking_up", "knockback",
    "falling", "torture", "ground_slam") for clip in CLIPS}
# Non-looping source clips that end far from where they began (last-to-first gap, in median frame
# steps, as molynx_assets.loop_gap measures it): never closed into a loop (see "File plan").
MAX_CLOSE_GAP = 10
ONE_WAY_GAPS: dict[str, float] = {"waking_up": 25.251}
ONCE_SUFFIX, LOOP_SUFFIX = "_once", "_loop"   # variant files: the clip opened / closed
GAME_FPS = 30

SLOT_CLIPS: dict[str, str] = {
    "walk": "walk", "run": "running", "draggedPose": "DraggedPose",
    "recoilHff": "getting_hit", "recoilHfb": "being_slapped",
    "melee1": "attack", "melee2": "attack", "special": "attack",
    "die": "die", "diePose": "DiePose", "happy": "happy", "angry": "angry",
    "swing": "being_picked_up", "sleep": "sleeping",
    "eat": "eating", "eatCorpse": "eating", "drink": "drinking",
    "building": "manufacturing", "pray": "praying", "fallback": "falling",
    "getUp": "waking_up", "dance": "celebrating",
    "drunk": "staggering", "drunk2": "staggering", "stunnedPose": "StunnedPose",
    "dejectedPose": "staggering",
    "idle1": "stand", "idle2": "fidgeting", "drunkIdle": "fidgeting",
    "poseFrame": "PoseFrame", "walk2": "patrolling",
    "entrance": "walk", "walkback": "walk",
    "research": "interacting_with_objects", "loot": "interacting_with_objects",
    "magic": "ground_slam", "torture": "torture", "elec": "ElecPose",
    "electrocute": "getting_hit",
}

# Slots that use a clip only as the nearest available, with why (for the in-game review).
_NEAREST_ONLY = {
    "elec": ("No electrocution pose in the library; the peak frame of flinching (getting_hit) is "
             "the nearest."),
    "electrocute": "No electrocution clip in the library; flinching (getting_hit) is the nearest.",
    "research": "No studying clip in the library; interacting_with_objects is the nearest.",
    "loot": "No looting clip in the library; interacting_with_objects is the nearest.",
    "special": "No special-attack clip in the library; the ordinary attack is reused.",
    "entrance": "No entrance clip in the library; the walk is the nearest.",
    "walkback": "No walking-backwards clip in the library; the forward walk is reused.",
    "melee2": "No second melee clip in the library; the ordinary attack is reused.",
    "drink": "Drinking is the eating motion without the chicken.",
    "eatCorpse": "No corpse-eating clip in the library; ordinary eating is reused.",
    "stunnedPose": "Lying where falling ends (and waking_up begins).",
    "dejectedPose": "No dejected pose in the library; staggering is the nearest.",
    "drunkIdle": "No drunken idle in the library; fidgeting is the nearest.",
}


def review_list() -> list[tuple[str, str, str]]:
    """(slot, clip used, reason it is only the nearest) for every slot that reuses a clip only as
    the nearest available one. A Bile Demon-empty slot (see RULING F) is listed but has no effect."""
    return [(slot, SLOT_CLIPS[slot], reason) for slot, reason in _NEAREST_ONLY.items()]


def slot_assignment(bile: Creature) -> dict[str, str]:
    """Animation slot field name -> stem, for every slot that is non-empty in `bile` (RULING F).
    `unk_1` is excluded; a slot not in SLOT_CLIPS takes `stand`; a static mesh slot whose clip has
    a static pose (STATIC_POSES) takes that pose."""
    table = {anim_slot(short): stem for short, stem in SLOT_CLIPS.items()}
    out = {}
    for slot in ANIM_SLOTS:
        if slot == "unk_1" or not bile.resource_name(slot):
            continue
        stem = table.get(slot, DEFAULT_STEM)
        if bile.art_tail(slot).type == ART_TYPE_MESH:
            stem = STATIC_POSES.get(stem, stem)
        out[slot] = stem
    return out


class WrapException(NamedTuple):
    """A one-way clip kept non-looping in slots whose Bile Demon file loops (or that need a loop)."""
    stem: str
    slots: tuple
    bile_wrap: bool | None   # the Bile Demon file's flag (None: a static mesh)
    gap: float               # last-to-first gap of the source clip, in median frame steps


@dataclass(frozen=True)
class FilePlan:
    slots: dict    # slot field name -> file stem (a clip, a pose stem or a _once / _loop variant)
    wraps: dict    # animation file stem -> whether that file loops
    wrap_exceptions: tuple = ()   # WrapException, one per one-way stem a slot wanted looped


def variant(stem: str, wrap: bool) -> str:
    return stem + (LOOP_SUFFIX if wrap else ONCE_SUFFIX)


def clip_of(file_stem: str) -> tuple[str, bool | None]:
    """(clip, forced wrap) of a file stem: ("walk", False) for walk_once, ("walk", None) for walk."""
    for suffix, wrap in ((ONCE_SUFFIX, False), (LOOP_SUFFIX, True)):
        if file_stem.endswith(suffix) and file_stem[:-len(suffix)] in CLIPS:
            return file_stem[:-len(suffix)], wrap
    return file_stem, None


def _slot_need(bile: Creature, slot: str, stem: str, files: dict) -> bool | None:
    """The wrap flag the slot's Molynx file needs (None: no need, a type NONE slot)."""
    kind = bile.art_tail(slot).type
    name = bile.resource_name(slot)
    if stem in POSE_STEMS:
        if kind != ART_TYPE_MESH:
            raise ValueError(f"{slot} takes the pose {stem} but is type {kind}, not a mesh (5)")
        return None
    if kind == ART_TYPE_NONE:
        return None
    if name.lower() not in files:
        raise ValueError(f"file_wraps has no entry for {name} (slot {slot})")
    wrap = files[name.lower()]
    if kind == ART_TYPE_MESH:
        raise ValueError(f"{slot} is a static mesh slot (type 5) but takes the clip {stem}, not a pose")
    if kind == ART_TYPE_ANIMATING_MESH:
        if wrap is None:
            raise ValueError(f"{slot} is an animation slot (type 6) but {name} is a static mesh")
        return bool(wrap)
    raise ValueError(f"{slot} has resource type {kind}; only 0, 5 and 6 are understood")


def file_plan(bile: Creature, file_wraps: Mapping[str, bool | None]) -> FilePlan:
    """Which file each assigned slot names and whether each animation file loops (module
    docstring, "File plan"). `file_wraps`: the Bile Demon's file name (any case) -> wrap flag, None
    for a static mesh."""
    files = {name.lower(): wrap for name, wrap in file_wraps.items()}
    assignment = slot_assignment(bile)
    needs = {slot: _slot_need(bile, slot, stem, files) for slot, stem in assignment.items()}
    held: dict[str, list] = {}  # one-way stem -> the slots that wanted it looped
    for slot, stem in assignment.items():
        if needs[slot] is True and stem in ONE_WAY_GAPS:
            held.setdefault(stem, []).append(slot)
            needs[slot] = False
    flags: dict[str, set] = {}
    for slot, stem in assignment.items():
        if stem not in POSE_STEMS:
            flags.setdefault(stem, set())
            if needs[slot] is not None:
                flags[stem].add(needs[slot])
    slots, wraps = {}, {}
    for slot, stem in assignment.items():
        if stem in POSE_STEMS:
            slots[slot] = stem
            continue
        need = flags[stem]
        plain = SOURCE_WRAP[stem] if len(need) != 1 else next(iter(need))
        wraps[stem] = plain
        if needs[slot] is None or needs[slot] == plain:
            slots[slot] = stem
        else:
            slots[slot] = variant(stem, needs[slot])
            wraps[slots[slot]] = needs[slot]
    exceptions = tuple(
        WrapException(stem, tuple(held[stem]),
                      files.get(bile.resource_name(held[stem][0]).lower()), ONE_WAY_GAPS[stem])
        for stem in held)
    return FilePlan(slots, wraps, exceptions)


def stems_used(bile: Creature, plan: FilePlan | None = None) -> set[str]:
    """The clip keys and pose stems that the Molynx slots for `bile` point at; with `plan`, the
    file stems (variants included) that the stage-2 record names."""
    return set((plan.slots if plan is not None else slot_assignment(bile)).values())


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


def _anim_frames(stem: str, wrap: bool, anim_info: Mapping[str, dict]) -> int:
    """The record's frame count for file `stem` (checked against the plan's wrap flag)."""
    if not isinstance(anim_info, Mapping):
        raise ValueError(f"anim_info must be a mapping of file stems, not {type(anim_info).__name__}")
    if stem not in anim_info:
        raise ValueError(f"anim_info has no entry for the file {PREFIX}{stem}")
    info = anim_info[stem]
    if not isinstance(info, Mapping):
        raise ValueError(f"anim_info[{stem!r}] must be a mapping with frames, fps and wrap, not "
                         f"{type(info).__name__}")
    if bool(info.get("wrap")) != wrap:
        raise ValueError(f"anim_info says {stem} {'loops' if info.get('wrap') else 'does not loop'}"
                         f", its slots need the opposite")
    if info.get("fps") != GAME_FPS:
        raise ValueError(f"anim_info gives {stem} {info.get('fps')} fps, every file is {GAME_FPS}")
    frames = info.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or not 2 <= frames <= 2 ** 32 - 1 + wrap:
        raise ValueError(f"anim_info gives {stem} {frames!r} frames")
    return frames - 1 if wrap else frames


def _write_slots(molynx: Creature, bile: Creature, plan: FilePlan, anim_info: Mapping[str, dict]):
    for slot, stem in plan.slots.items():
        molynx.set_resource_name(slot, PREFIX + stem)
        old = bile.art_tail(slot)
        if stem in POSE_STEMS or old.type == ART_TYPE_NONE:
            continue  # a pose keeps its mesh tail; a type NONE slot is not loaded by the game
        frames = _anim_frames(stem, plan.wraps[stem], anim_info)
        molynx.set_art_tail(slot, old._replace(
            flags=old.flags & ~(ART_FLAG_HAS_START_ANIMATION | ART_FLAG_HAS_END_ANIMATION),
            frames=frames, fps=GAME_FPS, start_af=0, end_af=0))


def build_molynx(bile: Creature, stage: int, order_in_editor: int,
                 text_ids: dict[str, int] | None = None,
                 anim_info: Mapping[str, dict] | None = None, *,
                 file_wraps: Mapping[str, bool | None] | None = None) -> Creature:
    """A new Molynx record built from `bile` (not modified), at stage 1, 2 or 3 (cumulative; see
    the module docstring). `text_ids` (keys name, tooltip, general, strength, weakness) is used at
    stage 3 only; without it stage 3 keeps the Bile Demon's text IDs. Stages 2 and 3 need
    `anim_info` ({file stem: {frames, fps, wrap}} of the exported files) and `file_wraps` (the
    Bile Demon's file name -> wrap flag, None for a mesh; see file_plan); stage 1 ignores both."""
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2 or 3, not {stage!r}")
    if not 0 <= order_in_editor <= _field_max("orderInEditor"):
        raise ValueError(f"order_in_editor {order_in_editor} does not fit a byte")
    if stage == 3 and text_ids is not None:
        _check_text_ids(text_ids)
    plan = None
    if stage >= 2:
        if anim_info is None:
            raise ValueError(f"stage {stage} needs anim_info (export_molynx_assets(...)['info'])")
        if file_wraps is None:
            raise ValueError(f"stage {stage} needs file_wraps (molynx_assets.bile_file_wraps)")
        plan = file_plan(bile, file_wraps)

    molynx = Creature(bile.raw)  # copies the bytes
    molynx.set("name", NAME)
    molynx.set("creatureId", CREATURE_ID)
    molynx.set("orderInEditor", order_in_editor)
    molynx.set("cloneCreatureId", CLONE_CREATURE_ID)
    if plan is not None:
        _write_slots(molynx, bile, plan, anim_info)
        for name, value in SIZE_FIELDS.items():
            molynx.set(name, value)
        molynx.set_resource_name("icon1Resource", ICON_RESOURCE)
        molynx.set_resource_name("portraitResource", PORTRAIT_RESOURCE)
    if stage >= 3:
        for name, value in {**STATS, **PERSONALITY}.items():
            molynx.set(name, value)
        if text_ids is not None:
            for key, field in TEXT_ID_FIELDS.items():
                molynx.set(field, text_ids[key])
    return molynx
