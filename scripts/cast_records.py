"""The creature records of the Soles Creatures Mod cast after the Necromancer: the Orc (49), Quuz's
Horned Reaper (50), the Dragon (51), Sejanus (52) and the Gargoyle (53).
DRAFT numbers for Soles to approve (14 Nov 2026); the stats table is `stats_table()`.

Every record is necro_record's way: a copy of a donor's record from the same table (each map's own
Globals, or the editor Creatures.kwd), with only what is needed changed: name, creatureId,
orderInEditor (the table's largest plus one), cloneCreatureId 0, his texts and panel pictures,
the file of every animation slot, sizes where his model differs, and a few numbers (STATS).

Animation slots (`set_slots`). Each slot the donor fills is pointed at the creature's own file for
it. When his file is a re-encode of the donor's own file for that slot (the Orc: every Troll file;
the Reaper: every Horny file), it has the same frame count and loop flag, so the donor's slot tail
(flags, frames, fps, Start / End flags, distances) is kept as it is, and his Start / End files exist
whenever the donor's flags ask for them. When his files are his own (the Dragon), the tail is the
donor's with the frame count taken from his file (frames - 1 for a looping file, as Molynx's and the
Necromancer's), fps 30, and the Start / End flags set exactly when he has those files; a slot the
donor leaves empty takes the Salamander's tail or a plain one (TAIL_FROM), and a slot he has no file
for is emptied. No slot is left pointing at a file the mod does not ship or the game does not have:
`slot_files` lists them all and the checker (cast_check) re-reads every table for it.

The donors:
  Orc       the Troll (10): DK1's Orc is a recoloured Troll; a fighter who trains, so a little more
            melee damage, health and training speed, the Troll's job preferences and attraction.
  Reaper    Horny (27): his own creature, Horny untouched; Horny's numbers a little lower (he must
            not replace Horny as the top creature). A regular creature (Soles, 15 Nov; the first
            install kept Horny's flags and he came out a summoned Horny): the regulars' flags without
            IS_HORNY (HREAPER_FLAGS), the Hand can pick him up, he is paid (the dearest) and takes the
            Black Knight's job preferences. He never comes through the Portal (Soles, 15 Nov: he will be
            a reward; he exists only where he is placed): no AVAILABLE_VIA_PORTAL and no attraction, as the
            Skeleton and the Vampire, and on no map's creature pool or availability (placed_only). Like
            Horny he never eats or sleeps (Horny's hunger and sleep times). Horny's 12 slots and their
            clips re-encoded on his body (hreaper_files), and the slots every portal creature fills
            that Horny leaves empty given the nearest of those files (HREAPER_FILL).
  Dragon    the Bile Demon (3): his files are made against the Bile Demon's (his Sleep lies in the
            Bile Demon's lair bed, his Eat carries the Bile Demon's chicken, his Torture and Pray are
            placed as the Bile Demon's), and he is DK1's slow heavy. His fire is the Salamander's (11):
            her two spells (Salamander Spit from level 4, Salamander Fireball from 8) and her
            resistances, timed to his own fire-breath clip. He comes through the Portal for a Lair
            and a Hatchery (DK1), 25 tiles each (draft).
  Sejanus   the Goblin (7): a plain melee fighter who trains (the Dark Elf is a ranged fighter, the
            Vampire undead with a coffin and research). The Hunter: walk and run speeds matched to
            his clips (the run above every game creature's), mid-tier health and damage between the
            Dark Elf's and the Vampire's, a longer sight range. No game creature can see invisible
            ones (the manual's claim for the Guard does not hold in the game; no flag or creature
            spell does it), so that part of his brief needs a Flame patch. His spell is the Monk's
            Haste Creature from level 4, cast with his Reveal clip. Drawn by a Training Room and a
            Combat Pit, 25 tiles each (draft).
  Gargoyle  the Goblin (7): a small cheap fighter; a little tougher (stone), a little lighter
            claws, cheaper pay, walk and run matched to his clips. His files carry the Goblin's own
            bed, chicken and bottle, so the Goblin's lair object stays. No spell (his Magic clip is a
            stone screech). Drawn by a Lair of any size and a 3 x 3 Training Room (draft).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scripts.dk2kwd_creatures import (ANIM_SLOTS, ART_FLAG_HAS_END_ANIMATION, ART_FLAG_HAS_START_ANIMATION,
                                      ART_TYPE_ANIMATING_MESH, ART_TYPE_MESH, FIELDS, ArtTail, Creature, anim_slot)

GAME_FPS = 30
SE = ART_FLAG_HAS_START_ANIMATION | ART_FLAG_HAS_END_ANIMATION
ROOM_LAIR, ROOM_HATCHERY = 2, 4                       # Rooms.kwd order
TROLL_ID, HORNY_ID, BILE_DEMON_ID, SALAMANDER_ID, GOBLIN_ID = 10, 27, 3, 11, 7
ROOM_TRAINING, ROOM_COMBAT_PIT = 7, 16
GOBLIN_BED = 22                                       # the Goblin's lair object
BILE_DEMON_BED = 18                                   # the Bile Demon's nest (the Dragon and the Gargoyle sleep in it)
SHARED_TOOLTIP_ID = 2408                              # the game's shared creature tooltip (as Molynx)

# Ids reserved for cast members whose records are not written yet (the id stays free so the creature
# table stays dense until each is ready).
RESERVED: dict = {}                        # every reserved id is now a cast member


@dataclass(frozen=True)
class Member:
    creature_id: int
    name: str                       # the record's name (char[32])
    donor_id: int
    donor_name: str
    prefix: str                     # his files: <prefix><stem>.kmf
    slots: dict                     # slot -> (file stem without prefix, has Start, has End, kind)
    keep_donor_tails: bool          # True: his files re-encode the donor's own, frame for frame
    stats: dict
    texts: dict
    icon: str                       # GUI\Creatures\Evil\<icon>, <icon>-bg
    textures: tuple                 # his skins (EngineTextures/<name>.png)
    extra: dict = field(default_factory=dict)   # fields set as they are (spells, attraction, sizes)
    spells_from: int | None = None  # creature whose spells and resistances he takes (by id)
    shot_delay: int | None = None   # his spells' release time (1/4096 s) in his own Magic clip
    shot_offset: tuple | None = None
    jobs_from: int | None = None    # creature whose job preferences (happy/unhappy/angry/hate/alternative) he takes
    fill_slots: dict = field(default_factory=dict)   # slot -> (file stem, "anim" / "mesh" / "copy:<slot>"):
                                    # slots his donor leaves empty, filled with one of his existing files
    placed_only: bool = False       # never through the Portal: on no creature pool or availability
    slot_fps: dict = field(default_factory=dict)     # slot -> fps where his file plays slower than 30 (Krefft's
                                    # Sleep at the Bile Demon's 15)


# ------------------------------------------------------------------------------ the Orc (49)

def _orc_slots() -> dict:
    from scripts.orc_build import TROLL_FILES, TROLL_START_END
    out = {}
    for short in TROLL_FILES:
        slot = anim_slot(short)
        out[slot] = (short, short + "Start" in TROLL_START_END, short + "End" in TROLL_START_END,
                     "mesh" if short in ("draggedPose", "stunnedPose", "elec", "poseFrame", "diePose") else "anim")
    return out


ORC = Member(
    49, "Orc", TROLL_ID, "Troll", "Orc-", _orc_slots(), True,
    stats={                                  # Troll's in the comments
        "hp": 1100,                          # 1000: a little tougher
        "meleeDamage": 160,                  # 130: a fighter first (DK1's Orc)
        "expPerSecondTraining": 36,          # 30: loves the Training Room
        "manufacturePerSecond": 50,          # 60: a fighter, not a craftsman
        "fear": 60, "threat": 25,            # 50, 20: follow the damage
        "pay": 550, "maxGoldHeld": 5500,     # 500, 5000
        "scoreRating": 70,                   # 65
    },
    texts={
        "name": "Orc",
        "general": "Orc.  A brute of the old wars who lives for the fight and the mallet.  {26}.  Fighting Style: {27}.",
        "strength": "Hits harder than a Troll and trains with a will. He is nearly as handy in the Workshop as his cousin.",
        "weakness": "Slow-witted: he will not set foot in the Library, and he is no match for the bigger creatures.",
    },
    icon="orc", textures=("Orc_Front", "Orc_Back", "Orc_Gear"),
)


# ------------------------------------------------------------------------------ Quuz's Horned Reaper (50)

def _reaper_slots() -> dict:
    from scripts.hreaper_files import HORNY_SLOTS, PREFIX, reaper_stem
    return {slot: (reaper_stem(stem)[len(PREFIX):], s, e, "mesh" if stem.lower() == "rea3-poseframe" else "anim")
            for slot, (stem, s, e) in HORNY_SLOTS.items()}


# Regular creature (Soles, 15 Nov: "make the quuz horny" a regular creature; in game record 50 came out as a
# summoned Horny). The creature flags are OpenKeeper's names (tools/convert/map/Creature.java). Horny's
# 0xA00652B4 has IS_HORNY (0x4000, his alone of the game's 46: the summoned Horned Reaper, slapped away,
# no portal, no Hand) and lacks CAN_BE_PICKED_UP, AVAILABLE_VIA_PORTAL, CAN_BE_HYPNOTIZED,
# CAN_BE_ELECTROCUTED and FREES_FRIENDS_ON_JAILBREAK, which every regular keeper creature has. His flags
# are the regulars' own word (0xA2098C96: the Bile Demon, Warlock, Dark Elf, Goblin, Black Knight, Molynx
# and the Necromancer all have exactly it) with an elite's: CAN_WALK_ON_LAVA (Horny), IS_IMMUNE_TO_TURNCOAT
# (Horny), IS_IMMUNE_TO_CHICKEN (Horny), IS_FEARLESS (Horny, the Vampire), and without GENERATE_DEAD_BODY
# (Horny, the Vampire and the Skeleton leave none; he has no clip to lie dead with).
FLAG_CAN_WALK_ON_LAVA, FLAG_IMMUNE_TURNCOAT, FLAG_GENERATE_DEAD_BODY = 0x20, 0x200, 0x8000
FLAG_IMMUNE_CHICKEN, FLAG_FEARLESS, FLAG_IS_HORNY = 0x20000, 0x40000, 0x4000
REGULAR_FLAGS = 0xA2098C96
# Not through the Portal (Soles, 15 Nov): AVAILABLE_VIA_PORTAL off, as the game's own regular creatures
# that never come through it, the Skeleton (0xA20D0896: the regulars' less 0x8400) and the Vampire
# (0xB2BD0886: less 0x8410): both stay picked up, slapped, paid and trained. The game's maps never put
# them (or Horny) in a creature pool or availability, and neither does cast_install for him.
FLAG_AVAILABLE_VIA_PORTAL = 0x400
HREAPER_FLAGS = (REGULAR_FLAGS | FLAG_CAN_WALK_ON_LAVA | FLAG_IMMUNE_TURNCOAT | FLAG_IMMUNE_CHICKEN
                 | FLAG_FEARLESS) & ~(FLAG_GENERATE_DEAD_BODY | FLAG_AVAILABLE_VIA_PORTAL)
BLACK_KNIGHT_ID, ROOM_TEMPLE = 22, 13

# The slots every portal creature of the game fills and Horny leaves empty (the game's 12 regular portal
# creatures: none leaves any of these empty), each given the nearest of his re-encoded Horny clips.
# Left empty as some portal creatures leave them: Drink (Firefly), Drunk / DrunkIdle (Bile Demon,
# Firefly, Maiden), Walkback (Maiden), Building, Electrocute, Melee2 (most).
HREAPER_FILL = {
    "animRunResource": ("Walk", "copy:animWalkResource"),     # he walks at his run speed, as Horny does
    "animRecoilHffResource": ("Melee3End", "anim"),           # the scythe swung back: a flinch
    "animRecoilHfbResource": ("Melee3End", "anim"),
    "animFallbackResource": ("Melee3End", "anim"),
    "animGetUpResource": ("WalkStart", "anim"),
    "animDieResource": ("Roar", "anim"),                      # a last roar; no body (GENERATE_DEAD_BODY off)
    "animDiePoseResource": ("PoseFrame", "mesh"),
    "animStunnedPoseResource": ("PoseFrame", "mesh"),
    "animHappyResource": ("Dance", "anim"),
    "animAngryResource": ("Roar", "anim"),
    "animEntranceResource": ("Roar", "anim"),                 # out of the Portal with a roar
    "animIdle2Resource": ("Footstamp", "anim"),
    "animSwingResource": ("Roar", "anim"),                    # in the Hand: no hanging clip yet (feet at the hand)
    "animSleepResource": ("Roar", "anim"),                    # never played: he never sleeps (Horny's timeSleep 0)
    "animTortureResource": ("StampOnChicken", "anim"),
}

HREAPER = Member(
    50, "Quuz Horned Reaper", HORNY_ID, "Horny", "HReaper-", _reaper_slots(), True,
    stats={                                  # Horny's in the comments
        "hp": 5400,                          # 6000
        "meleeDamage": 630,                  # 700
        "speed": 3000, "runSpeed": 3000,     # 3276, 3276
        "shuffleSpeed": 3000,                # 3276
        "fear": 1260, "threat": 504,         # 1400, 560
        "scoreRating": 120,                  # 130
        "tooltipStringId": SHARED_TOOLTIP_ID,   # Horny's own tooltip text is "Horny"
        # a regular creature (draft for Soles)
        "flags": HREAPER_FLAGS,              # 0xA00652B4 (see above)
        "pay": 1000, "maxGoldHeld": 10000,   # 0, 0: the dearest creature (Black Knight 750, King 1000)
        "angerNoPay": 60, "angerNoWork": 50, # 1, 1: the regulars' (the Black Knight's)
        "angerSlap": 3000, "angerInHand": -250,   # 8000, 1000: the Black Knight's
        "entranceEffectId": 184,             # 0: the Black Knight's portal entrance effect (unused now)
    },
    texts={
        "name": "Quuz's Horned Reaper",
        "general": "Quuz's Horned Reaper.  The Reaper of an older age, risen again with a silver scythe.  "
                   "{26}.  Fighting Style: {27}.",
        "strength": "A storm of horns and steel who never tires: he needs no food and no sleep.",
        "weakness": "Not quite Horny's equal, and the most expensive creature in your dungeon.",
    },
    icon="hreaper", textures=("HReaper_Legs", "HReaper_Torso", "HReaper_Back", "HReaper_Scythe", "HReaper_Face",
                              "HReaper_Gear"),
    # no attraction (Horny's: none), and no Portal flag
    jobs_from=BLACK_KNIGHT_ID,               # trains (95), guards (5); hates the Workshop and the Library
    fill_slots=HREAPER_FILL,
    placed_only=True,
)


# ------------------------------------------------------------------------------ the Dragon (51)

def _dragon_slots() -> dict:
    from scripts.dragon_build import POSES, SLOTS
    out = {}
    for slot, (stem, s, e) in SLOTS.items():
        if slot == "animDejectedPoseResource":
            continue                         # no game creature fills this slot: left empty, as theirs
        out[slot] = (stem, s, e, "mesh" if stem in POSES else "anim")
    return out


# A slot the Bile Demon leaves empty (Run, Research, Drunk) takes its tail from the Salamander's when
# she fills it, else a plain tail (no flags; frames and fps from his file).
TAIL_FROM = {"animResearchResource": None}
DRAGON_SHOT_DELAY = 2540                     # 0.62 s: the flame leaves his jaws (dragon_anims.breathe_fire)
DRAGON_SHOT_OFFSET = (0, -1720, 820)         # in front of his snout at that frame (KMF y -0.335, 0.18 up)

DRAGON = Member(
    51, "Dragon", BILE_DEMON_ID, "Bile Demon", "Dragon-", _dragon_slots(), False,
    stats={                                  # Bile Demon's in the comments
        "hp": 5500,                          # 6000: DK1's Dragon is a little less sturdy than the Bile Demon
        "meleeDamage": 500,                  # 525: his fire does the rest
        "meleeRecharge": 12288,              # 14336: a quicker bite (3 s)
        "fear": 340, "threat": 136,          # 300, 120
        "pay": 700, "maxGoldHeld": 7000,     # 600, 6000
        "scoreRating": 100,                  # 90
        "width": 1600, "height": 3640, "eyeHeight": 2660,   # 1740, 2662, 1515: his model is 0.74 tall
    },
    texts={
        "name": "Dragon",
        "general": "Dragon.  An ancient fire-breather, drawn up from the deep by a warm Lair and a full Hatchery.  "
                   "{26}.  Fighting Style: {27}.",
        "strength": "Thick scales and a savage bite. He breathes fire on his foes, and from level 8 hurls balls of flame.",
        "weakness": "Slow and heavy on his feet, and he wants a big Lair and a well-stocked Hatchery before he will come.",
    },
    icon="dragon", textures=("Dragon_Body", "Dragon_Head", "Dragon_Wing"),
    extra={
        "attractions[0].present": 1, "attractions[0].roomId": ROOM_LAIR, "attractions[0].roomSize": 25,
        "attractions[1].present": 1, "attractions[1].roomId": ROOM_HATCHERY, "attractions[1].roomSize": 25,
    },
    spells_from=SALAMANDER_ID, shot_delay=DRAGON_SHOT_DELAY, shot_offset=DRAGON_SHOT_OFFSET,
)

# ------------------------------------------------------------------------------ Sejanus (52)

def _sejanus_slots() -> dict:
    from scripts.sejanus_build import POSES, SLOTS
    return {slot: (stem, s, e, "mesh" if stem in POSES else "anim") for slot, (stem, s, e) in SLOTS.items()
            if slot != "animDejectedPoseResource"}       # no game creature fills it (as the Dragon's)


# His clips' ground speeds (the exporter, at 30 fps): walk 0.654, run 2.049 model units a second. The
# record's speeds are units a second x 4096 at scale 1.0 (the game's own: the Dark Elf's walk clip
# moves its planted feet 0.701 units/s for walk speed 2867 = 0.700; the Mistress's 0.739 for 0.700).
SEJANUS_WALK = round(0.654 * 4096)          # 2679
SEJANUS_RUN = round(2.049 * 4096)           # 8393: above every game creature's (Jack 7782, Imp 7536, Dark Angel 6594)
SPELL_HASTE = 13                            # Haste Creature (the Monk's, from level 4); no creature spell reveals the invisible
SEJANUS_CAST = round(0.45 * 4096)           # his Magic clip (sejanus_more.magic, "the Hunter's Reveal") peaks at 0.45 s

SEJANUS = Member(
    52, "Sejanus", GOBLIN_ID, "Goblin", "Sejanus-", _sejanus_slots(), False,
    stats={                                  # Goblin's in the comments (Dark Elf 1000 / 140, Vampire 2500 / 666)
        "hp": 1800,                          # 1500
        "meleeDamage": 320,                  # 170: a mid-tier fighter, between the Dark Elf and the Vampire
        "speed": SEJANUS_WALK,               # 2867: matched to his walk clip
        "runSpeed": SEJANUS_RUN,             # 5160: matched to his run clip, the fastest runner
        "fear": 110, "threat": 70,           # 50, 37: follow the damage
        "pay": 600, "maxGoldHeld": 6000,     # 500, 5000
        "scoreRating": 80,                   # 60
        "distanceCanSee": 20480,             # 16384 (every game creature): a hunter's eyes, 5 tiles further
        "lairObjectId": 18,                  # 22: his Sleep clip lies in the Bile Demon's bed (as the Dragon's)
        "width": 1384, "height": 3729, "eyeHeight": 2663,   # 651, 2867, 2048: in the Goblin's proportion to his model
    },
    texts={
        "name": "Sejanus",
        "general": "Sejanus.  The Hunter, a horse demon who runs down anything foolish enough to flee.  "
                   "{26}.  Fighting Style: {27}.",
        "strength": "The fastest runner in the dungeon, with eyes that miss nothing. From level 4 he can drive "
                    "himself and his packmates to an even greater pace.",
        "weakness": "Lean and light: he cannot take a heavy fighter's punishment, and he scorns the Library and "
                    "the Workshop.",
    },
    icon="sejanus", textures=("Sejanus_Body", "Sejanus_Head", "Sejanus_Limbs"),
    extra={
        "attractions[0].present": 1, "attractions[0].roomId": ROOM_TRAINING, "attractions[0].roomSize": 25,
        "attractions[1].present": 1, "attractions[1].roomId": ROOM_COMBAT_PIT, "attractions[1].roomSize": 25,
        "spells[0].creatureSpellId": SPELL_HASTE, "spells[0].levelAvailable": 4, "spells[0].playAnimation": 1,
        "spells[0].shotDelay": SEJANUS_CAST, "spells[0].shotOffset": (0, 0, 0),
    },
)

# ------------------------------------------------------------------------------ the Gargoyle (53)

def _gargoyle_slots() -> dict:
    from scripts.gargoyle_build import POSES, SLOTS
    return {slot: (stem, s, e, "mesh" if stem in POSES else "anim") for slot, (stem, s, e) in SLOTS.items()
            if slot != "animDejectedPoseResource"}       # no game creature fills it (as the Dragon's)


# His clips' ground speeds, measured from his files (a planted foot's steady backward speed, the
# measure that gives Sejanus' 0.654 / 2.05): walk 0.323, run 0.733 units a second -> x 4096 as for Sejanus.
# The Goblin's own walk file moves its feet 0.405 units/s for his speed 2867, so the game does not hold
# every creature to its clip; these are the clip-matched values Soles asked for (see the stats table).
GARGOYLE_WALK = round(0.323 * 4096)         # 1323
GARGOYLE_RUN = round(0.733 * 4096)          # 3002

GARGOYLE = Member(
    53, "Gargoyle", GOBLIN_ID, "Goblin", "Gargoyle-", _gargoyle_slots(), False,
    stats={                                  # Goblin's in the comments
        "hp": 1650,                          # 1500: stone, a little tougher
        "meleeDamage": 160,                  # 170: a little lighter claws
        "speed": GARGOYLE_WALK,              # 2867: matched to his walk clip
        "runSpeed": GARGOYLE_RUN,            # 5160: matched to his run clip
        "fear": 45, "threat": 33,            # 50, 37: follow the damage
        "pay": 400, "maxGoldHeld": 4000,     # 500, 5000: cheap
        "scoreRating": 55,                   # 60
        "lairObjectId": BILE_DEMON_BED,      # Soles, 15 Nov: the Bile Demon's nest, as the Dragon (was the
                                             # Goblin's bed, 22); his Sleep clip carries the nest
        "width": 827, "height": 2167, "eyeHeight": 1548,   # 651, 2867, 2048: in the Goblin's proportion
    },                                                       # to his model (head top 0.43, the Goblin's 0.569)
    texts={
        "name": "Gargoyle",
        "general": "Gargoyle.  A small stone demon on leathery wings, carved to keep watch and woken for war.  "
                   "{26}.  Fighting Style: {27}.",
        "strength": "Cheap to keep and hard as rock: his stone hide shrugs off blows a Goblin would feel.",
        "weakness": "Small and slow on his stubby legs, he is a foot soldier and no more. He will not study "
                    "or craft.",
    },
    icon="gargoyle", textures=("Gargoyle_Body", "Gargoyle_Head", "Gargoyle_Wing"),
    extra={
        # cheap and early: a Lair of any size and a small (3 x 3) Training Room (draft)
        "attractions[0].present": 1, "attractions[0].roomId": 2, "attractions[0].roomSize": 0,
        "attractions[1].present": 1, "attractions[1].roomId": ROOM_TRAINING, "attractions[1].roomSize": 9,
    },
)

CAST = (ORC, HREAPER, DRAGON, SEJANUS, GARGOYLE)
BY_ID = {m.creature_id: m for m in CAST}


# ------------------------------------------------------------------------------ building

def slot_files(member: Member) -> dict[str, list[str]]:
    """Slot -> every file it makes the game load (main, then Start / End)."""
    out = {}
    for slot, (stem, s, e, _kind) in member.slots.items():
        name = member.prefix + stem
        out[slot] = [name] + ([name + "Start"] if s else []) + ([name + "End"] if e else [])
    for slot, (stem, kind) in member.fill_slots.items():
        if kind.startswith("copy:"):
            out[slot] = list(out[kind[5:]])
        else:
            out[slot] = [member.prefix + stem]
    return out


def all_files(member: Member) -> list[str]:
    """Every .kmf his record names (no repeats), in slot order."""
    seen, out = set(), []
    for names in slot_files(member).values():
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _frames(info: dict) -> int:
    return info["frames"] - 1 if info.get("wrap") else info["frames"]


def set_slots(n: Creature, member: Member, donor: Creature, files_info: dict, table: dict[int, Creature]) -> None:
    """Point every animation slot at his files (see the module docstring)."""
    for slot in ANIM_SLOTS[1:]:                      # unk_1 holds no name; never touched
        old_tail = donor.art_tail(slot)
        if slot not in member.slots:
            if donor.resource_name(slot) or old_tail.type:
                n.set_resource_name(slot, "")
                n.set_art_tail(slot, ArtTail(0, 0, 0, 0, 0, 0, 0, 0, 0))
            continue
        stem, has_start, has_end, kind = member.slots[slot]
        name = member.prefix + stem
        n.set_resource_name(slot, name)
        if member.keep_donor_tails:
            continue
        info = files_info[name]
        want_type = ART_TYPE_MESH if kind == "mesh" else ART_TYPE_ANIMATING_MESH
        if old_tail.type != want_type:
            src = TAIL_FROM.get(slot, SALAMANDER_ID)
            old_tail = table[src].art_tail(slot) if src is not None else ArtTail(0, 0, 0, 0, 0, want_type, 0, 0, 0)
            if old_tail.type != want_type:
                old_tail = ArtTail(0, 0, 0, 0, 0, want_type, 0, 0, 0)
        if kind == "mesh":
            n.set_art_tail(slot, old_tail)          # a pose: the donor's mesh tail (scale, one frame)
            continue
        flags = (old_tail.flags & ~SE) | (ART_FLAG_HAS_START_ANIMATION if has_start else 0) \
            | (ART_FLAG_HAS_END_ANIMATION if has_end else 0)
        # startAf / endAf are the frame counts of the Start / End files (the game's own records:
        # Bile_Demon-WalkStart has 40 frames and startAf 40; 356 of the 379 Start / End files the
        # game's editor table names agree). The distances (startDist, endDist) stay the donor's, as Molynx's and the Necromancer's walks
        # kept theirs (tested in the game): the game's values follow no rule the files show.
        start_af = files_info[name + "Start"]["frames"] if has_start else 0
        end_af = files_info[name + "End"]["frames"] if has_end else 0
        n.set_art_tail(slot, old_tail._replace(flags=flags, frames=_frames(info), fps=member.slot_fps.get(slot, GAME_FPS),
                                               start_af=start_af, end_af=end_af))


JOB_FIELDS = ("happyJobs", "unhappyJobs", "angryJobs", "hateJobs", "alternativeJobs")


def fill_slots(n: Creature, member: Member, files_info: dict) -> None:
    """The slots of member.fill_slots: his file, with a plain tail (frames and fps from the file, as the
    game's own; a pose keeps his PoseFrame's mesh tail) or the tail of the slot it copies."""
    for slot, (stem, kind) in member.fill_slots.items():
        if n.resource_name(slot):
            raise ValueError(f"{member.name}: {slot} is not empty")
        if kind.startswith("copy:"):
            n.set_resource_name(slot, n.resource_name(kind[5:]))
            n.set_art_tail(slot, n.art_tail(kind[5:]))
            continue
        name = member.prefix + stem
        info = files_info[name]
        if kind == "mesh":
            if info["kind"] != "mesh":
                raise ValueError(f"{name} is not a pose")
            n.set_resource_name(slot, name)
            n.set_art_tail(slot, n.art_tail("animPoseFrameResource"))
        else:
            if info["kind"] != "anim":
                raise ValueError(f"{name} is not an animation")
            n.set_resource_name(slot, name)
            n.set_art_tail(slot, ArtTail(1, _frames(info), GAME_FPS, 0, 0, ART_TYPE_ANIMATING_MESH, 0, 0, 0))


def build(member: Member, table: list[Creature], order_in_editor: int, files_info: dict,
          text_ids: dict | None = None) -> Creature:
    """His record from the donor in `table` (the records of one Globals or Creatures.kwd), which is
    not changed. ValueError when the table has no single donor of the expected name."""
    by_id = {c.get("creatureId"): c for c in table}
    donors = [c for c in table if c.get("creatureId") == member.donor_id]
    if len(donors) != 1 or donors[0].get("name") != member.donor_name:
        raise ValueError(f"expected one {member.donor_name!r} with creatureId {member.donor_id}")
    donor = donors[0]
    n = Creature(donor.raw)
    n.set("name", member.name)
    n.set("creatureId", member.creature_id)
    n.set("orderInEditor", order_in_editor)
    n.set("cloneCreatureId", 0)
    set_slots(n, member, donor, files_info, by_id)
    fill_slots(n, member, files_info)
    if member.jobs_from is not None:
        src = by_id[member.jobs_from]
        for f in [k for k in FIELDS if k.split("[")[0] in JOB_FIELDS]:
            n.set(f, src.get(f))
    if member.spells_from is not None:
        src = by_id[member.spells_from]
        for i in range(3):
            for f in ("creatureSpellId", "levelAvailable", "playAnimation", "shotDelay", "shotOffset"):
                n.set(f"spells[{i}].{f}", src.get(f"spells[{i}].{f}"))
            if n.get(f"spells[{i}].creatureSpellId"):
                n.set(f"spells[{i}].shotDelay", member.shot_delay)
                n.set(f"spells[{i}].shotOffset", member.shot_offset)
        for i in range(4):
            n.set(f"resistances[{i}].attackType", src.get(f"resistances[{i}].attackType"))
            n.set(f"resistances[{i}].value", src.get(f"resistances[{i}].value"))
    for f, v in {**member.extra, **member.stats}.items():
        n.set(f, v)
    n.set_resource_name("icon1Resource", f"gui\\creatures\\evil\\{member.icon}")
    n.set_resource_name("portraitResource", f"gui\\creatures\\evil\\{member.icon}-bg")
    if text_ids:
        for key, f in (("name", "nameStringId"), ("general", "generalDescriptionStringId"),
                       ("strength", "strengthStringId"), ("weakness", "weaknessStringId")):
            n.set(f, text_ids[key])
        if "tooltipStringId" not in member.stats:
            n.set("tooltipStringId", SHARED_TOOLTIP_ID)
    return n


# ------------------------------------------------------------------------------ the table for Soles

STAT_ROWS = (("hp", "Health"), ("meleeDamage", "Melee damage"), ("meleeRecharge", "Time between hits (1/4096 s)"),
             ("speed", "Walk speed"), ("runSpeed", "Run speed"), ("fear", "Fear"), ("threat", "Threat"),
             ("pay", "Pay"), ("expPerSecondTraining", "Training speed"), ("researchPerSecond", "Research"),
             ("manufacturePerSecond", "Workshop"), ("distanceCanSee", "Sight range"), ("scoreRating", "Score rating"))


ROOM_NAMES = {2: "Lair", 4: "Hatchery", 6: "Library", 7: "Training Room", 10: "Workshop", 13: "Temple",
              14: "Graveyard", 16: "Combat Pit"}                       # Rooms.kwd order
SPELL_NAMES = {5: "Gas Cloud", 6: "Gas Missile", 13: "Haste Creature", 26: "Salamander Fireball", 38: "Reaper Fireball",
               39: "Salamander Spit"}                                   # CreatureSpells.kwd


def stats_table(table: list[Creature]) -> str:
    """Markdown: each cast member next to his donor, from built records in `table` (e.g. the
    installed editor Creatures.kwd)."""
    by_id = {c.get("creatureId"): c for c in table}
    lines = []
    for m in CAST:
        me, d = by_id[m.creature_id], by_id[m.donor_id]
        lines += [f"**{m.texts['name']} ({m.creature_id}) vs {m.donor_name} ({m.donor_id})**", "",
                  f"| | {m.donor_name} | {m.texts['name']} |", "|---|---:|---:|"]
        for f, label in STAT_ROWS:
            a, b = d.get(f), me.get(f)
            lines.append(f"| {label} | {a} | {b}{' *' if a != b else ''} |")
        sp = lambda c: ", ".join(f"{SPELL_NAMES.get(c.get(f'spells[{i}].creatureSpellId'))} (level "
                                 f"{c.get(f'spells[{i}].levelAvailable')})"
                                 for i in range(3) if c.get(f"spells[{i}].creatureSpellId")) or "none"
        def at(c):
            parts = []
            for i in range(2):
                if c.get(f"attractions[{i}].present"):
                    room, size = c.get(f"attractions[{i}].roomId"), c.get(f"attractions[{i}].roomSize")
                    parts.append(f"{ROOM_NAMES.get(room, room)} ({f'{size} tiles' if size else 'any size'})")
            return ", ".join(parts) or "none (not through the Portal)"
        lines.append(f"| Spells | {sp(d)} | {sp(me)} |")
        if m is HREAPER:
            for label, f in (("Gold held at most", "maxGoldHeld"), ("Anger: no pay", "angerNoPay"),
                             ("Anger: no work", "angerNoWork"), ("Anger: slapped", "angerSlap"),
                             ("Anger: in the Hand", "angerInHand"), ("Hunger rate", "hungerRate"),
                             ("Time awake / asleep", None), ("Flags", "flags")):
                if f is None:
                    lines.append(f"| {label} | {d.get('timeAwake')} / {d.get('timeSleep')} | "
                                 f"{me.get('timeAwake')} / {me.get('timeSleep')} |")
                elif f == "flags":
                    lines.append(f"| {label} | 0x{d.get(f):08X} | 0x{me.get(f):08X} * |")
                else:
                    star = " *" if d.get(f) != me.get(f) else ""
                    lines.append(f"| {label} | {d.get(f)} | {me.get(f)}{star} |")
        lines.append(f"| Attraction | {at(d)} | {at(me)} |")
        lines.append("")
    return "\n".join(lines)
