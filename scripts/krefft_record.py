"""Krefft (creature 54, the next id after the cast): his creature record, texts and panel pictures, the
Soles Creatures Mod way (cast_records.Member, built by cast_records.build from the Bile Demon's record in
each table). DRAFT numbers for Soles to approve (16 Nov 2026); nothing here installs anything.

Who he is. Soles' pet Krefft, a Krefft's river turtle, as the Casino's loan shark: slow, greedy and smug,
a turtle's snapping bite, a mouthful of coins spat at his debtors, a shell that turns blows aside.

The donor is the Bile Demon (3), as for his whole clip set (krefft_anims: every slot the Bile Demon fills
has a clip of the Bile Demon's own length; his Sleep lies in the Bile Demon's lair bed, his Eat carries the
Bile Demon's chicken). From the Bile Demon's record he keeps the regular creature's flags, the bed (lair
object 18), hunger, sleep, the anger strings, the unhappy and angry jobs and his sound category.

What is his own (STATS, PERSONALITY; the Bile Demon's numbers in the comments):

  Jobs        DK2's job table has no Casino job: its JobType runs 0..18 (OpenKeeper's Creature.JobType:
              sleep, eat, research, train, manufacture, guard, torture, pray, drink, leave, destroy rooms /
              walls, steal gold / spells, sulk, rebel, steal crates, kill creatures) and gambling is what a
              keeper's creature does in its own time when it holds more gold than the Casino's stake
              (OpenKeeper CreatureController.goGambling). So his Casino favourite is made with gold: the
              highest wages after Horny's and the Reaper's and a deep purse, so he is always flush and
              always at the tables; drawn to the Portal by a Casino. His one happy job is TORTURE (7, the
              Mistress' own: he goes to the Torture Chamber by choice). He hates training (4) and research
              (3); the Bile Demon's unhappy (sulk, drink) and angry (leave, rebel) jobs are kept.
  Fighting    a decent bite (melee 300, quicker than the Bile Demon's slam); Coin Spit from level 2
              (krefft_coin_spit: creature spell 35 and shot 48, the Salamander's spit with the game's Coin
              mesh, a Gold Sparkle trail and a Coin Spill where it hits), released at the frame his Magic
              clip spits (SPIT_AT) from in front of his lips; his shell resists projectiles (8) and
              physical blows (13) by half (attack categories; see notes_final.md on what is known).
  Body        slow: walk and run matched to his clips (as the Gargoyle's and Sejanus'); his sizes in the
              Bile Demon's proportion to his model.
  Gamble      DK2 has no casino slot. His Research slot (empty: he hates research, so he never researches)
              names Krefft-Gamble: OpenKeeper plays a gambler's Research, Happy and Idle1 clips
              (CreatureViewSystem.GAMBLE_ANIMATIONS), which, if DKII does the same, plays his lever pull at
              the fruit machines with no Flame change. To be checked in game; otherwise Flame needs a casino
              animation hook (notes_final.md).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from scripts.cast_records import BILE_DEMON_BED, BILE_DEMON_ID, Member

KREFFT_ID = 54
ROOM_CASINO = 15                       # Rooms.kwd order (the Rogue's and the Thief's attraction)
ROOM_TREASURY = 1                      # Rooms.kwd order (Treasury, Lair, Portal, Hatchery, ...)
JOB_RESEARCH, JOB_TRAIN, JOB_TORTURE = 3, 4, 7
ATTACK_PROJECTILE, ATTACK_PHYSICAL = 8, 13   # Shots' attack categories (OpenKeeper Shot.AttackCategory; 13 the
                                             # Tremor's and the Boulder's, molynx_ground_slam)
FX = 4096

# His files (krefft_anims_build): slot -> (stem, has Start, has End). DejectedPose is left out, as for the
# cast (no game creature fills that slot); Building is empty (he does not manufacture).
SLOTS = {
    "animIdle1Resource": ("Idle1", False, False), "animIdle2Resource": ("Idle2", False, False),
    "animWalkResource": ("Walk", True, True), "animRunResource": ("Run", False, False),
    "animWalkbackResource": ("Walkback", True, True),
    "animMelee1Resource": ("Melee", False, False), "animMelee2Resource": ("Melee2", False, False),
    "animMagicResource": ("Magic", False, False), "animAngryResource": ("Angry", False, False),
    "animHappyResource": ("Happy", False, False), "animDanceResource": ("Dance", False, False),
    "animEntranceResource": ("Entrance", False, False),
    "animRecoilHffResource": ("RecoilHff", False, False), "animRecoilHfbResource": ("RecoilHfb", False, False),
    "animElectrocuteResource": ("Electrocute", False, False), "animElecResource": ("ElecPose", False, False),
    "animDrunk2Resource": ("Drunk", False, False), "drunkIdle": ("DrunkIdle", False, False),
    "animDrinkResource": ("Drink", False, False), "animEatResource": ("Eat", False, False),
    "animResearchResource": ("Gamble", False, False),        # the casino (see the module notes)
    "animPrayResource": ("Pray", True, True), "animSleepResource": ("Sleep", False, False),
    "animFallbackResource": ("Fallback", False, False), "animGetUpResource": ("GetUp", False, False),
    "animDieResource": ("Die", False, False), "animSwingResource": ("Swing", False, False),
    "animTortureResource": ("Torture", False, False),
    "animPoseFrameResource": ("PoseFrame", False, False), "animDiePoseResource": ("DiePose", False, False),
    "animDraggedPoseResource": ("DraggedPose", False, False),
    "animStunnedPoseResource": ("StunnedPose", False, False),
}
POSES = {"PoseFrame", "DiePose", "DraggedPose", "StunnedPose", "ElecPose"}

# Sizes: the Bile Demon's record (width 1740, height 2662, eyeHeight 1515) for his file (BileDemon-Idle1:
# 0.845 wide, 0.535 tall), scaled to Krefft's PoseFrame (0.595 wide, 0.58 tall), as the Gargoyle's were.
HEIGHT_RATIO = 0.58 / 0.535
WIDTH_RATIO = 0.595 / 0.845
SIZES = {"width": round(1740 * WIDTH_RATIO), "height": round(2662 * HEIGHT_RATIO),
         "eyeHeight": round(1515 * HEIGHT_RATIO)}

# Speeds matched to his clips (the record's speed is a planted foot's speed in units a second x 4096, as
# for Sejanus and the Gargoyle): Walk slides each foot WALK_STRIDE (rig units) back over 62% of a 40-frame
# cycle, Run RUN_STRIDE over half of a 24-frame cycle; the export scales the rig by 0.9254.
WALK_SPEED = round(0.17 / (0.62 * 40 / 30) * 0.9254 * FX)      # 780 (Bile Demon 1638)
RUN_SPEED = round(0.12 / (0.5 * 24 / 30) * 0.9254 * FX)        # 1137 (Bile Demon 2293)

# Coin Spit (krefft_coin_spit): the shot leaves his lips at SPIT_AT (1.1 s, frame 33 of Magic); there his
# lips are 0.22 ahead and 0.587 up in the file. The record's offsets are game units, -y ahead, z up; the
# Bile Demon's record holds 2662 for his file's 0.535 (4976 a unit), the scale used here; the shot starts a
# little ahead of the lips.
COIN_SPIT_SPELL = 35
COIN_SPIT_LEVEL = 2
SPIT_DELAY = round(1.1 * FX)                                     # 4506
UNIT = 2662 / 0.535
SPIT_OFFSET = (0, -round((0.22 + 0.06) * UNIT), round(0.587 * UNIT))   # (0, -1393, 2921)

STATS = {                                  # the Bile Demon's in the comments
    "hp": 3200,                            # 6000: a turtle, not a mountain of fat
    "meleeDamage": 300,                    # 525: a decent bite
    "meleeRecharge": 8192,                 # 14336: a snap every 2 s, quicker than the slam
    "meleeRange": 2457,                    # 3276: the Salamander's (his neck shoots out)
    "speed": WALK_SPEED,                   # 1638: slow, matched to his walk
    "runSpeed": RUN_SPEED,                 # 2293: matched to his hurried waddle
    "fear": 200, "threat": 80,             # 300, 120: follow the damage
    "pay": 900, "maxGoldHeld": 12000,      # 600, 6000: greedy, the dearest after Horny and the Reaper
    "initialGoldHeld": 1500,               # 500: arrives with a purse (and goes gambling with it)
    "angerNoPay": 120,                     # 60: not paid is the worst thing you can do to a loan shark
    "expPerSecondTraining": 10,            # 20: he hates training anyway
    "scoreRating": 75,                     # 90
    "lairObjectId": BILE_DEMON_BED,        # 18 (the Bile Demon's own): his Sleep clip carries that bed
    **SIZES,                               # 1740 / 2662 / 1515
}
PERSONALITY = {
    # Soles, 21 Nov: a Casino and a Treasury of 5x5 at least (a loan shark comes to a keeper with money)
    "attractions[0].present": 1, "attractions[0].roomId": ROOM_CASINO, "attractions[0].roomSize": 25,
    "attractions[1].present": 1, "attractions[1].roomId": ROOM_TREASURY, "attractions[1].roomSize": 25,
    "happyJobs[0].jobType": JOB_TORTURE, "happyJobs[0].chance": 100, "happyJobs[0].moodChange": 0,
    "happyJobs[1].jobType": 0, "happyJobs[1].chance": 0,
    "hateJobs[0]": JOB_TRAIN, "hateJobs[1]": JOB_RESEARCH,
    "spells[0].creatureSpellId": COIN_SPIT_SPELL, "spells[0].levelAvailable": COIN_SPIT_LEVEL,
    "spells[0].playAnimation": 1, "spells[0].shotDelay": SPIT_DELAY, "spells[0].shotOffset": SPIT_OFFSET,
    "spells[1].creatureSpellId": 0, "spells[1].levelAvailable": 0, "spells[1].playAnimation": 0,
    "spells[1].shotDelay": 0, "spells[1].shotOffset": (0, 0, 0),
    "resistances[0].attackType": ATTACK_PROJECTILE, "resistances[0].value": 50,
    "resistances[1].attackType": ATTACK_PHYSICAL, "resistances[1].value": 50,
}

TEXTS = {
    "name": "Krefft",
    "general": "Krefft.  A river turtle up from the deep pools who keeps the Casino's books and lends your "
               "creatures gold at a ruinous rate.  {26}.  Fighting Style: {27}.",
    "strength": "His shell turns arrows and blows aside, his bite snaps shut like a trap, and from level 2 he "
                "spits a mouthful of coins at anyone who owes him.",
    "weakness": "Slow on his stubby legs and greedy: he wants the highest wages, he will not train or study, "
                "and he is happiest at the tables or the Torture Chamber.",
}

KREFFT = Member(
    KREFFT_ID, "Krefft", BILE_DEMON_ID, "Bile Demon", "Krefft-",
    {slot: (stem, s, e, "mesh" if stem in POSES else "anim") for slot, (stem, s, e) in SLOTS.items()}, False,
    stats=STATS, texts=TEXTS, icon="krefft", textures=("Krefft_Body", "Krefft_Head", "Krefft_Shell"),
    extra=PERSONALITY,
    slot_fps={"animSleepResource": 15},      # his Sleep plays at 15 fps, as the Bile Demon's bed(sleeping)
)

# Where he stands on the test maps (cast_install): beside the cast by Keeper 1's heart
PLACEMENTS = {"Dead Walking": (7, 40), "Furnace": (10, 55)}
MPD1_TILE = (27, 35)
