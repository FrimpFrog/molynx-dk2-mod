"""The Necromancer's creature record (creature 48): a modified copy of the Black Knight's record
(Creatures.kwd record 22). NOT applied to any game file yet: Soles will say when to install him.

What changes from the Black Knight (Soles' brief, 10 Nov 2026):
  * name "Necromancer", creatureId 48 (the slot after Molynx), cloneCreatureId 0;
  * every animation slot names his own file (necro_build.file_plan); the Black Knight's slot tails
    are kept (flags, Start / End files, start / end frames) with the frame count taken from his
    file (frames - 1 for a looping file, as Molynx's), except
      - Idle2: his own storm idle (150 frames, plays once), and
      - Research: a slot the Black Knight leaves empty, now his tome clip (120-frame loop);
  * he comes through the portal once there is a Graveyard of 5 x 5 (25 tiles) or more, and nothing
    else draws him (the Black Knight: any Combat Pit; Soles, 11 Nov, replacing the 10 x 10 Temple
    and Combat Pit of 10 Nov);
  * he researches: the Black Knight hates research and has researchPerSecond 0, so research is
    taken off his hated jobs, made a happy job beside training, and given a research speed
    between the Dark Angel's (3) and the Warlock's (5);
  * his own panel pictures (necromancer-bg, necromancer) and texts.
  * his own numbers (STATS, Soles 11 Nov): slightly less melee damage, about 10 % faster, better
    at prayer, researches, trains a bit faster, asks for more pay; 3000 health (Soles 11 Nov; the Black Knight 4000) and the Black Knight's
    stun time;
  * his own walk and run (no Start / End files; the flags are cleared, as Molynx's).
No ranged lightning attack (dropped, Soles 10 Nov); his one spell is Raise Dead from level 8 (SPELLS).
Everything else (health, stun time, melee range and speed, sounds, lair, hunger, slap, ...) stays
the Black Knight's.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from scripts.dk2kwd_creatures import (ART_FLAG_HAS_END_ANIMATION, ART_FLAG_HAS_START_ANIMATION,
                                      ART_TYPE_ANIMATING_MESH, ArtTail, Creature)
from scripts.necro_anims import MAGIC_RELEASE
from scripts.necro_build import OWN_SLOTS, SLOT_STEM, file_plan

NAME = "Necromancer"
CREATURE_ID = 48
BLACK_KNIGHT_ID = 22
GAME_FPS = 30
ROOM_GRAVEYARD = 14                                # Rooms.kwd order (Black Knight: 16 Combat Pit)
JOB_RESEARCH, JOB_TRAIN, JOB_MANUFACTURE = 3, 4, 5
ICON_RESOURCE = "gui\\creatures\\evil\\necromancer"
PORTRAIT_RESOURCE = "gui\\creatures\\evil\\necromancer-bg"

PERSONALITY = {
    # Soles, 11 Nov: comes through the portal once the keeper has a Graveyard of 5 x 5 or more
    # (room sizes count tiles, as Molynx's 5 x 5 Hatchery is 25)
    "attractions[0].present": 1, "attractions[0].roomId": ROOM_GRAVEYARD, "attractions[0].roomSize": 25,
    "attractions[1].present": 0, "attractions[1].roomId": 0, "attractions[1].roomSize": 0,
    "happyJobs[0].jobType": JOB_TRAIN, "happyJobs[0].chance": 60,
    "happyJobs[1].jobType": JOB_RESEARCH, "happyJobs[1].chance": 40,
    "hateJobs[0]": JOB_MANUFACTURE, "hateJobs[1]": 0,       # Black Knight: manufacture and research
}

# Raise Dead from level 8 (Soles, 11 Nov: first asked for "a spell at level 5 that raises skeletons, or if there is
# already a way to make dead bodies around him turn into skeletons, do that"). The game has one:
# creature spell 20, Raise Dead (shot 40), the Vampire's at level 10: dead bodies lying outside a
# Graveyard near him rise as Skeletons that fight for his keeper and crumble after a while. It plays
# his own magic clip (necro_anims.magic_keys) and leaves as his palm drives down at the ground
# (frame MAGIC_RELEASE of 30 fps; shotDelay is in 1/4096 s), at his feet like the Vampire's
# (shotOffset 0). The spell record itself is the game's, unchanged. Level 8, not 5 (Soles, 11 Nov:
# the Vampire and the Dark Angel raise the dead in combat at level 10; he gets it at 8).
SPELL_RAISE_DEAD = 20
RAISE_DEAD_LEVEL = 8
SPELLS = {
    "spells[0].creatureSpellId": SPELL_RAISE_DEAD, "spells[0].levelAvailable": RAISE_DEAD_LEVEL,
    "spells[0].playAnimation": 1, "spells[0].shotDelay": round(MAGIC_RELEASE / GAME_FPS * 4096),
    "spells[0].shotOffset": (0, 0, 0),
}

# His own numbers (Soles, 11 Nov: slightly less damage than the Black Knight, a bit faster, he
# researches, prays and trains a bit better; 3000 health and the Black Knight's stun time). Black Knight
# values in the comments.
STATS = {
    "meleeDamage": 350,          # 400: less damage (Soles and coolkid, 11 Nov: 350)
    "hp": 3000,                  # 4000: less health (Soles, 11 Nov)
    "speed": 2253,               # 2048: about 10 % faster walking
    "runSpeed": 4055,            # 3686: about 10 % faster running
    "shuffleSpeed": 1351,        # 1228: and shuffling
    "manaGenPrayer": 110,        # 90: a little more mana when he prays
    "researchPerSecond": 4,      # 0: he researches (Dark Angel 3, Warlock 5)
    "expPerSecondTraining": 22,  # 20: trains a little faster
    "fear": 480,                 # 533: scaled with his lower damage
    "threat": 192,               # 213: likewise
    "pay": 800,                  # 750: a scholar knight asks more
    "maxGoldHeld": 8000,         # 7500: in step with his pay
}

TEXTS = {
    "name": "Necromancer",
    "general": "Necromancer.  A fallen knight who bargained his soul for the storm, and now commands the dead.  "
               "{26}.  Fighting Style: {27}.",
    "strength": "Heavy armour and a sure sword arm. Unlike other knights he studies the dark arts in the Library, "
                "and from level 8 he raises the fallen around him as Skeletons to fight at his side.",
    "weakness": "Slow on his feet, and he asks for more gold than any other knight. He sees himself above common "
                "labour and refuses to work in the Workshop.",
}


def own_tails(black_knight: Creature, files_info: dict) -> dict:
    """Slot -> ArtTail for the slots that play his own clip: the Black Knight's tail (where he has
    one) without Start / End files, its frames from the Necromancer's file."""
    out = {}
    for slot in OWN_SLOTS:
        info = files_info["Necro-" + SLOT_STEM[slot]]
        frames = info["frames"] - 1 if info["wrap"] else info["frames"]
        old = black_knight.art_tail(slot)
        if old.type == ART_TYPE_ANIMATING_MESH:
            out[slot] = old._replace(flags=old.flags & ~(ART_FLAG_HAS_START_ANIMATION | ART_FLAG_HAS_END_ANIMATION),
                                     frames=frames, fps=GAME_FPS, start_af=0, end_af=0)
        else:
            out[slot] = ArtTail(flags=0, frames=frames, fps=GAME_FPS, start_dist=0, end_dist=0,
                                type=ART_TYPE_ANIMATING_MESH, start_af=0, end_af=0, sometimes_one=0)
    return out


def build_necromancer(black_knight: Creature, order_in_editor: int, files_info: dict,
                      text_ids: dict | None = None) -> Creature:
    """A new Necromancer record from the Black Knight's (which is not changed)."""
    if black_knight.get("creatureId") != BLACK_KNIGHT_ID:
        raise ValueError(f"expected the Black Knight's record (id {BLACK_KNIGHT_ID})")
    n = Creature(black_knight.raw)
    n.set("name", NAME)
    n.set("creatureId", CREATURE_ID)
    n.set("orderInEditor", order_in_editor)
    n.set("cloneCreatureId", 0)
    slots = {}
    for slot, name, src in file_plan():
        slots.setdefault(slot, name)                 # the slot names the main file; Start / End follow it
    for slot, name in slots.items():
        n.set_resource_name(slot, name)
        old = n.art_tail(slot)
        if old.type == ART_TYPE_ANIMATING_MESH:          # frames from his file (the Black Knight's Angry
            info = files_info[name]                      # tail says 46 for a 40-frame file)
            n.set_art_tail(slot, old._replace(frames=info["frames"] - 1 if info["wrap"] else info["frames"],
                                              fps=GAME_FPS))
    for slot, t in own_tails(black_knight, files_info).items():
        n.set_art_tail(slot, t)
    for field, value in {**PERSONALITY, **STATS, **SPELLS}.items():
        n.set(field, value)
    n.set_resource_name("icon1Resource", ICON_RESOURCE)
    n.set_resource_name("portraitResource", PORTRAIT_RESOURCE)
    if text_ids:
        for key, field in (("name", "nameStringId"), ("general", "generalDescriptionStringId"),
                           ("strength", "strengthStringId"), ("weakness", "weaknessStringId")):
            if key in text_ids:
                n.set(field, text_ids[key])
    return n
