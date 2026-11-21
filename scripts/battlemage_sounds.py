"""MegaGlest Battlemage sounds attached to his animations for the viewer. LICENCE: GPL-3.0
(sound files: MegaGlest data, CC BY-SA 3.0). Same pack format as golem_sounds."""
from pathlib import Path

from scripts.golem_sounds import build_sound_pack

GROUPS = {
    "attack": [f"battlemage_attack{i}" for i in range(1, 9)],
    "hit": [f"battlemage_hit{i}" for i in range(1, 5)],
    "die": [f"battlemage_die{i}" for i in range(1, 4)],
    "select": [f"battlemage_select{i}" for i in range(1, 15)],
    "ack": [f"battlemage_ack{i}" for i in range(1, 11)],
}

# MegaGlest: attack sound at 0.3 of the attack, the bolt lands (hit sound) later; die sound at 0. The rest are choices:
# select = a longer phrase, ack = a short one, attack = a spell shout, hit = the impact / a pained grunt.
EVENTS = {
    "attack": [("attack", 0.3, 1.0), ("hit", 0.75, 0.9)],
    "die": [("die", 0.0, 1.0)],
    "casting_spell": [("attack", 0.08, 1.0), ("hit", 0.7, 0.9)],
    "special_attack": [("attack", 0.1, 1.0), ("hit", 0.6, 1.0)],
    "training": [("attack", 0.0, 0.9), ("hit", 0.2, 0.7), ("attack", 0.33, 0.9), ("hit", 0.53, 0.7), ("attack", 0.66, 0.9), ("hit", 0.86, 0.7)],
    "fighting_another_creature": [("attack", 0.02, 1.0), ("hit", 0.14, 0.9), ("hit", 0.32, 0.8), ("attack", 0.5, 1.0), ("hit", 0.62, 0.9),
                                  ("hit", 0.82, 0.8)],
    "getting_hit": [("hit", 0.05, 1.0)],
    "being_slapped": [("hit", 0.08, 1.0)],
    "knockback": [("hit", 0.02, 1.0)],
    "blocking": [("hit", 0.65, 0.8)],
    "dropping": [("ack", 0.3, 0.6)],
    "picking_up": [("ack", 0.4, 0.6)],
    "manufacturing": [("hit", 0.2, 0.4), ("hit", 0.7, 0.4)],
    "researching": [("ack", 0.5, 0.5)],
    "angry": [("select", 0.05, 1.0)],
    "happy": [("ack", 0.2, 1.0)],
    "celebrating": [("ack", 0.1, 1.0), ("ack", 0.6, 1.0)],
    "victory_celebration": [("select", 0.02, 1.0), ("ack", 0.4, 1.0), ("select", 0.7, 1.0)],
    "frightened": [("select", 0.05, 0.8)],
    "unhappy": [("ack", 0.1, 0.7)],
    "looking_around": [("ack", 0.3, 0.6)],
    "waking_up": [("select", 0.1, 0.9)],
    "praying": [("select", 0.25, 0.6)],
    "eating": [("ack", 0.7, 0.6)],
    "being_picked_up": [("ack", 0.0, 0.9)],
    "staggering": [("ack", 0.4, 0.6)],
    "guarding": [("ack", 0.5, 0.5)],
}


def build_pack(sound_dir: Path, animations) -> dict:
    return build_sound_pack(sound_dir, animations, GROUPS, EVENTS)
