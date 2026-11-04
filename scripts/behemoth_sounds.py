"""MegaGlest Behemoth sounds attached to his animations for the viewer. LICENCE: GPL-3.0
(sound files: MegaGlest data, CC BY-SA 3.0). Same pack format as golem_sounds. The Behemoth has no hit-sound files."""
from pathlib import Path

from scripts.golem_sounds import build_sound_pack

GROUPS = {
    "walk": ["behemoth_walk1", "behemoth_walk2"],
    "attack": [f"behemoth_attack1{c}" for c in "abcdefgh"],
    "die": [f"behemoth_die{i}" for i in range(1, 5)],
    "select": [f"behemoth_select{i}" for i in range(1, 5)],
    "ack": [f"behemoth_ack{i}" for i in range(1, 7)],
}

# MegaGlest behemoth.xml: walk sound at 0.4 of the walk cycle, attack sound at 0, die sound at 0. The rest are choices:
# select = a longer roar, ack = a short grunt, walk = a footfall, attack = a swing roar.
EVENTS = {
    "walk": [("walk", 0.4, 1.0), ("walk", 0.9, 1.0)],
    "attack": [("attack", 0.0, 1.0)],
    "die": [("die", 0.0, 1.0)],
    "running": [("walk", 0.0, 1.0), ("walk", 0.5, 1.0)],
    "patrolling": [("walk", 0.0, 0.8), ("walk", 0.5, 0.8)],
    "staggering": [("walk", 0.2, 0.7), ("walk", 0.7, 0.7)],
    "training": [("attack", 0.0, 0.9), ("attack", 0.33, 0.9), ("attack", 0.66, 0.9)],
    "fighting_another_creature": [("attack", 0.02, 1.0), ("attack", 0.5, 1.0)],
    "getting_hit": [("ack", 0.05, 1.0)],
    "being_slapped": [("ack", 0.08, 1.0)],
    "knockback": [("ack", 0.02, 1.0), ("walk", 0.35, 0.8)],
    "manufacturing": [("walk", 0.2, 0.4), ("walk", 0.7, 0.4)],
    "interacting_with_objects": [("walk", 0.25, 0.4), ("walk", 0.75, 0.4)],
    "angry": [("select", 0.05, 1.0)],
    "happy": [("ack", 0.2, 1.0)],
    "celebrating": [("ack", 0.1, 1.0), ("ack", 0.6, 1.0)],
    "victory_celebration": [("select", 0.02, 1.0), ("ack", 0.4, 1.0), ("select", 0.7, 1.0)],
    "waking_up": [("select", 0.1, 0.9)],
    "praying": [("select", 0.25, 0.6)],
    "eating": [("ack", 0.7, 0.6)],
    "being_picked_up": [("ack", 0.0, 0.9)],
    "guarding": [("ack", 0.5, 0.5)],
}


def build_pack(sound_dir: Path, animations) -> dict:
    return build_sound_pack(sound_dir, animations, GROUPS, EVENTS)
