"""MegaGlest Golem sounds attached to Golem animations for the viewer. LICENCE: GPL-3.0
(sound files: MegaGlest data, CC BY-SA 3.0).

Pack format (data/<id>.sounds.json): {"files": {name: data URI}, "groups": {group: [names]},
"events": {animation: [{"group", "at": 0..1 of one cycle, "vol"}]}}. The viewer picks a random file of the group.
"""
import base64
import json
import subprocess
import tempfile
from pathlib import Path

GROUPS = {
    "walk": ["golem_walk1", "golem_walk2"],
    "attack": ["golem_attack1"],
    "hit": ["golem_hit1"],
    "die": ["golem_die1", "golem_die2", "golem_die3"],
    "select": [f"golem_select{i}" for i in range(1, 12)],
    "ack": [f"golem_ack{i}" for i in range(1, 12)],
}

# which sound plays when, as a fraction of one cycle of the animation (MegaGlest: walk/attack/die as in golem.xml,
# attack-start-time 0.4 for the impact; the rest are choices: select = a growl, ack = a short grunt)
EVENTS = {
    "walk": [("walk", 0.0, 1.0)],
    "attack": [("attack", 0.0, 1.0), ("hit", 0.4, 1.0)],
    "die": [("die", 0.0, 1.0)],
    "running": [("walk", 0.0, 1.0), ("walk", 0.5, 1.0)],
    "patrolling": [("walk", 0.0, 0.8), ("walk", 0.5, 0.8)],
    "turning": [("walk", 0.15, 0.8), ("walk", 0.6, 0.8)],
    "staggering": [("walk", 0.2, 0.7), ("walk", 0.7, 0.7)],
    "training": [("attack", 0.0, 1.0), ("hit", 0.14, 1.0), ("attack", 0.33, 1.0), ("hit", 0.47, 1.0),
                 ("attack", 0.66, 1.0), ("hit", 0.8, 1.0)],
    "special_attack": [("attack", 0.05, 1.0), ("hit", 0.5, 1.0)],
    "fighting_another_creature": [("attack", 0.02, 1.0), ("hit", 0.12, 1.0), ("hit", 0.3, 0.8), ("attack", 0.52, 1.0),
                                  ("hit", 0.62, 1.0), ("hit", 0.8, 0.8)],
    "getting_hit": [("hit", 0.05, 1.0)],
    "being_slapped": [("hit", 0.08, 1.0)],
    "knockback": [("hit", 0.02, 1.0), ("walk", 0.2, 1.0), ("walk", 0.3, 0.6)],
    "blocking": [("hit", 0.65, 0.9)],
    "dropping": [("hit", 0.35, 0.7)],
    "picking_up": [("walk", 0.3, 0.6)],
    "manufacturing": [("hit", 0.2, 0.6), ("hit", 0.7, 0.6)],
    "interacting_with_objects": [("hit", 0.25, 0.5), ("hit", 0.75, 0.5)],
    "angry": [("select", 0.05, 1.0)],
    "happy": [("ack", 0.2, 1.0)],
    "celebrating": [("ack", 0.1, 1.0), ("ack", 0.6, 1.0)],
    "victory_celebration": [("select", 0.02, 1.0), ("ack", 0.4, 1.0), ("select", 0.7, 1.0)],
    "frightened": [("select", 0.05, 0.8)],
    "unhappy": [("ack", 0.1, 0.7)],
    "looking_around": [("ack", 0.3, 0.7)],
    "waking_up": [("select", 0.1, 0.9)],
    "being_picked_up": [("ack", 0.0, 0.9)],
}


def _ogg_data_uri(wav: Path) -> str:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.ogg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "22050",
                        "-c:a", "libvorbis", "-q:a", "2", str(out)], check=True)
        return "data:audio/ogg;base64," + base64.b64encode(out.read_bytes()).decode()


def build_sound_pack(sound_dir: Path, animations, groups=None, events=None) -> dict:
    """animations: labels that exist on the creature (events for others are dropped). groups / events default to the Golem's."""
    groups = GROUPS if groups is None else groups
    events = EVENTS if events is None else events
    names = sorted({n for g in groups.values() for n in g})
    files = {n: _ogg_data_uri(Path(sound_dir) / f"{n}.wav") for n in names}
    out = {a: [{"group": g, "at": at, "vol": v} for g, at, v in ev] for a, ev in events.items() if a in set(animations)}
    return {"files": files, "groups": groups, "events": out}


def write_sound_pack(pack: dict, out_path: Path) -> Path:
    Path(out_path).write_text(json.dumps(pack, separators=(",", ":")))
    return Path(out_path)
