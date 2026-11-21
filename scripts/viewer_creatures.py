"""Which DK2 mesh files belong to which creature (for the web viewer).

Animation files are named inconsistently (GOB-dance, Goblin-Walk, Bil-Dance, Bile_Demon-Melee,
REA3_Walk), so each creature lists its file-name prefixes. Matching is case-insensitive and the
longest prefix wins. Files that match no listed creature are returned, never guessed.
"""
from dataclasses import dataclass
from pathlib import Path

from scripts.dk2kmf import parse_kmf


@dataclass(frozen=True)
class CreatureDef:
    label: str
    pose: str                 # stem of the static pose-frame file ('' when the game has none)
    prefixes: tuple[str, ...]


def _c(label: str, pose: str, *names: str) -> CreatureDef:
    """CreatureDef whose prefixes are each file-name stem followed by '-' or '_' (case-insensitive)."""
    return CreatureDef(label, pose, tuple(p for n in names for p in (f"{n}-", f"{n}_")))


# Labels for groups I could not name with confidence use the raw file prefix and say so.
CREATURES: dict[str, CreatureDef] = {
    "goblin": _c("Goblin", "Gob-PoseFrame", "goblin", "gob"),
    "bile_demon": _c("Bile Demon", "BD-PoseFrame", "bile_demon", "biledemon", "bil", "bd"),
    "horned_reaper": _c("Horned Reaper", "Rea3-poseframe", "rea3"),
    "imp": _c("Imp", "Imp-PoseFrame", "imp"),
    "mistress": _c("Mistress", "Mis-PoseFrame", "mistress", "mis"),
    "warlock": _c("Warlock", "War-PoseFrame", "warlock", "war"),
    "dark_elf": _c("Dark Elf", "DElf-PoseFrame", "darkelf", "dkelf", "delf"),
    "skeleton": _c("Skeleton", "Ske-PoseFrame", "skeleton", "ske"),
    "vampire": _c("Vampire", "Vam-PoseFrame", "vampire", "vam"),
    "troll": _c("Troll", "Tro-PoseFrame", "troll", "tro"),
    "salamander": _c("Salamander", "Sal-PoseFrame", "salamander", "sal"),
    "dwarf": _c("Dwarf", "Dwa-PoseFrame", "dwarf", "dwa"),
    "thief": _c("Thief", "Thf-PoseFrame", "thief", "thf"),
    "rogue": _c("Rogue", "Rog-PoseFrame", "rogue", "rog"),
    "wizard": _c("Wizard", "Wiz-PoseFrame", "wizard", "wiz"),
    "knight": _c("Knight", "Kni-PoseFrame", "knight", "kni"),
    "dark_knight": _c("Dark Knight", "DKni-PoseFrame", "drknight", "drkni", "dkni"),
    "stone_knight": _c("Stone Knight", "StKni-PoseFrame", "stoneknight", "stk", "stkni"),
    "giant": _c("Giant", "Gia-PoseFrame", "giant", "gia"),
    "monk": _c("Monk", "Mon-PoseFrame", "monk", "mon"),
    "guard": _c("Guard", "Gua-PoseFrame", "guard", "gua"),
    "royal_guard": _c("Royal Guard", "RGua-PoseFrame", "rguard", "rgua"),
    "archer": _c("Archer", "Arc-PoseFrame", "archer", "arc"),
    "dark_angel": _c("Dark Angel", "DAN-PoseFrame", "darkangel", "dangel", "dan"),
    "king": _c("King (name from file)", "Kin-PoseFrame", "king", "kin"),
    "fairy": _c("Fai (name from file)", "Fai-PoseFrame", "fai"),
    "fir": _c("Fir (name from file)", "fir-poseframe", "fir"),
    "lol": _c("LOL (name from file)", "LOL-PoseFrame", "lol"),
    "maiden": _c("Maiden (name from file)", "", "maiden"),
    "pr1": _c("PR1 (name from file)", "PR1-PoseFrame", "pr1"),
    "pr2": _c("PR2 (name from file)", "PR2-PoseFrame", "pr2"),
    "pr3": _c("PR3 (name from file)", "PR3-PoseFrame", "pr3"),
    "chicken": _c("Chicken", "", "chicken"),
}


def _owner(stem: str, creatures: dict[str, CreatureDef]):
    low = stem.lower()
    best = None
    for cid, d in creatures.items():
        for p in d.prefixes:
            if low.startswith(p) and (best is None or len(p) > len(best[1])):
                best = (cid, p)
    return best


def find_animations(meshes_dir: Path, creatures: dict[str, CreatureDef] | None = None):
    """Return ({creature_id: {label: path}}, unattributed_stems). Only animated, non-pose files count."""
    creatures = CREATURES if creatures is None else creatures
    result: dict[str, dict[str, Path]] = {cid: {} for cid in creatures}
    unattributed: list[str] = []
    for path in sorted(Path(meshes_dir).glob("*.kmf"), key=lambda p: p.name.lower()):
        stem = path.stem
        owner = _owner(stem, creatures)
        if owner is None:
            unattributed.append(stem)
            continue
        if "pose" in stem.lower():
            continue
        if parse_kmf(path.read_bytes()).kind != "anim":
            continue
        cid, prefix = owner
        label = stem[len(prefix):].lower()
        n = 2
        while label in result[cid]:          # same label from two spellings: keep both
            label = f"{stem[len(prefix):].lower()}_{n}"
            n += 1
        result[cid][label] = path
    return result, unattributed
