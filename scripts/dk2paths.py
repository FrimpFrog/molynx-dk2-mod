"""Where the Dungeon Keeper 2 game data lives on this machine (read-only inputs).

EDITOR_DIR holds the editor's .kwd definition files and MAPS_DIR the per-map files; TEXT_DIR the
text .str files; GAME_DIR the game's WAD archives; MESHES_DIR the extracted .kmf models; and
QUUZ_MAPS_DIR the same maps as patched by the Quuz level fixes; BEHEMOTH_DIR the MegaGlest
Behemoth (the base of Molynx, kept in the working copy but not in git) and CHARACTER_PICKLE a
reference build of the finished Molynx character (scratch). Tests skip when `have()` says a
location is missing.

LICENCE: GPL-3.0.
"""
from pathlib import Path

_SCRATCH = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")

EDITOR_DIR = _SCRATCH / "up" / "Data1" / "editor"
MAPS_DIR = EDITOR_DIR / "maps"
TEXT_DIR = _SCRATCH / "up" / "Data2" / "Text" / "Default"
GAME_DIR = _SCRATCH / "dk2game" / "Ai"
MESHES_DIR = _SCRATCH / "extracted" / "Meshes"
BEHEMOTH_DIR = Path(__file__).resolve().parent.parent / "megaglest_behemoth"
CHARACTER_PICKLE = _SCRATCH / "char_full.pkl"
QUUZ_MAPS_DIR = (_SCRATCH / "up" / "flame" / "flame" / "resources" / "Quuz_LevelFixes"
                 / "Data" / "editor" / "maps")


def have(p: Path) -> bool:
    """True when the game data at `p` is present."""
    return p.exists()
