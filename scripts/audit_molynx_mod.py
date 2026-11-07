"""Independent audit of a Molynx Flame mod folder: `audit(mod_dir) -> AuditReport`.

The audit re-reads every file of the folder from disk and compares it with the game's own source
files, so it proves what is on disk instead of trusting the builder. It never calls the builder or the
record module and takes no expectation from them: what Molynx's record must look like at each stage
is written out below from the stage definitions. Run it on a folder with
`python -m scripts.audit_molynx_mod <folder>`; the exit code is 0 only when every line passes.

The stage and the maps are inferred from the folder. The maps are the `<name>` of the files in
`Data/editor/maps`; each map's source is `map_source_dir(name)` (the Quuz copy when the user's game
loads that one, else the editor folder). The stage comes from record 47: it is the stage whose set
of changed fields (`_stage_field_sets`) is closest to the fields in which record 47 differs from the
Bile Demon; a field of that set left unchanged, or a field outside it, is a rule (b) FAIL that names
the field and its record bytes, and the other rules still run for that stage. Molynx's files are read
with the format reader (dk2kmf, dk2kmf_write.anim_invariants), never with the builder or exporter.

Rules, each one PASS or FAIL line per map (per folder for g and h):

  (a) every block of the patched `<map>Globals.kld` except Creatures is byte-identical to the source
      (same blocks in the same order, header and content).
  (b) the Creatures block has 47 records; the first 46 are identical to the source; record 47 is the
      source's Bile Demon with exactly the fields of one stage changed (and only to the ruled values;
      a wrong slot tail names the slot, the sub-field and its record bytes).
  (c) header invariants: every block header of the Globals file is consistent (signature, the
      constants 4 / 28 / 1, check words signature + 1 and + 2, file size and content size match the
      content), the Creatures block holds whole 5537-byte records numbered by its item count, and its
      header is the source's apart from file size, item count and content size.
  (d) `<map>Things.kld` differs from its source in exactly one byte: the creature id, 3 -> 47, of the
      FIRST type-200 creature thing of the source that is a Bile Demon owned by a keeper (player 3..7),
      and that thing is owned by player 3 ("Keeper 1", the human's side). The line names the thing.
  (e) every other map file is byte-identical to its source.
  (f) animation (mesh) names: every animation slot name of record 47 that differs from the Bile
      Demon's exists as `Data/Meshes/<name>.kmf` in the folder (exact case, not empty); every name left
      as the Bile Demon's exists in the game's Meshes.WAD. From stage 2 every such Molynx file must also
      parse to its last byte, store its own name as its internal name, pass the animation invariants
      (`anim_invariants`), keep every sprite within 255 vertices and 255 triangles with every triangle
      index in range, and name only textures that exist as `Data/EngineTextures/<texture>.png` (exact
      case; a PNG whose sides are powers of two of at most MAX_TEXTURE = 128 px, the largest texture
      in the game's cache). Each file's loop flag must be the loop flag of the Bile Demon's own file in
      that slot (for a slot whose Bile Demon file is a static mesh: a static pose), except the slots in WRAP_EXCEPTIONS, which the PASS line lists by name. A stage-3 folder
      always FAILs (f): its icon and portrait files are not audited before Task 10.
  (g) the folder holds nothing else: `resource.toml`, the seven files of each map (a folder with the flat test map:
      all three flat maps, below) and, from stage 2,
      exactly the `Data/Meshes/<name>.kmf` files record 47 names and the `Data/EngineTextures/<t>.png`
      files those meshes name (no unreferenced mesh or texture).
  (i) `Data/editor/Creatures.kwd` is there: the editor's Creatures.kwd with record 47 added, checked
      with rules (a), (b), (c) and (f) like a Globals file (rule g allows it). The game loads it as
      its base creature table and applies a map's Globals as an update by creatureId; without
      record 47 in it, picking a Molynx map crashes (Test B, 2026-11-07).
  (h) `resource.toml` is valid TOML with a [resource] table of name, version, authors, description,
      and its name is the name of the folder.

The flat test map. A folder that holds "Molynx Test" or "Molynx Test Portal" is a flat-test-map
folder: it must hold the three flat maps "Molynx Test", "Molynx Test Portal" and "Dead Walking"
and the copied map "Siege" (rule g), and in it "Dead Walking" is the flat map under the shipped name, not the shipped map, so
rules (d) and (e) no longer apply to it (an `N/A (d, e)` line says so; the same for the two new
names). Rules (a), (b), (c) and (f) apply to their Globals files with Siege's Globals (FLAT_GLOBALS,
the Quuz copy when there is one) as the source (the folder's record 47 spliced exactly as on a
copied map; Dead Walking's own creature table makes Imps impossible to pick up). The map files are
checked by these rules, written from the test-map design without the map generator (no import of
it: own parsing, own literals; the references are the shipped Dead Walking in the editor folder,
the user's Quuz copy of Siege for the portal, and the editor's AutoSaveTriggers.kld):

  (m1)  69 x 69 like Dead Walking; the Map header and the outer ring of tiles are Dead Walking's.
  (m2)  two Dungeon Hearts, of players 3 and 4, each Dead Walking's heart of that player (tiles in
        shape and bytes, and the things on it, its one type-204 room thing, moved with it); Keeper
        1's heart where Dead Walking has it (the Players file's camera looks there).
  (m3)  one Portal of player 3, Siege's Keeper 1 portal (9 tiles, its things: none), connected by
        floor to the heart.
  (m4)  Keeper 1's open area (the flood fill from its heart over its Claimed Path, Portal, heart and
        Dirt Path) has at least 900 floor tiles, 300 of them Claimed Path, holds the heart and the
        portal and encloses nothing else; 4 to 6 Gems and 2 Gold tiles, each touching the area and
        otherwise set in Rock; no terrain beyond the design's.
  (m5)  Keeper 2 is sealed: the flood fill over every passable tile (all but Impenetrable Rock, Rock,
        Gold, Gems, Reinforced Wall, Edge of Map) from its heart reaches no Keeper 1 tile; the
        chamber is only its heart and Claimed Path; Impenetrable Rock at least 2 tiles thick all
        round (the PASS line gives the thickness).
  (m6)  things: the two heart room things, Keeper 1's 6 Imps and Molynx, Keeper 2's 4 Imps, nothing
        else; each creature a copy of a Dead Walking creature thing of its player (Molynx of its
        Bile Demon) on its keeper's floor; Molynx on floor next to Keeper 1's heart.
  (m7)  Players: Dead Walking's records, Keeper 1 with starting gold 100000 and isAi 0.
  (m8)  Variables: Dead Walking's entries in order (player 3's availability values may change), then
        only player-3 entries; no duplicates; player 3 has exactly rooms 1-4 and 6-17, doors 1-6,
        traps 1-11 and 16, spells 1-11 and 13, all ENABLE (3); the creature pool and creature
        availability of Siege's player 3 (ids 3, 4, 5, 6, 7, 10, 12, 22, 23, 24, 45, with Siege's
        values), plus Molynx (47, pool 50, ENABLE) only in "Molynx Test Portal".
  (m9)  Triggers: the editor's empty AutoSaveTriggers.kld byte for byte, except the timestamps,
        which are Dead Walking's.
  (m10) the level file: Dead Walking's, changed only in the name ("Molynx Test"; "Molynx Test
        Portal" for the variant), the description (FLAT_DESCRIPTION) and the path texts, which
        name the stem's own files (Dead Walking: the shipped paths), all present in the folder; the
        path record heads (u32 type, u32 0) are Dead Walking's, and every text field (name,
        description, each path) holds only zero bytes after its NUL; and the script's root trigger
        (u16 at byte 4536) is 0, as in the shipped maps with an empty Triggers file (Small,
        Warrens): Dead Walking's root 1 with no such trigger hangs the game (Test B).
  (m11) containers: each file one block that read_blocks accepts, Dead Walking's signature, header
        size, check words and timestamps, consistent file size, content size and item count.
  (m12) exactly one creature thing has id 47, and it is owned by player 3.
  (m13) (folder) "Dead Walking" holds "Molynx Test"'s Map, Things, Players, Triggers, Variables and
        Globals byte for byte; "Molynx Test Portal" differs from "Molynx Test" only in its level
        file and its Variables, by exactly the two creature-47 entries.

The stages (written independently of the builder, from the Molynx design):
  1  record 47 = Bile Demon with name "Molynx", creatureId 47, orderInEditor = the source's largest + 1.
  2  + every animation slot that is non-empty in the Bile Demon (not unk_1) renamed `Molynx-<stem>`
     (64-byte name, NUL padded) with the stem of SLOT_FILES (a literal table; "stand" for a slot
     not in it); empty slots stay empty. The 20 bytes after the name (`TAIL`) of a
     renamed slot that plays one of Molynx's animation files follow the stage-2 rule, written out in
     `_expected_tail`: flags = the Bile Demon's without 0x8 / 0x10 (no Start / End files), frames =
     the file's frame count (minus 1 when the file loops), fps 30, type 6, startAf = endAf = 0,
     startDist / endDist / sometimesOne the Bile Demon's. A static mesh slot (type 5: the poses,
     stunnedPose and elec included) and a slot of type NONE (0, not loaded by the game) keep the
     Bile Demon's 20 bytes. The frame count and
     loop flag are read from the folder's own file.
  3  + his character: every field of STAGE3 at its value, and his texts (STAGE3_TEXTS) in the
     folder's Data/Text/Default/Text.str (the game's own entries kept byte for byte, his after
     them) at the IDs record 47 names; the tooltip stays the shared 2408.

LICENCE: GPL-3.0.
"""
import argparse
import functools
import os
import struct
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from scripts.dk2kld_splice import MAP_SUFFIXES, map_source_dir
from scripts.dk2kmf import parse_kmf
from scripts.dk2kmf_write import anim_invariants
from scripts.dk2kwd_blocks import HEADER_SIZE, read_blocks
from scripts.dk2kwd_creatures import ANIM_SLOTS, FIELDS, RESOURCE_NAME_SIZE, RESOURCE_SIZE, Creature
from scripts.dk2paths import EDITOR_DIR, GAME_DIR, MAPS_DIR, MESHES_DIR, QUUZ_MAPS_DIR
from scripts.dk2wad import read_wad

CREATURES = 0xAA
THINGS = 0xBE
CREATURE_THING = 200
KEEPER_PLAYERS = range(3, 8)
HUMAN_PLAYER = 3  # "Keeper 1", the first keeper slot: the player the human plays
BILE_DEMON_ID, BILE_DEMON_NAME = 3, "Bile Demon"
MOLYNX_ID, MOLYNX_NAME, MOLYNX_PREFIX = 47, "Molynx", "Molynx-"
SOURCE_RECORDS = 46
NAME_FIELD_SIZE = 32
ICON, PORTRAIT = r"gui\creatures\evil\molynx", r"gui\creatures\evil\molynx-bg"
# Stage 3: his own character, written out here from the design of 2026-11-07 (a tankier,
# stronger, slightly faster Bile Demon, the Horned Reaper's bodyguard, never eating (stone); drawn to
# a Guard Room and a Training Room, happiest guarding and training, hating research; Cast Armour,
# creature spell 15, from level 4 in place of the gas spells). Raw
# record values (float32 fields: value x 4096).
STAGE3 = {
    "hp": 8000, "meleeDamage": 600, "speed": 1843, "runSpeed": 2662, "meleeRecharge": 12288,
    "meleeRange": 2867, "stunDuration": 12288, "mass": 24576, "fear": 600, "threat": 250,
    "pay": 700, "maxGoldHeld": 7000, "timeAwake": 330,
    "hungerRate": 134213632, "hungerFill": 0, "hpFromChicken": 0, "angerNoFood": 1,
    "tortureTimeToConvert": 983040, "possessionManaCost": 400, "scoreRating": 115,
    "manufacturePerSecond": 80, "expPerSecondTraining": 25, "manaGenPrayer": 70, "angerSlap": 4500,
    "attractions[0].present": 1, "attractions[0].roomId": 9, "attractions[0].roomSize": 0,
    "attractions[1].present": 1, "attractions[1].roomId": 7, "attractions[1].roomSize": 25,
    "happyJobs[0].jobType": 6, "happyJobs[0].chance": 50,
    "happyJobs[1].jobType": 4, "happyJobs[1].chance": 50,
    "hateJobs[0]": 3, "hateJobs[1]": 0,
    "spells[0].creatureSpellId": 15, "spells[0].levelAvailable": 4,
    "spells[0].playAnimation": 1, "spells[0].shotDelay": 3563,
    **{f"spells[1].{f}": 0 for f in ("playAnimation", "shotDelay", "creatureSpellId", "levelAvailable")},
}
# Stage 3: his texts, in the folder's Data/Text/Default/Text.str at the IDs record 47 names (the
# game's own entries before them unchanged); the tooltip stays the game's shared one, 2408.
STAGE3_TEXTS = {
    "nameStringId": "Molynx",
    "generalDescriptionStringId":
        "Molynx.  Living stone, sworn to guard the Horned Reaper.  {26}.  Fighting Style: {27}.",
    "strengthStringId": "Rock-hard hide and a crushing mace. Made of stone, he never eats, shrugs off "
                        "blows and hardens into Stone Skin when the fighting starts.",
    "weaknessStringId": "Proud and slow-witted. He won't study in the Library, and slapping him makes "
                        "him furious.",
}
SHARED_TOOLTIP_ID = 2408
TEXT_FILE = Path("Data") / "Text" / "Default" / "Text.str"
# Stage 2 also gives him the Horned Reaper's size (editor Creatures.kwd, "Horny": scale 5734 = 1.4,
# height 4096 = 1.0, eyeHeight 2744 = 0.67), his model files being as tall as Horny's.
SIZE = {"scale": 5734, "height": 4096, "eyeHeight": 2744}
TEXT_ID_FIELDS = ("nameStringId", "tooltipStringId", "generalDescriptionStringId",
                  "strengthStringId", "weaknessStringId")
MAPS_PATH = Path("Data") / "editor" / "maps"
EDITOR_CREATURES = "Data/editor/Creatures.kwd"  # rule (i): the base creature table with record 47

BLOCK_NAMES = {0x6E: "Terrain", 0x78: "Rooms", 0x82: "Traps", 0x8C: "Doors", 0x96: "KeeperSpells",
               0xA0: "CreatureSpells", 0xAA: "Creatures", 0xF0: "Objects", 0xFA: "EffectElements",
               0x104: "Shots", 0x10E: "Effects"}

_PARSE_ERRORS = (ValueError, KeyError, IndexError, struct.error, OSError, UnicodeDecodeError)
_STAGE1_FIELDS = frozenset({"name", "creatureId", "orderInEditor"})
_RESOURCE_TAIL = slice(RESOURCE_NAME_SIZE, RESOURCE_SIZE)

# The 20 bytes after a slot's 64-byte name (resource.bt ArtResource; decoded in the Task 8a report):
# (sub-field, offset in the 20 bytes, struct format).
TAIL = (("flags", 0, "<I"), ("frames", 4, "<I"), ("fps", 8, "<I"), ("startDist", 12, "<H"),
        ("endDist", 14, "<H"), ("type", 16, "<B"), ("startAf", 17, "<B"), ("endAf", 18, "<B"),
        ("sometimesOne", 19, "<B"))
HAS_START_END = 0x8 | 0x10        # "has a <name>Start / <name>End file": Molynx has none
TYPE_NONE, TYPE_MESH, TYPE_ANIM = 0, 5, 6
MOLYNX_FPS = 30                   # every Molynx file is resampled to 30 fps
MAX_SPRITE = 255                  # vertices and triangles per sprite (corners are bytes)
MAX_TEXTURE = 128                 # no texture in the game's cache is larger (0 of 1911)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Slots whose Molynx file may end where the Bile Demon's file loops: slot -> (file, why).
# Which Molynx file each animation slot names (stage 2), written out here from the design's slot
# table (slot-table.md, plan Task 8) and the rulings on loop variants: entrance needs a walk that
# ends (walk_once); the static mesh slots stunnedPose and elec take static poses (StunnedPose, a
# middle frame of staggering until the review round, now the last frame of falling; ElecPose, the
# peak of getting_hit), like the game's creatures. Review round (2026-11-07): fallback falling
# (it ends lying), torture its own clip, drink its own (eating without the chicken). A slot
# not listed takes "stand"; a slot empty in the Bile Demon stays empty; a static mesh slot (type 5)
# whose entry is staggering (dejectedPose in some maps) takes StunnedPose (MESH_SLOT_FILES).
SLOT_FILES = {
    "animWalkResource": "walk", "animRunResource": "running",
    "animDraggedPoseResource": "DraggedPose", "animRecoilHffResource": "getting_hit",
    "animRecoilHfbResource": "being_slapped", "animMelee1Resource": "attack",
    "animMelee2Resource": "attack", "animSpecialResource": "attack", "animDieResource": "die",
    "animDiePoseResource": "DiePose", "animHappyResource": "happy", "animAngryResource": "angry",
    "animSwingResource": "being_picked_up", "animSleepResource": "sleeping",
    "animEatResource": "eating", "animEatCorpseResource": "eating", "animDrinkResource": "drinking",
    "animBuildingResource": "manufacturing", "animPrayResource": "praying",
    "animFallbackResource": "falling", "animGetUpResource": "waking_up",
    "animDanceResource": "celebrating", "animDrunkResource": "staggering",
    "animDrunk2Resource": "staggering", "animStunnedPoseResource": "StunnedPose",
    "animDejectedPoseResource": "staggering", "animIdle1Resource": "stand",
    "animIdle2Resource": "fidgeting", "drunkIdle": "fidgeting",
    "animPoseFrameResource": "PoseFrame", "animWalk2Resource": "patrolling",
    "animEntranceResource": "walk_once", "animWalkbackResource": "walk",
    "animResearchResource": "interacting_with_objects", "animLootResource": "interacting_with_objects",
    "animMagicResource": "getting_hit", "animTortureResource": "torture",
    "animElecResource": "ElecPose", "animElectrocuteResource": "getting_hit",
}
MESH_SLOT_FILES = {"staggering": "StunnedPose"}
DEFAULT_FILE = "stand"
# slot -> (file, why it may not loop although the Bile Demon's file does). Empty since the review
# round of 2026-11-07: sleeping is a closed breathing loop in the bed, like BileDemon-bed(sleeping).
WRAP_EXCEPTIONS: dict[str, tuple[str, str]] = {}


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    lines: list[str]


class _Lines:
    def __init__(self) -> None:
        self.items: list[tuple[bool, str]] = []

    def add(self, ok: bool, rule: str, scope: str, message: str) -> None:
        self.items.append((ok, f"{'PASS' if ok else 'FAIL'} ({rule}) {scope}: {message}"))

    def note(self, rules: str, scope: str, message: str) -> None:
        """A line for rules that do not apply here (it does not fail the audit)."""
        self.items.append((True, f"N/A ({rules}) {scope}: {message}"))


def _block_label(index: int, signature: int) -> str:
    return f"block #{index} {BLOCK_NAMES.get(signature, 'unknown')} (0x{signature:X})"


def _first_difference(a: bytes, b: bytes) -> int:
    """Index of the first byte at which `a` and `b` differ (the shorter length when one is a prefix)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


# ---- the game's own meshes ----------------------------------------------------------------------

@functools.cache
def _game_mesh_names() -> frozenset[str] | None:
    """File names (exact case) of the game's meshes: Meshes.WAD, else the extracted Meshes folder."""
    wad = GAME_DIR / "Meshes.WAD"
    try:
        if wad.is_file():
            return frozenset(entry.name for entry in read_wad(wad).entries)
        if MESHES_DIR.is_dir():
            return frozenset(os.listdir(MESHES_DIR))
    except _PARSE_ERRORS:
        pass
    return None


@functools.cache
def _game_wad():
    wad = GAME_DIR / "Meshes.WAD"
    return read_wad(wad) if wad.is_file() else None


@functools.cache
def _game_mesh(file_name: str) -> bytes | None:
    """The bytes of one game mesh file (exact case), from Meshes.WAD or the extracted folder."""
    try:
        wad = _game_wad()
        if wad is not None:
            entry = next((e for e in wad.entries if e.name == file_name), None)
            return None if entry is None else wad.extract(entry)
        path = MESHES_DIR / file_name
        return path.read_bytes() if file_name in os.listdir(MESHES_DIR) else None
    except _PARSE_ERRORS:
        return None


def _file_kind(data: bytes) -> tuple[str, int, bool | None]:
    """(kind, frames, loops) of a KMF file: ("anim", n, wrap) or ("mesh", 0, None)."""
    return _model_kind(parse_kmf(data))


def _model_kind(model) -> tuple[str, int, bool | None]:
    """(kind, frames, loops) of a parsed KMF: ("anim", n, wrap) or ("mesh", 0, None)."""
    if model.kind == "anim":
        return "anim", model.anim.frame_count, bool(model.anim.wrap)
    if model.kind == "mesh":
        return "mesh", 0, None
    raise ValueError(f"is a {model.kind} file, not a mesh or an animation")


# ---- the folder's own mesh files ----------------------------------------------------------------

class _Files:
    """The folder's Data/Meshes and Data/EngineTextures files, each read and parsed once."""

    def __init__(self, mod_dir: Path) -> None:
        self.mod_dir = mod_dir
        self.meshes = mod_dir / "Data" / "Meshes"
        self.textures = mod_dir / "Data" / "EngineTextures"
        self.mesh_names = set(os.listdir(self.meshes)) if self.meshes.is_dir() else set()
        self.texture_names = set(os.listdir(self.textures)) if self.textures.is_dir() else set()
        self._parsed: dict[str, tuple] = {}

    def parsed(self, name: str):
        """(model or None, problem or None) of `Data/Meshes/<name>.kmf`."""
        if name not in self._parsed:
            self._parsed[name] = self._parse(name + ".kmf")
        return self._parsed[name]

    def _parse(self, file_name: str):
        if file_name not in self.mesh_names:
            folded = {n.lower(): n for n in self.mesh_names}
            if file_name.lower() in folded:
                return None, (f"Data/Meshes/{file_name} is missing (only "
                              f"{folded[file_name.lower()]!r}, other case, exists)")
            return None, f"Data/Meshes/{file_name} is missing"
        data = _read(self.meshes / file_name)
        if not data:
            return None, f"Data/Meshes/{file_name} is empty"
        try:
            model = parse_kmf(data)
        except _PARSE_ERRORS as err:
            return None, f"Data/Meshes/{file_name} does not parse: {err}"
        if model.end != len(data):
            return None, (f"Data/Meshes/{file_name} is not read to its end: {len(data) - model.end} "
                          "bytes are left after the model")
        if model.kind not in ("anim", "mesh"):
            return None, f"Data/Meshes/{file_name} is a {model.kind} file, not a mesh or an animation"
        return model, None

    def kind(self, name: str) -> tuple[str, int, bool | None] | None:
        """(kind, frames, loops) of the folder's file `name`, None when it cannot be read."""
        model, _problem = self.parsed(name)
        return None if model is None else _model_kind(model)


def _png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise ValueError("is not a PNG file")
    return struct.unpack_from(">II", data, 16)


def _power_of_two(n: int) -> bool:
    return n > 0 and not n & (n - 1)


# ---- record fields ------------------------------------------------------------------------------

def _field_extents(size: int) -> list[tuple[int, int, str]]:
    return sorted((f.offset, f.end, name) for name, f in FIELDS.items() if f.end <= size)


def _differing_fields(a: bytes, b: bytes) -> set[str]:
    """Names of the record fields that hold at least one differing byte (the fields tile the record)."""
    extents = _field_extents(len(a))
    found = set()
    for start, end, name in extents:
        if a[start:end] != b[start:end]:
            found.add(name)
    covered = extents[-1][1] if extents else 0
    if covered != len(a):
        raise ValueError(f"the record fields cover {covered} of {len(a)} bytes")
    return found


def _resource_bytes(record: Creature, slot: str) -> bytes:
    start = FIELDS[slot].offset
    return bytes(record.raw[start:start + RESOURCE_SIZE])


def _anim_slots() -> list[str]:
    return [slot for slot in ANIM_SLOTS if slot != "unk_1"]  # unk_1 holds no name; never touched


def _stage_field_sets(bile: Creature) -> dict[int, frozenset[str]]:
    """The set of record fields in which record 47 must differ from the Bile Demon, per stage."""
    renamed = frozenset(slot for slot in _anim_slots() if bile.resource_name(slot))
    stage2 = _STAGE1_FIELDS | renamed | set(SIZE) | {"icon1Resource", "portraitResource"}
    stage3 = (stage2 | {f for f, v in STAGE3.items() if bile.get(f) != v}
              | {f for f in STAGE3_TEXTS if bile.get(f) is not None})
    return {1: _STAGE1_FIELDS, 2: stage2, 3: stage3}


def _tail(resource: bytes) -> dict[str, int]:
    """The 20 bytes after a slot's name, by sub-field (see TAIL)."""
    tail = resource[RESOURCE_NAME_SIZE:RESOURCE_SIZE]
    return {name: struct.unpack_from(fmt, tail, at)[0] for name, at, fmt in TAIL}


def _tail_range(slot: str, sub: str) -> str:
    """The record bytes of one tail sub-field, e.g. 'record 47 bytes 2124-2127'."""
    at, fmt = next((at, fmt) for name, at, fmt in TAIL if name == sub)
    start = FIELDS[slot].offset + RESOURCE_NAME_SIZE + at
    return f"record 47 bytes {start}-{start + struct.calcsize(fmt) - 1}"


def _expected_tail(bile_tail: dict[str, int], file: tuple | None) -> tuple[dict | None, str]:
    """The stage-2 rule for one renamed slot: (expected tail, or None for "the Bile Demon's or the
    converted one", and what the slot is). `file` is (kind, frames, loops) of the Molynx file, None
    when the file cannot be read; then `frames` of an animation tail is None (not checkable)."""
    kind = bile_tail["type"]
    if kind == TYPE_NONE:
        return dict(bile_tail), "type NONE (not loaded), kept"
    if kind == TYPE_MESH:
        return dict(bile_tail), "a static mesh (a pose), kept"
    frames = None if file is None or file[0] != "anim" else file[1] - (1 if file[2] else 0)
    rewritten = {**bile_tail, "flags": bile_tail["flags"] & ~HAS_START_END, "frames": frames,
                 "fps": MOLYNX_FPS, "startAf": 0, "endAf": 0}
    return rewritten, "an animation, rewritten"


def _tail_problems(slot: str, name: str, found: dict, bile_tail: dict, file: tuple | None) -> list[str]:
    problems = []
    if file is not None and file[0] == "mesh" and bile_tail["type"] == TYPE_ANIM:
        problems.append(f"{slot} names {name}, a static mesh, but the Bile Demon's slot plays an "
                        f"animation (type {TYPE_ANIM})")
    if file is not None and file[0] == "anim" and bile_tail["type"] == TYPE_MESH:
        problems.append(f"{slot} names {name}, an animation, but the Bile Demon's slot is a static "
                        f"mesh (type {TYPE_MESH}) and takes a static pose")
    expected, _what = _expected_tail(bile_tail, file)
    for sub, _at, _fmt in TAIL:
        want = expected[sub]
        if want is None or found[sub] == want:
            continue
        shown = (lambda v: f"0x{v:X}") if sub == "flags" else str
        note = " (the Start / End bits 0x8 and 0x10 must be clear)" if sub == "flags" else ""
        problems.append(f"{slot} {sub} ({_tail_range(slot, sub)}) is {shown(found[sub])}, "
                        f"expected {shown(want)}{note}")
    return problems


def _name_field_bytes(record: Creature) -> bytes:
    start = FIELDS["name"].offset
    return bytes(record.raw[start:start + NAME_FIELD_SIZE])


def _text_problems(molynx: Creature, mod_dir: Path) -> list[str]:
    """Stage 3: the folder's Text.str is the game's with his texts after its own entries, and record
    47's text IDs name them."""
    from scripts.dk2paths import TEXT_DIR
    from scripts.dk2str import StrFile
    path = mod_dir / TEXT_FILE
    if not path.is_file():
        return [f"{TEXT_FILE.as_posix()} is missing (stage 3 ships his texts)"]
    try:
        mine = StrFile.read(path, TEXT_DIR / "MBToUni.dat")
        game = StrFile.read(TEXT_DIR / "Text.str")
    except (ValueError, struct.error) as err:
        return [f"{TEXT_FILE.as_posix()} does not read: {err}"]
    problems = []
    if mine.entries[:len(game.entries)] != game.entries or mine.header != game.header:
        problems.append(f"{TEXT_FILE.as_posix()}: the game's own {len(game.entries)} entries are not "
                        "kept byte for byte")
    for field, want in STAGE3_TEXTS.items():
        sid = molynx.get(field)
        if not len(game.entries) <= sid < len(mine.entries):
            problems.append(f"{field} is {sid}, not one of the entries added after the game's "
                            f"{len(game.entries)}")
        elif mine.text(sid) != want:
            problems.append(f"{field} {sid} reads {mine.text(sid)!r}, expected {want!r}")
    return problems


def _value_problems(molynx: Creature, bile: Creature, stage: int, order: int,
                    files: "_Files") -> tuple[list[str], dict[str, int]]:
    """Where record 47 has the right fields changed but not the ruled values; also counts of how
    the renamed slots' tails were checked."""
    problems, counts = [], {"rewritten": 0, "kept": 0, "unchecked": 0}
    if _name_field_bytes(molynx) != MOLYNX_NAME.encode().ljust(NAME_FIELD_SIZE, b"\0"):
        problems.append(f"name is {molynx.get('name')!r} (or has bytes after its NUL), "
                        f"expected {MOLYNX_NAME!r} padded with NULs")
    if molynx.get("creatureId") != MOLYNX_ID:
        problems.append(f"creatureId is {molynx.get('creatureId')}, expected {MOLYNX_ID}")
    if molynx.get("orderInEditor") != order:
        problems.append(f"orderInEditor is {molynx.get('orderInEditor')}, expected {order} "
                        "(the source's largest + 1)")
    if stage >= 2:
        for slot in _anim_slots():
            if not bile.resource_name(slot):
                continue
            name = molynx.resource_name(slot)
            raw = _resource_bytes(molynx, slot)
            if raw == _resource_bytes(bile, slot):
                continue  # left as the Bile Demon's: already named as a missing stage-2 field
            start = FIELDS[slot].offset
            stem = SLOT_FILES.get(slot, DEFAULT_FILE)
            if _tail(_resource_bytes(bile, slot))["type"] == TYPE_MESH:
                stem = MESH_SLOT_FILES.get(stem, stem)
            expected = MOLYNX_PREFIX + stem
            if name != expected:
                problems.append(f"{slot} name (record 47 bytes {start}-{start + RESOURCE_NAME_SIZE - 1}) "
                                f"is {name!r}, expected {expected!r}")
            elif raw[len(name):RESOURCE_NAME_SIZE].strip(b"\0"):
                problems.append(f"{slot} name (record 47 bytes {start}-{start + RESOURCE_NAME_SIZE - 1}) "
                                "has bytes after its NUL")
            bile_tail = _tail(_resource_bytes(bile, slot))
            file = files.kind(expected)  # the tail describes the file the slot must name
            problems += _tail_problems(slot, expected, _tail(raw), bile_tail, file)
            what = _expected_tail(bile_tail, file)[1]
            if file is None and bile_tail["type"] == TYPE_ANIM:
                counts["unchecked"] += 1
            else:
                counts["rewritten" if what.endswith("rewritten") else "kept"] += 1
    if stage >= 2:
        for field, want in SIZE.items():
            if molynx.get(field) != want:
                problems.append(f"{field} is {molynx.get(field)}, expected {want} (the Horned Reaper's)")
    if stage >= 3:
        for field, want in STAGE3.items():
            if molynx.get(field) != want:
                problems.append(f"{field} is {molynx.get(field)}, expected {want} (stage 3)")
        if molynx.get("tooltipStringId") != SHARED_TOOLTIP_ID:
            problems.append(f"tooltipStringId is {molynx.get('tooltipStringId')}, expected the "
                            f"shared {SHARED_TOOLTIP_ID}")
        problems += _text_problems(molynx, files.mod_dir)
    if stage >= 2:
        for slot, expected in (("icon1Resource", ICON), ("portraitResource", PORTRAIT)):
            raw = _resource_bytes(molynx, slot)
            if molynx.resource_name(slot) != expected:
                problems.append(f"{slot} is {molynx.resource_name(slot)!r}, expected {expected!r}")
            elif raw[_RESOURCE_TAIL] != _resource_bytes(bile, slot)[_RESOURCE_TAIL]:
                problems.append(f"{slot}: the last 20 bytes after its name differ from the Bile Demon's")
    return problems, counts


# ---- one map ------------------------------------------------------------------------------------

def _split_records(content: bytes, size: int) -> list[Creature]:
    if size <= 0 or len(content) % size:
        raise ValueError(f"{len(content)} content bytes are not whole records of {size} bytes")
    return [Creature(content[i:i + size]) for i in range(0, len(content), size)]


def _creatures_block(blocks, what: str):
    found = [b for b in blocks if b.signature == CREATURES]
    if len(found) != 1:
        raise ValueError(f"{what} has {len(found)} Creatures blocks, expected 1")
    return found[0]


def _check_blocks(src_blocks, mod_blocks) -> tuple[bool, str]:
    """Rule (a)."""
    src_sigs, mod_sigs = [b.signature for b in src_blocks], [b.signature for b in mod_blocks]
    if src_sigs != mod_sigs:
        return False, ("the blocks are " + ", ".join(hex(s) for s in mod_sigs)
                       + " but the source has " + ", ".join(hex(s) for s in src_sigs))
    problems = []
    for i, (src, mod) in enumerate(zip(src_blocks, mod_blocks)):
        if src.signature == CREATURES:
            continue
        if src.header != mod.header:
            problems.append(f"{_block_label(i, src.signature)} differs from the source: header, "
                            f"first differing byte {_first_difference(src.header, mod.header)}")
        elif src.content != mod.content:
            problems.append(f"{_block_label(i, src.signature)} differs from the source: content, "
                            f"first differing byte {_first_difference(src.content, mod.content)}")
    if problems:
        return False, "; ".join(problems)
    return True, f"all {len(src_blocks) - 1} blocks other than Creatures are byte-identical to the source"


def _header_problems(block, index: int) -> list[str]:
    """Rule (c) for one block header."""
    label = _block_label(index, block.signature)
    header = block.header
    if len(header) != HEADER_SIZE:
        return [f"{label}: header is {len(header)} bytes, expected {HEADER_SIZE}"]
    sig, four, size, check_one, twenty_eight, _count, one = struct.unpack_from("<7I", header, 0)
    check_two, content_size = struct.unpack_from("<2I", header, 48)
    problems = []
    expectations = [
        (sig == block.signature, f"signature word is 0x{sig:X}"),
        (four == 4, f"word at byte 4 is {four}, expected 4"),
        (size == HEADER_SIZE + len(block.content),
         f"file size (byte 8) is {size}, expected {HEADER_SIZE + len(block.content)}"),
        (check_one == block.signature + 1,
         f"check word at byte 12 is 0x{check_one:X}, expected 0x{block.signature + 1:X}"),
        (twenty_eight == 28, f"word at byte 16 is {twenty_eight}, expected 28"),
        (one == 1, f"word at byte 24 is {one}, expected 1"),
        (check_two == block.signature + 2,
         f"check word at byte 48 is 0x{check_two:X}, expected 0x{block.signature + 2:X}"),
        (content_size == len(block.content),
         f"content size (byte 52) is {content_size}, expected {len(block.content)}"),
    ]
    for ok, text in expectations:
        if not ok:
            problems.append(f"{label}: {text}")
    return problems


def _check_headers(src_blocks, mod_blocks, record_size: int) -> tuple[bool, str]:
    """Rule (c)."""
    problems = []
    for i, block in enumerate(mod_blocks):
        problems += _header_problems(block, i)
    src = _creatures_block(src_blocks, "the source")
    mod = _creatures_block(mod_blocks, "the Globals file")
    at = mod_blocks.index(mod)
    count = mod.item_count
    if len(mod.content) != count * record_size:
        problems.append(f"{_block_label(at, CREATURES)}: item count {count} does not match its "
                        f"{len(mod.content)} content bytes of {record_size}-byte records")
    differing = [i for i in range(HEADER_SIZE) if mod.header[i] != src.header[i]]
    allowed = {*range(8, 12), *range(20, 24), *range(52, 56)}  # file size, item count, content size
    stray = [i for i in differing if i not in allowed]
    if stray:
        problems.append(f"{_block_label(at, CREATURES)}: header differs from the source's at byte "
                        f"{stray[0]} (only file size, item count and content size may change)")
    if problems:
        return False, "; ".join(problems)
    return True, (f"all {len(mod_blocks)} block headers are consistent; the Creatures header is the "
                  f"source's apart from file size, item count and content size")


def _missing_field_problems(missing: list[str]) -> list[str]:
    return [f"{field} is left as the Bile Demon's (record 47 bytes {FIELDS[field].offset}-"
            f"{FIELDS[field].end - 1})" for field in missing]


def _check_creatures(src_blocks, mod_blocks, files: "_Files") -> tuple[bool, str, tuple | None]:
    """Rule (b): (ok, message, (stage, record 47, Bile Demon, record size) or None)."""
    src = _creatures_block(src_blocks, "the source")
    mod = _creatures_block(mod_blocks, "the Globals file")
    if src.item_count != SOURCE_RECORDS:
        return False, f"the source has {src.item_count} creature records, expected {SOURCE_RECORDS}", None
    size = len(src.content) // src.item_count
    source = _split_records(src.content, size)
    try:
        records = _split_records(mod.content, size)
    except ValueError as err:
        return False, str(err), None
    if len(records) != SOURCE_RECORDS + 1:
        return False, f"has {len(records)} creature records, expected {SOURCE_RECORDS + 1}", None
    changed = [i + 1 for i in range(SOURCE_RECORDS) if records[i].raw != source[i].raw]
    if changed:
        return False, "record " + ", ".join(map(str, changed)) + " differs from the source", None

    biles = [c for c in source
             if c.get("creatureId") == BILE_DEMON_ID and c.get("name") == BILE_DEMON_NAME]
    if len(biles) != 1:
        return False, f"the source has {len(biles)} records named {BILE_DEMON_NAME!r} with id {BILE_DEMON_ID}", None
    bile, molynx = biles[0], records[-1]
    differing = _differing_fields(bytes(bile.raw), bytes(molynx.raw))
    sets = _stage_field_sets(bile)
    # the stage whose field set is closest to the differing fields (the lower one on a tie); a field
    # of that set left unchanged (a slot left as the Bile Demon's, ...) or a field outside it is a
    # problem, and the other rules still run for that stage
    stage = min(sorted(sets), key=lambda s: len(differing ^ sets[s]))
    order = max(c.get("orderInEditor") for c in source) + 1
    by_offset = lambda f: FIELDS[f].offset  # noqa: E731
    problems = [f"{field} differs from the Bile Demon's, but stage {stage} keeps it (record 47 bytes "
                f"{FIELDS[field].offset}-{FIELDS[field].end - 1})"
                for field in sorted(differing - sets[stage], key=by_offset)]
    problems += _missing_field_problems(sorted(sets[stage] - differing, key=by_offset))
    values, counts = _value_problems(molynx, bile, stage, order, files)
    problems += values
    info = (stage, molynx, bile, size)
    if problems:
        return False, f"record 47 (stage {stage}): " + "; ".join(problems), info
    if stage == 1:
        detail = ", ".join(sorted(differing))
    else:
        detail = (f"{len(differing)} fields (stage {stage} set); the 20 bytes after each renamed "
                  f"slot's name follow the stage-2 rule: {counts['rewritten']} rewritten from "
                  f"Molynx's animation files, {counts['kept']} kept (static meshes, type NONE)")
        if counts["unchecked"]:
            detail += (f", {counts['unchecked']} whose frame count cannot be checked because the "
                       "file is missing or unreadable (rule f)")
    return True, (f"{len(records)} records; the first {SOURCE_RECORDS} are identical to the source; "
                  f"record 47 is Molynx at stage {stage}, differing from the Bile Demon only in {detail}"
                  ), info


def _thing_records(data: bytes) -> list[tuple[int, int, int]]:
    """(offset, type, data size) of every record of a Things file: one block, records back to back."""
    if len(data) < HEADER_SIZE:
        raise ValueError("shorter than a block header")
    sig, count = struct.unpack_from("<I", data, 0)[0], struct.unpack_from("<I", data, 20)[0]
    if sig != THINGS:
        raise ValueError(f"signature 0x{sig:X}, expected 0x{THINGS:X}")
    pos, records = HEADER_SIZE, []
    for _ in range(count):
        kind, size = struct.unpack_from("<II", data, pos)
        records.append((pos, kind, size))
        pos += 8 + size
    if pos != len(data):
        raise ValueError(f"records end at byte {pos}, the file has {len(data)} bytes")
    return records


def _check_things(src: bytes, mod: bytes) -> tuple[bool, str]:
    """Rule (d): the one changed byte is the creature id of the FIRST keeper-owned Bile Demon thing of
    the source, 3 -> 47, and that thing belongs to Keeper 1 (player 3), the human's side."""
    if len(src) != len(mod):
        return False, f"is {len(mod)} bytes, the source {len(src)}"
    diffs = [i for i in range(len(src)) if src[i] != mod[i]]
    if len(diffs) != 1:
        return False, (f"differs from the source in {len(diffs)} bytes, expected exactly 1 "
                       f"(the creature id of one Bile Demon)")
    at = diffs[0]

    things = []  # (index among the creature things, record offset, creature id, owner)
    for n, (pos, _kind, size) in enumerate(r for r in _thing_records(src) if r[1] == CREATURE_THING):
        if size != 28:
            raise ValueError(f"creature thing #{n} at byte {pos} has {size} data bytes, expected 28")
        things.append((n, pos, src[pos + 34], src[pos + 35]))
    hit = [t for t in things if t[1] + 34 == at]
    if not hit:
        return False, f"the changed byte {at} is not the creature id byte of any creature thing"
    n, _pos, creature_id, owner = hit[0]
    if creature_id != BILE_DEMON_ID:
        return False, (f"the changed creature thing #{n} is creature id {creature_id}, not a Bile Demon "
                       f"({BILE_DEMON_ID})")
    if mod[at] != MOLYNX_ID:
        return False, f"the creature id of thing #{n} was changed to {mod[at]}, expected {MOLYNX_ID}"
    if owner not in KEEPER_PLAYERS:
        return False, f"the changed creature thing #{n} is owned by player {owner}, not a keeper (3..7)"
    first_n, _first_pos, _id, first_owner = next(
        t for t in things if t[2] == BILE_DEMON_ID and t[3] in KEEPER_PLAYERS)
    if n != first_n:
        return False, (f"the retargeted Bile Demon is creature thing #{n} (owner player {owner}), not "
                       f"the first keeper-owned one, #{first_n} (owner player {first_owner})")
    if owner != HUMAN_PLAYER:
        return False, (f"the retargeted Bile Demon, creature thing #{n}, is owned by player {owner}, "
                       f"not Keeper 1 (player {HUMAN_PLAYER}), the human's side")
    return True, (f"differs from the source in exactly 1 byte: creature thing #{n} (byte {at}), the "
                  f"first keeper-owned Bile Demon, {BILE_DEMON_ID} -> {MOLYNX_ID}, owned by "
                  f"Keeper 1 (player {owner})")


def _check_other_files(mod_dir: Path, name: str, source_dir: Path) -> tuple[bool, str]:
    """Rule (e)."""
    problems, count = [], 0
    for suffix in MAP_SUFFIXES:
        if suffix in ("Globals.kld", "Things.kld"):
            continue
        file_name = name + suffix
        mod, src = _read(mod_dir / MAPS_PATH / file_name), _read(source_dir / file_name)
        count += 1
        if mod is None:
            problems.append(f"{file_name} is missing")
        elif src is None:
            problems.append(f"{file_name} has no source")
        elif mod != src:
            where = _first_difference(src, mod)
            problems.append(f"{file_name} differs from the source (first differing byte {where}, "
                            f"sizes {len(mod)} and {len(src)})")
    if problems:
        return False, "; ".join(problems)
    return True, f"the other {count} files are byte-identical to the source"


def _model_problems(name: str, model, files: "_Files") -> tuple[list[str], set[str]]:
    """Rule (f) on one parsed Molynx file: (problems, the texture names its materials use)."""
    file_name = f"Data/Meshes/{name}.kmf"
    body = model.anim if model.kind == "anim" else model.mesh
    problems = []
    if body.name != name:
        problems.append(f"{file_name} calls itself {body.name!r} inside, not {name!r}")
    if model.kind == "anim":
        problems += [f"{file_name}: {text}" for text in anim_invariants(model.anim)[:3]]
    for s, sprite in enumerate(body.sprites):
        if len(sprite.vertices) > MAX_SPRITE:
            problems.append(f"{file_name} sprite {s} has {len(sprite.vertices)} vertices, at most "
                            f"{MAX_SPRITE} fit")
        for lod, triangles in enumerate(sprite.triangles):
            if len(triangles) > MAX_SPRITE:
                problems.append(f"{file_name} sprite {s} LOD {lod} has {len(triangles)} triangles, "
                                f"at most {MAX_SPRITE} fit")
            if any(not 0 <= i < len(sprite.vertices) for t in triangles for i in t):
                problems.append(f"{file_name} sprite {s} LOD {lod} has a triangle index outside its "
                                f"{len(sprite.vertices)} vertices")
        if not 0 <= sprite.material < len(model.materials):
            problems.append(f"{file_name} sprite {s} uses material {sprite.material} of "
                            f"{len(model.materials)}")
    textures = {t for material in model.materials for t in material.textures}
    return problems, textures


@functools.lru_cache(maxsize=1)
def _game_texture_names() -> frozenset[str]:
    """Lower-case names of the textures in the game's own EngineTextures cache (no MMn suffix):
    a prop the file carries (the chicken of an eat file, a bed) uses these, as the game's files do."""
    from scripts.dk2texture import TextureCache
    cache = TextureCache(GAME_DIR / "EngineTextures.dir", GAME_DIR / "EngineTextures.dat")
    return frozenset(n[:-3].lower() for n in cache.names if n[-3:-1].upper() == "MM")


def _texture_problems(texture: str, files: "_Files") -> list[str]:
    file_name = texture + ".png"
    shown = f"Data/EngineTextures/{file_name}"
    if file_name not in files.texture_names and texture.lower() in _game_texture_names():
        return []           # a game texture (a prop's): the game's own cache has it
    if file_name not in files.texture_names:
        folded = {n.lower(): n for n in files.texture_names}
        if file_name.lower() in folded:
            return [f"{shown} is missing (only {folded[file_name.lower()]!r}, other case, exists)"]
        return [f"{shown} is missing"]
    try:
        width, height = _png_size(_read(files.textures / file_name) or b"")
    except (ValueError, struct.error) as err:
        return [f"{shown} {err}"]
    if not (_power_of_two(width) and _power_of_two(height)
            and width <= MAX_TEXTURE and height <= MAX_TEXTURE):
        return [f"{shown} is {width}x{height}; each side must be a power of two of at most "
                f"{MAX_TEXTURE} px"]
    return []


def _wrap_problems(slots: dict[str, str], bile: Creature, files: "_Files") -> tuple[list[str], list[str]]:
    """Rule (f), loop flags: (problems, the WRAP_EXCEPTIONS used, as text)."""
    problems, used = [], []
    for slot, name in slots.items():
        kind = _tail(_resource_bytes(bile, slot))["type"]
        mine = files.kind(name)
        if kind == TYPE_NONE or mine is None:
            continue  # not loaded by the game / unreadable (named above)
        bile_name = bile.resource_name(slot)
        data = _game_mesh(bile_name + ".kmf")
        if data is None:
            problems.append(f"{slot}: the Bile Demon's {bile_name}.kmf is not in the game's meshes, "
                            "so its loop flag cannot be compared")
            continue
        theirs = _file_kind(data)
        if theirs[0] == "mesh":
            if mine[0] != "mesh":
                problems.append(f"{slot}: {name} is an animation, but it replaces the Bile Demon's "
                                f"static {bile_name}: it must be a static pose")
            continue
        if mine[0] != "anim" or mine[2] == theirs[2]:
            continue  # a static mesh in an animation slot is a rule (b) problem
        exception = WRAP_EXCEPTIONS.get(slot)
        if exception and exception[0] == name and theirs[2] and not mine[2]:
            used.append(f"{slot} {name} does not loop although the Bile Demon's {bile_name} does: "
                        f"{exception[1]}")
            continue
        problems.append(f"{slot}: {name} {'loops' if mine[2] else 'does not loop'}, but the Bile "
                        f"Demon's {bile_name} {'loops' if theirs[2] else 'does not'}")
    return problems, used


def _check_assets(stage: int, molynx: Creature, bile: Creature,
                  files: "_Files") -> tuple[bool, str, set[str], set[str]]:
    """Rule (f): (ok, message, Molynx mesh names used, texture names they use)."""
    game = _game_mesh_names()
    problems = []
    names = {slot: molynx.resource_name(slot) for slot in _anim_slots()}
    bile_names = {slot: bile.resource_name(slot) for slot in names}
    renamed = {slot: name for slot, name in names.items() if name != bile_names[slot] and name}
    changed = sorted(set(renamed.values()))
    kept = sorted({name for slot, name in names.items() if name and name == bile_names[slot]})

    textures: set[str] = set()
    for name in changed:
        model, problem = files.parsed(name)
        if problem:
            problems.append(problem)
        elif stage >= 2:
            more, used = _model_problems(name, model, files)
            problems += more
            textures |= used
    for texture in sorted(textures):
        problems += _texture_problems(texture, files)
    if kept:
        if game is None:
            problems.append("the game's Meshes.WAD (or the extracted Meshes folder) was not found, so "
                            f"the {len(kept)} unchanged Bile Demon names cannot be checked")
        else:
            problems += [f"{name}.kmf is not in the game's Meshes.WAD" for name in kept
                         if name + ".kmf" not in game]
    exceptions = []
    if stage >= 2:
        more, exceptions = _wrap_problems(renamed, bile, files)
        problems += more
    if problems:
        return False, "; ".join(problems), set(changed), textures
    if stage == 1:
        return True, (f"every animation slot name resolves: 0 Molynx meshes in Data/Meshes and "
                      f"{len(kept)} Bile Demon names in the game's Meshes.WAD"), set(), set()
    message = (f"every animation slot name resolves: {len(changed)} Molynx meshes in Data/Meshes (each "
               f"parses to its end, has its own name inside, passes the animation invariants, keeps "
               f"sprites within {MAX_SPRITE} vertices and triangles) using {len(textures)} textures in "
               f"Data/EngineTextures ({', '.join(sorted(textures))}; powers of two up to "
               f"{MAX_TEXTURE} px), and {len(kept)} Bile Demon names in the game's Meshes.WAD; each "
               f"file loops exactly when the Bile Demon's file in its slot does (where that file is "
               f"a static mesh: a static pose)")
    if exceptions:
        message += f", except {len(exceptions)} recorded exception(s): " + "; ".join(exceptions)
    return True, message, set(changed), textures


def _safe(check, *args) -> tuple:
    """Run a check; a file that does not parse becomes a FAIL result instead of a crash."""
    try:
        return check(*args)
    except _PARSE_ERRORS as err:
        return (False, f"could not be checked: {err}")


def _audit_map(mod_dir: Path, name: str, lines: _Lines, files: _Files) -> tuple:
    """Rules (a) to (f) for one copied map; returns (the stage found in record 47 or None, the
    Molynx mesh names its record uses, the texture names those meshes use)."""
    try:
        source_dir = map_source_dir(name)
    except FileNotFoundError as err:
        for rule in "abcdef":
            lines.add(False, rule, name, f"no source files to compare with: {err}")
        return None, set(), set()
    return _audit_globals(mod_dir, name, source_dir / f"{name}Globals.kld", lines, files,
                          later=lambda: _audit_things_and_files(mod_dir, name, source_dir, lines))


def _audit_things_and_files(mod_dir: Path, name: str, source_dir: Path, lines: _Lines) -> None:
    """Rules (d) and (e) of a copied map."""
    maps = mod_dir / MAPS_PATH
    mod_things = _read(maps / f"{name}Things.kld")
    src_things = _read(source_dir / f"{name}Things.kld")
    if mod_things is None or src_things is None:
        lines.add(False, "d", name, f"{name}Things.kld is missing" if mod_things is None
                  else "the source Things file is missing")
    else:
        ok, message = _safe(_check_things, src_things, mod_things)
        lines.add(ok, "d", name, message)

    ok, message = _safe(_check_other_files, mod_dir, name, source_dir)
    lines.add(ok, "e", name, message)


def _audit_globals(mod_dir: Path, name: str, source: Path, lines: _Lines, files: _Files,
                   later=None) -> tuple:
    """Rules (a), (b), (c) of `<name>Globals.kld` against the Globals file `source`, then `later()`
    (rules d and e of a copied map), then rule (f); returns what `_audit_map` returns."""
    maps = mod_dir / MAPS_PATH
    info = None

    mod_globals = _read(maps / f"{name}Globals.kld")
    src_globals = _read(source)
    if mod_globals is None or src_globals is None:
        which = f"{name}Globals.kld is missing" if mod_globals is None else "the source Globals is missing"
        for rule in "abc":
            lines.add(False, rule, name, which)
    else:
        try:
            src_blocks, mod_blocks = read_blocks(src_globals), read_blocks(mod_globals)
            record_size = len(_creatures_block(src_blocks, "the source").content) // SOURCE_RECORDS
        except ValueError as err:
            for rule in "abc":
                lines.add(False, rule, name, f"the Globals file does not parse: {err}")
        else:
            ok, message = _safe(_check_blocks, src_blocks, mod_blocks)
            lines.add(ok, "a", name, message)
            ok, message, *rest = _safe(_check_creatures, src_blocks, mod_blocks, files)
            info = rest[0] if rest else None
            lines.add(ok, "b", name, message)
            ok, message = _safe(_check_headers, src_blocks, mod_blocks, record_size)
            lines.add(ok, "c", name, message)

    if later is not None:
        later()

    if info:
        stage, molynx, bile, _size = info
        ok, message, *rest = _safe(_check_assets, stage, molynx, bile, files)
        lines.add(ok, "f", name, message)
        meshes, textures = rest if rest else (set(), set())
        return stage, meshes, textures
    lines.add(False, "f", name, "not checked because record 47 could not be read")
    return None, set(), set()


def _audit_editor_creatures(mod_dir: Path, lines: _Lines, files: _Files) -> tuple:
    """Rule (i): `Data/editor/Creatures.kwd` must be there, and it is checked like a map's Globals
    file against the editor's own Creatures.kwd: rules (a), (b), (c) and (f) on its one Creatures
    block (46 records unchanged, record 47 Molynx in the editor's 5449-byte record size). The game
    loads this file as its base creature table and applies a map's Globals as an update by
    creatureId, so without record 47 here picking a Molynx map crashes (the Test B crash)."""
    label = "editor Creatures.kwd"
    path = mod_dir / EDITOR_CREATURES
    if not path.is_file():
        lines.add(False, "i", label, f"{EDITOR_CREATURES} is missing: the game's base creature table "
                                     "would have no creature 47, and a map that updates creature 47 crashes")
        return None, set(), set()
    lines.add(True, "i", label, f"{EDITOR_CREATURES} is there (the base creature table)")
    return _audit_globals_file(path, EDITOR_DIR / "Creatures.kwd", label, lines, files)


def _audit_globals_file(mod_path: Path, source: Path, label: str, lines: _Lines, files: _Files) -> tuple:
    """Rules (a), (b), (c) and (f) of one creature-table file `mod_path` against `source`."""
    info = None
    mod_data, src_data = _read(mod_path), _read(source)
    if mod_data is None or src_data is None:
        for rule in "abc":
            lines.add(False, rule, label, "the file or its source is missing")
    else:
        try:
            src_blocks, mod_blocks = read_blocks(src_data), read_blocks(mod_data)
            record_size = len(_creatures_block(src_blocks, "the source").content) // SOURCE_RECORDS
        except ValueError as err:
            for rule in "abc":
                lines.add(False, rule, label, f"the file does not parse: {err}")
        else:
            ok, message = _safe(_check_blocks, src_blocks, mod_blocks)
            lines.add(ok, "a", label, message)
            ok, message, *rest = _safe(_check_creatures, src_blocks, mod_blocks, files)
            info = rest[0] if rest else None
            lines.add(ok, "b", label, message)
            ok, message = _safe(_check_headers, src_blocks, mod_blocks, record_size)
            lines.add(ok, "c", label, message)
    if info:
        stage, molynx, bile, _size = info
        ok, message, *rest = _safe(_check_assets, stage, molynx, bile, files)
        lines.add(ok, "f", label, message)
        meshes, textures = rest if rest else (set(), set())
        return stage, meshes, textures
    lines.add(False, "f", label, "not checked because record 47 could not be read")
    return None, set(), set()


# ---- the flat test map -------------------------------------------------------------------------
# Written from the test-map design and the layout rules, independently of the map generator: the
# audit parses every file itself and takes its expectations from the shipped files (Dead Walking in
# the editor folder, the user's Quuz copy of Siege for the portal, the editor's empty
# AutoSaveTriggers.kld) and from the literals below.

FLAT_BASE = "Dead Walking"
FLAT_GLOBALS = "Siege"   # the flat maps' creature table: a normal one (Dead Walking's pins its Imps)
# stem -> (Molynx also in the portal pool, the name the level file shows)
FLAT_VARIANTS = {"Molynx Test": (False, "Molynx Test"),
                 "Molynx Test Portal": (True, "Molynx Test Portal"),
                 FLAT_BASE: (False, "Molynx Test")}
FLAT_PLAIN, FLAT_PORTAL = "Molynx Test", "Molynx Test Portal"
FLAT_WITH = ("Siege",)  # the copied map a flat-test-map folder also holds (the shipped-map round's)
FLAT_SIX = (".kwd", "Map.kld", "Things.kld", "Players.kld", "Triggers.kld", "Variables.kld")
LEVEL_SCRIPT_ROOT = 4536  # u16 at 56 + 4480 of a level file: its script's root trigger (0: none)
FLAT_DESCRIPTION = "Flat test map: Keeper 2 is sealed in, no enemies, no script."
FLAT_RULES = tuple(f"m{i}" for i in range(1, 13))
# Terrain ids of Map.kld (1-based in Terrain.kwd).
T_IMPENETRABLE, T_ROCK, T_DIRT, T_GOLD, T_GEMS, T_CLAIMED = 1, 2, 3, 6, 7, 8
T_REINFORCED, T_PORTAL, T_HEART, T_EDGE = 9, 12, 14, 30
SOLID_TERRAIN = frozenset({T_IMPENETRABLE, T_ROCK, T_GOLD, T_GEMS, T_REINFORCED, T_EDGE})
FLAT_TERRAIN = frozenset({T_IMPENETRABLE, T_ROCK, T_DIRT, T_GOLD, T_GEMS, T_CLAIMED, T_PORTAL,
                          T_HEART, T_EDGE})
TERRAIN_NAMES = {T_IMPENETRABLE: "Impenetrable Rock", T_ROCK: "Rock", T_DIRT: "Dirt Path",
                 T_GOLD: "Gold", T_GEMS: "Gems", T_CLAIMED: "Claimed Path", T_PORTAL: "Portal",
                 T_HEART: "Dungeon Heart", T_EDGE: "Edge of Map"}
KEEPER_2 = 4
IMP_ID = 1
OBJECT_THING, ROOM_THING = 194, 204
FLAT_CREATURES = {(IMP_ID, HUMAN_PLAYER): 6, (MOLYNX_ID, HUMAN_PLAYER): 1, (IMP_ID, KEEPER_2): 4}
MIN_FLOOR, MIN_CLAIMED, MIN_WALL = 900, 300, 2
GEMS_RANGE, GOLD_COUNT = range(4, 7), 2
STARTING_GOLD = 100000
PLAYER_RECORD = 205
VARIABLE_ENTRY = 16
POOL_VARIABLE, AVAILABILITY_VARIABLE = 1, 2
AV_ROOM, AV_CREATURE, AV_TRAP, AV_DOOR, AV_SPELL = 1, 2, 3, 4, 5
AV_NAMES = {AV_ROOM: "room", AV_CREATURE: "creature", AV_TRAP: "trap", AV_DOOR: "door",
            AV_SPELL: "spell"}
ENABLE = 3
# What Keeper 1 must have ENABLE: every keeper room, door, trap and spell of the game.
KEEPER_1_ITEMS = {AV_ROOM: frozenset({1, 2, 3, 4, *range(6, 18)}), AV_DOOR: frozenset(range(1, 7)),
                  AV_TRAP: frozenset({*range(1, 12), 16}), AV_SPELL: frozenset({*range(1, 12), 13})}
# Siege's creature pool for player 3 (creature id -> pool value), copied into the flat map.
IMP = 1   # creature id of the Imp
SIEGE_POOL = {7: 50, 5: 50, 12: 10, 10: 50, 6: 50, 4: 50, 24: 50, 3: 50, 22: 50, 23: 20, 45: 20}
MOLYNX_POOL = 50
LEVEL_NAME_AT, LEVEL_NAME_BYTES = 56, 128
LEVEL_DESC_AT, LEVEL_DESC_BYTES = 184, 2048
LEVEL_FLAGS = 56 + 25484
PATH_LIST, PATH_RECORD, PATH_TEXT = 222, 72, 64
EDITOR_MAPS = "Data\\Editor\\Maps\\"


def _header_end(data: bytes) -> int:
    """The header size of a map file: 28 + the u32 at byte 16 (56; 36 for Map, 60 for Triggers)."""
    return 28 + struct.unpack_from("<I", data, 16)[0]


@functools.cache
def _flat_sources() -> dict:
    """The shipped files the flat maps are compared with (read once)."""
    base = {s: (MAPS_DIR / (FLAT_BASE + s)).read_bytes() for s in (*FLAT_SIX, "Globals.kld")}
    siege = {s: (QUUZ_MAPS_DIR / ("Siege" + s)).read_bytes() for s in ("Map.kld", "Things.kld")}
    return {"base": base, "siege": siege,
            "auto": (EDITOR_DIR / "AutoSaveTriggers.kld").read_bytes()}


class _Grid:
    """A Map.kld: width, height and 4-byte tiles (terrain, player, flag, unknown)."""

    def __init__(self, data: bytes) -> None:
        if struct.unpack_from("<I", data, 0)[0] != 0x64:
            raise ValueError("Map.kld signature is not 0x64")
        self.header = data[:_header_end(data)]
        self.w, self.h = struct.unpack_from("<II", data, 20)
        if len(data) != len(self.header) + 4 * self.w * self.h:
            raise ValueError(f"Map.kld of {len(data)} bytes does not hold {self.w} x {self.h} tiles")
        at = len(self.header)
        self.tiles = [data[at + 4 * i:at + 4 * i + 4] for i in range(self.w * self.h)]

    def at(self, x: int, y: int) -> bytes:
        return self.tiles[y * self.w + x]

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def cells(self, terrain: int, owner: int | None = None) -> list[tuple[int, int]]:
        return [(i % self.w, i // self.w) for i, t in enumerate(self.tiles)
                if t[0] == terrain and (owner is None or t[1] == owner)]

    def flood(self, starts, ok) -> set[tuple[int, int]]:
        """Tiles 4-connected to `starts` through tiles for which `ok(tile)` holds."""
        seen, todo = set(starts), list(starts)
        while todo:
            x, y = todo.pop()
            for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if self.inside(*c) and c not in seen and ok(self.at(*c)):
                    seen.add(c)
                    todo.append(c)
        return seen


def _things_of(data: bytes) -> list[tuple[int, bytes]]:
    """(type, data) of every record of a Things file."""
    return [(kind, data[pos + 8:pos + 8 + size]) for pos, kind, size in _thing_records(data)]


def _xy(thing_data: bytes) -> tuple[int, int]:
    return struct.unpack_from("<ii", thing_data, 0)


def _moved(thing_data: bytes, dx: int, dy: int) -> bytes:
    x, y = _xy(thing_data)
    return struct.pack("<ii", x + dx, y + dy) + thing_data[8:]


def _shape(grid: _Grid, terrain: int, owner: int, things) -> tuple:
    """(top-left corner, sorted (dx, dy, tile), sorted (type, data) of the object and room things
    on its bounding box, moved to the corner (0, 0)) of the tiles of one terrain and owner."""
    cells = grid.cells(terrain, owner)
    if not cells:
        return None, [], []
    x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
    x1, y1 = max(c[0] for c in cells), max(c[1] for c in cells)
    tiles = sorted((x - x0, y - y0, grid.at(x, y)) for x, y in cells)
    on = sorted((kind, _moved(d, -x0, -y0)) for kind, d in things
                if kind in (OBJECT_THING, ROOM_THING) and x0 <= _xy(d)[0] <= x1 and y0 <= _xy(d)[1] <= y1)
    return (x0, y0), tiles, on


def _records(data: bytes, size: int, what: str) -> list[bytes]:
    hsz = _header_end(data)
    content = data[hsz:]
    if len(content) % size or len(content) // size != struct.unpack_from("<I", data, 20)[0]:
        raise ValueError(f"{what}: {len(content)} content bytes are not the {struct.unpack_from('<I', data, 20)[0]} "
                         f"records of {size} bytes its header counts")
    return [content[i:i + size] for i in range(0, len(content), size)]


def _wtext(data: bytes, at: int, size: int) -> str:
    return data[at:at + size].decode("utf-16-le").split("\0")[0]


def _bytes_after_nul(field: bytes, unit: int) -> bool:
    """True when a NUL-terminated text field (`unit` 1: char, 2: wchar) has a non-zero byte after
    its first NUL (or has no NUL at all)."""
    for i in range(0, len(field) - unit + 1, unit):
        if field[i:i + unit] == bytes(unit):
            return any(field[i:])
    return True


def _level_paths(kwd: bytes) -> tuple[int, list[str]]:
    """(byte offset of the first path record, the paths) of a level file."""
    hsz = _header_end(kwd)
    end = hsz + struct.unpack_from("<I", kwd, hsz - 4)[0]
    if struct.unpack_from("<I", kwd, end)[0] != PATH_LIST:
        raise ValueError(f"the level file has no path list ({PATH_LIST}) after its content")
    first = end + 8
    count = struct.unpack_from("<H", kwd, 20)[0]
    return first, [kwd[first + PATH_RECORD * i + 8:first + PATH_RECORD * (i + 1)].split(b"\0")[0]
                   .decode("latin-1") for i in range(count)]


class _Flat:
    """One flat map of the folder, parsed once; each `m<n>` method is one rule: (ok, message)."""

    def __init__(self, mod_dir: Path, stem: str) -> None:
        self.mod_dir, self.stem = mod_dir, stem
        self.in_pool, self.shown = FLAT_VARIANTS[stem]
        self.src = _flat_sources()
        self.files = {s: _read(mod_dir / MAPS_PATH / (stem + s)) for s in FLAT_SIX}
        self._parsed = None

    def parsed(self):
        """(map, things, base map, base things, Keeper 1's area, Keeper 2's chamber)."""
        if self._parsed is None:
            missing = [self.stem + s for s, d in self.files.items() if d is None]
            if missing:
                raise ValueError(f"missing {', '.join(missing)}")
            grid, base = _Grid(self.files["Map.kld"]), _Grid(self.src["base"]["Map.kld"])
            things, bthings = _things_of(self.files["Things.kld"]), _things_of(self.src["base"]["Things.kld"])
            heart = grid.cells(T_HEART, HUMAN_PLAYER)
            area = grid.flood(heart, lambda t: (t[0] in (T_CLAIMED, T_PORTAL, T_HEART)
                                                and t[1] == HUMAN_PLAYER) or t[0] == T_DIRT)
            chamber = grid.flood(grid.cells(T_HEART, KEEPER_2), lambda t: t[0] not in SOLID_TERRAIN)
            self._parsed = grid, things, base, bthings, area, chamber
        return self._parsed

    def file(self, suffix: str) -> bytes:
        data = self.files[suffix]
        if data is None:
            raise ValueError(f"{self.stem}{suffix} is missing")
        return data

    # rule m1: size and border
    def m1(self):
        grid, _t, base, *_ = self.parsed()
        if (grid.w, grid.h) != (base.w, base.h):
            return False, f"the map is {grid.w} x {grid.h}, {FLAT_BASE} is {base.w} x {base.h}"
        problems = []
        if grid.header != base.header:
            problems.append(f"the Map header differs from {FLAT_BASE}'s")
        ring = [(x, y) for y in range(grid.h) for x in range(grid.w)
                if x in (0, grid.w - 1) or y in (0, grid.h - 1)]
        bad = [c for c in ring if grid.at(*c) != base.at(*c)]
        if bad:
            problems.append(f"{len(bad)} tiles of the outer ring differ from {FLAT_BASE}'s, e.g. "
                            f"{bad[0]} is {grid.at(*bad[0]).hex()}, {FLAT_BASE} has {base.at(*bad[0]).hex()}")
        if problems:
            return False, "; ".join(problems)
        return True, (f"{grid.w} x {grid.h} with the Map header and all {len(ring)} outer ring tiles "
                      f"of {FLAT_BASE}, byte for byte")

    # rule m2: two Dungeon Hearts made as the base makes them
    def m2(self):
        grid, things, base, bthings, *_ = self.parsed()
        problems = []
        owners = sorted({t[1] for t in grid.tiles if t[0] == T_HEART})
        if owners != [HUMAN_PLAYER, KEEPER_2]:
            problems.append(f"Dungeon Heart tiles belong to players {owners}, expected [3, 4]")
        corners = {}
        for owner in (HUMAN_PLAYER, KEEPER_2):
            corner, tiles, on = _shape(grid, T_HEART, owner, things)
            bcorner, btiles, bon = _shape(base, T_HEART, owner, bthings)
            corners[owner] = corner
            if tiles != btiles:
                problems.append(f"player {owner}'s heart tiles are not {FLAT_BASE}'s heart "
                                f"({len(btiles)} tiles) in shape and bytes")
            if on != bon:
                problems.append(f"the things on player {owner}'s heart are not {FLAT_BASE}'s "
                                f"({len(bon)} room thing(s)) moved with it")
            if owner == HUMAN_PLAYER and corner != bcorner:
                problems.append(f"Keeper 1's heart is at {corner}, not where {FLAT_BASE} has it "
                                f"({bcorner}, the Players file's camera)")
        rooms = sorted(d[19] for kind, d in things if kind == ROOM_THING and d[18] == T_HEART)
        if rooms != [HUMAN_PLAYER, KEEPER_2]:
            problems.append(f"heart room things belong to players {rooms}, expected one each for 3 and 4")
        if problems:
            return False, "; ".join(problems)
        return True, (f"two Dungeon Hearts, Keeper 1's at {corners[HUMAN_PLAYER]} (as in {FLAT_BASE}) "
                      f"and Keeper 2's at {corners[KEEPER_2]}, each {FLAT_BASE}'s 25 tiles with its "
                      f"one heart room thing at the centre")

    # rule m3: one Keeper 1 portal, Siege's, connected to the heart
    def m3(self):
        grid, things, _b, _bt, area, _c = self.parsed()
        siege = _Grid(self.src["siege"]["Map.kld"])
        corner, tiles, on = _shape(grid, T_PORTAL, HUMAN_PLAYER, things)
        _sc, stiles, son = _shape(siege, T_PORTAL, HUMAN_PLAYER, _things_of(self.src["siege"]["Things.kld"]))
        problems = []
        cells = grid.cells(T_PORTAL)
        if len(cells) != len(stiles) or any(grid.at(*c)[1] != HUMAN_PLAYER for c in cells):
            problems.append(f"{len(cells)} Portal tiles, expected one portal of {len(stiles)} tiles of player 3")
        if tiles != stiles:
            problems.append("the Portal tiles are not Siege's Keeper 1 portal in shape and bytes")
        if on != son:
            problems.append(f"the things on the portal are not Siege's ({len(son)})")
        if cells and not set(cells) <= area:
            problems.append("the portal is not connected by floor to Keeper 1's heart")
        if problems:
            return False, "; ".join(problems)
        return True, (f"one {len(stiles)}-tile Portal of Keeper 1 at {corner}, Siege's tiles and "
                      f"things ({len(son)}), connected by floor to the heart")

    # rule m4: Keeper 1's open area, gems and gold
    def m4(self):
        grid, _t, _b, _bt, area, _c = self.parsed()
        problems = []
        floor = [c for c in area if grid.at(*c)[0] in (T_DIRT, T_CLAIMED)]
        claimed = [c for c in area if grid.at(*c)[:2] == bytes([T_CLAIMED, HUMAN_PLAYER])]
        if len(floor) < MIN_FLOOR:
            problems.append(f"Keeper 1's open area has {len(floor)} floor tiles, fewer than {MIN_FLOOR}")
        if len(claimed) < MIN_CLAIMED:
            problems.append(f"Keeper 1's open area has {len(claimed)} Claimed Path tiles, fewer than {MIN_CLAIMED}")
        everything = {(x, y) for y in range(grid.h) for x in range(grid.w)}
        outside = everything - area
        reach, todo = {(0, 0)} & outside, [c for c in [(0, 0)] if c in outside]
        while todo:
            x, y = todo.pop()
            for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if c in outside and c not in reach:
                    reach.add(c)
                    todo.append(c)
        if reach != outside:
            inner = sorted(outside - reach)
            problems.append(f"{len(inner)} tiles inside Keeper 1's area are not floor, e.g. {inner[0]} "
                            f"({TERRAIN_NAMES.get(grid.at(*inner[0])[0], grid.at(*inner[0])[0])})")
        gems, gold = grid.cells(T_GEMS), grid.cells(T_GOLD)
        if len(gems) not in GEMS_RANGE:
            problems.append(f"{len(gems)} Gems tiles, expected 4 to 6")
        if len(gold) != GOLD_COUNT:
            problems.append(f"{len(gold)} Gold tiles, expected {GOLD_COUNT}")
        for x, y in gems + gold:
            near = [c for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if grid.inside(*c)]
            if not any(c in area for c in near):
                problems.append(f"the Gems/Gold tile {x},{y} does not touch Keeper 1's open area")
            if not all(c in area or grid.at(*c)[0] in (T_ROCK, T_GEMS, T_GOLD) for c in near):
                problems.append(f"the Gems/Gold tile {x},{y} is not set in Rock")
        other = sorted({t[0] for t in grid.tiles} - FLAT_TERRAIN)
        if other:
            problems.append(f"the map holds other terrain: ids {other}")
        if problems:
            return False, "; ".join(problems)
        return True, (f"Keeper 1's open area: {len(area)} tiles ({len(floor)} floor, {len(claimed)} "
                      f"Claimed Path of Keeper 1) holding the heart and the portal, nothing enclosed "
                      f"in it; {len(gems)} Gems and {len(gold)} Gold tiles set in Rock, each touching "
                      f"the area; only the terrain of the design")

    # rule m5: Keeper 2 sealed in
    def m5(self):
        grid, _t, _b, _bt, area, chamber = self.parsed()
        if not chamber:
            return False, "Keeper 2 has no Dungeon Heart"
        problems = []
        if chamber & area or any(grid.at(*c)[1] == HUMAN_PLAYER for c in chamber):
            problems.append("Keeper 2's chamber reaches Keeper 1's tiles (a flood fill over every "
                            "passable tile from Keeper 2's heart)")
        foreign = sorted(c for c in chamber
                         if grid.at(*c)[:2] not in (bytes([T_CLAIMED, KEEPER_2]), bytes([T_HEART, KEEPER_2])))
        if foreign:
            t = grid.at(*foreign[0])
            problems.append(f"Keeper 2's chamber holds {len(foreign)} tiles other than its Claimed Path "
                            f"and heart, e.g. {foreign[0]} ({TERRAIN_NAMES.get(t[0], t[0])} of player {t[1]})")
        # the wall: the distance to the nearest tile that is neither chamber nor Impenetrable Rock
        nearest = None
        for y in range(grid.h):
            for x in range(grid.w):
                if (x, y) in chamber or grid.at(x, y)[0] == T_IMPENETRABLE:
                    continue
                d = min(max(abs(x - cx), abs(y - cy)) for cx, cy in chamber)
                if nearest is None or d < nearest[0]:
                    nearest = (d, (x, y))
        thick = grid.w + grid.h if nearest is None else nearest[0] - 1
        on_edge = any(not grid.inside(cx + dx, cy + dy) for cx, cy in chamber
                      for dx in range(-MIN_WALL, MIN_WALL + 1) for dy in range(-MIN_WALL, MIN_WALL + 1))
        if thick < MIN_WALL or on_edge:
            where = "" if nearest is None else (f", e.g. {nearest[1]} "
                                                f"({TERRAIN_NAMES.get(grid.at(*nearest[1])[0], '?')})")
            problems.append(f"the wall around Keeper 2's chamber is {max(thick, 0)} tiles of "
                            f"Impenetrable Rock thick, at least {MIN_WALL} needed{where}")
        if problems:
            return False, "; ".join(problems)
        return True, (f"Keeper 2's chamber ({len(chamber)} tiles: its heart and Claimed Path) is "
                      f"sealed: no passable path to Keeper 1, walled by Impenetrable Rock at least "
                      f"{thick} tiles thick on every side, no Gems or Gold inside")

    # rule m6: things
    def m6(self):
        grid, things, base, bthings, area, chamber = self.parsed()
        siege = _Grid(self.src["siege"]["Map.kld"])
        sthings = _things_of(self.src["siege"]["Things.kld"])
        allowed = []
        for owner in (HUMAN_PLAYER, KEEPER_2):
            corner, _tl, _on = _shape(grid, T_HEART, owner, things)
            bcorner, _btl, bon = _shape(base, T_HEART, owner, bthings)
            if corner is not None:
                allowed += [(kind, _moved(d, *corner)) for kind, d in bon]
        corner, _tl, _on = _shape(grid, T_PORTAL, HUMAN_PLAYER, things)
        if corner is not None:
            allowed += [(kind, _moved(d, *corner))
                        for kind, d in _shape(siege, T_PORTAL, HUMAN_PLAYER, sthings)[2]]
        base_creatures = [d for kind, d in bthings if kind == CREATURE_THING]
        problems, counts, others = [], {}, []
        for kind, d in things:
            if kind != CREATURE_THING:
                if (kind, d) in allowed:
                    allowed.remove((kind, d))
                else:
                    others.append(kind)
                continue
            cid, owner = d[26], d[27]
            counts[(cid, owner)] = counts.get((cid, owner), 0) + 1
            x, y = _xy(d)
            region = area if owner == HUMAN_PLAYER else chamber if owner == KEEPER_2 else set()
            if (x, y) not in region or grid.at(x, y)[0] in SOLID_TERRAIN:
                problems.append(f"creature {cid} of player {owner} at {x},{y} is not on its keeper's floor")
            source_id = BILE_DEMON_ID if cid == MOLYNX_ID else cid
            like = d[:26] + bytes([source_id]) + d[27:]  # Molynx is a retargeted Bile Demon
            if not any(b[8:] == like[8:] and b[27] == owner for b in base_creatures):
                problems.append(f"creature {cid} of player {owner} at {x},{y} is not a copy of one of "
                                f"{FLAT_BASE}'s creature {source_id} things of that player")
            if cid == MOLYNX_ID:
                beside = any(grid.inside(x + dx, y + dy)
                             and grid.at(x + dx, y + dy)[:2] == bytes([T_HEART, HUMAN_PLAYER])
                             for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                if grid.at(x, y)[0] not in (T_DIRT, T_CLAIMED) or not beside:
                    problems.append(f"Molynx at {x},{y} is not on floor next to Keeper 1's heart")
        if others:
            problems.append(f"unexpected things of types {sorted(others)} (only the hearts' room "
                            f"things and the creatures belong on the map)")
        if allowed:
            problems.append(f"{len(allowed)} heart or portal thing(s) missing")
        if counts != FLAT_CREATURES:
            shown = ", ".join(f"{n} x creature {c} of player {o}" for (c, o), n in sorted(counts.items()))
            problems.append(f"creatures: {shown}; expected 6 Imps and Molynx of player 3, 4 Imps of player 4")
        if problems:
            return False, "; ".join(problems)
        return True, (f"{len(things)} things: the 2 heart room things, 6 Imps and Molynx (next to the "
                      f"heart) of Keeper 1 on its floor, 4 Imps of Keeper 2 in its chamber, each a copy "
                      f"of a {FLAT_BASE} creature thing; nothing else")

    # rule m7: players
    def m7(self):
        new = _records(self.file("Players.kld"), PLAYER_RECORD, "Players.kld")
        old = _records(self.src["base"]["Players.kld"], PLAYER_RECORD, f"{FLAT_BASE}Players.kld")
        if len(new) != len(old):
            return False, f"{len(new)} player records, {FLAT_BASE} has {len(old)}"
        problems = []
        for a, b in zip(new, old):
            player = b[-37]
            if player == HUMAN_PLAYER:
                gold, is_ai = struct.unpack_from("<iI", a, 0)
                if gold != STARTING_GOLD or is_ai != 0 or a[8:] != b[8:]:
                    problems.append(f"Keeper 1 (player 3) is not {FLAT_BASE}'s record with starting gold "
                                    f"{STARTING_GOLD} and isAi 0 (has {gold}, {is_ai})")
            elif a != b:
                problems.append(f"player {player}'s record differs from {FLAT_BASE}'s at byte "
                                f"{_first_difference(a, b)}")
        if problems:
            return False, "; ".join(problems)
        return True, (f"{len(new)} player records, {FLAT_BASE}'s, except Keeper 1's starting gold "
                      f"{STARTING_GOLD} (Keeper 1 human: isAi 0)")

    # rule m8: variables
    def m8(self):
        new = [struct.unpack("<i12s", e) for e in _records(self.file("Variables.kld"), VARIABLE_ENTRY, "Variables.kld")]
        old = [struct.unpack("<i12s", e) for e in _records(self.src["base"]["Variables.kld"], VARIABLE_ENTRY,
                                                           f"{FLAT_BASE}Variables.kld")]
        problems = []

        def av(rest):
            av_type, player, type_id, value = struct.unpack("<HHii", rest)
            return (av_type, player, type_id), value

        if len(new) < len(old):
            return False, f"{len(new)} entries, fewer than {FLAT_BASE}'s {len(old)}"
        for i, ((vid, rest), (bvid, brest)) in enumerate(zip(new, old)):
            k1_av = bvid == AVAILABILITY_VARIABLE and av(brest)[0][1] == HUMAN_PLAYER
            if (vid, rest[:8] if k1_av else rest) != (bvid, brest[:8] if k1_av else brest):
                problems.append(f"entry {i} of {FLAT_BASE}'s Variables moved or changed")
                break
        for vid, rest in new[len(old):]:
            if vid == AVAILABILITY_VARIABLE and av(rest)[0][1] == HUMAN_PLAYER:
                continue
            if vid == POOL_VARIABLE and struct.unpack("<iii", rest)[2] == HUMAN_PLAYER:
                continue
            problems.append(f"an added entry (variable {vid}) is not a pool or availability entry of player 3")
        keys = [av(rest)[0] for vid, rest in new if vid == AVAILABILITY_VARIABLE]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            problems.append(f"duplicate availability entries (type, player, id) {dup}")
        pool = [struct.unpack("<iii", rest) for vid, rest in new if vid == POOL_VARIABLE]
        pool_keys = [(c, p) for c, _v, p in pool]
        pdup = sorted({k for k in pool_keys if pool_keys.count(k) > 1})
        if pdup:
            problems.append(f"duplicate creature pool entries (creature, player) {pdup}")
        k1 = {k: v for k, v in (av(rest) for vid, rest in new if vid == AVAILABILITY_VARIABLE)
              if k[1] == HUMAN_PLAYER}
        for av_type, ids in KEEPER_1_ITEMS.items():
            have = {k[2] for k in k1 if k[0] == av_type}
            if have != ids:
                problems.append(f"player 3 has {AV_NAMES[av_type]} availability for {sorted(have)}, "
                                f"expected {sorted(ids)}")
        not_enabled = sorted((AV_NAMES.get(k[0], k[0]), k[2], v) for k, v in k1.items() if v != ENABLE)
        for what, type_id, value in not_enabled:
            problems.append(f"player 3's {what} {type_id} is {value}, not ENABLE ({ENABLE})")
        want_pool = {**SIEGE_POOL, **({MOLYNX_ID: MOLYNX_POOL} if self.in_pool else {})}
        got_pool = {c: v for c, v, p in pool if p == HUMAN_PLAYER}
        if got_pool != want_pool:
            problems.append(f"player 3's creature pool is {dict(sorted(got_pool.items()))}, expected "
                            f"{dict(sorted(want_pool.items()))}")
        got_av = {k[2] for k in k1 if k[0] == AV_CREATURE}
        want_av = set(want_pool) | {IMP}   # the Imp is available, not in the pool (review round)
        if got_av != want_av:
            problems.append(f"player 3's creature availability is {sorted(got_av)}, expected {sorted(want_av)}")
        if problems:
            return False, "; ".join(problems)
        molynx = " and Molynx (47, pool 50)" if self.in_pool else ", not Molynx"
        return True, (f"{len(new)} entries: {FLAT_BASE}'s {len(old)} in order (player 3's availability "
                      f"values set to ENABLE) then {len(new) - len(old)} added for player 3; every keeper "
                      f"room, door, trap and spell ENABLE; Siege's creature pool and availability "
                      f"({len(SIEGE_POOL)} creatures){molynx}; no duplicates")

    # rule m9: the editor's empty Triggers file
    def m9(self):
        data, auto, base = self.file("Triggers.kld"), self.src["auto"], self.src["base"]["Triggers.kld"]
        stamps = slice(_header_end(auto) - 28, _header_end(auto) - 8)
        if len(data) != len(auto) or data[:stamps.start] != auto[:stamps.start] or data[stamps.stop:] != auto[stamps.stop:]:
            return False, (f"{self.stem}Triggers.kld ({len(data)} bytes) is not the editor's empty "
                           f"AutoSaveTriggers.kld ({len(auto)} bytes) outside its timestamps")
        if data[stamps] != base[stamps]:
            return False, f"the Triggers timestamps (bytes {stamps.start}-{stamps.stop - 1}) are not {FLAT_BASE}'s"
        return True, (f"the editor's empty {len(auto)}-byte Triggers file (no script) with "
                      f"{FLAT_BASE}'s timestamps")

    # rule m10: the level file
    def m10(self):
        data, base = self.file(".kwd"), self.src["base"][".kwd"]
        if len(data) != len(base):
            return False, f"the level file is {len(data)} bytes, {FLAT_BASE}'s {len(base)}"
        problems = []
        first, paths = _level_paths(data)
        _bfirst, bpaths = _level_paths(base)
        regions = [(LEVEL_NAME_AT, LEVEL_NAME_AT + LEVEL_NAME_BYTES),
                   (LEVEL_DESC_AT, LEVEL_DESC_AT + LEVEL_DESC_BYTES),
                   (first, first + PATH_RECORD * len(bpaths)),
                   (LEVEL_SCRIPT_ROOT, LEVEL_SCRIPT_ROOT + 2)]
        if struct.unpack_from("<H", data, LEVEL_SCRIPT_ROOT)[0] != 0:
            problems.append(f"the script's root trigger (u16 at byte {LEVEL_SCRIPT_ROOT}) is "
                            f"{struct.unpack_from('<H', data, LEVEL_SCRIPT_ROOT)[0]}, not 0: the Triggers "
                            "file is empty, and a root with no trigger hangs the game")
        stray = [i for i in range(len(data)) if data[i] != base[i]
                 and not any(a <= i < b for a, b in regions)]
        if stray:
            problems.append(f"the level file differs from {FLAT_BASE}'s outside the name, description "
                            f"and paths (first at byte {stray[0]})")
        name, desc = _wtext(data, LEVEL_NAME_AT, LEVEL_NAME_BYTES), _wtext(data, LEVEL_DESC_AT, LEVEL_DESC_BYTES)
        if name != self.shown:
            problems.append(f"the level name is {name!r}, expected {self.shown!r}")
        if desc != FLAT_DESCRIPTION:
            problems.append(f"the description is {desc!r}, expected {FLAT_DESCRIPTION!r}")
        heads = [i for i in range(len(bpaths)) if data[first + PATH_RECORD * i:first + PATH_RECORD * i + 8]
                 != base[first + PATH_RECORD * i:first + PATH_RECORD * i + 8]]
        if heads:
            problems.append(f"path record {heads[0]}: its first 8 bytes (u32 type, u32 0) are not "
                            f"{FLAT_BASE}'s")
        fields = [("name", LEVEL_NAME_AT, LEVEL_NAME_BYTES, 2), ("description", LEVEL_DESC_AT, LEVEL_DESC_BYTES, 2)]
        fields += [(f"path {i}", first + PATH_RECORD * i + 8, PATH_TEXT, 1) for i in range(len(paths))]
        for what, at, size, unit in fields:
            if _bytes_after_nul(data[at:at + size], unit):
                problems.append(f"the {what} field (bytes {at}-{at + size - 1}) has non-zero bytes after its NUL")
        prefix = EDITOR_MAPS + FLAT_BASE
        want = [EDITOR_MAPS + self.stem + p[len(prefix):] if p.startswith(prefix) else p for p in bpaths]
        if paths != want:
            problems.append(f"the paths are {paths}, expected {want}")
        present = set(os.listdir(self.mod_dir / MAPS_PATH))
        absent = [p for p in paths if p.startswith(EDITOR_MAPS) and p[len(EDITOR_MAPS):] not in present]
        if absent:
            problems.append(f"the level file names files the folder lacks: {absent}")
        if any(len(p) >= PATH_TEXT for p in paths):
            problems.append("a path does not fit its 64-byte field")
        if problems:
            return False, "; ".join(problems)
        flags = struct.unpack_from("<H", data, LEVEL_FLAGS)[0]
        own = [p for p in paths if p.startswith(EDITOR_MAPS)]
        others = ", ".join(p.rsplit("\\", 1)[-1] for p in paths if not p.startswith(EDITOR_MAPS))
        return True, (f"{FLAT_BASE}'s level file (flags 0x{flags:X} kept) named {name!r} with the test "
                      f"map's description, every text field zero after its NUL, every path record head "
                      f"{FLAT_BASE}'s; its {len(own)} map paths name "
                      f"{'the shipped' if self.stem == FLAT_BASE else 'its own'} files, all in the folder; "
                      f"the {len(own) + 1}th is the game's {others}")

    # rule m11: containers
    def m11(self):
        problems = []
        for suffix in FLAT_SIX:
            data, like, name = self.files[suffix], self.src["base"][suffix], self.stem + suffix
            if data is None:
                problems.append(f"{name} is missing")
                continue
            try:
                read_blocks(data)
            except ValueError as err:
                problems.append(f"{name}: read_blocks refuses it ({err})")
                continue
            hsz, sig = _header_end(data), struct.unpack_from("<I", data, 0)[0]
            if hsz != _header_end(like) or sig != struct.unpack_from("<I", like, 0)[0]:
                problems.append(f"{name}: signature or header size is not {FLAT_BASE}'s")
                continue
            checks = [
                (struct.unpack_from("<I", data, 12)[0] == sig + 1, "check word at byte 12 is not signature + 1"),
                (data[hsz - 8:hsz - 4] == like[hsz - 8:hsz - 4], f"check word at byte {hsz - 8} is not {FLAT_BASE}'s"),
                (struct.unpack_from("<I", data, 8)[0] == len(data), "file size field is not the file's size"),
            ]
            if suffix == ".kwd":
                checks.append((data[hsz - 4:hsz] == like[hsz - 4:hsz], "content size is not the base's"))
            else:
                checks.append((struct.unpack_from("<I", data, hsz - 4)[0] == len(data) - hsz,
                               "content size field is wrong"))
            if suffix != "Map.kld":
                checks.append((data[hsz - 28:hsz - 8] == like[hsz - 28:hsz - 8],
                               f"timestamps differ from {FLAT_BASE}'s"))
            if suffix == "Things.kld":
                checks.append((struct.unpack_from("<I", data, 20)[0] == len(_thing_records(data)), "item count"))
            problems += [f"{name}: {text}" for ok, text in checks if not ok]
            try:
                if suffix == "Players.kld":
                    _records(data, PLAYER_RECORD, name)
                elif suffix == "Variables.kld":
                    _records(data, VARIABLE_ENTRY, name)
            except ValueError as err:
                problems.append(str(err))
        if problems:
            return False, "; ".join(problems)
        return True, (f"all six files are one block each that read_blocks accepts, with {FLAT_BASE}'s "
                      f"signatures, header sizes, check words and timestamps, and consistent sizes and counts")

    # rule m12: one Molynx
    def m12(self):
        _g, things, *_ = self.parsed()
        molynx = [d for kind, d in things if kind == CREATURE_THING and d[26] == MOLYNX_ID]
        if len(molynx) != 1 or molynx[0][27] != HUMAN_PLAYER:
            owners = [d[27] for d in molynx]
            return False, (f"{len(molynx)} creature things with id {MOLYNX_ID} (owners {owners}), "
                           f"expected exactly one, owned by player {HUMAN_PLAYER}")
        x, y = _xy(molynx[0])
        return True, f"exactly one creature thing with id {MOLYNX_ID}, owned by Keeper 1 (player 3), at {x},{y}"


def _audit_flat(mod_dir: Path, stem: str, lines: "_Lines", files: "_Files") -> tuple:
    """Rules (a), (b), (c), (f) on the Globals file against the base map's, and m1 to m12."""
    result = _audit_globals(mod_dir, stem, map_source_dir(FLAT_GLOBALS) / f"{FLAT_GLOBALS}Globals.kld",
                            lines, files)
    if stem == FLAT_BASE:
        note = (f"rules (d) and (e) no longer apply: {FLAT_BASE} is no longer the shipped map but "
                f"the flat test map (rules m1 to m13)")
    else:
        note = (f"rules (d) and (e) do not apply: {stem} is the flat test map, built from "
                f"{FLAT_BASE} (rules m1 to m13)")
    lines.note("d, e", stem, note)
    flat = _Flat(mod_dir, stem)
    for rule in FLAT_RULES:
        ok, message = _safe(getattr(flat, rule))
        lines.add(ok, rule, stem, message)
    return result


def _check_flat_variants(mod_dir: Path) -> tuple[bool, str]:
    """Rule m13: Dead Walking is Molynx Test under the shipped name; the portal variant differs
    from Molynx Test only in its Variables (the creature-47 entries) and its level file."""
    maps = mod_dir / MAPS_PATH
    same = ("Map.kld", "Things.kld", "Players.kld", "Triggers.kld", "Globals.kld")
    problems = []

    def read(stem, suffix):
        data = _read(maps / (stem + suffix))
        if data is None:
            raise ValueError(f"{stem}{suffix} is missing, so the maps cannot be compared")
        return data

    for suffix in (*same, "Variables.kld"):
        if read(FLAT_BASE, suffix) != read(FLAT_PLAIN, suffix):
            problems.append(f"{FLAT_BASE}{suffix} is not {FLAT_PLAIN}{suffix}")
    for suffix in same:
        if read(FLAT_PORTAL, suffix) != read(FLAT_PLAIN, suffix):
            problems.append(f"{FLAT_PORTAL}{suffix} is not {FLAT_PLAIN}{suffix}")
    plain, portal = read(FLAT_PLAIN, "Variables.kld"), read(FLAT_PORTAL, "Variables.kld")
    entries = lambda d: [d[i:i + VARIABLE_ENTRY] for i in range(_header_end(d), len(d), VARIABLE_ENTRY)]  # noqa: E731
    pool_47 = struct.pack("<iiii", POOL_VARIABLE, MOLYNX_ID, MOLYNX_POOL, HUMAN_PLAYER)
    av_47 = struct.pack("<iHHii", AVAILABILITY_VARIABLE, AV_CREATURE, HUMAN_PLAYER, MOLYNX_ID, ENABLE)
    added = [e for e in entries(portal) if e in (pool_47, av_47)]
    rest = [e for e in entries(portal) if e not in (pool_47, av_47)]
    hsz = _header_end(plain)
    sizes = {*range(8, 12), *range(20, 24), *range(hsz - 4, hsz)}
    header_ok = len(portal) >= hsz and all(portal[i] == plain[i] for i in range(hsz) if i not in sizes)
    if sorted(added) != sorted([pool_47, av_47]) or rest != entries(plain) or not header_ok:
        problems.append(f"{FLAT_PORTAL}Variables.kld is not {FLAT_PLAIN}Variables.kld plus exactly the "
                        f"two creature-47 entries (pool 50 and availability ENABLE for player 3)")
    if problems:
        return False, "; ".join(problems)
    return True, (f"{FLAT_BASE} holds {FLAT_PLAIN}'s Map, Things, Players, Triggers, Variables and Globals "
                  f"byte for byte; {FLAT_PORTAL} differs from {FLAT_PLAIN} only in its level file and in "
                  f"its Variables, by exactly the two creature-47 entries")


# ---- the folder ---------------------------------------------------------------------------------

def _map_names(files: list[str]) -> list[str]:
    """Map names that the files of Data/editor/maps point at (name = file name minus a map suffix)."""
    names = set()
    for file_name in files:
        for suffix in MAP_SUFFIXES:
            if file_name.endswith(suffix) and len(file_name) > len(suffix):
                names.add(file_name[:-len(suffix)])
    return sorted(names)


# His creature-panel pictures (stage 2 on): the textures the record's icon1 / portrait resources
# name, GUI\Creatures\Evil\molynx(-bg), as Flame reads them (a name's folders are folders).
PANEL_FOLDERS = ("Data/EngineTextures/GUI", "Data/EngineTextures/GUI/Creatures",
                 "Data/EngineTextures/GUI/Creatures/Evil")
PANEL_FILES = {"molynx.png", "molynx-bg.png"}


def _check_layout(mod_dir: Path, names: list[str], stage: int | None, meshes: set[str],
                  textures: set[str], required: tuple[str, ...] = ()) -> tuple[bool, str]:
    """Rule (g): `meshes` / `textures` are the names record 47 and its meshes use (stage >= 2);
    `required` are maps the folder must hold (the three flat maps of a flat-test-map folder)."""
    maps_rel = MAPS_PATH.as_posix()
    expected_maps = {name + suffix for name in names for suffix in MAP_SUFFIXES}
    stage2 = stage is not None and stage >= 2
    stage3 = stage is not None and stage >= 3
    expected_meshes = {n + ".kmf" for n in meshes} if stage2 else set()
    expected_textures = {t + ".png" for t in textures} if stage2 else set()
    problems, files, mesh_count, texture_count, panel_count = [], [], 0, 0, 0
    for root, dirs, names_here in os.walk(mod_dir):
        here = Path(root).relative_to(mod_dir).as_posix()
        for d in sorted(dirs):
            rel = d if here == "." else f"{here}/{d}"
            if not (rel in ("Data", "Data/editor", maps_rel)
                    or (stage2 and rel in ("Data/Meshes", "Data/EngineTextures") + PANEL_FOLDERS)
                    or (stage3 and rel in ("Data/Text", "Data/Text/Default"))):
                problems.append(f"unexpected folder {rel}")
        for f in sorted(names_here):
            rel = f if here == "." else f"{here}/{f}"
            files.append(rel)
            if rel in ("resource.toml", EDITOR_CREATURES) or (here == maps_rel and f in expected_maps):
                continue
            if stage3 and rel == TEXT_FILE.as_posix():
                continue
            if here == "Data/Meshes" and f in expected_meshes:
                mesh_count += 1
            elif stage2 and here == PANEL_FOLDERS[-1] and f in PANEL_FILES:
                panel_count += 1
            elif here == "Data/EngineTextures" and f in expected_textures:
                texture_count += 1
            elif here == "Data/Meshes" and f.endswith(".kmf") and stage2:
                problems.append(f"unexpected file {rel} (record 47 names no such mesh)")
            elif here == "Data/EngineTextures" and stage2:
                problems.append(f"unexpected file {rel} (no Molynx mesh names this texture)")
            else:
                problems.append(f"unexpected file {rel}")
    if "resource.toml" not in files:
        problems.append("resource.toml is missing")
    if stage2 and panel_count != len(PANEL_FILES):
        problems.append(f"{PANEL_FOLDERS[-1]} must hold " + " and ".join(sorted(PANEL_FILES)))
    if not names:
        problems.append(f"no map files in {maps_rel}")
    present = set(files)
    for name in required:
        if name not in names:
            problems.append(f"map {name} is missing (all {len(MAP_SUFFIXES)} files): a folder with the "
                            f"flat test map holds all of " + ", ".join(required))
    for name in names:
        missing = [name + s for s in MAP_SUFFIXES if f"{maps_rel}/{name}{s}" not in present]
        if missing:
            problems.append(f"map {name} is missing " + ", ".join(missing))
    if problems:
        return False, "; ".join(problems)
    extra = (f", {mesh_count} meshes that record 47 names, {texture_count} textures those "
             f"meshes name and his {panel_count} creature-panel pictures") if stage2 else ""
    return True, (f"holds only resource.toml and {len(names) * len(MAP_SUFFIXES)} map files of "
                  f"{len(names)} map(s) ({', '.join(names)}){extra}")


def _check_resource_toml(mod_dir: Path) -> tuple[bool, str]:
    """Rule (h)."""
    path = mod_dir / "resource.toml"
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        return False, "resource.toml is missing"
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as err:
        return False, f"resource.toml is not valid TOML: {err}"
    table = data.get("resource")
    if not isinstance(table, dict):
        return False, "resource.toml has no [resource] table"
    absent = [k for k in ("name", "version", "authors", "description") if not isinstance(table.get(k), str)]
    if absent:
        return False, f"[resource] lacks the text keys {absent}"
    folder = mod_dir.resolve().name
    if table["name"] != folder:
        return False, f'[resource] name is "{table["name"]}", expected the folder name "{folder}"'
    return True, f'resource.toml is valid TOML with [resource] name = "{table["name"]}", the folder name'


def audit(mod_dir: Path) -> AuditReport:
    """Audit the Molynx mod folder `mod_dir` against the game's source files (see the module docstring)."""
    mod_dir = Path(mod_dir)
    folder = _Lines()
    if not mod_dir.is_dir():
        folder.add(False, "g", "folder", f"{mod_dir} is not a folder")
        return AuditReport(False, [line for _ok, line in folder.items])

    maps_dir = mod_dir / MAPS_PATH
    names = _map_names(sorted(os.listdir(maps_dir)) if maps_dir.is_dir() else [])
    maps = _Lines()
    files = _Files(mod_dir)
    # A folder with "Molynx Test" or "Molynx Test Portal" is a flat-test-map folder: its Dead Walking is
    # the flat test map too, and all three flat maps must be there.
    flat = any(name in (FLAT_PLAIN, FLAT_PORTAL) for name in names)
    results = {name: (_audit_flat if flat and name in FLAT_VARIANTS else _audit_map)(mod_dir, name, maps, files)
               for name in names}
    results[EDITOR_CREATURES] = _audit_editor_creatures(mod_dir, maps, files)
    stages = {name: result[0] for name, result in results.items()}
    meshes = set().union(*(result[1] for result in results.values()))
    textures = set().union(*(result[2] for result in results.values()))

    distinct = {s for s in stages.values() if s is not None}
    stage = next(iter(distinct)) if len(distinct) == 1 and None not in stages.values() else None
    required = tuple(sorted([*FLAT_VARIANTS, *FLAT_WITH])) if flat else ()
    ok, message = _safe(_check_layout, mod_dir, names, stage, meshes, textures, required)
    folder.add(ok, "g", "folder", message)
    if flat:
        ok, message = _safe(_check_flat_variants, mod_dir)
        folder.add(ok, "m13", "folder", message)
    ok, message = _safe(_check_resource_toml, mod_dir)
    folder.add(ok, "h", "folder", message)
    if len(distinct) > 1:
        shown = ", ".join(f"{n}: {s if s else 'unreadable'}" for n, s in stages.items())
        maps.add(False, "b", "folder", f"the maps do not agree on one stage ({shown})")

    items = folder.items + maps.items
    return AuditReport(all(ok for ok, _line in items), [line for _ok, line in items])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a Molynx Flame mod folder.")
    parser.add_argument("folder", type=Path)
    report = audit(parser.parse_args(argv).folder)
    print("\n".join(report.lines))
    print("AUDIT " + ("OK: every rule passed" if report.ok else "FAILED: see the FAIL lines above"))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
