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
  (g) the folder holds nothing else: `resource.toml`, the seven files of each map and, from stage 2,
      exactly the `Data/Meshes/<name>.kmf` files record 47 names and the `Data/EngineTextures/<t>.png`
      files those meshes name (no unreferenced mesh or texture).
  (h) `resource.toml` is valid TOML with a [resource] table of name, version, authors, description,
      and its name is the name of the folder.

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
  3  + hp, meleeDamage, fear, threat, mass, pay above the Bile Demon's, icon1Resource
     `gui\\creatures\\evil\\molynx`, portraitResource `gui\\creatures\\evil\\molynx-bg`, and the five
     text id fields changed.

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
from scripts.dk2paths import GAME_DIR, MESHES_DIR
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
STATS = ("hp", "meleeDamage", "fear", "threat", "mass", "pay")
TEXT_ID_FIELDS = ("nameStringId", "tooltipStringId", "generalDescriptionStringId",
                  "strengthStringId", "weaknessStringId")
MAPS_PATH = Path("Data") / "editor" / "maps"

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
# middle frame of staggering; ElecPose, the peak of getting_hit), like the game's creatures. A slot
# not listed takes "stand"; a slot empty in the Bile Demon stays empty; a static mesh slot (type 5)
# whose entry is staggering (dejectedPose in some maps) takes StunnedPose (MESH_SLOT_FILES).
SLOT_FILES = {
    "animWalkResource": "walk", "animRunResource": "running",
    "animDraggedPoseResource": "DraggedPose", "animRecoilHffResource": "getting_hit",
    "animRecoilHfbResource": "being_slapped", "animMelee1Resource": "attack",
    "animMelee2Resource": "attack", "animSpecialResource": "attack", "animDieResource": "die",
    "animDiePoseResource": "DiePose", "animHappyResource": "happy", "animAngryResource": "angry",
    "animSwingResource": "being_picked_up", "animSleepResource": "sleeping",
    "animEatResource": "eating", "animEatCorpseResource": "eating", "animDrinkResource": "eating",
    "animBuildingResource": "manufacturing", "animPrayResource": "praying",
    "animFallbackResource": "knockback", "animGetUpResource": "waking_up",
    "animDanceResource": "celebrating", "animDrunkResource": "staggering",
    "animDrunk2Resource": "staggering", "animStunnedPoseResource": "StunnedPose",
    "animDejectedPoseResource": "staggering", "animIdle1Resource": "stand",
    "animIdle2Resource": "fidgeting", "drunkIdle": "fidgeting",
    "animPoseFrameResource": "PoseFrame", "animWalk2Resource": "patrolling",
    "animEntranceResource": "walk_once", "animWalkbackResource": "walk",
    "animResearchResource": "interacting_with_objects", "animLootResource": "interacting_with_objects",
    "animMagicResource": "getting_hit", "animTortureResource": "getting_hit",
    "animElecResource": "ElecPose", "animElectrocuteResource": "getting_hit",
}
MESH_SLOT_FILES = {"staggering": "StunnedPose"}
DEFAULT_FILE = "stand"
WRAP_EXCEPTIONS = {
    "animSleepResource": ("Molynx-sleeping", "a one-way clip (he lies down and stays down), so it "
                          "plays once and is not looped"),
}


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    lines: list[str]


class _Lines:
    def __init__(self) -> None:
        self.items: list[tuple[bool, str]] = []

    def add(self, ok: bool, rule: str, scope: str, message: str) -> None:
        self.items.append((ok, f"{'PASS' if ok else 'FAIL'} ({rule}) {scope}: {message}"))


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
    stage2 = _STAGE1_FIELDS | renamed
    stage3 = stage2 | set(STATS) | {"icon1Resource", "portraitResource"} | set(TEXT_ID_FIELDS)
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
    if stage >= 3:
        for stat in STATS:
            if not molynx.get(stat) > bile.get(stat):
                problems.append(f"{stat} is {molynx.get(stat)}, not above the Bile Demon's {bile.get(stat)}")
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


def _texture_problems(texture: str, files: "_Files") -> list[str]:
    file_name = texture + ".png"
    shown = f"Data/EngineTextures/{file_name}"
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
    if stage >= 3:
        problems.append("stage 3 is not audited yet (until Task 10): its icon and portrait files "
                        "(icon1Resource, portraitResource) and texts are not checked, so a stage-3 "
                        "folder never passes")
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
    """Rules (a) to (f) for one map; returns (the stage found in record 47 or None, the Molynx mesh
    names its record uses, the texture names those meshes use)."""
    try:
        source_dir = map_source_dir(name)
    except FileNotFoundError as err:
        for rule in "abcdef":
            lines.add(False, rule, name, f"no source files to compare with: {err}")
        return None, set(), set()
    maps = mod_dir / MAPS_PATH
    info = None

    mod_globals = _read(maps / f"{name}Globals.kld")
    src_globals = _read(source_dir / f"{name}Globals.kld")
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

    if info:
        stage, molynx, bile, _size = info
        ok, message, *rest = _safe(_check_assets, stage, molynx, bile, files)
        lines.add(ok, "f", name, message)
        meshes, textures = rest if rest else (set(), set())
        return stage, meshes, textures
    lines.add(False, "f", name, "not checked because record 47 could not be read")
    return None, set(), set()


# ---- the folder ---------------------------------------------------------------------------------

def _map_names(files: list[str]) -> list[str]:
    """Map names that the files of Data/editor/maps point at (name = file name minus a map suffix)."""
    names = set()
    for file_name in files:
        for suffix in MAP_SUFFIXES:
            if file_name.endswith(suffix) and len(file_name) > len(suffix):
                names.add(file_name[:-len(suffix)])
    return sorted(names)


def _check_layout(mod_dir: Path, names: list[str], stage: int | None, meshes: set[str],
                  textures: set[str]) -> tuple[bool, str]:
    """Rule (g): `meshes` / `textures` are the names record 47 and its meshes use (stage >= 2)."""
    maps_rel = MAPS_PATH.as_posix()
    expected_maps = {name + suffix for name in names for suffix in MAP_SUFFIXES}
    stage2 = stage is not None and stage >= 2
    expected_meshes = {n + ".kmf" for n in meshes} if stage2 else set()
    expected_textures = {t + ".png" for t in textures} if stage2 else set()
    problems, files, mesh_count, texture_count = [], [], 0, 0
    for root, dirs, names_here in os.walk(mod_dir):
        here = Path(root).relative_to(mod_dir).as_posix()
        for d in sorted(dirs):
            rel = d if here == "." else f"{here}/{d}"
            if not (rel in ("Data", "Data/editor", maps_rel)
                    or (stage2 and rel in ("Data/Meshes", "Data/EngineTextures"))):
                problems.append(f"unexpected folder {rel}")
        for f in sorted(names_here):
            rel = f if here == "." else f"{here}/{f}"
            files.append(rel)
            if rel == "resource.toml" or (here == maps_rel and f in expected_maps):
                continue
            if here == "Data/Meshes" and f in expected_meshes:
                mesh_count += 1
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
    if not names:
        problems.append(f"no map files in {maps_rel}")
    present = set(files)
    for name in names:
        missing = [name + s for s in MAP_SUFFIXES if f"{maps_rel}/{name}{s}" not in present]
        if missing:
            problems.append(f"map {name} is missing " + ", ".join(missing))
    if problems:
        return False, "; ".join(problems)
    extra = (f", {mesh_count} meshes that record 47 names and {texture_count} textures those "
             "meshes name") if stage2 else ""
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
    results = {name: _audit_map(mod_dir, name, maps, files) for name in names}
    stages = {name: result[0] for name, result in results.items()}
    meshes = set().union(*(result[1] for result in results.values()))
    textures = set().union(*(result[2] for result in results.values()))

    distinct = {s for s in stages.values() if s is not None}
    stage = next(iter(distinct)) if len(distinct) == 1 and None not in stages.values() else None
    ok, message = _safe(_check_layout, mod_dir, names, stage, meshes, textures)
    folder.add(ok, "g", "folder", message)
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
