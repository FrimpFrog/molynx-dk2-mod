"""Independent audit of a Molynx Flame mod folder: `audit(mod_dir) -> AuditReport`.

The audit re-reads every file of the folder from disk and compares it with the game's own source
files, so it proves what is on disk instead of trusting the builder. It never calls the builder or the
record module and takes no expectation from them: what Molynx's record must look like at each stage
is written out below from the stage definitions. Run it on a folder with
`python -m scripts.audit_molynx_mod <folder>`; the exit code is 0 only when every line passes.

The stage and the maps are inferred from the folder. The maps are the `<name>` of the files in
`Data/editor/maps`; each map's source is `map_source_dir(name)` (the Quuz copy when the user's game
loads that one, else the editor folder). The stage comes from record 47 (see `_STAGE_FIELDS`).

Rules, each one PASS or FAIL line per map (per folder for g and h):

  (a) every block of the patched `<map>Globals.kld` except Creatures is byte-identical to the source
      (same blocks in the same order, header and content).
  (b) the Creatures block has 47 records; the first 46 are identical to the source; record 47 is the
      source's Bile Demon with exactly the fields of one stage changed (and only to the ruled values).
  (c) header invariants: every block header of the Globals file is consistent (signature, the
      constants 4 / 28 / 1, check words signature + 1 and + 2, file size and content size match the
      content), the Creatures block holds whole 5537-byte records numbered by its item count, and its
      header is the source's apart from file size, item count and content size.
  (d) `<map>Things.kld` differs from its source in exactly one byte: the creature id of a keeper's
      Bile Demon (type-200 record), 3 -> 47.
  (e) every other map file is byte-identical to its source.
  (f) animation (mesh) names: every animation slot name of record 47 that differs from the Bile
      Demon's exists as `Data/Meshes/<name>.kmf` in the folder (exact case, not empty); every name left
      as the Bile Demon's exists in the game's Meshes.WAD. At stage 3 the icon and texture files are
      not checked yet, which is a FAIL, so a stage-3 folder cannot pass as verified.
  (g) the folder holds nothing else: `resource.toml`, the seven files of each map and, from stage 2,
      `Data/Meshes/*.kmf` and `Data/EngineTextures/**/*.png`.
  (h) `resource.toml` is valid TOML with a [resource] table of name, version, authors, description.

The stages (written independently of the builder, from the Molynx design):
  1  record 47 = Bile Demon with name "Molynx", creatureId 47, orderInEditor = the source's largest + 1.
  2  + every animation slot that is non-empty in the Bile Demon (not unk_1) renamed `Molynx-<stem>`;
     empty slots stay empty and the 20 bytes after a slot's 64-byte name stay the Bile Demon's.
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
from scripts.dk2kwd_blocks import HEADER_SIZE, read_blocks
from scripts.dk2kwd_creatures import ANIM_SLOTS, FIELDS, RESOURCE_NAME_SIZE, RESOURCE_SIZE, Creature
from scripts.dk2paths import GAME_DIR, MESHES_DIR
from scripts.dk2wad import read_wad

CREATURES = 0xAA
THINGS = 0xBE
CREATURE_THING = 200
KEEPER_PLAYERS = range(3, 8)
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

_PARSE_ERRORS = (ValueError, KeyError, IndexError, struct.error, OSError)
_STAGE1_FIELDS = frozenset({"name", "creatureId", "orderInEditor"})
_RESOURCE_TAIL = slice(RESOURCE_NAME_SIZE, RESOURCE_SIZE)


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


def _resource_name(record: Creature, slot: str) -> str:
    return record.resource_name(slot)


def _anim_slots() -> list[str]:
    return [slot for slot in ANIM_SLOTS if slot != "unk_1"]  # unk_1 holds no name; never touched


def _stage_field_sets(bile: Creature) -> dict[int, frozenset[str]]:
    """The set of record fields in which record 47 must differ from the Bile Demon, per stage."""
    renamed = frozenset(slot for slot in _anim_slots() if _resource_name(bile, slot))
    stage2 = _STAGE1_FIELDS | renamed
    stage3 = stage2 | set(STATS) | {"icon1Resource", "portraitResource"} | set(TEXT_ID_FIELDS)
    return {1: _STAGE1_FIELDS, 2: stage2, 3: stage3}


def _name_field_bytes(record: Creature) -> bytes:
    start = FIELDS["name"].offset
    return bytes(record.raw[start:start + NAME_FIELD_SIZE])


def _value_problems(molynx: Creature, bile: Creature, stage: int, order: int) -> list[str]:
    """Where record 47 has the right fields changed but not the ruled values."""
    problems = []
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
            if not _resource_name(bile, slot):
                continue
            name = _resource_name(molynx, slot)
            raw = _resource_bytes(molynx, slot)
            if not name.startswith(MOLYNX_PREFIX) or len(name) == len(MOLYNX_PREFIX):
                problems.append(f"{slot} is {name!r}, expected a name starting with {MOLYNX_PREFIX!r}")
            elif raw[len(name):RESOURCE_NAME_SIZE].strip(b"\0"):
                problems.append(f"{slot} has bytes after the NUL of its name")
            if raw[_RESOURCE_TAIL] != _resource_bytes(bile, slot)[_RESOURCE_TAIL]:
                problems.append(f"{slot}: the last 20 bytes after its name differ from the Bile Demon's")
    if stage >= 3:
        for stat in STATS:
            if not molynx.get(stat) > bile.get(stat):
                problems.append(f"{stat} is {molynx.get(stat)}, not above the Bile Demon's {bile.get(stat)}")
        for slot, expected in (("icon1Resource", ICON), ("portraitResource", PORTRAIT)):
            raw = _resource_bytes(molynx, slot)
            if _resource_name(molynx, slot) != expected:
                problems.append(f"{slot} is {_resource_name(molynx, slot)!r}, expected {expected!r}")
            elif raw[_RESOURCE_TAIL] != _resource_bytes(bile, slot)[_RESOURCE_TAIL]:
                problems.append(f"{slot}: the last 20 bytes after its name differ from the Bile Demon's")
    return problems


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


def _check_creatures(src_blocks, mod_blocks) -> tuple[bool, str, tuple | None]:
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
    stage = next((s for s, fields in sets.items() if differing == fields), None)
    if stage is None:
        closest = min(sets, key=lambda s: len(differing ^ sets[s]))
        extra, missing = sorted(differing - sets[closest]), sorted(sets[closest] - differing)
        return False, (f"record 47 differs from the Bile Demon in a set of fields that is no stage's; "
                       f"closest is stage {closest}: unexpected {extra}, missing {missing}"), None
    order = max(c.get("orderInEditor") for c in source) + 1
    problems = _value_problems(molynx, bile, stage, order)
    if problems:
        return False, f"record 47 (stage {stage}): " + "; ".join(problems), None
    detail = (", ".join(sorted(differing)) if stage == 1
              else f"{len(differing)} fields (stage {stage} set)")
    return True, (f"{len(records)} records; the first {SOURCE_RECORDS} are identical to the source; "
                  f"record 47 is Molynx at stage {stage}, differing from the Bile Demon only in {detail}"
                  ), (stage, molynx, bile, size)


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
    """Rule (d)."""
    if len(src) != len(mod):
        return False, f"is {len(mod)} bytes, the source {len(src)}"
    diffs = [i for i in range(len(src)) if src[i] != mod[i]]
    if len(diffs) != 1:
        return False, (f"differs from the source in {len(diffs)} bytes, expected exactly 1 "
                       f"(the creature id of one Bile Demon)")
    at = diffs[0]
    records = _thing_records(src)
    creature_things = [r for r in records if r[1] == CREATURE_THING]
    hit = [(n, r) for n, r in enumerate(creature_things) if r[0] + 34 == at]
    if not hit:
        return False, f"the changed byte {at} is not the creature id byte of any creature thing"
    n, (pos, _kind, size) = hit[0]
    owner = src[pos + 35]
    if size != 28:
        return False, f"the creature thing at byte {pos} has {size} data bytes, expected 28"
    if src[at] != BILE_DEMON_ID:
        return False, (f"the changed creature thing #{n} is creature id {src[at]}, not a Bile Demon "
                       f"({BILE_DEMON_ID})")
    if mod[at] != MOLYNX_ID:
        return False, (f"the creature id of thing #{n} was changed to {mod[at]}, expected {MOLYNX_ID}")
    if owner not in KEEPER_PLAYERS:
        return False, f"the changed creature thing #{n} is owned by player {owner}, not a keeper (3..7)"
    return True, (f"differs from the source in exactly 1 byte: creature thing #{n} (byte {at}), "
                  f"Bile Demon {BILE_DEMON_ID} -> {MOLYNX_ID}, owned by keeper player {owner}")


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


def _check_assets(mod_dir: Path, stage: int, molynx: Creature, bile: Creature) -> tuple[bool, str]:
    """Rule (f). Task 8 extends this with the textures and icon files (the `unchecked` list)."""
    game = _game_mesh_names()
    problems, unchecked = [], []
    changed = sorted({_resource_name(molynx, slot) for slot in _anim_slots()
                      if _resource_name(molynx, slot) != _resource_name(bile, slot)})
    kept = sorted({_resource_name(molynx, slot) for slot in _anim_slots()
                   if _resource_name(molynx, slot) and _resource_name(molynx, slot) == _resource_name(bile, slot)})

    meshes = mod_dir / "Data" / "Meshes"
    listing = set(os.listdir(meshes)) if meshes.is_dir() else set()
    folded = {n.lower(): n for n in listing}
    for name in changed:
        file_name = name + ".kmf"
        if file_name in listing:
            if (meshes / file_name).stat().st_size == 0:
                problems.append(f"Data/Meshes/{file_name} is empty")
        elif file_name.lower() in folded:
            problems.append(f"Data/Meshes/{file_name} is missing (only {folded[file_name.lower()]!r}, "
                            "other case, exists)")
        else:
            problems.append(f"Data/Meshes/{file_name} is missing")
    if kept:
        if game is None:
            problems.append("the game's Meshes.WAD (or the extracted Meshes folder) was not found, so "
                            f"the {len(kept)} unchanged Bile Demon names cannot be checked")
        else:
            problems += [f"{name}.kmf is not in the game's Meshes.WAD" for name in kept
                         if name + ".kmf" not in game]
    if stage >= 3:
        unchecked.append("the icon and texture files (icon1Resource, portraitResource, "
                         "Data/EngineTextures) are not checked by this audit version yet")
    if problems or unchecked:
        return False, "; ".join(problems + [u for u in unchecked])
    return True, (f"every animation slot name resolves: {len(changed)} Molynx meshes in Data/Meshes "
                  f"and {len(kept)} Bile Demon names in the game's Meshes.WAD")


def _safe(check, *args) -> tuple:
    """Run a check; a file that does not parse becomes a FAIL result instead of a crash."""
    try:
        return check(*args)
    except _PARSE_ERRORS as err:
        return (False, f"could not be checked: {err}")


def _audit_map(mod_dir: Path, name: str, lines: _Lines) -> int | None:
    """Rules (a) to (f) for one map; returns the stage found in record 47 (None if not found)."""
    try:
        source_dir = map_source_dir(name)
    except FileNotFoundError as err:
        for rule in "abcdef":
            lines.add(False, rule, name, f"no source files to compare with: {err}")
        return None
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
            ok, message, *rest = _safe(_check_creatures, src_blocks, mod_blocks)
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
        ok, message = _safe(_check_assets, mod_dir, stage, molynx, bile)
        lines.add(ok, "f", name, message)
        return stage
    lines.add(False, "f", name, "not checked because record 47 could not be read as a stage")
    return None


# ---- the folder ---------------------------------------------------------------------------------

def _map_names(files: list[str]) -> list[str]:
    """Map names that the files of Data/editor/maps point at (name = file name minus a map suffix)."""
    names = set()
    for file_name in files:
        for suffix in MAP_SUFFIXES:
            if file_name.endswith(suffix) and len(file_name) > len(suffix):
                names.add(file_name[:-len(suffix)])
    return sorted(names)


def _check_layout(mod_dir: Path, names: list[str], stage: int | None) -> tuple[bool, str]:
    """Rule (g)."""
    maps_rel = MAPS_PATH.as_posix()
    expected_maps = {name + suffix for name in names for suffix in MAP_SUFFIXES}
    stage2 = stage is not None and stage >= 2
    problems, files, mesh_count, texture_count = [], [], 0, 0
    for root, dirs, names_here in os.walk(mod_dir):
        here = Path(root).relative_to(mod_dir).as_posix()
        for d in dirs:
            rel = d if here == "." else f"{here}/{d}"
            if not (rel in ("Data", "Data/editor", maps_rel)
                    or (stage2 and (rel in ("Data/Meshes", "Data/EngineTextures")
                                    or rel.startswith("Data/EngineTextures/")))):
                problems.append(f"unexpected folder {rel}")
        for f in names_here:
            rel = f if here == "." else f"{here}/{f}"
            files.append(rel)
            if rel == "resource.toml" or (here == maps_rel and f in expected_maps):
                continue
            if stage2 and here == "Data/Meshes" and f.endswith(".kmf") and len(f) > 4:
                mesh_count += 1
            elif stage2 and here.startswith("Data/EngineTextures") and f.endswith(".png") and len(f) > 4:
                texture_count += 1
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
    extra = f", {mesh_count} meshes and {texture_count} textures" if stage2 else ""
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
    if not table["name"]:
        return False, "[resource] name is empty"
    return True, f'resource.toml is valid TOML with [resource] name = "{table["name"]}"'


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
    stages = {name: _audit_map(mod_dir, name, maps) for name in names}

    distinct = {s for s in stages.values() if s is not None}
    stage = next(iter(distinct)) if len(distinct) == 1 and None not in stages.values() else None
    ok, message = _safe(_check_layout, mod_dir, names, stage)
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
