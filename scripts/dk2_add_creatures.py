"""Add creature types 48, 49, 50, ... to a Flame resource folder: the general path for new creatures.

The game ships creature types 1..46. Type 47 (Molynx) works in the unpatched game; types past 46 come
through the portal and keep their "seen" flags only with Flame and the Molynx patches
(flame_patch/src/patches/molynx_creatures.cpp, v2): the portal list (46 slots per player) is
re-mapped so that types from 47 up get slots, and the per-type "seen" flag of types from 47 up is
kept outside the player structure.

`add_creatures(src, out, specs)` copies the resource folder `src` (for example a built Molynx folder,
build_molynx_mod) to `out` and adds one creature type per `NewCreature` spec:

- every creature table (`Data/editor/Creatures.kwd` and each map's `<map>Globals.kld`): a copy of
  that table's own record `clone_of` with creatureId and orderInEditor set to the new id, the
  portal and unique flags set as asked, and the name text set when the spec gives one. Records stay
  in creatureId order; every other record and block is unchanged. A copy of a unique creature,
  which has no panel picture of its own, gets the picture of the creature in its cloneCreatureId.
  New ids must follow the table's highest id with no gap (the game sizes some per-type tables by the number of types and finds
  records by id, so the ids are kept dense: 1..N);
- `Data/Text/Default/Text.str`: each new name is added at the end (the folder's own copy, or the
  game's when the folder has none);
- each map, when the spec has `follow` (an existing creature id): the new creature goes wherever
  `follow` is. Next to every creature thing of `follow` (same owner, the level of that thing unless
  the spec gives one) on the nearest free floor tile; and every creature-pool and
  creature-availability entry of `follow` is repeated for the new id.

The level files' creature id lists (`<map>.kwd`) are left alone: Molynx (47) works without being
listed there, which is the tested path for types beyond 46.

`check_added(src, out, specs)` re-reads `out` independently and returns a list of problems ([] when
it is right). Nothing is ever deleted: `out` must not exist.

Command line:
  python -m scripts.dk2_add_creatures SRC OUT --spec specs.json
  python -m scripts.dk2_add_creatures SRC OUT --elite-placeholders
A spec file is a JSON list of objects with the fields of NewCreature, for example
  [{"creature_id": 48, "clone_of": 35, "name": "Elite Troll", "follow": 47}]

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from scripts.dk2kld_splice import Thing, read_things, write_things
from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2kwd_creatures import FIELDS, RESOURCE_SIZE, Creature, creatures_block, read_creatures

CREATURES = 0xAA
CREATURE_THING = 200
THING_LEVEL_AT, THING_CREATURE_ID_AT, THING_OWNER_AT = 14, 26, 27
CREATURE_POOL, AVAILABILITY, AV_CREATURE = 1, 2, 2
FLAG_AVAILABLE_VIA_PORTAL = 0x400
FLAG_IS_UNIQUE = 0x8000000
DIRT, CLAIMED = 3, 8                     # floor terrain ids (Terrain.kwd)
FLOOR = frozenset({DIRT, CLAIMED})
MAX_ID = 255                             # a creature thing holds its type in one byte
MAPS = Path("Data") / "editor" / "maps"
EDITOR_CREATURES = Path("Data") / "editor" / "Creatures.kwd"
TEXT = Path("Data") / "Text" / "Default" / "Text.str"
TOML = Path("resource.toml")


@dataclass(frozen=True)
class NewCreature:
    creature_id: int
    clone_of: int
    name: str | None = None           # a new name text; None keeps the cloned record's name text
    portal: bool | None = True        # AVAILABLE_VIA_PORTAL; None keeps the cloned record's flag
    unique: bool | None = False       # IS_UNIQUE (never through a portal); None keeps it
    follow: int | None = None         # put it wherever this creature is (things, pool, availability)
    level: int | None = None          # level of its things; None: the level of the followed thing


# The 48/49/50 test: copies of unique ("elite") creatures, made ordinary portal creatures, placed
# next to Molynx and put in the portal pool wherever he is.
ELITE_PLACEHOLDERS = (
    NewCreature(48, 35, "Elite Troll", follow=47),
    NewCreature(49, 44, "Elite Dark Angel", follow=47),
    NewCreature(50, 39, "Elite Salamander", follow=47),
)


def _is_table(rel: Path) -> bool:
    return rel == EDITOR_CREATURES or (rel.parent == MAPS and rel.name.endswith("Globals.kld"))


def _check_specs(specs: list[NewCreature]) -> None:
    if not specs:
        raise ValueError("no creatures to add")
    ids = [s.creature_id for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"creature ids {ids} repeat")
    for s in specs:
        if not 47 <= s.creature_id <= MAX_ID:
            raise ValueError(f"creature id {s.creature_id}: new types are 47..{MAX_ID}")
        if s.level is not None and not 1 <= s.level <= 10:
            raise ValueError(f"creature {s.creature_id}: level {s.level} is not 1..10")
        if s.name is not None and not 0 < len(s.name) <= 31:
            raise ValueError(f"creature {s.creature_id}: name must be 1..31 characters")


def _flags(record: Creature, spec: NewCreature) -> int:
    flags = record.get("flags")
    for wanted, bit in ((spec.portal, FLAG_AVAILABLE_VIA_PORTAL), (spec.unique, FLAG_IS_UNIQUE)):
        if wanted is True:
            flags |= bit
        elif wanted is False:
            flags &= ~bit
    return flags


def _table(data: bytes, specs: list[NewCreature], name_ids: dict[int, int], where: str) -> bytes:
    blocks = read_blocks(data)
    found = [i for i, b in enumerate(blocks) if b.signature == CREATURES]
    if len(found) != 1:
        raise ValueError(f"{where}: expected one Creatures block, found {len(found)}")
    at = found[0]
    records = read_creatures(blocks[at])
    ids = [c.get("creatureId") for c in records]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError(f"{where}: creature ids are not 1..{len(ids)} in order: {ids}")
    want = list(range(len(ids) + 1, len(ids) + 1 + len(specs)))
    if sorted(s.creature_id for s in specs) != want:
        raise ValueError(f"{where}: the table has types 1..{len(ids)}, so the new ids must be {want}")
    by_id = dict(zip(ids, records))
    out = list(records)
    for s in sorted(specs, key=lambda s: s.creature_id):
        if s.clone_of not in by_id:
            raise ValueError(f"{where}: no creature {s.clone_of} to copy for {s.creature_id}")
        c = Creature(bytes(by_id[s.clone_of].raw))
        c.set("creatureId", s.creature_id)
        c.set("orderInEditor", s.creature_id)
        c.set("flags", _flags(c, s))
        if s.creature_id in name_ids:
            c.set("nameStringId", name_ids[s.creature_id])
        if s.name is not None:
            c.set("name", s.name)
        _portrait_from_clone(c, by_id)
        out.append(c)
    blocks[at] = creatures_block(out, blocks[at])
    return write_blocks(blocks)


PORTRAIT = "portraitResource"


def _portrait_from_clone(c: Creature, by_id: dict) -> None:
    """A record with no panel picture of its own (the unique creatures': they show the picture of
    the creature in their cloneCreatureId) gets that creature's whole portrait resource. Without it,
    types beyond 46 show an empty slot in the creature panel (tested 2026-11-08)."""
    base = by_id.get(c.get("cloneCreatureId"))
    if c.resource_name(PORTRAIT) or base is None or not base.resource_name(PORTRAIT):
        return
    at = FIELDS[PORTRAIT].offset
    c.raw[at:at + RESOURCE_SIZE] = base.raw[at:at + RESOURCE_SIZE]


def _map_tiles(data: bytes) -> tuple[int, int, list[bytes]]:
    from scripts.dk2_testmap import header_size, read_map
    m = read_map(data)
    assert header_size(data) == len(m.header)
    return m.width, m.height, list(m.tiles)


def _xy(t: Thing) -> tuple[int, int]:
    return struct.unpack_from("<ii", t.data)


def _free_tile(start, width, height, tiles, taken) -> tuple[int, int]:
    """The nearest floor tile to `start` (breadth-first over floor tiles, 8 neighbours) not in `taken`."""
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) not in taken and tiles[y * width + x][0] in FLOOR:
            return x, y
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = (x + dx, y + dy)
                if n not in seen and 0 <= n[0] < width and 0 <= n[1] < height \
                        and tiles[n[1] * width + n[0]][0] in FLOOR:
                    seen.add(n)
                    queue.append(n)
    raise ValueError(f"no free floor tile near {start}")


def _things(things_kld: bytes, map_kld: bytes, specs: list[NewCreature]) -> bytes:
    block, things = read_things(things_kld)
    width, height, tiles = _map_tiles(map_kld)
    taken = {_xy(t) for t in things if t.type == CREATURE_THING}
    added = []
    for s in specs:
        if s.follow is None:
            continue
        for t in things:
            if t.type != CREATURE_THING or t.data[THING_CREATURE_ID_AT] != s.follow:
                continue
            x, y = _free_tile(_xy(t), width, height, tiles, taken)
            taken.add((x, y))
            data = bytearray(struct.pack("<ii", x, y) + t.data[8:])
            data[THING_CREATURE_ID_AT] = s.creature_id
            if s.level is not None:
                data[THING_LEVEL_AT] = s.level
            added.append(Thing(t.type, bytes(data)))
    if not added:
        return things_kld
    return write_things(block, [*things, *added])


def _entries(data: bytes):
    (block,) = read_blocks(data)
    if len(block.content) % 16:
        raise ValueError("a Variables block whose content is not 16-byte entries")
    return block, [struct.unpack_from("<iiii", block.content, i) for i in range(0, len(block.content), 16)]


def _creature_of(e) -> int | None:
    """The creature id of a creature-pool or creature-availability entry, else None."""
    if e[0] == CREATURE_POOL:
        return e[1]
    if e[0] == AVAILABILITY and (e[1] & 0xFFFF) == AV_CREATURE:
        return e[2]
    return None


def _with_creature(e, cid: int):
    return (e[0], cid, e[2], e[3]) if e[0] == CREATURE_POOL else (e[0], e[1], cid, e[3])


def _variables(data: bytes, specs: list[NewCreature]) -> bytes:
    block, ents = _entries(data)
    added = []
    for s in specs:
        if s.follow is None:
            continue
        if any(_creature_of(e) == s.creature_id for e in ents):
            raise ValueError(f"the map already has Variables entries for creature {s.creature_id}")
        added += [_with_creature(e, s.creature_id) for e in ents if _creature_of(e) == s.follow]
    if not added:
        return data
    return write_blocks([rebuilt_block(block, [struct.pack("<iiii", *e) for e in [*ents, *added]])])


def _text(files: dict[Path, bytes], specs: list[NewCreature]) -> dict[int, int]:
    """Adds the new names to the folder's Text.str (the game's copy when it has none); their ids."""
    named = [s for s in specs if s.name is not None]
    if not named:
        return {}
    from scripts.dk2paths import TEXT_DIR
    from scripts.dk2str import StrFile
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Text.str"
        path.write_bytes(files.get(TEXT) or (TEXT_DIR / "Text.str").read_bytes())
        text = StrFile.read(path, TEXT_DIR / "MBToUni.dat")
    ids = text.append([s.name for s in named])
    files[TEXT] = text.to_bytes()
    return {s.creature_id: i for s, i in zip(named, ids)}


def add_creatures(src: Path, out: Path, specs: list[NewCreature] | tuple = ELITE_PLACEHOLDERS) -> Path:
    src, out, specs = Path(src), Path(out), list(specs)
    if out.exists():
        raise FileExistsError(f"{out} exists; it is never overwritten")
    _check_specs(specs)
    files: dict[Path, bytes] = {p.relative_to(src): p.read_bytes() for p in sorted(src.rglob("*")) if p.is_file()}
    tables = [rel for rel in files if _is_table(rel)]
    if not tables:
        raise ValueError(f"{src} holds no creature table")
    name_ids = _text(files, specs)
    for rel in tables:
        files[rel] = _table(files[rel], specs, name_ids, str(rel))
    maps = sorted({rel.name[:-len("Things.kld")] for rel in files
                   if rel.parent == MAPS and rel.name.endswith("Things.kld")})
    for m in maps:
        things, variables, map_kld = MAPS / f"{m}Things.kld", MAPS / f"{m}Variables.kld", MAPS / f"{m}Map.kld"
        if map_kld in files:
            files[things] = _things(files[things], files[map_kld], specs)
        if variables in files:
            files[variables] = _variables(files[variables], specs)
    if TOML in files:
        files[TOML] = _toml_note(files[TOML], specs)
    out.mkdir(parents=True)
    for rel, data in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    return out


def _toml_note(toml: bytes, specs: list[NewCreature]) -> bytes:
    """resource.toml with one sentence on the added creatures at the end of its description."""
    text = toml.decode("utf-8")
    end = text.rfind('"""')
    if end <= 0 or text.count('"""') < 2:
        return toml
    names = ", ".join(f"{s.creature_id} {s.name or f'(a copy of {s.clone_of})'}" for s in specs)
    note = f"Also adds the creatures {names}.\n"
    return (text[:end] + note + text[end:]).encode("utf-8")


# ---------------------------------------------------------------------------------------------
# Independent check


def check_added(src: Path, out: Path, specs: list[NewCreature] | tuple = ELITE_PLACEHOLDERS) -> list[str]:
    """Problems in `out` (made from `src` by add_creatures with `specs`), or [] when it is right."""
    src, out, specs = Path(src), Path(out), list(specs)
    problems: list[str] = []
    a = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
    b = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    if set(a) - set(b) or set(b) - set(a) - {TEXT}:
        problems.append(f"file lists differ: {sorted(set(a) ^ set(b))}")
    text_names = _names(out)
    for rel in b:
        o = (out / rel).read_bytes()
        s = (src / rel).read_bytes() if (src / rel).exists() else None
        if _is_table(rel):
            problems += _check_table(rel, s, o, specs, text_names)
        elif rel.parent == MAPS and rel.name.endswith("Things.kld"):
            problems += _check_things(rel, s, o, out, specs)
        elif rel.parent == MAPS and rel.name.endswith("Variables.kld"):
            problems += _check_variables(rel, s, o, specs)
        elif rel == TOML:
            if s is not None and not o.decode("utf-8").startswith(s.decode("utf-8")[:s.decode("utf-8").rfind('"""')]):
                problems.append(f"{rel}: more than the description note changed")
        elif rel == TEXT:
            if s is not None and not o.startswith(s[:4]):
                problems.append(f"{rel}: not a text file")
        elif s != o:
            problems.append(f"{rel}: changed")
    return problems


def _names(out: Path):
    """id -> text of the out folder's Text.str, or None when it cannot be read."""
    path = out / TEXT
    if not path.exists():
        return None
    from scripts.dk2paths import TEXT_DIR
    from scripts.dk2str import StrFile
    try:
        return StrFile.read(path, TEXT_DIR / "MBToUni.dat")
    except (OSError, ValueError):
        return None


def _check_table(rel, s, o, specs, text) -> list[str]:
    problems = []
    sb, ob = read_blocks(s), read_blocks(o)
    if [x.signature for x in sb] != [x.signature for x in ob]:
        return [f"{rel}: block list changed"]
    for x, y in zip(sb, ob):
        if x.signature != CREATURES:
            if x.content != y.content:
                problems.append(f"{rel}: block 0x{x.signature:X} changed")
            continue
        old, new = read_creatures(x), read_creatures(y)
        n = len(old)
        if len(new) != n + len(specs):
            problems.append(f"{rel}: {n} -> {len(new)} records, want {n + len(specs)}")
            continue
        if [c.raw for c in new[:n]] != [c.raw for c in old]:
            problems.append(f"{rel}: an existing record changed")
        if [c.get("creatureId") for c in new] != list(range(1, len(new) + 1)):
            problems.append(f"{rel}: ids are not 1..{len(new)}")
        by_id = {c.get("creatureId"): c for c in new}
        for sp in specs:
            c, donor = by_id.get(sp.creature_id), old[sp.clone_of - 1]
            if c is None:
                problems.append(f"{rel}: no record {sp.creature_id}")
                continue
            allowed = set()
            for f in ("name", "creatureId", "orderInEditor", "flags", "nameStringId"):
                fld = c._field(f)
                allowed |= set(range(fld.offset, fld.end))
            at = FIELDS[PORTRAIT].offset
            allowed |= set(range(at, at + RESOURCE_SIZE))
            if not c.resource_name(PORTRAIT):
                problems.append(f"{rel}: record {sp.creature_id} has no panel picture (portraitResource)")
            elif donor.resource_name(PORTRAIT) and c.raw[at:at + RESOURCE_SIZE] != donor.raw[at:at + RESOURCE_SIZE]:
                problems.append(f"{rel}: record {sp.creature_id} changed its donor's panel picture")
            if any(c.raw[k] != donor.raw[k] for k in range(len(c.raw)) if k not in allowed):
                problems.append(f"{rel}: record {sp.creature_id} differs from {sp.clone_of} beyond id/order/flags/name")
            if c.get("orderInEditor") != sp.creature_id:
                problems.append(f"{rel}: record {sp.creature_id} orderInEditor {c.get('orderInEditor')}")
            flags = c.get("flags")
            if sp.portal is not None and bool(flags & FLAG_AVAILABLE_VIA_PORTAL) != sp.portal:
                problems.append(f"{rel}: record {sp.creature_id} portal flag is not {sp.portal}")
            if sp.unique is not None and bool(flags & FLAG_IS_UNIQUE) != sp.unique:
                problems.append(f"{rel}: record {sp.creature_id} unique flag is not {sp.unique}")
            if (flags ^ donor.get("flags")) & ~(FLAG_AVAILABLE_VIA_PORTAL | FLAG_IS_UNIQUE):
                problems.append(f"{rel}: record {sp.creature_id} other flags changed")
            if sp.name is not None and c.get("name") != sp.name:
                problems.append(f"{rel}: record {sp.creature_id} is named {c.get('name')!r}")
            if sp.name is None:
                if c.get("nameStringId") != donor.get("nameStringId"):
                    problems.append(f"{rel}: record {sp.creature_id} name text changed")
            elif text is None or not 0 <= c.get("nameStringId") < len(text.entries) \
                    or text.text(c.get("nameStringId")) != sp.name:
                problems.append(f"{rel}: record {sp.creature_id} name text is not {sp.name!r}")
    return problems


def _creature_things(data: bytes):
    _b, things = read_things(data)
    return [t for t in things if t.type == CREATURE_THING]


def _check_things(rel, s, o, out, specs) -> list[str]:
    old, new = _creature_things(s), _creature_things(o)
    _b, all_old = read_things(s)
    _b, all_new = read_things(o)
    problems = []
    if [t.raw for t in all_new[:len(all_old)]] != [t.raw for t in all_old]:
        problems.append(f"{rel}: existing things changed")
    extra = all_new[len(all_old):]
    want = sum(1 for sp in specs if sp.follow is not None
               for t in old if t.data[THING_CREATURE_ID_AT] == sp.follow)
    if len(extra) != want:
        problems.append(f"{rel}: {len(extra)} things added, want {want}")
    map_path = out / MAPS / (rel.name[:-len("Things.kld")] + "Map.kld")
    width, height, tiles = _map_tiles(map_path.read_bytes())
    spots = [_xy(t) for t in new]
    if len(spots) != len(set(spots)):
        problems.append(f"{rel}: two creatures share a tile")
    by_id = {sp.creature_id: sp for sp in specs}
    for t in extra:
        cid = t.data[THING_CREATURE_ID_AT]
        sp = by_id.get(cid)
        x, y = _xy(t)
        if t.type != CREATURE_THING or sp is None:
            problems.append(f"{rel}: an added thing is not a new creature")
            continue
        if not (0 <= x < width and 0 <= y < height) or tiles[y * width + x][0] not in FLOOR:
            problems.append(f"{rel}: creature {cid} at {(x, y)} is not on a floor tile")
        followed = [f for f in old if f.data[THING_CREATURE_ID_AT] == sp.follow
                    and f.data[THING_OWNER_AT] == t.data[THING_OWNER_AT]]
        if not followed:
            problems.append(f"{rel}: creature {cid} has an owner that {sp.follow} does not")
            continue
        near = min(followed, key=lambda f: max(abs(_xy(f)[0] - x), abs(_xy(f)[1] - y)))
        if max(abs(_xy(near)[0] - x), abs(_xy(near)[1] - y)) > 4:
            problems.append(f"{rel}: creature {cid} at {(x, y)} is not next to {sp.follow}")
        want_level = sp.level if sp.level is not None else near.data[THING_LEVEL_AT]
        if t.data[THING_LEVEL_AT] != want_level:
            problems.append(f"{rel}: creature {cid} is level {t.data[THING_LEVEL_AT]}, want {want_level}")
    return problems


def _check_variables(rel, s, o, specs) -> list[str]:
    _b, old = _entries(s)
    _b, new = _entries(o)
    problems = []
    if new[:len(old)] != old:
        problems.append(f"{rel}: existing entries changed")
    extra = new[len(old):]
    for sp in specs:
        if sp.follow is None:
            continue
        want = sorted(_with_creature(e, sp.creature_id) for e in old if _creature_of(e) == sp.follow)
        got = sorted(e for e in extra if _creature_of(e) == sp.creature_id)
        if want != got:
            problems.append(f"{rel}: entries of {sp.creature_id} {got} != {want}")
    if any(_creature_of(e) not in {sp.creature_id for sp in specs} for e in extra):
        problems.append(f"{rel}: an added entry is not for a new creature")
    return problems


def load_specs(path: Path) -> list[NewCreature]:
    return [NewCreature(**d) for d in json.loads(Path(path).read_text())]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Add creature types 48, 49, ... to a Flame resource folder.")
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec", type=Path, help="JSON list of NewCreature fields")
    g.add_argument("--elite-placeholders", action="store_true", help="48/49/50: elite copies next to Molynx")
    args = ap.parse_args(argv)
    specs = load_specs(args.spec) if args.spec else list(ELITE_PLACEHOLDERS)
    out = add_creatures(args.src, args.out, specs)
    problems = check_added(args.src, out, specs)
    print("\n".join(problems) if problems else
          f"ADDED OK: creatures {[s.creature_id for s in specs]} in {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
