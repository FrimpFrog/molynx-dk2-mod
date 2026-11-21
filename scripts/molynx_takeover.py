"""Move Molynx from the new creatureId 47 into an existing, unused creature slot (default 46, "Jack").

No longer the shipped form (2026-11-08): Molynx runs as creature 47 in the game (build 260228 of
Flame with the Molynx patches, flame_patch/), Skirmish -> Dead Walking tested: he loads by the
heart, plays, and can be possessed. Slot 46 is not usable for him anyway: the game treats type 46
as the Jack In The Box (the spell-target search CThing sub_44DEC0 skips it, so Possession cannot
reach him, and processTakeDamage makes it explode with shot 65 when it dies). The notes below are
the reasoning of 2026-11-07.

Why. Creature 47 cannot work in the unpatched game: the player structure (CPlayer, 4389 bytes, Flame
mapping DKII_EXE_v170.sgmap) holds `fB57_creatureEntranceInfo[46][9]`, one entry per creature type
1..46, used by the portal code (Entrance_cpp_4DBF90) and `fFCE_creaturesUsed[47]` (types 0..46).
A 47th type indexes past the end of both. Test B (2026-11-07): with record 47 in the base table and
the map, the Skirmish lobby loads, then the game hangs as the level starts, the same with the
slot-only folder (an exact Bile Demon copy), so the type id is the cause, not his model. Every
table in the game is sized for types 1..46, so Molynx takes one of those slots instead. Type 46,
"Jack", is a creature the game never shows: its panel picture is missing from the game
(MESSAGE.LOG: "Unable to Load Bitmap Resource, 'gui\\creatures\\evil\\Jack-bg'").

`takeover(src, out, target)` copies a built Molynx folder (build_molynx_mod, creatureId 47) to
`out` and changes it so that Molynx is creature `target`:

- every creature table (each map's `<map>Globals.kld` and `Data/editor/Creatures.kwd`): the record
  of `target` is replaced by record 47 with its creatureId set to `target` and the replaced
  record's orderInEditor; record 47 is dropped (46 records again, every other byte unchanged);
- every `<map>Things.kld`: creature things with creature id 47 get `target`;
- every `<map>Variables.kld`: creature-pool and creature-availability entries of 47 get `target`
  (an entry of `target` that would then be repeated for the same player is dropped);
- every level `<map>.kwd` whose map holds Molynx (a thing or a pool entry) lists `target` in its
  creature id list (the count at byte 22 and the file size at byte 8 follow);
- every other file (meshes, textures, the other map files) is copied unchanged.

`check_takeover(out, target)` re-reads the result independently and returns a list of problems.
Nothing is ever deleted: `out` must not exist.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

from scripts.dk2kld_splice import find_creature_things, retarget_creature_thing
from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2kwd_creatures import Creature, creatures_block, read_creatures

OLD_ID = 47
DEFAULT_TARGET = 46            # "Jack": no panel picture in the game, never placed by it
CREATURES = 0xAA
VARIABLES = 0xE6
CREATURE_POOL, AVAILABILITY, AV_CREATURE = 1, 2, 2
MAPS = Path("Data") / "editor" / "maps"
EDITOR_CREATURES = Path("Data") / "editor" / "Creatures.kwd"
LEVEL_SIZE_AT, LEVEL_COUNT_AT = 8, 22


def _creature_table(data: bytes, target: int) -> bytes:
    blocks = read_blocks(data)
    (at,) = [i for i, b in enumerate(blocks) if b.signature == CREATURES]
    records = read_creatures(blocks[at])
    ids = [c.get("creatureId") for c in records]
    if ids.count(OLD_ID) != 1 or ids.count(target) != 1:
        raise ValueError(f"expected one record {OLD_ID} and one record {target}, found ids {ids}")
    molynx = Creature(records[ids.index(OLD_ID)].raw)
    replaced = records[ids.index(target)]
    molynx.set("creatureId", target)
    molynx.set("orderInEditor", replaced.get("orderInEditor"))
    out = [molynx if c.get("creatureId") == target else c
           for c in records if c.get("creatureId") != OLD_ID]
    blocks[at] = creatures_block(out, blocks[at])
    return write_blocks(blocks)


def _things(data: bytes, target: int) -> tuple[bytes, int]:
    moved = 0
    for i, ref in enumerate(find_creature_things(data)):
        if ref.creature_id == OLD_ID:
            data = retarget_creature_thing(data, i, target)
            moved += 1
    return data, moved


def _var_key(e: tuple) -> tuple | None:
    """("pool" | "av", creature id, player) of a creature-pool or creature-availability entry."""
    if e[0] == CREATURE_POOL:
        return ("pool", e[1], e[3])
    if e[0] == AVAILABILITY and (e[1] & 0xFFFF) == AV_CREATURE:
        return ("av", e[2], e[1] >> 16)
    return None


def _renamed(e: tuple, target: int) -> tuple:
    k = _var_key(e)
    if k and k[1] == OLD_ID:
        return (e[0], target, e[2], e[3]) if k[0] == "pool" else (e[0], e[1], target, e[3])
    return e


def _variables(data: bytes, target: int) -> tuple[bytes, int]:
    """Molynx's entries (47) become `target`'s; the slot's own entries for the same keys are dropped."""
    (block,) = read_blocks(data)
    ents = [struct.unpack_from("<iiii", block.content, i) for i in range(0, len(block.content), 16)]
    molynx_keys = {(k[0], target, k[2]) for k in map(_var_key, ents) if k and k[1] == OLD_ID}
    out, seen, moved = [], set(), 0
    for e in ents:
        k = _var_key(e)
        if k in molynx_keys:          # the slot's own entry, replaced by Molynx's
            continue
        if k and k[1] == OLD_ID:
            moved += 1
        e = _renamed(e, target)
        k = _var_key(e)
        if k is not None:
            if k in seen:
                raise ValueError(f"Variables entry {k} would be repeated")
            seen.add(k)
        out.append(struct.pack("<iiii", *e))
    return write_blocks([rebuilt_block(block, out)]), moved


def _level_ids(data: bytes) -> list[int]:
    count = struct.unpack_from("<H", data, LEVEL_COUNT_AT)[0]
    return list(struct.unpack_from(f"<{count}I", data, len(data) - 4 * count))


def _level_with(data: bytes, target: int) -> bytes:
    ids = _level_ids(data)
    if target in ids:
        return data
    count = len(ids) + 1
    out = bytearray(data + struct.pack("<I", target))
    struct.pack_into("<H", out, LEVEL_COUNT_AT, count)
    struct.pack_into("<I", out, LEVEL_SIZE_AT, len(out))
    return bytes(out)


def takeover(src: Path, out: Path, target: int = DEFAULT_TARGET, name: str | None = None) -> Path:
    src, out = Path(src), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; it is never overwritten")
    if not 1 <= target <= 46:
        raise ValueError("the target slot must be a shipped creature type, 1..46")
    files: dict[Path, bytes] = {}
    for p in sorted(src.rglob("*")):
        if p.is_file():
            files[p.relative_to(src)] = p.read_bytes()
    maps = sorted({p.name[:-len("Globals.kld")] for p in files if p.parent == MAPS and p.name.endswith("Globals.kld")})
    for rel in list(files):
        if rel == EDITOR_CREATURES or (rel.parent == MAPS and rel.name.endswith("Globals.kld")):
            files[rel] = _creature_table(files[rel], target)
    for m in maps:
        things, moved_t = _things(files[MAPS / f"{m}Things.kld"], target)
        files[MAPS / f"{m}Things.kld"] = things
        variables, moved_v = _variables(files[MAPS / f"{m}Variables.kld"], target)
        files[MAPS / f"{m}Variables.kld"] = variables
        if moved_t or moved_v:
            files[MAPS / f"{m}.kwd"] = _level_with(files[MAPS / f"{m}.kwd"], target)
    toml = files[Path("resource.toml")].decode("utf-8")
    folder = name or src.name
    toml = toml.replace(f'name = "{src.name}"', f'name = "{folder}"')
    toml = toml.replace("a 47th creature", f"creature {target} (Jack's unused slot)")
    files[Path("resource.toml")] = toml.encode("utf-8")
    out.mkdir(parents=True)
    for rel, data in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    return out


def check_takeover(src: Path, out: Path, target: int = DEFAULT_TARGET) -> list[str]:
    """Independent re-read of `out` against the 47-build `src`: problems, or [] when it is right."""
    src, out = Path(src), Path(out)
    problems = []
    a = sorted(p.relative_to(src) for p in src.rglob("*") if p.is_file())
    b = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    if a != b:
        problems.append(f"file lists differ: {sorted(set(a) ^ set(b))}")
    for rel in b:
        s, o = (src / rel).read_bytes(), (out / rel).read_bytes()
        is_table = rel == EDITOR_CREATURES or (rel.parent == MAPS and rel.name.endswith("Globals.kld"))
        if is_table:
            sb, ob = read_blocks(s), read_blocks(o)
            if [x.signature for x in sb] != [x.signature for x in ob]:
                problems.append(f"{rel}: block list changed")
                continue
            for x, y in zip(sb, ob):
                if x.signature != CREATURES:
                    if x.content != y.content:
                        problems.append(f"{rel}: block 0x{x.signature:X} changed")
                    continue
                sr, orr = read_creatures(x), read_creatures(y)
                if len(sr) != 47 or len(orr) != 46:
                    problems.append(f"{rel}: {len(sr)} -> {len(orr)} records (want 47 -> 46)")
                    continue
                for i in range(46):
                    want_id = sr[i].get("creatureId")
                    if want_id == target:
                        m = orr[i]
                        mol = sr[46]
                        if m.get("creatureId") != target or m.get("name") != "Molynx":
                            problems.append(f"{rel}: slot {target} is {m.get('name')!r} id {m.get('creatureId')}")
                        diff = [k for k in range(len(m.raw)) if m.raw[k] != mol.raw[k]]
                        allowed = set()
                        for f in ("creatureId", "orderInEditor"):
                            fld = Creature(m.raw)._field(f)
                            allowed |= set(range(fld.offset, fld.end))
                        if set(diff) - allowed:
                            problems.append(f"{rel}: slot {target} differs from record 47 outside id/order")
                        if m.get("orderInEditor") != sr[i].get("orderInEditor"):
                            problems.append(f"{rel}: orderInEditor not kept")
                    elif orr[i].raw != sr[i].raw:
                        problems.append(f"{rel}: record {want_id} changed")
        elif rel.parent == MAPS and rel.name.endswith("Things.kld"):
            st = [(r.offset, r.creature_id) for r in find_creature_things(s)]
            ot = [(r.offset, r.creature_id) for r in find_creature_things(o)]
            want = [(i, target if c == OLD_ID else c) for i, c in st]
            if ot != want:
                problems.append(f"{rel}: creature things {ot} != {want}")
            if any(c == OLD_ID for _i, c in ot):
                problems.append(f"{rel}: still a creature {OLD_ID}")
        elif rel.parent == MAPS and rel.name.endswith("Variables.kld"):
            (ob,) = read_blocks(o)
            ents = [struct.unpack_from("<iiii", ob.content, i) for i in range(0, len(ob.content), 16)]
            if any((e[0] == CREATURE_POOL and e[1] == OLD_ID) or
                   (e[0] == AVAILABILITY and (e[1] & 0xFFFF) == AV_CREATURE and e[2] == OLD_ID) for e in ents):
                problems.append(f"{rel}: still an entry for creature {OLD_ID}")
        elif rel.parent == MAPS and rel.suffix == ".kwd":
            if o != s:
                ids_s, ids_o = _level_ids(s), _level_ids(o)
                if ids_o != ids_s + [target] or len(o) != len(s) + 4 or o[:LEVEL_SIZE_AT] != s[:LEVEL_SIZE_AT] \
                        or struct.unpack_from("<I", o, LEVEL_SIZE_AT)[0] != len(o):
                    problems.append(f"{rel}: creature id list {ids_s} -> {ids_o} is not the old list plus {target}")
        elif rel.name != "resource.toml" and s != o:
            problems.append(f"{rel}: changed")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Move Molynx from creature 47 into an unused slot.")
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    args = ap.parse_args(argv)
    out = takeover(args.src, args.out, args.target)
    problems = check_takeover(args.src, out, args.target)
    print("\n".join(problems) if problems else f"TAKEOVER OK: Molynx is creature {args.target} in {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
