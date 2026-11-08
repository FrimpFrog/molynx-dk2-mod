"""Independent audit of the Molynx release folder (build_molynx_release): `audit_release(mod_dir,
map_folders) -> AuditReport`.

It re-reads the folder from disk. For the creature tables it runs audit_molynx_mod's own rules on
every map's Globals against that map's source (the first of `map_folders` that has it) and on the
folder's editor tables: (a) every other block byte-identical apart from Ground Slam, (b) exactly 47
records, the first 46 the source's, record 47 the source's Bile Demon changed by exactly the stage-3
fields, (c) header invariants, (f) his files, (i) the editor's Creatures.kwd, (j) Ground Slam, (h)
resource.toml. The release's own rules, written out here and not taken from the builder:

  (r1) maps: the folder holds `<map>Globals.kld` for exactly the maps whose name is mpd<n>, or whose
       level file (any copy in map_folders) has the multiplayer or skirmish flag, minus the campaign
       (level<n>, Secret<n>); `<map>Variables.kld` only for those maps; nothing else in maps.
  (r2) Variables: the source's entries in their order with only creature-47 entries added: per
       player with a creature pool exactly one pool entry (count: that player's Bile Demon count,
       else the player's largest), per player with creature availability exactly one ENABLE entry,
       none for other players. A map whose source has neither pool nor availability ships none.
  (r3) no record or entry for a creature above 47 anywhere (the slots after him stay empty).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import os
import re
import struct
import sys
from pathlib import Path

from scripts.audit_molynx_mod import (AuditReport, _audit_editor_creatures, _audit_globals_file,
                                      _check_ground_slam, _check_resource_toml, _Files, _Lines)
from scripts.dk2kwd_blocks import read_blocks

MAPS = "Data/editor/maps"
MOLYNX = 47
BILE = 3
CAMPAIGN = re.compile(r"(level\d+[a-c]?|secret\d+)", re.IGNORECASE)
MPD = re.compile(r"mpd\d+", re.IGNORECASE)
FLAGS_AT, MP_FLAG, SK_FLAG = 56 + 25484, 1 << 10, 1 << 11


def _sources(map_folders) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for folder in map_folders:
        for path in sorted(Path(folder).iterdir()):
            found.setdefault(path.name.lower(), []).append(path)
    return found


def _expected_maps(found) -> dict[str, str]:
    """lower-case map name -> its spelling, for every map Molynx must be on."""
    out = {}
    for key, paths in found.items():
        if not key.endswith(".kwd"):
            continue
        name = paths[0].name[:-4]
        if CAMPAIGN.fullmatch(name):
            continue
        flags = 0
        for p in paths:
            data = p.read_bytes()
            flags |= struct.unpack_from("<H", data, FLAGS_AT)[0] if len(data) >= FLAGS_AT + 2 else 0
        if MPD.fullmatch(name) or flags & (MP_FLAG | SK_FLAG):
            out[name.lower()] = name
    return out


def _variable_entries(data: bytes) -> list[tuple]:
    (sig,) = struct.unpack_from("<I", data, 0)
    if sig != 0xE6:
        raise ValueError(f"not a Variables block (signature {sig:#x})")
    body = data[56:]
    return [struct.unpack_from("<iiii", body, i) for i in range(0, len(body) - len(body) % 16, 16)]


def _creature_of(e) -> int | None:
    if e[0] == 1:
        return e[1]
    if e[0] == 2 and (e[1] & 0xFFFF) == 2:
        return e[2]
    return None


def _check_variables(src: bytes, mod: bytes | None) -> tuple[bool, str]:
    s = _variable_entries(src)
    pool_players = sorted({e[3] for e in s if e[0] == 1})
    av_players = sorted({e[1] >> 16 for e in s if e[0] == 2 and (e[1] & 0xFFFF) == 2})
    if not pool_players and not av_players:
        return (mod is None, "no creature pool or list in the map: no Variables shipped" if mod is None
                else "a Variables file is shipped for a map with no creature pool")
    if mod is None:
        return False, "the Variables file is missing"
    m = _variable_entries(mod)
    (count,) = struct.unpack_from("<I", mod, 20)
    (size,) = struct.unpack_from("<I", mod, 8)
    if count != len(m) or size != len(mod):
        return False, f"header says {count} entries / {size} bytes, the file holds {len(m)} / {len(mod)}"
    added = [e for e in m if _creature_of(e) == MOLYNX]
    rest = [e for e in m if _creature_of(e) != MOLYNX]
    if rest != s:
        return False, "the map's own entries are not kept as they were, in order"
    problems = []
    for p in pool_players:
        own = [e for e in s if e[0] == 1 and e[3] == p]
        biles = [e[2] for e in own if e[1] == BILE and e[2] > 0]
        want = (1, MOLYNX, biles[0] if biles else max(e[2] for e in own), p)
        got = [e for e in added if e[0] == 1 and e[3] == p]
        if got != [want]:
            problems.append(f"player {p} pool: {got}, expected {[want]}")
        # placed right after the player's last pool entry
        i = m.index(want) if want in m else -1
        if i > 0 and not (m[i - 1][0] == 1 and m[i - 1][3] == p):
            problems.append(f"player {p} pool entry is not right after that player's pool")
    for p in av_players:
        want = (2, (p << 16) | 2, MOLYNX, 3)
        got = [e for e in added if e[0] == 2 and e[1] >> 16 == p]
        if got != [want]:
            problems.append(f"player {p} availability: {got}, expected {[want]}")
    extra = [e for e in added if not ((e[0] == 1 and e[3] in pool_players) or (e[0] == 2 and e[1] >> 16 in av_players))]
    if extra:
        problems.append(f"entries for players without a pool or list: {extra}")
    if any((c := _creature_of(e)) is not None and c > MOLYNX for e in m):
        problems.append("an entry names a creature above 47 (r3)")
    if problems:
        return False, "; ".join(problems)
    return True, (f"own {len(s)} entries kept in order; Molynx in the pool of player(s) "
                  f"{pool_players} and ENABLE for {av_players}")


def audit_release(mod_dir: Path, map_folders: list[Path]) -> AuditReport:
    mod_dir = Path(mod_dir)
    lines, files = _Lines(), _Files(mod_dir)
    found = _sources(map_folders)
    expected = _expected_maps(found)
    shipped = sorted(os.listdir(mod_dir / MAPS))
    globals_ = {f[:-len("Globals.kld")].lower(): f for f in shipped if f.endswith("Globals.kld")}
    variables = {f[:-len("Variables.kld")].lower(): f for f in shipped if f.endswith("Variables.kld")}
    other = [f for f in shipped if not f.endswith(("Globals.kld", "Variables.kld"))]
    missing = sorted(set(expected) - set(globals_))
    extra = sorted(set(globals_) - set(expected)) + sorted(set(variables) - set(globals_))
    ok = not (missing or extra or other)
    lines.add(ok, "r1", "folder", f"{len(globals_)} maps, {len(variables)} Variables files" if ok else
              f"missing {missing}, not expected {extra}, other files {other}")
    stages = set()
    for key in sorted(expected):
        name = expected[key]
        if key not in globals_:
            continue
        label = globals_[key][:-len("Globals.kld")]
        source = found[(name + "Globals.kld").lower()][0]
        stage, _meshes, _textures = _audit_globals_file(mod_dir / MAPS / globals_[key], source, label, lines, files)
        stages.add(stage)
        src_var = found[(name + "Variables.kld").lower()][0].read_bytes()
        mod_var = (mod_dir / MAPS / variables[key]).read_bytes() if key in variables else None
        try:
            ok, message = _check_variables(src_var, mod_var)
        except (ValueError, struct.error) as err:
            ok, message = False, f"could not be checked: {err}"
        lines.add(ok, "r2", label, message)
        (block,) = [b for b in read_blocks((mod_dir / MAPS / globals_[key]).read_bytes()) if b.signature == 0xAA]
        lines.add(block.item_count == MOLYNX, "r3", label, f"{block.item_count} creature records")
    stage = stages.pop() if len(stages) == 1 else None
    lines.add(stage == 3, "b", "folder", f"every map is at stage {stage}" if stage == 3 else f"stages {stages | {stage}}")
    _audit_editor_creatures(mod_dir, lines, files)
    ok, message = _check_resource_toml(mod_dir)
    lines.add(ok, "h", "folder", message)
    ok, message = _check_ground_slam(mod_dir, [globals_[k][:-len("Globals.kld")] for k in globals_], stage, files)
    lines.add(ok, "j", "folder", message)
    return AuditReport(all(ok for ok, _ in lines.items), [line for _ok, line in lines.items])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    report = audit_release(Path(argv[0]), [Path(a) for a in argv[1:]])
    for line in report.lines:
        print(line)
    print("AUDIT", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
