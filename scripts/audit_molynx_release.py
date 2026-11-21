r"""Independent audit of the Molynx release folder (build_molynx_release): `audit_release(mod_dir,
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
  (k)  his sound bank: record 47's soundGategory CREATURE_MOLYNX (rule b) names
       Data/Sound/Sfx/Global/creature_molynxBANK.map, which must name the banks Global\Molynx and
       Global\MiscCreatures; creature_molynxSFX.map is the game's creature_bile_demonSFX.map byte
       for byte; MolynxHD.sdt and MolynxHW.sdt hold the same samples as the game's BileHD/HW.sdt
       (count, order, names, blanks), every other one MPEG-2 Layer II at 22050 Hz mono that decodes
       to the PCM length its header states (to within one 1152-sample frame) and not the Bile
       Demon's own data.

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


SOUNDS = "Data/Sound/Sfx/Global"


def _sdt(data: bytes):
    (n,) = struct.unpack_from("<I", data, 0)
    out = []
    for off in struct.unpack_from(f"<{n}I", data, 4):
        hs, size, name, fmt, _z1, pcm, _z2 = struct.unpack_from("<II16sIIII", data, off)
        out.append((name.split(b"\0")[0], fmt, pcm, data[off + hs:off + hs + size]))
    return out


def _check_sounds(mod_dir: Path, game_sounds: Path) -> tuple[bool, str]:
    import subprocess
    import tempfile
    d, g = mod_dir / SOUNDS, Path(game_sounds)
    problems = []
    bank = (d / "creature_molynxBANK.map").read_bytes()
    names, at = [], 0x1c + 11 * struct.unpack_from("<I", bank, 0x18)[0]
    while at < len(bank):
        (n,) = struct.unpack_from("<I", bank, at)
        names.append(bank[at + 4:at + 3 + n].decode("latin-1"))
        at += 4 + n
    if names != ["Global\\Molynx", "Global\\MiscCreatures"]:
        problems.append(f"BANK.map names {names}")
    if (d / "creature_molynxSFX.map").read_bytes() != (g / "creature_bile_demonSFX.map").read_bytes():
        problems.append("SFX.map is not the Bile Demon's")
    checked = 0
    for kind in ("HD", "HW"):
        mine, bile = _sdt((d / f"Molynx{kind}.sdt").read_bytes()), _sdt((g / f"Bile{kind}.sdt").read_bytes())
        if [(n, f) for n, f, _p, _d in mine] != [(n, f) for n, f, _p, _d in bile]:
            problems.append(f"Molynx{kind}.sdt does not list the Bile Demon's samples in order")
            continue
        for (name, fmt, pcm, data), (_n, _f, _p, bdata) in zip(mine, bile):
            if not bdata:
                if data:
                    problems.append(f"{kind} {name!r}: a blank with data")
                continue
            if data == bdata or data[:2] != b"\xff\xf5" or data[2] & 0x0c != 0 or data[3] >> 6 != 3:
                problems.append(f"{kind} {name!r}: not his own MPEG-2 Layer II 22050 Hz mono sample")
                continue
            with tempfile.TemporaryDirectory() as tmp:     # ffmpeg does not probe MPEG audio on a pipe
                (Path(tmp) / "s.mp2").write_bytes(data)
                raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(Path(tmp) / "s.mp2"), "-f",
                                      "s16le", "-"], capture_output=True).stdout
            if not 0 <= len(raw) - pcm < 1152 * 2:
                problems.append(f"{kind} {name!r}: decodes to {len(raw)} bytes, header says {pcm}")
            checked += 1
    if problems:
        return False, "; ".join(problems)
    return True, (f"creature_molynx bank (Global\\Molynx + MiscCreatures), the Bile Demon's events, "
                  f"{checked} of his own samples in MolynxHD/HW.sdt decoded and of the stated length")


def audit_release(mod_dir: Path, map_folders: list[Path], game_sounds: Path | None = None) -> AuditReport:
    mod_dir = Path(mod_dir)
    lines, files = _Lines(), _Files(mod_dir)
    found = _sources(map_folders)
    expected = _expected_maps(found)
    shipped = sorted(os.listdir(mod_dir / MAPS))
    globals_ = {f[:-len("Globals.kld")].lower(): f for f in shipped if f.endswith("Globals.kld")}
    variables = {f[:-len("Variables.kld")].lower(): f for f in shipped if f.endswith("Variables.kld")}
    from scripts.build_molynx_release import FLAT_TEST
    flat_key = FLAT_TEST.lower()
    other = [f for f in shipped if not f.endswith(("Globals.kld", "Variables.kld"))
             and not f.lower().startswith(flat_key)]
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
        if key == flat_key:     # the flat test map: its creature table is Siege's, its layout dk2_testmap's
            from scripts.build_molynx_mod import FLAT_GLOBALS
            from scripts.dk2_testmap import validate_testmap
            source = found[(FLAT_GLOBALS + "Globals.kld").lower()][0]
            stage, _m, _t = _audit_globals_file(mod_dir / MAPS / globals_[key], source, label, lines, files)
            stages.add(stage)
            tfiles = {f: (mod_dir / MAPS / f).read_bytes() for f in shipped if f.lower().startswith(flat_key)
                      and not f.endswith("Globals.kld")}
            errors = validate_testmap(tfiles, FLAT_TEST, molynx_in_pool=True)
            lines.add(not errors, "r2", label, "the flat test map (dk2_testmap rules, Mana Vaults)" if not errors
                      else "; ".join(errors))
            (block,) = [b for b in read_blocks((mod_dir / MAPS / globals_[key]).read_bytes()) if b.signature == 0xAA]
            lines.add(block.item_count == MOLYNX, "r3", label, f"{block.item_count} creature records")
            continue
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
    if game_sounds is not None:
        try:
            ok, message = _check_sounds(mod_dir, game_sounds)
        except (OSError, ValueError, struct.error) as err:
            ok, message = False, f"could not be checked: {err}"
        lines.add(ok, "k", "folder", message)
    ok, message = _check_resource_toml(mod_dir)
    lines.add(ok, "h", "folder", message)
    ok, message = _check_ground_slam(mod_dir, [globals_[k][:-len("Globals.kld")] for k in globals_], stage, files)
    lines.add(ok, "j", "folder", message)
    return AuditReport(all(ok for ok, _ in lines.items), [line for _ok, line in lines.items])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sounds = None
    if argv and argv[0].startswith("--game-sounds="):
        sounds, argv = Path(argv[0].split("=", 1)[1]), argv[1:]
    report = audit_release(Path(argv[0]), [Path(a) for a in argv[1:]], sounds)
    for line in report.lines:
        print(line)
    print("AUDIT", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
