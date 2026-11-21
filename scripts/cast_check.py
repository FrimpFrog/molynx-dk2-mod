"""Check a cast install (cast_install's out folder) against the folder it goes into, by reading
everything back. `check(out_dir, base_dir, furnace_dir, game_meshes, codepage, game_cache)` returns a list of
problems ([] when the result is right); `map_image` draws a map's heart and the cast on it.

The folder as it will be is base_dir with out_dir laid over it (a file in out_dir replaces the one
of the same path). In that folder:
  * every Globals and the editor Creatures.kwd parse, hold creature ids 1..N with no gap, records
    of one size, and records 47.. (Molynx, the Necromancer and every cast_records member) with their
    names;
  * no cast record has IS_HORNY (0x4000: the game treats him as the summoned Horny) and each has
    CAN_BE_PICKED_UP, CAN_BE_SLAPPED and AVAILABLE_VIA_PORTAL; a slot the Reaper's donor leaves empty
    and cast_records fills has a plain tail of its file (or is the copy it names);
  * every animation slot of records 47..52 (unk_1 aside: it holds no name in any record) that names a
    file has a mesh, animating-mesh or no type, and the file it names, and its Start / End files when
    its flags ask for them, exist in the folder's Data/Meshes or the game's meshes; a slot with a
    type and no name is a problem;
  * their panel pictures (icon1Resource, portraitResource) exist in the folder at the game's sizes
    (64 x 64, 64 x 128), their texts exist in Text.str, and every texture their own files use is in
    the folder or the game's texture cache;
  * every Variables file parses; any map that lists Molynx also lists the Necromancer; on Dead
    Walking and Furnace Keeper 1's creature pool (count > 0) and availability (ENABLE) hold 47..52,
    once each;
  * Dead Walking's and Furnace's Things parse and write back byte for byte; each of 47..52 is there
    once, owned by Keeper 1, at level 5, on a tile that is his Claimed Path touching his heart (two
    tiles at most) or, on Furnace, his heart's outer ring, and no two creatures share a tile; every
    thing of the source Things is still there.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from scripts.cast_records import CAST as _CAST, RESERVED  # noqa: E402

CAST_IDS = (47, 48, *(m.creature_id for m in _CAST))
NAMES = {47: "Molynx", 48: "Necromancer", **{m.creature_id: m.name for m in _CAST}}
OWN_FILES = tuple(m.creature_id for m in _CAST if not m.keep_donor_tails)   # tails set from his own files
IS_HORNY = 0x4000
REGULAR_BITS = 0x2 | 0x4                       # CAN_BE_PICKED_UP, CAN_BE_SLAPPED: every cast member
PORTAL = 0x400                                 # AVAILABLE_VIA_PORTAL: all but the placed_only ones
REGULAR = tuple(m.creature_id for m in _CAST)   # every cast member is a regular keeper creature
PLACED_ONLY = tuple(m.creature_id for m in _CAST if m.placed_only)
FILLED = {m.creature_id: m.fill_slots for m in _CAST if m.fill_slots}
MAPS = "Data/editor/maps"
KEEPER1, HEART, CLAIMED, ENABLE = 3, 14, 8, 3
FLOOR = {"Dead Walking": "claimed", "Furnace": "heart ring", "mpd1": "claimed near"}
ALWAYS = ("Dead Walking", "Furnace")   # the others are checked once the folder ships their Things


def configure(members) -> None:
    """Check these cast members (cast_records.Member) after Molynx and the Necromancer instead of
    cast_records.CAST: a creature after the cast (Krefft, krefft_install) is checked with the cast."""
    global _CAST, CAST_IDS, NAMES, OWN_FILES, REGULAR, PLACED_ONLY, FILLED
    _CAST = tuple(members)
    CAST_IDS = (47, 48, *(m.creature_id for m in _CAST))
    NAMES = {47: "Molynx", 48: "Necromancer", **{m.creature_id: m.name for m in _CAST}}
    OWN_FILES = tuple(m.creature_id for m in _CAST if not m.keep_donor_tails)
    REGULAR = tuple(m.creature_id for m in _CAST)
    PLACED_ONLY = tuple(m.creature_id for m in _CAST if m.placed_only)
    FILLED = {m.creature_id: m.fill_slots for m in _CAST if m.fill_slots}


class Folder:
    """base_dir with out_dir laid over it."""

    def __init__(self, base: Path, out: Path):
        self.base, self.out = Path(base), Path(out)

    def path(self, rel: str) -> Path | None:
        for root in (self.out, self.base):
            if (root / rel).exists():
                return root / rel
        return None

    def read(self, rel: str) -> bytes:
        p = self.path(rel)
        if p is None:
            raise FileNotFoundError(rel)
        return p.read_bytes()

    def find(self, rel: str) -> Path | None:
        """The file at `rel`, matching every part of the path without regard to case."""
        for root in (self.out, self.base):
            here = root
            for part in Path(rel).parts:
                if not here.is_dir():
                    here = None
                    break
                match = [p for p in here.iterdir() if p.name.lower() == part.lower()]
                here = match[0] if match else None
                if here is None:
                    break
            if here is not None and here.is_file():
                return here
        return None

    def names(self, rel_dir: str, pattern: str) -> list[str]:
        found = {p.name for root in (self.base, self.out) for p in (root / rel_dir).glob(pattern)}
        return sorted(found)


def _check_table(rel: str, data: bytes, meshes, game_meshes: set, out: list, records: dict,
                 frames_of: dict | None = None) -> None:
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.dk2kwd_creatures import (ANIM_SLOTS, ART_FLAG_HAS_END_ANIMATION as E,
                                          ART_FLAG_HAS_START_ANIMATION as S, read_creatures)
    try:
        blocks = read_blocks(data)
        tables = [b for b in blocks if b.signature == 0xAA]
        if len(tables) != 1:
            out.append(f"{rel}: {len(tables)} Creatures blocks")
            return
        recs = read_creatures(tables[0])
    except ValueError as e:
        out.append(f"{rel}: does not parse ({e})")
        return
    frames_of = frames_of or {}
    ids = [c.get("creatureId") for c in recs]
    if ids != list(range(1, len(ids) + 1)):
        out.append(f"{rel}: creature ids are not 1..{len(ids)} in order: ...{ids[-6:]}")
    if len({len(c.raw) for c in recs}) != 1:
        out.append(f"{rel}: records of different sizes")
    orders = [c.get("orderInEditor") for c in recs]
    if len(set(orders)) != len(orders):
        out.append(f"{rel}: repeated orderInEditor")
    by_id = {c.get("creatureId"): c for c in recs}
    for cid in CAST_IDS:
        c = by_id.get(cid)
        if c is None:
            out.append(f"{rel}: no record {cid}")
            continue
        if c.get("name") != NAMES[cid]:
            out.append(f"{rel}: record {cid} is named {c.get('name')!r}")
        records.setdefault(cid, c)
        if c.get("flags") & IS_HORNY:
            out.append(f"{rel}: record {cid} has IS_HORNY (0x4000): the game makes him a summoned Horny")
        if cid in REGULAR and c.get("flags") & REGULAR_BITS != REGULAR_BITS:
            out.append(f"{rel}: record {cid} flags 0x{c.get('flags'):08X} lack a regular creature's "
                       f"Hand / slap bits (0x{REGULAR_BITS:X})")
        if cid in REGULAR and bool(c.get("flags") & PORTAL) == (cid in PLACED_ONLY):
            out.append(f"{rel}: record {cid} flags 0x{c.get('flags'):08X}: Portal bit "
                       f"{'set on a creature that is only placed' if cid in PLACED_ONLY else 'missing'}")
        if cid in PLACED_ONLY and any(c.get(f"attractions[{i}].present") for i in range(2)):
            out.append(f"{rel}: record {cid} has an attraction but is only placed")
        for slot in ANIM_SLOTS[1:]:
            name, t = c.resource_name(slot), c.art_tail(slot)
            if not name and not t.type:
                continue
            if not name or t.type not in (0, 5, 6):
                out.append(f"{rel}: record {cid} slot {slot}: name {name!r} with type {t.type}")
                continue
            # a name with type 0 (NONE): Molynx's DejectedPose on 38 maps names Molynx-staggering
            # (the installed Molynx, tested in game on Dead Walking); the file must exist all the same
            want = [name] + ([name + "Start"] if t.flags & S else []) + ([name + "End"] if t.flags & E else [])
            files = frames_of.get(cid)
            if files and t.type == 6:
                # the cast's own tails: frames = the file's (less one when it loops), startAf /
                # endAf = the Start / End files' frame counts
                fr = files.get(name.lower())
                if fr is not None and t.frames != fr[0] - (1 if fr[1] else 0):
                    out.append(f"{rel}: record {cid} slot {slot}: tail frames {t.frames}, file {fr}")
                for flag, suffix, af in ((S, "Start", t.start_af), (E, "End", t.end_af)):
                    sub = files.get((name + suffix).lower())
                    if t.flags & flag and sub is not None and sub[0] != af:
                        out.append(f"{rel}: record {cid} slot {slot}: {suffix.lower()}Af {af}, file {sub[0]}")
            for n in want:
                if n.lower() not in meshes and n.lower() not in game_meshes:
                    out.append(f"{rel}: record {cid} slot {slot} needs {n}.kmf, which is in neither the folder nor the game")


def _entries(data: bytes):
    from scripts.build_molynx_release import _entries as entries
    return entries(data)[1]


def _xy(t) -> tuple[int, int]:
    return struct.unpack_from("<ii", t.data)[:2]


def _tile_problem(m, heart, xy, floor, keep_clear=frozenset()) -> str | None:
    from scripts.cast_install import NEAR
    x, y = xy
    if not (0 <= x < m.width and 0 <= y < m.height):
        return "off the map"
    if xy in keep_clear:
        return "on the way in and out of his dungeon (kept clear)"
    terr, owner = m.at(x, y)[0], m.at(x, y)[1]
    x0, y0, x1, y1 = heart
    inside = x0 <= x <= x1 and y0 <= y <= y1
    if floor in ("claimed", "claimed near"):
        if terr != CLAIMED or owner != KEEPER1:
            return f"terrain {terr} of player {owner}, not Keeper 1's Claimed Path"
        if inside or max(x0 - x, x - x1, y0 - y, y - y1) > (2 if floor == "claimed" else NEAR):
            return "not by the heart"
        return None
    if terr != HEART or owner != KEEPER1 or not inside or not (x in (x0, x1) or y in (y0, y1)):
        return "not on the outer ring of Keeper 1's heart"
    return None


def heart_box(m) -> tuple[int, int, int, int]:
    cells = [(i % m.width, i // m.width) for i, t in enumerate(m.tiles) if t[0] == HEART and t[1] == KEEPER1]
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def check(out_dir: Path, base_dir: Path, furnace_dir: Path, game_meshes: Path, codepage: Path,
          game_cache=None, map_dirs: dict | None = None) -> list[str]:
    """map_dirs: map name -> the folder the game loads that map's other files from (Furnace's is
    furnace_dir)."""
    from PIL import Image
    from scripts.dk2_testmap import read_map
    from scripts.dk2kld_splice import read_things, write_things
    from scripts.dk2kmf import parse_kmf
    from scripts.dk2str import StrFile
    out_dir, base_dir = Path(out_dir), Path(base_dir)
    f = Folder(base_dir, out_dir)
    problems: list[str] = []
    meshes = {n.lower()[:-4] for n in f.names("Data/Meshes", "*.kmf")}
    game = {p.name.lower()[:-4] for p in Path(game_meshes).glob("*.kmf")}
    records: dict = {}
    frames_of = {}                       # the cast's own files, wherever they are: name -> (frames, loops)
    prefixes = tuple(m.prefix.lower() for m in _CAST)
    for n in f.names("Data/Meshes", "*.kmf"):
        if n.lower().startswith(prefixes):
            k = parse_kmf(f.read(f"Data/Meshes/{n}"))
            frames_of[n[:-4].lower()] = (k.anim.frame_count, bool(k.anim.wrap)) if k.anim else (1, False)
    own = {cid: frames_of for cid in OWN_FILES}     # the Dragon's and Sejanus' tails are their own; 49, 50 keep the donor's
    problems += same_frames_as_donor(f, frames_of, game_meshes)
    _check_table("Data/editor/Creatures.kwd", f.read("Data/editor/Creatures.kwd"), meshes, game, problems, records, own)
    globals_ = f.names(MAPS, "*Globals.kld")
    for g in globals_:
        _check_table(f"{MAPS}/{g}", f.read(f"{MAPS}/{g}"), meshes, game, problems, records, own)
    # panel pictures, texts, textures of the cast's own files (from the editor table's records)
    text = StrFile.read(f.path("Data/Text/Default/Text.str"), codepage)
    cache_names = {n.lower() for n in game_cache.names} if game_cache is not None else set()
    for cid in CAST_IDS[2:]:
        c = records.get(cid)
        if c is None:
            continue
        for field, size in (("icon1Resource", (64, 64)), ("portraitResource", (64, 128))):
            res = c.resource_name(field).replace("\\", "/")
            hit = f.find(f"Data/EngineTextures/{res}.png")       # Windows: names match whatever their case
            if hit is None:
                problems.append(f"record {cid}: {field} {res}.png is not in the folder")
            else:
                im = Image.open(hit)
                if im.size != size:
                    problems.append(f"record {cid}: {hit.name} is {im.size}, not {size}")
        for field in ("nameStringId", "generalDescriptionStringId", "strengthStringId", "weaknessStringId"):
            if not 0 <= c.get(field) < len(text.entries):
                problems.append(f"record {cid}: {field} {c.get(field)} is not in Text.str")
        if text.text(c.get("nameStringId")) == "":
            problems.append(f"record {cid}: empty name text")
        from scripts.dk2kwd_creatures import ANIM_SLOTS
        used = {c.resource_name(s) for s in ANIM_SLOTS[1:] if c.resource_name(s)}
        textures = set()
        for n in used:
            p = f.path(f"Data/Meshes/{n}.kmf")
            if p is None:
                continue
            k = parse_kmf(p.read_bytes())
            textures |= {t for m in k.materials for t in m.textures}
        pngs = {n.lower()[:-4] for n in f.names("Data/EngineTextures", "*.png")}
        for t in sorted(textures):
            if t.lower() not in pngs and f"{t}mm0".lower() not in cache_names:
                problems.append(f"record {cid}: texture {t} is in neither the folder nor the game")
    # variables
    for v in f.names(MAPS, "*Variables.kld"):
        try:
            ents = _entries(f.read(f"{MAPS}/{v}"))
        except ValueError as e:
            problems.append(f"{v}: does not parse ({e})")
            continue
        pool = [(e[1], e[3]) for e in ents if e[0] == 1]
        av = [(e[2], e[1] >> 16) for e in ents if e[0] == 2 and (e[1] & 0xFFFF) == 2]
        if len(pool) != len(set(pool)) or len(av) != len(set(av)):
            problems.append(f"{v}: a creature is listed twice for one player")
        if {c for c, _p in pool} & {47} and 48 not in {c for c, _p in pool}:
            problems.append(f"{v}: Molynx in the pool without the Necromancer")
        name = v[:-len("Variables.kld")]
        placed_here = name in ALWAYS or f.path(f"{MAPS}/{name}Things.kld") is not None
        for cid in PLACED_ONLY:
            if [e for e in ents if (e[0] == 1 and e[1] == cid) or (e[0] == 2 and (e[1] & 0xFFFF) == 2 and e[2] == cid)]:
                problems.append(f"{v}: creature {cid} is in a creature pool or availability but is only placed")
        if name in FLOOR and placed_here:
            for cid in CAST_IDS:
                if cid in PLACED_ONLY:
                    continue
                if not [e for e in ents if e[0] == 1 and e[1] == cid and e[3] == KEEPER1 and e[2] > 0]:
                    problems.append(f"{v}: creature {cid} not in Keeper 1's pool")
                if not [e for e in ents if e[0] == 2 and e[1] == (KEEPER1 << 16) | 2 and e[2] == cid and e[3] == ENABLE]:
                    problems.append(f"{v}: creature {cid} not available to Keeper 1")
    # things
    from scripts.cast_install import KEEP_CLEAR
    dirs = {"Furnace": Path(furnace_dir), **(map_dirs or {})}
    for name, floor in FLOOR.items():
        rel = f"{MAPS}/{name}Things.kld"
        if name not in ALWAYS and f.path(rel) is None:
            continue
        try:
            data = f.read(rel)
            block, things = read_things(data)
        except (ValueError, FileNotFoundError) as e:
            problems.append(f"{rel}: {e}")
            continue
        if write_things(block, things) != data:
            problems.append(f"{rel}: does not write back byte for byte")
        mrel = f"{MAPS}/{name}Map.kld"
        mp = f.path(mrel) or dirs[name] / f"{name}Map.kld"
        m = read_map(mp.read_bytes())
        heart = heart_box(m)
        src = base_dir / rel if (base_dir / rel).exists() else dirs[name] / f"{name}Things.kld"
        _b, before = read_things(src.read_bytes())
        left = list(things)
        for t in before:
            if t in left:
                left.remove(t)
            else:
                problems.append(f"{rel}: a thing of type {t.type} of the source is gone")
        tiles = [_xy(t) for t in things if t.type == 200]
        if len(tiles) != len(set(tiles)):
            problems.append(f"{rel}: two creatures share a tile")
        for cid in CAST_IDS:
            mine = [t for t in things if t.type == 200 and t.data[26] == cid]
            if len(mine) != 1:
                problems.append(f"{rel}: {len(mine)} things of creature {cid}")
                continue
            t = mine[0]
            if t.data[27] != KEEPER1 or t.data[14] != 5:
                problems.append(f"{rel}: creature {cid} owner {t.data[27]} level {t.data[14]}")
            why = _tile_problem(m, heart, _xy(t), floor, KEEP_CLEAR.get(name, frozenset()))
            if why:
                problems.append(f"{rel}: creature {cid} at {_xy(t)}: {why}")
    return problems


REENCODED = {49: 10, 50: 27}              # the Orc's files re-encode the Troll's, the Reaper's Horny's


def same_frames_as_donor(f: "Folder", frames_of: dict, game_meshes: Path) -> list[str]:
    """For a creature whose files re-encode his donor's (and so keep the donor's slot tails): every
    file a slot names (and its Start / End files) has the frame count and loop flag of the donor's
    file for that slot (editor Creatures.kwd)."""
    from scripts.dk2kmf import parse_kmf
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.dk2kwd_creatures import ANIM_SLOTS, read_creatures
    game = {p.name.lower(): p for p in Path(game_meshes).glob("*.kmf")}
    recs = {c.get("creatureId"): c for b in read_blocks(f.read("Data/editor/Creatures.kwd")) if b.signature == 0xAA
            for c in read_creatures(b)}
    out = []
    for cid, did in REENCODED.items():
        me, donor = recs.get(cid), recs.get(did)
        if me is None or donor is None:
            continue
        for slot in ANIM_SLOTS[1:]:
            mine, theirs = me.resource_name(slot), donor.resource_name(slot)
            if slot in FILLED.get(cid, {}):
                # a slot his donor leaves empty, given one of his files: a plain tail of that file
                t, fr, kind = me.art_tail(slot), frames_of.get(mine.lower()), FILLED[cid][slot][1]
                if theirs or fr is None:
                    out.append(f"record {cid} slot {slot}: filled with {mine!r}, donor {theirs!r}, file {fr}")
                elif kind.startswith("copy:"):
                    if (mine, t) != (me.resource_name(kind[5:]), me.art_tail(kind[5:])):
                        out.append(f"record {cid} slot {slot}: not a copy of {kind[5:]}")
                elif t.type == 6 and (t.frames != fr[0] - (1 if fr[1] else 0) or t.flags & 0x18):
                    out.append(f"record {cid} slot {slot}: tail {t} for {mine} {fr}")
                elif t.type == 5 and fr[0] != 1:
                    out.append(f"record {cid} slot {slot}: a pose tail for {mine} {fr}")
                continue
            if bool(mine) != bool(theirs):
                out.append(f"record {cid} slot {slot}: {mine!r} where the donor has {theirs!r}")
                continue
            if not mine:
                continue
            for suffix in ("", "Start", "End"):
                g = game.get(f"{theirs}{suffix}.kmf".lower())
                m = frames_of.get(f"{mine}{suffix}".lower())
                if g is None and m is None:
                    continue
                if g is None or m is None:
                    out.append(f"record {cid} slot {slot}: {mine}{suffix} / {theirs}{suffix}: one of them is missing")
                    continue
                k = parse_kmf(g.read_bytes())
                theirs_fr = (k.anim.frame_count, bool(k.anim.wrap)) if k.anim else (1, False)
                if theirs_fr != m:
                    out.append(f"record {cid} slot {slot}: {mine}{suffix} {m} but {theirs}{suffix} {theirs_fr}")
    return out


# ------------------------------------------------------------------------------ the map picture

TERRAIN_COLOURS = {1: (20, 20, 20), 2: (70, 60, 50), 3: (120, 95, 60), 4: (40, 80, 160), 5: (200, 70, 20),
                   6: (210, 180, 40), 7: (90, 200, 220), 8: (150, 130, 100), 9: (95, 85, 75), 12: (150, 60, 170),
                   14: (190, 30, 40), 30: (0, 0, 0)}
CREATURE_COLOURS = {1: (240, 240, 240), 47: (255, 120, 40), 48: (60, 140, 255), 49: (230, 60, 160),
                    50: (255, 40, 40), 51: (120, 210, 60), 52: (170, 90, 230), 53: (160, 160, 160)}
LABELS = {1: "Imp", 47: "Molynx 47", 48: "Necromancer 48", 49: "Orc 49", 50: "Quuz Reaper 50", 51: "Dragon 51",
          52: "Sejanus 52", 53: "Gargoyle 53"}
LABELS.update({k: f"{v} (reserved)" for k, v in LABELS.items() if k in RESERVED})


def map_image(map_kld: bytes, things_kld: bytes, title: str, reserved: dict | None = None,
              cell: int = 9, zoom: tuple | None = None, zoom_cell: int = 40):
    """The whole map (cell px a tile) and a close-up of `zoom` = (x0, y0, x1, y1) tiles, with every
    creature of Keeper 1 marked and the cast labelled; reserved tiles drawn as rings."""
    from PIL import Image, ImageDraw
    from scripts.dk2_testmap import read_map
    from scripts.dk2kld_splice import read_things
    m = read_map(map_kld)
    _b, things = read_things(things_kld)
    crs = [(t.data[26], t.data[27], _xy(t)) for t in things if t.type == 200]

    def draw(x0, y0, x1, y1, c, labels):
        w, h = (x1 - x0 + 1) * c, (y1 - y0 + 1) * c
        im = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(im)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                terr, owner = m.at(x, y)[0], m.at(x, y)[1]
                col = TERRAIN_COLOURS.get(terr, (60, 60, 90))
                if terr == 8 and owner == KEEPER1:
                    col = (175, 150, 110)
                d.rectangle([(x - x0) * c, (y - y0) * c, (x - x0 + 1) * c - 1, (y - y0 + 1) * c - 1], fill=col,
                            outline=(0, 0, 0) if c >= 20 else None)
        for cid, (x, y) in (reserved or {}).items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                cx, cy = (x - x0 + .5) * c, (y - y0 + .5) * c
                d.ellipse([cx - c * .38, cy - c * .38, cx + c * .38, cy + c * .38], outline=CREATURE_COLOURS[cid],
                          width=max(2, c // 10))
                if labels:
                    d.text((cx - c * .45, cy - c * .2), str(cid), fill=CREATURE_COLOURS[cid])
        for cid, owner, (x, y) in crs:
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            cx, cy = (x - x0 + .5) * c, (y - y0 + .5) * c
            r = c * (.38 if cid in CREATURE_COLOURS and cid != 1 else .22)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CREATURE_COLOURS.get(cid, (200, 200, 0)),
                      outline=(0, 0, 0))
            if labels and cid != 1:
                d.text((cx - c * .2, cy - c * .2), str(cid), fill=(0, 0, 0))
        return im

    full = draw(0, 0, m.width - 1, m.height - 1, cell, False)
    parts = [full]
    if zoom:
        z = draw(*zoom, zoom_cell, True)
        if 0 <= zoom[0]:
            dd = ImageDraw.Draw(full)
            dd.rectangle([zoom[0] * cell, zoom[1] * cell, (zoom[2] + 1) * cell, (zoom[3] + 1) * cell],
                         outline=(255, 255, 0), width=2)
        parts.append(z)
    legend_h = 18 * (len(LABELS) + 2)
    W = sum(p.width for p in parts) + 20 * len(parts)
    H = max(max(p.height for p in parts) + 30, legend_h + 30)
    img = Image.new("RGB", (W + 230, H), (24, 24, 28))
    d = ImageDraw.Draw(img)
    d.text((10, 6), title, fill=(255, 255, 255))
    x = 10
    for p in parts:
        img.paste(p, (x, 26))
        x += p.width + 20
    ly = 30
    keys = [(TERRAIN_COLOURS[14], "Dungeon Heart"), ((175, 150, 110), "Keeper 1 Claimed Path"),
            (TERRAIN_COLOURS[9], "Reinforced Wall")] + [(CREATURE_COLOURS[k], v) for k, v in LABELS.items()]
    for col, label in keys:
        d.rectangle([x, ly, x + 12, ly + 12], fill=col)
        d.text((x + 18, ly), label, fill=(230, 230, 230))
        ly += 18
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check a cast install folder against the MoreCreatures folder it goes into.")
    p.add_argument("--out", type=Path, required=True, help="<install>/MoreCreatures")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--furnace", type=Path, required=True)
    p.add_argument("--meshes", type=Path, required=True, help="the game's meshes (extracted)")
    p.add_argument("--textures", type=Path, help="the game's Data folder (EngineTextures.dir/.dat)")
    p.add_argument("--codepage", type=Path, required=True, help="the game's MBToUni.dat")
    a = p.parse_args(argv)
    cache = None
    if a.textures:
        from scripts.dk2texture import TextureCache
        cache = TextureCache(a.textures / "EngineTextures.dir", a.textures / "EngineTextures.dat")
    problems = check(a.out, a.base, a.furnace, a.meshes, a.codepage, cache)
    for line in problems:
        print("PROBLEM", line)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
