"""Tests for scripts/dk2_testmap.py: the flat Molynx test map built from "Dead Walking".

Every layout rule of the task brief has a test on the real generated files, plus tests that the
validator rejects deliberately broken copies. Raw byte checks here use their own offsets (taken
from the reconnaissance notes) so they do not only repeat the module's own parsing.
"""
import struct

import pytest

from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2paths import EDITOR_DIR, MAPS_DIR, QUUZ_MAPS_DIR, have

needs_maps = pytest.mark.skipif(not (have(MAPS_DIR) and have(QUUZ_MAPS_DIR) and have(EDITOR_DIR)),
                                reason="DK2 editor maps or Quuz maps not present")

STEM = "Molynx Test"
STEM_POOL = "Molynx Test Portal"
SUFFIXES = (".kwd", "Map.kld", "Things.kld", "Players.kld", "Triggers.kld", "Variables.kld")
SIEGE_POOL_IDS = [3, 4, 5, 6, 7, 10, 12, 22, 23, 24, 45]
SOLID = {1, 2, 6, 7, 9, 30}


@pytest.fixture(scope="module")
def tm():
    from scripts import dk2_testmap
    return dk2_testmap


@pytest.fixture(scope="module")
def files(tm):
    return tm.testmap_files(STEM, molynx_in_pool=False)


@pytest.fixture(scope="module")
def files_pool(tm):
    return tm.testmap_files(STEM_POOL, molynx_in_pool=True, display_name="Molynx Test Portal")


def _base(name: str = "Dead Walking") -> dict[str, bytes]:
    from scripts.dk2kld_splice import map_source_dir
    folder = map_source_dir(name)
    return {s: (folder / (name + s)).read_bytes() for s in SUFFIXES}


def _f(files, stem, suffix):
    return files[stem + suffix]


def _tiles(map_kld: bytes):
    hsz = 28 + struct.unpack_from("<I", map_kld, 16)[0]
    w, h = struct.unpack_from("<II", map_kld, 20)
    return w, h, [map_kld[hsz + 4 * i:hsz + 4 * i + 4] for i in range(w * h)]


def _things(things_kld: bytes):
    pos, out = 56, []
    while pos < len(things_kld):
        kind, size = struct.unpack_from("<II", things_kld, pos)
        out.append((kind, things_kld[pos + 8:pos + 8 + size]))
        pos += 8 + size
    return out


def _with(files, stem, suffix, data):
    out = dict(files)
    out[stem + suffix] = data
    return out


def _set_tile(map_kld: bytes, x: int, y: int, tile: bytes) -> bytes:
    hsz = 28 + struct.unpack_from("<I", map_kld, 16)[0]
    w = struct.unpack_from("<I", map_kld, 20)[0]
    out = bytearray(map_kld)
    out[hsz + 4 * (y * w + x):hsz + 4 * (y * w + x) + 4] = tile
    return bytes(out)


def _component(w, h, tiles, start, ok):
    seen, todo = {start}, [start]
    while todo:
        i = todo.pop()
        x, y = i % w, i // w
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h:
                j = ny * w + nx
                if j not in seen and ok(tiles[j]):
                    seen.add(j)
                    todo.append(j)
    return seen


# ---- the whole file set ----------------------------------------------------------------------

@needs_maps
def test_six_files_named_after_the_stem(files):
    assert sorted(files) == sorted(STEM + s for s in SUFFIXES)


@needs_maps
def test_generated_files_validate(tm, files, files_pool):
    assert tm.validate_testmap(files, STEM, molynx_in_pool=False) == []
    assert tm.validate_testmap(files_pool, STEM_POOL, molynx_in_pool=True) == []


@needs_maps
def test_validator_rejects_pool_flag_mismatch(tm, files, files_pool):
    assert tm.validate_testmap(files, STEM, molynx_in_pool=True)
    assert tm.validate_testmap(files_pool, STEM_POOL, molynx_in_pool=False)


@needs_maps
def test_deterministic(tm, files):
    assert tm.testmap_files(STEM, molynx_in_pool=False) == files


@needs_maps
def test_variants_differ_only_in_variables_and_names(files, files_pool):
    for s in ("Map.kld", "Things.kld", "Players.kld", "Triggers.kld"):
        assert files[STEM + s] == files_pool[STEM_POOL + s], s
    assert files[STEM + "Variables.kld"] != files_pool[STEM_POOL + "Variables.kld"]


@needs_maps
def test_validator_works_on_files_read_back_from_disk(tm, files, tmp_path):
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    back = {name: (tmp_path / name).read_bytes() for name in files}
    assert tm.validate_testmap(back, STEM, molynx_in_pool=False) == []


# ---- rule 1: size and border ---------------------------------------------------------------------

@needs_maps
def test_rule1_size_and_border(files):
    bw, bh, bt = _tiles(_base()["Map.kld"])
    w, h, t = _tiles(_f(files, STEM, "Map.kld"))
    assert (w, h) == (bw, bh) == (69, 69)
    ring = [y * w + x for y in range(h) for x in range(w) if x in (0, w - 1) or y in (0, h - 1)]
    assert [t[i] for i in ring] == [bt[i] for i in ring]


@needs_maps
def test_rule1_validator_catches_changed_border_byte(tm, files):
    bad = _with(files, STEM, "Map.kld", _set_tile(_f(files, STEM, "Map.kld"), 30, 0, b"\x01\x02\x01\x00"))
    assert any("rule 1" in e for e in tm.validate_testmap(bad, STEM, molynx_in_pool=False))


# ---- rule 2: hearts ------------------------------------------------------------------------------

@needs_maps
def test_rule2_two_hearts_with_base_shape_and_room_things(files):
    w, h, t = _tiles(_f(files, STEM, "Map.kld"))
    bw, bh, bt = _tiles(_base()["Map.kld"])
    things = _things(_f(files, STEM, "Things.kld"))
    base_things = _things(_base()["Things.kld"])
    for owner in (3, 4):
        cells = [i for i, x in enumerate(t) if x[0] == 14 and x[1] == owner]
        bcells = [i for i, x in enumerate(bt) if x[0] == 14 and x[1] == owner]
        assert len(cells) == len(bcells) == 25
        x0, y0 = min(i % w for i in cells), min(i // w for i in cells)
        bx0, by0 = min(i % bw for i in bcells), min(i // bw for i in bcells)
        assert sorted((i % w - x0, i // w - y0, t[i]) for i in cells) == \
            sorted((i % bw - bx0, i // bw - by0, bt[i]) for i in bcells)
        rooms = [d for k, d in things if k == 204 and d[19] == owner]
        brooms = [d for k, d in base_things if k == 204 and d[19] == owner]
        assert len(rooms) == len(brooms) == 1
        x, y = struct.unpack_from("<ii", rooms[0])
        bx, by = struct.unpack_from("<ii", brooms[0])
        assert (x - x0, y - y0) == (bx - bx0, by - by0)
        assert rooms[0][8:] == brooms[0][8:]
    assert sum(1 for x in t if x[0] == 14) == 50


@needs_maps
def test_rule2_keeper1_heart_stays_where_its_camera_looks(files):
    w, h, t = _tiles(_f(files, STEM, "Map.kld"))
    assert t[37 * w + 4] == b"\x0e\x03\x01\x00"


@needs_maps
def test_rule2_validator_catches_moved_heart(tm, files):
    m = _f(files, STEM, "Map.kld")
    w, h, t = _tiles(m)
    cells = [i for i, x in enumerate(t) if x[0] == 14 and x[1] == 3]
    claimed = b"\x08\x03\x01\x00"
    for i in cells:  # shift the whole heart one tile right (the Things stay)
        m = _set_tile(m, i % w, i // w, claimed)
    for i in cells:
        m = _set_tile(m, i % w + 1, i // w, t[i])
    errors = tm.validate_testmap(_with(files, STEM, "Map.kld", m), STEM, molynx_in_pool=False)
    assert any("rule 2" in e for e in errors)


# ---- rule 3: portal ------------------------------------------------------------------------------

@needs_maps
def test_rule3_one_keeper1_portal_3x3_connected_to_heart(files):
    w, h, t = _tiles(_f(files, STEM, "Map.kld"))
    cells = [i for i, x in enumerate(t) if x[0] == 12]
    assert len(cells) == 9 and all(t[i] == b"\x0c\x03\x01\x00" for i in cells)
    xs, ys = {i % w for i in cells}, {i // w for i in cells}
    assert len(xs) == len(ys) == 3
    heart = next(i for i, x in enumerate(t) if x[0] == 14 and x[1] == 3)
    area = _component(w, h, t, heart, lambda x: x[0] in (3, 8, 12, 14) and x[1] in (2, 3))
    assert set(cells) <= area


@needs_maps
def test_rule3_validator_catches_missing_portal(tm, files):
    m = _f(files, STEM, "Map.kld")
    w, h, t = _tiles(m)
    for i, x in enumerate(t):
        if x[0] == 12:
            m = _set_tile(m, i % w, i // w, b"\x08\x03\x01\x00")
    errors = tm.validate_testmap(_with(files, STEM, "Map.kld", m), STEM, molynx_in_pool=False)
    assert any("rule 3" in e for e in errors)


# ---- rule 4: Keeper 1's open area ----------------------------------------------------------------

@needs_maps
def test_rule4_open_area(files):
    w, h, t = _tiles(_f(files, STEM, "Map.kld"))
    heart = next(i for i, x in enumerate(t) if x[0] == 14 and x[1] == 3)
    area = _component(w, h, t, heart, lambda x: (x[0] == 8 and x[1] == 3) or x[0] in (3, 12, 14) and x[1] in (2, 3))
    floor = [i for i in area if t[i][0] in (3, 8)]
    assert len(floor) >= 900
    assert sum(1 for i in area if t[i][0] == 8 and t[i][1] == 3) >= 300
    # no hole: every tile that is not in the area is reachable from the border without crossing it
    outside = _component(w, h, t, 0, lambda x: True) - area
    border_reach = _component(w, h, [b"o" if i not in area else b"a" for i in range(w * h)], 0,
                              lambda x: x == b"o")
    assert outside == border_reach
    gems = [i for i, x in enumerate(t) if x[0] == 7]
    gold = [i for i, x in enumerate(t) if x[0] == 6]
    assert 4 <= len(gems) <= 6 and len(gold) == 2
    for i in gems + gold:
        x, y = i % w, i // w
        near = [(y + dy) * w + x + dx for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        assert any(j in area for j in near)
        assert all(j in area or t[j][0] in (2, 6, 7) for j in near)
    assert {x[0] for x in t} <= {1, 2, 3, 6, 7, 8, 12, 14, 30}


@needs_maps
def test_rule4_validator_catches_rock_inside_area(tm, files):
    m = _set_tile(_f(files, STEM, "Map.kld"), 20, 30, b"\x02\x02\x01\x00")
    errors = tm.validate_testmap(_with(files, STEM, "Map.kld", m), STEM, molynx_in_pool=False)
    assert any("rule 4" in e for e in errors)


# ---- rule 5: Keeper 2 sealed ---------------------------------------------------------------------

@needs_maps
def test_rule5_keeper2_sealed(files):
    w, h, t = _tiles(_f(files, STEM, "Map.kld"))
    heart = next(i for i, x in enumerate(t) if x[0] == 14 and x[1] == 4)
    chamber = _component(w, h, t, heart, lambda x: x[0] not in SOLID)
    assert all(t[i][1] == 4 for i in chamber)
    floor = {i for i in chamber}
    for i in chamber:
        x, y = i % w, i // w
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                j = (y + dy) * w + x + dx
                if j not in floor:
                    assert t[j][0] == 1, (x + dx, y + dy)
    assert not any(t[i][0] in (6, 7) for i in chamber)


@needs_maps
def test_rule5_validator_catches_gap_in_wall(tm, files):
    m = _f(files, STEM, "Map.kld")
    w, h, t = _tiles(m)
    heart = [i for i, x in enumerate(t) if x[0] == 14 and x[1] == 4]
    x1, y = max(i % w for i in heart), min(i // w for i in heart) + 2
    m = _set_tile(m, x1 + 3, y, b"\x03\x02\x01\x00")  # a dirt tile in the wall, 2 tiles out of the ring
    errors = tm.validate_testmap(_with(files, STEM, "Map.kld", m), STEM, molynx_in_pool=False)
    assert any("rule 5" in e for e in errors)


# ---- rule 6: things ------------------------------------------------------------------------------

@needs_maps
def test_rule6_things(files):
    w, h, t = _tiles(_f(files, STEM, "Map.kld"))
    things = _things(_f(files, STEM, "Things.kld"))
    assert {k for k, _ in things} == {200, 204}
    creatures = [(d[26], d[27], *struct.unpack_from("<ii", d)) for k, d in things if k == 200]
    assert sorted(c[:2] for c in creatures) == sorted([(1, 3)] * 6 + [(47, 3)] + [(1, 4)] * 4)
    for cid, owner, x, y in creatures:
        tile = t[y * w + x]
        assert tile[0] not in SOLID and tile[1] == owner
    (mx, my), = [(c[2], c[3]) for c in creatures if c[0] == 47]
    assert t[my * w + mx][0] in (3, 8)
    assert any(t[(my + dy) * w + mx + dx][:2] == b"\x0e\x03" for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    assert sum(1 for k, _ in things if k == 204) == 2


@needs_maps
def test_rule6_molynx_record_is_the_base_bile_demon_record(files):
    base = [d for k, d in _things(_base()["Things.kld"]) if k == 200 and d[26] == 3 and d[27] == 3][0]
    mol = [d for k, d in _things(_f(files, STEM, "Things.kld")) if k == 200 and d[26] == 47][0]
    # the base Keeper 1 Bile Demon's, but at level 5 (review round 2: his level-4 ability at once)
    from scripts.dk2_testmap import MOLYNX_LEVEL
    assert mol[14] == MOLYNX_LEVEL == 5 and base[14] == 2
    assert mol[8:14] + mol[15:26] == base[8:14] + base[15:26] and mol[27] == 3


@needs_maps
def test_rule6_validator_catches_extra_creature(tm, files):
    from scripts.dk2kld_splice import Thing, read_things, write_things
    block, things = read_things(_f(files, STEM, "Things.kld"))
    extra = next(x for x in things if x.type == 200 and x.data[26] == 1 and x.data[27] == 3)
    bad = write_things(block, things + [Thing(200, extra.data[:26] + bytes([3]) + extra.data[27:])])
    errors = tm.validate_testmap(_with(files, STEM, "Things.kld", bad), STEM, molynx_in_pool=False)
    assert any("rule 6" in e for e in errors)


# ---- rule 7: players -----------------------------------------------------------------------------

@needs_maps
def test_rule7_players(files):
    base, new = _base()["Players.kld"], _f(files, STEM, "Players.kld")
    assert len(new) == len(base) == 56 + 4 * 205
    k1 = 56 + 2 * 205
    assert struct.unpack_from("<iI", new, k1) == (100000, 0)
    assert new[:k1] + new[k1 + 4:] == base[:k1] + base[k1 + 4:]


# ---- rule 8: variables ---------------------------------------------------------------------------

def _entries(v: bytes):
    n = struct.unpack_from("<I", v, 20)[0]
    return [struct.unpack_from("<iiii", v, 56 + 16 * i) for i in range(n)]


@needs_maps
@pytest.mark.parametrize("pool", [False, True])
def test_rule8_variables(files, files_pool, pool):
    v = _f(files_pool, STEM_POOL, "Variables.kld") if pool else _f(files, STEM, "Variables.kld")
    assert struct.unpack_from("<I", v, 20)[0] * 16 + 56 == len(v)
    assert struct.unpack_from("<I", v, 52)[0] == len(v) - 56
    assert struct.unpack_from("<I", v, 8)[0] == len(v)
    ents = _entries(v)
    base = _entries(_base()["Variables.kld"])
    assert len(ents) >= len(base)
    # the game settings the base overrides take the game's defaults (written out: Dead Walking's
    # 23 and 26 of 32000 instead of 60 flooded the hatchery in the 2026-11-08 test)
    defaults = {18: 16, 23: 60, 26: 60, 95: 5, 96: 2, 150: 30}
    for old, new in zip(base, ents):  # the base entries keep their place
        if old[0] == 2 and old[1] >> 16 == 3:
            assert new[:3] == old[:3] and new[3] == 3
        elif old[0] in defaults:
            assert new == (old[0], defaults[old[0]], old[2], old[3])
        else:
            assert new == old
    avail = [(e[1] & 0xFFFF, e[1] >> 16, e[2]) for e in ents if e[0] == 2]
    assert len(avail) == len(set(avail))
    k1 = {(t, i): e[3] for e in ents if e[0] == 2 and e[1] >> 16 == 3 for t, i in [(e[1] & 0xFFFF, e[2])]}
    assert all(val == 3 for val in k1.values())
    for i in (1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        assert (1, i) in k1
    for i in range(1, 7):
        assert (4, i) in k1
    for i in [*range(1, 12), 16]:
        assert (3, i) in k1
    for i in [*range(1, 12), 13]:
        assert (5, i) in k1
    pool_ids = sorted(e[1] for e in ents if e[0] == 1 and e[3] == 3)
    creature_av = sorted(i for t, i in k1 if t == 2)
    want = sorted(SIEGE_POOL_IDS + ([47] if pool else []))
    # the Imp is available but not in the pool (review round: he could not pick up his Imps)
    assert pool_ids == want and creature_av == sorted([1] + want)
    if pool:
        assert (1, 47, 50, 3) in ents
    pool_keys = [(e[1], e[3]) for e in ents if e[0] == 1]
    assert len(pool_keys) == len(set(pool_keys))


@needs_maps
def test_rule8_validator_catches_duplicate_availability(tm, files):
    v = _f(files, STEM, "Variables.kld")
    ents = _entries(v)
    dup = next(e for e in ents if e[0] == 2 and e[1] >> 16 == 3)
    from scripts.dk2kwd_blocks import rebuilt_block, write_blocks
    (block,) = read_blocks(v)
    bad = write_blocks([rebuilt_block(block, [struct.pack("<iiii", *e) for e in ents + [dup]])])
    errors = tm.validate_testmap(_with(files, STEM, "Variables.kld", bad), STEM, molynx_in_pool=False)
    assert any("rule 8" in e for e in errors)


# ---- rule 9: triggers ----------------------------------------------------------------------------

@needs_maps
def test_rule9_empty_triggers_like_the_editor(files):
    tr = _f(files, STEM, "Triggers.kld")
    auto = (EDITOR_DIR / "AutoSaveTriggers.kld").read_bytes()
    base = _base()["Triggers.kld"]
    assert len(tr) == len(auto) == 60
    # same layout as the editor's empty file, timestamps (bytes 32..51) from the base
    assert tr[:32] == auto[:32] and tr[52:] == auto[52:]
    assert tr[32:52] == base[32:52]


@needs_maps
def test_rule9_validator_compares_with_the_editors_file(tm, files):
    tr = _f(files, STEM, "Triggers.kld")
    auto = (EDITOR_DIR / "AutoSaveTriggers.kld").read_bytes()
    bad = bytearray(tr)
    bad[28] = 2  # the editor's file has 1 here (the u32 after the two counts)
    errors = tm.validate_testmap(_with(files, STEM, "Triggers.kld", bytes(bad)), STEM, molynx_in_pool=False)
    assert any("rule 9" in e for e in errors)
    # the editor's own file is not enough: rule 9 keeps the base's timestamps
    assert auto[32:52] != tr[32:52]
    errors = tm.validate_testmap(_with(files, STEM, "Triggers.kld", auto), STEM, molynx_in_pool=False)
    assert any("rule 9" in e for e in errors)


@needs_maps
def test_rule9_validator_does_not_use_the_generators_helper(tm):
    import inspect
    body = inspect.getsource(tm.validate_testmap)
    assert "_empty_triggers" not in body and "AutoSaveTriggers.kld" in body


# ---- rule 10: level file -------------------------------------------------------------------------

@needs_maps
def test_rule10_level_file(files, files_pool):
    base = _base()[".kwd"]
    for fs, stem, name in ((files, STEM, "Molynx Test"), (files_pool, STEM_POOL, "Molynx Test Portal")):
        k = fs[stem + ".kwd"]
        assert len(k) == len(base)
        assert k[:56] == base[:56]
        assert k[56:56 + 128].decode("utf-16-le").split("\0")[0] == name
        assert struct.unpack_from("<H", k, 56 + 25484) == struct.unpack_from("<H", base, 56 + 25484)
        paths_at = 56 + 25683 + 8
        paths = [k[paths_at + 72 * i + 8:paths_at + 72 * i + 72].split(b"\0")[0].decode() for i in range(7)]
        assert "Data\\Editor\\GlobalVariables.kwd" in paths
        for s in ("Globals.kld", "Players.kld", "Map.kld", "Triggers.kld", "Variables.kld", "Things.kld"):
            assert f"Data\\Editor\\Maps\\{stem}{s}" in paths
        changed = {i for i in range(len(k)) if k[i] != base[i]}
        allowed = set(range(56, 56 + 128)) | set(range(56 + 128, 56 + 128 + 2048)) | \
            set(range(paths_at, paths_at + 7 * 72)) | set(range(56 + 4480, 56 + 4482))  # script root
        assert changed <= allowed


@needs_maps
def test_rule10_base_stem_keeps_base_paths(tm):
    k = tm.testmap_files("Dead Walking", molynx_in_pool=False)["Dead Walking.kwd"]
    base = _base()[".kwd"]
    assert k[56 + 25683:] == base[56 + 25683:]
    # only the name (wchar[64] at 56) and the description (wchar[1024] at 184) differ from the base
    changed = {i for i in range(len(k)) if k[i] != base[i]}
    root = set(range(56 + 4480, 56 + 4482))  # the script root, 1 -> 0 (Test B)
    assert changed and changed <= set(range(56, 56 + 128)) | set(range(184, 184 + 2048)) | root
    assert k[56:56 + 128].decode("utf-16-le").split("\0")[0] == "Molynx Test"
    assert k[184:184 + 2048].decode("utf-16-le").split("\0")[0] == tm.DESCRIPTION


def _with_wstr(kwd: bytes, at: int, chars: int, text: str) -> bytes:
    out = bytearray(kwd)
    out[at:at + 2 * chars] = text.encode("utf-16-le").ljust(2 * chars, b"\0")
    return bytes(out)


@needs_maps
def test_rule10_validator_checks_the_description(tm, files):
    bad = _with_wstr(files[STEM + ".kwd"], 184, 1024, "Some other text.")
    errors = tm.validate_testmap(_with(files, STEM, ".kwd", bad), STEM, molynx_in_pool=False)
    assert any("rule 10" in e and "description" in e for e in errors)
    empty = _with_wstr(files[STEM + ".kwd"], 184, 1024, "")
    errors = tm.validate_testmap(_with(files, STEM, ".kwd", empty), STEM, molynx_in_pool=False)
    assert any("rule 10" in e and "description" in e for e in errors)


@needs_maps
def test_rule10_validator_rejects_an_empty_name(tm, files):
    bad = _with_wstr(files[STEM + ".kwd"], 56, 64, "")
    errors = tm.validate_testmap(_with(files, STEM, ".kwd", bad), STEM, molynx_in_pool=False)
    assert any("rule 10" in e and "name" in e for e in errors)


# ---- rule 11: containers -------------------------------------------------------------------------

@needs_maps
def test_rule11_containers(files):
    base = _base()
    for s in SUFFIXES:
        data, b = files[STEM + s], base[s]
        assert read_blocks(data)
        sig, four, size, check1, hend = struct.unpack_from("<5I", data, 0)
        hsz = 28 + hend
        assert (sig, four, check1, hend) == struct.unpack_from("<4I", b, 0)[:2] + struct.unpack_from("<2I", b, 12)
        assert size == len(data)
        assert struct.unpack_from("<I", data, hsz - 8) == struct.unpack_from("<I", b, hsz - 8)
        if s != ".kwd":
            assert struct.unpack_from("<I", data, hsz - 4)[0] == len(data) - hsz
    # timestamps copied from the base
    for s in ("Things.kld", "Players.kld", "Variables.kld"):
        assert files[STEM + s][28:48] == base[s][28:48]
    assert files[STEM + ".kwd"][28:48] == base[".kwd"][28:48]
    things = files[STEM + "Things.kld"]
    assert struct.unpack_from("<I", things, 20)[0] == len(_things(things))


# ---- writer exactness ----------------------------------------------------------------------------

@needs_maps
@pytest.mark.parametrize("name", ["Dead Walking", "Siege", "Small"])
def test_writers_round_trip_shipped_maps(tm, name):
    for s, data in _base(name).items():
        assert tm.rewrite_unchanged(s, data) == data, (name, s)


# ---- constants checked against the editor files ----------------------------------------------------

@needs_maps
def test_terrain_ids_match_terrain_kwd(tm):
    (block,) = read_blocks((EDITOR_DIR / "Terrain.kwd").read_bytes())
    size = len(block.content) // block.item_count
    names = {i + 1: block.content[i * size:i * size + 32].split(b"\0")[0].decode() for i in range(block.item_count)}
    assert names[tm.IMPENETRABLE] == "Impenetrable Rock" and names[tm.ROCK] == "Rock"
    assert names[tm.DIRT] == "Dirt Path" and names[tm.GOLD] == "Gold" and names[tm.GEMS] == "Gems"
    assert names[tm.CLAIMED] == "Claimed Path" and names[tm.PORTAL] == "Portal"
    assert names[tm.HEART] == "Dungeon Heart" and names[tm.EDGE] == "Edge of Map"


# ---- previews ------------------------------------------------------------------------------------

@needs_maps
def test_preview_text(tm, files):
    text = tm.preview_text(files, STEM)
    lines = text.splitlines()
    grid = lines[:69]
    assert all(len(line) == 69 for line in grid)
    assert "legend" in text.lower()
    assert sum(line.count("M") for line in grid) == 1


@needs_maps
def test_preview_png(tm, files, tmp_path):
    from PIL import Image
    out = tmp_path / "m.png"
    tm.preview_png(files, STEM, out)
    with Image.open(out) as img:
        assert img.size[0] >= 69 and img.size[0] == img.size[1]


# ---- the level's script root (Test B) -----------------------------------------------------------

@needs_maps
def test_level_script_root_is_zero_like_the_maps_with_an_empty_triggers_file(files):
    """Dead Walking's root trigger 1 with the empty Triggers file hung the game (Test B)."""
    kwd = _f(files, STEM, ".kwd")
    assert struct.unpack_from("<H", kwd, tm_module().LEVEL_TRIGGER_ROOT_AT)[0] == 0
    assert struct.unpack_from("<H", _base()[".kwd"], tm_module().LEVEL_TRIGGER_ROOT_AT)[0] == 1


@needs_maps
def test_rule10_validator_fails_on_the_bases_script_root(tm, files):
    bad = bytearray(_f(files, STEM, ".kwd"))
    struct.pack_into("<H", bad, tm.LEVEL_TRIGGER_ROOT_AT, 1)
    errors = tm.validate_testmap(_with(files, STEM, ".kwd", bytes(bad)), STEM, molynx_in_pool=False)
    assert any("trigger root" in e for e in errors)


def tm_module():
    import scripts.dk2_testmap as module
    return module
