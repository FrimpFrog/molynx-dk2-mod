import copy

import numpy as np
import pytest

from scripts.dk2kmf import (Anim, Kmf, Material, Mesh, Sprite, Vertex, evaluate_anim_frame,
                            parse_kmf)
from scripts.dk2kmf_write import anim_invariants, encode_anim, write_kmf
from scripts.dk2paths import MESHES_DIR, have

needs_meshes = pytest.mark.skipif(not have(MESHES_DIR), reason="extracted DK2 meshes not present")

# Game files that are not reproduced byte for byte, with the reason. Every other file must be.
NOT_BYTE_IDENTICAL: dict[str, str] = {}

LODS = 16  # every game sprite carries 16 LOD triangle lists


def _load(name: str) -> Kmf:
    return parse_kmf((MESHES_DIR / name).read_bytes())


def _sprite(material: int, rows: list[int]) -> Sprite:
    n = len(rows)
    tris = [[(i, (i + 1) % n, (i + 2) % n) for i in range(n - 2)]] + [[] for _ in range(LODS - 1)]
    verts = [Vertex((i * 1000, i * 500), (0.0, 0.0, -1.0), itab_index=r) for i, r in enumerate(rows)]
    return Sprite(material, tris, verts, mm_factor=1.5)


def _decoded(anim: Anim) -> list[list[np.ndarray]]:
    return [evaluate_anim_frame(anim, f) for f in range(anim.frame_count)]


def _max_error(anim: Anim, frames) -> float:
    """Largest per-axis distance between decoded positions and the input, over every frame."""
    return max(float(np.abs(got - want).max())
               for f, want_f in enumerate(frames)
               for got, want in zip(evaluate_anim_frame(anim, f), want_f))


def _keys(anim: Anim) -> int:
    return len(anim.geom_xyz) - 1


def _bound(anim: Anim) -> float:
    return anim.scale / 511 * 1.01


def _random_walk(frames: int, rows_per_sprite: list[list[int]], seed: int, step: float = 0.01):
    rng = np.random.default_rng(seed)
    nrows = max(r for rows in rows_per_sprite for r in rows) + 1
    tracks = rng.normal(0, 0.2, (1, nrows, 3)) + np.cumsum(rng.normal(0, step, (frames, nrows, 3)),
                                                           axis=0)
    return [[tracks[f, rows] for rows in rows_per_sprite] for f in range(frames)]


def _roundtrip_kmf(anim: Anim) -> Kmf:
    model = Kmf(17, "anim", [Material("Mat", ["Tex"], 0, 1.0, 0.0, "")], anim=anim)
    data = write_kmf(model)
    assert int.from_bytes(data[4:8], "little") == len(data)  # chunk sizes count their header
    back = parse_kmf(data)
    assert back.end == len(data)
    assert write_kmf(back) == data
    return back


# ---------------------------------------------------------------- lossless writer on game data

@needs_meshes
def test_every_game_file_is_written_back_byte_for_byte():
    files = sorted(MESHES_DIR.glob("*.kmf"))
    assert len(files) == 2015
    differ = []
    for path in files:
        data = path.read_bytes()
        if write_kmf(parse_kmf(data)) != data:
            differ.append(path.name)
    assert sorted(differ) == sorted(NOT_BYTE_IDENTICAL)


@needs_meshes
def test_anim_invariants_hold_for_every_game_animation():
    bad = {}
    for path in sorted(MESHES_DIR.glob("*.kmf")):
        model = parse_kmf(path.read_bytes())
        if model.kind == "anim" and (problems := anim_invariants(model.anim)):
            bad[path.name] = problems[:3]
    assert bad == {}


@needs_meshes
def test_parsed_controls_keep_the_raw_ctrl_entries():
    model = _load("Bile_Demon-Walk.kmf")
    assert len(model.anim.controls) == 20 * 8
    assert model.anim.controls[:8] == (0x570000).to_bytes(4, "little") + (1000).to_bytes(4, "little")


# ---------------------------------------------------------------- writer on fresh models

def test_fresh_static_mesh_round_trips():
    sprite = Sprite(0, [[(0, 1, 2)]] + [[] for _ in range(LODS - 1)],
                    [Vertex((0, 0), (0.0, 0.0, -1.0), geom=0),
                     Vertex((32768, 0), (0.0, 0.0, -1.0), geom=1),
                     Vertex((0, 32768), (0.0, 0.0, -1.0), geom=2)], mm_factor=2.0)
    mesh = Mesh("Tri", (0.5, 0.5, 0.0), 0.75, [sprite], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                                       (0.0, 1.0, 0.0)])
    model = Kmf(17, "mesh", [Material("M", ["T1", "T2"], 3, 1.0, 0.5, "env")], mesh=mesh)
    data = write_kmf(model)
    back = parse_kmf(data)
    assert back.end == len(data)
    assert back.mesh == mesh and back.materials == model.materials
    assert write_kmf(back) == data


def test_fresh_group_round_trips():
    model = Kmf(17, "grop", groups=[("Part_A", (1.0, 2.0, 3.0)), ("Part_B", (0.0, -1.0, 0.5))])
    data = write_kmf(model)
    back = parse_kmf(data)
    assert back.end == len(data) and back.groups == model.groups


# ---------------------------------------------------------------- encoder

def test_random_walk_245_frames_two_sprites_decodes_within_one_step():
    rows = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    frames = _random_walk(245, rows, seed=7)
    sprites = [_sprite(0, rows[0]), _sprite(1, rows[1])]
    anim = encode_anim("Walker", sprites, frames, wrap=False, controls=b"")
    assert anim.frame_count == 245 and anim.index_count == 9 and anim.itab.shape == (2, 9)
    assert anim_invariants(anim) == []
    assert _max_error(anim, frames) <= _bound(anim)
    # the second block on its own
    for f in range(128, 245):
        for got, want in zip(evaluate_anim_frame(anim, f), frames[f]):
            assert np.abs(got - want).max() <= _bound(anim)
    back = _roundtrip_kmf(anim)
    for f in (0, 127, 128, 200, 244):
        for a, b in zip(evaluate_anim_frame(back.anim, f), evaluate_anim_frame(anim, f)):
            np.testing.assert_allclose(a, b, atol=1e-6)


def test_encoder_follows_game_quantisation_conventions():
    rows = [[0, 1, 2], [3, 4, 5]]
    frames = _random_walk(200, rows, seed=3)
    anim = encode_anim("Conv", [_sprite(0, rows[0]), _sprite(0, rows[1])], frames, False, b"")
    pts = np.concatenate([np.concatenate(f) for f in frames])
    step = anim.scale / 511
    centre = (pts.max(0) + pts.min(0)) / 2
    np.testing.assert_allclose(anim.pos, centre, atol=1e-6)            # bbox centre
    assert abs(anim.scale - np.abs(pts).max()) < 1e-6                   # largest |coordinate|
    assert abs(anim.cube_scale - np.linalg.norm(pts - centre, axis=1).max()) < 2 * step
    # keyframe lists: block 0 ends on frame 128 (the next block's first frame), the last block
    # ends on the last frame, and block 1 starts with block 0's final keyframe
    bounds = list(np.sort(anim.itab.reshape(-1))) + [_keys(anim)]
    for r in range(anim.index_count):
        s0 = int(anim.itab[0, r]); e0 = bounds[bounds.index(s0) + 1]
        s1 = int(anim.itab[1, r]); e1 = bounds[bounds.index(s1) + 1]
        assert anim.geom_frame_base[s0] == 0 and anim.geom_frame_base[e0 - 1] == 128
        assert anim.geom_frame_base[s1] == 0 and anim.geom_frame_base[e1 - 1] == 199 - 128
        np.testing.assert_array_equal(anim.geom_xyz[e0 - 1], anim.geom_xyz[s1])


def test_vertices_on_one_row_share_a_track_and_a_still_vertex_decodes_exactly():
    rows = [[0, 1, 2], [2, 1, 3]]  # rows 1 and 2 are used by both sprites
    frames = _random_walk(150, rows, seed=11)
    still = np.array([0.25, -0.125, -0.5])
    for f in frames:  # row 2 never moves
        f[0][2] = still
        f[1][0] = still
    for f in frames:
        f[1][1] = f[0][1]
    anim = encode_anim("Still", [_sprite(0, rows[0]), _sprite(0, rows[1])], frames, False, b"")
    assert anim.index_count == 4 and anim_invariants(anim) == []
    first = evaluate_anim_frame(anim, 0)[0][2]
    for f in range(150):
        a, b = evaluate_anim_frame(anim, f)
        np.testing.assert_array_equal(a[2], first)
        np.testing.assert_array_equal(b[0], first)
        np.testing.assert_array_equal(a[1], b[1])
    q = np.round(still / anim.scale * 511) / 511 * anim.scale
    np.testing.assert_allclose(first, q, atol=1e-9)
    # a still row has exactly two keyframes per block (its first and last frame)
    starts = np.sort(anim.itab.reshape(-1))
    for k in range(2):
        s = int(anim.itab[k, 2])
        assert starts[np.searchsorted(starts, s) + 1] - s == 2


def test_one_frame_pose_round_trips():
    rows = [[0, 1, 2, 3]]
    frames = _random_walk(1, rows, seed=5)
    anim = encode_anim("Pose", [_sprite(0, rows[0])], frames, wrap=False, controls=b"")
    assert anim.frame_count == 1 and _keys(anim) == 4 and anim_invariants(anim) == []
    assert _max_error(anim, frames) <= _bound(anim)
    back = _roundtrip_kmf(anim)
    np.testing.assert_allclose(evaluate_anim_frame(back.anim, 0)[0], evaluate_anim_frame(anim, 0)[0])


def test_reduction_drops_keyframes_on_smooth_motion_but_not_on_noise():
    rows = [[0, 1, 2, 3, 4, 5]]
    t = np.arange(240)[:, None]
    smooth = [[np.stack([0.3 * np.sin(t[f] / 40.0 + np.arange(6)),
                         0.2 * np.cos(t[f] / 50.0 + np.arange(6)),
                         -0.4 + 0.01 * np.arange(6) + 0 * t[f]], axis=1)] for f in range(240)]
    anim = encode_anim("Smooth", [_sprite(0, rows[0])], smooth, False, b"")
    full = 6 * (240 + 1)  # every frame a keyframe, plus the shared block-boundary keyframe
    assert _keys(anim) < full / 4
    assert _max_error(anim, smooth) <= _bound(anim)
    rng = np.random.default_rng(1)
    noise = [[rng.uniform(-0.5, 0.5, (6, 3))] for _ in range(240)]
    anim = encode_anim("Noise", [_sprite(0, rows[0])], noise, False, b"")
    assert _keys(anim) == full
    assert _max_error(anim, noise) <= _bound(anim)


def test_reduce_false_keys_every_frame_like_the_fully_sampled_game_files():
    rows = [[0, 1, 2]]
    frames = [[np.zeros((3, 3)) + [0.1, 0.2, -0.3]] for _ in range(130)]
    anim = encode_anim("Flat", [_sprite(0, rows[0])], frames, False, b"", reduce=False)
    assert _keys(anim) == 3 * 131 and anim_invariants(anim) == []
    assert (anim.offsets[:, :128] == np.arange(128)).all()


def test_wrap_animation_must_end_on_its_first_frame():
    rows = [[0, 1, 2]]
    frames = _random_walk(40, rows, seed=2, step=0.05)
    with pytest.raises(ValueError, match="wrap"):
        encode_anim("Loop", [_sprite(0, rows[0])], frames, wrap=True, controls=b"")
    frames[-1] = [a.copy() for a in frames[0]]
    anim = encode_anim("Loop", [_sprite(0, rows[0])], frames, wrap=True, controls=b"")
    assert anim.wrap and _max_error(anim, frames) <= _bound(anim)


def test_encoder_rejects_inconsistent_input():
    frames = _random_walk(10, [[0, 1, 2]], seed=4)
    with pytest.raises(ValueError, match="row"):  # row 1 is never used
        encode_anim("Gap", [_sprite(0, [0, 2, 3])], _random_walk(10, [[0, 2, 3]], 4), False, b"")
    with pytest.raises(ValueError, match="row"):  # two vertices on row 0 disagree
        encode_anim("Clash", [_sprite(0, [0, 0, 1])], frames, False, b"")
    with pytest.raises(ValueError, match="controls"):
        encode_anim("Ctrl", [_sprite(0, [0, 1, 2])], frames, False, b"\0" * 5)
    with pytest.raises(ValueError, match="shape"):
        encode_anim("Shape", [_sprite(0, [0, 1, 2])], [[np.zeros((2, 3))]], False, b"")


# ---------------------------------------------------------------- invariants catch corruption

def _valid_anim() -> Anim:
    rows = [[0, 1, 2, 3]]
    return encode_anim("Inv", [_sprite(0, rows[0])], _random_walk(200, rows, seed=9), False, b"")


def test_invariants_report_itab_out_of_range():
    anim = copy.deepcopy(_valid_anim())
    anim.itab = anim.itab.copy()
    anim.itab[1, 3] = _keys(anim) + 5
    assert any("itab" in p for p in anim_invariants(anim))


def test_invariants_report_offset_past_the_keyframe_list():
    anim = copy.deepcopy(_valid_anim())
    anim.offsets = anim.offsets.copy()
    anim.offsets[2, 10] = 200
    assert any("offset" in p for p in anim_invariants(anim))


def test_invariants_report_decreasing_frame_numbers():
    anim = copy.deepcopy(_valid_anim())
    anim.geom_frame_base = anim.geom_frame_base.copy()
    s = int(anim.itab[0, 1])
    anim.geom_frame_base[s + 1] = 0
    assert any("do not increase" in p for p in anim_invariants(anim))


def test_invariants_report_a_block_that_stops_short():
    anim = copy.deepcopy(_valid_anim())
    anim.geom_frame_base = anim.geom_frame_base.copy()
    end = int(anim.itab[1, 0])  # row 3 of block 0 ends right before block 1 starts
    anim.geom_frame_base[end - 1] = 127
    assert any("last frame" in p for p in anim_invariants(anim))


# ---------------------------------------------------------------- re-encoding game animations

def _reencode(name: str, reduce: bool = True):
    orig = _load(name).anim
    frames = _decoded(orig)
    anim = encode_anim(orig.name, orig.sprites, frames, orig.wrap, orig.controls, reduce=reduce)
    return orig, frames, anim


def _assert_same_structure(orig: Anim, anim: Anim):
    assert anim_invariants(anim) == []
    assert anim.frame_count == orig.frame_count and anim.index_count == orig.index_count
    assert anim.itab.shape == orig.itab.shape
    assert [len(s.vertices) for s in anim.sprites] == [len(s.vertices) for s in orig.sprites]
    assert anim.controls == orig.controls and anim.wrap == orig.wrap
    step = orig.scale / 511
    assert abs(anim.scale - orig.scale) <= step
    assert np.abs(np.array(anim.pos) - orig.pos).max() <= step
    assert abs(anim.cube_scale - orig.cube_scale) <= 0.05 * orig.cube_scale


@needs_meshes
@pytest.mark.parametrize("name", ["Bile_Demon-Walk.kmf", "Goblin-Idle2.kmf",
                                  "Bile_Demon-Happy.kmf", "Imp_Idle1.kmf"])
def test_reencoded_game_animation_matches_the_original(name):
    orig, frames, anim = _reencode(name)
    _assert_same_structure(orig, anim)
    assert _max_error(anim, frames) <= _bound(anim)
    # the decoded original is piecewise linear on its own keyframes, so a reduced re-encode
    # never needs more keyframes than the original had
    assert _keys(anim) <= _keys(orig)
    back = _roundtrip_kmf(anim)
    assert anim_invariants(back.anim) == []


@needs_meshes
@pytest.mark.parametrize("name", ["Bile_Demon-Walk.kmf", "Bile_Demon-Happy.kmf"])
def test_unreduced_reencode_of_a_fully_sampled_file_is_the_same_file(name):
    """Every byte but HEAD pos and cube_scale (which the game's tool measured on unquantised
    source data we do not have) comes out identical."""
    data = (MESHES_DIR / name).read_bytes()
    model = parse_kmf(data)
    orig, frames, anim = _reencode(name, reduce=False)
    _assert_same_structure(orig, anim)
    out = write_kmf(Kmf(model.version, "anim", model.materials, anim=anim))
    head = data.index(b"ANIM") + 16
    at = data.index(b"\0", head) + 1 + 5 * 4  # name, then sprs/frames/index/geom/wrap
    assert len(out) == len(data)
    assert out[:at] == data[:at] and out[at + 16:] == data[at + 16:]
    assert _max_error(anim, frames) <= _bound(anim)


@needs_meshes
def test_reencoded_shortest_game_animation_matches_the_original():
    # the game's shortest animations have 2 frames (REA3_Melee3Start, Troll_RunStart)
    orig, frames, anim = _reencode("Troll_RunStart.kmf")
    assert orig.frame_count == 2
    _assert_same_structure(orig, anim)
    assert _max_error(anim, frames) <= _bound(anim)


@needs_meshes
def test_game_pose_mesh_encodes_as_a_one_frame_animation():
    model = _load("Bile_Demon-DiePose.kmf")
    geom = np.array(model.mesh.geometry)
    sprites = [Sprite(s.material, s.triangles,
                      [Vertex(v.uv, v.normal, itab_index=v.geom) for v in s.vertices], s.mm_factor)
               for s in model.mesh.sprites]
    frames = [[geom[[v.geom for v in s.vertices]] for s in model.mesh.sprites]]
    anim = encode_anim("Pose", sprites, frames, False, b"")
    assert anim.frame_count == 1 and _keys(anim) == len(geom) and anim_invariants(anim) == []
    assert _max_error(anim, frames) <= _bound(anim)
    _roundtrip_kmf(anim)
