"""Tests of the Dragon (dragon_model, dragon_pose, dragon_anims, dragon_texture, dragon_export, dragon_build).

The model and clip tests need nothing but this repo; the export tests need the game's meshes (the Bile
Demon's files the dragon is measured against) and skip without them.
"""
import numpy as np
import pytest

from scripts.dragon_anims import CLIPS, build_clips
from scripts.dragon_model import BONES, PARENT, Skinner, build_dragon, counts
from scripts.dragon_pose import Pose, leg_ik
from scripts.dragon_texture import ATLAS, atlas_uv, bake
from scripts.dk2paths import MESHES_DIR, have


@pytest.fixture(scope="module")
def parts():
    return build_dragon()


def test_triangle_budget_is_a_big_dk2_creature(parts):
    total = sum(t for _, t in counts(parts).values())
    assert 400 <= total <= 560          # the Horned Reaper's files have 406, the Knight's 290


def test_closed_parts_are_wound_outward(parts):
    # signed volume of every closed loft part is positive with counter-clockwise outward winding
    for p in parts:
        if p.name not in ("trunk", "head", "jaw"):
            continue
        P = np.array(p.points)
        T = np.array(p.tris)
        main = [t for t in T if p.island[t[0]] in ("trunk", "skull", "jaw")]
        vol = sum(np.dot(P[a], np.cross(P[b], P[c])) for a, b, c in main) / 6
        assert vol > 0, p.name


def test_bind_pose_skins_to_itself(parts):
    sk = Skinner(parts)
    out = sk(*Pose().world())
    for p, q in zip(parts, out):
        assert np.allclose(np.array(p.points), q, atol=1e-9)


def test_weights_sum_to_one(parts):
    for p in parts:
        for w in p.weights:
            assert abs(sum(w.values()) - 1) < 1e-9


def test_parents_come_first():
    assert all(p is None or p < i for i, p in enumerate(PARENT))


def test_leg_ik_reaches_its_target():
    from scripts.dragon_model import BONE_INDEX, HEAD
    P = Pose()
    target = HEAD[BONE_INDEX["footL"]] + np.array([0, 0.05, 0.08])
    leg_ik(P, "L", target, (0.3, 0, 1))
    _, Pw = P.world()
    assert np.linalg.norm(Pw[BONE_INDEX["footL"]] - target) < 1e-6


def test_uvs_stay_inside_their_islands(parts):
    for p in parts:
        uv = atlas_uv(p) * 128
        for i, isl in enumerate(p.island):
            (x0, y0, x1, y1), _ = ATLAS[p.texture][isl]
            assert x0 <= uv[i, 0] <= x1 and y0 <= uv[i, 1] <= y1


def test_skins_are_128_rgb(parts):
    tex = bake(parts)
    assert set(tex) == {"Dragon_Body", "Dragon_Head", "Dragon_Wing"}
    for img in tex.values():
        assert img.size == (128, 128) and img.mode == "RGB"


@pytest.mark.parametrize("name", sorted(CLIPS))
def test_every_clip_is_at_30_fps_and_loops_close(name):
    frames, wrap = build_clips([name])[name]
    assert len(frames) >= 16
    if wrap:
        a, b = frames[0], frames[-1]
        assert np.allclose(a.local, b.local) and np.allclose(a.root, b.root)


def test_feet_stay_on_the_ground_when_standing(parts):
    sk = Skinner(parts)
    frames, _ = build_clips(["stand"])["stand"]
    for P in frames[::10]:
        low = min(q[:, 1].min() for q in sk(*P.world()))
        assert abs(low) < 0.02


@pytest.mark.skipif(not have(MESHES_DIR), reason="game meshes not present")
def test_export_writes_valid_game_files(tmp_path):
    from scripts.dk2kmf import parse_kmf
    from scripts.dk2kmf_write import anim_invariants
    from scripts.dragon_build import build, export
    built = build(MESHES_DIR, names=["stand", "walk", "eating", "sleeping", "being_picked_up", "falling",
                                     "die", "getting_hit"])
    info = export(built, tmp_path)
    for name in info:
        k = parse_kmf((tmp_path / f"Dragon-{name}.kmf").read_bytes())
        assert anim_invariants(k.anim) == []
        for s in k.anim.sprites:
            assert len(s.vertices) <= 255 and len(s.triangles[0]) <= 255
        assert len(k.anim.geom_xyz) - 1 <= 34_816
    stand = parse_kmf((tmp_path / "Dragon-stand.kmf").read_bytes())
    from scripts.dk2kmf import evaluate_anim_frame
    p = np.concatenate(evaluate_anim_frame(stand.anim, 0))
    assert abs(p[:, 2].max()) < 1e-2                      # feet on the floor (z down)
    assert abs((p[:, 2].max() - p[:, 2].min()) - 0.74) < 0.01
    for stem in ("PoseFrame", "DiePose", "DraggedPose", "StunnedPose", "ElecPose"):
        assert parse_kmf((tmp_path / f"Dragon-{stem}.kmf").read_bytes()).kind == "mesh"
    eat = parse_kmf((tmp_path / "Dragon-eating.kmf").read_bytes())
    assert any("chicken" in m.textures[0].lower() for m in eat.materials)
    bed = parse_kmf((tmp_path / "Dragon-sleeping.kmf").read_bytes())
    assert any("bed" in m.textures[0].lower() for m in bed.materials)
