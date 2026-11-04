"""Molynx's exported game files: models and animations (.kmf), textures and portraits.

The export is built once per module (about half a minute) from the repo's character code and the
real game data; every test skips when that data is not on this machine.
"""
import pickle
import re

import numpy as np
import pytest
from PIL import Image

from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf
from scripts.dk2kmf_write import anim_invariants, write_kmf
from scripts.dk2kwd_creatures import ANIM_SLOTS
from scripts.dk2paths import (BEHEMOTH_DIR, CHARACTER_PICKLE, EDITOR_DIR, GAME_DIR, MESHES_DIR,
                              have)
from scripts.molynx_record import CLIPS, POSE_STEMS, PREFIX, build_molynx, stems_used
import scripts.molynx_assets as assets

needs_data = pytest.mark.skipif(
    not (have(MESHES_DIR) and have(EDITOR_DIR) and have(GAME_DIR / "EngineTextures.dir")
         and have(BEHEMOTH_DIR / "behemoth.xml")),
    reason="DK2 meshes, editor data, texture cache or the MegaGlest Behemoth not present")
needs_portrait = pytest.mark.skipif(not have(assets.PORTRAIT_SRC),
                                    reason="the user's portrait image is not present")

STEMS = list(CLIPS) + list(POSE_STEMS)
RATIO = 1.15
GAME_FPS = 30


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    if not (have(MESHES_DIR) and have(EDITOR_DIR) and have(GAME_DIR / "EngineTextures.dir")
            and have(BEHEMOTH_DIR / "behemoth.xml")):
        pytest.skip("game data not present")
    out = tmp_path_factory.mktemp("molynx") / "assets"
    portrait = assets.PORTRAIT_SRC if have(assets.PORTRAIT_SRC) else None
    plan = assets.plan_molynx(height_ratio=RATIO)
    result = assets.export_molynx_assets(out, height_ratio=RATIO, portrait_src=portrait, plan=plan)
    models = {stem: parse_kmf(p.read_bytes()) for stem, p in result["meshes"].items()}
    return {"out": out, "plan": plan, "result": result, "models": models}


@pytest.fixture(scope="module")
def bile():
    if not have(EDITOR_DIR):
        pytest.skip("editor data not present")
    return assets.bile_demon_record()


def _bile_file(bile, slot):
    return parse_kmf(assets.game_mesh_path(bile.anim_name(slot)).read_bytes())


def _points(model, frame=0):
    return evaluate_anim_frame(model.anim, frame) if model.kind == "anim" else mesh_points(model)


def _sprites(model):
    return (model.anim or model.mesh).sprites


# ---------------------------------------------------------------- source character

@needs_data
@pytest.mark.skipif(not have(CHARACTER_PICKLE), reason="reference character pickle not present")
def test_rebuilt_character_equals_the_reference_build(export):
    data, textures = export["plan"].source
    ref, ref_tex = pickle.loads(CHARACTER_PICKLE.read_bytes())
    assert list(data) == list(ref)
    for name in ref:
        a, b = data[name], ref[name]
        assert (a["wrap"], a["fps"], len(a["frames"])) == (b["wrap"], b["fps"], len(b["frames"]))
        for fa, fb in zip(a["frames"], b["frames"]):
            for pa, pb in zip(fa, fb):
                np.testing.assert_array_equal(pa, pb)
        for pa, pb in zip(a["parts"], b["parts"]):
            assert pa["texture"] == pb["texture"]
            np.testing.assert_array_equal(pa["indices"], pb["indices"])
            np.testing.assert_array_equal(pa["uv"], pb["uv"])
    for name, image in ref_tex.items():
        np.testing.assert_array_equal(np.asarray(textures[name]), np.asarray(image))


# ---------------------------------------------------------------- files and names

@needs_data
def test_exports_every_clip_and_pose_as_its_own_kmf(export):
    meshes = export["result"]["meshes"]
    assert sorted(meshes) == sorted(STEMS)
    for stem, path in meshes.items():
        assert path.name == f"{PREFIX}{stem}.kmf"
        assert path.parent == export["out"]
        assert path.exists()
    assert sorted(p.name for p in export["out"].glob("*.kmf")) == sorted(
        f"{PREFIX}{s}.kmf" for s in STEMS)


@needs_data
def test_file_stems_cover_every_stage_2_slot_name_exactly(export, bile):
    stems = {p.stem for p in export["result"]["meshes"].values()}
    molynx = build_molynx(bile, 2, 47)
    written = {molynx.resource_name(slot) for slot in ANIM_SLOTS if slot != "unk_1"} - {""}
    assert written == {PREFIX + s for s in stems_used(bile)}
    assert written <= stems  # exact case
    for stem in stems:
        assert len(stem.encode("ascii")) <= 63  # 64-byte resource name with its NUL


@needs_data
def test_every_file_parses_completely_and_writes_back_identically(export):
    for stem, path in export["result"]["meshes"].items():
        data = path.read_bytes()
        model = parse_kmf(data)
        assert model.end == len(data), stem
        assert int.from_bytes(data[4:8], "little") == len(data)
        assert write_kmf(model) == data, stem
        assert model.version == 17


@needs_data
def test_clips_are_animations_that_pass_the_format_invariants(export):
    for stem in CLIPS:
        model = export["models"][stem]
        assert model.kind == "anim", stem
        assert model.anim.name == f"{PREFIX}{stem}"
        assert anim_invariants(model.anim) == [], stem
        assert model.anim.frame_count >= 2, stem  # the game has no 1-frame animation


@needs_data
def test_every_frame_is_keyed_like_the_bile_demons_files(export):
    for stem in CLIPS:
        anim = export["models"][stem].anim
        assert len(anim.geom_xyz) - 1 == anim.index_count * (
            anim.frame_count + (anim.frame_count - 1) // 128), stem


@needs_data
def test_poses_follow_the_bile_demons_pose_files(export, bile):
    for stem, slot in (("PoseFrame", "poseFrame"), ("DiePose", "diePose"),
                       ("DraggedPose", "draggedPose")):
        theirs, ours = _bile_file(bile, slot), export["models"][stem]
        assert ours.kind == theirs.kind == "mesh", stem
        assert len(ours.mesh.controls) == len(theirs.mesh.controls)
        assert {len(s.triangles) for s in ours.mesh.sprites} == {16}
        geom = np.array(ours.mesh.geometry)
        lo, hi = geom.min(0), geom.max(0)
        np.testing.assert_allclose(ours.mesh.pos, (lo + hi) / 2, atol=1e-6)
        assert ours.mesh.scale == pytest.approx(
            np.linalg.norm(geom - np.array(ours.mesh.pos), axis=1).max(), rel=1e-6)


# ---------------------------------------------------------------- geometry

def _close(got, want, bound):
    return max(float(np.abs(g - w).max()) for g, w in zip(got, want)) <= bound


@needs_data
def test_decoded_frames_match_the_transformed_source_frames(export):
    plan = export["plan"]
    for stem in CLIPS:
        anim = export["models"][stem].anim
        want = plan.clips[stem].frames
        assert anim.frame_count == len(want), stem
        bound = anim.scale / 511 * 1.01
        for f in range(anim.frame_count):
            assert _close(evaluate_anim_frame(anim, f), want[f], bound), (stem, f)
    for stem in POSE_STEMS:
        model = export["models"][stem]
        assert _close(mesh_points(model), plan.poses[stem], 1e-5), stem


@needs_data
def test_plan_frames_are_the_source_frames_scaled_mirrored_and_placed(export):
    plan = export["plan"]
    data, _ = plan.source
    for stem in CLIPS:
        clip, src = plan.clips[stem], data[stem]
        if src["fps"] != GAME_FPS:
            continue  # resampled: checked by test_clips_are_resampled_to_the_game_rate
        for f, frame in enumerate(src["frames"]):
            want = plan.to_sprites(plan.to_kmf(frame))
            for got, w in zip(clip.frames[f], want):
                np.testing.assert_allclose(got, w, atol=1e-9)
    stand, die, picked = data["stand"]["frames"], data["die"]["frames"], data["being_picked_up"]
    for stem, frame in (("PoseFrame", stand[0]), ("DiePose", die[-1]),
                        ("DraggedPose", picked["frames"][0])):
        for got, w in zip(plan.poses[stem], plan.to_sprites(plan.to_kmf(frame))):
            np.testing.assert_allclose(got, w, atol=1e-9)


@needs_data
def test_mirror_is_the_flip_of_the_retargeting(export):
    plan = export["plan"]
    p = np.array([[1.0, 2.0, 3.0]])
    k = (plan.to_kmf([p])[0] - plan.offset) / plan.scale
    np.testing.assert_allclose(k, [[-1.0, 3.0, -2.0]])  # x mirrored; Y-up (X, Y, Z) -> (X, Z, -Y)


@needs_data
def test_clips_are_resampled_to_the_game_rate(export):
    plan = export["plan"]
    data, _ = plan.source
    for stem in CLIPS:
        src, clip = data[stem], plan.clips[stem]
        n = len(src["frames"])
        closed = src["wrap"] and np.allclose(np.concatenate(src["frames"][0]),
                                             np.concatenate(src["frames"][-1]))
        seconds = (n if src["wrap"] and not closed else n - 1) / src["fps"]
        played = len(clip.frames) - 1
        assert abs(played / GAME_FPS - seconds) <= 0.5 / GAME_FPS, stem
        assert clip.wrap == src["wrap"], stem
        assert plan.info[stem] == {"frames": len(clip.frames), "fps": GAME_FPS,
                                   "wrap": src["wrap"]}


@needs_data
def test_wrap_clips_end_on_a_copy_of_their_first_frame(export):
    for stem in CLIPS:
        anim = export["models"][stem].anim
        if not anim.wrap:
            continue
        first, last = evaluate_anim_frame(anim, 0), evaluate_anim_frame(anim, anim.frame_count - 1)
        for a, b in zip(first, last):
            np.testing.assert_array_equal(a, b)


@needs_data
def test_rest_height_is_the_bile_demons_times_the_ratio(export, bile):
    idle = _bile_file(bile, "idle1")
    pts = np.concatenate(evaluate_anim_frame(idle.anim, 0))
    target = (pts[:, 2].max() - pts[:, 2].min()) * RATIO
    assert export["plan"].target_height == pytest.approx(target)
    for stem in ("stand", "PoseFrame"):
        ours = np.concatenate(_points(export["models"][stem]))
        assert ours[:, 2].max() - ours[:, 2].min() == pytest.approx(target, rel=0.01), stem


@needs_data
def test_placed_like_the_bile_demon_feet_on_the_ground_and_centred(export, bile):
    model = export["models"]["PoseFrame"]
    pts = _points(model)
    allp = np.concatenate(pts)
    assert abs(allp[:, 2].max()) < 1e-6  # KMF z points down: the feet are the largest z, at 0
    body = np.concatenate([p for p, s in zip(pts, model.mesh.sprites)
                           if model.materials[s.material].textures == [assets.BODY_TEXTURE]])
    np.testing.assert_allclose(body[:, :2].mean(0), 0, atol=1e-6)
    theirs = np.concatenate(_points(_bile_file(bile, "idle1")))
    assert abs(theirs[:, 2].max()) < 0.005 and np.abs(theirs[:, :2].mean(0)).max() < 0.03


@needs_data
def test_faces_the_same_way_as_the_bile_demon(export, bile):
    def head_forward(pts):
        z = pts[:, 2]
        head = pts[z < z.min() + 0.15 * (z.max() - z.min())]  # the top 15 % (z points down)
        return head[:, 1].mean() - pts[:, 1].mean()
    model = export["models"]["PoseFrame"]
    body = np.concatenate([p for p, s in zip(_points(model), model.mesh.sprites)
                           if model.materials[s.material].textures == [assets.BODY_TEXTURE]])
    theirs = np.concatenate(_points(_bile_file(bile, "idle1")))
    assert head_forward(theirs) > 0 and head_forward(body) > 0


@needs_data
def test_winding_and_normals_follow_the_game_files(export, bile):
    def stats(model):
        dots, outward = [], []
        allp = np.concatenate(_points(model))
        for sprite, p in zip(_sprites(model), _points(model)):
            t = np.array(sprite.triangles[0])
            n = np.array([v.normal for v in sprite.vertices])
            np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-5)
            face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
            acc = np.zeros_like(p)
            for k in range(3):
                np.add.at(acc, t[:, k], -face)
            acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
            dots.append((acc * n).sum(1))
            outward.append(((face * (p[t].mean(1) - allp.mean(0))).sum(1) > 0))
        return np.median(np.concatenate(dots)), np.concatenate(outward).mean()
    their_dot, their_out = stats(_bile_file(bile, "poseFrame"))
    assert their_dot > 0.5 and their_out < 0.5  # the game: -(b-a)x(c-a) is the outward normal
    for stem in ("PoseFrame", "stand", "walk"):
        dot, out = stats(export["models"][stem])
        assert dot > 0.5 and out < 0.5, stem


@needs_data
def test_sprites_fit_the_byte_triangle_indices_and_carry_16_lods(export):
    for stem, model in export["models"].items():
        for sprite in _sprites(model):
            assert len(sprite.vertices) <= 255, stem
            assert len(sprite.triangles) == 16
            assert all(lod == sprite.triangles[0] for lod in sprite.triangles), stem
            assert max(max(t) for t in sprite.triangles[0]) < len(sprite.vertices)
            assert sprite.mm_factor > 0


@needs_data
def test_every_file_has_the_same_sprites_and_controls(export):
    models = export["models"]
    ref = models["stand"]
    for stem, model in models.items():
        assert [(s.material, s.triangles[0], [v.uv for v in s.vertices]) for s in _sprites(model)
                ] == [(s.material, s.triangles[0], [v.uv for v in s.vertices])
                      for s in ref.anim.sprites], stem


# ---------------------------------------------------------------- control points

def _controls(model):
    return np.frombuffer((model.anim or model.mesh).controls, "<u4").reshape(-1, 2)


@needs_data
def test_control_points_keep_the_bile_demons_ids_and_stay_in_range(export, bile):
    their_anim = _controls(_bile_file(bile, "idle1"))
    their_mesh = _controls(_bile_file(bile, "poseFrame"))
    for stem, model in export["models"].items():
        ours = _controls(model)
        if model.kind == "anim":
            assert ours[:, 1].tolist() == their_anim[:, 1].tolist(), stem
            for ref, _ in ours:
                sprite, vertex = ref & 0xFFFF, ref >> 16
                assert sprite < len(model.anim.sprites), stem
                assert vertex < len(model.anim.sprites[sprite].vertices), stem
        else:
            assert ours[:, 1].tolist() == their_mesh[:, 1].tolist(), stem
            assert (ours[:, 0] < len(model.mesh.geometry)).all(), stem


@needs_data
def test_control_points_name_the_same_spot_in_every_file(export):
    rest = export["models"]["stand"]
    for stem, model in export["models"].items():
        for (ref, _), (want, _) in zip(_controls(model), _controls(rest)):
            if model.kind == "anim":
                assert ref == want, stem
            else:
                pts = _points(model)
                geom = np.array(model.mesh.geometry)
                sprite, vertex = want & 0xFFFF, want >> 16
                np.testing.assert_allclose(geom[ref], pts[sprite][vertex], atol=1e-6)


@needs_data
def test_control_points_land_on_the_body(export):
    model = export["models"]["stand"]
    for ref, _ in _controls(model):
        sprite = model.anim.sprites[ref & 0xFFFF]
        assert model.materials[sprite.material].textures == [assets.BODY_TEXTURE]


# ---------------------------------------------------------------- materials and textures

@needs_data
def test_materials_name_the_texture_files_and_copy_a_game_material(export, bile):
    donor = _bile_file(bile, "idle1").materials[0]
    stems = {p.stem for p in export["result"]["textures"].values()}
    for stem, model in export["models"].items():
        used = {model.materials[s.material].textures[0] for s in _sprites(model)}
        assert used == stems, stem
        for m in model.materials:
            assert len(m.textures) == 1
            assert m.name == m.name.split(m.textures[0])[0] + m.textures[0]  # <part><texture>
            assert (m.flags, m.brightness, m.shininess, m.environment_map) == (
                donor.flags, donor.brightness, donor.shininess, donor.environment_map)


@needs_data
def test_textures_are_power_of_two_rgb_within_the_games_largest_size(export):
    textures = export["result"]["textures"]
    assert sorted(p.name for p in textures.values()) == ["Molynx_Body.png", "Molynx_Mace.png"]
    for path in textures.values():
        image = Image.open(path)
        w, h = image.size
        assert image.mode == "RGB"
        assert w == h and w & (w - 1) == 0 and w <= assets.MAX_TEXTURE <= 256


# ---------------------------------------------------------------- portraits

@needs_data
@needs_portrait
def test_portraits_have_the_games_sizes_and_frame(export):
    portraits = export["result"]["portraits"]
    assert sorted(p.name for p in portraits.values()) == ["molynx-bg.png", "molynx.png"]
    sizes = {p.name: Image.open(p).size for p in portraits.values()}
    assert sizes == {"molynx-bg.png": (64, 128), "molynx.png": (64, 64)}
    for p in portraits.values():
        assert Image.open(p).mode == "RGB"
    preview = export["result"]["preview"]
    assert preview.exists() and Image.open(preview).size[0] > 4 * 64


# ---------------------------------------------------------------- guards

@needs_data
def test_refuses_an_existing_non_empty_folder_and_deletes_nothing(export, tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("mine")
    with pytest.raises(FileExistsError):
        assets.export_molynx_assets(tmp_path, plan=export["plan"], portrait_src=None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_stem_pattern_is_the_slot_name_pattern():
    for stem in STEMS:
        assert re.fullmatch(r"[A-Za-z_]+", stem)
