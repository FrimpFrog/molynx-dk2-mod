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
from scripts.dk2paths import (BEHEMOTH_DIR, CHARACTER_PICKLE, EDITOR_DIR, GAME_DIR, MAPS_DIR,
                              MESHES_DIR, QUUZ_MAPS_DIR, have)
from scripts.molynx_record import (CLIPS, MAX_CLOSE_GAP, ONE_WAY_GAPS, POSE_STEMS, PREFIX,
                                   SOURCE_WRAP, WrapException, build_molynx, clip_of, file_plan,
                                   slot_assignment, stems_used)
import scripts.molynx_assets as assets
from molynx_files import ANIM_INFO, FILE_WRAPS

needs_data = pytest.mark.skipif(
    not (have(MESHES_DIR) and have(EDITOR_DIR) and have(GAME_DIR / "EngineTextures.dir")
         and have(BEHEMOTH_DIR / "behemoth.xml")),
    reason="DK2 meshes, editor data, texture cache or the MegaGlest Behemoth not present")
needs_portrait = pytest.mark.skipif(not (have(assets.DESIGN_PORTRAIT) and have(assets.DESIGN_ICON)),
                                    reason="the redesign's portrait pictures are not present")

VARIANTS = ["walk_once"]  # what the real Bile Demon's slots need (see below)
STEMS = list(CLIPS) + VARIANTS + list(POSE_STEMS)
ANIMS = list(CLIPS) + VARIANTS
# The loop flag each used file must have: its slots' Bile Demon files (walk / walkback loop,
# entrance does not -> walk_once; stunnedPose and elec take static poses, which have no flag).
REQUIRED = {
    "walk": True, "walk_once": False, "getting_hit": False,
    "stand": False, "fidgeting": False, "eating": False, "happy": False, "attack": False,
    "angry": False, "celebrating": False, "die": False, "being_slapped": False,
    "knockback": False, "waking_up": False, "sleeping": False, "being_picked_up": True,
    "manufacturing": True, "praying": True,
}
RATIO = 1.38  # the redesign: as tall as the Horned Reaper (was 1.15 x the Bile Demon)
GAME_FPS = 30


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    if not (have(MESHES_DIR) and have(EDITOR_DIR) and have(GAME_DIR / "EngineTextures.dir")
            and have(BEHEMOTH_DIR / "behemoth.xml")):
        pytest.skip("game data not present")
    out = tmp_path_factory.mktemp("molynx") / "assets"
    portrait = assets.DESIGN_PORTRAIT if have(assets.DESIGN_PORTRAIT) else None
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

def test_the_default_height_is_the_redesigns():
    assert assets.DESIGN_HEIGHT_RATIO == RATIO


@needs_data
@pytest.mark.skipif(not have(CHARACTER_PICKLE), reason="reference character pickle not present")
def test_rebuilt_character_without_the_design_equals_the_reference_build():
    data, textures = assets.build_source(design=False)
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
    file_wraps = assets.bile_file_wraps([bile])
    molynx = build_molynx(bile, 2, 47, None, export["result"]["info"], file_wraps=file_wraps)
    written = {molynx.resource_name(slot) for slot in ANIM_SLOTS if slot != "unk_1"} - {""}
    assert written == {PREFIX + s for s in stems_used(bile, file_plan(bile, file_wraps))}
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
    for stem in ANIMS:
        model = export["models"][stem]
        assert model.kind == "anim", stem
        assert model.anim.name == f"{PREFIX}{stem}"
        assert anim_invariants(model.anim) == [], stem
        assert model.anim.frame_count >= 2, stem  # the game has no 1-frame animation


# The largest GEOM (keyframe positions) of every animation file, reduced (reduce=True). The game's
# largest animation has 34,816 (Mis-Idle1); 65,535 is the largest a u16 index could reach.
# The redesign (2026-11-06): files that would pass the game's largest are reduced with a wider
# tolerance (molynx_assets.KEY_TOLERANCES): celebrating 46,079 -> 34,419 (2.5 steps), staggering
# 40,448 -> 33,988 (1.5), training 35,676 -> 31,920 (1.5), fighting_another_creature 43,518 ->
# 34,053 (2); victory_celebration (shipped by no slot) stays at one step.
GEOM = {
    "stand": 1834, "walk": 8068, "walk_once": 8068, "attack": 11420, "die": 5553,
    "fidgeting": 18932, "running": 14925, "eating": 19637, "praying": 16565, "happy": 21905,
    "angry": 9870, "celebrating": 34419, "victory_celebration": 73948, "getting_hit": 6220,
    "being_slapped": 10073, "staggering": 33988, "waking_up": 15987, "knockback": 25813,
    "sleeping": 21411, "training": 31920, "fighting_another_creature": 34053, "patrolling": 16911,
    "guarding": 1693, "manufacturing": 13467, "interacting_with_objects": 5971,
    "being_picked_up": 6492,
}
GAME_MAX_GEOM = 34_816
# The reduction tolerance (quantisation steps) of the files reduced wider than one step.
TOLERANCE = {"celebrating": 2.5, "staggering": 1.5, "training": 1.5, "fighting_another_creature": 2.0}


@needs_data
def test_keyframes_are_reduced_like_661_game_files(export):
    for stem in ANIMS:
        anim = export["models"][stem].anim
        full = anim.index_count * (anim.frame_count + (anim.frame_count - 1) // 128)
        keys = len(anim.geom_xyz) - 1
        assert anim.index_count <= keys < full, stem  # at least one key per row, fewer than all
        assert keys == GEOM[stem], stem


@needs_data
def test_shipped_animations_stay_within_the_index_range(export, bile):
    """Every animation the record names (and staggering, shipped for a type NONE slot of some
    maps' Bile Demons) stays within the game's own largest animation (34,816 keyframes); only the
    unshipped victory_celebration (73,948) is above it, and above 65,535."""
    shipped = {s for s in stems_used(bile, assets.molynx_files(bile)) if s not in POSE_STEMS}
    shipped.add("staggering")  # Siege's Bile Demon names it in dejectedPose (type NONE)
    for stem in shipped:
        assert GEOM[stem] <= 65_535, stem
    assert {s for s in shipped if GEOM[s] > GAME_MAX_GEOM} == set()
    assert {s for s in GEOM if GEOM[s] > GAME_MAX_GEOM} == {"victory_celebration"}
    assert assets.GAME_MAX_KEYS == GAME_MAX_GEOM
    assert {s for s in GEOM if GEOM[s] > 65_535} == {"victory_celebration"}
    assert "victory_celebration" not in shipped and "fighting_another_creature" not in shipped


@needs_data
def test_poses_follow_the_bile_demons_pose_files(export, bile):
    for stem, slot in (("PoseFrame", "poseFrame"), ("DiePose", "diePose"),
                       ("DraggedPose", "draggedPose"), ("StunnedPose", "stunnedPose"),
                       ("ElecPose", "elec")):
        theirs, ours = _bile_file(bile, slot), export["models"][stem]
        assert ours.kind == theirs.kind == "mesh", stem
        if stem == "ElecPose":  # Bil-elec carries no CTRL entries; ours keep the same 20 as every file
            assert theirs.mesh.controls == b"" and len(ours.mesh.controls) == 20 * 8
        else:
            assert len(ours.mesh.controls) == len(theirs.mesh.controls), stem
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
    for stem in ANIMS:
        anim = export["models"][stem].anim
        want = assets.file_clip(plan, stem, anim.wrap).frames
        assert anim.frame_count == len(want), stem
        bound = anim.scale / 511 * TOLERANCE.get(stem, 1.0) * 1.01
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
            if src["wrap"] and f == len(src["frames"]) - 1 and assets.loop_closes(src["frames"]):
                frame = src["frames"][0]  # a (nearly) closed loop's last frame becomes frame 0
            want = plan.to_sprites(plan.to_kmf(frame))
            for got, w in zip(clip.frames[f], want):
                np.testing.assert_allclose(got, w, atol=1e-9)
    stand, die, picked = data["stand"]["frames"], data["die"]["frames"], data["being_picked_up"]
    for stem, frame in (("PoseFrame", stand[0]), ("DiePose", die[-1]),
                        ("DraggedPose", picked["frames"][0]),
                        ("StunnedPose", data["staggering"]["frames"][48]),
                        ("ElecPose", data["getting_hit"]["frames"][5])):
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
        closed = src["wrap"] and assets.loop_closes(src["frames"])
        seconds = (n if src["wrap"] and not closed else n - 1) / src["fps"]
        played = len(clip.frames) - 1
        assert abs(played / GAME_FPS - seconds) <= 0.5 / GAME_FPS, stem
        assert clip.wrap == src["wrap"], stem
        assert plan.info[stem] == {"frames": len(clip.frames), "fps": GAME_FPS,
                                   "wrap": src["wrap"]}


@needs_data
def test_wrap_clips_end_on_a_copy_of_their_first_frame(export):
    for stem in ANIMS:
        anim = export["models"][stem].anim
        if not anim.wrap:
            continue
        first, last = evaluate_anim_frame(anim, 0), evaluate_anim_frame(anim, anim.frame_count - 1)
        for a, b in zip(first, last):
            np.testing.assert_array_equal(a, b)


def _median_gap(src_frames):
    track = np.stack([np.concatenate(f) for f in src_frames])
    steps = np.linalg.norm(np.diff(track, axis=0), axis=2).max(axis=1)
    return np.linalg.norm(track[-1] - track[0], axis=1).max() / np.median(steps)


@needs_data
def test_loops_have_no_near_duplicate_frame_at_the_loop_point(export):
    """A loop that nearly closes has its last frame replaced by frame 0 (frame count kept); any
    other loop gets frame 0 appended. Either way the last step is a real step. Loops whose source
    already ends exactly on frame 0 keep the source's own last step (an ease into the loop). The
    bound is 0.2 of the median step: interacting_with_objects, whose source eases its last steps
    (0.15 of a step), ends on a 0.23 step after the replacement (appending would leave 0.21)."""
    data, _ = export["plan"].source
    for stem in ANIMS:
        src, anim = data[clip_of(stem)[0]], export["models"][stem].anim
        if not anim.wrap:
            continue
        gap = _median_gap(src["frames"])
        n = len(src["frames"])
        if src["fps"] == GAME_FPS:
            assert anim.frame_count == (n if gap < 0.3 else n + 1), stem
        if gap < 1e-9:
            continue
        track = np.stack([np.concatenate(evaluate_anim_frame(anim, f))
                          for f in range(anim.frame_count)])
        steps = np.linalg.norm(np.diff(track, axis=0), axis=2).max(axis=1)
        assert steps[-1] >= 0.2 * np.median(steps), stem


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
    def stats(model, texture=None):
        dots, outward = [], []
        chosen = [(s, p) for s, p in zip(_sprites(model), _points(model))
                  if texture is None or model.materials[s.material].textures == [texture]]
        allp = np.concatenate([p for _, p in chosen])
        for sprite, p in chosen:
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
        for texture in (assets.BODY_TEXTURE, assets.MACE_TEXTURE):
            dot, out = stats(export["models"][stem], texture)
            assert dot > 0.5 and out < 0.5, (stem, texture)


@needs_data
def test_sprites_fit_the_byte_triangle_indices_and_carry_16_lods(export):
    for stem, model in export["models"].items():
        for sprite in _sprites(model):
            assert len(sprite.vertices) <= 255, stem
            assert len(sprite.triangles[0]) <= 255, stem  # the game's largest sprite has 255
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
def test_control_points_land_on_the_body_or_head_never_the_mace(export):
    model = export["models"]["stand"]
    where = {}
    for ref, ident in _controls(model):
        sprite = model.anim.sprites[ref & 0xFFFF]
        (texture,) = model.materials[sprite.material].textures
        assert texture != assets.MACE_TEXTURE
        where[ident] = texture
    assert where[1000] == where[5000] == assets.HEAD_TEXTURE  # the top of his head is the Golem's


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
    sizes = {p.name: Image.open(p).size for p in textures.values()}
    assert sizes == {"Molynx_Body.png": (128, 128), "Molynx_Head.png": (128, 128),
                     "Molynx_Mace.png": (128, 128), "Molynx_Neck.png": (128, 32)}
    for path in textures.values():
        image = Image.open(path)
        w, h = image.size
        assert image.mode == "RGB"
        for n in (w, h):
            assert n & (n - 1) == 0 and n <= assets.MAX_TEXTURE <= 256


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


@needs_data
def test_refuses_a_plan_made_for_another_height_ratio(export, tmp_path):
    with pytest.raises(ValueError):
        assets.export_molynx_assets(tmp_path / "out", 1.3, plan=export["plan"], portrait_src=None)
    assert not (tmp_path / "out").exists()


@needs_data
def test_a_missing_portrait_source_fails_before_anything_is_written(export, tmp_path):
    with pytest.raises(FileNotFoundError):
        assets.export_molynx_assets(tmp_path / "out", RATIO, plan=export["plan"],
                                    portrait_src=tmp_path / "missing.png")
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(not have(GAME_DIR / "EngineTextures.dir"), reason="texture cache not present")
def test_texture_limit_is_the_games_largest_texture():
    cache = assets.texture_cache()
    largest = max(max(cache.size_of(n)) for n in cache.names if n.endswith("MM0"))
    assert assets.MAX_TEXTURE == largest == 128


# ---------------------------------------------------------------- independent resampling check

RESAMPLED = ["stand", "walk", "attack", "die", "fidgeting", "victory_celebration", "sleeping",
             "training", "fighting_another_creature"]
# Files whose loop flag is not their source clip's: opened loops keep the loop's frames (timed as a
# loop, ending on the copy of frame 0); closed ones are resampled as a loop.
FLIPPED = ["stand", "fidgeting", "eating", "happy", "attack", "angry", "celebrating",
           "walk_once"]


def _axes(p):
    """Character Y-up, right-handed -> KMF axes, written out on its own: mirror x (DK2 is
    left-handed), then (X, Y, Z) -> (X, Z, -Y) (KMF y forward, z down)."""
    p = np.asarray(p, float)
    return np.stack([-p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _lerp_frame(frames, t, nxt):
    i = int(np.floor(t + 1e-9))
    u = t - i
    if u < 1e-9:
        return [np.asarray(p, float) for p in frames[i]]
    j = nxt(i)
    return [np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u
            for a, b in zip(frames[i], frames[j])]


@needs_data
def test_resampled_clips_decode_to_a_plain_interpolation_of_the_source(export, bile):
    """Independent of plan_molynx / _resample: the source clip (the redesigned character, built
    again here), interpolated linearly at 30 fps and moved to KMF space here, against the DECODED
    files."""
    data, _ = assets.build_source()
    layout = [(s.part, s.vertices) for s in export["plan"].sprites]  # which source vertex is which

    def sprites(parts):
        return [np.asarray(parts[part])[idx] for part, idx in layout]

    # scale: the Bile Demon's idle1 height x RATIO over the height of stand frame 0
    idle = np.concatenate(evaluate_anim_frame(_bile_file(bile, "idle1").anim, 0))
    rest = np.concatenate(sprites([_axes(p) for p in data["stand"]["frames"][0]]))
    scale = (idle[:, 2].max() - idle[:, 2].min()) * RATIO / (rest[:, 2].max() - rest[:, 2].min())
    # offset: whatever constant moves the scaled source rest pose onto the written PoseFrame
    pose = export["models"]["PoseFrame"]
    moved = np.concatenate(mesh_points(pose)) - rest * scale
    offset = moved.mean(0)
    assert np.abs(moved - offset).max() < 1e-6  # one translation for every vertex

    def kmf(frame):
        return [_axes(p) * scale + offset for p in sprites(frame)]

    for stem in RESAMPLED:
        assert data[stem]["fps"] != GAME_FPS, stem
    for stem in FLIPPED:
        assert export["models"][stem].anim.wrap != data[clip_of(stem)[0]]["wrap"], stem
    for stem in dict.fromkeys(RESAMPLED + FLIPPED):
        src = data[clip_of(stem)[0]]
        frames, fps = src["frames"], src["fps"]
        anim = export["models"][stem].anim
        wrap = src["wrap"] or anim.wrap  # timed as a loop
        count = anim.frame_count
        n = len(frames)
        if wrap:
            closes = _median_gap(frames) < 0.3
            cycle = n - 1 if closes else n
            assert count - 1 == round(cycle / fps * GAME_FPS), stem
            times = [k * cycle / (count - 1) for k in range(count - 1)] + [0.0]
            nxt = lambda i, c=cycle: (i + 1) % c
        else:
            assert count - 1 == round((n - 1) / fps * GAME_FPS), stem
            times = [k * (n - 1) / (count - 1) for k in range(count)]
            nxt = lambda i, last=n - 1: min(i + 1, last)
        bound = anim.scale / 511 * TOLERANCE.get(stem, 1.0) * 1.01
        check = sorted({0, count // 2, count - 1, *range(0, count, 5)})
        for f in check:
            want = kmf(_lerp_frame(frames, times[f], nxt))
            got = evaluate_anim_frame(anim, f)
            err = max(float(np.abs(g - w).max()) for g, w in zip(got, want))
            assert err <= bound, (stem, f, err, bound)
        first = max(float(np.abs(g - w).max())
                    for g, w in zip(evaluate_anim_frame(anim, 0), kmf(frames[0])))
        assert first <= bound, stem
        if not anim.wrap:  # an opened loop still ends on its copy of frame 0 (times[-1] = 0)
            assert times[-1] == (0.0 if src["wrap"] else n - 1), stem
        if not wrap:
            last = max(float(np.abs(g - w).max())
                       for g, w in zip(evaluate_anim_frame(anim, count - 1), kmf(frames[-1])))
            assert last <= bound, stem


# ---------------------------------------------------------------- loop flags per slot

@needs_data
def test_bile_file_wraps_are_the_bile_demons_files(bile):
    got = assets.bile_file_wraps([bile])
    for slot in slot_assignment(bile):
        name = bile.resource_name(slot)
        model = parse_kmf(assets.game_mesh_path(name).read_bytes())
        assert got[name.lower()] == (model.anim.wrap if model.kind == "anim" else None), slot
    # pinned: the 8a report's examples
    assert got["bile_demon-walk"] is True and got["bil-walkback"] is True
    assert got["biledemon-idle1"] is False and got["bil-entrance"] is False
    assert got["biledemon-bed(sleeping)"] is True and got["bil-elec"] is None


@needs_data
def test_required_wraps_for_the_real_bile_demon(bile):
    assert assets.required_wraps(bile) == REQUIRED
    assert set(REQUIRED) | set(POSE_STEMS) == stems_used(bile, file_plan(bile, assets.bile_file_wraps([bile])))


@needs_data
def test_source_wrap_is_the_sources_flag(export):
    data, _ = export["plan"].source
    assert {clip: data[clip]["wrap"] for clip in CLIPS} == SOURCE_WRAP


@needs_data
def test_default_export_writes_the_required_loop_flags(export):
    """The fixture's export uses the default wraps (required_wraps of the Bile Demon)."""
    models, info = export["models"], export["result"]["info"]
    for stem, wrap in REQUIRED.items():
        assert models[stem].anim.wrap is wrap, stem
    for clip in set(CLIPS) - set(REQUIRED):  # unused clips keep their own flag
        assert models[clip].anim.wrap is SOURCE_WRAP[clip], clip


@needs_data
def test_info_reports_the_final_files(export):
    models, info = export["models"], export["result"]["info"]
    assert sorted(info) == sorted(ANIMS)
    assert info == ANIM_INFO  # the record tests' pinned info is the real export's
    for stem in ANIMS:
        anim = models[stem].anim
        assert info[stem] == {"frames": anim.frame_count, "fps": GAME_FPS, "wrap": anim.wrap}, stem


@needs_data
def test_opened_loops_keep_the_loops_frames(export):
    plan, models = export["plan"], export["models"]
    for stem in ("stand", "fidgeting", "eating", "happy", "attack", "angry", "celebrating",
                 "walk_once"):
        clip = clip_of(stem)[0]
        loop = plan.clips[clip]
        assert loop.wrap and not models[stem].anim.wrap, stem
        assert models[stem].anim.frame_count == len(loop.frames), stem
        assert export["result"]["info"][stem]["frames"] == plan.info[clip]["frames"], stem


@needs_data
def test_closed_loops_end_on_frame_0_with_one_more_frame(export):
    """Closing a clip that does not close in the source (getting_hit, not shipped closed any more)
    appends the copy of frame 0."""
    plan = export["plan"]
    hit = export["models"]["getting_hit"].anim
    loop = assets.clip_kmf(plan, "getting_hit_loop", wrap=True).anim
    assert (loop.frame_count, hit.frame_count) == (18, 17) and loop.wrap
    for f in range(hit.frame_count):
        for a, b in zip(evaluate_anim_frame(hit, f), evaluate_anim_frame(loop, f)):
            assert np.abs(a - b).max() <= max(hit.scale, loop.scale) / 511 * 2.02, f


@needs_data
@pytest.mark.parametrize("wraps", [{"walk_twice": True}, {"walk_once": True}, {"walk_loop": False},
                                   {"PoseFrame": True}, {"walk": 1}, {"sleeping": True},
                                   {"waking_up_loop": True}])
def test_bad_wraps_are_refused_before_anything_is_written(export, tmp_path, wraps):
    with pytest.raises(ValueError):
        assets.export_molynx_assets(tmp_path / "out", RATIO, plan=export["plan"], portrait_src=None,
                                    wraps=wraps)
    assert not (tmp_path / "out").exists()


@needs_data
def test_empty_wraps_keep_every_clips_own_flag(export, tmp_path):
    result = assets.export_molynx_assets(tmp_path / "out", RATIO, plan=export["plan"],
                                         portrait_src=None, wraps={})
    assert sorted(result["info"]) == sorted(CLIPS)
    assert result["info"] == export["plan"].info


@needs_data
@pytest.mark.skipif(not have(MAPS_DIR), reason="DK2 editor maps not present")
def test_pinned_file_wraps_are_the_files_of_every_shipped_bile_demon(bile):
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.dk2kwd_creatures import read_creatures
    biles = [bile]
    for folder in (MAPS_DIR, QUUZ_MAPS_DIR):
        for path in sorted(folder.glob("*Globals.kld")) if have(folder) else []:
            block = next(b for b in read_blocks(path.read_bytes()) if b.signature == 0xAA)
            biles += [c for c in read_creatures(block) if c.get("creatureId") == 3]
    assert len(biles) > 240
    assert assets.bile_file_wraps(biles) == FILE_WRAPS


# ---------------------------------------------------------------- one-way clips are never looped

@needs_data
def test_one_way_gaps_are_measured_on_the_source(export):
    """Non-looping source clips whose last frame is more than 10 median steps from frame 0."""
    data, _ = export["plan"].source
    gaps = {clip: _median_gap(data[clip]["frames"]) for clip in CLIPS if not data[clip]["wrap"]}
    far = {clip: gap for clip, gap in gaps.items() if gap > 10}
    assert set(far) == set(ONE_WAY_GAPS) == {"sleeping", "waking_up"}
    for clip, gap in far.items():
        assert gap == pytest.approx(ONE_WAY_GAPS[clip], abs=0.01), clip
    assert gaps["getting_hit"] == pytest.approx(5.80, abs=0.01)  # could be closed (below 10)


@needs_data
def test_every_looping_file_closes_a_small_gap(export):
    """No file loops over a jump of more than MAX_CLOSE_GAP median steps of its source clip."""
    data, _ = export["plan"].source
    assert MAX_CLOSE_GAP == 10
    for stem in ANIMS:
        if export["models"][stem].anim.wrap:
            assert _median_gap(data[clip_of(stem)[0]]["frames"]) <= 10, stem


@needs_data
def test_sleeping_stays_one_way_and_is_reported(export, bile):
    model = export["models"]["sleeping"].anim
    assert model.wrap is False
    assert export["result"]["info"]["sleeping"] == {"frames": 136, "fps": 30, "wrap": False}
    (exception,) = export["result"]["wrap_exceptions"]
    assert exception == WrapException("sleeping", ("animSleepResource",), True,
                                      ONE_WAY_GAPS["sleeping"])
    assert assets.required_wraps(bile)["sleeping"] is False  # reported, not raised
    assert assets.molynx_files(bile).wrap_exceptions == (exception,)


# ---------------------------------------------------------------- static poses for mesh slots

def _mean_distance(frame, first):
    return float(np.mean([np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1).mean()
                          for a, b in zip(frame, first)]))


@needs_data
def test_stunned_and_elec_poses_are_the_middle_of_staggering_and_the_peak_of_getting_hit(export):
    data, _ = export["plan"].source
    assert assets.POSE_SOURCES["StunnedPose"] == ("staggering", len(data["staggering"]["frames"]) // 2) \
        == ("staggering", 48)
    hit = data["getting_hit"]["frames"]
    peak = max(range(len(hit)), key=lambda f: _mean_distance(hit[f], hit[0]))
    assert assets.POSE_SOURCES["ElecPose"] == ("getting_hit", peak) == ("getting_hit", 5)


@needs_data
def test_every_pose_but_the_rest_pose_differs_from_the_stand_frame(export):
    rest = mesh_points(export["models"]["PoseFrame"])
    for stem in ("StunnedPose", "ElecPose", "DiePose", "DraggedPose"):
        points = mesh_points(export["models"][stem])
        moved = max(float(np.abs(a - b).max()) for a, b in zip(points, rest))
        assert moved > 0.02, stem  # 0.62 is his height
