"""The approved redesign of Molynx (scripts/molynx_design.py), as built by molynx_assets.build_source.

Checks that each design change does what the user approved, measured on the built character: the
Troll clips, the free-hand claw riding on its hand, the mace arm / club / mace held in the locked
clips, the Golem head moving rigidly, his face taken off, and the pinned design textures. The build
is made once per module; tests that need the game data skip without it.
"""
import hashlib

import numpy as np
import pytest
from PIL import Image

import scripts.molynx_assets as assets
import scripts.molynx_design as design
from scripts.dk2paths import BEHEMOTH_DIR, GAME_DIR, MESHES_DIR, have
from scripts.molynx_record import CLIPS

needs_data = pytest.mark.skipif(
    not (have(MESHES_DIR) and have(GAME_DIR / "EngineTextures.dir")
         and have(BEHEMOTH_DIR / "behemoth.xml") and have(BEHEMOTH_DIR.parent / "megaglest_golem")),
    reason="DK2 meshes, texture cache or the MegaGlest Behemoth / Golem not present")

# The approved pictures (sha256), so any change to them is deliberate.
DESIGN_FILES = {
    "body_A2.png": ("9e3ae2eb48becabe622e4cbafefab1cec3f3516bc20302dcc4c193a094a5ff8d", (512, 512)),
    "head_A2.png": ("0a29e11eecf9f09f127742a5825952db41eeaab0c56337b033c8f76b12af1169", (512, 512)),
    "neck_A2.png": ("8b3ca5815113fcc549cd276d55e458ee07f4b95ed14c8fc82a65f0eb022dcf62", (685, 103)),
    "portrait_bg_A2.png": ("b5dc827e8d7115ec64b184e3aeb5a21ae78f3bea8d03e5e45110c91cd4149f5d", (1140, 1140)),
    "portrait_icon_A2.png": ("cdcf85498de30512204c7c3f2b444488fc64654bf53f6cbfab43d14325b05b25", (600, 600)),
}
NATIVE = ("stand", "walk", "attack", "die")
TEXTURES = ["custom_character/body", "custom_character/mace", "custom_character/golem_head",
            "custom_character/head_neck"]


def _pairwise(p):
    p = np.asarray(p, float)
    return np.linalg.norm(p[:, None] - p[None], axis=2)


@pytest.fixture(scope="module")
def built():
    if needs_data.args[0]:
        pytest.skip("game data not present")
    new, new_tex = assets.build_source()
    old, old_tex = assets.build_source(design=False)
    return new, new_tex, old, old_tex


def test_design_pictures_are_the_approved_ones():
    for name, (sha, size) in DESIGN_FILES.items():
        path = design.DATA_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, name
        assert Image.open(path).size == size, name
    assert assets.DESIGN_PORTRAIT == design.DATA_DIR / "portrait_bg_A2.png"
    assert assets.DESIGN_ICON == design.DATA_DIR / "portrait_icon_A2.png"


@needs_data
def test_every_clip_has_body_mace_head_and_neck(built):
    new, tex, _, _ = built
    assert sorted(tex) == sorted(TEXTURES)
    for name in CLIPS:
        parts = new[name]["parts"]
        assert [p["texture"] for p in parts] == TEXTURES, name
        for p in parts:
            np.testing.assert_array_equal(p["indices"], new["stand"]["parts"][parts.index(p)]["indices"])
        for frame in new[name]["frames"]:
            assert [len(v) for v in frame] == [len(p["uv"]) for p in parts], name
    head, neck = new["stand"]["parts"][2], new["stand"]["parts"][3]
    assert (len(np.unique(head["indices"])), len(head["indices"]) // 3) == (39, 58)
    assert (len(np.unique(neck["indices"])), len(neck["indices"]) // 3) == (38, 36)


@needs_data
def test_fidgeting_and_celebrating_are_the_troll_clips(built):
    new, _, old, _ = built
    assert (len(new["fidgeting"]["frames"]), new["fidgeting"]["fps"], new["fidgeting"]["wrap"]) == (113, 35.684, True)
    assert (len(new["celebrating"]["frames"]), new["celebrating"]["fps"], new["celebrating"]["wrap"]) == (97, 30.0, True)
    assert design.NEW_CLIP_SOURCES == {"fidgeting": "Troll idle2", "celebrating": "Troll dance"}
    for name in CLIPS:  # every other clip keeps its timing
        if name not in design.NEW_CLIPS:
            assert (len(new[name]["frames"]), new[name]["fps"], new[name]["wrap"]) == (
                len(old[name]["frames"]), old[name]["fps"], old[name]["wrap"]), name


@needs_data
def test_his_own_clips_change_only_by_the_claw_and_the_head(built):
    new, _, old, _ = built
    keep = np.setdiff1d(np.arange(728), design.CLAW)
    for name in NATIVE:
        for fn, fo in zip(new[name]["frames"], old[name]["frames"]):
            np.testing.assert_array_equal(np.asarray(fn[0])[keep], np.asarray(fo[0])[keep])
            np.testing.assert_array_equal(fn[1], fo[1])  # the mace


@needs_data
def test_the_claw_rides_on_its_hand_in_every_frame(built):
    new, _, _, _ = built
    from scripts.behemoth_rig import behemoth_rest
    rest, _, _ = behemoth_rest(BEHEMOTH_DIR)
    gap = np.linalg.norm(rest[design.CLAW].mean(0) - rest[design.HAND].mean(0))
    for name in CLIPS:
        for frame in new[name]["frames"]:
            body = np.asarray(frame[0])
            # the claw keeps its shape and its distance from the hand (it floated up to 2.8 off)
            np.testing.assert_allclose(_pairwise(body[design.CLAW]), _pairwise(rest[design.CLAW]), atol=1e-9)
            d = np.linalg.norm(body[design.CLAW].mean(0) - body[design.HAND].mean(0))
            assert abs(d - gap) < 0.05, (name, d, gap)


@needs_data
def test_locked_clips_keep_the_club_and_mace_as_in_the_idle_pose(built):
    new, _, _, _ = built
    ref = new["stand"]["frames"][0]
    want_club = _pairwise(np.asarray(ref[0])[design.CLUB])
    want_mace = _pairwise(ref[1])
    for name in design.LOCKED:
        for frame in new[name]["frames"]:
            np.testing.assert_allclose(_pairwise(np.asarray(frame[0])[design.CLUB]), want_club, atol=1e-9)
            np.testing.assert_allclose(_pairwise(frame[1]), want_mace, atol=1e-9)


@needs_data
def test_the_head_and_its_band_move_rigidly(built):
    new, _, _, _ = built
    ref = np.concatenate([new["stand"]["frames"][0][2], new["stand"]["frames"][0][3]])
    want = _pairwise(ref)
    for name in CLIPS:
        for frame in new[name]["frames"]:
            got = _pairwise(np.concatenate([frame[2], frame[3]]))
            np.testing.assert_allclose(got, want, atol=1e-9)


@needs_data
def test_his_face_is_taken_off_and_his_horns_stay(built):
    new, _, old, _ = built
    head, horn = design.molynx_masks(BEHEMOTH_DIR)
    ref = np.asarray(old["stand"]["frames"][0][0])
    face = head & (ref[:, 2] >= design.HEAD["face_cut"])
    tri_new = np.asarray(new["stand"]["parts"][0]["indices"], int).reshape(-1, 3)
    tri_old = np.asarray(old["stand"]["parts"][0]["indices"], int).reshape(-1, 3)
    assert not np.any(np.all(face[tri_new] & ~horn[tri_new], axis=1))
    for t in tri_old[np.any(horn[tri_old], axis=1)]:  # every horn triangle is kept
        assert np.any(np.all(tri_new == t, axis=1))
    assert len(tri_new) == 562 and len(tri_old) == 663


@needs_data
def test_the_game_files_use_four_textures_and_the_horned_reaper_size(built, tmp_path):
    new, tex, _, _ = built
    plan = assets.plan_molynx(source=(new, tex))
    assert [m.name for m in plan.materials] == ["BodyMolynx_Body", "MaceMolynx_Mace",
                                                "HeadMolynx_Head", "NeckMolynx_Neck"]
    assert [(s.texture, len(s.vertices), len(s.triangles)) for s in plan.sprites] == [
        ("Molynx_Body", 250, 255), ("Molynx_Body", 247, 255), ("Molynx_Body", 47, 52),
        ("Molynx_Mace", 246, 86), ("Molynx_Head", 39, 58), ("Molynx_Neck", 38, 36)]
    assert plan.scale == pytest.approx(0.17977, abs=1e-5)       # design 0.17999 (1.3817 x)
    assert plan.target_height == pytest.approx(0.7402, abs=1e-4)
    # the Golem face: the used part of its picture, whole pixels, the UVs moved into it
    u0, v0, u1, v1 = plan.crops["Molynx_Head"]
    assert (u0 * 512, v0 * 512, u1 * 512, v1 * 512) == (33, 207, 221, 485)
    head_uv = np.asarray(plan.parts[2]["uv"], float)[np.unique(plan.parts[2]["indices"])]
    assert head_uv.min() >= 0 and head_uv.max() <= 1
