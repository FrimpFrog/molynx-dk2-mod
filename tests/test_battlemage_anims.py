from pathlib import Path

import numpy as np
import pytest

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
UNIT = Path("/tmp/mg/mgd/techs/megapack/factions/magic/units/battlemage")
needs = pytest.mark.skipif(not ((S / "extracted/Meshes").exists() and UNIT.exists()), reason="game/battlemage files absent")

# stand, walk, attack, die and morph are the MegaGlest Battlemage's own; the rest is built here
NAMES = """looking_around fidgeting running turning sleeping waking_up eating training researching
manufacturing praying guarding celebrating angry frightened happy unhappy blocking special_attack casting_spell
getting_hit staggering knockback picking_up dropping being_picked_up being_slapped patrolling
interacting_with_objects fighting_another_creature victory_celebration""".split()


@pytest.fixture(scope="module")
def lib():
    from scripts.battlemage_anims import build_all
    from scripts.dk2texture import TextureCache
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    return build_all(S / "extracted/Meshes", cache, UNIT)


@needs
def test_every_requested_animation_exists_including_casting_and_research(lib):
    assert set(NAMES) <= set(lib), set(NAMES) - set(lib)


@needs
def test_animations_are_well_formed_smooth_and_not_torn(lib):
    from scripts.battlemage_rig import battlemage_rig
    rig = battlemage_rig(UNIT)
    idx = rig.indices
    e = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
    rest_max = np.linalg.norm(rig.rest[e[:, 0]] - rig.rest[e[:, 1]], axis=1).max()
    for name, a in lib.items():
        fr = np.array(a["frames"])
        assert fr.shape[1:] == rig.rest.shape, name
        assert np.isfinite(fr).all(), name
        assert len(fr) >= 20 and 20 <= a["fps"] <= 40, (name, len(fr), a["fps"])
        assert a["kind"] in ("dk2", "hand", "mix")
        assert np.abs(np.diff(fr, axis=0)).max() < 0.7, name                 # no teleporting vertices
        longest = np.linalg.norm(fr[:, e[:, 0]] - fr[:, e[:, 1]], axis=2).max()
        assert longest < rest_max * 1.5, (name, longest)                     # no torn / stretched triangles
        if name != "being_picked_up":
            assert fr[:, :, 1].min() > -0.05, name                           # never through the floor


@needs
def test_looping_animations_close(lib):
    for name, a in lib.items():
        if a["wrap"]:
            fr = np.array(a["frames"])
            assert np.abs(fr[0] - fr[-1]).max() < 0.5, name


@needs
def test_praying_kneels_and_a_fall_lies_down(lib):
    """Shins have their own bones, so kneeling and lying clips fold the legs instead of tilting a straight body."""
    pray = np.array(lib["praying"]["frames"])
    assert pray[:, :, 1].max(1).min() < 2.15                     # at some point he is clearly lower than standing (2.46 with hat)
    fall = np.array(lib["knockback"]["frames"])
    assert fall[:, :, 1].max(1).min() < 1.3                      # lying on the ground


@needs
def test_legs_are_together_when_standing(lib):
    """Regression: a walking-frame rest pose of the donor made every clip scissor the legs apart."""
    fr = np.array(lib["guarding"]["frames"])
    boots = fr[:, :, 1] < 0.12
    spread = np.array([f[b][:, 0].max() - f[b][:, 0].min() for f, b in zip(fr, boots)])
    assert np.median(spread) < 0.75


@needs
def test_no_gambling_animation_is_invented(lib):
    """DK2's creature files hold no casino clip, so none is made up."""
    assert "gambling" not in lib


@needs
def test_belt_and_dagger_stay_with_the_torso_and_no_cloth_wing_stretches_to_a_raised_arm(lib):
    """Regression: belt/dagger vertices were labelled arm, and seam triangles between sleeve and tunic became a wing of cloth."""
    from scripts.battlemage_anims import stretched_triangles
    from scripts.battlemage_rig import battlemage_rig
    rig = battlemage_rig(UNIT)
    b = rig.bones
    fore_arm = [b.index(n) for n in ("armN", "armP", "foreN", "foreP")]
    hip = (np.abs(rig.rest[:, 0]) < 0.2) & (rig.rest[:, 2] > 0.1) & (rig.rest[:, 1] > 1.0) & (rig.rest[:, 1] < 1.45)   # belt, dagger, tunic front
    assert rig.weights[hip][:, fore_arm].sum(1).max() < 0.35
    dropped = stretched_triangles(UNIT, lib)
    assert 0 < len(dropped) <= 40                      # a handful of seam triangles, not a hollowed-out body
