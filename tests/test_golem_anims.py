from pathlib import Path

import numpy as np
import pytest

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
UNIT = Path("/tmp/mg/mgd/techs/megapack/factions/magic/units/golem")
needs = pytest.mark.skipif(not ((S / "extracted/Meshes").exists() and UNIT.exists()), reason="game/golem files absent")

# the Golem is a brute like the Bile Demon: no spell casting or research animations
# standing idle, walking, attacking and dying are the MegaGlest Golem's own animations (stand, walk, attack, die)
NAMES = """looking_around fidgeting running turning sleeping waking_up eating training
manufacturing praying guarding celebrating angry frightened happy unhappy blocking special_attack
getting_hit staggering knockback picking_up dropping being_picked_up being_slapped patrolling
interacting_with_objects fighting_another_creature victory_celebration""".split()


@pytest.fixture(scope="module")
def lib():
    from scripts.dk2texture import TextureCache
    from scripts.golem_anims import build_all
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    return build_all(S / "extracted/Meshes", cache, UNIT)


@needs
def test_every_requested_animation_exists(lib):
    assert set(NAMES) <= set(lib), set(NAMES) - set(lib)


@needs
def test_animations_are_well_formed_and_slow(lib):
    from scripts.make_golem_anims import golem_rest
    g0, _ = golem_rest(UNIT)
    for name, a in lib.items():
        fr = np.array(a["frames"])
        assert fr.shape[1:] == g0.shape, name
        assert np.isfinite(fr).all(), name
        assert len(fr) >= 10 and 8 <= a["fps"] <= 60, (name, len(fr), a["fps"])
        assert a["kind"] in ("dk2", "golem", "hand")
        step = np.abs(np.diff(fr, axis=0)).max()          # no teleporting vertices between frames
        assert step < 1.2, (name, step)
        assert fr[:, :, 1].min() > -0.5, name              # nothing sinks through the floor


@needs
def test_looping_animations_close(lib):
    for name, a in lib.items():
        if a["wrap"]:
            fr = np.array(a["frames"])
            assert np.abs(fr[0] - fr[-1]).max() < 0.9, name


@needs
def test_face_and_chest_move_only_with_the_torso():
    """Regression: the face used to be skinned to the arms and stretched whenever an arm moved."""
    from scripts.golem_retarget import BONES, _weights
    from scripts.make_golem_anims import golem_rest
    g0, idx = golem_rest(UNIT)
    W = _weights(g0, idx)
    x, y, z = g0.T
    face = (np.abs(x) < 0.5) & (y > 0.9) & (y < 1.7) & (z > 0.9)
    assert face.sum() > 10
    assert (W[face, BONES.index("torso")] > 0.999).all()
    assert np.allclose(W.sum(1), 1)


@needs
def test_no_gambling_animation_is_invented(lib):
    assert "gambling" not in lib
