from pathlib import Path

import numpy as np
import pytest

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
UNIT = Path(__file__).resolve().parent.parent / "megaglest_behemoth"
needs = pytest.mark.skipif(not ((S / "extracted/Meshes").exists() and (UNIT / "behemoth.xml").exists()), reason="game/behemoth files absent")

# stand, walk, attack, die are the MegaGlest Behemoth's own; each of these is a real DK2 clip retargeted. Nothing is hand-posed and
# the clips DK2 has no animation for (casting, researching, blocking, ...) are not made.
NAMES = """fidgeting running eating praying happy angry celebrating victory_celebration getting_hit being_slapped staggering
waking_up knockback sleeping training fighting_another_creature patrolling guarding manufacturing interacting_with_objects
being_picked_up""".split()
MISSING = "casting_spell special_attack researching blocking frightened unhappy looking_around picking_up dropping turning".split()


@pytest.fixture(scope="module")
def lib():
    from scripts.behemoth_anims import build_all
    from scripts.dk2texture import TextureCache
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    return build_all(S / "extracted/Meshes", cache, UNIT)


@needs
def test_only_clips_with_a_real_dk2_source_exist(lib):
    assert set(lib) == set(NAMES)
    assert not set(MISSING) & set(lib)
    assert all(a["kind"] == "dk2" for a in lib.values())


@needs
def test_animations_are_well_formed_smooth_and_not_torn(lib):
    from scripts.behemoth_rig import behemoth_rig, stretched_triangles
    rig = behemoth_rig(UNIT)
    tri = rig.indices.reshape(-1, 3)
    keep = np.ones(len(tri), bool)
    keep[stretched_triangles(UNIT, lib)] = False
    t = tri[keep]
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    rest_max = np.linalg.norm(rig.rest[e[:, 0]] - rig.rest[e[:, 1]], axis=1).max()
    for name, a in lib.items():
        fr = np.array(a["frames"])
        assert fr.shape[1:] == rig.rest.shape, name
        assert np.isfinite(fr).all(), name
        assert len(fr) >= 17 and 20 <= a["fps"] <= 40, (name, len(fr), a["fps"])
        limit = 2.5 if name in ("training", "fighting_another_creature") else 1.0     # his own club strike really travels that fast
        assert np.abs(np.diff(fr, axis=0)).max() < limit, name                 # no teleporting vertices
        longest = np.linalg.norm(fr[:, e[:, 0]] - fr[:, e[:, 1]], axis=2).max()
        assert longest < rest_max * 1.3, (name, longest)                     # no torn / stretched triangles
        assert fr[:, :, 1].min() > -0.05, name                               # nothing sinks through the floor


@needs
def test_club_stays_in_the_hand_and_hip_seams_are_dropped(lib):
    from scripts.behemoth_rig import CLUB_VERTICES, behemoth_rig, stretched_triangles
    rig = behemoth_rig(UNIT)
    assert rig.weights[:CLUB_VERTICES].argmax(1).tolist() == [rig.bones.index("foreN")] * CLUB_VERTICES
    dropped = stretched_triangles(UNIT, lib)
    assert 0 < len(dropped) <= 30


@needs
def test_rig_weights_are_normalised_and_loincloth_follows_the_torso():
    from scripts.behemoth_rig import behemoth_rig
    rig = behemoth_rig(UNIT)
    assert np.allclose(rig.weights.sum(1), 1) and (rig.weights >= 0).all()
    skirt = (rig.rest[:, 1] > 1.0) & (rig.rest[:, 1] < 1.4) & (np.abs(rig.rest[:, 0]) > 0.3)
    assert rig.weights[skirt][:, rig.bones.index("torso")].mean() > 0.3
