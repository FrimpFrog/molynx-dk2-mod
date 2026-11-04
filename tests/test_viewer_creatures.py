from pathlib import Path

import pytest

from scripts.viewer_creatures import CREATURES, find_animations

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
MESHES = S / "extracted/Meshes"
needs_game = pytest.mark.skipif(not MESHES.exists(), reason="DK2 files not present")


@pytest.fixture(scope="module")
def found():
    return find_animations(MESHES, {k: CREATURES[k] for k in ("goblin", "bile_demon", "horned_reaper")})


@needs_game
def test_goblin_has_walk_and_dance(found):
    anims, _ = found
    assert {"walk", "dance"} <= set(anims["goblin"])
    assert all(p.exists() for p in anims["goblin"].values())


@needs_game
def test_bile_demon_has_walk_dance_melee(found):
    assert {"walk", "dance", "melee"} <= set(found[0]["bile_demon"])


@needs_game
def test_reaper_has_walk_roar_melee1(found):
    assert {"walk", "roar", "melee1"} <= set(found[0]["horned_reaper"])


@needs_game
def test_pose_files_are_not_animations(found):
    for labels in found[0].values():
        assert not any("pose" in k for k in labels)


@needs_game
def test_case_variants_attach_to_one_creature(found):
    anims = found[0]
    assert "walkback" in anims["bile_demon"]          # BIL-Walkback
    assert "dance" in anims["bile_demon"]             # Bil-Dance
    assert not any("bil" in k for k in anims["goblin"])


@needs_game
def test_unattributed_files_are_reported(found):
    assert "Arc-Walk" in found[1]
    assert "Goblin-Walk" not in found[1]
