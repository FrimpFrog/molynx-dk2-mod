"""Checks of the Necromancer's exported files (run after necro_build.export into NECRO_FILES)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import anim_invariants, write_kmf

FILES = Path(os.environ.get("NECRO_FILES", "/home/claude/necro/out/v5/game_files"))
INFO = FILES.parent / "files.json"
pytestmark = pytest.mark.skipif(not INFO.exists(), reason="Necromancer files not exported")


def _info():
    return json.loads(INFO.read_text())


def test_every_slot_file_and_start_end_file_is_there():
    from scripts.necro_build import file_plan
    names = {n for _, n, _ in file_plan()}
    assert names == set(_info())
    assert all((FILES / f"{n}.kmf").exists() for n in names)


@pytest.mark.parametrize("name", sorted(json.loads(INFO.read_text())) if INFO.exists() else [])
def test_file_is_valid(name):
    b = (FILES / f"{name}.kmf").read_bytes()
    k = parse_kmf(b)
    assert write_kmf(k) == b
    x = k.anim or k.mesh
    assert all(len(s.vertices) <= 255 and len(s.triangles[0]) <= 255 for s in x.sprites)
    if k.anim:
        assert anim_invariants(k.anim) == []
        if k.anim.wrap:
            a, z = evaluate_anim_frame(k.anim, 0), evaluate_anim_frame(k.anim, k.anim.frame_count - 1)
            assert max(np.abs(p - q).max() for p, q in zip(a, z)) <= k.anim.scale / 511 * 1.5


def test_no_lightning_on_plain_clips():
    """Lightning (the Necro_Gear streak) only in the clips Soles chose."""
    from scripts.necro_anims import LIGHTNING_IN
    allowed = set(LIGHTNING_IN) | {"Necro-Idle2", "Necro-Research", "Necro-Magic"}
    for name in _info():
        k = parse_kmf((FILES / f"{name}.kmf").read_bytes())
        gear = any(m.textures[0] == "Necro_Gear" for m in k.materials)
        assert gear == (name in allowed), name


def test_record_dry_run():
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.dk2kwd_creatures import read_creatures
    from scripts.necro_record import build_necromancer
    kwd = Path("/mnt/user-data/uploads/Dungeon Keeper 2/Data/editor/Creatures.kwd")
    if not kwd.exists():
        pytest.skip("no Creatures.kwd")
    cs = read_creatures([b for b in read_blocks(kwd.read_bytes()) if len(b.content) > 100000][0])
    bk = next(c for c in cs if c.get("creatureId") == 22)
    before = bytes(bk.raw)
    n = build_necromancer(bk, 48, _info())
    assert bytes(bk.raw) == before
    assert n.get("creatureId") == 48 and n.get("name") == "Necromancer"
    assert [(n.get(f"attractions[{i}].present"), n.get(f"attractions[{i}].roomId"), n.get(f"attractions[{i}].roomSize"))
            for i in range(2)] == [(1, 14, 25), (0, 0, 0)]
    assert n.get("hp") == 3000 and n.get("stunDuration") == bk.get("stunDuration")
    assert n.get("manaGenPrayer") > bk.get("manaGenPrayer") and n.get("expPerSecondTraining") > bk.get("expPerSecondTraining")
    assert 3 not in (n.get("hateJobs[0]"), n.get("hateJobs[1]"))
    assert n.resource_name("animResearchResource") == "Necro-Research"
    # Raise Dead (the game's creature spell 20, the Vampire's) from level 8, with his own cast clip
    assert n.resource_name("animMagicResource") == "Necro-Magic"
    assert (n.get("spells[0].creatureSpellId"), n.get("spells[0].levelAvailable"), n.get("spells[0].playAnimation")) == (20, 8, 1)
    assert 0 < n.get("spells[0].shotDelay") < _info()["Necro-Magic"]["frames"] / 30 * 4096
    assert n.art_tail("animMagicResource").frames == _info()["Necro-Magic"]["frames"]
    assert all(n.get(f"spells[{i}].creatureSpellId") == 0 for i in (1, 2))
    # his own stats: slightly less damage, a bit faster, he researches
    assert n.get("meleeDamage") < bk.get("meleeDamage")
    assert n.get("speed") > bk.get("speed") and n.get("runSpeed") > bk.get("runSpeed")
    assert n.get("researchPerSecond") > 0 == bk.get("researchPerSecond")
    # his own walk and run, without the Black Knight's Start / End files
    from scripts.dk2kwd_creatures import ART_FLAG_HAS_END_ANIMATION as E, ART_FLAG_HAS_START_ANIMATION as S
    for slot, stem in (("animWalkResource", "Necro-Walk"), ("animRunResource", "Necro-Run")):
        t = n.art_tail(slot)
        assert n.resource_name(slot) == stem and not t.flags & (S | E)
        assert t.frames == _info()[stem]["frames"] - 1
    assert _info()["Necro-Walk"]["from"] == "own:Walk" and _info()["Necro-Run"]["from"] == "own:Run"
    # every slot's files exist (main + Start / End where the tail asks for them)
    from scripts.dk2kwd_creatures import ANIM_SLOTS
    for slot in ANIM_SLOTS[1:]:
        name, t = n.resource_name(slot), n.art_tail(slot)
        if name and t.type:
            for x in [name] + ([name + "Start"] if t.flags & S else []) + ([name + "End"] if t.flags & E else []):
                assert x in _info(), (slot, x)


def test_walk_does_not_stretch():
    """No skin edge of his own walk or run grows past twice its rest length (the waist used to
    follow the bent sword arm, Soles 11 Nov)."""
    from scripts.necro_anims import (RUN_FRAMES, WALK_FRAMES, build_rig, posed, run_keys, to_yup,
                                     walk_keys)
    from scripts.necro_build import REFERENCE, Reference, load_clip
    meshes = Path("/home/claude/dk2data/Meshes")
    if not meshes.exists():
        pytest.skip("no meshes")
    ref = Reference(load_clip(meshes, REFERENCE))
    rig, tris = build_rig(ref)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), 1), axis=0)
    rest = np.linalg.norm(rig.rest[edges[:, 0]] - rig.rest[edges[:, 1]], axis=1)
    for keys, n in ((walk_keys(), WALK_FRAMES), (run_keys(), RUN_FRAMES)):
        q = to_yup(posed(rig, keys, n, loop=True, spline=True))
        for f in q:
            ratio = np.linalg.norm(f[edges[:, 0]] - f[edges[:, 1]], axis=1) / np.maximum(rest, 1e-4)
            assert ratio.max() < 2.0
