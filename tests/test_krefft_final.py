"""Krefft's final build: the Gamble clip's claw on the lever, the export plan constants it relies on, Coin Spit's
records and his creature record (built from the staged game tables; skipped when they are missing)."""
from pathlib import Path

import numpy as np
import pytest

GAME = Path("/mnt/user-data/uploads/Dungeon Keeper 2")
EDITOR = GAME / "flame/resources/MoreCreatures/Data/editor"
MESHES = Path("/home/claude/dk2data/Meshes")
need_meshes = pytest.mark.skipif(not MESHES.exists(), reason="no game meshes")
need_tables = pytest.mark.skipif(not EDITOR.exists(), reason="no staged MoreCreatures tables")


@need_meshes
def test_plan_constants_match_the_export_plan():
    from scripts.krefft_anims import PLAN_SCALE, PLAN_SHIFT
    from scripts.krefft_anims_build import plan
    pl = plan()
    assert pl.scale == pytest.approx(PLAN_SCALE)
    # file model space = (-k0, -k2, k1) of the KMF, k = rig-to-KMF of the plan
    assert np.allclose(PLAN_SHIFT, [-pl.offset[0], -pl.offset[2], pl.offset[1]])


@need_meshes
def test_claw_holds_the_lever_knob():
    from scripts import krefft_rig as KR
    from scripts.krefft_anims import CLIPS, grip, lever_angle, lever_knob
    from scripts.krefft_anims_build import plan, sample
    pl = plan()
    sk = KR.model()[1]
    _pts, bones, times = sample(CLIPS["gamble"]())
    for f in range(14, 29):                         # held from the top of the reach to the end of the hold
        k = pl.to_kmf([bones[f][1][sk.index["handFR"]][None]])[0][0]
        wrist = np.array([-k[0], -k[2], k[1]])
        a = lever_angle(times[f])
        assert np.linalg.norm(wrist - (lever_knob(a) + grip(a))) < .02, f


def test_lever_pull_is_a_real_pull():
    from scripts.krefft_anims import LEVER_PULLED, LEVER_REST, lever_angle, lever_knob
    assert lever_angle(0.0) == LEVER_REST and lever_angle(0.9) == LEVER_PULLED
    assert lever_knob(LEVER_REST)[1] - lever_knob(LEVER_PULLED)[1] > .25      # the knob comes down a long way


@need_tables
def test_coin_spit_records():
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.dk2kwd_tables import read_records
    from scripts.krefft_coin_spit import SHOT_ID, SPELL_ID, apply_coin_spit, coin_art
    art = coin_art(EDITOR)
    for name in ("CreatureSpells.kwd", "Shots.kwd"):
        old = (EDITOR / name).read_bytes()
        new = apply_coin_spit(old, art)
        assert len(new) == len(old)
        recs = {r.id: r for r in read_records(read_blocks(new)[0])}
        olds = {r.id: r for r in read_records(read_blocks(old)[0])}
        for i in recs:
            if i not in (SPELL_ID, SHOT_ID):
                assert recs[i].raw == olds[i].raw
    shots = {r.id: r for r in read_records(read_blocks(apply_coin_spit((EDITOR / "Shots.kwd").read_bytes(), art))[0])}
    s = shots[SHOT_ID]
    assert s.name == "Coin Spit" and s.get("mesh.type") == 5 and s.get("mesh.name").rstrip(b"\0") == b"Coin"
    assert s.get("generalEffectId") == 34 and s.get("hitThingEffectId") == 35 and s.get("attackCategory") == 13
    spells = {r.id: r for r in read_records(read_blocks(apply_coin_spit((EDITOR / "CreatureSpells.kwd").read_bytes(), art))[0])}
    assert spells[SPELL_ID].name == "Coin Spit" and spells[SPELL_ID].get("shotTypeId") == SHOT_ID
    with pytest.raises(ValueError):                 # applied twice: the slots are no longer the game's spares
        apply_coin_spit(apply_coin_spit((EDITOR / "Shots.kwd").read_bytes(), art), art)


@need_tables
def test_record_builds_from_the_bile_demon():
    from scripts.cast_records import build
    from scripts.dk2kld_splice import _creatures_index, _records
    from scripts.dk2kwd_blocks import read_blocks
    from scripts.cast_records import all_files
    from scripts.krefft_record import KREFFT
    b = read_blocks((EDITOR / "Creatures.kwd").read_bytes())
    recs = _records(b[_creatures_index(b)])
    info = {n: {"kind": "mesh" if n.endswith(("Pose", "PoseFrame")) else "anim", "frames": 41, "wrap": True}
            for n in all_files(KREFFT)}
    n = build(KREFFT, recs, 54, info, {"name": 1, "general": 2, "strength": 3, "weakness": 4})
    assert n.get("creatureId") == 54 and n.get("name") == "Krefft"
    assert n.resource_name("animResearchResource") == "Krefft-Gamble"
    assert n.get("spells[0].creatureSpellId") == 35 and n.get("attractions[0].roomId") == 15
    assert n.get("hateJobs[0]") == 4 and n.get("happyJobs[0].jobType") == 7
    assert n.art_tail("animSleepResource").fps == 15
