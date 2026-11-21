"""Tests of Sejanus, the horse demon (sejanus_model, sejanus_pose, sejanus_anims, sejanus_texture, sejanus_build).

The model and clip tests need nothing but this repo; the export test needs the game's meshes (the Bile Demon's
files the exporter measures against) and skips without them.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.dk2paths import MESHES_DIR, have
from scripts.sejanus_anims import build_clips
from scripts.sejanus_model import BONE_INDEX, HEAD, PARENT, Skinner, build_sejanus, counts
from scripts.sejanus_pose import Pose, leg_ik
from scripts.sejanus_texture import ATLAS, EXTENT, SIZE, atlas_uv, bake

LOFTS = {"trunk": ("torso", "neck"), "tail": ("tail",), "head": ("skull",), "jaw": ("jaw",),
         "legs": ("thigh", "shank"), "arms": ("arm",)}
_ALT = Path("/home/claude/dk2data/Meshes")
MESHES = MESHES_DIR if have(MESHES_DIR) else _ALT


@pytest.fixture(scope="module")
def parts():
    return build_sejanus()


def test_triangle_budget_is_a_big_dk2_creature(parts):
    total = sum(t for _, t in counts(parts).values())
    assert 380 <= total <= 424          # the Horned Reaper's PoseFrame has 406


def test_lofts_are_closed_and_wound_outward(parts):
    """Every loft is a closed surface (no hole, no open tube end: each edge shared by exactly two
    triangles, once each way) with a positive volume (counter-clockwise outward)."""
    for p in parts:
        P = np.array(p.points)
        ids, key = [], {}
        for q in P:
            ids.append(key.setdefault(tuple(np.round(q, 7)), len(key)))
        tris = [t for t in p.tris if p.island[t[0]] in LOFTS[p.name] and not p.tag[t[0]].get("cheek")]
        directed = Counter((ids[x], ids[y]) for a, b, c in tris for x, y in ((a, b), (b, c), (c, a)))
        assert all(n == 1 for n in directed.values()), p.name
        assert all((y, x) in directed for x, y in directed), p.name
        vol = sum(np.dot(P[a], np.cross(P[b], P[c])) for a, b, c in tris) / 6
        assert vol > 0, p.name


def test_bind_pose_skins_to_itself(parts):
    out = Skinner(parts)(*Pose().world())
    for p, q in zip(parts, out):
        assert np.allclose(np.array(p.points), q, atol=1e-9)


def test_weights_sum_to_one(parts):
    for p in parts:
        for w in p.weights:
            assert abs(sum(w.values()) - 1) < 1e-9


def test_parents_come_first():
    assert all(p is None or p < i for i, p in enumerate(PARENT))


def test_leg_ik_reaches_its_target():
    P = Pose()
    target = HEAD[BONE_INDEX["footL"]] + np.array([0, 0.04, 0.06])
    leg_ik(P, "L", target, (0.1, 0, 1))
    _, Pw = P.world()
    assert np.linalg.norm(Pw[BONE_INDEX["footL"]] - target) < 1e-6


def test_proportions_match_the_brief(parts):
    """Hooves on the floor, the horns on top; the torso pitched 35-45 degrees from vertical; the hands
    hanging to about the stifle (knee); the head out in front of the chest."""
    P = {p.name: np.array(p.points) for p in parts}
    allp = np.concatenate(list(P.values()))
    assert abs(allp[:, 1].min()) < 1e-6
    assert P["head"][:, 1].max() == allp[:, 1].max()
    spine = HEAD[BONE_INDEX["neck1"]] - HEAD[BONE_INDEX["root"]]
    lean = np.degrees(np.arctan2(spine[2], spine[1]))
    assert 35 <= lean <= 45
    stifle = HEAD[BONE_INDEX["shinL"]][1]
    arms = next(p for p in parts if p.name == "arms")
    hand = min(q[1] for q, isl in zip(arms.points, arms.island) if isl == "arm")    # the knuckles
    assert abs(hand - stifle) < 0.08
    assert P["head"][:, 2].max() > P["trunk"][:, 2].max() + 0.1


def test_uvs_stay_inside_their_islands(parts):
    for p in parts:
        uv = atlas_uv(p) * SIZE
        for i, isl in enumerate(p.island):
            (x0, y0, x1, y1), _ = ATLAS[p.texture][isl]
            assert x0 <= uv[i, 0] <= x1 and y0 <= uv[i, 1] <= y1


def test_uvs_are_not_stretched(parts):
    """Texels per model unit along an island's u and its v differ by less than half (area-weighted)."""
    for p in parts:
        P = np.array(p.points)
        uv = atlas_uv(p) * SIZE
        isl = np.array(p.island)
        for name in set(isl):
            acc, area = np.zeros(2), 0.0
            for a, b, c in p.tris:
                if isl[a] != name:
                    continue
                du, dp = uv[[b, c]] - uv[a], P[[b, c]] - P[a]
                if abs(np.linalg.det(du)) < 1e-9:
                    continue
                J = np.linalg.solve(du, dp)                 # model units per texel along x and y
                A = np.linalg.norm(np.cross(dp[0], dp[1])) / 2
                acc += A * np.linalg.norm(J, axis=1)
                area += A
            sx, sy = acc / area
            assert max(sx, sy) / min(sx, sy) < 1.5, (p.name, name, sx, sy)
            assert name in EXTENT


def test_skins_are_128_rgb(parts):
    tex = bake(parts)
    assert set(tex) == {"Sejanus_Body", "Sejanus_Limbs", "Sejanus_Head"}
    for img in tex.values():
        assert img.size == (128, 128) and img.mode == "RGB"


def test_idle_loops_and_keeps_the_hooves_down(parts):
    frames, wrap = build_clips(["stand"])["stand"]
    assert wrap and len(frames) >= 16
    assert np.allclose(frames[0].local, frames[-1].local) and np.allclose(frames[0].root, frames[-1].root)
    sk = Skinner(parts)
    for P in frames[::10]:
        low = min(q[:, 1].min() for q in sk(*P.world()))
        assert abs(low) < 0.01


@pytest.mark.skipif(not have(MESHES), reason="game meshes not present")
def test_export_writes_valid_game_files(tmp_path):
    from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf
    from scripts.dk2kmf_write import anim_invariants
    from scripts.sejanus_build import build, export
    export(build(MESHES), tmp_path)
    idle = parse_kmf((tmp_path / "Sejanus-Idle1.kmf").read_bytes())
    assert idle.kind == "anim" and anim_invariants(idle.anim) == []
    for s in idle.anim.sprites:
        assert len(s.vertices) <= 255 and len(s.triangles[0]) <= 255
    p = np.concatenate(evaluate_anim_frame(idle.anim, 0))
    assert abs(p[:, 2].max()) < 1e-2                      # hooves on the floor (z down)
    assert abs((p[:, 2].max() - p[:, 2].min()) - 0.74) < 0.01     # the Horned Reaper's file height
    pose = parse_kmf((tmp_path / "Sejanus-PoseFrame.kmf").read_bytes())
    assert pose.kind == "mesh"
    assert np.allclose(np.concatenate(mesh_points(pose)).max(0)[2], p[:, 2].max(), atol=1e-2)
    for name in ("Sejanus_Body", "Sejanus_Limbs", "Sejanus_Head"):
        assert (tmp_path / f"{name}.png").exists()


def _inside(pt, P, tris):
    d = np.array([0.31, 0.77, 0.55])
    d /= np.linalg.norm(d)
    n = 0
    for a, b, c in tris:
        e1, e2 = P[b] - P[a], P[c] - P[a]
        h = np.cross(d, e2)
        det = e1 @ h
        if abs(det) < 1e-12:
            continue
        s = pt - P[a]
        u = s @ h / det
        q = np.cross(s, e1)
        v = d @ q / det
        if 0 <= u <= 1 and v >= 0 and u + v <= 1 and e2 @ q / det > 0:
            n += 1
    return n % 2 == 1


def test_finger_roots_are_inside_the_hands(parts):
    """The fingers are open cones: their bases must be hidden inside the closed hands."""
    arms = next(p for p in parts if p.name == "arms")
    P = np.array(arms.points)
    hand = [t for t in arms.tris if arms.island[t[0]] == "arm"]
    roots = [i for i, (isl, tg) in enumerate(zip(arms.island, arms.tag)) if isl == "finger" and tg["along"] == 0]
    assert roots and all(_inside(P[i], P, hand) for i in roots)


def test_closed_mouth_shows_no_cheek(parts):
    """The open mouth's cheek wedges fold flat along the lip line when the jaw is closed (bind)."""
    jaw = next(p for p in parts if p.name == "jaw")
    P = np.array(jaw.points)
    for a, b, c in jaw.tris:
        if jaw.tag[a].get("cheek"):
            assert np.linalg.norm(np.cross(P[b] - P[a], P[c] - P[a])) / 2 < 2e-4


def test_every_slot_has_a_file_after_a_full_export(tmp_path):
    """Stage 3: every animation slot DK2 plays for a creature (the Dragon's set, with Run, Research Start /
    End and Melee2 too) gets a real file, Start / End files included; clips start / end on Idle1's frame 0
    or on their loop's frame 0, and stay within the game's keyframe limit."""
    from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
    from scripts.dk2kmf_write import anim_invariants
    from scripts.sejanus_build import SLOTS, build, export
    from scripts.dk2kwd_creatures import ANIM_SLOTS
    info = export(build(MESHES), tmp_path)
    assert all(v["keys"] <= 34_816 for v in info.values())
    assert set(SLOTS) <= set(ANIM_SLOTS)
    for slot, (stem, start, end) in SLOTS.items():
        for s in [stem] + ([stem + "Start"] if start else []) + ([stem + "End"] if end else []):
            k = parse_kmf((tmp_path / f"Sejanus-{s}.kmf").read_bytes())
            if k.kind == "anim":
                assert anim_invariants(k.anim) == [], s

    def frame(stem, i):
        a = parse_kmf((tmp_path / f"Sejanus-{stem}.kmf").read_bytes()).anim
        return np.concatenate(evaluate_anim_frame(a, i % a.frame_count))[:338]

    idle = frame("Idle1", 0)
    # a Start ends one frame before its loop's frame 0 (the loop's last frame is a copy of its first), an
    # End begins one frame after it: the motion runs straight on through the switch
    for loop in ("Walk", "Run", "Walkback", "Research", "Building", "Pray"):
        assert np.abs(frame(loop + "Start", 0) - idle).max() < 0.004, loop
        assert np.abs(frame(loop + "Start", -1) - frame(loop, -2)).max() < 0.004, loop
        assert np.abs(frame(loop + "End", 0) - frame(loop, 1)).max() < 0.004, loop
        if loop != "Pray":                    # (PrayEnd stands up where he prayed, at the temple's edge)
            assert np.abs(frame(loop + "End", -1) - idle).max() < 0.004, loop
    for once in ("Magic", "Happy", "Dance", "RecoilHff", "RecoilHfb", "Drink", "Eat", "Melee2"):
        assert np.abs(frame(once, 0) - idle).max() < 0.004, once
        assert np.abs(frame(once, -1) - idle).max() < 0.004, once


def test_lying_clips_rest_on_the_floor(parts):
    sk = Skinner(parts)
    for name in ("sleeping", "falling", "die", "waking_up", "dragged"):
        frames, _ = build_clips([name])[name]
        low = [min(v[:, 1].min() for v in sk(*P.world())) for P in frames]
        assert min(low) > -0.004, name
        assert abs(low[-1 if name != "waking_up" else 0]) < 0.004, name
