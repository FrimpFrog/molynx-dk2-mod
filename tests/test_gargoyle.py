"""Tests of the Gargoyle (gargoyle_model, gargoyle_pose, gargoyle_anims, gargoyle_texture, gargoyle_build).

The model, clip and skin tests need nothing but this repo; the export test needs the game's meshes (the
Bile Demon's files the export is measured against) and skips without them.
"""
from collections import Counter

import numpy as np
import pytest

from scripts.gargoyle_anims import CLIPS, build_clips, skinner
from scripts.gargoyle_model import BONE_INDEX, DOUBLE_SIDED, HEAD, PARENT, build_gargoyle, counts, \
    gargoyle_skinner, welded
from scripts.gargoyle_pose import GargoylePose, leg_ik
from scripts.gargoyle_texture import ATLAS, atlas_uv, bake


@pytest.fixture(scope="module")
def parts():
    return build_gargoyle()


def test_triangle_budget_is_a_small_dk2_creature(parts):
    total = sum(t for _, t in counts(parts).values())
    assert 250 <= total <= 360          # the Goblin's files have 216, the Imp's 187; he has wings


def test_closed_parts_are_watertight_and_wound_outward(parts):
    """Every closed piece uses each directed edge once (consistent winding) and has a positive volume;
    the only open edges are the tail's root, buried in the pelvis."""
    for p in parts:
        P = np.array(p.points)
        weld = welded(p)                   # UV seams duplicate vertices: compare by position
        T = [tuple(weld[v] for v in t) for t in p.tris if p.island[t[0]] not in DOUBLE_SIDED]
        directed = Counter(e for a, b, c in T for e in ((a, b), (b, c), (c, a)))
        assert max(directed.values()) == 1, p.name
        undirected = Counter(frozenset(e) for e in directed)
        open_edges = [e for e, n in undirected.items() if n == 1]
        if p.name == "tail":
            assert len(open_edges) == 3
        else:
            assert not open_edges, p.name
        vol = sum(np.dot(P[a], np.cross(P[b], P[c])) for a, b, c in T) / 6
        assert vol > 0, p.name


def test_double_sided_pieces_have_both_faces(parts):
    for p in parts:
        T = [t for t in p.tris if p.island[t[0]] in DOUBLE_SIDED and p.island[t[0]] != "throat"]
        faces = Counter(tuple(sorted(tuple(np.round(p.points[i], 6)) for i in t)) for t in T)
        assert all(n == 2 for n in faces.values()), p.name


def test_bind_pose_skins_to_itself(parts):
    out = gargoyle_skinner(parts)(*GargoylePose().world())
    for p, q in zip(parts, out):
        assert np.allclose(np.array(p.points), q, atol=1e-9)


def test_weights_sum_to_one(parts):
    for p in parts:
        for w in p.weights:
            assert abs(sum(w.values()) - 1) < 1e-9


def test_parents_come_first():
    assert all(p is None or p < i for i, p in enumerate(PARENT))


def test_leg_ik_reaches_its_target():
    P = GargoylePose()
    target = HEAD[BONE_INDEX["footL"]] + np.array([0, 0.04, 0.05])
    leg_ik(P, "L", target, (0.3, 0.4, 1))
    _, Pw = P.world()
    assert np.linalg.norm(Pw[BONE_INDEX["footL"]] - target) < 1e-6


def test_uvs_stay_inside_their_islands_and_are_not_collapsed(parts):
    for p in parts:
        uv = atlas_uv(p) * 128
        for i, isl in enumerate(p.island):
            (x0, y0, x1, y1), _ = ATLAS[p.texture][isl]
            assert x0 <= uv[i, 0] <= x1 and y0 <= uv[i, 1] <= y1
        # a visible face never has zero area in UV (only the small flat caps of cones and tube roots do)
        P = np.array(p.points)
        for a, b, c in p.tris:
            tg = [p.tag[v] for v in (a, b, c)]
            if len({t.get("ring", "-") for t in tg}) == 1 and "ring" in tg[0] or \
                    all(t.get("along", 1) == 0 for t in tg):
                continue                   # a flat cap (one ring, or a cone's base): hidden in the body
            area3 = np.linalg.norm(np.cross(P[b] - P[a], P[c] - P[a])) / 2
            f1, f2 = uv[b] - uv[a], uv[c] - uv[a]
            area2 = abs(f1[0] * f2[1] - f1[1] * f2[0]) / 2
            if area3 > 4e-4:
                assert area2 > 0.05, (p.name, p.island[a])


def test_skins_are_128_rgb(parts):
    tex = bake(parts)
    assert set(tex) == {"Gargoyle_Body", "Gargoyle_Head", "Gargoyle_Wing"}
    for img in tex.values():
        assert img.size == (128, 128) and img.mode == "RGB"
        lum = np.asarray(img, float).mean(2)
        assert 80 < lum.mean() < 160          # as bright as the game's own stone (the Stone Knight's: 95)


@pytest.mark.parametrize("name", sorted(CLIPS))
def test_every_clip_is_at_30_fps_and_loops_close(name):
    frames, wrap = build_clips([name])[name]
    assert len(frames) >= 11             # (the Start / End clips are short, as the Goblin's: 10 frames)
    if wrap:
        assert np.allclose(frames[0].local, frames[-1].local) and np.allclose(frames[0].root, frames[-1].root)


def _feet_sets():
    sk = skinner()
    li = next(i for i, p in enumerate(sk.parts) if p.name == "legs")
    P = np.array(sk.parts[li].points)
    return sk, li, {s: np.flatnonzero((P[:, 0] * (1 if s == "L" else -1) > 0) & (P[:, 1] < .06)) for s in "LR"}


@pytest.mark.parametrize("name", sorted(CLIPS))
def test_no_clip_goes_through_the_floor_and_planted_feet_hold(name):
    """Nothing below the floor; a foot on the floor stays put (Walk and Run: moves straight back at the
    gait's steady treadmill speed, as the game's own walk files do)."""
    from scripts.gargoyle_anims import TREADMILL_SPEED
    sk, li, feet = _feet_sets()
    from scripts.gargoyle_anims import ENTRANCE
    frames, _ = build_clips([name])[name]
    speed = TREADMILL_SPEED.get(name, 0.0)
    prev = None
    for i, P in enumerate(frames):
        pts = sk(*P.world())
        if name == "entrance" and i < ENTRANCE["land"] * 30 and P.root[1] < 0.05:
            prev = None                    # still rising out of the Portal, under the floor on purpose
            continue
        assert min(q[:, 1].min() for q in pts) > -0.003
        cur = {s: (pts[li][idx][:, 1].min(), pts[li][idx].mean(0)) for s, idx in feet.items()}
        if prev is not None:
            for s in "LR":
                if cur[s][0] < 0.006 and prev[s][0] < 0.006:
                    v = (cur[s][1] - prev[s][1]) * 30
                    if name in ("walk_start", "walk_end", "walkback_start", "walkback_end"):
                        # the planted foot moves straight back (forward, walking backwards) under him as he
                        # sets off / stops, never faster than in the gait itself
                        gait = name.rsplit("_", 1)[0]
                        tread = TREADMILL_SPEED.get(gait) or build_clips([gait]) and TREADMILL_SPEED[gait]
                        along = -v[2] * np.sign(tread)
                        back = abs(v[0]) < 0.02 and -0.02 < along < abs(tread) * 1.05
                        assert back or np.hypot(v[0], v[2]) < 0.06, (name, s)    # (or a step settling)
                    else:
                        assert np.hypot(v[0], v[2] + speed) < 0.06, (name, s)
        prev = cur


ONCE_TO_STAND = ("angry", "melee2", "magic", "happy", "getting_hit", "being_slapped", "drinking", "eating")


def test_one_shot_clips_end_in_the_stand():
    """The one-shot clips the game follows with the idle end in the stand (Idle1's first frame); the
    Entrance ends in the stand ENTRANCE["forward"] ahead of where he rose (as the game's own entrances end
    ahead of their start)."""
    from scripts.gargoyle_anims import ENTRANCE, GETUP_AT
    stand = build_clips(["stand"])["stand"][0][0]
    # (GetUp stands up where he lay, GETUP_AT behind where Fallback knocked him down, as the Goblin's does)
    for name, ahead in [(n, 0.0) for n in ONCE_TO_STAND] + [("entrance", ENTRANCE["forward"]),
                                                            ("waking_up", GETUP_AT)]:
        frames, wrap = build_clips([name])[name]
        assert not wrap
        assert np.allclose(frames[-1].local, stand.local, atol=1e-6), name
        assert np.allclose(frames[-1].root, stand.root + np.array([0, 0, ahead]), atol=1e-6), name


def test_get_up_starts_where_fallback_ends_and_die_stays_down():
    fall, _ = build_clips(["falling"])["falling"]
    up, _ = build_clips(["waking_up"])["waking_up"]
    assert np.allclose(fall[-1].local, up[0].local, atol=1e-5) and np.allclose(fall[-1].root, up[0].root, atol=1e-5)
    sk = skinner()
    die, wrap = build_clips(["die"])["die"]
    assert not wrap
    def top(P):                                                   # (his body: the wings slump over the floor)
        return max(q[:, 1].max() for q, p in zip(sk(*P.world()), sk.parts) if p.name != "wings")
    assert top(die[-1]) < 0.6 * top(build_clips(["stand"])["stand"][0][0])        # a heap on the floor


def test_entrance_rises_out_of_the_floor():
    sk = skinner()
    frames, _ = build_clips(["entrance"])["entrance"]
    top = max(q[:, 1].max() for q in sk(*frames[0].world()))
    assert top < -0.1                      # wholly under the floor at the start, like the game's own


@pytest.mark.parametrize("loop,start,end", [("walk", "walk_start", "walk_end"), ("running", "run_start", "run_end"),
                                            ("walkback", "walkback_start", "walkback_end"),
                                            ("interacting", "interacting_start", "interacting_end"),
                                            ("manufacturing", "manufacturing_start", "manufacturing_end"),
                                            ("praying", "praying_start", "praying_end")])
def test_start_and_end_clips_join_their_loop_and_the_stand(loop, start, end):
    stand = build_clips(["stand"])["stand"][0][0]
    first = build_clips([loop])[loop][0][0]
    s, _ = build_clips([start])[start]
    e, _ = build_clips([end])[end]
    from scripts.gargoyle_anims import PRAY_WALK
    ahead = np.array([0, 0, PRAY_WALK if loop == "praying" else 0.0])   # (he ends where he prayed)
    for a, b, off in ((s[0], stand, 0), (s[-1], first, 0), (e[0], first, 0), (e[-1], stand, ahead)):
        assert np.allclose(a.local, b.local, atol=1e-5) and np.allclose(a.root, b.root + off, atol=1e-6)


def test_every_slot_has_its_files():
    from scripts.gargoyle_build import POSES, SLOTS, STEMS
    made = set(STEMS.values()) | set(POSES)
    for slot, (stem, start, end) in SLOTS.items():
        assert stem in made, slot
        assert not start or stem + "Start" in made, slot
        assert not end or stem + "End" in made, slot


def _meshes():
    from scripts.gargoyle_build import meshes_dir
    d = meshes_dir()
    return d if (d / "BileDemon-Idle1.kmf").exists() else None


@pytest.mark.skipif(_meshes() is None, reason="game meshes not present")
def test_export_writes_valid_game_files(tmp_path):
    from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf
    from scripts.dk2kmf_write import anim_invariants
    from scripts.gargoyle_build import build, export
    export(build(_meshes()), tmp_path)
    from scripts.gargoyle_build import STEMS
    for stem in STEMS.values():
        k = parse_kmf((tmp_path / f"Gargoyle-{stem}.kmf").read_bytes())
        assert k.kind == "anim" and anim_invariants(k.anim) == [] and len(k.anim.geom_xyz) - 1 <= 34_816
    from scripts.gargoyle_build import POSES, SLOTS
    for stem in POSES:
        assert parse_kmf((tmp_path / f"Gargoyle-{stem}.kmf").read_bytes()).kind == "mesh", stem
    for slot, (stem, start, end) in SLOTS.items():
        for s in [stem] + [stem + "Start"] * start + [stem + "End"] * end:
            assert (tmp_path / f"Gargoyle-{s}.kmf").exists(), (slot, s)
    for stem, word in (("Sleep", "bed"), ("Eat", "chicken"), ("Drink", "bottle")):     # the Goblin's props
        k = parse_kmf((tmp_path / f"Gargoyle-{stem}.kmf").read_bytes())
        assert any(word in k.materials[sp.material].textures[0].lower() for sp in k.anim.sprites), stem
    idle = parse_kmf((tmp_path / "Gargoyle-Idle1.kmf").read_bytes())
    assert idle.kind == "anim" and anim_invariants(idle.anim) == []
    for s in idle.anim.sprites:
        assert len(s.vertices) <= 255 and len(s.triangles[0]) <= 255
    p = np.concatenate(evaluate_anim_frame(idle.anim, 0))
    assert abs(p[:, 2].max()) < 1e-2                                   # feet on the floor (z down)
    assert abs((p[:, 2].max() - p[:, 2].min()) - 0.50) < 0.01          # small: between the Imp and Goblin
    pose = parse_kmf((tmp_path / "Gargoyle-PoseFrame.kmf").read_bytes())
    assert pose.kind == "mesh"
    q = np.concatenate(list(mesh_points(pose)))
    assert np.allclose(np.sort(q[:, 2]), np.sort(p[:, 2]), atol=1e-3) or abs(q[:, 2].min() - p[:, 2].min()) < 1e-3
    for stem in ("Body", "Head", "Wing"):
        assert (tmp_path / f"Gargoyle_{stem}.png").exists()


# ---- review 4: the gaits and hops as big as the Goblin's files, the joins exact

def _foot_track(name):
    sk, li, feet = _feet_sets()
    frames, _ = build_clips([name])[name]
    out = {s: [] for s in "LR"}
    low = []
    for P in frames:
        pts = sk(*P.world())
        low.append(min(q[:, 1].min() for q in pts))
        for s, idx in feet.items():
            out[s].append(pts[li][idx][:, 1].min())
    return {s: np.array(v) for s, v in out.items()}, np.array(low)


@pytest.mark.parametrize("name", ["walkback", "staggering", "praying_start", "dance"])
def test_steps_lift_the_feet_clear(name):
    feet, _ = _foot_track(name)
    for s in "LR":
        assert feet[s].max() - feet[s].min() > 0.04 / 0.5948, (name, s)     # 0.04 of the game's units


@pytest.mark.parametrize("name", ["walkback", "staggering"])
def test_backing_and_drunk_gaits_move_at_the_goblins_speed(name):
    from scripts.gargoyle_anims import TREADMILL_SPEED
    build_clips([name])
    assert abs(TREADMILL_SPEED[name]) / 30 * 0.5948 > 0.0125                # Goblin: 0.0135, 0.014 a frame


def test_dance_hops_clear_of_the_floor_and_starts_on_the_idle():
    _, low = _foot_track("dance")
    assert low.max() > 0.05 / 0.5948
    stand = build_clips(["stand"])["stand"][0][0]
    first = build_clips(["dance"])["dance"][0][0]
    assert np.allclose(first.local, stand.local, atol=1e-5) and np.allclose(first.root, stand.root, atol=1e-6)


def test_drunk_and_drunk_idle_share_their_first_frame():
    c = build_clips(["staggering", "drunk_idle"])
    a, b = c["staggering"][0][0], c["drunk_idle"][0][0]
    assert np.allclose(a.local, b.local, atol=1e-5) and np.allclose(a.root, b.root, atol=1e-6)


def test_pray_start_walks_from_the_idle_spot():
    from scripts.gargoyle_anims import PRAY_WALK
    s, _ = build_clips(["praying_start"])["praying_start"]
    stand = build_clips(["stand"])["stand"][0][0]
    assert np.allclose(s[0].root, stand.root, atol=1e-6) and abs(s[-1].root[2] - PRAY_WALK) < 1e-6


def test_he_sleeps_in_the_bile_demon_nest(tmp_path):
    """Soles, 15 Nov: lair object 18. The Sleep file carries the nest as BileDemon-bed(sleeping) does at
    frame 0 (its material and texture, as the Dragon's Sleep), and in no frame is any part of him under the
    nest's top surface, while his lowest point rests on it (within a few thousandths)."""
    from pathlib import Path
    from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
    from scripts.dragon_build import _donor, _donor_sprites, export
    from scripts.gargoyle_build import body_samples, build, meshes_dir, nest_surface, surface_under
    md = meshes_dir()
    if not (Path(md) / "BileDemon-bed(sleeping).kmf").exists():
        pytest.skip("no game meshes")
    export(build(md, names=["sleeping"]), tmp_path, textures={})
    k = parse_kmf((tmp_path / "Gargoyle-Sleep.kmf").read_bytes())
    donor = _donor(md, "BileDemon-bed(sleeping)")
    (di,) = _donor_sprites(donor, "bed")
    tex = [k.materials[s.material].textures[0] for s in k.anim.sprites]
    assert "Creature Bed_Goblin IN" not in tex
    (ni,) = [i for i, t in enumerate(tex) if t == "Creature Bed_Bile Demon1"]
    nest0 = evaluate_anim_frame(donor.anim, 0)[di]
    assert k.anim.sprites[ni].triangles == donor.anim.sprites[di].triangles
    tris = np.asarray(donor.anim.sprites[di].triangles[0])
    surf = nest_surface(nest0, tris)
    lowest = []
    for f in range(k.anim.frame_count):
        F = evaluate_anim_frame(k.anim, f)
        assert np.abs(F[ni] - nest0).max() < 2e-3
        S = np.concatenate([body_samples(F[i], np.asarray(s.triangles[0]))
                            for i, s in enumerate(k.anim.sprites) if i != ni])
        clearance = -S[:, 2] - surface_under(surf, S)
        assert clearance.min() > -1e-3, f
        lowest.append(clearance.min())
    assert max(lowest) < 0.01                                   # resting on the straw, not floating
