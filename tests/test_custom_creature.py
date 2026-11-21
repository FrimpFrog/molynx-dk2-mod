import numpy as np
import pytest

from scripts import custom_creature as cc


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _body():
    """Two parts. Part 0 is a 3x3x3 grid (the 'head' is y>=1). Frames: pose, then rotated about z and shifted."""
    g = np.array([(x, y, z) for x in (0, .5, 1) for y in (0, .5, 1, 1.5) for z in (0, .5, 1)], float)
    other = np.array([[3, -1, 0], [3, -0.5, 0], [3, -1, 1.0]])
    pose = [g, other]
    frames = [[g, other]]
    for a in (0.3, 0.6, 1.0):
        R = _rot_z(a)
        frames.append([g @ R.T + [0.2 * a, 0, 0], other.copy()])
    return pose, frames


HEAD = {"y": [1.0, 2.0]}


def test_region_mask_hard_and_falloff():
    pts = np.array([[0, 0.5, 0], [0, 1.0, 0], [0, 1.5, 0], [0, 0.9, 0], [0, 0.5, 0.0]])
    w = cc.region_weights(pts, {"y": [1.0, 2.0]}, falloff=0)
    assert w.tolist() == [0, 1, 1, 0, 0]
    w = cc.region_weights(pts, {"y": [1.0, 2.0]}, falloff=0.2)
    assert w[1] == 1 and w[0] == 0 and w[3] == pytest.approx(0.5)


def test_reshape_move_adds_constant_offset_to_every_frame_only_in_region():
    pose, frames = _body()
    ops = [{"op": "move", "region": HEAD, "offset": [0, 0.25, 0]}]
    out = cc.apply_reshape(frames, pose, ops, anchors={})
    inside = pose[0][:, 1] >= 1.0
    for f in range(len(frames)):
        d = out[f][0] - frames[f][0]
        assert np.allclose(d[inside], [0, 0.25, 0])
        assert np.allclose(d[~inside], 0)
        assert np.allclose(out[f][1], frames[f][1])


def test_reshape_scale_about_pivot():
    pose, frames = _body()
    ops = [{"op": "scale", "region": HEAD, "pivot": [0.5, 1.0, 0.5], "factor": [2, 1, 1]}]
    out = cc.apply_reshape(frames, pose, ops, anchors={})
    inside = pose[0][:, 1] >= 1.0
    want = pose[0].copy()
    want[inside, 0] = 0.5 + (want[inside, 0] - 0.5) * 2
    assert np.allclose(out[0][0], want)


def test_reshape_with_anchor_rotates_offset_with_the_body():
    pose, frames = _body()
    ops = [{"op": "move", "region": HEAD, "offset": [0, 0.25, 0], "anchor": "head"}]
    anchors = {"head": {"region": HEAD}}
    out = cc.apply_reshape(frames, pose, ops, anchors)
    inside = pose[0][:, 1] >= 1.0
    R = _rot_z(1.0)
    d = (out[3][0] - frames[3][0])[inside]
    assert np.allclose(d, R @ [0, 0.25, 0], atol=1e-9)


def test_reshape_keeps_shapes():
    pose, frames = _body()
    out = cc.apply_reshape(frames, pose, [{"op": "scale", "region": HEAD, "pivot": [0, 0, 0], "factor": 1.2}], {})
    assert [p.shape for p in out[2]] == [p.shape for p in frames[2]]


def test_anchor_track_recovers_rotation():
    pose, frames = _body()
    R, c = cc.anchor_track(pose, frames, HEAD)
    assert np.allclose(R[0], np.eye(3), atol=1e-9)
    assert np.allclose(R[3], _rot_z(1.0), atol=1e-9)
    sel = pose[0][:, 1] >= 1.0
    assert np.allclose(c[3], (pose[0][sel] @ _rot_z(1.0).T + [0.2, 0, 0]).mean(0), atol=1e-9)


def test_anchor_needs_three_points():
    pose, frames = _body()
    with pytest.raises(ValueError):
        cc.anchor_track(pose, frames, {"y": [9, 10]})


def test_attachment_is_rigid_and_follows_anchor():
    pose, frames = _body()
    donor = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    att = {"scale": 0.5, "place": [0.5, 2.0, 0.5], "anchor": "head"}
    out = cc.make_attachment(pose, frames, {"head": {"region": HEAD}}, donor, att)
    assert out.shape == (len(frames), 4, 3)
    d0 = np.linalg.norm(out[0][:, None] - out[0][None], axis=-1)
    for f in range(len(frames)):
        df = np.linalg.norm(out[f][:, None] - out[f][None], axis=-1)
        assert np.allclose(df, d0, atol=1e-9)
    # starts where we placed it: origin = donor centroid -> place
    assert np.allclose(out[0].mean(0), [0.5, 2.0, 0.5] + (0.5 * (donor.mean(0) - donor.mean(0))), atol=1e-9)
    # distance from the anchor centroid is constant over frames
    R, c = cc.anchor_track(pose, frames, HEAD)
    d = [np.linalg.norm(out[f].mean(0) - c[f]) for f in range(len(frames))]
    assert np.allclose(d, d[0], atol=1e-9)


def test_attachment_mirror_flips_winding_flag():
    donor = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0, 1.0]])
    pose, frames = _body()
    att = {"scale": 1, "place": [0, 2, 0], "anchor": "head", "mirror": True}
    out = cc.make_attachment(pose, frames, {"head": {"region": HEAD}}, donor, att)
    # donor x is mirrored about its own centroid: ordering of x is reversed
    assert out[0][0, 0] > out[0][1, 0]
    assert cc.flips_winding(att) is True
    assert cc.flips_winding({"scale": 1}) is False
    assert cc.flips_winding({"scale": [-1, 1, 1]}) is True


def _mesh():
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0.5]], float) / 4
    idx = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], np.uint16)
    return {"uv": uv.astype(np.float32), "indices": idx.reshape(-1)}


def test_corner_map_identity_and_reordered_vertices():
    canon = _mesh()
    assert cc.corner_map(canon, canon).tolist() == [0, 1, 2, 3, 4]
    # same triangles, vertex list shuffled + one unused vertex appended, triangle corners rotated
    perm = [3, 0, 4, 1, 2]                      # file vertex i == canon vertex perm[i]
    inv = {c: f for f, c in enumerate(perm)}
    uv = np.vstack([canon["uv"][perm], [[9, 9]]]).astype(np.float32)
    tris = [[inv[a] for a in t] for t in canon["indices"].reshape(-1, 3)]
    tris = [t[1:] + t[:1] for t in tris]        # rotate corners
    part = {"uv": uv, "indices": np.array(tris, np.uint16).reshape(-1)}
    cm = cc.corner_map(canon, part)
    assert cm[:5].tolist() == perm and cm[5] == -1


def test_corner_map_with_missing_triangles():
    canon = _mesh()
    part = {"uv": canon["uv"], "indices": canon["indices"].reshape(-1, 3)[:2].reshape(-1)}
    cm = cc.corner_map(canon, part)
    assert cm.tolist() == [0, 1, 2, 3, -1]


def test_remap_pose_uses_far_placeholder_for_unmatched():
    pose = np.arange(15, dtype=float).reshape(5, 3)
    out = cc.remap_pose(pose, np.array([4, -1, 0]))
    assert out[0].tolist() == pose[4].tolist() and out[2].tolist() == pose[0].tolist()
    assert np.all(out[1] > 1e5)


def test_far_placeholder_vertices_never_move():
    pose = [np.array([[0, 0, 0], [cc.FAR, cc.FAR, cc.FAR]], float)]
    frames = [[pose[0].copy()]]
    out = cc.apply_reshape(frames, pose, [{"op": "scale", "region": {}, "pivot": [0, 0, 0], "factor": 2}], {})
    assert np.allclose(out[0][0], pose[0])


def test_region_absx_selects_both_sides():
    pts = np.array([[-0.3, 0, 0], [0.3, 0, 0], [0.0, 0, 0]])
    assert cc.region_weights(pts, {"absx": [0.2, 1.0]}).tolist() == [1, 1, 0]


def test_submesh_keeps_selected_triangles_and_compacts_vertices():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0.0]])
    part = {"uv": np.arange(12, dtype=np.float32).reshape(6, 2), "indices": np.array([0, 1, 2, 3, 4, 5], np.uint16)}
    sub, subpts = cc.submesh(part, pts, {"x": [4, 9]})
    assert len(sub["uv"]) == 3 and sub["indices"].tolist() == [0, 1, 2]
    assert subpts.tolist() == pts[3:].tolist() and sub["uv"].tolist() == part["uv"][3:].tolist()
    assert cc.submesh(part, pts, None)[0]["indices"].tolist() == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        cc.submesh(part, pts, {"x": [100, 200]})


def test_write_custom_obj_counts(tmp_path):
    from PIL import Image
    from scripts.custom_export import write_custom_obj
    part = {"texture": "custom/x/a", "uv": np.array([[0, 0], [1, 0], [0, 1]], np.float32),
            "indices": np.array([0, 1, 2], np.uint16)}
    anim = {"parts": [part], "frames": [[np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])]]}
    p = write_custom_obj(anim, 0, {"custom/x/a": Image.new("RGB", (4, 4))}, tmp_path, "m")
    text = p.read_text().splitlines()
    assert sum(l.startswith("v ") for l in text) == 3 and sum(l.startswith("vt ") for l in text) == 3
    assert "f 1/1 2/2 3/3" in text
    assert "map_Kd custom__x__a.png" in (tmp_path / "m.mtl").read_text() and (tmp_path / "custom__x__a.png").exists()
