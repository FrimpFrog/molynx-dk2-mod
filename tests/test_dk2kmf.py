from pathlib import Path

import numpy as np
import pytest

from scripts.dk2kmf import evaluate_anim_frame, parse_kmf, write_obj

MESHES = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/"
              "scratchpad/extracted/Meshes")
needs_game = pytest.mark.skipif(not MESHES.exists(), reason="extracted DK2 meshes not present")


def _load(name: str):
    return parse_kmf((MESHES / name).read_bytes())


def test_rejects_non_kmf():
    with pytest.raises(ValueError, match="KMSH"):
        parse_kmf(b"NOPE" + bytes(100))


@needs_game
def test_every_kmf_file_parses_and_is_fully_consumed():
    files = sorted(MESHES.glob("*.kmf"))
    assert len(files) == 2015
    for path in files:
        data = path.read_bytes()
        model = parse_kmf(data)
        assert model.end == len(data), path.name


@needs_game
def test_file_kinds_match_header_counts():
    kinds = {"mesh": 0, "anim": 0, "grop": 0}
    for path in MESHES.glob("*.kmf"):
        kinds[parse_kmf(path.read_bytes()).kind] += 1
    assert kinds == {"mesh": 862, "anim": 1145, "grop": 8}


@needs_game
def test_reaper_static_pose_is_consistent():
    model = _load("Rea3-poseframe.kmf")
    mesh = model.mesh
    assert model.kind == "mesh" and len(mesh.geometry) > 100
    assert model.materials
    for sprite in mesh.sprites:
        assert sprite.material < len(model.materials)
        verts = len(sprite.vertices)
        assert verts <= 256  # triangle indices are single bytes
        assert all(v.geom < len(mesh.geometry) for v in sprite.vertices)
        assert all(max(t) < verts for t in sprite.triangles[0])


@needs_game
def test_animation_frames_are_finite_and_match_static_size():
    static = _load("Rea3-poseframe.kmf")
    anim = _load("REA3_Walk.kmf").anim
    sp = np.concatenate([np.array(static.mesh.geometry)])
    static_size = sp.max(axis=0) - sp.min(axis=0)
    for frame in (0, anim.frame_count // 2, anim.frame_count - 1):
        pts = np.concatenate(evaluate_anim_frame(anim, frame))
        assert np.isfinite(pts).all()
        size = pts.max(axis=0) - pts.min(axis=0)
        # same character: animated extent within 2.5x of the static pose in every axis
        assert (size < 2.5 * static_size).all() and (size > 0.2 * static_size).all()


@needs_game
def test_write_obj_counts(tmp_path):
    model = _load("Rea3-poseframe.kmf")
    out = tmp_path / "reaper.obj"
    write_obj(model, out)
    text = out.read_text().splitlines()
    v = sum(1 for line in text if line.startswith("v "))
    f = sum(1 for line in text if line.startswith("f "))
    assert v == sum(len(s.vertices) for s in model.mesh.sprites)
    assert f == sum(len(s.triangles[0]) for s in model.mesh.sprites)
