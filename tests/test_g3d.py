import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.g3d import G3DMesh, G3DModel, parse_g3d, write_g3d_v4

GOLEM = Path("/tmp/mg/mgd/techs/megapack/factions/magic/units/golem/models")
needs_golem = pytest.mark.skipif(not GOLEM.exists(), reason="MegaGlest data not present")


def _mesh(**kw):
    rng = np.random.default_rng(1)
    base = dict(name="m", vertices=rng.random((3, 4, 3), dtype=np.float32), normals=rng.random((3, 4, 3), dtype=np.float32),
                uv=rng.random((4, 2), dtype=np.float32), indices=np.array([0, 1, 2, 2, 1, 3], np.uint32),
                diffuse_texture="a.png", custom_color=True)
    base.update(kw)
    return G3DMesh(**base)


def test_v4_roundtrip_keeps_everything():
    m = G3DModel(4, [_mesh(), _mesh(name="b", two_sided=True, glow=True, specular_texture="s.png", diffuse_texture="d.png")])
    back = parse_g3d(write_g3d_v4(m))
    assert [x.name for x in back.meshes] == ["m", "b"]
    assert back.meshes[0].frame_count == 3 and back.meshes[0].triangle_count == 2
    assert np.array_equal(back.meshes[1].vertices, m.meshes[1].vertices)
    assert (back.meshes[1].two_sided, back.meshes[1].glow, back.meshes[1].custom_color) == (True, True, True)
    assert (back.meshes[1].diffuse_texture, back.meshes[1].specular_texture) == ("d.png", "s.png")
    assert back.meshes[0].custom_color and not back.meshes[0].two_sided


def test_untextured_mesh_has_no_uv_block():
    m = G3DModel(4, [_mesh(uv=None, diffuse_texture=None, custom_color=False)])
    b = write_g3d_v4(m)
    back = parse_g3d(b)
    assert back.meshes[0].uv is None and back.meshes[0].diffuse_texture is None


def test_rejects_bad_files():
    with pytest.raises(ValueError):
        parse_g3d(b"XXX\x04" + b"\0" * 20)
    good = write_g3d_v4(G3DModel(4, [_mesh()]))
    with pytest.raises(ValueError):
        parse_g3d(good[:-5])


def test_v3_layout():
    nv, ni = 3, 3
    body = struct.pack("<I", 1)                       # one mesh
    body += struct.pack("<7I", 2, 2, 1, 1, nv, ni, 4)  # frames, normal frames, uv frames, colour frames, verts, indices, props=custom colour
    body += b"tex.tga".ljust(64, b"\0")
    body += np.arange(2 * nv * 3, dtype="<f4").tobytes() + np.zeros(2 * nv * 3, "<f4").tobytes()
    body += np.array([0, 0, 1, 0, 0, 1], "<f4").tobytes() + np.zeros(4, "<f4").tobytes()
    body += np.array([0, 1, 2], "<u4").tobytes()
    m = parse_g3d(b"G3D\x03" + body)
    me = m.meshes[0]
    assert m.version == 3 and me.frame_count == 2 and me.diffuse_texture == "tex.tga" and me.custom_color
    assert me.vertices[1, 0].tolist() == [9.0, 10.0, 11.0] and me.uv.tolist() == [[0, 0], [1, 0], [0, 1]]


@needs_golem
def test_real_golem_files_parse_and_roundtrip_byte_exact():
    for f in GOLEM.glob("*.g3d"):
        raw = f.read_bytes()
        m = parse_g3d(raw)
        for me in m.meshes:
            assert me.vertices.shape[1] == 0 or me.indices.max() < me.vertices.shape[1]
        assert write_g3d_v4(m) == raw, f.name


@needs_golem
def test_golem_animation_facts():
    walk = parse_g3d((GOLEM / "golem_walking.g3d").read_bytes())
    assert (walk.frame_count, walk.meshes[0].vertices.shape[1], walk.meshes[0].triangle_count) == (15, 407, 398)
    assert walk.meshes[0].diffuse_texture == "texture_golem.png" and walk.meshes[0].custom_color
