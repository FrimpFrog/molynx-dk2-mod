"""Reader/writer for Glest / MegaGlest .g3d models (versions 3 and 4).

Format (little-endian), as documented in the MegaGlest Blender add-on and confirmed against the
engine source (source/shared_lib/sources/graphics/model.cpp, interpolation.cpp):

  header      : "G3D" + uint8 version
  v4 model    : uint16 meshCount, uint8 type(0)
  v3 model    : uint32 meshCount
  per mesh v4 : char name[64], u32 frameCount, u32 vertexCount, u32 indexCount, f32 diffuse[3],
                f32 specular[3], f32 specularPower, f32 opacity, u32 properties, u32 textureFlags,
                then one char[64] file name per set bit of textureFlags (1 diffuse, 2 specular, 4 normal),
                then f32 vertices[frame][vertex][3], f32 normals[frame][vertex][3] (animation = one
                full vertex set per frame), f32 uv[vertex][2] (only if textured), u32 indices[indexCount].
  per mesh v3 : u32 frameCount, normalFrames, uvFrames, colorFrames, vertexCount, indexCount, properties,
                char diffuseTexture[64]; then vertices, normals, uvs, colours(4*colorFrames), indices.
  properties v4: 1 = custom (team) colour in the texture alpha, 2 = two-sided, 4 = not selectable, 8 = glow.
  properties v3: 1 = NO texture, 2 = two-sided, 4 = custom colour.

Animation: every frame stores all vertices; the engine blends linearly between neighbouring frames
(looping for cycled skills: prev = floor(t*n), next = (prev+1) % n) -- there is no skeleton.
LICENCE: GPL-3.0 (format knowledge derived from the MegaGlest sources, GPL-3.0).
"""
import struct
from dataclasses import dataclass, field

import numpy as np

PROP_CUSTOM_COLOR, PROP_TWO_SIDED, PROP_NO_SELECT, PROP_GLOW = 1, 2, 4, 8


@dataclass
class G3DMesh:
    name: str
    vertices: np.ndarray            # (frames, n, 3) float32
    normals: np.ndarray             # (frames, n, 3) float32
    uv: np.ndarray | None           # (n, 2) float32, None if untextured
    indices: np.ndarray             # (m,) uint32
    diffuse: tuple = (1.0, 1.0, 1.0)
    specular: tuple = (0.0, 0.0, 0.0)
    specular_power: float = 0.0
    opacity: float = 1.0
    custom_color: bool = False
    two_sided: bool = False
    no_select: bool = False
    glow: bool = False
    diffuse_texture: str | None = None
    specular_texture: str | None = None
    normal_texture: str | None = None

    @property
    def frame_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3


@dataclass
class G3DModel:
    version: int
    meshes: list = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return max((m.frame_count for m in self.meshes), default=0)


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1")


class _R:
    def __init__(self, data: bytes):
        self.d, self.p = data, 0

    def take(self, n: int) -> bytes:
        if self.p + n > len(self.d):
            raise ValueError("g3d file is truncated")
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def array(self, dtype, count):
        return np.frombuffer(self.take(count * np.dtype(dtype).itemsize), dtype).copy()


def parse_g3d(data: bytes) -> G3DModel:
    r = _R(data)
    magic, version = r.unpack("<3sB")
    if magic != b"G3D" or version not in (3, 4):
        raise ValueError(f"not a G3D v3/v4 file (magic {magic!r}, version {version})")
    model = G3DModel(version)
    if version == 4:
        count, mtype = r.unpack("<HB")
        if mtype != 0:
            raise ValueError(f"unsupported G3D model type {mtype}")
    else:
        (count,) = r.unpack("<I")
    for _ in range(count):
        if version == 4:
            name = _cstr(r.take(64))
            frames, nv, ni = r.unpack("<3I")
            diffuse = r.unpack("<3f"); specular = r.unpack("<3f")
            spow, opacity, props, tflags = r.unpack("<2f2I")
            tex = {}
            for bit, key in ((1, "diffuse"), (2, "specular"), (4, "normal")):
                if tflags & bit:
                    tex[key] = _cstr(r.take(64))
            verts = r.array("<f4", frames * nv * 3).reshape(frames, nv, 3)
            norms = r.array("<f4", frames * nv * 3).reshape(frames, nv, 3)
            uv = r.array("<f4", nv * 2).reshape(nv, 2) if tflags else None
            idx = r.array("<u4", ni)
            model.meshes.append(G3DMesh(
                name, verts, norms, uv, idx, diffuse, specular, spow, opacity,
                bool(props & PROP_CUSTOM_COLOR), bool(props & PROP_TWO_SIDED), bool(props & PROP_NO_SELECT),
                bool(props & PROP_GLOW), tex.get("diffuse"), tex.get("specular"), tex.get("normal")))
        else:
            frames, nframes, uvframes, colframes, nv, ni, props = r.unpack("<7I")
            texname = _cstr(r.take(64))
            textured = not (props & 1)
            verts = r.array("<f4", frames * nv * 3).reshape(frames, nv, 3)
            norms = r.array("<f4", nframes * nv * 3).reshape(nframes, nv, 3)
            uv = r.array("<f4", uvframes * nv * 2)[:nv * 2].reshape(nv, 2)
            r.array("<f4", colframes * 4)
            idx = r.array("<u4", ni)
            model.meshes.append(G3DMesh(
                f"mesh{len(model.meshes)}", verts, norms, uv if textured else None, idx,
                two_sided=bool(props & 2), custom_color=bool(props & 4),
                diffuse_texture=texname if textured else None))
    return model


def write_g3d_v4(model: G3DModel) -> bytes:
    out = bytearray(b"G3D\x04" + struct.pack("<HB", len(model.meshes), 0))
    for m in model.meshes:
        props = (PROP_CUSTOM_COLOR * m.custom_color) | (PROP_TWO_SIDED * m.two_sided) \
            | (PROP_NO_SELECT * m.no_select) | (PROP_GLOW * m.glow)
        names = [(1, m.diffuse_texture), (2, m.specular_texture), (4, m.normal_texture)]
        tflags = sum(bit for bit, n in names if n)
        out += m.name.encode("latin-1")[:63].ljust(64, b"\0")
        out += struct.pack("<3I3f3f2f2I", m.frame_count, m.vertices.shape[1], len(m.indices), *m.diffuse,
                           *m.specular, m.specular_power, m.opacity, props, tflags)
        for bit, n in names:
            if n:
                out += n.encode("latin-1")[:63].ljust(64, b"\0")
        out += m.vertices.astype("<f4").tobytes() + m.normals.astype("<f4").tobytes()
        if tflags:
            out += m.uv.astype("<f4").tobytes()
        out += m.indices.astype("<u4").tobytes()
    return bytes(out)
