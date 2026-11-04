"""Read Dungeon Keeper 2 KMF model files (static meshes, vertex animations, groups).

Format as documented by the community template (ufdada/dk2-tools, Formats/Model/kmf_struct.bt)
and cross-checked against OpenKeeper's loader (tonihele/OpenKeeper, GPL-3.0). This is an
independent implementation written from the format description.

Chunks (all little-endian; each chunk = 4-char tag + u32 size): KMSH > HEAD, MATL, then one of
MESH / ANIM / GROP. Strings are null-terminated.

Animations are NOT skeletal. Each vertex has a list of keyframe positions packed as 10 bits per
axis; per-128-frame block tables (ITAB, VGEO) pick the keyframe pair and the position is
interpolated between them.
"""
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KINDS = {1: "mesh", 2: "anim", 3: "grop"}


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def tag(self, expect: str) -> None:
        got = self.data[self.pos:self.pos + 4].decode("latin-1")
        if got != expect:
            raise ValueError(f"expected chunk {expect}, got {got!r} at byte {self.pos}")
        self.pos += 4

    def u32(self) -> int:
        (v,) = struct.unpack_from("<I", self.data, self.pos)
        self.pos += 4
        return v

    def f32(self) -> float:
        (v,) = struct.unpack_from("<f", self.data, self.pos)
        self.pos += 4
        return v

    def vec3(self) -> tuple[float, float, float]:
        v = struct.unpack_from("<3f", self.data, self.pos)
        self.pos += 12
        return v

    def string(self) -> str:
        end = self.data.index(b"\0", self.pos)
        s = self.data[self.pos:end].decode("latin-1")
        self.pos = end + 1
        return s

    def skip_size(self) -> None:
        self.pos += 4


@dataclass
class Material:
    name: str
    textures: list[str]
    flags: int
    brightness: float
    shininess: float
    environment_map: str


@dataclass
class Vertex:
    uv: tuple[int, int]
    normal: tuple[float, float, float]
    geom: int = 0       # static mesh: index into mesh geometry
    itab_index: int = 0  # animation: row in the ITAB/VGEO tables


@dataclass
class Sprite:
    """One material's worth of triangles. triangles[lod] is a list of (a, b, c) vertex indices."""
    material: int
    triangles: list[list[tuple[int, int, int]]]
    vertices: list[Vertex]


@dataclass
class Mesh:
    name: str
    pos: tuple[float, float, float]
    scale: float
    sprites: list[Sprite]
    geometry: list[tuple[float, float, float]]


@dataclass
class Anim:
    name: str
    frame_count: int
    index_count: int
    pos: tuple[float, float, float]
    cube_scale: float
    scale: float
    wrap: bool
    sprites: list[Sprite]
    itab: np.ndarray        # shape (blocks, index_count), uint32
    geom_xyz: np.ndarray    # shape (n + 1, 3), already multiplied by scale
    geom_frame_base: np.ndarray  # shape (n + 1,)
    offsets: np.ndarray     # shape (index_count, frame_count), uint8


@dataclass
class Kmf:
    version: int
    kind: str
    materials: list[Material] = field(default_factory=list)
    mesh: Mesh | None = None
    anim: Anim | None = None
    groups: list[tuple[str, tuple[float, float, float]]] = field(default_factory=list)
    end: int = 0  # byte position where parsing stopped


def _controls(r: _Reader) -> None:
    r.skip_size()
    count = r.u32()
    r.pos += count * 8


def _sprite_headers(r: _Reader, count: int, lods: int):
    heads = []
    for _ in range(count):
        r.tag("SPHD")
        r.skip_size()
        tri_counts = [r.u32() for _ in range(lods)]
        vert_count = r.u32()
        r.pos += 4  # mmFactor
        heads.append((tri_counts, vert_count))
    return heads


def _triangles(r: _Reader, tri_counts: list[int]):
    out = []
    for n in tri_counts:
        raw = np.frombuffer(r.data, np.uint8, n * 3, r.pos).reshape(n, 3)
        out.append([tuple(int(i) for i in t) for t in raw])
        r.pos += n * 3
    return out


def _parse_mesh(r: _Reader) -> Mesh:
    r.skip_size()
    r.tag("HEAD")
    r.skip_size()
    name = r.string()
    sprs_count, geom_count = r.u32(), r.u32()
    pos = r.vec3()
    scale = r.f32()
    lods = r.u32()
    r.tag("CTRL")
    _controls(r)
    r.tag("SPRS")
    r.skip_size()
    heads = _sprite_headers(r, sprs_count, lods)
    sprites = []
    for tri_counts, vert_count in heads:
        r.tag("SPRS")
        r.skip_size()
        material = r.u32()
        tris = _triangles(r, tri_counts)
        verts = []
        for _ in range(vert_count):
            geom, u, v = struct.unpack_from("<HHH", r.data, r.pos)
            normal = struct.unpack_from("<3f", r.data, r.pos + 6)
            r.pos += 18
            verts.append(Vertex((u, v), normal, geom=geom))
        sprites.append(Sprite(material, tris, verts))
    r.tag("GEOM")
    r.skip_size()
    geometry = [r.vec3() for _ in range(geom_count)]
    return Mesh(name, pos, scale, sprites, geometry)


def _parse_anim(r: _Reader) -> Anim:
    r.skip_size()
    r.tag("HEAD")
    r.skip_size()
    name = r.string()
    sprs_count, frame_count, index_count, geom_count = r.u32(), r.u32(), r.u32(), r.u32()
    wrap = r.u32() == 1
    pos = r.vec3()
    cube_scale, scale = r.f32(), r.f32()
    lods = r.u32()
    r.tag("CTRL")
    _controls(r)
    r.tag("SPRS")
    r.skip_size()
    heads = _sprite_headers(r, sprs_count, lods)
    sprites = []
    for tri_counts, vert_count in heads:
        r.tag("SPRS")
        r.skip_size()
        material = r.u32()
        r.tag("POLY")
        r.skip_size()
        tris = _triangles(r, tri_counts)
        r.tag("VERT")
        r.skip_size()
        verts = []
        for _ in range(vert_count):
            u, v = struct.unpack_from("<HH", r.data, r.pos)
            normal = struct.unpack_from("<3f", r.data, r.pos + 4)
            (itab_index,) = struct.unpack_from("<H", r.data, r.pos + 16)
            r.pos += 18
            verts.append(Vertex((u, v), normal, itab_index=itab_index))
        sprites.append(Sprite(material, tris, verts))
    r.tag("ITAB")
    r.skip_size()
    blocks = (frame_count - 1) // 128 + 1
    itab = np.frombuffer(r.data, "<u4", blocks * index_count, r.pos).reshape(blocks, index_count)
    r.pos += blocks * index_count * 4
    r.tag("GEOM")
    r.skip_size()
    raw = np.frombuffer(r.data, np.uint8, geom_count * 5, r.pos).reshape(geom_count, 5)
    r.pos += geom_count * 5
    packed = raw[:, :4].copy().view("<u4").reshape(-1).astype(np.int64)
    xyz = np.stack([((packed >> 20) & 0x3FF) - 512,
                    ((packed >> 10) & 0x3FF) - 512,
                    (packed & 0x3FF) - 512], axis=1) / 511.0 * scale
    frame_base = raw[:, 4].astype(np.int64)
    # the last keyframe is repeated so interpolation can always look one ahead
    xyz = np.vstack([xyz, xyz[-1:]])
    frame_base = np.append(frame_base, frame_base[-1])
    r.tag("VGEO")
    r.skip_size()
    offsets = np.frombuffer(r.data, np.uint8, index_count * frame_count, r.pos).reshape(
        index_count, frame_count)
    r.pos += index_count * frame_count
    return Anim(name, frame_count, index_count, pos, cube_scale, scale, wrap, sprites,
                itab, xyz, frame_base, offsets)


def parse_kmf(data: bytes) -> Kmf:
    r = _Reader(data)
    r.tag("KMSH")
    r.skip_size()
    version = r.u32()
    r.tag("HEAD")
    r.skip_size()
    fmt = r.u32()
    r.u32()
    if fmt not in KINDS:
        raise ValueError(f"unknown KMF format {fmt}")
    model = Kmf(version, KINDS[fmt])
    if model.kind != "grop":
        r.tag("MATL")
        r.skip_size()
        for _ in range(r.u32()):
            r.tag("MAT2")
            r.skip_size()
            name = r.string()
            textures = [r.string() for _ in range(r.u32())]
            flags = r.u32()
            brightness, shininess = r.f32(), r.f32()
            model.materials.append(Material(name, textures, flags, brightness, shininess,
                                            r.string()))
    section = data[r.pos:r.pos + 4].decode("latin-1")
    r.pos += 4
    if model.kind == "mesh" and section == "MESH":
        model.mesh = _parse_mesh(r)
    elif model.kind == "anim" and section == "ANIM":
        model.anim = _parse_anim(r)
    elif model.kind == "grop" and section == "GROP":
        r.skip_size()
        r.tag("HEAD")
        r.skip_size()
        for _ in range(r.u32()):
            r.tag("ELEM")
            r.skip_size()
            model.groups.append((r.string(), r.vec3()))
    else:
        raise ValueError(f"unexpected section {section!r} for {model.kind} file")
    model.end = r.pos
    return model


def evaluate_anim_frame(anim: Anim, frame: int) -> list[np.ndarray]:
    """Vertex positions (n, 3) per sprite at a frame, interpolating between keyframes."""
    block = frame >> 7
    results = []
    for sprite in anim.sprites:
        idx = np.array([v.itab_index for v in sprite.vertices], dtype=np.int64)
        geom_index = anim.itab[block, idx].astype(np.int64) + anim.offsets[idx, frame]
        base = anim.geom_frame_base[geom_index]
        nxt = anim.geom_frame_base[geom_index + 1]
        span = nxt - base
        factor = np.where(span > 0, ((frame & 0x7F) - base) / np.where(span > 0, span, 1), 0.0)
        factor = np.clip(factor, 0.0, 1.0)[:, None]
        a, b = anim.geom_xyz[geom_index], anim.geom_xyz[geom_index + 1]
        results.append(a + (b - a) * factor)
    return results


def mesh_points(model: Kmf) -> list[np.ndarray]:
    """Per-sprite vertex positions of a static mesh."""
    geom = np.array(model.mesh.geometry)
    return [geom[[v.geom for v in s.vertices]] for s in model.mesh.sprites]


def write_obj(model: Kmf, path: Path, frame: int | None = None) -> None:
    """Write LOD 0 as a Wavefront OBJ (one object per material, DK2 axes: y forward, -z up
    is converted to Y-up as x, -z, y). For animations pass a frame number."""
    if model.kind == "mesh":
        points, sprites = mesh_points(model), model.mesh.sprites
    elif model.kind == "anim":
        points, sprites = evaluate_anim_frame(model.anim, frame or 0), model.anim.sprites
    else:
        raise ValueError("only mesh and anim files can be exported")
    lines, base = ["# Dungeon Keeper 2 model exported by dk2kmf.py (Y-up)"], 1
    for i, (sprite, pts) in enumerate(zip(sprites, points)):
        material = model.materials[sprite.material].name if model.materials else str(i)
        lines.append(f"o part{i}_{material}")
        lines += [f"v {x:.6f} {-z:.6f} {y:.6f}" for x, y, z in pts]
        lines += [f"vt {v.uv[0] / 32768:.6f} {1 - v.uv[1] / 32768:.6f}" for v in sprite.vertices]
        lines += [f"vn {nx:.5f} {-nz:.5f} {ny:.5f}"
                  for nx, ny, nz in (v.normal for v in sprite.vertices)]
        for a, b, c in sprite.triangles[0]:
            lines.append("f " + " ".join(f"{base + k}/{base + k}/{base + k}" for k in (a, b, c)))
        base += len(sprite.vertices)
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: python3 -m scripts.dk2kmf <file.kmf> <out.obj> [frame]")
    write_obj(parse_kmf(Path(sys.argv[1]).read_bytes()), Path(sys.argv[2]),
              int(sys.argv[3]) if len(sys.argv) == 4 else None)
