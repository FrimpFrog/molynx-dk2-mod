"""Write Dungeon Keeper 2 KMF files and encode vertex animations the way the game's own files are.

`write_kmf(parse_kmf(b)) == b` for every one of the game's 2015 .kmf files. `encode_anim` turns
per-frame vertex positions into an `Anim`; `anim_invariants` checks an `Anim` against the format
rules (independently of the encoder).

Chunk layout (see scripts/dk2kmf.py for the field lists). Every chunk is tag + u32 size, and the
size counts the 8-byte chunk header too. In ANIM files the outer SPRS chunk also encloses ITAB,
GEOM and VGEO. KMSH HEAD is always (format, 1) and the version always 17.

Conventions of the game's 1145 animation files (measured over all of them; the encoder follows
each one):

* Rows. A vertex's `itab_index` names a row; all vertices at one position (across sprites, with
  their own uv and normal) share the row, so a row is one track of positions. Rows are not shared
  further: no two rows point at the same keyframes.
* Blocks. Frames are cut into blocks of 128. Block k of row r owns the keyframe list
  GEOM[itab[k, r] : next start]; the lists are stored block-major (all rows of block 0 in row
  order, then block 1, ...) back to back, with no gaps.
* Keyframe frame numbers (the GEOM frame-base byte) are local to the block, strictly increasing,
  start at 0 and end at the block's last frame. A block that is not the last ends on local frame
  128, a keyframe holding the next block's frame 0, so the next block's first keyframe repeats it
  exactly. The last block k ends on local frame frame_count - 1 - 128 * k.
* VGEO offset of frame f is the index of the last keyframe at or before f (the keyframe itself
  when f is one), so the last frame of the animation points at the final keyframe.
* Two families of files exist side by side, often for the same creature: fully sampled (every
  frame is a keyframe for every row, 484 files including 17 of 21 Bile Demon ones) and reduced
  (661 files, a row keeps only the keyframes it needs; a row that never moves keeps exactly two
  keyframes per block, its first and its last). The game's reduction tolerance cannot be
  recovered; this encoder reduces with a tolerance of one quantisation step (`reduce=True`) or
  samples every frame (`reduce=False`).
* Quantisation. Positions are absolute (not relative to `pos`): q = round(x / scale * 511) + 512
  per axis in 10 bits. `scale` is the largest |coordinate| over all keyframes (1142 files), so
  some coordinate reaches +-511; `pos` is the centre of the bounding box of all keyframes (1143
  files); `cube_scale` is the largest distance of a keyframe position from `pos` (1064 files; the
  rest were measured on data we do not have). The two top bits of each packed u32 are 0.
* Looping. Every `wrap` animation (209 files) ends on a copy of its first frame.
* No game animation has fewer than 2 frames; "pose" files are static meshes. A 1-frame
  animation is still valid (one keyframe per row).
* CTRL entries are pairs of u32; in creature animations the first appears to be
  (vertex << 16 | sprite) and the second a control-point id (1000.., 2000.., 3000.., 5000);
  they are copied through unchanged.

Static meshes (written as given): `pos` is the bounding-box centre of GEOM and `scale` the largest
distance of a GEOM point from `pos` (862 of 862 files).

LICENCE: GPL-3.0.
"""
import struct

import numpy as np

from scripts.dk2kmf import Anim, Kmf, Material, Mesh, Sprite

_FORMATS = {"mesh": 1, "anim": 2, "grop": 3}
_BLOCK = 128
_LODS = 16


def _chunk(tag: str, payload: bytes) -> bytes:
    return tag.encode("latin-1") + struct.pack("<I", len(payload) + 8) + payload


def _cstr(s: str) -> bytes:
    return s.encode("latin-1") + b"\0"


def _u32(*v: int) -> bytes:
    return struct.pack(f"<{len(v)}I", *v)


def _f32(*v: float) -> bytes:
    return struct.pack(f"<{len(v)}f", *v)


def _materials(materials: list[Material]) -> bytes:
    body = _u32(len(materials))
    for m in materials:
        body += _chunk("MAT2", _cstr(m.name) + _u32(len(m.textures))
                       + b"".join(_cstr(t) for t in m.textures) + _u32(m.flags)
                       + _f32(m.brightness, m.shininess) + _cstr(m.environment_map))
    return _chunk("MATL", body)


def _controls(raw: bytes) -> bytes:
    if len(raw) % 8:
        raise ValueError(f"controls must be whole (u32, u32) pairs, got {len(raw)} bytes")
    return _chunk("CTRL", _u32(len(raw) // 8) + raw)


def _lod_count(sprites: list[Sprite]) -> int:
    counts = {len(s.triangles) for s in sprites}
    if len(counts) > 1:
        raise ValueError(f"sprites disagree on the number of LODs: {sorted(counts)}")
    return counts.pop() if counts else _LODS


def _sprite_headers(sprites: list[Sprite]) -> bytes:
    return b"".join(_chunk("SPHD", _u32(*(len(t) for t in s.triangles), len(s.vertices))
                           + _f32(s.mm_factor)) for s in sprites)


def _triangles(sprite: Sprite) -> bytes:
    return bytes(i for lod in sprite.triangles for tri in lod for i in tri)


def _mesh(mesh: Mesh) -> bytes:
    head = _chunk("HEAD", _cstr(mesh.name) + _u32(len(mesh.sprites), len(mesh.geometry))
                  + _f32(*mesh.pos, mesh.scale) + _u32(_lod_count(mesh.sprites)))
    sprs = _sprite_headers(mesh.sprites)
    for s in mesh.sprites:
        verts = b"".join(struct.pack("<HHH3f", v.geom, *v.uv, *v.normal) for v in s.vertices)
        sprs += _chunk("SPRS", _u32(s.material) + _triangles(s) + verts)
    geom = b"".join(_f32(*p) for p in mesh.geometry)
    return _chunk("MESH", head + _controls(mesh.controls) + _chunk("SPRS", sprs)
                  + _chunk("GEOM", geom))


def _quantise(xyz: np.ndarray, scale: float) -> np.ndarray:
    if scale == 0:
        return np.full(xyz.shape, 512, dtype=np.int64)
    q = np.rint(np.asarray(xyz, dtype=np.float64) / scale * 511.0).astype(np.int64) + 512
    if q.min() < 0 or q.max() > 1023:
        raise ValueError("animation positions exceed the quantisation box (scale too small)")
    return q


def _anim(anim: Anim) -> bytes:
    n = len(anim.geom_xyz) - 1  # the reader repeats the final keyframe; it is not stored
    head = _chunk("HEAD", _cstr(anim.name) + _u32(len(anim.sprites), anim.frame_count,
                                                  anim.index_count, n, 1 if anim.wrap else 0)
                  + _f32(*anim.pos, anim.cube_scale, anim.scale) + _u32(_lod_count(anim.sprites)))
    sprs = _sprite_headers(anim.sprites)
    for s in anim.sprites:
        verts = b"".join(struct.pack("<HH3fH", *v.uv, *v.normal, v.itab_index) for v in s.vertices)
        sprs += _chunk("SPRS", _u32(s.material) + _chunk("POLY", _triangles(s))
                       + _chunk("VERT", verts))
    q = _quantise(anim.geom_xyz[:n], anim.scale)
    packed = (q[:, 0] << 20) | (q[:, 1] << 10) | q[:, 2]
    geom = np.zeros((n, 5), dtype=np.uint8)
    geom[:, :4] = packed.astype("<u4").view(np.uint8).reshape(n, 4)
    geom[:, 4] = np.asarray(anim.geom_frame_base[:n], dtype=np.uint8)
    sprs += _chunk("ITAB", np.asarray(anim.itab, dtype="<u4").tobytes())
    sprs += _chunk("GEOM", geom.tobytes())
    sprs += _chunk("VGEO", np.asarray(anim.offsets, dtype=np.uint8).tobytes())
    return _chunk("ANIM", head + _controls(anim.controls) + _chunk("SPRS", sprs))


def _groups(groups) -> bytes:
    body = _chunk("HEAD", _u32(len(groups)))
    body += b"".join(_chunk("ELEM", _cstr(name) + _f32(*pos)) for name, pos in groups)
    return _chunk("GROP", body)


def write_kmf(model: Kmf) -> bytes:
    """Serialise a parsed or freshly built `Kmf` to the game's byte layout."""
    if model.kind not in _FORMATS:
        raise ValueError(f"unknown KMF kind {model.kind!r}")
    body = _u32(model.version) + _chunk("HEAD", _u32(_FORMATS[model.kind], 1))
    if model.kind == "mesh":
        body += _materials(model.materials) + _mesh(model.mesh)
    elif model.kind == "anim":
        body += _materials(model.materials) + _anim(model.anim)
    else:
        body += _groups(model.groups)
    return _chunk("KMSH", body)


# ---------------------------------------------------------------------------- encoder

def _f32_up(v: float) -> float:
    """The smallest float32 >= v, so the stored scale still covers every coordinate."""
    f = np.float32(v)
    if float(f) < v:
        f = np.nextafter(f, np.float32(np.inf))
    return float(f)


def _row_tracks(sprites: list[Sprite], frames) -> np.ndarray:
    """(rows, frames, 3) positions, one track per itab row, checked for consistency."""
    if not frames:
        raise ValueError("an animation needs at least one frame")
    rows_of = [np.array([v.itab_index for v in s.vertices], dtype=np.int64) for s in sprites]
    for f, frame in enumerate(frames):
        if len(frame) != len(sprites):
            raise ValueError(f"frame {f} has {len(frame)} sprites, expected {len(sprites)} "
                             "(shape mismatch)")
        for s, (pts, rows) in enumerate(zip(frame, rows_of)):
            if np.shape(pts) != (len(rows), 3):
                raise ValueError(f"frame {f} sprite {s} has shape {np.shape(pts)}, "
                                 f"expected {(len(rows), 3)}")
    every = np.concatenate(rows_of) if rows_of else np.zeros(0, np.int64)
    if not len(every):
        raise ValueError("an animation needs at least one vertex")
    count = int(every.max()) + 1
    if count > 0x10000:
        raise ValueError(f"{count} rows do not fit the 16-bit itab_index")
    unused = sorted(set(range(count)) - set(every.tolist()))
    if unused:
        raise ValueError(f"row(s) {unused[:10]} are not used by any vertex")
    data = np.stack([np.concatenate([np.asarray(p, dtype=np.float64) for p in frame])
                     for frame in frames], axis=1)  # (vertices, frames, 3)
    tracks = np.zeros((count, len(frames), 3))
    first = np.full(count, -1)
    for i, r in enumerate(every):
        if first[r] < 0:
            first[r] = i
            tracks[r] = data[i]
    tol = 1e-9 + 1e-7 * float(np.abs(data).max())
    clash = np.abs(data - tracks[every]).max(axis=(1, 2)) > tol
    if clash.any():
        r = int(every[np.argmax(clash)])
        raise ValueError(f"vertices on row {r} have different positions")
    return tracks


def _keyframes(deq: np.ndarray, want: np.ndarray, tol: float, reduce: bool) -> list[int]:
    """Local keyframe frames of one row in one block. deq / want are (L + 1, 3): the quantised
    samples and the exact input. Greedy: from each keyframe jump to the furthest frame whose
    straight segment stays within `tol` of the input at every frame, interpolated exactly as
    the game (and evaluate_anim_frame) does."""
    last = len(deq) - 1
    if not reduce or last == 0:
        return list(range(last + 1))
    keys, i = [0], 0
    while i < last:
        best, width = i + 1, 8
        while True:
            hi = min(last, i + width)
            js = np.arange(i + 1, hi + 1)                           # candidate ends
            ts = np.arange(i, hi + 1)                               # frames covered
            span = (js - i)[:, None]
            fac = np.clip((ts[None, :] - i) / span, 0.0, 1.0)       # (cands, frames)
            seg = deq[i] + (deq[js] - deq[i])[:, None, :] * fac[:, :, None]
            err = np.abs(seg - want[ts][None, :, :]).max(axis=2)
            err = np.where(ts[None, :] <= js[:, None], err, 0.0).max(axis=1)
            ok = err <= tol
            bad = np.flatnonzero(~ok)
            if bad.size:  # stop at the first end that fails (i + 1 always holds: it only
                # covers two quantised samples, each within half a step)
                best = int(js[bad[0] - 1]) if bad[0] > 0 else i + 1
                break
            best = int(js[-1])
            if hi == last:
                break
            width *= 2
        keys.append(best)
        i = best
    return keys


def encode_anim(name: str, sprites: list[Sprite], frames: list[list[np.ndarray]], wrap: bool,
                controls: bytes, *, reduce: bool = True) -> Anim:
    """Encode `frames[f][s]` ((n_s, 3) positions of sprite s at frame f) as a game animation.

    Vertices that share an `itab_index` must have the same positions (they become one row).
    `pos`, `scale` and `cube_scale` are chosen from the data as the game does; every decoded
    coordinate is within one quantisation step (scale / 511) of the input. With `reduce`
    (default) each row keeps only the keyframes it needs per 128-frame block; with
    reduce=False every frame is a keyframe, like the game's fully sampled files. A `wrap`
    animation must end on a copy of its first frame, as all of the game's looping ones do."""
    if len(controls) % 8:
        raise ValueError(f"controls must be whole (u32, u32) pairs, got {len(controls)} bytes")
    tracks = _row_tracks(sprites, frames)
    rows, count = tracks.shape[0], tracks.shape[1]
    pts = tracks.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2
    scale = _f32_up(float(np.abs(pts).max())) or 1.0
    step = scale / 511.0
    if wrap and np.abs(tracks[:, -1] - tracks[:, 0]).max() > step * 1.01:
        raise ValueError("a wrap animation must end on a copy of its first frame")
    q = _quantise(tracks, scale)
    deq = (q - 512) / 511.0 * scale  # exactly what the reader computes
    blocks = (count - 1) // _BLOCK + 1
    itab = np.zeros((blocks, rows), dtype=np.uint32)
    offsets = np.zeros((rows, count), dtype=np.uint8)
    xyz, base = [], []
    total = 0
    for k in range(blocks):
        f0 = k * _BLOCK
        f1 = min(count - 1, f0 + _BLOCK)  # inclusive: non-last blocks end on the next frame 0
        local = np.arange(min(_BLOCK, count - f0))
        for r in range(rows):
            keys = _keyframes(deq[r, f0:f1 + 1], tracks[r, f0:f1 + 1], step, reduce)
            itab[k, r] = total
            total += len(keys)
            xyz.append(deq[r, f0 + np.array(keys)])
            base.append(np.array(keys))
            offsets[r, f0:f0 + len(local)] = np.searchsorted(keys, local, side="right") - 1
    xyz = np.vstack(xyz)
    base = np.concatenate(base).astype(np.int64)
    pos = tuple(float(np.float32(c)) for c in centre)
    cube = float(np.float32(np.linalg.norm(pts - centre, axis=1).max()))
    return Anim(name, count, rows, pos, cube, scale, bool(wrap), list(sprites), itab,
                np.vstack([xyz, xyz[-1:]]), np.append(base, base[-1]), offsets, bytes(controls))


# ---------------------------------------------------------------------------- invariants

def anim_invariants(anim: Anim) -> list[str]:
    """Format problems of an animation; empty when the game's rules all hold."""
    out: list[str] = []
    frames, rows = anim.frame_count, anim.index_count
    if frames < 1:
        return [f"frame_count is {frames}"]
    blocks = (frames - 1) // _BLOCK + 1
    itab, offsets = np.asarray(anim.itab), np.asarray(anim.offsets)
    if itab.shape != (blocks, rows):
        return [f"itab has shape {itab.shape}, expected {(blocks, rows)}"]
    if offsets.shape != (rows, frames):
        return [f"offsets have shape {offsets.shape}, expected {(rows, frames)}"]
    n = len(anim.geom_xyz) - 1
    if n < 1 or len(anim.geom_frame_base) != n + 1:
        return [f"{n} keyframes with {len(anim.geom_frame_base) - 1} frame numbers"]
    for s, sprite in enumerate(anim.sprites):
        for i, v in enumerate(sprite.vertices):
            if not 0 <= v.itab_index < rows:
                out.append(f"sprite {s} vertex {i}: itab_index {v.itab_index} not below {rows}")
        for lod, tris in enumerate(sprite.triangles):
            if any(max(t) >= len(sprite.vertices) for t in tris):
                out.append(f"sprite {s} lod {lod}: triangle index past its vertices")
    it = itab.astype(np.int64)
    for k, r in zip(*np.nonzero(it >= n)):
        out.append(f"itab[{k}, {r}] = {it[k, r]} is outside the {n} keyframes")
    if out:
        return out
    starts = np.unique(it)
    ends = np.append(starts[1:], n)
    fbase = np.asarray(anim.geom_frame_base[:n], dtype=np.int64)
    for k in range(blocks):
        f0 = k * _BLOCK
        width = min(_BLOCK, frames - f0)
        last = _BLOCK if k < blocks - 1 else frames - 1 - f0
        local = np.arange(width)
        for r in range(rows):
            s = it[k, r]
            e = ends[np.searchsorted(starts, s)]
            bases = fbase[s:e]
            where = f"block {k} row {r}"
            if bases[0] != 0:
                out.append(f"{where}: first keyframe is frame {bases[0]}, not 0")
            if (np.diff(bases) <= 0).any():
                out.append(f"{where}: keyframe frame numbers do not increase {bases.tolist()}")
            if bases[-1] != last:
                out.append(f"{where}: keyframes end at frame {bases[-1]}, not the block's "
                           f"last frame {last}")
            offs = offsets[r, f0:f0 + width].astype(np.int64)
            if (offs >= len(bases)).any():
                out.append(f"{where}: offset {offs.max()} past the row's {len(bases)} keyframes")
                continue
            nxt = offs + 1
            has_next = nxt < len(bases)
            after = bases[np.minimum(nxt, len(bases) - 1)]
            ok = (bases[offs] <= local) & np.where(has_next, local <= after,
                                                   bases[offs] == local)
            if not ok.all():
                f = int(local[np.argmin(ok)])
                out.append(f"{where}: offset at frame {f0 + f} does not bracket the frame")
    return out
