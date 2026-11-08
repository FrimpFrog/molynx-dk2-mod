"""A small kit for building DK2-style low-poly creatures from scratch: a skeleton, parts made of lofts,
cones and fins with skin weights and island UVs, linear blend skinning, poses, and the bake that paints
skins by tracing every texel back to the surface it covers.

It is the Dragon's toolset (dragon_model, dragon_pose, dragon_texture) without the Dragon in it: every
function takes the Skeleton it works on.

Model space: Y up, Z forward, X to the creature's LEFT (right-handed); triangles counter-clockwise seen
from outside (the KMF export mirrors x, which gives the game's winding).

LICENCE: GPL-3.0.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def mirror_name(n):
    return n[:-1] + "R" if n and n.endswith("L") else n


class Skeleton:
    """bones: [(name, parent, head, tail)] for the left side and the middle; ...L bones are mirrored to R."""

    def __init__(self, bones_left):
        out = []
        for name, parent, head, tail in bones_left:
            out.append((name, parent, np.array(head, float), np.array(tail, float)))
            if name.endswith("L"):
                m = np.array([-1.0, 1, 1])
                out.append((mirror_name(name), mirror_name(parent), np.array(head) * m, np.array(tail) * m))
        self.names = [b[0] for b in out]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parent = [None if b[1] is None else self.index[b[1]] for b in out]
        self.head = np.array([b[2] for b in out])
        self.tail = np.array([b[3] for b in out])
        for i, p in enumerate(self.parent):
            assert p is None or p < i, self.names[i]

    def __len__(self):
        return len(self.names)

    def forward(self, local, root_offset=(0, 0, 0)):
        B = len(self)
        Rw = np.zeros((B, 3, 3))
        Pw = np.zeros((B, 3))
        for i in range(B):
            p = self.parent[i]
            if p is None:
                Rw[i] = local[i]
                Pw[i] = self.head[i] + np.asarray(root_offset, float)
            else:
                Rw[i] = Rw[p] @ local[i]
                Pw[i] = Pw[p] + Rw[p] @ (self.head[i] - self.head[p])
        return Rw, Pw

    def weights(self, w):
        if isinstance(w, str):
            return {self.index[w]: 1.0}
        w = {self.index[k] if isinstance(k, str) else k: float(v) for k, v in w.items() if v > 1e-4}
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    def auto(self, p, bones, power=4.0, keep=2):
        def seg(b):
            a, c = self.head[self.index[b]], self.tail[self.index[b]]
            ab = c - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            return max(np.linalg.norm(p - (a + t * ab)), 1e-4)
        d = np.array([seg(b) for b in bones])
        order = np.argsort(d)[:keep]
        return {bones[i]: d[i] ** -power for i in order}


@dataclass
class Part:
    name: str
    texture: str
    skeleton: Skeleton
    points: list = field(default_factory=list)
    uv: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    tris: list = field(default_factory=list)
    island: list = field(default_factory=list)
    tag: list = field(default_factory=list)

    def add(self, p, uv, w, island, **tag):
        self.points.append(np.asarray(p, float))
        self.uv.append(tuple(map(float, uv)))
        self.weights.append(self.skeleton.weights(w))
        self.island.append(island)
        self.tag.append(tag)
        return len(self.points) - 1

    def tri(self, a, b, c, inside=None):
        if inside is not None:
            pa, pb, pc = (self.points[i] for i in (a, b, c))
            n = np.cross(pb - pa, pc - pa)
            if np.dot(n, (pa + pb + pc) / 3 - np.asarray(inside, float)) < 0:
                b, c = c, b
        self.tris.append((a, b, c))

    def tri_facing(self, a, b, c, normal):
        pa, pb, pc = (self.points[i] for i in (a, b, c))
        if np.dot(np.cross(pb - pa, pc - pa), normal) < 0:
            b, c = c, b
        self.tris.append((a, b, c))


def frames(path, hint):
    path = np.asarray(path, float)
    out = []
    for i in range(len(path)):
        a, b = path[max(i - 1, 0)], path[min(i + 1, len(path) - 1)]
        t = unit(b - a)
        h = np.asarray(hint[i] if np.ndim(hint) == 2 else hint, float)
        out.append((t, unit(h - np.dot(h, t) * t)))
    return out


def loft(part, island, path, radii, sides, hint, weights, *, start_point=None, end_point=None, shape=None,
         tag=None, v_range=(0.0, 1.0), u_offset=0.0):
    """Rings round `path` (radii (along hint, lateral)); angle 0 on the hint side; mirrored u (0 at angle 0,
    1 opposite), v along the length. `weights(i, point)`; `shape(i, theta) -> (scale hint, scale lateral)`."""
    path = np.asarray(path, float)
    fr = frames(path, hint)
    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    extra0 = np.linalg.norm(path[0] - start_point) if start_point is not None else 0.0
    extra1 = np.linalg.norm(end_point - path[-1]) if end_point is not None else 0.0
    full = extra0 + seglen[-1] + extra1
    vmap = lambda s: v_range[0] + (v_range[1] - v_range[0]) * s / full
    rings = []
    for i, (c, (t, d)) in enumerate(zip(path, fr)):
        lat = unit(np.cross(t, d))
        ring = []
        for k in range(sides):
            th = 2 * np.pi * k / sides + u_offset
            sd, sl = shape(i, th) if shape else (1.0, 1.0)
            p = c + radii[i][0] * sd * np.cos(th) * d + radii[i][1] * sl * np.sin(th) * lat
            a = (th % (2 * np.pi))
            u = (a if a <= np.pi else 2 * np.pi - a) / np.pi
            ring.append(part.add(p, (u, vmap(extra0 + seglen[i])), weights(i, p), island,
                                 **(tag(i, th) if tag else {})))
        rings.append(ring)
    for i in range(len(rings) - 1):
        mid = (path[i] + path[i + 1]) / 2
        for k in range(sides):
            a, b = rings[i][k], rings[i][(k + 1) % sides]
            c_, d_ = rings[i + 1][k], rings[i + 1][(k + 1) % sides]
            part.tri(a, b, c_, inside=mid)
            part.tri(b, d_, c_, inside=mid)
    if start_point is not None:
        s = part.add(start_point, (0.5, v_range[0]), weights(-1, start_point), island,
                     **(tag(-1, 0.0) if tag else {}))
        inner = path[0] + (path[1] - path[0]) * 0.5
        for k in range(sides):
            part.tri(s, rings[0][k], rings[0][(k + 1) % sides], inside=inner)
    if end_point is not None:
        e = part.add(end_point, (0.5, v_range[1]), weights(len(path), end_point), island,
                     **(tag(len(path), 0.0) if tag else {}))
        inner = path[-1] + (path[-2] - path[-1]) * 0.5
        for k in range(sides):
            part.tri(e, rings[-1][(k + 1) % sides], rings[-1][k], inside=inner)
    return rings


def cone(part, island, base, tip, radius, up, weights, sides=3, tag=None, uv_box=(0, 0, 1, 1)):
    base = np.asarray(base, float)
    t = unit(np.asarray(tip, float) - base)
    d = unit(np.asarray(up, float) - np.dot(up, t) * t)
    lat = np.cross(t, d)
    u0, v0, u1, v1 = uv_box
    ids = []
    for k in range(sides):
        th = 2 * np.pi * k / sides
        p = base + radius * (np.cos(th) * d + np.sin(th) * lat)
        u = (th if th <= np.pi else 2 * np.pi - th) / np.pi
        ids.append(part.add(p, (u0 + (u1 - u0) * u, v0), weights, island, **(tag or {}), along=0.0))
    tp = part.add(tip, ((u0 + u1) / 2, v1), weights, island, **(tag or {}), along=1.0)
    for k in range(sides):
        cen = (part.points[ids[k]] + part.points[ids[(k + 1) % sides]] + part.points[tp]) / 3
        part.tri(ids[k], ids[(k + 1) % sides], tp, inside=base + np.dot(cen - base, t) * t)
    return ids + [tp]


def double_sided(part, island, pts, uvs, weights, tris, normal, **tag):
    """A thin sheet: the same polygon twice, front (facing `normal`) and back. `island` is one name, or a
    (front, back) pair to paint the two sides differently (they then need separate texture areas)."""
    out = []
    islands = (island, island) if isinstance(island, str) else island
    for side, isl in zip((1.0, -1.0), islands):
        ids = [part.add(p, uv, w, isl, **tag, face=side) for p, uv, w in zip(pts, uvs, weights)]
        for a, b, c in tris:
            part.tri_facing(ids[a], ids[b], ids[c], np.asarray(normal(pts[a], pts[b], pts[c]) if callable(normal)
                                                                  else normal, float) * side)
        out.append(ids)
    return out


class Skinner:
    def __init__(self, parts, skeleton):
        self.sk = skeleton
        self.sizes = [len(p.points) for p in parts]
        P = np.concatenate([np.array(p.points) for p in parts])
        W = np.zeros((len(P), len(skeleton)))
        k = 0
        for p in parts:
            for w in p.weights:
                for b, x in w.items():
                    W[k, b] = x
                k += 1
        self.W = W
        self.rel = P[:, None, :] - skeleton.head[None, :, :]

    def __call__(self, Rw, Pw):
        moved = np.einsum("bij,vbj->vbi", Rw, self.rel) + Pw[None]
        out = np.einsum("vb,vbi->vi", self.W, moved)
        return np.split(out, np.cumsum(self.sizes)[:-1])


def euler(rx, ry, rz):
    return Rot.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()


def pose_locals(skeleton, spec):
    """{bone: (rx, ry, rz) degrees} -> local rotations; ...L entries are mirrored to R unless given."""
    local = np.tile(np.eye(3), (len(skeleton), 1, 1))
    full = dict(spec)
    for k, (rx, ry, rz) in spec.items():
        if k.endswith("L") and mirror_name(k) not in spec and not k.startswith("_"):
            full[mirror_name(k)] = (rx, -ry, -rz)
    for k, v in full.items():
        if k in skeleton.index:
            local[skeleton.index[k]] = euler(*v)
    return local


# ------------------------------------------------------------------------------------------- bake

def vertex_normals(P, T):
    n = np.zeros_like(P)
    f = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(n, T[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def atlas_uv(part, atlas, size=128):
    """Island-local (u, v) -> texture UVs from atlas[texture][island] = ((x0, y0, x1, y1), transpose)."""
    uv = np.array(part.uv, float)
    out = np.zeros_like(uv)
    for i, (isl, (u, v)) in enumerate(zip(part.island, uv)):
        (x0, y0, x1, y1), tr = atlas[part.texture][isl]
        a, b = (v, u) if tr else (u, v)
        out[i] = ((x0 + 0.5 + a * (x1 - x0 - 1)) / size, (y0 + 0.5 + b * (y1 - y0 - 1)) / size)
    return out


def bake(parts, atlas, painters, size=128, ss=4, occlusion=None, bleed=24):
    """-> {texture: PIL RGB}: every texel traced to its surface point and painted by
    painters[(texture, island)](samples) with samples u, v, pos, nrm, ao and each vertex tag interpolated."""
    from scripts.dragon_texture import _bleed
    big = size * ss
    layers = {tex: (np.zeros((big, big, 3), np.float32), np.zeros((big, big), bool)) for tex in atlas}
    for part in parts:
        P = np.array(part.points)
        T = np.array(part.tris)
        N = vertex_normals(P, T)
        loc = np.array(part.uv, float)
        tex_uv = atlas_uv(part, atlas, size) * big
        tagkeys = sorted({k for t in part.tag for k, v in t.items() if isinstance(v, (int, float))})
        tags = {k: np.array([float(t.get(k, 0.0)) for t in part.tag]) for k in tagkeys}
        ao = occlusion.get(part.name) if occlusion else None
        col, mask = layers[part.texture]
        for isl in sorted(set(part.island)):
            painter = painters[(part.texture, isl)]
            tris = [t for t in T if part.island[t[0]] == isl]
            S = {k: [] for k in ["u", "v", "pos", "nrm", "ao"] + tagkeys}
            where = []
            for a, b, c in tris:
                x, y = tex_uv[[a, b, c], 0], tex_uv[[a, b, c], 1]
                area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
                if abs(area) < 1e-9:
                    continue
                x0, x1 = int(max(np.floor(x.min()) - 1, 0)), int(min(np.ceil(x.max()) + 1, big))
                y0, y1 = int(max(np.floor(y.min()) - 1, 0)), int(min(np.ceil(y.max()) + 1, big))
                gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
                w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
                w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
                w2 = 1 - w0 - w1
                ins = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
                if not ins.any():
                    continue
                W = np.stack([w0[ins], w1[ins], w2[ins]], 1)
                S["u"].append(W @ loc[[a, b, c], 0])
                S["v"].append(W @ loc[[a, b, c], 1])
                S["pos"].append(W @ P[[a, b, c]])
                n = W @ N[[a, b, c]]
                S["nrm"].append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9))
                S["ao"].append(W @ (ao[[a, b, c]] if ao is not None else np.ones(3)))
                for k in tagkeys:
                    S[k].append(W @ tags[k][[a, b, c]])
                where.append((gy[ins].astype(int), gx[ins].astype(int)))
            if not where:
                continue
            t = {k: np.concatenate(v) for k, v in S.items() if v}
            rgb = painter(t) * (0.5 + 0.5 * t["ao"])[..., None]
            ys = np.concatenate([w[0] for w in where])
            xs = np.concatenate([w[1] for w in where])
            col[ys, xs] = rgb
            mask[ys, xs] = True
    out = {}
    for tex, (col, mask) in layers.items():
        col = _bleed(col, mask, bleed)
        out[tex] = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return out
