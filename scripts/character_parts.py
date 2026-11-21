"""The custom character built on the MegaGlest Behemoth: new parts that ride on the base model's animation frames.
LICENCE: GPL-3.0 (base model: MegaGlest, CC BY-SA 3.0).

Every new part is built once in the Behemoth's rest pose and then glued to a group of the base model's own vertices ("anchors").
For each animation frame the rigid move of that group (Kabsch fit, rest -> frame) is applied to the part, so the head spikes,
collar, chains, armour, shoulder guards and ankle bands follow any animation the base has, with no extra animation work.
The head is made a little bigger by growing its vertices around the neck, again frame by frame.

Parts (all UVs in the viewer's top-down convention, tiled on their own texture):  bone = chin spikes, steel = collar and chains,
plate = chest/back armour and shoulder guards, cuff = ankle bands (copied from the arm bands).
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripts.behemoth_rig import CLUB_VERTICES, behemoth_labels, behemoth_rest
from scripts.retarget import kabsch

HEAD_SCALE = 1.12


# ----------------------------------------------------------------------------------------------------------------- mesh builder
class Mesh:
    """Triangle soup with per-corner UVs (faces are never shared, so each keeps its own flat shading and texture tiling)."""

    def __init__(self):
        self.v, self.uv, self.t = [], [], []

    def quad(self, p0, p1, p2, p3, uv0=(0, 0), uv1=(1, 0), uv2=(1, 1), uv3=(0, 1)):
        n = sum(len(x) for x in self.v)
        self.v.append(np.array([p0, p1, p2, p3], float))
        self.uv.append(np.array([uv0, uv1, uv2, uv3], float))
        self.t.append(np.array([[n, n + 1, n + 2], [n, n + 2, n + 3]]))

    def tri(self, p0, p1, p2, uv0=(0, 0), uv1=(1, 0), uv2=(0.5, 1)):
        n = sum(len(x) for x in self.v)
        self.v.append(np.array([p0, p1, p2], float))
        self.uv.append(np.array([uv0, uv1, uv2], float))
        self.t.append(np.array([[n, n + 1, n + 2]]))

    def arrays(self):
        return np.concatenate(self.v), np.concatenate(self.uv), np.concatenate(self.t)


def basis(axis, hint=(0.0, 1.0, 0.0)):
    """Right-handed orthonormal (u, v, w) with w along `axis`."""
    w = np.asarray(axis, float)
    w = w / np.linalg.norm(w)
    h = np.asarray(hint, float)
    if abs(w @ h) > 0.95:
        h = np.array([1.0, 0.0, 0.0])
    u = np.cross(h, w)
    u /= np.linalg.norm(u)
    return u, np.cross(w, u), w


def add_box(m: Mesh, centre, axes, size, tile=1.0):
    """Box centred at `centre`, local axes (3,3 rows), full size (sx, sy, sz); UVs follow the real size (texture repeats every 1/tile)."""
    c = np.asarray(centre, float)
    ax = np.asarray(axes, float)
    s = np.asarray(size, float) / 2
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for sign in (1, -1):
            n = ax[i] * sign * s[i]
            a, b = ax[j] * s[j], ax[k] * s[k]
            pts = [c + n - a - b, c + n + a - b, c + n + a + b, c + n - a + b]
            if sign < 0:
                pts = pts[::-1]
            w, h = 2 * s[j] * tile, 2 * s[k] * tile
            uv = [(0, 0), (w, 0), (w, h), (0, h)]
            if sign < 0:
                uv = uv[::-1]
            m.quad(*pts, *uv)


def add_cone(m: Mesh, base, direction, length, radius, sides=6, bend=(0, 0, 0), tip_radius=0.0, rings=3, tile=1.0):
    """Tapered spike from `base` along `direction`; `bend` is the sideways offset of the tip (a slight curve)."""
    u, v, w = basis(direction)
    base = np.asarray(base, float)
    bend = np.asarray(bend, float)
    levels = []
    for r in range(rings + 1):
        f = r / rings
        c = base + w * length * f + bend * f * f
        rad = radius * (1 - f) + tip_radius * f
        levels.append([c + rad * (np.cos(a) * u + np.sin(a) * v) for a in np.linspace(0, 2 * np.pi, sides, endpoint=False)])
    for r in range(rings):
        for s in range(sides):
            s2 = (s + 1) % sides
            f0, f1 = r / rings * length * tile, (r + 1) / rings * length * tile
            if tip_radius == 0 and r == rings - 1:
                m.tri(levels[r][s], levels[r][s2], levels[r + 1][0] * 0 + (base + w * length + bend), (s / sides * 1.0, f0), ((s + 1) / sides, f0), ((s + .5) / sides, f1))
            else:
                m.quad(levels[r][s], levels[r][s2], levels[r + 1][s2], levels[r + 1][s], (s / sides, f0), ((s + 1) / sides, f0), ((s + 1) / sides, f1), (s / sides, f1))
    cap = levels[0]
    for s in range(sides):
        m.tri(cap[(s + 1) % sides], cap[s], base, (0, 0), (1, 0), (0.5, 1))


def add_ring(m: Mesh, centre, axes, radii_out, radii_in, height, sides=8, offset=0.0, tile=1.0, tilt=(0.0, 0.0)):
    """Angular (n-sided) collar: elliptic ring (outer/inner half-widths along local x, z), `height` along local y."""
    c = np.asarray(centre, float)
    ax = np.asarray(axes, float)
    angs = np.linspace(0, 2 * np.pi, sides, endpoint=False) + offset

    def pt(rx, rz, y, a):
        return c + ax[0] * rx * np.cos(a) + ax[2] * rz * np.sin(a) + ax[1] * y

    for s in range(sides):
        a0, a1 = angs[s], angs[(s + 1) % sides]
        u0, u1 = s / sides * tile * 4, (s + 1) / sides * tile * 4
        ho = height / 2
        # outer wall (slightly flared), top, inner wall, bottom
        m.quad(pt(*radii_out, -ho, a0), pt(*radii_out, -ho, a1), pt(*radii_out, ho, a1), pt(*radii_out, ho, a0), (u0, 0), (u1, 0), (u1, 1), (u0, 1))
        m.quad(pt(*radii_out, ho, a0), pt(*radii_out, ho, a1), pt(*radii_in, ho, a1), pt(*radii_in, ho, a0), (u0, 0), (u1, 0), (u1, .4), (u0, .4))
        m.quad(pt(*radii_in, ho, a0), pt(*radii_in, ho, a1), pt(*radii_in, -ho, a1), pt(*radii_in, -ho, a0), (u0, 0), (u1, 0), (u1, 1), (u0, 1))
        m.quad(pt(*radii_in, -ho, a0), pt(*radii_in, -ho, a1), pt(*radii_out, -ho, a1), pt(*radii_out, -ho, a0), (u0, 0), (u1, 0), (u1, .4), (u0, .4))


def add_dome(m: Mesh, centre, axes, radii, rings=3, sides=8, tile=1.0):
    """Upper half of an ellipsoid (a shoulder guard), local y up."""
    c = np.asarray(centre, float)
    ax = np.asarray(axes, float)
    levels = []
    for r in range(rings + 1):
        el = (np.pi / 2) * r / rings
        ring = []
        for a in np.linspace(0, 2 * np.pi, sides, endpoint=False):
            ring.append(c + ax[0] * radii[0] * np.cos(el) * np.cos(a) + ax[2] * radii[2] * np.cos(el) * np.sin(a) + ax[1] * radii[1] * np.sin(el))
        levels.append(ring)
    for r in range(rings):
        for s in range(sides):
            s2 = (s + 1) % sides
            uu0, uu1 = s / sides * 2 * tile, (s + 1) / sides * 2 * tile
            vv0, vv1 = r / rings * tile, (r + 1) / rings * tile
            if r == rings - 1:
                m.tri(levels[r][s], levels[r][s2], levels[r + 1][0], (uu0, vv0), (uu1, vv0), ((uu0 + uu1) / 2, vv1))
            else:
                m.quad(levels[r][s], levels[r][s2], levels[r + 1][s2], levels[r + 1][s], (uu0, vv0), (uu1, vv0), (uu1, vv1), (uu0, vv1))


def add_band(m: Mesh, centre, axis, rx, rz, height, sides=12, repeat=2.0):
    """Closed straight band around `axis` (an ankle ring); the texture repeats `repeat` times around it. Thin rims top and bottom."""
    u, v, w = basis(axis)
    c = np.asarray(centre, float)
    angs = np.linspace(0, 2 * np.pi, sides, endpoint=False)

    def ring(h, grow):
        return [c + w * h + (rx + grow) * np.cos(a) * u + (rz + grow) * np.sin(a) * v for a in angs]

    lo, hi = ring(-height / 2, 0.0), ring(height / 2, 0.0)
    lo_in, hi_in = ring(-height / 2, -0.05), ring(height / 2, -0.05)
    for s in range(sides):
        s2 = (s + 1) % sides
        u0, u1 = s / sides * repeat, (s + 1) / sides * repeat
        m.quad(lo[s], lo[s2], hi[s2], hi[s], (u0, 1), (u1, 1), (u1, 0), (u0, 0))
        m.quad(hi[s], hi[s2], hi_in[s2], hi_in[s], (u0, 0), (u1, 0), (u1, .1), (u0, .1))
        m.quad(lo_in[s], lo_in[s2], lo[s2], lo[s], (u0, .9), (u1, .9), (u1, 1), (u0, 1))


def add_link(m: Mesh, centre, tangent, face, length=0.2, width=0.12, bar=0.035):
    """One open chain link (a rectangular ring of four square bars) lying along `tangent`; `face` is the direction its flat side faces."""
    t = np.asarray(tangent, float)
    t = t / np.linalg.norm(t)
    f = np.asarray(face, float)
    f = f - t * (f @ t)
    f /= np.linalg.norm(f)
    side = np.cross(t, f)            # the link's wide direction; the flat side faces f
    c = np.asarray(centre, float)
    axes = np.array([side, f, t])
    for sgn in (-1, 1):
        add_box(m, c + side * sgn * (width / 2 - bar / 2), axes, (bar, bar, length), tile=6)
        add_box(m, c + t * sgn * (length / 2 - bar / 2), axes, (width, bar, bar), tile=6)


# ---------------------------------------------------------------------------------------------------------------- surface helpers
def surface_z(verts, tris, x, y, front=True):
    """Front-most (or rearmost) z of the triangle mesh at (x, y); NaN if the point is outside."""
    best = np.nan
    for a, b, c in tris:
        pa, pb, pc = verts[a], verts[b], verts[c]
        d = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(d) < 1e-9:
            continue
        w0 = ((pb[1] - pc[1]) * (x - pc[0]) + (pc[0] - pb[0]) * (y - pc[1])) / d
        w1 = ((pc[1] - pa[1]) * (x - pc[0]) + (pa[0] - pc[0]) * (y - pc[1])) / d
        w2 = 1 - w0 - w1
        if min(w0, w1, w2) < -1e-9:
            continue
        z = w0 * pa[2] + w1 * pb[2] + w2 * pc[2]
        if np.isnan(best) or (z > best if front else z < best):
            best = z
    return best


def slice_at(verts, tris, y):
    """Points where the triangle edges cross the horizontal plane at height y -> (k,3)."""
    pts = []
    for a, b, c in tris:
        for i, j in ((a, b), (b, c), (c, a)):
            ya, yb = verts[i][1], verts[j][1]
            if (ya - y) * (yb - y) < 0:
                f = (y - ya) / (yb - ya)
                pts.append(verts[i] + f * (verts[j] - verts[i]))
    return np.array(pts)


def welded_normals(verts, tris, axis_point=None):
    """Outward vertex normals with coincident vertices (UV seams) sharing one normal, so an offset shell has no cracks."""
    key = {}
    group = np.array([key.setdefault(tuple(np.round(p, 3)), len(key)) for p in verts])
    acc = np.zeros((len(key), 3))
    for a, b, c in tris:
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        for i in (a, b, c):
            acc[group[i]] += n
    nrm = acc[group]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    if axis_point is not None:
        centre = np.asarray(axis_point, float)
        out = verts - np.where(np.arange(3) == 1, verts, centre)          # horizontal direction away from the spine at each height
        flip = (nrm * out).sum(1) < 0
        nrm[flip] *= -1
    return nrm


@dataclass
class Part:
    name: str
    texture: str
    v: np.ndarray                  # (n,3) rest pose
    uv: np.ndarray
    t: np.ndarray                  # (m,3)
    anchors: np.ndarray            # indices into the base vertices that this part follows
    blend: tuple | None = None     # optional second anchor group and weight (shoulder guards sit between chest and arm)


# --------------------------------------------------------------------------------------------------------------------- the character
def _rot_between(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    sn = np.linalg.norm(axis)
    return np.zeros(3) if sn < 1e-9 else axis / sn * np.arctan2(sn, a @ b)


class Character:
    """Rest-pose data and the parts; `apply(base_frame)` returns the character's mesh parts for one frame of base vertices."""

    def __init__(self, unit_dir, uv_base: np.ndarray, base_tris: np.ndarray):
        rest, _, _ = behemoth_rest(unit_dir)
        self.base_rest = rest
        self.uv_base = uv_base
        self.base_tris = base_tris
        self.lab = behemoth_labels(rest)
        n = len(rest)
        i = np.arange(n)
        body = i >= CLUB_VERTICES
        # head: everything above the neck in front of the hump (includes the horns), minus the claws of the raised hand
        self.head = np.where(body & (rest[:, 1] > 3.3) & (rest[:, 2] > 0.0) & (rest[:, 2] < 0.85))[0]
        self.neck = np.array([rest[self.head, 0].mean(), 3.3, rest[self.head, 2].mean()])
        self.head_rest = self.scale_head(rest)
        self.torso = np.where(body & (self.lab == 0) & (rest[:, 1] > 2.3) & (rest[:, 1] < 3.33) & (np.abs(rest[:, 0] - 0.05) < 0.68))[0]
        self.parts: list[Part] = []
        self._build()

    # ................................................................................................................ head growth
    def head_weight(self, y):
        t = np.clip((y - 3.3) / 0.25, 0, 1)
        return t * t * (3 - 2 * t)

    def scale_head(self, frame):
        """Base frame with the head grown HEAD_SCALE about the neck (weights fade it out toward the shoulders)."""
        out = frame.copy()
        h = self.head
        piv = self.neck_of(frame)
        k = 1 + (HEAD_SCALE - 1) * self.head_weight(self.base_rest[h, 1])[:, None]
        out[h] = piv + (frame[h] - piv) * k
        return out

    def neck_of(self, frame):
        low = self.head[self.base_rest[self.head, 1] < 3.45]
        c = frame[low].mean(0)
        return np.array([frame[self.head, 0].mean(), c[1], frame[self.head, 2].mean()])

    def head_grown(self, frame):
        """Frame with the head grown: the rest-pose growth, turned by the head's own motion in this frame."""
        h = self.head
        R, _ = kabsch(self.base_rest[h], frame[h])
        out = frame.copy()
        out[h] = frame[h] + (self.head_rest[h] - self.base_rest[h]) @ R.T
        return out

    # ............................................................................................................... part building
    def add(self, name, texture, mesh: Mesh, anchors, blend=None):
        if not mesh.v:
            return
        v, uv, t = mesh.arrays()
        self.parts.append(Part(name, texture, v, uv, t, np.asarray(anchors), blend))

    def _build(self):
        """No invented shapes: the Horned Reaper's own two small jaw spikes (its real triangles and texture patch), and the Troll's mace
        in place of the club. Everything else is painted onto the existing polygons (character_paint.py)."""
        from scripts.character_mace import build_mace
        club = self.base_rest[:CLUB_VERTICES]
        m = build_mace(club)
        v, uv, t = m.arrays()
        self.parts.append(Part("mace", "mace", v, uv, t, np.arange(CLUB_VERTICES), None))

    # .................................................................................................................... per frame
    def apply(self, base_frame: np.ndarray):
        """-> list of (name, texture, verts) for one frame of base vertices (head already grown)."""
        grown = self.head_grown(base_frame)
        out = []
        for p in self.parts:
            R, t = kabsch(self.head_rest[p.anchors], grown[p.anchors])
            out.append((p.name, p.texture, p.v @ R.T + t))
        return grown, out
