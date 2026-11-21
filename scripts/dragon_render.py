"""A small software renderer for reviewing the Dragon (model space or decoded KMF): orthographic camera,
z-buffer, textured or flat triangles, Gouraud lighting, optional back-face culling (the game culls back
faces, so a review render culls too and a wrongly wound triangle shows as a hole).

Points are Y-up; `view(az, el)` turns the model about y by `az` degrees (0 = looking at its front... the
camera looks along -z of the turned model) and tilts it by `el`.

LICENCE: GPL-3.0.
"""
import numpy as np
from PIL import Image


def view(az, el=0.0):
    a, e = np.radians(az), np.radians(el)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ ry


def _normals(P, T):
    n = np.zeros_like(P)
    f = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    for k in range(3):
        np.add.at(n, T[:, k], f)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


class Canvas:
    def __init__(self, size=(512, 512), centre=(0, .5, 0), span=1.2, ss=2, bg=(46, 44, 50)):
        self.w, self.h = size
        self.ss = ss
        W, H = self.w * ss, self.h * ss
        self.col = np.zeros((H, W, 3), np.float32) + np.array(bg, np.float32)
        self.alpha = np.zeros((H, W), np.float32)
        self.z = np.full((H, W), np.inf, np.float32)
        self.centre = np.asarray(centre, float)
        self.span = span          # model units across the image width

    def project(self, Q):
        W, H = self.w * self.ss, self.h * self.ss
        k = W / self.span
        sx = W / 2 + (Q[:, 0] - self.centre[0]) * k
        sy = H / 2 - (Q[:, 1] - self.centre[1]) * k
        return sx, sy, Q[:, 2]

    def draw(self, P, T, rot, *, uv=None, tex=None, colour=(150, 140, 70), normals=None, cull=True,
             light=(0.45, 0.75, 0.55), ambient=0.42, flat_colours=None):
        """P (N,3) Y-up points, T (M,3) counter-clockwise-outward triangles (model winding)."""
        P = np.asarray(P, float)
        T = np.asarray(T, int)
        Q = P @ rot.T
        if normals is None:
            normals = _normals(P, T)
        Nq = normals @ rot.T
        L = np.asarray(light, float)
        L = L / np.linalg.norm(L)
        shade = ambient + (1 - ambient) * np.clip(Nq @ L, 0, 1)
        shade += 0.18 * np.clip(Nq @ np.array([-0.6, 0.2, -0.75]), 0, 1)   # rim from behind
        sx, sy, sz = self.project(Q)
        depth = -sz         # camera looks along -z: nearer = larger z
        Hh, Ww = self.z.shape
        if tex is not None:
            tex = np.asarray(tex, np.float32)
            th, tw = tex.shape[:2]
        for ti, (a, b, c) in enumerate(T):
            x = sx[[a, b, c]]
            y = sy[[a, b, c]]
            area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            # screen y points down: a counter-clockwise (outward, facing us) triangle has area < 0
            if cull and area >= 0:
                continue
            if abs(area) < 1e-9:
                continue
            x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()) + 1, Ww))
            y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()) + 1, Hh))
            if x0 >= x1 or y0 >= y1:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
            w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            zz = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
            zb = self.z[y0:y1, x0:x1]
            vis = inside & (zz < zb)
            if not vis.any():
                continue
            s = w0 * shade[a] + w1 * shade[b] + w2 * shade[c]
            if tex is not None:
                u = w0 * uv[a, 0] + w1 * uv[b, 0] + w2 * uv[c, 0]
                v = w0 * uv[a, 1] + w1 * uv[b, 1] + w2 * uv[c, 1]
                px = np.clip((u * tw).astype(int), 0, tw - 1)
                py = np.clip((v * th).astype(int), 0, th - 1)
                rgb = tex[py, px, :3]
            elif flat_colours is not None:
                rgb = np.broadcast_to(np.asarray(flat_colours[ti], np.float32), zz.shape + (3,))
            else:
                rgb = np.broadcast_to(np.asarray(colour, np.float32), zz.shape + (3,))
            out = rgb * s[..., None]
            zb[vis] = zz[vis]
            self.col[y0:y1, x0:x1][vis] = out[vis]
            self.alpha[y0:y1, x0:x1][vis] = 1

    def image(self, alpha=False):
        img = Image.fromarray(np.clip(self.col, 0, 255).astype(np.uint8))
        if alpha:
            a = Image.fromarray((self.alpha * 255).astype(np.uint8))
            img = img.convert("RGBA")
            img.putalpha(a)
        if self.ss > 1:
            img = img.resize((self.w, self.h), Image.LANCZOS)
        return img
