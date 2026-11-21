"""Software renderer for the high-quality design build (char_full layout: parts with uv/indices, frames = per-part vertex arrays)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _rot(az):
    a = np.radians(az)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

def view_for(clip, az, frames=None):
    rot = _rot(az)
    pts = np.concatenate([np.concatenate(clip["frames"][f]) for f in (frames or range(len(clip["frames"])))]) @ rot.T
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, float(max(hi - lo)) * 1.08

def render(clip, f, textures, az=200.0, size=400, view=None, ss=2, bg=(42, 42, 46)):
    rot = _rot(az)
    verts = [np.asarray(v, float) @ rot.T for v in clip["frames"][f]]
    if view is None:
        allp = np.concatenate(verts); lo, hi = allp.min(0), allp.max(0)
        view = ((lo + hi) / 2, float(max(hi - lo)) * 1.08)
    centre, span = view
    big = size * ss
    col = np.zeros((big, big, 3), np.float32); col[:] = bg
    zb = np.full((big, big), np.inf, np.float32)
    light = np.array([0.35, 0.55, 0.75]); light /= np.linalg.norm(light)
    for part, p in zip(clip["parts"], verts):
        tex = np.asarray(textures[part["texture"]].convert("RGB"), np.float32)
        th, tw = tex.shape[:2]
        tri = np.asarray(part["indices"], int).reshape(-1, 3)
        uv = np.asarray(part["uv"], float)
        fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        vn = np.zeros_like(p)
        for k in range(3):
            np.add.at(vn, tri[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
        v = p - centre
        sx = big / 2 + v[:, 0] / span * big
        sy = big / 2 - v[:, 1] / span * big
        depth = -v[:, 2]
        for a, b, c in tri:
            x0, x1 = int(max(min(sx[[a, b, c]]) - 1, 0)), int(min(max(sx[[a, b, c]]) + 2, big))
            y0, y1 = int(max(min(sy[[a, b, c]]) - 1, 0)), int(min(max(sy[[a, b, c]]) + 2, big))
            if x1 <= x0 or y1 <= y0:
                continue
            den = (sy[b] - sy[c]) * (sx[a] - sx[c]) + (sx[c] - sx[b]) * (sy[a] - sy[c])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            w0 = ((sy[b] - sy[c]) * (gx - sx[c]) + (sx[c] - sx[b]) * (gy - sy[c])) / den
            w1 = ((sy[c] - sy[a]) * (gx - sx[c]) + (sx[a] - sx[c]) * (gy - sy[c])) / den
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not ins.any():
                continue
            z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
            win = ins & (z < zb[y0:y1, x0:x1])
            if not win.any():
                continue
            uu = (w0 * uv[a, 0] + w1 * uv[b, 0] + w2 * uv[c, 0]) % 1.0
            vv = (w0 * uv[a, 1] + w1 * uv[b, 1] + w2 * uv[c, 1]) % 1.0
            texel = tex[np.minimum((vv * th).astype(int), th - 1), np.minimum((uu * tw).astype(int), tw - 1)]
            n = w0[..., None] * vn[a] + w1[..., None] * vn[b] + w2[..., None] * vn[c]
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
            shade = 0.45 + 0.6 * np.clip(np.abs(n @ light), 0, 1)
            reg = col[y0:y1, x0:x1]
            reg[win] = (texel * shade[..., None])[win]
            zb[y0:y1, x0:x1][win] = z[win]
    im = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")
    return im.resize((size, size), Image.LANCZOS) if ss > 1 else im

def label(im, text, size=16):
    d = ImageDraw.Draw(im); f = ImageFont.load_default(size=size)
    d.rectangle([0, im.height - size - 10, im.width, im.height], fill=(24, 24, 28))
    d.text((8, im.height - size - 6), text, fill=(235, 225, 210), font=f)
    return im

def fit_view(verts_list, az, pad=1.08):
    """(centre, span) in the rotated frame for a list of vertex arrays."""
    pts = np.concatenate(verts_list) @ _rot(az).T
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, float(max(hi - lo)) * pad
