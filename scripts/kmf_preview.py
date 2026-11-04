"""Quick software preview of a DK2 KMF model: flat-shaded, one colour per material part.

Usage: python3 -m scripts.kmf_preview <file.kmf> <out.png> [frame]
Draws front, side and three-quarter views side by side. Axes follow write_obj: Y-up from
DK2's (x, y, z) as (x, -z, y), so the model faces -Z in view space.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.dk2kmf import evaluate_anim_frame, mesh_points, parse_kmf

PALETTE = [(214, 70, 52), (150, 150, 160), (230, 190, 70), (90, 170, 220),
           (150, 200, 90), (200, 120, 200), (240, 240, 240), (120, 90, 60)]


def _yup(pts: np.ndarray) -> np.ndarray:
    return np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)


def render_view(parts, tris, azimuth_deg: float, size: int = 512, span=None) -> Image.Image:
    """parts: list of (n,3) Y-up point arrays; tris: matching lists of (a,b,c) index tuples."""
    az = np.radians(azimuth_deg)
    rot = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    allp = np.concatenate(parts) @ rot.T
    lo, hi = allp.min(0), allp.max(0)
    span = span or float(max(hi - lo)) * 1.1
    centre = (lo + hi) / 2
    img = Image.new("RGB", (size, size), (40, 40, 44))
    draw = ImageDraw.Draw(img)
    faces = []
    for i, (pts, tri) in enumerate(zip(parts, tris)):
        p = (pts @ rot.T - centre)
        for a, b, c in tri:
            v = p[[a, b, c]]
            n = np.cross(v[1] - v[0], v[2] - v[0])
            ln = np.linalg.norm(n)
            if ln == 0:
                continue
            shade = 0.35 + 0.65 * abs(float(n[2] / ln))
            faces.append((float(v[:, 2].mean()), v, shade, PALETTE[i % len(PALETTE)]))
    for _, v, shade, colour in sorted(faces, key=lambda f: -f[0]):
        xy = [(size / 2 + x / span * size, size / 2 - y / span * size) for x, y, _ in v]
        fill = tuple(int(c * shade) for c in colour)
        draw.polygon(xy, fill=fill, outline=(15, 15, 15))
    return img


def preview(path: Path, out: Path, frame: int | None = None) -> None:
    model = parse_kmf(Path(path).read_bytes())
    if model.kind == "mesh":
        raw, sprites = mesh_points(model), model.mesh.sprites
    else:
        raw, sprites = evaluate_anim_frame(model.anim, frame or 0), model.anim.sprites
    parts = [_yup(p) for p in raw]
    tris = [s.triangles[0] for s in sprites]
    span = float(max(np.concatenate(parts).max(0) - np.concatenate(parts).min(0))) * 1.1
    views = [render_view(parts, tris, az, span=span) for az in (0, 90, 40)]
    sheet = Image.new("RGB", (512 * 3, 512))
    for i, v in enumerate(views):
        sheet.paste(v, (i * 512, 0))
    sheet.save(out)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: python3 -m scripts.kmf_preview <file.kmf> <out.png> [frame]")
    preview(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
