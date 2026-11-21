"""2D silhouette preview of reaper_spec primitives, to compare against turnaround views."""
import math

import numpy as np
from PIL import Image, ImageDraw

from blender.reaper_spec import Prim


def _project(p, axis: str, w: int, h: int) -> tuple[float, float]:
    horizontal = p[0] if axis == "front" else -p[1]  # side view: character front is +x in the image
    return (w / 2 + horizontal * h, (1.0 - p[2]) * h)


def _extent(vec, axis: str) -> tuple[float, float]:
    return (vec[0], vec[2]) if axis == "front" else (vec[1], vec[2])


def rasterize(prims: list[Prim], axis: str, size_wh: tuple[int, int]) -> np.ndarray:
    """Draw prims (unit height 1.0) as a boolean mask of size (h, w); axis is 'front' or 'side'."""
    w, h = size_wh
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for pr in prims:
        if pr.kind in ("ellipsoid", "box"):
            cx, cy = _project(pr.p, axis, w, h)
            dx, dz = _extent(pr.q, axis)
            box = (cx - dx * h / 2, cy - dz * h / 2, cx + dx * h / 2, cy + dz * h / 2)
            (draw.ellipse if pr.kind == "ellipsoid" else draw.rectangle)(box, fill=255)
        else:
            ax, ay = _project(pr.p, axis, w, h)
            bx, by = _project(pr.q, axis, w, h)
            ra, rb = pr.r[0] * h, pr.r[1] * h
            length = math.hypot(bx - ax, by - ay) or 1.0
            nx, ny = -(by - ay) / length, (bx - ax) / length
            draw.polygon([(ax + nx * ra, ay + ny * ra), (bx + nx * rb, by + ny * rb),
                          (bx - nx * rb, by - ny * rb), (ax - nx * ra, ay - ny * ra)], fill=255)
            for (x, y, r) in ((ax, ay, ra), (bx, by, rb)):
                if r > 0:
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(img) > 0
