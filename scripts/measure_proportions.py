"""Measure Reaper proportions from the front and side turnaround silhouettes.

All values are ratios of the silhouette height (so height == 1.0).

- shoulder_width: widest row of the front view (pauldron spikes included)
- horn_span: widest row within the top 15% of the front view (horn tips)
- head_height: first row below the horns where the silhouette is a single run at
  least 10% of the shoulder width wide, down to the shoulder row (first row at 85%
  of full width)
- torso_height: shoulder row down to the waist row (narrowest central run, between
  the shoulders and 62% of the height)
- leg_length: waist row down to the soles
- arm_length: shoulder row down to the lowest row with silhouette in the outer 15%
  of the front view's columns (the hand)
- depth: widest row of the side view
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.refs import _silhouette_mask

ROOT = Path(__file__).resolve().parent.parent


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], row, [False]]).astype(int)
    edges = np.diff(padded)
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def _extent(mask: np.ndarray) -> np.ndarray:
    ext = np.zeros(mask.shape[0], dtype=int)
    for y, row in enumerate(mask):
        xs = np.where(row)[0]
        if xs.size:
            ext[y] = xs[-1] - xs[0] + 1
    return ext


def measure_proportions(front: Image.Image, side: Image.Image) -> dict[str, float]:
    fmask = _silhouette_mask(front)
    smask = _silhouette_mask(side)
    h, w = fmask.shape
    ext = _extent(fmask)
    max_ext = int(ext.max())

    shoulder_row = int(np.argmax(ext >= 0.85 * max_ext))
    horn_span = int(ext[: int(h * 0.15)].max())

    head_top = 0
    for y in range(h):
        runs = _runs(fmask[y])
        if len(runs) == 1 and runs[0][1] - runs[0][0] >= 0.10 * max_ext:
            head_top = y
            break

    outer = fmask[:, : int(w * 0.15)].any(axis=1)
    arm_bottom = int(np.where(outer)[0].max())

    centre = w // 2
    lo, hi = shoulder_row + int(0.05 * h), int(0.62 * h)
    best_row, best_width = lo, None
    for y in range(lo, hi):
        for x0, x1 in _runs(fmask[y]):
            if x0 <= centre < x1:
                if best_width is None or (x1 - x0) < best_width:
                    best_row, best_width = y, x1 - x0
    waist_row = best_row

    depth = int(_extent(smask).max())
    ratios = {
        "height": 1.0,
        "shoulder_width": max_ext / h,
        "horn_span": horn_span / h,
        "head_height": (shoulder_row - head_top) / h,
        "torso_height": (waist_row - shoulder_row) / h,
        "leg_length": (h - waist_row) / h,
        "arm_length": (arm_bottom - shoulder_row) / h,
        "depth": depth / smask.shape[0],
    }
    return {k: round(float(v), 4) for k, v in ratios.items()}


if __name__ == "__main__":
    views = ROOT / "refs" / "views"
    result = measure_proportions(Image.open(views / "front.png"),
                                 Image.open(views / "side_left.png"))
    out = ROOT / "data" / "reaper_proportions.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
