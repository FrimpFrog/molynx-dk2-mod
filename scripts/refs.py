"""Reference helpers: silhouette boxes and splitting the character turnaround sheet."""
from pathlib import Path

import numpy as np
from PIL import Image

VIEW_NAMES = ("front", "side_left", "back", "side_right")


def _silhouette_mask(img: Image.Image, bg_tol: int = 18) -> np.ndarray:
    """Boolean mask of pixels that differ from the border (background) colour."""
    rgb = np.asarray(img.convert("RGB")).astype(int)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    return np.abs(rgb - bg).max(axis=2) > bg_tol


def silhouette_bbox(img: Image.Image, bg_tol: int = 18) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the non-background pixels; x1 and y1 are exclusive."""
    mask = _silhouette_mask(img, bg_tol)
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if cols.size == 0:
        raise ValueError("no silhouette found: image is all background")
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def split_turnaround(path: Path, out_dir: Path) -> dict[str, Path]:
    """Split a four-view sheet at its empty background columns; save each view trimmed."""
    img = Image.open(path).convert("RGB")
    mask = _silhouette_mask(img)
    occupied = mask.any(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for x, filled in enumerate(occupied):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(occupied)))
    if len(runs) != len(VIEW_NAMES):
        raise ValueError(f"expected {len(VIEW_NAMES)} views, found {len(runs)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, Path] = {}
    for name, (x0, x1) in zip(VIEW_NAMES, runs):
        strip = img.crop((x0, 0, x1, img.height))
        trimmed = strip.crop(silhouette_bbox(strip))
        target = out_dir / f"{name}.png"
        trimmed.save(target)
        views[name] = target
    return views
