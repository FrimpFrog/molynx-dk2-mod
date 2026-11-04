"""Checks for turntable renders sent back from Blender."""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.refs import _silhouette_mask, silhouette_bbox

MIN_PIXEL_STD = 4.0


def check_render(path: Path, size: int = 1024) -> None:
    """Raise ValueError if the render is the wrong size or blank; FileNotFoundError if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        raise ValueError(f"{path.name}: size {img.size}, expected {(size, size)}")
    if float(np.asarray(img).std()) < MIN_PIXEL_STD:
        raise ValueError(f"{path.name}: image looks blank (pixel std under {MIN_PIXEL_STD})")


def _normalised_mask(img: Image.Image) -> np.ndarray:
    mask = _silhouette_mask(img)
    x0, y0, x1, y1 = silhouette_bbox(img)
    return mask[y0:y1, x0:x1]


def silhouette_iou(render: Image.Image, reference: Image.Image) -> float:
    """Overlap (intersection over union) of two silhouettes after trimming each to its own
    bounding box and resizing the render to the reference, so scale and offset do not matter."""
    ref = _normalised_mask(reference)
    mine = _normalised_mask(render)
    resized = Image.fromarray(mine.astype(np.uint8) * 255).resize(
        (ref.shape[1], ref.shape[0]), Image.NEAREST)
    mine = np.asarray(resized) > 0
    union = (mine | ref).sum()
    return float((mine & ref).sum() / union) if union else 0.0
