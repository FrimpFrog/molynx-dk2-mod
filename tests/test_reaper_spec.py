import json
from pathlib import Path

import numpy as np
from PIL import Image

from blender.reaper_spec import PART_NAMES, bounds, part_specs
from scripts.refs import _silhouette_mask
from scripts.silhouette_preview import rasterize

ROOT = Path(__file__).resolve().parent.parent
PROPORTIONS = json.loads((ROOT / "data" / "reaper_proportions.json").read_text())


def _body_prims(specs):
    return [p for name, prims in specs.items() if name != "Scythe" for p in prims]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / (a | b).sum())


def test_all_parts_present():
    specs = part_specs(PROPORTIONS, height_m=1.0)
    assert set(specs) == set(PART_NAMES)
    assert len(PART_NAMES) == 13
    assert all(len(prims) >= 1 for prims in specs.values())


def test_spec_bounds_match_ratios():
    (x0, y0, z0), (x1, y1, z1) = bounds(_body_prims(part_specs(PROPORTIONS, 1.0)))
    assert abs((x1 - x0) - PROPORTIONS["shoulder_width"]) / PROPORTIONS["shoulder_width"] < 0.05
    assert abs((y1 - y0) - PROPORTIONS["depth"]) / PROPORTIONS["depth"] < 0.10
    assert abs(z0) < 0.01 and abs(z1 - 1.0) < 0.02


def test_height_scales_with_height_m():
    (_, _, _), (_, _, top) = bounds(_body_prims(part_specs(PROPORTIONS, 2.4)))
    assert abs(top - 2.4) < 0.05


def _view_iou(view_name: str, axis: str) -> float:
    ref = _silhouette_mask(Image.open(ROOT / "refs" / "views" / f"{view_name}.png"))
    mine = rasterize(_body_prims(part_specs(PROPORTIONS, 1.0)), axis, ref.shape[::-1])
    return _iou(mine, ref)


def test_front_silhouette_overlap():
    assert _view_iou("front", "front") >= 0.80


def test_side_silhouette_overlap():
    assert _view_iou("side_left", "side") >= 0.80
