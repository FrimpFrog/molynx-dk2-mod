"""Run inside Blender:  blender --background --python tests/blender/test_reaper_build.py

Exits 0 if every check passes, 1 otherwise. Needs data/reaper_proportions.json.
"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "blender"))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from build_reaper import build_reaper  # noqa: E402
from reaper_spec import PART_NAMES  # noqa: E402

PROPORTIONS = json.loads((ROOT / "data" / "reaper_proportions.json").read_text())
HEIGHT_M = 2.4


def _world_bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p[i] for p in pts) for i in range(3)))
    hi = Vector((max(p[i] for p in pts) for i in range(3)))
    return lo, hi


def test_objects_exist(coll):
    names = {o.name for o in coll.objects}
    missing = set(PART_NAMES) - names
    assert not missing, f"missing parts: {sorted(missing)}"
    assert len(PART_NAMES) == 13


def test_height_ratio(coll):
    body = [o for o in coll.objects if o.name != "Scythe"]
    lo, hi = _world_bounds(body)
    height = hi.z - lo.z
    width_ratio = (hi.x - lo.x) / height
    depth_ratio = (hi.y - lo.y) / height
    assert abs(width_ratio - PROPORTIONS["shoulder_width"]) / PROPORTIONS["shoulder_width"] < 0.05, width_ratio
    assert abs(depth_ratio - PROPORTIONS["depth"]) / PROPORTIONS["depth"] < 0.10, depth_ratio
    assert abs(height - HEIGHT_M) < 0.05 * HEIGHT_M, height


def test_no_empty_meshes(coll):
    for obj in coll.objects:
        assert obj.type == "MESH", obj.name
        assert len(obj.data.vertices) >= 8, f"{obj.name} has {len(obj.data.vertices)} vertices"


def main() -> int:
    coll = build_reaper(PROPORTIONS, height_m=HEIGHT_M)
    failures = 0
    for test in (test_objects_exist, test_height_ratio, test_no_empty_meshes):
        try:
            test(coll)
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"{3 - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
