import os
import subprocess
import sys
from pathlib import Path

import pytest

BLENDER_DIR = Path(__file__).resolve().parent.parent / "blender"
SCRIPTS = ["build_reaper.py", "render_turntable.py"]

STUB_BPY = "class app:\n    version = (3, 6, 0)\n    version_string = '3.6.0'\n"
STUB_BMESH = "ops = None\ntypes = None\n"
STUB_MATHUTILS = "class Matrix: pass\nclass Vector: pass\nclass Euler: pass\n"


@pytest.mark.parametrize("script", SCRIPTS)
def test_exits_2_outside_blender(script):
    proc = subprocess.run([sys.executable, str(BLENDER_DIR / script)],
                          capture_output=True, text=True)
    assert proc.returncode == 2
    assert "Blender" in proc.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_exits_2_on_old_blender(script, tmp_path):
    (tmp_path / "bpy.py").write_text(STUB_BPY)
    (tmp_path / "bmesh.py").write_text(STUB_BMESH)
    (tmp_path / "mathutils.py").write_text(STUB_MATHUTILS)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    proc = subprocess.run([sys.executable, str(BLENDER_DIR / script)],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 2
    assert "4.0" in proc.stderr and "3.6.0" in proc.stderr
