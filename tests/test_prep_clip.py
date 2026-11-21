import subprocess
from pathlib import Path

import pytest
from PIL import Image

from scripts.prep_clip import detect_crop, extract_frames

REAL_CLIP = Path(
    "/mnt/user-data/attachments/76eb4112-412f-58e4-a30a-b275f992be89/"
    "c0cefca4-Dungeon_Keeper_Remastered_Video.mp4"
)


def _make_video(path: Path, graph: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", graph, "-r", "30", "-t", "1", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def pillarbox(tmp_path):
    graph = ("color=black:s=640x360:r=30:d=1[bg];"
             "testsrc=s=480x360:r=30:d=1[fg];[bg][fg]overlay=80:0")
    return _make_video(tmp_path / "pillarbox.mp4", graph)


@pytest.fixture
def full_frame(tmp_path):
    return _make_video(tmp_path / "full.mp4", "testsrc=s=640x360:r=30:d=1")


def test_detect_crop_pillarbox(pillarbox):
    assert detect_crop(pillarbox) == (480, 360, 80, 0)


def test_detect_crop_no_bars(full_frame):
    assert detect_crop(full_frame) == (640, 360, 0, 0)


def test_extract_frames_count_and_size(pillarbox, tmp_path):
    out = tmp_path / "frames"
    frames = extract_frames(pillarbox, (480, 360, 80, 0), 0.0, 1.0, out)
    assert len(frames) == 30
    assert all(p.exists() for p in frames)
    assert Image.open(frames[0]).size == (480, 360)


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="real clip not present")
def test_real_clip_crop():
    assert detect_crop(REAL_CLIP) == (1440, 1080, 240, 0)
