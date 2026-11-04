"""Clip prep: find the real picture area (drop black bars) and extract frames."""
import re
import subprocess
from collections import Counter
from pathlib import Path

Crop = tuple[int, int, int, int]  # (w, h, x, y)


def _frame_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def detect_crop(video: Path) -> Crop:
    """Return (w, h, x, y) of the picture area, the most frequent cropdetect result."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video),
         "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", proc.stderr)
    if not found:
        w, h = _frame_size(video)
        return (w, h, 0, 0)
    best = Counter(found).most_common(1)[0][0]
    return tuple(int(v) for v in best)  # type: ignore[return-value]


def extract_frames(video: Path, crop: Crop, start_s: float, end_s: float,
                   out_dir: Path) -> list[Path]:
    """Extract cropped PNG frames between start_s and end_s; return the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    w, h, x, y = crop
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start_s), "-t", str(end_s - start_s), "-i", str(video),
         "-vf", f"crop={w}:{h}:{x}:{y}",
         str(out_dir / "frame_%05d.png")],
        check=True,
    )
    return sorted(out_dir.glob("frame_*.png"))
