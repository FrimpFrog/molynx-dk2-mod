from pathlib import Path

from PIL import Image

from scripts.refs import silhouette_bbox, split_turnaround

TURNAROUND = Path(__file__).resolve().parent.parent / "refs" / "turnaround.png"


def test_bbox_synthetic():
    img = Image.new("RGB", (100, 100), (110, 110, 110))
    img.paste((255, 255, 255), (10, 20, 30, 60))  # 20 wide, 40 tall
    assert silhouette_bbox(img) == (10, 20, 30, 60)


def test_split_turnaround_views(tmp_path):
    views = split_turnaround(TURNAROUND, tmp_path)
    assert set(views) == {"front", "side_left", "back", "side_right"}
    sizes = {k: Image.open(p).size for k, p in views.items()}
    heights = [h for _, h in sizes.values()]
    assert max(heights) / min(heights) < 1.03
    assert sizes["front"][0] > 1.8 * sizes["side_left"][0]
    assert sizes["back"][0] > 1.8 * sizes["side_right"][0]
