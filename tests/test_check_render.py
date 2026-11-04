import numpy as np
import pytest
from PIL import Image

from scripts.check_render import check_render, silhouette_iou

GREY = (110, 110, 110)


def _noise(size, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _shape(size, boxes, fill=(255, 255, 255)):
    img = Image.new("RGB", (size, size), GREY)
    for box in boxes:
        img.paste(fill, box)
    return img


def test_blank_image_raises(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (1024, 1024), GREY).save(path)
    with pytest.raises(ValueError, match="blank"):
        check_render(path)


def test_wrong_size_raises(tmp_path):
    path = tmp_path / "small.png"
    _noise(512).save(path)
    with pytest.raises(ValueError, match="size"):
        check_render(path)


def test_normal_image_passes(tmp_path):
    path = tmp_path / "ok.png"
    _noise(1024).save(path)
    check_render(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_render(tmp_path / "nope.png")


def test_iou_identical_is_one():
    a = _shape(100, [(10, 10, 40, 40), (60, 60, 90, 90)])
    assert silhouette_iou(a, a) == 1.0


def test_iou_complementary_shapes_is_zero():
    left = _shape(100, [(10, 10, 50, 50), (50, 50, 90, 90)])
    right = _shape(100, [(50, 10, 90, 50), (10, 50, 50, 90)])
    assert silhouette_iou(left, right) == 0.0


def test_iou_ignores_scale_and_offset():
    small = _shape(100, [(10, 10, 30, 50), (30, 30, 40, 50)])
    big = _shape(300, [(60, 60, 120, 180), (120, 120, 150, 180)])
    assert silhouette_iou(small, big) > 0.95
