from pathlib import Path

from PIL import Image

from scripts.measure_proportions import measure_proportions

VIEWS = Path(__file__).resolve().parent.parent / "refs" / "views"
KEYS = {"height", "shoulder_width", "horn_span", "head_height",
        "torso_height", "leg_length", "arm_length", "depth"}


def _measure():
    front = Image.open(VIEWS / "front.png")
    side = Image.open(VIEWS / "side_left.png")
    return measure_proportions(front, side), front, side


def test_ratios_keys_and_range():
    ratios, _, _ = _measure()
    assert set(ratios) == KEYS
    assert ratios["height"] == 1.0
    for key, value in ratios.items():
        if key != "height":
            assert 0.05 <= value <= 1.2, (key, value)


def test_front_side_height_agree():
    _, front, side = _measure()
    assert abs(front.height - side.height) / front.height < 0.03


def test_head_torso_leg_fit_inside_height():
    ratios, _, _ = _measure()
    total = ratios["head_height"] + ratios["torso_height"] + ratios["leg_length"]
    assert 0.80 < total <= 1.0, total
