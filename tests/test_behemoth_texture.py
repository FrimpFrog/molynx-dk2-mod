from pathlib import Path

import numpy as np
import pytest

from scripts.behemoth_texture import bake_behemoth

PNG = Path(__file__).resolve().parent.parent / "megaglest_behemoth" / "models" / "texture_daemon_giant.png"


@pytest.mark.skipif(not PNG.exists(), reason="Behemoth texture absent")
def test_team_colour_areas_are_red_whatever_team_the_exporter_asks_for():
    a = np.asarray(bake_behemoth(PNG, True, team=(40, 80, 200))).astype(int)
    b = np.asarray(bake_behemoth(PNG, True)).astype(int)
    assert (a == b).all()
    assert b[..., 0].mean() > b[..., 2].mean()                   # red-brown overall, not the blue team colour
