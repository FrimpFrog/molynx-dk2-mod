from pathlib import Path

import numpy as np
import pytest

TEX = Path("/tmp/mg/mgd/techs/megapack/factions/magic/units/battlemage/models/texture_battle_mage.png")


@pytest.mark.skipif(not TEX.exists(), reason="MegaGlest data absent")
def test_battlemage_is_yellow_with_red_crabs():
    from scripts.battlemage_texture import bake_battlemage
    px = np.asarray(bake_battlemage(TEX)).astype(int)
    tunic = px[100:150, 125:190]                           # front of the tunic
    assert tunic[..., 0].mean() > 190 and tunic[..., 2].mean() < 80          # yellow, not blue
    chest = px[78:104, 148:182]
    red = (chest[..., 0] > 150) & (chest[..., 1] < 90) & (chest[..., 2] < 90)
    assert red.sum() > 150                                                    # the red crab on the chest
