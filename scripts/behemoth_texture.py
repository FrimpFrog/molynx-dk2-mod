"""Texture of the MegaGlest Behemoth (Magic faction daemon giant) in the dark-red colours of its in-game art. LICENCE: GPL-3.0
(texture: MegaGlest data, CC BY-SA 3.0).

The engine paints the semi-transparent parts of the texture in the player's team colour; here a fixed dark red is baked in,
which gives the red-brown hide with pale rib stripes of the reference picture. Meshes without team colour (club, horns) keep their own look.
"""
from pathlib import Path

from PIL import Image

from scripts.glest_export import PLAIN_BACKDROP

TEAM_RED = (150, 36, 28)


def bake_behemoth(path: Path, custom_color: bool = True, team=None) -> Image.Image:
    """`team` is accepted for the exporter's baker signature and ignored: the Behemoth is always dark red."""
    im = Image.open(path).convert("RGBA")
    backdrop = Image.new("RGBA", im.size, (*(TEAM_RED if custom_color else PLAIN_BACKDROP), 255))
    backdrop.alpha_composite(im)
    return backdrop.convert("RGB")
