"""The Battlemage in his original colours: yellow hat and tunic, red crab emblems. LICENCE: GPL-3.0.

The MegaGlest texture marks the team-coloured cloth with partial alpha (hat, tunic) and the emblems are opaque
grey; in the original artwork the cloth is yellow and the crabs are red. Boxes are in 256 px texture coordinates.
"""
import numpy as np
from PIL import Image

CREST_BOXES = [(58, 30, 118, 66, True), (145, 76, 184, 106, False), (84, 184, 106, 204, False)]   # x0,y0,x1,y1,emblem is team-coloured
YELLOW = np.array([248, 208, 22.0])
RED = np.array([205, 38, 22.0])


def bake_battlemage(path, custom_color=True, team=None) -> Image.Image:
    a = np.asarray(Image.open(path).convert("RGBA")).astype(float)
    rgb, alpha = a[..., :3], a[..., 3]
    lum = (0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]) / 255
    cloth = alpha < 240
    out = rgb.copy()
    out[cloth] = np.clip(YELLOW * (0.55 + 0.95 * np.clip(lum * 1.6, 0, 1.2))[..., None], 0, 255)[cloth]
    emblem = np.zeros(alpha.shape, bool)
    for x0, y0, x1, y1, on_cloth in CREST_BOXES:
        sub = (cloth if on_cloth else ~cloth)[y0:y1, x0:x1] & (rgb[y0:y1, x0:x1].max(-1) > 40)
        emblem[y0:y1, x0:x1] = sub
    out[emblem] = np.clip(RED * (0.5 + 1.0 * np.clip(lum * 1.6, 0, 1.2))[..., None], 0, 255)[emblem]
    return Image.fromarray(out.astype(np.uint8), "RGB")
