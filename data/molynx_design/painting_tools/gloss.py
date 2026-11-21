"""DESIGN v8 A2: grade Molynx's rock skin to the real DK2 creatures (brighter, more contrast, painted gloss).
Targets measured on the game's ReaperTorso / ReaperBack (red mean ~145, std ~53, light-value median ~55)."""
import sys; sys.path.insert(0, '/home/claude/molynx_design/tools')
import numpy as np
from PIL import Image
from scipy import ndimage
from head_paint_v8 import skin_mask

def grade(img: Image.Image, mean_r=140.0, std_r=50.0, gloss=0.55, gloss_px=5.0) -> Image.Image:
    a = np.asarray(img.convert('RGB'), float)
    m = skin_mask(a)
    sel = m > 0.9
    r = a[..., 0]
    mu, sd = r[sel].mean(), r[sel].std()
    new_r = np.clip((r - mu) / max(sd, 1e-6) * std_r + mean_r, 4, 255)
    f = new_r / np.maximum(r, 4)                              # same factor on G and B: Molynx keeps his own hue
    g = a * f[..., None]
    # painted gloss: raised detail (lighter than its surroundings) on the lit parts gets a warm highlight, like DK2's skins
    L = g @ np.array([0.3, 0.59, 0.11])
    relief = L - ndimage.gaussian_filter(L, gloss_px)
    lit = np.clip((ndimage.gaussian_filter(L, gloss_px) - np.percentile(L[sel], 40)) / (np.percentile(L[sel], 92) - np.percentile(L[sel], 40)), 0, 1)
    hi = np.clip(relief / np.percentile(relief[sel], 97), 0, 1) ** 1.4 * (0.35 + 0.65 * lit)
    g = g + hi[..., None] * np.array([255.0, 150.0, 70.0]) * gloss * (1 - g / 300.0)
    out = a * (1 - m[..., None]) + np.clip(g, 0, 255) * m[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
