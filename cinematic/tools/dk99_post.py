"""1999-cinematic finish for rendered frames: warm grade with brown blacks, bloom around lights and eyes,
softness, vignette, grain. python3 dk99_post.py in_dir_or_files... --out DIR"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageFilter

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

files = []
for p in A.inputs:
    files += sorted(glob.glob(os.path.join(p, '*.png'))) if os.path.isdir(p) else [p]


def blur(x, r):
    im = Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r)), np.float32) / 255


def grade(x, k):
    h, w, _ = x.shape
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    # bloom: two radii from the bright parts (torch flames, eyes, archway light)
    br = np.clip((x - 0.62) / 0.38, 0, 1) ** 1.5
    bl = 0.55 * blur(br, w * 0.008) + 0.45 * blur(br, w * 0.03)
    x = x + bl * np.array([1.0, 0.82, 0.62], np.float32) * 0.85
    # contrast S-curve around mid-grey, then warm lift in the shadows / warm-yellow highlights
    x = np.clip(x, 0, 1.5)
    x = x / (1 + 0.15 * x)
    m = 0.32
    x = np.where(x < m, m * (x / m) ** 1.22, x)
    sat = 1.18
    g = x @ np.array([0.299, 0.587, 0.114], np.float32)
    x = g[..., None] + (x - g[..., None]) * sat
    x = x * np.array([1.06, 0.98, 0.86], np.float32) + np.array([0.022, 0.012, 0.004], np.float32) * (1 - lum[..., None])
    # vignette
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    x *= (1 - 0.32 * np.clip(r - 0.45, 0, 1) ** 1.6)[..., None]
    # softness of a period video master
    x = 0.7 * x + 0.3 * blur(x, 0.8)
    # grain (fine, slightly chromatic)
    rng = np.random.default_rng(A.seed * 100003 + k)
    n = rng.normal(0, 1, (h, w, 1)).astype(np.float32) * 0.018 + rng.normal(0, 1, (h, w, 3)).astype(np.float32) * 0.006
    x = x + n * (0.4 + 0.6 * np.sqrt(np.clip(lum, 0, 1)))[..., None]
    return np.clip(x, 0, 1)


for k, f in enumerate(files):
    x = np.asarray(Image.open(f).convert('RGB'), np.float32) / 255
    y = grade(x, k)
    Image.fromarray((y * 255 + 0.5).astype(np.uint8)).save(os.path.join(A.out, os.path.basename(f)))
print('POST_DONE', len(files))
