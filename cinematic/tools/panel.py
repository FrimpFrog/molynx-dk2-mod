"""DK2 creature-intro style panel: full body + face close-up on the blue backdrop, in a mossy
stone frame with iron studs (procedural, in the spirit of the game's intro panels)."""
import sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
sys.path.insert(0, '/home/claude/molynx_cinematic/tools')
from hd_build import fbm

W, H, B = 1920, 1080, 46
BODY, FACE, OUT = (sys.argv[1:4] if len(sys.argv) > 3 else ('work/panel_body.png', 'work/panel_face.png', 'out/Molynx_cinematic_panel.png'))
body = Image.open(BODY).convert('RGB')
face = Image.open(FACE).convert('RGB')
inner = np.zeros((H - 2 * B, W - 2 * B, 3))
iw, ih = inner.shape[1], inner.shape[0]
# backdrop: the face render's own blue, stretched
yy0, xx0 = np.mgrid[0:ih, 0:iw]
rr = np.hypot((xx0 - iw * 0.42) / (iw * 0.75), (yy0 - ih * 0.4) / (ih * 0.9))
c0, c1, c2 = np.array([29, 58, 110]), np.array([12, 29, 64]), np.array([3, 7, 15])
t1 = np.clip(rr / 0.55, 0, 1)[..., None]; t2 = np.clip((rr - 0.55) / 0.6, 0, 1)[..., None]
inner[:] = np.where(rr[..., None] < 0.55, c0 * (1 - t1) + c1 * t1, c1 * (1 - t2) + c2 * t2)
# face on the right, fading into the backdrop on its left edge
fw = int(ih * face.width / face.height)
f = np.asarray(face.resize((fw, ih), Image.LANCZOS)).astype(float)
x0 = iw - fw
ramp = np.clip((np.arange(fw) - 0) / (fw * 0.28), 0, 1) ** 1.5
inner[:, x0:] = inner[:, x0:] * (1 - ramp[None, :, None]) + f * ramp[None, :, None]
# full body on the left, over it
bwid = int(ih * body.width / body.height)
b = np.asarray(body.resize((bwid, ih), Image.LANCZOS)).astype(float)
bx = 40
rampb = np.clip((bwid - np.arange(bwid)) / (bwid * 0.18), 0, 1)[None, :, None] * np.clip(np.arange(bwid) / 30, 0, 1)[None, :, None]
inner[:, bx:bx + bwid] = inner[:, bx:bx + bwid] * (1 - rampb) + b * rampb
# vignette
yy, xx = np.mgrid[0:ih, 0:iw]
v = 1 - 0.35 * (((xx - iw / 2) / (iw / 2)) ** 2 + ((yy - ih / 2) / (ih / 2)) ** 2) ** 1.4
inner *= np.clip(v, 0.55, 1)[..., None]
# frame
n1 = fbm((H, W), (2, 4, 8, 16), seed=31); n2 = fbm((H, W), (16, 32, 64), seed=32)
stone = 70 + 18 * n1 + 10 * n2
yy, xx = np.mgrid[0:H, 0:W]
d_edge = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]).astype(float)
d_in = np.minimum.reduce([xx - B, yy - B, W - 1 - B - xx, H - 1 - B - yy]).astype(float)
bevel = np.where(d_edge < 8, ((xx < yy) * 1.0 - 0.5) * 0, 0)
light = np.clip(1.25 - d_edge / 60, 0.85, 1.25)
frame = np.stack([stone * 0.95, stone * 1.0, stone * 0.92], -1) * light[..., None]
moss = np.clip(n2 * 0.8 + 0.2 * n1 - 0.2, 0, 1) * np.clip(1 - d_edge / (B * 1.2), 0, 1)
frame = frame * (1 - 0.55 * moss[..., None]) + np.array([52, 78, 30]) * 0.55 * moss[..., None] * (0.8 + 0.4 * n1[..., None])
# inner lip: dark groove and a lit bevel
groove = (d_in > -6) & (d_in < 0)
frame[groove] *= 0.45
lip = (d_in > -12) & (d_in <= -6)
frame[lip] *= 1.25
# iron studs along the frame
out = frame.copy()
out[B:H - B, B:W - B] = inner
for sx in list(np.linspace(B * 2, W - B * 2, 14)) :
    for sy in (B / 2, H - B / 2):
        r = np.hypot(xx - sx, yy - sy); m = r < 11
        sh = np.clip(1 - r / 11, 0, 1)
        out[m] = (np.array([95, 92, 88]) * (0.5 + 0.9 * sh[m][:, None] * np.clip(1 - (xx[m] - sx + yy[m] - sy)[:, None] / 22, 0.3, 1.4)))
for sy in list(np.linspace(B * 2, H - B * 2, 8)):
    for sx in (B / 2, W - B / 2):
        r = np.hypot(xx - sx, yy - sy); m = r < 11
        sh = np.clip(1 - r / 11, 0, 1)
        out[m] = (np.array([95, 92, 88]) * (0.5 + 0.9 * sh[m][:, None] * np.clip(1 - (xx[m] - sx + yy[m] - sy)[:, None] / 22, 0.3, 1.4)))
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(OUT)
print('ok')
