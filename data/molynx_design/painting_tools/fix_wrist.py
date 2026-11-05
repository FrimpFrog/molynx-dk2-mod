"""A2 fix: the leftover arm-band patch on the wrist/back of the hand (not red, so the skin grade skipped it) gets the
surrounding rock skin."""
import sys; sys.path.insert(0, '/home/claude/molynx_design/tools')
from common import *
from scipy import ndimage
from PIL import Image, ImageDraw
from scripts.behemoth_rig import behemoth_rest, behemoth_labels
from head_paint_v8 import skin_mask

def arm_mask(data):
    rest, _, _ = behemoth_rest(UNIT); lab = behemoth_labels(rest)
    b = data['stand']['parts'][0]; tri = np.asarray(b['indices'], int).reshape(-1, 3); uv = np.asarray(b['uv'], float) * 512
    fore = np.isin(lab, [2, 3, 6, 7, 10])
    m = Image.new('L', (512, 512), 0); d = ImageDraw.Draw(m)
    for t in tri[np.all(fore[tri], axis=1)]:
        d.polygon([tuple(uv[i]) for i in t], fill=255)
    return np.asarray(m) > 0

def fix(path_in, path_out, data):
    a = np.asarray(Image.open(path_in).convert('RGB'), float)
    arm = arm_mask(data)
    sk = skin_mask(a) > 0.5
    L = a @ np.array([0.3, 0.59, 0.11])
    odd = arm & ~sk & (L > 40)
    lab_, n = ndimage.label(odd)
    big = np.isin(lab_, [i for i in range(1, n + 1) if (lab_ == i).sum() >= 20])   # the patch, not single stray texels
    region = ndimage.binary_dilation(big, iterations=2) & arm
    src = arm & sk & ~ndimage.binary_dilation(region, iterations=2)
    idx = ndimage.distance_transform_edt(~src, return_distances=False, return_indices=True)
    yy, xx = np.mgrid[0:512, 0:512]
    ny, nx = idx[0], idx[1]
    my = np.clip(2 * ny - yy, 0, 511); mx = np.clip(2 * nx - xx, 0, 511)      # reflect through the nearest skin texel
    ok = src[my, mx]
    fill = np.where(ok[..., None], a[my, mx], a[ny, nx])
    w = np.clip(ndimage.gaussian_filter(region.astype(float), 1.2) * 1.5, 0, 1)
    out = a * (1 - w[..., None]) + fill * w[..., None]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path_out)
    return int(region.sum())

if __name__ == '__main__':
    data, _ = load_char()
    W = '/home/claude/molynx_design/work/'
    print('A2 texels repainted', fix(W + 'A2_body.png', W + 'A2_body.png', data))
    print('A texels repainted', fix(W + 'A_body.png', W + 'A_body.png', data))
