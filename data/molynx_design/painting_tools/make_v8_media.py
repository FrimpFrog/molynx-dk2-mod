import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from multiprocessing import Pool
from head_swap_v8 import *
import head_swap as v7
from render_hq import render, view_for, label
from PIL import Image
W = '/home/claude/molynx_design/work/'
OUT = Path('/home/claude/molynx_design/out_v8')
data, tex = load_char()
new, _ = build_v8(data)
NAMES = {'A': 'Version A - red rock body', 'B': 'Version B - smooth, matches body'}

def texset(v):
    T = dict(tex)
    T['custom_character/body'] = Image.open(W + ('A_body.png' if v == 'A' else 'body_v6.png')).convert('RGB')
    T[HEAD_TEX] = Image.open(W + f'{v}_head.png').convert('RGB'); T[SKIRT_TEX] = Image.open(W + f'{v}_strip.png').convert('RGB')
    return T

def save_gif(frames, path, ms):
    pal = frames[0].quantize(colors=220, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, disposal=2, optimize=True)

def job(j):
    v, kind = j
    T = texset(v); c = new['stand']
    if kind in ('turn_body', 'turn_head'):
        pts = np.concatenate(c['frames'][0]); r = float(np.hypot(pts[:, 0] - 0.05, pts[:, 2]).max())
        view = ((np.array([0.05, (pts[:, 1].min() + pts[:, 1].max()) / 2, 0]), max(2 * r, np.ptp(pts[:, 1])) * 1.05)
                if kind == 'turn_body' else (np.array([0.05, 3.55, 0.0]), 1.45))
        fr = [label(render(c, 0, T, az=az, size=420, view=view), NAMES[v]) for az in range(0, 360, 8)]
        save_gif(fr, OUT / f'Molynx_v8{v}_{"turntable" if kind == "turn_body" else "head_turntable"}.gif', 80)
    else:
        cl = new[kind]; n = len(cl['frames']); view = view_for(cl, 330)
        fr = [label(render(cl, f, T, az=330, size=400, view=view), f'{NAMES[v]} - {kind}')
              for f in range(0, n - (1 if cl['wrap'] else 0))]
        save_gif(fr, OUT / f'Molynx_v8{v}_{kind}.gif', int(1000 / cl['fps']))
    return v, kind

if __name__ == '__main__':
    # sheet 1: the shape fixes, v7 -> v8 (version B textures so only the shape differs)
    n7, _ = v7.build(data)
    T7 = dict(tex); T7['custom_character/body'] = Image.open(W + 'body_v6.png').convert('RGB'); T7[v7.HEAD_TEX] = Image.open(W + 'golem_head_v7.png').convert('RGB')
    TB = texset('B')
    views = [(0, (0.05, 3.62, 0.1), 0.95, 'front'), (320, (0.05, 3.55, 0.1), 1.3, 'three-quarter'), (285, (0.05, 3.5, 0.1), 1.5, 'side / neck'),
             (0, (0.30, 3.76, 0.0), 0.55, 'horn joint')]
    s = Image.new('RGB', (4 * 380, 760), (30, 30, 34))
    for i, (az, ctr, span, t) in enumerate(views):
        vw = (np.array(ctr) * [1, 1, 0], span)
        s.paste(label(render(n7['stand'], 0, T7, az=az, size=380, ss=3, view=vw), f'Before (v7) - {t}'), (i * 380, 0))
        s.paste(label(render(new['stand'], 0, TB, az=az, size=380, ss=3, view=vw), f'After (v8) - {t}'), (i * 380, 380))
    s.save(OUT / 'Molynx_v8_fixes_before_after.png')
    # sheet 2: A and B side by side
    rows = []
    for v in ('A', 'B'):
        T = texset(v); c = new['stand']
        rows.append([label(render(c, 0, T, az=0, size=380, ss=3, view=(np.array([0.05, 3.6, 0.1]), 0.95)), f'{v}: face'),
                     label(render(c, 0, T, az=320, size=380, ss=3, view=(np.array([0.05, 3.55, 0.1]), 1.3)), f'{v}: three-quarter'),
                     label(render(c, 0, T, az=285, size=380, ss=3, view=(np.array([0.05, 3.5, 0.1]), 1.5)), f'{v}: side / neck'),
                     label(render(c, 0, T, az=0, size=380, ss=2, view=view_for(c, 0, [0])), f'{v}: whole body')])
    s = Image.new('RGB', (1520, 760))
    for r, row in enumerate(rows):
        for i, im in enumerate(row): s.paste(im, (i * 380, r * 380))
    s.save(OUT / 'Molynx_v8_versions_A_and_B.png')
    for v in ('A', 'B'):
        label(render(new['stand'], 0, texset(v), az=0, size=1000, ss=3, view=(np.array([0.05, 3.62, 0.1]), 0.95)), f'{NAMES[v]} - face', size=22).save(OUT / f'Molynx_v8{v}_face_closeup.png')
    jobs = [(v, k) for v in ('A', 'B') for k in ('turn_body', 'turn_head', 'walk', 'angry')]
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            print(r, flush=True)
