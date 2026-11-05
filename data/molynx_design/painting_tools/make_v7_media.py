import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from multiprocessing import Pool
from head_swap import *
from render_hq import render, view_for, label
from PIL import Image
OUT = Path('/home/claude/molynx_design/out_v7')
data, tex = load_char()
V6 = dict(tex); V6['custom_character/body'] = Image.open('/home/claude/molynx_design/work/body_v6.png').convert('RGB')
T = dict(V6); T[HEAD_TEX] = Image.open('/home/claude/molynx_design/work/golem_head_v7.png').convert('RGB')
new, _ = build(data)
TAG = 'Golem head'

def save_gif(frames, path, ms):
    pal = frames[0].quantize(colors=220, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, disposal=2, optimize=True)

def clip_gif(job):
    name, text, step, az = job
    c = new[name]; n = len(c['frames'])
    view = view_for(c, az)
    fr = [label(render(c, f, T, az=az, size=400, view=view), text) for f in range(0, n - (1 if c['wrap'] else 0), step)]
    save_gif(fr, OUT / f'Molynx_v7_{name}.gif', int(1000 / c['fps'] * step))
    return name, len(fr)

def turntable(kind):
    c = new['stand']
    pts = np.concatenate(c['frames'][0]); r = float(np.hypot(pts[:, 0] - 0.05, pts[:, 2]).max())
    if kind == 'body':
        view = (np.array([0.05, (pts[:, 1].min() + pts[:, 1].max()) / 2, 0]), max(2 * r, np.ptp(pts[:, 1])) * 1.05)
        name, text = 'Molynx_v7_turntable.gif', 'New head: Golem, angry, with horns'
    else:
        view = (np.array([0.05, 3.55, 0.0]), 1.5)
        name, text = 'Molynx_v7_head_turntable.gif', 'Head close-up, all the way round'
    fr = [label(render(c, 0, T, az=az, size=420, view=view), text) for az in range(0, 360, 8)]
    save_gif(fr, OUT / name, 80)
    return name, len(fr)

if __name__ == '__main__':
    c6, c7 = data['stand'], new['stand']
    hv = (np.array([0.05, 3.60, 0.1]), 1.05)
    pairs = [(label(render(c6, 0, V6, az=az, size=500, ss=3, view=hv), f'Before - {t}'),
              label(render(c7, 0, T, az=az, size=500, ss=3, view=hv), f'After - {t}')) for az, t in ((0, 'front'), (330, 'three-quarter'), (290, 'side'))]
    sheet = Image.new('RGB', (1500, 1000), (30, 30, 34))
    for i, (b, a) in enumerate(pairs):
        sheet.paste(b, (i * 500, 0)); sheet.paste(a, (i * 500, 500))
    sheet.save(OUT / 'Molynx_v7_head_before_after.png')
    label(render(c7, 0, T, az=0, size=1000, ss=3, view=(np.array([0.05, 3.62, 0.1]), 0.95)), 'Molynx v7 - face close-up', size=22).save(OUT / 'Molynx_v7_face_closeup.png')
    label(render(c7, 0, T, az=0, size=1000, ss=3, view=view_for(c7, 0, [0])), 'Molynx v7 - front', size=22).save(OUT / 'Molynx_v7_front_full.png')
    jobs = [('walk', 'Walk', 1, 200), ('attack', 'Attack', 1, 200), ('angry', 'Angry', 1, 330), ('celebrating', 'Dance / celebrating', 2, 200)]
    with Pool(2) as p:
        print(p.map(turntable, ['body', 'head']))
        for r in p.imap_unordered(clip_gif, jobs):
            print(r, flush=True)
