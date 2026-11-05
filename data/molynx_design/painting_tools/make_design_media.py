import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from common import *
from multiprocessing import Pool
from render_hq import render, view_for, label
from PIL import Image
OUT = Path('/home/claude/molynx_design/out')
data, tex = load_char()
new = Image.open('/home/claude/molynx_design/work/body_v6.png').convert('RGB')
NEW = dict(tex); NEW['custom_character/body'] = new

def save_gif(frames, path, ms):
    pal = frames[0].quantize(colors=220, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, disposal=2, optimize=True)

def clip_gif(job):
    name, text, step = job
    c = data[name]; n = len(c['frames'])
    view = view_for(c, 200)
    fr = [label(render(c, f, NEW, az=200, size=400, view=view), text) for f in range(0, n - (1 if c['wrap'] else 0), step)]
    save_gif(fr, OUT / f'Molynx_v6_{name}.gif', int(1000 / c['fps'] * step))
    return name, len(fr)

def turntable(_):
    c = data['stand']
    pts = np.concatenate(c['frames'][0]); r = float(np.hypot(pts[:,0]-0.05, pts[:,2]).max())
    view = (np.array([0.05, (pts[:,1].min()+pts[:,1].max())/2, 0]), max(2*r, np.ptp(pts[:,1]))*1.05)
    fr = [label(render(c, 0, NEW, az=az, size=400, view=view), 'New design: harness off, Battlemage belt') for az in range(0, 360, 8)]
    save_gif(fr, OUT / 'Molynx_v6_turntable.gif', 80)
    return 'turntable', len(fr)

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    c = data['stand']
    tiles = []
    for az, t in ((0, 'front'), (180, 'back'), (90, 'side')):
        v = view_for(c, az, [0])
        tiles.append((label(render(c, 0, tex, az=az, size=420, view=v), f'Before - {t}'), label(render(c, 0, NEW, az=az, size=420, view=v), f'After - {t}')))
    sheet = Image.new('RGB', (420 * 3, 840), (30, 30, 34))
    for i, (b, a) in enumerate(tiles):
        sheet.paste(b, (i * 420, 0)); sheet.paste(a, (i * 420, 420))
    sheet.save(OUT / 'Molynx_v6_before_after.png')
    jobs = [('walk', 'Walk', 1), ('attack', 'Attack', 1), ('fidgeting', 'Fidgeting', 2), ('celebrating', 'Dance / celebrating', 2)]
    with Pool(2) as p:
        print(p.map(turntable, [0]))
        for r in p.imap_unordered(clip_gif, jobs):
            print(r, flush=True)
