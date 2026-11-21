import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from multiprocessing import Pool
from scene import *
from render_hq import render, label
OUT = Path('/home/claude/molynx_design/out_poses')

def save_gif(frames, path, ms):
    pal = frames[len(frames)//2].quantize(colors=230, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, disposal=2, optimize=True)

def setup(clip):
    tx = {}; a = MolynxActor(clip, tx)
    return tx, a

def view_all(a, times, pad=1.06):
    pts = np.concatenate([np.concatenate(a.frame(t)) for t in times])
    r = float(np.hypot(pts[:, 0], pts[:, 2]).max()); lo, hi = pts[:, 1].min(), pts[:, 1].max()
    return np.array([0, (lo + hi) / 2, 0]), max(2 * r, hi - lo) * pad

def job(kind):
    if kind == 'sheets':
        for clip, name in (('tpose', 'T-pose'), ('stand', 'Idle')):
            tx, a = setup(clip); fr = a.frame(0)
            c = {'parts': a.parts, 'frames': [fr], 'wrap': False, 'fps': 1}
            v = view_all(a, [0.0], 1.04)
            ims = [label(render(c, 0, tx, az=az, size=600, ss=3, view=v), f'Molynx {name} - {t}') for az, t in ((0, 'front'), (90, 'side'), (180, 'back'))]
            s = Image.new('RGB', (1800, ims[0].height))
            for i, im in enumerate(ims): s.paste(im, (i * 600, 0))
            s.save(OUT / f'Molynx_{clip}_front_side_back.png')
            label(render(c, 0, tx, az=0, size=1100, ss=3, view=v), f'Molynx {name} - front', 22).save(OUT / f'Molynx_{clip}_front_large.png')
        return kind
    if kind in ('tpose_turn', 'stand_turn'):
        clip = kind.split('_')[0]; tx, a = setup(clip); fr = a.frame(0)
        c = {'parts': a.parts, 'frames': [fr], 'wrap': False, 'fps': 1}
        v = view_all(a, [0.0])
        frames = [label(render(c, 0, tx, az=az, size=480, ss=2, view=v), f'Molynx {"T-pose" if clip == "tpose" else "idle pose"} - turning') for az in range(0, 360, 8)]
        save_gif(frames, OUT / f'Molynx_{clip}_turntable.gif', 80)
        return kind
    if kind == 'idle_anim':
        out = []
        for clip, title in (('stand', 'Idle (his stand loop)'), ('fidgeting', 'Idle 2 (fidgeting)')):
            tx, a = setup(clip)
            times = np.arange(0, a.duration, 1 / 15)
            v = view_all(a, times[::3])
            frames = [label(render({'parts': a.parts, 'frames': [a.frame(t)], 'wrap': False, 'fps': 1}, 0, tx, az=20, size=480, ss=2, view=v), f'Molynx {title}') for t in times]
            save_gif(frames, OUT / f'Molynx_{clip}_animation.gif', 67)
            out.append(len(frames))
        return kind, out

if __name__ == '__main__':
    with Pool(4) as p:
        for r in p.imap_unordered(job, ['sheets', 'tpose_turn', 'stand_turn', 'idle_anim']):
            print(r, flush=True)
