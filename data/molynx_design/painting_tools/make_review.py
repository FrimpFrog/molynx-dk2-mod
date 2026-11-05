import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from multiprocessing import Pool
from scene import *
from render_hq import render, label, fit_view
OUT = Path('/home/claude/molynx_design/out_review')
JOBS = [('fid_troll_idle2', 'Molynx_05_fidgeting', '05 Idle 2 (fidgeting) - NEW: Troll idle'),
        ('praying', 'Molynx_08_praying', '08 Praying - mace arm fixed'),
        ('happy', 'Molynx_09_happy', '09 Happy - mace arm fixed'),
        ('dance_troll', 'Molynx_11_celebrating', '11 Celebrating - NEW: Troll dance')]

def save_gif(frames, path, ms):
    pal = frames[len(frames) // 2].quantize(colors=230, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, disposal=2, optimize=True)

def job(j):
    clip, name, text = j
    tx = {}; a = MolynxActor(clip, tx)
    times = np.arange(0, a.duration, 1 / 15)
    pts = np.concatenate([np.concatenate(a.frame(t)) for t in times])
    lo, hi = pts[:, 1].min(), pts[:, 1].max()
    v = (np.array([0, (lo + hi) / 2, 0]), max(hi - lo, np.ptp(pts[:, 0]), np.ptp(pts[:, 2])) * 1.12)
    frames = [label(render({'parts': a.parts, 'frames': [a.frame(t)], 'wrap': False, 'fps': 1}, 0, tx, az=330, size=420, ss=2, view=v), text) for t in times]
    save_gif(frames, OUT / f'{name}.gif', 67)
    return name, len(frames)

if __name__ == '__main__':
    with Pool(4) as p:
        for r in p.imap_unordered(job, JOBS):
            print(r, flush=True)
