import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from multiprocessing import Pool
from scene import *
from render_hq import render, label
from common import load_char
OUT = Path('/home/claude/molynx_design/out_all')
NAMES = {'stand': 'Idle (stand)', 'walk': 'Walk', 'attack': 'Attack', 'die': 'Die', 'fidgeting': 'Idle 2 (fidgeting)',
 'running': 'Run', 'eating': 'Eating', 'praying': 'Praying', 'happy': 'Happy', 'angry': 'Angry', 'celebrating': 'Celebrating',
 'victory_celebration': 'Dance (victory celebration)', 'getting_hit': 'Getting hit', 'being_slapped': 'Being slapped',
 'staggering': 'Staggering', 'waking_up': 'Waking up', 'knockback': 'Knocked back', 'sleeping': 'Sleeping',
 'training': 'Training', 'fighting_another_creature': 'Fighting', 'patrolling': 'Patrolling', 'guarding': 'Guarding',
 'manufacturing': 'Working (manufacturing)', 'interacting_with_objects': 'Interacting with objects',
 'being_picked_up': 'Picked up (in the hand)'}
ORDER = list(NAMES)

def save_gif(frames, path, ms):
    pal = frames[len(frames) // 2].quantize(colors=230, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, disposal=2, optimize=True)

def job(clip):
    tx = {}; a = MolynxActor(clip, tx)
    times = np.arange(0, a.duration, 1 / 15)
    pts = np.concatenate([np.concatenate(a.frame(t)) for t in times])
    r = float(np.hypot(pts[:, 0], pts[:, 2]).max()); lo, hi = pts[:, 1].min(), pts[:, 1].max()
    cx, cz = pts[:, 0].mean(), pts[:, 2].mean()
    v = (np.array([0, (lo + hi) / 2, 0]), max(hi - lo, np.ptp(pts[:, 0]), np.ptp(pts[:, 2])) * 1.12)
    v = (np.array([(pts[:,0].min()+pts[:,0].max())/2 * 0, (lo + hi) / 2, 0]), v[1])
    n = ORDER.index(clip) + 1
    frames = [label(render({'parts': a.parts, 'frames': [a.frame(t)], 'wrap': False, 'fps': 1}, 0, tx, az=330, size=420, ss=2, view=v),
                    f'{n:02d}  {NAMES[clip]}') for t in times]
    if not a.c['wrap']:
        frames += [frames[-1]] * 8                       # hold the end of one-way clips a moment
    save_gif(frames, OUT / f'Molynx_{n:02d}_{clip}.gif', 67)
    return clip, len(frames), frames[len(frames) // 2]

if __name__ == '__main__':
    mids = {}
    with Pool(4) as p:
        for clip, n, mid in p.imap_unordered(job, ORDER):
            mids[clip] = mid; print(clip, n, flush=True)
    W = 210
    sheet = Image.new('RGB', (5 * W, 5 * W), (30, 30, 34))
    for i, c in enumerate(ORDER):
        sheet.paste(mids[c].resize((W, W)), ((i % 5) * W, (i // 5) * W))
    sheet.save(OUT / 'Molynx_all_animations_sheet.png')
