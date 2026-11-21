import sys; sys.path.insert(0, '/home/claude/molynx_design/tools')
from multiprocessing import Pool
from scene import *
from render_hq import render, label
OUT = Path('/home/claude/molynx_design/out_scenes')
GIF_FPS = 15

def view_of(ap, times, az, pad=1.12):
    a = np.radians(az)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pts = np.concatenate([np.concatenate(scene_clip(ap, t)['frames'][0]) for t in times]) @ R.T
    lo, hi = pts.min(0), pts.max(0)
    span = float(max(hi - lo)) * pad
    view_of.box = ((hi - lo)[0] * pad / span, (hi - lo)[1] * pad / span)    # width, height as fractions of the square
    return (lo + hi) / 2, span

def crop(im, box, size, text, lab=17):
    w, h = box
    W_, H_ = int(min(1, w + 0.04) * size), int(min(1, h + 0.06) * size)
    x0, y0 = (size - W_) // 2, (size - H_) // 2
    return label(im.crop((x0, y0, x0 + W_, y0 + H_ + lab + 10)) if False else _pad(im.crop((x0, y0, x0 + W_, y0 + H_)), lab), text, lab)

def _pad(im, lab):
    out = Image.new('RGB', (im.width, im.height + lab + 10), (42, 42, 46)); out.paste(im, (0, 0)); return out

def save_gif(frames, path, ms):
    pal = frames[len(frames) // 2].quantize(colors=230, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=ms, loop=0, disposal=2, optimize=True)

def make(spec):
    kind, name, text, actors, az, secs = spec
    tx = {}
    ap = [(A(stem, tx) if A is KmfActor else A(stem, tx), place) for (A, stem, place) in actors]
    if kind == 'still':
        v = view_of(ap, [0.0], az, 1.15)
        crop(render(scene_clip(ap, secs), 0, tx, az=az, size=1100, ss=3, view=v), view_of.box, 1100, text, 22).save(OUT / f'{name}.png')
        return name
    times = np.arange(0, secs, 1 / GIF_FPS)
    v = view_of(ap, times[::3], az)
    box = view_of.box
    frames = [crop(render(scene_clip(ap, t), 0, tx, az=az, size=560, ss=2, view=v), box, 560, text) for t in times]
    save_gif(frames, OUT / f'{name}.gif', int(1000 / GIF_FPS))
    return name, len(frames)

M, K = MolynxActor, KmfActor
SPECS = [
    # stills
    ('still', 'Molynx_and_Horned_Reaper', 'Molynx (rock) and the Horned Reaper - same size',
     [(M, 'stand', (15, (-0.36, 0, 0))), (K, 'REA3_Dance', (-15, (0.36, 0, 0)))], 15, 0.0),
    ('still', 'Molynx_and_Knight', 'Molynx (rock, Horned Reaper size) and a Knight',
     [(M, 'stand', (15, (-0.32, 0, 0))), (K, 'Knight-Idle1', (-15, (0.30, 0, 0)))], 15, 0.0),
    ('still', 'Molynx_size_lineup', 'Size line-up (Molynx now Horned Reaper size): Bile Demon, Molynx, Knight, Horned Reaper',
     [(K, 'BileDemon-Idle1', (0, (-1.15, 0, 0))), (M, 'stand', (0, (-0.40, 0, 0))), (K, 'Knight-Idle1', (0, (0.35, 0, 0))),
      (K, 'REA3_Dance', (0, (1.15, 0, 0)))], 20, 0.0),
    # with the Horned Reaper
    ('gif', 'Molynx_vs_Horned_Reaper_fight', 'Molynx vs the Horned Reaper',
     [(M, 'attack', (90, (-0.38, 0, 0))), (K, 'REA3_Melee3', (-90, (0.38, 0, 0)))], 25, 4.0),
    ('gif', 'Molynx_and_Horned_Reaper_dance', 'Molynx and the Horned Reaper dancing',
     [(M, 'victory_celebration', (20, (-0.38, 0, 0))), (K, 'REA3_Dance', (-20, (0.38, 0, 0)))], 10, 8.2),
    ('gif', 'Molynx_and_Horned_Reaper_angry', 'Face-off: Molynx angry, the Reaper roars',
     [(M, 'angry', (60, (-0.38, 0, 0))), (K, 'REA3_Roar', (-60, (0.38, 0, 0)))], 15, 4.0),
    # with the Knight
    ('gif', 'Molynx_vs_Knight_fight', 'Molynx vs a Knight',
     [(M, 'attack', (90, (-0.35, 0, 0))), (K, 'Knight-Melee', (-90, (0.33, 0, 0)))], 25, 4.0),
    ('gif', 'Molynx_and_Knight_walk', 'Molynx and a Knight walking side by side',
     [(M, 'walk', (0, (-0.31, 0, 0))), (K, 'Knight-Walk', (0, (0.29, 0, 0)))], 30, 3.0),
    ('gif', 'Molynx_and_Knight_dance', 'Molynx and a Knight dancing',
     [(M, 'victory_celebration', (20, (-0.35, 0, 0))), (K, 'Kni-Dance', (-20, (0.33, 0, 0)))], 10, 8.2),
]

if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1:] or None
    specs = [s for s in SPECS if not only or s[1] in only]
    with Pool(4) as p:
        for r in p.imap_unordered(make, specs):
            print(r, flush=True)
