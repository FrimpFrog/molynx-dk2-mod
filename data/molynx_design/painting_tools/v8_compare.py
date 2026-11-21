import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from head_swap_v8 import *
from render_hq import render, view_for, label
from PIL import Image
W='/home/claude/molynx_design/work/'
data, tex = load_char()
new,_ = build_v8(data)
def texset(v):
    T = dict(tex)
    T['custom_character/body'] = Image.open(W + ('A_body.png' if v == 'A' else 'body_v6.png')).convert('RGB')
    T[HEAD_TEX] = Image.open(W + f'{v}_head.png').convert('RGB'); T[SKIRT_TEX] = Image.open(W + f'{v}_strip.png').convert('RGB')
    return T
c = new['stand']
rows = []
for v in ('A', 'B'):
    T = texset(v)
    rows.append([label(render(c, 0, T, az=0, size=380, ss=3, view=(np.array([0.05, 3.6, 0.1]), 0.95)), f'{v} face'),
                 label(render(c, 0, T, az=320, size=380, ss=3, view=(np.array([0.05, 3.55, 0.1]), 1.3)), f'{v} three-quarter'),
                 label(render(c, 0, T, az=285, size=380, ss=3, view=(np.array([0.05, 3.5, 0.1]), 1.5)), f'{v} side / neck'),
                 label(render(c, 0, T, az=0, size=380, ss=2, view=view_for(c, 0, [0])), f'{v} whole body')])
s = Image.new('RGB', (1520, 760))
for r, row in enumerate(rows):
    for i, im in enumerate(row): s.paste(im, (i * 380, r * 380))
s.save(W + 'v8_AB.png')
