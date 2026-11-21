import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from head_swap_v8 import *
from render_hq import render, label
from PIL import Image
data, tex = load_char()
body_png = sys.argv[3] if len(sys.argv) > 3 else '/home/claude/molynx_design/work/body_v6.png'
T = dict(tex); T['custom_character/body'] = Image.open(body_png).convert('RGB'); T[HEAD_TEX] = Image.open(sys.argv[2]).convert('RGB'); T[SKIRT_TEX] = Image.open('/home/claude/molynx_design/work/head_neck_strip.png').convert('RGB')
new, _ = build_v8(data)
c = new['stand']
azs = (0, 330, 290, 250, 180, 30)
ims = [label(render(c, 0, T, az=az, size=400, ss=3, view=(np.array([0.05, 3.55, 0.1]), 1.35)), f'az {az}') for az in azs]
s = Image.new('RGB', (1200, 800))
for i, im in enumerate(ims): s.paste(im, ((i % 3) * 400, (i // 3) * 400))
s.save(sys.argv[1])
