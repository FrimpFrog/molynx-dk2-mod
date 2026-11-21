import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
import numpy as np
from PIL import Image
from scene import MolynxActor
from render_hq import render, _rot
tx = {}; a = MolynxActor('stand', tx)
pts = a.frame(0.0); head = pts[2]
out = sys.argv[1]; az = float(sys.argv[2])
hc = head.mean(0) @ _rot(az).T
clip = {'parts': a.parts, 'frames': [pts], 'wrap': False, 'fps': 1}
bg = render(clip, 0, tx, az=az, size=1140, view=(np.array([hc[0] + 0.01, 0.585, 0]), 0.40), ss=3, bg=(10, 23, 89))
icon = render(clip, 0, tx, az=az, size=600, view=(np.array([hc[0], hc[1] + 0.005, 0]), 0.17), ss=3, bg=(10, 23, 89))
bg.save(f'{out}/portrait_bg_A2.png'); icon.save(f'{out}/portrait_icon_A2.png')
w = round(1140 * 50 / 114); x = (1140 - w) // 2
prev = Image.new('RGB', (w + 600 + 20, 1140), (30, 30, 30)); prev.paste(bg.crop((x, 0, x + w, 1140)), (0, 0)); prev.paste(icon, (w + 20, 0))
prev.resize((prev.width // 2, prev.height // 2)).save(f'{out}/portrait_src_preview.png')
