import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
import head_swap
from head_swap import *
from render_hq import render, label
from PIL import Image
data, tex = load_char()
T = dict(tex); T['custom_character/body'] = Image.open('/home/claude/molynx_design/work/body_v6.png').convert('RGB')
new, gtex = build(data)
T[HEAD_TEX] = Image.open(sys.argv[2]).convert('RGB') if len(sys.argv) > 2 else gtex
c = new['stand']
azs=(0,330,300,270,225,180)
ims=[label(render(c,0,T,az=az,size=400,view=(np.array([0.05,3.55,0.1]),1.4)),f'az {az}') for az in azs]
s=Image.new('RGB',(1200,800))
for i,im in enumerate(ims): s.paste(im,((i%3)*400,(i//3)*400))
s.save(sys.argv[1])
