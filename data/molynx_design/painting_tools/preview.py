import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from common import *
import importlib, design_paint
from render_hq import render, label
from PIL import Image
body, uv, idx = paint_inputs()
new = design_paint.paint_character(UNIT, body, uv, idx)
new.save('/home/claude/molynx_design/work/body_v6.png')
data, tex = load_char()
newtex = dict(tex); newtex['custom_character/body'] = new
clip = data['stand']
ims=[]
for az,t in ((0,'old front'),(0,'new front'),(180,'old back'),(180,'new back'),(90,'old side'),(90,'new side')):
    T = tex if t.startswith('old') else newtex
    ims.append(label(render(clip,0,T,az=az,size=400,view=(np.array([0.05,2.75,0]),1.9)),t))
sheet=Image.new('RGB',(800,1200))
for i,im in enumerate(ims): sheet.paste(im,((i%2)*400,(i//2)*400))
sheet.save('/home/claude/molynx_design/work/zoom_torso.png')
