"""Assemble the Blender renders into the deliverables: two DK2-style panels, the expressions sheet
and the turnaround."""
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

F = Path('/home/claude/molynx_blender/final'); OUT = Path('/home/claude/molynx_blender/out'); OUT.mkdir(exist_ok=True)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 34)
TOOLS = '/home/claude/molynx_cinematic/tools'

for e in ('grin', 'closed'):
    subprocess.run(['python3', f'{TOOLS}/panel.py', str(F / 'panel_body.png'), str(F / f'face_{e}.png'), str(OUT / f'Molynx_blender_panel_{e}.png')],
                   check=True, cwd='/home/claude/molynx_cinematic')


def sheet(paths, labels, out, h):
    ims = [Image.open(p).convert('RGB') for p in paths]
    ims = [im.resize((int(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
    pad = 16; W = sum(im.width for im in ims) + pad * (len(ims) + 1); H = h + pad * 2 + 56
    S = Image.new('RGB', (W, H), (8, 9, 13)); d = ImageDraw.Draw(S); x = pad
    for im, lab in zip(ims, labels):
        S.paste(im, (x, pad))
        tw = d.textlength(lab, font=FONT); d.text((x + (im.width - tw) / 2, pad + h + 10), lab, font=FONT, fill=(214, 186, 120))
        x += im.width + pad
    S.save(out)


sheet([F / 'face_grin.png', F / 'face_closed.png'], ['Grin', 'Mouth closed'], OUT / 'Molynx_blender_expressions.png', 900)
sheet([F / f'turn_{v}.png' for v in ('front', 'left', 'back', 'right')], ['Front', 'Side', 'Back', 'Side'], OUT / 'Molynx_blender_turnaround.png', 900)
Image.open(F / 'hero.png').save(OUT / 'Molynx_blender_hero.png')
print('composed', sorted(p.name for p in OUT.iterdir()))
