"""Cut the rendered shots together, apply the 1999 finish, fades, and encode.
python3 dk99_edit.py OUT.mp4"""
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = '/home/claude/molynx_blender/dk99'
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/Molynx_1999_cinematic.mp4'
ORDER = [1, 2, 3, 4, 5, 6, 7]
seq = f'{ROOT}/seq'; graded = f'{ROOT}/graded'
for d in (seq, graded):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

frames = []
for s in ORDER:
    fs = sorted(glob.glob(f'{ROOT}/shots/s{s}/f_*.png'))
    assert fs, f'shot {s} has no frames'
    frames += fs
for i, f in enumerate(frames):
    os.symlink(f, f'{seq}/f_{i:04d}.png')
print('frames', len(frames))
subprocess.run(['python3', '/home/claude/molynx_cinematic/tools/dk99_post.py', seq, '--out', graded], check=True)

# fades: in from black over 12 frames, out over 18
n = len(frames)
for i in list(range(12)) + list(range(n - 18, n)):
    p = f'{graded}/f_{i:04d}.png'
    k = min(1.0, i / 12) if i < 12 else (n - 1 - i) / 18
    x = np.asarray(Image.open(p), np.float32) * (k ** 1.5)
    Image.fromarray(x.astype(np.uint8)).save(p)

# period compression: an MPEG-2 pass at the native 480x360, then a clean upscale to 4:3 1440x1080
mid = f'{ROOT}/mid.mpg'
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '25', '-i', f'{graded}/f_%04d.png', '-c:v', 'mpeg2video',
                '-b:v', '1600k', '-maxrate', '2000k', '-bufsize', '1000k', '-g', '12', '-pix_fmt', 'yuv420p', mid], check=True)
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mid, '-vf', 'scale=1440:1080:flags=bicubic,unsharp=5:5:0.25',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-pix_fmt', 'yuv420p', '-r', '25',
                '-movflags', '+faststart', OUT], check=True)
print('EDIT_DONE', OUT, n / 25.0, 's')
