"""Re-export only the two head sculpts (after face edits) into the Blender parts folder."""
import sys
import numpy as np
sys.path.insert(0, '/home/claude/molynx_cinematic/tools'); sys.path.insert(0, '/home/claude/molynx')
import hd_build as H
import head_sdf as HS
from subdiv import laplacian_smooth
for expr in ('grin', 'closed'):
    Vh, Fh = HS.polygonize(expr)
    Vh = laplacian_smooth(Vh, Fh, 2, 0.4)
    np.savez_compressed(f'/home/claude/molynx_blender/parts/head_{expr}.npz', V=(Vh + HS.O) * H.MIRROR, F=Fh[:, ::-1])
    print(expr, Vh.shape, Fh.shape)
