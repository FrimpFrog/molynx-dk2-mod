"""Re-export the head sculpts, teeth and eyeballs only (after face edits) into the Blender parts folder."""
import sys
import numpy as np
sys.path.insert(0, '/home/claude/molynx_cinematic/tools'); sys.path.insert(0, '/home/claude/molynx')
import hd_build as H
import head_sdf as HS
from hd_build import tooth_mesh
from subdiv import laplacian_smooth, loop_subdivide
P = '/home/claude/molynx_blender/parts/'
for expr in ('grin', 'closed'):
    Vh, Fh = HS.polygonize(expr)
    Vh = laplacian_smooth(Vh, Fh, 2, 0.4)
    np.savez_compressed(P + f'head_{expr}.npz', V=(Vh + HS.O) * H.MIRROR, F=Fh[:, ::-1])
    TP, TF = [], []
    for base, up, out, w, h, dpt, fang in HS.teeth_set(expr):
        Pt, Ft = tooth_mesh(base, up, out, w, h, dpt, fang=fang)
        TF.append(Ft + sum(len(p) for p in TP)); TP.append(Pt)
    TPv, TFv = np.vstack(TP), np.vstack(TF)
    TPv, TFv, _ = loop_subdivide(TPv, TFv); TPv, TFv, _ = loop_subdivide(TPv, TFv)
    tip = np.clip((TPv[:, 1] - (-0.17)) / 0.06, 0, 1)
    np.savez_compressed(P + f'teeth_{expr}.npz', V=(TPv + HS.O) * H.MIRROR, F=TFv[:, ::-1], t=tip)
    print(expr, Vh.shape, TPv.shape)
