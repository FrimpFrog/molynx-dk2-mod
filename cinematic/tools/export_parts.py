"""Exports Molynx's cinematic parts as clean geometry (no baked rock) for Blender: one .npz per part
(V, F, optional corner UVs, colours, per-triangle region class) plus the belt/cloth texture.
Coordinates: the game's (x mirrored), y up; Blender's import script turns them z-up."""
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/claude/molynx_cinematic/tools')
sys.path.insert(0, '/home/claude/molynx')
import hd_build as H
import head_sdf as HS
from subdiv import vertex_normals, laplacian_smooth, loop_subdivide

OUT = Path('/home/claude/molynx_blender/parts'); OUT.mkdir(parents=True, exist_ok=True)


def save(name, **kw):
    np.savez_compressed(OUT / f'{name}.npz', **kw)
    print(name, {k: getattr(v, 'shape', v) for k, v in kw.items()})


def main():
    from scripts.character_paint import HORNS
    data, tex = pickle.load(open(H.SRC, 'rb'))
    clip = data['stand']
    H.TOPO = data['stand']
    H.POSE_PARTS[id(clip)] = [0, 1, 2, 3]
    V, F, UVc, _ = H.part_mesh(clip, 0)
    raw = np.asarray(clip['frames'][0][0], float) * H.MIRROR
    key = np.round(raw / 1e-6).astype(np.int64); wkey = np.round(V / 1e-6).astype(np.int64)
    lookup = {tuple(k): i for i, k in enumerate(wkey)}
    horn = np.zeros(len(V), bool); claw = np.zeros(len(V), bool)
    for a, c in HORNS:
        for i in range(a, c + 1):
            horn[lookup[tuple(key[i])]] = True
    for grp in ([244, 571, 572, 573, 574], [585, 586, 587], [706, 707, 708]):
        for i in grp:
            claw[lookup[tuple(key[i])]] = True
    tri_horn = np.all(horn[F], axis=1); tri_claw = np.all(claw[F], axis=1)
    # body: smooth (Loop level 3), with UVs and a region class per triangle
    Fb, UVb = F[~tri_horn & ~tri_claw], UVc[~tri_horn & ~tri_claw]
    Vb, Fb, UVb = H.subdivide(V, Fb, UVb, 3)
    maps = H.body_maps()
    cuv = UVb.mean(1)
    cls = maps['cls'][np.clip((cuv[:, 1] * H.TEX).astype(int), 0, H.TEX - 1), np.clip((cuv[:, 0] * H.TEX).astype(int), 0, H.TEX - 1)]
    save('body', V=Vb, F=Fb, UV=UVb, cls=cls)
    Image.fromarray(np.clip(maps['albedo'], 0, 255).astype(np.uint8)).save(OUT / 'body_albedo.png')
    Hn = maps['H'] * (maps['cls'] != 0); Hn = (Hn - Hn.min()) / (np.ptp(Hn) + 1e-9)
    Image.fromarray((Hn * 255).astype(np.uint8)).save(OUT / 'body_height.png')
    # horns (with growth rings) and claws
    Hv, Hf, _ = H.horn_geometry(V, F, UVc, horn, 3)
    head_c = np.array([0.0, 3.62, 0.45]) * H.MIRROR
    dist = np.linalg.norm(Hv - head_c, axis=1); tpar = (dist - dist.min()) / (np.ptp(dist) + 1e-9)
    save('horns', V=Hv, F=Hf, t=tpar)
    Fc = F[tri_claw]; used = np.unique(Fc); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    Vc, Fc2, _ = H.subdivide(V[used], remap[Fc], UVc[tri_claw], 2)
    save('claws', V=Vc, F=Fc2)
    # heads: the smooth sculpt (rock comes from Blender), eyes, teeth
    from hd_build import tooth_mesh
    for expr in ('grin', 'closed'):
        Vh, Fh = HS.polygonize(expr)
        Vh = laplacian_smooth(Vh, Fh, 2, 0.4)
        save(f'head_{expr}', V=(Vh + HS.O) * H.MIRROR, F=Fh[:, ::-1])
        TP, TF = [], []
        for base, up, out, w, h, dpt, fang in HS.teeth_set(expr):
            P, Ft = tooth_mesh(base, up, out, w, h, dpt, fang=fang)
            TF.append(Ft + sum(len(p) for p in TP)); TP.append(P)
        TPv, TFv = np.vstack(TP), np.vstack(TF)
        TPv, TFv, _ = loop_subdivide(TPv, TFv); TPv, TFv, _ = loop_subdivide(TPv, TFv)
        tip = np.clip((TPv[:, 1] - (-0.17)) / 0.06, 0, 1)
        save(f'teeth_{expr}', V=(TPv + HS.O) * H.MIRROR, F=TFv[:, ::-1], t=tip)
    for i, e in enumerate(HS.EYES):
        P, Fe, UV = HS.eyeball(e)
        save(f'eye{i}', V=(P + HS.O) * H.MIRROR, F=Fe[:, ::-1], UV=UV)
    # stone spurs, from the smooth body surface
    Nb = vertex_normals(Vb, Fb)
    mpts = np.asarray(clip['frames'][0][1], float) * H.MIRROR
    SP, SN, SF, SC, SG = H.spurs(Vb, Nb, mpts)
    save('spurs', V=SP, F=SF)
    # the mace
    mg = H.mace_geometry(mpts)
    for nm, (P, Fm, UVm) in mg.items():
        kw = dict(V=P, F=np.asarray(Fm))
        if UVm is not None:
            kw['UVv'] = UVm
        save(f'mace_{nm}', **kw)


if __name__ == '__main__':
    main()
