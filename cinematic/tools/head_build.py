"""Builds the cinematic head meshes (one expression) in the game data's coordinates:
rocky skin (vertex coloured), eyeballs (textured, glowing), teeth. -> list of mesh dicts."""
import numpy as np
from scipy import ndimage as ndi
import head_sdf as HS
from rock3d import rock
from subdiv import vertex_normals, laplacian_smooth

# linear-light colours (vertex colours are not gamma decoded): sRGB ~ (175,45,26), (110,24,14), (28,8,6)
RED_HI = np.array([0.215, 0.014, 0.0065]); RED_LO = np.array([0.075, 0.005, 0.003]); CRACK = np.array([0.011, 0.0024, 0.0018])


def skin_colour(P, r, extra_dark=None):
    tone = r['tone'][:, None]
    base = RED_LO + (RED_HI - RED_LO) * (0.55 + 0.45 * tone)
    # warmer, brighter plate centres; ash-dark cracks
    col = base * (1 - 0.92 * r['crack'][:, None]) + CRACK * r['crack'][:, None]
    grain = np.sin(P[:, 0] * 900) * np.sin(P[:, 1] * 770) * np.sin(P[:, 2] * 830)
    col = col * (0.93 + 0.07 * grain[:, None])
    if extra_dark is not None:
        col = col * (1 - extra_dark[:, None]) + np.array([0.012, 0.0015, 0.001]) * extra_dark[:, None]
    return np.clip(col, 0, 1)


def build_head(expr):
    V, F = HS.polygonize(expr)
    V = laplacian_smooth(V, F, 2, 0.4)
    N = vertex_normals(V, F)
    r = rock(V, cell=0.06, seed=21, crack_w=0.11, fine_cell=0.026)
    # keep the eyelids, the nose and the lips calmer (cracks there would hide the features)
    near_eye = np.min([np.linalg.norm(V - e, axis=1) for e in HS.EYES], axis=0)
    calm = np.clip((near_eye - (HS.EYE_R + 0.02)) / 0.035, 0.1, 1)
    nose_d = np.linalg.norm((V - np.array([0, -0.03, 0.265])) / np.array([1.0, 0.6, 1.0]), axis=1)
    calm = calm * np.clip((nose_d - 0.03) / 0.03, 0.25, 1)
    mouth_d = HS.ellipsoid(V, np.array([0, -0.145, 0.215]), np.array([0.13, 0.045, 0.09]))
    calm = calm * np.clip(mouth_d / 0.02, 0.35, 1)
    disp = r['height'] * 0.0055 * calm
    V2 = V + N * disp[:, None]
    V2 = laplacian_smooth(V2, F, 1, 0.2)
    N2 = vertex_normals(V2, F)
    # creases, furrows, folds and sockets go dark: concavity of the sculpt before the rock
    Vs = laplacian_smooth(V, F, 6, 0.5)
    conc = np.sum((Vs - V) * N, 1)                       # > 0 where the surface dips in
    dark = np.clip(conc / 0.006, 0, 1) * 0.85
    if expr == 'grin':
        cav = HS.ellipsoid(V, np.array([0, -0.142, 0.21]), np.array([0.12, 0.026, 0.085]))
        dark = np.maximum(dark, np.clip(1 - (cav + 0.006) / 0.018, 0, 1))
    sock = np.min([HS.sphere(V, e, HS.EYE_R + 0.03) for e in HS.EYES], axis=0)
    dark = np.maximum(dark, np.clip(1 - (sock + 0.004) / 0.02, 0, 1) * 0.7)
    col = skin_colour(V, r, dark)
    # the deepest cracks smoulder (an ember glow inside the rock), never on the features
    g0 = np.clip((r['crack'] - 0.9) / 0.1, 0, 1) * np.clip(calm * 1.2, 0, 1) * (1 - np.clip(dark * 1.5, 0, 1))
    Fe = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    for _ in range(3):                                    # smooth along the surface: a continuous seam, not dots
        acc = np.zeros(len(V)); cnt = np.zeros(len(V)); np.add.at(acc, Fe[:, 0], g0[Fe[:, 1]]); np.add.at(cnt, Fe[:, 0], 1)
        g0 = 0.5 * g0 + 0.5 * acc / np.maximum(cnt, 1)
    glow = np.clip(g0 * 1.6, 0, 1) * 0.6
    meshes = [dict(name='head_skin', P=V2 + HS.O, N=N2, F=F, color=col, kind='skin', glow=glow)]
    # eyes
    col_t, em_t = HS.eye_texture()
    for i, e in enumerate(HS.EYES):
        P, Fe, UV = HS.eyeball(e)
        meshes.append(dict(name=f'eye{i}', P=P + HS.O, N=vertex_normals(P, Fe), F=Fe, UV=UV, kind='eye'))
    meshes[-1]['tex'] = (col_t, em_t)
    # teeth
    from hd_build import tooth_mesh
    from subdiv import loop_subdivide
    TP, TF = [], []
    for base, up, out, w, h, dpt, fang in HS.teeth_set(expr):
        P, Ft = tooth_mesh(base, up, out, w, h, dpt, fang=fang)
        TF.append(Ft + sum(len(p) for p in TP)); TP.append(P)
    TPv, TFv = np.vstack(TP), np.vstack(TF)
    TPv, TFv, _ = loop_subdivide(TPv, TFv)
    TPv, TFv, _ = loop_subdivide(TPv, TFv)
    # a little enamel wear
    TN = vertex_normals(TPv, TFv)
    # yellowed, stained teeth: dark brown at the gum, old ivory at the points (linear light)
    tcol = np.tile(np.array([[0.55, 0.40, 0.17]]), (len(TPv), 1))
    tip = np.clip((TPv[:, 1] - (-0.17)) / 0.06, 0, 1)
    tcol = tcol * (0.25 + 0.75 * tip[:, None] ** 0.7)
    meshes.append(dict(name='teeth', P=TPv + HS.O, N=TN, F=TFv, color=tcol, kind='teeth'))
    return meshes
