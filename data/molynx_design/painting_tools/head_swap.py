"""DESIGN v7: the Golem's head on Molynx. Builds a new character dict from the v6 one (never writes to the repo).

The Golem's face/dome (its front upper body above y 0.9, |x| < 0.62, z > 0.35) becomes a new part with its own texture.
Molynx keeps his neck, the back of his skull and his horns; his face triangles are taken out. In every frame of every clip
the new head follows his head by a rigid fit (Kabsch) of his skull vertices against the reference pose (stand, frame 0)."""
import sys; sys.path.insert(0, '/home/claude/molynx_design/tools')
from common import *
from golem_src import golem
from scripts.behemoth_rig import behemoth_rest, behemoth_labels, CLUB_VERTICES
from scripts.character_paint import HORNS

HEAD_TEX = 'custom_character/golem_head'
P = dict(scale=0.44, centre=(0.05, 3.62, 0.50), face_cut=0.30, skirt=0.44, taper=1.0)    # placement (tuned by eye)

def molynx_masks():
    rest, idx, _ = behemoth_rest(UNIT); lab = behemoth_labels(rest)
    horn = np.zeros(len(rest), bool)
    for a, c in HORNS: horn[a:c + 1] = True
    hand = (lab == 1) & (rest[:, 2] > 0.8)
    head = (lab == 1) & (np.arange(len(rest)) >= CLUB_VERTICES) & ~hand & ~horn
    return head, horn

def golem_head():
    v, tri, uv, tex = golem()
    sel = np.all((v[tri][:, :, 2] > 0.35) & (v[tri][:, :, 1] > 0.9) & (np.abs(v[tri][:, :, 0]) < 0.62), axis=1)
    t = tri[sel]; keep = np.unique(t)
    remap = -np.ones(len(v), int); remap[keep] = np.arange(len(keep))
    return v[keep], remap[t], uv[keep], tex

def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    H = (a - ca).T @ (b - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca

def boundary_edges(tri):
    from collections import Counter
    cnt = Counter()
    for a, b, c in tri:
        for e in ((a, b), (b, c), (c, a)):
            cnt[tuple(sorted(e))] += 1
    out = []
    for a, b, c in tri:                                       # keep each open edge in its triangle's winding order
        for e in ((a, b), (b, c), (c, a)):
            if cnt[tuple(sorted(e))] == 1:
                out.append(e)
    return out

SKIN_BOX = (200 / 256, 76 / 256, 250 / 256, 140 / 256)    # an empty (black) area of the Golem texture: the head painter fills it with skin

def add_skirt(v, tri, uv, depth, taper):
    """Close the shell: every open edge is extruded straight back (-z) by `depth`, narrowing towards the head's centre line,
    so the face becomes a solid head that runs back into the skull and down into the neck. The skirt has its own vertices
    (same positions as the rim) so it can take plain skin from SKIN_BOX instead of stretching the rim's rock."""
    edges = boundary_edges(tri)
    rim = sorted({i for e in edges for i in e})
    n = len(v)
    front_id = {i: n + k for k, i in enumerate(rim)}
    back_id = {i: n + len(rim) + k for k, i in enumerate(rim)}
    cx, cy = v[:, 0].mean(), v[:, 1].mean()
    back = v[rim].copy()
    back[:, 2] -= depth
    back[:, 0] = cx + (back[:, 0] - cx) * taper
    back[:, 1] = cy + (back[:, 1] - cy) * (0.5 + 0.5 * taper)
    pts = np.vstack([v[rim], back])
    a = pts[:, 0] + pts[:, 2]                                 # planar skin mapping (the skin is plain, any projection works)
    b = pts[:, 1]
    u0, v0, u1, v1 = SKIN_BOX
    su = u0 + (a - a.min()) / max(np.ptp(a), 1e-9) * (u1 - u0)
    sv = v0 + (b.max() - b) / max(np.ptp(b), 1e-9) * (v1 - v0)
    v2 = np.vstack([v, pts]); uv2 = np.vstack([uv, np.stack([su, sv], 1)])
    quads = []
    for a_, b_ in edges:
        quads += [(front_id[b_], front_id[a_], back_id[a_]), (front_id[b_], back_id[a_], back_id[b_])]
    return v2, np.vstack([tri, np.array(quads)]), uv2

def place(gv, p=P):
    lo, hi = gv.min(0), gv.max(0)
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
    return (gv - c) * p['scale'] + np.array(p['centre'])

def place_skirted(gv, n_face, p=P):
    lo, hi = gv[:n_face].min(0), gv[:n_face].max(0)              # placement measured on the face itself, not the skirt
    c = (lo + hi) / 2
    return (gv - c) * p['scale'] + np.array(p['centre'])

def build(data, p=P, head_verts=None):
    head, horn = molynx_masks()
    gv, gt, guv, gtex = golem_head()
    gv, gt, guv = add_skirt(gv, gt, guv, p['skirt'] / p['scale'], p['taper'])
    placed = place_skirted(gv, len(golem_head()[0]), p) if head_verts is None else head_verts
    ref = np.asarray(data['stand']['frames'][0][0])
    skull = head & (ref[:, 2] < p['face_cut'])                  # what stays of his head (back of the skull): drives the fit
    face = head & ~skull
    fit = head & (ref[:, 1] > 3.6)                              # the upper skull moves rigidly (the neck below it bends)
    out = {}
    for name, clip in data.items():
        parts = []
        for k, part in enumerate(clip['parts']):
            tri = np.asarray(part['indices'], int).reshape(-1, 3)
            if part['texture'] == 'custom_character/body' and k == 0:
                tri = tri[~np.any(face[tri], axis=1)]              # his face comes off
            parts.append({**part, 'indices': tri.reshape(-1).astype(np.uint16)})
        parts.append({'texture': HEAD_TEX, 'uv': guv.astype(np.float32), 'indices': gt.reshape(-1).astype(np.uint16)})
        frames = []
        for fr in clip['frames']:
            body = np.asarray(fr[0])
            R, t = kabsch(ref[fit], body[fit])
            frames.append(list(fr) + [placed @ R.T + t])
        out[name] = {**clip, 'parts': parts, 'frames': frames}
    return out, gtex
