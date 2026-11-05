"""DESIGN v8 (v7 + rounder top, jutting jaw, horn roots kept, neck textured from his own skin). DESIGN v7: the Golem's head on Molynx. Builds a new character dict from the v6 one (never writes to the repo).

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

def golem_head(reshape=True):
    v, tri, uv, tex = golem()
    sel = np.all((v[tri][:, :, 2] > 0.35) & (v[tri][:, :, 1] > 0.9) & (np.abs(v[tri][:, :, 0]) < 0.62), axis=1)
    t = tri[sel]; keep = np.unique(t)
    remap = -np.ones(len(v), int); remap[keep] = np.arange(len(keep))
    gv = v[keep].copy()
    if reshape:
        gv = reshape_head(gv)
    return gv, remap[t], uv[keep], tex

def reshape_head(gv):
    """v8: a rounder, less pointed top, and a heavier jaw that juts forward (Golem units, before scaling)."""
    gv = gv.copy()
    y, x = gv[:, 1], np.abs(gv[:, 0])
    top = y > 2.05
    # the dome: a rounded top instead of a point (centre a little higher than its shoulders)
    for i in np.nonzero(top)[0]:
        xi, front = x[i], gv[i, 2] > 0.6
        if xi < 0.1:
            gv[i, 1] = 2.27 if front else 2.33
        elif xi < 0.3:
            gv[i, 1] = 2.24
        else:
            gv[i, 1] = 2.15 if gv[i, 2] < 0.6 else 2.14
    # the jaw: everything below the mouth comes forward and a little down, most at the chin
    jaw = y < 1.235
    w = np.clip((1.235 - y) / 0.30, 0, 1)
    gv[jaw, 2] += 0.13 * w[jaw] * np.clip(1.2 - x[jaw] / 0.6, 0.2, 1)
    gv[jaw, 1] -= 0.05 * w[jaw]
    return gv

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

JAW_TAPER = 0.30
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
    jawk = np.clip((1.30 - v[rim, 1]) / 0.30, 0, 1)            # 0 above the mouth, 1 at the chin
    tp = taper - JAW_TAPER * jawk                                # v8: the jaw line turns in under the face
    back[:, 2] -= depth * (1 - 0.35 * jawk)
    back[:, 0] = cx + (back[:, 0] - cx) * tp
    back[:, 1] = cy + (back[:, 1] - cy) * (0.5 + 0.5 * tp) + 0.10 * jawk
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
                tri = tri[~np.any(face[tri], axis=1) | np.any(horn[tri], axis=1)]   # his face comes off; horn roots stay
            parts.append({**part, 'indices': tri.reshape(-1).astype(np.uint16)})
        parts.append({'texture': HEAD_TEX, 'uv': guv.astype(np.float32), 'indices': gt.reshape(-1).astype(np.uint16)})
        frames = []
        for fr in clip['frames']:
            body = np.asarray(fr[0])
            R, t = kabsch(ref[fit], body[fit])
            frames.append(list(fr) + [placed @ R.T + t])
        out[name] = {**clip, 'parts': parts, 'frames': frames}
    return out, gtex


# ------------------------------------------------------------------------------------------------ v8 build
SKIRT_TEX = 'custom_character/head_neck'     # the head's sides/back/underside: a strip of his own muscle skin

def rim_loops(tri):
    """Open edges chained into closed loops, following the triangles' winding."""
    nxt = {a: b for a, b in boundary_edges(tri)}
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            seen.add(cur); loop.append(cur); cur = nxt[cur]
        loops.append(loop)
    return loops

SKIRT_UV = None

def skirt_geometry(v, tri, depth, taper, scale):
    """The closing band as its own vertices/triangles, with strip UVs: u = distance round the rim (in final model units),
    v = 0 at the face's edge to 1 at the back. -> (points, triangles, (u_len, depth_len))"""
    global SKIRT_UV
    cx, cy = v[:, 0].mean(), v[:, 1].mean()
    pts, tris, uvs = [], [], []
    total = 0.0
    for loop in rim_loops(tri):
        ring = loop + [loop[0]]                                     # the seam vertex twice (u = 0 and u = loop length)
        front = v[ring].copy()
        back = front.copy()
        jawk = np.clip((1.30 - front[:, 1]) / 0.30, 0, 1)
        tp = taper - JAW_TAPER * jawk
        back[:, 2] -= depth * (1 - 0.35 * jawk)
        back[:, 0] = cx + (back[:, 0] - cx) * tp
        back[:, 1] = cy + (back[:, 1] - cy) * (0.5 + 0.5 * tp) + 0.10 * jawk
        topk = np.clip((front[:, 1] - 1.85) / 0.35, 0, 1)
        back[:, 1] -= 0.30 * topk                                   # the back of the head slopes down (no flat lid)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(front, axis=0), axis=1))] * scale
        base = len(pts and np.vstack(pts)) if pts else 0
        n = len(ring)
        pts += [front, back]
        uvs += [np.stack([total + seg, np.zeros(n)], 1), np.stack([total + seg, np.ones(n)], 1)]
        for k in range(n - 1):
            f0, f1, b0, b1 = base + k, base + k + 1, base + n + k, base + n + k + 1
            tris += [(f1, f0, b0), (f1, b0, b1)]
        total += seg[-1] + 0.05
    P_ = np.vstack(pts)
    UV = np.vstack(uvs)
    dlen = float(np.max(np.linalg.norm(P_[len(P_) // 2:] - P_[:len(P_) // 2], axis=1)) * scale) if len(rim_loops(tri)) == 1 else depth * scale
    SKIRT_UV = np.stack([UV[:, 0] / total, UV[:, 1]], 1)
    return P_, np.array(tris), (total, depth * scale)

def closest_uv(points, verts, tris, uvs):
    """UV of the closest point on the triangles (verts[tris]) for each point: barycentric on the nearest triangle."""
    A, B, C = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    out = np.zeros((len(points), 2))
    for k, p in enumerate(points):
        best, buv = np.inf, None
        ab, ac, ap = B - A, C - A, p - A
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        bb, cc, bc = (ab * ab).sum(1), (ac * ac).sum(1), (ab * ac).sum(1)
        den = np.maximum(bb * cc - bc * bc, 1e-12)
        v_ = np.clip((cc * d1 - bc * d2) / den, 0, 1); w_ = np.clip((bb * d2 - bc * d1) / den, 0, 1)
        s_ = v_ + w_; over = s_ > 1; v_[over] /= s_[over]; w_[over] /= s_[over]
        q = A + v_[:, None] * ab + w_[:, None] * ac
        d = np.linalg.norm(q - p, axis=1)
        j = int(np.argmin(d))
        u0, u1, u2 = uvs[tris[j, 0]], uvs[tris[j, 1]], uvs[tris[j, 2]]
        out[k] = u0 + v_[j] * (u1 - u0) + w_[j] * (u2 - u0)
    return out

def build_v8(data, p=P):
    head, horn = molynx_masks()
    gv, gt, guv, gtex = golem_head()
    n_face = len(gv)
    sv, st, STRIP = skirt_geometry(gv, gt, p['skirt'] / p['scale'], p['taper'], p['scale'])
    lo, hi = gv.min(0), gv.max(0); c = (lo + hi) / 2
    face_pl = (gv - c) * p['scale'] + np.array(p['centre'])
    skirt_pl = (sv - c) * p['scale'] + np.array(p['centre'])
    ref = np.asarray(data['stand']['frames'][0][0])
    skull = head & (ref[:, 2] < p['face_cut'])
    face = head & ~skull
    fit = head & (ref[:, 1] > 3.6)
    skirt_uv = SKIRT_UV
    build_v8.strip = STRIP                                       # (length round the rim, depth) in model units
    out = {}
    for name, clip in data.items():
        parts = []
        for k, part in enumerate(clip['parts']):
            tri = np.asarray(part['indices'], int).reshape(-1, 3)
            if part['texture'] == 'custom_character/body' and k == 0:
                tri = tri[~np.any(face[tri], axis=1) | np.any(horn[tri], axis=1)]
            parts.append({**part, 'indices': tri.reshape(-1).astype(np.uint16)})
        parts.append({'texture': HEAD_TEX, 'uv': guv.astype(np.float32), 'indices': gt.reshape(-1).astype(np.uint16)})
        parts.append({'texture': SKIRT_TEX, 'uv': skirt_uv.astype(np.float32), 'indices': st.reshape(-1).astype(np.uint16)})
        frames = []
        for fr in clip['frames']:
            b = np.asarray(fr[0])
            R, t = kabsch(ref[fit], b[fit])
            frames.append(list(fr) + [face_pl @ R.T + t, skirt_pl @ R.T + t])
        out[name] = {**clip, 'parts': parts, 'frames': frames}
    return out, gtex
