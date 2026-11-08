"""Soles as a low-poly DK2 creature: his face and curly hair from his photos, his red T-shirt, and below the
waist the Dark Angel's red loincloth, trousers and boots, holding the Dark Angel's sword.

The lower body and the sword are the game's own triangles (DarkAngel-Idle2, frame 0), kept with their UVs
on the game's DarkAngelFrnt / DarkAngelBak skins, so nothing of the game's art is copied: the files only
point at it. Everything above the waist is built here (lowpoly) on two new skins, Soles_Head (face, hair)
and Soles_Shirt (shirt, arms, neck).

Model space as lowpoly: Y up, Z forward, X to his LEFT; 1.0 tall to the top of his hair (the Dark Angel's
file height is scaled to the same).

LICENCE: GPL-3.0.
"""
from pathlib import Path

import numpy as np

from scripts.lowpoly import Part, Skeleton, loft, unit

SKELETON = Skeleton([
    ("root", None, (0, .5, 0), (0, .6, 0)),
    ("spine", "root", (0, .6, 0), (0, .68, 0)),
    ("chest", "spine", (0, .68, 0), (0, .8, -.005)),
    ("neck", "chest", (0, .8, -.005), (0, .845, 0)),
    ("head", "neck", (0, .845, 0), (0, 1.0, 0)),
    ("shoulderL", "chest", (.03, .785, -.01), (.105, .785, -.01)),
    ("armL", "shoulderL", (.105, .785, -.01), (.13, .63, -.02)),
    ("forearmL", "armL", (.13, .63, -.02), (.15, .495, .0)),
    ("handL", "forearmL", (.15, .495, .0), (.155, .43, .008)),
    ("thighL", "root", (.06, .48, 0), (.07, .26, .01)),
    ("shinL", "thighL", (.07, .26, .01), (.075, .05, -.01)),
    ("footL", "shinL", (.075, .05, -.01), (.08, .0, .08)),
])
SK = SKELETON
HEAD, SHIRT = "Soles_Head", "Soles_Shirt"
DA_FRONT, DA_BACK = "DarkAngelFrnt", "DarkAngelBak"

WAIST = .47            # the Dark Angel's triangles below this (and within HIP_X of the middle) are kept
HIP_X = .15
DA_HEIGHT = .706       # DarkAngel-Idle2 frame 0, in KMF units

# ------------------------------------------------------------------------------------------- upper body

TORSO = [((0, .45, 0), (.07, .1)), ((0, .52, 0), (.066, .1)), ((0, .6, .0), (.064, .098)),
         ((0, .68, .004), (.07, .108)), ((0, .74, .0), (.066, .112)), ((0, .778, -.006), (.054, .1)),
         ((0, .806, -.008), (.034, .05)), ((0, .826, -.008), (.028, .032)), ((0, .855, -.004), (.027, .029))]
TORSO_BONES = ["root", "spine", "chest", "neck"]


def build_torso(parts):
    p = Part("torso", SHIRT, SK)

    def shape(i, th):
        c = np.cos(th)
        if i == 5:
            return (0.9, 1.0) if abs(np.sin(th)) > 0.7 else (1.0, 1.0)
        return (1.0, 1.0) if c > -0.2 else (0.94, 1.0)

    loft(p, "torso", [c for c, _ in TORSO], [r for _, r in TORSO], 10, (0, 0, 1.0),
         lambda i, q: SK.auto(q, TORSO_BONES), shape=shape, u_offset=np.pi / 10)
    parts.append(p)
    return p


ARM_PATH = [(.108, .772, -.01), (.122, .705, -.016), (.13, .63, -.02), (.141, .56, -.01), (.15, .495, .0)]
ARM_RADII = [(.04, .04), (.037, .035), (.03, .028), (.027, .025), (.021, .02)]


def build_arms(parts):
    p = Part("arms", SHIRT, SK)
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        nm = (lambda b: b) if s > 0 else (lambda b: b[:-1] + "R")
        arm = [nm(b) for b in ("armL", "forearmL", "handL")]
        loft(p, "arm", np.array(ARM_PATH) * m, ARM_RADII, 7, (0, 0, 1.0),
             lambda i, q, arm=arm: {arm[0]: 1.0} if i == 0 else SK.auto(q, arm))
        w = np.array(ARM_PATH[-1]) * m
        h = [w + (0, -.006, .004), w + (.004 * s, -.055, .012)]
        loft(p, "hand", np.array(h), [(.017, .024), (.013, .02)], 5, (0, 0, 1.0),
             lambda i, q, a=arm: {a[2]: 1.0}, end_point=w + (.005 * s, -.08, .012))
        # the thumb
        loft(p, "hand", np.array([w + (-.012 * s, -.02, .014), w + (-.016 * s, -.045, .026)]),
             [(.007, .007), (.006, .006)], 4, (0, 0, 1.0), lambda i, q, a=arm: {a[2]: 1.0},
             end_point=w + (-.017 * s, -.056, .03))
    parts.append(p)
    return p


# head: his face is a front projection of his photo
HC = np.array([0, .905, .006])
HRX, HRY, HRZ = .05, .07, .058
HEAD_SECTIONS = [(.75, .82, .72, -.85), (.35, .98, .92, -1.0), (-.05, 1.0, .98, -1.0), (-.45, .9, .96, -.82),
                 (-.75, .7, .9, -.55), (-1.0, .34, .8, -.3)]
HEAD_RING = [(0, 1), (.55, .9), (.9, .55), (1, 0), (.7, -.65), (-.7, -.65), (-1, 0), (-.9, .55), (-.55, .9)]
HEAD_BACK = (3, 4, 5, 6)


def build_head(parts):
    p = Part("head", HEAD, SK)
    face, scalp = [], []
    for j, (h, w, f, b) in enumerate(HEAD_SECTIONS):
        fr, sr = [], {}
        for k, (x, z) in enumerate(HEAD_RING):
            q = HC + np.array([x * w * HRX, h * HRY, z * (f if z > 0 else -b) * HRZ])
            fr.append(p.add(q, (0, 0), "head", "face"))
            if k in HEAD_BACK:
                sr[k] = p.add(q, ((k - 3) / 3, j / (len(HEAD_SECTIONS) - 1)), "head", "scalp")
        face.append(fr)
        scalp.append(sr)
    n = len(HEAD_RING)
    backq = HEAD_BACK[:-1]
    vid = lambda j, k, back: scalp[j][k] if back else face[j][k]
    for i in range(len(face) - 1):
        mid = HC + (0, (HEAD_SECTIONS[i][0] + HEAD_SECTIONS[i + 1][0]) / 2 * HRY, 0)
        for k in range(n):
            back = k in backq
            a, b = vid(i, k, back), vid(i, (k + 1) % n, back)
            c, d = vid(i + 1, k, back), vid(i + 1, (k + 1) % n, back)
            p.tri(a, b, c, inside=mid)
            p.tri(b, d, c, inside=mid)
    top = HC + (0, HRY, -.004)
    tf, ts = p.add(top, (0, 0), "head", "face"), p.add(top, (.5, 0), "head", "scalp")
    chin = HC + (0, -HRY * 1.02, .034)
    cf, cs = p.add(chin, (0, 0), "head", "face"), p.add(chin, (.5, 1), "head", "scalp")
    last = len(face) - 1
    for k in range(n):
        back = k in backq
        p.tri(vid(0, k, back), vid(0, (k + 1) % n, back), ts if back else tf, inside=HC)
        p.tri(vid(last, (k + 1) % n, back), vid(last, k, back), cs if back else cf, inside=HC - (0, HRY * .7, 0))
    # nose and ears
    tip = HC + (0, -.026, HRZ * .98 + .016)
    b1, b2, b3 = HC + (.011, -.03, HRZ * .95), HC + (-.011, -.03, HRZ * .95), HC + (0, .006, HRZ * .97)
    ids = [p.add(q, (0, 0), "head", "face") for q in (b1, b2, b3, tip)]
    inner = HC + (0, -.012, HRZ * .8)
    p.tri(ids[0], ids[2], ids[3], inside=inner)
    p.tri(ids[2], ids[1], ids[3], inside=inner)
    p.tri(ids[1], ids[0], ids[3], inside=inner)
    for s in (1, -1):
        a, b = HC + (s * HRX * .97, .012, -.004), HC + (s * HRX * .95, -.03, .004)
        t = HC + (s * (HRX + .014), -.004, -.016)
        for side in (1, -1):
            ids = [p.add(q, (0, 0), "head", "face") for q in (a, b, t)]
            p.tri_facing(*ids, np.array([s * side, 0, 0]))
    # face UVs: the front projection (set by the texture module, which knows the photo)
    parts.append(p)
    return p


# hair: a full mop of curls, a fringe over the forehead, curls over the ears and the nape
HAIR = [  # height, front, side, back radius, open (half-angle at the face), curl depth of the lower edge
    (1.0, .03, .035, .035, 0, 0), (.984, .064, .07, .074, 0, 0), (.962, .07, .078, .084, 28, 0),
    (.925, .068, .08, .088, 52, .0), (.88, .064, .078, .086, 70, .012), (.845, .06, .072, .08, 78, .018)]
HAIR_COLS = 10


def build_hair(parts):
    p = Part("hair", HEAD, SK)
    grid = []
    rng = np.random.default_rng(7)
    for j, (y, rf, rs, rb, op, curl) in enumerate(HAIR):
        row = []
        for k in range(HAIR_COLS):
            s = k / (HAIR_COLS - 1)
            a = np.radians(op + s * (360 - 2 * op))
            ca, sa = np.cos(a), np.sin(a)
            r = (rf if ca > 0 else rb) * (1 + .06 * rng.standard_normal() * (j > 0))
            yy = y - curl * (k % 2) * 1.0
            q = np.array([HC[0] + rs * sa * (1 + .05 * rng.standard_normal() * (j > 0)), yy,
                          HC[2] - .004 + r * ca])
            row.append(p.add(q, (abs(s - .5) * 2, j / (len(HAIR) - 1)), "head", "hair", inner=0.0))
        grid.append(row)
    centre = lambda j: np.array([HC[0], HAIR[j][0], HC[2] - .004])
    for j in range(len(HAIR) - 1):
        mid = (centre(j) + centre(j + 1)) / 2
        for k in range(HAIR_COLS - 1):
            a, b = grid[j][k], grid[j][k + 1]
            c, d = grid[j + 1][k], grid[j + 1][k + 1]
            p.tri(a, b, c, inside=mid)
            p.tri(b, d, c, inside=mid)
    t = p.add(HC + (0, 1.006 - HC[1], -.004), (1.0, 0.0), "head", "hair", inner=0.0)
    for k in range(HAIR_COLS - 1):
        p.tri(grid[0][k], grid[0][k + 1], t, inside=HC)
    # (his fringe is in the photo projected on the forehead)
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- the Dark Angel's parts

def dark_angel(meshes_dir):
    """-> (lower body: [(texture, points, uvs, tris)], sword: (texture, points, uvs, tris)) in model space."""
    from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
    m = parse_kmf((Path(meshes_dir) / "DarkAngel-Idle2.kmf").read_bytes())
    frame = evaluate_anim_frame(m.anim, 0)
    lower, sword = [], None
    for i, (sp, pts) in enumerate(zip(m.anim.sprites, frame)):
        q = np.stack([-pts[:, 0], -pts[:, 2], pts[:, 1]], 1) / DA_HEIGHT
        uv = np.array([(v.uv[0] / 32768, v.uv[1] / 32768) for v in sp.vertices])
        tex = m.materials[sp.material].textures[0]
        T = np.array(sp.triangles[0])
        if i == 3:
            sword = (tex, q, uv, T)
            continue
        if i == 0:
            continue                                         # the wings
        c = q[T].mean(1)
        keep = T[(c[:, 1] < WAIST) & (np.abs(c[:, 0]) < HIP_X)]
        if len(keep):
            lower.append((tex, q, uv, keep))
    return lower, sword


def build_from_game(parts, meshes_dir):
    lower, sword = dark_angel(meshes_dir)
    legs = ["root", "thighL", "shinL", "footL", "thighR", "shinR", "footR"]
    for k, (tex, q, uv, T) in enumerate(lower):
        p = Part(f"lower{k}", tex, SK)
        used = sorted(set(T.ravel()))
        remap = {}
        for v in used:
            remap[v] = p.add(q[v], uv[v], SK.auto(q[v], legs), "game")
        for a, b, c in T:
            p.tris.append((remap[a], remap[b], remap[c]))
        p.game_uv = True
        parts.append(p)
    tex, q, uv, T = sword
    p = Part("sword", tex, SK)
    for v in range(len(q)):
        p.add(q[v], uv[v], "handL", "game")
    p.tris = [tuple(t) for t in T]
    p.game_uv = True
    p.grip = q[:4].mean(0)
    parts.append(p)
    return parts


def build_soles(meshes_dir):
    parts = []
    build_torso(parts)
    build_arms(parts)
    build_head(parts)
    build_hair(parts)
    build_from_game(parts, meshes_dir)
    return parts


# a relaxed stance holding the sword in his left hand, its blade up and out
STANCE = {"armL": (-30, 0, 8), "forearmL": (-50, 0, 0), "armR": (0, 0, -6), "forearmR": (-10, 0, 0),
          "head": (0, -4, 0)}
