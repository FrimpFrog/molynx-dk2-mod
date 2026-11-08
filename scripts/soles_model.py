"""Soles as a low-poly DK2 creature, built like the game's own humans (the Thief, the Rogue, the Guard): one
body whose parts meet (the legs start inside the hips under a belt and loincloth, the arms inside the
shoulders, the neck inside the head), a sculpted head whose hair is part of its shape, front and back
projected skins like the game's, and small armour.

His face is painted in his likeness on the front of the head (soles_texture). He wears
his red T-shirt with an original flame emblem, a studded leather baldric, a dark steel pauldron on his
sword arm, leather bracers, a belt, a red loincloth in the Dark Angel's style over black leather trousers
and boots, and holds the Dark Angel's sword (the game's own triangles, still pointing at DarkAngelFrnt).

Model space as lowpoly: Y up, Z forward, X to his LEFT; 1.0 tall to the top of his hair.

LICENCE: GPL-3.0.
"""
from pathlib import Path

import numpy as np

from scripts.lowpoly import Part, Skeleton, unit

SK = Skeleton([
    ("root", None, (0, .5, 0), (0, .6, 0)),
    ("spine", "root", (0, .6, 0), (0, .69, 0)),
    ("chest", "spine", (0, .69, 0), (0, .8, -.006)),
    ("neck", "chest", (0, .8, -.006), (0, .85, 0)),
    ("head", "neck", (0, .85, 0), (0, 1.0, 0)),
    ("shoulderL", "chest", (.03, .78, -.01), (.108, .772, -.012)),
    ("armL", "shoulderL", (.108, .772, -.012), (.128, .628, -.02)),
    ("forearmL", "armL", (.128, .628, -.02), (.146, .49, .0)),
    ("handL", "forearmL", (.146, .49, .0), (.15, .42, .008)),
    ("thighL", "root", (.062, .49, .0), (.07, .265, .012)),
    ("shinL", "thighL", (.07, .265, .012), (.074, .055, -.01)),
    ("footL", "shinL", (.074, .055, -.01), (.078, .008, .08)),
])
HEAD, BODY, LIMBS = "Soles_Head", "Soles_Body", "Soles_Limbs"
DA_HEIGHT = .706


def nm_for(s):
    return (lambda b: b) if s > 0 else (lambda b: b[:-1] + "R")


def ring_pts(c, t, up, rx, rz, n, phase=0.0):
    """n points round centre c in the plane normal to t: rx across (t x up'), rz along up'."""
    t = unit(t)
    u = unit(np.asarray(up, float) - np.dot(up, t) * t)
    side = np.cross(t, u)
    return [np.asarray(c, float) + side * rx * np.sin(2 * np.pi * k / n + phase)
            + u * rz * np.cos(2 * np.pi * k / n + phase) for k in range(n)]


def tube(p, island, rings, weights, inside_pts, cap0=None, cap1=None):
    """Stitch rings into a tube. UVs: u mirrored round the ring (0 at its first point, 1 opposite),
    v along the rings."""
    n = len(rings[0])
    L = max(len(rings) - 1, 1)
    ids = [[p.add(q, (min(k, n - k) / (n / 2), i / L), weights(i, q), island) for k, q in enumerate(r)]
           for i, r in enumerate(rings)]
    for i in range(len(rings) - 1):
        mid = (np.asarray(inside_pts[i]) + np.asarray(inside_pts[i + 1])) / 2
        for k in range(n):
            a, b = ids[i][k], ids[i][(k + 1) % n]
            c, d = ids[i + 1][k], ids[i + 1][(k + 1) % n]
            p.tri(a, b, c, inside=mid)
            p.tri(b, d, c, inside=mid)
    for cap, ring in ((cap0, 0), (cap1, len(rings) - 1)):
        if cap is not None:
            inner = np.asarray(inside_pts[ring]) + (np.asarray(inside_pts[ring]) - np.asarray(cap)) * .5
            ci = p.add(cap, (.5, ring / L), weights(ring, cap), island)
            for k in range(n):
                p.tri(ids[ring][k], ids[ring][(k + 1) % n], ci, inside=inner)
    return ids


# ------------------------------------------------------------------------------------------- torso

# rings: y, half width, front, back, centre z
TORSO = [(.445, .09, .064, .068, .0), (.475, .096, .066, .07, .0), (.51, .094, .063, .066, .0),
         (.56, .088, .062, .062, .004), (.62, .094, .066, .062, .008), (.68, .106, .074, .06, .01),
         (.735, .114, .072, .058, .004), (.772, .108, .058, .05, -.004), (.8, .062, .042, .04, -.006),
         (.822, .034, .031, .031, -.006), (.87, .03, .029, .029, -.002)]
TORSO_N = 12


def torso_ring(y, hw, f, b, cz, n=TORSO_N):
    r = []
    for k in range(n):
        a = 2 * np.pi * k / n
        ca, sa = np.cos(a), np.sin(a)
        e = .75 if y > .6 else .92                     # squarer chest and shoulders, rounder waist
        r.append(np.array([hw * np.sign(sa) * abs(sa) ** e, y, cz + (f if ca > 0 else b) * np.sign(ca) * abs(ca) ** e]))
    return r


def build_torso(parts):
    p = Part("torso", BODY, SK)
    rings = [torso_ring(*row) for row in TORSO]
    bones = ["root", "spine", "chest", "neck"]
    tube(p, "torso", rings, lambda i, q: SK.auto(q, bones), [np.array([0, r[0], r[4]]) for r in TORSO],
         cap0=np.array([0, .43, 0]))
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- arms

ARM = [((.096, .772, -.012), .047, .043), ((.116, .735, -.014), .044, .04), ((.124, .68, -.018), .036, .034),
       ((.128, .628, -.02), .03, .029), ((.135, .58, -.014), .031, .029), ((.142, .53, -.006), .027, .024),
       ((.146, .49, .0), .02, .018)]


def build_arms(parts):
    p = Part("arms", LIMBS, SK)
    for s in (1, -1):
        nm = nm_for(s)
        bones = [nm("armL"), nm("forearmL"), nm("handL")]
        cs = [np.array(c) * (s, 1, 1) for c, _, _ in ARM]
        rings = [ring_pts(cs[i], cs[min(i + 1, 6)] - cs[max(i - 1, 0)], (0, 0, 1), rx, rz, 8)
                 for i, (_, rx, rz) in enumerate(ARM)]
        tube(p, "arm", rings, lambda i, q, b=bones: {b[0]: 1.0} if i <= 1 else SK.auto(q, b), cs)
        w = cs[-1]
        hand = [w + (0, -.004, .002), w + (.002 * s, -.03, .008), w + (.004 * s, -.058, .01)]
        rh = [(.021, .014), (.024, .013), (.019, .011)]
        rings = [ring_pts(hand[i], hand[min(i + 1, 2)] - hand[max(i - 1, 0)], (s, 0, 0), rh[i][1], rh[i][0], 6)
                 for i in range(3)]
        tube(p, "hand", rings, lambda i, q, b=bones: {b[2]: 1.0}, hand, cap1=w + (.005 * s, -.074, .01))
        th = [w + (-.01 * s, -.014, .014), w + (-.015 * s, -.036, .026)]
        rings = [ring_pts(q, th[1] - th[0], (0, 0, 1), .0075, .0075, 4) for q in th]
        tube(p, "hand", rings, lambda i, q, b=bones: {b[2]: 1.0}, th, cap1=w + (-.017 * s, -.048, .03))
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- legs

LEG = [((.06, .505, .004), .062, .06), ((.064, .43, .008), .058, .056), ((.067, .35, .012), .048, .046),
       ((.07, .27, .014), .039, .038), ((.071, .2, .01), .044, .043), ((.072, .13, .0), .037, .037),
       ((.074, .062, -.008), .031, .031)]
BOOT_TOP = .2


def build_legs(parts):
    p = Part("legs", LIMBS, SK)
    for s in (1, -1):
        nm = nm_for(s)
        bones = [nm("thighL"), nm("shinL"), nm("footL")]
        cs = [np.array(c) * (s, 1, 1) for c, _, _ in LEG]
        rings = [ring_pts(cs[i], cs[min(i + 1, 6)] - cs[max(i - 1, 0)], (0, 0, 1), rx, rz, 8)
                 for i, (_, rx, rz) in enumerate(LEG)]
        tube(p, "leg", rings, lambda i, q, b=bones: {"root": .5, b[0]: .5} if i == 0 else SK.auto(q, b), cs)
        a = cs[-1]
        foot = [a + (0, -.03, -.008), a + (.002 * s, -.044, .03), a + (.004 * s, -.052, .07)]
        rf = [(.033, .03), (.03, .018), (.025, .011)]
        rings = [ring_pts(foot[i], foot[min(i + 1, 2)] - foot[max(i - 1, 0)], (0, 1, .4), rf[i][0], rf[i][1], 6)
                 for i in range(3)]
        tube(p, "foot", rings, lambda i, q, b=bones: {b[2]: 1.0}, foot, cap1=a + (.005 * s, -.054, .1))
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- loincloth

def flap(p, island, top_y, bot_y, z0, z1, w0, w1, front=True, cols=3, rows=3):
    """A cloth panel hanging from the belt, curved round the hips and flaring out at the hem."""
    pts, uvs, ws = [], [], []
    sgn = 1 if front else -1
    for j in range(rows + 1):
        v = j / rows
        y = top_y + (bot_y - top_y) * v
        hw = w0 + (w1 - w0) * v
        z = z0 + (z1 - z0) * v
        for k in range(cols + 1):
            u = k / cols
            x = (u - .5) * 2 * hw
            pts.append(np.array([x, y, sgn * (z - .03 * (2 * u - 1) ** 2)]))
            uvs.append((u, v))
            ws.append({"root": 1 - .7 * v, ("thighL" if x > 0 else "thighR"): .7 * v + 1e-6})
    W = cols + 1
    tris = []
    for j in range(rows):
        for k in range(cols):
            a, b, c, d = j * W + k, j * W + k + 1, (j + 1) * W + k, (j + 1) * W + k + 1
            tris += [(a, b, c), (b, d, c)]
    normal = np.array([0, 0, float(sgn)])
    for side, isl in ((1, island), (-1, island + "_in")):
        ids = [p.add(q, uv, w, isl, face=side) for q, uv, w in zip(pts, uvs, ws)]
        for a, b, c in tris:
            p.tri_facing(ids[a], ids[b], ids[c], normal * side)


def build_loincloth(parts):
    p = Part("loincloth", LIMBS, SK)
    flap(p, "cloth", .478, .3, .07, .088, .064, .07, front=True)
    flap(p, "cloth", .478, .29, .072, .092, .07, .078, front=False)
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- armour

def build_armour(parts):
    p = Part("armour", LIMBS, SK)
    # a dark steel pauldron on the sword arm: three overlapping plates down the shoulder
    for i, (dy, r) in enumerate(((0, .054), (-.026, .05), (-.05, .044))):
        c = np.array([.112 + .008 * i, .79 + dy, -.012])
        e1 = np.array([0, 0, 1.0])
        e2 = unit(np.array([1, -.5 - .3 * i, 0]))
        nrm = unit(np.cross(e1, e2))
        if nrm[1] < 0:
            nrm = -nrm
        ring = [c + np.cos(a) * e1 * r * .95 + np.sin(a) * e2 * r * .75 for a in np.linspace(0, 2 * np.pi, 9)[:-1]]
        w = {"shoulderL": .3, "armL": .7}
        ids = [p.add(q - nrm * .006, (min(k, 8 - k) / 4, (i + .02) / 3), w, "pauldron", along=0.0)
               for k, q in enumerate(ring)]
        apex = p.add(c + nrm * .018, (.5, (i + .98) / 3), w, "pauldron", along=1.0)
        for k in range(8):
            p.tri(ids[k], ids[(k + 1) % 8], apex, inside=c - nrm * .05)
    # leather bracers on both forearms
    for s in (1, -1):
        nm = nm_for(s)
        a0 = np.array(ARM[4][0]) * (s, 1, 1)
        a1 = np.array(ARM[6][0]) * (s, 1, 1) + (0, .008, 0)
        rings = [ring_pts(a0, a1 - a0, (0, 0, 1), .036, .034, 8), ring_pts(a1, a1 - a0, (0, 0, 1), .027, .025, 8)]
        tube(p, "bracer", rings, lambda i, q, b=nm("forearmL"): {b: 1.0}, [a0, a1])
    # the belt round the hips, over the shirt's hem and the top of the loincloth
    rings = [torso_ring(y, .1, .072, .076, 0) for y in (.462, .5)]
    tube(p, "belt", rings, lambda i, q: {"root": 1.0}, [np.array([0, .462, 0]), np.array([0, .5, 0])])
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- head

HC = np.array([0, .9, .008])
# rows: y, half width, front, back: the skull and, above the hairline and round the back, his hair
HEAD_ROWS = [
    (.838, .02, .03, .018), (.846, .036, .05, .044), (.862, .046, .058, .06), (.88, .053, .062, .066),
    (.897, .055, .064, .066), (.912, .056, .064, .07), (.927, .058, .063, .074), (.944, .059, .061, .076),
    (.96, .056, .055, .072), (.975, .048, .044, .062), (.988, .032, .028, .042)]
HEAD_TOP = (0, .996, -.01)
HEAD_N = 14
HAIRLINE = .942
EAR_Y = .893


def build_head(parts):
    p = Part("head", HEAD, SK)
    rng = np.random.default_rng(3)
    rings = []
    for y, hw, f, b in HEAD_ROWS:
        r = []
        for k in range(HEAD_N):
            a = 2 * np.pi * k / HEAD_N
            ca, sa = np.cos(a), np.sin(a)
            e = .82 if y > .87 else 1.0
            q = np.array([hw * np.sign(sa) * abs(sa) ** e, y, HC[2] + (f if ca > 0 else b) * np.sign(ca) * abs(ca) ** e])
            # the brow juts over the eyes; cheekbones; a narrower jaw at the front
            if ca > .5:
                q[2] += .005 * np.exp(-((y - .922) / .007) ** 2)
            if .3 < ca < .9:
                q[0] += np.sign(sa) * .004 * np.exp(-((y - .893) / .01) ** 2)
            hair = (y > HAIRLINE) or (ca < .3 and y > .846)
            if hair:                                       # wavy lumps in his hair
                d = q - np.array([HC[0], y, HC[2]])
                q = q + d / np.linalg.norm(d) * .005 * rng.standard_normal()
            r.append(q)
        rings.append(r)
    tube(p, "head", rings, lambda i, q: "head", [np.array([0, y, HC[2]]) for y, *_ in HEAD_ROWS],
         cap0=np.array([0, .834, .024]), cap1=np.array(HEAD_TOP))
    # the nose
    z0 = HC[2]
    pts = dict(br=(0, .916, z0 + .066), tip=(0, .884, z0 + .084), wl=(.013, .877, z0 + .066),
               wr=(-.013, .877, z0 + .066), base=(0, .873, z0 + .072), sl=(.011, .906, z0 + .064),
               sr=(-.011, .906, z0 + .064))
    ids = {k: p.add(np.array(q), (0, 0), "head", "head") for k, q in pts.items()}
    inner = np.array([0, .89, z0 + .05])
    for a, b, c in (("br", "sl", "tip"), ("br", "tip", "sr"), ("sl", "wl", "tip"), ("sr", "tip", "wr"),
                    ("wl", "base", "tip"), ("wr", "tip", "base")):
        p.tri(ids[a], ids[b], ids[c], inside=inner)
    # ears, half under his hair
    for s in (1, -1):
        c = np.array([s * .057, EAR_Y, HC[2] - .004])
        pts = [c + (0, .016, .004), c + (s * .012, .008, -.006), c + (s * .01, -.012, -.004), c + (0, -.016, .006)]
        for side in (1, -1):
            ids = [p.add(q, (0, 0), "head", "head") for q in pts]
            p.tri_facing(ids[0], ids[1], ids[2], np.array([s * side, 0, 0]))
            p.tri_facing(ids[0], ids[2], ids[3], np.array([s * side, 0, 0]))
    parts.append(p)
    return p


def split_front_back(part, front_island, back_island, uv_front, uv_back, z_split=0.0):
    """Each triangle goes to the front or back island by its centroid's z; seam vertices are duplicated;
    UVs from the two projections."""
    P = np.array(part.points)
    pts, uvs, ws, isls, tags, tris, cache = [], [], [], [], [], [], {}
    for t in part.tris:
        isl = front_island if P[list(t)][:, 2].mean() >= z_split else back_island
        out = []
        for v in t:
            if (v, isl) not in cache:
                cache[(v, isl)] = len(pts)
                pts.append(P[v])
                uvs.append((uv_front if isl == front_island else uv_back)(P[v]))
                ws.append(part.weights[v])
                isls.append(isl)
                tags.append(part.tag[v])
            out.append(cache[(v, isl)])
        tris.append(tuple(out))
    part.points, part.uv, part.weights, part.island, part.tag, part.tris = (
        [np.asarray(q) for q in pts], uvs, ws, isls, tags, tris)
    return part


# ------------------------------------------------------------------------------------------- the sword

SWORD_DIR = (0, -.15, 1.0)


def build_sword(parts, meshes_dir):
    from scipy.spatial.transform import Rotation as Rot

    from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
    m = parse_kmf((Path(meshes_dir) / "DarkAngel-Idle2.kmf").read_bytes())
    sp = m.anim.sprites[3]
    pts = evaluate_anim_frame(m.anim, 0)[3]
    q = np.stack([-pts[:, 0], -pts[:, 2], pts[:, 1]], 1) / DA_HEIGHT
    p = Part("sword", m.materials[sp.material].textures[0], SK)
    g, tip = q[:4].mean(0), q[4:7].mean(0)
    R = Rot.align_vectors([unit(SWORD_DIR)], [unit(tip - g)])[0].as_matrix()
    fist = np.array(ARM[-1][0]) + (.004, -.034, .012)
    for v, vx in zip(q, sp.vertices):
        p.add(R @ (v - g) + fist, (vx.uv[0] / 32768, vx.uv[1] / 32768), "handL", "game")
    p.tris = [tuple(t) for t in sp.triangles[0]]
    p.game_uv = True
    parts.append(p)
    return p


def build_soles(meshes_dir):
    parts = []
    build_torso(parts)
    build_arms(parts)
    build_legs(parts)
    build_loincloth(parts)
    build_armour(parts)
    build_head(parts)
    build_sword(parts, meshes_dir)
    return parts


# relaxed: weight on his right leg, hips tilted, the sword arm bent with the blade up and out, the free
# hand loose at his side, the head turned a touch toward the blade
STANCE = {
    "root": (0, 6, 3), "spine": (2, -3, -2), "chest": (0, -4, -1), "neck": (0, 3, 1), "head": (2, 4, 2),
    "armL": (-28, 10, 10), "forearmL": (-62, 0, 4), "handL": (12, 0, -6),
    "armR": (4, 0, -9), "forearmR": (-16, 0, 0), "handR": (0, 0, 6),
    "thighL": (-6, 0, 5), "shinL": (10, 0, 0), "footL": (-4, 0, -4),
    "thighR": (2, 0, -2), "shinR": (2, 0, 0),
}
