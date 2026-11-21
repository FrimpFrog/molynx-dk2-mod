"""Soles as a low-poly DK2 creature, built like the game's own humans (the Thief, the Rogue, the Guard): one
body whose parts meet (the legs start inside the hips under a belt and loincloth, the arms inside the
shoulders, the neck inside the head), a sculpted head under a layered shell of long hair, front and back
projected skins like the game's, and small armour.

His face is painted in his likeness on the front of the head (soles_texture). He is on
the evil side, in knightly plate of his own design in dark grey, red, white, silver and gold, bare-armed:
an engraved breastplate carrying the Molynx crest, faulds, layered pauldrons, vambraces, cuisses, knee cops
and greaves, a red skirt with a white and gold hem, and the Dark Angel's sword (the game's own triangles,
still pointing at DarkAngelFrnt).

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
    ("shoulderL", "chest", (.03, .78, -.01), (.122, .772, -.012)),
    ("armL", "shoulderL", (.122, .772, -.012), (.142, .628, -.02)),
    ("forearmL", "armL", (.142, .628, -.02), (.16, .49, .0)),
    ("handL", "forearmL", (.16, .49, .0), (.164, .42, .008)),
    ("thighL", "root", (.062, .49, .0), (.07, .265, .012)),
    ("shinL", "thighL", (.07, .265, .012), (.074, .055, -.01)),
    ("footL", "shinL", (.074, .055, -.01), (.078, .008, .08)),
])
HEAD, BODY, LIMBS, HAIR_TEX = "Soles_Head", "Soles_Body", "Soles_Limbs", "Soles_Hair"
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
TORSO = [(.445, .09, .064, .068, .0), (.5, .095, .064, .067, .0), (.56, .082, .062, .062, .004),
         (.64, .104, .068, .062, .008), (.71, .128, .074, .06, .008), (.772, .112, .058, .05, -.004),
         (.81, .04, .034, .034, -.006), (.87, .03, .029, .029, -.002)]
TORSO_N = 8


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

ARM_RINGS = (0, 2, 4, 6)
ARM = [((.112, .772, -.012), .05, .046), ((.13, .735, -.014), .046, .042), ((.138, .68, -.018), .042, .04),
       ((.142, .628, -.02), .031, .029), ((.149, .58, -.014), .035, .033), ((.156, .53, -.006), .029, .026),
       ((.16, .49, .0), .021, .019)]


def build_arms(parts):
    p = Part("arms", LIMBS, SK)
    for s in (1, -1):
        nm = nm_for(s)
        bones = [nm("armL"), nm("forearmL"), nm("handL")]
        use = [ARM[i] for i in ARM_RINGS]
        cs = [np.array(c) * (s, 1, 1) for c, _, _ in use]
        L = len(cs) - 1
        rings = [ring_pts(cs[i], cs[min(i + 1, L)] - cs[max(i - 1, 0)], (0, 0, 1), rx, rz, 6)
                 for i, (_, rx, rz) in enumerate(use)]
        tube(p, "arm", rings, lambda i, q, b=bones: {b[0]: 1.0} if i == 0 else SK.auto(q, b), cs)
        w = cs[-1]
        hand = [w + (0, -.004, .002), w + (.002 * s, -.03, .008), w + (.003 * s, -.054, .01)]
        rh = [(.024, .022), (.027, .025), (.022, .02)]
        rings = [ring_pts(hand[i], hand[min(i + 1, 2)] - hand[max(i - 1, 0)], (s, 0, 0), rh[i][1], rh[i][0], 5)
                 for i in (0, 2)]
        hand = [hand[0], hand[2]]
        tube(p, "hand", rings, lambda i, q, b=bones: {b[2]: 1.0}, hand, cap1=w + (.005 * s, -.074, .01))
        cs = [np.array(c) * (s, 1, 1) for c, _, _ in ARM]
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- legs

LEG_RINGS = (0, 2, 3, 4, 6)
LEG = [((.05, .47, .004), .05, .048), ((.06, .41, .008), .05, .048), ((.067, .35, .012), .048, .046),
       ((.07, .27, .014), .039, .038), ((.071, .2, .01), .044, .043), ((.072, .13, .0), .037, .037),
       ((.074, .062, -.008), .031, .031)]
BOOT_TOP = .2


def build_legs(parts):
    p = Part("legs", LIMBS, SK)
    for s in (1, -1):
        nm = nm_for(s)
        bones = [nm("thighL"), nm("shinL"), nm("footL")]
        use = [LEG[i] for i in LEG_RINGS]
        cs = [np.array(c) * (s, 1, 1) for c, _, _ in use]
        L = len(cs) - 1
        rings = [ring_pts(cs[i], cs[min(i + 1, L)] - cs[max(i - 1, 0)], (0, 0, 1), rx, rz, 6)
                 for i, (_, rx, rz) in enumerate(use)]
        tube(p, "leg", rings, lambda i, q, b=bones: {"root": .5, b[0]: .5} if i == 0 else SK.auto(q, b), cs)
        a = cs[-1]
        foot = [a + (0, -.02, -.012), a + (.003 * s, -.042, .05)]
        rf = [(.036, .034), (.03, .016)]
        rings = [ring_pts(foot[i], foot[1] - foot[0], (0, 1, .4), rf[i][0], rf[i][1], 6) for i in range(2)]
        tube(p, "foot", rings, lambda i, q, b=bones: {b[2]: 1.0}, foot, cap1=a + (.003 * s, -.048, .085))
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
    """A red skirt all round the hips, from under the belt to the knee, flaring, split at the sides."""
    p = Part("loincloth", LIMBS, SK)
    flap(p, "cloth", .48, .29, .066, .086, .085, .1, front=True, rows=2, cols=3)
    flap(p, "cloth", .48, .28, .07, .09, .09, .104, front=False, rows=2, cols=3)
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- armour

def plate(p, island, c, e1, e2, r1, r2, lift, w, n=8, inside_off=.05, along0=0.0):
    """A low dome of plate: an ellipse (e1 * r1, e2 * r2) round c, its apex `lift` out along e1 x e2 (made
    to face away from `inside`)."""
    nrm = unit(np.cross(e1, e2))
    ring = [c + np.cos(a) * e1 * r1 + np.sin(a) * e2 * r2 for a in np.linspace(0, 2 * np.pi, n + 1)[:-1]]
    ids = [p.add(q - nrm * .004, (min(k, n - k) / (n / 2), along0), w, island, along=0.0)
           for k, q in enumerate(ring)]
    apex = p.add(c + nrm * lift, (.5, along0 + .3), w, island, along=1.0)
    for k in range(n):
        p.tri(ids[k], ids[(k + 1) % n], apex, inside=c - nrm * inside_off)
    return nrm


def spike(p, island, base, tip, r, w):
    t = unit(np.asarray(tip) - base)
    a = np.cross(t, [0, 0, 1.0]) if abs(t[2]) < .9 else np.cross(t, [1.0, 0, 0])
    a, b = unit(a), unit(np.cross(t, a))
    ids = [p.add(base + (np.cos(th) * a + np.sin(th) * b) * r, (k / 3, .8), w, island, along=0.0)
           for k, th in enumerate(np.linspace(0, 2 * np.pi, 4)[:-1])]
    tp = p.add(tip, (.5, 1), w, island, along=1.0)
    for k in range(3):
        p.tri(ids[k], ids[(k + 1) % 3], tp, inside=base - t * .01)


def build_armour(parts):
    """Knightly plate in dark grey with silver edges: four stacked lames over each shoulder, vambraces on
    the forearms, a knee cop on each knee, and the belt. (The breastplate, faulds, cuisses and greaves are
    painted on the torso and legs.)"""
    p = Part("armour", LIMBS, SK)
    for s in (1, -1):
        nm = nm_for(s)
        w = {nm("shoulderL"): .35, nm("armL"): .65}
        for i, (dy, r) in enumerate(((0, .068), (-.034, .058))):
            c = np.array([s * (.125 + .006 * i), .8 + dy, -.012])
            e1 = np.array([0, 0, 1.0])
            e2 = unit(np.array([s, -.35 - .35 * i, 0]))
            nrm = unit(np.cross(e1, e2))
            if nrm[1] < 0:
                e1 = -e1
            plate(p, "pauldron", c, e1, e2, r * .95, r * .7, .03 - .005 * i, w, n=6, along0=i / 4)
        # vambraces: plate over the forearm, stopping short of the fist
        a0 = np.array(ARM[4][0]) * (s, 1, 1)
        a1 = np.array(ARM[6][0]) * (s, 1, 1) + (0, .018, 0)
        rings = [ring_pts(a0 + (0, .012, 0), a1 - a0, (0, 0, 1), .041, .039, 6),
                 ring_pts(a1, a1 - a0, (0, 0, 1), .029, .027, 6)]
        tube(p, "bracer", rings, lambda i, q, b=nm("forearmL"): {b: 1.0}, [a0 + (0, .012, 0), a1])
        # knee cops
        k = np.array(LEG[3][0]) * (s, 1, 1)
        plate(p, "pauldron", k + (0, .004, .034), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), .036, .04,
              .016, {nm("thighL"): .3, nm("shinL"): .7}, n=6, along0=.75)
    rings = [torso_ring(y, .1, .072, .076, 0) for y in (.462, .5)]
    tube(p, "belt", rings, lambda i, q: {"root": 1.0}, [np.array([0, .462, 0]), np.array([0, .5, 0])])
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- hair

# a long, layered shag: volume at the crown, over the ears, to the jaw and the nape, the ends flicking out
# rows: y, side radius, front radius, back radius, half-angle of the opening at the face (deg)
HAIR_ROWS = [(.993, .046, .046, .056, 0), (.967, .063, .062, .076, 0), (.945, .066, .067, .08, 0),
             (.905, .071, .062, .08, 58), (.872, .074, .057, .079, 64), (.848, .077, .052, .079, 72)]
HAIR_BACK_DROP = .013                 # the last row hangs lower behind (y .835 at the back)
HAIR_COLS = 9


# a cap over the skull, then separate locks hanging from it round the sides and back (each a gently
# curved strip, so the silhouette breaks up into locks), and a fringe of two locks from the parting
CAP_ROWS = [(.997, .03, .03, .036), (.982, .052, .05, .062), (.962, .064, .062, .076), (.94, .066, .066, .08)]
CAP_COLS = 10
# locks: angle round the head (deg, 0 = his face, + = his left), top y, length, width, outward flick
LOCKS = [(60, .975, .122, .05, .012), (90, .98, .13, .056, .02), (120, .98, .136, .058, .016),
         (150, .98, .136, .06, .012), (180, .98, .14, .062, .01), (-150, .98, .134, .06, .016),
         (-120, .98, .138, .058, .012), (-90, .98, .128, .056, .022), (-60, .975, .12, .05, .014)]


def head_radius(y, a):
    """The hair's surface radius at height y, angle a (from the cap rows / the skull)."""
    ys = [r[0] for r in CAP_ROWS] + [.9, .86]
    rs = [r[1] for r in CAP_ROWS] + [.064, .062]
    rf = [r[2] for r in CAP_ROWS] + [.062, .058]
    rb = [r[3] for r in CAP_ROWS] + [.076, .07]
    side = np.interp(-y, [-v for v in ys], rs)
    fb = np.interp(-y, [-v for v in ys], rf if np.cos(a) > 0 else rb)
    return side, fb


def build_hair(parts):
    p = Part("hair", HAIR_TEX, SK)
    # the cap
    grid = []
    for j, (y, rs, rf, rb) in enumerate(CAP_ROWS):
        row = []
        for k in range(CAP_COLS):
            a = 2 * np.pi * k / CAP_COLS
            r = rf if np.cos(a) > 0 else rb
            q = np.array([HC[0] + rs * np.sin(a), y, HC[2] - .006 + r * np.cos(a)])
            row.append(p.add(q, (k / CAP_COLS, j / (len(CAP_ROWS) - 1)), "head", "hair"))
        grid.append(row)
    for j in range(len(grid) - 1):
        mid = np.array([HC[0], (CAP_ROWS[j][0] + CAP_ROWS[j + 1][0]) / 2, HC[2] - .006])
        for k in range(CAP_COLS):
            a, b = grid[j][k], grid[j][(k + 1) % CAP_COLS]
            c, d = grid[j + 1][k], grid[j + 1][(k + 1) % CAP_COLS]
            p.tri(a, b, c, inside=mid)
            p.tri(b, d, c, inside=mid)
    top = p.add(np.array([HC[0], 1.003, HC[2] - .008]), (.5, 0), "head", "hair")
    for k in range(CAP_COLS):
        p.tri(grid[0][k], grid[0][(k + 1) % CAP_COLS], top, inside=HC)
    # the locks: 3 rows x 2 columns, curved round the head, flicking out at the end
    for i, (deg, y0, length, width, flick) in enumerate(LOCKS):
        a = np.radians(deg)
        out = np.array([np.sin(a), 0, np.cos(a)])
        tang = np.array([np.cos(a), 0, -np.sin(a)])
        pts, uvs = [], []
        for j, v in enumerate((0, .5, 1)):
            y = y0 - length * v
            side, fb = head_radius(y, a)
            r = 1 / np.sqrt((np.sin(a) / side) ** 2 + (np.cos(a) / fb) ** 2) + .003 + .004 * v + flick * v ** 2
            c = np.array([HC[0], y, HC[2] - .006]) + out * r
            w = width * (1 - .25 * v)
            for u in (-.5, 0, .5):
                q = c + tang * w * u + out * (.004 * (1 - abs(u) * 2))       # a rounded lock
                pts.append(q)
                uvs.append((u + .5, v))
        isl = "lock"
        tris = [(0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4), (3, 4, 6), (4, 7, 6), (4, 5, 7), (5, 8, 7)]
        both = abs(deg) < 100                                       # beside the face: seen from inside too
        for side_, il in ((1, isl), (-1, "lock_in")) if both else ((1, isl),):
            ids = [p.add(q, uv, "head", il, lock=float(i)) for q, uv in zip(pts, uvs)]
            for t in tris:
                pa = pts[t[0]]
                p.tri_facing(ids[t[0]], ids[t[1]], ids[t[2]], (out + np.array([0, .2, 0])) * side_)
    # the fringe: two locks from the parting on his left, swept across the forehead to his right
    for (x0, y0, x1, y1, w0) in ((.03, .995, -.025, .935, .05), (.0, .99, -.058, .932, .046)):
        pts, uvs = [], []
        for j, v in enumerate((0, .5, 1)):
            x, y = x0 + (x1 - x0) * v, y0 + (y1 - y0) * v
            side, fb = head_radius(y, 0.0)
            z = HC[2] - .006 + fb * np.sqrt(max(1 - (x / side) ** 2, .05)) + .006
            for u in (-.5, 0, .5):
                pts.append(np.array([x + .012 * u * (1 - v), y + w0 * u * .55, z + .002 * (1 - abs(u) * 2)]))
                uvs.append((u + .5, v))
        tris = [(0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4), (3, 4, 6), (4, 7, 6), (4, 5, 7), (5, 8, 7)]
        for side_, il in ((1, "lock"), (-1, "lock_in")):
            ids = [p.add(q, uv, "head", il, lock=9.0) for q, uv in zip(pts, uvs)]
            for t in tris:
                p.tri_facing(ids[t[0]], ids[t[1]], ids[t[2]], np.array([0, .3, 1.0]) * side_)
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- head

HC = np.array([0, .9, .008])
# rows: y, half width, front, back: the skull and, above the hairline and round the back, his hair
HEAD_ROWS = [
    (.84, .022, .044, .02), (.862, .04, .058, .06), (.893, .057, .064, .066), (.925, .059, .063, .074),
    (.958, .056, .055, .072), (.984, .036, .032, .046)]
HEAD_TOP = (0, .996, -.01)
HEAD_N = 8
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
                q[0] += np.sign(sa) * .006 * np.exp(-((y - .893) / .01) ** 2)
                q[0] -= np.sign(sa) * .003 * np.exp(-((y - .874) / .006) ** 2)     # the hollow under it
            r.append(q)
        rings.append(r)
    tube(p, "head", rings, lambda i, q: "head", [np.array([0, y, HC[2]]) for y, *_ in HEAD_ROWS],
         cap0=np.array([0, .834, .024]), cap1=np.array(HEAD_TOP))
    # the nose
    z0 = HC[2]
    pts = dict(br=(0, .916, z0 + .064), tip=(0, .884, z0 + .074), wl=(.015, .878, z0 + .062),
               wr=(-.015, .878, z0 + .062), base=(0, .874, z0 + .068), sl=(.011, .906, z0 + .062),
               sr=(-.011, .906, z0 + .062))
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

SWORD_DIR = (.15, -.5, 1.0)


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
    fist = np.array(ARM[-1][0]) + (.002, -.03, .008)
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
    build_hair(parts)
    build_sword(parts, meshes_dir)
    return parts


# relaxed: weight on his right leg, hips tilted, the sword arm bent with the blade up and out, the free
# hand loose at his side, the head turned a touch toward the blade
STANCE = {
    "root": (0, 12, 4), "spine": (2, -4, -2), "chest": (8, -10, 0), "neck": (-4, 3, 1), "head": (-2, 6, 2),
    "armL": (-15, 25, 20), "forearmL": (-55, 0, 4), "handL": (10, 0, -6),
    "armR": (10, 0, -18), "forearmR": (-35, 0, 0), "handR": (0, 0, 8),
    "thighL": (-14, -10, 8), "shinL": (22, 0, 0), "footL": (-8, 10, 0),
    "thighR": (8, 12, -6), "shinR": (10, 0, 0),
}
STANCE_DROP = .01
