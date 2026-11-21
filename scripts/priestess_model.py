"""The Priestess (Dungeon Keeper 1's hero, the "Witch" in its data) as a low-poly DK2 creature.

Built from her DK1 sprites (CREATURE.JTY 2980-3189; the five-direction stand 3110-3114 and the walk
2980-3009 set her proportions): a tall, slim woman with a shoulder-length blonde bob, a lavender armoured
top and briefs with a bronze belt over a bare midriff, long lavender gloves, small grey pauldrons
holding a long pale cape, brown thigh-high boots, and a sceptre topped with a lavender orb.

Model space as in lowpoly: Y up, Z forward, X to her LEFT; she is 1.0 tall to the crown of her hair.
She is bound in an A-pose (arms a little out, legs straight); `ICON_POSE` is the stance of her DK1
portrait (left hand on her hip, the sceptre held out in her right hand).

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.lowpoly import Part, Skeleton, cone, double_sided, loft, unit

SKELETON = Skeleton([
    ("root", None, (0, .47, 0), (0, .575, 0)),
    ("spine", "root", (0, .575, 0), (0, .66, 0)),
    ("chest", "spine", (0, .66, 0), (0, .79, -.005)),
    ("neck", "chest", (0, .79, -.005), (0, .845, 0)),
    ("head", "neck", (0, .845, 0), (0, .99, 0)),
    ("shoulderL", "chest", (.02, .765, -.008), (.088, .765, -.008)),
    ("armL", "shoulderL", (.088, .765, -.008), (.12, .605, -.02)),
    ("forearmL", "armL", (.12, .605, -.02), (.142, .47, .0)),
    ("handL", "forearmL", (.142, .47, .0), (.148, .405, .008)),
    ("thighL", "root", (.052, .455, .0), (.06, .245, .012)),
    ("shinL", "thighL", (.06, .245, .012), (.064, .052, -.008)),
    ("footL", "shinL", (.064, .052, -.008), (.068, .008, .085)),
    ("capeA", "chest", (0, .765, -.05), (0, .5, -.1)),
    ("capeB", "capeA", (0, .5, -.1), (0, .18, -.13)),
])
SK = SKELETON

BODY, LIMBS, HEAD, CAPE = "Priestess_Body", "Priestess_Limbs", "Priestess_Head", "Priestess_Cape"

# torso rings: centre, (front-back half depth, half width)
TORSO = [((0, .41, .004), (.034, .04)), ((0, .455, 0), (.062, .1)), ((0, .5, .0), (.058, .092)),
         ((0, .548, .0), (.047, .072)), ((0, .595, .0), (.043, .062)), ((0, .632, .004), (.047, .07)),
         ((0, .672, .008), (.054, .082)), ((0, .712, .006), (.056, .088)), ((0, .762, -.006), (.042, .1)),
         ((0, .8, -.006), (.026, .03)), ((0, .845, -.004), (.024, .027))]
TORSO_BONES = ["root", "spine", "chest", "neck"]
TORSO_BUST = {6: 1.5, 7: 1.32}         # chest rings pushed forward at the front corners


def build_torso(parts):
    p = Part("torso", BODY, SK)
    path = [c for c, _ in TORSO]
    radii = [r for _, r in TORSO]

    def shape(i, th):
        c = np.cos(th)
        if i in TORSO_BUST and abs(np.sin(th)) > .5 and c > .5:
            return (TORSO_BUST[i], 1.0)
        if i == 8:                                   # shoulders: square across, thin front to back
            return (0.9, 1.0) if abs(np.sin(th)) > 0.7 else (1.0, 1.0)
        return (1.0, 1.0) if c > -0.2 else (0.92, 1.0)

    loft(p, "torso", path, radii, 10, (0, 0, 1.0), lambda i, q: SK.auto(q, TORSO_BONES), shape=shape,
         u_offset=np.pi / 10,
         start_point=np.array([0, .392, .0]))
    # (the bust is shaped into the chest rings: TORSO_BUST)
    parts.append(p)
    return p


ARM_PATH = [(.098, .768, -.008), (.114, .69, -.016), (.128, .605, -.02), (.139, .535, -.01), (.148, .47, .0)]
ARM_RADII = [(.034, .034), (.03, .029), (.025, .023), (.024, .022), (.018, .017)]
LEG_PATH = [(.056, .46, .002), (.06, .36, .008), (.064, .245, .016), (.066, .16, .0), (.068, .055, -.008)]
LEG_RADII = [(.064, .058), (.05, .046), (.038, .035), (.04, .034), (.026, .024)]   # thick thigh boots


def build_limbs(parts):
    p = Part("limbs", LIMBS, SK)
    for s in (1, -1):
        m = np.array([s, 1, 1.0])
        nm = (lambda b: b) if s > 0 else (lambda b: b[:-1] + "R")
        arm = [nm(b) for b in ("armL", "forearmL", "handL")]
        loft(p, "arm", np.array(ARM_PATH) * m, ARM_RADII, 7, (0, 0, 1.0),
             lambda i, q, arm=arm: {arm[0]: 1.0} if i == 0 else SK.auto(q, arm))
        # the hand: a flat mitten, then the thumb
        w = np.array(ARM_PATH[-1]) * m
        h = [w + (0, -.008, .004), w + (0.005 * s, -.06, .012)]
        loft(p, "hand", np.array(h), [(.015, .021), (.011, .017)], 4, (0, 0, 1.0),
             lambda i, q, a=arm: {a[2]: 1.0}, end_point=w + (.006 * s, -.078, .012))
        cone(p, "hand", w + (-.008 * s, -.025, .014), w + (-.012 * s, -.05, .03), .007, (0, 0, 1), arm[2])
        leg = [nm(b) for b in ("thighL", "shinL", "footL")]
        loft(p, "leg", np.array(LEG_PATH) * m, LEG_RADII, 8, (0, 0, 1.0),
             lambda i, q, leg=leg: {leg[0]: 1.0} if i == 0 else SK.auto(q, leg))
        # the boot's foot: ankle -> ball -> a pointed toe, with a small heel
        a = np.array(LEG_PATH[-1]) * m
        foot = [a, a + (.002 * s, -.042, .05)]
        loft(p, "foot", np.array(foot), [(.028, .025), (.014, .022)], 4, (0, 1.0, 0.3),
             lambda i, q, f=leg[2]: {f: 1.0}, end_point=a + (.003 * s, -.05, .1))
        cone(p, "foot", a + (0, -.03, -.012), a + (0, -.054, -.016), .009, (0, 0, -1), leg[2], sides=3)
    parts.append(p)
    return p


# head: sections (height in head units, half width, front, back), the face a front projection
HC = np.array([0, .9, .006])
HRX, HRY, HRZ = .044, .066, .054
HEAD_SECTIONS = [(.75, .8, .7, -.85), (.35, .97, .92, -1.0), (-.05, 1.0, .98, -1.0), (-.45, .88, .96, -.8),
                 (-.75, .64, .88, -.5), (-1.0, .28, .78, -.25)]
HEAD_RING = [(0, 1), (.55, .9), (.9, .55), (1, 0), (.7, -.65), (-.7, -.65), (-1, 0), (-.9, .55), (-.55, .9)]
HEAD_BACK = (3, 4, 5, 6)


def face_uv(p):
    return ((p[0] - HC[0]) / HRX * .5 + .5, (HRY - (p[1] - HC[1])) / (2 * HRY))


def build_head(parts):
    """The face is projected from the front; the back of the head (under her hair) has its own small
    island, so it never paints over the face."""
    p = Part("head", HEAD, SK)
    BACK = HEAD_BACK                          # ring corners that also get a scalp copy
    face, scalp = [], []
    for j, (h, w, f, b) in enumerate(HEAD_SECTIONS):
        pts = [(x * w, z * (f if z > 0 else -b)) for x, z in HEAD_RING]
        fr, sr = [], {}
        for k, (x, z) in enumerate(pts):
            q = HC + np.array([x * HRX, h * HRY, z * HRZ])
            fr.append(p.add(q, face_uv(q), "head", "face"))
            if k in BACK:
                sr[k] = p.add(q, ((k - BACK[0]) / 3, j / (len(HEAD_SECTIONS) - 1)), "head", "scalp")
        face.append(fr)
        scalp.append(sr)
    n = len(HEAD_RING)
    BACKQ = BACK[:-1]

    def vid(j, k, back):
        return scalp[j][k] if back else face[j][k]

    for i in range(len(face) - 1):
        mid = HC + (0, (HEAD_SECTIONS[i][0] + HEAD_SECTIONS[i + 1][0]) / 2 * HRY, 0)
        for k in range(n):
            k2 = (k + 1) % n
            back = k in BACKQ
            a, b = vid(i, k, back), vid(i, k2, back)
            c, d = vid(i + 1, k, back), vid(i + 1, k2, back)
            p.tri(a, b, c, inside=mid)
            p.tri(b, d, c, inside=mid)
    top = HC + (0, HRY, -.004)
    tf = p.add(top, face_uv(top), "head", "face")
    ts = p.add(top, (.5, 0), "head", "scalp")
    for k in range(n):
        back = k in BACKQ
        p.tri(vid(0, k, back), vid(0, (k + 1) % n, back), ts if back else tf, inside=HC)
    chin = HC + (0, -HRY * 1.02, .03)
    cf = p.add(chin, face_uv(chin), "head", "face")
    cs = p.add(chin, (.5, 1), "head", "scalp")
    last = len(face) - 1
    for k in range(n):
        back = k in BACKQ
        p.tri(vid(last, (k + 1) % n, back), vid(last, k, back), cs if back else cf, inside=HC - (0, HRY * .7, 0))
    # the nose
    tip = HC + (0, -.012, HRZ * .98 + .012)
    b1, b2, b3 = HC + (.009, -.022, HRZ * .96), HC + (-.009, -.022, HRZ * .96), HC + (0, .012, HRZ * .97)
    ids = [p.add(q, face_uv(q), "head", "face") for q in (b1, b2, b3, tip)]
    inner = HC + (0, -.008, HRZ * .8)
    p.tri(ids[0], ids[2], ids[3], inside=inner)
    p.tri(ids[2], ids[1], ids[3], inside=inner)
    p.tri(ids[1], ids[0], ids[3], inside=inner)
    parts.append(p)
    return p


# hair: rings round the head from the crown down to a bob on the shoulders; the lower rings open at the
# face (open = half-angle of the opening, degrees); 6 columns from one side of the face round the back
HAIR = [  # height, front radius, side radius, back radius, open (a full bob, as in her sprites)
    (.988, .024, .03, .03, 0), (.964, .06, .066, .07, 0), (.925, .062, .072, .076, 42),
    (.865, .06, .078, .082, 52), (.795, .064, .088, .086, 58)]
HAIR_COLS = 8


def build_hair(parts):
    p = Part("hair", HEAD, SK)
    grid = []
    for j, (y, rf, rs, rb, op) in enumerate(HAIR):
        row = []
        for k in range(HAIR_COLS):
            s = k / (HAIR_COLS - 1)
            a = np.radians(op + s * (360 - 2 * op))       # 0 = the front
            ca, sa = np.cos(a), np.sin(a)
            r_fb = rf if ca > 0 else rb
            q = np.array([HC[0] + rs * sa, y, HC[2] - .004 + r_fb * ca])
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
    t = p.add(HC + (0, .998 - HC[1], -.004), (1.0, 0.0), "head", "hair", inner=0.0)
    for k in range(HAIR_COLS - 1):
        p.tri(grid[0][k], grid[0][k + 1], t, inside=HC)
    # the inside of the bob (seen past her face): the lower rows again, facing in, darker
    inner = []
    for j in range(2, len(HAIR)):
        inner.append([p.add(p.points[i], p.uv[i], "head", "hair_in", inner=1.0) for i in grid[j]])
    for j in range(len(inner) - 1):
        mid = (centre(j + 2) + centre(j + 3)) / 2
        for k in range(HAIR_COLS - 1):
            a, b = inner[j][k], inner[j][k + 1]
            c, d = inner[j + 1][k], inner[j + 1][k + 1]
            # facing the centre: wind so the normal points toward `mid`
            pa, pb, pc = (p.points[i] for i in (a, b, c))
            nrm = np.cross(pb - pa, pc - pa)
            if np.dot(nrm, mid - (pa + pb + pc) / 3) > 0:
                p.tris += [(a, b, c), (b, d, c)]
            else:
                p.tris += [(a, c, b), (b, c, d)]
    parts.append(p)
    return p


def build_pauldrons(parts):
    p = Part("pauldrons", HEAD, SK)
    for s in (1, -1):
        b = "armL" if s > 0 else "armR"
        # a rounded plate lying over the shoulder, sloping down and out: a low dome, open underneath
        top = np.array([.1 * s, .782, -.008])
        w = {b: .6, ("shoulderL" if s > 0 else "shoulderR"): .4}
        e1 = np.array([0, 0, 1.0])
        e2 = unit(np.array([s, -.45, 0]))
        nrm = unit(np.cross(e2, e1)) * (1 if np.cross(e2, e1)[1] > 0 else -1)
        ring = []
        for k in range(6):
            a = 2 * np.pi * k / 6
            q = top + (np.cos(a) * e1 * .042 + np.sin(a) * e2 * .046) - nrm * .006
            ring.append(p.add(q, (k / 6, 0), w, "pauldron", along=0.0))
        apex = p.add(top + nrm * .016, (.5, 1), w, "pauldron", along=1.0)
        for k in range(6):
            p.tri(ring[k], ring[(k + 1) % 6], apex, inside=top - nrm * .03)
    parts.append(p)
    return p


# the cape: rows from the shoulders down to the calves, columns across the back, flaring out
CAPE_ROWS = [(.772, .1, -.048), (.66, .125, -.075), (.52, .15, -.098), (.36, .175, -.115),
             (.2, .2, -.13)]   # y, half width, z
CAPE_COLS = 6
CAPE_PLEAT = .014      # depth of the folds, growing toward the hem


def build_cape(parts):
    p = Part("cape", CAPE, SK)
    pts, uvs, ws = [], [], []
    for j, (y, hw, z) in enumerate(CAPE_ROWS):
        for k in range(CAPE_COLS + 1):
            s = k / CAPE_COLS
            x = (s - .5) * 2 * hw
            bulge = .035 * (1 - (2 * s - 1) ** 2) * (j > 0)          # round over her back
            pleat = CAPE_PLEAT * (j / (len(CAPE_ROWS) - 1)) * (1 if k % 2 else -1) * (0 < k < CAPE_COLS)
            q = np.array([x, y, z - bulge + .03 * (2 * s - 1) ** 2 * (j > 0) - pleat])
            pts.append(q)
            uvs.append((s, j / (len(CAPE_ROWS) - 1)))
            if j == 0:
                ws.append({"chest": 1.0})
            elif j == 1:
                ws.append({"capeA": .7, "chest": .3})
            elif j == 2:
                ws.append({"capeA": .8, "capeB": .2})
            elif j == 3:
                ws.append({"capeA": .35, "capeB": .65})
            else:
                ws.append({"capeB": 1.0})
    tris = []
    W = CAPE_COLS + 1
    for j in range(len(CAPE_ROWS) - 1):
        for k in range(CAPE_COLS):
            a, b, c, d = j * W + k, j * W + k + 1, (j + 1) * W + k, (j + 1) * W + k + 1
            tris += [(a, b, c), (b, d, c)]
    double_sided(p, ("cape", "cape_in"), pts, uvs, ws, tris, (0, 0, -1.0))
    parts.append(p)
    return p


# the sceptre in her right hand: a wooden shaft, a flanged collar and the lavender orb
def build_sceptre(parts):
    p = Part("sceptre", HEAD, SK)
    hand = np.array([-.152, .432, .014])            # in her right fist (bind pose)
    lo, hi = hand + (0, -.16, 0), hand + (0, .33, 0)
    loft(p, "staff", np.array([lo, hand, hi]), [(.009, .009), (.01, .01), (.01, .01)], 3, (0, 0, 1),
         lambda i, q: "handR")
    orb = hi + (0, .044, 0)
    r = .037
    # a faceted ball: the poles and two rings of 8 (rotated half a step)
    rings = []
    for j, (lat, off) in enumerate(((np.radians(40), 0), (np.radians(-40), np.pi / 8))):
        ring = []
        for k in range(8):
            a = 2 * np.pi * k / 8 + off
            q = orb + r * np.array([np.cos(lat) * np.cos(a), np.sin(lat), np.cos(lat) * np.sin(a)])
            ring.append(p.add(q, (k / 8, .5), "handR", "orb", along=.5))
        rings.append(ring)
    t = p.add(orb + (0, r, 0), (.5, 1), "handR", "orb", along=1.0)
    b = p.add(orb - (0, r, 0), (.5, 0), "handR", "orb", along=0.0)
    for k in range(8):
        u0, u1 = rings[0][k], rings[0][(k + 1) % 8]
        l0, l1 = rings[1][k], rings[1][(k + 1) % 8]
        p.tri(u0, u1, t, inside=orb)
        p.tri(l0, l1, b, inside=orb)
        p.tri(u0, l0, u1, inside=orb)
        p.tri(u1, l0, l1, inside=orb)
    # the collar: three prongs cupping the orb
    for a in np.radians([90, 210, 330]):
        d = np.array([np.cos(a), 0, np.sin(a)])
        cone(p, "staff", hi + d * .005, hi + d * .03 + (0, .036, 0), .007, (0, 1, 0), "handR", sides=3)
    parts.append(p)
    return p


def build_priestess():
    parts = []
    build_torso(parts)
    build_limbs(parts)
    build_head(parts)
    build_hair(parts)
    build_pauldrons(parts)
    build_cape(parts)
    build_sceptre(parts)
    return parts


# her portrait's stance: weight on the right leg, left hand on the hip, the sceptre held out at her side
ICON_POSE = {
    "root": (0, -6, 2), "spine": (0, 2, -2), "chest": (0, 4, -1), "head": (-2, -8, 3),
    "armL": (-8, 0, 32), "forearmL": (-15, 75, -78), "handL": (0, 0, 20),
    "armR": (-14, 0, -14), "forearmR": (-40, 0, 0), "handR": (12, 0, 34),
    "thighL": (-4, 0, 6), "thighR": (2, 0, -3), "shinL": (6, 0, 0), "footL": (0, 0, -4),
}
