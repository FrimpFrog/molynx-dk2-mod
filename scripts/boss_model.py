"""The Boss: a greedy corporate kingpin with a big white box for a head, as a low-poly DK2 creature.

A triangle mesh on a bone skeleton (soles_model.SK), each vertex weighted to its bones, at DK2's scale:
a three-piece suit with a paunch, jacket tails, a briefcase in his left hand, his right arm pointing (giving
orders), polished shoes; his head a bevelled white box with a printed corporate mark on its face and a
cigar jutting from it. The mark is an original one (a $, or the initials of whatever corporation the
user invents), never a real company's logo.

Model space as lowpoly: Y up, Z forward, X to his LEFT; 1.0 tall to the top of the box.

LICENCE: GPL-3.0.
"""
import numpy as np
from scipy.spatial import ConvexHull

from scripts.lowpoly import Part, unit
from scripts.soles_model import SK, flap, nm_for, ring_pts, torso_ring, tube

BODY, LIMBS, HEAD = "Boss_Body", "Boss_Limbs", "Boss_Head"

# ------------------------------------------------------------------------------------------- torso
# a paunchy kingpin: rings (y, half width, front, back, centre z)
TORSO = [(.445, .096, .07, .07, .0), (.5, .104, .082, .07, .006), (.56, .108, .09, .066, .012),
         (.63, .11, .084, .064, .01), (.7, .124, .074, .062, .006), (.765, .118, .06, .052, -.004),
         (.805, .045, .038, .036, -.006), (.835, .036, .032, .032, -.004)]


def build_torso(parts):
    p = Part("torso", BODY, SK)
    rings = [torso_ring(*row, n=8) for row in TORSO]
    bones = ["root", "spine", "chest", "neck"]
    tube(p, "torso", rings, lambda i, q: SK.auto(q, bones), [np.array([0, r[0], r[4]]) for r in TORSO],
         cap0=np.array([0, .43, 0]))
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- limbs
ARM = [((.118, .772, -.012), .046, .044), ((.142, .68, -.016), .04, .038), ((.152, .58, -.012), .034, .032),
       ((.16, .5, .0), .03, .028)]
LEG = [((.054, .47, .004), .056, .054), ((.062, .36, .01), .05, .048), ((.068, .26, .012), .044, .042),
       ((.07, .15, .004), .042, .04), ((.072, .06, -.006), .038, .036)]


def build_limbs(parts):
    p = Part("limbs", LIMBS, SK)
    for s in (1, -1):
        nm = nm_for(s)
        bones = [nm("armL"), nm("forearmL"), nm("handL")]
        cs = [np.array(c) * (s, 1, 1) for c, _, _ in ARM]
        rings = [ring_pts(cs[i], cs[min(i + 1, 3)] - cs[max(i - 1, 0)], (0, 0, 1), rx, rz, 6)
                 for i, (_, rx, rz) in enumerate(ARM)]
        tube(p, "arm", rings, lambda i, q, b=bones: {b[0]: 1.0} if i == 0 else SK.auto(q, b), cs)
        # the shirt cuff, then the hand: a fist (5 sides)
        w = cs[-1]
        cuff = [w, w + (0, -.014, .002)]
        rings = [ring_pts(q, cuff[1] - cuff[0], (0, 0, 1), .026, .024, 6) for q in cuff]
        tube(p, "cuff", rings, lambda i, q, b=bones: {b[2]: 1.0}, cuff)
        hand = [w + (0, -.014, .004), w + (.002 * s, -.06, .01)]
        rings = [ring_pts(hand[i], hand[1] - hand[0], (s, 0, 0), .02, .023, 5) for i in range(2)]
        tube(p, "hand", rings, lambda i, q, b=bones: {b[2]: 1.0}, hand, cap1=w + (.003 * s, -.074, .01))
        lb = [nm("thighL"), nm("shinL"), nm("footL")]
        cs = [np.array(c) * (s, 1, 1) for c, _, _ in LEG]
        rings = [ring_pts(cs[i], cs[min(i + 1, 4)] - cs[max(i - 1, 0)], (0, 0, 1), rx, rz, 6)
                 for i, (_, rx, rz) in enumerate(LEG)]
        tube(p, "leg", rings, lambda i, q, b=lb: {"root": .5, b[0]: .5} if i == 0 else SK.auto(q, b), cs)
        # polished shoes: a long rounded toe and a heel
        a = cs[-1]
        foot = [a + (0, -.034, -.016), a + (.002 * s, -.046, .05)]
        rings = [ring_pts(q, foot[1] - foot[0], (0, 1, .3), rx, rz, 6)
                 for q, (rx, rz) in zip(foot, ((.036, .03), (.032, .016)))]
        tube(p, "shoe", rings, lambda i, q, b=lb: {b[2]: 1.0}, foot, cap1=a + (.003 * s, -.05, .088))
    # jacket tails below the waist, front and back
    flap(p, "tails", .47, .38, .08, .084, .098, .1, front=True, rows=1, cols=2)
    flap(p, "tails", .47, .37, .072, .076, .098, .1, front=False, rows=1, cols=2)
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- the box head

BOX_C = np.array([0, .918, .008])
BOX_H = np.array([.098, .1, .094])          # half sizes
BOX_BEVEL = .016
BOX_TILT = 6.0                              # degrees, cocked to one side: a little swagger


def build_head(parts):
    """A bevelled white box: the 24 inset face corners and their convex hull; every triangle gets its own
    vertices (flat shading) and the island of the side it faces."""
    from scipy.spatial.transform import Rotation as Rot
    p = Part("head", HEAD, SK)
    pts = []
    for axis in range(3):
        for sgn in (-1, 1):
            u, v = [a for a in range(3) if a != axis]
            for su in (-1, 1):
                for sv in (-1, 1):
                    q = np.zeros(3)
                    q[axis] = sgn * BOX_H[axis]
                    q[u] = su * (BOX_H[u] - BOX_BEVEL)
                    q[v] = sv * (BOX_H[v] - BOX_BEVEL)
                    pts.append(q)
    pts = np.array(pts)
    R = Rot.from_euler("z", BOX_TILT, degrees=True).as_matrix()
    hull = ConvexHull(pts)
    names = {(2, 1): "front", (2, -1): "back", (0, 1): "left", (0, -1): "right", (1, 1): "top", (1, -1): "bottom"}
    for tri in hull.simplices:
        P = pts[tri]
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        if np.dot(nrm, P.mean(0)) < 0:
            tri = tri[[0, 2, 1]]
            P = pts[tri]
            nrm = -nrm
        c = P.mean(0)
        axis = int(np.argmax(np.abs(c / BOX_H)))
        isl = names[(axis, int(np.sign(c[axis])))]
        ids = []
        for q in P:
            u, v = [a for a in range(3) if a != axis]
            uv = (q[u] / BOX_H[u] * .5 + .5, .5 - q[v] / BOX_H[v] * .5)
            if axis == 2:
                uv = ((q[0] / BOX_H[0] * .5 + .5) if isl == "front" else (.5 - q[0] / BOX_H[0] * .5),
                      .5 - q[1] / BOX_H[1] * .5)
            if axis == 0:
                uv = (.5 + np.sign(c[0]) * q[2] / BOX_H[2] * .5 * -1, .5 - q[1] / BOX_H[1] * .5)
            if axis == 1:
                uv = (q[0] / BOX_H[0] * .5 + .5, q[2] / BOX_H[2] * .5 + .5)
            ids.append(p.add(R @ q + BOX_C, (float(np.clip(uv[0], 0, 1)), float(np.clip(uv[1], 0, 1))),
                             "head", isl))
        p.tris.append(tuple(ids))
    # the neck: a short tube up into the box
    rings = [ring_pts(np.array([0, y, -.004]), (0, 1, 0), (0, 0, 1), .034, .032, 6) for y in (.82, .84)]
    tube(p, "neck", rings, lambda i, q: {"neck": 1.0}, [np.array([0, .82, -.004]), np.array([0, .84, -.004])])
    # the cigar, jutting from the box's lower right, its tip glowing
    base = R @ np.array([-.03, -.045, BOX_H[2] - .004]) + BOX_C
    tip = base + R @ np.array([-.014, -.012, .07])
    rings = [ring_pts(q, tip - base, (0, 1, 0), .0075, .0075, 5) for q in (base, tip)]
    ids = tube(p, "cigar", rings, lambda i, q: "head", [base, tip], cap1=tip + unit(tip - base) * .004)
    for i, q in enumerate(p.points[-11:]):
        p.tag[len(p.points) - 11 + i]["along"] = float(np.dot(q - base, unit(tip - base)) / np.linalg.norm(tip - base))
    parts.append(p)
    return p


# ------------------------------------------------------------------------------------------- the briefcase

def build_case(parts):
    p = Part("case", HEAD, SK)
    hand = np.array(ARM[-1][0]) + (.003, -.07, .01)
    c = hand + (0, -.07, 0)
    h = np.array([.022, .06, .085])
    pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    hull = ConvexHull(pts)
    for tri in hull.simplices:
        P = pts[tri]
        if np.dot(np.cross(P[1] - P[0], P[2] - P[0]), P.mean(0) - c) < 0:
            tri = tri[[0, 2, 1]]
        ids = []
        for q in pts[tri]:
            d = (q - c) / h
            ids.append(p.add(q, ((d[2] * .5 + .5), (.5 - d[1] * .5)), "handL", "case"))
        p.tris.append(tuple(ids))
    # the handle
    for x in (-.01, .01):
        a, b = c + (x, h[1], -.03), c + (x, h[1], .03)
        top = (a + b) / 2 + (0, .016, 0)
        rings = [ring_pts(q, (0, 0, 1), (0, 1, 0), .004, .004, 3) for q in (a, top)]
    rings = [ring_pts(q, (0, 0, 1), (0, 1, 0), .005, .005, 4)
             for q in (c + (0, h[1] + .002, -.03), c + (0, h[1] + .018, 0), c + (0, h[1] + .002, .03))]
    tube(p, "handle", rings, lambda i, q: "handL", [c + (0, h[1], -.03), c + (0, h[1] + .016, 0),
                                                       c + (0, h[1], .03)])
    parts.append(p)
    return p


def build_boss():
    parts = []
    build_torso(parts)
    build_limbs(parts)
    build_head(parts)
    build_case(parts)
    return parts


# chest out, head cocked, the right arm pointing ahead (giving orders), the briefcase hanging in his left
STANCE = {
    "root": (0, -8, 0), "spine": (-4, 0, 0), "chest": (-6, 6, 0), "neck": (2, 4, 0), "head": (0, 6, -4),
    "armR": (-78, -10, -8), "forearmR": (-8, 0, 0), "handR": (-10, 0, 0),
    "armL": (6, 0, 6), "forearmL": (-6, 0, 0),
    "thighL": (-4, 0, 6), "thighR": (4, 6, -5), "shinR": (6, 0, 0), "footR": (0, -10, 0), "footL": (0, 8, 0),
}
