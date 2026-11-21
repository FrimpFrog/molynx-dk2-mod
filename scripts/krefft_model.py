"""Krefft (Soles' Creatures Mod) as a low-poly Dungeon Keeper 2 creature: skeleton, mesh, skin weights,
UVs. Built the way the Gargoyle and the Dragon are (scripts.dragon_model: Part, loft, cone, forward,
Skinner; scripts.gargoyle_model: gloft, ccone, flat_cap), on Krefft's own Skeleton.

Soles' pet: a Krefft's river turtle. In the dungeon he is the Casino's loan shark and the Torture
Chamber's debt collector: a low, flat, olive-khaki shell with gold coins wedged in its cracks and duckweed
on it, a short thick neck in a battered gold chain, a broad flat head with a smooth grey head plate, a
wide smug mouth with a chipped gold tooth, one scarred half-closed eye, and four big webbed paddle feet
with five long black claws each. Quadruped, low to the ground, no humanoid parts.

Model space as the Gargoyle's: Y up, Z forward, X to the creature's LEFT; triangles counter-clockwise
seen from outside (the KMF export mirrors x into the game's winding). Units are KMF units (krefft_build
exports at scale 1): about 0.85 long from the snout to the hind claws and 0.28 tall at the top of the
shell, between the Salamander's files (0.91 long, 0.41 tall, 218 triangles) and the Bile Demon's
(0.85 wide, 0.54 tall, 212 triangles) in bulk.

Parts (each its own group, so the later retarget from the Salamander can drive them by bone):
  shell   carapace (a ridge, a costal ring, the marginal rim) and the underside (the bridge and the
          plastron) - one closed piece; three gold coins wedged in it (double-sided, half sunk)
  neck    a 5-sided loft from inside the shell to inside the head (bones neck1, neck2), and the gold
          chain round it (an 8-sided band, sagging at the throat)
  head    the skull (6-sided sections, unmirrored UVs: the scarred eye is only on his right), the
          chipped gold tooth hanging over the lower lip
  jaw     the lower jaw (4-sided, hinged under the eyes, built almost shut: the smug line) and the throat
  legs    four 3-sided lofts shoulder/hip - elbow/knee - wrist/ankle, each ending in a webbed paddle foot
          whose outline runs out into five long claws
  tail    a short 3-sided point under the back of the shell

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_model import Part, Skeleton, Skinner, _mirror_name, _seg_dist, _unit, cone, forward, loft
from scripts.gargoyle_model import flat_cap, gloft

# ------------------------------------------------------------------------------------------- skeleton

# name, parent, head (bind position), tail. Laid out like a quadruped donor's segments (the Salamander's
# body, neck, head, jaw, tail and four three-segment legs) so the retarget can fit each one.
_BONES_L = [
    ("root", None, (0, .17, -.02), (0, .17, .12)),
    ("neck1", "root", (0, .125, .15), (0, .145, .235)),
    ("neck2", "neck1", (0, .145, .235), (0, .163, .285)),
    ("head", "neck2", (0, .163, .285), (0, .135, .44)),
    ("jaw", "head", (0, .15, .31), (0, .128, .44)),
    ("tail1", "root", (0, .12, -.25), (0, .1, -.29)),
    ("tail2", "tail1", (0, .1, -.29), (0, .087, -.33)),
    ("upperFL", "root", (.10, .13, .13), (.235, .105, .185)),
    ("foreFL", "upperFL", (.235, .105, .185), (.255, .07, .215)),
    ("handFL", "foreFL", (.255, .07, .215), (.29, .05, .3)),
    ("thighL", "root", (.10, .125, -.15), (.24, .1, -.2)),
    ("shinL", "thighL", (.24, .1, -.2), (.258, .07, -.235)),
    ("footL", "shinL", (.258, .07, -.235), (.33, .05, -.31)),
]


def _build_bones():
    out = []
    for name, parent, head, tail in _BONES_L:
        out.append((name, parent, np.array(head, float), np.array(tail, float)))
        if name.endswith("L"):
            m = np.array([-1.0, 1, 1])
            out.append((_mirror_name(name), _mirror_name(parent), np.array(head) * m, np.array(tail) * m))
    return out


SKEL = Skeleton(_build_bones())
BONES, BONE_NAMES, BONE_INDEX = SKEL.bones, SKEL.names, SKEL.index
PARENT, HEAD, TAIL = SKEL.parent, SKEL.head, SKEL.tail
FRONT = {s: tuple(b + s for b in ("upperF", "foreF", "handF")) for s in "LR"}
HIND = {s: tuple(b + s for b in ("thigh", "shin", "foot")) for s in "LR"}

BODY, HEADT, SHELL = "Krefft_Body", "Krefft_Head", "Krefft_Shell"


class KrefftPart(Part):
    bone_index = BONE_INDEX


def auto_weights(p, bones, power=4.0, keep=2):
    d = np.array([max(_seg_dist(p, HEAD[BONE_INDEX[b]], TAIL[BONE_INDEX[b]]), 1e-4) for b in bones])
    order = np.argsort(d)[:keep]
    return {bones[i]: d[i] ** -power for i in order}


# ------------------------------------------------------------------------------------------- shell

SHELL_Z = -.02               # the middle of the carapace (z)
SHELL_A = .235               # half width at the widest (a little behind the middle): a broad, round oval
SHELL_B = (.26, .285)        # half length to the front (over the neck) and to the back
RIDGE = [(.10, .238), (-.03, .246), (-.165, .241)]      # (z, y) down the vertebral keel: a low, even dome
COSTAL_K, COSTAL_F, COSTAL_Y = 8, .58, .226             # ring of the costal scutes' outer corners
RIM_K, RIM_Y = 12, .148                                  # the marginal edge
PLASTRON_K, PLASTRON_F, PLASTRON_Y = 6, .74, .082        # where the underside flattens into the plastron
PLASTRON_C = (0, .072, -.03)


def outline(phi, f=1.0):
    """A point on the carapace's outline (top view), phi = 0 at the front, + to his left; f = fraction."""
    b = SHELL_B[0] if np.cos(phi) >= 0 else SHELL_B[1]
    wide = 1 + .05 * -np.cos(phi)                                   # widest behind the middle
    x = SHELL_A * wide * np.sin(phi) * f
    z = SHELL_Z + b * np.cos(phi) * f
    return x, z


def rim_y(phi):
    # the nuchal edge over the neck rides up, the back edge flares down a little
    return RIM_Y + .016 * max(np.cos(phi), 0) ** 3 - .01 * max(-np.cos(phi), 0) ** 2


def _shell_uv(p):
    """Carapace: planar from above, unmirrored (coins and cracks are his own on each side)."""
    zmin, zmax = SHELL_Z - SHELL_B[1] * 1.0, SHELL_Z + SHELL_B[0] * 1.0
    a = SHELL_A * 1.05
    return ((p[0] + a) / (2 * a), (p[2] - zmin) / (zmax - zmin))


def _under_uv(p):
    """The underside: planar from below, mirrored."""
    zmin, zmax = SHELL_Z - SHELL_B[1], SHELL_Z + SHELL_B[0]
    return (abs(p[0]) / (SHELL_A * 1.05), (p[2] - zmin) / (zmax - zmin))


def _merge_rings(part, inner, outer, inner_phi, outer_phi, centre):
    """Triangulate the band between two closed rings (angles increasing), sweeping by angle."""
    ni, no = len(inner), len(outer)
    i = j = 0
    ip = list(inner_phi) + [inner_phi[0] + 2 * np.pi]
    op = list(outer_phi) + [outer_phi[0] + 2 * np.pi]
    while i < ni or j < no:
        a, b = inner[i % ni], outer[j % no]
        if j < no and (i >= ni or op[j + 1] <= ip[i + 1]):
            part.tri(a, b, outer[(j + 1) % no], inside=centre)
            j += 1
        else:
            part.tri(a, b, inner[(i + 1) % ni], inside=centre)
            i += 1


SHELL_DEPTH = 1.0            # the standing version B deepens the shell (krefft_standing)
BELLY = 0.0                  # and gives the plastron a bulge (a centre point pushed out)


def build_shell(parts):
    part = KrefftPart("shell", SHELL)
    deep = lambda y: RIM_Y + (y - RIM_Y) * SHELL_DEPTH
    centre = np.array([0, .17, SHELL_Z])
    add_c = lambda p, **t: part.add(p, _shell_uv(p), "root", "carapace", **t)
    add_u = lambda p, **t: part.add(p, _under_uv(p), "root", "under", **t)
    ridge = [add_c(np.array([0, deep(y), z]), ring="ridge") for z, y in RIDGE]
    cphi = [2 * np.pi * k / COSTAL_K for k in range(COSTAL_K)]
    costal = []
    for ph in cphi:
        x, z = outline(ph, COSTAL_F)
        y = COSTAL_Y + .012 * max(-np.cos(ph), 0) - .006 * max(np.cos(ph), 0)
        costal.append(add_c(np.array([x, deep(y), z]), ring="costal"))
    rphi = [2 * np.pi * k / RIM_K for k in range(RIM_K)]
    rim_pts = []
    for ph in rphi:
        x, z = outline(ph)
        rim_pts.append(np.array([x, rim_y(ph), z]))
    rim_c = [add_c(p, ring="rim") for p in rim_pts]
    rim_u = [add_u(p, ring="rim") for p in rim_pts]
    pphi = [2 * np.pi * (k + .5) / PLASTRON_K for k in range(PLASTRON_K)]
    plast = []
    for ph in pphi:
        x, z = outline(ph, PLASTRON_F)
        plast.append(add_u(np.array([x, deep(PLASTRON_Y), z]), ring="plastron"))
    # the top: the keel (front F, middle M, back B) to the costal ring (0 front, 2 his left, 4 back)
    F, M, B = ridge
    c = costal
    top = [(F, c[0], c[1]), (F, c[1], M), (M, c[1], c[2]), (M, c[2], c[3]), (M, c[3], B), (B, c[3], c[4]),
           (F, c[7], c[0]), (F, M, c[7]), (M, c[6], c[7]), (M, c[5], c[6]), (M, B, c[5]), (B, c[4], c[5])]
    for a, b, d in top:
        part.tri(a, b, d, inside=centre)
    _merge_rings(part, costal, rim_c, cphi, rphi, centre)
    _merge_rings(part, plast, rim_u, pphi, rphi, centre)
    pc = np.array(PLASTRON_C, float)
    if BELLY > 0:
        bp = np.array([0, deep(PLASTRON_Y) - BELLY, PLASTRON_C[2]])
        b = part.add(bp, (.22, _under_uv(bp)[1]), "root", "under", ring="belly")   # (off the midline seam's texels)
        for k in range(PLASTRON_K):
            part.tri(plast[k], plast[(k + 1) % PLASTRON_K], b, inside=pc + (0, .2, 0))
    else:
        for k in range(1, PLASTRON_K - 1):
            part.tri(plast[0], plast[k], plast[k + 1], inside=pc + (0, .2, 0))
    parts.append(part)
    return part


# gold coins wedged in the shell's cracks: (centre on the shell (x, z), radius, the crack's direction
# (deg about the vertical), how far the disc is propped up out of the crack (deg from lying flat)).
# Big flat round discs (6-sided) so they read from DK2's high camera.
COINS = [((.09, -.12), .054, 30, 9),     # the big one, his left, behind the middle
         ((-.11, .07), .047, -55, 9),     # his right, front
         ((-.1, -.19), .043, 60, 8)]       # his right, at the back (off the keel, so no edge sticks up)
COIN_SIDES = 6


def shell_top_y(part, x, z):
    """y of the carapace above (x, z) (a ray from above)."""
    P = np.array(part.points)
    best = None
    for a, b, c in part.tris:
        if part.island[a] != "carapace":
            continue
        A, B_, C = P[a], P[b], P[c]
        d = (B_[2] - C[2]) * (A[0] - C[0]) + (C[0] - B_[0]) * (A[2] - C[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((B_[2] - C[2]) * (x - C[0]) + (C[0] - B_[0]) * (z - C[2])) / d
        w1 = ((C[2] - A[2]) * (x - C[0]) + (A[0] - C[0]) * (z - C[2])) / d
        w2 = 1 - w0 - w1
        if min(w0, w1, w2) >= -1e-9:
            y = w0 * A[1] + w1 * B_[1] + w2 * C[1]
            best = y if best is None else max(best, y)
    return best


def build_coins(parts, shell):
    part = KrefftPart("coins", SHELL)
    for ci, ((x, z), r, yaw, prop) in enumerate(COINS):
        y = shell_top_y(shell, x, z)
        e = .01                                                 # the shell's own slope there
        n = _unit(np.array([-(shell_top_y(shell, x + e, z) - shell_top_y(shell, x - e, z)) / (2 * e), 1.0,
                            -(shell_top_y(shell, x, z + e) - shell_top_y(shell, x, z - e)) / (2 * e)]))
        a = np.radians(yaw)
        ax = np.array([np.cos(a), 0, -np.sin(a)])              # along the crack
        ax = _unit(ax - n * np.dot(ax, n))
        across = np.cross(n, ax)                                # across it, in the shell's surface
        if np.dot(across, [x, 0, z + .02]) > 0:                 # the raised edge toward the keel (uphill)
            across = -across
        P = np.radians(prop)
        tilt = across * np.cos(P) + n * np.sin(P)               # propped 'prop' degrees off the surface
        nrm = _unit(np.cross(ax, tilt))
        if nrm[1] < 0:
            nrm = -nrm
        cen = np.array([x, y, z])
        ts = [2 * np.pi * k / COIN_SIDES for k in range(COIN_SIDES)]
        pts = [cen + r * (np.cos(t) * ax + np.sin(t) * across) for t in ts]
        # each corner laid onto the shell (the coin follows its curve, so no edge stands proud of the
        # dome), the uphill half lifted `prop` degrees out of the crack: wedged
        pts = [np.array([q[0], shell_top_y(shell, q[0], q[2]) + .006 + max(np.sin(t), 0) * r * np.sin(P), q[2]])
               for q, t in zip(pts, ts)]
        uvs = [(.5 + .5 * np.cos(t), .5 + .5 * np.sin(t)) for t in ts]
        for face in (1,):                                        # flush on the shell: its top face only
            ids = [part.add(p, uv, "root", "coin", face=face, coin=ci) for p, uv in zip(pts, uvs)]
            # a centre vertex on the shell too (a fan from a corner would cut under the dome's curve)
            mid = np.array([x, shell_top_y(shell, x, z) + .006 + r * np.sin(P) * .5, z])
            # lift the whole coin clear of any ridge between the shell's facets under it (else a facet's
            # edge pokes through and bites a wedge out of the disc)
            deficit = 0.0
            for k in range(COIN_SIDES):
                A_, B_, C_ = mid, part.points[ids[k]], part.points[ids[(k + 1) % COIN_SIDES]]
                for w1 in np.linspace(0, 1, 9):
                    for w2 in np.linspace(0, 1 - w1, 9):
                        q = A_ * (1 - w1 - w2) + B_ * w1 + C_ * w2
                        sy = shell_top_y(shell, q[0], q[2])
                        if sy is not None:
                            deficit = max(deficit, sy + .0025 - q[1])
            for j in ids:
                part.points[j] = part.points[j] + np.array([0, deficit, 0])
            mid = mid + np.array([0, deficit, 0])
            cid = part.add(mid, (.5, .5), "root", "coin", face=face, coin=ci)
            for k in range(COIN_SIDES):
                a_, b_, c_ = cid, ids[k], ids[(k + 1) % COIN_SIDES]
                pa, pb, pc = (part.points[j] for j in (a_, b_, c_))
                if np.dot(np.cross(pb - pa, pc - pa), nrm) * face >= 0:
                    part.tri(a_, b_, c_)
                else:
                    part.tri(a_, c_, b_)
    parts.append(part)
    return part


# v4 (Soles' changes to B): one coin only (the big one), wedged into a crack that splits the shell's
# plates: the disc stands in the split along the crack, leaning COIN_LEAN degrees off the shell's normal,
# sunk so about 60% of it shows; the crack's broken lips (a raised ridge each side, khaki outside, dark
# inside) hide the bottom of the disc. Double-sided (the back of the coin shows from behind).
V4 = False
COIN_V4 = ((.09, -.12), .056, 30)        # (x, z) on the shell, radius, the crack's direction (deg)
COIN_LEAN = 60.0                          # degrees the disc leans off the shell's normal (toward `across`)
COIN_RISE = .03                           # the disc's centre this many radii above the shell (in its plane)
COIN_SIDES_V4 = 8
CRACK_GAP = .006                          # half the split's width at the coin
LIP_LEN = 1.3                             # the lips run this many radii each way along the crack
LIP_H = {1: .0035, -1: .009}              # the lips' height: low under the leaning coin, high behind it


def coin_frame(shell):
    """-> (the point on the shell under the coin, the shell's normal there, along the crack, across it)."""
    (x, z), r, yaw = COIN_V4
    e = .01
    y = shell_top_y(shell, x, z)
    n = _unit(np.array([-(shell_top_y(shell, x + e, z) - shell_top_y(shell, x - e, z)) / (2 * e), 1.0,
                        -(shell_top_y(shell, x, z + e) - shell_top_y(shell, x, z - e)) / (2 * e)]))
    a = np.radians(yaw)
    ax = np.array([np.cos(a), 0, -np.sin(a)])
    ax = _unit(ax - n * np.dot(ax, n))
    across = np.cross(n, ax)
    if np.dot(across, [x, 0, z + .02]) < 0:                # lean away from the keel (downhill)
        across = -across
    return np.array([x, y, z]), n, ax, across


def build_wedged_coin(parts, shell):
    part = KrefftPart("coins", SHELL)
    S, n, ax, across = coin_frame(shell)
    r = COIN_V4[1]
    L = np.radians(COIN_LEAN)
    up = n * np.cos(L) + across * np.sin(L)                 # the disc's 'up' in its own plane
    nrm = _unit(np.cross(ax, up))                            # its face (toward the lean side, upward)
    if nrm[1] < 0:
        nrm = -nrm
    C = S + up * r * COIN_RISE
    ts = [2 * np.pi * k / COIN_SIDES_V4 + np.pi / COIN_SIDES_V4 for k in range(COIN_SIDES_V4)]
    pts = [C + r * (np.cos(t) * ax + np.sin(t) * up) for t in ts]
    for face in (1, -1):                                     # the front, and the back (mirrored stamp)
        ids = [part.add(p, (.5 + .5 * np.cos(t) * face, .5 + .5 * np.sin(t)), "root", "coin", face=face)
               for p, t in zip(pts, ts)]
        for k in range(1, COIN_SIDES_V4 - 1):
            a_, b_, c_ = ids[0], ids[k], ids[k + 1]
            pa, pb, pc = (part.points[j] for j in (a_, b_, c_))
            if np.dot(np.cross(pb - pa, pc - pa), nrm) * face >= 0:
                part.tri(a_, b_, c_)
            else:
                part.tri(a_, c_, b_)

    def on_shell(q, lift=0.0):
        return np.array([q[0], shell_top_y(shell, q[0], q[2]), q[2]]) + n * lift

    # the crack's lips: a broken ridge each side of the split; outside khaki (v 0 .. .45), inside dark
    # (v .55 .. 1), the inside running down into the split under the coin
    g, Lh = CRACK_GAP, LIP_LEN * r
    for s in (1, -1):
        sd = across * s
        e0 = on_shell(S - ax * Lh + sd * g * .4, -.002)
        e1 = on_shell(S + ax * Lh + sd * g * .4, -.002)
        top = on_shell(S + sd * (g + .004) + ax * .006 * s, LIP_H[s])
        out = on_shell(S + sd * (g + .026), .0012)
        inner = S + sd * g * .2 - n * .016
        o = [part.add(e0, (0, .3), "root", "crack", lip=s), part.add(out, (.5, 0), "root", "crack", lip=s),
             part.add(e1, (1, .3), "root", "crack", lip=s), part.add(top, (.5, .45), "root", "crack", lip=s)]
        i_ = [part.add(e0, (0, .7), "root", "crack", lip=s), part.add(top, (.5, .55), "root", "crack", lip=s),
              part.add(e1, (1, .7), "root", "crack", lip=s), part.add(inner, (.5, 1.0), "root", "crack", lip=s)]
        below = S - n * .05
        part.tri(o[0], o[1], o[3], inside=below + sd * .0)
        part.tri(o[1], o[2], o[3], inside=below)
        part.tri(i_[0], i_[1], i_[3], inside=S + sd * .05 - n * .02)
        part.tri(i_[1], i_[2], i_[3], inside=S + sd * .05 - n * .02)
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- neck

NECK_PATH = np.array([(0, .125, .15), (0, .145, .235), (0, .163, .285)])   # under the shell's front lip
NECK_RADII = [(.08, .104), (.07, .09), (.04, .07)]          # short, thick (0.9 x the head), flaring into the shell
NECK_SIDES = 5
NECK_BONES = ["root", "neck1", "neck2", "head"]
# the chain: an 8-sided band round the neck between ring 0 and ring 1, tilted down at the throat
CHAIN_CENTRE = np.array([0, .147, .25])
CHAIN_TILT = 70.0            # degrees: the band square round the neck (a collar), a little slack at the throat
CHAIN_R = (.066, .087)       # (dorsal-ventral, lateral) of its top edge: 5% outside the neck
CHAIN_DROP = (.03, .006)     # its lower edge: down and out
CHAIN_K = 8


def build_neck(parts):
    part = KrefftPart("neck", BODY)
    gloft(part, "neck", NECK_PATH, NECK_RADII, NECK_SIDES, (0, 1, 0),
          lambda i, p: auto_weights(p, NECK_BONES), tag=lambda i, th: {"theta": th, "ring": i})
    parts.append(part)
    # the chain (on the head's skin with the gold tooth)
    ch = KrefftPart("chain", HEADT)
    a = np.radians(CHAIN_TILT)
    axis = _unit(np.array([0, np.cos(a), np.sin(a)]))      # the band's axis (up, leaning forward)
    d0 = _unit(np.array([0, 0, 1.0]) - axis * axis[2])       # towards the throat in the band's plane
    lat = np.array([1.0, 0, 0])
    top, bot = [], []
    for k in range(CHAIN_K + 1):
        th = 2 * np.pi * k / CHAIN_K + np.pi / CHAIN_K
        dirn = np.cos(th) * d0 + np.sin(th) * lat
        p_top = CHAIN_CENTRE + CHAIN_R[0] * np.cos(th) * d0 + CHAIN_R[1] * np.sin(th) * lat
        p_bot = p_top + dirn * CHAIN_DROP[1] - axis * CHAIN_DROP[0]
        sag = .005 * max(np.cos(th), 0) ** 2                    # a little slack at the throat
        p_top, p_bot = p_top - axis * sag, p_bot - axis * sag
        u = k / CHAIN_K
        w = auto_weights(p_top, ["neck1", "neck2", "root"])
        top.append(ch.add(p_top, (u, 0.0), w, "chain", k=k))
        bot.append(ch.add(p_bot, (u, 1.0), w, "chain", k=k))
    for k in range(CHAIN_K):
        mid = CHAIN_CENTRE
        ch.tri(top[k], top[k + 1], bot[k], inside=mid)
        ch.tri(top[k + 1], bot[k + 1], bot[k], inside=mid)
    parts.append(ch)
    return part


# ------------------------------------------------------------------------------------------- head

# v3: a broad, flat, rounded spade of a head, as in Soles' photos (crop1, crop3): 7-point sections (the
# head plate on top, a bevel each side, the upper side where the eye sits, the lip), four rings and a
# blunt snout. Every face gets a planar UV island by the way it faces (the plate from above, each side
# from the side, the snout from the front, the chin from below), so the face's features get real texels
# and are painted where they are in 3D; each eye is a modelled dome with its own island.
#   (z, centre y, half width, up to the plate, down to the lip)
HEAD_RINGS = [
    (.262, .168, .074, .036, .030),    # the back (the neck's end is inside it), at the shell lip
    (.325, .171, .094, .046, .033),    # the temples behind the eyes: the widest
    (.390, .164, .078, .039, .030),    # in front of the eyes
    (.437, .151, .048, .026, .024),    # the snout: broad and blunt, tapering (a rounded spade from above)
]
SNOUT_TIP = np.array([0, .129, .463])     # at the lip line: the snout's front slopes down to it
SKULL_BACK = np.array([0, .17, .262])
SKULL_BACK_CAP = False                 # the back of the skull is open: the neck's end fills it
# a section's left half: (x / half width, y / up (+) or down (-)): the top, the bevel, the upper side, the lip
SECTION = [(0.0, 1.0), (.62, .86), (.97, .12), (.84, -1.0)]     # the cheeks slope in to the lip: no wall
SMILE = (.01, .019, .004, 0.0)        # the lip line lifted per ring: up at the back, the smug curl
EYE_YZ = (.19, .372)                  # (y, z) of the eyes' centres, high on the sides (crop1)
EYE_R = .023
EYE_BULGE = .013                      # how far the dome stands off the head
SCARRED = -1                           # his right eye (x < 0): the half-shut lid is modelled over it
JAW_OPEN = 0.0                         # degrees: shut
HINGE = HEAD[BONE_INDEX["jaw"]]
JAW_DEPTH = (.036, .031, .022)         # under the lip at rings 1, 2, 3
CHIN = np.array([0, .123, .456])          # the lower jaw's front, just under the snout's tip
JAW_BACK = np.array([0, .118, .292])
# the chipped gold tooth: on his left, in the front third of the upper lip, hanging straight down over
# the lower lip: a thick three-sided peg, tapered, its end broken off aslant (the chip)
TOOTH_Z, TOOTH_LEN, TOOTH_R = .416, .022, .0125
# planar UV boxes (A's bind space): the sides (z, y), the plate (x, z), the snout front (x, y), the chin
SIDE_BOX = ((.245, .47), (.085, .225))
TOP_BOX = ((-.1, .1), (.245, .47))
FRONT_BOX = ((-.075, .075), (.1, .19))
UNDER_BOX = ((-.09, .09), (.27, .47))
EYES = {}                              # side -> (centre, normal, along, up): filled by build_head


def _section_pts(z, yc, w, up, down, lift=0.0):
    """Left to right over the top: lip L, side L, bevel L, top, bevel R, side R, lip R."""
    pts = []
    for s in (1, -1):
        half = []
        for k, (fx, fy) in enumerate(SECTION):
            y = yc + (fy * up if fy > 0 else fy * down) + (lift if k == 3 else 0.0)
            half.append(np.array([s * fx * w, y, z]))
        pts.append(half)
    L, R = pts
    return [L[3], L[2], L[1], L[0], R[1], R[2], R[3]]


def _face_island(n, c, kind):
    """Which planar island a face goes to, by its normal (kind 'skull' or 'jaw')."""
    ax, ay, az = abs(n[0]), n[1], n[2]
    if kind == "skull" and ay > .78:
        return "top"
    if ay < -.55 or (kind == "jaw" and ax < .45):
        return "under"
    if ax >= max(abs(ay), abs(az)) * .8 or ax > .5:
        return "sideL" if c[0] > 0 else "sideR"
    if az > abs(ay):
        return "front"
    return "top" if ay > 0 else "under"


def head_uv(island, p):
    p = np.asarray(p, float)
    if island in ("sideL", "sideR"):
        (z0, z1), (y0, y1) = SIDE_BOX
        u = (p[2] - z0) / (z1 - z0)
        return (u if island == "sideL" else 1 - u, (y1 - p[1]) / (y1 - y0))
    if island == "top":
        (x0, x1), (z0, z1) = TOP_BOX
        return ((p[0] - x0) / (x1 - x0), (z1 - p[2]) / (z1 - z0))
    if island == "front":
        (x0, x1), (y0, y1) = FRONT_BOX
        return ((p[0] - x0) / (x1 - x0), (y1 - p[1]) / (y1 - y0))
    if island == "under":
        (x0, x1), (z0, z1) = UNDER_BOX
        return ((p[0] - x0) / (x1 - x0), (p[2] - z0) / (z1 - z0))
    return (.5, .5)


def _emit(part, P, tris, bone, kind, inside_of, extra_island=None):
    """Add the faces `tris` over points P to `part`, each face's corners on its own planar island
    (shared between faces of the same island). `inside_of(face index)` -> a point inside the volume."""
    ids = {}
    for fi, (a, b, c) in enumerate(tris):
        pa, pb, pc = P[a], P[b], P[c]
        cen = (pa + pb + pc) / 3
        n = _unit(np.cross(pb - pa, pc - pa))
        if np.dot(n, cen - inside_of(fi)) < 0:
            n = -n
            b, c = c, b
        isl = extra_island(fi) if extra_island and extra_island(fi) else _face_island(n, cen, kind)
        vs = []
        for v in (a, b, c):
            key = (v, isl)
            if key not in ids:
                ids[key] = part.add(P[v], head_uv(isl, P[v]), bone, isl, k=v, along=1.0 if kind == "jaw" else 0.0)
            vs.append(ids[key])
        part.tris.append(tuple(vs))


def build_head(parts):
    part = KrefftPart("head", HEADT)
    rings = []
    for i, (z, yc, w, up, down) in enumerate(HEAD_RINGS):
        rings.append(_section_pts(z, yc, w, up, down, SMILE[i]))
    P = [p for r in rings for p in r]
    n = 7
    idx = lambda i, k: i * n + k
    tris, inside, mouth = [], [], []
    cen = [np.array([0, yc, z]) for z, yc, *_ in HEAD_RINGS]
    for i in range(len(rings) - 1):
        mid = (cen[i] + cen[i + 1]) / 2
        for k in range(n - 1):
            tris += [(idx(i, k), idx(i, k + 1), idx(i + 1, k)), (idx(i, k + 1), idx(i + 1, k + 1), idx(i + 1, k))]
            inside += [mid, mid]
            mouth += [False, False]
    tip = len(P)
    P.append(SNOUT_TIP)
    back = len(P)
    P.append(SKULL_BACK)
    last = len(rings) - 1
    for k in range(n - 1):
        tris.append((idx(last, k), idx(last, k + 1), tip))
        inside.append(cen[last] - (SNOUT_TIP - cen[last]))
        mouth.append(False)
        if SKULL_BACK_CAP:
            tris.append((idx(0, k), idx(0, k + 1), back))
            inside.append(cen[0] + (cen[0] - SKULL_BACK) + (0, 0, .02))
            mouth.append(False)
    # the palate (inside the shut mouth): the lip line's ring 0 / 1 / 3 corners and the tip
    for j, (a_, b_, c_) in enumerate(((idx(0, 6), idx(0, 0), idx(1, 6)), (idx(0, 0), idx(1, 0), idx(1, 6)),
                                      (idx(1, 6), idx(1, 0), idx(last, 6)), (idx(1, 0), idx(last, 0), idx(last, 6)),
                                      (idx(last, 6), idx(last, 0), tip))):
        tris.append((a_, b_, c_))
        inside.append(cen[1] + (0, .08, 0))
        mouth.append(j >= 2)          # behind the mouth corner it is the throat (skin, seen from below)
    P = [np.asarray(p, float) for p in P]
    _emit(part, P, tris, "head", "skull", lambda fi: inside[fi], lambda fi: "mouth" if mouth[fi] else None)
    part.lips = [(r[0].copy(), r[6].copy()) for r in rings]
    for s in (1, -1):
        build_eye(part, s)
    build_lid(part)
    build_tooth(part)
    parts.append(part)
    return part, rings


def side_x(part, y, z, s, islands=("sideL", "sideR", "top", "front")):
    """x of the skull's outer surface on side s at (y, z) (a ray from outside along -s x)."""
    P = np.array(part.points)
    best = None
    for a, b, c in part.tris:
        if part.island[a] not in islands:
            continue
        A, B, C = P[a], P[b], P[c]
        d = (B[2] - C[2]) * (A[1] - C[1]) + (C[1] - B[1]) * (A[2] - C[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((B[2] - C[2]) * (y - C[1]) + (C[1] - B[1]) * (z - C[2])) / d
        w1 = ((C[2] - A[2]) * (y - C[1]) + (A[1] - C[1]) * (z - C[2])) / d
        w2 = 1 - w0 - w1
        if min(w0, w1, w2) >= -1e-9:
            x = w0 * A[0] + w1 * B[0] + w2 * C[0]
            if x * s > 0:
                best = x if best is None else (max(best, x) if s > 0 else min(best, x))
    return best


def surf(part, y, z, s, out=0.0):
    x = side_x(part, y, z, s)
    if x is None:
        raise ValueError(("no skull at", y, z))
    return np.array([x + out * s, y, z])


def onto(part, q, s, out=0.0):
    """q moved onto the skull's side s (along x; stepping down if the head is lower than q there)."""
    for k in range(80):
        x = side_x(part, q[1] - .0005 * k, q[2], s)
        if x is not None:
            return np.array([x + out * s, q[1] - .0005 * k, q[2]])
    raise ValueError(("no skull under", q))


def eye_frame(part, s):
    """The eye's centre on the head's surface and its frame: the normal (out, a little forward and up),
    `along` (toward the snout in the eye's plane) and `up`."""
    y, z = EYE_YZ
    c = surf(part, y, z, s)
    e = .004
    dy = (surf(part, y + e, z, s) - surf(part, y - e, z, s)) / (2 * e)
    dz = (surf(part, y, z + e, s) - surf(part, y, z - e, s)) / (2 * e)
    nrm = _unit(np.cross(dz, dy) * s)
    if nrm[0] * s < 0:
        nrm = -nrm
    nrm = _unit(nrm + np.array([0, .12, .3]))       # out, a little forward and up (crop1)
    up = _unit(np.array([0, 1.0, 0]) - nrm * nrm[1])
    along = np.cross(up, nrm) * s
    if along[2] < 0:
        along = -along
    return c, nrm, along, up


def build_eye(part, s):
    """A low dome (6 faces) standing EYE_BULGE off the side of the head: the eye, on its own island
    (planar in the eye's frame, so the round iris and pupil get 20 texels across)."""
    c, nrm, along, up = eye_frame(part, s)
    EYES[s] = (c, nrm, along, up)
    isl = "eyeL" if s > 0 else "eyeR"
    k = 6
    base = []
    for j in range(k):
        th = 2 * np.pi * (j + .5) / k
        q = c + EYE_R * (np.cos(th) * along + np.sin(th) * up)
        base.append(q - nrm * .003)             # sunk a little into the head round the dome
    apex = c + nrm * EYE_BULGE
    uv = lambda q: (.5 + np.dot(q - c, along) / (2.3 * EYE_R), .5 - np.dot(q - c, up) / (2.3 * EYE_R))
    bid = [part.add(q, uv(q), "head", isl, eye=s) for q in base]
    aid = part.add(apex, uv(apex), "head", isl, eye=s)
    inside = c - nrm * .03
    for j in range(k):
        part.tri(bid[j], bid[(j + 1) % k], aid, inside=inside)


def build_lid(part):
    """The scarred eye's half-shut upper lid, modelled: a heavy flap (2 faces) from the brow above the eye
    out and down over the dome's upper half, its rim just below the eye's middle."""
    s = SCARRED
    c, nrm, along, up = EYES[s]
    top = [c + up * EYE_R * 1.25 + along * EYE_R * d for d in (-1.25, 1.25)]
    top = [q - nrm * .002 for q in top]
    # the rim droops toward the snout (a glower), just below the eye's middle
    rim = []
    for d in (-.95, .95):                       # on the dome (its height there) plus a lid's thickness
        dn, du = d, -(.42 + .2 * d)                    # covering 70% of the eye
        h = EYE_BULGE * max(1 - np.hypot(dn, du), 0) + .0022
        rim.append(c + up * EYE_R * du + along * EYE_R * dn + nrm * h)
    pts = [top[0], top[1], rim[1], rim[0]]
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    ids = [part.add(p, q, "head", "lid", along=float(q[1])) for p, q in zip(pts, uv)]
    inside = c - nrm * .03
    part.tri(ids[0], ids[1], ids[2], inside=inside)
    part.tri(ids[0], ids[2], ids[3], inside=inside)


def build_tooth(part):
    """The chipped gold tooth: a thick three-sided peg from inside the upper lip, straight down over the
    outside of the lower lip, tapered, its end cut aslant (one corner short: the chip). 7 faces."""
    zs = [r[0] for r in HEAD_RINGS]
    lipx = np.interp(TOOTH_Z, zs, [lp[0][0] for lp in part.lips])
    lipy = np.interp(TOOTH_Z, zs, [lp[0][1] for lp in part.lips])
    root = np.array([lipx * 1.03, lipy + .006, TOOTH_Z])
    axis = np.array([0.08, -1.0, 0.04])
    axis = axis / np.linalg.norm(axis)
    ring0, ring1 = [], []
    for k in range(3):
        th = 2 * np.pi * k / 3 + np.pi / 6
        off = TOOTH_R * np.array([np.cos(th), 0, np.sin(th)])     # one face outward (+x), one forward
        ring0.append(part.add(root + off, (k / 3, 0.0), "head", "tooth", along=0.0, k=k))
        cut = (TOOTH_LEN + .006) * (.55 if k == 2 else 1.0)          # the chipped corner (the front one)
        ring1.append(part.add(root + off * .18 + axis * cut, (k / 3, 1.0), "head", "tooth",
                              along=1.0 if k != 2 else .66, k=k))
    cen = root + axis * TOOTH_LEN * .5
    for k in range(3):
        a, b = ring0[k], ring0[(k + 1) % 3]
        c, d = ring1[k], ring1[(k + 1) % 3]
        part.tri(a, b, c, inside=cen)
        part.tri(b, d, c, inside=cen)
    part.tri(ring1[0], ring1[1], ring1[2], inside=cen)


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def build_jaw(parts, head):
    """The lower jaw under the head's lip line: 5-point sections (lip, the rounded side, the chin's
    middle), from the mouth corner (ring 1) to the chin; the tongue bed (hidden when shut) on top.
    Its faces share the head's planar islands (the sides from the side, the chin from below)."""
    part = KrefftPart("jaw", HEADT)
    R = _rx(JAW_OPEN)
    op = lambda p: HINGE + R @ (np.asarray(p, float) - HINGE)
    P, secs = [], []
    for i, hr in enumerate((1, 2, 3)):
        z, yc, w, up, down = HEAD_RINGS[hr]
        lipL, lipR = (np.asarray(q, float) for q in head.lips[hr])
        d = JAW_DEPTH[i]
        wide = np.array([1.04, 1, 1])
        sec = [lipL * wide, np.array([.72 * w, lipL[1] - d * .62, z]), np.array([0, lipL[1] - d, z]),
               np.array([-.72 * w, lipL[1] - d * .62, z]), lipR * wide]
        secs.append(len(P))
        P += [op(q) for q in sec]
    chin = len(P)
    P.append(op(CHIN))
    back = len(P)
    P.append(op(JAW_BACK))
    cen = [np.mean(P[s:s + 5], 0) for s in secs]
    tris, inside, mouth = [], [], []
    for i in range(2):
        mid = (cen[i] + cen[i + 1]) / 2
        for k in range(4):
            a, b = secs[i] + k, secs[i] + k + 1
            c_, d_ = secs[i + 1] + k, secs[i + 1] + k + 1
            tris += [(a, b, c_), (b, d_, c_)]
            inside += [mid, mid]
            mouth += [False, False]
    for k in range(4):
        tris.append((secs[2] + k, secs[2] + k + 1, chin))
        inside.append(cen[2] - (P[chin] - cen[2]))
        mouth.append(False)
        tris.append((secs[0] + k, secs[0] + k + 1, back))
        inside.append(cen[0] + (cen[0] - P[back]))
        mouth.append(False)
    for a_, b_, c_ in ((secs[0] + 4, secs[0], secs[2] + 4), (secs[0], secs[2], secs[2] + 4),
                       (secs[2] + 4, secs[2], chin)):     # the tongue bed (hidden when shut)
        tris.append((a_, b_, c_))
        inside.append(cen[1] - (0, .08, 0))
        mouth.append(True)
    _emit(part, [np.asarray(p, float) for p in P], tris, "jaw", "jaw", lambda fi: inside[fi],
          lambda fi: "mouth" if mouth[fi] else None)
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- legs, feet

# legs (left; mirrored): shoulder / hip inside the shell, elbow / knee out under the rim, wrist / ankle
FLEG_PATH = np.array([(.10, .13, .13), (.235, .105, .185), (.255, .072, .215)])
FLEG_RADII = [(.064, .064), (.058, .06), (.046, .05)]
HLEG_PATH = np.array([(.10, .125, -.15), (.24, .1, -.2), (.258, .072, -.235)])
HLEG_RADII = [(.068, .07), (.06, .062), (.048, .052)]
# paddle feet: (wrist, forward direction (x, z), length to the web, width, claw length, claw spread deg)
FFOOT = ((.255, .065, .215), (.38, 1.0), .094, .225, .019, 84)
HFOOT = ((.258, .065, -.235), (.85, -1.0), .1, .235, .02, 86)
FLOOR = .045                 # A's feet stand this high in model space (the export puts them on the floor):
#                              the body sits low between the sprawled legs, nearly on its plastron


def _limb_weights(bones, n_rings):
    def w(i, p):
        if i <= 0:
            return {bones[0]: 1.0}
        if i >= n_rings:
            return {bones[-1]: 1.0}
        return auto_weights(p, list(bones))
    return w


def build_foot(part, island, wrist, fwd, length, width, claw, spread, s, bones, floor=None, sole=True, palm=False):
    """A webbed paddle: a raised knuckle over an outline that runs out into five claw points (top: 11
    triangles round the knuckle), a flat sole under the claw tips (5). UV: planar in the foot's frame,
    u across (0 inner, 1 outer, mirrored left and right), v from the wrist (0) to the claw tips (1)."""
    wrist = np.asarray(wrist, float)
    f = _unit(np.array([fwd[0] * s, 0, fwd[1]]))
    side = np.array([f[2], 0, -f[0]]) * (1 if fwd[1] > 0 else -1) * s     # outward (away from the body)
    if np.dot(side, [s, 0, 0]) < 0:
        side = -side
    up = np.array([0, 1.0, 0])
    wb = bones[-1]
    total = length + claw
    floor = FLOOR if floor is None else floor

    def P(across, along, y):
        return wrist + side * across + f * along + up * (y + floor - wrist[1])

    def uv(across, along):
        return (.5 + across / (width * 1.25), along / total)

    pts = []           # outline, inner wrist -> claws -> outer wrist
    tags = []
    pts.append((-width * .4, .0, .026)); tags.append("wrist")
    angs = np.radians(np.linspace(-spread / 2, spread / 2, 5))
    for j, a in enumerate(angs):
        d = np.array([np.sin(a), np.cos(a)])
        pts.append((d[0] * total * .92, d[1] * total, .004)); tags.append("claw")
        if j < 4:                                                # a shallow V notch in the web
            a2 = (angs[j] + angs[j + 1]) / 2
            d2 = np.array([np.sin(a2), np.cos(a2)])
            pts.append((d2[0] * length * .99, d2[1] * length * .97, .006)); tags.append("web")   # shallow notches: a paddle
    pts.append((width * .4, .0, .026)); tags.append("wrist")
    knuckle = (0.0, length * .38, .04)
    ids = [part.add(P(*q), uv(q[0], q[1]), wb, island, kind=t, side=s) for q, t in zip(pts, tags)]
    k = part.add(P(*knuckle), uv(knuckle[0], knuckle[1]), wb, island, kind="knuckle", side=s)
    n = len(ids)
    for j in range(n):
        a, b = ids[j], ids[(j + 1) % n]
        pa, pb, pk = part.points[a], part.points[b], part.points[k]
        if np.dot(np.cross(pb - pa, pk - pa), up) >= 0:
            part.tri(a, b, k)
        else:
            part.tri(a, k, b)
    if palm:            # v5 (B): a closed paddle - the whole outline fanned underneath on the paddle's own
        #                   skin (the dark "sole" island read as a hole on a raised hand), so the paddle is a
        #                   solid wedge from every side (a hand lifted, a sole kicking in the air)
        cq = (0.0, length * .38, .004)          # (flat with the claw tips: the lowest point stays theirs)
        # on the limb's own pebbled skin from its lit side (a turtle's paddle is paler underneath): the
        # paddle's dark top texture read as a black slab whenever a raised hand or a sole faced the camera
        limb = {"ffoot": "fleg", "hfoot": "hleg"}.get(island, island)

        def puv(across, along):
            return (.04 + .16 * (across / (width * .5) * .5 + .5), .3 + .4 * min(along / total, 1.0))
        c = part.add(P(*cq), puv(cq[0], cq[1]), wb, limb, kind="palm", side=s)
        under = [part.add(P(q[0], q[1], q[2]), puv(q[0], q[1]), wb, limb, kind="palm", side=s) for q in pts]
        for j in range(n):
            a, b = under[j], under[(j + 1) % n]
            pa, pb, pc = part.points[a], part.points[b], part.points[c]
            if np.dot(np.cross(pb - pa, pc - pa), up) <= 0:
                part.tri(a, b, c)
            else:
                part.tri(a, c, b)
        return
    # the sole (its own island, shared by the four feet): wrists and claw tips, flat on the floor
    claws = [q for q, t in zip(pts, tags) if t == "claw"]
    if not sole:
        return
    sole_q = [pts[0], claws[0], claws[4], pts[-1]]               # a flat quad under the paddle
    sid = [part.add(P(q[0], q[1], .002), uv(q[0], q[1]), wb, "sole", kind="sole", side=s) for q in sole_q]
    for j in range(1, len(sid) - 1):
        a, b, c = sid[0], sid[j], sid[j + 1]
        pa, pb, pc = (part.points[x] for x in (a, b, c))
        if np.dot(np.cross(pb - pa, pc - pa), up) <= 0:
            part.tri(a, b, c)
        else:
            part.tri(a, c, b)


def build_legs(parts):
    part = KrefftPart("legs", BODY)
    for s, side in ((1, "L"), (-1, "R")):
        m = np.array([s, 1, 1.0])
        for path, radii, bones, foot, isl in ((FLEG_PATH, FLEG_RADII, FRONT[side], FFOOT, "fleg"),
                                              (HLEG_PATH, HLEG_RADII, HIND[side], HFOOT, "hleg")):
            wrist, fwd, length, width, claw, spread = foot
            end = np.array(wrist) * m + np.array([0, .004, 0])
            gloft(part, isl, path * m, radii, 3, (0, 1.0, .3 * np.sign(path[1][2])), _limb_weights(bones, 3),
                  end_point=end, tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
            build_foot(part, isl[0] + "foot", np.array(wrist) * m, fwd, length, width, claw, spread, s, bones)
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- tail

TAIL_BASE = np.array([0, .112, -.245])        # (the floor is at FLOOR: the tip stays off it)
TAIL_TIP = np.array([0, .087, -.329])        # a short stub (60% of the first one)


def build_tail(parts):
    part = KrefftPart("tail", BODY)
    w = lambda: None
    ids = cone(part, "tail", TAIL_BASE, TAIL_TIP, .03, (0, 1, 0), {"tail1": 1.0}, sides=3)
    for j in ids[:-1]:
        part.weights[j] = {BONE_INDEX["tail1"]: 1.0}
    part.weights[ids[-1]] = {BONE_INDEX["tail2"]: 1.0}
    parts.append(part)
    return part


# ------------------------------------------------------------------------------------------- assembly

MANUAL = ("coin", "crack", "medal", "chain", "ffoot", "hfoot", "sole", "lid", "brow", "eyeL", "eyeR", "tooth")      # wound by hand (open or double-sided on purpose)


def welded(part):
    first = {}
    return [first.setdefault(tuple(np.round(p, 7)), i) for i, p in enumerate(part.points)]


def orient(part):
    """Consistent outward winding of every closed (or tube-like) piece (the Gargoyle's `orient`, its
    volume measured about the piece's own centre so open-ended tubes judge right too)."""
    P = np.array(part.points)
    weld = welded(part)
    idx = [i for i, t in enumerate(part.tris) if part.island[t[0]] not in MANUAL]
    edges = {}
    for i in idx:
        a, b, c = (weld[v] for v in part.tris[i])
        for e in ((a, b), (b, c), (c, a)):
            edges.setdefault(frozenset(e), []).append(i)
    seen = set()
    for start in idx:
        if start in seen:
            continue
        comp, todo = [], [start]
        seen.add(start)
        while todo:
            i = todo.pop()
            comp.append(i)
            a, b, c = (weld[v] for v in part.tris[i])
            for e in ((a, b), (b, c), (c, a)):
                for j in edges[frozenset(e)]:
                    if j in seen:
                        continue
                    x, y, z = (weld[v] for v in part.tris[j])
                    if e in ((x, y), (y, z), (z, x)):
                        x, y, z = part.tris[j]
                        part.tris[j] = (x, z, y)
                    seen.add(j)
                    todo.append(j)
        cen = np.mean([P[v] for i in comp for v in part.tris[i]], 0)
        vol = sum(np.dot(P[a] - cen, np.cross(P[b] - cen, P[c] - cen)) for a, b, c in (part.tris[i] for i in comp))
        if vol < 0:
            for i in comp:
                a, b, c = part.tris[i]
                part.tris[i] = (a, c, b)


def build_krefft(shell_depth=1.0, belly=0.0, v4=False):
    """-> list of Parts in bind pose (standing, head up a little: the idle's rest). The standing version
    B passes a deeper shell and a belly; `v4`: the one coin wedged in its crack (Soles' v4 changes)."""
    global SHELL_DEPTH, BELLY, V4
    SHELL_DEPTH, BELLY, V4 = shell_depth, belly, v4
    parts = []
    shell = build_shell(parts)
    (build_wedged_coin if v4 else build_coins)(parts, shell)
    build_neck(parts)
    head, _ = build_head(parts)
    build_jaw(parts, head)
    build_legs(parts)
    build_tail(parts)
    for p in parts:
        orient(p)
    return parts


def counts(parts):
    return {p.name: (len(p.points), len(p.tris)) for p in parts}


def krefft_skinner(parts):
    return Skinner(parts, SKEL)


def krefft_forward(local, root_offset=(0, 0, 0)):
    return forward(local, root_offset, SKEL)


if __name__ == "__main__":
    ps = build_krefft()
    c = counts(ps)
    print(c, "total tris", sum(t for _, t in c.values()))
