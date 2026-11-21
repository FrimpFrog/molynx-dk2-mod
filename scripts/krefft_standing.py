"""Krefft, version B: standing upright on his two hind legs, the shell on his back (the plastron his
front), the forelegs stubby webbed, clawed arms. Same head, neck, chain, tooth, scar, coins and skins as
the four-legged version A (scripts.krefft_model): those parts are A's own, posed upright through A's
skeleton (the shell pitched up, leaning forward a little: a slight hunch; the neck bent forward so he
looks ahead), and the limbs are built new for standing - short, thick turtle's legs on big paddle feet,
and short arms ending in the same webbed paddles, hanging forward. A's three skins fit B unchanged (the
limbs use A's limb islands and UV conventions).

Same bone names as A (so the same skins and the same code paths work); in B the front legs' bones
(upperF / foreF / handF) are the arms and the hind legs' (thigh / shin / foot) the legs. Planned donor
for B's clips: the Bile Demon (a round heavy body on short legs with stubby arms; see notes).

LICENCE: GPL-3.0.
"""
import numpy as np

from scripts.dragon_model import Skeleton, _seg_dist, _unit
from scripts.gargoyle_model import gloft
from scripts.krefft_model import (BODY, BONE_INDEX, BONES, FFOOT, HFOOT, KrefftPart, SKEL, _rx, build_foot,
                                  build_krefft, krefft_forward, krefft_skinner, orient)

LEAN = 24.0                  # degrees the shell leans forward from upright: a slight hunch
NECK_BEND = (50.0, 12.0, -4.0)     # (v3: the head raised, the chin out: a boss, not dejected)     # neck1, neck2, head: the neck straight forward off the top of the shell
SHELL_DEPTH_B = 1.65         # a deeper shell (so it is not a board from the side)
BELLY_B = .085                # and a pot belly on the plastron
LEG_TOP = .115
HEAD_SCALE = 1.25
HEAD_DROP = np.array([0, -.05, -.02])     # down into the shell's opening: only a short collar of neck               # how high the bottom of the shell stands off the floor


def pose():
    """-> (world rotations, world heads) of A's skeleton posed upright."""
    local = np.repeat(np.eye(3)[None], len(BONES), 0)
    local[BONE_INDEX["root"]] = _rx(-(90.0 - LEAN))
    for name, deg in zip(("neck1", "neck2", "head"), NECK_BEND):
        local[BONE_INDEX[name]] = _rx(deg)
    local[BONE_INDEX["tail1"]] = _rx(45.0)                  # the tail pointing down and back behind the legs
    return krefft_forward(local)


# limbs in B's space (left; mirrored), relative to the shell: filled in by build_standing
LEG_RADII = [(.094, .1), (.08, .085), (.052, .056)]     # tapering 35% to the ankle       # short, thick, round stumps tapering to the ankle
ARM_RADII = [(.1, .1), (.09, .094), (.068, .072)]     # short and thick


def _weights_v5(bones):
    """v5: a limb's rings follow its joints - the root ring the upper bone, the middle ring (the knee /
    elbow) half and half, the end ring (the ankle / wrist) the lower bone (v4's nearest-two weights gave
    the knee ring half to the foot, so the thigh stretched as soon as a foot moved)."""
    def w(i, p):
        if i <= 0:
            return {bones[0]: 1.0}
        if i == 1:
            return {bones[0]: .5, bones[1]: .5}
        return {bones[1]: 1.0}
    return w


def _weights(skel, bones):
    def w(i, p):
        if i <= 0:
            return {bones[0]: 1.0}
        if i >= 3:
            return {bones[-1]: 1.0}
        d = np.array([max(_seg_dist(p, skel.head[skel.index[b]], skel.tail[skel.index[b]]), 1e-4) for b in bones])
        o = np.argsort(d)[:2]
        return {bones[k]: d[k] ** -4.0 for k in o}
    return w


def _paddle(part, island, wrist, R, foot, s, bones, palm=False):
    """A's paddle foot built flat at the origin, turned by R and set at `wrist` (for the hands). `palm`
    (v5): closed underneath."""
    tmp = KrefftPart("tmp", BODY)
    _, fwd, length, width, claw, spread = foot
    build_foot(tmp, island, np.array([0, .02, 0]), fwd, length * .95, width * .85, claw * 1.1, spread, s, bones, floor=0.0,
               sole=False, palm=palm)
    base = len(part.points)
    for p, uv, w, isl, tg in zip(tmp.points, tmp.uv, tmp.weights, tmp.island, tmp.tag):
        part.points.append(np.asarray(wrist, float) + R @ (p - np.array([0, .02, 0])))
        part.uv.append(uv)
        part.weights.append(dict(w))
        part.island.append(isl)
        part.tag.append(dict(tg))
    for a, b, c in tmp.tris:
        part.tris.append((a + base, b + base, c + base))


def cap_ring(part, ring, point, bone, prev=None):
    """v5: close a loft's open ring with a cone to `point` (weighted to `bone`); the ring's own UVs are
    kept, and each cone face gets its own copy of the tip whose UV lies a little back along the limb's
    strip (toward the ring `prev`), so the cap shows the limb's own skin, not a smear of one texel."""
    point = np.asarray(point, float)
    cen = np.mean([part.points[j] for j in ring], 0)
    inside = cen - (point - cen) * 0.5
    n = len(ring)
    for k in range(n):
        a, b = ring[k], ring[(k + 1) % n]
        u = (part.uv[a][0] + part.uv[b][0]) / 2
        v = (part.uv[a][1] + part.uv[b][1]) / 2
        if prev is not None:
            v += (np.mean([part.uv[j][1] for j in prev]) - v) * .35
        e = part.add(point, (u, v), bone, part.island[a], **dict(part.tag[a]))
        part.tri(e, a, b, inside=inside)


def drop_island(part, island):
    """Remove an island's triangles and the vertices only they used."""
    keep = [t for t in part.tris if part.island[t[0]] != island]
    used = sorted({v for t in keep for v in t})
    remap = {v: i for i, v in enumerate(used)}
    for attr in ("points", "uv", "weights", "island", "tag"):
        setattr(part, attr, [getattr(part, attr)[v] for v in used])
    part.tris = [tuple(remap[v] for v in t) for t in keep]


def _cast(parts, origin, direction):
    """-> (the nearest hit point, its face normal) of a ray against the parts' triangles, or None."""
    best = None
    for p in parts:
        P = np.array(p.points)
        for a, b, c in p.tris:
            A, e1, e2 = P[a], P[b] - P[a], P[c] - P[a]
            pv = np.cross(direction, e2)
            det = e1 @ pv
            if abs(det) < 1e-12:
                continue
            tv = origin - A
            u = (tv @ pv) / det
            qv = np.cross(tv, e1)
            v = (direction @ qv) / det
            t = (e2 @ qv) / det
            if u < 0 or v < 0 or u + v > 1 or t <= 0:
                continue
            if best is None or t < best[0]:
                nn = _unit(np.cross(e1, e2))
                best = (t, origin + direction * t, nn if nn @ direction < 0 else -nn)
    return None if best is None else best[1:]


# v4: the gold chain as jewellery - a thin, loose chain over his nape, draped low over the plastron in a
# soft U (not tight round the throat), with a small gold medallion (the coin's face) at the bottom.
CHAIN_W = .02               # the chain's width
CHAIN_SHOULDER = (.112, .53)     # (x, y) where it comes out from beside the neck onto the plastron
CHAIN_BOTTOM = .39          # y of the bottom of the U, on the plastron
CHAIN_FRONT_N = 6            # segments across the front U
CHAIN_LINKS = 24             # painted links along the whole loop (krefft_texture.paint_chain_v4)
PENDANT_R = .017
BACK_XZ = ((.088, .124), (.048, .13), (0, .128), (-.048, .13), (-.088, .124))    # (x, z) round the nape, his left to his right


def build_jewellery_chain(parts, skel):
    from scripts.krefft_model import HEADT
    body = [p for p in parts if p.name in ("shell", "neck", "head", "jaw")]
    skin = [p for p in body if p.name in ("shell", "neck")]       # (behind the jaw: never on it)
    lift = .0045
    pts, outs = [], []
    X, Y = CHAIN_SHOULDER
    # over the nape, between the back of the head and the shell's rim (cast from above), from his
    # left shoulder round to his right: seen behind the head from DK2's high camera
    back = []
    for x, z in BACK_XZ:
        hit = _cast(skin, np.array([x, 1.0, z]), np.array([0, -1.0, 0]))
        back.append((hit[0] + hit[1] * lift, hit[1]))
    # the front: a soft U on the plastron, cast onto the body from the front
    front = []
    for k in range(CHAIN_FRONT_N + 1):
        sx = -1 + 2 * k / CHAIN_FRONT_N
        x = X * sx
        y = CHAIN_BOTTOM + (Y - CHAIN_BOTTOM) * abs(sx) ** 2.2        # a soft U: rounded at the bottom
        hit = _cast(skin, np.array([x, y, 1.0]), np.array([0, 0, -1.0]))
        front.append((hit[0] + hit[1] * lift, hit[1]))
    # the loop: his right shoulder (x < 0) across the front to his left, then back round the nape
    loop = front + back
    P = np.array([p for p, _ in loop])
    O = np.array([o for _, o in loop])
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    ch = KrefftPart("chain", HEADT)
    n = len(P)
    ids = []
    for i in range(n + 1):
        j = i % n
        t = _unit(P[(j + 1) % n] - P[(j - 1) % n])
        side = _unit(np.cross(O[j], t))
        w = {"root": 1.0} if j < len(front) else {"neck1": 1.0}
        u = L[i] / L[-1]
        cw = CHAIN_W * (1.0 if j < len(front) else 1.2)       # a little wider over the nape (the high camera)
        ids.append((ch.add(P[j] + side * cw / 2, (u, 0.0), w, "chain", k=i),
                    ch.add(P[j] - side * cw / 2, (u, 1.0), w, "chain", k=i), O[j]))
    for i in range(n):
        (a, b, oa), (c, d, oc) = ids[i], ids[i + 1]
        inside = (ch.points[a] + ch.points[d]) / 2 - (oa + oc) * .05
        ch.tri(a, c, b, inside=inside)
        ch.tri(c, d, b, inside=inside)
    # the medallion: a small gold disc hanging just under the U's bottom, lying on the plastron
    bot, nb_ = front[CHAIN_FRONT_N // 2]
    cen = bot + np.array([0, -PENDANT_R * .85, 0]) + nb_ * .002
    hit = _cast(skin, np.array([cen[0], cen[1], 1.0]), np.array([0, 0, -1.0]))
    if hit:
        cen = hit[0] + hit[1] * (lift + .0015)
        nb_ = hit[1]
    upd = _unit(np.array([0, 1.0, 0]) - nb_ * nb_[1])
    lat = np.cross(upd, nb_)
    coins = next(p for p in parts if p.name == "coins")
    ts = [2 * np.pi * k / 6 + np.pi / 2 for k in range(6)]
    pid = [coins.add(cen + PENDANT_R * (np.cos(t) * lat + np.sin(t) * upd),
                     (.5 + .5 * np.cos(t), .5 + .5 * np.sin(t)), "root", "medal", face=1) for t in ts]
    for k in range(1, 5):
        coins.tri(pid[0], pid[k], pid[k + 1], inside=cen - nb_ * .05)
    return ch


def build_standing(v4=False, v5=False):
    """-> (parts in B's bind pose, B's Skeleton). `v4`: Soles' v4 changes (one coin wedged in its crack,
    no gold tooth, the chain as jewellery). `v5` (implies v4): no chain or medallion at all, and every limb
    joint closed (the arms capped into the hands, the hands and feet closed underneath, the legs capped
    into the feet, the limbs' inner ends capped inside the shell)."""
    v4 = v4 or v5
    parts = [p for p in build_krefft(SHELL_DEPTH_B, BELLY_B, v4=v4) if p.name != "legs"]
    Rw, Pw = pose()
    posed = krefft_skinner(parts)(Rw, Pw)
    lo = min(P[:, 1].min() for P, p in zip(posed, parts) if p.name == "shell")
    lift = np.array([0, LEG_TOP - lo, 0])
    for p, P in zip(parts, posed):
        p.points = [q + lift for q in P]
    # the shell's frame in B: its middle, its front (the plastron) and top
    shell = np.array(parts[0].points)
    smin, smax = shell.min(0), shell.max(0)
    front = smax[2]                                       # the plastron's front face
    # the skeleton in B's space: A's bones moved by the pose, the limbs' bones laid on the new limbs
    heads = {n: Pw[i] + lift for i, n in enumerate(SKEL.names)}
    tails = {n: Pw[i] + lift + Rw[i] @ (SKEL.tail[i] - SKEL.head[i]) for i, n in enumerate(SKEL.names)}
    yb = smin[1]
    leg = np.array([(.105, yb + .06, front - .11), (.17, yb - .012, front - .05), (.162, .05, front - .068)])  # knees out
    sh_y = smax[1] - .2
    arm = np.array([(.15, sh_y, front - .1), (.25, sh_y - .075, front - .025), (.215, sh_y - .125, front + .07)])  # elbows out
    for s, side in ((1, "L"), (-1, "R")):
        m = np.array([s, 1, 1.0])
        for names, path in ((("thigh", "shin", "foot"), leg), (("upperF", "foreF", "handF"), arm)):
            pts = list(path * m) + [path[-1] * m + (np.array([0, -.05, .07]) if names[0] == "thigh"
                                                    else np.array([.0, -.03, .08]) * m)]
            for k, n in enumerate(names):
                heads[n + side] = pts[k]
                tails[n + side] = pts[k + 1]
    bones = [(n, SKEL.bones[i][1], heads[n], tails[n]) for i, n in enumerate(SKEL.names)]
    skel = Skeleton(bones)
    # the limbs
    part = KrefftPart("legs", BODY)
    for s, side in ((1, "L"), (-1, "R")):
        m = np.array([s, 1, 1.0])
        lb = tuple(n + side for n in ("thigh", "shin", "foot"))
        ab = tuple(n + side for n in ("upperF", "foreF", "handF"))
        # the legs: short and thick, the knee forward, a big paddle foot flat on the floor
        ankle = leg[-1] * m
        # 6-sided and open at the ankle (the paddle foot closes it): no boot-like cap or sole
        rings = gloft(part, "hleg", leg * m, LEG_RADII, 6, (0, .3, 1.0), _weights_v5(lb) if v5 else _weights(skel, lb),
                      tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
        if v5:            # the hip end inside the shell, the ankle end down into the foot
            hip = leg[0] * m
            # (the inner caps ride on the shell: they stay inside it when the limb swings)
            cap_ring(part, rings[0], hip + np.array([-.045 * s, .01, -.03]), "root", rings[1])
            cap_ring(part, rings[-1], np.array([ankle[0], .022, ankle[2] - .01]), {lb[1]: .5, lb[2]: .5}, rings[-2])
        wrist, fwd, length, width, claw, spread = HFOOT
        build_foot(part, "hfoot", np.array([ankle[0], .02, ankle[2] - .045]), (.2, 1.0), length * 1.45,
                   width * 1.25, claw * 1.15, 84, s, lb, floor=0.0, sole=False, palm=v5)
        # the arms: stubby, hanging forward, the webbed paddle hands turned down and in
        wr = arm[-1] * m
        rings = gloft(part, "fleg", arm * m, ARM_RADII, 3, (0, 1.0, .2), _weights_v5(ab) if v5 else _weights(skel, ab),
                      tag=lambda i, th, s=s: {"theta": th, "ring": i, "side": s}, side_axis_sign=s)
        f = _unit(np.array([-.15 * s, -.35, 1.0]))                          # the hand: forward, a little down
        if v5:            # the shoulder end inside the shell; the wrist end runs on into the hand
            cap_ring(part, rings[0], arm[0] * m + np.array([-.05 * s, 0, -.035]), "root", rings[1])
            cap_ring(part, rings[-1], wr + f * .055 + np.array([0, -.006, 0]), {ab[1]: .5, ab[2]: .5}, rings[-2])
        up = _unit(np.cross(np.array([1.0, 0, 0]), f))                       # its back faces forward-up
        if up[2] < 0:
            up = -up
        R = np.stack([np.cross(up, f), up, f], 1)       # columns: x (left), y (the paddle's up), z (along)
        _paddle(part, "ffoot", wr + np.array([0, -.01, .015]), R, FFOOT, s, ab, palm=v5)
    parts.insert(len(parts) - 1, part)
    # a bigger head (x HEAD_SCALE about the head joint), sat down into the shell's opening
    hj = heads["head"]
    hb = BONE_INDEX["head"]
    for p in parts:
        if p.name in ("head", "jaw"):
            p.points = [hj + (q - hj) * HEAD_SCALE + HEAD_DROP for q in p.points]
        elif p.name == "neck":            # the neck's end follows the head by its weight to the head bone
            p.points = [q + w.get(hb, 0.0) * (hj + (q - hj) * HEAD_SCALE + HEAD_DROP - q)
                        for q, w in zip(p.points, p.weights)]
    if v4:
        drop_island(next(p for p in parts if p.name == "head"), "tooth")
        i = next(k for k, p in enumerate(parts) if p.name == "chain")
        if v5:            # Soles (16 Nov): "remove the gold chain completely" - no chain, no medallion
            del parts[i]
        else:
            parts[i] = build_jewellery_chain(parts, skel)
    for p in parts:
        orient(p)
    return parts, skel


if __name__ == "__main__":
    import sys
    ps, sk = build_standing("--v4" in sys.argv)
    c = {p.name: len(p.tris) for p in ps}
    allp = np.concatenate([np.array(p.points) for p in ps])
    print(c, "total", sum(c.values()), "extent", allp.max(0) - allp.min(0))
