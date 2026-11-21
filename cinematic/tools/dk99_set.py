"""Dungeon set for the 1999-style Molynx cinematic (after the Dungeon Keeper 2 intro: stone-block hall,
pointed arches on pillars, torches, a bright cool archway at the back, haze).
blender -b anim.blend --python dk99_set.py -- --save dk99.blend
Blender coords: z up, Molynx stands at the origin facing -y (towards the usual camera)."""
import argparse
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument('--save', default='/home/claude/molynx_blender/dk99/dk99.blend')
A = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
S = bpy.context.scene
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- clear the studio
for n in ('backdrop', 'floor', 'key', 'fill', 'top', 'rim warm'):
    o = bpy.data.objects.get(n)
    if o:
        bpy.data.objects.remove(o, do_unlink=True)

coll = bpy.data.collections.new('dungeon'); S.collection.children.link(coll)

# ---------------------------------------------------------------- hall dimensions
WX = 5.6          # inner half-width of the hall (wall faces)
Y0, Y1 = -16.0, 9.0   # front wall (behind the usual camera), back wall
PX = 4.75         # pillar centre x
PILLARS_Y = [-12.0, -6.0, 0.0, 6.0]
SPRING = 5.2      # arch springing height
ARCH_R = 5.6      # pointed-arch radius for the cross ribs


# ---------------------------------------------------------------- mesh helpers
def mesh_obj(name, V, F, mat=None, attrs=None, smooth=False, bevel=0.0, parent=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V], [], [tuple(int(i) for i in f) for f in F])
    me.update()
    if attrs:
        for k, (dom, vals) in attrs.items():
            a = me.attributes.new(k, 'FLOAT', dom); a.data.foreach_set('value', np.asarray(vals, np.float32))
    for p in me.polygons:
        p.use_smooth = smooth
    o = bpy.data.objects.new(name, me); coll.objects.link(o)
    if mat:
        me.materials.append(mat)
    if bevel:
        m = o.modifiers.new('bevel', 'BEVEL'); m.width = bevel; m.segments = 2; m.limit_method = 'ANGLE'
        m.harden_normals = False
    if parent:
        o.parent = parent
    return o


class Builder:
    def __init__(s):
        s.V = []; s.F = []; s.rnd = []

    def box(s, c, h, r=None, rot=0.0):
        """axis-aligned box (optionally rotated about z) centre c, half size h, one random value per box"""
        c = np.asarray(c, float); h = np.asarray(h, float)
        r = rng.random() if r is None else r
        n = len(s.V)
        cs, sn = math.cos(rot), math.sin(rot)
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    x, y = dx * h[0], dy * h[1]
                    s.V.append(c + (x * cs - y * sn, x * sn + y * cs, dz * h[2]))
        for f in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            s.F.append([n + i for i in f]); s.rnd.append(r)

    def quad_strip(s, P, Q, r=0.5):
        """faces between two point rows"""
        n = len(s.V); m = len(P)
        s.V += list(P) + list(Q)
        for i in range(m - 1):
            s.F.append([n + i, n + i + 1, n + m + i + 1, n + m + i]); s.rnd.append(r)

    def obj(s, name, mat, bevel=0.0, smooth=False):
        return mesh_obj(name, s.V, s.F, mat, {'rnd': ('FACE', s.rnd)}, smooth=smooth, bevel=bevel)


# ---------------------------------------------------------------- materials
def nodes(mat):
    mat.use_nodes = True; nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); return nt, out


def stone_mat(name, c0, c1, scale=1.6, bump=0.35, rough=0.72):
    m = bpy.data.materials.new(name); nt, out = nodes(m); N = nt.nodes; L = nt.links
    bs = N.new('ShaderNodeBsdfPrincipled'); L.new(bs.outputs[0], out.inputs[0])
    tc = N.new('ShaderNodeTexCoord')
    at = N.new('ShaderNodeAttribute'); at.attribute_name = 'rnd'; at.attribute_type = 'GEOMETRY'
    nz = N.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = scale; nz.inputs['Detail'].default_value = 6
    nz.inputs['Roughness'].default_value = 0.6
    L.new(tc.outputs['Object'], nz.inputs['Vector'])
    cr = N.new('ShaderNodeValToRGB'); cr.color_ramp.elements[0].color = (*c0, 1); cr.color_ramp.elements[1].color = (*c1, 1)
    cr.color_ramp.elements[0].position = 0.42; cr.color_ramp.elements[1].position = 0.98
    mx = N.new('ShaderNodeMath'); mx.operation = 'MULTIPLY_ADD'; mx.inputs[1].default_value = 0.42
    L.new(at.outputs['Fac'], mx.inputs[0]); L.new(nz.outputs['Fac'], mx.inputs[2])        # block tint + noise
    L.new(mx.outputs[0], cr.inputs[0])
    # painted grime: big soft blotches darken
    nz2 = N.new('ShaderNodeTexNoise'); nz2.inputs['Scale'].default_value = 0.35; nz2.inputs['Detail'].default_value = 3
    L.new(tc.outputs['Object'], nz2.inputs['Vector'])
    g = N.new('ShaderNodeMapRange'); g.inputs[1].default_value = 0.35; g.inputs[2].default_value = 0.7
    g.inputs[3].default_value = 0.55; g.inputs[4].default_value = 1.1
    L.new(nz2.outputs['Fac'], g.inputs[0])
    mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1
    L.new(cr.outputs[0], mul.inputs[6]); L.new(g.outputs[0], mul.inputs[7])
    st = N.new('ShaderNodeTexNoise'); st.inputs['Scale'].default_value = scale * 3.5; st.inputs['Detail'].default_value = 8
    st.inputs['Roughness'].default_value = 0.7; L.new(tc.outputs['Object'], st.inputs['Vector'])
    stc = N.new('ShaderNodeValToRGB'); stc.color_ramp.elements[0].position = 0.42; stc.color_ramp.elements[0].color = (0.5, 0.52, 0.42, 1)
    stc.color_ramp.elements[1].position = 0.6; stc.color_ramp.elements[1].color = (1, 1, 1, 1)
    L.new(st.outputs['Fac'], stc.inputs[0])
    ms = N.new('ShaderNodeMix'); ms.data_type = 'RGBA'; ms.blend_type = 'MULTIPLY'; ms.inputs['Factor'].default_value = 1
    L.new(mul.outputs[2], ms.inputs[6]); L.new(stc.outputs[0], ms.inputs[7]); mul = ms
    sz = N.new('ShaderNodeSeparateXYZ'); L.new(tc.outputs['Object'], sz.inputs[0])
    hz_ = N.new('ShaderNodeMapRange'); hz_.inputs[1].default_value = 2.5; hz_.inputs[2].default_value = 9.5
    hz_.inputs[3].default_value = 1.0; hz_.inputs[4].default_value = 0.3
    L.new(sz.outputs['Z'], hz_.inputs[0])
    mz = N.new('ShaderNodeMix'); mz.data_type = 'RGBA'; mz.blend_type = 'MULTIPLY'; mz.inputs['Factor'].default_value = 1
    L.new(mul.outputs[2], mz.inputs[6]); L.new(hz_.outputs[0], mz.inputs[7]); mul = mz
    ao = N.new('ShaderNodeAmbientOcclusion'); ao.inputs['Distance'].default_value = 0.25; ao.only_local = True; ao.samples = 3
    L.new(mul.outputs[2], ao.inputs['Color'])
    mao = N.new('ShaderNodeMix'); mao.data_type = 'RGBA'; mao.blend_type = 'MULTIPLY'; mao.inputs['Factor'].default_value = 0.75
    L.new(mul.outputs[2], mao.inputs[6]); L.new(ao.outputs['AO'], mao.inputs[7])
    L.new(mao.outputs[2], bs.inputs['Base Color'])
    bs.inputs['Roughness'].default_value = rough
    bs.inputs['Specular IOR Level'].default_value = 0.35
    # chisel bumps
    vo = N.new('ShaderNodeTexVoronoi'); vo.feature = 'DISTANCE_TO_EDGE'; vo.inputs['Scale'].default_value = scale * 2.2
    L.new(tc.outputs['Object'], vo.inputs['Vector'])
    ad = N.new('ShaderNodeMath'); ad.operation = 'ADD'
    sm = N.new('ShaderNodeMath'); sm.operation = 'SMOOTH_MIN'; sm.inputs[1].default_value = 0.12; sm.inputs[2].default_value = 0.1
    L.new(vo.outputs['Distance'], sm.inputs[0])
    L.new(sm.outputs[0], ad.inputs[0]); L.new(nz.outputs['Fac'], ad.inputs[1])
    bp = N.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump; bp.inputs['Distance'].default_value = 0.05
    L.new(ad.outputs[0], bp.inputs['Height']); L.new(bp.outputs[0], bs.inputs['Normal'])
    return m


def simple_mat(name, col, rough=0.5, metal=0.0, emit=None, estr=0.0):
    m = bpy.data.materials.new(name); nt, out = nodes(m)
    bs = nt.nodes.new('ShaderNodeBsdfPrincipled'); nt.links.new(bs.outputs[0], out.inputs[0])
    bs.inputs['Base Color'].default_value = (*col, 1); bs.inputs['Roughness'].default_value = rough
    bs.inputs['Metallic'].default_value = metal
    if emit:
        bs.inputs['Emission Color'].default_value = (*emit, 1); bs.inputs['Emission Strength'].default_value = estr
    return m


def gold_mat():
    m = bpy.data.materials.new('dk gold'); nt, out = nodes(m); N = nt.nodes; L = nt.links
    bs = N.new('ShaderNodeBsdfPrincipled'); L.new(bs.outputs[0], out.inputs[0])
    bs.inputs['Base Color'].default_value = (0.75, 0.48, 0.16, 1); bs.inputs['Metallic'].default_value = 1
    bs.inputs['Roughness'].default_value = 0.32
    tc = N.new('ShaderNodeTexCoord'); w = N.new('ShaderNodeTexWave'); w.wave_type = 'RINGS'; w.rings_direction = 'SPHERICAL'
    w.inputs['Scale'].default_value = 3.0; w.inputs['Distortion'].default_value = 6; w.inputs['Detail'].default_value = 2
    L.new(tc.outputs['Object'], w.inputs['Vector'])
    bp = N.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.6; bp.inputs['Distance'].default_value = 0.02
    L.new(w.outputs['Fac'], bp.inputs['Height']); L.new(bp.outputs[0], bs.inputs['Normal'])
    return m


def flame_mat():
    m = bpy.data.materials.new('flame'); nt, out = nodes(m); N = nt.nodes; L = nt.links
    tc = N.new('ShaderNodeTexCoord')
    nz = N.new('ShaderNodeTexNoise'); nz.noise_dimensions = '4D'; nz.inputs['Scale'].default_value = 3.5
    nz.inputs['Detail'].default_value = 3; nz.name = 'flame noise'
    L.new(tc.outputs['Generated'], nz.inputs['Vector'])
    gr = N.new('ShaderNodeSeparateXYZ'); L.new(tc.outputs['Generated'], gr.inputs[0])
    # hotter at the bottom, flickery top
    mr = N.new('ShaderNodeMath'); mr.operation = 'MULTIPLY_ADD'; mr.inputs[1].default_value = -0.9; mr.inputs[2].default_value = 1.15
    L.new(gr.outputs['Z'], mr.inputs[0])
    mm = N.new('ShaderNodeMath'); mm.operation = 'MULTIPLY'
    L.new(mr.outputs[0], mm.inputs[0]); L.new(nz.outputs['Fac'], mm.inputs[1])
    cr = N.new('ShaderNodeValToRGB'); E = cr.color_ramp.elements
    E[0].position = 0.22; E[0].color = (0.0, 0.0, 0.0, 1); E[1].position = 0.85; E[1].color = (1.0, 0.7, 0.3, 1)
    e = E.new(0.4); e.color = (0.85, 0.18, 0.01, 1)
    L.new(mm.outputs[0], cr.inputs[0])
    em = N.new('ShaderNodeEmission'); em.inputs['Strength'].default_value = 7
    L.new(cr.outputs[0], em.inputs['Color'])
    tr = N.new('ShaderNodeBsdfTransparent')
    mx = N.new('ShaderNodeMixShader')
    al = N.new('ShaderNodeMath'); al.operation = 'GREATER_THAN'; al.inputs[1].default_value = 0.3
    L.new(mm.outputs[0], al.inputs[0])
    L.new(al.outputs[0], mx.inputs[0]); L.new(tr.outputs[0], mx.inputs[1]); L.new(em.outputs[0], mx.inputs[2])
    L.new(mx.outputs[0], out.inputs[0])
    m.blend_method = 'BLEND'
    return m


M_wall = stone_mat('dk wall stone', (0.05, 0.035, 0.025), (0.22, 0.16, 0.1))
M_vault = stone_mat('dk vault stone', (0.03, 0.022, 0.016), (0.1, 0.075, 0.05))
M_floor = stone_mat('dk floor stone', (0.045, 0.033, 0.024), (0.2, 0.15, 0.1), scale=1.2, bump=0.25, rough=0.5)
M_pillar = stone_mat('dk pillar stone', (0.07, 0.05, 0.033), (0.3, 0.21, 0.13), scale=2.0)
M_mortar = simple_mat('dk mortar', (0.03, 0.025, 0.02), 0.95)
M_gold = gold_mat()
M_iron = bpy.data.materials.get('armour iron') or simple_mat('dk iron', (0.12, 0.12, 0.13), 0.45, 1.0)
M_flame = flame_mat()
M_wood = stone_mat('dk wood', (0.09, 0.05, 0.025), (0.22, 0.13, 0.06), scale=0.9, bump=0.2, rough=0.6)
M_portal = simple_mat('portal light', (0, 0, 0), 1.0, emit=(0.62, 0.8, 1.0), estr=9)

# ---------------------------------------------------------------- floor: irregular flagstones over mortar
B = Builder()
y = Y0
while y < Y1:
    d = rng.uniform(0.8, 1.3)
    x = -WX
    while x < WX:
        w = min(rng.uniform(0.8, 1.5), WX - x)
        if w > 0.15:
            B.box((x + w / 2, y + d / 2, -0.06 + rng.uniform(-0.012, 0.012)), (w / 2 - 0.025, d / 2 - 0.025, 0.06),
                  rot=rng.uniform(-0.01, 0.01))
        x += w
    y += d
B.obj('flagstones', M_floor, bevel=0.03)
mesh_obj('mortar floor', [(-WX - 1, Y0 - 1, -0.07), (WX + 1, Y0 - 1, -0.07), (WX + 1, Y1 + 6, -0.07), (-WX - 1, Y1 + 6, -0.07)],
         [(0, 1, 2, 3)], M_mortar)


# ---------------------------------------------------------------- walls: staggered blocks
def block_wall(name, axis_pts, height, depth_dir, skip=None, mortar=True):
    """blocks along a horizontal line from p0 to p1 (x,y), rising to height; depth_dir = unit (x,y) out of the hall"""
    p0, p1 = np.array(axis_pts[0], float), np.array(axis_pts[1], float)
    L = np.linalg.norm(p1 - p0); u = (p1 - p0) / L; dd = np.array(depth_dir, float)
    rot = math.atan2(u[1], u[0])
    B = Builder(); z = 0.0; row = 0
    while z < height:
        hgt = rng.uniform(0.5, 0.85)
        s = -rng.uniform(0, 0.9)
        while s < L:
            w = rng.uniform(0.7, 1.9)
            a, b = max(s, 0), min(s + w, L)
            pieces = [(a, b)]
            if skip:                                   # clip the block around an opening (interval in s at this row)
                hole = skip(z)
                if hole:
                    lo, hi = hole; pieces = [(a, min(b, lo)), (max(a, hi), b)]
            for a, b in pieces:
                if b - a > 0.1:
                    mid = (a + b) / 2
                    c2 = p0 + u * mid
                    out = rng.uniform(-0.03, 0.05)
                    c3 = (c2[0] + dd[0] * (0.2 - out), c2[1] + dd[1] * (0.2 - out), z + hgt / 2)
                    B.box(c3, ((b - a) / 2 - 0.03, 0.2, hgt / 2 - 0.03), rot=rot + rng.uniform(-0.012, 0.012))
            s += w
        z += hgt; row += 1
    o = B.obj(name, M_wall, bevel=0.06)
    if not mortar:
        return o
    # mortar backing
    q = [p0 + dd * 0.38, p1 + dd * 0.38]
    mesh_obj(name + ' mortar', [(q[0][0], q[0][1], 0), (q[1][0], q[1][1], 0), (q[1][0], q[1][1], height), (q[0][0], q[0][1], height)],
             [(0, 1, 2, 3)], M_mortar)
    return o


WALL_H = 11.0
block_wall('wall L', ((-WX, Y0), (-WX, Y1)), WALL_H, (-1, 0))
block_wall('wall R', ((WX, Y1), (WX, Y0)), WALL_H, (1, 0))
DOOR_W, DOOR_SPRING = 2.0, 4.2      # half-width and springing height of the back archway


def door_shape(x, z, pad=0.0):
    """inside the pointed archway on the back wall (x across, z up)"""
    hw = DOOR_W + pad
    if abs(x) > hw:
        return False
    if z < DOOR_SPRING:
        return True
    R = 2 * hw * 0.85                    # pointed arch: arcs centred at +-(R - hw)
    cx = R - hw
    return (abs(x) + cx) ** 2 + (z - DOOR_SPRING) ** 2 < R * R


def door_hole(z, pad=0.25):
    """interval of the back wall's s coordinate (s = x + WX) cut away for the archway at height z"""
    hw = DOOR_W + pad
    if z < DOOR_SPRING:
        x = hw
    else:
        R = 2 * hw * 0.85; cx = R - hw; q = R * R - (z - DOOR_SPRING) ** 2
        if q <= 0:
            return None
        x = math.sqrt(q) - cx
        if x <= 0:
            return None
    return (WX - x, WX + x)


block_wall('wall back', ((-WX, Y1), (WX, Y1)), WALL_H, (0, 1), skip=door_hole, mortar=False)
Vm = []; Fm = []
for i_ in range(int(2 * WX / 0.1)):
    for j_ in range(int(WALL_H / 0.1)):
        x_ = -WX + (i_ + 0.5) * 0.1; z_ = (j_ + 0.5) * 0.1
        if door_shape(x_, z_, 0.42):
            continue
        n_ = len(Vm); x0_, z0_ = x_ - 0.05, z_ - 0.05
        Vm += [(x0_, Y1 + 0.39, z0_), (x0_ + 0.1, Y1 + 0.39, z0_), (x0_ + 0.1, Y1 + 0.39, z0_ + 0.1), (x0_, Y1 + 0.39, z0_ + 0.1)]
        Fm.append((n_, n_ + 1, n_ + 2, n_ + 3))
mo_ = mesh_obj('wall back mortar', Vm, Fm, M_mortar)
mo_.modifiers.new('weld', 'WELD')
block_wall('wall front', ((WX, Y0), (-WX, Y0)), WALL_H, (0, -1))


# ---------------------------------------------------------------- pointed-arch geometry
def pointed_arc(hw, spring, R, n=24):
    """points (x,z) of a pointed arch of half-span hw from left springing to right springing"""
    cx = R - hw
    ta = math.acos(-cx / R)      # left arc (centre +cx): from angle pi (left springing) to ta (apex, x = 0)
    tb = math.acos(cx / R)       # right arc (centre -cx): from tb (apex) to 0 (right springing)
    pts = []
    for i in range(n + 1):
        a = math.pi + (ta - math.pi) * i / n
        pts.append((cx + R * math.cos(a), spring + R * math.sin(a)))
    for i in range(1, n + 1):
        a = tb * (1 - i / n)
        pts.append((-cx + R * math.cos(a), spring + R * math.sin(a)))
    return pts


def sweep_rect(pts3, normal_side, w, d):
    """box section swept along a polyline in a vertical plane. pts3: list of 3D points;
    normal_side: unit vector across the plane (thickness direction); w: radial half-depth, d: half-thickness"""
    P = np.array(pts3); n = len(P)
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    S_ = np.array(normal_side, float)
    Nn = np.cross(T, S_); Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
    V = []; F = []
    for i in range(n):
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            V.append(P[i] + Nn[i] * a * w + S_ * b * d)
    for i in range(n - 1):
        for k in range(4):
            a, b = i * 4 + k, i * 4 + (k + 1) % 4
            F.append((a, b, b + 4, a + 4))
    F.append((0, 1, 2, 3)); F.append(((n - 1) * 4 + 3, (n - 1) * 4 + 2, (n - 1) * 4 + 1, (n - 1) * 4))
    return V, F


def segmented_rib(name, pts3, side, w, d, mat, nseg=9):
    """an arch rib built from voussoir blocks (separate bevelled pieces)"""
    P = np.array(pts3); cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    edges = np.linspace(0, cum[-1], nseg + 1)
    VV = []; FF = []; rr = []
    for i in range(nseg):
        a, b = edges[i] + 0.02, edges[i + 1] - 0.02
        ts = np.linspace(a, b, 5)
        Q = np.stack([np.interp(ts, cum, P[:, k]) for k in range(3)], 1)
        V, F = sweep_rect(Q, side, w * rng.uniform(0.97, 1.04), d)
        n0 = len(VV); VV += V; FF += [tuple(x + n0 for x in f) for f in F]; rr += [rng.random()] * len(F)
    o = mesh_obj(name, VV, FF, mat, {'rnd': ('FACE', rr)}, bevel=0.03)
    return o


ribs = []
for i, py in enumerate(PILLARS_Y):
    # pillar: square shaft with base and capital, built from stacked drums
    B = Builder(); z = 0.0
    B.box((PX, py, 0.25), (0.62, 0.62, 0.25)); B.box((-PX, py, 0.25), (0.62, 0.62, 0.25))
    z = 0.5
    while z < SPRING - 0.4:
        hgt = rng.uniform(0.55, 0.75)
        for sx in (-1, 1):
            B.box((sx * PX, py, z + hgt / 2), (0.46, 0.46, hgt / 2 - 0.015))
        z += hgt
    for sx in (-1, 1):
        B.box((sx * PX, py, SPRING - 0.2), (0.6, 0.6, 0.2))
    B.obj(f'pillars {i}', M_pillar, bevel=0.04)
    # cross rib (pointed arch across the hall)
    arc = pointed_arc(PX, SPRING, ARCH_R)
    ribs.append(segmented_rib(f'rib {i}', [(x, py, z_) for x, z_ in arc], (0, 1, 0), 0.32, 0.36, M_pillar, nseg=13))
    # gold band under the capital
    G = Builder()
    for sx in (-1, 1):
        G.box((sx * PX, py, SPRING - 0.45), (0.5, 0.5, 0.06))
    G.obj(f'gold band {i}', M_gold, bevel=0.02)

# wall arches between pillars (along each side wall)
for i in range(len(PILLARS_Y) - 1):
    ya, yb = PILLARS_Y[i], PILLARS_Y[i + 1]; hw = (yb - ya) / 2 - 0.45; cy = (ya + yb) / 2
    arc = pointed_arc(hw, SPRING - 0.4, hw * 1.7)
    for sx in (-1, 1):
        segmented_rib(f'wall arch {i}{sx}', [(sx * (WX - 0.15), cy + x, z_) for x, z_ in arc], (1, 0, 0), 0.22, 0.2, M_pillar, nseg=9)

# vault: pointed barrel along y, closing the top (dark, mostly lost in shadow)
arc = pointed_arc(WX, SPRING + 0.6, ARCH_R * 1.12, n=30)
V = []; F = []; ny = 40
ys = np.linspace(Y0, Y1, ny)
for j, yy in enumerate(ys):
    for x, z_ in arc:
        V.append((x, yy, z_ + 0.35))
m = len(arc)
for j in range(ny - 1):
    for i in range(m - 1):
        a = j * m + i; F.append((a, a + m, a + m + 1, a + 1))
mesh_obj('vault', V, F, M_vault, {'rnd': ('FACE', rng.random(len(F)))}, smooth=True)

# ---------------------------------------------------------------- back archway: gold-trimmed, bright cool light beyond
arc = pointed_arc(DOOR_W + 0.2, DOOR_SPRING, 2 * (DOOR_W + 0.2) * 0.85)
pts = [(-(DOOR_W + 0.2), Y1 - 0.05, 0.0)] + [(x, Y1 - 0.05, z_) for x, z_ in arc] + [(DOOR_W + 0.2, Y1 - 0.05, 0.0)]
V, F = sweep_rect(pts, (0, 1, 0), 0.4, 0.3); mesh_obj('door frame', V, F, M_pillar, {'rnd': ('FACE', rng.random(len(F)))}, bevel=0.03)
arcg = pointed_arc(DOOR_W + 0.5, DOOR_SPRING, 2 * (DOOR_W + 0.5) * 0.85)
pts2 = [(-(DOOR_W + 0.5), Y1 - 0.32, 0.0)] + [(x, Y1 - 0.32, z_) for x, z_ in arcg] + [(DOOR_W + 0.5, Y1 - 0.32, 0.0)]
V, F = sweep_rect(pts2, (0, 1, 0), 0.08, 0.07); mesh_obj('door gold', V, F, M_gold, smooth=True)
# corridor beyond the arch and the light source
B = Builder()
for sx in (-1, 1):
    B.box((sx * (DOOR_W + 0.6), Y1 + 3, 3.5), (0.4, 3, 3.5))
B.box((0, Y1 + 3, 7.4), (DOOR_W + 1, 3, 0.4))
B.obj('door corridor', M_wall, bevel=0.03)
mesh_obj('portal glow', [(-4, Y1 + 5.8, -0.1), (4, Y1 + 5.8, -0.1), (4, Y1 + 5.8, 8), (-4, Y1 + 5.8, 8)], [(0, 1, 2, 3)], M_portal)
sp = bpy.data.lights.new('door beam', 'SPOT'); sp.energy = 3200; sp.spot_size = math.radians(48); sp.spot_blend = 0.4
sp.color = (0.66, 0.8, 1.0); sp.shadow_soft_size = 0.4
so = bpy.data.objects.new('door beam', sp); coll.objects.link(so); so.location = (0.0, Y1 + 5.0, 6.2)
so.rotation_euler = (Vector((0, -2.5, 0)) - so.location).to_track_quat('-Z', 'Y').to_euler()

# ---------------------------------------------------------------- torches on the pillars' inner faces
torches = []
TORCH_E = 95
TORCH_POS = [(sx, py) for py in PILLARS_Y for sx in (-1, 1)]
for k, (sx, py) in enumerate(TORCH_POS):
    x = sx * (PX - 0.46); z = 3.3
    B = Builder()
    B.box((x - sx * 0.18, py, z - 0.25), (0.18, 0.04, 0.04))                    # bracket arm
    B.box((x - sx * 0.02, py, z - 0.48), (0.03, 0.03, 0.26))                    # strap down the wall
    o = B.obj(f'torch bracket {k}', M_iron, bevel=0.01)
    # holder: tapered cup
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.05, radius2=0.12, depth=0.32)
    me = bpy.data.meshes.new(f'torch cup {k}'); bm.to_mesh(me); bm.free()
    cup = bpy.data.objects.new(f'torch cup {k}', me); coll.objects.link(cup); me.materials.append(M_iron)
    cx = x - sx * 0.36; cup.location = (cx, py, z - 0.14)
    for p in me.polygons:
        p.use_smooth = True
    # flame: stretched sphere, noise-animated emission
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
    for v in bm.verts:
        t = (v.co.z + 1) / 2
        v.co.x *= 0.1 * (1 - 0.8 * t ** 1.3); v.co.y *= 0.1 * (1 - 0.8 * t ** 1.3); v.co.z = 0.32 * v.co.z + 0.1
    me = bpy.data.meshes.new(f'flame {k}'); bm.to_mesh(me); bm.free()
    fl = bpy.data.objects.new(f'flame {k}', me); coll.objects.link(fl); me.materials.append(M_flame)
    fl.location = (cx, py, z + 0.05); fl.scale = (1, 1, 1.25)
    fl.visible_shadow = False
    for p in me.polygons:
        p.use_smooth = True
    li = bpy.data.lights.new(f'torch light {k}', 'POINT'); li.energy = TORCH_E; li.color = (1.0, 0.52, 0.18)
    li.shadow_soft_size = 0.18
    lo = bpy.data.objects.new(f'torch light {k}', li); coll.objects.link(lo); lo.location = (cx - sx * 0.12, py, z + 0.35)
    torches.append((fl, lo, k))

# flicker: keyed per frame over the whole range (a few overlapping sines per torch)
for fl, lo, k in torches:
    ph = rng.uniform(0, 100, 4)
    for f in range(-30, 400):
        t = f / 24.0
        fk = 1 + 0.16 * math.sin(7.3 * t + ph[0]) + 0.1 * math.sin(13.1 * t + ph[1]) + 0.07 * math.sin(23.7 * t + ph[2])
        lo.data.energy = TORCH_E * fk; lo.data.keyframe_insert('energy', frame=f)
        fl.scale = (1, 1, 1.25 * (0.9 + 0.12 * fk)); fl.keyframe_insert('scale', frame=f)
# animated flame noise (shared material): drive the 4D noise W by frame
nz = M_flame.node_tree.nodes['flame noise']
for f in (-30, 400):
    nz.inputs['W'].default_value = f * 0.09; nz.inputs['W'].keyframe_insert('default_value', frame=f)
for fc in M_flame.node_tree.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'

# ---------------------------------------------------------------- dressing: chains, rubble, bones
B = Builder()
for _ in range(60):          # rubble along the wall feet
    sx = rng.choice([-1, 1]); yy = rng.uniform(Y0 + 1, Y1 - 1)
    s = rng.uniform(0.06, 0.22)
    B.box((sx * (WX - rng.uniform(0.15, 0.9)), yy, s * 0.7), (s, s * rng.uniform(0.6, 1.2), s * 0.7), rot=rng.uniform(0, 3))
o = B.obj('rubble', M_wall, bevel=0.04)
o.modifiers['bevel'].segments = 1


def chain(name, p0, p1, sag, links=26):
    p0, p1 = Vector(p0), Vector(p1)
    verts = []; faces = []
    for i in range(links):
        t = (i + 0.5) / links
        p = p0.lerp(p1, t); p.z -= sag * 4 * t * (1 - t)
        d = ((p1 - p0) + Vector((0, 0, -sag * 4 * (1 - 2 * t)))).normalized()
        side = d.cross(Vector((0, 0, 1))).normalized() if abs(d.z) < 0.99 else Vector((1, 0, 0))
        if i % 2:
            side = d.cross(side).normalized()
        n0 = len(verts); M, Nm = 12, 6
        for a in range(M):
            th = 2 * math.pi * a / M
            c = p + d * (0.085 * math.cos(th)) + side * (0.05 * math.sin(th))
            rad = (d * math.cos(th) * 0.085 + side * math.sin(th) * 0.05).normalized()
            nrm = d.cross(side).normalized()
            for b in range(Nm):
                ph = 2 * math.pi * b / Nm
                verts.append(c + rad * 0.018 * math.cos(ph) + nrm * 0.018 * math.sin(ph))
        for a in range(M):
            for b in range(Nm):
                i0 = n0 + a * Nm + b; i1 = n0 + ((a + 1) % M) * Nm + b
                faces.append((i0, i1, n0 + ((a + 1) % M) * Nm + (b + 1) % Nm, n0 + a * Nm + (b + 1) % Nm))
    return mesh_obj(name, verts, faces, M_iron, smooth=True)


chain('chain 0', (-WX + 0.3, -3.5, 4.6), (-WX + 0.3, -1.0, 4.4), 1.0)
chain('chain 1', (WX - 0.3, 2.5, 4.7), (WX - 0.3, 4.6, 4.5), 0.9)
chain('chain 2', (-WX + 0.3, 3.0, 4.8), (-WX + 0.35, 4.8, 4.0), 1.2)

# skulls: simple sculpted lumps near the walls (sphere + jaw box), bone colour
M_bone = simple_mat('dk bone', (0.55, 0.47, 0.33), 0.55)
for k, (x, yy, rz) in enumerate([(-4.6, -2.6, 0.6), (-4.9, -1.9, 2.0), (4.7, 3.2, -0.6), (4.4, -4.0, 1.4), (-4.2, 4.6, 3.0)]):
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=10, radius=0.17)
    for v in bm.verts:
        v.co.y *= 1.18
        if v.co.y < -0.08 and v.co.z < 0:
            v.co.z *= 0.6
    me = bpy.data.meshes.new(f'skull {k}'); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(f'skull {k}', me); coll.objects.link(o); me.materials.append(M_bone)
    o.location = (x, yy, 0.13); o.rotation_euler = (0.2, 0.1, rz)
    for p in me.polygons:
        p.use_smooth = True

# ---------------------------------------------------------------- haze, world, lights on the creature
beam = bpy.data.materials.new('light beam'); nt, out = nodes(beam); N = nt.nodes; Lk = nt.links
tcb = N.new('ShaderNodeTexCoord'); sxyz = N.new('ShaderNodeSeparateXYZ'); Lk.new(tcb.outputs['Generated'], sxyz.inputs[0])
lw = N.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.6
inv = N.new('ShaderNodeMath'); inv.operation = 'SUBTRACT'; inv.inputs[0].default_value = 1.0
Lk.new(lw.outputs['Facing'], inv.inputs[1])
fall = N.new('ShaderNodeMath'); fall.operation = 'POWER'; fall.inputs[1].default_value = 1.2
alongm = N.new('ShaderNodeMath'); alongm.operation = 'SUBTRACT'; alongm.inputs[0].default_value = 1.0
Lk.new(sxyz.outputs['Y'], fall.inputs[0])          # generated Y: 0 at the floor end, 1 at the archway
nzb = N.new('ShaderNodeTexNoise'); nzb.inputs['Scale'].default_value = 2.5; Lk.new(tcb.outputs['Object'], nzb.inputs['Vector'])
m1 = N.new('ShaderNodeMath'); m1.operation = 'MULTIPLY'; Lk.new(inv.outputs[0], m1.inputs[0]); Lk.new(fall.outputs[0], m1.inputs[1])
m2 = N.new('ShaderNodeMath'); m2.operation = 'MULTIPLY'; Lk.new(m1.outputs[0], m2.inputs[0]); Lk.new(nzb.outputs['Fac'], m2.inputs[1])
m3 = N.new('ShaderNodeMath'); m3.operation = 'MULTIPLY'; m3.inputs[1].default_value = 1.6; Lk.new(m2.outputs[0], m3.inputs[0])
emb = N.new('ShaderNodeEmission'); emb.inputs['Color'].default_value = (0.6, 0.78, 1.0, 1)
Lk.new(m3.outputs[0], emb.inputs['Strength'])
trb = N.new('ShaderNodeBsdfTransparent'); add = N.new('ShaderNodeAddShader')
Lk.new(trb.outputs[0], add.inputs[0]); Lk.new(emb.outputs[0], add.inputs[1]); Lk.new(add.outputs[0], out.inputs['Surface'])
beam.blend_method = 'BLEND'
# frustum from the archway down and forward onto the floor around him
ring0 = [(DOOR_W * 0.9 * math.cos(a_), Y1 + 0.2, 3.0 + 2.6 * math.sin(a_)) for a_ in np.linspace(0, 2 * math.pi, 25)[:-1]]
ring1 = [(3.0 * math.cos(a_), 0.6, 0.1 + 1.6 * math.sin(a_)) for a_ in np.linspace(0, 2 * math.pi, 25)[:-1]]
Vb = ring0 + ring1; Fb = [(i, (i + 1) % 24, 24 + (i + 1) % 24, 24 + i) for i in range(24)]
bo = mesh_obj('light beam', Vb, Fb, beam, smooth=True)
bo.visible_shadow = False; bo.visible_diffuse = False; bo.visible_glossy = False
S.view_layers[0].use_pass_mist = True
S.world.mist_settings.start = 3.0; S.world.mist_settings.depth = 28.0; S.world.mist_settings.falloff = 'QUADRATIC'

W = S.world; W.use_nodes = True; bg = W.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.012, 0.009, 0.007, 1); bg.inputs['Strength'].default_value = 1.0

rb = bpy.data.objects.get('rim blue')
if rb:      # cool rim from the archway side, behind him
    rb.location = (2.2, 6.5, 6.0); rb.data.energy = 1500; rb.data.spread = math.radians(35); rb.data.size = 1.5; rb.data.color = (0.62, 0.78, 1.0)
    rb.rotation_euler = (Vector((0, 0, 2.6)) - rb.location).to_track_quat('-Z', 'Y').to_euler()
# a soft warm bounce from the floor/torches on the camera side so the shadow side reads (DK2 shadows are deep but not black)
fb = bpy.data.lights.new('warm bounce', 'AREA'); fb.energy = 90; fb.size = 4; fb.color = (1.0, 0.62, 0.35)
fo = bpy.data.objects.new('warm bounce', fb); coll.objects.link(fo); fo.location = (-1.5, -6.5, 1.2)
fo.rotation_euler = (Vector((0, 0, 2.4)) - fo.location).to_track_quat('-Z', 'Y').to_euler()

# ---------------------------------------------------------------- mace impact: sparks and a dust burst
IMP = Vector((-0.7, -2.0, 0.04)); F_HIT = 219
bm = bmesh.new(); bmesh.ops.create_circle(bm, cap_ends=True, segments=12, radius=0.22)
me = bpy.data.meshes.new('impact emitter'); bm.to_mesh(me); bm.free()
emo = bpy.data.objects.new('impact emitter', me); coll.objects.link(emo); emo.location = IMP
spark_m = simple_mat('spark', (0, 0, 0), 1.0, emit=(1.0, 0.55, 0.15), estr=60)
bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.013)
me = bpy.data.meshes.new('spark'); bm.to_mesh(me); bm.free(); me.materials.append(spark_m)
spk = bpy.data.objects.new('spark', me); coll.objects.link(spk); spk.location = (0, 0, -50)
dust_m = bpy.data.materials.new('dust'); nt, out = nodes(dust_m)
pv = nt.nodes.new('ShaderNodeVolumePrincipled'); pv.inputs['Color'].default_value = (0.42, 0.33, 0.24, 1)
pv.inputs['Density'].default_value = 0.0
nzd = nt.nodes.new('ShaderNodeTexNoise'); nzd.inputs['Scale'].default_value = 3.0; nzd.inputs['Detail'].default_value = 3
grd = nt.nodes.new('ShaderNodeTexGradient'); grd.gradient_type = 'SPHERICAL'
tcd = nt.nodes.new('ShaderNodeTexCoord'); nt.links.new(tcd.outputs['Object'], grd.inputs['Vector']); nt.links.new(tcd.outputs['Object'], nzd.inputs['Vector'])
md = nt.nodes.new('ShaderNodeMath'); md.operation = 'MULTIPLY'
nt.links.new(grd.outputs['Fac'], md.inputs[0]); nt.links.new(nzd.outputs['Fac'], md.inputs[1])
dens = nt.nodes.new('ShaderNodeMath'); dens.operation = 'MULTIPLY'; dens.name = 'dust density'
nt.links.new(md.outputs[0], dens.inputs[0]); nt.links.new(dens.outputs[0], pv.inputs['Density'])
nt.links.new(pv.outputs[0], out.inputs['Volume'])
for f_, dv in ((F_HIT - 1, 0.0), (F_HIT, 0.0), (F_HIT + 3, 9.0), (F_HIT + 20, 4.0), (F_HIT + 60, 0.0)):
    dens.inputs[1].default_value = dv; dens.inputs[1].keyframe_insert('default_value', frame=f_)
bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
me = bpy.data.meshes.new('dust puff'); bm.to_mesh(me); bm.free(); me.materials.append(dust_m)
puff = bpy.data.objects.new('dust puff', me); coll.objects.link(puff); puff.location = (0, 0, -60)
for f_, sc in ((F_HIT, 0.12), (F_HIT + 6, 0.32), (F_HIT + 30, 0.6), (F_HIT + 60, 0.8)):
    puff.scale = (sc, sc, sc * 0.8); puff.keyframe_insert('scale', frame=f_)


def burst(name, count, life, normal, rnd, inst, size, size_rnd, grav, damp, f_end):
    ps = emo.modifiers.new(name, 'PARTICLE_SYSTEM').particle_system
    st = ps.settings; st.count = count; st.frame_start = F_HIT; st.frame_end = f_end
    st.lifetime = life; st.lifetime_random = 0.5; st.emit_from = 'FACE'
    st.normal_factor = normal; st.factor_random = rnd
    st.render_type = 'OBJECT'; st.instance_object = inst; st.particle_size = size; st.size_random = size_rnd
    st.effector_weights.gravity = grav; st.damping = damp
    ps.seed = 3
    return ps


burst('sparks', 140, 16, 3.2, 3.0, spk, 1.0, 0.6, 1.0, 0.0, F_HIT + 1.5)
burst('dust', 26, 70, 0.9, 1.6, puff, 1.0, 0.5, 0.0, 0.12, F_HIT + 2)
emo.show_instancer_for_render = False

# old camera animation is replaced per shot later
bpy.ops.wm.save_as_mainfile(filepath=A.save, compress=True)
print('SET_SAVED', A.save, len(bpy.data.objects))
