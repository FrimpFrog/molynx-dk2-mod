"""Rig Molynx for animation: builds the cinematic scene (blender_scene.py, grin head), switches the
rock to bump-only shading (fast enough for video), decimates the sculpted body, adds an armature
(spine, neck, head, jaw, brow, arms, fingers, IK legs), skins everything and saves rig.blend.

blender -b --factory-startup -P anim_rig.py
"""
import math
import sys

import bpy
import mathutils
import numpy as np

TOOLS = '/home/claude/molynx_cinematic/tools/'
OUT = '/home/claude/molynx_blender/anim/rig.blend'
sys.argv = [sys.argv[0], '--', '--expr', 'grin', '--shot', 'hero', '--nodisp', '--buildonly', '--samples', '12', '--res', '960x540']
try:
    exec(compile(open(TOOLS + 'blender_scene.py').read(), 'blender_scene.py', 'exec'))
except SystemExit as e:
    assert str(e) == '__BUILD_ONLY__', e
S = bpy.context.scene
P = '/home/claude/molynx_blender/parts/'
MIRROR = np.array([-1.0, 1.0, 1.0])
O_HEAD = np.array([0.0275, 3.60, 0.45])


def Z(p):
    p = np.asarray(p, float)
    return mathutils.Vector((p[0], -p[2], p[1]))


def nrm(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


# ---------------------------------------------------------------- video-speed shading
for m in bpy.data.materials:
    m.displacement_method = 'BUMP'
OB = bpy.data.objects
body = OB['Molynx body']
dec = body.modifiers.new('decimate', 'DECIMATE'); dec.ratio = 0.3
bpy.context.view_layer.objects.active = body
bpy.ops.object.modifier_apply(modifier='decimate')
print('body faces after decimate', len(body.data.polygons))

# ---------------------------------------------------------------- skeleton (parts coords: y up, +z forward)
aL = nrm([0.03, -0.95, 0.31]); WL = np.array([-0.872, 1.775, 0.02])
GC = np.array([0.249, 2.991, 0.946]); GA = nrm([-0.189, -0.391, 0.901]); HC = GC + GA * -0.085
ELB_R = np.array([0.9, 2.35, 0.3]); uR = ELB_R - HC; uR = nrm(uR - np.dot(uR, GA) * GA); WR = HC + uR * 0.2
FEET = [(np.array([-0.595, -0.28]), nrm([-0.362, 0.932])), (np.array([0.606, 0.167]), nrm([0.275, 0.961]))]


def toe(c, f):
    return np.array([c[0] + f[0] * 0.38, 0.06, c[1] + f[1] * 0.38])


hinge = (O_HEAD + np.array([0, -0.06, -0.03])) * MIRROR
chin = (O_HEAD + np.array([0, -0.24, 0.2])) * MIRROR
BONES = [  # name, head, tail, parent, deform
    ('root', (0, 0, 0), (0, 0.5, 0), None, False),
    ('pelvis', (0, 1.72, -0.2), (0, 1.95, -0.2), 'root', True),
    ('spine', (0, 1.95, -0.2), (0, 2.72, -0.15), 'pelvis', True),
    ('chest', (0, 2.72, -0.15), (0, 3.28, -0.05), 'spine', True),
    ('neck', (0, 3.28, -0.05), (-0.02, 3.45, 0.18), 'chest', True),
    ('head', (-0.02, 3.45, 0.18), (-0.02, 3.85, 0.35), 'neck', True),
    ('jaw', tuple(hinge), tuple(chin), 'head', True),
    ('brow', tuple((O_HEAD + np.array([0, 0.09, 0.12])) * MIRROR), tuple((O_HEAD + np.array([0, 0.09, 0.3])) * MIRROR), 'head', True),
    ('clav.L', (-0.15, 3.05, -0.2), (-0.6, 3.05, -0.3), 'chest', True),
    ('upper.L', (-0.6, 3.05, -0.3), (-0.85, 2.13, -0.27), 'clav.L', True),
    ('fore.L', (-0.85, 2.13, -0.27), tuple(WL), 'upper.L', True),
    ('hand.L', tuple(WL), tuple(WL + aL * 0.2), 'fore.L', True),
    ('fingers.L', tuple(WL + aL * 0.22), tuple(WL + aL * 0.44), 'hand.L', True),
    ('clav.R', (0.15, 3.05, -0.15), (0.58, 3.0, -0.12), 'chest', True),
    ('upper.R', (0.58, 3.0, -0.12), tuple(ELB_R), 'clav.R', True),
    ('fore.R', tuple(ELB_R), tuple(WR), 'upper.R', True),
    ('hand.R', tuple(WR), tuple(HC), 'fore.R', True),
    ('thigh.L', (-0.3, 1.62, -0.25), (-0.41, 0.95, -0.29), 'pelvis', True),
    ('shin.L', (-0.41, 0.95, -0.29), (-0.41, 0.27, -0.54), 'thigh.L', True),
    ('foot.L', (-0.41, 0.27, -0.54), tuple(toe(*FEET[0])), 'shin.L', True),
    ('thigh.R', (0.3, 1.62, -0.1), (0.38, 0.96, 0.05), 'pelvis', True),
    ('shin.R', (0.38, 0.96, 0.05), (0.44, 0.27, -0.13), 'thigh.R', True),
    ('foot.R', (0.44, 0.27, -0.13), tuple(toe(*FEET[1])), 'shin.R', True),
    ('ik.L', (-0.41, 0.27, -0.54), (-0.41, 0.27, -0.34), 'root', False),
    ('ik.R', (0.44, 0.27, -0.13), (0.44, 0.27, 0.07), 'root', False),
    ('pole.L', (-0.41, 0.95, 0.9), (-0.41, 0.95, 1.1), 'root', False),
    ('pole.R', (0.38, 0.96, 1.2), (0.38, 0.96, 1.4), 'root', False),
    ('pole.armR', (1.45, 2.2, -0.45), (1.45, 2.4, -0.45), 'chest', False),
    ('pole.armL', (-1.35, 2.2, -0.75), (-1.35, 2.4, -0.75), 'chest', False),
    ('ikrest.L', tuple(WL), tuple(WL + aL * 0.2), 'root', False),
]
arm_d = bpy.data.armatures.new('Molynx rig'); arm = bpy.data.objects.new('Molynx rig', arm_d); S.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
for name, h, t, par, dfm in BONES:
    eb = arm_d.edit_bones.new(name); eb.head = Z(h); eb.tail = Z(t); eb.use_deform = dfm
    if par:
        eb.parent = arm_d.edit_bones[par]
    eb.roll = 0
# mace control: a copy of hand.R (the mace hand) parented to root; the arm reaches for it with IK
eb = arm_d.edit_bones.new('mace.ctl'); hr = arm_d.edit_bones['hand.R']
eb.head = hr.head.copy(); eb.tail = hr.tail.copy(); eb.roll = hr.roll; eb.parent = arm_d.edit_bones['root']; eb.use_deform = False
# left (claw) hand grip on the shaft below the fist: the hand frame (down the hand, palm normal, thumb side)
# is mapped onto (toward the shaft from the wrist side, the fist's knuckle side, toward the mace head)
GRIP_T = 0.27
_uR = nrm(ELB_R - HC); _uR = nrm(_uR - np.dot(_uR, GA) * GA); _vR = np.cross(GA, _uR)
nL0 = nrm(np.array([1.0, 0, 0]) - aL[0] * aL); wL0 = np.cross(aL, nL0)
a2, n2, w2 = -_uR, np.cross(-GA, -_uR), -GA
SH = HC + GA * GRIP_T
WL2 = SH - a2 * 0.24 - n2 * 0.115
B0 = mathutils.Matrix([list(Z(aL)), list(Z(nL0)), list(Z(wL0))]).transposed()
B1 = mathutils.Matrix([list(Z(a2)), list(Z(n2)), list(Z(w2))]).transposed()
Rg = B1 @ B0.transposed()
hl = arm_d.edit_bones['hand.L']
D = (Rg @ hl.matrix.to_3x3()).to_4x4(); D.translation = Z(WL2)
eb = arm_d.edit_bones.new('grip.L'); eb.length = hl.length; eb.matrix = D; eb.length = hl.length
eb.parent = arm_d.edit_bones['mace.ctl']; eb.use_deform = False
bpy.ops.object.mode_set(mode='OBJECT')

# IK legs: feet follow the ik bones, knees aim at the poles; feet keep the ik bone's orientation
for s in ('L', 'R'):
    pb = arm.pose.bones['shin.' + s]
    c = pb.constraints.new('IK'); c.target = arm; c.subtarget = 'ik.' + s; c.pole_target = arm; c.pole_subtarget = 'pole.' + s; c.chain_count = 2
    fc = arm.pose.bones['foot.' + s].constraints.new('COPY_ROTATION'); fc.target = arm; fc.subtarget = 'ik.' + s; fc.mix_mode = 'REPLACE'
    fc.target_space = fc.owner_space = 'WORLD'
    # the ik bone points forward while the foot bone points at the toe: copy the *change* only
    fc.mix_mode = 'BEFORE'; fc.target_space = fc.owner_space = 'LOCAL'


def chain_error(s):
    bpy.context.view_layer.update()
    e = 0
    for b in ('thigh.', 'shin.'):
        pb = arm.pose.bones[b + s]
        e += (pb.matrix.to_3x3() - pb.bone.matrix_local.to_3x3()).median_scale if False else sum(
            (pb.matrix.col[i].xyz - pb.bone.matrix_local.col[i].xyz).length for i in range(3))
    return e


for s in ('L', 'R'):                       # pole angle that leaves the rest pose untouched
    c = arm.pose.bones['shin.' + s].constraints['IK']
    best = min(range(-180, 181, 2), key=lambda a: (setattr(c, 'pole_angle', math.radians(a)), chain_error(s))[1])
    best = min([best + d * 0.25 for d in range(-8, 9)], key=lambda a: (setattr(c, 'pole_angle', math.radians(a)), chain_error(s))[1])
    c.pole_angle = math.radians(best); print('pole', s, best, 'err', chain_error(s))

# arms: the mace arm reaches for mace.ctl (always), the claw arm for grip.L (influence animated)
def arm_chain_error(s):
    bpy.context.view_layer.update()
    return sum((arm.pose.bones[b + s].matrix.col[i].xyz - arm.pose.bones[b + s].bone.matrix_local.col[i].xyz).length
               for b in ('upper.', 'fore.') for i in range(3))


cR = arm.pose.bones['fore.R'].constraints.new('IK'); cR.target = arm; cR.subtarget = 'mace.ctl'
cR.pole_target = arm; cR.pole_subtarget = 'pole.armR'; cR.chain_count = 2
cL = arm.pose.bones['fore.L'].constraints.new('IK'); cL.target = arm; cL.subtarget = 'ikrest.L'
cL.pole_target = arm; cL.pole_subtarget = 'pole.armL'; cL.chain_count = 2
for s, c in (('R', cR), ('L', cL)):
    best = min(range(-180, 181, 2), key=lambda a: (setattr(c, 'pole_angle', math.radians(a)), arm_chain_error(s))[1])
    best = min([best + d * 0.25 for d in range(-8, 9)], key=lambda a: (setattr(c, 'pole_angle', math.radians(a)), arm_chain_error(s))[1])
    c.pole_angle = math.radians(best); print('arm pole', s, best, 'err', arm_chain_error(s))
cL.subtarget = 'grip.L'; cL.influence = 0.0
for hb_, tg, infl in (('hand.R', 'mace.ctl', 1.0), ('hand.L', 'grip.L', 0.0)):
    c = arm.pose.bones[hb_].constraints.new('COPY_ROTATION'); c.name = 'grip'; c.target = arm; c.subtarget = tg
    c.target_space = c.owner_space = 'WORLD'; c.influence = infl
bpy.context.view_layer.update()
print('rest check R', arm_chain_error('R'), 'L', arm_chain_error('L'))

# ---------------------------------------------------------------- skinning
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True); arm.select_set(True); bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
print('body groups', len(body.vertex_groups))


def groups_of(ob):
    return {g.name: g for g in ob.vertex_groups}


def rig_object(ob, weights):
    """weights: dict bone -> array per vertex"""
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    for bn, w in weights.items():
        g = ob.vertex_groups.new(name=bn)
        w = np.asarray(w, float)
        for val in np.unique(np.round(w[w > 1e-3], 3)):
            idx = np.where(np.abs(np.round(w, 3) - val) < 1e-9)[0].tolist()
            g.add(idx, float(val), 'REPLACE')
    mod = ob.modifiers.new('rig', 'ARMATURE'); mod.object = arm
    ob.parent = arm


def verts(ob):
    a = np.empty(len(ob.data.vertices) * 3, np.float32); ob.data.vertices.foreach_get('co', a); return a.reshape(-1, 3)


def to_parts(Vb):
    return np.stack([Vb[:, 0], Vb[:, 2], -Vb[:, 1]], 1)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


# body skin weights + a lookup of the nearest skin vertex (used to glue head, spurs and claws)
Vbody = verts(body); kd = mathutils.kdtree.KDTree(len(Vbody))
for i, v in enumerate(Vbody):
    kd.insert(v, i)
kd.balance()
names = [g.name for g in body.vertex_groups]
Wb = np.zeros((len(Vbody), len(names)))
for v in body.data.vertices:
    for g in v.groups:
        Wb[v.index, g.group] = g.weight
Wb /= np.maximum(Wb.sum(1, keepdims=True), 1e-9)
Nbody = np.empty(len(Vbody) * 3, np.float32); body.data.vertices.foreach_get('normal', Nbody); Nbody = Nbody.reshape(-1, 3)


def nearest(Vo):
    idx = np.empty(len(Vo), int); dist = np.empty(len(Vo))
    for j, v in enumerate(Vo):
        _, i, dd = kd.find(mathutils.Vector(v)); idx[j] = i; dist[j] = dd
    return idx, dist


# head: the neck takes the skin's own weights (no seam), jaw below the mouth line, brow over the eyes
head = OB['Molynx head']
Vh = verts(head); L = to_parts(Vh) * MIRROR - O_HEAD
w_neck = sstep(-0.22, -0.34, L[:, 1])
w_jaw = sstep(-0.125, -0.158, L[:, 1]) * sstep(-0.03, 0.06, L[:, 2]) * (1 - w_neck)
w_brow = np.exp(-((L[:, 0] / 0.16) ** 2 + ((L[:, 1] - 0.095) / 0.045) ** 2)) * sstep(0.1, 0.19, L[:, 2])
w_brow *= (1 - w_jaw - w_neck)
hidx, _ = nearest(Vh)
AXIAL = [k for k, n in enumerate(names) if n in ('pelvis', 'spine', 'chest', 'neck', 'head')]   # never the arms
Wn = np.zeros_like(Wb[hidx]); Wn[:, AXIAL] = Wb[hidx][:, AXIAL]
ssum = Wn.sum(1, keepdims=True); Wn = np.where(ssum > 1e-6, Wn / np.maximum(ssum, 1e-9), 0)
Wn[ssum[:, 0] <= 1e-6, names.index('neck')] = 1.0
Wh = {n: Wn[:, k] * w_neck for k, n in enumerate(names)}
for n, w in (('jaw', w_jaw), ('brow', w_brow * 0.9), ('head', 1 - w_neck - w_jaw - w_brow * 0.9)):
    Wh[n] = Wh.get(n, 0) + w
rig_object(head, {n: w for n, w in Wh.items() if np.max(w) > 1e-3})


def components(ob):
    from collections import defaultdict
    n = len(ob.data.vertices); par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for e in ob.data.edges:
        a, b = find(e.vertices[0]), find(e.vertices[1])
        if a != b:
            par[a] = b
    lab = np.array([find(i) for i in range(n)])
    return lab


# teeth: lower ones ride the jaw
teeth = OB['teeth']
Lt = to_parts(verts(teeth)) * MIRROR - O_HEAD; lab = components(teeth)
lower = np.zeros(len(Lt), bool)
for c in np.unique(lab):
    m = lab == c
    if Lt[m, 1].min() < -0.15:
        lower[m] = True
rig_object(teeth, {'jaw': lower * 1.0, 'head': (~lower) * 1.0})
for nm in ('horns', 'eye 0', 'eye 1'):
    rig_object(OB[nm], {'head': np.ones(len(OB[nm].data.vertices))})
for nm in [o.name for o in OB if o.name.startswith('mace ')]:
    rig_object(OB[nm], {'hand.R': np.ones(len(OB[nm].data.vertices))})

# stone spurs: sink the open base rims into the skin, then glue: base vertices take the weights of the
# skin under them, the rest of the spur rides the average of its base (stays a rigid spike)
sp_ob = OB['stone spurs']; me = sp_ob.data
from collections import Counter
ec = Counter(tuple(sorted(e.vertices)) for e in me.edges)
edge_faces = Counter()
for p in me.polygons:
    vs = list(p.vertices)
    for a, b in zip(vs, vs[1:] + vs[:1]):
        edge_faces[tuple(sorted((a, b)))] += 1
rim = np.zeros(len(me.vertices), bool)
for e, n in edge_faces.items():
    if n == 1:
        rim[list(e)] = True
Vs = verts(sp_ob); sidx, sdist = nearest(Vs)
Vs[rim] -= Nbody[sidx[rim]] * 0.05
me.vertices.foreach_set('co', Vs.astype(np.float32).ravel()); me.update()
Vs = verts(sp_ob); sidx, sdist = nearest(Vs); lab = components(sp_ob)
W = np.zeros((len(Vs), len(names)))
for c in np.unique(lab):
    m = np.where(lab == c)[0]
    base = m[sdist[m] < np.percentile(sdist[m], 40)]
    wc = Wb[sidx[base]].mean(0)
    alpha = np.clip((sdist[m] - 0.015) / 0.06, 0, 1)[:, None]
    W[m] = (1 - alpha) * Wb[sidx[m]] + alpha * wc
rig_object(sp_ob, {n: W[:, k] for k, n in enumerate(names) if W[:, k].max() > 1e-3})
print('spurs: rim verts sunk', int(rim.sum()))

# claws: each one rigid on the strongest bone of the skin it sits on
for nm in ('claws',):
    ob = OB[nm]; Vo = verts(ob); lab = components(ob); W = np.zeros((len(Vo), len(names)))
    for c in np.unique(lab):
        m = lab == c
        _, i, _ = kd.find(mathutils.Vector(Vo[m].mean(0)))
        w = Wb[i] / max(Wb[i].sum(), 1e-9)
        if nm == 'claws':                       # claws are rigid: their single strongest bone
            w = (w == w.max()) * 1.0
        W[m] = w
    rig_object(ob, {n: W[:, k] for k, n in enumerate(names) if W[:, k].max() > 0})

# ---------------------------------------------------------------- gear: armour rides its bone, the belt the skin,
# the skirt is cloth pinned along its top (collides with hidden hip/thigh shapes)
for o in [o for o in OB if 'bone' in o.keys()]:
    bn = o['bone']; Vo = verts(o)
    if bn and bn != 'cloth':
        rig_object(o, {bn: np.ones(len(Vo))})
        continue
    idx, _ = nearest(Vo)
    Wg = Wb[idx]
    if bn == 'cloth':                                   # the skirt hangs from the hips: pelvis + thighs only
        keepb = [k for k, n in enumerate(names) if n in ('pelvis', 'spine', 'thigh.L', 'thigh.R')]
        W2 = np.zeros_like(Wg); W2[:, keepb] = Wg[:, keepb]; ssum = W2.sum(1, keepdims=True)
        W2 = np.where(ssum > 1e-6, W2 / np.maximum(ssum, 1e-9), 0); W2[ssum[:, 0] <= 1e-6, names.index('pelvis')] = 1
        N_, NT_ = o['grid']; row = np.tile(np.arange(NT_), N_)
        top = np.clip(1 - row / 6.0, 0, 1)                   # upper rows follow the pelvis, lower rows are free cloth
        Wg = W2 * top[:, None]; Wg[:, names.index('pelvis')] += 1 - top
    rig_object(o, {n: Wg[:, k] for k, n in enumerate(names) if Wg[:, k].max() > 1e-3})
    if bn == 'cloth':
        pin = o.vertex_groups.new(name='pin')
        for r in range(NT_):
            wv = 1.0 if r <= 1 else (0.5 if r == 2 else 0.0)
            if wv > 0:
                pin.add([i * NT_ + r for i in range(N_)], wv, 'REPLACE')
        for m in list(o.modifiers):
            if m.type == 'SOLIDIFY':
                o.modifiers.remove(m)
        cm = o.modifiers.new('cloth', 'CLOTH'); cs = cm.settings
        cs.quality = 7; cs.mass = 0.25; cs.air_damping = 1.5
        cs.tension_stiffness = cs.compression_stiffness = 10; cs.shear_stiffness = 6; cs.bending_stiffness = 0.4
        cs.vertex_group_mass = 'pin'; cs.pin_stiffness = 2.0
        cm.collision_settings.distance_min = 0.012; cm.collision_settings.use_self_collision = False
        cm.collision_settings.collision_quality = 3
        sol = o.modifiers.new('thickness', 'SOLIDIFY'); sol.thickness = 0.008; sol.offset = 1.0
        SKIRT = o

# hidden collision shapes for the cloth (follow their bones)
import bmesh


def collider(name, bone, centre, radii, along=None):
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=1.0); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); S.collection.objects.link(ob)
    M = mathutils.Matrix.Identity(4)
    if along is not None:
        a = mathutils.Vector(along).normalized(); M = a.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    ob.matrix_world = mathutils.Matrix.Translation(Z(centre)) @ M @ mathutils.Matrix.Diagonal((*radii, 1))
    wm = ob.matrix_world.copy()
    ob.parent = arm; ob.parent_type = 'BONE'; ob.parent_bone = bone
    bb = arm.data.bones[bone]
    ob.matrix_parent_inverse = (arm.matrix_world @ bb.matrix_local @ mathutils.Matrix.Translation((0, bb.length, 0))).inverted()
    ob.matrix_basis = wm
    ob.modifiers.new('collision', 'COLLISION'); ob.collision.thickness_outer = 0.01; ob.collision.cloth_friction = 2
    ob.hide_render = True; ob.visible_camera = False
    return ob


collider('col pelvis', 'pelvis', (0, 1.88, -0.12), (0.43, 0.37, 0.24))
for s_, top, kn in (('L', (-0.3, 1.62, -0.25), (-0.41, 0.95, -0.29)), ('R', (0.3, 1.62, -0.1), (0.38, 0.96, 0.05))):
    c = (np.array(top) + np.array(kn)) / 2; ax = Z(np.array(kn) - np.array(top))
    collider('col thigh ' + s_, 'thigh.' + s_, c, (0.2, 0.2, 0.42), along=ax)

# eyes: a squint shape key (lids narrow the glowing slit)
for nm in ('eye 0', 'eye 1'):
    ob = OB[nm]; ob.shape_key_add(name='Basis'); sk = ob.shape_key_add(name='squint')
    V = verts(ob); c = V.mean(0); V2 = V.copy(); V2[:, 2] = c[2] + (V[:, 2] - c[2]) * 0.3
    sk.data.foreach_set('co', V2.astype(np.float32).ravel())

# eye-glow lights follow the head
hb = arm.data.bones['head']
for nm in ('eye glow 0', 'eye glow 1'):
    lo = OB[nm]; wm = lo.matrix_world.copy()
    lo.parent = arm; lo.parent_type = 'BONE'; lo.parent_bone = 'head'
    lo.matrix_parent_inverse = (arm.matrix_world @ hb.matrix_local @ mathutils.Matrix.Translation((0, hb.length, 0))).inverted()
    lo.matrix_basis = wm

bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
print('RIG_SAVED', OUT)
