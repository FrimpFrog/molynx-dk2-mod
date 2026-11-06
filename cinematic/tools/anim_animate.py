"""Molynx test animation (12 s, 24 fps) on rig.blend.

  0.0-2.0  idle breathing, eyes ignite
  2.0-5.4  looks to his left, then right
  5.4-6.0  scowl: brow drops, eyes narrow to slits
  6.0-7.5  roar: jaw drops, head back, eyes flare, chest heaves, claws spread
  7.6-10.9 mace: wind-up, step forward, overhead slam into the floor, recover, step back
  10.9-12  clenches his claws, settles, eyes pulse

blender -b rig.blend -P anim_animate.py -- --test 0,3,6.7,9.4 --res 480x270   (stills of poses)
blender -b rig.blend -P anim_animate.py -- --save anim.blend                     (bake + save for rendering)
"""
import argparse
import math
import sys

import bpy
import mathutils
from mathutils import Quaternion, Vector

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--test', default=''); ap.add_argument('--res', default='960x540'); ap.add_argument('--samples', type=int, default=12)
ap.add_argument('--save', default=''); ap.add_argument('--outdir', default='/home/claude/molynx_blender/anim/test/')
ap.add_argument('--nocam', action='store_true'); ap.add_argument('--zoom', default='')   # lens multiplier,target height,target x
A = ap.parse_args(argv)

S = bpy.context.scene
FPS = 24; DUR = 12.0; NF = int(DUR * FPS)
arm = bpy.data.objects['Molynx rig']
X, Y, Zax = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))


def sm(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


def env(t, a, b, c, d):
    """0 before a, eases to 1 by b, holds, eases back to 0 from c to d"""
    if t <= a or t >= d:
        return 0.0
    if t < b:
        return sm((t - a) / (b - a))
    if t <= c:
        return 1.0
    return 1.0 - sm((t - c) / (d - c))


def ramp(t, a, b):
    return sm((t - a) / (b - a))


# ---------------------------------------------------------------- the mace path (parts coords: y up, +z front,
# the mace hand on +x; his right side is -x). Each key: time, grip centre, direction from the grip to the ball.
def P2B(v):
    return Vector((v[0], -v[2], v[1]))


def nrm(v):
    v = Vector(v); return v.normalized()


GC = Vector((0.249, 2.991, 0.946)); GA = nrm((-0.189, -0.391, 0.901)); HC = GC - GA * 0.085
ELB_R = Vector((0.9, 2.35, 0.3)); SHO_R = Vector((0.58, 3.0, -0.12))
_u = ELB_R - HC; U0 = (_u - _u.dot(GA) * GA).normalized()
MACE = [  # time, grip centre, toward the ball
    (7.5, HC, -GA),
    (7.95, Vector((0.4, 3.15, 0.85)), nrm((0.2, 0.85, -0.5))),                 # raise, the claw hand grabs on
    (8.55, Vector((0.5, 3.4, 0.35)), nrm((-0.75, 0.25, -0.62))),             # wind-up: hands by his ear, mace behind his head
    (8.72, Vector((0.52, 3.45, 0.3)), nrm((-0.68, 0.1, -0.73))),               # coil
    (9.05, Vector((-0.2, 1.85, 0.95)), nrm((-0.5, -1.35, 1.05))),             # chop: into the floor, front-right
    (9.25, Vector((-0.22, 1.9, 0.95)), nrm((-0.5, -1.3, 1.05))),
    (9.75, Vector((-0.45, 2.15, 0.65)), nrm((-0.85, 0.05, 0.5))),             # sweep out to his right
    (10.45, Vector((0.32, 2.5, 0.95)), nrm((0.12, 1.0, -0.15))),             # recover: upright in both hands
    (10.9, Vector((0.32, 2.5, 0.95)), nrm((0.12, 1.0, -0.15))),
    (11.55, HC, -GA),                                                         # back on the shoulder
]
TWIST = [(7.5, 0), (7.95, 5), (8.55, 18), (8.72, 22), (9.05, -22), (9.25, -24), (9.75, -38), (10.45, -6), (10.9, -4), (11.55, 0)]
LEAN = [(7.5, 0), (8.55, -8), (8.72, -9), (9.05, 18), (9.25, 19), (9.75, 10), (10.45, 2), (11.55, 0)]
HEADZ = [(7.5, 0), (8.55, 8), (9.05, -12), (9.75, -14), (10.45, -2), (11.55, 0)]
HEADX = [(7.5, 0), (8.55, -6), (9.05, 12), (9.25, 12), (9.75, 4), (10.45, 0), (11.55, 0)]
DROP = [(7.5, 0), (8.55, 0.02), (9.05, -0.14), (9.25, -0.15), (9.75, -0.1), (10.45, -0.02), (11.55, 0)]
_aL = nrm((0.03, -0.95, 0.31)); _nL = (Vector((1.0, 0, 0)) - _aL.x * _aL).normalized()
WLAX = P2B(_aL.cross(_nL))                     # across the claw hand: fingers curl about this axis


def keyed(t, keys):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, a), (t1, b) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            return a + (b - a) * sm((t - t0) / (t1 - t0))
    return keys[-1][1]


def grip_l(t):
    return env(t, 7.7, 8.0, 10.9, 11.2)


def frame_of(c, h):
    """grip frame: toward the ball, the wrist side (toward his shoulder), and their cross product"""
    u = SHO_R - c; u = (u - u.dot(h) * h).normalized()
    return h, u, h.cross(u)


def basis(h, u, w):
    return mathutils.Matrix([list(P2B(h)), list(P2B(u)), list(P2B(w))]).transposed()


Q_REST = basis(-GA, U0, (-GA).cross(U0)).to_quaternion()


def key_q(i):
    t0, c, h = MACE[i]
    if (c - HC).length < 1e-9 and (h + GA).length < 1e-9:
        return Q_REST
    return basis(*frame_of(c, h)).to_quaternion()


def mace_key(t):
    """(grip centre, orientation) eased between keys; the grip centre follows a Catmull-Rom curve"""
    if t <= MACE[0][0] or t >= MACE[-1][0]:
        return HC, Q_REST
    for i in range(len(MACE) - 1):
        t0, c0, _ = MACE[i]; t1, c1, _ = MACE[i + 1]
        if t <= t1:
            u = sm((t - t0) / (t1 - t0))
            cp = MACE[max(i - 1, 0)][1]; cn = MACE[min(i + 2, len(MACE) - 1)][1]
            c = 0.5 * ((2 * c0) + (-cp + c1) * u + (2 * cp - 5 * c0 + 4 * c1 - cn) * u * u + (-cp + 3 * c0 - 3 * c1 + cn) * u ** 3)
            q0, q1 = key_q(i), key_q(i + 1)
            if q0.dot(q1) < 0:
                q1 = -q1
            return c, q0.slerp(q1, u)
    return HC, Q_REST


def mace_q(t):
    return mace_key(t)[1] @ Q_REST.inverted()


def mace_t(t):
    c, q = mace_key(t); dq = q @ Q_REST.inverted()
    wr = arm.pose.bones['mace.ctl'].bone.head_local
    return P2B(c) - wr - dq @ (P2B(HC) - wr)


def pose(t):
    """bone -> (list of (world axis, degrees), world translation)"""
    R = {}; Lc = {}

    def rot(b, axis, deg):
        R.setdefault(b, []).append((axis, deg))

    def mv(b, v):
        Lc[b] = Lc.get(b, Vector()) + Vector(v)

    br = math.sin(2 * math.pi * t / 2.7)                       # breathing
    rot('chest', X, -1.6 * br); rot('spine', X, -0.6 * br); rot('clav.L', Y, 1.2 * br); rot('clav.R', Y, -1.2 * br)
    rot('head', X, 0.8 * br)
    # look around
    look = 30 * env(t, 2.1, 2.8, 3.3, 4.0) - 32 * env(t, 3.9, 4.6, 5.0, 5.6)
    rot('neck', Zax, 0.4 * look); rot('head', Zax, 0.6 * look); rot('chest', Zax, 0.15 * look)
    rot('head', Y, -0.12 * look)                                # slight tilt into the turn
    # scowl then roar
    sc = env(t, 5.3, 5.8, 6.0, 6.3)
    roar = env(t, 6.0, 6.35, 7.0, 7.5)
    bd = max(sc, roar)                                          # brow stays crushed down through the roar
    rot('head', X, 7 * sc); mv('brow', (0, 0.004 * bd, -0.013 * bd))
    rot('jaw', X, 11 * roar); rot('head', X, -6 * roar); rot('neck', X, -4 * roar); rot('chest', X, -6 * roar)
    rot('clav.L', Y, 6 * roar); rot('clav.R', Y, -6 * roar)
    rot('upper.L', Y, 14 * roar); rot('fingers.L', X, -18 * roar)       # arm out, claws spread
    rot('head', Y, 4 * math.sin(2 * math.pi * 7 * t) * env(t, 6.3, 6.5, 6.9, 7.1))   # roar shake
    # mace attack (two-handed, after the game's attack): the mace itself is driven (mace.ctl), both arms
    # follow by IK; here only the body: coil, chop with a twist and a step, sweep, recover
    tw = keyed(t, TWIST)
    rot('chest', Zax, tw); rot('spine', Zax, 0.5 * tw); rot('pelvis', Zax, 0.3 * tw)
    lean = keyed(t, LEAN)
    rot('chest', X, lean); rot('spine', X, 0.5 * lean)
    rot('head', Zax, keyed(t, HEADZ)); rot('head', X, keyed(t, HEADX))
    mv('pelvis', (0, 0, keyed(t, DROP)))
    step = env(t, 8.6, 9.0, 10.6, 11.2)
    lift = env(t, 8.6, 8.8, 8.8, 9.0) + env(t, 10.6, 10.9, 10.9, 11.2)
    mv('ik.L', (-0.08 * step, -0.36 * step, 0.15 * lift))
    # claw hand: curls round the shaft while it grips; after letting go it clenches
    g = grip_l(t)
    cl = env(t, 11.3, 11.55, 11.7, 11.98)
    rot('fingers.L', WLAX, 78 * g + 34 * cl); rot('hand.L', WLAX, 8 * cl)
    R['mace.ctl'] = [(mace_q(t), None)]
    Lc['mace.ctl'] = mace_t(t)
    return R, Lc


def apply_pose(t, key=None):
    R, Lc = pose(t)
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        rest = pb.bone.matrix_local.to_quaternion()
        q = Quaternion()
        for axis, deg in R.get(pb.name, []):
            q = (axis if deg is None else Quaternion(axis, math.radians(deg))) @ q
        pb.rotation_quaternion = rest.inverted() @ q @ rest
        pb.location = rest.inverted() @ Lc.get(pb.name, Vector())
        if key is not None:
            pb.keyframe_insert('rotation_quaternion', frame=key); pb.keyframe_insert('location', frame=key)
    g = grip_l(t)
    for c in (arm.pose.bones['fore.L'].constraints['IK'], arm.pose.bones['hand.L'].constraints['grip']):
        c.influence = g
        if key is not None:
            c.keyframe_insert('influence', frame=key)


# ---------------------------------------------------------------- face channels
eye_mat = bpy.data.materials['eye']; eye_in = eye_mat.node_tree.nodes['Principled BSDF'].inputs['Emission Strength']
lights = [bpy.data.objects['eye glow 0'].data, bpy.data.objects['eye glow 1'].data]
keys = [bpy.data.objects[f'eye {i}'].data.shape_keys.key_blocks['squint'] for i in (0, 1)]


def face(t, key=None):
    ign = ramp(t, 0.3, 1.4)
    flick = 1 + 0.05 * math.sin(2 * math.pi * 3.1 * t) + 0.03 * math.sin(2 * math.pi * 7.7 * t)
    flare = env(t, 6.0, 6.3, 7.0, 7.6) + 0.6 * env(t, 9.1, 9.25, 9.4, 9.9) + 0.5 * env(t, 11.2, 11.45, 11.6, 12.0)
    e = 1.4 * ign * flick * (1 + 2.2 * flare)          # a faint glow; flares only on the roar and the hits
    eye_in.default_value = e
    for L in lights:
        L.energy = 0.25 * ign * (1 + 3.0 * flare)
    sq = env(t, 5.3, 5.75, 5.95, 6.15) + 0.7 * env(t, 8.6, 8.8, 9.1, 9.35)
    blink = env(t, 2.0, 2.06, 2.08, 2.16) + env(t, 9.95, 10.0, 10.02, 10.1)
    for k in keys:
        k.value = min(1.0, sq + blink)
    if key is not None:
        eye_in.keyframe_insert('default_value', frame=key)
        for L in lights:
            L.keyframe_insert('energy', frame=key)
        for k in keys:
            k.keyframe_insert('value', frame=key)


# ---------------------------------------------------------------- camera
cam = S.camera; cam.rotation_mode = 'QUATERNION'
CAM = [  # time, azimuth, elevation, distance, target height, target x, lens
    (0.0, 28, 4, 10.5, 2.15, 0.1, 50), (2.0, 22, 4, 9.0, 2.6, 0.05, 50), (4.2, 12, 3, 4.6, 3.45, 0.0, 55),
    (5.6, 6, 1, 3.3, 3.55, 0.0, 60), (7.3, 4, 1, 3.5, 3.55, 0.0, 60), (8.3, -8, 3, 12.5, 2.0, 0.15, 50),
    (10.6, -16, 3, 12.0, 2.0, 0.2, 50), (12.0, 18, 5, 9.0, 2.6, 0.1, 55)]


def camstate(t):
    for (t0, *a), (t1, *b) in zip(CAM[:-1], CAM[1:]):
        if t <= t1:
            u = sm((t - t0) / (t1 - t0)); return [x + (y - x) * u for x, y in zip(a, b)]
    return CAM[-1][1:]


def place_cam(t, key=None):
    az, el, dist, th, tx, lens = camstate(t)
    a, e = math.radians(az), math.radians(el)
    tgt = Vector((-tx, -0.3, th))
    loc = tgt + Vector((dist * math.sin(a) * math.cos(e), -dist * math.cos(a) * math.cos(e), dist * math.sin(e)))
    cam.location = loc; cam.rotation_quaternion = (tgt - loc).to_track_quat('-Z', 'Y')
    if A.zoom:
        zm, zh, zx = map(float, A.zoom.split(','))
        tgt = Vector((-zx, -0.3, zh)); loc = tgt + (loc - tgt) * 1.0; cam.rotation_quaternion = (tgt - loc).to_track_quat('-Z', 'Y'); lens *= zm
    cam.data.lens = lens; cam.data.dof.focus_distance = (tgt - loc).length; cam.data.dof.aperture_fstop = 4.0
    if key is not None:
        cam.keyframe_insert('location', frame=key); cam.keyframe_insert('rotation_quaternion', frame=key)
        cam.data.keyframe_insert('lens', frame=key); cam.data.dof.keyframe_insert('focus_distance', frame=key)


# ---------------------------------------------------------------- render settings for video
w, h = map(int, A.res.split('x'))
S.render.resolution_x, S.render.resolution_y = w, h
S.cycles.samples = A.samples; S.cycles.adaptive_threshold = 0.05; S.cycles.use_denoising = True
S.cycles.max_bounces = 4; S.cycles.diffuse_bounces = 2; S.cycles.glossy_bounces = 2
S.render.use_persistent_data = True
S.view_settings.exposure = 0.1
S.render.fps = FPS; S.frame_start = 1; S.frame_end = NF

if A.test:
    import os
    os.makedirs(A.outdir, exist_ok=True)
    for ts in A.test.split(','):
        t = float(ts)
        apply_pose(t); face(t)
        if not A.nocam:
            place_cam(t)
        bpy.context.view_layer.update()
        S.render.filepath = f'{A.outdir}/t{t:05.2f}.png'
        bpy.ops.render.render(write_still=True)
        print('TEST_DONE', t)
else:
    for f in range(1, NF + 1):
        t = (f - 1) / FPS
        apply_pose(t, key=f); face(t, key=f); place_cam(t, key=f)
    for act in [o.animation_data.action for o in bpy.data.objects if o.animation_data and o.animation_data.action] + \
               [l.animation_data.action for l in lights if l.animation_data] + [eye_mat.node_tree.animation_data.action] + \
               [cam.data.animation_data.action] + [bpy.data.objects['eye 0'].data.shape_keys.animation_data.action, bpy.data.objects['eye 1'].data.shape_keys.animation_data.action]:
        for fc in getattr(act, 'fcurves', []):
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'
    S.render.image_settings.file_format = 'PNG'
    S.render.filepath = '/home/claude/molynx_blender/anim/frames/f_'
    # the skirt: hold the first pose for a second before frame 1 so the cloth settles, cache to disk, bake
    sk = bpy.data.objects.get('skirt')
    if sk and 'cloth' in sk.modifiers:
        pc = sk.modifiers['cloth'].point_cache; pc.frame_start = -24; pc.frame_end = NF; pc.use_disk_cache = True
        bpy.ops.wm.save_as_mainfile(filepath=A.save, compress=True)
        import time as _t; t0 = _t.time()
        S.frame_set(-24)
        for f in range(-24, NF + 1):
            S.frame_set(f)
        with bpy.context.temp_override(scene=S, point_cache=pc):
            bpy.ops.ptcache.bake_from_cache()
        print('CLOTH_BAKED', round(_t.time() - t0, 1), 's')
    bpy.ops.wm.save_as_mainfile(filepath=A.save, compress=True)
    print('ANIM_SAVED', A.save, NF, 'frames')
