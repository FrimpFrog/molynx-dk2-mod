"""Shots for the 1999-style Molynx cinematic. Each shot re-photographs a range of the baked performance
(frames of anim.blend, 24 per second of performance time) with its own camera.
blender -b dk99.blend --python dk99_shots.py -- --shot 3 --test 120,140          (stills)
blender -b dk99.blend --python dk99_shots.py -- --shot 3 --render                 (all frames of the shot)"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

ap = argparse.ArgumentParser()
ap.add_argument('--shot', type=int, default=1)
ap.add_argument('--test', default='')
ap.add_argument('--render', action='store_true')
ap.add_argument('--res', default='480x360')
ap.add_argument('--samples', type=int, default=10)
ap.add_argument('--out', default='/home/claude/molynx_blender/dk99')
ap.add_argument('--nomb', action='store_true')
A = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
S = bpy.context.scene


def sm(x):
    x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)


# shot: anim frame range, camera keys (frame, location, look-at target, lens), shake (frame from, to, amp deg)
SHOTS = {
    1: dict(name='establish', f=(-24, 40),
            keys=[(-24, (0.9, -14.6, 1.25), (0.0, 0.0, 2.6), 22), (40, (0.6, -10.8, 1.5), (0.0, 0.0, 2.9), 24)]),
    2: dict(name='torch', f=(48, 100),
            keys=[(48, (-3.0, -7.2, 2.7), (0.1, -0.2, 2.9), 30), (100, (-2.4, -6.8, 2.6), (0.0, -0.2, 3.0), 30)]),
    3: dict(name='tilt up', f=(100, 140),
            keys=[(100, (0.35, -3.6, 0.7), (0.1, -0.4, 1.9), 24), (140, (0.3, -2.9, 3.0), (0.0, -0.4, 3.6), 26)], hide=['light beam']),
    4: dict(name='roar', f=(138, 184),
            keys=[(138, (0.4, -2.1, 3.5), (0.02, -0.5, 3.62), 27), (184, (0.32, -1.85, 3.45), (0.02, -0.5, 3.62), 27)],
            shake=(150, 178, 0.6), hide=['light beam']),
    5: dict(name='wind up', f=(186, 214),
            keys=[(186, (-2.2, -3.6, 1.0), (0.2, 0.0, 3.4), 28), (214, (-2.5, -3.3, 0.9), (0.2, 0.0, 3.7), 28)]),
    6: dict(name='impact', f=(212, 236),
            keys=[(212, (-2.0, -4.6, 0.35), (-0.45, -1.4, 1.3), 25), (236, (-2.15, -4.85, 0.33), (-0.45, -1.4, 1.3), 25)],
            shake=(219, 234, 1.1)),
    7: dict(name='hero', f=(236, 296),
            keys=[(236, (-3.2, -5.8, 1.0), (0.0, -0.2, 2.9), 26), (296, (-1.9, -3.7, 1.5), (0.0, -0.4, 3.2), 30)]),
}
SH = SHOTS[A.shot]

cam = S.camera
for nm in SH.get('hide', []):
    o = bpy.data.objects.get(nm)
    if o:
        o.hide_render = True
cam.animation_data_clear(); cam.data.animation_data_clear()
cam.rotation_mode = 'QUATERNION'
cam.data.dof.use_dof = False
cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = 36
cam.data.clip_start = 0.05; cam.data.clip_end = 200


def shake(f):
    if 'shake' not in SH:
        return Euler((0, 0, 0))
    a, b, amp = SH['shake']
    k = sm((f - a) / 2.0) * (1 - sm((f - b + 6) / 6.0)) if a <= f else 0.0
    if f > b:
        k = 0
    t = f / 24.0
    return Euler((math.radians(amp * k * (math.sin(t * 61) * 0.6 + math.sin(t * 37 + 1) * 0.4)),
                  math.radians(amp * k * 0.5 * math.sin(t * 53 + 2)),
                  math.radians(amp * k * (math.sin(t * 47 + 3) * 0.6 + math.sin(t * 29) * 0.4))))


def place(f):
    ks = SH['keys']
    if f <= ks[0][0]:
        k0 = k1 = ks[0]; u = 0
    else:
        for k0, k1 in zip(ks[:-1], ks[1:]):
            if f <= k1[0]:
                break
        u = min(1.0, (f - k0[0]) / max(1, (k1[0] - k0[0])))
        u = 0.5 * u + 0.5 * sm(u)          # mostly-linear dolly, eased
    loc = Vector(k0[1]).lerp(Vector(k1[1]), u); tgt = Vector(k0[2]).lerp(Vector(k1[2]), u)
    lens = k0[3] + (k1[3] - k0[3]) * u
    q = (tgt - loc).to_track_quat('-Z', 'Y')
    q = q @ shake(f).to_quaternion()
    cam.location = loc; cam.rotation_quaternion = q; cam.data.lens = lens


f0, f1 = SH['f']
for f in range(f0 - 1, f1 + 2):
    place(f); cam.keyframe_insert('location', frame=f); cam.keyframe_insert('rotation_quaternion', frame=f)
    cam.data.keyframe_insert('lens', frame=f)

w, h = map(int, A.res.split('x'))
S.render.resolution_x, S.render.resolution_y = w, h; S.render.resolution_percentage = 100
S.render.pixel_aspect_x = S.render.pixel_aspect_y = 1
S.cycles.samples = A.samples; S.cycles.adaptive_threshold = 0.06; S.cycles.use_denoising = True
S.cycles.max_bounces = 3; S.cycles.diffuse_bounces = 1; S.cycles.glossy_bounces = 1; S.cycles.volume_bounces = 0
S.cycles.transparent_max_bounces = 8
S.cycles.volume_step_rate = 4.0
S.render.use_persistent_data = True
S.render.use_motion_blur = not A.nomb; S.render.motion_blur_shutter = 0.5
S.view_settings.exposure = 0.0
S.render.image_settings.file_format = 'PNG'
od = f"{A.out}/shots/s{A.shot}"; os.makedirs(od, exist_ok=True)

emo = bpy.data.objects.get('impact emitter')
if emo:
    for ps in emo.particle_systems:
        ps.point_cache.frame_start = 215; ps.point_cache.frame_end = 300
    with bpy.context.temp_override(scene=S):
        bpy.ops.ptcache.bake_all(bake=True)

if A.test:
    for fs in A.test.split(','):
        f = int(fs); S.frame_set(f)
        S.render.filepath = f'{A.out}/tests/s{A.shot}_{f:04d}.png'
        bpy.ops.render.render(write_still=True)
        print('TEST_DONE', A.shot, f)
elif A.render:
    # frame by frame (shot 1 starts before frame 0, which an animation render cannot); names offset by 1000
    for f in range(f0, f1 + 1):
        fp = f'{od}/f_{f + 1000:04d}.png'
        if os.path.exists(fp):
            continue
        S.frame_set(f); S.render.filepath = fp
        bpy.ops.render.render(write_still=True)
    print('SHOT_DONE', A.shot)
