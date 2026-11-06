"""Square head-and-shoulders portrait of Molynx from dk99.blend.
blender -b dk99.blend --python portrait.py -- --frames 120,160 --res 360 --samples 10 --out DIR [--cam x,y,z] [--tgt x,y,z] [--lens 70]"""
import argparse, math, os, sys
import bpy
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument('--frames', default='160')
ap.add_argument('--res', type=int, default=360)
ap.add_argument('--samples', type=int, default=10)
ap.add_argument('--out', required=True)
ap.add_argument('--cam', default='0.75,-2.75,3.55')
ap.add_argument('--tgt', default='0.02,-0.45,3.58')
ap.add_argument('--lens', type=float, default=72)
ap.add_argument('--key', type=float, default=60)
A = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
S = bpy.context.scene

for nm in ['light beam']:
    o = bpy.data.objects.get(nm)
    if o: o.hide_render = True

c = S.camera
c.animation_data_clear(); c.data.animation_data_clear()
c.rotation_mode = 'QUATERNION'
c.location = Vector(map(float, A.cam.split(','))); t = Vector(map(float, A.tgt.split(',')))
c.rotation_quaternion = (t - c.location).to_track_quat('-Z', 'Y')
c.data.lens = A.lens; c.data.sensor_fit = 'HORIZONTAL'; c.data.sensor_width = 36
c.data.dof.use_dof = True; c.data.dof.focus_distance = (t - c.location).length; c.data.dof.aperture_fstop = 2.8

# warm portrait key, low and to camera-right like a torch beside him
if A.key > 0:
    ld = bpy.data.lights.new('portrait key', 'AREA'); ld.energy = A.key; ld.size = 0.6
    ld.color = (1.0, 0.62, 0.32)
    lo = bpy.data.objects.new('portrait key', ld); S.collection.objects.link(lo)
    lo.location = Vector((1.3, -1.9, 3.1))
    lo.rotation_mode = 'QUATERNION'; lo.rotation_quaternion = (t - lo.location).to_track_quat('-Z', 'Y')

S.render.resolution_x = S.render.resolution_y = A.res; S.render.resolution_percentage = 100
S.cycles.samples = A.samples; S.cycles.adaptive_threshold = 0.03; S.cycles.use_denoising = True
S.cycles.max_bounces = 4; S.cycles.diffuse_bounces = 2; S.cycles.glossy_bounces = 2; S.cycles.volume_bounces = 0
S.render.use_motion_blur = False
S.render.image_settings.file_format = 'PNG'
os.makedirs(A.out, exist_ok=True)
for fs in A.frames.split(','):
    f = int(fs); S.frame_set(f)
    S.render.filepath = f'{A.out}/p_{f:04d}.png'
    bpy.ops.render.render(write_still=True)
    print('P_DONE', f)
