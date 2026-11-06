"""Molynx in Blender/Cycles: imports the exported parts, builds procedural CGI materials (rock with
real displacement and subsurface scattering, bone horns, stained teeth, burning eyes, iron, wood,
leather), lights, cameras and a compositor glow; renders one shot and saves the .blend.

blender -b --factory-startup -P blender_scene.py -- --expr grin --shot face --out x.png [--samples 96] [--res 1280x720] [--blend f.blend]
"""
import argparse
import math
import sys

import bpy
import numpy as np

P = '/home/claude/molynx_blender/parts/'
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--expr', default='grin'); ap.add_argument('--shot', default='face'); ap.add_argument('--out', default='/tmp/x.png')
ap.add_argument('--samples', type=int, default=96); ap.add_argument('--res', default='1280x720'); ap.add_argument('--blend', default='')
ap.add_argument('--cam', default='');  ap.add_argument('--bg', default='dk2'); ap.add_argument('--dicing', type=float, default=1.5); ap.add_argument('--nodisp', action='store_true'); ap.add_argument('--clay', action='store_true'); ap.add_argument('--gamebody', action='store_true'); ap.add_argument('--buildonly', action='store_true'); ap.add_argument('--nogear', action='store_true')
A = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
S = bpy.context.scene
S.render.engine = 'CYCLES'
S.cycles.device = 'CPU'
S.cycles.feature_set = 'EXPERIMENTAL'          # adaptive subdivision: true micro-displacement
S.cycles.samples = A.samples
S.cycles.use_adaptive_sampling = True
S.cycles.adaptive_threshold = 0.02
S.cycles.use_denoising = True
S.cycles.denoiser = 'OPENIMAGEDENOISE'
S.cycles.max_bounces = 6; S.cycles.diffuse_bounces = 3; S.cycles.glossy_bounces = 3; S.cycles.transmission_bounces = 4
S.cycles.dicing_rate = A.dicing; S.cycles.offscreen_dicing_scale = 8
w, h = map(int, A.res.split('x')); S.render.resolution_x, S.render.resolution_y = w, h; S.render.resolution_percentage = 100
S.render.film_transparent = False
S.view_settings.view_transform = 'Khronos PBR Neutral'; S.view_settings.look = 'None'; S.view_settings.exposure = -0.3
S.render.threads_mode = 'AUTO'


# ---------------------------------------------------------------- geometry
def to_z(V):
    V = np.asarray(V, float)
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)


def make(name, V, F, uv_corner=None, uv_vert=None, smooth=True, attrs=None):
    V = to_z(V); F = np.asarray(F, np.int64)
    attrs = dict(attrs or {}); attrs['rest'] = V            # rest position: procedural textures stick to the skin when it moves
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V)); me.vertices.foreach_set('co', V.astype(np.float32).ravel())
    me.loops.add(len(F) * 3); me.loops.foreach_set('vertex_index', F.astype(np.int32).ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set('loop_start', np.arange(0, len(F) * 3, 3, dtype=np.int32)); me.polygons.foreach_set('loop_total', np.full(len(F), 3, np.int32))
    if uv_vert is not None:
        uv_corner = np.asarray(uv_vert)[F]
    if uv_corner is not None:
        uvl = me.uv_layers.new(name='UVMap')
        u = np.asarray(uv_corner, float).reshape(-1, 2).copy(); u[:, 1] = 1 - u[:, 1]
        uvl.data.foreach_set('uv', u.astype(np.float32).ravel())
    if attrs:
        for an, av in attrs.items():
            av = np.asarray(av, np.float32)
            if av.ndim == 2:
                at = me.attributes.new(an, 'FLOAT_VECTOR', 'POINT'); at.data.foreach_set('vector', av.ravel())
            else:
                at = me.attributes.new(an, 'FLOAT', 'POINT'); at.data.foreach_set('value', av)
    me.polygons.foreach_set('use_smooth', np.full(len(F), smooth))
    me.update(); me.validate(clean_customdata=False)
    ob = bpy.data.objects.new(name, me); S.collection.objects.link(ob)
    return ob


def load(name):
    return dict(np.load(P + name + '.npz'))


# ---------------------------------------------------------------- node helpers
def mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return m, nt, bsdf, out


def node(nt, kind, **kw):
    n = nt.nodes.new(kind)
    for k, v in kw.items():
        if k.startswith('in_'):
            n.inputs[k[3:].replace('_', ' ')].default_value = v
        else:
            setattr(n, k, v)
    return n


def link(nt, a, b):
    nt.links.new(a, b)


def math_node(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new('ShaderNodeMath'); n.operation = op; n.use_clamp = clamp
    if isinstance(a, (int, float)):
        n.inputs[0].default_value = a
    else:
        link(nt, a, n.inputs[0])
    if b is not None:
        if isinstance(b, (int, float)):
            n.inputs[1].default_value = b
        else:
            link(nt, b, n.inputs[1])
    return n.outputs[0]


def ramp(nt, inp, stops):
    r = nt.nodes.new('ShaderNodeValToRGB'); el = r.color_ramp.elements
    el[0].position, el[0].color = stops[0][0], (*stops[0][1], 1)
    el[1].position, el[1].color = stops[-1][0], (*stops[-1][1], 1)
    for pos, col in stops[1:-1]:
        e = el.new(pos); e.color = (*col, 1)
    link(nt, inp, r.inputs['Fac'])
    return r.outputs['Color']


# ---------------------------------------------------------------- materials
def rest(nt):
    """the vertex's rest position (stored per vertex by make): texture space that moves with the skin"""
    a = nt.nodes.new('ShaderNodeAttribute'); a.attribute_type = 'GEOMETRY'; a.attribute_name = 'rest'
    return a.outputs['Vector']


def rock_material(name='rock', dark=1.0, ember=1.0, scale=1.0, calm=False):
    """Cracked red stone: plates (each its own height and tint), deep cracks with a faint ember,
    pitted surface; real displacement + subsurface scattering."""
    m, nt, bsdf, out = mat(name)
    tc = node(nt, 'ShaderNodeTexCoord')
    co = rest(nt)
    vor = node(nt, 'ShaderNodeTexVoronoi', feature='F1', in_Scale=11.0 * scale, in_Randomness=0.85)
    link(nt, co, vor.inputs['Vector'])
    edge = node(nt, 'ShaderNodeTexVoronoi', feature='DISTANCE_TO_EDGE', in_Scale=11.0 * scale, in_Randomness=0.85)
    link(nt, co, edge.inputs['Vector'])
    fine = node(nt, 'ShaderNodeTexVoronoi', feature='DISTANCE_TO_EDGE', in_Scale=30.0 * scale, in_Randomness=1.0)
    link(nt, co, fine.inputs['Vector'])
    noise = node(nt, 'ShaderNodeTexNoise', in_Scale=38.0 * scale, in_Detail=10.0, in_Roughness=0.62)
    link(nt, co, noise.inputs['Vector'])
    pits = node(nt, 'ShaderNodeTexNoise', in_Scale=140.0 * scale, in_Detail=4.0, in_Roughness=0.7)
    link(nt, co, pits.inputs['Vector'])
    # crack masks (1 in a crack)
    crack = math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'DIVIDE', edge.outputs['Distance'], 0.045, clamp=True), clamp=True)
    crack = math_node(nt, 'POWER', crack, 1.6)
    fcrack = math_node(nt, 'SUBTRACT', 1.0, math_node(nt, 'DIVIDE', fine.outputs['Distance'], 0.05, clamp=True), clamp=True)
    fcrack = math_node(nt, 'POWER', fcrack, 2.2)
    if calm:        # features (nose, lips, eyelids) are smoother stone: fewer, finer cracks so they read
        ca = nt.nodes.new('ShaderNodeAttribute'); ca.attribute_type = 'GEOMETRY'; ca.attribute_name = 'calm'
        cf = math_node(nt, 'ADD', 0.15, math_node(nt, 'MULTIPLY', ca.outputs['Fac'], 0.85))
        crack = math_node(nt, 'MULTIPLY', crack, cf); fcrack = math_node(nt, 'MULTIPLY', fcrack, cf)
    # per-plate random level & tint
    sep = node(nt, 'ShaderNodeSeparateColor'); link(nt, vor.outputs['Color'], sep.inputs['Color'])
    level = sep.outputs['Red']; tint = sep.outputs['Green']
    # height: plate level - cracks + pitting
    hgt = math_node(nt, 'MULTIPLY', level, 0.45)
    if calm:
        hgt = math_node(nt, 'MULTIPLY', hgt, cf)
    hgt = math_node(nt, 'SUBTRACT', hgt, math_node(nt, 'MULTIPLY', crack, 1.2))
    hgt = math_node(nt, 'SUBTRACT', hgt, math_node(nt, 'MULTIPLY', fcrack, 0.45))
    hgt = math_node(nt, 'ADD', hgt, math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', noise.outputs['Fac'], 0.5), 0.5))
    hgt = math_node(nt, 'ADD', hgt, math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', pits.outputs['Fac'], 0.5), 0.25))
    disp = node(nt, 'ShaderNodeDisplacement', in_Midlevel=0.0, in_Scale=0.022 * dark ** 0)
    link(nt, hgt, disp.inputs['Height'])
    link(nt, disp.outputs['Displacement'], out.inputs['Displacement'])
    m.displacement_method = 'BOTH'
    # colour: dark red to blood red per plate, weathered by noise, near-black cracks
    base = ramp(nt, tint, [(0.0, (0.03 * dark, 0.0025, 0.003)), (0.5, (0.075 * dark, 0.005, 0.006)), (1.0, (0.15 * dark, 0.01, 0.011))])
    weather = ramp(nt, noise.outputs['Fac'], [(0.3, (0.55, 0.5, 0.48)), (0.7, (1.1, 1.0, 1.0))])
    mixw = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', in_Factor=1.0)
    link(nt, base, mixw.inputs[6]); link(nt, weather, mixw.inputs[7])
    dark_c = math_node(nt, 'MAXIMUM', crack, math_node(nt, 'MULTIPLY', fcrack, 0.7))
    mixc = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MIX')
    link(nt, dark_c, mixc.inputs['Factor']); link(nt, mixw.outputs[2], mixc.inputs[6]); mixc.inputs[7].default_value = (0.006, 0.0015, 0.001, 1)
    # cavity shading: grooves between muscles and plates go darker (local ambient occlusion)
    ao = node(nt, 'ShaderNodeAmbientOcclusion'); ao.samples = 8; ao.only_local = True; ao.inputs['Distance'].default_value = 0.09
    mixa = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', in_Factor=0.7)
    link(nt, mixc.outputs[2], mixa.inputs[6]); link(nt, ao.outputs['AO'], mixa.inputs[7])
    link(nt, mixa.outputs[2], bsdf.inputs['Base Color'])
    rough = math_node(nt, 'ADD', 0.62, math_node(nt, 'MULTIPLY', dark_c, 0.25))
    link(nt, rough, bsdf.inputs['Roughness'])
    bsdf.inputs['Subsurface Weight'].default_value = 0.06
    bsdf.inputs['Subsurface Radius'].default_value = (0.6, 0.1, 0.1)
    bsdf.inputs['Subsurface Scale'].default_value = 0.012
    bsdf.inputs['Specular IOR Level'].default_value = 0.45
    # ember: the deepest main cracks smoulder faintly
    deep = math_node(nt, 'POWER', crack, 10.0)
    em = math_node(nt, 'MULTIPLY', deep, 0.35 * ember)
    bsdf.inputs['Emission Color'].default_value = (1.0, 0.22, 0.03, 1)
    link(nt, em, bsdf.inputs['Emission Strength'])
    return m


def leather_proc(name='belt leather'):
    """worn dark leather: scuffs, creases, a darker edge"""
    m, nt, bsdf, out = mat(name)
    n = node(nt, 'ShaderNodeTexNoise', in_Scale=30.0, in_Detail=10.0, in_Roughness=0.65); link(nt, rest(nt), n.inputs['Vector'])
    c = ramp(nt, n.outputs['Fac'], [(0.3, (0.018, 0.009, 0.005)), (0.7, (0.075, 0.04, 0.02))])
    link(nt, c, bsdf.inputs['Base Color'])
    r = ramp(nt, n.outputs['Fac'], [(0.3, (0.38, 0.38, 0.38)), (0.7, (0.62, 0.62, 0.62))]); link(nt, r, bsdf.inputs['Roughness'])
    s2 = node(nt, 'ShaderNodeTexNoise', in_Scale=160.0, in_Detail=6.0); link(nt, rest(nt), s2.inputs['Vector'])
    h = math_node(nt, 'ADD', n.outputs['Fac'], math_node(nt, 'MULTIPLY', s2.outputs['Fac'], 0.6))
    b = node(nt, 'ShaderNodeBump', in_Strength=0.35, in_Distance=0.006); link(nt, h, b.inputs['Height']); link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
    return m


def cloth_proc(name='skirt cloth'):
    """coarse, dirty, sun-bleached cloth; grime gathers toward the torn hem (attribute t: 0 top, 1 hem)"""
    m, nt, bsdf, out = mat(name)
    g = node(nt, 'ShaderNodeTexNoise', in_Scale=9.0, in_Detail=8.0, in_Roughness=0.6); link(nt, rest(nt), g.inputs['Vector'])
    base = ramp(nt, g.outputs['Fac'], [(0.3, (0.11, 0.085, 0.06)), (0.72, (0.3, 0.25, 0.17))])
    ta = nt.nodes.new('ShaderNodeAttribute'); ta.attribute_type = 'GEOMETRY'; ta.attribute_name = 't'
    hem = ramp(nt, ta.outputs['Fac'], [(0.55, (1.0, 1.0, 1.0)), (1.0, (0.35, 0.28, 0.22))])
    mx = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', in_Factor=1.0)
    link(nt, base, mx.inputs[6]); link(nt, hem, mx.inputs[7]); link(nt, mx.outputs[2], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.95
    bsdf.inputs['Sheen Weight'].default_value = 0.6; bsdf.inputs['Sheen Tint'].default_value = (0.9, 0.85, 0.7, 1)
    bsdf.inputs['Subsurface Weight'].default_value = 0.0
    w = node(nt, 'ShaderNodeTexWave', wave_type='BANDS', in_Scale=260.0, in_Distortion=1.5); link(nt, rest(nt), w.inputs['Vector'])
    w2 = node(nt, 'ShaderNodeTexWave', wave_type='BANDS', bands_direction='Y', in_Scale=260.0, in_Distortion=1.5); link(nt, rest(nt), w2.inputs['Vector'])
    h = math_node(nt, 'ADD', math_node(nt, 'MULTIPLY', w.outputs['Fac'], 0.5), math_node(nt, 'MULTIPLY', w2.outputs['Fac'], 0.5))
    h = math_node(nt, 'ADD', h, g.outputs['Fac'])
    b = node(nt, 'ShaderNodeBump', in_Strength=0.25, in_Distance=0.004); link(nt, h, b.inputs['Height']); link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
    return m


def simple(name, color, rough, metal=0.0, sss=0.0, coat=0.0, emit=None, emit_str=0.0):
    m, nt, bsdf, out = mat(name)
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = rough; bsdf.inputs['Metallic'].default_value = metal
    bsdf.inputs['Subsurface Weight'].default_value = sss
    bsdf.inputs['Coat Weight'].default_value = coat
    if emit:
        bsdf.inputs['Emission Color'].default_value = (*emit, 1); bsdf.inputs['Emission Strength'].default_value = emit_str
    return m, nt, bsdf, out


def bump_noise(nt, bsdf, scale, strength, dist=0.01):
    tc = node(nt, 'ShaderNodeTexCoord'); n = node(nt, 'ShaderNodeTexNoise', in_Scale=scale, in_Detail=8.0)
    link(nt, rest(nt), n.inputs['Vector'])
    b = node(nt, 'ShaderNodeBump', in_Strength=strength, in_Distance=dist); link(nt, n.outputs['Fac'], b.inputs['Height'])
    link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
    return n


def horn_material():
    m, nt, bsdf, out = mat('horn')
    at = node(nt, 'ShaderNodeAttribute', attribute_name='t')
    col = ramp(nt, at.outputs['Fac'], [(0.0, (0.02, 0.011, 0.006)), (0.3, (0.09, 0.055, 0.03)), (0.75, (0.3, 0.24, 0.15)), (1.0, (0.42, 0.36, 0.25))])
    tc = node(nt, 'ShaderNodeTexCoord'); n = node(nt, 'ShaderNodeTexNoise', in_Scale=60.0, in_Detail=6.0); link(nt, rest(nt), n.inputs['Vector'])
    # growth rings: uneven, wobbling ridges rather than regular stripes
    wob = math_node(nt, 'ADD', math_node(nt, 'MULTIPLY', at.outputs['Fac'], 34.0), math_node(nt, 'MULTIPLY', n.outputs['Fac'], 2.5))
    rings = math_node(nt, 'POWER', math_node(nt, 'ABSOLUTE', math_node(nt, 'SINE', wob)), 6.0)
    grain = node(nt, 'ShaderNodeTexNoise', in_Scale=8.0, in_Detail=10.0, in_Distortion=0.5)
    gm = node(nt, 'ShaderNodeMapping'); gm.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
    link(nt, rest(nt), gm.inputs['Vector']); link(nt, gm.outputs['Vector'], grain.inputs['Vector'])
    streak = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', in_Factor=0.6)
    link(nt, col, streak.inputs[6]); link(nt, n.outputs['Color'], streak.inputs[7])
    dark_r = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
    link(nt, math_node(nt, 'MULTIPLY', rings, 0.55), dark_r.inputs['Factor']); link(nt, streak.outputs[2], dark_r.inputs[6]); dark_r.inputs[7].default_value = (0.35, 0.3, 0.25, 1)
    link(nt, dark_r.outputs[2], bsdf.inputs['Base Color'])
    hb = math_node(nt, 'ADD', math_node(nt, 'MULTIPLY', rings, -0.25), math_node(nt, 'MULTIPLY', n.outputs['Fac'], 0.5))
    b = node(nt, 'ShaderNodeBump', in_Strength=0.35, in_Distance=0.01); link(nt, hb, b.inputs['Height']); link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
    bsdf.inputs['Roughness'].default_value = 0.55; bsdf.inputs['Subsurface Weight'].default_value = 0.08
    bsdf.inputs['Subsurface Radius'].default_value = (0.5, 0.35, 0.2); bsdf.inputs['Subsurface Scale'].default_value = 0.02
    return m


def teeth_material():
    m, nt, bsdf, out = mat('teeth')
    at = node(nt, 'ShaderNodeAttribute', attribute_name='t')
    col = ramp(nt, at.outputs['Fac'], [(0.0, (0.02, 0.008, 0.004)), (0.35, (0.16, 0.09, 0.035)), (1.0, (0.36, 0.28, 0.15))])
    link(nt, col, bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.3; bsdf.inputs['Subsurface Weight'].default_value = 0.25
    bsdf.inputs['Subsurface Radius'].default_value = (0.6, 0.45, 0.25); bsdf.inputs['Subsurface Scale'].default_value = 0.01
    bsdf.inputs['Coat Weight'].default_value = 0.4
    bump_noise(nt, bsdf, 220.0, 0.25, 0.002)
    return m


def eye_material():
    """amber eyes with a faint inner glow (Horned-Reaper-like, not lamps) and Molynx's own pupil:
    a horizontal bar with chipped, chiselled edges - goat-like (his horns), like a crack in stone"""
    m, nt, bsdf, out = mat('eye')
    tc = node(nt, 'ShaderNodeTexCoord'); sep = node(nt, 'ShaderNodeSeparateXYZ'); link(nt, tc.outputs['UV'], sep.inputs['Vector'])
    v = math_node(nt, 'SUBTRACT', 1.0, sep.outputs['Y'])         # angle from the front pole (0..1 = 0..pi)
    th = math_node(nt, 'MULTIPLY', v, math.pi); ph = math_node(nt, 'MULTIPLY', sep.outputs['X'], 2 * math.pi)
    st = math_node(nt, 'SINE', th)
    X = math_node(nt, 'MULTIPLY', st, math_node(nt, 'COSINE', ph))     # across (horizontal)
    Y = math_node(nt, 'MULTIPLY', st, math_node(nt, 'SINE', ph))       # up/down
    nz = node(nt, 'ShaderNodeTexNoise', in_Scale=9.0, in_Detail=2.0)
    cx = node(nt, 'ShaderNodeCombineXYZ'); link(nt, X, cx.inputs[0]); link(nt, Y, cx.inputs[1]); link(nt, cx.outputs[0], nz.inputs['Vector'])
    jag = math_node(nt, 'MULTIPLY', math_node(nt, 'SUBTRACT', nz.outputs['Fac'], 0.5), 0.09)
    # bar: |X| < 0.36, |Y| < 0.075 (+ jagged edges), corners chipped
    bx = math_node(nt, 'SUBTRACT', math_node(nt, 'ABSOLUTE', X), math_node(nt, 'ADD', 0.36, jag))
    by = math_node(nt, 'SUBTRACT', math_node(nt, 'ABSOLUTE', Y), math_node(nt, 'ADD', 0.075, math_node(nt, 'MULTIPLY', jag, 0.6)))
    box = math_node(nt, 'MAXIMUM', bx, by)
    pupil = math_node(nt, 'LESS_THAN', box, 0.0)
    # iris: amber with radial striations, darker toward its rim
    stri = node(nt, 'ShaderNodeTexWave', wave_type='RINGS', in_Scale=1.0, in_Distortion=0.0)
    ang = math_node(nt, 'ARCTAN2', Y, X)
    sv = node(nt, 'ShaderNodeCombineXYZ'); link(nt, math_node(nt, 'MULTIPLY', ang, 6.0), sv.inputs[0]); link(nt, sv.outputs[0], stri.inputs['Vector'])
    col = ramp(nt, v, [(0.0, (1.0, 0.82, 0.3)), (0.22, (0.95, 0.58, 0.08)), (0.42, (0.45, 0.13, 0.01)), (0.6, (0.12, 0.03, 0.0))])
    mixs = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', in_Factor=0.35)
    link(nt, col, mixs.inputs[6]); link(nt, stri.outputs['Color'], mixs.inputs[7])
    final = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MIX')
    link(nt, pupil, final.inputs['Factor']); link(nt, mixs.outputs[2], final.inputs[6]); final.inputs[7].default_value = (0.008, 0.004, 0.002, 1)
    link(nt, final.outputs[2], bsdf.inputs['Base Color'])
    glow = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MIX')
    link(nt, pupil, glow.inputs['Factor']); link(nt, col, glow.inputs[6]); glow.inputs[7].default_value = (0, 0, 0, 1)
    link(nt, glow.outputs[2], bsdf.inputs['Emission Color'])
    bsdf.inputs['Emission Strength'].default_value = 1.2
    bsdf.inputs['Roughness'].default_value = 0.08; bsdf.inputs['Coat Weight'].default_value = 1.0
    return m


def leather_cloth_metal(img):
    mats = {}
    for nm, rough, metal, sheen, bstr in (('leather', 0.55, 0.0, 0.0, 0.5), ('iron studs', 0.32, 1.0, 0.0, 0.3), ('loincloth', 0.92, 0.0, 0.8, 0.7)):
        m, nt, bsdf, out = mat(nm)
        t = node(nt, 'ShaderNodeTexImage'); t.image = img
        grime = node(nt, 'ShaderNodeTexNoise', in_Scale=18.0, in_Detail=8.0)
        tcg = node(nt, 'ShaderNodeTexCoord'); link(nt, rest(nt), grime.inputs['Vector'])
        gr = ramp(nt, grime.outputs['Fac'], [(0.3, (0.09, 0.07, 0.05)), (0.75, (0.26, 0.22, 0.16))] if nm == 'loincloth' else [(0.3, (0.6, 0.6, 0.6)), (0.7, (1.0, 1.0, 1.0))])
        mx = node(nt, 'ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY', in_Factor=1.0)
        link(nt, t.outputs['Color'], mx.inputs[6]); link(nt, gr, mx.inputs[7])
        link(nt, mx.outputs[2], bsdf.inputs['Base Color'])
        bsdf.inputs['Roughness'].default_value = rough; bsdf.inputs['Metallic'].default_value = metal
        bsdf.inputs['Sheen Weight'].default_value = sheen; bsdf.inputs['Sheen Tint'].default_value = (0.9, 0.85, 0.7, 1)
        n = node(nt, 'ShaderNodeTexNoise', in_Scale=90.0, in_Detail=10.0)
        tc = node(nt, 'ShaderNodeTexCoord'); link(nt, rest(nt), n.inputs['Vector'])
        hb = math_node(nt, 'ADD', math_node(nt, 'MULTIPLY', t.outputs['Color'], 0.6), n.outputs['Fac'])
        b = node(nt, 'ShaderNodeBump', in_Strength=bstr, in_Distance=0.01); link(nt, hb, b.inputs['Height']); link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
        mats[nm] = m
    mats['iron studs'] = iron('iron studs', 0.12)
    return mats


def iron(name, base=0.22):
    m, nt, bsdf, out = simple(name, (base, base * 0.96, base * 0.92), 0.45, metal=1.0)
    tc = node(nt, 'ShaderNodeTexCoord'); n = node(nt, 'ShaderNodeTexNoise', in_Scale=25.0, in_Detail=12.0); link(nt, rest(nt), n.inputs['Vector'])
    r = ramp(nt, n.outputs['Fac'], [(0.35, (0.28, 0.28, 0.28)), (0.65, (0.62, 0.62, 0.62))]); link(nt, r, bsdf.inputs['Roughness'])
    rust = ramp(nt, n.outputs['Fac'], [(0.55, (base, base * 0.96, base * 0.92)), (0.75, (0.18, 0.07, 0.03))]); link(nt, rust, bsdf.inputs['Base Color'])
    s = node(nt, 'ShaderNodeTexNoise', in_Scale=400.0, in_Detail=2.0); link(nt, rest(nt), s.inputs['Vector'])
    b = node(nt, 'ShaderNodeBump', in_Strength=0.15, in_Distance=0.003); link(nt, s.outputs['Fac'], b.inputs['Height']); link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
    return m


def wood():
    m, nt, bsdf, out = mat('wood')
    tc = node(nt, 'ShaderNodeTexCoord')
    wv = node(nt, 'ShaderNodeTexWave', wave_type='BANDS', bands_direction='X', in_Scale=14.0, in_Distortion=6.0, in_Detail=6.0)
    link(nt, tc.outputs['UV'], wv.inputs['Vector'])
    c = ramp(nt, wv.outputs['Fac'], [(0.0, (0.025, 0.012, 0.006)), (1.0, (0.09, 0.045, 0.02))]); link(nt, c, bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.68
    b = node(nt, 'ShaderNodeBump', in_Strength=0.3, in_Distance=0.005); link(nt, wv.outputs['Fac'], b.inputs['Height']); link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
    return m


def grip():
    m, nt, bsdf, out = mat('grip')
    tc = node(nt, 'ShaderNodeTexCoord'); sep = node(nt, 'ShaderNodeSeparateXYZ'); link(nt, tc.outputs['UV'], sep.inputs['Vector'])
    wrap = math_node(nt, 'SINE', math_node(nt, 'ADD', math_node(nt, 'MULTIPLY', sep.outputs['Y'], 120.0), math_node(nt, 'MULTIPLY', sep.outputs['X'], 6.283)))
    bsdf.inputs['Base Color'].default_value = (0.022, 0.011, 0.006, 1); bsdf.inputs['Roughness'].default_value = 0.6
    b = node(nt, 'ShaderNodeBump', in_Strength=0.7, in_Distance=0.01); link(nt, wrap, b.inputs['Height']); link(nt, b.outputs['Normal'], bsdf.inputs['Normal'])
    return m


# ---------------------------------------------------------------- build
def subsurf_adaptive(ob, levels=2):
    mod = ob.modifiers.new('subd', 'SUBSURF'); mod.levels = 1; mod.render_levels = levels
    if not A.nodisp:
        ob.cycles.use_adaptive_subdivision = True
    return mod


rock = rock_material('rock skin', calm=True)
rock_head = rock_material('rock face', scale=1.45, calm=True)
basalt = rock_material('basalt spurs', dark=0.45, ember=0.4, scale=2.2)
img = bpy.data.images.load(P + 'body_albedo.png')
lcm = leather_cloth_metal(img)

b = load('body' if A.gamebody else 'body_sculpt')
def torso_calm(V):
    """finer, softer cracks over the chest and stomach so the muscles read (1 elsewhere)"""
    V = np.asarray(V, float)
    def band(x, a, b, w):
        return np.clip((x - a) / w, 0, 1) * np.clip((b - x) / w, 0, 1)
    f = band(V[:, 1], 1.95, 3.2, 0.12) * band(V[:, 0], -0.42, 0.42, 0.12) * np.clip((V[:, 2] + 0.05) / 0.15, 0, 1)
    return 1 - 0.6 * f


body = make('Molynx body', b['V'], b['F'], uv_corner=b['UV'], attrs={'calm': torso_calm(b['V']) if 'cls' in b else np.ones(len(b['V']))})
for mm in (rock, lcm['leather'], lcm['iron studs'], lcm['loincloth']):
    body.data.materials.append(mm)
cls = b['cls'].astype(int)
idx = np.select([cls == 0, cls == 1, cls == 2, cls == 3], [0, 1, 2, 3], 0)
body.data.polygons.foreach_set('material_index', idx.astype(np.int32))
if A.gamebody:
    subsurf_adaptive(body, 2)

# gear fitted to the sculpt: belt, cloth skirt, arm armour
GEAR = []
if not A.gamebody and not A.nogear:
    sys.path.insert(0, '/home/claude/molynx_cinematic/tools')
    import gear
    gmats = {'leather': leather_proc(), 'iron': iron('belt iron', 0.2), 'armour': iron('armour iron', 0.17), 'cloth': cloth_proc()}
    for g in gear.build(b['V'], b['F']):
        o = make(g['name'], g['V'], g['F'], attrs=g.get('attrs'))
        o.data.materials.append(gmats[g['mat']]); o['bone'] = g['bone'] or ''
        if g['mat'] == 'cloth':
            sol = o.modifiers.new('thickness', 'SOLIDIFY'); sol.thickness = 0.008; sol.offset = 1.0
            o['grid'] = list(g['grid'])
        elif g['name'] in ('belt', ) or g['mat'] == 'armour':
            o.modifiers.new('subd', 'SUBSURF').render_levels = 1; o.modifiers['subd'].levels = 0
        GEAR.append(o)

hd = load(f'head_{A.expr}')
def face_calm(V):
    """1 on plain rock, down to ~0 on the nose, lips and eyelids (head-local coordinates)"""
    L = np.asarray(V, float) * np.array([-1.0, 1.0, 1.0]) - np.array([0.0275, 3.60, 0.45])
    def ell(c, r):
        return np.linalg.norm((L - np.array(c)) / np.array(r), axis=1)
    nose = np.clip((ell((0, -0.025, 0.27), (0.07, 0.085, 0.06)) - 0.75) / 0.5, 0, 1)
    lips = np.clip((ell((0, -0.145, 0.215), (0.13, 0.05, 0.09)) - 0.85) / 0.45, 0, 1)
    eyes = np.minimum(*[np.clip((ell((sx, 0.045, 0.152), (0.07, 0.05, 0.06)) - 0.8) / 0.5, 0, 1) for sx in (-0.072, 0.072)])
    return np.minimum.reduce([nose, 0.2 + 0.8 * lips, 0.25 + 0.75 * eyes])


head = make('Molynx head', hd['V'], hd['F'], attrs={'calm': face_calm(hd['V'])}); head.data.materials.append(rock_head)
if not A.nodisp:                       # dense sculpt: plain per-vertex displacement on one subdivision level
    pass

hr = load('horns'); horns = make('horns', hr['V'], hr['F'], attrs={'t': hr['t']}); horns.data.materials.append(horn_material())
horns.modifiers.new('subd', 'SUBSURF').render_levels = 1
cl = load('claws' if A.gamebody else 'claws_sculpt'); claws = make('claws', cl['V'], cl['F']); claws.data.materials.append(simple('claw', (0.012, 0.008, 0.006), 0.25, coat=0.6)[0])
claws.modifiers.new('subd', 'SUBSURF').render_levels = 2 if A.gamebody else 1
tt = load(f'teeth_{A.expr}'); teeth = make('teeth', tt['V'], tt['F'], attrs={'t': tt['t']}); teeth.data.materials.append(teeth_material())
em = eye_material()
for i in (0, 1):
    e = load(f'eye{i}'); eo = make(f'eye {i}', e['V'], e['F'], uv_vert=e['UV']); eo.data.materials.append(em)
sp = load('spurs'); spurs = make('stone spurs', sp['V'], sp['F'], smooth=False); spurs.data.materials.append(basalt)
if not A.nodisp:
    spurs.cycles.use_adaptive_subdivision = True; sm = spurs.modifiers.new('subd', 'SUBSURF'); sm.subdivision_type = 'SIMPLE'; sm.render_levels = 2
ir = iron('iron', 0.2); spk = iron('spike iron', 0.3)
for nm, mm in (('ball', ir), ('spikes', spk), ('iron', ir), ('shaft', wood()), ('grip', grip())):
    d = load(f'mace_{nm}')
    o = make(f'mace {nm}', d['V'], d['F'], uv_vert=d.get('UVv'))
    o.data.materials.append(mm)
    if nm in ('ball', 'iron'):
        o.modifiers.new('subd', 'SUBSURF').render_levels = 1

if A.clay:
    clay = simple('clay', (0.35, 0.33, 0.31), 0.6)[0]
    for o in (body, head):
        o.data.materials.clear(); o.data.materials.append(clay)

# ---------------------------------------------------------------- world, floor, lights
world = bpy.data.worlds.new('world'); S.world = world; world.use_nodes = True
wn = world.node_tree; bg = wn.nodes['Background']
if A.bg == 'dk2':
    bg.inputs['Color'].default_value = (0.006, 0.012, 0.03, 1); bg.inputs['Strength'].default_value = 1.0
else:
    bg.inputs['Color'].default_value = (0.006, 0.004, 0.003, 1); bg.inputs['Strength'].default_value = 1.0
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0)); floor = bpy.context.active_object; floor.name = 'floor'
fm, fnt, fb, fo = simple('floor stone', (0.005, 0.0055, 0.007) if A.bg == 'dk2' else (0.005, 0.004, 0.003), 0.55)
fv = node(fnt, 'ShaderNodeTexVoronoi', feature='DISTANCE_TO_EDGE', in_Scale=1.3)
ftc = node(fnt, 'ShaderNodeTexCoord'); link(fnt, ftc.outputs['Object'], fv.inputs['Vector'])
fbump = node(fnt, 'ShaderNodeBump', in_Strength=0.6, in_Distance=0.02); link(fnt, math_node(fnt, 'MINIMUM', fv.outputs['Distance'], 0.04), fbump.inputs['Height']); link(fnt, fbump.outputs['Normal'], fb.inputs['Normal'])
fb.inputs['Specular IOR Level'].default_value = 0.15; fb.inputs['Roughness'].default_value = 0.9
floor.data.materials.append(fm)
# backdrop: a big curved sweep with a DK2-blue (or dungeon) glow behind him
bpy.ops.mesh.primitive_plane_add(size=140, location=(0, 14, 10), rotation=(math.radians(90), 0, 0)); back = bpy.context.active_object; back.name = 'backdrop'
bm_, bnt, bbs, bout = mat('backdrop'); bnt.nodes.remove(bbs)
emi = node(bnt, 'ShaderNodeEmission'); link(bnt, emi.outputs['Emission'], bout.inputs['Surface'])
gtc = node(bnt, 'ShaderNodeTexCoord'); grad = node(bnt, 'ShaderNodeTexGradient', gradient_type='SPHERICAL')
mp = node(bnt, 'ShaderNodeMapping'); mp.inputs['Location'].default_value = (0.0, 0.0, -0.05); mp.inputs['Scale'].default_value = (0.09, 0.12, 0.09)
link(bnt, gtc.outputs['Object'], mp.inputs['Vector']); link(bnt, mp.outputs['Vector'], grad.inputs['Vector'])
cols = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.035, 0.075, 0.2))] if A.bg == 'dk2' else [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.12, 0.05, 0.02))]
link(bnt, ramp(bnt, grad.outputs['Fac'], cols), emi.inputs['Color']); emi.inputs['Strength'].default_value = 1.0
back.data.materials.append(bm_)


def area(name, loc, target, power, color, size):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = power; L.color = color; L.size = size; L.shape = 'DISK'
    o = bpy.data.objects.new(name, L); o.location = loc; S.collection.objects.link(o)
    d = np.array(target) - np.array(loc); o.rotation_euler = (0, 0, 0)
    o.rotation_mode = 'QUATERNION'
    import mathutils
    o.rotation_quaternion = mathutils.Vector(d).to_track_quat('-Z', 'Y')
    return o


# hard-ish warm key from high and to the side (his right), cool and warm rims, almost no front fill
area('key', (-7.0, -2.5, 7.0), (0, 0, 2.6), 1300, (1.0, 0.86, 0.74), 1.6)
area('rim blue', (5.0, 6.5, 5.5), (0, 0, 2.4), 3400, (0.45, 0.6, 1.0), 3.0)
area('rim warm', (-6.0, 5.0, 3.5), (0, 0, 2.2), 2400, (1.0, 0.45, 0.2), 2.5)
area('fill', (2.0, -9.0, 2.5), (0, 0, 2.5), 120, (0.6, 0.7, 1.0), 6.0)
area('top', (-0.4, -1.6, 7.0), (0, -0.3, 3.4), 260, (1.0, 0.9, 0.82), 1.2)          # raking top light: brow, nose, cheekbones
if A.bg != 'dk2':
    area('torch', (-3.0, -3.5, 3.5), (0, 0, 2.5), 900, (1.0, 0.5, 0.18), 0.6)
# eye glow spill (small, warm)
for i in (0, 1):
    e = load(f'eye{i}'); c = to_z(e['V']).mean(0)
    L = bpy.data.lights.new(f'eye glow {i}', 'POINT'); L.energy = 0.3; L.color = (1.0, 0.55, 0.12); L.shadow_soft_size = 0.02
    o = bpy.data.objects.new(f'eye glow {i}', L); o.location = (c[0], c[1] - 0.05, c[2]); S.collection.objects.link(o)

# ---------------------------------------------------------------- camera
import mathutils
cam_d = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cam_d); S.collection.objects.link(cam); S.camera = cam
SHOTS = {   # azimuth (deg, 0 = in front), elevation (deg), distance, target (x, height), lens (mm), f-stop
    'face': (-20, 2, 3.8, (0.0, 3.56), 105, 2.8),
    'face34': (38, 3, 3.0, (0.0, 3.56), 105, 2.8),
    'body': (22, 4, 13.0, (0.05, 2.05), 70, 8.0),
    'hero': (30, -2, 13.5, (0.1, 2.1), 50, 5.6),
    'front': (0, 4, 13.0, (0.0, 2.05), 70, 11.0), 'left': (90, 4, 13.0, (0.0, 2.05), 70, 11.0),
    'back': (180, 4, 13.0, (0.0, 2.05), 70, 11.0), 'right': (270, 4, 13.0, (0.0, 2.05), 70, 11.0),
}
az, el, dist, (tx, th), lens, fstop = SHOTS[A.shot]
a, e = math.radians(az), math.radians(el)
tgt = mathutils.Vector((-tx, -0.45 if A.shot.startswith('face') else 0.0, th))   # face shots aim at the face (front is -y)
loc = tgt + mathutils.Vector((-dist * math.sin(a) * math.cos(e) * -1, -dist * math.cos(a) * math.cos(e), dist * math.sin(e)))
if A.cam:                                  # free camera in game coords: px,py,pz,tx,ty,tz,lens
    c = [float(v) for v in A.cam.split(',')]
    loc = mathutils.Vector(to_z([c[:3]])[0]); tgt = mathutils.Vector(to_z([c[3:6]])[0]); lens = c[6]; fstop = 4.0
cam.location = loc
cam.rotation_mode = 'QUATERNION'; cam.rotation_quaternion = (tgt - loc).to_track_quat('-Z', 'Y')
cam_d.lens = lens; cam_d.clip_start = 0.05
cam_d.dof.use_dof = True; cam_d.dof.focus_distance = (tgt - loc).length; cam_d.dof.aperture_fstop = fstop

# ---------------------------------------------------------------- compositor: glow on the eyes
S.use_nodes = True
ct = S.node_tree; ct.nodes.clear()
rl = ct.nodes.new('CompositorNodeRLayers'); comp = ct.nodes.new('CompositorNodeComposite')
gl = ct.nodes.new('CompositorNodeGlare'); gl.glare_type = 'FOG_GLOW'; gl.threshold = 1.0; gl.size = 8; gl.mix = -0.35
ct.links.new(rl.outputs['Image'], gl.inputs['Image']); ct.links.new(gl.outputs['Image'], comp.inputs['Image'])
if not A.shot.startswith('face'):
    S.view_settings.exposure = 0.35          # full-body shots: lift the body out of the dark
S.render.filepath = A.out
S.render.image_settings.file_format = 'PNG'
if A.buildonly:
    raise SystemExit('__BUILD_ONLY__')
if A.blend:
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=A.blend, compress=True)
import time
t0 = time.time()
bpy.ops.render.render(write_still=True)
print('RENDER_DONE', A.shot, A.expr, round(time.time() - t0, 1), 's')
