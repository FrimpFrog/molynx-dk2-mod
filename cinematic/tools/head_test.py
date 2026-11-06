import sys, time, numpy as np
sys.path.insert(0, '/home/claude/molynx_cinematic/tools')
from head_build import build_head
from glb import GLB
from hd_build import img_bytes
MIRROR = np.array([-1.0, 1, 1])
expr = sys.argv[1]; out = sys.argv[2]
t = time.time(); ms = build_head(expr); print('built', round(time.time() - t, 1), [(m['name'], len(m['F'])) for m in ms])
g = GLB()
skin = g.material('head', roughness=0.72, extras={'vertexColors': True, 'clearcoat': 0.12, 'clearcoatRoughness': 0.6})
teeth = g.material('teeth', roughness=0.35, extras={'vertexColors': True, 'clearcoat': 0.5, 'clearcoatRoughness': 0.3})
col_t, em_t = [m for m in ms if 'tex' in m][0]['tex']
te = g.material('eye', g.image(*img_bytes(col_t)), roughness=0.12, emissive_tex=g.image(*img_bytes(em_t)), emissive=(1, 1, 1), extras={'clearcoat': 1.0, 'clearcoatRoughness': 0.05, 'emissiveIntensity': 2.2})
for m in ms:
    P = m['P'] * MIRROR; F = m['F'][:, ::-1]; N = m['N'] * MIRROR
    mat = {'skin': skin, 'teeth': teeth, 'eye': te}[m['kind']]
    g.mesh(m['name'], P, N, F, mat, UV=m.get('UV'), color=m.get('color'))
print('bytes', g.save(out))
