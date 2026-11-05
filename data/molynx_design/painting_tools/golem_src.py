import sys; sys.path.insert(0,'/home/claude/molynx_design/tools')
from common import *
from scripts.g3d import parse_g3d
from PIL import Image
G = REPO / 'megaglest_golem/models'
def golem():
    m = parse_g3d((G / 'golem_standing.g3d').read_bytes())
    me = m.meshes[0]
    v = np.asarray(me.vertices[0], float)
    uv = np.asarray(me.uv, float).copy(); uv[:, 1] = 1 - uv[:, 1]
    tri = np.asarray(me.indices, int).reshape(-1, 3)
    tex = Image.open(G / 'texture_golem.png').convert('RGB')
    return v, tri, uv, tex
def as_clip(v, tri, uv, name='g'):
    return {'frames': [[v]], 'parts': [{'texture': name, 'uv': uv, 'indices': tri.reshape(-1)}], 'wrap': False, 'fps': 1}
