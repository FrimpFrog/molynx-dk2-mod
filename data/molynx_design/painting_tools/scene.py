"""Scenes of the rocky Molynx (design v8, version A) with real DK2 creatures, all in game space (y up, KMF units)."""
import sys; sys.path.insert(0, '/home/claude/molynx_design/tools')
from pathlib import Path
import numpy as np
from PIL import Image
from common import REPO, load_char
from scripts.dk2kmf import parse_kmf, evaluate_anim_frame, mesh_points
from scripts.dk2texture import TextureCache
from scripts.dk2paths import GAME_DIR, MESHES_DIR

W = Path('/home/claude/molynx_design/work')
SKIN = 'A2'                              # 'A' = first rock version, 'A2' = graded brighter + gloss to match DK2
FLAME = Path('/home/claude/dk2data/up/flame/flame/resources')
LOOSE = [FLAME / 'Quuz_FullHD/Data/EngineTextures', FLAME / 'Quuz_HD/Data/EngineTextures']   # his alt-resources order
MOLYNX_SCALE = 0.14980878818609805 * (0.715 / 0.6168) * (0.71 / 0.685)   # design: top of his head level with the Horned Reaper skull (0.71)
GAME_FPS = 30.0

_cache = None
def cache():
    global _cache
    if _cache is None:
        _cache = TextureCache(GAME_DIR / 'EngineTextures.dir', GAME_DIR / 'EngineTextures.dat')
    return _cache

_loose_index = None
def texture(base: str) -> Image.Image:
    """A material's texture as the user's game shows it: his HD packs first (loose PNG), then the game's cache."""
    global _loose_index
    if _loose_index is None:
        _loose_index = {}
        for d in reversed(LOOSE):
            for p in d.rglob('*.png'):
                _loose_index[p.stem.lower()] = p
    p = _loose_index.get(base.lower())
    if p is not None:
        return Image.open(p).convert('RGB')
    names = {n.lower(): n for n in cache().names}
    return cache().get(names[f'{base}MM0'.lower()]).convert('RGB')

def yup(p):
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)

class Actor:
    """parts: [{'texture','uv','indices'}]; frames(t) -> list of vertex arrays at time t (seconds)."""
    def at(self, t, place):
        yaw, off = place
        a = np.radians(yaw)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        return [v @ R.T + np.asarray(off) for v in self.frame(t)]

class KmfActor(Actor):
    def __init__(self, stem, textures, loop=True):
        f = {p.stem.lower(): p for p in MESHES_DIR.glob('*.kmf')}[stem.lower()]
        self.m = parse_kmf(f.read_bytes()); self.loop = loop; self.parts = []
        sprites = self.m.anim.sprites if self.m.kind == 'anim' else self.m.mesh.sprites
        for k, s in enumerate(sprites):
            base = self.m.materials[s.material].textures[0]
            key = f'dk2/{base}'
            if key not in textures:
                textures[key] = texture(base)
            uv = np.array([(v.uv[0] / 32768.0, v.uv[1] / 32768.0) for v in s.vertices])
            self.parts.append({'texture': key, 'uv': uv, 'indices': np.array(s.triangles[0], int).reshape(-1), 'k': k})
        self.n = self.m.anim.frame_count if self.m.kind == 'anim' else 1
        self.duration = self.n / GAME_FPS
    def frame(self, t):
        if self.m.kind == 'mesh':
            pts = mesh_points(self.m)
        else:
            f = int(t * GAME_FPS)
            f = f % (self.n - (1 if self.m.anim.wrap else 0)) if self.loop else min(f, self.n - 1)
            pts = evaluate_anim_frame(self.m.anim, f)
        return [yup(np.asarray(pts[p['k']], float)) for p in self.parts]

class MolynxActor(Actor):
    """The design character (v8 rock) in game space: x mirrored like his export, scaled, feet on the ground."""
    _built = None
    def __init__(self, clip, textures, loop=True):
        if MolynxActor._built is None:
            from head_swap_v8 import build_v8, HEAD_TEX, SKIRT_TEX
            data, tex = load_char()
            new, _ = build_v8(data)
            T = dict(tex)
            T['custom_character/body'] = Image.open(W / f'{SKIN}_body.png').convert('RGB')
            T[HEAD_TEX] = Image.open(W / f'{SKIN}_head.png').convert('RGB'); T[SKIRT_TEX] = Image.open(W / f'{SKIN}_strip.png').convert('RGB')
            ref = np.concatenate(new['stand']['frames'][0][:1])
            off = np.array([ref[:, 0].mean(), ref[:, 1].min(), ref[:, 2].mean()])
            MolynxActor._built = (new, T, off)
        new, T, off = MolynxActor._built
        textures.update({k: v for k, v in T.items() if k not in textures})
        self.c = new[clip]; self.loop = loop; self.off = off
        self.parts = [{**p, 'texture': p['texture']} for p in self.c['parts']]
        self.n = len(self.c['frames']); self.fps = self.c['fps']
        self.duration = (self.n - (1 if self.c['wrap'] else 0)) / self.fps
    def frame(self, t):
        f = int(t * self.fps)
        f = f % (self.n - (1 if self.c['wrap'] else 0)) if self.loop else min(f, self.n - 1)
        out = []
        for v in self.c['frames'][f]:
            p = (np.asarray(v, float) - self.off) * MOLYNX_SCALE
            p[:, 0] *= -1
            out.append(p)
        return out

def scene_clip(actors_places, t):
    """One render_hq 'clip' (single frame) holding every actor at time t."""
    parts, verts = [], []
    for actor, place in actors_places:
        parts += actor.parts
        verts += actor.at(t, place)
    return {'parts': parts, 'frames': [verts], 'wrap': False, 'fps': 1}

def height(actor, t=0.0):
    return float(np.concatenate(actor.frame(t))[:, 1].max())
