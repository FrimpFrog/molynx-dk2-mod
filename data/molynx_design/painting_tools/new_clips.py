"""Design: new clips for Molynx from other real DK2 creatures, retargeted exactly like his others (scripts/behemoth_anims.py)."""
import sys, pickle, warnings; sys.path.insert(0, '/home/claude/molynx_design/tools')
from common import *
from scripts.behemoth_rig import behemoth_rig
from scripts.donor_tracks import donor_tracks
from scripts.humanoid_retarget import retarget
from scripts.golem_anims import _loop_close, _resample
from scripts.behemoth_anims import FLIP, MIRROR, DK2_FPS, CLUB_ARM, CLUB_LOCKED, LEGS_NARROW
from scripts.dk2paths import MESHES_DIR, GAME_DIR
from scripts.dk2texture import TextureCache

_tracks = {}
def rt(creature, anim, mirror=False, **kw):
    rig = behemoth_rig(UNIT)
    if creature not in _tracks:
        cache = TextureCache(GAME_DIR / 'EngineTextures.dir', GAME_DIR / 'EngineTextures.dat')
        _tracks[creature] = donor_tracks(MESHES_DIR, cache, creature)
    T = _tracks[creature]
    r = 'idle1' if 'idle1' in T else ('walk' if 'walk' in T else sorted(T)[0])
    kw = {**kw, 'amp': {**CLUB_ARM, **kw.get('amp', {})}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = retarget(rig, T[anim][0] * (MIRROR if mirror else FLIP), T[r][0][0] * (MIRROR if mirror else FLIP), **kw)
    return out, len(T[anim][0]) / DK2_FPS

def clip(creature, anim, loop=True, **kw):
    fr, secs = rt(creature, anim, **kw)
    fr = np.asarray(fr)
    if len(fr) / secs < 24 or len(fr) / secs > 36:
        fr = _resample(fr, int(secs * 30))
    if loop:
        fr = _loop_close(fr)
    return {'frames': [np.asarray(f) for f in fr], 'wrap': loop, 'fps': round(len(fr) / secs, 3)}

def dress(clips):
    """Turn body-only frames into the character's full frames (body + mace + head parts) like build_character does."""
    from scripts.custom_character import build_character
    data, _, _ = build_character(UNIT, clips=('stand',), extra=clips)
    return {k: data[k] for k in clips}
