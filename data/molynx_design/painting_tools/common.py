"""Shared bits for the Molynx design work (reads the repo, never writes to it)."""
import sys, pickle
from pathlib import Path
import numpy as np
REPO = Path("/home/claude/molynx")
sys.path.insert(0, str(REPO))
UNIT = REPO / "megaglest_behemoth"
PKL = Path("/home/claude/dk2data/char_full.pkl")

def paint_inputs():
    from scripts.g3d import parse_g3d
    from scripts.behemoth_rig import behemoth_rest
    from scripts.character_texture import build_atlas
    model = parse_g3d((UNIT / "models/behemoth_standing.g3d").read_bytes())
    uv = np.concatenate([me.uv for me in model.meshes]).astype(float)
    uv[:, 1] = 1 - uv[:, 1]
    rest, idx, _ = behemoth_rest(UNIT)
    body = build_atlas(UNIT / "models/texture_daemon_giant.png").crop((0, 0, 256, 256))
    return body, uv, idx

def load_char(fixed=True):
    """The saved character (never modified on disk). fixed=True applies the design fixes (the free hand's claw)."""
    data, tex = pickle.load(open(PKL, "rb"))
    extra = Path("/home/claude/molynx_design/work/candidate_clips.pkl")
    if extra.exists():                                   # design clips retargeted from other DK2 creatures (new_clips.py)
        for k, c in pickle.load(open(extra, "rb")).items():
            data[k] = {**c, "parts": data["stand"]["parts"]}
    if fixed:
        from scripts.behemoth_rig import behemoth_rest
        from claw_fix import fix_claw, fix_club_arm, lock_club_arm
        from scripts.behemoth_rig import behemoth_rig
        rest, _, _ = behemoth_rest(UNIT)
        shoulder = np.asarray(behemoth_rig(UNIT).joints['armN'], float)
        rig = behemoth_rig(UNIT)
        data = fix_club_arm(fix_claw(data, rest), rest, shoulder)
        from claw_fix import LOCKED
        locked = set(LOCKED) | {k for k in data if k.startswith(('dance_', 'fid_reaper'))}
        data = lock_club_arm(data, np.asarray(rig.weights), list(rig.bones), shoulder, locked)
    return data, tex
