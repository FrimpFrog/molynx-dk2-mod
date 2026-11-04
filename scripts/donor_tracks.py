"""Per-animation vertex tracks of one DK2 creature in a single vertex numbering. LICENCE: GPL-3.0.

Animation files of a creature number their vertices differently, so every file is matched back to the
creature's idle1 pose through its triangles' corner UVs (custom_creature.corner_map).
"""
import numpy as np

from scripts.custom_creature import corner_map
from scripts.viewer_creatures import CREATURES, find_animations
from scripts.viewer_export import _collect


def donor_tracks(meshes_dir, cache, creature="giant", canon_anim="idle1"):
    """-> {anim: (frames (F,N,3) with NaN for vertices the file lacks, wrap)}, in the creature's own
    (left-handed DK2, Y-up) coordinates."""
    files, _ = find_animations(meshes_dir, {creature: CREATURES[creature]})
    fl = files[creature]
    miss: set = set()
    if canon_anim not in fl:
        canon_anim = "walk" if "walk" in fl else sorted(fl)[0]
    canon = _collect(fl[canon_anim], cache, miss)
    tex = [p["texture"] for p in canon["parts"]]
    offs = np.cumsum([0] + [len(p["uv"]) for p in canon["parts"]])
    out = {}
    for lab, path in fl.items():
        a = _collect(path, cache, miss)
        if not set(tex) <= {q["texture"] for q in a["parts"]}:
            continue                                     # the file swaps the model (electrocute)
        arr = np.full((len(a["frames"]), offs[-1], 3), np.nan)
        for pi, part in enumerate(a["parts"]):
            if part["texture"] not in tex:
                continue
            ci = tex.index(part["texture"])
            cm = corner_map(canon["parts"][ci], part)
            ok = cm >= 0
            for f in range(len(a["frames"])):
                arr[f, offs[ci] + cm[ok]] = a["frames"][f][pi][ok]
        out[lab] = (arr, a["wrap"])
    return out
