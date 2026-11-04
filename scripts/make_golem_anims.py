"""Build the retargeted Golem animation library. LICENCE: GPL-3.0."""
import numpy as np

from scripts.donor_tracks import donor_tracks
from scripts.g3d import parse_g3d
from scripts.golem_retarget import retarget

FLIP = np.array([-1.0, 1.0, 1.0])       # DK2 data is left-handed; the Golem (MegaGlest) is right-handed


def golem_rest(unit_dir):
    m = parse_g3d((unit_dir / "models/golem_standing.g3d").read_bytes())
    me = m.meshes[0]
    return me.vertices[0].astype(float), me.indices.reshape(-1, 3)


def raw_library(meshes_dir, cache, unit_dir, creature="giant", rest_anim="walk", params=None):
    """Every donor animation retargeted with default settings -> {label: frames (F,M,3)}."""
    tracks = donor_tracks(meshes_dir, cache, creature)
    g0, idx = golem_rest(unit_dir)
    rest = tracks[rest_anim][0][0] * FLIP
    return {lab: retarget(arr * FLIP, rest, g0, idx, **(params or {})) for lab, (arr, wrap) in tracks.items()}, \
        {lab: wrap for lab, (arr, wrap) in tracks.items()}
