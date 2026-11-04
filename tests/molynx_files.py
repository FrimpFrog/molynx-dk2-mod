"""Literal Molynx file data shared by the record, splice and mod tests (pinned values, so the
record tests do not need the game's meshes or a slow export).

FILE_WRAPS: the loop flag of every file the Bile Demons' Molynx slots replace (None: a static
mesh), as molynx_assets.bile_file_wraps reads them from Meshes.WAD (test_molynx_assets checks
the files themselves). ANIM_INFO: the final `info` of the default export (test_molynx_assets pins
the same frame counts on the written files).
"""
FILE_WRAPS = {
    "bile_demon-walk": True, "bil-dragged_pose": None, "bile_demon-recoil(hff)": False,
    "bile_demon-melee": False, "bile_demon-magic": False, "bile_demon-die": False,
    "bile_demon-happy": False, "bile_demon-angry": False, "bile_demon-stunned_pose": None,
    "bile_demon-swing": True, "biledemon-bed(sleeping)": True, "bile_demon-eat": False,
    "bil-torture": False, "bile_demon-drink": False, "biledemon-idle1": False,
    "bile_demon-recoil(hfb)": False, "bile_demon-building": True, "bile_demon-pray": True,
    "bile_demon-fallback": False, "bil-elec": None, "bil-electrocute": False,
    "bile_demon-getup": False, "bil-dance": False, "bil-entrance": False,
    "biledemon-idle2": False, "bil-drunkidle": False, "bil-walkback": True, "bd-poseframe": None,
    "bile_demon-diepose": None, "bile_demon-dejected_pose": None,
}
ANIM_INFO = {stem: {"frames": frames, "fps": 30, "wrap": wrap} for stem, frames, wrap in [
    ("stand", 61, False), ("walk", 47, True), ("walk_once", 47, False), ("attack", 41, False),
    ("die", 31, False), ("fidgeting", 63, False), ("running", 31, True), ("eating", 83, False),
    ("praying", 121, True), ("happy", 59, False), ("angry", 23, False), ("celebrating", 94, False),
    ("victory_celebration", 246, False), ("getting_hit", 17, False),
    ("getting_hit_loop", 18, True), ("being_slapped", 35, False), ("staggering", 96, True),
    ("waking_up", 46, False), ("knockback", 61, False), ("sleeping", 137, True),
    ("training", 121, True), ("fighting_another_creature", 133, True), ("patrolling", 42, True),
    ("guarding", 24, True), ("manufacturing", 34, True), ("interacting_with_objects", 18, True),
    ("being_picked_up", 61, True),
]}
