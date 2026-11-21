# Reaper Proxy and Clip Prep (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the clip frames and produce a proportion-accurate proxy Reaper in Blender with turntable renders for the user's review.

**Architecture:** Everything verifiable without Blender (clip prep, reference splitting, proportion measurement, render checks) is plain Python tested with pytest here. Blender work is scripts run on the user's PC with `blender --background --python`, because PyPI and `bpy` are blocked in this workspace (HTTP 403 verified). The user sends the renders back and the plain-Python checks verify them here.

**Tech Stack:** Python 3.13 with Pillow and numpy (installed), pytest 9, ffmpeg, Blender 4.x (user's PC, free download).

**Spec:** `docs/superpowers/specs/2026-11-02-dk2-remaster-design.md`

## Global Constraints

- Source clip: `/mnt/user-data/attachments/76eb4112-412f-58e4-a30a-b275f992be89/c0cefca4-Dungeon_Keeper_Remastered_Video.mp4`, 1920x1080, 30 fps, 906 frames, 30.2 s. Verified crop of the picture area: `1440:1080:240:0` (w:h:x:y).
- User hardware is a GTX 1070 (8 GB VRAM). Nothing in this stage needs the GPU beyond Blender's Eevee or Workbench viewport render.
- Claude's workspace cannot install or run `bpy`. Blender scripts must run on the user's machine.
- The fan-made War for the Overworld Reaper image is a proportion cross-check only, never a base asset.
- The working directory `/home/claude` is not a git repo, so this plan has no commit steps. Each task ends with a saved, verified file instead.
- Reference files (copy into `refs/`): turnaround `b5857b2a-image.png` (1024x547, front, side, back, side) as `refs/turnaround.png`; promo render `75f85ed2-image.png` as `refs/promo.png`; figurine `48ff1f0f-image.png` as `refs/figurine.png`; fan WIP `aae6a58e-image.jpg` as `refs/fan_wip.jpg`. All sit in `/mnt/user-data/attachments/76eb4112-412f-58e4-a30a-b275f992be89/`.

## Review Focus

- A clip with no black bars: crop detection returns the full frame, not an error.
- A clip whose frame count differs from 906 (dropped or duplicated frames): extraction reports the actual count, not an assumed one.
- A turnaround view whose figure is offset or at a different scale: proportions are measured relative to each view's own silhouette.
- Blender not installed or the wrong version: scripts exit with a clear message, not a stack trace.
- A render that is blank or the wrong size: the render check fails loudly.

## File Structure

- `scripts/prep_clip.py`: crop detection and frame extraction.
- `scripts/refs.py`: silhouette bounding box and turnaround splitting.
- `scripts/measure_proportions.py`: ratios from the silhouettes to `data/reaper_proportions.json`.
- `blender/build_reaper.py`: builds the proxy from the JSON (runs in Blender).
- `blender/render_turntable.py`: five-view renders (runs in Blender).
- `scripts/check_render.py`: sanity and silhouette-overlap checks on returned renders.
- `tests/`: pytest files, one per script.

### Task 1: Clip prep

**Files:**
- Create: `scripts/prep_clip.py`, `tests/test_prep_clip.py`

**Interfaces:**
- Produces: `detect_crop(video: Path) -> tuple[int, int, int, int]` returning `(w, h, x, y)`; `extract_frames(video: Path, crop: tuple[int, int, int, int], start_s: float, end_s: float, out_dir: Path) -> list[Path]` returning the PNGs in order.

- [ ] **Step 1: Write failing tests.** `test_detect_crop_pillarbox`: generate with `ffmpeg -f lavfi` a 640x360 video with a 480x360 coloured box centred on black; assert `(480, 360, 80, 0)`. `test_detect_crop_no_bars`: a full-frame test video returns `(640, 360, 0, 0)`. `test_extract_frames_count_and_size`: one second at 30 fps returns 30 PNGs of 480x360. `test_real_clip_crop` (skipped if the clip is absent): `(1440, 1080, 240, 0)`.
- [ ] **Step 2: Run** `pytest tests/test_prep_clip.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement both functions in `scripts/prep_clip.py`** using ffmpeg `cropdetect` (take the most frequent result) and `-vf crop=...`; return the actual files written, not an assumed count.
- [ ] **Step 4: Run** the same command. Expected: 4 passed.
- [ ] **Step 5: Extract the three pilot ranges** (about 14.5 to 15.5 s, 17.5 to 18.5 s, 26.5 to 27.5 s) to `frames/pilot_*/` and read one frame from each to confirm they show the furnace, the Reaper in flames and the old wizard. Adjust the ranges if not.

### Task 2: Reference splitting

**Files:**
- Create: `scripts/refs.py`, `tests/test_refs.py`, `refs/` (copies listed in Global Constraints)

**Interfaces:**
- Produces: `silhouette_bbox(img: Image.Image, bg_tol: int = 18) -> tuple[int, int, int, int]` returning `(x0, y0, x1, y1)` of pixels differing from the border colour; `split_turnaround(path: Path, out_dir: Path) -> dict[str, Path]` with keys `front`, `side_left`, `back`, `side_right`.

- [ ] **Step 1: Write failing tests.** `test_bbox_synthetic`: a grey image with a white 20x40 rectangle returns exactly that box. `test_split_turnaround_views`: `refs/turnaround.png` yields four images, each about 256 px wide, each with a silhouette whose height is within 3% of the others.
- [ ] **Step 2: Run** `pytest tests/test_refs.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement both functions** by splitting the sheet into four equal vertical strips and trimming each to its silhouette box.
- [ ] **Step 4: Run.** Expected: 2 passed. Save the four views in `refs/views/`.

### Task 3: Proportion measurement

**Files:**
- Create: `scripts/measure_proportions.py`, `tests/test_proportions.py`, `data/reaper_proportions.json`

**Interfaces:**
- Consumes: `silhouette_bbox` from Task 2.
- Produces: `measure_proportions(front: Image.Image, side: Image.Image) -> dict[str, float]` with keys `height`, `shoulder_width`, `horn_span`, `head_height`, `torso_height`, `leg_length`, `arm_length`, `depth`, each as a ratio of total silhouette height (so `height == 1.0`).

- [ ] **Step 1: Write failing tests.** `test_ratios_keys_and_range`: all eight keys present, `height == 1.0`, every other ratio between 0.05 and 1.2. `test_front_side_height_agree`: front and side silhouette heights differ by under 3%.
- [ ] **Step 2: Run** `pytest tests/test_proportions.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `measure_proportions`.** Take widths from row-wise silhouette extents on the front view and depth from the side view. Write the result with a `__main__` entry to `data/reaper_proportions.json`.
- [ ] **Step 4: Run.** Expected: 2 passed. Open the JSON and compare each ratio against the turnaround by eye.

### Task 4: Blender proxy build

**Files:**
- Create: `blender/build_reaper.py`, `tests/blender/test_reaper_build.py`, `tests/test_build_guard.py`

**Interfaces:**
- Consumes: `data/reaper_proportions.json` (keys from Task 3).
- Produces: `build_reaper(proportions: dict[str, float], height_m: float = 2.4) -> bpy.types.Collection` named `Reaper`; objects named `Body`, `Head`, `Horn_L`, `Horn_R`, `Pauldron_L`, `Pauldron_R`, `Arm_L`, `Arm_R`, `Leg_L`, `Leg_R`, `Hoof_L`, `Hoof_R`, `Scythe`. Saves `build/reaper_proxy.blend` and `build/reaper_proxy.glb`.

- [ ] **Step 1: Write the tests.** In Blender (`tests/blender/test_reaper_build.py`): `test_objects_exist` (all 13 names present), `test_height_ratio` (bounding-box width and depth divided by height each within 5% of `shoulder_width` and `depth` in the JSON), `test_no_empty_meshes` (every object has at least 8 vertices). In plain Python (`tests/test_build_guard.py`): `test_exits_2_outside_blender` runs `python3 blender/build_reaper.py` and asserts exit code 2 with "Blender" in stderr.
- [ ] **Step 2: Run both.** Here: `pytest tests/test_build_guard.py -v`. On the user's PC: `blender --background --python tests/blender/test_reaper_build.py`. Expected: both FAIL (builder missing).
- [ ] **Step 3: Implement `build_reaper`** from primitives with subdivision modifiers, sized from the ratios; print a clear message and exit with code 2 if run outside Blender. No rig yet: rigging belongs to the Stage 2 plan, where the pose to match is known.
- [ ] **Step 4: Run both commands again.** Expected: guard test passes here; 3 passed in Blender, and both files saved.

### Task 5: Turntable renders

**Files:**
- Create: `blender/render_turntable.py`, `scripts/check_render.py`, `tests/test_check_render.py`

**Interfaces:**
- Produces: `render_turntable(blend: Path, out_dir: Path, size: int = 1024) -> list[Path]` writing `front.png`, `side.png`, `back.png`, `three_quarter.png`, `head.png`; `check_render(path: Path, size: int = 1024) -> None` raising `ValueError` if the image is the wrong size or its pixel standard deviation is under 4; `silhouette_iou(render: Image.Image, reference: Image.Image) -> float`.

- [ ] **Step 1: Write failing tests** for `check_render` (blank image raises, wrong size raises, a normal image passes) and for `silhouette_iou` (identical images return 1.0, disjoint ones return 0.0).
- [ ] **Step 2: Run** `pytest tests/test_check_render.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `check_render` and `silhouette_iou`** in `scripts/check_render.py`, then `render_turntable` in Blender using the Workbench engine with a neutral grey background so the silhouette check works.
- [ ] **Step 4: Run** the pytest command (expected: 5 passed). Then on the user's PC: `blender --background build/reaper_proxy.blend --python blender/render_turntable.py`, and the user sends the five PNGs back.
- [ ] **Step 5: Verify returned renders here:** run `check_render` on all five, then `silhouette_iou` for `front` and `side` against `refs/views/`. Expected: all pass the sanity check, and IoU is at least 0.80 for both. If under, adjust `build_reaper.py` and repeat Steps 4 and 5.
- [ ] **Step 6: Review gate.** Send the user the renders next to the promo and figurine references. Stage 2 (pilot shot scaffolds) starts only after the user approves the model.
