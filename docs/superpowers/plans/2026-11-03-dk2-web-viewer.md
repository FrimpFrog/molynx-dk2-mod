# DK2 Creature Web Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A private web page where the user picks a DK2 creature, plays its animations, orbits it, and toggles Original / HD textures.

**Architecture:** A Python exporter evaluates every animation frame with the existing `evaluate_anim_frame` and writes one compact `.dk2v` file per creature (JSON header plus a binary blob of 16-bit positions). A texture exporter and an HD pack builder fill two PNG folders with identical names. A three.js page loads a `.dk2v`, blends between stored frames, and picks the texture folder from the toggle.

**Tech Stack:** Python 3 + numpy + Pillow + pytest (existing), three.js (vendored, from npm), Node (parser test), Playwright (already installed) for browser checks.

**Spec:** `/home/claude/docs/superpowers/specs/2026-11-03-dk2-web-viewer-design.md`

## Global Constraints

- /home/claude is not a git repo: no commit steps; record each finished task in `/home/claude/.superpowers/sdd/2026-11-03-dk2-web-viewer/progress.md`.
- Run tests as `python3 -m pytest tests -q` from /home/claude (plain `pytest` also picks up `deliverables/` copies).
- Positions are 16-bit integers scaled to each creature's bounding box; LOD 0 only; playback 30 fps (unconfirmed against the game).
- HD pack is exactly 4x the original size, same file names, labelled "Enhanced (non-AI)".
- Page is a private artifact; assets are EA's, for personal use. The texture decoder is GPL-3.0, so the exporter that imports it is GPL-3.0 too (state in README).
- Tests that read game files skip themselves when the files are absent (same pattern as `tests/test_kmf_textured.py`).
- Game-file paths: `S = /tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad`; meshes in `S/extracted/Meshes`, textures at `S/dk2game/Ai/EngineTextures.dir|.dat`.

## Review Focus

- An animation whose part list differs from the creature's pose (Bile_Demon-Drink and -Eat have 5 parts, base has 3): each animation must be self-contained and render, not be dropped or crash. Owned by Task 2.
- Frames at index ≥128 and the final frame (Goblin dance has 176): exported positions must match `evaluate_anim_frame`. Owned by Task 2.
- A material whose texture name is not in the cache, or a part with no LOD 0 triangles: export reports it and the viewer draws that part grey, not a crash. Owned by Tasks 2 and 4.
- HD file missing for a texture: viewer falls back to the original, and the toggle still reports HD. Owned by Task 4.
- Animation files that belong to no listed creature, or differ only by case (`GOB-dance`, `Bil-Dance`): reported in a list, never silently dropped or attached to the wrong creature. Owned by Task 1.

---

### Task 1: Creature and animation mapping

**Files:**
- Create: `scripts/viewer_creatures.py`
- Test: `tests/test_viewer_creatures.py`

**Interfaces:**
- Produces: `CREATURES: dict[str, CreatureDef]` with `CreatureDef(label: str, pose: str, prefixes: tuple[str, ...])`; `find_animations(meshes_dir: Path) -> tuple[dict[str, dict[str, Path]], list[str]]` returning `{creature_id: {animation_label: kmf_path}}` and the list of unattributed file stems.

- [ ] **Step 1: Write failing tests** `test_goblin_has_walk_and_dance`, `test_bile_demon_has_walk_dance_melee`, `test_reaper_has_walk_roar_melee1` (labels are the file stem minus the creature prefix, lower-cased, e.g. `walk`, `dance`; assert those keys and that paths exist), `test_pose_files_are_not_animations` (no key contains `poseframe`), `test_case_variants_attach_to_one_creature` (`Bil-Dance` and `BIL-Walkback` are both under `bile_demon`), `test_unattributed_files_are_reported` (the returned list is non-empty and contains `Arc-Walk` while the three creatures are the only ones defined).
- [ ] **Step 2:** Run `python3 -m pytest tests/test_viewer_creatures.py -q`; expect FAIL (module missing).
- [ ] **Step 3: Implement** in `scripts/viewer_creatures.py`. Initial table: `goblin` (pose `Gob-PoseFrame`, prefixes `goblin-`, `gob-`), `bile_demon` (`BD-PoseFrame`; `bile_demon-`, `biledemon-`, `bil-`, `bile_demon-`), `horned_reaper` (`Rea3-poseframe`; `rea3_`, `rea3-`). Match case-insensitively on file stem; longest prefix wins; skip stems containing `pose`. Only `kind == "anim"` files count (check via `parse_kmf`, or the `model_index.csv` kind column).
- [ ] **Step 4:** Run the tests; expect PASS. Record in ledger.

### Task 2: `.dk2v` exporter

**Files:**
- Create: `scripts/viewer_export.py`
- Test: `tests/test_viewer_export.py`

**Interfaces:**
- Consumes: `find_animations`, `CREATURES` (Task 1); `parse_kmf`, `evaluate_anim_frame` (`scripts/dk2kmf.py`); `material_texture_name` (`scripts/kmf_textured.py`); `TextureCache` (`scripts/dk2texture.py`).
- Produces: `export_creature(creature_id: str, meshes_dir: Path, cache: TextureCache, out_dir: Path) -> Path` writing `<out_dir>/<creature_id>.dk2v`; `read_dk2v(path: Path) -> dict` returning `{"meta": dict, "animations": {label: {"parts": [...], "frames": list[list[np.ndarray]]}}}` with positions de-quantised; `write_index(out_dir: Path, paths: list[Path]) -> Path` writing `index.json`.
- File layout (fixed): bytes `DK2V`, `u32` JSON length, JSON header, binary blob. JSON holds `label`, `fps` (30), `bbox` (`min`, `max`), `animations[label] = {frame_count, wrap, parts[{texture, vertex_count, uv_offset, index_offset, index_count, position_offset}]}`, and `missing_textures`. Positions are `uint16` per component, little-endian, frame-major per part; UVs `float32`; indices `uint16`.

- [ ] **Step 1: Write failing tests** `test_roundtrip_matches_evaluate_anim_frame` (Goblin-Walk, frames 0, 10, last; max abs error ≤ bbox span / 65535 × 1.5), `test_frames_past_128` (Goblin dance, frames 127, 128, 150, last), `test_animation_with_extra_parts_is_self_contained` (Bile_Demon-Drink has 5 parts; exported part count equals the file's non-empty LOD0 sprites), `test_missing_texture_is_reported_not_fatal` (cache stub whose `names` lacks one texture → name appears in `missing_textures`, export still writes), `test_parts_without_lod0_triangles_are_skipped`, `test_index_lists_creatures_and_animations`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3: Implement** the layout above. Use one shared bbox per creature across all frames of all animations for quantisation. Evaluate pose-less: each animation is exported from its own file only.
- [ ] **Step 4:** Run `python3 -m pytest tests/test_viewer_export.py -q`; expect PASS. Then export the three creatures to `S/viewer/data/` and print file sizes (record in the ledger; size feeds Task 6).

### Task 3: Texture sets and HD pack

**Files:**
- Create: `scripts/viewer_textures.py`
- Test: `tests/test_viewer_textures.py`

**Interfaces:**
- Consumes: `read_dk2v` (Task 2) for the texture names a creature needs; `TextureCache`.
- Produces: `export_textures(cache: TextureCache, names: set[str], out_dir: Path) -> list[Path]` (PNG, file name `<texture name>.png`); `build_hd_pack(original_dir: Path, hd_dir: Path, scale: int = 4) -> list[Path]` (Lanczos then mild unsharp mask, alpha preserved); `texture_names(paths: list[Path]) -> set[str]`.

- [ ] **Step 1: Write failing tests** `test_original_pngs_written_for_every_needed_texture` (Goblin → `GoblinFrontMM0.png`, `GoblinBackMM0.png`), `test_hd_is_exactly_4x_with_same_names`, `test_hd_differs_from_plain_resize` (sharpening applied: mean abs difference > 0), `test_alpha_is_preserved` (use a synthetic RGBA image in `tmp_path`), `test_hd_pack_is_idempotent`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3: Implement** as specified; sanitise texture names for file names (backslashes in GUI names become `__`) and keep the same sanitising function used by the viewer index (`texture_file(name) -> str`, exported here).
- [ ] **Step 4:** Run; expect PASS. Generate both folders for the three creatures to `S/viewer/textures/original` and `S/viewer/textures/hd`.

### Task 4: Viewer page

**Files:**
- Create: `viewer/index.html`, `viewer/app.js`, `viewer/dk2v.js` (parser and frame blending, no DOM), `viewer/vendor/` (three.js module and OrbitControls via `npm install three@0.160.0`, copied in)
- Test: `tests/viewer/test_dk2v.mjs` (Node), `tests/test_viewer_page.py` (Playwright)

**Interfaces:**
- Consumes: `.dk2v` and `index.json` (Task 2), texture folders (Task 3).
- Produces (`dk2v.js`): `parseDk2v(buffer: ArrayBuffer) -> Model`; `framePositions(part, frameIndex: number, bbox) -> Float32Array`; `blend(a: Float32Array, b: Float32Array, t: number) -> Float32Array`. `app.js` reads `index.json`, builds the creature and animation lists, and exposes `window.viewer = {setCreature, setAnimation, setPack('original'|'hd'), currentFrame()}` for tests.

- [ ] **Step 1: Write failing tests.** Node: `parseDk2v` on the exported Goblin file returns the same part count, vertex counts and frame 10 positions (within 1e-4) as a JSON dump the Python test helper writes; `blend` at t=0.5 is the midpoint. Playwright: page loads with zero console errors; creature list shows Goblin, Bile Demon, Horned Reaper; choosing an animation advances `currentFrame()`; `setPack('hd')` makes texture requests go to `textures/hd/`; deleting one HD file (served from a temp copy) falls back to `textures/original/` and the page still shows the model; a part whose texture is missing renders (grey) with no console error.
- [ ] **Step 2:** Run both; expect FAIL.
- [ ] **Step 3: Implement.** Fixed camera framing from `bbox`; orbit and zoom; play/pause, frame scrub, speed slider, loop; Original / HD toggle with the label "Enhanced (non-AI)" on HD; texture load with `onerror` fallback to original; texture filtering set to nearest-neighbour for Original and linear for HD (the original is a 128px game texture; the HD pack should look smooth).
- [ ] **Step 4:** Run both test sets; expect PASS.

### Task 5: Visual verification against the reference renderer

**Files:**
- Create: `scripts/check_viewer_frames.py`
- Test: runs Playwright; output `S/viewer/check/*.png`

**Interfaces:**
- Consumes: `window.viewer` (Task 4), `render_textured` (`scripts/kmf_textured.py`).

- [ ] **Step 1:** Write `check_viewer_frames.py`: for each of the three creatures, one walk frame and one melee frame, screenshot the page (`deviceScaleFactor` 1) next to the Python render of the same frame, and report the mean silhouette overlap (non-background pixels) as an IoU.
- [ ] **Step 2:** Run it; the IoU must be ≥ 0.85 per pair. Look at the contact sheet by eye: same pose, textures the right way up, not mirrored. Fix the viewer if not (likely candidates: handedness, UV flip, axis order).
- [ ] **Step 3:** Record IoUs and what was seen in the ledger.

### Task 6: Package, widen, publish

**Files:**
- Create: `scripts/build_viewer.py` (runs the exporter and texture steps for every creature in `CREATURES`, then writes `index.json`), `viewer/README.md`
- Modify: `scripts/viewer_creatures.py` (add the remaining creatures and heroes to `CREATURES`)
- Test: `tests/test_build_viewer.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `S/viewer/` ready to publish and `deliverables/dk2_viewer.zip`.

- [ ] **Step 1: Write failing test** `test_every_creature_in_table_has_pose_and_at_least_one_animation` and `test_unattributed_list_contains_no_creature_prefix_in_table`.
- [ ] **Step 2:** Run; expect FAIL once the table is widened. Add creatures from the model index prefixes (imp, warlock, maiden, rogue, wizard, vampire, troll, knight, giant, thief, salamander, monk, dwarf, skeleton, dark elf, dark knight, archer, fairy, guard, king and others found by listing); check each by running the Task 1 mapping and confirming nothing is misattributed. Report to the user any file group whose creature cannot be named confidently.
- [ ] **Step 3:** Run `python3 scripts/build_viewer.py`; print total size and the largest files. If the page exceeds the artifact limits (16 MB for the page and each file, 255 files per publish, 64 MB per publish), publish in stages (several publishes to the same URL) or ship fewer creatures first, and say so to the user.
- [ ] **Step 4:** Run the full suite (`python3 -m pytest tests -q`), re-run Task 5 on two newly added creatures, then publish with Artifact (load the `artifact-design` skill first, add `files` for data, textures and vendor) and build `deliverables/dk2_viewer.zip` with a README explaining local use (`python -m http.server` in the folder, since browsers block file loading from `file://`) and the GPL-3.0 and EA-assets notices.
- [ ] **Step 5:** Tell the user what is in the viewer, that the HD pack is non-AI, how to drop in their own AI-upscaled PNGs (same file names, 4x), and what is not verified.
