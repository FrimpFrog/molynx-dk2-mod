# Molynx test map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task by task.

**Spec:** `docs/superpowers/specs/2026-11-05-molynx-testmap-design.md`

**Goal:** Build a flat test map (and a portal-pool variant) for Dungeon Keeper 2 and ship it inside the existing Molynx mod folders, with an independent audit, an updated guide and a new bundle zip.

## Global Constraints

- Repo conventions: scripts in `scripts/`, tests in `tests/`, each module docstring ends `LICENCE: GPL-3.0.`, tests skip when game data is absent, machine paths only in `scripts/dk2paths.py`. `/home/claude` is a git repo (HEAD at the start: 3990c61): commit with explicit paths, end each commit message with the two attribution lines given in your dispatch.
- Reuse, do not duplicate: `scripts/dk2kwd_blocks.py` (`read_blocks`, `rebuilt_block`, `write_blocks`), `scripts/dk2kld_splice.py` (`read_things`, `write_things`, `splice_creatures`, `molynx_for_map`, `map_source_dir`), `scripts/build_molynx_mod.py`, `scripts/audit_molynx_mod.py`, `scripts/make_molynx_bundle.py`, `scripts/molynx_record.py`. Format templates (ufdada dk2-tools, `.bt` files) are under the scratchpad `bt/dk2-tools/Formats/Level/` (map_struct, things_struct, player_struct, variables_struct, level_struct, triggers_struct, rooms_struct, terrains_struct). The game's code is disassembled in the scratchpad `exe/dk2.asm` (named functions in `flame_src/repo/mapping/DKII_EXE_v170.sgmap`).
- Terrain ids in `Map.kld` are 1-based (`id = index in Terrain.kwd + 1`): 1 Impenetrable Rock, 2 Rock, 3 Dirt Path, 6 Gold, 7 Gems, 8 Claimed Path, 12 Portal, 14 Dungeon Heart, 30 Edge of Map. A Map tile is 4 bytes: terrainId, playerId (1 Good, 2 Neutral, 3..7 keepers), flag, unknown. Things coordinates are tile coordinates. Verify every one of these against the data before relying on it.
- Base map: "Dead Walking" (editor folder `MAPS_DIR`, via `map_source_dir`). Keeper 1 is owner 3, Keeper 2 owner 4. Keep the map's width and height (69 x 69) and its outer border.
- The user cannot be asked and the game cannot be run here: decide with the evidence in the files, record each decision as a `Ruling:` line in the report.
- Nothing about Molynx or Behemoth is added to `deliverables/dk2_tools*`, the viewer site or any published artifact. Generated output goes to `deliverables/molynx_mod/` only. `SendUserFile` belongs to the controller, not to you. You never dispatch subagents.
- Deliverable names: new zip `deliverables/molynx_mod/Molynx_test_bundle_v2.zip`; never touch the existing `Molynx_test_bundle.zip` that was already sent. The scratchpad directory is given in your dispatch.

---

## Task 1: The map generator and its validator

**Files:**
- Create: `scripts/dk2_testmap.py`, `tests/test_dk2_testmap.py`
- Maybe extend: `scripts/dk2kwd_blocks.py` helpers only if a missing helper is needed (with tests)

**Interfaces:**
- Consumes: the base map's files (`<name>.kwd`, `<name>Map.kld`, `...Things.kld`, `...Players.kld`, `...Triggers.kld`, `...Variables.kld`), Siege's Variables file (pool and creature availability for player 3: ids 3,4,5,6,7,10,12,22,23,24,45), the editor's `AutoSaveTriggers.kld` (60 bytes, an empty Triggers block) as the pattern for a minimal Triggers file, `Terrain.kwd`, `Rooms.kwd`.
- Produces (names are fixed so Task 2 can call them):
  - `testmap_files(stem: str, *, molynx_in_pool: bool, display_name: str = "Molynx Test") -> dict[str, bytes]`: the six map files named `<stem>.kwd`, `<stem>Map.kld`, `<stem>Things.kld`, `<stem>Players.kld`, `<stem>Triggers.kld`, `<stem>Variables.kld` (no Globals: the builder splices it). Deterministic: the same call returns the same bytes. `display_name` is the name the level file shows in the Skirmish list (find out where the level file stores it and its description; keep the file's other fields).
  - `validate_testmap(files: dict[str, bytes], stem: str, *, molynx_in_pool: bool) -> list[str]`: empty list when every layout rule below holds, otherwise one line per violation. It parses the bytes itself; it must be usable on files read back from disk.
  - `preview_png(files: dict[str, bytes], stem: str, path: Path) -> None` and `preview_text(files, stem) -> str` (one character per tile, a legend): a picture so a human can see the map.
- Layout rules (the validator checks each; thresholds are minimums, you may exceed them):
  1. Map width and height equal the base's; the outermost ring of tiles equals the base's ring byte for byte.
  2. Exactly two Dungeon Hearts: for each of owner 3 and 4 the heart tiles (id 14) have the same shape and relative tile data as the base's heart for that owner, and the heart's Things (the room thing, objects, whatever the base uses to make a working heart: find out) are present exactly as the base has them, moved with the heart.
  3. Keeper 1 has one Portal (id 12 tiles shaped like the base's portal, owner 3) with the things the base gives its portal, connected by floor to the heart.
  4. Keeper 1's open area: at least 900 contiguous floor tiles (Claimed Path owner 3 at least 300, the rest Dirt Path or Claimed Path), containing the heart and portal, with no rock, door, trap or water inside it. 4 to 6 Gems tiles (id 7) and 2 Gold tiles (id 6) embedded in Rock (id 2) so that each touches the open area on at least one side; no other terrain.
  5. Keeper 2 is sealed: breadth-first search over passable terrain (every floor-like terrain: Dirt Path, Claimed Path, room tiles) from Keeper 2's heart never reaches any Keeper 1 tile; every tile within Chebyshev distance 2 of Keeper 2's chamber floor that is not chamber floor is Impenetrable Rock (id 1); the chamber holds no Gems or Gold.
  6. Things: only what rules 2 and 3 need, plus creatures: for owner 3 six Imps (id 1) and exactly one creature thing with creatureId 47 placed on floor next to the heart; for owner 4 four Imps inside the sealed chamber. No other creature, hero, neutral, door, trap, hero gate or action point unless you can show from the base's data or the game's code that the game needs it (then record the reason). Every thing sits on a passable tile of the right owner's area.
  7. Players file: all fields copied from the base except Keeper 1's starting gold = 100000; Keeper 1 is human (isAi NO), Keeper 2 keeps the base's AI settings.
  8. Variables file: starts from the base map's Variables; for player 3 every room, door, trap and spell availability entry has value ENABLE (3), no duplicate entries are added (change the existing ones, add only the missing ones); the creature pool and creature availability for player 3 are Siege's (ids 3,4,5,6,7,10,12,22,23,24,45) and, only when `molynx_in_pool`, also creature 47 (pool value 50, availability ENABLE); the block header (item count, sizes) is recomputed.
  9. Triggers file: the editor's minimal empty Triggers (like `AutoSaveTriggers.kld`). If the base's Triggers or the game's loader shows that an empty file cannot work, say so in the report and choose the smallest working alternative.
  10. Level file (`.kwd`): copy of the base's with the display name (and description, if the file has one) changed; keep every flag (Skirmish and Multiplayer bits stay as the base's); all header check fields correct; its stored width/height unchanged.
  11. Container rules for every file: `read_blocks` accepts it, header check fields = signature + 1 / + 2 as in the base files, sizes and item counts consistent, timestamps copied from the base (deterministic).
- Writer exactness: re-writing each base file through the same code path with no change returns the base file byte for byte (a regression test over the base map and one more shipped map).

- [ ] **Step 1: Reconnaissance.** In `.superpowers/sdd/2026-11-05-molynx-testmap/task-1-recon.md` (the workspace folder you are given) record, with evidence from the base map's bytes and the templates: how a Dungeon Heart and a Portal are represented (Map tiles, Things types 204 / 194 / others, offsets and meaning of each field you use); what Things the base places at each keeper's base; where the level file stores the display name, description, flags, width and height; what the Players and Variables fields you change are; whether the Triggers file of the base does anything a skirmish needs. Do not write the generator on guesses: if a layout fact cannot be established, say so and use the base's own bytes.
- [ ] **Step 2: Write failing tests** (`tests/test_dk2_testmap.py`): one test per layout rule and per container rule above, each on the real generated files, plus tests that the validator fails on deliberately broken copies (a gap in the wall, an extra creature, a heart moved, a missing portal, a changed border byte, a duplicate availability entry), determinism, writer exactness, and the preview functions run. Tests skip when game data is absent.
- [ ] **Step 3: Run; expect failures. Step 4: Implement. Step 5: Run; all pass.** Also run the nine-file suite named in the Molynx plan's last report (`tests/test_dk2kmf.py tests/test_dk2kmf_write.py tests/test_molynx_assets.py tests/test_molynx_record.py tests/test_dk2kwd_creatures.py tests/test_dk2kwd_blocks.py tests/test_dk2kld_splice.py tests/test_dk2kld_things.py tests/test_build_molynx_mod.py`) plus your new file; all must pass with no warnings.
- [ ] **Step 6: Preview.** Write `preview_png` output for the three stems to the scratchpad and look at it; put the text preview in the report.
- [ ] **Step 7: Commit** (explicit paths).

---

## Task 2: Ship the map: builder, audit, guide, bundle

**Files:**
- Modify: `scripts/build_molynx_mod.py`, `scripts/audit_molynx_mod.py`, `docs/molynx-test-guide.md`, tests for both
- Create: `deliverables/molynx_mod/*` (generated, untracked), `Molynx_test_bundle_v2.zip`

**Interfaces:**
- Consumes: Task 1's `testmap_files`, `validate_testmap`, `preview_png`; `splice_creatures`, `molynx_for_map`, the existing builder pieces.
- Produces: `build_mod` also writes, for each stage and folder, the generated maps: stems `Molynx Test` (no pool), `Molynx Test Portal` (`molynx_in_pool=True`) and `Dead Walking` (the flat map under the shipped name, same content as `Molynx Test`, so it is found by the game even if new names are not listed); `Siege` stays the Test A map. Each generated map's Globals is the base map's Globals spliced with that folder's record 47 (the stage-specific record, `anim_info` and `file_wraps` as for the other maps). The folder holds exactly the files the audit lists. The resource name equals the folder name (rule h stays).
- Audit: for generated maps the audit needs its own checks written from the spec and the layout rules above, independently of `dk2_testmap.py` (it may share `dk2kwd_blocks`, the low-level parser, and `dk2paths`): parse the six files and Globals itself and check rules 1-11 above on the bytes found in the folder (the same rules, your own code and literals), that Globals = base Globals with record 47 appended exactly as the other maps (reuse the existing rule (b) logic), that exactly one creature thing has id 47 and owner 3, that "Dead Walking" and "Molynx Test" hold identical Map/Things/Players/Triggers/Variables bytes, and that "Molynx Test Portal" differs from "Molynx Test" only in the Variables file by exactly the added creature-47 entries. For Siege the Test A rules stay. The old rules (a)-(e) no longer apply to "Dead Walking" (it is no longer the shipped map): the audit says so in its output lines. Rule (f) (files named by record 47 exist) and (g) (folder holds only expected files) cover the new files. The audit must FAIL on deliberately broken copies (tests).
- Guide: rewrite the "What to do" steps of `docs/molynx-test-guide.md` for the new map. Keep what is still true (folder first in `alt-resources`, one folder at a time, what to send back, undo steps, the quirks). New: start the map "Molynx Test" (in the Skirmish list it appears under the name the level file shows; if it is not in the list, pick "Dead Walking", which is the same flat map; or set `Level = "Molynx Test"`); what to look for first (Molynx beside the heart; if he is not there, say what is there); then the checks of Test A; then the second map "Molynx Test Portal" (to see him arrive through the portal; tell me if the game crashes then). Plain short words, as in the current guide.
- Bundle: `scripts/make_molynx_bundle.py` zips both folders + guide into `deliverables/molynx_mod/Molynx_test_bundle_v2.zip` (deterministic, built twice and compared). Previous outputs `Molynx`, `Molynx_slot_only` folders are regenerated (remove only the generated folders; keep the old zip file untouched). Also write the map previews (PNG) to the scratchpad for the controller to send.

- [ ] **Step 1: Write failing tests** for the builder (the new files exist in both stages, Globals have 47 records, the portal variant differs only as stated, no Molynx assets are needed by the generated maps beyond the existing ones) and for the audit (passes on a fresh build, fails on: a hole in Keeper 2's wall, a second creature-47 thing, a changed border byte, Variables of the portal variant equal to the plain one, a missing file, an unexpected extra file, a changed Globals record).
- [ ] **Step 2: Run; expect failures. Step 3: Implement. Step 4: Run; all pass.**
- [ ] **Step 5: Build** both stages with the CLI (stage 2 needs the assets: scratchpad `molynx_assets_r4/`), audit both folders (AUDIT OK), build the zip twice (byte-identical), extract it into a fresh directory and audit the copies, record sha256 manifest and size in the report.
- [ ] **Step 6: Update the guide** and re-read it against the real files (names, steps).
- [ ] **Step 7: Run the full nine-file suite plus `tests/test_dk2_testmap.py`; all pass with no warnings. Commit (explicit paths).**
