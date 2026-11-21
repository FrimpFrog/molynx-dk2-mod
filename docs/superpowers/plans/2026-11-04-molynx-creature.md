# Molynx (47th DK2 creature) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Flame resource folder that puts the custom character Molynx into Dungeon Keeper 2 as a new 47th creature, delivered as three user-run tests, with an audit that proves every file is authentic.

**Architecture:** Small, single-purpose modules read and write DK2's own formats losslessly (KWD/KLD blocks, creature records, KMF models, text strings). A builder assembles `flame/resources/Molynx/` in three stages. An independent audit re-reads the built files from disk and diffs them against the originals.

**Tech Stack:** Python 3, numpy, Pillow, pytest. Existing modules: `scripts/dk2kmf.py` (reader), `scripts/dk2wad.py`, `scripts/custom_character.py` (`build_character`), `scripts/behemoth_anims.py` (`build_all`).

**Spec:** `docs/superpowers/specs/2026-11-04-molynx-creature-design.md`

## Global Constraints

- Delivery is a Flame resource folder (`flame/resources/Molynx/`); the user adds `Molynx` FIRST in `alt-resources` of Flame's `config.toml` (see Amendments).
- Molynx is the 47th creature: creature ID 47, name "Molynx", a Creatures block of 47 records. If the slot fails, the same record, model and icon go into Horny's slot.
- Record sizes are 5449 bytes (editor `Creatures.kwd`) and 5537 bytes (205 of 241 maps). Block header is 56 bytes; its check fields are signature + 1 and signature + 2. The test map uses the map's own record size.
- The test map is a copy. No original map or game file is modified.
- Molynx gets no new sounds: he uses the Bile Demon's sound category. Lair, spells and small icons are the Bile Demon's.
- Slots with no matching clip reuse the nearest clip and are listed for review. No clip is invented.
- Stat multipliers over the Bile Demon: health ×1.6, melee damage ×1.4, fear ×1.2, threat ×1.4, mass ×1.2, pay ×1.3.
- Test ladder (user decision: Tasks 1–8 are one bundled test): Test A = the 47th creature with his model, textures and animations (stages 1 and 2 together); Test B = his portrait, stats, text (stage 3). The bundle also carries the stage 1 folder (`Molynx_slot_only`, exact Bile Demon copy as ID 47) so a failure can be traced to the slot or to the assets.
- The panel portrait is the user's uploaded image (`/mnt/user-data/attachments/76eb4112-412f-58e4-a30a-b275f992be89/8582356e-image.png`). All other icons are placeholders.
- Out of scope: redrawn icons, new sounds, campaign portal pools beyond the test map, multiplayer compatibility.
- Repo conventions: scripts in `scripts/`, tests in `tests/`, each module docstring ends `LICENCE: GPL-3.0.`, tests skip when game data is absent. `/home/claude` is a git repo (base 1826ae7), so every "Checkpoint" step (run the listed tests plus `pytest tests/test_dk2kmf.py -q`) is a commit.
- Molynx and the Behemoth are not added to `deliverables/dk2_tools*`, the viewer site or any published artifact. Output goes to `deliverables/molynx_mod/`.
- User-facing text (the test guide) is short and in simple language.

## Review Focus

- A map with 44 creature records, or with 5449-byte records, is spliced without mixing sizes (Task 4).
- A clip longer than 128 frames (`victory_celebration`, 245 frames) decodes correctly across the second block (Task 7).
- Bytes the reader does not understand survive every read and write unchanged (Tasks 1, 2, 6).
- Names fit their fields (creature name 32 bytes, resource names 64 bytes) and match file names exactly, including case (Tasks 3, 8, 10).
- Vertices that never move, and 1-frame poses, encode and decode (Task 7).

## File Structure

| File | Responsibility |
|---|---|
| `scripts/dk2paths.py` | Locations of the user's game data; `have(path)` for test skips |
| `scripts/dk2kwd_blocks.py` | Read and write the block sequence in `.kwd` and `*Globals.kld` |
| `scripts/dk2kwd_creatures.py` | Named access to a creature record |
| `scripts/molynx_record.py` | Build the Molynx record per stage |
| `scripts/dk2kld_splice.py` | Splice a Creatures block into a Globals file; retarget a creature in a Things file |
| `scripts/dk2kmf_write.py` | Lossless KMF writer and the animation encoder |
| `scripts/molynx_assets.py` | Export Molynx's KMFs, textures and portrait |
| `scripts/dk2str.py` | Read and write `Text.str` |
| `scripts/build_molynx_mod.py` | Assemble `flame/resources/Molynx/` for stage 1, 2 or 3 |
| `scripts/audit_molynx_mod.py` | Independent audit of a built folder |
| `docs/molynx-test-guide.md` | What the user does and reports for each test |

---

### Task 1: Game data paths and the block container

**Files:** Create `scripts/dk2paths.py`, `scripts/dk2kwd_blocks.py`, `tests/test_dk2kwd_blocks.py`

**Interfaces:**
- Produces: `dk2paths.EDITOR_DIR`, `MAPS_DIR`, `TEXT_DIR`, `GAME_DIR`, `MESHES_DIR`, `QUUZ_MAPS_DIR` (all `Path`, under `/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad/up/` and `.../dk2game/Ai`, `.../extracted/Meshes`); `have(p: Path) -> bool`.
- Produces: `Block(signature: int, header: bytes, content: bytes)` with property `item_count`; `read_blocks(data: bytes) -> list[Block]`; `write_blocks(blocks: list[Block]) -> bytes`; `rebuilt_block(like: Block, records: list[bytes]) -> Block` (new header: item count, file size, content size updated, everything else copied from `like`).

- [ ] **Step 1: Write failing tests.** `test_every_globals_kld_roundtrips`: all 241 `*Globals.kld` in `MAPS_DIR` satisfy `write_blocks(read_blocks(d)) == d`. `test_editor_kwd_files_roundtrip`: same for the 13 `.kwd` files in `EDITOR_DIR`. `test_header_invariants`: for every block, `check_one == signature + 1`, `check_two == signature + 2`, `file_size == 56 + len(content)`. `test_rebuilt_block_counts`: rebuilding the Creatures block with 47 equal-size records gives `item_count == 47` and `read_blocks` accepts the result.
- [ ] **Step 2: Run** `pytest tests/test_dk2kwd_blocks.py -q`; expect failures (module missing).
- [ ] **Step 3: Implement.** Header layout (verified): signature u32, 4, file size u32, signature+1, 28, item count u32, 1, created 10 bytes, modified 10 bytes, signature+2, content size u32 (56 bytes). Keep the header bytes verbatim.
- [ ] **Step 4: Run tests; all pass.** If any file fails the invariants, record it in the test as a named exception with its reason.
- [ ] **Step 5: Checkpoint.**

### Task 2: Creature record fields

**Files:** Create `scripts/dk2kwd_creatures.py`, `tests/test_dk2kwd_creatures.py`

**Interfaces:**
- Consumes: `Block`, `read_blocks`, `rebuilt_block` (Task 1).
- Produces: `FIELDS: dict[str, Field]` (`Field(offset: int, fmt: str)`) covering every field of the community template `ufdada/dk2-tools` `Formats/Level/creatures_struct.bt` (names as in the template, e.g. `hp`, `meleeDamage`, `fear`, `threat`, `mass`, `pay`, `creatureId`, `orderInEditor`, `cloneCreatureId`, `lairObjectId`, `soundGategory`, `nameStringId`); `ANIM_SLOTS: list[str]` (the 38 `anim*Resource` names plus `animWalkback`, `animPoseFrame`, `animWalk2`, `animDiePose`, `animMelee2`, `animExtra43..47`, `drunkIdle`), each an 84-byte resource whose name is the first 64 bytes; `class Creature` with `raw: bytearray`, `get(name)`, `set(name, value)`, `anim_name(slot) -> str`, `set_anim_name(slot, name)`; `read_creatures(block) -> list[Creature]`; `creatures_block(creatures, like: Block) -> Block`.

- [ ] **Step 1: Write failing tests** (editor `Creatures.kwd` and one 5537-byte map): record sizes are 5449 / 5537; the 46 names equal `['Imp','Prince Tristran','Bile Demon', ..., 'Maiden Of The Nest','Jack']` in order; `creatureId` of record *i* equals *i*+1 (if not, report the real values: the game looks up by ID from index id−1); `Imp`'s walk slot is `Imp_Run` and its portrait resource is `gui\creatures\evil\imp-bg`; `hp` and `pay` are in 1..65535 for all; for every field `set(name, get(name))` leaves `raw` unchanged; the end offset of the last field is 5449 (5537 with the tail); `creatures_block(read_creatures(b), b)` reproduces the original block bytes.
- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement.** Take sizes of `Light`, `Attraction`, `Spell`, `Resistance`, `JobPreference`, `JobAlternative`, `Footstep`, `Swipe`, `SpecialAbility`, `GammaEffect` from the template file; bytes with no name stay in `raw` untouched.
- [ ] **Step 4: Run tests; all pass.**
- [ ] **Step 5: Checkpoint.** Also print a table of `creatureId, name, cloneCreatureId, hp` for the 46 creatures to `scratchpad/creature_table.txt` (used in Task 3).

### Task 3: The Molynx record

**Files:** Create `scripts/molynx_record.py`, `tests/test_molynx_record.py`

**Interfaces:**
- Consumes: `Creature` (Task 2).
- Produces: `MULTIPLIERS = {"hp": 1.6, "meleeDamage": 1.4, "fear": 1.2, "threat": 1.4, "mass": 1.2, "pay": 1.3}`; `SLOT_CLIPS: dict[str, str]` (anim slot → Molynx clip, Task 8 table); `build_molynx(bile: Creature, stage: int, order_in_editor: int, text_ids: dict[str, int] | None = None) -> Creature` (`text_ids` keys: `name`, `tooltip`, `general`, `strength`, `weakness`; used at stage 3 only); `review_list() -> list[tuple[str, str, str]]` (slot, clip used, reason it is only the nearest).

- [ ] **Step 1: Write failing tests.** Stage 1: result differs from `bile.raw` only at the offsets of `name`, `creatureId`, `orderInEditor`; `name == "Molynx"`, `creatureId == 47`. Stage 2: additionally only anim slot names differ, and each is `Molynx-<ClipName>` ≤ 64 bytes. Stage 3: additionally only `hp, meleeDamage, fear, threat, mass, pay`, the portrait resources and the five text ID fields differ; each stat equals `min(65535, round(bile * multiplier))`; the portrait names are `gui\creatures\evil\molynx` and `gui\creatures\evil\molynx-bg`.
- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement.** `cloneCreatureId`: use the pattern the 46 creatures follow in `creature_table.txt` for non-unique creatures (own ID or 0), and write the decision in the module docstring. `soundGategory`, lair, spells, attractions, jobs, anger values stay the Bile Demon's. Stage 1 and 2 keep the Bile Demon's text IDs; stage 3 uses `text_ids` from Task 9.
- [ ] **Step 4: Run tests; all pass.**
- [ ] **Step 5: Checkpoint.**

### Task 4: Splice into a Globals file

**Files:** Create `scripts/dk2kld_splice.py`, `tests/test_dk2kld_splice.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `splice_creatures(globals_kld: bytes, extra: list[Creature]) -> bytes` (appends `extra` to the map's Creatures block, ordered by `creatureId`; raises `ValueError` if a record size differs from the map's); `molynx_for_map(globals_kld: bytes, stage: int) -> Creature` (Bile Demon of that map, in that map's record size, run through `build_molynx`).

- [ ] **Step 1: Write failing tests**, each run on three maps (one 46×5537, one 44×5537, one 46×5449): result has `item_count` 47 / 45 / 47 and `get("creatureId")` of the last record is 47; the first N records are byte-identical to the input; every other block is byte-identical; `read_blocks` accepts the result and the header invariants hold; mixing a 5449-byte Molynx into a 5537 map raises `ValueError`.
- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement** with `read_blocks`, `rebuilt_block`, `write_blocks`.
- [ ] **Step 4: Run tests; all pass.**
- [ ] **Step 5: Checkpoint.**

### Task 5: Place Molynx in a map's Things file; pick the test map

**Files:** Create `tests/test_dk2kld_things.py`; modify `scripts/dk2kld_splice.py`

**Interfaces:**
- Produces: `find_creature_things(things_kld: bytes) -> list[ThingRef]` (`ThingRef(offset: int, creature_id: int, player: int)`); `retarget_creature_thing(things_kld: bytes, index: int, new_creature_id: int) -> bytes`; `pick_test_map() -> str` (name of the smallest map whose Globals has 46×5537 records, whose Things holds at least one Bile Demon owned by a player, and which exists in both `MAPS_DIR` and `QUUZ_MAPS_DIR` if possible); `map_source_dir(name) -> Path` (prefers `QUUZ_MAPS_DIR`, since the user's game loads those first).

- [ ] **Step 1: Derive the Things layout** from the community template `ufdada/dk2-tools` `Formats/Level/things_struct.bt` and write the findings into the module docstring.
- [ ] **Step 2: Write failing tests**: parsing every `*Things.kld` in `MAPS_DIR` then re-serialising gives identical bytes; `retarget_creature_thing` changes exactly the creature ID bytes of one record and nothing else; the retargeted file lists that record with `creature_id == 47`; `pick_test_map()` returns a name whose seven map files all exist in `map_source_dir`.
- [ ] **Step 3: Run; expect failures. Step 4: Implement. Step 5: Run; all pass.**
- [ ] **Step 6: Checkpoint.**

### Task 6: Stage 1 mod, audit, and the Test 1 hand-off

**Files:** Create `scripts/build_molynx_mod.py`, `scripts/audit_molynx_mod.py`, `tests/test_build_molynx_mod.py`, `docs/molynx-test-guide.md`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `build_mod(stage: int, out_dir: Path) -> Path` (creates `out_dir/Molynx/` with `resource.toml` (name "Molynx"), and `Data/editor/maps/<test map files>`; for stage 1 only the patched `...Globals.kld` and `...Things.kld` differ from their sources); `audit(mod_dir: Path) -> AuditReport` with `ok: bool`, `lines: list[str]`.
- Audit rules, each re-reading files from disk with fresh parsing: (a) every non-Creatures block in each patched Globals is byte-identical to its source; (b) the Creatures block has 47 records, the first 46 identical to the source, record 47 matches the stage's expected differences from the Bile Demon; (c) header invariants; (d) the Things file differs from its source only in one creature ID; (e) every other map file is byte-identical to its source; (f) for stage ≥ 2, every animation name, mesh and texture the record references exists (exact case) in the mod folder or in the game's `Meshes.WAD` / `EngineTextures.WAD`.

- [ ] **Step 1: Write failing tests** with `tmp_path`: stage 1 builds, `audit(...).ok` is true, the folder holds only `resource.toml` and map files; corrupting one byte of a copied non-Creatures block makes `audit` return `ok == False` with a line naming the block.
- [ ] **Step 2: Run; expect failures. Step 3: Implement both modules. Step 4: Run; all pass.**
- [ ] **Step 5: Write `docs/molynx-test-guide.md`** (Test 1 only for now): copy the folder to `<DK2>\flame\resources\`, add `Molynx;` at the START of `alt-resources`, start the test map by name (Dead Walking; see Amendments), what to check (map loads, Molynx walks, pick up and drop, creature panel opens), and what to send back (`flame/latest.log`, a screenshot, the exact point of any crash). Include: if a checksum error appears, set `NoFileChecksum = true`.
- [ ] **Step 6: Build stage 1 into `deliverables/molynx_mod/Molynx_slot_only/` and run the audit. Do not send it yet; continue to Task 7 (the user bundled Tasks 1–8 into one test).**

### Task 7: Lossless KMF writer and animation encoder

**Files:** Modify `scripts/dk2kmf.py` (keep every field the writer needs, including the raw CTRL bytes, so a parse loses nothing); create `scripts/dk2kmf_write.py`, `tests/test_dk2kmf_write.py`

**Interfaces:**
- Produces: `write_kmf(model: Kmf) -> bytes`; `encode_anim(name: str, sprites: list[Sprite], frames: list[list[np.ndarray]], wrap: bool, controls: bytes) -> Anim` (`frames[f][s]` is an `(n_s, 3)` array for sprite *s*; picks `pos` and `scale` from the data; reduces keyframes per vertex per 128-frame block while keeping error within one quantisation step, `scale / 511`); `anim_invariants(anim: Anim) -> list[str]` (empty when valid: `itab` values index into `geom_xyz`, offsets stay inside the vertex's keyframe list, keyframe frame numbers do not decrease within a block and reach the block's last frame).

- [ ] **Step 1: Write failing tests**: `write_kmf(parse_kmf(b)) == b` for all 2015 files in `MESHES_DIR`; `anim_invariants` is empty for every game animation; encode then `evaluate_anim_frame` of a synthetic 245-frame two-sprite random walk has error ≤ `scale / 511 * 1.01` at every frame (checks the second block); a vertex that never moves shares a track and decodes exactly; a 1-frame pose round-trips; re-encoding the decoded frames of `Bile_Demon-Walk.kmf` and of one game animation with more than 128 frames decodes within the same bound and passes `anim_invariants`; `parse_kmf(write_kmf(...))` of the encoded result succeeds and is fully consumed.
- [ ] **Step 2: Run; expect failures. Step 3: Implement. Step 4: Run; all pass.** If any game file is not byte-identical, list it in the test with the reason and keep the other tests strict.
- [ ] **Step 5: Checkpoint.**

### Task 8: Export Molynx's models, textures and portrait

**Files:** Create `scripts/molynx_assets.py`, `tests/test_molynx_assets.py`

**Interfaces:**
- Consumes: `build_character`, `build_all` (existing), `write_kmf`, `encode_anim` (Task 7).
- Produces: `export_molynx_assets(out_dir: Path, height_ratio: float = 1.15) -> dict` with `meshes: dict[str, Path]` (`Molynx-<ClipName>.kmf` for each of the 25 clips, plus `Molynx-PoseFrame.kmf`, `Molynx-DiePose.kmf`, `Molynx-DraggedPose.kmf`, one frame each from `stand` frame 0, `die` last frame, `being_picked_up` frame 0), `textures: dict[str, Path]` (`Molynx_Body.png` 128×128, `Molynx_Mace.png` 128×128), `portraits: dict[str, Path]` (`molynx-bg.png` 64×128 from the user's image cropped to 1:2 about the centre; `molynx.png` 64×64 cropped to his face), `preview: Path` (the two portraits next to Bile Demon's, for the user to approve).
- Scale: every vertex is multiplied by one factor so the rest height equals the Bile Demon's rest mesh height × `height_ratio`; x is mirrored as the game's left-handed data requires (`FLIP` in `behemoth_anims.py`); normals are computed from the triangles.
- Slot table (`SLOT_CLIPS` in `molynx_record.py`; slot names are the template's `anim<Name>Resource` fields without the `anim` prefix and `Resource` suffix, case-insensitive, e.g. `walk` is `animWalkResource`): walk→walk, run→running, draggedPose→DraggedPose, recoilHff→getting_hit, recoilHfb→being_slapped, melee1/melee2/special→attack, die→die, diePose→DiePose, happy→happy, angry→angry, swing→being_picked_up, sleep→sleeping, eat/eatCorpse/drink→eating, building→manufacturing, pray→praying, fallback→knockback, getUp→waking_up, dance→celebrating, drunk/drunk2/stunnedPose/dejectedPose→staggering, idle1→stand, idle2/drunkIdle→fidgeting, poseFrame→PoseFrame, walk2→patrolling, entrance/walkback→walk, research/loot→interacting_with_objects, magic/torture/elec/electrocute→getting_hit. Every slot not in this list takes `stand`. Slots that use a clip only as the nearest (magic, torture, elec, electrocute, research, loot, special, entrance, walkback, melee2, drink, eatCorpse, stunnedPose, dejectedPose, drunkIdle) go in `review_list()`.

- [ ] **Step 1: Write failing tests**: 25 clip files plus 3 poses exist; each file parses, is fully consumed, and passes `anim_invariants`; decoding every frame of every clip matches the source frames within `scale / 511 * 1.01` after the same scale and flip; the rest height is within 1% of the target; material texture names equal the texture file stems; every texture is a power of two ≤ 128 (see Amendments); portraits have the stated sizes; every file stem is ≤ 64 bytes and matches the names written by `build_molynx` at stage 2 (exact case).
- [ ] **Step 2: Run; expect failures. Step 3: Implement. Step 4: Run; all pass.**
- [ ] **Step 5: Render check.** Using `scripts/kmf_textured.py`, render the rest pose and frames at 0%, 50%, 100% of `walk`, `eating` and `victory_celebration` from the written `.kmf` files and from the source frames; save the contact sheet to the scratchpad and confirm they match by eye and by mean pixel difference < 2%.
- [ ] **Step 6: Build and hand off the bundle.** Extend `build_mod` for stage 2 (adds `Data/Meshes/*.kmf`, `Data/EngineTextures/Molynx_*.png`, the stage 2 record) and `audit` for rule (f). Build stage 2 into `deliverables/molynx_mod/Molynx/`, run the audit on both folders, update `docs/molynx-test-guide.md` for Test A (walk, run, fight, sleep, fall, picked up and dropped; the review list of reused clips) and send the bundle zip and guide with `SendUserFile`. STOP and wait for the result. If it fails, diagnose from the log (the disassembly in `scratchpad/exe/dk2.asm` is the reference), try `Molynx_slot_only` to separate slot from assets, and decide with the user between patching and Horny's slot.
- [ ] **Step 7: Checkpoint.**

### Task 9: Text strings

**Files:** Create `scripts/dk2str.py`, `tests/test_dk2str.py`

**Interfaces:**
- Produces: `read_str(data: bytes) -> list[str]`; `write_str(strings: list[str]) -> bytes`; `append_strings(data: bytes, new: list[str]) -> tuple[bytes, list[int]]` (new IDs in order).

- [ ] **Step 1: Derive the `BFST` layout** from `Data2/Text/Default/Text.str` (header `BFST`, then count and offset table) and the community templates; document it in the module docstring.
- [ ] **Step 2: Write failing tests**: `write_str(read_str(d)) == d` for every `.str` in `TEXT_DIR`; `append_strings` on `Text.str` keeps the original strings unchanged and returns IDs equal to the old count, +1, +2, +3, +4; the Bile Demon's `nameStringId` string reads as "Bile Demon".
- [ ] **Step 3: Run; expect failures. Step 4: Implement. Step 5: Run; all pass.**
- [ ] **Step 6: Checkpoint.** If the layout cannot be recovered exactly, stop: Molynx then keeps the Bile Demon's text IDs (already the stage 1 and 2 behaviour) and the user is told.

### Task 10: Stage 3, final audit, Test B hand-off

**Files:** Modify `scripts/build_molynx_mod.py`, `scripts/audit_molynx_mod.py`, `docs/molynx-test-guide.md`, `tests/test_build_molynx_mod.py`

**Interfaces:**
- Consumes: Tasks 3, 8, 9. `build_mod(stage, out_dir)` now adds stage 3 → `Data/EngineTextures/molynx*.png`, `Data/Text/Default/Text.str` (original plus five strings: name, tooltip, general description, strength, weakness), the stage-3 record, and `Data/editor/Creatures.kwd` (47 records in the 5449-byte form, for the user's map editor; untested until the user opens the editor).

- [ ] **Step 1: Write failing tests**: stage 3 builds and passes `audit`; stage 3 contains every file of stage 2; the audit also checks that the appended strings read back, that no file is larger than 16 MB, and writes the diff report `AUDIT.txt` next to the mod (list of every file, size, and "identical to source" or the exact differing offsets).
- [ ] **Step 2: Run; expect failures. Step 3: Implement. Step 4: Run; all pass.**
- [ ] **Step 5: Extend the guide** with Test B (portrait, stats, text; a full cycle of portal, lair, eat, payday, train, slap, drop, fight).
- [ ] **Step 6: Final triple check.** Run the whole suite (`pytest tests -q`), run `audit` on stage 3, and run the Task 8 render check on the built folder. Build stage 3 into `deliverables/molynx_mod/Molynx/` (replacing stage 2) and send it once Test A has passed.
- [ ] **Step 7: Checkpoint.**

## Amendments (as built, Tasks 1-8; ruled by the controller, see the SDD ledger)

These replace the matching plan text above where they differ:

- **alt-resources order:** the mod folder is listed FIRST (Flame uses the first listed folder that has a file; `Quuz_LevelFixes` has its own Siege). Guide and audit follow this.
- **Test maps:** Dead Walking (first; Keeper 1 owns the first Bile Demon) and Siege (second; multiplayer only, no skirmish flag). `Level =` with either was never run in the game.
- **Bundle:** `Molynx_test_bundle.zip` holds two folders, `Molynx` (stage 2: slot + model, animations, textures) and `Molynx_slot_only` (stage 1: exact Bile Demon copy), plus the guide. The user enables ONE folder at a time. `build_mod(stage, out_dir, name="Molynx", maps=None, text_ids=None, *, anim_info=None, file_wraps=None, assets_dir=None)`; stage 1 builds `Molynx_slot_only`; resource.toml name equals the folder name. The zip is made by `scripts/make_molynx_bundle.py` (deterministic).
- **Textures:** 128 x 128 (the engine and the Quuz HD pack allow no more); no 256 px fallback exists.
- **Slots (Ruling F):** a slot that is empty in the Bile Demon stays empty; "every other slot takes `stand`" only when the Bile slot is non-empty. The Bile Demon has no `run` slot. Each shipped slot also gets its 20-byte tail rewritten (frames, fps 30, flags, type) so the record describes Molynx's files (stage 2 = names + tails, using `file_wraps` and `anim_info`).
- **Files:** one KMF per clip (not one shared mesh); variants `walk_once` (entrance). Poses are static meshes (type 5): `PoseFrame`, `DiePose`, `DraggedPose`, `StunnedPose` (staggering frame 48), `ElecPose` (getting_hit frame 5). `stunnedPose` and `elec` stay static slots like every game creature; a type-5 `dejectedPose` slot takes `StunnedPose`. 24 `.kmf` ship (23 for Dead Walking alone plus `staggering` for Siege's inert type-NONE slot). 7 clips are not shipped (no slot uses them).
- **Encoding:** every clip is encoded with reduced keyframes (`reduce=True`), like 661 game files; the earlier `reduce=False` decision was withdrawn (GEOM sizes up to 2.2x the game maximum). Largest shipped GEOM: `staggering` 42,185 (game maximum 34,816); every other shipped file is at most 27,974.
- **Sleeping** is a non-looping one-way clip (exception listed in the audit); fallback if it misbehaves: a static sleep pose from the last frame.
- **Task 7:** poses are static meshes, not 1-frame animations (the game has no animation with fewer than 2 frames).
- **Task 8:** `SendUserFile` belongs to the controller, not the implementer. The guide step "Create `docs/molynx-test-guide.md`" is Test A (two folders).
- **Checkpoint:** `/home/claude` is a git repo (base 1826ae7); every Checkpoint is a commit.
