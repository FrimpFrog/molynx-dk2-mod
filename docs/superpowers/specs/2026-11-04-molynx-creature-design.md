# Molynx (custom DK2 creature, 47th slot) — design

Date: 2026-11-04. Status: approved; Tasks 1-8 built (see Amendments at the end). (/home/claude is a git repo; this file is committed.)

## Purpose

Put the custom character **Molynx** (built on the MegaGlest Behemoth, with the 25 finished clips) into Dungeon Keeper 2 as a **new, separate, 47th creature**, behaving like an official creature: stats, likes and dislikes, happiness, recovery after being dropped, jobs, lair, pay, portal attraction. The user's own panel portrait is his panel icon. Every other icon is a placeholder to be redrawn later. If a 47th creature turns out to be impossible, the same assets go into the Horned Reaper's slot ("Horny") instead.

## What the user asked for

- Molynx is an evil-side creature, stronger and tougher than the Bile Demon: "the Horned Reaper of DK1, in DK2".
- Separate creature first. Replacing an existing creature is the fallback only.
- He must work exactly like other creatures: likes and dislikes, happiness meter, time to stand up after being dropped by the hand, and the rest of the AI data.
- The panel image the user uploaded is his panel icon. Other icons are placeholders.
- Success: the user starts the game and sees Molynx walking, fighting, sleeping and so on, with the other creatures' rules applied to him.

## Assumptions (user can correct)

- The user tests in the game, because DK2 cannot run in this workspace.
- Delivery is a Flame resource folder (`flame/resources/Molynx/`), because the user already runs Flame with `EnableFilePatching` and `EnableArtPatching` on.
- Molynx gets no new sounds. He borrows the Bile Demon's sound category.
- Slots with no matching clip of his 25 reuse the nearest clip and are listed for review. No clip is invented.
- Final balance numbers are tuned after he runs in the game.

## Facts that shape the design

Read from the user's files and the game program (`DKII-DX.exe`, 2,987,568 bytes):

- **Creature data is a plain data block.** `Data/editor/Creatures.kwd` holds 46 records. Likes and dislikes (happy, unhappy, angry and hate job lists), anger values (no lair, no food, no pay, no work, slapped, in hand), stun and slap-fearless durations, hunger, pay, health, melee damage, about 44 animation slots, sound category, icons and text IDs are all fields of that record (layout from the community template `ufdada/dk2-tools`, `creatures_struct.bt`).
- **Two record sizes exist.** The editor's `Creatures.kwd` uses 5449-byte records. 205 of the 241 maps in the editor folder use 5537-byte records (88 extra tail bytes), 12 use 5449, and 24 hold only 44 creatures. The game derives the record size from the block's content size divided by its item count, so varying counts and sizes already load.
- **Each map carries its own copy.** `<Map>Globals.kld` (about 680 KB) is a plain sequence of blocks: Terrain `0x6E`, Rooms `0x78`, Creatures `0xAA`, Doors `0x8C`, Traps `0x82`, Objects `0xF0`, CreatureSpells `0xA0`, KeeperSpells `0x96`, Shots `0x104`, Effects `0x10E`, EffectElements `0xFA`. Every block starts with a header whose check fields are signature + 1 and signature + 2. Changing the editor's `Creatures.kwd` alone does not affect existing maps.
- **The loader is count-driven.** In the game program, `CWorld_loadCreature` inserts each record into a table sorted by creature ID. The table's capacity comes from the block's item count (the table is heap-allocated with 2199 bytes per entry). A block with 47 records therefore loads, and lookup by ID (`CWorld::fun_50D390`) works for ID 47. This is a static reading and has not been run.
- **Textures are loose PNGs.** The user's HD packs place `Data/EngineTextures/<Name>.png`. The Flame log says artwork patching is on.
- **Models:** the game program builds mesh paths as `data\Meshes\<name>.kmf`. A loose `.kmf` mod has not been seen yet, so loose model loading is unconfirmed.
- **Unknown, to be settled by testing:** anything else indexed by creature ID that may be sized to 46: per-player creature info, the portal pool, the creature panel and the icon sheets. The test ladder below isolates these.

## Design

### 1. Creature record tools (`scripts/dk2kwd_creatures.py`)

A reader and writer for the Creatures block in both record sizes.

- Fields are exposed by name for everything Molynx needs. Unknown bytes are kept exactly, so a read, write round trip is byte-identical.
- Round-trip test: re-save the editor's `Creatures.kwd` and a shipped map's creature block, and require identical bytes.
- Header handling: item count, file size and content size are recomputed when records are added.

### 2. The Molynx record

- Start from a copy of the Bile Demon record in the same size as the target map (5537 bytes).
- Set: name "Molynx", creature ID 47, order in editor after the last creature.
- Stats above the Bile Demon: health, melee damage, pay, fear, threat, mass. Starting values are the Bile Demon's multiplied by: health ×1.6, melee damage ×1.4, fear ×1.2, threat ×1.4, mass ×1.2, pay ×1.3. These are starting points, tuned from the user's feedback after Test 3.
- Behaviour copied from the Bile Demon so he follows the same rules: job lists, anger values, unhappy threshold, hunger, sleep and wake times, stun and slap-fearless durations, flags, job class, fight style.
- Animation slots: each of the ~44 slots points at a Molynx animation name. His 25 clips fill the slots they match (walk, run, die, happy, angry, eat, pray, getting hit, dragged and so on). Remaining slots reuse the nearest clip and are written to a review list.
- Lair object, sound category, spells and small icons: the Bile Demon's (placeholders).
- Text: new name, tooltip and description strings appended to the text files with fresh IDs. The string file format (`Data/Text/Default/*.str`) is inspected first. If loose text is not picked up, Molynx reuses the Bile Demon's string IDs until it is solved.

### 3. Map data

- A splice tool replaces or appends the Creatures block inside a `*Globals.kld` and rewrites that block's header. Every other block is copied byte for byte.
- Test maps: Dead Walking and Siege (see Amendments), patched with the 47-record block and one Molynx placed in its Things file. The original map is never modified; a copy goes in the resource folder.
- The edited `Creatures.kwd` is also written for the user's map editor, if the editor accepts it.

### 4. Model and animation (`scripts/dk2kmf_write.py`)

- A KMF writer: mesh, material list and the per-vertex keyframe animation format (10 bits per axis, per-128-frame block tables) that `dk2kmf.py` already reads.
- Check: decode the encoder's output with the existing reader and require the original frames within the format's precision. A second check re-encodes one existing creature animation from the game and compares it with the original.
- Molynx's 25 clips become named animations on a shared mesh with the body part and the mace part.
- Output: loose `.kmf` files in `Data/Meshes/` of the resource folder.
- Texture: one PNG for the body and one for the mace in `Data/EngineTextures/`, sized to what the engine accepts (confirmed by the tests, with 128 px as the ceiling, see Amendments).

### 5. Icons

- The panel portrait is the user's uploaded image, converted to the game's portrait format. That format has not been inspected yet; the first implementation step for this part is reading the equivalent Bile Demon portrait out of `Sprite.WAD` / `FrontEnd.WAD` and matching it.
- All other icons are the Bile Demon's.

### 6. Test ladder

Each test needs the previous one to pass, so a failure points at one cause. The user runs them in the game.

1. **Slot test.** 47th creature, an exact Bile Demon copy (different name and ID only). No new files. Passes if the test map loads, Molynx spawns and walks, can be picked up and dropped, and the creature panel opens.
2. **Visual test.** Same record plus Molynx's model, textures and animations.
3. **Behaviour test.** His icon, stats and tuned behaviour. The user plays him through a full cycle: portal, lair, eating, payday, training, being slapped, being dropped, fighting.

### 7. Fallback

If Test 1 fails on the creature ID, the cause is found in the game program's code (the static analysis is already set up) and we choose between patching it and using Horny's slot with the same record, model and icon. No work from parts 1 to 5 is lost.

## Out of scope

- Redrawn icons (placeholders only).
- New sounds and speech.
- Adding him to campaign maps' portal pools beyond the test map, and to the skirmish creature list. These come after Test 3.
- Network and multiplayer compatibility (Flame's `original-compatible` stays off).

## Risks

- Another table sized to 46 creatures may crash the game when ID 47 appears. Test 1 finds it; the analysis then decides between patching and the fallback.
- A loose `.kmf` may not load. If so, the model goes into a rebuilt `Meshes.WAD` using the WAD reader we already have plus a writer.
- Texture size limits. 128 px is the engine ceiling (see Amendments).
- The writer's animation encoding can be wrong in a way the reader cannot see. The re-encode check on a game animation covers this.

## Amendments (as built, Tasks 1-8)

Where the text above differs, this section is what was built:

- The file is committed (the repo is `/home/claude`, base 1826ae7).
- **Animation slots (section 2):** a slot that is empty in the Bile Demon stays empty; every filled slot names a Molynx file and gets its tail (frames, fps 30, flags, type) rewritten to describe that file. `stunnedPose` and `elec` stay static pose slots (type 5) like all game creatures.
- **Test maps (section 3):** Dead Walking and Siege (Siege is multiplayer only). The edited editor `Creatures.kwd` is NOT written yet (pending, after Test A).
- **Model and animation (section 4):** one KMF per clip (24 shipped), keyframes reduced like 661 game files; poses are static meshes. Textures are 128 x 128 (the ceiling; there is no 256 px fallback).
- **Test ladder (section 6):** Tests 1 and 2 are bundled as Test A: two folders, `Molynx_slot_only` (exact Bile Demon copy) and `Molynx` (his model and animations), enabled ONE at a time. The folder must be listed FIRST in `alt-resources`, because Flame uses the first listed folder that has a file.
- Risks: loose `.kmf` loading and `Level =` with the test maps are untested in the game; a loose texture with a brand-new name is likely fine (Quuz_UniqueBatTex already loads `Unique_Bat.png`, which is in no cache).
