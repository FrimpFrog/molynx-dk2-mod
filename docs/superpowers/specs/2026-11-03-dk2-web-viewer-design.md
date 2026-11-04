# DK2 creature web viewer — design

Date: 2026-11-03. Status: awaiting user review. (/home/claude is not a git repo, so this file is not committed.)

## Purpose

A web page for browsing Dungeon Keeper 2 creatures: pick a creature, play its animations, orbit and zoom, and switch between the original textures and an HD texture pack.

This is a stepping stone. The user's main goal, creating their own creatures in game, comes later. The data format therefore has to be something a custom creature could also be exported into.

## What the user asked for

- A web viewer ("Can you make a webviewer").
- An HD texture pack toggle inside it.
- The HD pack made both ways: a quick non-AI sharpened pack now, and a folder the user can later fill with AI-upscaled textures from their GTX 1070.

## Assumptions (user can correct)

- First scope is creatures and heroes with their animations. Rooms, traps and objects are later.
- Playback is 30 fps. The game's real frame rate is not confirmed.
- The HD pack is a folder of PNGs with the same file names as the originals.
- The page is private. The assets belong to EA and come from the user's own copy; they are not for public redistribution.

## Existing pieces this builds on

All in `/home/claude/scripts`: `dk2wad.py` (WAD extraction), `dk2kmf.py` (KMF parsing, `evaluate_anim_frame`), `dk2texture.py` and `dk2texture_tables.py` (texture decoder, GPL-3.0), `kmf_textured.py` (reference renderer, `fixed_view`). Extracted game data is in the session scratchpad (`extracted/Meshes`, `dk2game/Ai/EngineTextures.*`).

## Parts

### 1. Exporter (Python)

Reads a creature's static pose file and each of its animation files and writes one binary file per creature plus a JSON index.

Per creature file:
- Parts, each with a texture name, UVs, triangle indices (LOD 0) and normals.
- Per animation: name, frame count, and the vertex positions of every part for every frame, stored as 16-bit integers scaled to the creature's bounding box. Frames are evaluated in Python with `evaluate_anim_frame`, so the browser only blends between stored frames and does not implement the keyframe decoder.

The index lists creatures, their animations and the texture names each needs. The layout is the same for every model, so a custom creature can be exported into it later.

Creature-to-file mapping is by file-name convention (for example `Goblin-Walk`, `GOB-dance`, `Bile_Demon-Melee`, `REA3_Walk`). The exporter reports any animation file it cannot attribute to a creature rather than guessing.

### 2. Texture sets

- `textures/original/`: decoded textures at full size as PNG, named by texture name.
- `textures/hd/`: the HD pack, same file names.
- The viewer loads from the folder the toggle selects and falls back to the original for any missing HD file.

### 3. HD pack builder (Python)

Makes a 4x pack without AI: Lanczos resize plus a mild unsharp mask. It is labelled "Enhanced (non-AI)" in the viewer. Output goes to `textures/hd/`. The user can later replace those files with AI-upscaled ones (for example from Real-ESRGAN on their own PC) with no viewer changes.

### 4. Viewer (HTML + three.js)

- Left: creature list. Right or bottom: animation list for the chosen creature.
- Centre: 3D view with orbit and zoom, fixed camera framing across an animation so it does not jitter.
- Controls: play and pause, frame scrub, speed slider, loop.
- Top: Original / HD toggle, showing which pack is active.
- Animation: the viewer blends between the stored frames each tick, the way the game does, and updates vertex positions.

## Testing and verification

- Exporter: a test that exports a creature, reads the file back and checks vertex positions equal `evaluate_anim_frame` within the 16-bit rounding error, on several frames including past frame 128.
- Pack builder: every HD file is exactly 4x the original size and has the same name.
- Viewer: render a few chosen frames in a headless browser (Playwright is installed) and compare them to the Python reference renderer by eye, plus a check that the page loads with no console errors and that the toggle changes the texture source.
- Honest limit: the non-AI pack will be smoother but adds no real detail, and I have not yet seen how it looks.

## Delivery

- Private artifact page. Size is checked before publishing (the page and each file have size limits). If it is too large, ship fewer creatures first and say so.
- Also provide the same files as a zip so the viewer can run locally from a folder.
- First pass: Goblin, Bile Demon, Horned Reaper. Widen to all creatures and heroes once those match the reference renders.

## Out of scope for now

Rooms, traps, objects, sound, camera paths, writing models back to KMF, repacking WADs, creature registration. These belong to the custom-creature work.

## Risks

- Total size for all creatures may exceed the page limits; mitigated by quantisation and staged scope.
- Some animation files may not map cleanly to a creature name; the exporter lists them for the user instead of guessing.
- Non-AI HD pack may disappoint; mitigated by the drop-in folder for AI upscales.
