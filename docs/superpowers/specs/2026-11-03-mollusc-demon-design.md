# Mollusc Demon (custom DK2 creature) — design

Date: 2026-11-03. Status: awaiting user review. (/home/claude is not a git repo, so this file is not committed.)

## Purpose

Design a new creature in Dungeon Keeper 2's own low-poly, dark style, as a 3D model with animations that the user can view and judge. This is the design stage only. Getting it into the game (writing a KMF, repacking a WAD, registering the creature) is a later, separate project, started when the user says the design is final.

## What the user asked for

- A new creature made by repainting textures and reusing parts from other creatures ("remix").
- Concept: a large, tanky, demon-like mollusc. A lungfish idea was dropped ("fish doesn't make sense in the underworld"). Inspiration images: a crustacean/fish monster concept painting (teal, purple, pearly colours) and an Australian lungfish drawing (scales, ancient tanky body).
- Base body: the **Giant**, with elements of the **Bile Demon**.
- For now: the 3D model and, if possible, animations. In-game work later.

## Assumptions (user can correct)

- Working name "Mollusc Demon", id `mollusc_demon`; the user can rename it.
- It appears in the existing web viewer as one more creature, with the Original/HD toggle working on its textures.
- Playback 30 fps, as in the viewer (unconfirmed against the game).

## Facts that shape the design

- DK2 models are split by texture, not by anatomy. The Giant has two body parts (back, front). The Bile Demon has front, back and a Spikes part (horns with spiked balls). The Salamander has a body and a tongue part.
- Animation is per vertex keyframes, with no skeleton. A new model gets the Giant's animations only if it keeps the Giant's vertices; extra geometry must be attached to follow chosen vertices.

## The creature

- **Body:** the Giant, with all its animations (32 found).
- **Skin:** wet, dark violet-teal with mottling.
- **Armour:** overlapping chiton-style shell plates over shoulders, chest and back; a ridged spiral shell pattern on the back texture.
- **Belly:** pale lilac with sucker-like dots (the Bile Demon influence).
- **Head:** a reshaped crest or mantle like a squid hood, heavier brow, yellow slit eyes.
- **Bile Demon parts:** its horns-with-balls part attached to the head, repainted pearly coral and barnacle-like.
- **Tentacles:** two, from the Salamander tongue part, hanging from the jaw. They follow the head; they do not wiggle on their own.

## Parts

### 1. Creature spec (data)

A JSON file `creatures_custom/mollusc_demon.json` describes the creature: base creature id and its animation set, texture edit recipes, reshape operations, and attachments. The build code reads it; nothing about this creature is hard-coded in Python.

### 2. Reshape

Moves existing vertices of the base mesh by a per-vertex offset (a vector for each vertex of the base). The offset is added to that vertex in every frame of every animation. Triangle count, indices and UVs do not change. Operations: select vertices by region (a box in the base pose, optionally smooth falloff) and apply move or scale about a point.

### 3. Attachments

A part taken from another creature's pose (geometry, UVs, texture) rigidly follows an anchor on the base. The anchor is three chosen base vertices; each frame, their positions define position and orientation, and the part keeps its starting relation to them. Each attachment has a scale, an offset and optional texture replacement.

### 4. Textures

New textures are generated from the Giant's two body textures at the original size, so UVs stay valid:
- Recolour (hue and saturation shift) plus overlay patterns (shell plates, spiral ridges, sucker dots) applied through masks painted by region of the UV layout.
- Parts taken from other creatures get their own recoloured copies.
- Export each texture as a PNG together with a UV-outline guide image so the user can repaint by hand. Hand-painted PNGs placed in the creature's folder replace generated ones.
- The HD pack builder already makes the 4x set from these.

### 5. Integration

`build_viewer` gains a "custom creatures" input folder. The exporter produces the same `.dk2v` as for game creatures (extra parts for attachments, new texture names). The viewer needs no change beyond listing the creature.

### 6. Outputs

- The creature in the web viewer (published page updated, and in the local build).
- GIFs: walk, dance, melee, idle.
- Textured OBJ + MTL + PNG of the pose, for opening in Blender.
- Texture PNGs with UV guides.

## Testing and verification

- Reshape: triangle count and index buffers equal the base; each animation frame equals base frame + offset within 16-bit rounding; a vertex outside every selected region does not move.
- Attachments: the distance between an attached part's points and its anchor stays constant across all frames of an animation (rigid), and its orientation follows the anchor.
- Textures: every generated PNG has the same size as its source; UV guides have the same size as the texture.
- Export: the creature appears in `index.json`, all 32 animations load, and no part is missing a texture.
- Visual: look at renders of the creature from several angles and in walk, dance and melee frames; the user reviews GIFs.
- Honest limit: patterns made by code are not hand-painted art; look quality is the user's call, and the textures can be repainted.

## Out of scope for now

KMF writing, WAD repack, creature registration (Creatures.kwd), sound, new animations, in-game testing.

## Risks

- Shell and sucker patterns may look crude when painted by code; mitigated by exporting editable PNGs and iterating with the user.
- Anchors on the Giant's head may jitter if chosen vertices move non-rigidly in some animations; mitigated by testing every animation and choosing a stable triple.
- Giant's UV layout may make region masks fiddly; I will look at the UV guide before choosing regions.
