# Dungeon Keeper 2 Cinematic Remaster: 3D-Guided AI Rendering

Date: 2026-11-02

## Intent

Turn a 30-second Dungeon Keeper 2 CGI cinematic clip into a version that looks like a modern high-detail render (reference look: realistic skin pores, scratched metal, lit flame), as a video file. It is a personal project using the user's own game files.

Not in scope: a playable game asset, redistributing the output, or the longer 97-second ending clip (kept as a possible later project).

## Source and constraints

- Source clip: `Dungeon_Keeper_Remastered_Video.mp4`. 1920x1080, 30 fps, 30.2 s, 906 frames, stereo AAC audio. The picture appears to be 4:3 with black side bars (real image about 1440x1080); crop bars before processing, confirm with cropdetect.
- Content: the Reaper, a silver-armoured knight, a wizard in a wide hat, an old white-bearded wizard, and fire in nearly every shot. About 10 shots.
- Hardware: NVIDIA GTX 1070 (8 GB VRAM, Pascal, no tensor cores, slow fp16). Use SD 1.5 or SDXL-class models with ControlNet. Newest large video models are out of scope.
- Estimated render time: 1 to 2 minutes per final frame, so about 15 to 30 hours for the full clip.
- Claude's workspace has no GPU and probably cannot download AI model weights. Blender work (via the `bpy` pip module) and scripting happen there. Heavy AI generation and final renders run on the user's machine from scripts and settings Claude provides.
- Upscaling alone cannot invent the missing detail. Detail comes from the AI generation stage, guided by 3D or estimated geometry.

## Pipeline

1. **Prep.** Crop bars, extract frames, split into shots, light deblock and denoise. Skip a separate upscale since the clip is already 1080p.
2. **Geometry guidance.** Two tiers:
   - Tier A (Reaper hero shots): a proxy Reaper model in Blender, camera and pose matched to the footage, rendering depth, normal, edge and mask passes per frame.
   - Tier B (everything else: knight, wizards, sets, fire): depth and edge passes estimated directly from each frame with a depth-estimation model. No hand-built 3D.
3. **Character lock.** Train a small LoRA on a few enhanced hero frames so the Reaper stays recognisable. Training runs on the user's GPU.
4. **Generation.** Per frame, image-to-image seeded from the cleaned source frame at low denoise strength, with ControlNet from the guidance passes and the LoRA. Use fixed seeds, optical-flow warping from the previous frame, and keyframe propagation to limit flicker.
5. **Finish.** Tile-based detail upscale (1080p, 4K if it holds up), temporal deflicker, mux with the original audio.

Fire is treated as a separate problem: test it in the pilots and consider compositing fire if AI fire flickers.

## Reaper proxy model (Tier A)

- Sources: the in-game low-poly turnaround (front, side, back) for proportions; the high-resolution promo render and the figurine photo for surface detail, scythe, hooves and armour; clip frames for pose and expression. The fan-made War for the Overworld Reaper is a proportion cross-check only, not a base asset.
- Build in Blender: blocked-out body, armour, horns, hooves and scythe. It must be accurate in proportions, silhouette and pose, not in fine detail, because the AI stage adds the detail.
- Rig simply enough to pose per shot.
- Deliverable before any shot work: turntable renders (front, side, back, three-quarter, head close-up) for the user to check.

## Staged delivery

1. Reaper proxy model, with turntable renders for user review.
2. Pilot shot scaffolds: camera and pose match, guidance passes, user checks alignment against the footage.
3. AI render of a few frames per pilot shot, for look and real timings.
4. Full clip render, only when the user says go.

## Pilot shots (approximate times in the clip)

1. Reaper in flames, front-facing, about 18 s. Tests the proxy and fire.
2. Furnace and flames, about 15 s. Tests fire with no character.
3. Old wizard close-up, about 27 s. Tests a face with Tier B guidance.

## Risks

- Frame-to-frame flicker, especially in fire.
- Character drift between frames, mitigated by the LoRA and 3D guidance.
- GTX 1070 speed and VRAM limits, mitigated by smaller models and tiling.
- Model weights must be downloaded by the user, since Claude's workspace likely cannot.

## Success criteria

- Pilot frames visibly approach the reference look (detail, lighting) without changing the character's identity or the shot's composition.
- Fire is stable enough across a shot to watch without distracting flicker.
- Full clip renders in a few days of unattended GPU time, with a final file that keeps the original audio in sync.
