# MegaGlest: how its models, textures and animations work

Everything below was checked against the files in your MegaGlest folder, the official MegaGlest data
(github.com/MegaGlest/megaglest-data, CC BY-SA 3.0) and the engine source (github.com/MegaGlest/megaglest-source, GPL-3.0).
Your zip has an empty `techs/` folder (no factions), so the Golem came from the official data repository
(`techs/megapack/factions/magic/units/golem`).

## Models: G3D
- One file = one model made of meshes. Versions 3 and 4 exist. Of the 620 `.g3d` files in your zip, 617 are v4, 2 are v3
  and 1 is an old v2; the v3/v4 ones all parse, with every index in range. The Golem's files rewrite byte for byte.
- Mesh header: name, frame count, vertex count, index count, diffuse/specular colour, opacity, property flags
  (1 team-colour alpha, 2 two-sided, 4 not selectable, 8 glow) and texture flags (diffuse / specular / normal map).
- Data: vertices for every frame, normals for every frame, one UV set, 32-bit triangle indices. No skeleton, no bones.
- Look: plain low-poly meshes. The magic faction's units run 250-1,000 triangles (Golem 398 triangles / 407 vertices;
  Archmage 472, Battle Mage 602, Behemoth 779, Dragon Rider 984). Tileset objects have a median of 242 triangles.

## Textures
- Ordinary image files next to the model (PNG, TGA, BMP, JPG). The Golem uses a 256x256 RGBA PNG.
- Engine texturing is plain diffuse mapping, with optional specular and normal maps per mesh.
- The alpha channel of a mesh flagged "custom colour" marks the pixels the engine paints in the player's team colour
  (the Golem's glowing runes). The viewer bakes a team colour (blue) into those pixels.
- Because these are normal PNGs, an AI-upscaled replacement is a straight file swap (the viewer's HD pack toggle works on them too).

## Animation
- Vertex-morph animation like Dungeon Keeper 2: each frame stores every vertex position. The engine blends linearly between
  neighbouring frames and wraps the last frame back to the first for repeating skills.
- Each skill in `golem.xml` names its own animation file and an `anim-speed`. Engine rule (unit.cpp, 40 updates per second):
  one cycle takes 100 / anim-speed seconds, so frames per second = frames x anim-speed / 100 (die: (frames - 1) x anim-speed / 100).

| Skill | File | Frames | anim-speed | Plays at | One cycle |
|---|---|---|---|---|---|
| stop (stand) | golem_standing.g3d | 12 | 50 | 6 fps | 2.0 s, loops |
| move (walk) | golem_walking.g3d | 15 | 20 | 3 fps | 5.0 s, loops |
| attack | golem_attacking.g3d | 15 | 60 | 9 fps | 1.67 s, loops |
| die | golem_dying.g3d | 11 | 60 | 6 fps | 1.67 s, once |
| be_built | golem_construction.g3d | 1 | 300 | n/a | progress-driven |

`stone_model.g3d` (26 vertices, 1 frame) and `texture_stone.png` are a small stone block that sits with the Golem's files
but is not referenced by `golem.xml`.

## Tools in this folder
- `scripts/g3d.py`: read and write G3D v3/v4. `scripts/glest_export.py`: turn a unit folder (xml + models + textures) into the web viewer's format.
- `python3 -m scripts.build_viewer <site> <Meshes> <EngineTextures.dir> <EngineTextures.dat> --glest=golem:Golem:<path to unit folder>`
- MegaGlest's own Blender add-on (`blender/g3d_support.py` in your zip) can import and export G3D directly.
