# Behemoth animations

The MegaGlest Behemoth (Magic faction, CC BY-SA 3.0 data; 728 vertices, 779 triangles, one 256x256 texture) is in the viewer in his
own dark red-brown colours (`scripts/behemoth_texture.py`; the team-colour pixels are baked red). His four MegaGlest animations are kept
as exported at the engine's real speeds (`stand` 9 frames at 50 %, `walk` 31 at 65 %, `attack` 24 at 75 %, `die` 8 at 100 %), and his
MegaGlest sounds play on them (Sound button, off by default). The spiked club is its own 142-vertex mesh part.

Everything else is a real DK2 animation retargeted onto him. The donor is the DK2 Giant (a clean rigid-limbed biped with run, eat, pray,
happy, angry, dance, melee, melee2, recoil, fall-back, get-up, drunk and swing clips); the Bile Demon supplies manufacturing (its
building clip), the Dwarf interacting (claim wall) and the Skeleton guarding (attention). Nothing is hand-posed, and a clip DK2 has no
animation for (casting, researching, blocking, frightened, unhappy, looking around, picking up, dropping, turning, special attack, gambling)
is not made.

| animation | source | length | |
|---|---|---|---|
| fidgeting | Giant idle2 | 2.1 s | loop |
| running | Giant run | 1.0 s | loop |
| eating | Giant eat | 2.8 s | loop |
| praying | Giant pray | 4.0 s | loop |
| happy | Giant happy | 2.0 s | loop |
| angry | Giant angry | 0.8 s | loop |
| celebrating | Giant dance, first half | 3.1 s | loop |
| victory_celebration | Giant dance, then happy | 8.2 s | once |
| getting_hit | Giant recoil (hit from the front) | 0.6 s | once |
| being_slapped | Giant recoil (hit from behind) | 1.2 s | once |
| staggering | Giant drunk (staggering) | 3.2 s | loop |
| waking_up | Giant getup | 1.5 s | once |
| knockback | Giant fallback, then getup | 2.0 s | once |
| sleeping | Giant fallback, then breathing | 4.0 s | once |
| training | Giant melee, melee2, melee | 4.5 s | loop |
| fighting_another_creature | Giant melee, melee2 with Giant recoils | 5.8 s | loop |
| patrolling | Giant walk | 1.4 s | loop |
| guarding | Skeleton attention | 0.8 s | loop |
| manufacturing | Bile Demon building | 1.1 s | loop |
| interacting_with_objects | Dwarf claim wall | 0.6 s | loop |
| being_picked_up | Giant swing (the animation DK2 plays when a creature is picked up) | 2.0 s | loop |

## Method

Same as the Golem and Battlemage (`golem-animations.md`, `battlemage-animations.md`), with `scripts/behemoth_rig.py`:
rest pose = frame 0 of `behemoth_standing.g3d` (club on the left shoulder), recentred on the feet. The four MegaGlest files number
their vertices identically (club first in the standing file; the dying file lists the body first, so the standing layout is used for
every added clip). Bones come from the body plan (legs below the hips with a shin split at y 0.62, arms outside the chest with a forearm split
half way from the shoulder, head above the neck); I checked the labelling by clustering the vertices by how they move across 72 frames of
the four native animations (rigid-segment fit, mean residual about 1 % of his height). Joints are the mean of the closest vertex pairs
between a segment and its parent. The club follows the forearm that holds it. The loincloth (hips down to y 1.0) takes half torso weight,
the shoulder pads half, so neither is dragged by a swinging limb. Skin weights are smoothed over the mesh with limb sharpening.
10 small triangles that join a hand resting on the hip to the loincloth are left out of the retargeted clips (`stretched_triangles`):
against the body at rest, long spikes as soon as the arm swings away.
The donor's rest pose is its standing idle; the body's lowest point is planted on the floor each frame, and the root rotation is exact so
falls lie flat. Timing: DK2 clips keep their length at an assumed 30 frames per second. Praying starts prone because the Giant's own
`pray` clip does.

## Towards a custom creature on this base

The viewer-side work (mesh, rig, skin weights, 21 animations, sounds) is the part that is done. Getting a creature of your own
into Dungeon Keeper 2 itself still needs, and these are deliberately deferred:
1. a KMF writer for the mesh and a per-vertex keyframe animation encoder (DK2 has no skeleton, just like MegaGlest),
2. new texture entries in `EngineTextures` (and the matching texture indices in the KMF),
3. a rebuilt `Meshes.WAD` containing the new files,
4. registering the creature in the game data (creature definition, animation names to clips, sounds).

## Not done

* A casting, research, block or any other clip DK2 has no animation for (see above).
* Hand-posed animations: none, by design.
* Not yet converted into a DK2 creature (items above).
