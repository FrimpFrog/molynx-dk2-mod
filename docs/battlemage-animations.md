# Battlemage animations

The MegaGlest Battlemage (Magic faction, CC BY-SA 3.0 data) is in the viewer in his original colours (yellow hat and tunic,
red crab emblems; `scripts/battlemage_texture.py`). His own five MegaGlest animations are kept at the engine's real speeds
(`stand` 40 %, `walk` 100 %, `attack` 60 %, `morph` 80 %, `die` 50 %), and his MegaGlest sounds play on matching animations
(Sound button, off by default).

Everything else is built from real DK2 mage animations. The donor is the DK2 Priest (clean rigid limbs, and it has magic, research, pray, melee, run, drunk, fall and get-up clips); the Dark Elf supplies the special attack. Warlock and Wizard were tried first and rejected: their robes deform too much, so kneels came out straight-legged.

| animation | origin | source | length | |
|---|---|---|---|---|
| fidgeting | dk2 | Pr1 idle2 | 2.2 s | loop |
| running | dk2 | Pr1 run | 0.8 s | loop |
| eating | dk2 | Pr1 eat | 4.3 s | loop |
| praying | dk2 | Pr1 pray | 1.4 s | loop |
| researching | dk2 | Pr1 research | 2.0 s | loop |
| happy | dk2 | Pr1 happy | 3.3 s | loop |
| angry | dk2 | Pr1 angry | 2.6 s | loop |
| celebrating | dk2 | Priest dance, first half | 1.6 s | loop |
| victory_celebration | dk2 | Priest dance, then happy | 6.6 s | once |
| casting_spell | dk2 | Priest magic | 2.7 s | once |
| special_attack | dk2 | Dark Elf magic | 2.1 s | once |
| getting_hit | dk2 | Priest recoil (hit from the front) | 1.2 s | once |
| being_slapped | dk2 | Priest recoil (hit from behind) | 1.3 s | once |
| staggering | dk2 | Priest drunk (staggering) | 2.1 s | loop |
| waking_up | dk2 | Pr1 getup | 1.2 s | once |
| knockback | dk2 | Priest fallback, then getup | 2.0 s | once |
| sleeping | dk2 | Priest fallback, then breathing | 4.3 s | once |
| training | dk2 | Priest staff swing, three times | 4.2 s | loop |
| fighting_another_creature | dk2 | Priest and Dark Elf melee with Priest recoils | 6.4 s | loop |
| turning | dk2 | Priest walk steps while the body turns 100 degrees | 1.3 s | once |
| patrolling | mix | Priest walk with the head scanning left and right | 1.0 s | loop |
| guarding | mix | Priest idle with a slow head scan | 6.3 s | loop |
| being_picked_up | dk2 | Priest swing (the animation DK2 plays when a creature is picked up) | 2.0 s | loop |
| looking_around | hand | hand-posed head and shoulder turns left and right | 6.0 s | loop |
| frightened | hand | hand-posed cowering with trembling | 1.6 s | loop |
| blocking | hand | hand-posed guard: forearms crossed in front, one impact shake | 2.2 s | once |
| unhappy | hand | hand-posed slump, head down and a slow head shake | 6.0 s | loop |
| picking_up | hand | hand-posed bow, grab, lift to the chest | 3.2 s | once |
| dropping | hand | hand-posed open hands, let go | 2.2 s | once |
| manufacturing | hand | hand-posed work at a bench: forearms hammering alternately | 3.0 s | loop |
| interacting_with_objects | hand | hand-posed reach and pull, like working a lever | 3.0 s | loop |

Origins: `dk2` = a DK2 clip of the same meaning retargeted onto the Battlemage; `hand` = hand-posed with the same skeleton;
`mix` = DK2 motion plus a hand-posed head scan, or hand-posed play followed by real DK2 reactions.

## Gambling and the casino: not made

DK2's `Meshes.WAD` has the casino props (`RouletteTable`, `RouletteWheel`, `Bar_Fruit_Machine`, `Casino_Pillar`, `Coin`, `Bar_Table`) and
the engine has casino code (`CasinoState`, `SJ-Casino`, "big winner is slapped"), but there is **no creature animation file for gambling**:
creature files are Idle1/2, Walk, Run, Melee, Magic, Pray, Research, Drink, Drunk, Eat, Dance, Happy, Angry, Recoil, Fallback, Getup, ...
The creature-to-animation mapping lives in game data outside `Meshes.WAD`, so which clips the game plays at the casino is not known from
these files. Since no such animation exists, none is invented: the Battlemage (and the Golem) have no `gambling` animation.
If you can tell me which existing clips the casino uses, they can be added.

## Method

Same as the Golem (`golem-animations.md`) with a humanoid rig (`scripts/battlemage_rig.py`, `scripts/humanoid_retarget.py`):
the donor body is split into rigid segments (torso, head, arms, forearms, legs, shins), each frame's rotation is measured against the
donor's rest pose (Kabsch fit) and applied to the matching bones of the Battlemage; the mesh follows by linear-blend skinning.
Rest pose = frame 0 of `battlemage_dying.g3d`; the donor's rest pose is its standing idle (not a walking frame, which made every clip scissor the legs). Skin weights come from the texture's UV islands and colours, so cape, arms and legs are separated properly. Body tilt comes from the legs only, and the root rotation is exact so a fall lies flat; the lowest body point is planted on the floor each frame. The tunic's whole side edge (belt ends, dagger, side seam, |x| up to 0.27) is torso and takes no arm weight below the shoulder; the yellow shoulder cloth follows the torso half way. 27 small seam triangles between the sleeve and the tunic side are left out of the mesh for the retargeted clips (`stretched_triangles`): hidden against the body at rest, but a raised arm turned them into a wing of cloth between wrist and hip. Shins have their own bones, so kneeling clips (praying, falling,
getting up) fold the legs. The cape has its own bone and a small procedural swing (it hangs plumb as he leans, flares when he strides) and rests on the floor in falls. Timing: DK2 clips keep their length at an assumed
30 frames per second.

## Not done

* A gambling animation (does not exist in the DK2 files, see above).
* The cape is only about 28 vertices, so it swings stiffly rather than flutters.
* Hand-posed animations are approximations; the low-poly arms limit how far the elbows bend.
* Not yet converted into a DK2 creature (KMF writer, animation encoder, WAD repack: deferred).
