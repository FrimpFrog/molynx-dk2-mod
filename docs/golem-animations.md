# Golem animations (red-moss Golem, 35 in the viewer)

The Golem is a brute like the Bile Demon, so it has the Bile Demon's kind of animations: no spell casting and no research.

The Golem keeps its own MegaGlest `stand`, `walk`, `attack`, `die` and `build_up` (made for this body, real engine speeds).
The other 30 are new. Origin of each (kind): **dk2** = a DK2 animation of the same meaning (mostly the Bile Demon), retargeted; **golem** = built from
the Golem's own MegaGlest walk/attack; **hand** = no DK2 equivalent, posed by hand on the same skeleton.

| Animation | Kind | Source |
|---|---|---|
| fidgeting | dk2 | Bile Demon idle2 |
| eating | dk2 | Bile Demon eat |
| praying | dk2 | Bile Demon pray |
| happy | dk2 | Bile Demon happy |
| angry | dk2 | Bile Demon angry |
| celebrating | dk2 | Bile Demon dance (first half) |
| victory_celebration | dk2 | Bile Demon dance, then happy |
| manufacturing | dk2 | Bile Demon building |
| getting_hit | dk2 | Bile Demon recoil (hit from front) |
| being_slapped | dk2 | Bile Demon recoil (hit from behind) |
| staggering | dk2 | Bile Demon drunk idle |
| waking_up | dk2 | Bile Demon getup |
| knockback | dk2 | Bile Demon fallback, then getup |
| sleeping | dk2 | Bile Demon fallback, then breathing |
| guarding | dk2 | Skeleton attention |
| interacting_with_objects | dk2 | Dwarf claim wall (amplified) |
| running | golem | MegaGlest walk, faster, leaning forward |
| turning | golem | MegaGlest walk steps while the body turns 100 degrees |
| patrolling | golem | MegaGlest walk, upper body scanning left and right |
| training | golem | MegaGlest attack, three swings |
| special_attack | golem | MegaGlest attack, slower, with a lunge |
| fighting_another_creature | golem | MegaGlest attack, taking a hit, attacking again |
| looking_around | hand | torso turns left and right |
| frightened | hand | cowering and trembling |
| blocking | hand | forearms crossed, one impact shake |
| unhappy | hand | slump and slow head shake |
| picking_up | hand | bow, grab, lift to chest |
| dropping | hand | open hands, let go |
| being_picked_up | hand | dangling, limbs swaying |

## How the retarget works
DK2 animation is per-vertex (no skeleton). `scripts/retarget.py` measures each limb's rotation per frame (Kabsch fit of each
donor segment to its rest pose) and `scripts/golem_retarget.py` applies those rotations to a skeleton on the Golem (root, torso,
two upper arms, two forearms, two legs) with smooth skin weights, soft rotation limits (a heavy golem does not whip its limbs),
in-between frames when a limb turns too fast, and ground planting. The Giant was tried first and looked wrong on the Golem's
hunched body; the Bile Demon (wide, heavy, bent arms) fits much better.

## Sounds
The viewer plays the Golem's MegaGlest sounds (`scripts/golem_sounds.py`): walk/attack/hit/die exactly as in `golem.xml`
(the hit lands at the attack's 0.4 mark), and the select/acknowledge growls on angry, happy, victory and similar animations.
Use the "Sound" button in the viewer header (off by default). Sounds: MegaGlest data, CC BY-SA 3.0.

## Not done
Writing these into the game (KMF with per-vertex frames, WAD repack, creature registration) is still deferred.
