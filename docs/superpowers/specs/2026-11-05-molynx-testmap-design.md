# Molynx test map: design

Date: 2026-11-05. Status: approved by the user's request ("make a completely flat map with a few gems, a portal and the Enemy keeper trapped by impenetrable wall so I can test the creature without being attacked and digging everywhere"). The user's DK2 folder holds the map editor and all shipped maps.

## Why

Test A (Molynx in Dead Walking / Siege) loaded without a crash, but Molynx was not seen at the start and the user had to wait for portal arrivals. Placed creature things may not be honoured by a Skirmish start, and the shipped maps mean digging, fighting and waiting. A purpose-made map removes all of that.

## What the user gets

A new map, "Molynx Test", plus a variant "Molynx Test Portal" (Molynx is also in the portal pool), delivered inside the two existing mod folders (`Molynx` stage 2 and `Molynx_slot_only` stage 1, so each folder's map carries that folder's record 47). "Dead Walking" in the same folders is replaced by the same flat map under the shipped name, because a file that replaces an existing map is certain to be found by the game, while a brand-new map name may not appear in the Skirmish list (Flame's alt-resources layer is only known to override files that the game opens by name). Siege stays as in Test A.

The map (base: the shipped 2-keeper skirmish map "Dead Walking", 69 x 69, owners 3 = Keeper 1 (human), 4 = Keeper 2 (AI)):

- Completely flat: one big open floor area for Keeper 1, no hills, no water, no lava, no hero gates, no traps, no doors, no neutral or good creatures, no action points the script would need.
- Keeper 1: Dungeon Heart, a Portal, a large pre-claimed area (rooms can be built at once), at least 30 x 30 open floor, 4 to 6 gem tiles and 2 gold tiles each next to the open floor, 6 Imps, Molynx placed next to the heart, 100000 starting gold, every room, door, trap and spell enabled.
- Keeper 2: its Dungeon Heart (and what the base puts with it) in a sealed chamber walled by Impenetrable Rock at least 2 tiles thick on every side, no way out, no way in. A few Imps only.
- Variants differ only in the Variables file (portal pool).
- Win/lose script: none (an empty Triggers file), so nothing ends the game.

## Rules

- Reuse verified structures: hearts, portal and their objects are copied from the base map's own bases, not invented. Every file parses with `read_blocks`; header fields follow the container rules; the generator is deterministic.
- Placed Molynx = creature thing with creatureId 47 owned by player 3, like Test A. The Globals file of each generated map is the base map's Globals spliced with that folder's record 47 (the existing `splice_creatures` + `molynx_for_map`).
- The audit stays independent of the generator (own parsing, own literals).
- Nothing about Molynx or Behemoth goes into the dk2_tools zips, the viewer site or any published artifact.

## Risks (accepted, to be found by the in-game test)

- A brand-new map name may not show in the Skirmish list (hence the "Dead Walking" replacement and the `Level = "Molynx Test"` route).
- An empty Triggers file may change how the game starts or ends a skirmish. The Players/Variables/Things layout is copied from shipped maps, but this exact combination has never been run.
- The portal-pool variant can make the game read the per-player entrance table (46 entries, found in the game's player structure) at index 46. This is the real "47th creature" test; if it crashes when he enters, that is a useful answer.
- Pre-claimed floor might not be accepted for building until imps "claim" it; the map uses the same Claimed Path tiles the shipped maps use around hearts.

## Out of scope

Pre-built rooms, scripted tutorials, a bigger or prettier map, multiplayer.
