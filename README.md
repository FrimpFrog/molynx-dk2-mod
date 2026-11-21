# Soles Creatures Mod for Dungeon Keeper II

New creatures for Dungeon Keeper II, made in the game's own low-poly style and added next to the original
roster (nothing is replaced). Work in progress; the release is planned for December 2026.

This repository holds the **tools that build the mod** from your own copy of Dungeon Keeper II, plus the
design notes and tests. It contains no game files: everything the mod uses is read from your install
and written into a Flame resource folder called `MoreCreatures`.

## The creatures

| Id | Creature | Made from | How you get him |
|---|---|---|---|
| 47 | **Molynx**, the rock golem | MegaGlest's Golem/Behemoth, rebuilt for DK2 | Portal: a 5x5 Hatchery and a 5x5 Combat Pit |
| 48 | **Necromancer** | DK2's Black Knight, reworked | Portal: a Graveyard of 5x5 or more |
| 49 | **Orc** | DK1's Orc, on DK2's Troll | Test maps for now |
| 50 | **Classic Reaper** | Quuz's DK1-style Horned Reaper design | Placed only, never through the Portal (in game he is still named "Quuz's Horned Reaper" for now) |
| 51 | **Dragon** | DK1's Dragon, rebuilt for DK2 | Test maps for now |
| 52 | **Sejanus**, the demon horse hunter | Original design | Test maps for now |
| 53 | **Gargoyle** | Original design | Test maps for now |
| 54 | **Krefft**, the Casino's loan shark | Original design, after Soles' own Krefft's river turtle | Test maps for now; his Portal lure is a Casino and a Treasury, each 5x5 or more |

## Where to find them

Molynx and the Necromancer can come through the Portal on every Skirmish and My Pet Dungeon map.

While the newer creatures (49 to 54) are being tested, the whole cast stands next to Keeper 1's Dungeon
Heart (at level 5) on a few test maps, **My Pet Dungeon level 1** among them, and they can come through
Keeper 1's Portal there (except the Classic Reaper, who is only ever placed).

At release every new creature will be on all Skirmish maps, the new campaign and My Pet Dungeon.

## Installing a built mod folder

You need Dungeon Keeper II (GOG) with the **Flame** mod loader. The mod is a Flame resource folder.

1. Copy the `MoreCreatures` folder into `<DK2>\flame\resources\`.
2. Open `<DK2>\flame\config.toml` in Notepad and put `MoreCreatures;` at the **start** of the
   `alt-resources` line in the `[flame]` part, for example:
   ```
   alt-resources = "MoreCreatures;Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```
   Flame takes each file from the first folder on that line that has it, so `MoreCreatures` must come first.
3. Start the game through Flame (the `DKII-DX` shortcut). The GOG launcher does not load Flame.
4. Start My Pet Dungeon level 1 to meet the whole cast.

To remove the mod, take `MoreCreatures;` out of the `alt-resources` line (and delete the folder if you like).
`docs/install-guide.md` has the step-by-step guide, including what to do about a file-checksum error.

## Building the mod from source

Python 3.11 or newer with `numpy`, `scipy` and `Pillow`. The scripts read the game's data from a staged
copy of your install (see `scripts/dk2paths.py`; set `MOLYNX_DK2_DATA` to its folder). The mod is built in
layers, each written on top of the last:

1. **Molynx (47)**: `scripts/build_molynx_release.py` writes the `MoreCreatures` folder with Molynx on
   every Skirmish and My Pet Dungeon map (his record, meshes, skins, panel pictures and his own sound bank).
2. **Necromancer (48)**: `scripts/necro_install.py` (its `install()`) adds him to every map the folder ships.
3. **The cast (49 to 53)**: `python -m scripts.cast_build ...` builds the Orc and the Classic Reaper, takes
   the approved builds of the Dragon, Sejanus and the Gargoyle, writes all their records into every map
   table, places the cast on the Skirmish test maps and checks the result (`scripts/cast_check.py`). Its
   docstring lists the arguments. `scripts/cast_install.py` explains how a later creature is added, and
   its `map_install` adds the cast to My Pet Dungeon level 1.
4. **Krefft (54)**: `python -m scripts.krefft_anims_build` builds his files, then
   `python -m scripts.krefft_install ... --files <those files>` adds him the same way (his record, his Coin
   Spit, texts, files and panel pictures, My Pet Dungeon level 1 included) and runs the same check.

Each step writes only the files that change into a new folder; nothing in the game folder is touched.

Run the tests with `python -m pytest tests`. Tests that need game data are skipped when it is missing.

## Repository layout

- `scripts/`: the builders, installers and checks, plus the DK2 file readers and writers (`.kmf` models,
  `.kld` / `.kwd` tables, `.str` text, WAD archives, sound banks)
- `tests/`: the test suite
- `docs/`: the test guide, animation notes and design documents
- `data/`: our own design art (Molynx's painted skins), a MegaGlest skull and the credits for Quuz's Reaper art
- `cinematic/`: the 1999-style Molynx cinematic and the tools that made it
- `flame_patch/`, `.github/`: a small Flame patch and the workflow that builds it

## How the textures were made

No textures from Dungeon Keeper II or from Quuz's pack are stored here. When you build the mod, the
scripts read them from your own copy and paint the new skins from them. (The only outside art kept here is
MegaGlest's, which its licence lets us share: the skull in `data/character_art/` and Molynx's skins, which
were painted over MegaGlest textures.)

- **Skins made from DK2's own:** the Orc is the Troll's skin recoloured to DK1's Orc, and the Necromancer is
  the Black Knight's skin repainted, keeping the game's own folds, rivets and plate edges as the shading.
  Molynx's mace uses small pieces of the Troll's mace, which `scripts/game_art.py` cuts from your game.
- **Skins painted from scratch:** the Dragon, Sejanus, the Gargoyle and Krefft are painted by our scripts
  in the game's style: every texel is traced back to the model and painted with light from above and
  shadow in the folds, like DK2's own hand-painted skins (the Gargoyle's stone uses the colours of the
  game's own gargoyle statue; Krefft is painted from photos of Soles' turtle, and his coins use the game's
  own coin and gold textures).
- **Molynx's red rock skin** (`data/molynx_design/`) was painted by us over MegaGlest's Golem and Behemoth
  textures, with MegaGlest's Battlemage belt.
- **The Classic Reaper** wears **Quuz**'s own art on DK2's Horned Reaper: his DK1-style reaper face,
  shoulder plates, chains, belt and loin flap from his "HD textures" pack for Flame. We turned the shoulder
  plates shiny silver and evened out the skin; the legs, scythe and the rest come from Horny's own
  textures. The scripts read Quuz's art from your copy of his pack (`flame/resources/Quuz_HD`), so you need
  it installed to build the Classic Reaper (a built mod folder already has his skins). Details in
  `data/reaper/CREDITS.md`.

## Still to come

- More new creatures
- Proper sounds for all the new creatures
- Balancing, bug fixes and polish from playtesting

## Credits

- **Mod by Soles**
- **Concept art:** @toadly_toons (https://www.instagram.com/toadly_toons/)
- **Special thanks:** the Flame team, the KeeperFX team, Keeper Klan, Quuz and Coolkid
- **The Classic Reaper's design and textures:** Quuz
- **Dungeon Keeper and Dungeon Keeper II** (the models, textures and sounds the mod is built from, read from
  your own copy): Peter Molyneux and the Bullfrog team, Electronic Arts
- **MegaGlest** (megaglest.org): Molynx's base model and head and a skull texture, under CC BY-SA 3.0; the
  parts of the mod made from them are shared under the same licence
- **OpenKeeper** (github.com/tonihele/OpenKeeper, GPL-3.0): `scripts/dk2texture.py` is a Python port of the
  texture decoder the OpenKeeper project worked out, and its notes on DK2's file formats helped throughout

## About AI

We are against most uses of AI, especially where it replaces artists. Dungeon Keeper II is an old game
with a small community and no funding, so this mod uses AI tools for a lot of the work (building
models and animations from the game's own files, the tools in this repository, testing and reviews).
The creature concepts, design decisions and final say are ours, and everything is checked in game.

## Licence and disclaimer

The code in this repository is GPL-3.0.

The Soles Creatures Mod is an unofficial, non-commercial fan modification for Dungeon Keeper II. It is
independently developed and is not affiliated with, endorsed by or supported by Electronic Arts, Bullfrog
Productions or their partners. Third-party assets remain the property of their owners and are credited
above. The mod does not include or distribute Dungeon Keeper II: a legitimate copy of the game is
required. AI-assisted tools were used in certain aspects of development, as disclosed above. The mod is
free of charge and made without commercial intent.
