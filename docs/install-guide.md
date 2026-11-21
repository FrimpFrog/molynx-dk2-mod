# Installing and testing the Soles Creatures Mod

The mod is one Flame resource folder, `MoreCreatures`. Flame loads it on top of the game, so nothing in
your own game files changes: take it out and the game is as before.

## What you need

Dungeon Keeper II (GOG) with the **Flame** mod loader. Nothing else: the folder already holds every new
creature's files.

## Install

1. **Copy the folder.** Copy `MoreCreatures` into `<DK2>\flame\resources\`.
2. **Turn it on.** Open `<DK2>\flame\config.toml` in Notepad. In the `[flame]` part, find the line that
   starts with `alt-resources` and put `MoreCreatures;` right after the first quote mark, for example:
   ```
   alt-resources = "Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```
   becomes
   ```
   alt-resources = "MoreCreatures;Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```
   `MoreCreatures` must be **first**: Flame takes each file from the first folder on that line that has it.
   Save the file. (Or in the Flame start window: enable `MoreCreatures` and drag it to the top.)
3. **Start the game through Flame** (the `DKII-DX` shortcut). The GOG launcher does not load Flame.

## Meet the creatures

- **Molynx** and the **Necromancer** can come through your Portal on every Skirmish and My Pet Dungeon map.
  Molynx is drawn by a Hatchery and a Combat Pit of 5 x 5 or more, the Necromancer by a Graveyard of
  5 x 5 or more.
- **The whole cast** (also the Orc, Classic Reaper, Dragon, Sejanus, Gargoyle and Krefft) stands next to
  your Dungeon Heart on **My Pet Dungeon level 1** while they are tested. Krefft is drawn by a Casino and
  a Treasury of 5 x 5 or more there.

## If something goes wrong

- **A file-checksum error:** open `config.toml`, find `NoFileChecksum = false` (in the `[dk2]` part),
  make it `NoFileChecksum = true`, save and start again.
- **A crash or freeze:** note what you were doing at that moment, then copy these straight after the game
  closes (the next start overwrites them) and send them to Soles:
  - `<DK2>\flame\latest.log`
  - the newest file in `<DK2>\flame\CrashInfo\` and in `<DK2>\flame\StackInfo\` (if there is one)

## Remove it

Take `MoreCreatures;` out of the `alt-resources` line (and delete the folder if you like).
