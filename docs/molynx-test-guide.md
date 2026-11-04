# Molynx - Test A

## What you are testing

Molynx is a new, 47th creature: a red horned brute with a mace. This test checks two things at once:

- **Does the game accept a 47th creature?**
- **Do his own model and animations work?**

He lives by the same rules as the Bile Demon in this test: what he likes, when he gets angry, how
long he stays stunned, his strength and his pay are all copied from the Bile Demon. His name and
panel picture are still the Bile Demon's in this test; they come next.

This is an honest first test. The game might crash, and that is a useful answer too. Nothing in your
real game files changes: the test is one folder that Flame loads on top of the game. Take it out and
the game is as before.

The zip holds two folders. Use **only one at a time**:

- `Molynx` - the real test (start with this one).
- `Molynx_slot_only` - a backup test, only for step 6.

## What to do

1. **Copy the folder.** Copy the folder `Molynx` from the zip into `<DK2>\flame\resources\`
   (next to `Quuz_LevelFixes`, `Quuz_HD` and the others).

2. **Turn it on.** Open `<DK2>\flame\config.toml` in Notepad. Find the line that starts with
   `alt-resources` (in the `[flame]` part). Today it reads:

   ```
   alt-resources = "Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```

   Put `Molynx;` **at the very start**, inside the quotes, so it reads:

   ```
   alt-resources = "Molynx;Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```

   It must be first: Flame reads the list from the left and uses the first folder that has a file,
   and `Quuz_LevelFixes` has its own copy of the Siege map. Save the file.

   (You can do the same in the Flame start window: double-click `Molynx` so it moves to *Enabled*,
   then drag it to the top.)

3. **Start the test map.** The test maps are **Siege** and **Dead Walking**. Start with Siege.
   - Quickest: in `config.toml`, in the `[dk2]` part, find the line `Level = ""` and make it
     `Level = "Siege"`. Start the game as usual. It should go straight into the map.
     (I could not try this myself, so it may just show the menu.)
   - Or pick **Siege** in the Skirmish list. It may not be listed on your install.
   - If Siege does not start, use **Dead Walking** the same way: `Level = "Dead Walking"`, or pick it
     in the Skirmish list.

4. **Look at these things.** Molynx starts in **your** dungeon (Keeper 1). For each point, "pass"
   means what is written after the dash.
   - **The map loads** - you can play, and the game keeps running.
   - **Find him** - a red horned brute with a mace, about 15% taller than a Bile Demon. (When you
     hover over him the name may still say "Bile Demon"; that is expected in this test.)
   - **He walks, stands and fidgets** - he moves his legs when he walks, and stands and shifts about
     when he is idle. He is not invisible, not white or grey, not stretched or torn.
   - **Pick him up and drop him** - in your hand he dangles. Dropped, he staggers dazed for a moment,
     then carries on.
   - **Slap him** - he flinches.
   - **Make him fight** - he swings his mace, flinches when he is hit, can be knocked back and gets
     up again. If he is killed, he falls and lies there.
   - **Put him to sleep** - in a lair he lies down and stays lying (see the quirks below).
   - **Eat** - in the hatchery he eats.
   - **Pray** - in a temple he prays.
   - **Work** - in a workshop he works.
   - **Moods** - when he is happy or angry he shows it.
   - **His creature panel opens** - click his picture or open his panel; the game does not crash.

   **Fail** is: a crash or freeze (while loading, or at one of the points above), Molynx invisible,
   white or grey, or a Bile Demon where Molynx should be. If every creature is a normal one, the
   folder is probably not first in the list (step 2).

5. **Send me:**
   - `<DK2>\flame\latest.log`. Copy it **right after you close the game**; the next start
     overwrites it.
   - A screenshot or two (Molynx on screen, if you can).
   - The exact moment of any crash: while loading, on the map, when picking him up, when he went to
     sleep, and so on.
   - If a new file appeared in `<DK2>\flame\CrashInfo\`, send the newest one too.

6. **If the game fails or crashes on start:** open `config.toml` again and take `Molynx;` out of the
   `alt-resources` line. Copy the folder `Molynx_slot_only` from the zip into
   `<DK2>\flame\resources\` and put it **first** instead:

   ```
   alt-resources = "Molynx_slot_only;Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```

   Try the same map again. In this folder Molynx is an exact copy of the Bile Demon: same look, same
   name. Tell me whether the map starts and plays. This tells me whether the trouble is the 47th
   creature itself or his model and animations.

7. **If a file-checksum error appears,** open `config.toml`, find `NoFileChecksum = false` (in the
   `[dk2]` part), make it `NoFileChecksum = true`, save, and try again. Tell me you did.

## Known quirks to watch

Some of his actions borrow the nearest animation he has. These are expected in this test:

- **Sleeping** plays once: he lies down and stays lying until he wakes up. If he keeps standing up
  and lying down again, or freezes standing, tell me.
- **Dazed after a drop** (and when stunned) uses his staggering animation.
- **Being electrocuted, tortured, or casting a spell** uses his flinch (being hit). When electrocuted
  the flinch repeats.
- **Coming out of the portal** uses his walk once; **walking backwards** uses his normal walk.
- **Drinking** uses his eating animation; **drunk** uses his fidgeting.
- His feet may slide a little when he walks.

## When you are done

- Take `Molynx;` (or `Molynx_slot_only;`) out of the `alt-resources` line, so it is the old line
  again.
- Put `Level = ""` back, so the game starts normally.
- You can delete the folder from `<DK2>\flame\resources\` if you like.
