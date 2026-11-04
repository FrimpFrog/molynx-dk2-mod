# Molynx - Test 1 (slot only)

This is the first test. Molynx is only a **copy of the Bile Demon** here, with the name "Molynx" and
creature number 47. He does not have his own look yet. We only want to know: **does the game accept a
47th creature?** It might crash, and that is a useful answer too.

Nothing in your real game files is changed. The test is one folder, `Molynx_slot_only`, that Flame
loads on top of the game. Remove it and the game is as before.

## What to do

1. **Copy the folder.** Copy `Molynx_slot_only` into `<DK2>\flame\resources\`
   (next to `Quuz_LevelFixes`, `Quuz_HD` and the others).

2. **Turn it on.** Open `<DK2>\flame\config.toml` in Notepad. Find the line that starts with
   `alt-resources` (in the `[flame]` part). Today it reads:

   ```
   alt-resources = "Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```

   Put `Molynx_slot_only;` **at the very start**, inside the quotes:

   ```
   alt-resources = "Molynx_slot_only;Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```

   Why first, not last: Flame reads the list from the left and uses the first folder that has a file.
   `Quuz_LevelFixes` has its own copy of the Siege map, so ours must come before it.
   Only **one** Molynx folder may be in the list at a time. Save the file.

   (You can do the same in the Flame start window: double-click `Molynx_slot_only` so it moves to
   *Enabled*, then drag it to the top.)

3. **Start the test map.** The two test maps are **Siege** and **Dead Walking**. Start with Siege.
   - Quickest: in `config.toml`, in the `[dk2]` part, find the line `Level = ""` and make it
     `Level = "Siege"`. Start the game as usual. It should go straight into the map.
     (I have not been able to run this myself, so it may just show the menu.)
   - Or pick the map **Siege** in the Skirmish list. On your install it may not be listed
     (your Quuz copy is not marked as a skirmish map), which is why there is a second map.
   - If Siege does not start (for example you only get the menu), use **Dead Walking** the same
     way: `Level = "Dead Walking"`. This map is a normal skirmish map, so it is also in the
     Skirmish list. If the list shows it under another name, use the `Level` line.

4. **Look at these things** (about 5 minutes is enough):
   - The map loads and you can play.
   - One of the creatures that start in **your** dungeon (Keeper 1) is called **Molynx**. Hold the
     mouse over your creatures or open their panels. He looks like a Bile Demon.
   - He **walks** around.
   - You can **pick him up** and **drop** him.
   - His **creature panel opens**.

   **Pass:** all of that works and the game keeps running.
   **Fail:** the game crashes or freezes (while loading, or when you pick him up or open his panel),
   he is invisible, or there is no creature called Molynx.
   If every creature is a normal one and none is called Molynx, the folder is probably not first
   in the list (step 2).

5. **Send me:**
   - `<DK2>\flame\latest.log`. Copy it **right after you close the game**; the next start
     overwrites it.
   - A screenshot (Molynx on screen, if you can).
   - The exact moment of any crash: while loading, on the map, when picking him up, when opening his
     panel, and so on.
   - If a new file appeared in `<DK2>\flame\CrashInfo\`, send the newest one too.

6. **If a file-checksum error appears,** open `config.toml` again, find `NoFileChecksum = false`
   (in the `[dk2]` part), make it `NoFileChecksum = true`, save, and try again. Tell me you did.

## When you are done

- Remove `Molynx_slot_only;` from the `alt-resources` line (so it is the old line again).
- Put `Level = ""` back, so the game starts normally.
- You can delete the folder from `<DK2>\flame\resources\` if you like.
