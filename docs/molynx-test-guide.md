# Molynx - Test A, map 2 (the flat test map)

## What you are testing

Molynx is a new, 47th creature: a red, rocky, horned brute with a stone face and a mace (his new
look: the Golem-style head, the rock skin and the Battlemage belt, as tall as the Horned Reaper).
This test checks two things at once:

- **Does the game accept a 47th creature?**
- **Do his own model and animations work?**

On the first Test A maps he did not show up at the start. So this zip brings a new map made only
for this test: **Molynx Test**. It is flat and open. You have a Dungeon Heart, a Portal, 6 Imps, a few gems and
gold, 100000 gold to start, and every room, door, trap and spell. Molynx stands **right next to
your heart**. The enemy keeper is shut in behind thick Impenetrable Rock, so nobody attacks you and
there is nothing to dig except the gems and gold. There is no win or lose, so the game does not
end by itself.

He lives by the same rules as the Bile Demon in this test: what he likes, when he gets angry, how
long he stays stunned, his strength and his pay are all copied from the Bile Demon. His name and
panel picture are still the Bile Demon's in this test; they come next.

This is an honest test. The game might crash, and that is a useful answer too. Nothing in your
real game files changes: the test is one folder that Flame loads on top of the game. Take it out and
the game is as before.

The zip holds two folders. Use **only one at a time**:

- `Molynx` - the real test (start with this one).
- `Molynx_slot_only` - a backup test, only for step 8.

Each folder has three maps: **Molynx Test**, **Molynx Test Portal** (step 7) and **Siege** (as in
the first zip). While the folder is on, **Dead Walking** is also the flat test map (its own map is
back when you take the folder out).

## What to do

1. **Copy the folder.** Copy the folder `Molynx` from the zip into `<DK2>\flame\resources\`
   (next to `Quuz_LevelFixes`, `Quuz_HD` and the others). If the old `Molynx` folder from the first zip
   is still there, delete it first (do not copy over it: the new look has two more texture files
   and every model file is new).

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

3. **Start the map "Molynx Test".** There are two ways. Try the first one first.
   - **First try:** in `config.toml`, in the `[dk2]` part, find the line `Level = ""` and make it
     `Level = "Molynx Test"`. Save, and start the game as usual. The game should go straight into
     the map, like a campaign level, so the creatures placed in the map should be there.
     (I could not try this myself, so it may just show the menu.)
   - **If you get the menu instead:** put `Level = ""` back, start a Skirmish and pick
     **Molynx Test** in the list. It may be there twice: the flat map is also stored under the
     shipped name **Dead Walking**, so if you only see Dead Walking, pick that. While the folder is
     on, it is the same flat map.
   - Tell me which way you used.

4. **Look first: is Molynx next to your heart?** Your camera starts on your heart. Molynx should
   stand right beside it, with your 6 Imps around it.
   - Tell me: **were the Imps there at the start? Was Molynx there?**
   - If he is not there, tell me what is there instead (only Imps? a Bile Demon? nothing?). In
     **Molynx Test** he does not come through the portal; only the usual creatures do.

5. **Look at these things.** For each point, "pass" means what is written after the dash. Build a
   lair, a hatchery, a workshop, a temple and a training room for him (you have the gold).
   - **The map loads** - you can play, and the game keeps running.
   - **Find him** - a red, rocky, horned brute with a stone face, glowing angry eyes, bottom teeth
     over his lip and a mace on his shoulder, about as tall as the Horned Reaper (clearly bigger
     than a Bile Demon). (When you hover over him the name may still say "Bile Demon"; that is
     expected in this test.) If he has the old look (smooth skin, a harness with a skull), the old
     folder is still being used.
   - **He walks, stands and fidgets** - he moves his legs when he walks, and stands and shifts about
     when he is idle. He is not invisible, not white or grey, not stretched or torn.
   - **Pick him up and drop him** - in your hand he dangles. Dropped, he stands dazed for a moment
     (one still pose), then carries on.
   - **Slap him** - he flinches.
   - **Make him fight** - in the training room he swings his mace. There are no enemies on this
     map, so being hit, knocked back, getting up and dying are hard to see here; if you do see
     them anywhere, tell me how they looked.
   - **Put him to sleep** - in a lair he lies in his bed (the bed stays visible) and breathes.
   - **Eat** - in the hatchery a chicken appears; he picks it up with his free hand (not the mace
     hand) and eats it.
   - **Pray** - in a temple he prays.
   - **Work** - in a workshop he smashes his mace down on the bench, twice per cycle.
   - **Torture** - on the torture wheel both arms are spread the same way and his mace is gone.
   - **Stun** - stunned, he falls to a crouch and stays down until he gets up.
   - **Moods** - when he is happy or angry he shows it.
   - **Dance** - if he dances, it is his own dance (a stompy Troll-style dance, mace on his shoulder).
   - **His creature panel opens** - click his picture or open his panel; the game does not crash.

   **Fail** is: a crash or freeze (while loading, or at one of the points above), or Molynx
   invisible, white or grey, or his head or face missing, white or grey (his head has its own two
   pictures, `Molynx_Head.png` and `Molynx_Neck.png`). If he looks like a plain Bile Demon, the
   folder is probably not first in the list (step 2).

6. **Send me:**
   - Which way you started the map (step 3), and whether Imps and Molynx were there at the start.
   - `<DK2>\flame\latest.log`. Copy it **right after you close the game**; the next start
     overwrites it.
   - A photo or two (Molynx on screen, if you can). See "Taking screenshots and video" below.
   - The exact moment of any crash (while loading, when picking him up, when he went to sleep, ...).
   - If a new file appeared in `<DK2>\flame\CrashInfo\`, send the newest one too.

7. **Second map: "Molynx Test Portal".** The same flat map, but here Molynx can also **come through
   the portal**, like any creature. Start it the same way: `Level = "Molynx Test Portal"`, or pick
   **Molynx Test Portal** in the Skirmish list. Build a lair and a hatchery and wait at the portal.
   If you cannot find "Molynx Test Portal" in the list, skip this step and tell me.
   - Does a second Molynx walk out of the portal?
   - **Tell me if the game crashes then** (when he comes out, or when his panel or the creature list
     opens). That is a real answer: this is the riskiest part of the test.

8. **If anything fails** (a crash on start or later, for example when he sleeps): first copy
   `flame\latest.log` (the next start overwrites it). Then take `Molynx;` out of the
   `alt-resources` line, copy the folder `Molynx_slot_only` from the zip into
   `<DK2>\flame\resources\` and put it **first** instead:

   ```
   alt-resources = "Molynx_slot_only;Quuz_AudioFix;LevelFixes;Quuz_UniqueBatTex;Quuz_FullHD;Quuz_HD;Quuz_LevelFixes"
   ```

   Play the same map again. Here Molynx is an exact copy of the Bile Demon (same look, same name).
   Send me the new `latest.log` too, and tell me whether this one works. That shows whether the
   trouble is the 47th creature itself or his model and animations.

9. **If a file-checksum error appears,** open `config.toml`, find `NoFileChecksum = false` (in the
   `[dk2]` part), make it `NoFileChecksum = true`, save, and try again. Tell me you did.

## Taking screenshots and video

A photo of the monitor with your phone is fine. If you want real screenshots, run the game in a
window:

- **Windowed:** in the Flame start window, tick **Windowed mode** (in *Simple settings*). Or in
  `config.toml`, in the `[flame]` part, change `windowed = false` to `windowed = true`.
  (Flame turns its GOG patch off while the game runs in a window.)
- **A picture:** press **Win+Shift+S** and drag over the game window.
- **A video:** press **Win+Alt+R** to start and stop recording (the Xbox Game Bar), or use OBS with
  a **Window Capture** source. This is the usual Windows way; I could not try it with this game.

## Known quirks to watch

Some of his actions borrow the nearest animation he has. These are expected in this test:

- **Sleeping** is a breathing loop lying in the bed. If he floats above it or lies beside it, tell me.
- **Electrocuted**, he holds a still flinch pose or flinches. **Casting a spell** uses his flinch.
- **Coming out of the portal** uses his walk once; **walking backwards** uses his normal walk.
- **Drinking** uses his eating animation without the chicken; **drunk** uses his fidgeting.
- His feet may slide a little when he walks.

About the map:

- Nothing ends the game by itself (no script). Quit from the menu when you are done.
- The enemy keeper is sealed in. You cannot reach it, and it cannot reach you.

## When you are done

- Take `Molynx;` (or `Molynx_slot_only;`) out of the `alt-resources` line, so it is the old line
  again.
- Put `Level = ""` back, so the game starts normally.
- If you turned on the window, put `windowed = false` back (or untick **Windowed mode**) for
  fullscreen.
- You can delete the folder from `<DK2>\flame\resources\` if you like. Dead Walking is then the
  normal map again.
