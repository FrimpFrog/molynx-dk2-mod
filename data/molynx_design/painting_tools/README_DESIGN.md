# Molynx design v6 (design only; NOT applied to the mod)

Changes vs v5 (the high-quality character in char_full.pkl):
- Whole harness removed: shoulder straps (front+back), chest strap, diagonal belly strap, skull medallion,
  and the Behemoth's own original leather harness underneath (masked fully incl. dark edges, refilled with
  local skin colour + skin grain, feathered).
- Belt: old black belt, bone plate and chain row removed; replaced by MegaGlest Battlemage belt
  (texture_battle_mage.png: back strap crop (17,130)-(60,139) repeated, front piece (133,115)-(193,133) with
  the round buckle), band height 0.18 model units (old 0.14), centred y 2.49, buckle centred on the spine (x 0.05).
- Model, mace, animations unchanged (texture-only change).

Files: tools/design_paint.py (copy of scripts/character_paint.py with the v6 edits), tools/render_hq.py,
tools/make_design_media.py; out/ = previews + Molynx_v6_body_texture_512.png.

To apply later (only when the user says so): swap paint_character for the v6 painter in the character build,
re-export Molynx assets (needs Meshes.WAD), rebuild/audit the mod folders.

# Molynx design v7 (on top of v6; design only, NOT applied to the mod)
- Head: the MegaGlest Golem's face and peaked dome (golem_standing.g3d frame 0, triangles with y > 0.9, |x| < 0.62, z > 0.35)
  as a new part `custom_character/golem_head`, scaled 0.44 and centred at (0.05, 3.62, 0.50). Its open rim is extruded back
  0.44 (taper 1.0) into his skull/neck so it is a solid head. His own face triangles (head vertices with z > 0.30 in stand
  frame 0) are removed; the back of his skull, his neck and both horns stay.
- Motion: every frame of every clip (and the T-pose) moves the new head by a rigid fit to his upper skull (head vertices
  above y 3.6), max fit error about 0.1-0.2.
- Texture (tools/head_paint.py -> out_v7/Molynx_v7_head_texture_512.png): the Golem's light/shade recoloured to his body skin
  colour with his skin grain; cracks softened; angry slanted glowing eyes, heavy brows and frown lines like Molynx's; snarl
  mouth (corners down) with ivory upper teeth and corner fangs hanging over the lower lip (slight overbite), small lower
  teeth behind them.
- Build: tools/head_swap.py build(data) -> new character dict; tools/make_v7_media.py renders out_v7/.
- Still to do when applying: game version needs the head as KMF sprites (golem head 103+ verts fits the 255 limit) and its
  texture at 128 px; re-export needs Meshes.WAD.

# Molynx design v8 (on top of v7; design only, NOT applied to the mod) -- tools/head_swap_v8.py build_v8(), tools/head_paint_v8.py
Shape (both versions):
- Horn roots: triangles touching a horn are kept even where they touch his old face (v7 cut them: notch + dark wedge).
- Top: the Golem apex lowered/rounded (centre 2.33/2.27, shoulders 2.24, sides 2.15/2.14 in Golem units).
- Jaw: Golem vertices below the mouth (y < 1.235) pushed forward up to 0.13 and down 0.05 (jutting jaw); the closing band
  turns in under the jaw (JAW_TAPER 0.30, shorter) and the back of the head slopes down (no flat lid).
- Sides/back/underside of the head: own part `custom_character/head_neck` = a strip of his own muscle skin (body texture
  125..245 x 262..330, cleaned of pale flecks), UV = distance round the rim x depth, at 2x his body texel density, with a
  shadow along the face edge (jaw line).
- Teeth: big BOTTOM teeth like the original (two square front teeth, smaller ones, corner tusks) rising over the top lip;
  small upper teeth behind them; snarl mouth; chin shadow.
Version A (red rock): work/A_body.png = his body with the Golem rock cracks over every red-skin texel (cell size matched to
  the head; belt, loincloth, horns, teeth, ankle bands and dark claws masked out); A_head.png rocky; A_strip.png rocky.
Version B (smooth): body unchanged (v6); B_head.png = his own skin grain at his texel size and exact average colour, only the
  Golem face's large forms + painted features; B_strip.png his skin.
Media: out_v8/.

# Scenes and design A2 (2026-11-06)
- Chosen: version A (red rock). A2 = A graded to the real DK2 skins (tools/gloss.py): red channel mean 140 / std 50 like
  ReaperTorso/Back (was 71/23), same factor on G/B (his hue kept), plus a painted warm highlight on raised rock detail (gloss).
  Textures: work/A2_body.png, A2_head.png, A2_strip.png (copies in out_scenes/).
- Design size: the top of his head level with the Horned Reaper's skull: scale = export scale x (0.715/0.6168) x (0.71/0.685)
  = about 1.20x his old in-game size (tools/scene.py MOLYNX_SCALE). When applied, height_ratio in molynx_assets would be ~1.39.
- Scenes (tools/scene.py, tools/make_scenes.py -> out_scenes/): real DK2 KMF models + the user's Quuz HD textures.
  Dances use his victory_celebration clip (the Bile-dance retarget), not celebrating.
- 2026-11-06: wrist fix (tools/fix_wrist.py): a leftover arm-band patch on the forearm/back of the hand (not red, so the
  A2 grade skipped it and it showed as a beige-grey smear) repainted with the surrounding rock skin (A and A2 bodies;
  backups *_before_wristfix.png). T-pose / idle renders in out_poses/. The idle pose's free hand is a closed fist without
  claws in the original MegaGlest stand pose (not a fault).
- 2026-11-06: claw fix (tools/claw_fix.py, applied in common.load_char(fixed=True)): the free hand's claw (vertices 244,
  571-574, labelled torso/leg in the Behemoth rig) floated by the hip in the 21 retargeted DK2 clips (up to 2.8 units off);
  now moved rigidly with its 12 hand vertices in every frame. Native clips unchanged in effect. char_full.pkl not modified.
  When applying to the game files, the export must use the fixed data.

## Animation review round (fidget / pray / happy / celebrate)
- Fidgeting (idle 2) -> Troll idle 2 (`fid_troll_idle2`, tools/new_clips.py, work/candidate_clips.pkl).
- Celebrating -> Troll dance (`dance_troll`).
- Mace arm fix: tools/claw_fix.py `fix_club_arm` (arm verts follow the club rigidly) and `lock_club_arm`
  (arm + club + mace held at their stand-frame-0 pose, carried by the upper torso) for LOCKED clips
  (eating, praying, happy, celebrating, victory_celebration, manufacturing) and the dance_/fid_reaper candidates.
- Alternatives on hand: Dark Knight idle2/dance, Bile Demon dance. Horned Reaper clips don't retarget (leg proportions).
- Review GIFs: out_review/.
