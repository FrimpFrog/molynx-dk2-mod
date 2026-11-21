# Molynx 1999-style cinematic

A DK2-intro-style short of Molynx in a torch-lit dungeon hall, rendered in Blender 4.5 (Cycles).

- `output/Molynx_1999_cinematic.mp4`: the finished 12.8 s film (4:3, 1440x1080, 25 fps)
- `output/Molynx_profile.png`: 1024x1024 head portrait
- `blend/anim.blend`: the rigged, animated Molynx (cloth baked)
- `blend/dk99.blend`: anim.blend plus the dungeon set, lights and impact effects (built by `tools/dk99_set.py`)
- `tools/`: the model, rig, animation, set, shot, grade and edit scripts

Rebuild the film:

```
blender -b blend/anim.blend --python tools/dk99_set.py -- --save blend/dk99.blend
tools/render_shots.sh 1 2 3 4 5 6 7      # one PNG per frame into shots/sN/ (edit the paths at the top first)
python3 tools/dk99_edit.py OUT.mp4        # grade, fades, MPEG-2 pass, 1440x1080 upscale
```

The scripts still contain the absolute paths of the machine they were written on (`/home/claude/...`); change those before running them elsewhere.
