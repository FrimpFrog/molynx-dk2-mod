#!/bin/bash
# render the listed shots of the 1999 cinematic in sequence (frames already on disk are skipped)
cd /home/claude/molynx_blender/dk99
for s in "$@"; do
  /home/claude/blender/blender -b dk99.blend --python /home/claude/molynx_cinematic/tools/dk99_shots.py -- --shot $s --render 2>&1 | grep --line-buffered -E "^Saved|rror|Killed|SHOT_DONE"
done
echo JOB_DONE "$@"
