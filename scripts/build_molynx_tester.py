"""The Molynx tester package: everything a tester needs, laid out as the files to copy into their
Dungeon Keeper 2 folder, with Molynx on two test maps only.

`build_tester(out, assets_dir, game_sounds, wavs, flame_build)` writes `out/` holding

  Copy into your Dungeon Keeper 2 folder/
    PATCH.dll, flame/DKII.dll, flame/Flame.dll
                      the Flame build with the Molynx patches (branch flame-build-out): creature
                      types beyond 46 (portal slots and "seen" flags) and the hang watchdog
    flame/resources/Molynx/...
                      the stage-3 Molynx folder of build_molynx_mod (model, animations, textures,
                      panel pictures, Ground Slam, texts, the editor's Creatures.kwd with record 47)
                      plus his sound bank (molynx_soundbank), and no maps
    Data/editor/maps/MolynxTest1*, MolynxTest2*
                      the two flat test maps (dk2_testmap): MolynxTest1 with Molynx beside the
                      heart, MolynxTest2 also with Molynx in the portal pool. They go in the game's
                      own maps folder, not in the resource folder, because the Skirmish list is a
                      listing of that folder (Flame's fs_DirIter_init calls FindFirstFileA on the
                      real path; alt-resources only redirects files opened by name), so a new map
                      name in a resource folder never shows in the list.

No other map is touched: every other map keeps the game's 46-creature table.

`python -m scripts.build_molynx_tester --out DIR --assets EXPORT --game-sounds Data/Sound/Sfx/Global
 --wavs WAVS --flame FLAME_BUILD`

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from scripts.build_molynx_mod import MAPS_PATH, build_mod
from scripts.molynx_soundbank import sound_files

TEST_MAPS = {"MolynxTest1": False, "MolynxTest2": True}   # stem -> Molynx also in the portal pool
COPY_DIR = "Copy into your Dungeon Keeper 2 folder"
FLAME_FILES = ("PATCH.dll", "flame/DKII.dll", "flame/Flame.dll")
README_NAME = "READ ME FIRST - Molynx test.txt"   # docs/<this>, copied with Windows line ends


def build_tester(out: Path, assets_dir: Path, game_sounds: Path, wavs: Path, flame_build: Path) -> Path:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; it is never overwritten")
    with tempfile.TemporaryDirectory() as tmp:
        folder = build_mod(3, Path(tmp), "Molynx", maps=[], flat_maps=TEST_MAPS, assets_dir=assets_dir)
        for rel, data in sound_files(game_sounds, wavs).items():
            p = folder / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        root = out / COPY_DIR
        maps_src = folder / MAPS_PATH
        maps_dst = root / MAPS_PATH
        maps_dst.mkdir(parents=True)
        for f in sorted(maps_src.iterdir()):
            shutil.move(str(f), maps_dst / f.name)
        maps_src.rmdir()
        toml = folder / "resource.toml"
        toml.write_bytes(toml.read_bytes().replace(b'version = ""', b'version = "tester 1"')
                         .replace(b'authors = ""', b'authors = "Soles"'))
        shutil.copytree(folder, root / "flame" / "resources" / "Molynx")
    readme = Path(__file__).resolve().parent.parent / "docs" / README_NAME
    (out / README_NAME).write_bytes(readme.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    for rel in FLAME_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(flame_build) / rel, dst)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Molynx tester package (two test maps only).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True, help="molynx_assets export folder")
    p.add_argument("--game-sounds", type=Path, required=True, help="the game's Data/Sound/Sfx/Global")
    p.add_argument("--wavs", type=Path, required=True, help="Molynx's WAVs (molynx_sounds)")
    p.add_argument("--flame", type=Path, required=True, help="the flame-build-out files (PATCH.dll, flame/)")
    a = p.parse_args(argv)
    print(build_tester(a.out, a.assets, a.game_sounds, a.wavs, a.flame))
    return 0


if __name__ == "__main__":
    sys.exit(main())
