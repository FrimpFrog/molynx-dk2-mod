"""One-step launcher: extract models from your Dungeon Keeper 2 folder, build the viewer, open it in a browser.

Usage: python scripts/run_viewer.py [path to the DK2 folder that contains Meshes.WAD]
With no path it asks once and remembers the answer. Everything is written to ./dk2_viewer_data.
LICENCE: GPL-3.0 (uses the OpenKeeper-derived texture decoder).
"""
import functools
import http.server
import json
import sys
import threading
import webbrowser
from pathlib import Path

NEEDED = ("Meshes.WAD", "EngineTextures.dir", "EngineTextures.dat")


def _find(folder: Path, name: str):
    for f in folder.iterdir():
        if f.is_file() and f.name.lower() == name.lower():
            return f
    return None


def find_game_files(game_dir: Path):
    """(Meshes.WAD, EngineTextures.dir, EngineTextures.dat). Each is searched case-insensitively in the pasted
    folder, its sub-folders and its parent folder (not the parent's other sub-folders), so the Data folder or the game folder both work."""
    game_dir = Path(game_dir)
    folders = [game_dir] + sorted(p for p in game_dir.iterdir() if p.is_dir())
    if game_dir.parent != game_dir:
        folders += [game_dir.parent]
    found = {n: next((f for folder in folders if (f := _find(folder, n))), None) for n in NEEDED}
    missing = [n for n, f in found.items() if f is None]
    if missing:
        hint = ""
        if any(n.startswith("EngineTextures.") for n in missing):
            hint = (" EngineTextures.dir and .dat hold the creature skins and are not part of the .WAD files: "
                    "copy both files next to Meshes.WAD (search your game folder for them).")
        raise FileNotFoundError(f"Could not find {', '.join(missing)} near {game_dir}.{hint}")
    return tuple(found[n] for n in NEEDED)


CUSTOM_DIR = Path(__file__).resolve().parent.parent / "creatures_custom"
GOLEM_DIR = Path(__file__).resolve().parent.parent / "megaglest_golem"      # the MegaGlest Golem (CC BY-SA 3.0 data)


def golem_units(meshes: Path, cache, golem_dir: Path | None = None) -> list:
    """The red-moss MegaGlest Golem with its new DK2-based animations and sounds (empty if the folder is not there)."""
    golem_dir = GOLEM_DIR if golem_dir is None else golem_dir
    if not (golem_dir / "golem.xml").exists():
        return []
    from scripts.glest_export import bake_inferno
    from scripts.golem_anims import build_all
    opts = {"baker": bake_inferno, "extra_anims": build_all(meshes, cache, golem_dir)}
    if (golem_dir / "golem.sounds.json").exists():
        opts["sound_pack"] = golem_dir / "golem.sounds.json"
    return [("golem", "Golem (MegaGlest, red moss)", str(golem_dir), opts)]


BATTLEMAGE_DIR = Path(__file__).resolve().parent.parent / "megaglest_battlemage"     # the MegaGlest Battlemage (CC BY-SA 3.0 data)


def battlemage_units(meshes: Path, cache, unit_dir: Path | None = None) -> list:
    """The MegaGlest Battlemage in his original yellow/red colours with DK2-based animations and sounds."""
    unit_dir = BATTLEMAGE_DIR if unit_dir is None else unit_dir
    if not (unit_dir / "battlemage.xml").exists():
        return []
    from scripts.battlemage_anims import build_all
    from scripts.battlemage_texture import bake_battlemage
    from scripts.battlemage_anims import stretched_triangles
    lib = build_all(meshes, cache, unit_dir)
    opts = {"baker": bake_battlemage, "extra_anims": lib, "extra_base": "models/battlemage_dying.g3d",
            "drop_triangles": stretched_triangles(unit_dir, lib)}
    if (unit_dir / "battlemage.sounds.json").exists():
        opts["sound_pack"] = unit_dir / "battlemage.sounds.json"
    return [("battlemage", "Battlemage (MegaGlest)", str(unit_dir), opts)]


BEHEMOTH_DIR = Path(__file__).resolve().parent.parent / "megaglest_behemoth"     # the MegaGlest Behemoth (CC BY-SA 3.0 data)


def behemoth_units(meshes: Path, cache, unit_dir: Path | None = None) -> list:
    """The MegaGlest Behemoth in his original dark red-brown colours with DK2-based animations and sounds."""
    unit_dir = BEHEMOTH_DIR if unit_dir is None else unit_dir
    if not (unit_dir / "behemoth.xml").exists():
        return []
    from scripts.behemoth_anims import build_all
    from scripts.behemoth_rig import stretched_triangles
    from scripts.behemoth_texture import bake_behemoth
    lib = build_all(meshes, cache, unit_dir)
    opts = {"baker": bake_behemoth, "extra_anims": lib, "drop_triangles": stretched_triangles(unit_dir, lib)}
    if (unit_dir / "behemoth.sounds.json").exists():
        opts["sound_pack"] = unit_dir / "behemoth.sounds.json"
    return [("behemoth", "Behemoth (MegaGlest)", str(unit_dir), opts)]


def _fingerprint(custom_dir: Path) -> str:
    """Changes whenever a custom creature spec or hand-painted texture changes (forces a rebuild)."""
    import hashlib
    h = hashlib.sha1()
    if custom_dir and custom_dir.exists():
        for f in sorted(custom_dir.rglob("*")):
            if f.is_file() and "generated" not in f.parts:
                h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def prepare(game_dir: Path, work_dir: Path, creature_ids=None, custom_dir: Path | None = None) -> Path:
    """Extract Meshes.WAD and build the site under work_dir; reuses a finished build. Returns the site folder."""
    from scripts.build_viewer import build
    from scripts.dk2texture import TextureCache
    from scripts.dk2wad import extract_all
    work_dir = Path(work_dir)
    wad, tex_dir, tex_dat = find_game_files(game_dir)
    meshes, site, marker = work_dir / "Meshes", work_dir / "site", work_dir / "site/.built.json"
    custom_dir = CUSTOM_DIR if custom_dir is None else custom_dir
    wanted = {"ids": sorted(creature_ids or []), "custom": _fingerprint(custom_dir), "golem": _fingerprint(GOLEM_DIR), "battlemage": _fingerprint(BATTLEMAGE_DIR), "behemoth": _fingerprint(BEHEMOTH_DIR)}
    if marker.exists() and json.loads(marker.read_text()) == wanted:
        return site
    if not meshes.exists() or not any(meshes.glob("*.kmf")):
        print("Extracting models from Meshes.WAD ...", flush=True)
        extract_all(wad, meshes)
    print("Building the viewer (about a minute the first time) ...", flush=True)
    cache = TextureCache(tex_dir, tex_dat)
    build(site, meshes, cache, creature_ids, custom_dir=custom_dir, glest_units=golem_units(meshes, cache) + battlemage_units(meshes, cache) + behemoth_units(meshes, cache))
    marker.write_text(json.dumps(wanted))
    return site


def serve(site: Path, port: int = 8765):
    handler = functools.partial(type("Q", (http.server.SimpleHTTPRequestHandler,),
                                     {"log_message": lambda *a: None}), directory=str(site))
    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/index.html"


def main(argv: list[str]) -> int:
    work = Path("dk2_viewer_data")
    work.mkdir(exist_ok=True)
    remembered = work / "game_path.txt"
    if len(argv) > 1:
        game = Path(argv[1].strip('"'))
    elif remembered.exists():
        game = Path(remembered.read_text().strip())
    else:
        game = Path(input("Paste the path to your Dungeon Keeper 2 folder (the one with Meshes.WAD): ").strip().strip('"'))
    try:
        site = prepare(game, work)
    except FileNotFoundError as e:
        print(e)
        remembered.unlink(missing_ok=True)
        return 1
    remembered.write_text(str(game))
    srv, url = serve(site)
    print(f"Viewer running at {url}\nLeave this window open while you use it; close it to stop.", flush=True)
    webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main(sys.argv))
