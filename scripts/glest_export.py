"""Export a MegaGlest unit (unit.xml + its .g3d animation files + textures) to the web viewer's .dk2v format.

LICENCE: GPL-3.0. MegaGlest game data is CC BY-SA 3.0 (see docs/LICENSE.data.txt in megaglest-data).

How MegaGlest plays animations (from the engine source, unit.cpp / interpolation.cpp):
  * each skill in unit.xml names one .g3d file and an <anim-speed>; the model is a vertex-morph animation
    (every frame holds all vertices), blended linearly between frames, looping for repeating skills;
  * progress per game tick = animSpeed / (100 * 40 ticks)  ->  one full cycle takes 100 / animSpeed seconds,
    so playback speed = frames * animSpeed / 100 frames per second (die: (frames - 1) * animSpeed / 100).
Texture alpha: for meshes flagged "custom colour" the alpha channel marks the areas the engine paints in the
player's team colour; here that colour is baked into the texture (default blue).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import shutil

import numpy as np
from PIL import Image

from scripts.g3d import parse_g3d
from scripts.viewer_export import write_dk2v

SKILL_LABELS = {"stop": "stand", "move": "walk", "attack": "attack", "die": "die", "be_built": "build_up", "morph": "morph"}
LOOPING = {"stop", "move", "attack"}
TEAM_BLUE = (50, 110, 255)
PLAIN_BACKDROP = (24, 24, 24)


def parse_skills(xml_path: Path) -> list[dict]:
    """[{type, name, anim_speed, paths[]}] in file order; skills without an animation are skipped."""
    root = ET.parse(xml_path).getroot()
    out = []
    for sk in root.find("skills"):
        paths = [a.get("path") for a in sk.findall("animation") if a.get("path")]
        if not paths:
            continue
        speed = sk.find("anim-speed")
        out.append({"type": sk.find("type").get("value"), "name": sk.find("name").get("value"),
                    "anim_speed": float(speed.get("value")) if speed is not None else 100.0, "paths": paths})
    return out


def playback_fps(frame_count: int, anim_speed: float, looping: bool) -> float:
    frames = frame_count if looping else max(frame_count - 1, 1)
    return frames * anim_speed / 100.0


def bake_texture(path: Path, custom_color: bool, team=TEAM_BLUE) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    backdrop = Image.new("RGBA", im.size, (*(team if custom_color else PLAIN_BACKDROP), 255))
    backdrop.alpha_composite(im)
    return backdrop.convert("RGB")


def _model_data(g3d_path: Path, unit_id: str, textures: dict, team, baker=bake_texture):
    model = parse_g3d(g3d_path.read_bytes())
    frames_n = model.frame_count
    parts, mesh_list = [], []
    for me in model.meshes:
        if me.vertices.shape[1] == 0 or me.triangle_count == 0:
            continue
        if me.vertices.shape[1] > 65535:
            raise ValueError(f"{g3d_path.name}: mesh has too many vertices for 16-bit indices")
        tex = ""
        if me.diffuse_texture:
            stem = Path(me.diffuse_texture).stem
            tex = f"{unit_id}/{stem}" + ("" if me.custom_color else "_plain")
            if tex not in textures:
                textures[tex] = baker(g3d_path.parent / me.diffuse_texture, me.custom_color, team)
        uv = me.uv.copy() if me.uv is not None else np.zeros((me.vertices.shape[1], 2), np.float32)
        uv[:, 1] = 1.0 - uv[:, 1]                       # OpenGL v runs bottom-up; the viewer reads top-down
        parts.append({"texture": tex, "uv": uv, "indices": me.indices.astype(np.uint16)})
        mesh_list.append(me)
    frames = [[m.vertices[min(f, m.frame_count - 1)].astype(np.float64) for m in mesh_list] for f in range(frames_n)]
    return frames, parts


def bake_inferno(path: Path, custom_color: bool, team=None) -> Image.Image:
    """Red version of the Golem that keeps the grey/brown stone: only the green moss turns blood red and the
    blue team-colour runes glow red-orange. (Name kept from the first, fully red attempt.)"""
    im = Image.open(path).convert("RGBA")
    px = np.asarray(im, float)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    moss = np.clip((g - np.maximum(r, b) + 4) / 14.0, 0, 1)[..., None]   # how green-dominant a pixel is (neutral stone gives about 0.3)
    red = np.stack([g, 0.8 * r, 0.9 * b], -1)                            # swap green into red: moss -> brick red
    rgb = px[..., :3] * (1 - moss) + red * moss
    glow = (1 - np.asarray(im, float)[..., 3] / 255.0)[..., None]       # alpha < 1 marks the team-colour runes
    ember = np.array([255, 70, 30]) if custom_color else np.array(PLAIN_BACKDROP)
    out = rgb * (1 - glow) + ember * glow
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def export_glest_unit(unit_dir: Path, unit_id: str, label: str, out_dir: Path, team=TEAM_BLUE,
                      extra_models: dict | None = None, baker=None, tex_prefix: str | None = None,
                      extra_anims: dict | None = None, sound_dir: Path | None = None, sound_pack: Path | None = None,
                      extra_base: str | None = None, drop_triangles=None):
    """extra_anims = {label: {"frames": [(n,3) array per frame], "wrap": bool, "fps": float}} are added on the standing model's mesh (see golem_retarget).
    baker(path, custom_color, team) -> image replaces the default texture baking (e.g. bake_inferno);
    extra_base = relative g3d path whose mesh parts (and UVs) carry the extra animations (default: the standing model);
    drop_triangles = triangle numbers removed from the extra animations' (first) mesh part;
    tex_prefix renames the textures (default: unit_id) so a recoloured copy can live next to the original."""
    """-> (dk2v path, {texture name: PIL image}). extra_models = {animation label: relative g3d path}."""
    unit_dir = Path(unit_dir)
    textures: dict = {}
    data, seen = {}, set()
    xml = unit_dir / f"{unit_dir.name}.xml"
    if not xml.exists():                                  # e.g. the bundled folder megaglest_golem/golem.xml
        xml = unit_dir / f"{unit_id.split('_')[0]}.xml"
    for sk in parse_skills(xml):
        for i, rel in enumerate(sk["paths"]):
            if rel in seen:
                continue
            seen.add(rel)
            name = SKILL_LABELS.get(sk["type"], sk["name"]) + (f"_{i + 1}" if i else "")
            frames, parts = _model_data(unit_dir / rel, tex_prefix or unit_id, textures, team, baker or bake_texture)
            looping = sk["type"] in LOOPING
            data[name] = {"frames": frames, "parts": parts, "wrap": looping,
                          "fps": playback_fps(len(frames), sk["anim_speed"], looping)}
    for name, rel in (extra_models or {}).items():
        if rel not in seen:
            frames, parts = _model_data(unit_dir / rel, tex_prefix or unit_id, textures, team, baker or bake_texture)
            data[name] = {"frames": frames, "parts": parts, "wrap": True, "fps": 10.0}
    if extra_anims:
        base = data["stand"] if "stand" in data else next(iter(data.values()))
        if extra_base:
            base = {"parts": _model_data(unit_dir / extra_base, tex_prefix or unit_id, textures, team, baker or bake_texture)[1]}
        if drop_triangles is not None and len(drop_triangles):
            counts = np.array([len(pt["indices"]) // 3 for pt in base["parts"]])
            starts = np.concatenate([[0], np.cumsum(counts)])
            parts = []
            for k, pt in enumerate(base["parts"]):          # triangle numbers run through the mesh parts in order
                gone = np.asarray(drop_triangles)
                gone = gone[(gone >= starts[k]) & (gone < starts[k + 1])] - starts[k]
                keep = np.ones(counts[k], bool)
                keep[gone] = False
                parts.append({**pt, "indices": pt["indices"].reshape(-1, 3)[keep].reshape(-1)})
            base = {"parts": parts}
        for name, ea in extra_anims.items():
            sizes = np.cumsum([len(pt["uv"]) for pt in base["parts"]])[:-1]       # a unit with several mesh parts (club + body): split the rows
            data[name] = {"frames": [np.split(np.asarray(fr, np.float64), sizes) for fr in ea["frames"]], "parts": base["parts"],
                          "wrap": ea["wrap"], "fps": ea["fps"]}
    path = write_dk2v(unit_id, label, data, set(), out_dir, mirror_x=False)
    if sound_pack:                                      # a ready-made pack (no ffmpeg needed)
        shutil.copy2(sound_pack, Path(out_dir) / f"{unit_id}.sounds.json")
    elif sound_dir:                                     # sounds the viewer plays on matching animations (golem_sounds.py)
        from scripts.golem_sounds import build_sound_pack, write_sound_pack
        write_sound_pack(build_sound_pack(Path(sound_dir), data), Path(out_dir) / f"{unit_id}.sounds.json")
    return path, textures
