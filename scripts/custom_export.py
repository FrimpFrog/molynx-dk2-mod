"""Export custom creatures (creatures_custom/*.json) into the viewer's data format.

LICENCE: GPL-3.0 when combined with the rest of dk2_tools (texture decoder).
Per creature folder creatures_custom/<id>/ : generated/ gets the generated textures and UV guides
(copy a texture to textures/<name>.png and repaint it; it then replaces the generated one).
"""
from pathlib import Path

from scripts import custom_creature as cc
from scripts import custom_textures as ct
from scripts.viewer_export import write_dk2v


def custom_specs(custom_dir: Path) -> list[Path]:
    return sorted(Path(custom_dir).glob("*.json"))


def export_custom(spec_path: Path, meshes_dir: Path, cache, out_dir: Path):
    """-> (dk2v path, {texture name: PIL image}, skipped animations)."""
    spec = cc.load_spec(spec_path)
    folder = Path(spec_path).parent / spec["id"]
    built = cc.build_data(spec, meshes_dir, cache)
    textures = ct.generate_textures(spec, built, cache, custom_dir=folder)
    gen = folder / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    prefix = f"custom/{spec['id']}/"
    for name, (img, guide) in textures.items():
        short = name[len(prefix):]
        img.save(gen / f"{short}.png")
        guide.save(gen / f"{short}_uv.png")
    path = write_dk2v(spec["id"], spec["label"], built["data"], built["missing"], out_dir)
    return path, {n: img for n, (img, _) in textures.items()}, built["skipped"]


def write_custom_obj(data_anim: dict, frame: int, textures: dict, out_dir: Path, stem: str) -> Path:
    """Textured OBJ + MTL + PNGs of one frame (Y-up), one object/material per part, for Blender."""
    from scripts.viewer_textures import texture_file
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mtl, obj, base = [], [f"mtllib {stem}.mtl"], 1
    for i, part in enumerate(data_anim["parts"]):
        name = f"part{i}_" + texture_file(part["texture"]).rsplit(".", 1)[0].replace("custom__nautiloth__", "")
        png = out_dir / texture_file(part["texture"])
        if part["texture"] in textures and not png.exists():
            textures[part["texture"]].save(png)
        mtl += [f"newmtl {name}", "Kd 1 1 1", f"map_Kd {png.name}", ""]
        obj += [f"o {name}", f"usemtl {name}"]
        for x, y, z in data_anim["frames"][frame][i]:
            obj.append(f"v {x:.5f} {y:.5f} {z:.5f}")
        for u, v in part["uv"]:
            obj.append(f"vt {u:.5f} {1 - v:.5f}")
        for a, b, c in part["indices"].reshape(-1, 3):
            obj.append("f " + " ".join(f"{base + int(k)}/{base + int(k)}" for k in (a, b, c)))
        base += len(part["uv"])
    (out_dir / f"{stem}.mtl").write_text("\n".join(mtl))
    (out_dir / f"{stem}.obj").write_text("\n".join(obj))
    return out_dir / f"{stem}.obj"
