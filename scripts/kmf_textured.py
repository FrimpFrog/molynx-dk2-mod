"""Textured preview and OBJ/MTL export for DK2 KMF models.

LICENCE: GPL-3.0, because it uses scripts.dk2texture (a port of OpenKeeper's texture decoder).

Usage: python3 -m scripts.kmf_textured <Meshes dir> <model.kmf> <EngineTextures dir base> <out.png>
(see the __main__ block). Uses texture mip level 0 (full size) for every material.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.dk2kmf import Kmf, Material, evaluate_anim_frame, mesh_points, parse_kmf
from scripts.dk2texture import TextureCache


def material_texture_name(cache: TextureCache, material: Material) -> str | None:
    """Cache name of a material's main texture at full resolution (first texture listed)."""
    if not material.textures:
        return None
    base = material.textures[0]
    lowered = {n.lower(): n for n in cache.names}
    return lowered.get(f"{base}MM0".lower())


def _sprites_and_points(model: Kmf, frame: int | None):
    if model.kind == "mesh":
        return model.mesh.sprites, mesh_points(model)
    return model.anim.sprites, evaluate_anim_frame(model.anim, frame or 0)


def _yup(p: np.ndarray) -> np.ndarray:
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def _rotation(azimuth: float) -> np.ndarray:
    az = np.radians(azimuth)
    return np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])


def fixed_view(models_frames, azimuth: float = 180.0):
    """Camera (centre, span) enclosing every (model, frame) pair, so clips don't jitter."""
    rot = _rotation(azimuth)
    pts = []
    for model, frame in models_frames:
        _, raw = _sprites_and_points(model, frame)
        pts += [_yup(p) @ rot.T for p in raw]
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    return (lo + hi) / 2, float(max(hi - lo)) * 1.1


def render_textured(model: Kmf, cache: TextureCache, azimuth: float = 180.0, size: int = 512,
                    frame: int | None = None, supersample: int = 2, view=None) -> Image.Image:
    """Software-render a model with its real textures (orthographic, z-buffered, lit).

    view: optional (centre, span) from fixed_view(); default frames each call to the model.
    """
    sprites, raw = _sprites_and_points(model, frame)
    pts = [_yup(p) for p in raw]
    rot = _rotation(azimuth)
    rotated = [p @ rot.T for p in pts]
    if view is None:
        allp = np.concatenate(rotated)
        lo, hi = allp.min(0), allp.max(0)
        centre, span = (lo + hi) / 2, float(max(hi - lo)) * 1.1
    else:
        centre, span = view
    big = size * supersample
    colour = np.zeros((big, big, 3), dtype=np.float32)
    colour[:] = (38, 38, 42)
    zbuf = np.full((big, big), np.inf, dtype=np.float32)
    texcache: dict[str, np.ndarray] = {}
    light = np.array([0.35, 0.55, -0.75])  # toward the viewer, from upper left
    light /= np.linalg.norm(light)
    for sprite, p3 in zip(sprites, rotated):
        if not sprite.triangles[0]:
            continue
        name = material_texture_name(cache, model.materials[sprite.material])
        if name is None:
            continue
        if name not in texcache:
            texcache[name] = np.asarray(cache.get(name).convert("RGB"), dtype=np.float32)
        tex = texcache[name]
        th, tw = tex.shape[:2]
        v = (p3 - centre)
        sx = big / 2 + v[:, 0] / span * big
        sy = big / 2 - v[:, 1] / span * big
        uv = np.array([(x.uv[0] / 32768.0, x.uv[1] / 32768.0) for x in sprite.vertices])
        nrm = _yup(np.array([x.normal for x in sprite.vertices])) @ rot.T
        for a, b, c in sprite.triangles[0]:
            x0, x1 = int(max(min(sx[[a, b, c]]) - 1, 0)), int(min(max(sx[[a, b, c]]) + 2, big))
            y0, y1 = int(max(min(sy[[a, b, c]]) - 1, 0)), int(min(max(sy[[a, b, c]]) + 2, big))
            if x1 <= x0 or y1 <= y0:
                continue
            den = ((sy[b] - sy[c]) * (sx[a] - sx[c]) + (sx[c] - sx[b]) * (sy[a] - sy[c]))
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            w0 = ((sy[b] - sy[c]) * (gx - sx[c]) + (sx[c] - sx[b]) * (gy - sy[c])) / den
            w1 = ((sy[c] - sy[a]) * (gx - sx[c]) + (sx[a] - sx[c]) * (gy - sy[c])) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not inside.any():
                continue
            z = w0 * v[a, 2] + w1 * v[b, 2] + w2 * v[c, 2]
            win = inside & (z < zbuf[y0:y1, x0:x1])
            if not win.any():
                continue
            u = (w0 * uv[a, 0] + w1 * uv[b, 0] + w2 * uv[c, 0]) % 1.0
            t = (w0 * uv[a, 1] + w1 * uv[b, 1] + w2 * uv[c, 1]) % 1.0
            texel = tex[np.minimum((t * th).astype(int), th - 1), np.minimum((u * tw).astype(int), tw - 1)]
            n = w0[..., None] * nrm[a] + w1[..., None] * nrm[b] + w2[..., None] * nrm[c]
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
            shade = 0.5 + 0.5 * np.clip(np.abs(n @ light), 0, 1)
            region = colour[y0:y1, x0:x1]
            region[win] = (texel * shade[..., None])[win]
            zbuf[y0:y1, x0:x1][win] = z[win]
    img = Image.fromarray(np.clip(colour, 0, 255).astype(np.uint8), "RGB")
    return img.resize((size, size), Image.LANCZOS) if supersample > 1 else img


def write_textured_obj(model: Kmf, cache: TextureCache, out_dir: Path, stem: str,
                       frame: int | None = None) -> Path:
    """Write stem.obj, stem.mtl and PNG textures into out_dir (Y-up, v flipped for OBJ)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sprites, raw = _sprites_and_points(model, frame)
    obj = ["# Dungeon Keeper 2 model with textures (Y-up)", f"mtllib {stem}.mtl"]
    mtl, written, base = [], set(), 1
    for i, (sprite, p) in enumerate(zip(sprites, raw)):
        if not sprite.triangles[0]:
            continue
        material = model.materials[sprite.material]
        tex_name = material_texture_name(cache, material)
        mat_id = f"{i}_{material.name}".replace(" ", "_")
        if tex_name and mat_id not in written:
            png = f"{tex_name}.png"
            cache.get(tex_name).save(out_dir / png)
            mtl += [f"newmtl {mat_id}", "Kd 1 1 1", f"map_Kd {png}", ""]
            written.add(mat_id)
        y = _yup(p)
        obj.append(f"o part{i}_{material.name}".replace(" ", "_"))
        obj += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in y]
        obj += [f"vt {v.uv[0] / 32768:.6f} {1 - v.uv[1] / 32768:.6f}" for v in sprite.vertices]
        obj += [f"vn {a:.5f} {b:.5f} {c:.5f}" for a, b, c in _yup(
            np.array([v.normal for v in sprite.vertices]))]
        obj.append(f"usemtl {mat_id}")
        for a, b, c in sprite.triangles[0]:
            obj.append("f " + " ".join(f"{base + k}/{base + k}/{base + k}" for k in (a, b, c)))
        base += len(sprite.vertices)
    path = out_dir / f"{stem}.obj"
    path.write_text("\n".join(obj) + "\n")
    (out_dir / f"{stem}.mtl").write_text("\n".join(mtl) + "\n")
    return path


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        sys.exit("usage: python3 -m scripts.kmf_textured <model.kmf> <EngineTextures.dir> "
                 "<EngineTextures.dat> <out.png> [frame]")
    cache = TextureCache(Path(sys.argv[2]), Path(sys.argv[3]))
    model = parse_kmf(Path(sys.argv[1]).read_bytes())
    render_textured(model, cache, frame=int(sys.argv[5]) if len(sys.argv) == 6 else None).save(
        sys.argv[4])
