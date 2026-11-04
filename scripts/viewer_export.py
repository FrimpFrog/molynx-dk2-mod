"""Export DK2 creatures to compact .dk2v files for the web viewer.

LICENCE: GPL-3.0 (imports scripts.kmf_textured, which uses the OpenKeeper-derived texture decoder).

File layout (little-endian):  b"DK2V" | u32 json_len | json (space-padded to a multiple of 4) | blob
JSON: {label, fps, bbox{min,max}, missing_textures[], animations{label: {frame_count, wrap,
parts[{texture, vertex_count, uv_offset, index_offset, index_count, position_offset}]}}}
Blob offsets are bytes from the blob start. Per part: float32 UVs (2*n), uint16 indices (padded to
4 bytes), uint16 positions frame-major (frames*n*3, padded to 4 bytes). Positions are Y-up and
quantised to 16 bits across the creature's bounding box over all of its frames.
"""
import base64
import gzip
import json
import struct
from pathlib import Path

import numpy as np

from scripts.dk2kmf import evaluate_anim_frame, parse_kmf
from scripts.kmf_textured import _yup, material_texture_name
from scripts.viewer_creatures import CREATURES, find_animations

FPS = 30
MAGIC = b"DK2V"


def _pad4(b: bytes) -> bytes:
    return b + b"\0" * (-len(b) % 4)


def _collect(path: Path, cache, missing: set[str]):
    model = parse_kmf(path.read_bytes())
    anim = model.anim
    keep = [i for i, s in enumerate(anim.sprites) if s.triangles[0]]
    frames = []
    for f in range(anim.frame_count):
        pts = evaluate_anim_frame(anim, f)
        frames.append([_yup(pts[i]) for i in keep])
    parts = []
    for i in keep:
        s = anim.sprites[i]
        mat = model.materials[s.material]
        tex = material_texture_name(cache, mat)
        if tex is None:
            missing.add(mat.textures[0] if mat.textures else mat.name)
        parts.append({
            "texture": tex or "",
            "uv": np.array([(v.uv[0] / 32768.0, v.uv[1] / 32768.0) for v in s.vertices], np.float32),
            "indices": np.array(s.triangles[0], np.uint16).reshape(-1),
        })
    return {"frames": frames, "parts": parts, "wrap": bool(anim.wrap)}


def write_dk2v(creature_id: str, label: str, data: dict, missing: set[str], out_dir: Path,
               mirror_x: bool = True) -> Path:
    """data[anim_label] = {"frames": [[(n,3) per part] per frame], "parts": [{texture, uv, indices}], "wrap": bool, "fps": optional}.
    mirror_x False = right-handed source data (MegaGlest); the viewer then does not mirror it."""
    allp = np.concatenate([p for a in data.values() for fr in a["frames"] for p in fr])
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1e-9)
    blob = bytearray()
    meta_anims = {}
    for alabel, a in data.items():
        parts_meta = []
        for pi, part in enumerate(a["parts"]):
            n = len(part["uv"])
            uv_off = len(blob); blob += _pad4(np.asarray(part["uv"], np.float32).tobytes())
            ix_off = len(blob); blob += _pad4(np.asarray(part["indices"], np.uint16).tobytes())
            q = np.stack([np.round((fr[pi] - lo) / span * 65535) for fr in a["frames"]]).astype(np.uint16)
            pos_off = len(blob); blob += _pad4(q.tobytes())
            parts_meta.append({"texture": part["texture"], "vertex_count": n, "uv_offset": uv_off,
                               "index_offset": ix_off, "index_count": len(part["indices"]),
                               "position_offset": pos_off})
        meta_anims[alabel] = {"frame_count": len(a["frames"]), "wrap": a["wrap"], "parts": parts_meta}
        if a.get("fps"):
            meta_anims[alabel]["fps"] = a["fps"]      # per-animation playback speed (frames per second)
    header = {"label": label, "fps": FPS, "mirror_x": mirror_x,
              "bbox": {"min": lo.tolist(), "max": hi.tolist()},
              "missing_textures": sorted(missing), "animations": meta_anims}
    js = json.dumps(header, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{creature_id}.dk2v"
    path.write_bytes(MAGIC + struct.pack("<I", len(js)) + js + bytes(blob))
    return path


def export_creature(creature_id: str, meshes_dir: Path, cache, out_dir: Path, creatures=None) -> Path:
    creatures = CREATURES if creatures is None else creatures
    files, _ = find_animations(meshes_dir, {creature_id: creatures[creature_id]})
    missing: set[str] = set()
    data = {label: _collect(p, cache, missing) for label, p in files[creature_id].items()}
    return write_dk2v(creature_id, creatures[creature_id].label, data, missing, out_dir)


def read_header(path: Path) -> tuple[dict, int]:
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError("not a .dk2v file")
    (n,) = struct.unpack_from("<I", raw, 4)
    return json.loads(raw[8:8 + n]), 8 + n


def read_dk2v(path: Path) -> dict:
    """Python reader used by tests: de-quantised positions per frame per part."""
    raw = Path(path).read_bytes()
    meta, base = read_header(path)
    lo, hi = np.array(meta["bbox"]["min"]), np.array(meta["bbox"]["max"])
    span = np.maximum(hi - lo, 1e-9)
    anims = {}
    for label, am in meta["animations"].items():
        parts, frames = [], []
        for pm in am["parts"]:
            n = pm["vertex_count"]
            uv = np.frombuffer(raw, "<f4", 2 * n, base + pm["uv_offset"]).reshape(n, 2)
            ix = np.frombuffer(raw, "<u2", pm["index_count"], base + pm["index_offset"])
            q = np.frombuffer(raw, "<u2", am["frame_count"] * n * 3, base + pm["position_offset"])
            parts.append({**pm, "uv": uv, "indices": ix})
            frames.append(q.reshape(am["frame_count"], n, 3) / 65535.0 * span + lo)
        anims[label] = {"meta": am, "parts": parts,
                        "frames": [[fr[f] for fr in frames] for f in range(am["frame_count"])]}
    return {"meta": meta, "animations": anims}


def pack_text(path: Path) -> Path:
    """Write <name>.dk2v.txt = base64(gzip(file)) next to a .dk2v. Used for hosts that only serve text types."""
    path = Path(path)
    out = path.with_name(path.name + ".txt")
    out.write_bytes(base64.b64encode(gzip.compress(path.read_bytes(), 9, mtime=0)))
    return out


def write_index(out_dir: Path, paths: list[Path]) -> Path:
    creatures = []
    for p in paths:
        meta, _ = read_header(p)
        textures = sorted({pm["texture"] for a in meta["animations"].values()
                           for pm in a["parts"] if pm["texture"]})
        sounds = Path(p).with_name(Path(p).stem + ".sounds.json")
        creatures.append({"id": Path(p).stem, "label": meta["label"], "file": Path(p).name,
                          **({"sounds": sounds.name} if sounds.exists() else {}),
                          "missing_textures": meta["missing_textures"], "textures": textures,
                          "animations": [{"label": k, "frames": v["frame_count"], "wrap": v["wrap"]}
                                         for k, v in meta["animations"].items()]})
    out = Path(out_dir) / "index.json"
    out.write_text(json.dumps({"fps": FPS, "creatures": creatures}, indent=1))
    return out
