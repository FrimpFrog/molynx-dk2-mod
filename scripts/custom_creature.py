"""Build a custom DK2 creature by remixing existing ones: reshape the base mesh, attach parts from
other creatures, and keep everything following the base animations.

All geometry is Y-up (the same space the .dk2v exporter uses). Frames are lists (per frame) of lists
(per part) of (n, 3) arrays; a pose is a list (per part) of (n, 3) arrays.

Regions are dicts with optional "x", "y", "z" = [min, max] in pose space (a missing axis is unbounded).
LICENCE: GPL-3.0 when combined with the rest of dk2_tools (texture decoder).
"""
import numpy as np

FAR = 1e6   # pose placeholder for vertices that have no counterpart in the base pose


def region_weights(points: np.ndarray, region: dict, falloff: float = 0.0) -> np.ndarray:
    """1 inside the box, falling linearly to 0 at `falloff` distance outside it."""
    excess = np.zeros_like(points)
    for k, axis in zip("xyz", range(3)):
        lo, hi = region.get(k, [-np.inf, np.inf])
        excess[:, axis] = np.maximum(lo - points[:, axis], 0) + np.maximum(points[:, axis] - hi, 0)
    if "absx" in region:          # symmetric: |x| inside [lo, hi]
        lo, hi = region["absx"]
        ax = np.abs(points[:, 0])
        excess[:, 0] = np.maximum(excess[:, 0], np.maximum(lo - ax, 0) + np.maximum(ax - hi, 0))
    dist = np.linalg.norm(excess, axis=1)
    w = (dist == 0).astype(float) if falloff <= 0 else np.clip(1 - dist / falloff, 0, 1)
    w[np.abs(points).max(axis=1) >= FAR / 2] = 0.0     # placeholder for vertices we could not match
    return w


def submesh(part: dict, points: np.ndarray, region):
    """Triangles of `part` whose centroid is inside `region` (None = all), with vertices compacted."""
    idx = np.asarray(part["indices"]).reshape(-1, 3)
    keep = np.ones(len(idx), bool) if region is None else region_weights(points[idx].mean(1), region) > 0
    if not keep.any():
        raise ValueError(f"region {region} selects no triangles")
    used = np.unique(idx[keep])
    remap = np.full(len(points), -1)
    remap[used] = np.arange(len(used))
    return ({"uv": np.asarray(part["uv"])[used], "indices": remap[idx[keep]].reshape(-1).astype(np.uint16)},
            points[used])


def _select(pose, region):
    return [region_weights(p, region) > 0 for p in pose]


def anchor_track(pose, frames, region):
    """Rigid motion (R_f, c_f) of the pose points inside `region`, per frame (Kabsch fit).

    A point q in the pose maps to c_f + R_f (q - c_0) at frame f.
    """
    sel = _select(pose, region)
    P = np.concatenate([p[m] for p, m in zip(pose, sel)])
    if len(P) < 3:
        raise ValueError(f"anchor region {region} selects {len(P)} vertices; need at least 3")
    c0 = P.mean(0)
    Pc = P - c0
    Rs, cs = [], []
    for fr in frames:
        Q = np.concatenate([p[m] for p, m in zip(fr, sel)])
        cq = Q.mean(0)
        U, _, Vt = np.linalg.svd(Pc.T @ (Q - cq))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        Rs.append(R); cs.append(cq)
    return np.array(Rs), np.array(cs)


def pose_centre(pose, region):
    sel = _select(pose, region)
    return np.concatenate([p[m] for p, m in zip(pose, sel)]).mean(0)


def _scale3(v):
    return np.array([v, v, v], float) if np.isscalar(v) else np.array(v, float)


def apply_reshape(frames, pose, ops, anchors):
    """Add per-vertex offsets (computed once on the pose) to every frame.

    ops: {"op": "move", region, offset, falloff?, anchor?} or
         {"op": "scale", region, pivot, factor (number or [x,y,z]), falloff?, anchor?}.
    An op with "anchor" has its offset rotated with that anchor each frame (use it for the head).
    """
    cur = [p.copy() for p in pose]
    offsets = [np.zeros((len(p), 3)) for p in pose]   # per op layers: (anchor name or None, offsets)
    layers = []
    for op in ops:
        layer = [np.zeros((len(p), 3)) for p in cur]
        for i, p in enumerate(cur):
            w = region_weights(p, op["region"], op.get("falloff", 0.0))[:, None]
            if op["op"] == "move":
                layer[i] = w * np.array(op["offset"], float)
            elif op["op"] == "scale":
                pivot, f = np.array(op["pivot"], float), _scale3(op["factor"])
                layer[i] = w * (pivot + (p - pivot) * f - p)
            else:
                raise ValueError(f"unknown reshape op {op['op']!r}")
            cur[i] = p + layer[i]
        layers.append((op.get("anchor"), layer))
    tracks = {name: anchor_track(pose, frames, anchors[name]["region"])
              for name in {a for a, _ in layers if a}}
    out = []
    for f, fr in enumerate(frames):
        parts = [p.copy() for p in fr]
        for name, layer in layers:
            for i in range(len(parts)):
                off = layer[i] if name is None else layer[i] @ tracks[name][0][f].T
                parts[i] = parts[i] + off
        out.append(parts)
    return out


def _euler(deg):
    rx, ry, rz = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X


def _placement(att):
    """Linear part of the donor placement (scale, mirror, rotation)."""
    s = _scale3(att.get("scale", 1.0))
    if att.get("mirror"):
        s = s * np.array([-1, 1, 1])
    return _euler(att.get("rotate_deg", [0, 0, 0])) @ np.diag(s)


def flips_winding(att) -> bool:
    return bool(np.linalg.det(_placement(att)) < 0)


def make_attachment(pose, frames, anchors, donor_points, att):
    """Positions (F, n, 3) of a donor part rigidly following the anchor named in `att`.

    The donor is placed in pose space (origin = its centroid or att["origin"] -> att["place"]); at
    frame f a point X moves to c_f + R_f (X - c_0) where (R_f, c_f) is the anchor's rigid motion.
    """
    region = anchors[att["anchor"]]["region"]
    R, c = anchor_track(pose, frames, region)
    c0 = pose_centre(pose, region)
    origin = np.array(att["origin"], float) if "origin" in att else donor_points.mean(0)
    X = (donor_points - origin) @ _placement(att).T + np.array(att["place"], float)
    return np.stack([c[f] + (X - c0) @ R[f].T for f in range(len(frames))])



def _norm_tri(tri, uv):
    """Triangle as a tuple of its corner UVs, rotated so the smallest UV comes first -> (key, rotation)."""
    corners = [tuple(float(x) for x in uv[i]) for i in tri]
    k = min(range(3), key=lambda i: corners[i])
    return tuple(corners[k:] + corners[:k]), k


def corner_map(canon: dict, part: dict) -> np.ndarray:
    """For each vertex of `part`, the matching vertex index in `canon` (-1 if none).

    Animation files of one creature number their vertices differently (and may omit triangles), so
    vertices are matched through their triangles: same corner UVs = same triangle.
    """
    cidx = np.asarray(canon["indices"]).reshape(-1, 3)
    pool: dict = {}
    for t in cidx:
        key, k = _norm_tri(t, canon["uv"])
        pool.setdefault(key, []).append([int(t[(k + j) % 3]) for j in range(3)])
    cm = np.full(len(part["uv"]), -1, int)
    for t in np.asarray(part["indices"]).reshape(-1, 3):
        key, k = _norm_tri(t, part["uv"])
        if not pool.get(key):
            continue
        corners = pool[key].pop(0)
        for j in range(3):
            v = int(t[(k + j) % 3])
            if cm[v] < 0:
                cm[v] = corners[j]
    return cm


def remap_pose(canon_pose: np.ndarray, cm: np.ndarray) -> np.ndarray:
    """Pose positions in the file's own vertex numbering; unmatched vertices are placed far away."""
    out = np.full((len(cm), 3), FAR)
    ok = cm >= 0
    out[ok] = canon_pose[cm[ok]]
    return out


# ---------------------------------------------------------------- building from a spec file
import json
from pathlib import Path


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    for key in ("id", "label", "base", "pose"):
        if key not in spec:
            raise ValueError(f"creature spec {path} is missing {key!r}")
    return spec


def build_data(spec: dict, meshes_dir: Path, cache):
    """-> (data, missing, sources). data matches viewer_export.write_dk2v; sources maps each new
    texture name to ("base"|"donor", original cache texture name) so textures can be generated."""
    from scripts.viewer_creatures import CREATURES, find_animations
    from scripts.viewer_export import _collect

    cid = spec["id"]
    missing: set[str] = set()
    files, _ = find_animations(meshes_dir, {spec["base"]: CREATURES[spec["base"]]})
    base = {lab: _collect(p, cache, missing) for lab, p in files[spec["base"]].items()}
    pose_anim = spec["pose"]["animation"]
    pose = [p.copy() for p in base[pose_anim]["frames"][spec["pose"].get("frame", 0)]]
    anchors = spec.get("anchors", {})
    tex_map = {k: f"custom/{cid}/{v}" for k, v in spec["body_textures"].items()}
    sources = {tex_map[k]: ("base", k) for k in tex_map}
    geometry = {}   # new texture name -> {"uv", "indices", "points"} used to build texture masks

    donors = []   # (att, [(part, points, texture name)])
    for att in spec.get("attachments", []):
        dfiles, _ = find_animations(meshes_dir, {att["donor"]: CREATURES[att["donor"]]})
        dd = _collect(dfiles[att["donor"]][att.get("donor_animation", "idle1")], cache, missing)
        pieces = []
        for dp in att["donor_parts"]:
            part, pts = submesh(dd["parts"][dp["part"]],
                                dd["frames"][att.get("donor_frame", 0)][dp["part"]], dp.get("tri_region"))
            tex = f"custom/{cid}/{dp['texture']}"
            sources[tex] = ("donor", dd["parts"][dp["part"]]["texture"])
            geometry[tex] = {"uv": part["uv"], "indices": part["indices"], "points": pts}
            pieces.append((part, pts, tex))
        att = dict(att)
        att.setdefault("origin", np.concatenate([p for _, p, _ in pieces]).mean(0).tolist())
        donors.append((att, pieces))

    canon = base[pose_anim]
    canon_by_tex = {p["texture"]: i for i, p in enumerate(canon["parts"])}
    canon_pose = pose
    for i, cp in enumerate(canon["parts"]):
        if cp["texture"] in tex_map:
            geometry[tex_map[cp["texture"]]] = {"uv": cp["uv"], "indices": cp["indices"], "points": pose[i]}
    data, skipped = {}, {}
    for lab, a in base.items():
        in_file = {p["texture"] for p in a["parts"]}
        if not set(canon_by_tex) <= in_file:
            skipped[lab] = "this animation swaps the creature's model (e.g. electrocute -> skeleton)"
            continue
        # per-file pose in the file's own vertex numbering, so regions/anchors line up
        file_pose, cms = [], []
        for part in a["parts"]:
            ci = canon_by_tex.get(part["texture"])
            if ci is None:                       # a prop (bottle, chicken...): leave it alone
                cms.append(None); file_pose.append(np.full((len(part["uv"]), 3), FAR)); continue
            cm = corner_map(canon["parts"][ci], part)
            cms.append(cm); file_pose.append(remap_pose(canon_pose[ci], cm))
        frames = apply_reshape(a["frames"], file_pose, spec.get("reshape", []), anchors)
        parts = [{**p, "texture": tex_map.get(p["texture"], p["texture"])} for p in a["parts"]]
        for att, pieces in donors:
            for part, pts, tex in pieces:
                traj = make_attachment(file_pose, a["frames"], anchors, pts, att)
                idx = np.asarray(part["indices"], np.uint16).reshape(-1, 3)
                if flips_winding(att):
                    idx = idx[:, ::-1]
                parts.append({"texture": tex, "uv": part["uv"], "indices": idx.reshape(-1).copy()})
                frames = [fr + [traj[f]] for f, fr in enumerate(frames)]
        data[lab] = {"frames": frames, "parts": parts, "wrap": a["wrap"]}
    return {"data": data, "missing": missing, "sources": sources, "pose": pose, "skipped": skipped,
            "geometry": geometry}
