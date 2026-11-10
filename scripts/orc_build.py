"""The Orc (from Dungeon Keeper 1) as a DK2 creature, made the way DK1 made him: the Troll, recoloured,
with hair added and his own weapons.

Every Orc file is the DK2 Troll's file of the same slot, re-encoded with
  * the Troll's two skins recoloured (`orc_texture`), shipped under new names so the Troll keeps his,
  * extra sprites carried rigidly with a body part: hair on the head, a buckler on the free forearm.

Body parts are found by texture position (UV), since the Troll's files each order their vertices
differently but all map the same two skins the same way:
  head      the face block of trollfrnt and the back-of-head block of trollback
  forearm   the hand end of the arm strips (both arms share these UVs; the free arm is the one on the
            other side of the body from the mallet in the file's first frame)
  mallet    the grey head and wooden handle blocks of trollfrnt
Each part's pose in a frame is the rigid fit (Kabsch) of its vertices to their place in the reference
frame (Troll-Idle1 frame 0), and an accessory follows that fit.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.dk2kmf import Kmf, Material, Mesh, Sprite, Vertex, evaluate_anim_frame, parse_kmf
from scripts.dk2kmf_write import encode_anim, write_kmf
from scripts.molynx_assets import GAME_MAX_KEYS, KEY_TOLERANCES

PREFIX = "Orc-"
REFERENCE = ("Troll-Idle1", 0)
# Troll record slot -> Troll file (Creatures.kwd record 10); the Orc's files keep the slot, not the name
TROLL_FILES = {
    "walk": "Troll_Walk", "run": "Troll_Run", "draggedPose": "TRO-Dragged_Pose", "recoilHff": "Troll_Recoil(hff)",
    "melee1": "Troll_Melee", "die": "Troll-Die", "happy": "Troll_Happy", "angry": "Troll_Angry",
    "stunnedPose": "Troll_Stunned", "swing": "Troll-Swing", "sleep": "Troll-bed(sleeping)", "eat": "Troll-Eat",
    "torture": "Tro-Torture", "drink": "Troll-Drink", "idle1": "Troll-Idle1", "recoilHfb": "Troll_Recoil(hfb)",
    "pray": "Troll-Pray", "fallback": "Troll-Fallback", "elec": "tro-elec", "getUp": "Troll-Getup",
    "dance": "Tro-Dance", "entrance": "tro-Entrance", "idle2": "Troll-Idle2", "drunk2": "Tro-Drunk",
    "drunkIdle": "TRO-Drunkidle", "walkback": "TRO_WalkBack", "poseFrame": "Tro-PoseFrame",
    "diePose": "Troll-DiePose",
}
SKINS = {"trollfrnt": "Orc_Front", "trollback": "Orc_Back"}

# UV boxes (u0, v0, u1, v1) in 0..1 of the named texture
HEAD_UV = {"trollfrnt": [(0.38, 0.26, 0.90, 0.66)], "trollback": [(0.0, 0.56, 0.40, 1.0)]}
HEAD_MIN_HEIGHT = 0.38      # of the reference frame (the Troll's top is 0.498): keys seen lower are not head
MALLET_UV = {"trollfrnt": [(0.0, 0.66, 0.45, 1.0), (0.45, 0.66, 0.90, 0.86)]}


def _in_boxes(uv: np.ndarray, boxes) -> np.ndarray:
    m = np.zeros(len(uv), bool)
    for u0, v0, u1, v1 in boxes:
        m |= (uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
    return m


@dataclass
class Clip:
    stem: str
    model: Kmf
    frames: list            # frames[f] = list of (n_s, 3) per sprite (all sprites, props included)
    troll: list             # indices of the sprites drawn with the Troll's skins

    @property
    def is_anim(self) -> bool:
        return self.model.anim is not None

    def sprites(self):
        return (self.model.anim or self.model.mesh).sprites

    def texture(self, s: int) -> str:
        return self.model.materials[self.sprites()[s].material].textures[0].lower()

    def uv(self, s: int) -> np.ndarray:
        return np.array([v.uv for v in self.sprites()[s].vertices], float) / 32768.0

    def part(self, boxes: dict) -> list[tuple[int, np.ndarray]]:
        """(sprite, vertex mask) for every Troll sprite with vertices in the part's UV boxes."""
        out = []
        for s in self.troll:
            t = self.texture(s)
            if t in boxes:
                m = _in_boxes(self.uv(s), boxes[t])
                if m.any():
                    out.append((s, m))
        return out

    def points(self, f: int, part) -> np.ndarray:
        return np.concatenate([self.frames[f][s][m] for s, m in part])

    def keys(self, part) -> list:
        """UV keys of a part's vertices (to pair them with the reference's)."""
        return [(self.texture(s), tuple(np.round(u, 4))) for s, m in part for u in self.uv(s)[m]]


def load_clip(meshes_dir: Path, stem: str) -> Clip:
    model = parse_kmf((Path(meshes_dir) / f"{stem}.kmf").read_bytes())
    if model.anim is not None:
        frames = [evaluate_anim_frame(model.anim, f) for f in range(model.anim.frame_count)]
    else:
        g = np.array(model.mesh.geometry, float)
        frames = [[g[[v.geom for v in s.vertices]] for s in model.mesh.sprites]]
    sprites = (model.anim or model.mesh).sprites
    troll = [i for i, s in enumerate(sprites) if model.materials[s.material].textures[0].lower() in SKINS]
    return Clip(stem, model, frames, troll)


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with R @ src_i + t ~ dst_i (least squares, proper rotation)."""
    cs, cd = src.mean(0), dst.mean(0)
    h = (src - cs).T @ (dst - cd)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cd - r @ cs


def _restrict(part, mask: np.ndarray):
    out, at = [], 0
    for s, m in part:
        n = int(m.sum())
        sub = m.copy()
        sub[np.flatnonzero(m)] = mask[at:at + n]
        at += n
        if sub.any():
            out.append((s, sub))
    return out


class PartTracker:
    """A body part's rigid pose in each frame of a clip, relative to the reference frame."""

    def __init__(self, ref: Clip, ref_frame: int, boxes: dict, min_height: float | None = None, select=None):
        self.boxes = boxes
        self.select = select
        part = ref.part(boxes)
        if select is not None:
            part = select(ref, part)
        pts = ref.points(ref_frame, part)
        self.ref = {}
        low = set()
        for k, p in zip(ref.keys(part), pts):
            if min_height is not None and -p[2] < min_height:
                low.add(k)
            self.ref.setdefault(k, []).append(p)
        for k in low:                      # a key also used below the part (a seam) is not the part's
            self.ref.pop(k, None)
        self.ref_body = np.concatenate([ref.frames[ref_frame][s] for s in ref.troll])

    def poses(self, clip: Clip, select=None):
        """Per frame (R, t) taking the reference part onto the clip's part.

        Within a clip every vertex keeps its identity, so frame f is fitted to the clip's own frame 0
        exactly; frame 0 is tied to the reference once, through the UV keys (a key the reference uses
        for several vertices, e.g. a mirrored half, is paired with the nearest after a first fit on the
        unambiguous ones, and the worst fifth of the pairs is dropped)."""
        part = clip.part(self.boxes)
        select = select or self.select
        if select is not None:
            try:
                part = select(clip, part)
            except ValueError:            # no piece of the part at all (TRO_WalkBack's other mapping)
                return self._poses_by_shape(clip)
        keys = clip.keys(part)
        mine = np.array([k in self.ref for k in keys])
        # keep only the vertices whose key is the part's (the UV box also catches seam vertices)
        part = _restrict(part, mine)
        keys = [k for k, m in zip(keys, mine) if m]
        if len(keys) < 5:
            return self._poses_by_shape(clip)
        p0 = clip.points(0, part)
        use = np.ones(len(keys), bool)
        uniq = [i for i, k in enumerate(keys) if use[i] and len(self.ref[k]) == 1]
        if len(uniq) >= 4:
            r, t = kabsch(np.array([self.ref[keys[i]][0] for i in uniq]), p0[uniq])
        else:
            r, t = np.eye(3), p0[use].mean(0) - np.concatenate([np.array(v) for v in self.ref.values()]).mean(0)
        for _ in range(3):
            idx = np.flatnonzero(use)
            src = np.array([np.array(self.ref[keys[i]])[np.argmin(np.linalg.norm(
                np.array(self.ref[keys[i]]) @ r.T + t - p0[i], axis=1))] for i in idx])
            err = np.linalg.norm(src @ r.T + t - p0[idx], axis=1)
            keep = err <= np.quantile(err, 0.8) + 1e-9
            r, t = kabsch(src[keep], p0[idx][keep])
        tie = (r, t)
        self.residual = float(np.quantile(err, 0.8))
        out = []
        for f in range(len(clip.frames)):
            rf, tf = kabsch(p0, clip.points(f, part))
            out.append((rf @ tie[0], rf @ tie[1] + tf))
        return out

    def _poses_by_shape(self, clip: Clip):
        """For a file mapped differently from the others (the game's TRO_WalkBack uses an older UV
        layout): the whole body of the clip's frame 0 is tied to the reference body by nearest-point
        ICP, and the part is the clip's vertices nearest the reference part's."""
        body = np.concatenate([clip.frames[0][s] for s in clip.troll])
        r, t = np.eye(3), body.mean(0) - self.ref_body.mean(0)
        for _ in range(40):
            moved = self.ref_body @ r.T + t
            near = body[np.argmin(np.linalg.norm(moved[:, None] - body[None], axis=2), axis=1)]
            r, t = kabsch(self.ref_body, near)
        refp = np.concatenate([np.array(v) for v in self.ref.values()]) @ r.T + t
        # the clip's vertex nearest each mapped reference vertex, and all vertices at those places
        cand = [(s, i) for s in clip.troll for i in range(len(clip.frames[0][s]))]
        cpos = np.array([clip.frames[0][s][i] for s, i in cand])
        picked = {int(np.argmin(np.linalg.norm(cpos - q, axis=1))) for q in refp}
        places = cpos[sorted(picked)]
        part = []
        for s in clip.troll:
            d = np.linalg.norm(clip.frames[0][s][:, None] - places[None], axis=2).min(1)
            m = d < 1e-4
            if m.any():
                part.append((s, m))
        p0 = clip.points(0, part)
        self.residual = float(np.quantile(np.linalg.norm(self.ref_body @ r.T + t - near, axis=1), 0.8))
        out = []
        for f in range(len(clip.frames)):
            rf, tf = kabsch(p0, clip.points(f, part))
            out.append((rf @ r, rf @ t + tf))
        return out


# ------------------------------------------------------------------------------------- writing


def _normals(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    n = np.zeros_like(p)
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    f = -np.cross(b - a, c - a)               # the game's outward normal
    for k in range(3):
        np.add.at(n, tris[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


@dataclass
class Accessory:
    """A rigid prop: points in reference-frame KMF space, triangles, UVs (0..1), its texture."""
    name: str
    points: np.ndarray
    tris: np.ndarray
    uv: np.ndarray
    texture: str
    tracker: str            # which PartTracker carries it


def _material(base: Material, name: str, texture: str) -> Material:
    return Material(name, [texture], base.flags, base.brightness, base.shininess, base.environment_map)


def orc_kmf(clip: Clip, accessories: list[Accessory], poses: dict, lods: int = 16, name: str | None = None) -> Kmf:
    """The clip with Orc skins and the accessories, re-encoded."""
    model = clip.model
    mats = []
    for m in model.materials:
        t = m.textures[0].lower()
        if t in SKINS:
            label = m.name[: len(m.name) - len(m.textures[0])] if m.name.endswith(m.textures[0]) else m.name
            mats.append(_material(m, label + SKINS[t], SKINS[t]))
        else:
            mats.append(m)
    base = model.materials[clip.sprites()[clip.troll[0]].material]
    sprites = [Sprite(s.material, s.triangles, list(s.vertices), s.mm_factor) for s in clip.sprites()]
    frames = [list(f) for f in clip.frames]
    rows_used = max(v.itab_index if clip.is_anim else v.geom for s in sprites for v in s.vertices) + 1
    for acc in accessories:
        tex_index = next((i for i, m in enumerate(mats) if m.textures == [acc.texture]), None)
        if tex_index is None:
            mats.append(_material(base, acc.name + acc.texture, acc.texture))
            tex_index = len(mats) - 1
        nrm = _normals(acc.points, acc.tris)
        verts = []
        for i in range(len(acc.points)):
            uv = tuple(int(round(np.clip(c, 0, 1) * 32768)) for c in acc.uv[i])
            v = Vertex(uv, tuple(float(c) for c in nrm[i]))
            if clip.is_anim:
                v.itab_index = rows_used + i
            else:
                v.geom = rows_used + i
            verts.append(v)
        rows_used += len(acc.points)
        tris = [tuple(int(c) for c in t) for t in acc.tris]
        sprites.append(Sprite(tex_index, [list(tris) for _ in range(lods)], verts, _mm(acc)))
        for f, (r, t) in enumerate(poses[acc.tracker]):
            frames[f].append(acc.points @ r.T + t)
    if clip.is_anim:
        a = model.anim
        anim = None
        for tol in KEY_TOLERANCES:
            anim = encode_anim(name or PREFIX + clip.stem, sprites, frames, a.wrap, a.controls, tolerance=tol)
            if len(anim.geom_xyz) - 1 <= GAME_MAX_KEYS:
                break
        return Kmf(model.version, "anim", mats, anim=anim, groups=model.groups)
    g = np.zeros((rows_used, 3))
    for s, p in zip(sprites, frames[0]):
        for v, q in zip(s.vertices, p):
            g[v.geom] = q
    old = model.mesh
    centre = (g.min(0) + g.max(0)) / 2
    mesh = Mesh(name or PREFIX + clip.stem, tuple(map(float, centre)), float(np.linalg.norm(g - centre, axis=1).max()),
                sprites, [tuple(map(float, q)) for q in g], old.controls)
    return Kmf(model.version, "mesh", mats, mesh=mesh, groups=model.groups)


def _mm(acc: Accessory) -> float:
    """mmFactor like molynx_assets: texels per model unit of the prop."""
    from scripts.molynx_assets import _mm_factor
    return _mm_factor(acc.points, acc.uv * 32768.0, acc.tris)


def write(model: Kmf, path: Path) -> None:
    Path(path).write_bytes(write_kmf(model))


# ------------------------------------------------------------------------------------- skins

ORC_SKIN_HUE = 344 / 360        # the DK1 Orc's magenta-crimson (sprite median (105, 36, 44), lit ~(190, 70, 120))
STONE = np.array([0.70, 0.64, 0.50])   # the mallet head: pale sandstone, not grey iron


def _hsv(a):
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    nz = d > 1e-9
    i = nz & (mx == r)
    h[i] = ((g - b)[i] / d[i]) % 6
    i = nz & (mx == g) & ~(mx == r)
    h[i] = (b - r)[i] / d[i] + 2
    i = nz & (mx == b) & ~(mx == r) & ~(mx == g)
    h[i] = (r - g)[i] / d[i] + 4
    return h / 6, np.where(mx > 0, d / np.maximum(mx, 1e-9), 0), mx


def _rgb(h, s, v):
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    out = np.stack([np.choose(i, [v, q, p, p, t, v]), np.choose(i, [t, v, v, q, p, p]),
                    np.choose(i, [p, p, t, v, v, q])], -1)
    return out


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def orc_skin(troll_tex, name: str):
    """The Troll's skin with his green turned the Orc's magenta (shading kept), and on the front skin the
    mallet's grey iron head turned pale stone."""
    from PIL import Image
    a = np.asarray(troll_tex.convert("RGB"), np.float32) / 255
    h, s, v = _hsv(a)
    deg = h * 360
    # how green a pixel is: 1 inside 70..165 deg, fading out to 45 and 185; greyish greens count less
    green = _smooth(deg, 45, 70) * (1 - _smooth(deg, 165, 185)) * _smooth(s, 0.08, 0.22)
    nh = np.full_like(h, ORC_SKIN_HUE)
    ns = np.clip(s * 0.95, 0, 0.85)
    nv = np.clip(v * 1.05, 0, 1)
    orc = _rgb(nh, ns, nv)
    out = a * (1 - green[..., None]) + orc * green[..., None]
    hgt, wid = a.shape[:2]
    yy, xx = np.mgrid[0:hgt, 0:wid] / np.array([hgt, wid])[:, None, None]
    loin = np.zeros((hgt, wid), bool)
    for u0, v0, u1, v1 in LOIN_UV[name]:
        loin |= (xx >= u0) & (xx <= u1) & (yy >= v0) & (yy <= v1)
    deg_ = h * 360
    cloth = loin & (deg_ > 8) & (deg_ < 50) & (s > 0.22)
    # grow the cloth mask by a texel so no brown fringe is left, then paint skin with its shading
    from PIL import ImageFilter as _F
    from PIL import Image as _I
    cloth = np.asarray(_I.fromarray((cloth * 255).astype(np.uint8)).filter(_F.MaxFilter(3)), np.uint8) > 0
    cloth &= loin
    lum = a.mean(-1)
    skin = _rgb(np.full_like(h, ORC_SKIN_HUE), np.full_like(h, 0.62), np.clip(0.20 + 0.85 * lum, 0, 0.75))
    out[cloth] = skin[cloth]
    if name == "trollfrnt":
        hgt, wid = a.shape[:2]
        yy, xx = np.mgrid[0:hgt, 0:wid] / np.array([hgt, wid])[:, None, None]
        box = np.zeros((hgt, wid), bool)
        for u0, v0, u1, v1 in MALLET_HEAD_UV["trollfrnt"]:
            box |= (xx >= u0) & (xx <= u1) & (yy >= v0) & (yy <= v1)
        grey = box & (s < 0.2)
        lum = a.mean(-1)
        stone = STONE[None, None] * (0.45 + 0.9 * lum[..., None])
        out[grey] = stone[grey]
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


# the Troll's loincloth (front apron, back band and seat): painted over with skin, the Orc wears none
LOIN_UV = {"trollfrnt": [(0.0, 0.42, 0.42, 0.64)], "trollback": [(0.0, 0.43, 0.44, 0.54), (0.42, 0.30, 0.90, 0.52)]}
MALLET_HEAD_UV = {"trollfrnt": [(0.0, 0.66, 0.45, 1.0), (0.45, 0.66, 0.84, 0.87), (0.76, 0.0, 1.0, 0.19)]}


class TextureOverlay:
    """A TextureCache with some textures replaced or added (for previews)."""

    def __init__(self, base, extra: dict):
        self.base = base
        self.extra = {f"{k}MM0": v for k, v in extra.items()}
        self.names = list(base.names) + list(self.extra)

    def get(self, name):
        return self.extra[name] if name in self.extra else self.base.get(name)


# ------------------------------------------------------------------------------------- the free hand

FOREARM_UV = {"trollfrnt": [(0.80, 0.18, 1.0, 0.36)], "trollback": [(0.84, 0.20, 1.0, 0.36)]}


def _pieces(clip: Clip, part):
    """The part's vertices split into connected pieces (triangles inside the part, and vertices at one
    place joined across sprites); each piece a list of (sprite, vertex)."""
    p0 = clip.frames[0]
    items = [(s, i) for s, m in part for i in np.flatnonzero(m)]
    index = {it: k for k, it in enumerate(items)}
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sprites = clip.sprites()
    for s, m in part:
        for tri in sprites[s].triangles[0]:
            ks = [index.get((s, int(v))) for v in tri]
            ks = [k for k in ks if k is not None]
            for k in ks[1:]:
                join(ks[0], k)
    pts = np.array([p0[s][i] for s, i in items])
    for a in range(len(items)):
        for b in np.flatnonzero(np.linalg.norm(pts - pts[a], axis=1) < 1e-4):
            join(a, int(b))
    groups = {}
    for k in range(len(items)):
        groups.setdefault(find(k), []).append(items[k])
    return [g for g in groups.values() if len(g) >= 3]


def _as_part(clip: Clip, part, keep: set):
    out = []
    for s, m in part:
        mm = np.zeros_like(m)
        for i in np.flatnonzero(m):
            mm[i] = (s, i) in keep
        if mm.any():
            out.append((s, mm))
    return out


def _key(clip: Clip, s: int, i: int):
    return (clip.texture(s), tuple(np.round(clip.uv(s)[i], 4)))


def free_forearm(clip: Clip, part):
    """The reference's free forearm: of the two forearm pieces, the one farther from the mallet head."""
    mallet = clip.points(0, clip.part(MALLET_HEAD_UV)).mean(0)
    best = max(_pieces(clip, part), key=lambda g: np.linalg.norm(
        np.mean([clip.frames[0][s][i] for s, i in g], 0) - mallet))
    return _as_part(clip, part, set(best))


class ForearmPicker:
    """In any clip, the forearm piece whose UV keys are most like the reference's free forearm's (the
    two forearms map mostly the same texels but not all; the keys only one of them uses decide)."""

    def __init__(self, ref: Clip):
        part = ref.part(FOREARM_UV)
        free = free_forearm(ref, part)
        free_keys = set(ref.keys(free))
        all_pieces = _pieces(ref, part)
        other = set()
        for g in all_pieces:
            ks = {_key(ref, s, i) for s, i in g}
            if not ks <= free_keys:
                other |= ks
        self.only_free = free_keys - other
        self.only_other = other - free_keys

    def __call__(self, clip: Clip, part):
        def score(g):
            ks = [_key(clip, s, i) for s, i in g]
            return sum(k in self.only_free for k in ks) - sum(k in self.only_other for k in ks)
        best = max(_pieces(clip, part), key=score)
        return _as_part(clip, part, set(best))


# ------------------------------------------------------------------------------------- everything


def export(meshes_dir: Path, cache, out_dir: Path) -> dict:
    """Write the Orc's 28 files (Orc-<slot>.kmf, one per Troll slot) and his three skins (PNG) into
    out_dir. Returns per file: frames, keys, triangles of the first LOD."""
    from scripts.dk2kmf_write import anim_invariants
    from scripts.orc_gear import GEAR, ball, buckler, gear_texture, mane
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = load_clip(meshes_dir, REFERENCE[0])
    beef_arms(ref)
    head = PartTracker(ref, REFERENCE[1], HEAD_UV, HEAD_MIN_HEIGHT)
    hand = PartTracker(ref, REFERENCE[1], FOREARM_UV, select=ForearmPicker(ref))
    hips = PartTracker(ref, REFERENCE[1], LOIN_UV)
    accs = [mane(np.concatenate([np.array(v) for v in head.ref.values()])),
            buckler(np.concatenate([np.array(v) for v in hand.ref.values()])),
            ball(crotch(ref, REFERENCE[1]))]
    info = {}
    for slot, stem in TROLL_FILES.items():
        clip = load_clip(meshes_dir, stem)
        beef_arms(clip)
        k = orc_kmf(clip, accs, {"head": head.poses(clip), "hand": hand.poses(clip), "hips": hips.poses(clip)},
                    name=PREFIX + slot)
        if k.anim is not None:
            bad = anim_invariants(k.anim)
            if bad:
                raise ValueError(f"{slot}: {bad}")
        (out_dir / f"{PREFIX}{slot}.kmf").write_bytes(write_kmf(k))
        x = k.anim or k.mesh
        info[slot] = {"from": stem, "frames": k.anim.frame_count if k.anim else 1,
                      "keys": len(k.anim.geom_xyz) - 1 if k.anim else 0,
                      "tris": sum(len(s.triangles[0]) for s in x.sprites)}
    for troll, orc in SKINS.items():
        orc_skin(cache.get(troll + "MM0"), troll).save(out_dir / f"{orc}.png")
    gear_texture().save(out_dir / f"{GEAR}.png")
    return info


# ------------------------------------------------------------------------------------- the Orc's body

ARM_UV = {"trollfrnt": [(0.50, 0.0, 1.0, 0.36)], "trollback": [(0.55, 0.0, 1.0, 0.36)]}
ARM_NOT_UV = {"trollfrnt": [(0.76, 0.0, 1.0, 0.19)]}     # the mallet's face sits in the arm strip's corner
ARM_THICK = 1.30        # upper arm and forearm girth, against the Troll's
SHOULDER_RAISE = 0.012  # shoulders and upper arms lifted (KMF units; the Troll stands 0.50)


def _smooth01(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def beef_arms(clip: Clip, thick: float = ARM_THICK, raise_: float = SHOULDER_RAISE) -> None:
    """The DK1 Orc's arms: thicker than the Troll's from shoulder to wrist and carried higher at the
    shoulder. Each arm is the larger connected piece of the arm strips (UV u runs 0.5 at the shoulder
    to 1.0 at the hand, t = 0..1); per frame its axis is the centroid of its vertices in bins of t,
    every vertex is pushed away from the axis by the girth factor (fading to none at the wrist so
    the hands, the mallet and the buckler keep their grip), and the shoulder end is lifted (fading out
    by the elbow). Rows shared between sprites move together. Frames are changed in place."""
    part = []
    for s in clip.troll:
        t = clip.texture(s)
        if t not in ARM_UV:
            continue
        m = _in_boxes(clip.uv(s), ARM_UV[t])
        if t in ARM_NOT_UV:
            m &= ~_in_boxes(clip.uv(s), ARM_NOT_UV[t])
        if m.any():
            part.append((s, m))
    if not part:
        return                                   # TRO_WalkBack: its UVs do not show the arms
    pieces = []
    for g in _pieces(clip, part):
        u = np.array([clip.uv(s)[i][0] for s, i in g])
        if u.max() - u.min() > 0.3:
            pieces.append((g, (u - 0.5) / 0.5))
    pieces = sorted(pieces, key=lambda p: -len(p[0]))[:2]
    sprites = clip.sprites()
    row_of = (lambda s, i: sprites[s].vertices[i].itab_index) if clip.is_anim else \
        (lambda s, i: sprites[s].vertices[i].geom)
    by_row = {}
    for s, sp in enumerate(sprites):
        for i in range(len(sp.vertices)):
            by_row.setdefault(row_of(s, i), []).append((s, i))
    bins = np.linspace(0, 1, 9)
    for g, t in pieces:
        t = np.clip(t, 0, 1)
        k = np.clip(np.digitize(t, bins) - 1, 0, len(bins) - 2)
        centres = (bins[:-1] + bins[1:]) / 2
        grow = 1 + (thick - 1) * (1 - _smooth01(t, 0.72, 0.93))
        lift = raise_ * (1 - _smooth01(t, 0.12, 0.5))
        rows = [row_of(s, i) for s, i in g]
        for f, fr in enumerate(clip.frames):
            P = np.array([fr[s][i] for s, i in g])
            cen = np.array([P[k == b].mean(0) if (k == b).any() else np.full(3, np.nan) for b in range(len(centres))])
            ok = ~np.isnan(cen[:, 0])
            axis = np.stack([np.interp(t, centres[ok], cen[ok, j]) for j in range(3)], 1)
            newp = axis + (P - axis) * grow[:, None]
            newp[:, 2] -= lift
            done = set()
            for (s, i), r, q in zip(g, rows, newp):
                if r in done:
                    continue
                done.add(r)
                for s2, i2 in by_row[r]:
                    fr[s2][i2] = q


def crotch(ref: Clip, frame: int) -> np.ndarray:
    """Where the front apron of the Troll's loincloth ends between the legs (reference frame)."""
    pts = []
    for s in ref.troll:
        if ref.texture(s) == "trollfrnt":
            m = _in_boxes(ref.uv(s), LOIN_UV["trollfrnt"])
            pts.append(ref.frames[frame][s][m])
    p = np.concatenate(pts)
    low = p[p[:, 2] >= np.quantile(p[:, 2], 0.8)]          # the lowest fifth (z down)
    c = low.mean(0)
    c[1] = p[:, 1].max() - 0.004                           # at the front of the body
    return c
