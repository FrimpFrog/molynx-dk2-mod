"""Molynx's creature-panel pictures, made from his own model.

DK2 gives every keeper creature two pictures in GUI\\Creatures\\Evil: `<name>-bg` (64 x 128, the
painted portrait on a deep blue ground, shown on the creature panel's tabs) and `<name>` (64 x 64, a
white emblem on a dark green ground). Both sit in the same riveted metal frame (FRAME pixels).

`render_portrait` renders one frame of one of his clips the way those portraits look: a low camera,
a warm key light from the upper left, a cold rim light from behind, on the blue ground, framed
about his head and shoulders. `mace_emblem` draws his emblem, a spiked mace, in the white line style
of the game's emblems (the Troll's hammer, the Bile Demon's face). `make_panel_pictures` puts both
in the Bile Demon's frames (the frames are the game's own pictures with the inside replaced).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scripts.dk2kmf import Kmf, evaluate_anim_frame, parse_kmf
from scripts.kmf_textured import _rotation, _yup, material_texture_name

FRAME = 7                                   # the game portraits' frame, pixels on every side
BG_SIZE, ICON_SIZE = (64, 128), (64, 64)
BLUE_CENTRE, BLUE_EDGE = (34, 52, 128), (14, 22, 62)      # the portraits' blue ground
GREEN_CENTRE, GREEN_EDGE = (46, 60, 14), (4, 6, 2)        # the emblems' green ground
EMBLEM_WHITE = (228, 230, 214)

# The picture: a frame of one of his clips, turned a little to one side, the camera a little below
# his chin.
PORTRAIT_CLIP, PORTRAIT_FRAME = "attack", 20     # facing ahead, mace raised behind his head
PORTRAIT_AZIMUTH, PORTRAIT_ELEVATION = 200.0, 8.0
KEY_LIGHT = np.array([-0.55, 0.55, -0.65])    # x right, y up, z away from the camera
RIM_LIGHT = np.array([0.7, 0.35, 0.6])        # from behind
AMBIENT, KEY, RIM = 0.30, 0.95, 0.85
RIM_COLOUR = np.array([1.0, 0.82, 0.62])


def _camera(azimuth: float, elevation: float) -> np.ndarray:
    e = np.radians(elevation)
    tilt = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return tilt @ _rotation(azimuth)


PN_LEVEL = 4          # each triangle drawn as PN_LEVEL**2 curved pieces (point-normal triangles)
BEHIND_BIAS = 100.0   # depth added to the mace (shaft and head): drawn only where nothing of him is


def _smooth_normals(points: list[np.ndarray], sprites) -> list[np.ndarray]:
    """Vertex normals averaged over every triangle at the same place (across sprites too), so the
    curved pieces of neighbouring triangles meet without cracks."""
    allp = np.concatenate(points)
    keys = np.round(allp / (np.ptp(allp, axis=0).max() * 1e-4)).astype(np.int64)
    _, weld = np.unique(keys, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    acc = np.zeros((weld.max() + 1, 3))
    start = 0
    for sprite, p in zip(sprites, points):
        t = np.asarray(sprite.triangles[0], int).reshape(-1, 3)
        if len(t):
            face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
            for k in range(3):
                np.add.at(acc, weld[start + t[:, k]], face)
        start += len(p)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    out, start = [], 0
    for p in points:
        out.append(acc[weld[start:start + len(p)]])
        start += len(p)
    return out


def _pn_patch(P: np.ndarray, N: np.ndarray, level: int):
    """(points, barycentric weights, sub-triangles) of a point-normal triangle cut `level` times
    along each side (Vlachos et al. 2001)."""
    P1, P2, P3 = P
    N1, N2, N3 = N
    w = lambda i, j: float((P[j] - P[i]) @ N[i])
    b210 = (2 * P1 + P2 - w(0, 1) * N1) / 3
    b120 = (2 * P2 + P1 - w(1, 0) * N2) / 3
    b021 = (2 * P2 + P3 - w(1, 2) * N2) / 3
    b012 = (2 * P3 + P2 - w(2, 1) * N3) / 3
    b102 = (2 * P3 + P1 - w(2, 0) * N3) / 3
    b201 = (2 * P1 + P3 - w(0, 2) * N1) / 3
    E = (b210 + b120 + b021 + b012 + b102 + b201) / 6
    b111 = E + (E - (P1 + P2 + P3) / 3) / 2
    bary, index = [], {}
    for i in range(level + 1):
        for j in range(level + 1 - i):
            index[i, j] = len(bary)
            bary.append(((level - i - j) / level, i / level, j / level))
    bary = np.array(bary)
    a, u, v = bary[:, :1], bary[:, 1:2], bary[:, 2:3]
    pts = (P1 * a ** 3 + P2 * u ** 3 + P3 * v ** 3 + 3 * b210 * a * a * u + 3 * b120 * a * u * u
           + 3 * b201 * a * a * v + 3 * b021 * u * u * v + 3 * b102 * a * v * v
           + 3 * b012 * u * v * v + 6 * b111 * a * u * v)
    tris = []
    for i in range(level):
        for j in range(level - i):
            tris.append((index[i, j], index[i + 1, j], index[i, j + 1]))
            if i + j < level - 1:
                tris.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    return pts, bary, tris


def _bilinear(tex: np.ndarray, u: np.ndarray, t: np.ndarray) -> np.ndarray:
    th, tw = tex.shape[:2]
    x, y = u * tw - 0.5, t * th - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0, y0 = x0 % tw, y0 % th
    x1, y1 = (x0 + 1) % tw, (y0 + 1) % th
    return ((tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy)
            + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy)


def render_portrait(model: Kmf, cache, frame: int, size: tuple[int, int], box: tuple, *,
                    azimuth: float = PORTRAIT_AZIMUTH, elevation: float = PORTRAIT_ELEVATION,
                    supersample: int = 3, behind: list | None = None,
                    level: int = PN_LEVEL) -> Image.Image:
    """RGBA picture (`size` = width, height) of `model` at `frame`, seeing the camera-space window
    `box` = (x centre, y centre, height); transparent where there is no model. Smooth: every
    triangle is drawn curved (point-normal triangles on normals averaged across the mesh) with
    filtered textures. `behind` (per sprite, a flag per vertex) marks triangles drawn behind all
    the others (the mace: the hands that hold it come in front, and it never pokes through him)."""
    sprites = model.anim.sprites
    rot = _camera(azimuth, elevation)
    pts = [_yup(p) @ rot.T for p in evaluate_anim_frame(model.anim, frame)]
    nrms = _smooth_normals(pts, sprites)
    w, h = size[0] * supersample, size[1] * supersample
    cx, cy, span = box
    scale = h / span
    colour = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    zbuf = np.full((h, w), np.inf, np.float32)
    key = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
    rim = RIM_LIGHT / np.linalg.norm(RIM_LIGHT)
    texcache = {}
    for k, (sprite, v, nrm) in enumerate(zip(sprites, pts, nrms)):
        if not sprite.triangles[0]:
            continue
        name = material_texture_name(cache, model.materials[sprite.material])
        if name is None:
            continue
        if name not in texcache:
            texcache[name] = np.asarray(cache.get(name).convert("RGB"), np.float32)
        tex = texcache[name]
        uv = np.array([(x.uv[0] / 32768.0, x.uv[1] / 32768.0) for x in sprite.vertices])
        flag = behind[k] if behind is not None and k < len(behind) else np.zeros(len(v), bool)
        for tri in sprite.triangles[0]:
            tri = list(tri)
            P, N = v[tri], nrm[tri]
            sub, bary, pieces = _pn_patch(P, N, level)
            subn = bary @ N
            subuv = bary @ uv[tri]
            bias = BEHIND_BIAS if all(flag[tri]) else 0.0
            sx = w / 2 + (sub[:, 0] - cx) * scale
            sy = h / 2 - (sub[:, 1] - cy) * scale
            z = sub[:, 2] + bias                                 # nearer is smaller
            for a, b, c in pieces:
                x0, x1 = int(max(min(sx[[a, b, c]]) - 1, 0)), int(min(max(sx[[a, b, c]]) + 2, w))
                y0, y1 = int(max(min(sy[[a, b, c]]) - 1, 0)), int(min(max(sy[[a, b, c]]) + 2, h))
                if x1 <= x0 or y1 <= y0:
                    continue
                den = (sy[b] - sy[c]) * (sx[a] - sx[c]) + (sx[c] - sx[b]) * (sy[a] - sy[c])
                if abs(den) < 1e-9:
                    continue
                gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
                w0 = ((sy[b] - sy[c]) * (gx - sx[c]) + (sx[c] - sx[b]) * (gy - sy[c])) / den
                w1 = ((sy[c] - sy[a]) * (gx - sx[c]) + (sx[a] - sx[c]) * (gy - sy[c])) / den
                w2 = 1 - w0 - w1
                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
                if not inside.any():
                    continue
                zz = w0 * z[a] + w1 * z[b] + w2 * z[c]
                win = inside & (zz < zbuf[y0:y1, x0:x1])
                if not win.any():
                    continue
                u = (w0 * subuv[a, 0] + w1 * subuv[b, 0] + w2 * subuv[c, 0]) % 1.0
                t = (w0 * subuv[a, 1] + w1 * subuv[b, 1] + w2 * subuv[c, 1]) % 1.0
                texel = _bilinear(tex, u, t)
                n = w0[..., None] * subn[a] + w1[..., None] * subn[b] + w2[..., None] * subn[c]
                n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
                n = np.where((n[..., 2:3] > 0), -n, n)            # face the camera (two-sided)
                diffuse = AMBIENT + KEY * np.clip(n @ key, 0, 1)
                edge = (1 - np.clip(-n[..., 2], 0, 1)) ** 2
                rimlight = RIM * edge * np.clip(n @ rim + 0.35, 0, 1)
                lit = texel * diffuse[..., None] + 255 * rimlight[..., None] * RIM_COLOUR * 0.55
                colour[y0:y1, x0:x1][win] = lit[win]
                alpha[y0:y1, x0:x1][win] = 255
                zbuf[y0:y1, x0:x1][win] = zz[win]
    rgba = np.dstack([np.clip(colour, 0, 255), alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA").resize(size, Image.LANCZOS)


def _radial(size: tuple[int, int], centre, edge, focus=(0.5, 0.4)) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx / max(w - 1, 1) - focus[0]) * 1.2, (yy / max(h - 1, 1) - focus[1]))
    t = np.clip(d / 0.75, 0, 1)[..., None]
    img = np.asarray(centre, float) * (1 - t) + np.asarray(edge, float) * t
    return Image.fromarray(img.astype(np.uint8), "RGB")


HEAD_WEIGHT = 0.85    # across, the picture centres on his head, drawn a little toward the mace


def portrait_box(model: Kmf, frame: int, inner: tuple[int, int], *, azimuth=PORTRAIT_AZIMUTH,
                 elevation=PORTRAIT_ELEVATION, headroom: float = 0.04, depth: float = 0.36,
                 mace_cut: float = 0.25) -> tuple:
    """The camera window of the panel portrait, tight on him like the game's portraits: his head
    (horns included) with a little room above, down to `depth` of his height; the mace's spiked
    head may run off the top, keeping `mace_cut` of its top cut off at most (it is the grip and the
    spikes that read)."""
    rot = _camera(azimuth, elevation)
    pts = evaluate_anim_frame(model.anim, frame)
    tex = [model.materials[s.material].textures[0] for s in model.anim.sprites]
    allv = _yup(np.concatenate(pts)) @ rot.T
    head = np.concatenate([_yup(p) @ rot.T for p, t in zip(pts, tex) if t == "Molynx_Head"])
    mace = np.concatenate([_yup(p) @ rot.T for p, t in zip(pts, tex) if t == "Molynx_Mace"])
    mace = mace[mace[:, 1] > np.quantile(mace[:, 1], 0.6)]          # its spiked head end
    tall = allv[:, 1].max() - allv[:, 1].min()
    mtop, mlow = mace[:, 1].max(), mace[:, 1].min()
    top = max(head[:, 1].max() + headroom * tall, mtop - mace_cut * (mtop - mlow))
    height = depth * tall + max(0.0, top - head[:, 1].max() - headroom * tall)
    across = HEAD_WEIGHT * head[:, 0].mean() + (1 - HEAD_WEIGHT) * mace[:, 0].mean()
    return float(across), float(top - height / 2), float(height)


def panel_portrait(model: Kmf, cache, frame: int = PORTRAIT_FRAME, behind=None) -> Image.Image:
    """The inside of the -bg picture (64 - 2 FRAME by 128 - 2 FRAME), at 4x for the downscale."""
    inner = (BG_SIZE[0] - 2 * FRAME, BG_SIZE[1] - 2 * FRAME)
    big = (inner[0] * 8, inner[1] * 8)
    box = portrait_box(model, frame, inner)
    fig = render_portrait(model, cache, frame, big, box, behind=behind)
    ground = _radial(big, BLUE_CENTRE, BLUE_EDGE, focus=(0.5, 0.3))
    glow = fig.split()[3].filter(ImageFilter.GaussianBlur(big[0] / 18))
    halo = Image.new("RGB", big, (60, 80, 170))
    ground = Image.composite(halo, ground, glow.point(lambda a: a * 0.35))
    ground.paste(fig, (0, 0), fig)
    return ground.resize(inner, Image.LANCZOS)


def mace_emblem(size: int) -> Image.Image:
    """His emblem: a spiked mace, slanted, drawn as white strokes on the dark green ground."""
    k = 8
    S = size * k
    img = _radial((S, S), GREEN_CENTRE, GREEN_EDGE, focus=(0.5, 0.55))
    d = ImageDraw.Draw(img)
    line = max(2, int(S * 0.045))
    head = np.array([0.62, 0.36]) * S
    r = S * 0.17
    grip = np.array([0.24, 0.84]) * S
    # the shaft, as an outline (two lines and an end cap), like the Troll's hammer
    axis = (head - grip) / np.linalg.norm(head - grip)
    side = np.array([-axis[1], axis[0]]) * S * 0.035
    end = head - axis * r * 0.95
    for s in (side, -side):
        d.line([tuple(grip + s), tuple(end + s)], fill=EMBLEM_WHITE, width=line)
    d.line([tuple(grip + side), tuple(grip - side)], fill=EMBLEM_WHITE, width=line)
    # the grip bands
    for f in (0.12, 0.22):
        p = grip + (head - grip) * f
        d.line([tuple(p + side * 1.6), tuple(p - side * 1.6)], fill=EMBLEM_WHITE, width=line)
    # the spikes: eight triangles round the head, then the head itself
    for i in range(8):
        a = np.radians(22.5 + 45 * i)
        u = np.array([np.cos(a), np.sin(a)])
        v = np.array([-u[1], u[0]])
        base = head + u * r * 0.8
        tip = head + u * r * 1.75
        d.polygon([tuple(base + v * r * 0.32), tuple(tip), tuple(base - v * r * 0.32)], fill=EMBLEM_WHITE)
    d.ellipse([tuple(head - r), tuple(head + r)], fill=EMBLEM_WHITE)
    d.ellipse([tuple(head - r * 0.55), tuple(head + r * 0.55)], outline=(150, 152, 140), width=line // 2)
    return img.resize((size, size), Image.LANCZOS)


def _framed(frame: Image.Image, picture: Image.Image) -> Image.Image:
    out = frame.convert("RGB").copy()
    out.paste(picture, (FRAME, FRAME))
    return out


def make_panel_pictures(model: Kmf, cache, bg_frame: Image.Image, icon_frame: Image.Image,
                        behind=None) -> tuple[Image.Image, Image.Image]:
    """(molynx-bg, molynx): the portrait and the emblem in the game's frames."""
    if bg_frame.size != BG_SIZE or icon_frame.size != ICON_SIZE:
        raise ValueError(f"the frames are {bg_frame.size} and {icon_frame.size}, "
                         f"not {BG_SIZE} and {ICON_SIZE}")
    bg = _framed(bg_frame, panel_portrait(model, cache, behind=behind))
    icon = _framed(icon_frame, mace_emblem(ICON_SIZE[0] - 2 * FRAME))
    return bg, icon


MACE_PART = 1    # the character's parts: 0 body (its first CLUB_VERTICES are the mace's shaft), 1 mace


def mace_flags(plan) -> list[np.ndarray]:
    """Per sprite of `plan` (molynx_assets.MolynxPlan), a flag per vertex: the mace's (its shaft
    in the body part and the whole mace part), for render_portrait's `behind`."""
    from scripts.behemoth_rig import CLUB_VERTICES
    return [np.full(len(sp.vertices), True) if sp.part == MACE_PART
            else (np.asarray(sp.vertices) < CLUB_VERTICES) if sp.part == 0
            else np.full(len(sp.vertices), False) for sp in plan.sprites]


class PanelCache:
    """The game's texture cache with his own textures added ({name: image}), as the renderer
    looks textures up (`names`, `get`; his are listed as <name>MM0)."""

    def __init__(self, game_cache, mine: dict):
        self.game, self.mine = game_cache, {f"{k}MM0": v for k, v in mine.items()}
        self.names = list(game_cache.names) + list(self.mine)

    def get(self, name):
        return self.mine[name] if name in self.mine else self.game.get(name)


def load_clip(assets_dir: Path, clip: str = PORTRAIT_CLIP) -> Kmf:
    return parse_kmf((Path(assets_dir) / f"Molynx-{clip}.kmf").read_bytes())
