"""Creature-panel pictures of the Orc, the Classic Reaper and the Dragon, made from their own
game files the molynx_portrait way.

Each creature gets the two pictures DK2 gives a keeper creature in GUI\\Creatures\\Evil:
  <icon>-bg  64 x 128: a frame of one of his own clips rendered by molynx_portrait.render_portrait
             (low camera, warm key light, cold rim light, smooth shading, his own skins) on the
             portraits' blue ground, framed on his head and shoulders like the game's portraits;
  <icon>     64 x 64: a white emblem on the emblems' dark green ground, in the game's line style.
             The Orc's and the Reaper's start from their donor's own emblem (as the Necromancer's
             is the Black Knight's with a horn knocked off): the Troll's hammer with the Orc's round
             buckler, a skull on crossed bones, over it; Horny's horned face with a clean scythe
             beside, behind or under it that never crosses the face (REAPER_EMBLEM_STYLES). The Dragon's is drawn: a horned dragon's head in profile, jaws open; Sejanus'
             a horse's head in profile with pricked ears and a spiked mane.
Both are set into the donor's own frame (the game's picture with its inside replaced).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scripts.dk2kmf import parse_kmf, evaluate_anim_frame
from scripts.kmf_textured import _yup
from scripts.molynx_portrait import (BG_SIZE, BLUE_CENTRE, BLUE_EDGE, EMBLEM_WHITE, FRAME, GREEN_CENTRE,
                                     GREEN_EDGE, ICON_SIZE, PanelCache, _camera, _framed, _radial,
                                     render_portrait)

EVIL = "GUI\\Creatures\\Evil\\"


@dataclass(frozen=True)
class Shot:
    clip: str             # his file (without .kmf)
    frame: int
    azimuth: float
    elevation: float
    head: tuple           # textures of his head: the picture is centred across on them (() = his top part)
    top: str              # "model": the picture's top is his highest point (horns); "head": his head's
    frac: float           # picture height, as a share of his height in this frame
    across: float = 0.0   # shift right, as a share of the picture height
    down: float = 0.0     # shift the picture down (the figure up), as a share of the picture height
    light: tuple | None = None   # (key direction (x right, y up, z away), ambient, key, rim); None: Molynx's
    head_uv: tuple | None = None # (texture, u0, v0, u1, v1): his head is these points (no head texture of its own)
    by_head: bool = False        # frac is a share of his head's height (else of his whole height)
    grade: tuple | None = None   # (r, g, b gains, gamma) on the figure: deeper colour and shadows, as the
                                 # game's painted portraits have (None: as rendered)
    grade_textures: tuple = ()   # if given, the grade goes on these skins before the render instead of on
                                 # the whole picture (the Orc: his skin, not his bronze buckler or mane)
    sharpen: float = 0.0         # unsharp-mask strength on the final picture (a tight crop magnifies the skins)
    pose: tuple = ()             # arm turns for the picture only (his files are untouched): each
                                 # (side, pivot, axis, degrees, d0, d1) turns the points on that side
                                 # (sign of x, y up) about the axis through the pivot, by the full angle where
                                 # a point has moved more than d1 since the clip's frame 0 and not at all under
                                 # d0 (so the arm and what it holds turn, the shoulder bends, the body stays)


# The vanilla portraits' light: a warm key from the upper left and front, darker shadows than Molynx's
VANILLA_LIGHT = ((-0.55, 0.55, -0.65), 0.16, 1.1, 0.6)
ORC_FACE = ("Orc_Front", 0.38, 0.26, 0.90, 0.66)        # orc_build.HEAD_UV: the face block of his front skin
_R_SHOULDER, _L_SHOULDER = (0.07, 0.37, 0.05), (-0.06, 0.37, 0.05)   # Orc-prayStart, y up
ORC_ARMS = ((1, _R_SHOULDER, (1, 0, 0), -55.0, 0.04, 0.1),          # mallet arm up beside his head
            (1, _R_SHOULDER, (0, 1, 0), -12.0, 0.04, 0.1),          # ... and a little in toward it
            (-1, _L_SHOULDER, (0, 1, 0), 30.0, 0.04, 0.1),          # buckler arm in, in front of him
            ("part", 4, 0.7, (0.0, -0.01, 0.0)))                    # his belt knob smaller and lower

SHOTS = {
    # Soles, 14 Nov (round 2: "too wide", tight like the game's): the face fills the upper half, brows near
    # the top, the mane at the edges, the belly cropped, the skull buckler's rim low right; a darker
    # shadow side and a crimson grade so the skin reads deep crimson, not pink
    # (round 3: the face centred, both eyes, nose and tusk inside, a slight 3/4 turn; the crimson grade on
    # his skins only, so the buckler's rim stays bronze; sharpened after the downsample)
    # (round 5: "do it where his weapon and shield are visible", composed like troll-bg): the start of his
    # pray (arms coming up), with his mallet arm turned up so the sandstone head stands beside his face
    # and his buckler arm turned in so the skull buckler sits in front, low; the face 3/4, centred. The
    # crimson grade only on skin texels, so the stone and the handle keep their own colours
    # (round 6, Soles: option B, the small round knob below the buckler lower and smaller)
    "orc": Shot("Orc-prayStart", 3, 190.0, -8.0, (), "model", 0.68, 0.0, -0.02,
                light=((-0.6, 0.55, -0.6), 0.10, 1.25, 0.5), head_uv=ORC_FACE,
                grade=(1.1, 0.82, 0.74, 1.15), grade_textures=("Orc_Front", "Orc_Back"), sharpen=0.6,
                pose=ORC_ARMS),
    # the frontal stare, head and horns filling the top half (the horn tips run off the edges, as the
    # game's reaper-bg), the chest chain at the bottom edge
    "hreaper": Shot("HReaper-Roar", 0, 182.0, 0.0, ("HReaper_Face",), "model", 0.36, 0.0, 0.14,
                    light=VANILLA_LIGHT, grade=(1.0, 0.88, 0.84, 1.15)),
    "sejanus": Shot("Sejanus-Idle1", 0, 160.0, 6.0, ("Sejanus_Head",), "model", 0.47, 0.05),
    "gargoyle": Shot("Gargoyle-Idle1", 0, 195.0, -6.0, ("Gargoyle_Head",), "head", 2.2, 0.0, -0.06,
                     light=VANILLA_LIGHT, by_head=True, sharpen=0.6),
    # a portrait of his face: head turned toward the viewer, snout, eye and horns large in the upper half,
    # the neck's curve and a wing edge below
    "dragon": Shot("Dragon-Angry", 20, 160.0, 6.0, ("Dragon_Head",), "head", 1.8, -0.04, -0.03,
                   light=VANILLA_LIGHT, by_head=True, grade=(1.0, 0.95, 0.85, 1.12)),
    # Krefft (16 Nov): the end of the idle's look round (Idle1 130: head turned toward the viewer, the smirk),
    # seen from above (a turtle's eyes sit on top of the head) and 3/4: the eye with its catch-light and the
    # smug mouth line in the upper third against the blue, the shell's rim behind, the plastron and a paddle
    # below; the key light from his face's side so the head stands out from the body
    # (round 2: the side-on first framing cut the snout off and read as a shell; round 3: the eye read as a
    # dot and the head was lost against the plastron)
    "krefft": Shot("Krefft-Idle1", 130, 205.0, -28.0, ("Krefft_Head",), "model", 0.85, 0.02, 0.0,
                   light=((0.4, 0.6, -0.65), 0.12, 1.25, 0.8), grade=(1.0, 0.97, 0.88, 1.12), sharpen=0.5),
}
DONOR_ICON = {"orc": "troll", "hreaper": "reaper", "dragon": "bile_demon", "sejanus": "goblin", "gargoyle": "goblin",
              "krefft": "bile_demon"}


def _turn(axis, degrees):
    a = np.array(axis, float) / np.linalg.norm(axis)
    t = np.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def posed_frame(anim, frame, pose):
    """evaluate_anim_frame with Shot.pose's arm turns and part moves (in the y-up space of _yup)."""
    pts = evaluate_anim_frame(anim, frame)
    if not pose:
        return pts
    rest = evaluate_anim_frame(anim, 0)
    out = []
    for k, (p, q) in enumerate(zip(pts, rest)):
        p, q = _yup(p), _yup(q)
        moved = np.linalg.norm(p - q, axis=1)
        side = np.sign(p[:, 0])
        for entry in pose:
            if entry[0] == "part":
                # ("part", sprite, scale, (dx, dy, dz)): one small separate piece scaled about its
                # centre and moved (the Orc's belt knob)
                _tag, sprite, scale, shift = entry
                if k == sprite:
                    c = p.mean(axis=0)
                    p = c + (p - c) * scale + np.array(shift, float)
                continue
            s, pivot, axis, degrees, d0, d1 = entry
            w = np.clip((moved - d0) / (d1 - d0), 0, 1) * (side == s)
            w = w * w * (3 - 2 * w)
            pivot = np.array(pivot, float)
            for j in np.nonzero(w)[0]:
                p[j] = pivot + _turn(axis, degrees * w[j]) @ (p[j] - pivot)
        out.append(np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=-1))    # back from _yup
    return out


@contextmanager
def _posed(pose):
    """molynx_portrait's frames with these arm turns, for one render."""
    import scripts.molynx_portrait as mp
    if not pose:
        yield
        return
    saved = mp.evaluate_anim_frame
    mp.evaluate_anim_frame = lambda anim, frame: posed_frame(anim, frame, pose)
    try:
        yield
    finally:
        mp.evaluate_anim_frame = saved


def _points(model, frame, pose=()):
    if model.anim is not None:
        pts = posed_frame(model.anim, frame, pose)
        sprites = model.anim.sprites
    else:
        g = np.array(model.mesh.geometry, float)
        sprites = model.mesh.sprites
        pts = [g[[v.geom for v in s.vertices]] for s in sprites]
    tex = [model.materials[s.material].textures[0] for s in sprites]
    return pts, tex


def shot_box(model, shot: Shot) -> tuple:
    """(x centre, y centre, height) in camera space for the -bg picture."""
    rot = _camera(shot.azimuth, shot.elevation)
    pts, tex = _points(model, shot.frame, shot.pose)
    allv = _yup(np.concatenate(pts)) @ rot.T
    tall = allv[:, 1].max() - allv[:, 1].min()
    if shot.head_uv:
        name, u0, v0, u1, v1 = shot.head_uv
        sprites = (model.anim or model.mesh).sprites
        parts = []
        for p, t, sp in zip(pts, tex, sprites):
            if t == name:
                uv = np.array([v.uv for v in sp.vertices], float) / 32768.0
                m = (uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
                parts.append(_yup(p[m]) @ rot.T)
        head = np.concatenate(parts)
        head = head[head[:, 1] > allv[:, 1].max() - 0.3 * tall]       # the face block also maps lower seams
    elif shot.head:
        head = np.concatenate([_yup(p) @ rot.T for p, t in zip(pts, tex) if t in shot.head])
        head = head[head[:, 1] > head[:, 1].max() - 0.3 * tall]       # the head, not the neck below it
    else:
        head = allv[allv[:, 1] > allv[:, 1].max() - 0.2 * tall]
    height = shot.frac * (np.ptp(head[:, 1]) if shot.by_head else tall)
    top = (allv if shot.top == "model" else head)[:, 1].max() + 0.03 * height
    return (float(np.median(head[:, 0]) + shot.across * height), float(top - height / 2 - shot.down * height),
            float(height))


class _GradedCache:
    """A texture cache with some skins colour-graded ((r, g, b gains, gamma), as Shot.grade)."""

    def __init__(self, base, names, grade):
        self.base, self.names = base, base.names
        self.graded = {n.lower() for n in names}
        self.grade = grade
        self.memo = {}

    def get(self, name):
        stem = name[:-3] if name.upper().endswith(("MM0", "MM1", "MM2", "MM3")) else name
        if stem.lower() not in self.graded:
            return self.base.get(name)
        if name not in self.memo:
            a = np.asarray(self.base.get(name).convert("RGB"), float) / 255.0
            g = np.clip(a * np.array(self.grade[:3]), 0, 1) ** self.grade[3]
            # only the skin (magenta: more blue than green); grey stone and brown wood stay as painted
            m = np.clip((a[..., 2] - a[..., 1] + 0.01) / 0.05, 0, 1)[..., None]
            a = m * g + (1 - m) * a
            self.memo[name] = Image.fromarray((a * 255).astype(np.uint8), "RGB")
        return self.memo[name]


@contextmanager
def _lighting(light):
    """molynx_portrait's lights for one render (its module values), put back afterwards."""
    import scripts.molynx_portrait as mp
    if light is None:
        yield
        return
    saved = mp.KEY_LIGHT, mp.AMBIENT, mp.KEY, mp.RIM
    mp.KEY_LIGHT, mp.AMBIENT, mp.KEY, mp.RIM = np.array(light[0], float), light[1], light[2], light[3]
    try:
        yield
    finally:
        mp.KEY_LIGHT, mp.AMBIENT, mp.KEY, mp.RIM = saved


def portrait(model, cache, shot: Shot) -> Image.Image:
    inner = (BG_SIZE[0] - 2 * FRAME, BG_SIZE[1] - 2 * FRAME)
    big = (inner[0] * 8, inner[1] * 8)
    box = shot_box(model, shot)
    if shot.grade and shot.grade_textures:
        cache = _GradedCache(cache, shot.grade_textures, shot.grade)
    with _lighting(shot.light), _posed(shot.pose):
        fig = render_portrait(model, cache, shot.frame, big, box, azimuth=shot.azimuth, elevation=shot.elevation)
    if shot.grade and not shot.grade_textures:
        a = np.asarray(fig, float)
        rgb = np.clip(a[..., :3] * np.array(shot.grade[:3]), 0, 255) / 255.0
        a[..., :3] = 255.0 * rgb ** shot.grade[3]
        fig = Image.fromarray(a.astype(np.uint8), "RGBA")
    ground = _radial(big, BLUE_CENTRE, BLUE_EDGE, focus=(0.5, 0.3))
    glow = fig.split()[3].filter(ImageFilter.GaussianBlur(big[0] / 18))
    halo = Image.new("RGB", big, (60, 80, 170))
    ground = Image.composite(halo, ground, glow.point(lambda a: a * 0.35))
    ground.paste(fig, (0, 0), fig)
    out = ground.resize(inner, Image.LANCZOS)
    if shot.sharpen:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(shot.sharpen * 100), threshold=1))
    return out


# ------------------------------------------------------------------------------ emblems

K = 8                                   # drawn at K x, then reduced


def _inner(img: Image.Image) -> Image.Image:
    return img.crop((FRAME, FRAME, img.width - FRAME, img.height - FRAME))


def _white_mask(inner: Image.Image) -> np.ndarray:
    a = np.asarray(inner.convert("RGB"), float)
    return (a.min(2) > 150) & (np.ptp(a, axis=2) < 60)


def orc_emblem(troll_icon: Image.Image) -> Image.Image:
    """The Troll's hammer, with the Orc's round buckler (a skull on crossed bones) over its lower right."""
    inner = _inner(troll_icon).convert("RGB")
    n = inner.width
    S = n * K
    img = inner.resize((S, S), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    c = np.array([0.66, 0.66]) * S
    r = 0.25 * S
    d.ellipse([tuple(c - r * 1.08), tuple(c + r * 1.08)], fill=(14, 18, 6))          # cut the hammer away
    line = int(S * 0.04)
    d.ellipse([tuple(c - r), tuple(c + r)], outline=EMBLEM_WHITE, width=line)
    # crossed bones
    for sx in (1, -1):
        a = c + np.array([-0.62 * sx, 0.55]) * r
        b = c + np.array([0.62 * sx, -0.55]) * r
        d.line([tuple(a), tuple(b)], fill=EMBLEM_WHITE, width=int(line * 0.9))
        for p in (a, b):
            for k in (-1, 1):
                q = p + np.array([k * 0.09, 0.0]) * r
                d.ellipse([tuple(q - 0.09 * r), tuple(q + 0.09 * r)], fill=EMBLEM_WHITE)
    # the skull: a dome and a jaw, no eyes or mouth (the buckler's embossed skull)
    sk = c + np.array([0, -0.08]) * r
    d.ellipse([tuple(sk - np.array([0.38, 0.36]) * r), tuple(sk + np.array([0.38, 0.30]) * r)], fill=EMBLEM_WHITE)
    d.rectangle([tuple(sk + np.array([-0.22, 0.1]) * r), tuple(sk + np.array([0.22, 0.44]) * r)], fill=EMBLEM_WHITE)
    return img.resize((n, n), Image.LANCZOS)


REAPER_EMBLEM_STYLES = ("beside", "behind", "under")
REAPER_EMBLEM_STYLE = "upright"         # Soles, 15 Nov: the first try ("hreaper v2"), not the crescent


def _face_layer(reaper_icon: Image.Image, S: int, scale: float, at: tuple) -> tuple[Image.Image, np.ndarray]:
    """Horny's emblem face (the white shape and its inner lines, from the game's icon) on a fresh green
    ground, scaled about its centre and moved so its centre is at `at` (shares of S). Returns the
    picture and the face's mask (grown a little: nothing else is drawn there)."""
    inner = _inner(reaper_icon).convert("RGB")
    big = np.asarray(inner.resize((S, S), Image.LANCZOS), float)
    lum = big @ np.array([0.3, 0.59, 0.11])
    white = lum > 105                                  # the face (its lower half is shaded grey), not the ground
    keep = np.asarray(Image.fromarray((white * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)), bool)
    body = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)), bool)
    ys, xs = np.nonzero(body)
    c = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    ground = np.asarray(_radial((S, S), GREEN_CENTRE, GREEN_EDGE, focus=(0.5, 0.55)), float)
    # where the face is: the icon's pixels (white, its grey edges and its dark inner lines)
    region = np.asarray(Image.fromarray((body * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)), float) / 255
    face = Image.fromarray((big * region[..., None] + ground * (1 - region[..., None])).astype(np.uint8))
    alpha = Image.fromarray((region * 255).astype(np.uint8))
    size = int(round(S * scale))
    face, alpha = face.resize((size, size), Image.LANCZOS), alpha.resize((size, size), Image.LANCZOS)
    off = (int(round(at[0] * S - c[0] * scale)), int(round(at[1] * S - c[1] * scale)))
    out = Image.fromarray(ground.astype(np.uint8))
    out.paste(face, off, alpha)
    mask = Image.new("L", (S, S), 0)
    mask.paste(alpha.point(lambda v: 255 if v > 40 else 0), off)
    grown = np.asarray(mask.filter(ImageFilter.MaxFilter(int(S * 0.06) | 1)), bool)
    return out, grown


def _blade(S: int, tip_from: tuple, toward: float, length: float, bend: float, thick: float) -> Image.Image:
    """A scythe blade as a mask: a crescent from `tip_from` (where it meets the shaft, shares of S)
    running `length` (share of S) in the direction `toward` (degrees, 0 = right, 90 = down), curving
    down by `bend`, `thick` at the root and tapering to a point."""
    n = 40
    t = np.linspace(0, 1, n)
    a = np.radians(toward)
    d = np.array([np.cos(a), np.sin(a)])
    nrm = np.array([-d[1], d[0]])
    spine = np.array(tip_from)[None] + t[:, None] * length * d[None] + (bend * length * t ** 2)[:, None] * nrm[None]
    w = thick * (1 - t) ** 0.8
    outer = spine - w[:, None] * nrm[None] * 0.35
    inner = spine + w[:, None] * nrm[None] * 0.65
    poly = np.concatenate([outer, inner[::-1]]) * S
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=255)
    return m


def _crescent(S: int, root: tuple, R: float, sweep: float, thick: float, side: int = -1) -> Image.Image:
    """A scythe blade as a mask: a true crescent hooking from `root` (the shaft's top, shares of S) to
    the left (side -1) and down: the disc of radius R whose top is at the root, less the same disc
    moved down by `thick`, kept from the top through `sweep` degrees (thick at the root, a point at
    the tip)."""
    yy, xx = np.mgrid[0:S, 0:S] / S
    cx, cy = root[0], root[1] + R
    outer = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    inner = (xx - cx) ** 2 + (yy - cy - thick) ** 2 <= R ** 2
    ang = np.degrees(np.arctan2(side * (xx - cx), -(yy - cy)))
    return Image.fromarray(((outer & ~inner & (ang >= -4) & (ang <= sweep)) * 255).astype(np.uint8))


def reaper_emblem(reaper_icon: Image.Image, style: str = REAPER_EMBLEM_STYLE) -> Image.Image:
    """Horny's horned-face emblem with a clean scythe that never crosses the face, kept inside the
    picture (Soles, 14 Nov: the first version's scythe cut across the face and ran off the frame):
      upright the first try (round 1): the face smaller and to the left, the scythe upright on the right
              edge, a thin shaft and a straight-backed blade over the right horn (the installed one);
      beside  the face a little smaller and to the left, the scythe upright on the right, its blade
              hooking left over the right horn;
      behind  the face nearly as the game draws it, the shaft behind it (hidden where the face is),
              the blade above the left horn;
      under   the face a little smaller and higher, the scythe lying under its chin, blade up at the right."""
    n = _inner(reaper_icon).width
    S = n * K
    line = int(S * 0.045)
    if style == "upright":                                                   # round 1's "beside" (Soles'
        pic, face = _face_layer(reaper_icon, S, 0.80, (0.40, 0.57))        # pick, 15 Nov: "hreaper v2")
        shaft = [(0.86, 0.93), (0.86, 0.11)]
        blade = _blade(S, (0.87, 0.11), 176, 0.44, 0.22, 0.12)
    elif style == "beside":
        pic, face = _face_layer(reaper_icon, S, 0.70, (0.35, 0.65))        # (round 2: a crescent
        shaft = [(0.84, 0.93), (0.84, 0.10)]                                 # blade, a thicker shaft)
        line = int(S * 0.075)
        blade = _crescent(S, (0.84, 0.09), 0.40, 62, 0.14)
    elif style == "behind":
        pic, face = _face_layer(reaper_icon, S, 0.92, (0.5, 0.55))
        shaft = [(0.90, 0.93), (0.16, 0.09)]
        blade = _blade(S, (0.15, 0.08), 8, 0.40, 0.30, 0.11)
    elif style == "under":
        pic, face = _face_layer(reaper_icon, S, 0.78, (0.5, 0.42))
        shaft = [(0.10, 0.86), (0.82, 0.86)]
        blade = _blade(S, (0.82, 0.86), 262, 0.36, -0.40, 0.11)
    else:
        raise ValueError(style)
    layer = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(layer)
    d.line([tuple(np.array(p) * S) for p in shaft], fill=255, width=line)
    if style == "upright":                                                   # its foot rounded, as drawn then
        d.ellipse([tuple(np.array(shaft[0]) * S - line * 0.7), tuple(np.array(shaft[0]) * S + line * 0.7)], fill=255)
    else:
        for end in shaft:                                                    # round ends
            d.ellipse([tuple(np.array(end) * S - line / 2), tuple(np.array(end) * S + line / 2)], fill=255)
    layer = Image.fromarray(np.maximum(np.asarray(layer), np.asarray(blade)))
    layer = Image.fromarray(np.where(face, 0, np.asarray(layer)).astype(np.uint8))      # never over the face
    out = Image.composite(Image.new("RGB", (S, S), EMBLEM_WHITE), pic, layer)
    return out.resize((n, n), Image.LANCZOS)


def dragon_emblem(size: int) -> Image.Image:
    """A horned dragon's head in profile, jaws open, as white shapes on the green ground."""
    S = size * K
    img = _radial((S, S), GREEN_CENTRE, GREEN_EDGE, focus=(0.5, 0.55))
    d = ImageDraw.Draw(img)
    P = lambda pts: [(x * S, y * S) for x, y in pts]
    # skull and upper jaw (snout to the left)
    d.polygon(P([(0.10, 0.44), (0.20, 0.36), (0.40, 0.30), (0.56, 0.22), (0.70, 0.22), (0.80, 0.30),
                 (0.84, 0.42), (0.80, 0.54), (0.66, 0.56), (0.50, 0.50), (0.30, 0.50), (0.12, 0.52)]),
              fill=EMBLEM_WHITE)
    # lower jaw, dropped open
    d.polygon(P([(0.16, 0.66), (0.34, 0.58), (0.52, 0.56), (0.66, 0.60), (0.74, 0.70), (0.62, 0.72),
                 (0.40, 0.70), (0.20, 0.72)]), fill=EMBLEM_WHITE)
    # neck
    d.polygon(P([(0.66, 0.50), (0.84, 0.44), (0.92, 0.70), (0.90, 0.94), (0.70, 0.94), (0.70, 0.72)]),
              fill=EMBLEM_WHITE)
    # horns, swept back
    d.polygon(P([(0.62, 0.24), (0.78, 0.04), (0.74, 0.22)]), fill=EMBLEM_WHITE)
    d.polygon(P([(0.72, 0.26), (0.96, 0.14), (0.82, 0.32)]), fill=EMBLEM_WHITE)
    # teeth into the open mouth
    for x in (0.20, 0.28, 0.36, 0.44):
        d.polygon(P([(x, 0.50), (x + 0.03, 0.56), (x + 0.06, 0.50)]), fill=EMBLEM_WHITE)
        d.polygon(P([(x + 0.02, 0.60), (x + 0.05, 0.54), (x + 0.08, 0.60)]), fill=EMBLEM_WHITE)
    # the eye and the nostril, cut out
    dark = (20, 26, 8)
    d.polygon(P([(0.56, 0.32), (0.66, 0.30), (0.62, 0.37)]), fill=dark)
    d.ellipse(P([(0.14, 0.41), (0.18, 0.45)]), fill=dark)
    # the line between the jaws' backs
    d.line(P([(0.50, 0.50), (0.68, 0.58)]), fill=dark, width=int(S * 0.02))
    return img.resize((size, size), Image.LANCZOS)


def sejanus_emblem(size: int) -> Image.Image:
    """A horse's head in profile, ears pricked and mane in spikes, as white shapes on the green ground."""
    S = size * K
    img = _radial((S, S), GREEN_CENTRE, GREEN_EDGE, focus=(0.5, 0.55))
    d = ImageDraw.Draw(img)
    P = lambda pts: [(x * S, y * S) for x, y in pts]
    # head and neck: the long face down to the muzzle on the left, the neck to the lower right
    d.polygon(P([(0.12, 0.62), (0.14, 0.52), (0.30, 0.38), (0.46, 0.24), (0.58, 0.20), (0.68, 0.26),
                 (0.76, 0.40), (0.86, 0.62), (0.90, 0.92), (0.62, 0.92), (0.56, 0.66), (0.42, 0.62),
                 (0.30, 0.70), (0.16, 0.72)]), fill=EMBLEM_WHITE)
    # ears, pricked
    d.polygon(P([(0.56, 0.22), (0.58, 0.04), (0.65, 0.22)]), fill=EMBLEM_WHITE)
    d.polygon(P([(0.63, 0.24), (0.70, 0.08), (0.71, 0.27)]), fill=EMBLEM_WHITE)
    # the spiked mane down the back of the neck
    for i in range(5):
        y = 0.30 + i * 0.12
        x = 0.72 + i * 0.035
        d.polygon(P([(x, y), (x + 0.13, y + 0.02), (x + 0.03, y + 0.09)]), fill=EMBLEM_WHITE)
    dark = (20, 26, 8)
    d.polygon(P([(0.46, 0.34), (0.55, 0.32), (0.51, 0.38)]), fill=dark)              # the eye
    d.ellipse(P([(0.15, 0.56), (0.19, 0.60)]), fill=dark)                            # the nostril
    d.line(P([(0.13, 0.66), (0.30, 0.64)]), fill=dark, width=int(S * 0.02))          # the mouth
    d.line(P([(0.56, 0.66), (0.66, 0.50)]), fill=dark, width=int(S * 0.018))         # the jaw line
    return img.resize((size, size), Image.LANCZOS)


def gargoyle_emblem(size: int) -> Image.Image:
    """A gargoyle's horned head from the front with bat wings spread behind it, as white shapes on the
    green ground (the game's emblem style: the Goblin's and the Bile Demon's are front-on heads)."""
    S = size * K
    img = _radial((S, S), GREEN_CENTRE, GREEN_EDGE, focus=(0.5, 0.55))
    d = ImageDraw.Draw(img)
    P = lambda pts: [(x * S, y * S) for x, y in pts]
    for sx in (1, -1):                       # the wings: three finger bones and a scalloped edge each side
        m = lambda pts: P([(0.5 + sx * (x - 0.5), y) for x, y in pts])
        d.polygon(m([(0.64, 0.46), (0.82, 0.26), (0.97, 0.30), (0.91, 0.46), (0.95, 0.60), (0.84, 0.58),
                     (0.82, 0.72), (0.72, 0.64), (0.64, 0.70)]), fill=EMBLEM_WHITE)
    dark = (20, 26, 8)
    for sx in (1, -1):                       # wing membranes' dark seams between the bones
        m = lambda pts: P([(0.5 + sx * (x - 0.5), y) for x, y in pts])
        d.line(m([(0.69, 0.48), (0.89, 0.38)]), fill=dark, width=int(S * 0.02))
        d.line(m([(0.69, 0.55), (0.86, 0.55)]), fill=dark, width=int(S * 0.02))
    # the head: a squat skull with a heavy brow, pointed ears and curled horns
    d.polygon(P([(0.34, 0.36), (0.40, 0.30), (0.60, 0.30), (0.66, 0.36), (0.68, 0.52), (0.62, 0.70),
                 (0.56, 0.78), (0.44, 0.78), (0.38, 0.70), (0.32, 0.52)]), fill=EMBLEM_WHITE)
    for sx in (1, -1):
        m = lambda pts: P([(0.5 + sx * (x - 0.5), y) for x, y in pts])
        d.polygon(m([(0.57, 0.31), (0.62, 0.12), (0.72, 0.04), (0.68, 0.16), (0.65, 0.33)]), fill=EMBLEM_WHITE)
        d.polygon(m([(0.52, 0.48), (0.64, 0.42), (0.61, 0.53)]), fill=dark)               # the eye, slanted
    d.polygon(P([(0.48, 0.52), (0.52, 0.52), (0.50, 0.57)]), fill=dark)                  # the snout
    d.polygon(P([(0.40, 0.61), (0.60, 0.61), (0.56, 0.71), (0.44, 0.71)]), fill=dark)    # the mouth
    for x in (0.42, 0.54):                                                                  # fangs
        d.polygon(P([(x, 0.61), (x + 0.02, 0.68), (x + 0.04, 0.61)]), fill=EMBLEM_WHITE)
    return img.resize((size, size), Image.LANCZOS)


def krefft_emblem(size: int) -> Image.Image:
    """A river turtle seen from above, as white shapes on the green ground (the game's emblem style: one bold
    silhouette with a few heavy dark cuts that hold at 64 px): the domed shell with its central plate and the
    seams to the rim cut in, the head pushed out at the top with two eye cuts, four paddle legs and the stub of
    a tail. (Round 2: the reviewers found the first one's 1-pixel plate lines and the coin ring on the head too
    fine to hold at real size; the coin is left to his portrait and clips.)"""
    S = size * K
    img = _radial((S, S), GREEN_CENTRE, GREEN_EDGE, focus=(0.5, 0.55))
    d = ImageDraw.Draw(img)
    P = lambda pts: [(x * S, y * S) for x, y in pts]
    dark = (20, 26, 8)
    for sx in (1, -1):                       # the paddles: front ones reaching forward, back ones splayed
        m = lambda pts: P([(0.5 + sx * (x - 0.5), y) for x, y in pts])
        d.polygon(m([(0.66, 0.38), (0.88, 0.22), (0.97, 0.28), (0.91, 0.38), (0.72, 0.50)]), fill=EMBLEM_WHITE)
        d.polygon(m([(0.68, 0.70), (0.87, 0.80), (0.85, 0.90), (0.74, 0.88), (0.62, 0.77)]), fill=EMBLEM_WHITE)
    d.polygon(P([(0.45, 0.84), (0.50, 0.96), (0.55, 0.84)]), fill=EMBLEM_WHITE)            # the tail
    d.ellipse(P([(0.37, 0.06), (0.63, 0.34)]), fill=EMBLEM_WHITE)                       # the head
    d.ellipse(P([(0.20, 0.26), (0.80, 0.90)]), fill=EMBLEM_WHITE)                       # the shell
    w = int(S * 0.05)                        # about 3 px at 64: the cuts hold at the panel's size
    hexa = P([(0.50, 0.44), (0.62, 0.51), (0.62, 0.66), (0.50, 0.73), (0.38, 0.66), (0.38, 0.51)])
    d.polygon(hexa, outline=dark, width=w)                                               # the central plate
    for a_, b_ in (((0.50, 0.44), (0.50, 0.33)), ((0.50, 0.73), (0.50, 0.83)), ((0.62, 0.51), (0.71, 0.46)),
                   ((0.38, 0.51), (0.29, 0.46)), ((0.62, 0.66), (0.71, 0.71)), ((0.38, 0.66), (0.29, 0.71))):
        d.line(P([a_, b_]), fill=dark, width=w)                                          # the plates' seams
    d.ellipse(P([(0.26, 0.32), (0.74, 0.84)]), outline=dark, width=int(w * .7))       # the rim, left whole outside
    for sx in (1, -1):                                                                   # the eyes
        m = lambda pts: P([(0.5 + sx * (x - 0.5), y) for x, y in pts])
        d.polygon(m([(0.53, 0.17), (0.60, 0.15), (0.57, 0.22)]), fill=dark)
    return img.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------------------ everything

def make(icon: str, files_dir: Path, game_cache, out_dir: Path) -> tuple[Image.Image, Image.Image]:
    """Write <icon>-bg.png and <icon>.png into out_dir from his files in files_dir."""
    from scripts.molynx_assets import _cache_image
    files_dir, out_dir = Path(files_dir), Path(out_dir)
    shot = SHOTS[icon]
    model = parse_kmf((files_dir / f"{shot.clip}.kmf").read_bytes())
    mine = {p.stem: Image.open(p).convert("RGB") for p in files_dir.glob("*.png")}
    cache = PanelCache(game_cache, mine)
    donor = DONOR_ICON[icon]
    bg_frame = _cache_image(game_cache, EVIL + donor + "-bg")
    icon_frame = _cache_image(game_cache, EVIL + donor)
    bg = _framed(bg_frame, portrait(model, cache, shot))
    inner = ICON_SIZE[0] - 2 * FRAME
    if icon == "orc":
        emblem = orc_emblem(icon_frame)
    elif icon == "hreaper":
        emblem = reaper_emblem(icon_frame)
    elif icon == "sejanus":
        emblem = sejanus_emblem(inner)
    elif icon == "gargoyle":
        emblem = gargoyle_emblem(inner)
    elif icon == "krefft":
        emblem = krefft_emblem(inner)
    else:
        emblem = dragon_emblem(inner)
    ic = _framed(icon_frame, emblem)
    out_dir.mkdir(parents=True, exist_ok=True)
    bg.save(out_dir / f"{icon}-bg.png")
    ic.save(out_dir / f"{icon}.png")
    return bg, ic
