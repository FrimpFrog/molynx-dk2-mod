"""Molynx's own identity pieces that still came from the Bile Demon: his first-person attack icon,
and the mentor messages his record names. Preview only: nothing here writes the game's files.

- `make_melee_icon(cache)`: GUI\\Icons\\1st-person\\m-molynx (64 x 64), the icon at the lower left of
  the screen while he is possessed. The game's own icons are a silver weapon on a dark green
  riveted field (m-bile_demon is a bare spiked ball); his is his own weapon: a spiked mace head on a
  banded haft, pointing up and to the right like the other weapons.
- MENTOR: the Speech.str messages his record points at. The game's creature records name three
  blocks of the mentor's spoken messages (Data\\Text\\Default\\Speech.str and the mentor's speech
  bank): nine "has become unhappy because ..." (angerStringIdLair), nine "has become angry
  because ..." (angerStringIdFood) and "has left your Dungeon" (angerStringIdPay). His still point
  at the Bile Demon's (594, 864, 1118: "A Bile Demon has ..."). The game has a generic set that the
  mentor also speaks ("One of your creatures has ...": 828, 1098, 1144, used by Prince Tristran),
  so pointing his record there keeps the voice and drops the Bile Demon. OWN_MESSAGES are his own
  lines, for the case where text without the mentor's voice is wanted (untested: the mentor's
  voice for a new message number does not exist).

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import math

ICON_SIZE = (64, 64)
MELEE_DONORS = ("m-bile_demon", "m-troll", "m-reaper", "m-black_knight", "m-giant", "m-dwarf",
                "m-skeleton", "m-goblin", "m-salamander", "m-dark_angel", "m-guard", "m-king",
                "m-knight", "m-thief", "m-monk", "m-wizard", "m-rogue", "m-vampire", "m-warlock")

# record field -> (the Bile Demon's message number, the generic voiced message number)
MENTOR = {"angerStringIdLair": (594, 828), "angerStringIdFood": (864, 1098), "angerStringIdPay": (1118, 1144)}

REASONS = ("has no Lair", "has no food", "hasn't been paid", "has no work to do", "has been slapped",
           "has remained in the Hand of Evil for too long", "has been left alone in the Combat Pit",
           "is in the company of creatures he hates", "has been tortured")
OWN_MESSAGES = {
    "unhappy": [f"Molynx has become unhappy because he {r}" for r in REASONS],
    "angry": [f"Molynx has become angry because he {r}" for r in REASONS],
    "left": ["Molynx has left your Dungeon"],
}


def _frame(cache):
    """The empty first-person icon (riveted frame, dark green field): per pixel, the second darkest
    of the game's own icons. Their glyphs are light and most lie on the same diagonal, so only the
    darkest values are free of them."""
    import numpy as np
    from PIL import Image
    lowered = {n.lower(): n for n in cache.names}
    stack = [np.asarray(cache.get(lowered[f"gui\\icons\\1st-person\\{n}mm0"]).convert("RGB"), float)
             for n in MELEE_DONORS if f"gui\\icons\\1st-person\\{n}mm0" in lowered]
    if len(stack) < 9:
        raise ValueError("the texture cache lacks the first-person icons")
    stack = np.stack(stack)
    order = np.argsort(stack.sum(axis=3), axis=0)
    pick = np.take_along_axis(stack, order[1][None, ..., None].repeat(3, axis=3), axis=0)[0]   # second darkest
    # that field is darker than the icons' own: lift it to their median's broad light (blurred, so
    # no glyph shape comes back), leaving the frame as it is
    from scipy.ndimage import gaussian_filter
    median = np.median(stack, axis=0)
    gain = (gaussian_filter(median, (6, 6, 0)) + 1) / (gaussian_filter(pick, (6, 6, 0)) + 1)
    inside = np.zeros(pick.shape[:2], bool)
    inside[7:57, 7:57] = True
    pick[inside] = (pick * gain.clip(1, 1.8))[inside]
    return Image.fromarray(pick.clip(0, 255).round().astype("uint8"))


def _mace_mask(size: int):
    """His mace, white on black: a spiked ball at the upper right on a banded haft to the lower left."""
    from PIL import Image, ImageDraw
    s = size / 64.0
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    P = lambda pts: [(x * s, y * s) for x, y in pts]
    # the haft: lower left to the head, thick, with a pommel and two raised bands
    a, b = (13, 51), (36, 28)
    d.line(P([a, b]), fill=255, width=int(round(7 * s)))
    d.ellipse(P([(10.5, 46.5), (17.5, 53.5)]), fill=255)
    ux, uy = (b[0] - a[0]), (b[1] - a[1])
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    px, py = -uy, ux
    for t in (0.25, 0.62):
        cx, cy = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        w, h = 4.6, 1.6
        d.polygon(P([(cx + px * w - ux * h, cy + py * w - uy * h), (cx + px * w + ux * h, cy + py * w + uy * h),
                     (cx - px * w + ux * h, cy - py * w + uy * h), (cx - px * w - ux * h, cy - py * w - uy * h)]),
                  fill=255)
    # the head: a ball with eight spikes
    cx, cy, r, spike = 39.5, 24.5, 10.5, 7.5
    d.ellipse(P([(cx - r, cy - r), (cx + r, cy + r)]), fill=255)
    for k in range(8):
        ang = math.radians(22.5 + 45 * k)
        tip = (cx + math.cos(ang) * (r + spike), cy + math.sin(ang) * (r + spike))
        side = math.radians(90)
        base_w = 3.4
        l = (cx + math.cos(ang) * (r - 1) + math.cos(ang + side) * base_w,
             cy + math.sin(ang) * (r - 1) + math.sin(ang + side) * base_w)
        rr = (cx + math.cos(ang) * (r - 1) - math.cos(ang + side) * base_w,
              cy + math.sin(ang) * (r - 1) - math.sin(ang + side) * base_w)
        d.polygon(P([l, tip, rr]), fill=255)
    return img


def make_melee_icon(cache):
    """m-molynx, 64 x 64 RGB, lit like the game's first-person icons: white at the upper left, pale
    green to the lower right, darker at the glyph's edges, with a soft shadow to the lower right."""
    import numpy as np
    from PIL import Image, ImageFilter
    base = np.asarray(_frame(cache), float)
    big = 512
    mask = _mace_mask(big)
    shift = int(big / 64 * 2)
    shadow = Image.fromarray(np.roll(np.roll(np.asarray(mask), shift, 0), shift, 1)).filter(
        ImageFilter.GaussianBlur(big / 64 * 1.5))
    inner = mask.filter(ImageFilter.GaussianBlur(big / 64 * 1.2))
    a = np.asarray(mask.resize(ICON_SIZE, Image.LANCZOS), float) / 255
    sh = np.asarray(shadow.resize(ICON_SIZE, Image.LANCZOS), float) / 255 * 0.8
    edge = np.asarray(inner.resize(ICON_SIZE, Image.LANCZOS), float) / 255
    field = np.zeros(ICON_SIZE)
    field[7:57, 7:57] = 1
    a, sh = a * field, sh * field
    yy, xx = np.mgrid[0:ICON_SIZE[1], 0:ICON_SIZE[0]].astype(float)
    t = ((0.6 * yy + 0.4 * xx - 14) / 42).clip(0, 1)[..., None]
    top, bottom = np.array([252, 252, 248]), np.array([150, 170, 120])
    colour = (top * (1 - t) + bottom * t) * (0.62 + 0.38 * edge[..., None])
    out = base * (1 - sh[..., None])
    out = out * (1 - a[..., None]) + colour * a[..., None]
    return Image.fromarray(out.clip(0, 255).round().astype("uint8"), "RGB")
