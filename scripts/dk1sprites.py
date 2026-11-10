"""LICENCE: GPL-3.0. Read Dungeon Keeper 1 creature sprites (DATA/CREATURE.TAB + CREATURE.JTY), per KeeperFX's reader.
CREATURE.TAB: 16-byte records {u32 DataOffset; u8 SWidth, SHeight, FrameWidth, FrameHeight, Rotable,
FramesCount, FrameOffsW, FrameOffsH; i16 offset_x, offset_y}. A creature animation number n maps to a
record index through KeeperFX's creature_list; a Rotable animation holds 5 views (front, front
diagonal, side, back diagonal, back) of FramesCount frames each. Pixels: rows of signed run bytes,
<0 skip, 0 end of row, n>0 n palette indices follow."""
import os, re, struct, sys
from pathlib import Path
import numpy as np
from PIL import Image

KFX = Path(os.environ.get('KEEPERFX_SRC', 'keeperfx/src')) / 'creature_graphics.c'   # dkfans/keeperfx


def creature_list():
    src = KFX.read_text()
    body = src[src.index('creature_list[CREATURE_FRAMELIST_LENGTH] = {'):]
    body = body[body.index('{') + 1: body.index('};')]
    return [int(x) for x in re.findall(r'\d+', body)]


class Sprites:
    def __init__(self, data_dir: Path, palette='MAIN.PAL'):
        d = Path(data_dir)
        tab = (d / 'CREATURE.TAB').read_bytes()
        self.recs = [struct.unpack_from('<I8B2h', tab, i) for i in range(0, len(tab) - 15, 16)]
        self.jty = (d / 'CREATURE.JTY').read_bytes()
        pal = np.frombuffer((d / palette).read_bytes()[:768], np.uint8).reshape(256, 3).astype(int)
        self.pal = np.clip(pal * 4 + pal // 16, 0, 255).astype(np.uint8)
        self.list = creature_list()

    def anim(self, n):
        i = self.list[n]
        r = self.recs[i]
        return i, r[5], r[6]        # first record, rotable, frames

    def frame(self, rec_index) -> Image.Image:
        off, sw, sh, fw, fh, rot, fc, ow, oh, ox, oy = self.recs[rec_index]
        end = self.recs[rec_index + 1][0]
        data = self.jty[off:end]
        img = np.zeros((max(sh, 1), max(sw, 1)), np.int16) - 1
        p = 0
        for y in range(sh):
            x = 0
            while p < len(data):
                c = struct.unpack_from('b', data, p)[0]; p += 1
                if c == 0:
                    break
                if c < 0:
                    x += -c
                else:
                    row = np.frombuffer(data[p:p + c], np.uint8)
                    img[y, x:x + c] = row[: max(0, min(c, sw - x))] if x < sw else img[y, x:x]
                    x += c; p += c
        rgba = np.zeros((img.shape[0], img.shape[1], 4), np.uint8)
        m = img >= 0
        rgba[m, :3] = self.pal[img[m]]
        rgba[m, 3] = 255
        return Image.fromarray(rgba, 'RGBA')

    def views(self, n, frame=0):
        i, rot, fc = self.anim(n)
        if not rot:
            return [self.frame(i + min(frame, fc - 1))]
        return [self.frame(i + v * fc + min(frame, fc - 1)) for v in range(5)]


def sheet(sp, anims, scale=3, frame=0):
    rows = []
    for n in anims:
        ims = sp.views(n, frame)
        rows.append(ims)
    W = max(sum(im.width for im in r) + 8 * len(r) for r in rows) * scale
    H = sum(max(im.height for im in r) + 8 for r in rows) * scale
    out = Image.new('RGBA', (W, H), (40, 40, 48, 255))
    y = 0
    for r in rows:
        x = 0
        for im in r:
            out.alpha_composite(im.resize((im.width * scale, im.height * scale), Image.NEAREST), (x, y))
            x += (im.width + 8) * scale
        y += (max(im.height for im in r) + 8) * scale
    return out
