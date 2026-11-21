"""Small pieces of Dungeon Keeper 2's own textures that the painters work from, cut from the player's
own game at build time, so the repository holds no game files.

  troll_metal, troll_wood   the Troll's mace (trollfrnt): the iron band and the wooden haft, for
                            Molynx's mace (scripts.character_mace)
  chain_arm                 one arm of the door chain (Padlock_Chains), turned level, the motif the
                            first-approved Molynx's painted chest chains were tiled from
                            (scripts.character_paint; the approved redesign's skins replace that body,
                            so this near copy of the first cut changes nothing in the game)

LICENCE: GPL-3.0. The pixels are Bullfrog's / EA's and come from the player's copy of the game.
"""
from __future__ import annotations

from PIL import Image

# name -> (texture, box in its MM0 level)
CROPS = {
    "troll_metal": ("trollfrnt", (6, 90, 46, 107)),
    "troll_wood": ("trollfrnt", (52, 117, 92, 123)),
}
CHAIN = ("Padlock_Chains", 18.8, (4, 6, 68, 52))   # texture, turn (degrees), box after the turn


def _texture(cache, name: str) -> Image.Image:
    lowered = {n.lower(): n for n in cache.names}
    return cache.get(lowered[f"{name}MM0".lower()])


def _cache(cache):
    if cache is None:
        from scripts.molynx_assets import texture_cache
        cache = texture_cache()
    return cache


def crop(name: str, cache=None) -> Image.Image:
    """One of CROPS (RGB) or "chain_arm" (RGBA), cut from the game's texture cache."""
    cache = _cache(cache)
    if name == "chain_arm":
        tex, turn, box = CHAIN
        return _texture(cache, tex).convert("RGBA").rotate(turn, resample=Image.BICUBIC).crop(box)
    tex, box = CROPS[name]
    return _texture(cache, tex).convert("RGB").crop(box)
