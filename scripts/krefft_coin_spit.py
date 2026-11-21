"""Coin Spit: Krefft's ranged attack, made from two records the game does not use (the way Molynx's Ground
Slam was, molynx_ground_slam), so it needs no Flame change.

  CreatureSpells 35  "Dark Angel Firebomb " -> "Coin Spit". No creature of the game or of the mod has
                     spell 35 in any table (the editor's Creatures.kwd and all 123 map Globals of the
                     installed MoreCreatures; the Dark Angel casts 31, her Hail Storm; spell 29 "Fairy
                     Freeze" is the other unused one). It becomes a copy of Salamander Spit (39) with its
                     range (12288, three tiles), recharge (20480, five seconds), damage (150 in the editor
                     table; each map's Globals keeps its own, some have 100) and sound (event 277 in the
                     shot's SPELL_FIREBALL bank: the Salamander's wet spit), naming shot 48.
  Shots 48           "Elven  Archer Sure Shot" -> "Coin Spit". Nothing names shot 48: no creature spell,
                     keeper spell, trap or shot (death / hit shots), and the game's code creates no shot by
                     that number (molynx_ground_slam: only 53, 63 and 65). It becomes a copy of the
                     Salamander Spit shot (47) - its speed, damage type, collision and life - with:
                       mesh            the game's Coin (art type 5, the whole art resource of Coin Spill
                                       Element 0, effect element 122) in place of the FB0 fire sprite
                       generalEffectId 34 Gold Sparkle (the treasury's glint) as its trail, in place of 328
                       hitThingEffectId 35 Coin Spill (coins bouncing out of the hit), in place of 234
                       hitSolidEffectId 35 Coin Spill too: a miss scatters coins off the wall
                       attackCategory  13 physical (the Tremor's and the Boulder's), in place of 4 fire, so
                                       it does not set things burning and fire-proof creatures feel it
                     The effects 34 and 35 are the game's own, unchanged (nothing new in Effects or
                     EffectElements).

Krefft's record (krefft_record) casts spell 35 from level 2 with his Magic clip, released at the frame his
lips open (1.1 s).

`apply_coin_spit(data)` returns a Globals.kld or an editor .kwd with these two records changed (the blocks
of the four tables it has; every other byte the same) after checking that both slots are still the game's
unused records and that the donors are the game's.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

from pathlib import Path

from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2kwd_tables import Record, read_records
from scripts.molynx_ground_slam import CREATURE_SPELLS, ELEMENTS, SHOTS, _by_id, _check_names, references

SPELL_ID, SHOT_ID = 35, 48
NAME = "Coin Spit"
REUSED = {CREATURE_SPELLS: {SPELL_ID: "Dark Angel Firebomb "}, SHOTS: {SHOT_ID: "Elven  Archer Sure Shot"}}
SALAMANDER_SPIT_SPELL, SALAMANDER_SPIT_SHOT = 39, 47
DONORS = {CREATURE_SPELLS: {SALAMANDER_SPIT_SPELL: "Salamander Spit"}, SHOTS: {SALAMANDER_SPIT_SHOT: "Salamander Spit"}}
COIN_ELEMENT = 122                          # Coin Spill Element 0: its art resource is the game's Coin mesh
GOLD_SPARKLE, COIN_SPILL = 34, 35           # Effects
ATTACK_PHYSICAL = 13
ART = ("name", "flags", "data", "type", "extra", "sometimesOne")

SPELL_FIELDS = {"name": NAME, "shotTypeId": SHOT_ID, "creatureSpellId": SPELL_ID}
SHOT_FIELDS = {"name": NAME, "shotId": SHOT_ID, "generalEffectId": GOLD_SPARKLE, "hitThingEffectId": COIN_SPILL,
               "hitSolidEffectId": COIN_SPILL, "attackCategory": ATTACK_PHYSICAL}
# what the copy keeps of the Salamander Spit shot (checked, so a changed donor is noticed)
# (only the kind of record: a map's own Salamander Spit numbers - some maps' Globals have damage 100, not
# the editor's 150 - are copied as that map has them, so Coin Spit is as strong as her spit on every map)
SHOT_KEPT = {"damageType": 7, "collideType": 2, "processType": 0}
SPELL_KEPT = {"shotTypeId": SALAMANDER_SPIT_SHOT}


def _text(v):
    return v.rstrip(b"\0").decode("latin-1") if isinstance(v, bytes) else v


def coin_spit_records(blocks, coin_art: dict | None = None) -> dict[int, dict[int, Record]]:
    """{signature: {id: new record}} for the CreatureSpells and Shots blocks `blocks` has. The Coin art
    resource comes from the EffectElements block when present, else `coin_art` (the editor's, for a
    Globals.kld without that block)."""
    by_sig = {b.signature: b for b in blocks}
    present = [s for s in (CREATURE_SPELLS, SHOTS) if s in by_sig]
    if not present:
        raise ValueError("the file has neither a CreatureSpells nor a Shots block")
    names = {CREATURE_SPELLS: "CreatureSpells", SHOTS: "Shots"}
    tables = {s: _by_id(read_records(by_sig[s]), names[s]) for s in present}
    for s in present:
        _check_names(tables[s], REUSED[s], names[s], "to be reused")
        _check_names(tables[s], DONORS[s], names[s], "copied")
    refs = references(blocks)
    if SHOTS in tables and refs["shot"].get(SHOT_ID):
        raise ValueError(f"shot {SHOT_ID} is not free: named by " + ", ".join(refs["shot"][SHOT_ID]))
    out: dict[int, dict[int, Record]] = {}
    if CREATURE_SPELLS in tables:
        t = tables[CREATURE_SPELLS]
        d = t[SALAMANDER_SPIT_SPELL]
        wrong = {k: d.get(k) for k, v in SPELL_KEPT.items() if d.get(k) != v}
        if wrong:
            raise ValueError(f"Salamander Spit (spell {SALAMANDER_SPIT_SPELL}) is not the game's: {wrong}")
        r = Record(CREATURE_SPELLS, bytearray(d.raw))
        for k, v in SPELL_FIELDS.items():
            r.set(k, v)
        out[CREATURE_SPELLS] = {SPELL_ID: r}
    if SHOTS in tables:
        t = tables[SHOTS]
        d = t[SALAMANDER_SPIT_SHOT]
        wrong = {k: _text(d.get(k)) for k, v in SHOT_KEPT.items() if _text(d.get(k)) != v}
        if wrong:
            raise ValueError(f"the Salamander Spit shot ({SALAMANDER_SPIT_SHOT}) is not the game's: {wrong}")
        if ELEMENTS in by_sig:
            el = _by_id(read_records(by_sig[ELEMENTS]), "EffectElements")[COIN_ELEMENT]
            coin_art = {k: el.get(f"art.{k}") for k in ART}
        if coin_art is None or _text(coin_art["name"]) != "Coin" or coin_art["type"] != 5:
            raise ValueError(f"no Coin mesh art resource (effect element {COIN_ELEMENT})")
        r = Record(SHOTS, bytearray(d.raw))
        for k, v in SHOT_FIELDS.items():
            r.set(k, v)
        for k in ART:
            r.set(f"mesh.{k}", coin_art[k])
        out[SHOTS] = {SHOT_ID: r}
    return out


def coin_art(editor_dir: Path) -> dict:
    el = _by_id(read_records(read_blocks((Path(editor_dir) / "EffectElements.kwd").read_bytes())[0]),
                "EffectElements")[COIN_ELEMENT]
    return {k: el.get(f"art.{k}") for k in ART}


def apply_coin_spit(data: bytes, art: dict | None = None) -> bytes:
    """`data` (a Globals.kld or an editor .kwd) with Coin Spit's spell and shot in place of the reused
    records; every other record and block byte for byte the same."""
    blocks = read_blocks(data)
    new = coin_spit_records(blocks, art)
    for k, b in enumerate(blocks):
        if b.signature not in new:
            continue
        records = read_records(b)
        blocks[k] = rebuilt_block(b, [bytes(new[b.signature].get(r.id, r).raw) for r in records])
    return write_blocks(blocks)


EDITOR_FILES = ("CreatureSpells.kwd", "Shots.kwd")
