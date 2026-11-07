"""Ground Slam: Molynx's own close-range special attack, made from records the game does not use.

He brings his mace down on the floor in front of him; the ground bursts (rocks, black and red-hot
chips, dust and a red glow) and every enemy within two tiles is hurt, most at the point of impact.
Nothing ranged: he casts it only on an enemy within one tile, like the Bile Demon's Gas Cloud.

Which records (all in the four tables of dk2kwd_tables; every map's Globals.kld carries all four,
and the game also loads the editor's .kwd files as its base tables, so both get the same change):

  CreatureSpells 23  "Chicken Arrow" -> "Ground Slam". No creature has spell 23 (the Dark Elf and
                     the Elven Archer fire the Chicken Arrow SHOT, 36, as their alternative shot in
                     room 4, so shot 36 is left alone).
  Shots 51           "Dark Angel Hailstorm" -> "Ground Slam". Nothing names shot 51: no creature
                     spell, keeper spell, trap or shot (death / hit shots), and the game's code
                     creates no shot 51 by number (the createCShot calls with a constant id ask for
                     53, 63 and 65). The Dark Angel's Hail Storm uses shots 37 -> 17.
  Effects 36, 40, 56, 127 ("Alex Spare 0", "Michaels Crappy Place Holder", "Test", "Spare") and
  EffectElements 77, 490 ("do not use ", "test"): placeholders nothing names; they become the
  burst's effects and its two recoloured particles.

How the game runs it (DKII.EXE 1.70, function names from the Flame mapping DKII_EXE_v170.sgmap):

- A creature spell's shot is created at the creature plus the spell's shotOffset turned to his
  facing (sub_4982B0), and gets the SPELL's shotData1 / shotData2 as its data (CShot +0x13D,
  +0x141), passed on to any shot it creates (CShot::tick). It flies at the shot's speed: 0 here, so
  it stays where the mace lands (no division by the speed, sub_4AB8A0; the Mistress's Lightning
  Shot is cast with speed 0 too).
- processType 0 runs no special process; damageType 2 (RANGE_FROM_ORIGIN, fun_4AC6D0) hurts, every
  game tick the shot lives, every thing of another owner within data2 tiles by
  data1 * (1 - distance / (data2 tiles)): never his own side, never himself (only the owner is
  tested: another player's creatures, doors and traps are hit too). As in every such shot of the
  game (the Grenade Explosion's too), the damage dealt to one thing becomes the damage for the next
  thing it finds in that tick (fun_4AC6D0 keeps the result in the same register), so with several
  enemies around, the later ones take less. The shot lives `health` ticks (DIE_OVER_TIME). The shot record is the Grenade Explosion's (19, the explosion a
  creature's Grenade spell makes), so only the fields set below differ from a record the game
  already runs from a creature's spell.
- attackCategory 13 (physical, as the keeper's Tremor and the Boulder trap): the game tests the
  category only for fire (4: burning) and gas (7: not on objects); no creature resists 13.
- Its creation effect is spawned where it is created. Effect generation type 18
  (EffectGeneration_cpp_439100) spawns each of the first four generateIds once per 4
  elementsPerTurn, as the keeper's Tremor Spell Effect (103) does with Tremor Rocks and Tremor Dust.
- The sound at the cast is the spell's soundEvent in the shot's sound category (sub_4982B0). The
  keeper's Tremor plays event 0x2EC (748) of KEEPER_SPELL_TREMOR for each falling rock
  (World_cpp_510440); Ground Slam plays that one (the rumble, 251, is a long sound the world
  tremor keeps alive and ends itself, so it is not used).

The burst (effect ids are the reused slots):
  36 Ground Slam Effect   type 18, spawns 40, 127 and 56 once (a copy of Tremor Spell Effect 103)
  40 Ground Slam Rocks    Tremor Rocks (43) cut from 108 ticks to RESOURCE_TICKS: a few of the
                          tremor's rocks thrown up, splintering when they land
  127 Ground Slam Dust    Tremor Dust (44) in a 1.5-tile area: brown dust, red dust (490), grey
                          mist (373)
  56 Ground Slam Chips    Earth blast (325), short: black chips (536) and red-hot chips (77),
                          with a red glow
  77 Ground Slam Ember    Earth blast Element (536), red-hot instead of black
  490 Ground Slam Red Dust Dust (6), dark red instead of brown

`apply_ground_slam(data)` returns a Globals.kld / editor .kwd file with these changes (the blocks
it has of the four tables; every other byte unchanged), after checking that the reused records
are still the game's unused ones and that nothing in the file names them.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import struct
from pathlib import Path

from scripts.dk2kwd_blocks import read_blocks, rebuilt_block, write_blocks
from scripts.dk2kwd_tables import Record, read_records

CREATURE_SPELLS, SHOTS, EFFECTS, ELEMENTS = 0xA0, 0x104, 0x10E, 0xFA
KEEPER_SPELLS, TRAPS = 0x96, 0x82
KEEPER_SPELL_SIZE, TRAP_SIZE = 406, 579
KEEPER_SPELL_SHOT_AT, KEEPER_SPELL_BONUS_SHOT_AT, TRAP_SHOT_AT = 240, 275, 525  # OpenKeeper readers

SPELL_ID = 23
SHOT_ID = 51
ROOT, ROCKS, DUST, CHIPS = 36, 40, 127, 56          # effects
EMBER, RED_DUST = 77, 490                           # effect elements
NAME = "Ground Slam"

# id -> the name the record must have before it is reused (the game's own placeholders)
REUSED = {
    CREATURE_SPELLS: {SPELL_ID: "Chicken Arrow"},
    SHOTS: {SHOT_ID: "Dark Angel Hailstorm"},
    EFFECTS: {ROOT: "Alex Spare 0", ROCKS: "Michaels Crappy Place Holder", DUST: "Spare",
              CHIPS: "Test"},
    ELEMENTS: {EMBER: "do not use ", RED_DUST: "test"},
}
# the game records the new ones are copied from (id -> name, checked)
GAS_CLOUD, GRENADE_EXPLOSION = 5, 19
TREMOR_SPELL_EFFECT, TREMOR_ROCKS, TREMOR_DUST, EARTH_BLAST = 103, 43, 44, 325
EARTH_BLAST_ELEMENT, DUST_ELEMENT, GREY_MIST = 536, 6, 373
TREMOR_ROCK_1, TREMOR_ROCK_2 = 305, 306
DONORS = {
    CREATURE_SPELLS: {GAS_CLOUD: "Gas Cloud"},
    SHOTS: {GRENADE_EXPLOSION: "Grenade Explosion"},
    EFFECTS: {TREMOR_SPELL_EFFECT: "Tremor Spell Effect", TREMOR_ROCKS: "Tremor Rocks",
              TREMOR_DUST: "Tremor Dust", EARTH_BLAST: "Earth blast",
              TREMOR_ROCK_1: "Tremor Rock 1", TREMOR_ROCK_2: "Tremor Rock 2"},
    ELEMENTS: {EARTH_BLAST_ELEMENT: "Earth blast Element", DUST_ELEMENT: "Dust",
               GREY_MIST: "Skeliton to Bones Element 1"},
}

FX = 4096
# The spell. shotData1 is the damage per tick at the point of impact, shotData2 the radius in
# tiles (both read by RANGE_FROM_ORIGIN; the radius must be at least 1, it is divided by).
DAMAGE_PER_TICK = 240
RADIUS_TILES = 2
SHOT_TICKS = 4               # the shot's health: 4 ticks of damage (Grenade Explosion: 8 of 60)
RANGE = 1 * FX               # cast only on an enemy within one tile (Gas Cloud: the same)
RECHARGE = 8 * FX            # seconds between slams (Gas Cloud 5, Wind 10)
SPELL_FLAGS = 0x109          # IS_ATTACKING | CAST_ONLY_ON_CREATURES | CAST_ON_OTHERS (Gas Cloud's)
SOUND_CATEGORY = "KEEPER_SPELL_TREMOR"
SOUND_EVENT = 0x2EC          # a falling rock of the keeper's Tremor (World_cpp_510440)
ICON_RESOURCE = "GUI\\C-Spells\\c-ground_slam"
ICON_FILE = "c-ground_slam.png"   # export file name; shipped as Data/EngineTextures/GUI/C-Spells/
ICON_SIZE = (64, 64)
ATTACK_CATEGORY = 13         # physical (Tremor, Boulder)

SPELL_FIELDS = {
    "name": NAME, "shotData1": DAMAGE_PER_TICK, "shotData2": RADIUS_TILES, "range": RANGE,
    "flags": SPELL_FLAGS, "soundEvent": SOUND_EVENT, "shotTypeId": SHOT_ID,
    "alternativeShotId": 0, "alternativeRoomId": 0, "alternativeShot": 0, "rechargeTime": RECHARGE,
}
SHOT_FIELDS = {   # set on a copy of the Grenade Explosion
    "name": NAME, "shotId": SHOT_ID, "speed": 0, "data1": DAMAGE_PER_TICK, "data2": RADIUS_TILES,
    "health": SHOT_TICKS, "creationEffectId": ROOT, "attackCategory": ATTACK_CATEGORY,
    "soundCategory": SOUND_CATEGORY,
}
# What the shot keeps of the Grenade Explosion (checked, so a changed donor is noticed).
SHOT_KEPT = {"processType": 0, "damageType": 2, "collideType": 2, "flags": 0x104,
             "processFlags": 1, "deathShotId": 0, "mesh.type": 0}

RED_GLOW = bytes((0xFF, 0x46, 0x14, 0x00))     # light colour (r, g, b, 0), as the Fireburst's ff4040
EMBER_COLOUR = bytes((0xB4, 0x32, 0x0A))        # red-hot chip (the Earth blast's chips are 0a0a0a)
RED_DUST_COLOUR = bytes((0x8C, 0x28, 0x14))     # dark red dust (Dust is 826e64)


def _gen(ids):
    ids = list(ids) + [0] * (8 - len(ids))
    return {f"generateIds[{k}]": v for k, v in enumerate(ids)}


# effect id -> (donor id, fields set on the copy)
EFFECT_COPIES = {
    ROOT: (TREMOR_SPELL_EFFECT, {"name": "Ground Slam Effect", **_gen([ROCKS, DUST, CHIPS]),
                                 "elementsPerTurn": 4, "minHp": 1, "maxHp": 1}),
    ROCKS: (TREMOR_ROCKS, {"name": "Ground Slam Rocks", "minHp": 4, "maxHp": 4,
                           **_gen([TREMOR_ROCK_1, TREMOR_ROCK_2, TREMOR_ROCK_1, TREMOR_ROCK_2])}),
    DUST: (TREMOR_DUST, {"name": "Ground Slam Dust", "radius": 6144, "minHp": 2, "maxHp": 2,
                         "elementsPerTurn": 6, **_gen([DUST_ELEMENT, RED_DUST, GREY_MIST])}),
    CHIPS: (EARTH_BLAST, {"name": "Ground Slam Chips", "minHp": 3, "maxHp": 3,
                          "elementsPerTurn": 16, **_gen([EARTH_BLAST_ELEMENT, EMBER]),
                          "light.radius": 6144, "light.colour": RED_GLOW, "light.flags": 0}),
}
ELEMENT_COPIES = {
    EMBER: (EARTH_BLAST_ELEMENT, {"name": "Ground Slam Ember", "colour": EMBER_COLOUR}),
    RED_DUST: (DUST_ELEMENT, {"name": "Ground Slam Red Dust", "colour": RED_DUST_COLOUR}),
}
# Which flags make an effect's generateIds name effects (0x40) or elements (0x80).
GENERATE_EFFECTS, GENERATE_ELEMENTS = 0x40, 0x80


def _by_id(records: list[Record], what: str) -> dict[int, Record]:
    out = {}
    for r in records:
        if r.id in out:
            raise ValueError(f"{what}: id {r.id} appears twice")
        out[r.id] = r
    return out


def _check_names(table: dict[int, Record], want: dict[int, str], what: str, role: str) -> None:
    for i, name in want.items():
        if i not in table:
            raise ValueError(f"{what} has no record {i} ({name!r}, {role})")
        if table[i].name != name:
            raise ValueError(f"{what} {i} is {table[i].name!r}, not the game's {name!r} ({role}): "
                             "this file is not the one Ground Slam was made for")


def references(blocks) -> dict[str, dict[int, list[str]]]:
    """{"shot" | "effect" | "element": {id: [who names it]}} in one file's blocks: the four
    tables' own links, every creature spell's shots, and the keeper spells' and traps' shots."""
    by_sig = {b.signature: b for b in blocks}
    refs = {"shot": {}, "effect": {}, "element": {}}

    def add(kind, i, who):
        if i:
            refs[kind].setdefault(i, []).append(who)

    if SHOTS in by_sig:
        for r in read_records(by_sig[SHOTS]):
            f = r.fields()
            for k in ("generalEffectId", "creationEffectId", "deathEffectId", "timedEffectId",
                      "hitSolidEffectId", "hitLavaEffectId", "hitWaterEffectId", "hitThingEffectId"):
                add("effect", f[k], f"shot {r.id} {k}")
            for k in ("deathShotId", "hitSolidShotId", "hitLavaShotId", "hitWaterShotId",
                      "hitThingShotId"):
                add("shot", f[k], f"shot {r.id} {k}")
    if EFFECTS in by_sig:
        for r in read_records(by_sig[EFFECTS]):
            f = r.fields()
            for k in ("nextEffectId", "deathEffectId", "hitSolidEffectId", "hitLavaEffectId",
                      "hitWaterEffectId"):
                add("effect", f[k], f"effect {r.id} {k}")
            for k in range(8):
                if f["flags"] & GENERATE_EFFECTS:
                    add("effect", f[f"generateIds[{k}]"], f"effect {r.id} generates")
                if f["flags"] & GENERATE_ELEMENTS:
                    add("element", f[f"generateIds[{k}]"], f"effect {r.id} generates")
    if ELEMENTS in by_sig:
        for r in read_records(by_sig[ELEMENTS]):
            f = r.fields()
            for k in ("deathElementId", "hitSolidElementId", "hitLavaElementId", "hitWaterElementId"):
                add("element", f[k], f"element {r.id} {k}")
            add("effect", f["nextEffectId"], f"element {r.id} nextEffectId")
    if CREATURE_SPELLS in by_sig:
        for r in read_records(by_sig[CREATURE_SPELLS]):
            add("shot", r.get("shotTypeId"), f"creature spell {r.id}")
            add("shot", r.get("alternativeShotId"), f"creature spell {r.id} alternative")
    for sig, size, offsets, what in ((KEEPER_SPELLS, KEEPER_SPELL_SIZE,
                                      (KEEPER_SPELL_SHOT_AT, KEEPER_SPELL_BONUS_SHOT_AT), "keeper spell"),
                                     (TRAPS, TRAP_SIZE, (TRAP_SHOT_AT,), "trap")):
        if sig in by_sig:
            b = by_sig[sig]
            if len(b.content) != size * b.item_count:
                raise ValueError(f"{what}s: {len(b.content)} bytes for {b.item_count} records of {size}")
            for i in range(b.item_count):
                for at in offsets:
                    add("shot", b.content[i * size + at], f"{what} {i}")
    return refs


_KIND = {SHOTS: "shot", EFFECTS: "effect", ELEMENTS: "element"}


def _copy(donor: Record, slot: Record, fields: dict, id_field: str) -> Record:
    out = Record(donor.signature, bytearray(donor.raw))
    for name, value in fields.items():
        out.set(name, value)
    out.set(id_field, slot.id)
    return out


def ground_slam_records(blocks) -> dict[int, dict[int, Record]]:
    """{signature: {id: new record}} for the tables `blocks` has, checked (see the module notes)."""
    by_sig = {b.signature: b for b in blocks}
    present = [s for s in (CREATURE_SPELLS, SHOTS, EFFECTS, ELEMENTS) if s in by_sig]
    if not present:
        raise ValueError("the file has none of the CreatureSpells, Shots, Effects and EffectElements "
                         "blocks")
    names = {CREATURE_SPELLS: "CreatureSpells", SHOTS: "Shots", EFFECTS: "Effects",
             ELEMENTS: "EffectElements"}
    tables = {s: _by_id(read_records(by_sig[s]), names[s]) for s in present}
    for s in present:
        _check_names(tables[s], REUSED[s], names[s], "to be reused")
        _check_names(tables[s], DONORS[s], names[s], "copied")
    refs = references(blocks)
    for s in present:
        if s == CREATURE_SPELLS:
            continue
        for i in REUSED[s]:
            if refs[_KIND[s]].get(i):
                raise ValueError(f"{names[s]} {i} ({REUSED[s][i]!r}) is not free: named by "
                                 + ", ".join(refs[_KIND[s]][i]))
    out: dict[int, dict[int, Record]] = {}
    if CREATURE_SPELLS in tables:
        t = tables[CREATURE_SPELLS]
        spell = Record(CREATURE_SPELLS, bytearray(t[SPELL_ID].raw))
        for name, value in SPELL_FIELDS.items():
            spell.set(name, value)
        gas = t[GAS_CLOUD]   # its whole 84-byte GUI icon resource, renamed
        for name in ("flags", "data", "type", "extra", "sometimesOne"):
            spell.set(f"guiIcon.{name}", gas.get(f"guiIcon.{name}"))
        spell.set("guiIcon.name", ICON_RESOURCE)
        out[CREATURE_SPELLS] = {SPELL_ID: spell}
    if SHOTS in tables:
        t = tables[SHOTS]
        donor = t[GRENADE_EXPLOSION]
        wrong = {k: donor.get(k) for k, v in SHOT_KEPT.items() if donor.get(k) != v}
        if wrong:
            raise ValueError(f"the Grenade Explosion (shot {GRENADE_EXPLOSION}) is not the game's: {wrong}")
        out[SHOTS] = {SHOT_ID: _copy(donor, t[SHOT_ID], SHOT_FIELDS, "shotId")}
    if EFFECTS in tables:
        t = tables[EFFECTS]
        out[EFFECTS] = {i: _copy(t[d], t[i], f, "effectId") for i, (d, f) in EFFECT_COPIES.items()}
    if ELEMENTS in tables:
        t = tables[ELEMENTS]
        out[ELEMENTS] = {i: _copy(t[d], t[i], f, "effectElementId")
                         for i, (d, f) in ELEMENT_COPIES.items()}
    return out


def apply_ground_slam(data: bytes) -> bytes:
    """`data` (a Globals.kld or an editor .kwd) with Ground Slam's records in place of the reused
    ones; every other record and block is byte for byte the same. ValueError if the file is not
    the game's (see ground_slam_records)."""
    blocks = read_blocks(data)
    new = ground_slam_records(blocks)
    for k, b in enumerate(blocks):
        if b.signature not in new:
            continue
        records = read_records(b)
        blocks[k] = rebuilt_block(b, [bytes(new[b.signature].get(r.id, r).raw) for r in records])
    return write_blocks(blocks)


EDITOR_FILES = ("CreatureSpells.kwd", "Shots.kwd", "Effects.kwd", "EffectElements.kwd")


def editor_files(editor_dir: Path) -> dict[str, bytes]:
    """{file name: bytes} of the four editor tables with Ground Slam."""
    return {name: apply_ground_slam((Path(editor_dir) / name).read_bytes()) for name in EDITOR_FILES}


# ---------------------------------------------------------------- the spell icon

ICON_DONORS = ("c-gas_cloud", "c-grenade", "c-hailstorm", "c-wind", "c-skeleton_army",
               "c-raise_dead", "c-slow", "c-knives", "c-lightning", "c-freeze", "c-drain",
               "c-fireball", "c-arrow", "c-dark_arrow", "c-guided_bolt", "c-spit", "c-haste",
               "c-invisible", "c-invulnerable", "c-touch_of_death")


def _icon_frame(cache):
    """The empty creature-spell icon (riveted frame and dark green field): per pixel, the darker
    third of the game's own icons (their glyphs are light, so every glyph drops out)."""
    import numpy as np
    from PIL import Image
    lowered = {n.lower(): n for n in cache.names}
    stack = []
    for name in ICON_DONORS:
        key = f"gui\\c-spells\\{name}mm0"
        if key in lowered:
            stack.append(np.asarray(cache.get(lowered[key]).convert("RGB"), float))
    if len(stack) < 9:
        raise ValueError("the texture cache lacks the creature-spell icons")
    stack = np.stack(stack)
    order = np.argsort(stack.sum(axis=3), axis=0)          # darkest first, per pixel
    k = len(stack) // 3
    pick = np.take_along_axis(stack, order[k][None, ..., None].repeat(3, axis=3), axis=0)[0]
    return Image.fromarray(pick.round().astype("uint8"))


def _glyph_mask(size: int):
    """The Ground Slam glyph (white on black, `size` square): a spiked mace head on its haft
    struck into the ground, the ground cracked under it, chunks thrown up and two shock arcs."""
    import math
    from PIL import Image, ImageDraw
    s = size / 64.0
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)

    def P(pts):
        return [(x * s, y * s) for x, y in pts]

    # the ground: a slab across the lower part, broken where the head strikes
    d.polygon(P([(9, 46), (26, 46), (29, 50), (25, 55), (9, 55)]), fill=255)
    d.polygon(P([(55, 46), (38, 46), (35, 50), (39, 55), (55, 55)]), fill=255)
    # the haft, from upper right down to the head, and its pommel
    d.line(P([(49, 8), (36, 30)]), fill=255, width=int(round(5 * s)))
    d.ellipse(P([(46, 4), (53, 11)]), fill=255)
    # the head: a round spiked ball resting in the break
    cx, cy, r = 32, 38, 8.5
    d.ellipse(P([(cx - r, cy - r), (cx + r, cy + r)]), fill=255)
    for k in range(8):
        a = math.radians(k * 45 + 22.5)
        tip = (cx + math.cos(a) * (r + 5), cy + math.sin(a) * (r + 5))
        l = (cx + math.cos(a + 0.33) * (r - 1), cy + math.sin(a + 0.33) * (r - 1))
        rr = (cx + math.cos(a - 0.33) * (r - 1), cy + math.sin(a - 0.33) * (r - 1))
        if tip[1] < 46:                                       # no spike into the ground
            d.polygon(P([l, tip, rr]), fill=255)
    # chunks thrown up on both sides
    for chunk in ([(9, 33), (13, 30), (16, 34), (12, 37)], [(14, 21), (17, 19), (19, 22), (16, 25)],
                  [(51, 27), (55, 25), (57, 29), (53, 31)], [(52, 39), (55, 37), (57, 40), (54, 42)],
                  [(21, 14), (23, 13), (24, 15), (22, 16)]):
        d.polygon(P(chunk), fill=255)
    # two shock arcs low on each side
    w = max(1, int(round(2 * s)))
    d.arc(P([(3, 37), (21, 59)]), 195, 255, fill=255, width=w)
    d.arc(P([(43, 37), (61, 59)]), 285, 345, fill=255, width=w)
    return img


def make_icon(cache):
    """The Ground Slam icon (64 x 64 RGB) in the style of the game's creature-spell icons: their
    frame and field, a silver glyph lit from the upper left (white there, pale green to the lower
    right, darker at its edges) and a soft dark shadow to its lower right."""
    import numpy as np
    from PIL import Image, ImageFilter
    base = np.asarray(_icon_frame(cache), float)
    big = 512
    mask = _glyph_mask(big)
    shift = int(big / 64 * 2)
    shadow = Image.fromarray(np.roll(np.roll(np.asarray(mask), shift, 0), shift, 1)).filter(
        ImageFilter.GaussianBlur(big / 64 * 1.5))
    inner = mask.filter(ImageFilter.GaussianBlur(big / 64 * 1.2))
    a = np.asarray(mask.resize(ICON_SIZE, Image.LANCZOS), float) / 255
    sh = np.asarray(shadow.resize(ICON_SIZE, Image.LANCZOS), float) / 255 * 0.8
    edge = np.asarray(inner.resize(ICON_SIZE, Image.LANCZOS), float) / 255
    field = np.zeros(ICON_SIZE)
    field[8:56, 8:56] = 1                       # the glyph stays inside the icon's field
    a, sh = a * field, sh * field
    yy, xx = np.mgrid[0:ICON_SIZE[1], 0:ICON_SIZE[0]].astype(float)
    t = ((0.8 * yy + 0.2 * xx - 12) / 40).clip(0, 1)[..., None]   # 0 at the top, 1 at the bottom
    top, bottom = np.array([250, 252, 250]), np.array([84, 170, 142])
    colour = top * (1 - t) + bottom * t
    colour = colour * (0.6 + 0.4 * edge[..., None])
    out = base * (1 - sh[..., None])
    out = out * (1 - a[..., None]) + colour * a[..., None]
    return Image.fromarray(out.clip(0, 255).round().astype("uint8"), "RGB")
