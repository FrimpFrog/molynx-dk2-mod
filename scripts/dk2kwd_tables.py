"""Record layouts of four DK2 .kwd / Globals blocks: CreatureSpells (0xA0), Shots (0x104),
Effects (0x10E) and EffectElements (0xFA), read and written field by field.

The layouts follow OpenKeeper's KwdFile reader (readCreatureSpells, readShots, readEffects,
readEffectElements; GPL-3.0), and every record size is checked against the game's own files:
CreatureSpells 266 bytes, Shots 239, Effects 246, EffectElements 182. Fixed-point fields
("fx", int32 / 4096) are kept as the raw integers; strings are NUL-padded bytes.

A Record keeps its raw bytes, so a field nobody sets stays exactly as the game had it.

LICENCE: GPL-3.0.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

# ArtResource: name (64), flags (u32), 12 bytes of type data, type (u8), 2 bytes, sometimesOne (u8)
ART = [("name", "64s"), ("flags", "<I"), ("data", "12s"), ("type", "<B"), ("extra", "2s"),
       ("sometimesOne", "<B")]
LIGHT = [("posX", "<i"), ("posY", "<i"), ("posZ", "<i"), ("radius", "<i"), ("flags", "<I"),
         ("colour", "4s")]


def _art(prefix: str):
    return [(f"{prefix}.{n}", f) for n, f in ART]


CREATURE_SPELL = [
    ("name", "32s"), *_art("editorIcon"), *_art("guiIcon"),
    ("shotData1", "<I"), ("shotData2", "<I"), ("range", "<i"), ("flags", "<I"),
    ("combatPoints", "<H"), ("soundEvent", "<H"), ("nameStringId", "<H"), ("tooltipStringId", "<H"),
    ("generalDescriptionStringId", "<H"), ("strengthStringId", "<H"), ("weaknessStringId", "<H"),
    ("creatureSpellId", "<B"), ("shotTypeId", "<B"), ("alternativeShotId", "<B"),
    ("alternativeRoomId", "<B"), ("rechargeTime", "<i"), ("alternativeShot", "<B"), ("unused", "27s"),
]
SHOT = [
    ("name", "32s"), *_art("mesh"), *[(f"light.{n}", f) for n, f in LIGHT],
    ("airFriction", "<i"), ("mass", "<i"), ("speed", "<i"), ("data1", "<I"), ("data2", "<I"),
    ("processFlags", "<I"), ("radius", "<i"), ("flags", "<I"),
    ("generalEffectId", "<H"), ("creationEffectId", "<H"), ("deathEffectId", "<H"),
    ("timedEffectId", "<H"), ("hitSolidEffectId", "<H"), ("hitLavaEffectId", "<H"),
    ("hitWaterEffectId", "<H"), ("hitThingEffectId", "<H"), ("health", "<H"),
    ("shotId", "<B"), ("deathShotId", "<B"), ("timedDelay", "<B"), ("hitSolidShotId", "<B"),
    ("hitLavaShotId", "<B"), ("hitWaterShotId", "<B"), ("hitThingShotId", "<B"),
    ("damageType", "<B"), ("collideType", "<B"), ("processType", "<B"), ("attackCategory", "<B"),
    ("soundCategory", "32s"), ("threat", "<H"), ("burnDuration", "<i"),
]
EFFECT = [
    ("name", "32s"), *_art("art"), *[(f"light.{n}", f) for n, f in LIGHT],
    ("mass", "<i"), ("airFriction", "<i"), ("elasticity", "<i"), ("radius", "<i"),
    ("minSpeedXy", "<i"), ("maxSpeedXy", "<i"), ("minSpeedYz", "<i"), ("maxSpeedYz", "<i"),
    ("minScale", "<i"), ("maxScale", "<i"), ("flags", "<I"),
    ("effectId", "<H"), ("minHp", "<H"), ("maxHp", "<H"), ("fadeDuration", "<H"),
    ("nextEffectId", "<H"), ("deathEffectId", "<H"), ("hitSolidEffectId", "<H"),
    ("hitLavaEffectId", "<H"), ("hitWaterEffectId", "<H"),
    *[(f"generateIds[{k}]", "<H") for k in range(8)],
    ("outerOriginRange", "<H"), ("lowerHeightLimit", "<H"), ("upperHeightLimit", "<H"),
    ("orientationRange", "<H"), ("spriteSpinRateRange", "<H"), ("whirlpoolRate", "<H"),
    ("directionalSpread", "<H"), ("circularPathRate", "<H"), ("innerOriginRange", "<H"),
    ("generateRandomness", "<H"), ("misc2", "<H"), ("misc3", "<H"),
    ("generationType", "<B"), ("elementsPerTurn", "<B"), ("unknown3", "<H"),
]
EFFECT_ELEMENT = [
    ("name", "32s"), *_art("art"),
    ("mass", "<i"), ("airFriction", "<i"), ("elasticity", "<i"), ("minSpeedXy", "<i"),
    ("maxSpeedXy", "<i"), ("minSpeedYz", "<i"), ("maxSpeedYz", "<i"), ("minScale", "<i"),
    ("maxScale", "<i"), ("scaleRatio", "<i"), ("flags", "<I"),
    ("effectElementId", "<H"), ("minHp", "<H"), ("maxHp", "<H"), ("deathElementId", "<H"),
    ("hitSolidElementId", "<H"), ("hitLavaElementId", "<H"), ("hitWaterElementId", "<H"),
    ("colour", "3s"), ("randomColourIndex", "<B"), ("tableColourIndex", "<B"),
    ("fadePercentage", "<B"), ("nextEffectId", "<H"),
]

# signature -> (layout, record size, the field that holds the record's own id)
TABLES = {
    0xA0: (CREATURE_SPELL, 266, "creatureSpellId"),
    0x104: (SHOT, 239, "shotId"),
    0x10E: (EFFECT, 246, "effectId"),
    0xFA: (EFFECT_ELEMENT, 182, "effectElementId"),
}
NAMES = {0xA0: "CreatureSpells", 0x104: "Shots", 0x10E: "Effects", 0xFA: "EffectElements"}


def _offsets(layout):
    out, at = {}, 0
    for name, fmt in layout:
        size = struct.calcsize(fmt)
        out[name] = (at, fmt, size)
        at += size
    return out, at


_OFFSETS = {sig: _offsets(layout) for sig, (layout, _size, _id) in TABLES.items()}
for _sig, (_layout, _size, _id) in TABLES.items():
    assert _OFFSETS[_sig][1] == _size, (hex(_sig), _OFFSETS[_sig][1], _size)


@dataclass
class Record:
    signature: int
    raw: bytearray

    def _field(self, name):
        try:
            return _OFFSETS[self.signature][0][name]
        except KeyError:
            raise KeyError(f"{NAMES[self.signature]} has no field {name!r}") from None

    def get(self, name):
        at, fmt, _ = self._field(name)
        (value,) = struct.unpack_from(fmt, self.raw, at)
        return value

    def set(self, name, value) -> None:
        at, fmt, size = self._field(name)
        if fmt.endswith("s"):
            if isinstance(value, str):
                value = value.encode("latin-1")
            if len(value) > size:
                raise ValueError(f"{name}: {len(value)} bytes do not fit {size}")
            value = value.ljust(size, b"\0")
        struct.pack_into(fmt, self.raw, at, value)

    def text(self, name) -> str:
        return self.get(name).split(b"\0")[0].decode("latin-1")

    @property
    def id(self) -> int:
        return self.get(TABLES[self.signature][2])

    @property
    def name(self) -> str:
        return self.text("name")

    def fields(self) -> dict:
        return {n: self.get(n) for n in _OFFSETS[self.signature][0]}


def read_records(block) -> list[Record]:
    """The records of a block (dk2kwd_blocks.Block) of one of TABLES."""
    _layout, size, _id = TABLES[block.signature]
    if len(block.content) != size * block.item_count:
        raise ValueError(f"{NAMES[block.signature]}: {len(block.content)} bytes for "
                         f"{block.item_count} records of {size}")
    return [Record(block.signature, bytearray(block.content[i * size:(i + 1) * size]))
            for i in range(block.item_count)]


def records_content(records: list[Record]) -> bytes:
    return b"".join(bytes(r.raw) for r in records)


def by_id(records: list[Record]) -> dict[int, Record]:
    return {r.id: r for r in records}


def by_name(records: list[Record]) -> dict[str, Record]:
    return {r.name: r for r in records}
