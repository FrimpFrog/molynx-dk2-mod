"""Named fields of the Dungeon Keeper 2 creature record (Creatures block, signature 0xAA).

Layout, names and order follow the community template ufdada/dk2-tools
Formats/Level/creatures_struct.bt (with util_struc.bt and Level/resource.bt for the nested types),
and every size was checked against the editor Creatures.kwd and all 241 *Globals.kld maps: the
fields tile the record with no gap, 5449 bytes, plus an 88-byte tail (unk_40 .. unk_42) in records
of 5537 bytes. All values are little-endian. Nested structs are flattened into dotted / indexed
names ("light.radius", "spells[2].creatureSpellId", "footstepFrames[47].right").

The template's float32 is a fixed-point int32 (value / 4096) and float16 an int16 (value / 4096);
both are read and written here as the raw integers. Bitfield structs (CreatureFlag, CreatureFlag2,
LightFlag) are whole 32-bit words. Where the data disagrees with the template it wins: angerSlap and
angerInHand are signed (Mistress: slapping removes 5000 anger), not the template's ushort. ArtResources (animation slots, icons, portrait, ...) are 84
bytes whose name is the NUL-terminated first 64; `get` returns their raw bytes and
`resource_name` / `anim_name` their names.

LICENCE: GPL-3.0.
"""
import struct
from dataclasses import dataclass

from scripts.dk2kwd_blocks import Block, rebuilt_block

RECORD_SIZE = 5449
RECORD_SIZE_WITH_TAIL = 5537
RESOURCE_SIZE = 84
RESOURCE_NAME_SIZE = 64
FIXED_ONE = 4096  # float32 / float16 fixed-point scale

_ART = f"{RESOURCE_SIZE}s"
_V3 = "<3i"  # Vector3f: int32 x, y, z, fixed point
# Nested template structs, flattened: {type: [(member, fmt), ...]}.
_STRUCTS: dict[str, list[tuple[str, str]]] = {
    "Light": [("mKPos", _V3), ("radius", "<i"), ("flags", "<I"), ("color", "4B")],
    "Attraction": [("present", "<I"), ("roomId", "<H"), ("roomSize", "<H")],
    "Spell": [("shotOffset", _V3), ("unk_1", "<B"), ("playAnimation", "<B"), ("unk_2", "<B"),
              ("unk_3", "<B"), ("shotDelay", "<i"), ("unk_4", "<B"), ("unk_5", "<B"),
              ("creatureSpellId", "<B"), ("levelAvailable", "<B")],
    "Resistance": [("attackType", "<B"), ("value", "<B")],
    "JobPreference": [("jobType", "<I"), ("moodChange", "<H"), ("manaChange", "<H"), ("chance", "<B"),
                      ("pad_1", "<B"), ("pad_2", "<B"), ("pad_3", "<B")],
    "JobAlternative": [("jobType", "<I"), ("moodChange", "<H"), ("manaChange", "<H")],
    "Footstep": [("left", "<H"), ("right", "<H")],
    "HateJobsUnused": [("value", "4B")],
}
# Template scalar and enum types -> struct format.
_TYPES = {
    "ubyte": "<B", "ushort": "<H", "short": "<h", "uint": "<I", "int": "<i",
    "float32": "<i", "float16": "<h", "Vector3f": _V3, "ArtResource": _ART, "UnkArtResource": _ART,
    "JobType": "<I", "Swipe": "<B", "SpecialAbility": "<B", "AttackType": "<B", "JobClass": "<B",
    "FightStyle": "<B", "Direction": "<B", "GammaEffect": "<B", "Material": "<b",
    "CreatureFlag": "<I", "CreatureFlag2": "<I",
}
_ANIMS_1_36 = ("Walk Run DraggedPose RecoilHff Melee1 Magic Die Happy Angry StunnedPose Swing Sleep Eat "
               "Research Unused15 DejectedPose Torture Unused18 Drink Idle1 RecoilHfb Building Pray "
               "Fallback Elec Electrocute GetUp Dance Drunk Entrance Idle2 Special EatCorpse Drunk2 Loot "
               "Unused36").split()
_EXTRAS = ["Melee2", "Extra43", "Extra44", "Extra45", "Extra46", "Extra47"]

# (name, type, count) in record order; count 0 = not an array. "char[n]" is a text field.
_LAYOUT: list[tuple[str, str, int]] = [
    ("name", "char[32]", 0), ("unk_1", "UnkArtResource", 0),
    *[(f"anim{a}Resource", "ArtResource", 0) for a in _ANIMS_1_36],
    ("icon1Resource", "ArtResource", 0), ("icon2Resource", "ArtResource", 0),
    ("swipeSpeed", "ushort", 0), ("workModifierDefault", "uint", 0),
    ("workModifierLevelIncrease", "uint", 0), ("orderInEditor", "ubyte", 0),
    ("angerStringIdGeneral", "ushort", 0), ("shotDelay", "uint", 0), ("olhiEffectId", "ushort", 0),
    ("introductionStringId", "ushort", 0), ("perceptionRange", "float32", 0),
    *[(f"angerStringId{a}", "ushort", 0)
      for a in "Lair Food Pay Work Slap Held Lonely Hatred Torture".split()],
    ("translationSoundGategory", "char[32]", 0), ("shuffleSpeed", "float32", 0),
    ("cloneCreatureId", "ubyte", 0), ("firstPersonGammaEffect", "GammaEffect", 0),
    ("firstPersonWalkCycleScale", "ubyte", 0), ("introCameraPathIndex", "ubyte", 0),
    ("unk_12", "ubyte", 0), ("portraitResource", "ArtResource", 0), ("light", "Light", 0),
    ("attractions", "Attraction", 2), ("firstPersonWaddleScale", "float32", 0),
    ("firstPersonOscillateScale", "float32", 0), ("spells", "Spell", 3),
    ("resistances", "Resistance", 4), ("happyJobs", "JobPreference", 3),
    ("unhappyJobs", "JobPreference", 2), ("angryJobs", "JobPreference", 3),
    ("hateJobs", "JobType", 2), ("alternativeJobs", "JobAlternative", 3),
    ("animationOffset", "Vector3f", 0),
    *[(n, "float32", 0) for n in "width height mass airFriction eyeHeight speed runSpeed hungerRate".split()],
    ("timeAwake", "int", 0), ("timeSleep", "int", 0),
    *[(n, "float32", 0) for n in ("distanceCanSee distanceCanHear stunDuration guardDuration "
                                  "idleDuration slapFearlessDuration").split()],
    ("unk_17", "int", 0), ("abilityCreepPercentage", "int", 0), ("possessionManaCost", "short", 0),
    ("ownLandHealthIncrease", "short", 0), ("meleeRange", "float32", 0), ("unk_19", "int", 0),
    ("tortureTimeToConvert", "float32", 0), ("meleeRecharge", "float32", 0),
    ("flags", "CreatureFlag", 0), ("expForNextLevel", "ushort", 0), ("jobClass", "JobClass", 0),
    ("fightStyle", "FightStyle", 0),
    *[(n, "ushort", 0) for n in ("expPerSecond expPerSecondTraining researchPerSecond "
                                 "manufacturePerSecond hp hpFromChicken fear threat meleeDamage "
                                 "slapDamage manaGenPrayer unk_20 pay maxGoldHeld maxAngleChange "
                                 "decomposeValue nameStringId tooltipStringId angerNoLair angerNoFood "
                                 "angerNoPay angerNoWork").split()],
    ("angerSlap", "short", 0), ("angerInHand", "short", 0),  # template: ushort; data: signed
    *[(n, "ushort", 0) for n in ("initialGoldHeld "
                                 "entranceEffectId generalDescriptionStringId strengthStringId "
                                 "weaknessStringId slapEffectId deathEffectId").split()],
    *[(n, "Swipe", 0) for n in "melee1Swipe melee2Swipe melee3Swipe spellSwipe".split()],
    ("firstPersonSpecialAbility1", "SpecialAbility", 0),
    ("firstPersonSpecialAbility2", "SpecialAbility", 0), ("unk_23", "ubyte", 3),
    *[(n, "ubyte", 0) for n in ("creatureId scoreRating roomCapacityNeeded hungerFill "
                                "unhappyThreshold").split()],
    ("meleeAttackType", "AttackType", 0), ("disarmFailurePercentage", "ubyte", 0),
    ("lairObjectId", "ubyte", 0), ("unk_26", "ubyte", 0), ("deathFallDirection", "Direction", 0),
    ("unk_27", "ubyte", 0), ("soundGategory", "char[32]", 0), ("material", "Material", 0),
    ("firstPersonFilterResource", "ArtResource", 0), ("satisfyAngerTreat", "short", 0),
    ("scale", "float32", 0), ("drunkIdle", "ArtResource", 0), ("special1Swipe", "Swipe", 0),
    ("special2Swipe", "Swipe", 0), ("firstPersonMeleeResource", "ArtResource", 0),
    ("sniperZoomMultiplier", "float32", 0), ("tortureHpChange", "short", 0),
    ("tortureMoodChange", "short", 0),
    *[(f"anim{a}Resource", "ArtResource", 0) for a in _EXTRAS],
    ("animationOffsets", "Vector3f", 7), ("animWalkbackResource", "ArtResource", 0),
    ("footstepFrames", "Footstep", 48), ("animPoseFrameResource", "ArtResource", 0),
    ("animWalk2Resource", "ArtResource", 0), ("animDiePoseResource", "ArtResource", 0),
    ("uniqueNameTextId", "ushort", 0), ("hateJobs34", "HateJobsUnused", 2),
    ("firstPersonSpecialAbility1Count", "int", 0), ("firstPersonSpecialAbility2Count", "int", 0),
    ("uniqueResource", "ArtResource", 0), ("flags2", "CreatureFlag2", 0),
    # Tail, only in 5537-byte records (maps' globals).
    ("unk_40", "ubyte", 80), ("flags3", "CreatureFlag2", 0), ("unk_41", "ushort", 0),
    ("unk_42", "float16", 0),
]


@dataclass(frozen=True)
class Field:
    offset: int
    fmt: str

    @property
    def end(self) -> int:
        return self.offset + struct.calcsize(self.fmt)


def _build_fields() -> tuple[dict[str, Field], set[str], list[str]]:
    fields: dict[str, Field] = {}
    text: set[str] = set()
    resources: list[str] = []
    pos = 0

    def add(name: str, fmt: str) -> None:
        nonlocal pos
        fields[name] = Field(pos, fmt)
        pos += struct.calcsize(fmt)

    for name, kind, count in _LAYOUT:
        if kind.startswith("char["):
            text.add(name)
            add(name, f"{kind[5:-1]}s")
        elif kind in _STRUCTS:
            for i in range(count or 1):
                prefix = f"{name}[{i}]" if count else name
                for member, fmt in _STRUCTS[kind]:
                    add(f"{prefix}.{member}", fmt)
        elif count and kind == "ubyte":
            add(name, f"{count}B")
        elif count:
            for i in range(count):
                add(f"{name}[{i}]", _TYPES[kind])
        else:
            if kind in ("ArtResource", "UnkArtResource"):
                resources.append(name)
            add(name, _TYPES[kind])
    assert fields["flags2"].end == RECORD_SIZE and pos == RECORD_SIZE_WITH_TAIL
    return fields, text, resources


FIELDS, TEXT_FIELDS, _ALL_RESOURCES = _build_fields()
# The 48 animation slots, in record order; the template comments number unk_1 as slot 0,
# walk .. unused36 as 1 .. 36 and melee2, extra43 .. extra47 as 42 .. 47.
ANIM_SLOTS: list[str] = [n for n in _ALL_RESOURCES if n.startswith("anim") or n in ("unk_1", "drunkIdle")]
# The other 84-byte resources (icons, portrait, first-person, unique).
RESOURCE_FIELDS: list[str] = [n for n in _ALL_RESOURCES if n not in ANIM_SLOTS]


def _short(slot: str) -> str:
    if slot.startswith("anim") and slot.endswith("Resource"):
        slot = slot[4:-8]
    return slot.lower()


_SLOT_BY_SHORT = {_short(s): s for s in ANIM_SLOTS}


def anim_slot(slot: str) -> str:
    """The field name of an animation slot given by field name or short name ("walk", "poseFrame",
    "drunkIdle", "extra43", "unk_1", ...; case-insensitive)."""
    return _SLOT_BY_SHORT[_short(slot)]


def _put_text(raw: bytearray, offset: int, size: int, text: str) -> None:
    data = text.encode("ascii")
    if len(data) >= size:
        raise ValueError(f"{text!r} needs {len(data) + 1} bytes with its NUL, the field has {size}")
    raw[offset:offset + size] = data.ljust(size, b"\0")


def _text(raw: bytearray, offset: int, size: int) -> str:
    return bytes(raw[offset:offset + size]).split(b"\0", 1)[0].decode("latin-1")


class Creature:
    """One creature record; `raw` is the record itself, edited in place."""

    def __init__(self, raw: bytes | bytearray):
        if len(raw) not in (RECORD_SIZE, RECORD_SIZE_WITH_TAIL):
            raise ValueError(f"creature record of {len(raw)} bytes, expected "
                             f"{RECORD_SIZE} or {RECORD_SIZE_WITH_TAIL}")
        self.raw = bytearray(raw)

    def _field(self, name: str) -> Field:
        field = FIELDS[name]
        if field.end > len(self.raw):
            raise KeyError(f"{name} is not in a {len(self.raw)}-byte record")
        return field

    def get(self, name: str):
        """A single value as int (or bytes / str), a multi-value field as a tuple. Text fields
        (char[n]) give the text before the NUL."""
        field = self._field(name)
        if name in TEXT_FIELDS:
            return _text(self.raw, field.offset, struct.calcsize(field.fmt))
        values = struct.unpack_from(field.fmt, self.raw, field.offset)
        return values[0] if len(values) == 1 else values

    def set(self, name: str, value) -> None:
        """Write a value in the shape `get` returns; text is NUL-padded to the field size."""
        field = self._field(name)
        if name in TEXT_FIELDS:
            _put_text(self.raw, field.offset, struct.calcsize(field.fmt), value)
            return
        values = value if isinstance(value, tuple) else (value,)
        struct.pack_into(field.fmt, self.raw, field.offset, *values)

    def resource_name(self, field: str) -> str:
        """The name of an 84-byte resource (animation slot by field name, icon, portrait, ...)."""
        if field not in _ALL_RESOURCES:
            raise KeyError(f"{field} is not a resource")
        return _text(self.raw, FIELDS[field].offset, RESOURCE_NAME_SIZE)

    def set_resource_name(self, field: str, name: str) -> None:
        """Set a resource's name (first 64 bytes, NUL-padded); its other 20 bytes are kept."""
        if field not in _ALL_RESOURCES:
            raise KeyError(f"{field} is not a resource")
        _put_text(self.raw, FIELDS[field].offset, RESOURCE_NAME_SIZE, name)

    def anim_name(self, slot: str) -> str:
        return self.resource_name(anim_slot(slot))

    def set_anim_name(self, slot: str, name: str) -> None:
        self.set_resource_name(anim_slot(slot), name)


def read_creatures(block: Block) -> list[Creature]:
    """The records of a Creatures block (record size = content size / item count)."""
    size, rest = divmod(len(block.content), block.item_count)
    if rest:
        raise ValueError(f"{len(block.content)} content bytes do not split into {block.item_count} records")
    return [Creature(block.content[i * size:(i + 1) * size]) for i in range(block.item_count)]


def creatures_block(creatures: list[Creature], like: Block) -> Block:
    """A Creatures block holding `creatures`, its header otherwise copied from `like`."""
    return rebuilt_block(like, [bytes(c.raw) for c in creatures])
