"""dk2kwd_tables: the CreatureSpells, Shots, Effects and EffectElements record layouts, checked on the
game's own files (they skip when the DK2 data is absent)."""
import pytest

from scripts.dk2kwd_blocks import read_blocks
from scripts.dk2kwd_tables import NAMES, TABLES, Record, by_id, by_name, read_records, records_content
from scripts.dk2paths import EDITOR_DIR, MAPS_DIR, QUUZ_MAPS_DIR, have

needs_data = pytest.mark.skipif(not (have(EDITOR_DIR) and have(MAPS_DIR)), reason="DK2 data not present")
EDITOR = {0xA0: "CreatureSpells.kwd", 0x104: "Shots.kwd", 0x10E: "Effects.kwd", 0xFA: "EffectElements.kwd"}
# (signature, id) -> name, from the editor's own lists (OpenKeeper's tables, the World Editor)
KNOWN = {(0xA0, 5): "Gas Cloud", (0xA0, 15): "Cast Armour", (0xA0, 23): "Chicken Arrow",
         (0x104, 3): "Fart", (0x104, 19): "Grenade Explosion", (0x104, 24): "Tremor",
         (0x104, 36): "Chicken Arrow", (0x10E, 103): "Tremor Spell Effect", (0x10E, 43): "Tremor Rocks",
         (0xFA, 6): "Dust", (0xFA, 536): "Earth blast Element"}


def test_record_sizes():
    assert {sig: size for sig, (_l, size, _i) in TABLES.items()} == {0xA0: 266, 0x104: 239, 0x10E: 246,
                                                                       0xFA: 182}


def _globals_files():
    files = sorted(MAPS_DIR.glob("*Globals.kld"))
    if have(QUUZ_MAPS_DIR):
        files += sorted(QUUZ_MAPS_DIR.glob("*Globals.kld"))
    return files


@needs_data
@pytest.mark.parametrize("sig", sorted(EDITOR))
def test_editor_tables_read_and_write_back_byte_for_byte(sig):
    (block,) = read_blocks((EDITOR_DIR / EDITOR[sig]).read_bytes())
    assert block.signature == sig
    records = read_records(block)
    assert len(records) == block.item_count
    assert records_content(records) == block.content
    assert [r.id for r in records] == list(range(1, len(records) + 1))   # id = place + 1


@needs_data
def test_every_globals_table_reads_and_writes_back():
    files = _globals_files()
    assert files
    for path in files:
        for block in read_blocks(path.read_bytes()):
            if block.signature in TABLES:
                assert records_content(read_records(block)) == block.content, (path.name, NAMES[block.signature])


@needs_data
def test_known_records_have_their_names_and_ids():
    tables = {sig: by_id(read_records(read_blocks((EDITOR_DIR / f).read_bytes())[0]))
              for sig, f in EDITOR.items()}
    for (sig, i), name in KNOWN.items():
        assert tables[sig][i].name == name
    gas = tables[0xA0][5]
    assert gas.get("shotTypeId") == 3 and gas.get("range") == 4096 and gas.get("flags") == 265
    assert gas.text("guiIcon.name") == r"gui\c-spells\c-gas_cloud"
    tremor = tables[0x104][24]
    assert tremor.get("processType") == 27 and tremor.get("creationEffectId") == 103
    assert tremor.text("soundCategory") == "KEEPER_SPELL_TREMOR"
    assert by_name(list(tables[0x10E].values()))["Tremor Dust"].id == 44


@needs_data
def test_set_and_get_round_trip_and_refuse_too_long_text():
    (block,) = read_blocks((EDITOR_DIR / "Shots.kwd").read_bytes())
    shot = Record(0x104, bytearray(read_records(block)[0].raw))
    shot.set("data1", 1234)
    shot.set("name", "X")
    assert shot.get("data1") == 1234 and shot.name == "X"
    assert shot.get("name") == b"X".ljust(32, b"\0")
    with pytest.raises(ValueError):
        shot.set("name", "y" * 33)
    with pytest.raises(KeyError, match="no field"):
        shot.get("colour")
