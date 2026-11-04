import struct

import pytest

from scripts.dk2kld_splice import molynx_for_map, splice_creatures
from scripts.dk2kwd_blocks import HEADER_SIZE, Block, read_blocks, write_blocks
from scripts.dk2kwd_creatures import Creature, creatures_block, read_creatures
from scripts.dk2paths import MAPS_DIR, QUUZ_MAPS_DIR, have
from scripts.molynx_record import build_molynx

CREATURES = 0xAA
BILE_DEMON_ID = 3
TEXT_IDS = {"name": 3001, "tooltip": 3002, "general": 3003, "strength": 3004, "weakness": 3005}

# The three map shapes the splice must handle, pinned by file name so the test is deterministic:
# id -> (Globals file in MAPS_DIR, Creatures records, record size). test_pinned_maps_have_their_shape
# fails if a data refresh changes any of them. Each is the first file of its shape in sorted order.
FIXTURES = {
    "46x5537": ("01coopGlobals.kld", 46, 5537),
    "44x5537": ("Level10Globals.kld", 44, 5537),
    "46x5449": ("06coopGlobals.kld", 46, 5449),
}

needs_maps = pytest.mark.skipif(not have(MAPS_DIR), reason="DK2 editor maps not present")


# ---- helpers ------------------------------------------------------------------------------------

def _creatures_index(blocks: list[Block]) -> int:
    (index,) = [i for i, b in enumerate(blocks) if b.signature == CREATURES]
    return index


def _records(data: bytes) -> list[bytes]:
    blocks = read_blocks(data)
    return [bytes(c.raw) for c in read_creatures(blocks[_creatures_index(blocks)])]


def _synthetic_block(signature: int, count: int, record: bytes) -> bytes:
    content = record * count
    header = (struct.pack("<7I", signature, 4, HEADER_SIZE + len(content), signature + 1, 28, count, 1)
              + bytes(range(10)) + bytes(range(10, 20))
              + struct.pack("<2I", signature + 2, len(content)))
    return header + content


def _broken_header_invariants(block: Block) -> list[str]:
    sig, four, size, check_one, twenty_eight, _count, one = struct.unpack_from("<7I", block.header, 0)
    check_two, content_size = struct.unpack_from("<2I", block.header, 48)
    checks = {
        "header_size": len(block.header) == HEADER_SIZE, "signature": sig == block.signature,
        "four": four == 4, "twenty_eight": twenty_eight == 28, "one": one == 1,
        "check_one": check_one == block.signature + 1, "check_two": check_two == block.signature + 2,
        "file_size": size == HEADER_SIZE + len(block.content), "content_size": content_size == len(block.content),
        "whole_records": len(block.content) % block.item_count == 0,
    }
    return [name for name, ok in checks.items() if not ok]


def _with_creatures(data: bytes, edit) -> bytes:
    """`data` with its Creatures records changed by `edit(list[Creature]) -> list[Creature]`."""
    blocks = read_blocks(data)
    at = _creatures_index(blocks)
    blocks[at] = creatures_block(edit(read_creatures(blocks[at])), blocks[at])
    return write_blocks(blocks)


@pytest.fixture(scope="module", params=list(FIXTURES), ids=list(FIXTURES))
def fixture(request):
    """(Globals file bytes, record count, record size) of one pinned map."""
    name, count, size = FIXTURES[request.param]
    path = MAPS_DIR / name
    if not have(path):
        pytest.skip(f"{name} not present")
    return path.read_bytes(), count, size


@pytest.fixture(scope="module", params=[1, 2, 3])
def stage(request):
    return request.param


@pytest.fixture(scope="module")
def spliced(fixture, stage):
    """(original bytes, Molynx record, spliced bytes) for each pinned map at each stage."""
    data, _count, _size = fixture
    molynx = molynx_for_map(data, stage, TEXT_IDS)
    return data, molynx, splice_creatures(data, [molynx])


# ---- the fixtures are what they are -------------------------------------------------------------

@needs_maps
@pytest.mark.parametrize("name,count,size", list(FIXTURES.values()), ids=list(FIXTURES))
def test_pinned_maps_have_their_shape(name, count, size):
    blocks = read_blocks((MAPS_DIR / name).read_bytes())
    block = blocks[_creatures_index(blocks)]
    assert (block.item_count, len(block.content) // block.item_count) == (count, size)
    assert len(block.content) == count * size
    assert [c.get("creatureId") for c in read_creatures(block)] == list(range(1, count + 1))


# ---- splice_creatures ---------------------------------------------------------------------------

def test_one_more_record_with_creature_id_47(spliced, fixture):
    _data, count, size = fixture
    _original, molynx, result = spliced
    blocks = read_blocks(result)
    block = blocks[_creatures_index(blocks)]
    assert block.item_count == count + 1
    assert len(block.content) == (count + 1) * size
    creatures = read_creatures(block)
    assert creatures[-1].get("creatureId") == 47
    assert creatures[-1].get("name") == "Molynx"
    assert creatures[-1].raw == molynx.raw


def test_the_existing_records_are_untouched(spliced, fixture):
    original, _molynx, result = spliced
    count = fixture[1]
    old, new = _records(original), _records(result)
    assert len(old) == count
    assert new[:count] == old


def test_every_other_block_is_byte_identical(spliced):
    original, _molynx, result = spliced
    before, after = read_blocks(original), read_blocks(result)
    assert [b.signature for b in after] == [b.signature for b in before]
    at = _creatures_index(before)
    for i, (old, new) in enumerate(zip(before, after)):
        if i != at:
            assert new == old, f"block {i} (0x{old.signature:X}) changed"
    # The file is nothing but blocks, so what sits before and after the Creatures block is unchanged.
    start = sum(HEADER_SIZE + len(b.content) for b in before[:at])
    end = start + HEADER_SIZE + len(before[at].content)
    new_end = start + HEADER_SIZE + len(after[at].content)
    assert result[:start] == original[:start]
    assert result[new_end:] == original[end:]


def test_creatures_header_changes_only_count_and_sizes(spliced):
    original, _molynx, result = spliced
    before, after = read_blocks(original), read_blocks(result)
    at = _creatures_index(before)
    old, new = before[at].header, after[at].header
    # Everything but file size (8:12), item count (20:24) and content size (52:56) is copied verbatim.
    for lo, hi in [(0, 8), (12, 20), (24, 52)]:
        assert new[lo:hi] == old[lo:hi]
    (old_size, old_count, old_content), (new_size, new_count, new_content) = (
        (struct.unpack_from("<I", h, 8)[0], struct.unpack_from("<I", h, 20)[0],
         struct.unpack_from("<I", h, 52)[0]) for h in (old, new))
    record = len(before[at].content) // old_count
    assert (new_count, new_size, new_content) == (old_count + 1, old_size + record, old_content + record)


def test_result_reads_back_with_the_header_invariants(spliced):
    _original, _molynx, result = spliced
    blocks = read_blocks(result)
    assert write_blocks(blocks) == result
    for block in blocks:
        assert _broken_header_invariants(block) == [], f"0x{block.signature:X}"


def test_the_input_is_not_modified(fixture, stage):
    data = fixture[0]
    snapshot = bytes(data)
    molynx = molynx_for_map(data, stage, TEXT_IDS)
    raw_before = bytes(molynx.raw)
    splice_creatures(data, [molynx])
    assert data == snapshot
    assert bytes(molynx.raw) == raw_before


def test_splicing_is_deterministic(spliced):
    original, molynx, result = spliced
    assert splice_creatures(original, [Creature(molynx.raw)]) == result


# ---- record sizes -------------------------------------------------------------------------------

@needs_maps
def test_a_5449_byte_molynx_does_not_go_into_a_5537_byte_map():
    small = (MAPS_DIR / FIXTURES["46x5449"][0]).read_bytes()
    big = (MAPS_DIR / FIXTURES["46x5537"][0]).read_bytes()
    molynx = molynx_for_map(small, 1)
    assert len(molynx.raw) == 5449
    with pytest.raises(ValueError, match="5449.*5537"):
        splice_creatures(big, [molynx])


@needs_maps
def test_a_5537_byte_molynx_does_not_go_into_a_5449_byte_map():
    small = (MAPS_DIR / FIXTURES["46x5449"][0]).read_bytes()
    big = (MAPS_DIR / FIXTURES["46x5537"][0]).read_bytes()
    molynx = molynx_for_map(big, 1)
    assert len(molynx.raw) == 5537
    with pytest.raises(ValueError, match="5537.*5449"):
        splice_creatures(small, [molynx])


def test_one_wrong_sized_record_among_good_ones_refuses_all(fixture):
    data, _count, size = fixture
    good = molynx_for_map(data, 1)
    other = Creature(bytes(good.raw[:5449]) if size == 5537 else bytes(good.raw) + bytes(88))
    other.set("creatureId", 48)
    with pytest.raises(ValueError, match="bytes"):
        splice_creatures(data, [good, other])


# ---- what is refused ----------------------------------------------------------------------------

def test_no_records_is_refused(fixture):
    with pytest.raises(ValueError, match="no creature"):
        splice_creatures(fixture[0], [])


def test_a_creature_id_the_map_already_has_is_refused(spliced):
    _original, molynx, result = spliced
    with pytest.raises(ValueError, match="47"):
        splice_creatures(result, [molynx])  # already spliced once


def test_a_repeated_creature_id_in_extra_is_refused(fixture):
    data = fixture[0]
    molynx = molynx_for_map(data, 1)
    with pytest.raises(ValueError, match="47"):
        splice_creatures(data, [molynx, Creature(molynx.raw)])


def test_no_creatures_block_is_refused():
    data = _synthetic_block(0x6E, 2, b"abcd") + _synthetic_block(0x78, 1, b"efgh")
    with pytest.raises(ValueError, match="Creatures block"):
        splice_creatures(data, [Creature(bytes(5537))])
    with pytest.raises(ValueError, match="Creatures block"):
        molynx_for_map(data, 1)


def test_two_creatures_blocks_are_refused(fixture):
    data = fixture[0]
    blocks = read_blocks(data)
    doubled = write_blocks([*blocks, blocks[_creatures_index(blocks)]])
    with pytest.raises(ValueError, match="Creatures block"):
        splice_creatures(doubled, [molynx_for_map(data, 1)])
    with pytest.raises(ValueError, match="Creatures block"):
        molynx_for_map(doubled, 1)


def test_an_empty_creatures_block_is_refused():
    data = _synthetic_block(CREATURES, 0, b"")
    with pytest.raises(ValueError, match="no records"):
        splice_creatures(data, [Creature(bytes(5537))])
    with pytest.raises(ValueError, match="no records"):
        molynx_for_map(data, 1)


def test_a_truncated_file_is_refused(fixture):
    data = fixture[0]
    with pytest.raises(ValueError, match="truncated"):
        splice_creatures(data[:-1], [molynx_for_map(data, 1)])


def test_records_out_of_creature_id_order_are_refused_not_reordered(fixture):
    data = fixture[0]
    unsorted = _with_creatures(data, lambda cs: [cs[1], cs[0], *cs[2:]])
    with pytest.raises(ValueError, match="ascending"):
        splice_creatures(unsorted, [molynx_for_map(data, 1)])


# ---- order --------------------------------------------------------------------------------------

@needs_maps
def test_extra_records_are_ordered_by_creature_id():
    data = (MAPS_DIR / FIXTURES["44x5537"][0]).read_bytes()  # holds ids 1..44
    molynx = molynx_for_map(data, 1)
    gap = Creature(molynx.raw)
    gap.set("creatureId", 45)
    high = Creature(molynx.raw)
    high.set("creatureId", 48)
    result = splice_creatures(data, [high, molynx, gap])  # deliberately not in order
    blocks = read_blocks(result)
    creatures = read_creatures(blocks[_creatures_index(blocks)])
    assert [c.get("creatureId") for c in creatures] == [*range(1, 45), 45, 47, 48]
    assert _records(result)[:44] == _records(data)


# ---- molynx_for_map -----------------------------------------------------------------------------

def _bile_and_max_order(data: bytes) -> tuple[Creature, int]:
    blocks = read_blocks(data)
    creatures = read_creatures(blocks[_creatures_index(blocks)])
    (bile,) = [c for c in creatures if c.get("name") == "Bile Demon" and c.get("creatureId") == BILE_DEMON_ID]
    return bile, max(c.get("orderInEditor") for c in creatures)


def test_molynx_is_the_maps_bile_demon_run_through_build_molynx(fixture, stage):
    data, count, size = fixture
    molynx = molynx_for_map(data, stage, TEXT_IDS)
    bile, top = _bile_and_max_order(data)
    assert top == count  # orderInEditor runs 1..N in every shipped map, so max + 1 is "last"
    expected = build_molynx(bile, stage, top + 1, TEXT_IDS)
    assert molynx.raw == expected.raw
    assert len(molynx.raw) == size
    assert molynx.get("name") == "Molynx"
    assert molynx.get("creatureId") == 47
    assert molynx.get("orderInEditor") == count + 1  # 47 in a 46-creature map, 45 in a 44-creature map


def test_molynx_stages_differ_the_way_build_molynx_says(fixture):
    data = fixture[0]
    one, two, three = (molynx_for_map(data, s, TEXT_IDS) for s in (1, 2, 3))
    assert one.raw != two.raw != three.raw
    assert one.get("hp") == two.get("hp")  # stats come at stage 3
    assert three.get("hp") > two.get("hp")
    assert three.get("nameStringId") == TEXT_IDS["name"]
    assert two.get("nameStringId") != TEXT_IDS["name"]


def test_molynx_stage_3_without_text_ids_keeps_the_bile_demons(fixture):
    data = fixture[0]
    bile, _ = _bile_and_max_order(data)
    assert molynx_for_map(data, 3).get("nameStringId") == bile.get("nameStringId")


def test_molynx_for_map_does_not_touch_the_map(fixture, stage):
    data = fixture[0]
    snapshot = bytes(data)
    molynx_for_map(data, stage, TEXT_IDS)
    assert data == snapshot


def test_a_bad_stage_is_refused(fixture):
    with pytest.raises(ValueError, match="stage"):
        molynx_for_map(fixture[0], 4)


def test_bad_text_ids_are_refused_at_stage_3(fixture):
    with pytest.raises(ValueError, match="text_ids"):
        molynx_for_map(fixture[0], 3, {"name": 1})


def test_a_map_without_a_bile_demon_is_refused(fixture):
    data = fixture[0]

    def rename(creatures):
        creatures[BILE_DEMON_ID - 1].set("name", "Not A Demon")
        return creatures

    with pytest.raises(ValueError, match="Bile Demon"):
        molynx_for_map(_with_creatures(data, rename), 1)


def test_creature_id_3_must_also_be_named_bile_demon(fixture):
    data = fixture[0]

    def renumber(creatures):
        creatures[BILE_DEMON_ID - 1].set("creatureId", 99)
        return creatures

    with pytest.raises(ValueError, match="Bile Demon"):
        molynx_for_map(_with_creatures(data, renumber), 1)


def test_two_bile_demons_are_refused(fixture):
    data = fixture[0]

    def twin(creatures):
        creatures[0] = Creature(creatures[BILE_DEMON_ID - 1].raw)
        return creatures

    with pytest.raises(ValueError, match="Bile Demon"):
        molynx_for_map(_with_creatures(data, twin), 1)


def test_the_order_in_editor_must_fit_a_byte(fixture):
    data = fixture[0]

    def top(creatures):
        creatures[0].set("orderInEditor", 255)
        return creatures

    with pytest.raises(ValueError, match="order_in_editor"):
        molynx_for_map(_with_creatures(data, top), 1)


# ---- every shipped map --------------------------------------------------------------------------

def _all_globals():
    return [p for d in (MAPS_DIR, QUUZ_MAPS_DIR) if have(d) for p in sorted(d.glob("*Globals.kld"))]


@needs_maps
def test_molynx_splices_into_every_shipped_globals_file():
    files = _all_globals()
    assert len(files) >= 241
    for path in files:
        data = path.read_bytes()
        old = _records(data)
        for stage in (1, 2, 3):
            molynx = molynx_for_map(data, stage, TEXT_IDS)
            result = splice_creatures(data, [molynx])
            new = _records(result)
            assert new[:-1] == old, (path.name, stage)
            assert new[-1] == bytes(molynx.raw), (path.name, stage)
            assert len(read_blocks(result)) == len(read_blocks(data)), (path.name, stage)
