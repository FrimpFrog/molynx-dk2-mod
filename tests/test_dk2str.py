"""The DK2 text file reader / writer (scripts.dk2str) on the game's own Text.str."""
import pytest

from scripts.dk2paths import TEXT_DIR, have
from scripts.dk2str import StrFile

needs_text = pytest.mark.skipif(not have(TEXT_DIR / "Text.str"), reason="the game's Text.str is not present")


@needs_text
def test_reads_and_writes_text_str_byte_for_byte():
    path = TEXT_DIR / "Text.str"
    text = StrFile.read(path)
    assert text.to_bytes() == path.read_bytes()
    assert text.text(2356) == "Bile Demon"
    assert text.text(2357) == "Bile Demon.  {26}.  Fighting Style: {27}."
    assert text.text(2408) == "{28} the {29} is {30}.  Status: {31}.  Mood: {32}."


@needs_text
def test_most_entries_encode_back_to_their_own_bytes():
    text = StrFile.read(TEXT_DIR / "Text.str")
    same = sum(text.encode(text.text(i)) == text.entries[i] for i in range(len(text.entries)))
    assert same > 0.98 * len(text.entries)


@needs_text
def test_appended_entries_read_back_and_leave_the_rest_alone(tmp_path):
    text = StrFile.read(TEXT_DIR / "Text.str")
    old = list(text.entries)
    ids = text.append(["Molynx", "Molynx.  {26}.  Fighting Style: {27}.", "Won't"])
    assert ids == [len(old), len(old) + 1, len(old) + 2]
    out = tmp_path / "Text.str"
    out.write_bytes(text.to_bytes())
    back = StrFile.read(out, TEXT_DIR / "MBToUni.dat")
    assert back.entries[:len(old)] == old
    assert [back.text(i) for i in ids] == ["Molynx", "Molynx.  {26}.  Fighting Style: {27}.", "Won't"]


@needs_text
def test_a_character_outside_the_code_page_is_refused():
    text = StrFile.read(TEXT_DIR / "Text.str")
    with pytest.raises(ValueError, match="code page"):
        text.encode("Molynx ☺")
