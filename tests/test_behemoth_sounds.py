from pathlib import Path

import pytest

from scripts.behemoth_sounds import EVENTS, GROUPS, build_pack

SND = Path(__file__).resolve().parent.parent / "megaglest_behemoth" / "sounds"
needs = pytest.mark.skipif(not SND.exists(), reason="MegaGlest Behemoth sounds absent")


def test_events_use_known_groups_and_valid_times():
    for anim, evs in EVENTS.items():
        for g, at, vol in evs:
            assert g in GROUPS and 0 <= at < 1 and 0 < vol <= 1, (anim, g, at, vol)


def test_no_event_for_a_clip_that_does_not_exist():
    assert not {"casting_spell", "researching", "blocking", "looking_around", "frightened", "unhappy"} & set(EVENTS)


@needs
def test_every_group_file_exists_and_pack_filters_animations():
    pack = build_pack(SND, ["attack", "die"])
    assert set(pack["events"]) == {"attack", "die"}
    assert all(v.startswith("data:audio/ogg;base64,") for v in pack["files"].values())
    assert len(pack["files"]) == 2 + 8 + 4 + 4 + 6
