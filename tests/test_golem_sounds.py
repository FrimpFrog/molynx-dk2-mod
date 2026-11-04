from pathlib import Path

import pytest

from scripts.golem_sounds import EVENTS, GROUPS, build_sound_pack

SND = Path("/tmp/mg/mgd/techs/megapack/commondata/sounds")
needs = pytest.mark.skipif(not SND.exists() or not list(SND.glob("golem_*.wav")), reason="MegaGlest sounds absent")


def test_every_event_uses_a_known_group_and_a_valid_time():
    for anim, evs in EVENTS.items():
        for g, at, vol in evs:
            assert g in GROUPS, (anim, g)
            assert 0 <= at < 1 and 0 < vol <= 1, (anim, at, vol)


@needs
def test_pack_contains_every_file_and_drops_events_for_missing_animations():
    pack = build_sound_pack(SND, ["walk", "attack"])
    assert set(pack["events"]) == {"walk", "attack"}
    for names in GROUPS.values():
        for n in names:
            assert pack["files"][n].startswith("data:audio/ogg;base64,")
