from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import glest_export as ge
from scripts.viewer_export import read_dk2v

UNIT = Path("/tmp/mg/mgd/techs/megapack/factions/magic/units/golem")
needs_golem = pytest.mark.skipif(not UNIT.exists(), reason="MegaGlest data not present")


def test_playback_fps_matches_engine_rule():
    # engine: one cycle takes 100 / animSpeed seconds; walk anim-speed 20, 15 frames -> 3 fps
    assert ge.playback_fps(15, 20, True) == pytest.approx(3.0)
    assert ge.playback_fps(12, 50, True) == pytest.approx(6.0)
    assert ge.playback_fps(11, 60, False) == pytest.approx(6.0)     # non-looping: (11 - 1) * 60 / 100


def test_parse_skills_reads_types_speeds_paths(tmp_path):
    (tmp_path / "u.xml").write_text("""<unit><skills>
      <skill><type value="stop"/><name value="s"/><anim-speed value="50"/><animation path="models/a.g3d"/></skill>
      <skill><type value="attack"/><name value="x"/><animation path="models/b.g3d"/><animation path="models/c.g3d"/></skill>
      <skill><type value="produce"/><name value="noanim"/></skill></skills></unit>""")
    s = ge.parse_skills(tmp_path / "u.xml")
    assert [(k["type"], k["anim_speed"], k["paths"]) for k in s] == [
        ("stop", 50.0, ["models/a.g3d"]), ("attack", 100.0, ["models/b.g3d", "models/c.g3d"])]


def test_bake_texture_uses_team_colour_only_for_custom_meshes(tmp_path):
    im = Image.new("RGBA", (2, 1)); im.putpixel((0, 0), (255, 0, 255, 0)); im.putpixel((1, 0), (10, 200, 10, 255))
    im.save(tmp_path / "t.png")
    a = ge.bake_texture(tmp_path / "t.png", True, team=(1, 2, 3))
    b = ge.bake_texture(tmp_path / "t.png", False)
    assert a.getpixel((0, 0)) == (1, 2, 3) and a.getpixel((1, 0)) == (10, 200, 10)
    assert b.getpixel((0, 0)) == ge.PLAIN_BACKDROP


@needs_golem
def test_golem_exports_all_animations_with_engine_speeds(tmp_path):
    path, tex = ge.export_glest_unit(UNIT, "golem", "Golem", tmp_path)
    d = read_dk2v(path)
    m = d["meta"]
    assert m["mirror_x"] is False
    assert set(m["animations"]) == {"stand", "walk", "attack", "die", "build_up"}
    assert m["animations"]["walk"]["frame_count"] == 15 and m["animations"]["walk"]["fps"] == pytest.approx(3.0)
    assert m["animations"]["die"]["wrap"] is False and m["animations"]["stand"]["wrap"] is True
    walk = d["animations"]["walk"]
    assert len(walk["parts"]) == 1 and walk["parts"][0]["vertex_count"] == 407
    assert sorted(tex) == ["golem/texture_golem", "golem/texture_golem_plain"]
    assert all(im.size == (256, 256) for im in tex.values())
    frames = np.array([f[0] for f in walk["frames"]])
    assert frames.std(axis=0).max() > 0.001           # the vertices actually move between frames


def test_inferno_recolour_turns_moss_red_keeps_grey_stone(tmp_path):
    im = Image.new("RGBA", (4, 1), (0, 0, 0, 255))
    im.putpixel((0, 0), (60, 130, 50, 255))        # moss green
    im.putpixel((1, 0), (130, 125, 120, 255))      # grey stone
    im.putpixel((2, 0), (110, 90, 70, 255))        # brown stone
    im.putpixel((3, 0), (60, 90, 255, 40))         # blue rune (low alpha)
    im.save(tmp_path / "t.png")
    out = ge.bake_inferno(tmp_path / "t.png", True)
    r, g, b = out.getpixel((0, 0))
    assert r > g + 30 and r > b + 30                              # moss is now red
    assert max(abs(x - y) for x, y in zip(out.getpixel((1, 0)), (130, 125, 120))) <= 6      # grey stone untouched
    assert max(abs(x - y) for x, y in zip(out.getpixel((2, 0)), (110, 90, 70))) <= 6        # brown stone untouched
    er, eg, eb = out.getpixel((3, 0))
    assert er > 200 and eb < 120                                   # rune glows red-orange, not blue
    assert ge.bake_inferno(tmp_path / "t.png", False).getpixel((3, 0))[0] < 120


@needs_golem
def test_inferno_export_uses_its_own_texture_names(tmp_path):
    path, tex = ge.export_glest_unit(UNIT, "golem_inferno", "Golem (Inferno)", tmp_path,
                                     baker=ge.bake_inferno, tex_prefix="golem_inferno")
    assert sorted(tex) == ["golem_inferno/texture_golem", "golem_inferno/texture_golem_plain"]
