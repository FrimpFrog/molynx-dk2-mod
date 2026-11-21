import functools
import http.server
import shutil
import threading
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

S = Path("/tmp/claude-0/-home-claude/76eb4112-412f-58e4-a30a-b275f992be89/scratchpad")
SITE_SRC = S / "viewer/full"
VIEWER = Path("/home/claude/viewer")
needs_site = pytest.mark.skipif(not (SITE_SRC / "data/index.json").exists(), reason="viewer data not built")
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def make_site(tmp: Path, data_dir: Path | None = None, textures_dir: Path | None = None) -> Path:
    site = tmp / "site"
    shutil.copytree(VIEWER, site, ignore=shutil.ignore_patterns("package.json"))
    shutil.copytree(data_dir or SITE_SRC / "data", site / "data")
    shutil.copytree(textures_dir or SITE_SRC / "textures", site / "textures")
    return site


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture
def serve():
    servers = []

    def _serve(site: Path) -> str:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(site)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}/index.html"

    yield _serve
    for s in servers:
        s.shutdown()


@pytest.fixture
def page():
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(args=ARGS)
        pg = browser.new_page(viewport={"width": 1100, "height": 700})
        pg.errors = []
        pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        yield pg
        browser.close()


def load(page, url, accept=True):
    page.goto(url)
    page.evaluate("window.viewer.ready")
    if accept:
        page.click("#disclaimerOk")


@needs_site
def test_loads_without_errors_and_lists_creatures(page, serve, tmp_path):
    load(page, serve(make_site(tmp_path)))
    names = page.locator("#creatures button").all_inner_texts()
    assert names[:3] == ["Goblin", "Bile Demon", "Horned Reaper"] and len(names) >= 3
    assert page.errors == []


@needs_site
def test_animation_advances(page, serve, tmp_path):
    load(page, serve(make_site(tmp_path)))
    page.evaluate("window.viewer.setAnimation('dance')")
    f0 = page.evaluate("window.viewer.currentFrame()")
    page.wait_for_timeout(700)
    assert page.evaluate("window.viewer.currentFrame()") != f0


@needs_site
def test_hd_toggle_requests_hd_textures(page, serve, tmp_path):
    load(page, serve(make_site(tmp_path)))
    page.click("#packHd")
    page.wait_for_timeout(300)
    reqs = page.evaluate("window.viewer.requests")
    assert any(r.startswith("textures/hd/") for r in reqs)
    assert page.locator("#packNote").inner_text() == "Enhanced (non-AI) 4x pack"
    page.click("#packOriginal")
    assert page.errors == []


@needs_site
def test_missing_hd_file_falls_back_to_original(page, serve, tmp_path):
    site = make_site(tmp_path)
    (site / "textures/hd/GoblinFrontMM0.png").unlink()
    load(page, serve(site))
    page.click("#packHd")
    page.wait_for_timeout(500)
    assert page.evaluate("window.viewer.state.fallbacks") >= 1
    assert "textures/original/GoblinFrontMM0.png" in page.evaluate("window.viewer.requests")
    assert page.locator("#packHd").get_attribute("aria-pressed") == "true"
    assert [e for e in page.errors if "Failed to load resource" not in e] == []


@needs_site
def test_part_without_texture_renders_grey_without_errors(page, serve, tmp_path):
    from scripts.dk2texture import TextureCache
    from scripts.viewer_creatures import CREATURES
    from scripts.viewer_export import export_creature, write_index

    class Fake:
        names = [n for n in TextureCache(S / "dk2game/Ai/EngineTextures.dir",
                                         S / "dk2game/Ai/EngineTextures.dat").names if n != "GoblinFrontMM0"]

    data = tmp_path / "d"
    p = export_creature("goblin", S / "extracted/Meshes", Fake(), data, {"goblin": CREATURES["goblin"]})
    write_index(data, [p])
    load(page, serve(make_site(tmp_path, data_dir=data)))
    assert "GoblinFront" in page.locator("#status").inner_text()
    assert page.errors == []


@needs_site
def test_text_encoded_data_loads_and_plays(page, serve, tmp_path):
    from scripts.build_viewer import build
    from scripts.dk2texture import TextureCache
    cache = TextureCache(S / "dk2game/Ai/EngineTextures.dir", S / "dk2game/Ai/EngineTextures.dat")
    site = tmp_path / "textsite"
    build(site, S / "extracted/Meshes", cache, ["goblin", "chicken"], text_data=True)
    load(page, serve(site))
    assert page.locator("#creatures button").all_inner_texts() == ["Goblin", "Chicken"]
    page.evaluate("window.viewer.setCreature('chicken')")
    page.evaluate("window.viewer.setAnimation(window.viewer.state.index.creatures[1].animations[0].label)")
    f0 = page.evaluate("window.viewer.currentFrame()")
    page.wait_for_timeout(600)
    assert page.evaluate("window.viewer.state.model.label") == "Chicken"
    assert page.errors == []


@needs_site
def test_disclaimer_is_shown_before_the_viewer_opens_and_can_be_accepted(page, serve, tmp_path):
    load(page, serve(make_site(tmp_path)), accept=False)
    box = page.locator("#disclaimer")
    assert box.is_visible()
    text = box.inner_text()
    assert text.startswith("Disclaimer") and "built using AI" in text and "Electronic Arts" in text and "NetDragon" in text and "extreme interest in Dungeon Keeper 3" in text and "Keeper Klan forum wrote letters and organised online petitions" in text and "cancelled" not in text and "Bright Rock Games" in text and "War for the Overworld" not in text and "Subterranean" not in text
    assert page.locator("#packHd").bounding_box() is not None                  # the viewer is loaded behind it
    page.click("#disclaimerOk")
    assert not box.is_visible()
    page.click("#packHd")                                                      # and nothing blocks the viewer afterwards


@needs_site
def test_soles_logo_sits_in_the_header_and_does_not_block_clicks(page, serve, tmp_path):
    load(page, serve(make_site(tmp_path)))
    logo = page.locator("#logo")
    assert logo.is_visible() and page.locator("#logo svg").get_attribute("aria-label") == "SOLES"
    assert logo.evaluate("e => getComputedStyle(e).pointerEvents") == "none"
    head, box = page.locator("header").bounding_box(), logo.bounding_box()
    assert head["y"] <= box["y"] and box["y"] + box["height"] <= head["y"] + head["height"]      # inside the top bar
    assert abs((box["x"] + box["width"] / 2) - (head["x"] + head["width"] / 2)) < 4              # centred
    assert "watermark" not in page.content()
