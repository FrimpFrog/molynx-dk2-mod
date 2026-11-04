"""Take screenshots of a creature in the web viewer (headless Chromium) and make contact sheets / GIFs.

Usage: python3 -m scripts.shoot_viewer <site dir> <creature id> <out dir> [--hd]
"""
import functools
import http.server
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
BG = (38, 38, 42)


@contextmanager
def viewer_page(site: Path, size: int = 480, hd: bool = False):
    from playwright.sync_api import sync_playwright
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        type("Q", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None}), directory=str(site)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=ARGS)
        page = browser.new_page(viewport={"width": size, "height": size})
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html")
        page.evaluate("window.viewer.ready")
        page.add_style_tag(content="#disclaimer{display:none!important}")
        page.add_style_tag(content=f".app{{display:block;height:{size}px}}header,footer,.list{{display:none}}#stage{{height:{size}px}}")
        page.evaluate("window.viewer.scene.background = new window.viewer.THREE.Color(0x26262a)")
        page.evaluate("window.viewer.camera.fov = 20; window.viewer.camera.updateProjectionMatrix()")
        if hd:
            page.evaluate("window.viewer.setPack('hd')")
        yield page
        browser.close()
    srv.shutdown()


def shot(page, creature, anim, frame, az, path=None, wait=700) -> Image.Image:
    page.evaluate(f"window.viewer.setCreature('{creature}')")
    page.evaluate(f"window.viewer.setAnimation('{anim}')")
    page.evaluate(f"window.viewer.setView({az})")
    page.evaluate(f"window.viewer.showFrame({frame})")
    page.wait_for_timeout(wait)
    if path:
        page.screenshot(path=str(path))
        return Image.open(path).convert("RGB")
    import io
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def sheet(images, cols):
    w, h = images[0].size
    rows = -(-len(images) // cols)
    out = Image.new("RGB", (w * cols, h * rows), BG)
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out


if __name__ == "__main__":
    site, cid, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    with viewer_page(site, hd="--hd" in sys.argv) as page:
        first = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "idle1"
        views = [shot(page, cid, first, 0, az) for az in (25, 115, 205, 295)]
        sheet(views, 4).save(out / "turnaround.png")
        extra = [shot(page, cid, a, f, 25) for a, f in (("walk", 8), ("melee", 10), ("dance", 12), ("angry", 10))]
        sheet(extra, 4).save(out / "actions.png")
