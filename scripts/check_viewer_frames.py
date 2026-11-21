"""Compare the web viewer's rendering with the Python reference renderer (silhouette IoU + contact sheet).

Usage: python3 -m scripts.check_viewer_frames <site dir> <Meshes dir> <EngineTextures.dir> <EngineTextures.dat> <out dir>
LICENCE: GPL-3.0 (uses scripts.kmf_textured).
"""
import functools
import http.server
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.dk2kmf import parse_kmf
from scripts.dk2texture import TextureCache
from scripts.kmf_textured import fixed_view, render_textured
from scripts.viewer_creatures import CREATURES, find_animations

BG = (255, 0, 255)
CASES = [("goblin", "walk", 10), ("goblin", "melee", 8), ("bile_demon", "walk", 10),
         ("bile_demon", "melee", 30), ("horned_reaper", "walk", 10), ("horned_reaper", "melee1", 12)]
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def silhouette(img: Image.Image, bg) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=int)
    return np.abs(arr - np.array(bg)).sum(axis=2) > 30


def normalised_iou(a: np.ndarray, b: np.ndarray, size: int = 256, flip: bool = False) -> float:
    """IoU of two masks after cropping each to its bounding box and resizing (framing-independent)."""
    def prep(m):
        ys, xs = np.nonzero(m)
        crop = Image.fromarray((m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8))
        return np.asarray(crop.resize((size, size), Image.BILINEAR)) > 127
    pa, pb = prep(a), prep(b)
    if flip:
        pb = pb[:, ::-1]
    return float((pa & pb).sum() / max((pa | pb).sum(), 1))


def run(site: Path, meshes: Path, cache: TextureCache, out: Path, az_ref: float = 205, az_view: float = 25,
        cases=None):
    """Viewer azimuth = reference azimuth - 180 (viewer front is 0, reference front is 180)."""
    from playwright.sync_api import sync_playwright
    out.mkdir(parents=True, exist_ok=True)
    cases = cases or CASES
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(
        type("Q", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None}), directory=str(site)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=ARGS)
        page = browser.new_page(viewport={"width": 640, "height": 640})
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html")
        page.evaluate("window.viewer.ready")
        page.add_style_tag(content="#disclaimer{display:none!important}")
        page.add_style_tag(content=".app{display:block;height:640px}header,footer,.list{display:none}#stage{height:640px}")
        page.evaluate("window.viewer.scene.background = new window.viewer.THREE.Color(0xff00ff)")
        # the reference renderer is orthographic; a narrow field of view makes the perspective camera comparable
        page.evaluate("window.viewer.camera.fov = 6; window.viewer.camera.updateProjectionMatrix()")
        found = {c: find_animations(meshes, {c: CREATURES[c]})[0][c] for c in {c for c, _, _ in cases}}
        for cid, label, frame in cases:
            page.evaluate(f"window.viewer.setCreature('{cid}')")
            page.evaluate(f"window.viewer.setAnimation('{label}')")
            page.evaluate(f"window.viewer.setView({az_view})")
            page.evaluate(f"window.viewer.showFrame({frame})")
            page.wait_for_timeout(1200)   # textures load
            shot = out / f"{cid}_{label}_viewer.png"
            page.screenshot(path=str(shot))
            model = parse_kmf(found[cid][label].read_bytes())
            ref = render_textured(model, cache, azimuth=az_ref, size=640, frame=frame,
                                  view=fixed_view([(model, frame)], az_ref))
            ref.save(out / f"{cid}_{label}_ref.png")
            ms, mr = silhouette(Image.open(shot), BG), silhouette(ref, (38, 38, 42))
            results.append((cid, label, normalised_iou(ms, mr), normalised_iou(ms, mr, flip=True)))
        browser.close()
    srv.shutdown()
    return results


if __name__ == "__main__":
    site, meshes, d, dat, out = map(Path, sys.argv[1:6])
    for cid, label, iou, flipped in run(site, meshes, TextureCache(d, dat), out):
        print(f"{cid:14s} {label:8s} IoU {iou:.3f}   (mirrored {flipped:.3f})")
