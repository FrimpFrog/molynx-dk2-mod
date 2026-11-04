"""Render GIFs of a creature from the web viewer: python3 -m scripts.make_custom_gifs <site> <id> <out dir> [anim...]"""
import sys
from pathlib import Path

from scripts.shoot_viewer import shot, viewer_page


def make_gif(page, cid, anim, out: Path, az=25, step=2, size=None, fps=30):
    n = page.evaluate(f"(async()=>{{await window.viewer.setCreature('{cid}');window.viewer.setAnimation('{anim}');return window.viewer.state.anim.frameCount}})()")
    shot(page, cid, anim, 0, az)   # first call loads the textures
    frames = [shot(page, cid, anim, f, az, wait=0) for f in range(0, n, step)]
    if size:
        frames = [im.resize((size, size)) for im in frames]
    frames = [im.quantize(colors=128, method=2) for im in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps * step), loop=0, disposal=2)
    return out, n


if __name__ == "__main__":
    site, cid, outdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    anims = sys.argv[4:] or ["walk", "dance", "melee", "idle1"]
    with viewer_page(site, size=400) as page:
        for a in anims:
            p, n = make_gif(page, cid, a, outdir / f"{cid}_{a}.gif")
            print(p, n, "frames")


def make_timed_gif(page, cid, anim, out: Path, az=25, out_fps=20, max_frames=110, hold_last=0):
    """GIF at the creature's real playback speed (per-animation fps), blending between keyframes like the game."""
    info = page.evaluate(f"(async()=>{{await window.viewer.setCreature('{cid}');window.viewer.setAnimation('{anim}');"
                         f"const a=window.viewer.state.anim;return [a.frameCount,a.fps||window.viewer.state.model.fps,a.wrap]}})()")
    n, fps, wrap = info
    span = n if wrap else n - 1                       # frames of travel; wrapping animations end back at frame 0
    seconds = span / fps
    count = max(2, min(max_frames, round(seconds * out_fps)))
    shot(page, cid, anim, 0, az)
    frames = [shot(page, cid, anim, span * i / count, az, wait=0) for i in range(count)]
    frames += [frames[-1]] * hold_last
    frames = [im.quantize(colors=128, method=2) for im in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(seconds / count * 1000), loop=0, disposal=2)
    return out, n, fps, seconds
