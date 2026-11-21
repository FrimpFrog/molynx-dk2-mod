"""Strip the document wrapper from viewer/index.html for hosts that add their own (writes a sibling file).

Usage: python3 -m scripts.make_publish_page viewer/index.html viewer/_publish_index.html
"""
import re
import sys
from pathlib import Path


def strip_wrapper(html: str) -> str:
    html = re.sub(r"<!doctype html>\s*", "", html, flags=re.I)
    html = re.sub(r"</?(?:html|head|body)(?:\s[^>]*)?>\s*", "", html)   # not <header>/<hr>
    html = re.sub(r'<meta charset[^>]*>\s*', "", html)
    html = re.sub(r'<meta name="viewport"[^>]*>\s*', "", html)
    return html


if __name__ == "__main__":
    src, dst = map(Path, sys.argv[1:3])
    dst.write_text(strip_wrapper(src.read_text()))
