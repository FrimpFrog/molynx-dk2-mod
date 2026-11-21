from pathlib import Path

from scripts.make_publish_page import strip_wrapper

SRC = Path("/home/claude/viewer/index.html")


def test_wrapper_tags_removed_but_header_and_footer_kept():
    out = strip_wrapper(SRC.read_text())
    for tag in ("<!doctype", "<html", "<head>", "</head>", "<body>", "</body>", "</html>", "<meta charset"):
        assert tag not in out.lower(), tag
    assert "<header>" in out and "</header>" in out and "<footer>" in out
    assert out.lstrip().startswith("<title>")


def test_every_other_element_is_unchanged():
    import re
    src = SRC.read_text()
    out = strip_wrapper(src)
    count = lambda s: {t: len(re.findall(rf"<{t}[\s>]", s)) for t in ("header", "footer", "nav", "main", "button", "script")}
    assert count(src) == count(out)
