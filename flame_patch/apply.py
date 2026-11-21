"""Lay the Molynx patches over a Flame source tree (the flame-build workflow runs this).

Usage: python apply.py <flame source dir>
Copies flame_patch/src/** into <flame>/src, adds the new .cpp files to src/CMakeLists.txt and
calls each patch's init() at the end of patch::flameInit (src/patches/flame_main.cpp).
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHES = ["molynx_creatures"]          # src/patches/<name>.cpp with patch::<name>::init()


def insert_after(text: str, anchor: str, new: str, path: Path) -> str:
    if anchor not in text:
        sys.exit(f"apply.py: {path} has no {anchor!r}; is this Flame f6e2a3d?")
    if new in text:
        return text
    return text.replace(anchor, anchor + new, 1)


def main(flame: Path) -> None:
    src = flame / "src"
    for f in (HERE / "src").rglob("*"):
        if f.is_file():
            out = src / f.relative_to(HERE / "src")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out)
            print("copied", out.relative_to(flame))
    cmake = src / "CMakeLists.txt"
    text = cmake.read_text()
    for name in PATCHES:
        text = insert_after(text, "        patches/alternative_resources.cpp\n",
                            f"        patches/{name}.cpp\n", cmake)
    cmake.write_text(text)
    main_cpp = src / "patches" / "flame_main.cpp"
    text = main_cpp.read_text()
    for name in PATCHES:
        text = insert_after(text, '#include "alternative_resources.h"\n',
                            f'#include "patches/{name}.h"\n', main_cpp)
        text = insert_after(text, "    patch::screen_resolution::init();\n",
                            f"    patch::{name}::init();\n", main_cpp)
    main_cpp.write_text(text)
    print("applied:", ", ".join(PATCHES))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
