"""The Molynx Test A bundle: one zip with the two mod folders and the guide.

`make_bundle(mods_dir, guide, out)` writes `out` holding exactly

  Molynx/...              the stage-2 folder (`mods_dir/Molynx`)
  Molynx_slot_only/...    the stage-1 folder (`mods_dir/Molynx_slot_only`)
  Molynx_test_guide.md    a byte copy of `guide`

deterministically: entries sorted by name, one fixed timestamp (TIMESTAMP), mode 0644, deflate
level 9, no directory entries, so the same input gives the same zip byte for byte. It refuses
(ValueError, nothing written) anything that does not belong in a mod folder: a file other than
`resource.toml` at a folder's top, a file outside `Data/`, a hidden or backup file (`.x`, `x~`), a
symbolic link, or a missing folder or `resource.toml`; it never overwrites `out`
(FileExistsError). It does not judge the contents: audit both folders first
(`python -m scripts.audit_molynx_mod <folder>`).

`python -m scripts.make_molynx_bundle --mods DIR --guide docs/molynx-test-guide.md --out ZIP`

LICENCE: GPL-3.0.
"""
import argparse
import sys
import zipfile
from pathlib import Path

FOLDERS = ("Molynx", "Molynx_slot_only")
GUIDE_NAME = "Molynx_test_guide.md"
TIMESTAMP = (2026, 11, 5, 0, 0, 0)
MODE = 0o644


def _folder_entries(folder: Path) -> list[tuple[str, Path]]:
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError(f"{folder} is not a folder")
    if not (folder / "resource.toml").is_file():
        raise ValueError(f"{folder} has no resource.toml")
    entries = []
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder)
        if path.is_symlink():
            raise ValueError(f"{folder.name}/{rel.as_posix()} is a symbolic link")
        if any(part.startswith(".") or part.endswith("~") for part in rel.parts):
            raise ValueError(f"{folder.name}/{rel.as_posix()} is a hidden or backup file")
        if path.is_dir():
            if rel.parts[0] != "Data":
                raise ValueError(f"{folder.name}/{rel.as_posix()} is a folder outside Data")
            continue
        if rel.as_posix() != "resource.toml" and rel.parts[0] != "Data":
            raise ValueError(f"{folder.name}/{rel.as_posix()} is a stray file")
        entries.append((f"{folder.name}/{rel.as_posix()}", path))
    return entries


def make_bundle(mods_dir: Path, guide: Path, out: Path) -> list[str]:
    """Write the bundle (see the module docstring); returns the entry names in zip order."""
    mods_dir, guide, out = Path(mods_dir), Path(guide), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; it is never overwritten")
    if not guide.is_file():
        raise ValueError(f"guide {guide} is not a file")
    entries = [(GUIDE_NAME, guide)]
    for name in FOLDERS:
        entries += _folder_entries(mods_dir / name)
    entries.sort()
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, path in entries:
            info = zipfile.ZipInfo(name, date_time=TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = MODE << 16
            bundle.writestr(info, path.read_bytes(), compresslevel=9)
    return [name for name, _ in entries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zip the Molynx Test A bundle (two mod folders and "
                                                 "the guide), deterministically.")
    parser.add_argument("--mods", type=Path, required=True,
                        help="folder holding Molynx and Molynx_slot_only")
    parser.add_argument("--guide", type=Path, required=True, help="the guide (Markdown)")
    parser.add_argument("--out", type=Path, required=True, help="the zip to write (must not exist)")
    args = parser.parse_args(argv)
    names = make_bundle(args.mods, args.guide, args.out)
    print(f"{args.out}: {len(names)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
