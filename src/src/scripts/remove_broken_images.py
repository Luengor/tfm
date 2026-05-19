"""Remove broken/unreadable image files from a dataset directory."""

import argparse
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def check_and_remove(directory: Path, dry_run: bool, extensions: set[str]) -> tuple[int, int]:
    files = [f for f in directory.rglob("*") if f.suffix.lower() in extensions]
    broken = []

    for path in tqdm(files, desc="Checking images", unit="img"):
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, SyntaxError):
            broken.append(path)

    for path in broken:
        print(f"{'[dry-run] ' if dry_run else ''}Removing: {path}")
        if not dry_run:
            path.unlink()

    return len(files), len(broken)


def main():
    parser = argparse.ArgumentParser(description="Remove broken images from a dataset directory.")
    parser.add_argument("directory", type=Path, help="Dataset directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Report broken files without deleting")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"],
        help="File extensions to check (default: common image formats)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}
    total, broken = check_and_remove(args.directory, args.dry_run, extensions)

    print(f"\nScanned {total} files. {'Found' if args.dry_run else 'Removed'} {broken} broken image(s).")


if __name__ == "__main__":
    main()
