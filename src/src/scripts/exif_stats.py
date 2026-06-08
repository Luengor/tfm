"""Aggregate EXIF statistics from a dataset of images.

Usage:
    uv run python -m src.scripts.exif_stats --dataset /opt/grafiti3
    uv run python -m src.scripts.exif_stats --dataset /opt/grafiti3 /other/path --csv exif_data.csv
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".heic", ".heif"}

TAGS_OF_INTEREST = {
    "Make",
    "Model",
    "Software",
    "DateTime",
    "DateTimeOriginal",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "FocalLength",
    "Flash",
    "WhiteBalance",
    "ExposureMode",
    "MeteringMode",
    "LensModel",
    "GPSInfo",
    "Orientation",
    "ResolutionUnit",
    "XResolution",
    "YResolution",
    "ExifImageWidth",
    "ExifImageHeight",
    "PixelXDimension",
    "PixelYDimension",
}

# Tag id -> name reverse map
TAG_ID = {v: k for k, v in TAGS.items()}


def rational_to_float(val) -> float | None:
    try:
        if hasattr(val, "numerator"):  # IFDRational
            return float(val)
        if isinstance(val, tuple) and len(val) == 2:
            return val[0] / val[1] if val[1] != 0 else None
        return float(val)
    except Exception:
        return None


def extract_exif(path: Path) -> dict:
    row = {"path": str(path), "width": None, "height": None}
    try:
        with Image.open(path) as img:
            row["width"], row["height"] = img.size
            row["format"] = img.format
            exif_data = img._getexif()  # noqa: SLF001
            if not exif_data:
                return row
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if tag_name not in TAGS_OF_INTEREST:
                    continue
                if tag_name == "GPSInfo":
                    row["has_gps"] = True
                    continue
                if tag_name in ("ExposureTime",):
                    row[tag_name] = rational_to_float(value)
                elif tag_name in ("FNumber", "FocalLength"):
                    row[tag_name] = rational_to_float(value)
                elif tag_name in ("XResolution", "YResolution"):
                    row[tag_name] = rational_to_float(value)
                elif isinstance(value, bytes):
                    row[tag_name] = value.decode("utf-8", errors="replace").strip("\x00")
                elif isinstance(value, (list, tuple)) and not isinstance(value, str):
                    row[tag_name] = str(value)
                else:
                    row[tag_name] = value
    except Exception as exc:
        row["error"] = str(exc)
    return row


def collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def print_counter(title: str, counter: Counter, top: int = 10) -> None:
    print(f"\n  {title}:")
    for val, count in counter.most_common(top):
        bar = "█" * min(30, int(30 * count / counter.most_common(1)[0][1]))
        print(f"    {str(val):<40s} {count:>6d}  {bar}")


def summarize(rows: list[dict]) -> None:
    total = len(rows)
    errors = sum(1 for r in rows if "error" in r)
    with_exif = sum(1 for r in rows if "Make" in r or "Model" in r)

    print(f"\n{'='*60}")
    print(f"  Dataset EXIF Summary")
    print(f"{'='*60}")
    print(f"  Total images  : {total}")
    print(f"  Parse errors  : {errors}")
    print(f"  With EXIF     : {with_exif} ({100*with_exif/total:.1f}%)")
    print(f"  With GPS      : {sum(1 for r in rows if r.get('has_gps'))}")

    cameras = Counter(
        f"{r.get('Make', '?').strip()} / {r.get('Model', '?').strip()}"
        for r in rows if r.get("Make") or r.get("Model")
    )
    print_counter("Camera (Make / Model)", cameras)

    resolutions = Counter(
        f"{r['width']}×{r['height']}"
        for r in rows if r.get("width") and r.get("height")
    )
    print_counter("Resolution (px)", resolutions)

    megapixels = [
        round(r["width"] * r["height"] / 1e6, 1)
        for r in rows if r.get("width") and r.get("height")
    ]
    if megapixels:
        avg_mp = sum(megapixels) / len(megapixels)
        min_mp, max_mp = min(megapixels), max(megapixels)
        print(f"\n  Megapixels    : avg {avg_mp:.1f} MP  |  min {min_mp:.1f}  max {max_mp:.1f}")

    formats = Counter(r.get("format") for r in rows if r.get("format"))
    print_counter("Format", formats, top=5)

    # Year from DateTimeOriginal or DateTime (format: "YYYY:MM:DD HH:MM:SS")
    years: Counter = Counter()
    for r in rows:
        dt = r.get("DateTimeOriginal") or r.get("DateTime")
        if dt and isinstance(dt, str) and len(dt) >= 4:
            years[dt[:4]] += 1
    if years:
        print_counter("Year (DateTimeOriginal)", years, top=10)

    isos = [r["ISOSpeedRatings"] for r in rows if r.get("ISOSpeedRatings")]
    if isos:
        numeric = [x for x in isos if isinstance(x, (int, float))]
        if numeric:
            print(f"\n  ISO           : avg {sum(numeric)/len(numeric):.0f}  |  min {min(numeric)}  max {max(numeric)}")

    fnumbers = [r["FNumber"] for r in rows if r.get("FNumber") is not None]
    if fnumbers:
        print(f"  f-number      : avg f/{sum(fnumbers)/len(fnumbers):.1f}  |  min f/{min(fnumbers):.1f}  max f/{max(fnumbers):.1f}")

    focal = [r["FocalLength"] for r in rows if r.get("FocalLength") is not None]
    if focal:
        print(f"  Focal length  : avg {sum(focal)/len(focal):.1f} mm  |  min {min(focal):.1f}  max {max(focal):.1f}")

    software = Counter(str(r.get("Software", "")).strip() for r in rows if r.get("Software"))
    if software:
        print_counter("Software / Processing", software, top=5)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EXIF data from an image dataset.")
    parser.add_argument("--dataset", required=True, nargs="+", help="Root directory/directories of images")
    parser.add_argument("--csv", default=None, help="Optional path to dump per-image CSV")
    parser.add_argument("--top", type=int, default=10, help="Top N values per category (default 10)")
    args = parser.parse_args()

    roots = [Path(d) for d in args.dataset]
    for root in roots:
        if not root.is_dir():
            print(f"Error: {root} is not a directory", file=sys.stderr)
            sys.exit(1)

    images = []
    for root in roots:
        found = collect_images(root)
        print(f"Found {len(found)} images under {root}")
        images.extend(found)

    rows = []
    for path in tqdm(images, desc="Reading EXIF", unit="img"):
        rows.append(extract_exif(path))

    summarize(rows)

    if args.csv:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-image CSV written to {args.csv}")


if __name__ == "__main__":
    main()
