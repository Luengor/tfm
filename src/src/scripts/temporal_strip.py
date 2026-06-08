"""
Cumulative line plot of photo capture dates from EXIF DateTimeOriginal.

Shows total images accumulated over time, making campaign bursts visible as
steep rises and idle periods as flat segments.

Output: PNG suitable for thesis inclusion.

Usage (from src/):
    uv run python -m src.scripts.temporal_strip \
        --dataset "/opt/grafiti3/Takeout/Google Fotos" \
        --output plots/temporal_distribution.png
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from PIL import Image


EXIF_DATETIME_TAG = 36867  # DateTimeOriginal
EXIF_DATETIME_FMT = "%Y:%m:%d %H:%M:%S"


def extract_datetime(image_path: str) -> datetime | None:
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
        if not exif:
            return None
        raw = exif.get(EXIF_DATETIME_TAG)
        if not raw:
            return None
        return datetime.strptime(raw, EXIF_DATETIME_FMT)
    except Exception:
        return None


def collect_dates(dataset_roots: list[str]) -> list[datetime]:
    dates = []
    for root in dataset_roots:
        for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
            dt = extract_datetime(p)
            if dt:
                dates.append(dt)
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal histogram of graffiti photos")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["/opt/grafiti3/Takeout/Google Fotos"],
        help="One or more root folders to scan for images",
    )
    parser.add_argument(
        "--output",
        default="plots/temporal_distribution.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Figure DPI"
    )
    parser.add_argument(
        "--min-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Drop images captured before this date",
    )
    args = parser.parse_args()

    min_date = datetime.strptime(args.min_date, "%Y-%m-%d") if args.min_date else None

    print("Extracting DateTimeOriginal from EXIF…", flush=True)
    dates = collect_dates(args.dataset)
    if not dates:
        sys.exit("No EXIF dates found — check dataset path.")
    if min_date:
        before = len(dates)
        dates = [d for d in dates if d >= min_date]
        print(f"  Dropped {before - len(dates)} images before {args.min_date}")
    print(f"  {len(dates)} images with valid DateTimeOriginal")

    dates_sorted = sorted(dates)
    dates_num = mdates.date2num(dates_sorted)
    cumulative = np.arange(1, len(dates_num) + 1)

    # Campaign bands — detected visually from the data; adjust if dataset changes
    campaigns = [
        (datetime(2022, 5, 21), datetime(2022, 7, 14), "Comienzo del dataset"),
        (datetime(2022, 9, 21), datetime(2022, 11, 7), "Primera campaña"),
        (datetime(2022, 12, 14), datetime(2023, 3, 21), "Segunda campaña"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.0))

    # Subtle horizontal grid behind everything
    ax.yaxis.grid(True, color="gray", alpha=0.25, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Campaign shading (labels added after ylim is known)
    for start, end, _label in campaigns:
        ax.axvspan(
            mdates.date2num(start), mdates.date2num(end),
            color="#1f77b4", alpha=0.08, zorder=1,
        )

    # Step line
    ax.plot(dates_num, cumulative, color="#1f77b4", linewidth=1.8,
            drawstyle="steps-post", zorder=3)

    # End-point annotation
    ax.scatter(dates_num[-1], cumulative[-1], color="#1f77b4", s=30, zorder=4)
    ax.annotate(
        f"{cumulative[-1]:,} imágenes",
        xy=(dates_num[-1], cumulative[-1]),
        xytext=(-8, -18), textcoords="offset points",
        ha="right", fontsize=8, color="#333333",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.set_ylabel("Fotografías acumuladas")
    ax.set_xlabel("Fecha de captura")
    ax.set_ylim(bottom=0, top=cumulative[-1] * 1.18)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title(
        f"Crecimiento acumulado del conjunto de datos ({len(dates)} imágenes)",
        fontsize=11,
    )

    # Re-draw campaign labels now that ylim is set
    for start, end, label in campaigns:
        mid = mdates.date2num(start) + (mdates.date2num(end) - mdates.date2num(start)) / 2
        ax.text(
            mid, ax.get_ylim()[1] * 0.95,
            label, ha="center", va="top", fontsize=8, color="#1f77b4",
            style="italic",
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
