"""
Generate a spatial cluster map of graffiti photo locations in Salamanca.

GPS coordinates are extracted from EXIF data, spatially clustered with DBSCAN,
and plotted on an OpenStreetMap basemap via contextily. A small random jitter
(~15 m) is applied to every point before plotting to avoid pinpointing exact
shooting locations.

Output: a PNG figure suitable for inclusion in the thesis.

Usage (from src/):
    uv run python -m src.scripts.map_locations \
        --dataset "/opt/grafiti3/Takeout/Google Fotos" \
        --output plots/mapa_ubicaciones.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyproj import Transformer


# Salamanca bounding box (WGS84) — filters out stray GPS readings
SALAMANCA_LAT = (40.85, 41.10)
SALAMANCA_LON = (-6.00, -5.30)

# Privacy jitter: uniform noise ±JITTER_M metres
JITTER_M = 15


def dms_to_dd(dms, ref: str) -> float:
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    dd = d + m / 60 + s / 3600
    return -dd if ref in ("S", "W") else dd


def extract_gps(image_path: str, filter_salamanca: bool = True) -> tuple[float, float] | None:
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
        if not exif:
            return None
        gps = exif.get(34853)
        if not gps:
            return None
        lat = dms_to_dd(gps[2], gps[1])
        lon = dms_to_dd(gps[4], gps[3])
        if filter_salamanca:
            if not (SALAMANCA_LAT[0] <= lat <= SALAMANCA_LAT[1]):
                return None
            if not (SALAMANCA_LON[0] <= lon <= SALAMANCA_LON[1]):
                return None
        return lat, lon
    except Exception:
        return None


def collect_coords(dataset_roots: list[str], filter_salamanca: bool = True) -> tuple[np.ndarray, np.ndarray]:
    lats, lons = [], []
    for root in dataset_roots:
        for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
            result = extract_gps(p, filter_salamanca=filter_salamanca)
            if result:
                lats.append(result[0])
                lons.append(result[1])
    return np.array(lats), np.array(lons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Salamanca graffiti location map")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["/opt/grafiti3/Takeout/Google Fotos"],
        help="One or more root folders to scan for images",
    )
    parser.add_argument(
        "--output",
        default="plots/mapa_ubicaciones.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--jitter", type=float, default=JITTER_M,
        help="Privacy jitter radius in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Figure DPI"
    )
    parser.add_argument(
        "--no-location-filter",
        action="store_true",
        help="Disable Salamanca bounding box filter and include all GPS coordinates",
    )
    args = parser.parse_args()

    filter_salamanca = not args.no_location_filter
    print("Extracting GPS coordinates…", flush=True)
    lats, lons = collect_coords(args.dataset, filter_salamanca=filter_salamanca)
    if len(lats) == 0:
        sys.exit("No GPS coordinates found — check the dataset path.")
    location_label = "Salamanca " if filter_salamanca else ""
    print(f"  {len(lats)} images with valid {location_label}GPS")

    # --- WGS84 → Web Mercator (EPSG:3857) for contextily compatibility ---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(lons, lats)  # type: ignore[misc]

    # --- Privacy jitter in metres (applied in Web Mercator space) ---
    rng = np.random.default_rng(42)
    xs = xs + rng.uniform(-args.jitter, args.jitter, size=xs.shape)
    ys = ys + rng.uniform(-args.jitter, args.jitter, size=ys.shape)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 12))

    ax.scatter(xs, ys, c="#1f77b4", s=32, alpha=0.40, linewidths=0, zorder=3)

    # Basemap (OpenStreetMap)
    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.Esri.WorldTopoMap, zoom="auto")
    except Exception as e:
        print(f"  Warning: could not load basemap tiles ({e}). Figure saved without map background.")

    ax.set_axis_off()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
