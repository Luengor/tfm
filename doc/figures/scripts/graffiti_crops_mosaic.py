#!/usr/bin/env python3
"""Build a 2x3 (vertical) mosaic of graffiti crops.

Prefers near-square source photos so center-cropping to a square cell
throws away as little of the image as possible.

Usage (from repo root):
    python presen/figures/scripts/graffiti_crops_mosaic.py
    python presen/figures/scripts/graffiti_crops_mosaic.py --source data/style/train_crop --seed 11
"""
import argparse
import glob
import os
import random

from PIL import Image

# repo root = three levels up from this file (presen/figures/scripts/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def aspect_penalty(path):
    """0 for a perfect square; grows as the image gets more elongated."""
    with Image.open(path) as im:
        w, h = im.size
    return abs(w / h - 1.0)


def collect(source, max_penalty):
    imgs = glob.glob(os.path.join(source, "**", "*.jpg"), recursive=True)
    scored = [(aspect_penalty(p), p) for p in imgs]
    squareish = [p for pen, p in scored if pen <= max_penalty]
    # fall back to all images if the filter is too strict
    return squareish if len(squareish) >= 6 else [p for _, p in scored]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/style/train_crop",
                    help="folder with crops (searched recursively)")
    ap.add_argument("--out", default="presen/figures/graffiti_crops_mosaic.jpg")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cell", type=int, default=400, help="cell size in px")
    ap.add_argument("--gap", type=int, default=8, help="gap between cells in px")
    ap.add_argument("--max-penalty", type=float, default=0.35,
                    help="max |w/h - 1| for a photo to count as 'square-ish'")
    args = ap.parse_args()

    random.seed(args.seed)
    source = os.path.join(ROOT, args.source)
    out = os.path.join(ROOT, args.out)

    candidates = collect(source, args.max_penalty)
    if len(candidates) < 6:
        raise SystemExit(f"need >=6 images, found {len(candidates)} in {source}")
    picks = random.sample(candidates, 6)

    cols, rows, cell, gap = 2, 3, args.cell, args.gap
    W = cols * cell + (cols + 1) * gap
    H = rows * cell + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for i, p in enumerate(picks):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2,
                      (w - s) // 2 + s, (h - s) // 2 + s)).resize((cell, cell), Image.LANCZOS)
        r, c = divmod(i, cols)
        canvas.paste(im, (gap + c * (cell + gap), gap + r * (cell + gap)))

    canvas.save(out, quality=92)
    print("saved", out, canvas.size)
    for p in picks:
        print("  ", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
