"""E5 — 2×2 grid of YOLO detection examples from doc/figures/yolo/.

Source images: doc/figures/yolo/det_{0,1}_*.jpg.
Output: doc/figures/e5_yolo_detections.pdf — a single PDF combining the four
images in a 2×2 layout. Used to illustrate the variability of YOLO crops
discussed in the segmenter subsection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from _style import FIG_DIR, save, setup

YOLO_DIR = FIG_DIR / "yolo"


def main() -> None:
    setup()
    paths = sorted(YOLO_DIR.glob("det_*.jpg"))
    if len(paths) != 4:
        raise SystemExit(f"expected 4 images in {YOLO_DIR}, found {len(paths)}")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    for ax, path in zip(axes.flat, paths):
        ax.imshow(mpimg.imread(path))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)

    save(fig, "e5_yolo_detections")


if __name__ == "__main__":
    main()
