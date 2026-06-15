"""1×3 image strip — cluster: Elaborado (elaborate)."""
from __future__ import annotations

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from _style import FIG_DIR, save, setup

CLUSTERS_DIR = FIG_DIR / "clusters"
LABEL = "Elaborado"
SUBDIR = "elaborate"
FILES = ["e1.png", "e2.png", "e3.png"]


def main() -> None:
    setup()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.5, 2.2),
        gridspec_kw={"wspace": 0.04},
    )

    for ax, fname in zip(axes, FILES):
        ax.imshow(mpimg.imread(CLUSTERS_DIR / SUBDIR / fname))
        ax.axis("off")

    save(fig, "cluster_elaborate")


if __name__ == "__main__":
    main()
