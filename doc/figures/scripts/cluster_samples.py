"""3×3 image grid showing sample crops from each of the three style clusters."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from _style import FIG_DIR, save, setup

CLUSTERS_DIR = FIG_DIR / "clusters"

CLUSTERS = [
    ("Personajes", "characters", ["c1.png", "c2.png", "c3.png"]),
    ("Elaborado",  "elaborate",  ["e1.png", "e2.png", "e3.png"]),
    ("Simple",     "simple",     ["s1.png", "s2.png", "s3.png"]),
]


def main() -> None:
    setup()

    n_rows = len(CLUSTERS)
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5, 5.0),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    for row_idx, (label, subdir, files) in enumerate(CLUSTERS):
        for col_idx, fname in enumerate(files):
            ax = axes[row_idx][col_idx]
            img_path = CLUSTERS_DIR / subdir / fname
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold", rotation=90,
                              labelpad=6, va="center")
                ax.yaxis.set_label_position("left")
                ax.yaxis.set_label_coords(-0.08, 0.5)

    fig.suptitle("Muestras representativas por clúster", fontsize=10, fontweight="bold", y=1.01)

    save(fig, "cluster_samples")


if __name__ == "__main__":
    main()
