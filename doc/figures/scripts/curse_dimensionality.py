"""Curse of dimensionality — distance concentration (Aggarwal et al., 2001).

Synthetic illustration for the dimensionality-reduction slide: as the dimension
grows, the relative contrast (d_max - d_min) / d_min between the nearest and
farthest points collapses toward zero, so "nearest neighbour" stops being
meaningful. Pure synthetic data, no benchmark inputs.
"""
from __future__ import annotations

import numpy as np

from _style import HIGHLIGHT, PALETTE, save, setup

import matplotlib.pyplot as plt

DIMS = np.unique(np.round(np.logspace(0, 3, 30)).astype(int))  # 1 .. 1000
N_POINTS = 1000
N_QUERIES = 50
RNG = np.random.default_rng(0)


def relative_contrast(dim: int) -> float:
    """Mean over queries of (d_max - d_min) / d_min for uniform points in [0,1]^dim."""
    contrasts = []
    for _ in range(N_QUERIES):
        pts = RNG.random((N_POINTS, dim))
        q = RNG.random(dim)
        d = np.linalg.norm(pts - q, axis=1)
        dmin = d.min()
        if dmin > 0:
            contrasts.append((d.max() - dmin) / dmin)
    return float(np.mean(contrasts))


def main() -> None:
    setup()
    contrast = np.array([relative_contrast(int(d)) for d in DIMS])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(DIMS, contrast, color=PALETTE[0], lw=1.6, zorder=2)
    ax.scatter(DIMS, contrast, color=PALETTE[0], s=18, zorder=3)

    # Highlight a representative high-dimensional embedding size.
    ref = int(DIMS[np.argmin(np.abs(DIMS - 384))])
    ref_val = contrast[np.where(DIMS == ref)[0][0]]
    ax.scatter([ref], [ref_val], color=HIGHLIGHT, s=60, zorder=4)
    ax.annotate(
        f"d={ref}\n(p. ej. DINOv2)",
        xy=(ref, ref_val),
        xytext=(6, 14),
        textcoords="offset points",
        fontsize=7,
        color=HIGHLIGHT,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimensión")
    ax.set_ylabel(r"Contraste relativo $(d_{max}-d_{min})/d_{min}$")
    ax.set_title("Concentración de distancias al crecer la dimensión")

    save(fig, "curse_dimensionality")


if __name__ == "__main__":
    main()
