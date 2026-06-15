"""E3 — scaling with N per clustering family.

For each family, picks the best-silhouette variant at each N and plots three
panels: clustering wall-time (s), peak RSS delta (MB) and cluster count k,
all on log-log axes. Backs claims in the prose about cost and memory blowing
up for OPTICS / agglomerative while HDBSCAN stays cheap.
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _style import HIGHLIGHT, load_runs, save, setup

FAMILY_LABEL = {
    "hdbscan": "HDBSCAN",
    "optics": "OPTICS",
    "gmm": "GMM",
    "kmeans": "KMeans",
    "dbscan": "DBSCAN",
    "agglomerative": "Aglomerativo",
    "spectral": "Espectral",
}
FAMILY_ORDER = list(FAMILY_LABEL.values())

MEM_FLOOR = 0.1
COST_FLOOR = 1e-3


def main() -> None:
    setup()

    by_fam_n = defaultdict(list)
    for r in load_runs("exp03"):
        ct = r["clustering_type"]
        if ct not in FAMILY_LABEL:
            continue
        if r["clustering_quality_silhouette"] is None:
            continue
        by_fam_n[(ct, int(r["limit_parameter"]))].append(r)

    rows = []
    for (ct, N), runs in by_fam_n.items():
        best = max(runs, key=lambda x: x["clustering_quality_silhouette"])
        peak = best.get("clustering_peak_rss_delta_mb") or 0.0
        rows.append(
            {
                "family": FAMILY_LABEL[ct],
                "N": N,
                "cost": max(best["clustering_wall_time_s"], COST_FLOOR),
                "mem": max(peak, MEM_FLOOR),
                "k": best.get("cluster_count") or 0,
            }
        )
    df = pd.DataFrame(rows).sort_values(["family", "N"]).reset_index(drop=True)

    palette = dict(
        zip(FAMILY_ORDER, sns.color_palette("colorblind", n_colors=len(FAMILY_ORDER)))
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True)
    metrics = [
        ("cost", "Coste de agrupamiento (s)", axes[0]),
        ("mem", "Pico de RSS (MB)", axes[1]),
        ("k", "Número de clústeres", axes[2]),
    ]

    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam].sort_values("N")
        if sub.empty:
            continue
        is_base = fam == "HDBSCAN"
        color = palette[fam]
        lw = 2.2 if is_base else 1.3
        ms = 8 if is_base else 5
        zorder = 5 if is_base else 3
        for metric, _, ax in metrics:
            ax.plot(
                sub["N"],
                sub[metric],
                marker="o",
                markersize=ms,
                color=color,
                lw=lw,
                label=fam,
                zorder=zorder,
                markeredgecolor=HIGHLIGHT if is_base else "white",
                markeredgewidth=1.2 if is_base else 0.6,
            )

    for metric, ylabel, ax in metrics:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (imágenes)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].legend(loc="upper left", fontsize=7.5, ncol=2, columnspacing=0.8)
    fig.suptitle(
        "E3 · Escalado de las familias con N (mejor variante por silueta)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save(fig, "e3_scaling_with_n")


if __name__ == "__main__":
    main()
