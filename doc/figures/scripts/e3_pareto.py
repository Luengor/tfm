"""E3 — clustering families at N=6416, quality vs cost scatter.

One point per family at its best-silhouette variant (big labeled marker);
all other variants shown as small translucent dots in the same family color.
X axis is clustering wall-time (log scale) so the 3-orders-of-magnitude span
across families is legible. HDBSCAN base configuration highlighted in red.
"""
from __future__ import annotations

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

COST_FLOOR = 1e-3


def main() -> None:
    setup()

    rows = []
    for r in load_runs("exp03"):
        if int(r["limit_parameter"]) != 6416:
            continue
        ct = r["clustering_type"]
        if ct not in FAMILY_LABEL:
            continue
        sil = r["clustering_quality_silhouette"]
        clu_t = r["clustering_wall_time_s"]
        if sil is None or clu_t is None:
            continue
        rows.append({"family": FAMILY_LABEL[ct], "silhouette": sil, "cost": clu_t})
    raw = pd.DataFrame(rows)
    raw["cost_plot"] = raw["cost"].clip(lower=COST_FLOOR)

    best_idx = raw.groupby("family")["silhouette"].idxmax()
    best = raw.loc[best_idx].reset_index(drop=True)

    palette = dict(
        zip(FAMILY_ORDER, sns.color_palette("colorblind", n_colors=len(FAMILY_ORDER)))
    )

    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    for fam, sub in raw.groupby("family"):
        if len(sub) > 1:
            ax.scatter(
                sub["cost_plot"],
                sub["silhouette"],
                s=28,
                color=palette[fam],
                alpha=0.35,
                edgecolor="none",
                zorder=2,
            )

    for row in best.itertuples():
        is_base = row.family == "HDBSCAN"
        ax.scatter(
            row.cost_plot,
            row.silhouette,
            s=170,
            color=palette[row.family],
            edgecolor=HIGHLIGHT if is_base else "white",
            linewidth=2.2 if is_base else 1.2,
            zorder=4,
        )

    pareto = best.sort_values("cost_plot").reset_index(drop=True)
    front_x, front_y = [], []
    best_sil = -float("inf")
    for row in pareto.itertuples():
        if row.silhouette > best_sil:
            front_x.append(row.cost_plot)
            front_y.append(row.silhouette)
            best_sil = row.silhouette
    ax.plot(
        front_x,
        front_y,
        color="0.4",
        lw=1.0,
        ls="--",
        zorder=3,
        label="Frente de Pareto",
    )

    for row in best.itertuples():
        ax.annotate(
            row.family,
            xy=(row.cost_plot, row.silhouette),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=HIGHLIGHT if row.family == "HDBSCAN" else "#222222",
            zorder=5,
        )

    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Coste (s, escala log) — tiempo de agrupamiento")
    ax.set_ylabel("Silueta (mejor variante por familia)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")

    save(fig, "e3_pareto_quality_cost")


if __name__ == "__main__":
    main()
