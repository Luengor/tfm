"""E7 — log-log wall time vs N per stage and per clustering algorithm.

Stages (ingest, UMAP reduction, similarity search) read from HDBSCAN run.
Clustering algorithms shown for HDBSCAN, KMeans, DBSCAN, OPTICS, Agglomerative.
Annotates regression slope per series; marks UMAP N=6416 anomaly.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _style import HIGHLIGHT, load_runs, save, setup

ALGORITHMS = ["hdbscan", "kmeans", "dbscan", "optics", "agglomerative"]
ALGO_LABEL = {
    "hdbscan": "HDBSCAN",
    "kmeans": "KMeans",
    "dbscan": "DBSCAN",
    "optics": "OPTICS",
    "agglomerative": "Aglomerativo",
}


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan")
    lx = np.log10(xs[mask])
    ly = np.log10(ys[mask])
    m, _ = np.polyfit(lx, ly, 1)
    return float(m)


def main() -> None:
    setup()
    runs = load_runs("exp07")
    rows = []
    for r in runs:
        rows.append(
            {
                "algo": r["clustering_type"],
                "N": int(r["limit_parameter"]),
                "ingest": r.get("ingest_wall_time_s"),
                "umap": r.get("reduction_wall_time_s"),
                "clustering": r.get("clustering_wall_time_s"),
                "search": r.get("similarity_search_wall_time_s"),
            }
        )
    df = pd.DataFrame(rows)

    fig, (ax_stages, ax_algo) = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=False)

    # === Panel A: pipeline stages (use HDBSCAN run for stages + search) ===
    base = df[df.algo == "hdbscan"].sort_values("N")
    stages_palette = sns.color_palette("colorblind", n_colors=3)

    for (col, label), color in zip(
        [("ingest", "Ingesta"), ("umap", "Reducción (UMAP)"), ("search", "Búsqueda (SQLite)")],
        stages_palette,
    ):
        sub = base.dropna(subset=[col])
        ax_stages.plot(
            sub["N"], sub[col], marker="o", linestyle="-", label=label, color=color
        )
        # slope on N>=500
        sub2 = sub[sub["N"] >= 500]
        s = _slope(sub2["N"].to_numpy(), sub2[col].to_numpy())
        ax_stages.annotate(
            f"  m≈{s:.2f}",
            xy=(sub["N"].iloc[-1], sub[col].iloc[-1]),
            fontsize=7,
            color=color,
            va="center",
        )

    # Mark UMAP anomaly at N=6416
    anom = base[base.N == 6416]
    if not anom.empty:
        ax_stages.scatter(
            [6416], [anom["umap"].iloc[0]], s=120, facecolors="none",
            edgecolors=HIGHLIGHT, linewidths=1.4, zorder=5,
        )
        ax_stages.annotate(
            "pynndescent",
            xy=(6416, anom["umap"].iloc[0]),
            xytext=(-8, -18),
            textcoords="offset points",
            fontsize=7,
            color=HIGHLIGHT,
            ha="right",
        )

    ax_stages.set_xscale("log")
    ax_stages.set_yscale("log")
    ax_stages.set_xlabel("N (imágenes)")
    ax_stages.set_ylabel("Tiempo de pared (s)")
    ax_stages.set_title("A · Etapas del pipeline")
    ax_stages.legend(loc="upper left")

    # === Panel B: clustering algorithms ===
    algo_palette = sns.color_palette("colorblind", n_colors=len(ALGORITHMS))
    for algo, color in zip(ALGORITHMS, algo_palette):
        sub = df[df.algo == algo].sort_values("N").dropna(subset=["clustering"])
        if sub.empty:
            continue
        ax_algo.plot(
            sub["N"],
            sub["clustering"],
            marker="o",
            linestyle="-",
            label=ALGO_LABEL[algo],
            color=color,
        )
        sub2 = sub[sub["N"] >= 500]
        s = _slope(sub2["N"].to_numpy(), sub2["clustering"].to_numpy())
        ax_algo.annotate(
            f"  m≈{s:.2f}",
            xy=(sub["N"].iloc[-1], sub["clustering"].iloc[-1]),
            fontsize=7,
            color=color,
            va="center",
        )

    ax_algo.set_xscale("log")
    ax_algo.set_yscale("log")
    ax_algo.set_xlabel("N (imágenes)")
    ax_algo.set_ylabel("Tiempo de pared (s)")
    ax_algo.set_title("B · Algoritmos de agrupamiento")
    ax_algo.legend(loc="upper left")

    fig.suptitle("E7 · Escalabilidad temporal (log–log)", y=1.02)
    save(fig, "e7_scalability_loglog")


if __name__ == "__main__":
    main()
