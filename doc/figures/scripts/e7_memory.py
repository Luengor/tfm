"""E7 — peak RSS by stage vs N.

Panel A: peak clustering RSS per algorithm (agglomerative blows up; others flat).
Panel B: peak RSS of similarity search (HDBSCAN run is the only one with ss).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
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


def main() -> None:
    setup()
    rows = []
    for r in load_runs("exp07"):
        rows.append(
            {
                "algo": r["clustering_type"],
                "N": int(r["limit_parameter"]),
                "clu_peak": r.get("clustering_peak_rss_delta_mb"),
                "ss_peak": r.get("similarity_search_peak_rss_delta_mb"),
            }
        )
    df = pd.DataFrame(rows)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    palette = sns.color_palette("colorblind", n_colors=len(ALGORITHMS))
    for algo, color in zip(ALGORITHMS, palette):
        sub = df[df.algo == algo].sort_values("N").dropna(subset=["clu_peak"])
        if sub.empty:
            continue
        is_agglo = algo == "agglomerative"
        axA.plot(
            sub["N"],
            sub["clu_peak"],
            marker="o",
            label=ALGO_LABEL[algo],
            color=HIGHLIGHT if is_agglo else color,
            lw=2.0 if is_agglo else 1.2,
        )
    axA.set_xlabel("N (imágenes)")
    axA.set_ylabel("RSS pico (MB)")
    axA.set_title("A · Pico RSS del agrupamiento")
    axA.set_yscale("symlog", linthresh=1)
    axA.legend(loc="upper left")

    # Panel B: similarity search RSS — only HDBSCAN run has ss enabled
    ss = df[df.algo == "hdbscan"].sort_values("N").dropna(subset=["ss_peak"])
    axB.plot(
        ss["N"],
        ss["ss_peak"],
        marker="o",
        color=sns.color_palette("colorblind")[2],
    )
    axB.set_xlabel("N (imágenes)")
    axB.set_ylabel("RSS pico (MB)")
    axB.set_title("B · Pico RSS de la búsqueda por similitud (SQLite)")
    for _, row in ss.iterrows():
        axB.annotate(
            f"{row['ss_peak']:.1f}",
            xy=(row["N"], row["ss_peak"]),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=7,
            ha="center",
        )

    fig.suptitle("E7 · Memoria pico por etapa", y=1.02)
    save(fig, "e7_memory")


if __name__ == "__main__":
    main()
