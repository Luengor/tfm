"""E3 — DBSCAN sensibilidad a epsilon.

Visualización compacta de cómo la silueta y la fracción de ruido cambian con
epsilon en DBSCAN a N=6416. El barrido incluye tres valores fijos
(epsilon=0.2, 0.5, 1.0) más el automático. Apoya la afirmación en prosa de
que DBSCAN es inestable bajo este criterio.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from _style import load_runs, save, setup


def _params(r):
    p = r["clustering_params"]
    return json.loads(p) if isinstance(p, str) else p


def main() -> None:
    setup()

    rows = []
    for r in load_runs("exp03"):
        if r["clustering_type"] != "dbscan":
            continue
        if int(r["limit_parameter"]) != 6416:
            continue
        if r["clustering_quality_silhouette"] is None:
            continue
        eps = _params(r).get("eps")
        label = f"eps={eps}" if eps is not None else "auto"
        rows.append(
            {
                "label": label,
                "eps_num": eps if eps is not None else float("nan"),
                "sil": r["clustering_quality_silhouette"],
                "noise": r["clustering_quality_noise_ratio"] or 0.0,
                "k": r.get("cluster_count") or 0,
            }
        )
    df = pd.DataFrame(rows)

    df["sort_key"] = df["eps_num"].fillna(-1.0)
    df = df.sort_values("sort_key").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    x = range(len(df))
    bar_w = 0.38
    bars_sil = ax.bar(
        [i - bar_w / 2 for i in x],
        df["sil"],
        width=bar_w,
        color="#1f77b4",
        label="Silueta",
    )
    bars_noise = ax.bar(
        [i + bar_w / 2 for i in x],
        df["noise"],
        width=bar_w,
        color="#d62728",
        alpha=0.8,
        label="Fracción de ruido",
    )

    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Silueta / fracción de ruido")
    ax.set_title("E3 · Sensibilidad de DBSCAN a $\\varepsilon$ (N=6416)")
    ax.legend(loc="lower right", fontsize=8)

    for i, row in df.iterrows():
        ax.annotate(
            f"k={int(row['k'])}",
            xy=(i, max(row["sil"], row["noise"])),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#444",
        )

    ax.grid(axis="y", alpha=0.3)
    save(fig, "e3_dbscan_eps_sensitivity")


if __name__ == "__main__":
    main()
