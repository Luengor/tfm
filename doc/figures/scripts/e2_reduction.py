"""E2 — reduction family: quality (panel A) and stage cost (panel B) at N=6416.

Identity, PCA-10, PCA-50, UMAP-10, UMAP-50, Isomap-10. Highlights how
identidad pushes cost into the clustering stage.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _style import HIGHLIGHT, load_runs, save, setup

LABELS = [
    ("identity", None, "identidad"),
    ("pca", 10, "PCA-10"),
    ("pca", 50, "PCA-50"),
    ("umap", 10, "UMAP-10"),
    ("umap", 50, "UMAP-50"),
    ("isomap", 10, "Isomap-10"),
]


def label_of(r: dict) -> str | None:
    p = r.get("reduction_params") or {}
    if isinstance(p, str):
        import json as _j

        p = _j.loads(p)
    n = p.get("n_components")
    rt = r["reduction_type"]
    for t, nc, lbl in LABELS:
        if t == rt and (nc is None or nc == n):
            return lbl
    return None


def main() -> None:
    setup()
    runs = [r for r in load_runs("exp02") if int(r["limit_parameter"]) == 6416]
    rows = []
    for r in runs:
        lbl = label_of(r)
        if lbl is None:
            continue
        rows.append(
            {
                "reduction": lbl,
                "silhouette": r["clustering_quality_silhouette"],
                "clusters": r["cluster_count"],
                "noise": r["clustering_quality_noise_ratio"],
                "red_t": r["reduction_wall_time_s"] or 0.0,
                "clu_t": r["clustering_wall_time_s"] or 0.0,
            }
        )
    df = pd.DataFrame(rows)
    order = [lbl for _, _, lbl in LABELS]
    df["reduction"] = pd.Categorical(df["reduction"], categories=order, ordered=True)
    df = df.sort_values("reduction")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.1))

    palette = sns.color_palette("colorblind", n_colors=len(order))
    color_map = {lbl: palette[i] for i, lbl in enumerate(order)}

    for _, row in df.iterrows():
        lbl = row["reduction"]
        color = HIGHLIGHT if lbl == "UMAP-10" else color_map[lbl]
        axA.scatter(
            row["noise"],
            row["silhouette"],
            color=color,
            s=70,
            zorder=3,
        )
        axA.annotate(
            f"{lbl}\nk={int(row['clusters'])}",
            xy=(row["noise"], row["silhouette"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
        )
    # Pareto front: non-dominated on (min noise, max silhouette)
    pareto = df[
        df.apply(
            lambda row: not any(
                (df["noise"] <= row["noise"])
                & (df["silhouette"] >= row["silhouette"])
                & ((df["noise"] < row["noise"]) | (df["silhouette"] > row["silhouette"]))
            ),
            axis=1,
        )
    ].sort_values("noise")
    axA.plot(
        pareto["noise"],
        pareto["silhouette"],
        color="0.5",
        lw=1.2,
        ls="--",
        zorder=1,
        label="Pareto",
    )

    axA.axhline(0, color="0.4", lw=0.6)
    axA.set_xlabel("Ruido (ratio)")
    axA.set_ylabel("Silueta")
    axA.set_title("A · Calidad: silueta vs. ruido")
    axA.set_xlim(-0.05, 1.05)
    ymin = min(-0.05, df["silhouette"].min() - 0.05)
    axA.set_ylim(ymin, df["silhouette"].max() + 0.07)

    # Panel B: stage cost
    long = df.melt(
        id_vars=["reduction"],
        value_vars=["red_t", "clu_t"],
        var_name="stage",
        value_name="seconds",
    )
    long["stage"] = long["stage"].map({"red_t": "reducción", "clu_t": "agrupamiento"})
    sns.barplot(
        data=long,
        y="reduction",
        x="seconds",
        hue="stage",
        ax=axB,
        order=order,
        edgecolor="white",
        linewidth=0.4,
    )
    axB.set_xlabel("Tiempo (s)")
    axB.set_ylabel("")
    axB.set_title("B · Coste por etapa, N=6416")
    axB.legend(loc="upper right")

    # Annotate totals
    totals = df.set_index("reduction").apply(lambda r: r["red_t"] + r["clu_t"], axis=1)
    for i, lbl in enumerate(order):
        t = totals.get(lbl, np.nan)
        if np.isfinite(t):
            axB.annotate(
                f"{t:.1f} s",
                xy=(t, i),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color="#333333",
            )

    save(fig, "e2_reduction_quality_cost")


if __name__ == "__main__":
    main()
