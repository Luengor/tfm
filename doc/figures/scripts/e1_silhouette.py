"""E1 — silhouette per extractor at N in {250, 1000, 6416}.

Highlights dinov2 base vs dinov2_graffiti_style_head and annotates CH on the
style-head bar at N=6416 to show the silhouette/CH inversion.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _style import EXTRACTOR_SHORT, HIGHLIGHT, load_runs, save, setup


def main() -> None:
    setup()
    rows = []
    for r in load_runs("exp01"):
        rows.append(
            {
                "extractor": r["embedding_type"],
                "N": int(r["limit_parameter"]),
                "silhouette": r["clustering_quality_silhouette"],
                "ch": r["clustering_quality_calinski_harabasz"],
            }
        )
    df = pd.DataFrame(rows)

    order = (
        df[df.N == 6416]
        .sort_values("silhouette", ascending=False)["extractor"]
        .tolist()
    )
    df["extractor"] = pd.Categorical(df["extractor"], categories=order, ordered=True)
    df["extractor_short"] = df["extractor"].map(EXTRACTOR_SHORT).astype(str)
    short_order = [EXTRACTOR_SHORT[e] for e in order]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    sns.barplot(
        data=df,
        x="extractor_short",
        y="silhouette",
        hue="N",
        order=short_order,
        ax=ax,
        edgecolor="white",
        linewidth=0.4,
    )

    # Highlight dinov2 and style head bars at N=6416
    highlight = {
        "dinov2": "dinov2",
        "dinov2+style": "dinov2_graffiti_style_head",
    }
    for patch, (label, xt) in zip(
        [p for p in ax.patches if p.get_height() != 0][-len(short_order) :],
        zip(short_order, short_order),
    ):
        if xt in highlight:
            patch.set_edgecolor(HIGHLIGHT)
            patch.set_linewidth(1.4)

    # CH annotation on style head bar at N=6416
    style_row = df[(df.N == 6416) & (df.extractor == "dinov2_graffiti_style_head")]
    if not style_row.empty:
        x_idx = short_order.index("dinov2+style")
        ch = float(style_row["ch"].iloc[0])
        sil = float(style_row["silhouette"].iloc[0])
        ax.annotate(
            f"CH={ch:.1f}",
            xy=(x_idx + 0.25, sil),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=HIGHLIGHT,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=HIGHLIGHT, lw=0.8),
        )

    ax.set_xlabel("")
    ax.set_ylabel("Silueta")
    ax.set_title("E1 · Silueta por extractor a tres tamaños de banco")
    ax.legend(title="N", loc="upper right", ncol=3)
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")

    save(fig, "e1_silhouette_by_embedding")


if __name__ == "__main__":
    main()
