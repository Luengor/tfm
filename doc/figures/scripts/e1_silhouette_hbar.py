"""E1 (Option B) — silhouette per extractor as a horizontal lollipop.

Presentation variant: tall 3:4 panel for the left column of the E1 results
slide. Extractors on the y axis sorted by silhouette at N=6416 (best on top).
Per extractor a thin stem spans the three N readings, with a coloured dot per N.
No title (the slide carries its own caption/footnote).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from _style import EXTRACTOR_SHORT, HIGHLIGHT, PALETTE, load_runs, save, setup

N_ORDER = [250, 1000, 6416]
# Distinct, visible colours per N (colorblind palette: green / orange / blue).
N_COLOR = {250: PALETTE[2], 1000: PALETTE[1], 6416: PALETTE[0]}
HIGHLIGHT_EXTRACTORS = {"dinov2_vits14", "dinov2_graffiti_style_head"}


def main() -> None:
    setup()
    rows = []
    for r in load_runs("exp01"):
        rows.append(
            {
                "extractor": r["embedding_type"],
                "N": int(r["limit_parameter"]),
                "silhouette": r["clustering_quality_silhouette"],
            }
        )
    df = pd.DataFrame(rows)

    # Sort extractors by silhouette at N=6416, best on top (reversed y axis).
    order = (
        df[df.N == 6416]
        .sort_values("silhouette", ascending=True)["extractor"]
        .tolist()
    )
    ypos = {e: i for i, e in enumerate(order)}

    fig, ax = plt.subplots(figsize=(4.5, 6.0))  # 3:4 tall panel for slide

    # Stems: span min→max silhouette across N per extractor.
    for extractor in order:
        g = df[df.extractor == extractor]
        y = ypos[extractor]
        lo, hi = g["silhouette"].min(), g["silhouette"].max()
        ax.plot([lo, hi], [y, y], color="0.8", lw=1.4, zorder=1)

    # Dots per (extractor, N), coloured by N.
    for n in N_ORDER:
        sub = df[df.N == n]
        ax.scatter(
            sub["silhouette"],
            [ypos[e] for e in sub["extractor"]],
            color=N_COLOR[n],
            s=42,
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
            label=f"N={n}",
        )

    ax.set_yticks(list(ypos.values()))
    labels = [EXTRACTOR_SHORT[e] for e in order]
    ax.set_yticklabels(labels)
    # Bold the two highlighted extractor labels.
    for tick, extractor in zip(ax.get_yticklabels(), order):
        if extractor in HIGHLIGHT_EXTRACTORS:
            tick.set_fontweight("bold")
            tick.set_color(HIGHLIGHT)

    ax.set_xlabel("Silueta")
    ax.set_ylabel("")
    ax.legend(title="", loc="lower right", ncol=1)
    ax.margins(x=0.08)
    ax.grid(axis="y", visible=False)

    save(fig, "e1_silhouette_hbar")


if __name__ == "__main__":
    main()
