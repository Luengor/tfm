"""E6+E9 — per-query latency and recall at N=6416.

Horizontal bar chart, sorted by latency (log x). Each bar = one backend/config.
Recall@5 annotated on the bar end. HNSW configurations grouped + anchor
highlighted. SQLite kept on the same log axis as the high-latency reference.
"""
from __future__ import annotations

import json as _json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _style import HIGHLIGHT, load_runs, save, setup


def per_query(t: float | None, n: int | None) -> float | None:
    if t is None or not n:
        return None
    return 1000.0 * t / n  # ms


ANCHOR = (16, 64, 40)  # m, ef_c, ef_s
COLORS = {
    "SQLite": "#777777",
    "pgvector exacto": sns.color_palette("colorblind")[1],
    "pgvector + HNSW": sns.color_palette("colorblind")[0],
}


def collect() -> pd.DataFrame:
    rows = []

    # E6 at N=6416: SQLite + pgvector exacto (HNSW dropped, duplicates anchor)
    for r in load_runs("exp06"):
        if int(r["limit_parameter"]) != 6416:
            continue
        sp = r.get("storage_params") or {}
        if isinstance(sp, str):
            sp = _json.loads(sp)
        if r["storage_type"] == "sqlite":
            backend, label = "SQLite", "SQLite (Python)"
        elif sp.get("hnsw"):
            continue
        else:
            backend, label = "pgvector exacto", "pgvector exacto"
        rows.append(
            {
                "backend": backend,
                "label": label,
                "recall": r.get("ann_recall_at_k") or 1.0,
                "lat_ms": per_query(
                    r["similarity_search_wall_time_s"], r["similarity_search_sample_n"]
                ),
                "is_anchor": False,
                "is_exact": backend != "pgvector + HNSW",
            }
        )

    # E9 HNSW sweep (exact baseline already covered by E6)
    for r in load_runs("exp09"):
        sp = r.get("storage_params") or {}
        if isinstance(sp, str):
            sp = _json.loads(sp)
        h = sp.get("hnsw")
        if not h:
            continue
        lat = per_query(
            r.get("similarity_search_wall_time_s"), r.get("similarity_search_sample_n")
        )
        recall = r.get("ann_recall_at_k")
        if lat is None or recall is None:
            continue
        m, efc, efs = int(h["m"]), int(h["ef_construction"]), int(h["ef_search"])
        is_anchor = (m, efc, efs) == ANCHOR
        tag = (
            "HNSW ancla (m=16, efc=64, efs=40)"
            if is_anchor
            else f"HNSW m={m}, efc={efc}, efs={efs}"
        )
        rows.append(
            {
                "backend": "pgvector + HNSW",
                "label": tag,
                "recall": recall,
                "lat_ms": lat,
                "is_anchor": is_anchor,
                "is_exact": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup()
    df = collect().dropna(subset=["lat_ms"]).sort_values("lat_ms").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    colors = [COLORS[b] for b in df["backend"]]
    bars = ax.barh(
        df["label"],
        df["lat_ms"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    # Highlight anchor
    for bar, anc in zip(bars, df["is_anchor"]):
        if anc:
            bar.set_edgecolor(HIGHLIGHT)
            bar.set_linewidth(2.0)

    ax.set_xscale("log")
    ax.set_xlabel("Latencia por consulta (ms, log)")
    ax.set_title("E6+E9 · Latencia y recall@5 de la búsqueda por similitud (N=6416)")
    ax.invert_yaxis()  # fastest at top

    # Recall annotation at bar end
    for bar, lat, recall, is_exact in zip(
        bars, df["lat_ms"], df["recall"], df["is_exact"]
    ):
        txt = "exacto" if is_exact else f"r@5 = {recall:.3f}"
        ax.annotate(
            f"  {lat:.2f} ms · {txt}",
            xy=(lat, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#222222",
        )

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[b])
        for b in ["SQLite", "pgvector exacto", "pgvector + HNSW"]
    ]
    ax.legend(
        handles,
        ["SQLite (Python)", "pgvector exacto", "pgvector + HNSW"],
        loc="center right",
    )

    # Extra room on right for annotations
    xmax = df["lat_ms"].max() * 6
    ax.set_xlim(df["lat_ms"].min() * 0.6, xmax)
    ax.grid(axis="x", which="both", alpha=0.3)

    save(fig, "e6_e9_pareto")


if __name__ == "__main__":
    main()
