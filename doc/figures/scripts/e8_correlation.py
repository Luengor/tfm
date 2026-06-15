"""E8 — silhouette vs ARI scatter, internal-external correlation.

Two panels (E8a styles | E8b authors) sharing the ARI axis, so the near-chance
ARI floor of the author experiment is visible against the style experiment that
spans up to ARI 0.72. Points are coloured by clusterer family (fixed-k
partitional vs HDBSCAN). Reproduces the Pearson r's quoted in the results
section: E8a r=+0.78 (n=48, fixed-k) / +0.55 (n=89, all); E8b r=-0.54 (n=89).

NOTE: this script does NOT use _style.load_runs(), whose merge key
(embedding, limit, reduction, clustering) collapses the HDBSCAN min_cluster_size
and agglomerative n_clusters variants into one cell. The correlation is a
cell-level quantity, so cells are merged by run_id (the per-config hash),
newest file wins — matching exp/exp08{a,b}/report.md (104/96 cells).

Produces e8_correlation.pdf.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _style import EXP_DIR, PALETTE, save, setup

FIXED_K = {"kmeans", "agglomerative", "spectral"}


def _merge_cells(exp: str) -> list[dict]:
    """All benchmark JSONs for an exp, merged by run_id, newest file wins."""
    out = EXP_DIR / exp / "output"
    files = sorted(out.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime)
    merged: dict[str, dict] = {}
    for path in files:
        for r in json.loads(path.read_text())["results"]:
            if r.get("status") == "success":
                merged[r["run_id"]] = r  # newer mtime overwrites
    return list(merged.values())


def _n_clusters(r: dict) -> int | None:
    cpr = r.get("clusters_per_repeat") or []
    return cpr[0] if cpr else None


def _cells(exp: str, ari_key: str) -> list[tuple[float, float, bool]]:
    """(silhouette, ARI, is_hdbscan) for cells with valid sil+ARI and >1 cluster."""
    pts = []
    for r in _merge_cells(exp):
        ari = r.get(ari_key)
        sil = r.get("clustering_quality_silhouette")
        if ari is None or sil is None:
            continue
        k = _n_clusters(r)
        if k is not None and k <= 1:
            continue
        pts.append((sil, ari, r["clustering_type"] == "hdbscan"))
    return pts


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _regline(ax, pairs, color, ls):
    """Draw the OLS fit y = a + b x over the x-range of the given (x, y) pairs."""
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    x0, x1 = min(xs), max(xs)
    ax.plot([x0, x1], [a + b * x0, a + b * x1], color=color, ls=ls, lw=1.6, zorder=3)


def _panel(ax, pts, title, annotate_fixed, draw_lines=True):
    sil = [p[0] for p in pts]
    ari = [p[1] for p in pts]
    fixed = [(s, a) for s, a, hb in pts if not hb]
    hdb = [(s, a) for s, a, hb in pts if hb]

    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=0)
    ax.scatter([s for s, _ in fixed], [a for _, a in fixed],
               c=[PALETTE[0]], s=24, linewidths=0, alpha=0.6, zorder=2)
    ax.scatter([s for s, _ in hdb], [a for _, a in hdb],
               c=[PALETTE[1]], s=24, linewidths=0, alpha=0.6, marker="^", zorder=2)

    if draw_lines:
        all_pairs = [(s, a) for s, a, _ in pts]
        _regline(ax, all_pairs, color="#444444", ls="--")
        if annotate_fixed and fixed:
            _regline(ax, fixed, color=PALETTE[0], ls="-")

    r_all = _pearson(sil, ari)
    lines = [f"$r$ (todas, $n={len(pts)}$) $= {r_all:+.2f}$"]
    if annotate_fixed and fixed:
        fs = [s for s, _ in fixed]
        fa = [a for _, a in fixed]
        lines.insert(0, f"$r$ ($k$ fijo, $n={len(fixed)}$) $= {_pearson(fs, fa):+.2f}$")
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=0.8))

    ax.set_title(title)
    ax.set_xlabel("Silueta")


def main() -> None:
    setup()
    a = _cells("exp08a", "extrinsic_ari")
    b = _cells("exp08b", "extrinsic_author_ari")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.2, 4.0), sharey=True)
    _panel(axa, a, "E8a · estilo (4 clases)", annotate_fixed=True)
    _panel(axb, b, "E8b · autoría (87 clases)", annotate_fixed=False, draw_lines=False)

    axa.set_ylabel("ARI")

    handles = [
        Line2D([], [], marker="o", ls="", color=PALETTE[0], label="$k$ fijo (KMeans/aglo/espectral)"),
        Line2D([], [], marker="^", ls="", color=PALETTE[1], label="HDBSCAN ($k$ libre)"),
        Line2D([], [], ls="-", color=PALETTE[0], label="ajuste $k$ fijo"),
        Line2D([], [], ls="--", color="#444444", label="ajuste todas las celdas"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "e8_correlation")


if __name__ == "__main__":
    main()
