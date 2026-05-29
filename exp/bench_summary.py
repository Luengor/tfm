#!/usr/bin/env python3
"""
bench_summary.py — Markdown table summaries of benchmark JSON outputs.

Usage examples:
  # Quality table (default)
  python bench_summary.py exp04/output/*.json

  # Cost table
  python bench_summary.py exp04/output/*.json --view cost

  # All views stacked
  python bench_summary.py exp04/output/*.json --view all

  # Pivot by a param (e.g., min_cluster_size) with limit_parameter as row key
  python bench_summary.py exp04/output/*.json --pivot clustering_params.min_cluster_size --row limit_parameter

  # Add extra columns from params
  python bench_summary.py exp05/output/*.json --extra reduction_params.n_components

  # Highlight best values per column
  python bench_summary.py exp04/output/*.json --highlight

  # Filter to specific embedding type
  python bench_summary.py exp01/output/*.json --filter embedding_type=dinov2_graffiti_style_head

  # CSV output
  python bench_summary.py exp04/output/*.json --csv
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Column definitions: (header, key, fmt, higher_is_better | None)
# fmt: None=str, f1/f2/f3=float dp, pct=percentage, s=seconds, mb=megabytes
# higher_is_better: True=bold max, False=bold min, None=no highlight
# ---------------------------------------------------------------------------

COLS_IDENTITY = [
    ("run", "run_name", None, None),
    ("status", "status", None, None),
    ("embed", "embedding_type", None, None),
    ("cluster", "clustering_type", None, None),
    ("reduce", "reduction_type", None, None),
    ("seg", "segmenter_type", None, None),
    ("n", "image_count", None, None),
    ("k", "cluster_count", None, None),
]

COLS_QUALITY = [
    ("sil", "clustering_quality_silhouette", "f3", True),
    ("silM", "clustering_quality_silhouette_macro", "f3", True),
    ("CH", "clustering_quality_calinski_harabasz", "f1", True),
    ("DB", "clustering_quality_davies_bouldin", "f3", False),
    ("noise", "clustering_quality_noise_ratio", "pct2", False),
    ("csCV", "clustering_quality_cluster_size_cv", "f3", False),
]

COLS_COST = [
    ("ingest_s", "ingest_wall_time_s", "s", False),
    ("ingest_MB", "ingest_peak_rss_delta_mb", "mb", False),
    ("ingest_ips", "ingest_throughput_ips", "f1", True),
    ("red_s", "reduction_wall_time_s", "s", False),
    ("red_MB", "reduction_peak_rss_delta_mb", "mb", False),
    ("clu_s", "clustering_wall_time_s", "s", False),
    ("clu_MB", "clustering_peak_rss_delta_mb", "mb", False),
    ("ss_s", "similarity_search_wall_time_s", "s", False),
    ("ss_ips", "similarity_search_throughput_ips", "f1", True),
    ("ann_recall", "ann_recall_at_k", "f3", True),
]

COLS_EXTRINSIC = [
    ("ARI", "extrinsic_ari", "f3", True),
    ("NMI", "extrinsic_nmi", "f3", True),
    ("F1", "extrinsic_pairwise_f1", "f3", True),
    ("ARI_nn", "extrinsic_ari_no_noise", "f3", True),
    ("NMI_nn", "extrinsic_nmi_no_noise", "f3", True),
    ("F1_nn", "extrinsic_pairwise_f1_no_noise", "f3", True),
    ("cov", "extrinsic_coverage", "pct2", True),
    ("n_match", "extrinsic_n_matched", None, None),
]

COLS_EXTRINSIC_AUTHOR = [
    ("ARIa", "extrinsic_author_ari", "f3", True),
    ("NMIa", "extrinsic_author_nmi", "f3", True),
    ("F1a", "extrinsic_author_pairwise_f1", "f3", True),
]

COLS_SIMILARITY = [
    ("P@k", "similarity_extrinsic_precision_at_k", "f3", True),
    ("R@k", "similarity_extrinsic_recall_at_k", "f3", True),
    ("MAP@k", "similarity_extrinsic_map_at_k", "f3", True),
    ("MRR", "similarity_extrinsic_mrr", "f3", True),
]

COLS_SIMILARITY_AUTHOR = [
    ("P@ka", "similarity_extrinsic_author_precision_at_k", "f3", True),
    ("MAPa", "similarity_extrinsic_author_map_at_k", "f3", True),
]

VIEW_MAP = {
    "identity": COLS_IDENTITY,
    "quality": COLS_QUALITY,
    "cost": COLS_COST,
    "extrinsic": COLS_EXTRINSIC,
    "extrinsic_author": COLS_EXTRINSIC_AUTHOR,
    "similarity": COLS_SIMILARITY,
    "similarity_author": COLS_SIMILARITY_AUTHOR,
}


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------

def _fmt(val, fmt: str | None) -> str:
    if val is None:
        return "—"
    if fmt is None:
        return str(val)
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if fmt == "f1":
        return f"{v:.1f}"
    if fmt == "f2":
        return f"{v:.2f}"
    if fmt == "f3":
        return f"{v:.3f}"
    if fmt == "f4":
        return f"{v:.4f}"
    if fmt == "pct":
        return f"{v*100:.1f}%"
    if fmt == "pct2":
        return f"{v*100:.3f}"  # raw fraction to 3 dp like 0.063 in reports
    if fmt == "s":
        if v >= 3600:
            return f"{v/3600:.1f}h"
        if v >= 60:
            return f"{v/60:.1f}m"
        return f"{v:.1f}s"
    if fmt == "mb":
        return f"{v:.1f}"
    return str(val)


# ---------------------------------------------------------------------------
# Param extraction: "clustering_params.min_cluster_size" → value from record
# ---------------------------------------------------------------------------

def _extract(record: dict, path: str):
    """Extract a value by dotted path; handles JSON-string sub-dicts."""
    parts = path.split(".", 1)
    top = record.get(parts[0])
    if len(parts) == 1:
        return top
    # sub-key from a JSON-string field
    if isinstance(top, str):
        try:
            top = json.loads(top)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(top, dict):
        return top.get(parts[1])
    return None


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def _load_records(paths: list[str]) -> list[dict]:
    records = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict) and "results" in data:
            records.extend(data["results"])
        else:
            records.append(data)
    return records


def _apply_filter(records: list[dict], filters: list[str]) -> list[dict]:
    """Filter by key=value pairs; supports dotted param paths."""
    for f in filters:
        if "=" not in f:
            print(f"Warning: ignoring malformed filter {f!r} (expected key=value)", file=sys.stderr)
            continue
        k, v = f.split("=", 1)
        records = [r for r in records if str(_extract(r, k)) == v]
    return records


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

def _has_any(records: list[dict], keys: list[str]) -> bool:
    return any(_extract(r, k) is not None for r in records for k in keys)


def _build_col_list(views: list[str], records: list[dict], extra: list[str]) -> list[tuple]:
    """Build the final column list for the chosen views."""
    cols = []

    # Always include a minimal identity block
    if "all" in views or "identity" in views:
        cols += COLS_IDENTITY
    else:
        # Minimal: run name + key config fields
        cols += [
            ("run", "run_name", None, None),
            ("embed", "embedding_type", None, None),
            ("cluster", "clustering_type", None, None),
            ("n", "image_count", None, None),
            ("k", "cluster_count", None, None),
        ]

    if "all" in views or "quality" in views:
        cols += COLS_QUALITY
    if "all" in views or "cost" in views:
        cols += COLS_COST
    if "all" in views or "extrinsic" in views:
        if _has_any(records, ["extrinsic_ari", "extrinsic_nmi"]):
            cols += COLS_EXTRINSIC
    if "all" in views or "extrinsic_author" in views:
        if _has_any(records, ["extrinsic_author_ari"]):
            cols += COLS_EXTRINSIC_AUTHOR
    if "all" in views or "similarity" in views:
        if _has_any(records, ["similarity_extrinsic_precision_at_k"]):
            cols += COLS_SIMILARITY
    if "all" in views or "similarity_author" in views:
        if _has_any(records, ["similarity_extrinsic_author_precision_at_k"]):
            cols += COLS_SIMILARITY_AUTHOR

    # Extra columns from dotted param paths
    for path in extra:
        header = path.split(".")[-1]
        cols.append((header, path, None, None))

    return cols


# ---------------------------------------------------------------------------
# Highlight: bold best value per column (among non-None)
# ---------------------------------------------------------------------------

def _best_values(records: list[dict], cols: list[tuple]) -> dict[str, float]:
    """Return {key: best_numeric_value} for highlight-eligible columns."""
    best = {}
    for header, key, fmt, hib in cols:
        if hib is None:
            continue
        vals = []
        for r in records:
            v = _extract(r, key)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if not vals:
            continue
        best[key] = max(vals) if hib else min(vals)
    return best


# ---------------------------------------------------------------------------
# Pivot table: rows = unique values of row_field, cols = unique values of
# pivot_field, cells = a quality/cost value.  Simplified: produces a
# record-per-unique-(row_val, pivot_val) combination.
# ---------------------------------------------------------------------------

def _pivot_records(records: list[dict], pivot_field: str, row_field: str) -> list[dict]:
    """
    Annotate each record with extracted pivot and row fields as top-level keys
    named '__pivot__' and '__row__', and sort by (row_val, pivot_val).
    """
    out = []
    for r in records:
        nr = dict(r)
        nr["__pivot__"] = _extract(r, pivot_field)
        nr["__row__"] = _extract(r, row_field)
        out.append(nr)
    out.sort(key=lambda r: (
        _sort_key_val(r.get("__row__")),
        _sort_key_val(r.get("__pivot__")),
    ))
    return out


def _sort_key_val(v):
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


# ---------------------------------------------------------------------------
# Markdown table rendering
# ---------------------------------------------------------------------------

def _md_table(records: list[dict], cols: list[tuple], highlight: bool, pivot: str | None, row: str | None) -> str:
    best = _best_values(records, cols) if highlight else {}

    effective_cols = list(cols)
    if pivot:
        effective_cols = [("pivot→" + pivot.split(".")[-1], "__pivot__", None, None),
                          ("↓" + (row or "").split(".")[-1], "__row__", None, None)] + cols
    elif row:
        effective_cols = [("↓" + row.split(".")[-1], "__row__", None, None)] + cols

    # Build header
    headers = [c[0] for c in effective_cols]
    alignments = []
    for _, key, fmt, _ in effective_cols:
        if fmt in ("f1", "f2", "f3", "f4", "pct", "pct2", "s", "mb") or key in (
            "image_count", "cluster_count", "ingest_peak_rss_delta_mb",
            "reduction_peak_rss_delta_mb", "clustering_peak_rss_delta_mb",
        ):
            alignments.append("--:")
        else:
            alignments.append("---")

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(alignments) + "|")

    for r in records:
        if r.get("status") != "success":
            continue
        cells = []
        for header, key, fmt, hib in effective_cols:
            v = _extract(r, key)
            cell = _fmt(v, fmt)
            if highlight and hib is not None and v is not None:
                try:
                    fv = float(v)
                    if key in best and abs(fv - best[key]) < 1e-9:
                        cell = f"**{cell}**"
                except (TypeError, ValueError):
                    pass
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    # Failed rows at the bottom
    failed = [r for r in records if r.get("status") != "success"]
    if failed:
        lines.append("")
        lines.append(f"† {len(failed)} failed run(s): " +
                     ", ".join(r.get("run_name", "?") for r in failed))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV rendering
# ---------------------------------------------------------------------------

def _csv_table(records: list[dict], cols: list[tuple]) -> str:
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([c[0] for c in cols])
    for r in records:
        w.writerow([_fmt(_extract(r, c[1]), c[2]) for c in cols])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Print benchmark results as Markdown tables for report writing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("files", nargs="+", help="benchmark JSON file(s)")
    p.add_argument(
        "--view", nargs="+",
        choices=["quality", "cost", "extrinsic", "extrinsic_author",
                 "similarity", "similarity_author", "identity", "all"],
        default=["quality"],
        help="Which metric groups to include (default: quality).",
    )
    p.add_argument(
        "--sort", default=None,
        help="Dotted field path to sort by (descending). E.g. clustering_quality_silhouette.",
    )
    p.add_argument(
        "--sort-asc", action="store_true",
        help="Sort ascending instead of descending.",
    )
    p.add_argument(
        "--filter", nargs="*", default=[],
        metavar="KEY=VAL",
        help="Keep only records where KEY equals VAL. Supports dotted param paths.",
    )
    p.add_argument(
        "--filter-status", choices=["success", "failed", "all"], default="success",
        help="Status filter (default: success).",
    )
    p.add_argument(
        "--extra", nargs="*", default=[],
        metavar="PATH",
        help="Extra columns from dotted param paths, e.g. clustering_params.min_cluster_size.",
    )
    p.add_argument(
        "--pivot", default=None,
        metavar="PATH",
        help="Add a pivot column from this dotted param path (e.g. clustering_params.min_cluster_size).",
    )
    p.add_argument(
        "--row", default=None,
        metavar="PATH",
        help="Field to use as the row grouping key when --pivot is set.",
    )
    p.add_argument(
        "--highlight", action="store_true",
        help="Bold the best value in each numeric column.",
    )
    p.add_argument(
        "--no-identity", action="store_true",
        help="Drop the identity columns (run name, embed, etc.) from the table.",
    )
    p.add_argument(
        "--csv", action="store_true",
        help="Output CSV instead of Markdown.",
    )
    args = p.parse_args()

    records = _load_records(args.files)

    if args.filter_status != "all":
        records = [r for r in records if r.get("status") == args.filter_status]

    if args.filter:
        records = _apply_filter(records, args.filter)

    if not records:
        print("No records after filtering.", file=sys.stderr)
        sys.exit(1)

    if args.pivot or args.row:
        pivot_field = args.pivot
        row_field = args.row or "run_name"
        records = _pivot_records(records, pivot_field or row_field, row_field)

    sort_field = args.sort
    if sort_field:
        def _sk(r):
            v = _extract(r, sort_field)
            try:
                return (0, float(v))
            except (TypeError, ValueError):
                return (1, str(v or ""))
        records = sorted(records, key=_sk, reverse=not args.sort_asc)
    else:
        # default: sort by silhouette desc, then run name
        def _default_sk(r):
            sil = r.get("clustering_quality_silhouette")
            return (-(sil if sil is not None else -999), r.get("run_name", ""))
        records = sorted(records, key=_default_sk)

    cols = _build_col_list(args.view, records, args.extra or [])
    if args.no_identity:
        identity_keys = {c[1] for c in COLS_IDENTITY}
        cols = [c for c in cols if c[1] not in identity_keys]

    print(f"<!-- {len(records)} run(s) from {len(args.files)} file(s) -->")
    print()
    if args.csv:
        print(_csv_table(records, cols))
    else:
        print(_md_table(records, cols, args.highlight, args.pivot, args.row))


if __name__ == "__main__":
    main()
