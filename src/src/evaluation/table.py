"""Display benchmark JSON results as a compact comparison table."""

import argparse
import json
import sys
from pathlib import Path


# Column definitions: (header, key, width, fmt)
# fmt: None = str, "f2" = 2dp float, "f3" = 3dp float, "f4" = 4dp float, "s" = seconds
_COLS_IDENTITY = [
    ("Run", "run_name", 28, None),
    ("Status", "status", 7, None),
    ("Embed", "embedding_type", 16, None),
    ("Cluster", "clustering_type", 13, None),
    ("Reduce", "reduction_type", 10, None),
    ("Seg", "segmenter_type", 8, None),
    ("N", "cluster_count", 4, None),
    ("Imgs", "image_count", 5, None),
    ("Sil", "clustering_quality_silhouette", 6, "f3"),
    ("CH", "clustering_quality_calinski_harabasz", 8, "f1"),
    ("DB", "clustering_quality_davies_bouldin", 6, "f3"),
    ("Noise%", "clustering_quality_noise_ratio", 6, "pct"),
    ("CV", "clustering_quality_cluster_size_cv", 5, "f2"),
    ("Ingest", "ingest_wall_time_s", 7, "s"),
    ("Clust", "clustering_wall_time_s", 6, "s"),
]

_COLS_EXTRINSIC = [
    ("ARI", "extrinsic_ari", 6, "f3"),
    ("NMI", "extrinsic_nmi", 6, "f3"),
    ("F1", "extrinsic_pairwise_f1", 6, "f3"),
    ("Cov%", "extrinsic_coverage", 5, "pct"),
]

_COLS_SIMILARITY = [
    ("P@k", "similarity_extrinsic_precision_at_k", 6, "f3"),
    ("R@k", "similarity_extrinsic_recall_at_k", 6, "f3"),
    ("MAP@k", "similarity_extrinsic_map_at_k", 7, "f3"),
    ("MRR", "similarity_extrinsic_mrr", 6, "f3"),
]


def _fmt(val, fmt: str | None) -> str:
    if val is None:
        return "-"
    if fmt is None:
        s = str(val)
        return s
    if fmt == "f1":
        return f"{float(val):.1f}"
    if fmt == "f2":
        return f"{float(val):.2f}"
    if fmt == "f3":
        return f"{float(val):.3f}"
    if fmt == "f4":
        return f"{float(val):.4f}"
    if fmt == "pct":
        return f"{float(val)*100:.1f}"
    if fmt == "s":
        v = float(val)
        if v >= 3600:
            return f"{v/3600:.1f}h"
        if v >= 60:
            return f"{v/60:.1f}m"
        return f"{v:.1f}s"
    return str(val)


def _truncate(s: str, width: int) -> str:
    if len(s) > width:
        return s[: width - 1] + "…"
    return s


def _build_header(cols: list) -> tuple[str, str]:
    parts = []
    sep_parts = []
    for header, _, width, _ in cols:
        parts.append(header.ljust(width)[:width])
        sep_parts.append("-" * width)
    return "  ".join(parts), "  ".join(sep_parts)


def _build_row(record: dict, cols: list) -> str:
    parts = []
    for _, key, width, fmt in cols:
        val = record.get(key)
        cell = _truncate(_fmt(val, fmt), width)
        parts.append(cell.ljust(width))
    return "  ".join(parts)


def _has_any(records: list[dict], keys: list[str]) -> bool:
    return any(r.get(k) is not None for r in records for k in keys)


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


def _sort_key(r: dict) -> tuple:
    sil = r.get("clustering_quality_silhouette")
    return (r.get("status", ""), -(sil if sil is not None else -999))


def print_table(records: list[dict]) -> None:
    records = sorted(records, key=_sort_key)

    has_extrinsic = _has_any(records, ["extrinsic_ari", "extrinsic_nmi"])
    has_similarity = _has_any(records, ["similarity_extrinsic_precision_at_k"])

    cols = list(_COLS_IDENTITY)
    if has_extrinsic:
        cols += _COLS_EXTRINSIC
    if has_similarity:
        cols += _COLS_SIMILARITY

    header, sep = _build_header(cols)
    print(header)
    print(sep)
    for r in records:
        print(_build_row(r, cols))

    # Failed runs summary
    failed = [r for r in records if r.get("status") != "success"]
    if failed:
        print()
        print(f"Failed runs ({len(failed)}):")
        for r in failed:
            print(f"  {r.get('run_name', '?')}: {r.get('error', 'unknown error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print benchmark results as a comparison table.")
    parser.add_argument("files", nargs="+", help="benchmark JSON file(s)")
    parser.add_argument("--sort", choices=["silhouette", "ari", "name", "embed", "cluster"], default="silhouette",
                        help="Sort column (default: silhouette desc)")
    parser.add_argument("--filter-status", choices=["success", "failed", "all"], default="all")
    args = parser.parse_args()

    records = _load_records(args.files)

    if args.filter_status != "all":
        records = [r for r in records if r.get("status") == args.filter_status]

    if not records:
        print("No records.", file=sys.stderr)
        sys.exit(1)

    sort_map = {
        "silhouette": lambda r: -(r.get("clustering_quality_silhouette") or -999),
        "ari": lambda r: -(r.get("extrinsic_ari") or -999),
        "name": lambda r: r.get("run_name", ""),
        "embed": lambda r: r.get("embedding_type", ""),
        "cluster": lambda r: r.get("clustering_type", ""),
    }
    records = sorted(records, key=sort_map[args.sort])

    print(f"Results: {len(records)} run(s) from {len(args.files)} file(s)\n")
    print_table(records)


if __name__ == "__main__":
    main()
