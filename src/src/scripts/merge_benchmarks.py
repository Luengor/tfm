"""Merge multiple benchmark JSON files into one.

For runs with the same run_id, keeps the entry with the latest finished_at.
The output runtime block is taken from the most recently finished file.
"""

import argparse
import json
from pathlib import Path


def _run_sort_key(run: dict) -> str:
    return run.get("finished_at") or run.get("started_at") or ""


def merge_benchmark_files(input_files: list[Path]) -> dict:
    best: dict[str, dict] = {}
    latest_runtime: dict | None = None
    latest_ts: str = ""

    for path in input_files:
        with path.open() as f:
            data = json.load(f)

        runtime = data.get("runtime", {})
        results = data.get("results", [])

        file_ts = max((_run_sort_key(r) for r in results), default="")
        if file_ts > latest_ts:
            latest_ts = file_ts
            latest_runtime = runtime

        seen_in_file: set[str] = set()
        for run in results:
            rid = run["run_id"]
            if rid in seen_in_file:
                raise ValueError(f"Duplicate run_id '{rid}' within {path}")
            seen_in_file.add(rid)
            existing = best.get(rid)
            if existing is None or _run_sort_key(run) > _run_sort_key(existing):
                best[rid] = run

    return {
        "runtime": latest_runtime or {},
        "results": list(best.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Benchmark JSON files to merge",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help="Output merged JSON file",
    )
    args = parser.parse_args()

    input_files = [Path(p) for p in args.input]
    merged = merge_benchmark_files(input_files)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(merged, f, indent=2)

    n_runs = len(merged["results"])
    print(f"Merged {len(input_files)} file(s) → {n_runs} unique run(s) → {out}")


if __name__ == "__main__":
    main()
