import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline-benchmark",
        description=(
            "Benchmark Storage + Embedding + Clustering configurations and "
            "export computational cost metrics."
        ),
    )
    parser.add_argument("--whitelist", required=True, help="Path to whitelist JSON file.")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder with images.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory where benchmark CSV/JSON files will be written.",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        help="Optional run names or run IDs to execute from the whitelist.",
    )
    parser.add_argument(
        "--prefix",
        default="benchmark",
        help="Output filename prefix.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    from src.evaluation.runner import run_benchmarks, write_results
    from src.evaluation.whitelist import load_whitelist

    whitelist_path = Path(args.whitelist)
    dataset_path = Path(args.dataset)

    run_specs = load_whitelist(str(whitelist_path))

    if args.run:
        selected = set(args.run)
        run_specs = [spec for spec in run_specs if spec.name in selected or spec.run_id in selected]
        if not run_specs:
            parser.error("None of the provided --run values matched whitelist run names or run IDs.")

    results = run_benchmarks(run_specs=run_specs, dataset_path=str(dataset_path), output_dir=args.output_dir)
    csv_path, json_path = write_results(results=results, output_dir=args.output_dir, prefix=args.prefix)

    success = sum(1 for result in results if result.status == "success")
    failed = len(results) - success
    print(f"Completed {len(results)} runs: {success} success, {failed} failed")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
