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
    parser.add_argument("--configuration", required=True, help="Path to configuration JSON file.")
    parser.add_argument("--dataset", required=True, help="Path to dataset folder with images.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory where benchmark JSON files will be written.",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        help="Optional run names or run IDs to execute from the configuration.",
    )
    parser.add_argument(
        "--prefix",
        default="benchmark",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of images to process per run.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help=(
            "Optional path to a ground-truth CSV with at least 'filename' and "
            "'style' columns. When provided, the runner computes extrinsic "
            "metrics (ARI, NMI, pairwise F1) for runs that use the identity "
            "segmenter. Filenames are matched by basename."
        ),
    )
    parser.add_argument(
        "--cluster-plot",
        action="store_true",
        default=False,
        help=(
            "Generate an interactive 2D UMAP Vega-Lite scatter of the final "
            "clusters per run. Writes <output-dir>/<run_id>_cluster.html. "
            "Overrides the 'cluster_plot.enabled' field in the configuration if set."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    from src.evaluation.runner import load_ground_truth, run_benchmarks, write_results
    from src.evaluation.configuration import load_configuration_with_options

    configuration_path = Path(args.configuration)
    dataset_path = Path(args.dataset)

    run_specs, cluster_plot_options = load_configuration_with_options(str(configuration_path))
    if args.cluster_plot:
        cluster_plot_options["enabled"] = True

    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else None

    if args.run:
        selected = set(args.run)
        run_specs = [spec for spec in run_specs if spec.name in selected or spec.run_id in selected]
        if not run_specs:
            parser.error("None of the provided --run values matched configuration run names or run IDs.")

    if args.limit is not None:
        from src.evaluation.configuration import make_run_id
        for spec in run_specs:
            spec.limit = args.limit
            spec.run_id = make_run_id(spec)

    results = run_benchmarks(
        run_specs=run_specs,
        dataset_path=str(dataset_path),
        output_dir=args.output_dir,
        limit=args.limit,
        ground_truth=ground_truth,
        cluster_plot_options=cluster_plot_options,
    )
    json_path = write_results(results=results, output_dir=args.output_dir, prefix=args.prefix)

    success = sum(1 for result in results if result.status == "success")
    failed = len(results) - success
    print(f"Completed {len(results)} runs: {success} success, {failed} failed")
    print(f"JSON: {json_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
