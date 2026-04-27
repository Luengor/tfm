import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def load_data(input_paths: List[str]) -> pd.DataFrame:
    """Loads and flattens benchmark results from one or more JSON files."""
    all_results = []
    for path in input_paths:
        p = Path(path)
        if not p.exists():
            print(f"Warning: File {path} not found. Skipping.")
            continue
        
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # The JSON structure from runner.py is {"runtime": ..., "results": [...]}
            if "results" in data:
                all_results.extend(data["results"])
            else:
                # Fallback if it's just a list of results
                all_results.extend(data if isinstance(data, list) else [data])
                
    if not all_results:
        raise ValueError("No valid results found in the provided input files.")
        
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark metrics.")
    parser.add_argument(
        "--inputs", "-i", nargs="+", required=True, help="Path(s) to JSON results files."
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["ingest_wall_time_s", "clustering_wall_time_s", "clustering_quality_silhouette"],
        help="Metrics to plot (columns in the results JSON).",
    )
    parser.add_argument(
        "--x-axis", "-x", default="run_name", help="Grouping variable for the x-axis."
    )
    parser.add_argument(
        "--hue", "-H", help="Optional sub-grouping variable (color)."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots",
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--show", "-s", action="store_true", help="Display plots interactively."
    )
    parser.add_argument(
        "--style", default="whitegrid", help="Seaborn plot style."
    )

    args = parser.parse_args()

    # Load data
    try:
        df = load_data(args.inputs)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_theme(style=args.style)

    # Plot each metric
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data. Skipping.")
            continue

        plt.figure(figsize=(12, 6))
        
        # Determine plot type: if x-axis is categorical or unique
        # For benchmarks, barplot is usually best for comparing across runs
        ax = sns.barplot(data=df, x=args.x_axis, y=metric, hue=args.hue)
        
        plt.title(f"Benchmark: {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save plot
        safe_metric_name = metric.replace("/", "_").replace("\\", "_")
        plot_file = output_path / f"{safe_metric_name}.png"
        plt.savefig(plot_file)
        print(f"Saved plot: {plot_file}")

        if args.show:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    main()
