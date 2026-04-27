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
        help=(
            "Metrics to plot. Can be individual columns or comma-separated groups "
            "(e.g., 'ingest_wall_time_s,clustering_wall_time_s') to show multiple series in one plot."
        ),
    )
    parser.add_argument(
        "--x-axis", "-x", default="run_name", help="Grouping variable for the x-axis."
    )
    parser.add_argument(
        "--hue", "-H", help="Optional sub-grouping variable (color). Ignored if multiple metrics are grouped in one plot."
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
    parser.add_argument(
        "--filter",
        "-f",
        nargs="+",
        help=(
            "Filter expressions to apply to the data (e.g., 'image_count > 100' "
            "or 'storage_type == \"sqlite\"'). Uses pandas.query() syntax."
        ),
    )

    args = parser.parse_args()

    # Load data
    try:
        df = load_data(args.inputs)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Apply filters
    if args.filter:
        for filter_expr in args.filter:
            try:
                before_count = len(df)
                df = df.query(filter_expr)
                after_count = len(df)
                print(f"Applied filter '{filter_expr}': {before_count} -> {after_count} rows.")
            except Exception as e:
                print(f"Error applying filter '{filter_expr}': {e}")
                continue

    if df.empty:
        print("Error: No data left after filtering.")
        return

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_theme(style=args.style)

    # Process metrics
    for metric_group in args.metrics:
        metrics = [m.strip() for m in metric_group.split(",")]
        
        # Validate metrics
        valid_metrics = [m for m in metrics if m in df.columns]
        missing_metrics = [m for m in metrics if m not in df.columns]
        
        if missing_metrics:
            print(f"Warning: Metrics {missing_metrics} not found in data. Skipping them.")
        
        if not valid_metrics:
            continue

        plt.figure(figsize=(12, 6))
        
        if len(valid_metrics) > 1:
            # Multi-metric plot: We need to melt the dataframe to have 'metric_name' and 'value' columns
            # This allows seaborn to use 'metric_name' as hue
            id_vars = [args.x_axis]
            if args.hue and args.hue in df.columns:
                id_vars.append(args.hue)
            
            # Melt only the columns we need
            plot_df = df.melt(
                id_vars=id_vars,
                value_vars=valid_metrics,
                var_name="Metric",
                value_name="Value"
            )
            
            # Determine hue: if user provided a hue, we might have a complex grouping.
            # Usually, for multiple metrics, the metric itself is the hue.
            # If user provided a hue, we combine them for visualization or prioritize the metric.
            hue_col = "Metric"
            if args.hue and args.hue in df.columns:
                # If both metric and user-hue are present, we combine them for the legend
                plot_df["Group"] = plot_df[args.hue].astype(str) + " (" + plot_df["Metric"] + ")"
                hue_col = "Group"
                
            ax = sns.barplot(data=plot_df, x=args.x_axis, y="Value", hue=hue_col)
            plt.ylabel("Value")
            title_suffix = f"({', '.join(valid_metrics)})"
        else:
            # Single metric plot
            metric = valid_metrics[0]
            ax = sns.barplot(data=df, x=args.x_axis, y=metric, hue=args.hue)
            plt.ylabel(metric)
            title_suffix = metric

        plt.title(f"Benchmark: {title_suffix}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save plot
        safe_name = metric_group.replace(",", "_").replace("/", "_").replace("\\", "_")
        plot_file = output_path / f"{safe_name}.png"
        plt.savefig(plot_file)
        print(f"Saved plot: {plot_file}")

        if args.show:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    main()
