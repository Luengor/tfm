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
        
    df = pd.DataFrame(all_results)

    # Expand JSON string columns for easier filtering
    json_cols = ["storage_params", "embedding_params", "clustering_params", "reduction_params"]
    for col in json_cols:
        if col in df.columns:
            try:
                # Parse JSON strings and handle possible None values
                expanded = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else (x if x is not None else {}))
                # Convert to DataFrame and join
                expanded_df = pd.json_normalize(expanded.tolist()).add_prefix(f"{col}.")
                # Reset index to ensure proper alignment during concat
                df = pd.concat([df.reset_index(drop=True), expanded_df.reset_index(drop=True)], axis=1)
            except Exception as e:
                print(f"Warning: Could not expand JSON column {col}: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark metrics.")
    parser.add_argument(
        "--inputs", "-i", nargs="+", action="append", required=True, help="Path(s) to JSON results files."
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        action="append",
        help=(
            "Metrics to plot. Can be individual columns or comma-separated groups "
            "(e.g., 'ingest_wall_time_s,clustering_wall_time_s') to show multiple series in one plot. "
            "Defaults to ingest_wall_time_s, clustering_wall_time_s, and clustering_quality_silhouette."
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
        action="append",
        help=(
            "Filter expressions to apply to the data (e.g., 'image_count > 100' "
            "or 'storage_type == \"sqlite\"'). Uses pandas.query() syntax. "
            "JSON parameters are expanded (e.g., '`clustering_params.n_clusters` == 5'). "
            "Note: Use backticks for columns with dots."
        ),
    )
    parser.add_argument(
        "--sort-by",
        "-S",
        nargs="+",
        action="append",
        help="Column(s) to sort the data by before plotting. Supports expanded JSON columns."
    )
    parser.add_argument(
        "--sort-order",
        choices=["asc", "desc"],
        default="asc",
        help="Sort order (asc or desc). Default is asc."
    )

    args = parser.parse_args()

    # Flatten list arguments that use action="append"
    if args.inputs:
        args.inputs = [item for sublist in args.inputs for item in sublist]

    if args.metrics:
        args.metrics = [item for sublist in args.metrics for item in sublist]
    else:
        args.metrics = ["ingest_wall_time_s", "clustering_wall_time_s", "clustering_quality_silhouette"]

    if args.filter:
        args.filter = [item for sublist in args.filter for item in sublist]
    if args.sort_by:
        args.sort_by = [item for sublist in args.sort_by for item in sublist]

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
                if "." in filter_expr and "`" not in filter_expr:
                    print(f"Hint: Use backticks for columns with dots, e.g., '`{filter_expr.split()[0]}`' if it contains a dot.")
                continue

    # Apply sorting
    if args.sort_by:
        ascending = args.sort_order == "asc"
        # Validate sort columns
        valid_sort_cols = [c for c in args.sort_by if c in df.columns]
        missing_sort_cols = [c for c in args.sort_by if c not in df.columns]
        
        if missing_sort_cols:
            print(f"Warning: Sort columns {missing_sort_cols} not found in data. Ignoring them.")
        
        if valid_sort_cols:
            try:
                df = df.sort_values(by=valid_sort_cols, ascending=ascending)
                print(f"Sorted data by {valid_sort_cols} ({args.sort_order}).")
            except Exception as e:
                print(f"Error sorting data: {e}")

    if df.empty:
        print("Error: No data left after filtering.")
        return

    # Handle multi-variable x-axis
    x_axis_cols = [c.strip() for c in args.x_axis.split(",")]
    plot_x_axis = args.x_axis
    
    if len(x_axis_cols) > 1:
        # Check if all columns exist
        missing_cols = [c for c in x_axis_cols if c not in df.columns]
        if missing_cols:
            print(f"Error: X-axis columns {missing_cols} not found in data.")
            return
        
        # Create combined column
        plot_x_axis = " + ".join(x_axis_cols)
        # We use a lambda to ensure strings and handle NaNs gracefully
        df[plot_x_axis] = df[x_axis_cols].apply(
            lambda row: " | ".join(row.fillna("N/A").astype(str)), axis=1
        )
        print(f"Combined x-axis variables: {x_axis_cols} -> {plot_x_axis}")
    elif args.x_axis not in df.columns:
        print(f"Error: X-axis column '{args.x_axis}' not found in data.")
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
            id_vars = [plot_x_axis]
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
                
            sns.barplot(data=plot_df, x=plot_x_axis, y="Value", hue=hue_col)
            plt.ylabel("Value")
            title_suffix = f"({', '.join(valid_metrics)})"
        else:
            # Single metric plot
            metric = valid_metrics[0]
            sns.barplot(data=df, x=plot_x_axis, y=metric, hue=args.hue)
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
