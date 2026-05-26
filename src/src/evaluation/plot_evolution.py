import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the existing load_data function to ensure consistent parsing of the benchmark JSON
from src.evaluation.plot_metrics import load_data

def main():
    parser = argparse.ArgumentParser(description="Plot the evolution of parameters/metrics against each other.")
    parser.add_argument(
        "--inputs", "-i", nargs="+", action="append", required=True, help="Path(s) to JSON results files."
    )
    parser.add_argument(
        "--metrics", "-m", nargs="+", action="append", required=True, help="Metrics to plot. Supply 2 for a single plot, or 3+ for a pairwise grid."
    )
    parser.add_argument(
        "--hue", "-H", help="Optional sub-grouping variable (color)."
    )
    parser.add_argument(
        "--plot-type", "-t", choices=["scatter", "line"], default="line", help="Type of plot (scatter or line)."
    )
    parser.add_argument(
        "--output", "-o", default="plots/evolution.png", help="Output file path for the plot."
    )
    parser.add_argument(
        "--show", "-s", action="store_true", help="Display the plot interactively."
    )
    parser.add_argument(
        "--filter", "-f", nargs="+", action="append", help="Filter expressions (e.g., '`clustering_params.n_clusters` > 5')."
    )
    parser.add_argument(
        "--style", default="whitegrid", help="Seaborn plot style."
    )

    args = parser.parse_args()

    # Flatten list arguments
    if args.inputs:
        args.inputs = [item for sublist in args.inputs for item in sublist]
    if args.metrics:
        args.metrics = [item for sublist in args.metrics for item in sublist]
    if args.filter:
        args.filter = [item for sublist in args.filter for item in sublist]

    if len(args.metrics) < 2:
        print("Error: At least two metrics must be provided (-m metric1 -m metric2).")
        return

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

    if df.empty:
        print("Error: No data to plot.")
        return

    missing_metrics = [m for m in args.metrics if m not in df.columns]
    if missing_metrics:
        print(f"Warning: Metrics {missing_metrics} not found in data. Skipping.")
    
    valid_metrics = [m for m in args.metrics if m in df.columns]
    if len(valid_metrics) < 2:
        print("Error: Not enough valid metrics left to plot (need at least 2).")
        return

    if args.hue:
        hue_cols = [c.strip() for c in args.hue.split(",")]
        missing_hues = [c for c in hue_cols if c not in df.columns]
        if missing_hues:
            print(f"Warning: Hue column(s) {missing_hues} not found in data. Ignoring hue.")
            args.hue = None
        elif len(hue_cols) > 1:
            combined_hue = " + ".join(hue_cols)
            df[combined_hue] = df[hue_cols].apply(
                lambda row: " | ".join(row.fillna("N/A").astype(str)), axis=1
            )
            args.hue = combined_hue
            print(f"Combined hue variables: {hue_cols} -> {combined_hue}")
        else:
            args.hue = hue_cols[0]

    # Drop missing values
    df_plot = df.dropna(subset=valid_metrics)

    if df_plot.empty:
        print("Error: Data is empty after dropping missing values for the selected metrics.")
        return

    # Set style
    sns.set_theme(style=args.style)

    plot_kwargs = {}
    if args.plot_type == "line":
        plot_kwargs["marker"] = "o"  # Add a marker for line plots
        plot_kwargs["dashes"] = False

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(valid_metrics) == 2:
        # Single plot for 2 metrics
        plt.figure(figsize=(8, 6))
        x_metric, y_metric = valid_metrics[0], valid_metrics[1]
        
        if args.plot_type == "scatter":
            sns.scatterplot(data=df_plot, x=x_metric, y=y_metric, hue=args.hue, **plot_kwargs)
        else:
            sns.lineplot(data=df_plot, x=x_metric, y=y_metric, hue=args.hue, **plot_kwargs)
            
        plt.title(f"{y_metric} vs {x_metric}")
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
        
    else:
        # PairGrid for 3 or more metrics
        g = sns.PairGrid(df_plot, vars=valid_metrics, hue=args.hue, height=2.5, aspect=1.2)
        
        if args.plot_type == "scatter":
            g.map(sns.scatterplot, **plot_kwargs)
        else:
            g.map(sns.lineplot, **plot_kwargs)
            
        if args.hue:
            g.add_legend()
            
        # Adjust layout
        g.figure.subplots_adjust(top=0.95)
        g.figure.suptitle(f"Pairwise Evolution ({len(valid_metrics)} metrics)")
        
        g.savefig(output_path)
        print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
