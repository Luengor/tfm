import json
import argparse
import itertools
from pathlib import Path

def generate_name(combination, keys):
    parts = []
    for key, value in zip(keys, combination):
        if isinstance(value, dict):
            if "type" in value:
                type_val = value["type"]
                params = value.get("params", {})
                if type_val == "kmeans" and "n_clusters" in params:
                    parts.append(f"kmeans{params['n_clusters']}")
                else:
                    parts.append(type_val)
            else:
                # For generic dicts like distance_query, just use the key if it varies
                parts.append(key)
        elif key == "limit":
            parts.append(f"l{value}")
        elif isinstance(value, bool):
            if value:
                parts.append(key)
            else:
                parts.append(f"no-{key}")
        else:
            parts.append(str(value))
    return "-".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark whitelist from a grid of options.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON grid file.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output JSON whitelist file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return

    with open(input_path, "r") as f:
        grid = json.load(f)

    # Separate keys that are lists (grid) from static values
    grid_keys = []
    grid_values = []
    static_config = {}

    for key, val in grid.items():
        if isinstance(val, list):
            grid_keys.append(key)
            grid_values.append(val)
        else:
            static_config[key] = val

    # Determine which keys actually vary (have more than 1 option)
    varying_indices = [i for i, values in enumerate(grid_values) if len(values) > 1]
    varying_keys = [grid_keys[i] for i in varying_indices]

    # Generate all combinations
    combinations = list(itertools.product(*grid_values))
    
    runs = []
    for combo in combinations:
        run_config = static_config.copy()
        for key, val in zip(grid_keys, combo):
            run_config[key] = val
        
        # Generate a name if not provided in static or grid
        if "name" not in run_config:
            varying_values = [combo[i] for i in varying_indices]
            run_config["name"] = generate_name(varying_values, varying_keys)
        
        runs.append(run_config)

    output_data = {"runs": runs}
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully generated {len(runs)} runs and saved to {args.output}")

if __name__ == "__main__":
    main()
