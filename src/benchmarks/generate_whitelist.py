import json
import argparse
import itertools
import hashlib
import copy
from pathlib import Path

def get_storage_key(run_config):
    # We care about storage (excluding db_path), embedding and limit
    storage = run_config.get("storage", {})
    embedding = run_config.get("embedding", {})
    limit = run_config.get("limit")
    
    storage_type = str(storage.get("type", "")).lower()
    storage_params = dict(storage.get("params", {}))
    
    # We ignore db_path for the key to group runs that COULD share a database
    if "db_path" in storage_params:
        del storage_params["db_path"]
        
    key_data = {
        "storage_type": storage_type,
        "storage_params": storage_params,
        "embedding": embedding,
        "limit": limit
    }
    
    # Stable JSON string for hashing
    key_str = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(key_str.encode()).hexdigest()[:12]
    return key

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
    seen_storage_keys = {} # key -> stable_db_path

    for combo in combinations:
        run_config = copy.deepcopy(static_config)
        for key, val in zip(grid_keys, combo):
            run_config[key] = copy.deepcopy(val)
        
        # Generate a name if not provided in static or grid
        if "name" not in run_config:
            varying_values = [combo[i] for i in varying_indices]
            run_config["name"] = generate_name(varying_values, varying_keys)
        
        # Database reuse logic
        storage_key = get_storage_key(run_config)
        storage_type = str(run_config.get("storage", {}).get("type", "")).lower()

        if storage_key not in seen_storage_keys:
            # First time seeing this combination, we must clear storage
            run_config["clear_storage"] = True
            
            # For SQLite, we assign a stable path based on the storage key if none provided
            if storage_type == "sqlite":
                storage_params = run_config.get("storage", {}).get("params", {})
                db_path = storage_params.get("db_path", "")
                
                if not db_path or "{run_id}" in db_path:
                    # Use a stable path that doesn't depend on the run_id
                    if not db_path:
                        stable_path = "{output_dir}/db/storage_" + storage_key + ".sqlite"
                    else:
                        # Replace {run_id} with the storage_key to keep the user's naming preference but make it stable
                        stable_path = db_path.replace("{run_id}", "storage_" + storage_key)
                        
                    if "storage" not in run_config:
                        run_config["storage"] = {"type": "sqlite", "params": {}}
                    if "params" not in run_config["storage"]:
                        run_config["storage"]["params"] = {}
                    run_config["storage"]["params"]["db_path"] = stable_path
                    seen_storage_keys[storage_key] = stable_path
                else:
                    # User provided a stable path, we respect it but still use it for reuse
                    seen_storage_keys[storage_key] = db_path
            else:
                # For Postgres or others, we just mark it as seen
                seen_storage_keys[storage_key] = True
        else:
            # Reusing a previously seen storage/embedding/limit combination
            run_config["clear_storage"] = False
            
            # If it's SQLite, ensure we use the same stable path
            if storage_type == "sqlite" and isinstance(seen_storage_keys[storage_key], str):
                if "storage" not in run_config:
                    run_config["storage"] = {"type": "sqlite", "params": {}}
                if "params" not in run_config["storage"]:
                    run_config["storage"]["params"] = {}
                run_config["storage"]["params"]["db_path"] = seen_storage_keys[storage_key]
        
        runs.append(run_config)

    output_data = {"runs": runs}
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Successfully generated {len(runs)} runs and saved to {args.output}")

if __name__ == "__main__":
    main()
