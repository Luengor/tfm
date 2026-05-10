import argparse
import os
import sys
from pathlib import Path

# Add the parent directory of 'src' to sys.path to allow imports from the 'src' package
# Assuming the script is in src/src/scripts/visualize_detections.py
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from src.storage.sqlite import SQLiteStorage

def main():
    parser = argparse.ArgumentParser(description="Visualize image detections from a SQLite database.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database.")
    parser.add_argument("--dataset", help="Optional: Base path for images if they are stored as relative paths in the DB.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of images to show.")
    parser.add_argument("--output-dir", help="If provided, save the visualized images to this directory instead of showing them.")
    parser.add_argument("--no-show", action="store_true", help="Do not show images using matplotlib (useful for headless environments).")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database file not found: {args.db}")
        sys.exit(1)

    try:
        storage = SQLiteStorage(args.db)
        images_data = storage.get_all_images()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Filter only those with bounding boxes
    detections = [img for img in images_data if img.bbox is not None]
    
    if not detections:
        print("No detections found in the database.")
        return

    print(f"Found {len(detections)} total detections in the database. Processing up to {args.limit}.")
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving images to: {args.output_dir}")

    count = 0
    for data in detections:
        if count >= args.limit:
            break
            
        img_path = data.filename
        # Try to resolve the path
        if not os.path.isabs(img_path):
            if args.dataset:
                resolved_path = os.path.join(args.dataset, img_path)
            else:
                # Try relative to CWD
                resolved_path = os.path.abspath(img_path)
        else:
            resolved_path = img_path
            
        if not os.path.exists(resolved_path):
            print(f"Warning: Image not found: {resolved_path}")
            continue
            
        try:
            with Image.open(resolved_path) as img:
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                bbox = data.bbox
                
                # Draw the bounding box
                # BoundingBox is (x1, y1, x2, y2)
                draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline="red", width=3)
                
                # Draw confidence
                label = f"{bbox.confidence:.2f}"
                draw.text((bbox.x1, bbox.y1), label, fill="red")
                
                if args.output_dir:
                    filename = os.path.basename(resolved_path)
                    save_path = os.path.join(args.output_dir, f"det_{count}_{filename}")
                    img.save(save_path)
                    print(f"[{count+1}/{args.limit}] Saved: {save_path}")
                
                if not args.no_show:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img)
                    plt.title(f"{os.path.basename(resolved_path)} - Conf: {bbox.confidence:.2f}")
                    plt.axis("off")
                    plt.show()
                
                count += 1
        except Exception as e:
            print(f"Error processing {resolved_path}: {e}")

    print(f"Done. Processed {count} images.")

if __name__ == "__main__":
    main()
