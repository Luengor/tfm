import argparse
import os
import random
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
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of images to show (each image is drawn once with all of its bounding boxes).")
    parser.add_argument("--output-dir", help="If provided, save the visualized images to this directory instead of showing them.")
    parser.add_argument("--no-show", action="store_true", help="Do not show images using matplotlib (useful for headless environments).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling images. If omitted, uses system entropy.")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    print(f"Using random seed: {seed}")
    rng = random.Random(seed)

    if not os.path.exists(args.db):
        print(f"Error: Database file not found: {args.db}")
        sys.exit(1)

    try:
        storage = SQLiteStorage(args.db)
        images_data = storage.get_all_images()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Group all bounding boxes by source image filename.
    detections_by_image: dict[str, list] = {}
    for img in images_data:
        if img.bbox is None:
            continue
        detections_by_image.setdefault(img.filename, []).append(img.bbox)

    if not detections_by_image:
        print("No detections found in the database.")
        return

    filenames = list(detections_by_image.keys())
    rng.shuffle(filenames)

    print(f"Found {len(filenames)} images with detections in the database. Processing up to {args.limit}.")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving images to: {args.output_dir}")

    count = 0
    for filename in filenames:
        if count >= args.limit:
            break

        bboxes = detections_by_image[filename]

        # Drop default full-image bboxes (IdentitySegmenter sentinel).
        bboxes = [
            b for b in bboxes
            if not (b.confidence == 1.0 and b.x1 == 0 and b.y1 == 0 and b.x2 == 1 and b.y2 == 1)
        ]
        if not bboxes:
            print(f"Skipping image with only default full-image bboxes: {filename}")
            continue

        img_path = filename
        if not os.path.isabs(img_path):
            if args.dataset:
                resolved_path = os.path.join(args.dataset, img_path)
            else:
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
                w, h = img.size

                for bbox in bboxes:
                    x1 = bbox.x1 * w
                    y1 = bbox.y1 * h
                    x2 = bbox.x2 * w
                    y2 = bbox.y2 * h

                    draw.rectangle([x1, y1, x2, y2], outline="red", width=10)
                    label = f"{bbox.confidence:.2f}"
                    draw.text((x1, y1), label, fill="red")

                print(f"{os.path.basename(resolved_path)}: {len(bboxes)} bbox(es)")

                if args.output_dir:
                    base = os.path.basename(resolved_path)
                    save_path = os.path.join(args.output_dir, f"det_{count}_{base}")
                    img.save(save_path)
                    print(f"[{count+1}/{args.limit}] Saved: {save_path}")

                if not args.no_show:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img)
                    plt.title(f"{os.path.basename(resolved_path)} - {len(bboxes)} bbox(es)")
                    plt.axis("off")
                    plt.show()

                count += 1
        except Exception as e:
            print(f"Error processing {resolved_path}: {e}")

    print(f"Done. Processed {count} images.")

if __name__ == "__main__":
    main()
