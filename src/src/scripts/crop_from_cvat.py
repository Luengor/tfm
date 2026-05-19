"""
Crop images from a CVAT 1.1 annotations.xml export, grouping crops by label
into subfolders. Useful for turning a CVAT-labelled folder into a per-class
image dataset that can be consumed directly (e.g. as ground truth for
clustering metrics).

Boxes whose label is in --skip-labels (default: "other") are skipped.

Also emits ``labels.csv`` at the root of the output directory with the
``filename,style`` columns expected by ``pipeline-benchmark --ground-truth``.

Example:

    python -m src.scripts.crop_from_cvat \\
        --input ../dataset/sample_500 \\
        --output ../dataset/sample_500_crops

    uv run pipeline-benchmark \\
        --configuration benchmarks/style_eval.json \\
        --dataset ../dataset/sample_500_crops \\
        --ground-truth ../dataset/sample_500_crops/labels.csv
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def crop_from_cvat(
    image_dir: Path,
    annotations_xml: Path,
    output_dir: Path,
    skip_labels: set[str],
) -> None:
    if not image_dir.is_dir():
        sys.exit(f"Image directory does not exist: {image_dir}")
    if not annotations_xml.is_file():
        sys.exit(f"Annotations file does not exist: {annotations_xml}")

    print(f"Parsing {annotations_xml}...")
    tree = ET.parse(annotations_xml)
    root = tree.getroot()

    images = root.findall("image")
    print(f"Found {len(images)} images in annotations.")

    output_dir.mkdir(parents=True, exist_ok=True)

    label_counts: Counter[str] = Counter()
    rows: list[tuple[str, str]] = []
    skipped = 0
    missing = 0
    errors = 0

    for image_elem in tqdm(images):
        filename = image_elem.get("name")
        if not filename:
            continue
        img_path = image_dir / filename
        if not img_path.is_file():
            missing += 1
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size

                for box_idx, box in enumerate(image_elem.findall("box")):
                    label = box.get("label", "")
                    if label in skip_labels:
                        text_attr = box.find("attribute[@name='text']")
                        text_val = (text_attr.text or "").strip().lower() if text_attr is not None else ""
                        if text_val:
                            label = text_val
                        else:
                            skipped += 1
                            continue

                    try:
                        xtl = float(box.get("xtl", "0"))
                        ytl = float(box.get("ytl", "0"))
                        xbr = float(box.get("xbr", "0"))
                        ybr = float(box.get("ybr", "0"))
                    except ValueError:
                        errors += 1
                        continue

                    x0 = max(0, int(xtl))
                    y0 = max(0, int(ytl))
                    x1 = min(img_w, int(xbr))
                    y1 = min(img_h, int(ybr))
                    if x1 <= x0 or y1 <= y0:
                        errors += 1
                        continue

                    crop = img.crop((x0, y0, x1, y1))
                    label_dir = output_dir / label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    out_name = f"{Path(filename).stem}_{box_idx:03d}.jpg"
                    crop.save(label_dir / out_name, quality=95)
                    label_counts[label] += 1
                    rows.append((out_name, label))
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")
            errors += 1
            continue

    labels_csv = output_dir / "labels.csv"
    with labels_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "style"])
        writer.writerows(rows)

    print("\nDone.")
    print(f"  Skipped boxes (in --skip-labels): {skipped}")
    print(f"  Missing image files: {missing}")
    print(f"  Errors: {errors}")
    print(f"  Crops per label:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")
    print(f"  Total crops: {sum(label_counts.values())}")
    print(f"  Wrote labels CSV: {labels_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crop images from a CVAT 1.1 annotations.xml export, grouping "
            "crops by label into subfolders."
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing images and annotations.xml",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory (one subfolder per label)",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Path to annotations.xml (default: <input>/annotations.xml)",
    )
    parser.add_argument(
        "--skip-labels",
        nargs="+",
        default=["other"],
        help="Box labels to skip (default: other)",
    )

    args = parser.parse_args()

    image_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    annotations_xml = (
        Path(args.annotations).resolve()
        if args.annotations
        else image_dir / "annotations.xml"
    )

    crop_from_cvat(
        image_dir=image_dir,
        annotations_xml=annotations_xml,
        output_dir=output_dir,
        skip_labels=set(args.skip_labels),
    )


if __name__ == "__main__":
    main()
