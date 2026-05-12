"""
Sample N images from a directory tree (recursing into subfolders) and copy
them into a flat output folder. Optionally auto-annotate the sampled images
with a YOLO model and emit a CVAT 1.1 annotations.xml.

Examples:

    # Sample 300 images, no annotation
    python -m src.scripts.sample_and_annotate \\
        --input ../dataset/images_5k \\
        --output ../dataset/sample_300 \\
        --num 300

    # Sample 300 images and auto-annotate with a YOLO model
    python -m src.scripts.sample_and_annotate \\
        --input ../dataset/images_5k \\
        --output ../dataset/sample_300 \\
        --num 300 \\
        --model models/yolo26m.pt
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(input_dir: Path) -> list[Path]:
    return [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def make_dest_name(src_path: Path, input_root: Path) -> str:
    """Flatten a nested path to a single filename, prefixing with subfolders
    to avoid collisions between e.g. ``a/img.jpg`` and ``b/img.jpg``."""
    rel = src_path.relative_to(input_root)
    if len(rel.parts) == 1:
        return rel.name
    return "__".join(rel.parts)


def sample_and_copy(input_dir: Path, output_dir: Path, n: int, seed: int) -> list[Path]:
    if not input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {input_dir}")

    print(f"Scanning {input_dir} for images...")
    images = find_images(input_dir)
    print(f"Found {len(images)} images.")

    if n > len(images):
        sys.exit(f"Requested {n} images but only {len(images)} available.")

    rng = random.Random(seed)
    sampled = rng.sample(images, n)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying {n} images to {output_dir} (seed={seed})...")
    copied: list[Path] = []
    for src in tqdm(sampled):
        dest_path = output_dir / make_dest_name(src, input_dir)
        shutil.copy2(src, dest_path)
        copied.append(dest_path)
    return copied


def write_cvat_xml(images: list[Path], model_path: str, output_xml: Path) -> None:
    from ultralytics import YOLO

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    class_names = model.names

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    job = ET.SubElement(meta, "job")
    ET.SubElement(job, "id").text = "0"
    ET.SubElement(job, "size").text = str(len(images))
    ET.SubElement(job, "mode").text = "annotation"

    labels = ET.SubElement(meta, "labels")
    for _cls_id, name in class_names.items():
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "color").text = "#00ff00"
        ET.SubElement(label, "type").text = "rectangle"

    print(f"Running YOLO inference on {len(images)} images...")
    box_count = 0
    for i, img_path in enumerate(tqdm(images)):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            tqdm.write(f"Error opening {img_path}: {e}")
            continue

        results = model(str(img_path), verbose=False)

        image_elem = ET.SubElement(
            root, "image",
            id=str(i),
            name=img_path.name,
            width=str(width),
            height=str(height),
        )

        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                ET.SubElement(image_elem, "box", {
                    "label": class_names[cls_id],
                    "source": "manual",
                    "occluded": "0",
                    "xtl": f"{coords[0]:.2f}",
                    "ytl": f"{coords[1]:.2f}",
                    "xbr": f"{coords[2]:.2f}",
                    "ybr": f"{coords[3]:.2f}",
                    "z_order": "0",
                })
                box_count += 1

    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ", level=0)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotations to {output_xml} ({box_count} boxes total)...")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample N images from a directory tree and copy them into an "
            "output folder. Optionally auto-annotate with a YOLO model and "
            "emit a CVAT 1.1 annotations.xml."
        ),
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory (recursive)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-n", "--num", type=int, required=True, help="Number of images to sample")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Optional YOLO model path. If provided, auto-annotates sampled images and writes CVAT XML.",
    )
    parser.add_argument(
        "--output-xml",
        default=None,
        help="Path for CVAT XML output (default: <output>/annotations.xml). Only used with --model.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    copied = sample_and_copy(input_dir, output_dir, args.num, args.seed)

    if args.model:
        output_xml = Path(args.output_xml).resolve() if args.output_xml else output_dir / "annotations.xml"
        write_cvat_xml(copied, args.model, output_xml)


if __name__ == "__main__":
    main()
