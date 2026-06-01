"""
Crop all images in a folder (recursively) using a pretrained YOLO detector,
mirroring the input directory structure in the output.

Exposes the same segmentation knobs as ``YoloSegmenter`` (the segmenter used
by ``pipeline-benchmark``), so the produced crops match what a benchmark run
would feed to the embedding stage:

  - ``--threshold`` (detection confidence)
  - ``--no-full-image-fallback`` (disable full-image fallback when nothing
    is detected; enabled by default)
  - ``--merge-threshold`` (overlap-area ratio required to merge two boxes)
  - ``--max-boxes-per-image``
  - ``--touch-merge-gap`` (normalized gap distance below which two boxes
    are merged even without overlap)
  - ``--max-merged-area`` (cap on merged-box area, fraction of image)
  - ``--padding`` (extra normalized padding applied around each kept box)

Example:

    uv run python -m src.scripts.crop_with_yolo \\
        --input ../dataset/images \\
        --output ../dataset/images_crops \\
        --model ../models/yolo11m.pt \\
        --threshold 0.4 --merge-threshold 0.6 --touch-merge-gap 0.02
"""

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "False")

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.embedding.segmenters import YoloSegmenter

try:
    from ultralytics.utils import LOGGER as _ULTRA_LOGGER  # type: ignore
    _ULTRA_LOGGER.setLevel(logging.WARNING)
except Exception:
    pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def crop_with_yolo(
    input_dir: Path,
    output_dir: Path,
    segmenter: YoloSegmenter,
    quality: int,
) -> None:
    if not input_dir.is_dir():
        sys.exit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_images(input_dir))
    if not image_paths:
        sys.exit(f"No images found in {input_dir}")

    total_crops = 0
    skipped = 0
    errors = 0

    for img_path in tqdm(image_paths):
        rel = img_path.relative_to(input_dir).parent
        out_subdir = output_dir / rel
        out_subdir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                boxes = segmenter.segment(img)

                if not boxes:
                    skipped += 1
                    continue

                stem = img_path.stem
                ext = img_path.suffix.lower()
                if ext not in {".jpg", ".jpeg"}:
                    ext = ".jpg"

                for idx, box in enumerate(boxes):
                    x0 = max(0, int(box.x1 * w))
                    y0 = max(0, int(box.y1 * h))
                    x1 = min(w, int(box.x2 * w))
                    y1 = min(h, int(box.y2 * h))
                    if x1 <= x0 or y1 <= y0:
                        errors += 1
                        continue
                    crop = img.crop((x0, y0, x1, y1))
                    out_name = f"{stem}_{idx:03d}{ext}"
                    crop.save(out_subdir / out_name, quality=quality)
                    total_crops += 1
        except Exception as e:
            tqdm.write(f"Error processing {img_path}: {e}")
            errors += 1
            continue

    print("\nDone.")
    print(f"  Images processed: {len(image_paths)}")
    print(f"  Images with no boxes (skipped): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total crops: {total_crops}")
    print(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crop all images of a folder structure using a pretrained YOLO. "
            "Mirrors the input subdirectory layout in the output."
        ),
    )
    parser.add_argument("-i", "--input", required=True, help="Input image folder (recursed)")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt weights")

    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument(
        "--no-full-image-fallback",
        action="store_true",
        help="Skip images that have no detections (default: emit full-image crop as fallback)",
    )
    parser.add_argument("--merge-threshold", type=float, default=0.8, help="Overlap-area ratio to merge boxes")
    parser.add_argument("--max-boxes-per-image", type=int, default=None, help="Cap kept boxes per image")
    parser.add_argument("--touch-merge-gap", type=float, default=0.0, help="Normalized gap below which two boxes are merged")
    parser.add_argument("--max-merged-area", type=float, default=1.0, help="Max normalized area allowed for a merged box")
    parser.add_argument("--padding", type=float, default=0.0, help="Extra normalized padding around each kept box")
    parser.add_argument("--batch-size", type=int, default=1, help="(Unused here; segment_batch not invoked)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for saved crops")

    args = parser.parse_args()

    segmenter = YoloSegmenter(
        model_path=args.model,
        threshold=args.threshold,
        merge_threshold=args.merge_threshold,
        padding=args.padding,
        max_boxes_per_image=args.max_boxes_per_image,
        allow_full_image_fallback=not args.no_full_image_fallback,
        batch_size=args.batch_size,
        touch_merge_gap=args.touch_merge_gap,
        max_merged_area=args.max_merged_area,
    )

    crop_with_yolo(
        input_dir=Path(args.input).resolve(),
        output_dir=Path(args.output).resolve(),
        segmenter=segmenter,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
