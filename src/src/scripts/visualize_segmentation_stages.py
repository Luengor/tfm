"""Visualize the YOLO segmentation process as a staged pipeline figure.

Renders, left to right: original photo -> raw detections -> fused boxes ->
padded boxes -> resulting crops. Used by methods.tex (fig:segmentation-stages)
to illustrate how a photograph is turned into the crops fed to the extractor.

Unlike ``visualize_detections.py`` (which draws the *final* boxes stored in a
DB), this reproduces every intermediate stage of ``YoloSegmenter._postprocess``
so the figure shows what happens between detection and crop. The reproduction is
asserted against ``YoloSegmenter.segment`` so the last stage is faithful.

Requires the ``src`` venv (torch + ultralytics). Run from ``src/``:

    uv run python -m src.scripts.visualize_segmentation_stages \
        --model models/yolo11m-train-10.pt \
        --image-dir ../data/detect/a/images \
        --output ../doc/figures/segmentation_stages.pdf --seed 7
"""
import argparse
import os
import random
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ruff: noqa: E402
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from PIL.Image import Image as PILImage

from src.abstractions import BoundingBox
from src.embedding.segmenters import YoloSegmenter

# Default segmenter parameters mirror the experiment configs (e.g. exp05).
THRESHOLD = 0.5
MERGE_THRESHOLD = 0.8
PADDING = 0.05

DETECT_COLOR = "#1f77b4"
MERGE_COLOR = "#d62728"
PAD_COLOR = "#2ca02c"


def raw_boxes(segmenter: YoloSegmenter, image: PILImage) -> list[BoundingBox]:
    """Detections before any merge/padding (the head of ``_postprocess``)."""
    width, height = image.size
    result = segmenter.model(image, conf=segmenter.threshold, verbose=False)[0]
    boxes: list[BoundingBox] = []
    for box in result.boxes:
        coords = box.xyxy[0].tolist()
        boxes.append(BoundingBox(
            x1=coords[0] / width, y1=coords[1] / height,
            x2=coords[2] / width, y2=coords[3] / height,
            confidence=float(box.conf[0]),
        ))
    return boxes


def stages(segmenter: YoloSegmenter, image: PILImage):
    """Reproduce the three box stages: raw, fused, padded."""
    raw = raw_boxes(segmenter, image)
    fused = segmenter._merge_boxes(raw)
    padded = [segmenter._apply_padding(b) for b in fused]
    return raw, fused, padded


def draw_boxes(ax, image: PILImage, boxes: list[BoundingBox], color: str) -> None:
    w, h = image.size
    ax.imshow(image)
    for b in boxes:
        ax.add_patch(Rectangle(
            (b.x1 * w, b.y1 * h), (b.x2 - b.x1) * w, (b.y2 - b.y1) * h,
            fill=False, edgecolor=color, linewidth=2.2,
        ))
    clean(ax)


def clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def crop(image: PILImage, b: BoundingBox) -> PILImage:
    w, h = image.size
    return image.crop((int(b.x1 * w), int(b.y1 * h), int(b.x2 * w), int(b.y2 * h)))


def stack_crops(image: PILImage, boxes: list[BoundingBox], target_h: int = 320, gap: int = 16) -> PILImage:
    """Composite the crops into a single horizontal strip on a white canvas."""
    crops = []
    for b in boxes:
        c = crop(image, b)
        scale = target_h / c.height
        crops.append(c.resize((max(1, int(c.width * scale)), target_h)))
    total_w = sum(c.width for c in crops) + gap * (len(crops) - 1)
    canvas = Image.new("RGB", (total_w, target_h), "white")
    x = 0
    for c in crops:
        canvas.paste(c, (x, 0))
        x += c.width + gap
    return canvas


def pick_image(segmenter: YoloSegmenter, candidates: list[Path], rng: random.Random) -> tuple[Path, PILImage, tuple]:
    """Find an image where merging *and* multiple crops are both visible."""
    rng.shuffle(candidates)
    fallback = None
    for path in candidates:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        raw, fused, padded = stages(segmenter, image)
        if len(fused) >= 2 and len(raw) > len(fused):
            return path, image, (raw, fused, padded)
        if fallback is None and len(fused) >= 2:
            fallback = (path, image, (raw, fused, padded))
    if fallback is None:
        raise SystemExit("No candidate image produced >=2 fused boxes; try a different --seed or --image-dir.")
    print("Warning: no image had both a merge and multiple crops; using a multi-crop fallback.")
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo11m-train-10.pt", help="Fine-tuned graffiti YOLO weights.")
    parser.add_argument("--image", help="Specific image to use (skips auto-selection).")
    parser.add_argument("--image-dir", default="../data/detect/a/images", help="Folder to auto-select an image from.")
    parser.add_argument("--output", default="../doc/figures/segmentation_stages.pdf", help="Output figure path.")
    parser.add_argument("--max-candidates", type=int, default=40, help="Max images to scan when auto-selecting.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for candidate shuffling (regenerable figure).")
    args = parser.parse_args()

    segmenter = YoloSegmenter(
        model_path=args.model, threshold=THRESHOLD,
        merge_threshold=MERGE_THRESHOLD, padding=PADDING,
    )
    rng = random.Random(args.seed)

    if args.image:
        image = Image.open(args.image).convert("RGB")
        chosen = Path(args.image)
        boxes = stages(segmenter, image)
    else:
        all_imgs = sorted(Path(args.image_dir).glob("*.jpg")) + sorted(Path(args.image_dir).glob("*.png"))
        if not all_imgs:
            raise SystemExit(f"No images in {args.image_dir}")
        candidates = all_imgs[:args.max_candidates]
        chosen, image, boxes = pick_image(segmenter, candidates, rng)

    raw, fused, padded = boxes
    # Faithfulness check: our reproduced final stage must equal the real output.
    assert padded == segmenter.segment(image), "stage reproduction diverged from YoloSegmenter.segment"
    print(f"Image: {chosen.name}  raw={len(raw)} fused={len(fused)} padded={len(padded)}")

    fig = plt.figure(figsize=(11.0, 6.0))
    # Two rows of three cells. Row 1: foto, detecciones, fusión.
    # Row 2: margen, recortes (spanning the last two cells).
    gs = fig.add_gridspec(2, 3, wspace=0.1, hspace=0.06)
    ax_photo = fig.add_subplot(gs[0, 0])
    ax_detect = fig.add_subplot(gs[0, 1])
    ax_fusion = fig.add_subplot(gs[0, 2])
    ax_margin = fig.add_subplot(gs[1, 0])
    ax_cuts = fig.add_subplot(gs[1, 1:])

    draw_boxes(ax_photo, image, [], DETECT_COLOR)
    draw_boxes(ax_detect, image, raw, DETECT_COLOR)
    draw_boxes(ax_fusion, image, fused, MERGE_COLOR)
    draw_boxes(ax_margin, image, padded, PAD_COLOR)
    ax_cuts.imshow(stack_crops(image, padded))
    clean(ax_cuts)

    # Row 1 labels above the panels; row 2 labels below, so the wrap arrow can
    # descend into the top of Margen without landing on its label.
    for ax, t in [(ax_photo, "Fotografía"), (ax_detect, "Detecciones"), (ax_fusion, "Fusión")]:
        ax.set_title(t, fontsize=12, fontweight="bold")
    for ax, t in [(ax_margin, "Margen"), (ax_cuts, "Recortes")]:
        ax.set_xlabel(t, fontsize=12, fontweight="bold")

    fig.canvas.draw()

    def arrow(p0, p1, **kw):
        fig.add_artist(FancyArrowPatch(
            p0, p1, transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=16, color="0.35", linewidth=1.5, **kw,
        ))

    def hgap(left, right):
        """Horizontal arrow in the gap between two side-by-side panels."""
        y = (left.get_position().y0 + left.get_position().y1) / 2
        arrow((left.get_position().x1 + 0.004, y), (right.get_position().x0 - 0.004, y))

    hgap(ax_photo, ax_detect)
    hgap(ax_detect, ax_fusion)
    hgap(ax_margin, ax_cuts)

    # Wrap arrow: drops from the bottom of Fusión (end of row 1), sweeps left
    # through the empty band between the rows, and descends into the top of
    # Margen (start of row 2) — so it points clearly down into Margen and
    # crosses neither the upper nor the lower panels.
    f_pos = ax_fusion.get_position()
    m_pos = ax_margin.get_position()
    arrow(
        ((f_pos.x0 + f_pos.x1) / 2, f_pos.y0 - 0.004),
        ((m_pos.x0 + m_pos.x1) / 2, m_pos.y1 + 0.004),
        connectionstyle="arc,angleA=-90,angleB=90,armA=30,armB=30,rad=10",
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
