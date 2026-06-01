"""Zero-shot CLIP baseline for graffiti author identification.

Each subfolder name under --dataset is treated as the author label and fed
through the CLIP text encoder (optionally with a prompt template, or an
ensemble of templates whose embeddings are averaged). Image embeddings are
matched to the text-anchor matrix by cosine similarity. Reports top-1 and
top-5 accuracy plus per-class top-1 accuracy.

No fine-tuning. Establishes baseline before any text-supervised training.
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")

# ruff: noqa: E402
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.train.evaluate import list_eligible_classes


DEFAULT_TEMPLATES = [
    "a photo of graffiti by {}",
    "graffiti tag '{}'",
    "street art by the artist {}",
    "a wall painted with the tag {}",
    "{}",
]


def humanize(name: str) -> str:
    """Folder name -> a string fed to the text encoder.

    Replaces underscores/dashes with spaces; everything else (case, digits) is
    kept as-is since graffiti tags are case- and orthography-sensitive.
    """
    return name.replace("_", " ").replace("-", " ").strip()


@torch.no_grad()
def build_text_anchors(
    model,
    tokenizer,
    classes: list[str],
    templates: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Returns an [n_classes, dim] L2-normalized matrix of text anchors.

    When multiple templates are given, embeddings are averaged per class then
    re-normalized (standard CLIP zero-shot ensembling).
    """
    anchors = []
    for cls in classes:
        name = humanize(cls)
        prompts = [t.format(name) for t in templates]
        tokens = tokenizer(prompts).to(device)
        emb = model.encode_text(tokens).float()
        emb = F.normalize(emb, p=2, dim=-1)
        emb = emb.mean(dim=0)
        emb = F.normalize(emb, p=2, dim=-1)
        anchors.append(emb)
    return torch.stack(anchors, dim=0)


@torch.no_grad()
def encode_image_dataset(
    model,
    preprocess,
    classes: list[str],
    crops_dir: str,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Encodes every image under crops_dir/<class>/. Returns (embeddings, labels, paths)."""
    embs: list[torch.Tensor] = []
    labels: list[int] = []
    paths: list[str] = []
    for label_idx, cls in enumerate(tqdm(classes, desc="Encoding images")):
        cls_dir = os.path.join(crops_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device)
                e = model.encode_image(x).float()
                e = F.normalize(e, p=2, dim=-1).squeeze(0).cpu()
                embs.append(e)
                labels.append(label_idx)
                paths.append(img_path)
            except Exception as e:
                print(f"  Error {img_path}: {e}")
    if not embs:
        raise RuntimeError("No images encoded.")
    return torch.stack(embs, dim=0), np.array(labels), paths


def top_k_accuracy(scores: torch.Tensor, labels: np.ndarray, k: int) -> float:
    topk = scores.topk(k=min(k, scores.shape[1]), dim=1).indices.cpu().numpy()
    correct = np.any(topk == labels[:, None], axis=1)
    return float(correct.mean())


def per_class_accuracy(scores: torch.Tensor, labels: np.ndarray, n_classes: int) -> np.ndarray:
    preds = scores.argmax(dim=1).cpu().numpy()
    acc = np.zeros(n_classes, dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for p, y in zip(preds, labels):
        counts[y] += 1
        if p == y:
            acc[y] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, acc / counts, np.nan)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot CLIP author baseline using text-anchor cosine matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to crops directory (one subfolder per author).")
    parser.add_argument("--min-samples", type=int, default=1,
                        help="Minimum crops per class to include.")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--template", action="append", default=None,
                        help="Prompt template containing '{}'. Repeatable; embeddings are "
                             "averaged across templates. If omitted, uses a built-in ensemble.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5],
                        help="Top-k accuracies to report.")
    parser.add_argument("--save-per-class", default=None,
                        help="Optional path to write per-class top-1 accuracy as CSV.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes, skipped = list_eligible_classes(args.dataset, args.min_samples)
    print(f"Classes: {len(classes)} with >={args.min_samples} samples ({skipped} skipped)")
    if not classes:
        raise SystemExit("No eligible classes.")

    templates = args.template if args.template else DEFAULT_TEMPLATES
    print(f"Templates ({len(templates)}):")
    for t in templates:
        print(f"  - {t}")

    print(f"Loading CLIP {args.model} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    model.eval()

    print("Building text anchors...")
    text_anchors = build_text_anchors(model, tokenizer, classes, templates, device)
    print(f"  text_anchors: {tuple(text_anchors.shape)}")

    img_emb, labels, _paths = encode_image_dataset(
        model, preprocess, classes, args.dataset, device
    )
    print(f"  image_emb:    {tuple(img_emb.shape)}")

    scores = img_emb.to(device) @ text_anchors.t()

    print("\nResults:")
    for k in args.top_k:
        acc = top_k_accuracy(scores, labels, k)
        print(f"  top-{k} accuracy: {acc:.4f}")

    per_cls = per_class_accuracy(scores, labels, len(classes))
    valid = ~np.isnan(per_cls)
    print(f"  macro-avg top-1: {per_cls[valid].mean():.4f} "
          f"(over {valid.sum()} classes with >=1 sample)")

    if args.save_per_class:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_per_class)) or ".", exist_ok=True)
        with open(args.save_per_class, "w") as f:
            f.write("class,top1_accuracy\n")
            for cls, a in zip(classes, per_cls):
                f.write(f"{cls},{a if not np.isnan(a) else ''}\n")
        print(f"  per-class accuracy -> {args.save_per_class}")


if __name__ == "__main__":
    main()
