import argparse

from src.train.dataset import _STYLE_AUG_TRANSFORM
from src.train.trainer import run_training


def build_arg_parser(description: str, default_save_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="../dataset_cropped",
        help="Path to dataset root with one subdirectory per style class.",
    )
    parser.add_argument(
        "--save-path", default=default_save_path,
        help="Where to save the best model weights.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs.")
    parser.add_argument(
        "--classes-per-batch", type=int, default=None,
        help="P: number of distinct classes per PK batch. Default: all classes "
             "(matches the original balanced-batch behaviour for style training).",
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=4,
        help="K: number of samples per class per PK batch (batch size = P × K).",
    )
    parser.add_argument(
        "--batches-per-epoch", type=int, default=None,
        help="Batches per epoch. Default: len(dataset) // (P × K).",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument(
        "--loss", default="supcon", choices=["supcon", "triplet"],
        help="Metric-learning loss: SupCon (Khosla 2020) or batch-hard triplet (Hermans 2017).",
    )
    parser.add_argument(
        "--margin", type=float, default=0.3,
        help="Triplet margin in squared-L2 space on normalized embeddings (only used when --loss=triplet).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07,
        help="SupCon softmax temperature (only used when --loss=supcon; lower = sharper contrast).",
    )
    parser.add_argument(
        "--output-dim", type=int, default=128,
        help="Projection head output dimensionality.",
    )
    parser.add_argument(
        "--train-min-samples", type=int, default=2,
        help="Minimum crops per class to include in training. Default 2 ensures "
             "every class can supply a non-trivial positive pair within a batch.",
    )
    parser.add_argument(
        "--val-dataset", default=None,
        help="Optional path to validation crops directory (one subfolder per style class). "
             "If set, validation metrics are computed each epoch.",
    )
    parser.add_argument(
        "--val-min-samples", type=int, default=2,
        help="Minimum crops per class for validation evaluation.",
    )
    parser.add_argument(
        "--val-metric", default="accuracy_1nn",
        choices=["silhouette", "accuracy_1nn", "ari", "nmi"],
        help="Validation metric used to select the best model.",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early-stopping patience in epochs (no improvement in --val-metric). "
             "Requires --val-dataset. Disabled if unset.",
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated list of class folder names to exclude from training "
             "and validation (e.g. 'other,tag').",
    )
    return parser


def train_style_head(model, args: argparse.Namespace) -> None:
    skip = {s.strip() for s in args.skip.split(",") if s.strip()} or None
    run_training(
        model=model,
        save_path=args.save_path,
        dataset_dir=args.dataset,
        num_epochs=args.epochs,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        batches_per_epoch=args.batches_per_epoch,
        learning_rate=args.lr,
        loss=args.loss,
        margin=args.margin,
        temperature=args.temperature,
        train_min_samples=args.train_min_samples,
        val_dataset_dir=args.val_dataset,
        val_min_samples=args.val_min_samples,
        val_metric=args.val_metric,
        patience=args.patience,
        train_transform=_STYLE_AUG_TRANSFORM,
        skip_classes=skip,
    )
