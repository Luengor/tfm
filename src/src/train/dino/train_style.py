import argparse
import torch
from src.train.models import ModelWithHead
from src.train.style_trainer import run_style_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a DINOv2 ViT-S/14 projection head for graffiti style classification "
                    "using Supervised Contrastive Loss on the dataset_cropped folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="../dataset_cropped",
        help="Path to dataset root with one subdirectory per style class.",
    )
    parser.add_argument(
        "--save-path", default="models/dinov2_graffiti_style_head.pth",
        help="Where to save the best model weights.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Maximum training epochs.")
    parser.add_argument(
        "--samples-per-class", type=int, default=4,
        help="Images per class per batch (batch size = classes × this value).",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument(
        "--temperature", type=float, default=0.07,
        help="SupCon softmax temperature (lower = sharper contrast).",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated list of class folder names to exclude from training (e.g. 'other,tag').",
    )
    return parser.parse_args()


def train_dino_style_head(args: argparse.Namespace) -> None:
    print("Loading DINOv2 model and freezing backbone...")
    base_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    for param in base_model.parameters():
        param.requires_grad = False

    # DINOv2 ViT-S/14 output dim is 384
    model = ModelWithHead(base_model, input_dim=384)

    run_style_training(
        model=model,
        save_path=args.save_path,
        dataset_root=args.dataset,
        num_epochs=args.epochs,
        samples_per_class=args.samples_per_class,
        learning_rate=args.lr,
        temperature=args.temperature,
        patience=args.patience,
        skip={s.strip() for s in args.skip.split(",") if s.strip()},
    )


if __name__ == "__main__":
    train_dino_style_head(parse_args())
