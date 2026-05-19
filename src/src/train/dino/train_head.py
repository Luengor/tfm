import argparse
import torch
from src.train.models import ModelWithHead
from src.train.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a DINOv2 ViT-S/14 projection head for graffiti author similarity "
                    "using triplet loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset root with one subdirectory per author.",
    )
    parser.add_argument(
        "--save-path", default="models/dinov2_graffiti_author_head.pth",
        help="Where to save the trained model weights.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--margin", type=float, default=1.0, help="Triplet loss margin.")
    parser.add_argument(
        "--hard-negative-prob", type=float, default=0.5,
        help="Probability of sampling a singleton class as hard negative.",
    )
    return parser.parse_args()


def train_dino_head(args: argparse.Namespace) -> None:
    print("Loading DINOv2 model and freezing backbone...")
    base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    for param in base_model.parameters():
        param.requires_grad = False

    # DINOv2 ViT-S/14 output dim is 384
    model = ModelWithHead(base_model, input_dim=384)

    run_training(
        model=model,
        save_path=args.save_path,
        dataset_dir=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        margin=args.margin,
        hard_negative_prob=args.hard_negative_prob,
    )


if __name__ == "__main__":
    train_dino_head(parse_args())
