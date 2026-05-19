import argparse
import torch.nn as nn
from torchvision import models
from src.train.models import ModelWithHead
from src.train.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a MobileNetV3-Large projection head for graffiti author similarity "
                    "using triplet loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset root with one subdirectory per author.",
    )
    parser.add_argument(
        "--save-path", default="models/mobilenet_graffiti_head.pth",
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


def train_mobilenet_head(args: argparse.Namespace) -> None:
    print("Loading MobileNetV3-Large model and freezing backbone...")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    base_model = models.mobilenet_v3_large(weights=weights)
    # Remove final classifier to get embeddings (output dim 1280)
    base_model.classifier[3] = nn.Identity()

    for param in base_model.parameters():
        param.requires_grad = False

    # MobileNetV3-Large feature size is 1280
    model = ModelWithHead(base_model, input_dim=1280)

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
    train_mobilenet_head(parse_args())
