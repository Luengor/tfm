import torch.nn as nn
from torchvision import models
from src.train.author_head import build_arg_parser, train_author_head
from src.train.models import ModelWithHead


def train_mobilenet_head(args) -> None:
    print("Loading MobileNetV3-Large model and freezing backbone...")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    base_model = models.mobilenet_v3_large(weights=weights)
    # Remove final classifier to get embeddings (output dim 1280)
    base_model.classifier[3] = nn.Identity()

    for param in base_model.parameters():
        param.requires_grad = False

    # MobileNetV3-Large feature size is 1280
    model = ModelWithHead(base_model, input_dim=1280, output_dim=args.output_dim)
    train_author_head(model, args)


if __name__ == "__main__":
    parser = build_arg_parser(
        description="Fine-tune a MobileNetV3-Large projection head for graffiti author similarity "
                    "using batch-hard triplet loss.",
        default_save_path="models/mobilenet_graffiti_author_head.pth",
    )
    train_mobilenet_head(parser.parse_args())
