import torch.nn as nn
from torchvision import models
from src.train.models import ModelWithHead
from src.train.style_head import build_arg_parser, train_style_head


def train_mobilenet_style_head(args) -> None:
    print("Loading MobileNetV3-Large model and freezing backbone...")
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    base_model = models.mobilenet_v3_large(weights=weights)
    # Remove final classifier to get features (output dim 1280)
    base_model.classifier[3] = nn.Identity()

    for param in base_model.parameters():
        param.requires_grad = False

    # MobileNetV3-Large feature size is 1280
    model = ModelWithHead(base_model, input_dim=1280, output_dim=args.output_dim)
    train_style_head(model, args)


if __name__ == "__main__":
    parser = build_arg_parser(
        description="Fine-tune a MobileNetV3-Large projection head for graffiti style similarity.",
        default_save_path="models/mobilenet_graffiti_style_head.pth",
    )
    train_mobilenet_style_head(parser.parse_args())
