import torch
from src.train.models import ModelWithHead
from src.train.style_head import build_arg_parser, train_style_head


def train_dino_style_head(args) -> None:
    print("Loading DINOv2 model and freezing backbone...")
    base_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    for param in base_model.parameters():
        param.requires_grad = False

    # DINOv2 ViT-S/14 output dim is 384
    model = ModelWithHead(base_model, input_dim=384, output_dim=args.output_dim)
    train_style_head(model, args)


if __name__ == "__main__":
    parser = build_arg_parser(
        description="Fine-tune a DINOv2 ViT-S/14 projection head for graffiti style similarity.",
        default_save_path="models/dinov2_graffiti_style_head.pth",
    )
    train_dino_style_head(parser.parse_args())
