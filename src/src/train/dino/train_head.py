import torch
from src.train.models import ModelWithHead
from src.train.trainer import run_training

def train_dino_head():
    print("Loading DINOv2 model and freezing backbone...")
    base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    for param in base_model.parameters():
        param.requires_grad = False
    
    # DINOv2 ViT-S/14 output dim is 384
    model = ModelWithHead(base_model, input_dim=384)
    
    run_training(
        model=model,
        save_path="models/dinov2_graffiti_head.pth",
        num_epochs=20,
        batch_size=8,
        learning_rate=1e-4
    )

if __name__ == "__main__":
    train_dino_head()
