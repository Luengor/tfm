import torch.nn as nn
from torchvision import models
from src.train.models import ModelWithHead
from src.train.trainer import run_training

def train_mobilenet_head():
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
        save_path="models/mobilenet_graffiti_head.pth",
        num_epochs=20,
        batch_size=8,
        learning_rate=1e-4
    )

if __name__ == "__main__":
    train_mobilenet_head()
