import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.train.dataset import GraffitiTripletDataset

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def run_training(
    model,
    save_path,
    dataset_dir: str = os.path.join("dataset", "crops"),
    num_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    margin: float = 1.0,
    hard_negative_prob: float = 0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = GraffitiTripletDataset(root_dir=dataset_dir, hard_negative_prob=hard_negative_prob)
    print(
        f"Dataset: {len(dataset)} anchor images across {len(dataset.anchor_classes)} classes, "
        f"{len(dataset.singleton_classes)} singleton hard-negative classes"
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    criterion = TripletLoss(margin=margin)
    # Only optimize parameters that require grad (the head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for anchors, positives, negatives in pbar:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            optimizer.zero_grad()
            
            anchor_embeds = model(anchors)
            positive_embeds = model(positives)
            negative_embeds = model(negatives)
            
            # Normalize embeddings to unit hypersphere
            anchor_embeds = F.normalize(anchor_embeds, p=2, dim=1)
            positive_embeds = F.normalize(positive_embeds, p=2, dim=1)
            negative_embeds = F.normalize(negative_embeds, p=2, dim=1)
            
            loss = criterion(anchor_embeds, positive_embeds, negative_embeds)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
