import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

class GraffitiTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_images = {cls: [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))] for cls in self.classes}
        self.classes = [cls for cls in self.classes if len(self.class_to_images[cls]) >= 2]
        self.all_images = []
        for cls in self.classes:
            for img_path in self.class_to_images[cls]:
                self.all_images.append((img_path, cls))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_path, anchor_class = self.all_images[idx]
        positive_path = random.choice(self.class_to_images[anchor_class])
        while positive_path == anchor_path and len(self.class_to_images[anchor_class]) > 1:
            positive_path = random.choice(self.class_to_images[anchor_class])
        negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])
        negative_path = random.choice(self.class_to_images[negative_class])
        
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class DINOWithHead(nn.Module):
    def __init__(self, base_model):
        super(DINOWithHead, self).__init__()
        self.base_model = base_model
        # Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 384)
        )
        
    def forward(self, x):
        features = self.base_model(x)
        return self.projection_head(features)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 20
    batch_size = 8
    learning_rate = 1e-4 # Higher LR for head-only
    margin = 1.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = GraffitiTripletDataset(root_dir=os.path.join("dataset", "crops"), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Loading DINOv2 model and freezing backbone...")
    base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    for param in base_model.parameters():
        param.requires_grad = False
    
    model = DINOWithHead(base_model)
    model.to(device)
    model.train()

    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.projection_head.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for anchors, positives, negatives in pbar:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            optimizer.zero_grad()
            
            anchor_embeds = model(anchors)
            positive_embeds = model(positives)
            negative_embeds = model(negatives)
            
            anchor_embeds = F.normalize(anchor_embeds, p=2, dim=1)
            positive_embeds = F.normalize(positive_embeds, p=2, dim=1)
            negative_embeds = F.normalize(negative_embeds, p=2, dim=1)
            
            loss = criterion(anchor_embeds, positive_embeds, negative_embeds)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
    # Save the model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "dinov2_graffiti_head.pth")
    # We only need the state dict of the head if we reconstruct the model, 
    # but for simplicity let's save the whole thing or just enough to load later.
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
