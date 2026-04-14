import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import os, random
from PIL import Image as PILImage
from tqdm import tqdm

from src.embedding.custom import CustomEmbeddingModel
from src.embedding.embeddings import EmbeddingModelNames, TorchEmbeddingModel

class TripletDataset(Dataset):
    def __init__(self, clusters: list[list[str]], images_path: str, transforms=None):
        super().__init__()
        self.clusters = clusters
        self.images_path = images_path
        self.len = sum(len(cluster) for cluster in self.clusters)
        self.first_noise = self.len 
        self.transforms = transforms
        for ci, cluster in enumerate(self.clusters):
            if len(cluster) == 1:
                self.first_noise = ci
                break

    def __len__(self):
        return self.len 

    def __getitem__(self, index: int):
        anchor_class = random.randint(0, self.first_noise - 1)
        first_index = random.randint(0, len(self.clusters[anchor_class]) - 1)
        second_index = first_index
        while second_index == first_index:
            second_index = random.randint(0, len(self.clusters[anchor_class]) - 1)
            
        second_class = random.randint(0, len(self.clusters) - 1)
        while second_class == anchor_class:
            second_class = random.randint(0, len(self.clusters) - 1)
        third_index = random.randint(0, len(self.clusters[second_class]) - 1)

        name1 = os.path.join(
                self.images_path, self.clusters[anchor_class][first_index])
        name2 = os.path.join(
                self.images_path, self.clusters[anchor_class][second_index])
        name3 = os.path.join(
                self.images_path, self.clusters[second_class][third_index])

        image1 = PILImage.open(name1).convert('RGB')
        image2 = PILImage.open(name2).convert('RGB')
        image3 = PILImage.open(name3).convert('RGB')

        if self.transforms:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            image3 = self.transforms(image3)

        return image1, image2, image3

class CustomTrainer:
    def __init__(self, model: TorchEmbeddingModel, cluster_path: str, images_path: str, device: torch.device):
        self.model = model.to(device)
        self.cluster_path = cluster_path
        self.images_path = images_path
        self.device = device

        with open(cluster_path, 'r') as f:
            self.clusters = json.load(f)['clusters']

        self.dataset = TripletDataset(self.clusters, self.images_path, self.model.preprocessor)
        self.dataloader = DataLoader(self.dataset, batch_size=8, shuffle=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    # Perform one epoch of training
    def train(self):
        self.model.train()

        criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        for anchors, positives, negatives in tqdm(self.dataloader):
            anchors, positives, negatives = anchors.to(self.device), positives.to(self.device), negatives.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(anchors), self.model(positives), self.model(negatives)
            loss = criterion(*outputs)
            loss.backward()
            self.optimizer.step()

        # Print last loss for monitoring
        print(f"Last loss: {loss.item():.6f}")

if __name__ == "__main__":
    from sys import argv

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        assert device
    else:
        device = torch.device('cpu')

    custom_resnet = TorchEmbeddingModel(EmbeddingModelNames.RESNET50)
    trainer = CustomTrainer(custom_resnet, argv[1], argv[2], device) 
    trainer.train()

