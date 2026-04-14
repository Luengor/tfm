from src.embedding.embeddings import EmbeddingModelNames, TorchEmbeddingModel
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image as PILImage
import os

class CustomEmbeddingModel(TorchEmbeddingModel):
    def __init__(self, name: EmbeddingModelNames, weight_path: str):
        super().__init__(name)

        # Load custom weights
        self.model.load_state_dict(torch.load(weight_path))


