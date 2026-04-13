from src.abstractions import SiameseBase, ImageData
from src.embedding.embeddings import TorchEmbeddingModel, EmbeddingModelNames
import torch
import json
import os
from PIL import Image
from torch import nn

class Siamese(SiameseBase):
    def __init__(self, model_path: str):
        with open(os.path.join(model_path, "data.json"), "r") as f:
            self.data = json.load(f)
        model_name = EmbeddingModelNames(self.data['model_name'])

        self.embedding_model = TorchEmbeddingModel(model_name)
        model_state = torch.load(os.path.join(model_path, "model.pt"))
        self.embedding_model.model.load_state_dict(model_state)
        self.embedding_model.model.eval()  # Set to evaluation mode

        self.connector = nn.Sequential(
                nn.Linear(self.embedding_model.embedding_size * 2,
                          self.embedding_model.embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_model.embedding_size, 1)
        )
        connector_state = torch.load(os.path.join(model_path, "connector.pt"))
        self.connector.load_state_dict(connector_state)
        self.connector.eval()  # Set to evaluation mode

    def gen_embedding(self, image: Image.Image) -> list[float]:
        return self.embedding_model.gen_embedding(image)

    def distance(self, image1: ImageData, image2: ImageData) -> float:
        # Join the two embeddings and pass through the connector
        emb1 = torch.tensor(image1.embedding)
        emb2 = torch.tensor(image2.embedding)

        input_tensor = torch.cat((emb1, emb2)).unsqueeze(0)  # Add batch dimension
        output = self.connector(input_tensor)
        return output.item()  # Return the distance as a float

    @property
    def embedding_size(self) -> int:
        return self.data['embedding_size']

if __name__ == "__main__":
    s = Siamese("models/resnet50-siamese")

