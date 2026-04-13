import torch
from torch import nn
from PIL import Image as PILImage
from src.embedding.embeddings import EmbeddingModelNames, TorchEmbeddingModel
import os
import json

class SiameseModel(nn.Module):
    def __init__(self, embedding_model: TorchEmbeddingModel):
        super().__init__()
        self.embedding_model = embedding_model
        self.connector = nn.Sequential(
                nn.Linear(embedding_model.embedding_size * 2,
                          embedding_model.embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_model.embedding_size, 1)
        )

        self.embedding_model.model.train()

    def forward_once(self, image: PILImage.Image) -> torch.Tensor:
        image_tensor = self.embedding_model.preprocessor(image).unsqueeze(0)
        return self.embedding_model.model(image_tensor)

    def forward(self, image1: PILImage.Image, image2: PILImage.Image) -> torch.Tensor:
        emb1 = self.forward_once(image1)
        emb2 = self.forward_once(image2)

        # Concatenate the embeddings and pass through the connector
        embs = torch.cat((emb1, emb2), dim=1)
        output = self.connector(embs)

        return output

    def save_siamese(self, base_path: str):
        folder_path = os.path.join(base_path, f"{self.embedding_model.name.value}-siamese")

        # Create a folder for the model
        os.makedirs(folder_path, exist_ok=False)

        # Save the model, preprocessor, and connector
        torch.save(self.embedding_model.model.state_dict(), os.path.join(folder_path, "model.pt"))
        torch.save(self.connector.state_dict(), os.path.join(folder_path, "connector.pt"))

        # Save some info
        info = {
            "model_name": self.embedding_model.name.value,
        }
        with open(os.path.join(folder_path, "data.json"), "w") as f:
            json.dump(info, f)

if __name__ == "__main__":
    s = SiameseModel(TorchEmbeddingModel(EmbeddingModelNames.RESNET50))
    s.save_siamese("models")

