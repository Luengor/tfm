from src.embedding.embeddings import EmbeddingModelNames, TorchEmbeddingModel
import torch

class CustomEmbeddingModel(TorchEmbeddingModel):
    def __init__(self, name: EmbeddingModelNames, weight_path: str):
        super().__init__(name)

        # Load custom weights
        self.model.load_state_dict(torch.load(weight_path))


