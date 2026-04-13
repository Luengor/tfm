from torchvision.models import (
    ResNet50_Weights,
    VGG16_Weights,
    Inception_V3_Weights,
    resnet50,
    vgg16,
    inception_v3,
)
import torch
from enum import Enum
from PIL.Image import Image as ImageImage
from ultralytics import YOLO # pyright: ignore
from src.abstractions import EmbeddingBase

class EmbeddingModelNames(str, Enum):
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    INCEPTION_V3 = "inception_v3"
    YOLOn = "yolon"
    YOLOs = "yolos"
    YOLOm = "yolom"

MODELS = {
    EmbeddingModelNames.RESNET50: {
        'model': resnet50,
        'weights': ResNet50_Weights.DEFAULT,
        'embedding_size': 2048,
    },

    EmbeddingModelNames.VGG16: {
        'model': vgg16,
        'weights': VGG16_Weights.DEFAULT,
        'embedding_size': 4096,
    },

    EmbeddingModelNames.INCEPTION_V3: {
        'model': inception_v3,
        'weights': Inception_V3_Weights.DEFAULT,
        'embedding_size': 2048,
    },
    EmbeddingModelNames.YOLOn: {
        'model': 'yolo26n.pt',
        'embedding_size': 256,
    },
    EmbeddingModelNames.YOLOs: {
        'model': 'yolo26s.pt',
        'embedding_size': 512,
    },
    EmbeddingModelNames.YOLOm: {
        'model': 'yolo26m.pt',
        'embedding_size': 512,
    },
}

class TorchEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        self.name = name

        m_data = MODELS[name]
        self.preprocessor = m_data['weights'].transforms()
        self.model = m_data['model'](weights=m_data['weights'])

        # Remove final layer
        match name:
            case EmbeddingModelNames.RESNET50:
                self.model.fc = torch.nn.Identity()
            case EmbeddingModelNames.VGG16:
                self.model.classifier[6] = torch.nn.Identity()
            case EmbeddingModelNames.INCEPTION_V3:
                self.model.fc = torch.nn.Identity()

        self.model.eval()  # Set model to evaluation mode

    def gen_embedding(self, image: ImageImage) -> list[float]:
        # Get the embedding
        with torch.no_grad():
            input_tensor = self.preprocessor(image).unsqueeze(0)  # Add batch dimension
            output = self.model(input_tensor)

        return output.squeeze().tolist()  # Convert to list for storage

    @property
    def embedding_size(self) -> int:
        return MODELS[self.name]['embedding_size']

class YoloEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        self.name = name
        self.model = YOLO(MODELS[name]['model'])

    def gen_embedding(self, image: ImageImage) -> list[float]:
        return self.model.embed(image)[0].cpu().tolist() # type: ignore

    @property
    def embedding_size(self) -> int:
        return MODELS[self.name]['embedding_size']

def get_model(name: EmbeddingModelNames) -> EmbeddingBase:
    match name:
        case EmbeddingModelNames.YOLOm | EmbeddingModelNames.YOLOs | EmbeddingModelNames.YOLOn:
            return YoloEmbeddingModel(name)

        case _:
            return TorchEmbeddingModel(name)


if __name__ == "__main__":
    for name in EmbeddingModelNames:
        model = get_model(name)
        print(name, model.embedding_size)
