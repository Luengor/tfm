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
    },

    EmbeddingModelNames.VGG16: {
        'model': vgg16,
        'weights': VGG16_Weights.DEFAULT,
    },

    EmbeddingModelNames.INCEPTION_V3: {
        'model': inception_v3,
        'weights': Inception_V3_Weights.DEFAULT
    },
    EmbeddingModelNames.YOLOn: {
        'model': 'yolo26n.pt',
    },
    EmbeddingModelNames.YOLOs: {
        'model': 'yolo26s.pt',
    },
    EmbeddingModelNames.YOLOm: {
        'model': 'yolo26m.pt',
    },
}

class TorchEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        self.name = name

        m_data = MODELS[name]
        self.preprocessor = m_data['weights'].transforms()
        self.model = m_data['model'](weights=m_data['weights'])
        if hasattr(self.model, 'classifier'):
            # Remove the final layer
            self.model.classifier = self.model.classifier[:-1]
        elif hasattr(self.model, 'fc'):
            self.model.fc = torch.nn.Identity()
        else:
            print(f"Nothing done for model {name}")
        self.model.eval()  # Set model to evaluation mode

    def gen_embedding(self, image: ImageImage) -> list[float]:
        # Get the embedding
        with torch.no_grad():
            input_tensor = self.preprocessor(image).unsqueeze(0)  # Add batch dimension
            output = self.model(input_tensor)

        return output.squeeze().tolist()  # Convert to list for storage

class YoloEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        self.model = YOLO(MODELS[name]['model'])

    def gen_embedding(self, image: ImageImage) -> list[float]:
        return self.model.embed(image)[0].cpu().tolist() # type: ignore

def get_model(name: EmbeddingModelNames) -> EmbeddingBase:
    match name:
        case EmbeddingModelNames.YOLOm | EmbeddingModelNames.YOLOs | EmbeddingModelNames.YOLOn:
            return YoloEmbeddingModel(name)

        case _:
            return TorchEmbeddingModel(name)

