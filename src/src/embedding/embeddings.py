import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.hub")
warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")

from torchvision.models import (
    ResNet50_Weights,
    VGG16_Weights,
    Inception_V3_Weights,
    MobileNet_V3_Large_Weights,
    resnet50,
    vgg16,
    inception_v3,
    mobilenet_v3_large,
)
import torch
import torch.nn as nn
from enum import Enum
from PIL.Image import Image as ImageImage
from ultralytics import YOLO # pyright: ignore
import open_clip
from src.abstractions import EmbeddingBase

def get_device() -> torch.device:
    return torch.accelerator.current_accelerator() or torch.device("cpu")
class EmbeddingModelNames(str, Enum):
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    INCEPTION_V3 = "inception_v3"
    MOBILENET_V3 = "mobilenet_v3"
    DINOV2_VITS14 = "dinov2_vits14"
    CLIP_VIT_B32 = "clip_vit_b32"
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
    EmbeddingModelNames.MOBILENET_V3: {
        'model': mobilenet_v3_large,
        'weights': MobileNet_V3_Large_Weights.DEFAULT,
        'embedding_size': 960,
    },
    EmbeddingModelNames.DINOV2_VITS14: {
        'embedding_size': 384,
    },
    EmbeddingModelNames.CLIP_VIT_B32: {
        'embedding_size': 512,
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


class TorchEmbeddingModel(nn.Module, EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        super().__init__()
        self.name = name
        self.device = get_device()

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
            case EmbeddingModelNames.MOBILENET_V3:
                self.model.classifier[3] = torch.nn.Identity()

        self.model.to(self.device)

    def gen_embedding(self, image: ImageImage) -> list[float]:
        # Ensure model is in evaluation mode
        self.model.eval()

        # Get the embedding
        with torch.no_grad():
            input_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to device
            output = self.model(input_tensor)

        return output.squeeze().cpu().tolist()  # Convert to list for storage, moving back to CPU

    def forward(self, x):
        return self.model(x)

    @property
    def embedding_size(self) -> int:
        return MODELS[self.name]['embedding_size']

class DinoEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        self.name = name
        self.device = get_device()
        # Using torch hub for DINOv2
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.to(self.device)
        self.model.eval()
        
        # Standard DINOv2 transforms
        from torchvision import transforms
        self.preprocessor = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def gen_embedding(self, image: ImageImage) -> list[float]:
        with torch.no_grad():
            input_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
        return output.squeeze().cpu().tolist()

    @property
    def embedding_size(self) -> int:
        return MODELS[self.name]['embedding_size']

class ClipEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        self.name = name
        self.device = get_device()
        # Using open_clip for CLIP
        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        self.model.to(self.device)
        self.model.eval()

    def gen_embedding(self, image: ImageImage) -> list[float]:
        with torch.no_grad():
            input_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
            output = self.model.encode_image(input_tensor)
            # Normalize to unit length (standard for CLIP)
            output /= output.norm(dim=-1, keepdim=True)
        return output.squeeze().cpu().tolist()

    @property
    def embedding_size(self) -> int:
        return MODELS[self.name]['embedding_size']

class YoloEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames):
        self.name = name
        self.device = get_device()
        self.model = YOLO(MODELS[name]['model'])
        self.model.to(self.device)

    def gen_embedding(self, image: ImageImage) -> list[float]:
        return self.model.embed(image)[0].cpu().tolist() # type: ignore

    @property
    def embedding_size(self) -> int:
        return MODELS[self.name]['embedding_size']

def get_model(name: EmbeddingModelNames) -> EmbeddingBase:
    match name:
        case EmbeddingModelNames.YOLOm | EmbeddingModelNames.YOLOs | EmbeddingModelNames.YOLOn:
            return YoloEmbeddingModel(name)
        
        case EmbeddingModelNames.DINOV2_VITS14:
            return DinoEmbeddingModel(name)
            
        case EmbeddingModelNames.CLIP_VIT_B32:
            return ClipEmbeddingModel(name)

        case _:
            return TorchEmbeddingModel(name)


if __name__ == "__main__":
    for name in EmbeddingModelNames:
        model = get_model(name)
        print(name, model.embedding_size)
