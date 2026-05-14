import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.hub")
warnings.filterwarnings("ignore", category=UserWarning, module="open_clip")

# ruff: noqa: E402
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
from torchvision import models
import torch
import torch.nn as nn
import os
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
    MOBILENET_V3_GRAFFITI_HEAD = "mobilenet_v3_graffiti_head"
    DINOV2_VITS14 = "dinov2_vits14"
    DINOV2_GRAFFITI_AUTHOR_HEAD = "dinov2_graffiti_author_head"
    DINOV2_GRAFFITI_STYLE_HEAD = "dinov2_graffiti_style_head"
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
        'embedding_size': 1280,
    },
    EmbeddingModelNames.MOBILENET_V3_GRAFFITI_HEAD: {
        'embedding_size': 1280,
        'weights_path': "models/mobilenet_graffiti_head.pth"
    },
    EmbeddingModelNames.DINOV2_VITS14: {
        'embedding_size': 384,
    },
    EmbeddingModelNames.DINOV2_GRAFFITI_AUTHOR_HEAD: {
        'embedding_size': 384,
        'weights_path': "models/dinov2_graffiti_head.pth"
    },
    EmbeddingModelNames.DINOV2_GRAFFITI_STYLE_HEAD: {
        'embedding_size': 384,
        'weights_path': "models/dinov2_graffiti_style_head.pth"
    },
    EmbeddingModelNames.CLIP_VIT_B32: {
        'embedding_size': 512,
    },
    EmbeddingModelNames.YOLOn: {
        'model': 'models/yolo26n.pt',
        'embedding_size': 256,
    },
    EmbeddingModelNames.YOLOs: {
        'model': 'models/yolo26s.pt',
        'embedding_size': 512,
    },
    EmbeddingModelNames.YOLOm: {
        'model': 'models/yolo26m.pt',
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
        self.model.eval()

    def gen_embedding(self, image: ImageImage) -> list[float]:
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
        
        # Load custom weights if available
        m_data = MODELS[name]
        if 'weights_path' in m_data and os.path.exists(m_data['weights_path']):
            print(f"Loading custom weights for {name} from {m_data['weights_path']}")
            self.model.load_state_dict(torch.load(m_data['weights_path'], map_location=self.device))
        
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

class HeadEmbeddingModel(EmbeddingBase):
    def __init__(self, name: EmbeddingModelNames, base_model, preprocessor, input_dim):
        self.name = name
        self.device = get_device()
        self.base_model = base_model
        self.preprocessor = preprocessor
        
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        m_data = MODELS[name]
        if os.path.exists(m_data['weights_path']):
            print(f"Loading custom weights for {name} from {m_data['weights_path']}")
            state_dict = torch.load(m_data['weights_path'], map_location=self.device)
            
            base_state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items() if k.startswith('base_model.')}
            head_state_dict = {k.replace('projection_head.', ''): v for k, v in state_dict.items() if k.startswith('projection_head.')}
            
            self.base_model.load_state_dict(base_state_dict)
            self.projection_head.load_state_dict(head_state_dict)
            
        self.base_model.to(self.device)
        self.projection_head.to(self.device)
        self.base_model.eval()
        self.projection_head.eval()

    def gen_embedding(self, image: ImageImage) -> list[float]:
        with torch.no_grad():
            input_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
            features = self.base_model(input_tensor)
            output = self.projection_head(features)
            output = torch.nn.functional.normalize(output, p=2, dim=-1)
        return output.squeeze().cpu().tolist()

    @property
    def embedding_size(self) -> int:
        return MODELS[self.name]['embedding_size']

def get_model(name: EmbeddingModelNames) -> EmbeddingBase:
    match name:
        case EmbeddingModelNames.YOLOm | EmbeddingModelNames.YOLOs | EmbeddingModelNames.YOLOn:
            return YoloEmbeddingModel(name)
        
        case EmbeddingModelNames.MOBILENET_V3_GRAFFITI_HEAD:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            base = models.mobilenet_v3_large(weights=weights)
            base.classifier[3] = nn.Identity()
            return HeadEmbeddingModel(name, base, weights.transforms(), input_dim=1280)
        
        case EmbeddingModelNames.DINOV2_VITS14:
            return DinoEmbeddingModel(name)
        
        case EmbeddingModelNames.DINOV2_GRAFFITI_AUTHOR_HEAD | EmbeddingModelNames.DINOV2_GRAFFITI_STYLE_HEAD:
            base = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            from torchvision import transforms
            preprocessor = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            return HeadEmbeddingModel(name, base, preprocessor, input_dim=384)
            
        case EmbeddingModelNames.CLIP_VIT_B32:
            return ClipEmbeddingModel(name)

        case _:
            return TorchEmbeddingModel(name)


if __name__ == "__main__":
    for name in EmbeddingModelNames:
        model = get_model(name)
        print(name, model.embedding_size)
