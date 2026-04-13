from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL.Image import Image as PILImage

@dataclass
class ImageData:
    filename: str
    embedding: list[float]

class StorageBase(ABC):
    @abstractmethod
    def save(self, data: ImageData) -> None:
        ...

    @abstractmethod
    def load(self, filename:str) -> ImageData:
        ...

    @abstractmethod
    def get_all_images(self) -> list[ImageData]:
        ...

    @abstractmethod
    def get_by_distance(self, embedding: list[float], cos_distance: bool = True) -> list[ImageData]:
        ...

class EmbeddingBase(ABC):
    @abstractmethod
    def gen_embedding(self, image: PILImage) -> list[float]:
        ...

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        ...

class ClusteringBase(ABC):
    @abstractmethod
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        ...

class ClusterCombo(EmbeddingBase, ClusteringBase):
    def __init__(self, embedding: EmbeddingBase, clustering: ClusteringBase):
        self.embedding = embedding
        self.clustering = clustering

    def gen_embedding(self, image: PILImage) -> list[float]:
        return self.embedding.gen_embedding(image)

    @property
    def embedding_size(self) -> int:
        return self.embedding.embedding_size

    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        return self.clustering.cluster(images, **kwargs)

class SiameseBase(EmbeddingBase):
    @abstractmethod
    def gen_embedding(self, image: PILImage) -> list[float]:
        ...

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        ...

    @abstractmethod
    def distance(self, image1: ImageData, image2: ImageData) -> float:
        ...

