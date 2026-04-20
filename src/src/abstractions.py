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
    def has(self, filename: str) -> bool:
        ...

    @abstractmethod
    def get_all_images(self) -> list[ImageData]:
        ...

    @abstractmethod
    def get_by_distance(self, embedding: list[float], cos_distance: bool = True) -> list[ImageData]:
        ...

    def clear(self) -> None:
        raise NotImplementedError("Storage backend does not implement clear().")

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

