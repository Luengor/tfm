from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL.Image import Image as PILImage

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

@dataclass
class ImageData:
    filename: str
    embedding: list[float]
    bbox: BoundingBox | None = None


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
    def get_by_distance(self, embedding: list[float], max_images:int = -1, cos_distance: bool = True) -> list[ImageData]:
        ...

    @abstractmethod
    def set_metadata(self, key: str, value: str) -> None:
        ...

    @abstractmethod
    def get_metadata(self, key: str) -> str | None:
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

class SegmenterBase(ABC):
    @abstractmethod
    def segment(self, image: PILImage, threshold: float = 0.5) -> list[BoundingBox]:
        ...

class ClusteringBase(ABC):
    @abstractmethod
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        ...

class ReductionBase(ABC):
    @abstractmethod
    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        ...

