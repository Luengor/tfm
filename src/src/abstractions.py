from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from PIL.Image import Image as PILImage

@dataclass
class BoundingBox:
    x1: float = 0
    y1: float = 0
    x2: float = 1
    y2: float = 1
    confidence: float = 1

@dataclass
class ImageData:
    filename: str
    embedding: list[float]
    bbox: BoundingBox = field(default_factory=BoundingBox)


class StorageBase(ABC):
    @abstractmethod
    def save(self, data: ImageData) -> None:
        ...

    @abstractmethod
    def load(self, filename:str) -> list[ImageData]:
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

    def close(self) -> None:
        pass

class EmbeddingBase(ABC):
    @abstractmethod
    def gen_embedding(self, image: PILImage) -> list[float]:
        ...

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        ...

class SegmenterBase(ABC):
    @property
    @abstractmethod
    def padding(self) -> float:
        ...

    @abstractmethod
    def segment(self, image: PILImage) -> list[BoundingBox]:
        ...

    @property
    def batch_size(self) -> int:
        return 1

    def segment_batch(self, images: list[PILImage]) -> list[list[BoundingBox]]:
        return [self.segment(image) for image in images]

class ClusteringBase(ABC):
    @abstractmethod
    def cluster(self, images: list[ImageData], **kwargs) -> list[int]:
        ...

class ReductionBase(ABC):
    @abstractmethod
    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        ...

