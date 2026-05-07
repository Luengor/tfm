from ultralytics import YOLO # type: ignore
from src.abstractions import SegmenterBase, BoundingBox
from PIL.Image import Image as PILImage

class YoloSegmenter(SegmenterBase):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def segment(self, image: PILImage, threshold: float = 0.5) -> list[BoundingBox]:
        results = self.model(image, conf=threshold)
        boxes = []
        for r in results:
            for box in r.boxes:
                # box.xyxy returns [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append(BoundingBox(
                    x1=coords[0],
                    y1=coords[1],
                    x2=coords[2],
                    y2=coords[3],
                    confidence=conf
                ))
        return boxes

class IdentitySegmenter(SegmenterBase):
    def segment(self, image: PILImage, threshold: float = 0.5) -> list[BoundingBox]:
        width, height = image.size
        return [BoundingBox(
            x1=0.0,
            y1=0.0,
            x2=float(width),
            y2=float(height),
            confidence=1.0
        )]
