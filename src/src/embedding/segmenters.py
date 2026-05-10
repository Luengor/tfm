from ultralytics import YOLO # type: ignore
from src.abstractions import SegmenterBase, BoundingBox
from PIL.Image import Image as PILImage

class YoloSegmenter(SegmenterBase):
    def __init__(self, model_path: str, threshold: float = 0.5, merge_threshold: float = 0.8, padding: float = 0.0):
        self.model = YOLO(model_path)
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self._padding = padding

    @property
    def padding(self) -> float:
        return self._padding

    def segment(self, image: PILImage) -> list[BoundingBox]:
        width, height = image.size
        results = self.model(image, conf=self.threshold)
        boxes = []
        for r in results:
            for box in r.boxes:
                # box.xyxy returns [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append(BoundingBox(
                    x1=coords[0] / width,
                    y1=coords[1] / height,
                    x2=coords[2] / width,
                    y2=coords[3] / height,
                    confidence=conf
                ))

        if self.merge_threshold < 1.0:
            boxes = self._merge_boxes(boxes)

        if self._padding > 0:
            boxes = [self._apply_padding(box) for box in boxes]

        return boxes

    def _apply_padding(self, box: BoundingBox) -> BoundingBox:
        w = box.x2 - box.x1
        h = box.y2 - box.y1
        return BoundingBox(
            x1=max(0.0, box.x1 - w * self._padding),
            y1=max(0.0, box.y1 - h * self._padding),
            x2=min(1.0, box.x2 + w * self._padding),
            y2=min(1.0, box.y2 + h * self._padding),
            confidence=box.confidence
        )

    def _merge_boxes(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        if not boxes:
            return []

        # Sort by confidence descending
        sorted_boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
        merged = []

        while sorted_boxes:
            current = sorted_boxes.pop(0)
            to_merge = []
            remaining = []

            for other in sorted_boxes:
                if self._should_merge(current, other):
                    to_merge.append(other)
                else:
                    remaining.append(other)
            
            if to_merge:
                # Merge current with all in to_merge
                new_x1 = min(current.x1, *(b.x1 for b in to_merge))
                new_y1 = min(current.y1, *(b.y1 for b in to_merge))
                new_x2 = max(current.x2, *(b.x2 for b in to_merge))
                new_y2 = max(current.y2, *(b.y2 for b in to_merge))
                new_conf = max(current.confidence, *(b.confidence for b in to_merge))
                
                merged_box = BoundingBox(x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2, confidence=new_conf)
                # We put it back at the start to check if it needs more merging
                sorted_boxes = [merged_box] + remaining
            else:
                merged.append(current)
                sorted_boxes = remaining

        return merged

    def _should_merge(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        # Calculate intersection
        ix1 = max(box1.x1, box2.x1)
        iy1 = max(box1.y1, box2.y1)
        ix2 = min(box1.x2, box2.x2)
        iy2 = min(box1.y2, box2.y2)

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        # Merge if intersection is more than merge_threshold of EITHER box
        return (intersection_area / area1 > self.merge_threshold) or (intersection_area / area2 > self.merge_threshold)

class IdentitySegmenter(SegmenterBase):
    def __init__(self, padding: float = 0.0):
        self._padding = padding

    @property
    def padding(self) -> float:
        return self._padding

    def segment(self, image: PILImage) -> list[BoundingBox]:
        return [BoundingBox(
            x1=0.0,
            y1=0.0,
            x2=1.0,
            y2=1.0,
            confidence=1.0
        )]
