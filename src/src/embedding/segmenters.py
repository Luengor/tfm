from ultralytics import YOLO # type: ignore
from src.abstractions import SegmenterBase, BoundingBox
from PIL.Image import Image as PILImage

class YoloSegmenter(SegmenterBase):
    def __init__(self, model_path: str, threshold: float = 0.5, merge_threshold: float = 0.8, padding: float = 0.0, max_boxes_per_image: int | None = None, allow_full_image_fallback: bool = True, batch_size: int = 1, touch_merge_gap: float = 0.0, max_merged_area: float = 1.0):
        self.model = YOLO(model_path)
        self.model.eval()
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self._padding = padding
        self.max_boxes_per_image = max_boxes_per_image
        self.allow_full_image_fallback = allow_full_image_fallback
        self._batch_size = max(1, int(batch_size))
        self.touch_merge_gap = touch_merge_gap
        self.max_merged_area = max_merged_area

    @property
    def padding(self) -> float:
        return self._padding

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def segment(self, image: PILImage) -> list[BoundingBox]:
        results = self.model(image, conf=self.threshold)
        return self._postprocess(image, results[0])

    def segment_batch(self, images: list[PILImage]) -> list[list[BoundingBox]]:
        if not images:
            return []
        results = self.model(images, conf=self.threshold, verbose=False)
        return [self._postprocess(img, r) for img, r in zip(images, results)]

    def _postprocess(self, image: PILImage, result) -> list[BoundingBox]:
        width, height = image.size
        boxes: list[BoundingBox] = []
        for box in result.boxes:
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

        if self.max_boxes_per_image is not None and len(boxes) > self.max_boxes_per_image:
            boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)[:self.max_boxes_per_image]

        if self._padding > 0:
            boxes = [self._apply_padding(box) for box in boxes]

        if not boxes and self.allow_full_image_fallback:
            boxes = [BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0, confidence=1.0)]

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
        ix1 = max(box1.x1, box2.x1)
        iy1 = max(box1.y1, box2.y1)
        ix2 = min(box1.x2, box2.x2)
        iy2 = min(box1.y2, box2.y2)

        contain_merge = False
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
            area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
            contain_merge = (intersection_area / area1 > self.merge_threshold) or (intersection_area / area2 > self.merge_threshold)

        touch_merge = False
        if self.touch_merge_gap > 0:
            gap_x = max(0.0, max(box1.x1, box2.x1) - min(box1.x2, box2.x2))
            gap_y = max(0.0, max(box1.y1, box2.y1) - min(box1.y2, box2.y2))
            touch_merge = max(gap_x, gap_y) <= self.touch_merge_gap

        if not (contain_merge or touch_merge):
            return False

        if self.max_merged_area < 1.0:
            union_x1 = min(box1.x1, box2.x1)
            union_y1 = min(box1.y1, box2.y1)
            union_x2 = max(box1.x2, box2.x2)
            union_y2 = max(box1.y2, box2.y2)
            if (union_x2 - union_x1) * (union_y2 - union_y1) > self.max_merged_area:
                return False

        return True

class IdentitySegmenter(SegmenterBase):
    # `padding` is accepted and exposed only to satisfy the abstract
    # SegmenterBase.padding contract shared with YoloSegmenter; it has no effect
    # here since segment() always returns the full-image box.
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
