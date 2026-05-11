# Segmenters

Segmenters detect graffiti regions in each image and return bounding boxes. The pipeline crops
each detected region and generates an independent embedding per crop. The `segmenter` key in a
whitelist run selects the backend; omitting it is equivalent to using `"identity"`.

---

## YOLO

**Type string:** `yolo`

Runs a YOLO detection model on the image and returns the resulting bounding boxes above the
confidence threshold. Overlapping boxes are merged by IoU and an optional padding factor is
applied to expand each crop.

**Merging logic:** boxes are merged when their intersection area exceeds `merge_threshold`
relative to *either* box. Merging is applied iteratively in confidence-descending order.
Set `merge_threshold` to `1.0` to disable merging.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | *(required)* | Path to the YOLO `.pt` weights file, resolved from the `src/` working directory. |
| `threshold` | `float` | `0.5` | Confidence threshold; detections below this value are discarded. |
| `merge_threshold` | `float` | `0.8` | IoU fraction above which two boxes are merged into one. |
| `padding` | `float` | `0.0` | Fractional padding added to each box edge (relative to box width/height). Clamped to image bounds. |

**Example:**
```json
{
  "type": "yolo",
  "params": {
    "model_path": "models/yolo26s.pt",
    "threshold": 0.5,
    "merge_threshold": 0.8,
    "padding": 0.05
  }
}
```

---

## Identity (full image)

**Type string:** `identity`

Returns a single bounding box covering the entire image. No model is loaded. Use when
graffiti detection is not needed and the full image should be embedded as-is.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `padding` | `float` | `0.0` | Unused in practice (the bounding box already spans [0, 1] × [0, 1]). Accepted for interface compatibility. |

**Example:**
```json
{
  "type": "identity",
  "params": {}
}
```
