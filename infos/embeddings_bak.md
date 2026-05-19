# Embedding models

Embedding models convert an image (or cropped bounding box) into a fixed-length float vector.
The `embedding` key in a configuration run selects the model.

All models auto-select GPU via `torch.accelerator.current_accelerator()` and fall back to CPU.
Model weights are resolved relative to the `src/` working directory.

---

## Standard pretrained models

These use ImageNet-pretrained torchvision backbones with the final classification layer removed.

| Type string | Class | Embedding size | Model file |
|---|---|---|---|
| `resnet50` | `TorchEmbeddingModel` | 2048 | Downloaded automatically |
| `vgg16` | `TorchEmbeddingModel` | 4096 | Downloaded automatically |
| `inception_v3` | `TorchEmbeddingModel` | 2048 | Downloaded automatically |
| `mobilenet_v3` | `TorchEmbeddingModel` | 1280 | Downloaded automatically |

**Example:**
```json
{ "type": "resnet50", "params": {} }
```

No configurable parameters. Weights are downloaded from `torchvision` on first use.

---

## DINOv2 ViT-S/14

**Type string:** `dinov2_vits14`  
**Class:** `DinoEmbeddingModel`  
**Embedding size:** 384

Self-supervised Vision Transformer loaded via `torch.hub` from `facebookresearch/dinov2`.
Input images are resized to 256 px, center-cropped to 224 px, and ImageNet-normalized.

**Example:**
```json
{ "type": "dinov2_vits14", "params": {} }
```

---

## CLIP ViT-B/32

**Type string:** `clip_vit_b32`  
**Class:** `ClipEmbeddingModel`  
**Embedding size:** 512

OpenAI CLIP image encoder loaded via `open_clip`. Output embeddings are L2-normalized to unit
length before storage, which makes cosine similarity equivalent to dot product.

**Example:**
```json
{ "type": "clip_vit_b32", "params": {} }
```

---

## YOLO embedding models

Graffiti-specific YOLO models. The backbone is used directly as a feature extractor via
`YOLO.embed()`. Model files must be present in `models/`.

| Type string | Class | Embedding size | Model file |
|---|---|---|---|
| `yolon` | `YoloEmbeddingModel` | 256 | `models/yolo26n.pt` |
| `yolos` | `YoloEmbeddingModel` | 512 | `models/yolo26s.pt` |
| `yolom` | `YoloEmbeddingModel` | 512 | `models/yolo26m.pt` |

**Example:**
```json
{ "type": "yolos", "params": {} }
```

---

## Fine-tuned graffiti projection heads

These wrap a frozen pretrained backbone with a two-layer projection head
(`Linear → ReLU → Linear`) trained for graffiti similarity. Output is L2-normalized.

### MobileNetV3 Graffiti Author Head

**Type string:** `mobilenet_v3_graffiti_author_head`  
**Embedding size:** 1280  
**Model file:** `models/mobilenet_graffiti_author_head.pth`

```json
{ "type": "mobilenet_v3_graffiti_author_head", "params": {} }
```

### DINOv2 Graffiti Author Head

**Type string:** `dinov2_graffiti_author_head`  
**Embedding size:** 384  
**Model file:** `models/dinov2_graffiti_author_head.pth`

```json
{ "type": "dinov2_graffiti_author_head", "params": {} }
```

---

## Custom model

Loads a `TorchEmbeddingModel` base architecture with user-supplied weights via `torch.load`.

**Type string:** `custom`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | *(required)* | Base model identifier. Must be a valid `EmbeddingModelNames` value (e.g. `"mobilenet_v3"`). |
| `weight_path` | `str` | *(required)* | Path to the `.pth` state-dict file. |

**Example:**
```json
{
  "type": "custom",
  "params": {
    "model": "mobilenet_v3",
    "weight_path": "models/my_weights.pth"
  }
}
```
