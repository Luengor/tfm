# §8a — Supervised Validation on Style Crops: Report

**Source:** `exp/exp08a/output/benchmark_20260529T125933Z.json`
**Date:** 2026-05-29 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 96/96 runs `success`.

## Setup

Pillar 3 (supervised validation). Question: when ground-truth **style** labels
are available, which embedding × reduction × clustering pipeline best recovers
them — and in particular, does a graffiti-specific projection head beat the
pretrained backbone, and is the lift task-specific (style head vs. author head)?

Ground truth: [`data/style/eval_crop/`](../../data/style/eval_crop/) — 294
pre-segmented crops across 4 styles, held out from the disjoint style-head
training split. Segmenter is `identity` (required for extrinsic metrics); no
`limit` (full 294 each run).

Fixed / varied axes:

| Axis | Values |
|---|---|
| Embedding (8) | `dinov2_vits14`, `dinov2_graffiti_style_head`, `dinov2_graffiti_author_head`, `mobilenet_v3`, `mobilenet_v3_graffiti_style_head`, `mobilenet_v3_graffiti_author_head`, `clip_vit_b32`, `resnet50` |
| Reduction (2) | `identity` (raw space), UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Clustering (6) | `kmeans` k=4, `agglomerative` (avg) k=4, `spectral` k=4, `hdbscan` mcs∈{5,10,20} |
| Storage | SQLite · Segmenter | `identity` · Repeats | `K=3` |

8 × 2 × 6 = 96 runs. ARI is the headline (chance-corrected, comparable across
k); NMI flags the many-tiny-clusters inflation mode; pairwise F1 is the most
interpretable.

> **Reading note — the `*_no_noise` extrinsic variants inflate exactly when
> they should be distrusted.** The default ARI/NMI/F1 treat HDBSCAN noise (−1)
> as its own cluster; the `_no_noise` variants drop −1 points before scoring.
> At high noise the no-noise reading grades only the dense survivor core, not a
> partition: `mobilenet_v3_graffiti_style_head` identity hdb20 posts ARI 0.224
> but **ARI_no_noise 0.854 while discarding 57.5 % of points as noise**;
> `resnet50` identity hdb5 posts ARI 0.040 vs **ARI_no_noise 0.639 at 82.7 %
> noise**. The no-noise column must always be read jointly with `noise_ratio`.

## Summary — best pipeline per embedding (default ARI)

`best` = highest default ARI over the 12 (reduction × clustering) cells for that
embedding; `sil`/`NMI`/`F1` are that same winning cell.

| Embedding | best ARI | NMI | F1 | sil | winning cell |
|---|--:|--:|--:|--:|---|
| **`dinov2_graffiti_style_head`** | **0.698** | **0.653** | **0.790** | 0.310 | identity-kmeans4 |
| `mobilenet_v3_graffiti_style_head` | 0.560 | 0.474 | 0.693 | 0.166 | identity-spectral |
| `clip_vit_b32` | 0.368 | 0.392 | 0.556 | 0.047 | umap-agglomerative |
| `dinov2_vits14` | 0.356 | 0.374 | 0.558 | 0.135 | identity-kmeans4 |
| `mobilenet_v3` | 0.356 | 0.336 | 0.550 | 0.038 | umap-agglomerative |
| `resnet50` | 0.334 | 0.277 | 0.548 | 0.058 | identity-spectral |
| `mobilenet_v3_graffiti_author_head` | 0.267 | 0.256 | 0.513 | 0.079 | umap-hdbscan10 |
| `dinov2_graffiti_author_head` | 0.234 | 0.207 | 0.453 | 0.101 | umap-agglomerative |

The §8a answer is unambiguous: **`dinov2_graffiti_style_head` + identity +
KMeans-4 wins decisively** (ARI 0.698, F1 0.790) — roughly 2× the ARI of the
pretrained DINOv2 backbone (0.356) and of every off-the-shelf encoder.

## The central result: author heads *damage* style structure

The expected finding was "fine-tuned head beats backbone." The data shows
something sharper — the **style** head beats its backbone, but the **author**
head falls *below* its own pretrained backbone on the style task:

| Family | pretrained | style head | author head |
|---|--:|--:|--:|
| DINOv2 | 0.356 | **0.698** (+0.342) | 0.234 (−0.122) |
| MobileNetV3 | 0.356 | **0.560** (+0.204) | 0.267 (−0.089) |

(best default ARI per embedding.) The author head is not merely *less useful*
for style — it actively destroys style-discriminative geometry, dropping ARI
below the untuned backbone on both families. This is a strongly
diagonal-dominant transfer matrix: each head concentrates the axis it was
trained on and suppresses the other. It is the cleanest possible evidence that
the two heads encode genuinely different information rather than a shared
"graffiti exists" representation (hypothesis 2).

## Fine-tuned style head: reduction × clustering detail

`cls` = cluster count, `noise` = fraction −1, others as above. Both DINOv2- and
MobileNet-style heads peak in **raw (identity) space** under oracle-k=4
partitional clustering; UMAP costs a little and HDBSCAN collapses (next
section).

| emb | red | clu | cls | noise | sil | CH | DB | ARI | NMI | F1 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| dino-style | identity | kmeans4 | 4 | 0.00 | 0.310 | 172.7 | 1.26 | **0.698** | 0.653 | 0.790 |
| dino-style | identity | spectral | 4 | 0.00 | 0.312 | 169.4 | 1.23 | 0.673 | 0.652 | 0.776 |
| dino-style | identity | agglo | 4 | 0.00 | 0.332 | 133.0 | 1.40 | 0.630 | 0.638 | 0.762 |
| dino-style | umap | agglo | 4 | 0.00 | 0.313 | 169.6 | 1.24 | 0.658 | 0.644 | 0.765 |
| dino-style | umap | kmeans4 | 4 | 0.00 | 0.287 | 161.0 | 1.38 | 0.599 | 0.631 | 0.726 |
| dino-style | umap | hdb{5,10,20} | 3 | 0.00 | 0.406 | 193.2 | 1.00 | 0.632 | 0.647 | 0.764 |
| mob-style | identity | spectral | 4 | 0.00 | 0.166 | 57.5 | 1.98 | **0.560** | 0.474 | 0.693 |
| mob-style | umap | agglo | 4 | 0.00 | 0.114 | 51.9 | 2.55 | 0.556 | 0.491 | 0.689 |
| mob-style | identity | kmeans4 | 4 | 0.00 | 0.161 | 58.6 | 2.01 | 0.514 | 0.454 | 0.655 |

## HDBSCAN does not recover k≈4 on the fine-tuned style head

Hypothesis 4 expected HDBSCAN to auto-discover k near the true 4 on the
fine-tuned head. **It does not.** On `dinov2_graffiti_style_head`:

- **identity space:** every `mcs` collapses to **k=2** — a single binary split
  (mcs5/10) or a 2-cluster high-noise partition (mcs20, noise 0.35), never 4.
- **UMAP space:** all three `mcs` values produce the **identical k=3 partition**
  (ARI 0.632, sil 0.406, noise 0) — the sweep is degenerate; the knob has no
  effect because UMAP has already fused the 4 styles into 3 density basins.

So on the embedding where the true k=4 is most linearly recoverable (KMeans-4
hits 0.698), density-based auto-k still under-segments. The oracle-k partitional
methods are the right tool when k is known; HDBSCAN's value is elsewhere (the
unsupervised pipeline, §§1–4, where k is unknown).

## Repeat instability at small `mcs` under UMAP

As in §4, the unstable cells sit at the HDBSCAN collapse boundary —
`mcs=5` under UMAP flips cluster count across the K=3 seeds:

- `dinov2_graffiti_author_head` umap hdb5 → `clusters_per_repeat = [5, 11, 4]`
- `clip_vit_b32` umap hdb5 → `[2, 10, 13]`
- `mobilenet_v3` umap hdb5 → `[7, 13, 13]`

The aggregated row for these cells is a mean over qualitatively different
partitions and should not be read as a single result. Oracle-k partitional rows
(kmeans/agglo/spectral) are stable: `[4, 4, 4]` everywhere.

## Internal–external correlation

Pearson r(silhouette, ARI):

- **r = 0.66** over the 48 oracle-k=4 partitional runs (kmeans/agglo/spectral).
- **r = 0.40** over all 83 runs with valid ARI (adds the HDBSCAN rows).

Silhouette is a **defensible proxy for label agreement at fixed k**, but the
correlation weakens markedly once cluster count is allowed to vary — exactly the
regime (HDBSCAN) where silhouette peaks on degenerate low-k / high-noise cells,
mirroring the §2 and §4 hazard. On this 4-style label granularity, silhouette
tracks semantic quality only when k is pinned.

## Secondary signal — supervised retrieval (similarity search, top_k=5)

Similarity search runs over the raw stored embeddings (per-embedding, reduction-
independent), scored against the style labels. Precision@5 reproduces the
clustering ranking and is the highest-signal single number for the style head:

| Embedding | P@5 | MAP@5 | MRR |
|---|--:|--:|--:|
| `dinov2_graffiti_style_head` | **0.820** | 0.877 | 0.895 |
| `mobilenet_v3_graffiti_style_head` | 0.737 | 0.821 | 0.843 |
| `dinov2_vits14` | 0.699 | 0.805 | 0.837 |
| `mobilenet_v3` | 0.687 | 0.806 | 0.830 |
| `clip_vit_b32` | 0.669 | 0.792 | 0.825 |
| `dinov2_graffiti_author_head` | 0.669 | 0.776 | 0.802 |
| `resnet50` | 0.654 | 0.789 | 0.829 |
| `mobilenet_v3_graffiti_author_head` | 0.568 | 0.721 | 0.753 |

Same story: style heads on top, author heads at the bottom (the MobileNet
author head is the single worst retriever at P@5 0.568).

## Summary — cost

Cost is not the §8a deliverable (§§6–7/9 own it), and quality is invariant to
storage, but for completeness: **ingest dominates** and is paid once per
embedding (shared across that embedding's 12 cells):

| Embedding | ingest wall (s) | throughput (ips) | sim-search wall (s) |
|---|--:|--:|--:|
| dinov2 (×3 heads) | 11.4–12.5 | 23–26 | ~5.0 |
| mobilenet (×3 heads) | 8.3–12.5 | 23–36 | ~15.0 |
| clip_vit_b32 | 11.1 | 26.4 | ~6.3 |
| resnet50 | 9.6 | 30.7 | ~23.0 |

Reduction is trivial (UMAP ~0.35 s, identity ~0); clustering sub-0.02 s on 294
points. Similarity-search wall time scales with embedding **dimension** under
the SQLite linear scan (DINOv2 384-d ~5 s, ResNet50 2048-d ~23 s) — a property
of the storage backend, not the head; §6/§9 quantify it properly.

## Hypotheses — disposition

1. **Style head > backbone.** ✅ Confirmed on both families (DINOv2 +0.342 ARI,
   MobileNet +0.204). Held-out split ⇒ genuine transfer, not memorisation.
2. **Cross-task specificity.** ✅ Confirmed *more strongly* than predicted: the
   author head does not just under-help, it drops below the pretrained backbone
   on style (DINOv2 −0.122, MobileNet −0.089). Diagonal-dominant transfer.
3. **CLIP competitive / ResNet50 weakest.** ⚠ Partial. CLIP (0.368) ≈ DINOv2
   base (0.356) ✅; but ResNet50 (0.334) is **not** clearly the weakest
   off-the-shelf encoder — it sits with the DINOv2/MobileNet base tier. Refuted
   for ResNet50.
4. **HDBSCAN recovers k≈4 on the fine-tuned head.** ❌ Refuted. Identity space
   collapses to k=2; UMAP space gives an `mcs`-invariant k=3. Oracle-k
   partitional methods, not density auto-k, recover the 4 styles.
5. **identity ≥ UMAP on the fine-tuned head.** ✅ Confirmed. DINOv2-style
   identity-kmeans 0.698 > umap-kmeans 0.599; the head has already concentrated
   the discriminative axes, so further reduction is information loss.

## Conclusion for downstream experiments

1. **`dinov2_graffiti_style_head` is the right baseline embedding for the style
   task** and validates the pipeline-wide default used in §§1–9. Its win over
   every untuned encoder (ARI 0.698 vs ≤0.368) is the supervised anchor the
   unsupervised silhouette/CH/DB metrics lean on elsewhere.
2. **The head is task-specific, not generically "graffiti-aware."** The author
   head harms style recovery; §8b must show the symmetric result (author head
   wins author recovery) for the specificity claim to close.
3. **Silhouette is a usable proxy only at fixed k** (r=0.66) — when reporting
   unsupervised quality in §§1–4, pin k or read silhouette jointly with
   `noise_ratio`/`cluster_count`, never alone.
4. **Use oracle-k partitional clustering when k is known.** HDBSCAN under-
   segments even the cleanly-separable fine-tuned head; its role is the
   unknown-k unsupervised setting, not label recovery.

**Degenerate cells (all-noise, no metrics):** `mobilenet_v3`,
`mobilenet_v3_graffiti_author_head` (all three identity-HDBSCAN), and
`dinov2_graffiti_author_head` / `clip_vit_b32` / `resnet50` (identity HDBSCAN
mcs10/20) collapse to 0 clusters (noise 1.0) — HDBSCAN finds no density
structure in those raw spaces. Listed here once; excluded from all tables above.
