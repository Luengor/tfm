# §8a — Supervised Validation on Style Crops: Report

**Source:** `exp/exp08a/output/benchmark_20260529T125933Z.json` (96 base runs +
8 follow-up runs merged on 2026-05-31 for the `umap × hdbscan mcs=50` cell),
`exp/exp08a/output/benchmark_20260601T152849Z.json` (author heads, re-run
on 2026-06-01; supersedes the 8 cells per author head that were rerun —
agglomerative + HDBSCAN mcs∈{5,10,20} under identity and UMAP; kmeans /
spectral / mcs=50 author-head cells kept from the 29-May benchmark) and
`exp/exp08a/output/benchmark_20260602T095234Z.json` (style heads, re-run on
2026-06-02; supersedes all 12 cells per style head plus the `umap × mcs=50`
follow-up cell — 26 cells total).
**Date:** 2026-05-29 (base) · 2026-05-31 (mcs=50 follow-up) · 2026-06-01 (author-head re-run) · 2026-06-02 (style-head re-run)
**Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 104/104 + 16/16 + 26/26 runs `success`.

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
| Clustering (7) | `kmeans` k=4, `agglomerative` (avg) k=4, `spectral` k=4, `hdbscan` mcs∈{5,10,20,50} |
| Storage | SQLite · Segmenter | `identity` · Repeats | `K=3` |

96 base runs (8 × 2 × 6) + 8 follow-up runs (`umap × hdbscan mcs=50` only,
identity reduction not retested) = **104 runs**. The follow-up cell tests the
hypothesis that a larger `mcs` could push HDBSCAN towards the supervised k=4
basin (motivated by the §4 collapse pattern at `mcs=50`/n=6416 → k=4 on the
full corpus). ARI is the headline (chance-corrected, comparable across k); NMI
flags the many-tiny-clusters inflation mode; pairwise F1 is the most
interpretable.

> **Reading note — the `*_no_noise` extrinsic variants inflate exactly when
> they should be distrusted.** The default ARI/NMI/F1 treat HDBSCAN noise (−1)
> as its own cluster; the `_no_noise` variants drop −1 points before scoring.
> At high noise the no-noise reading grades only the dense survivor core, not a
> partition: `dinov2_graffiti_style_head` identity hdb20 posts ARI 0.422 but
> **ARI_no_noise 0.769 while discarding 34.7 % of points as noise**;
> `resnet50` identity hdb5 posts ARI 0.040 vs **ARI_no_noise 0.639 at 82.7 %
> noise**. The no-noise column must always be read jointly with `noise_ratio`.

## Summary — best pipeline per embedding (default ARI)

`best` = highest default ARI over the 12 (reduction × clustering) cells for that
embedding; `sil`/`NMI`/`F1` are that same winning cell.

| Embedding | best ARI | NMI | F1 | sil | winning cell |
|---|--:|--:|--:|--:|---|
| **`dinov2_graffiti_style_head`** | **0.722** | **0.678** | **0.806** | 0.284 | identity-kmeans4 |
| `mobilenet_v3_graffiti_style_head` | 0.627 | 0.539 | 0.740 | 0.272 | identity-kmeans4 |
| `clip_vit_b32` | 0.368 | 0.392 | 0.556 | 0.047 | umap-agglomerative |
| `dinov2_vits14` | 0.356 | 0.374 | 0.558 | 0.135 | identity-kmeans4 |
| `mobilenet_v3` | 0.356 | 0.336 | 0.550 | 0.038 | umap-agglomerative |
| `resnet50` | 0.334 | 0.277 | 0.548 | 0.058 | identity-spectral |
| `mobilenet_v3_graffiti_author_head` | 0.318 | 0.295 | 0.517 | 0.046 | umap-agglomerative |
| `dinov2_graffiti_author_head` | 0.302 | 0.273 | 0.519 | 0.094 | umap-agglomerative |

The §8a answer is unambiguous: **`dinov2_graffiti_style_head` + identity +
KMeans-4 wins decisively** (ARI 0.722, F1 0.806) — roughly 2× the ARI of the
pretrained DINOv2 backbone (0.356) and of every off-the-shelf encoder. After
the 2026-06-02 re-run the MobileNet style head's winning cell also flips from
identity-spectral (0.560) to identity-kmeans4 (0.627); both style heads now
peak under the same partitional clusterer.

## The central result: author heads underperform on style

The expected finding was "fine-tuned head beats backbone." The data shows the
sharper diagonal pattern — the **style** head beats its backbone substantially,
while the **author** head falls *modestly* below its own pretrained backbone on
the style task:

| Family | pretrained | style head | author head |
|---|--:|--:|--:|
| DINOv2 | 0.356 | **0.722** (+0.366) | 0.302 (−0.054) |
| MobileNetV3 | 0.356 | **0.627** (+0.271) | 0.318 (−0.038) |

(best default ARI per embedding.) After the 2026-06-01 author-head re-run and
the 2026-06-02 style-head re-run, the author head is *less useful* than its
backbone for style on both families, but the gap is much smaller than the
style head's lift in the other direction (DINOv2 6.8×, MobileNet 7.1×). This
is a diagonal-dominant transfer matrix with **asymmetric magnitudes**: each
head concentrates the axis it was trained on, and the style head's lift is
large while the author head's drag is modest. It remains the cleanest
evidence that the two heads encode different information rather than a shared
"graffiti exists" representation (hypothesis 2).

## Fine-tuned style head: reduction × clustering detail

`cls` = cluster count, `noise` = fraction −1, others as above. Both DINOv2- and
MobileNet-style heads peak in **raw (identity) space** under oracle-k=4
partitional clustering; UMAP costs a little and HDBSCAN collapses (next
section).

| emb | red | clu | cls | noise | sil | CH | DB | ARI | NMI | F1 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| dino-style | identity | kmeans4 | 4 | 0.00 | 0.284 | 143.7 | 1.35 | **0.722** | 0.678 | 0.806 |
| dino-style | identity | spectral | 4 | 0.00 | 0.283 | 141.1 | 1.33 | 0.719 | 0.696 | 0.806 |
| dino-style | identity | agglo | 4 | 0.00 | 0.311 | 112.1 | 1.31 | 0.645 | 0.668 | 0.772 |
| dino-style | umap | agglo | 4 | 0.00 | 0.280 | 137.8 | 1.37 | 0.708 | 0.682 | 0.799 |
| dino-style | umap | kmeans4 | 4 | 0.00 | 0.251 | 121.9 | 1.63 | 0.684 | 0.654 | 0.783 |
| dino-style | umap | hdb{5,10} | 3 | 0.00 | 0.379 | 160.7 | 0.97 | 0.652 | 0.690 | 0.779 |
| dino-style | umap | hdb{20,50} | 2 | 0.00 | 0.355 | 197.5 | 1.16 | 0.569 | 0.593 | 0.732 |
| mob-style | identity | kmeans4 | 4 | 0.00 | 0.272 | 105.8 | 1.42 | **0.627** | 0.539 | 0.740 |
| mob-style | umap | kmeans4 | 4 | 0.00 | 0.270 | 103.5 | 1.42 | 0.612 | 0.531 | 0.731 |
| mob-style | identity | spectral | 4 | 0.00 | 0.270 | 103.5 | 1.41 | 0.613 | 0.549 | 0.733 |
| mob-style | umap | spectral | 4 | 0.00 | 0.271 | 103.8 | 1.41 | 0.610 | 0.529 | 0.730 |
| mob-style | umap | agglo | 4 | 0.00 | 0.271 | 103.6 | 1.41 | 0.609 | 0.526 | 0.730 |
| mob-style | identity | agglo | 4 | 0.00 | 0.257 | 97.1 | 1.48 | 0.561 | 0.495 | 0.698 |

## HDBSCAN does not recover k≈4 on the fine-tuned style head

Hypothesis 4 expected HDBSCAN to auto-discover k near the true 4 on the
fine-tuned head. **It does not.** On `dinov2_graffiti_style_head`:

- **identity space:** mcs=5 produces k=3 with 26.5 % noise (ARI 0.469,
  ARI_no_noise 0.750); mcs=10 produces a k=2 partition with low noise (6.1 %)
  but poor ARI (0.103); mcs=20 produces k=2 with 34.7 % noise (ARI 0.422,
  ARI_no_noise 0.769). None reach k=4.
- **UMAP space:** mcs=5 and mcs=10 produce the **identical k=3 partition**
  (ARI 0.652, sil 0.379, noise 0) — the knob has no effect at small mcs
  because UMAP has already fused the 4 styles into 3 density basins. mcs=20
  drops to **k=2** (ARI 0.569, sil 0.355, noise 0).
- **UMAP + `mcs=50` (follow-up):** lands on the **same k=2 partition as
  mcs=20** (ARI 0.569, sil 0.355). Raising `mcs` past the §4 collapse ceiling
  does not push HDBSCAN towards k=4 — it consolidates the k=2 binary split.
  This is the controlled test of the §4 hypothesis on a labelled subset: the
  `mcs=50`/k=4 cell observed on the full corpus (§4, n=6416) is **not** a
  recovery of the supervised style basins but a coincidence of cardinality at
  the collapse boundary.

So on the embedding where the true k=4 is most linearly recoverable (KMeans-4
hits 0.722), density-based auto-k still under-segments at every `mcs` swept
({5, 10, 20, 50}). The oracle-k partitional methods are the right tool when k
is known; HDBSCAN's value is elsewhere (the unsupervised pipeline, §§1–4,
where k is unknown).

### `mcs=50` across the whole panel

The follow-up cell also clarifies the wider behaviour of large `mcs` on a
small (n=294) labelled set. Per-embedding results under `umap × hdbscan mcs=50`:

| Embedding | k | sil | noise | ARI |
|---|--:|--:|--:|--:|
| `dinov2_graffiti_style_head` | 2 | 0.355 | 0.000 | 0.569 |
| `mobilenet_v3_graffiti_style_head` | 2 | 0.306 | 0.000 | 0.473 |
| `dinov2_vits14` | 2 | 0.149 | 0.156 | 0.324 |
| `mobilenet_v3` | 2 | 0.055 | 0.059 | 0.284 |
| `resnet50` | 2 | 0.109 | 0.448 | 0.185 |
| `dinov2_graffiti_author_head` | 2 | 0.153 | 0.278 | 0.122 |
| `clip_vit_b32` | 0 | — | 1.000 | — |
| `mobilenet_v3_graffiti_author_head` | 0 | — | 1.000 | — |

Three regimes. (i) Fine-tuned **style** heads keep their ranking: both
survive the larger `mcs` with zero noise (after the 2026-06-02 re-run the
MobileNet style head's UMAP partition is fully assigned at mcs=50, no longer
12 % noise) and the highest ARI under this cell. (ii) Off-the-shelf encoders
collapse to k=2 but with noticeable noise (15–45 %), so their default-ARI /
no-noise-ARI split widens — the no-noise reading must be paired with the
noise column to avoid the §8a `silhouette without noise` trap. (iii)
`clip_vit_b32` and `mobilenet_v3_graffiti_author_head` join the all-noise
list at `mcs=50` in UMAP space, mirroring their identity-HDBSCAN collapses.
Across the panel, `mcs=50` never recovers the k=4 supervised basin and never
beats the per-embedding best cell; it is uniformly dominated by oracle-k
partitional clustering.

## Repeat instability at small `mcs` under UMAP

As in §4, the unstable cells sit at the HDBSCAN collapse boundary —
`mcs=5` under UMAP flips cluster count across the K=3 seeds:

- `clip_vit_b32` umap hdb5 → `[2, 10, 13]`
- `mobilenet_v3` umap hdb5 → `[7, 13, 13]`
- `mobilenet_v3_graffiti_style_head` umap hdb5/hdb10 → `[3, 2, 3]` (post
  2026-06-02 re-run — mild flip at the k=2/k=3 boundary).
- (post 2026-06-01 author-head re-run: `dinov2_graffiti_author_head` umap
  hdb5 now `[3, 4, 3]` — no longer in the wildly-unstable bucket.)

The aggregated row for these cells is a mean over qualitatively different
partitions and should not be read as a single result. Oracle-k partitional rows
(kmeans/agglo/spectral) are stable: `[4, 4, 4]` everywhere.

## Internal–external correlation

Pearson r(silhouette, ARI), recomputed over the merged 104-cell dataset
(29-May base ← 01-Jun author-head re-run ← 02-Jun style-head re-run):

- **r = 0.781** over the 48 oracle-k=4 partitional runs (kmeans/agglo/spectral).
- **r = 0.545** over all 89 runs with valid ARI (adds the HDBSCAN rows; n
  grew from 83 → 89 because the MobileNet-style identity-HDBSCAN cells and
  some author-head HDBSCAN cells exited the all-noise set after the re-runs).

Silhouette is a **defensible proxy for label agreement at fixed k** (r≈0.78),
but the correlation weakens once cluster count is allowed to vary — exactly
the regime (HDBSCAN) where silhouette peaks on degenerate low-k / high-noise
cells, mirroring the §2 and §4 hazard. On this 4-style label granularity,
silhouette tracks semantic quality best when k is pinned.

## Secondary signal — supervised retrieval (similarity search, top_k=5)

Similarity search runs over the raw stored embeddings (per-embedding, reduction-
independent), scored against the style labels. Precision@5 reproduces the
clustering ranking and is the highest-signal single number for the style head:

| Embedding | P@5 | MAP@5 | MRR |
|---|--:|--:|--:|
| `dinov2_graffiti_style_head` | **0.851** | 0.909 | 0.922 |
| `mobilenet_v3_graffiti_style_head` | 0.760 | 0.818 | 0.835 |
| `dinov2_vits14` | 0.699 | 0.805 | 0.837 |
| `mobilenet_v3` | 0.687 | 0.806 | 0.830 |
| `dinov2_graffiti_author_head` | 0.674 | 0.804 | 0.832 |
| `clip_vit_b32` | 0.669 | 0.792 | 0.825 |
| `resnet50` | 0.654 | 0.789 | 0.829 |
| `mobilenet_v3_graffiti_author_head` | 0.639 | 0.775 | 0.799 |

Same story: style heads on top, author heads at the bottom (the MobileNet
author head is the single worst retriever at P@5 0.639). DINOv2 author head
edges past CLIP at P@5 after the re-run, so the retrieval ranking is now
slightly more spread than the clustering one.

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

1. **Style head > backbone.** ✅ Confirmed on both families (DINOv2 +0.366 ARI,
   MobileNet +0.271). Held-out split ⇒ genuine transfer, not memorisation.
2. **Cross-task specificity.** ✅ Confirmed: the author head drops below the
   pretrained backbone on style (DINOv2 −0.054, MobileNet −0.038 after the
   2026-06-01 re-run). Diagonal-dominant transfer with **asymmetric
   magnitude**: the style head's +0.366 / +0.271 lift on its trained axis far
   exceeds the author head's −0.054 / −0.038 drag on the other (6.8× / 7.1×).
3. **CLIP competitive / ResNet50 weakest.** ⚠ Partial. CLIP (0.368) ≈ DINOv2
   base (0.356) ✅; but ResNet50 (0.334) is **not** clearly the weakest
   off-the-shelf encoder — it sits with the DINOv2/MobileNet base tier. Refuted
   for ResNet50.
4. **HDBSCAN recovers k≈4 on the fine-tuned head.** ❌ Refuted at every `mcs`
   swept. Identity space lands on k=3 (mcs=5) or k=2 (mcs=10/20); UMAP space
   gives an `mcs`-invariant k=3 for `mcs∈{5,10}` and drops to k=2 at
   `mcs∈{20,50}` (ARI 0.569). The `mcs=50` follow-up specifically rules out
   the §4 hypothesis that a larger `mcs` could land on the supervised k=4
   basin: it lands on the same k=2 partition as `mcs=20`. Oracle-k partitional
   methods, not density auto-k, recover the 4 styles.
5. **identity ≥ UMAP on the fine-tuned head.** ✅ Confirmed. DINOv2-style
   identity-kmeans 0.722 > umap-kmeans 0.684; MobileNet-style identity-kmeans
   0.627 > umap-kmeans 0.612. The head has already concentrated the
   discriminative axes, so further reduction is information loss.

## Conclusion for downstream experiments

1. **`dinov2_graffiti_style_head` is the right baseline embedding for the style
   task** and validates the pipeline-wide default used in §§1–9. Its win over
   every untuned encoder (ARI 0.722 vs ≤0.368) is the supervised anchor the
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

**Degenerate cells (all-noise, no metrics):** `mobilenet_v3`
(all three identity-HDBSCAN), `mobilenet_v3_graffiti_author_head`
(identity-HDBSCAN mcs=20 — after the 2026-06-01 re-run mcs=5/10 produce
high-noise partitions instead of full collapse), `clip_vit_b32` /
`resnet50` (identity HDBSCAN mcs10/20). The two author-head identity-HDBSCAN
mcs=5/10 cells, which were all-noise in the 29-May benchmark, now post
high-noise (65–81 %) k=2–4 partitions after the re-run and are no longer
fully degenerate. After the 2026-06-02 style-head re-run, the two
`mobilenet_v3_graffiti_style_head` identity-HDBSCAN cells (mcs=5/10/20) also
exit the degenerate set entirely — they now produce k=2/k=3 partitions with
26–32 % noise and ARI 0.42–0.46. The `mcs=50` follow-up adds two more cells:
`clip_vit_b32` and `mobilenet_v3_graffiti_author_head` under UMAP collapse to
all-noise at `mcs=50` (too restrictive once those embeddings' UMAP density
basins are smaller than 50 points). Listed here once; excluded from all
tables above.
