<!-- LTeX: language=en-US -->

# Unified Big Report — All Experiments (§§1–9)

**Purpose.** Single consolidated source for writing the *Results and Discussion*
chapter of the TFM. Combines proposal objectives ([`doc/enunciado.md`](../doc/enunciado.md)),
the experiment catalogue ([`infos/experiments.md`](../infos/experiments.md)) and
the nine individual reports ([`exp/exp{01..09}/report.md`](.)) into one
structured document.

**Sources (raw):**

| Exp | Report | Benchmark JSON |
|---|---|---|
| §1 | `exp/exp01/report.md` | `exp/exp01/output/benchmark_20260528T231810Z.json` |
| §2 | `exp/exp02/report.md` | `exp/exp02/output/benchmark_20260528T093147Z.json` |
| §3 | `exp/exp03/report.md` | `exp/exp03/output/benchmark_20260527T155755Z.json` |
| §4 | `exp/exp04/report.md` | `exp/exp04/output/benchmark_20260527T133844Z.json` |
| §5 | `exp/exp05/report.md` | `exp/exp05/output/benchmark_20260529T103834Z.json` |
| §6 | `exp/exp06/report.md` | `exp/exp06/output/benchmark_20260529T172522Z.json` |
| §7 | `exp/exp07/report.md` | `exp/exp07/output/benchmark_20260529T225953Z.json` |
| §8a | `exp/exp08a/report.md` | `exp/exp08a/output/benchmark_20260529T125933Z.json` |
| §8b | `exp/exp08b/report.md` | `exp/exp08b/output/benchmark_20260529T145725Z.json` |
| §9 | `exp/exp09/report.md` | `exp/exp09/output/benchmark_20260529T180633Z.json` |

All runs executed on the same host: **FullCreamMilk (Linux, 16 logical CPU,
Python 3.14)**. Aggregate execution status: **370/371 runs `success`** (1
failure in §9, `hnsw_m64_efc64_efs40`, due to pgvector constraint
`ef_construction ≥ 2·m`).

---

## 1. Cross-reference to the TFM brief

[`doc/enunciado.md`](../doc/enunciado.md) (TFM21) defines three deliverables
that the catalogue maps onto three *pillars*:

| Brief deliverable | Catalogue pillar | Experiments |
|---|---|---|
| Caracterización del **coste computacional** en función del tamaño del conjunto de imágenes, de forma experimental | (1) Computational cost | §6 storage, §7 scalability (HEADLINE), §9 HNSW tuning |
| Estudio de las agrupaciones con **métricas no supervisadas** | (2) Unsupervised quality | §1 embedding, §2 reduction, §3 clustering, §4 mcs tuning, §5 segmenter |
| **Evaluación supervisada** en un subconjunto etiquetado | (3) Supervised validation | §8a style (294 crops, 4 classes), §8b author (185 crops, 87 classes) |

The brief explicitly stresses the *technical* framing — no commentary on
graffiti as a phenomenon, only the imaging / ML side. The §7 headline
characterisation is the brief's primary ask; §6 and §9 quantify the
infrastructure-side trade-offs of similarity search; §§1–5 fix the quality
pipeline; §8 anchors the unsupervised metrics against ground truth.

---

## 2. Baseline pipeline (constant across experiments)

Every experiment varies **one axis** around the same baseline so comparisons
are clean:

| Component | Baseline value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style-discriminative projection head, L2-normalised output, 384-d) |
| Reduction | UMAP, `n_components=10`, `n_neighbors=15`, `min_dist=0.0`, `metric="cosine"` |
| Clustering | HDBSCAN, `min_cluster_size=5` |
| Segmenter | `identity` (full image — isolates the segmenter question to §5) |
| Storage | SQLite (zero-dependency default; replaced by PostgreSQL+pgvector for §6/§9) |
| Corpus sweep | `limit ∈ {250, 1000, 6416}` for quality experiments (§§1–5, §8); `limit ∈ {500, 1000, 2500, 6416}` for §6; `limit ∈ {250, 500, 1000, 2000, 3500, 6416}` for §7; `limit = 6416` fixed for §9 |
| Repeats | `K=3` for quality experiments (§§1–5, §8); `K=5` for cost experiments (§§6, §7, §9) |

### Dataset

- **Working corpus:** 6 416 images.
- **§8a Style eval:** [`data/style/eval_crop/`](../data/style/eval_crop/) —
  294 crops across 4 styles, held out from the disjoint style-head training
  split (`data/style/train_crop/`: 385 crops = 110 Salamanca + 275 Cuenca,
  trained via supervised contrastive loss).
- **§8b Author eval:** [`data/author/eval_crop/`](../data/author/eval_crop/) —
  185 crops across 87 authors, drawn from the Salamanca primary dataset. The
  author heads were trained on a **disjoint 169-crop split from the Cuenca
  (stopgrafiti) dataset** via triplet loss — §8b is therefore a *cross-city*
  held-out test. Class distribution: 53 singletons, 17 pairs, top author 19
  crops (mean class size ≈ 2.1).

### Repeat protocol

Reduction + clustering loop K times in the same process with seeds `0..K-1`.

- For **wall time / CPU time**: iter 0 is dropped (UMAP/numba JIT + CUDA cache
  warm-up); the remaining K−1 samples aggregate as mean ± std.
- For **memory (RSS, VRAM)**: all K samples are kept (after iter 0 the
  allocator holds pages, so per-iter deltas under-report the true peak).
- For **clustering quality**: every iteration is kept (quality is unaffected by
  JIT warm-up) and aggregated as mean ± std under
  `clustering_quality_*` / `clustering_quality_*_std` / `clustering_quality_*_n`.
- **Ingest** and **similarity_search** are single-shot per run; the metadata
  table persists ingest metrics so re-used DBs report the original time.

### Component implementations referenced

| Abstract | Implementations actually swept |
|---|---|
| `StorageBase` | `SQLiteStorage`, `PostgreSQLStorage` (exact, HNSW with cosine/L2/IP/both index ops) |
| `EmbeddingBase` | ResNet50, VGG16, InceptionV3, MobileNetV3, DINOv2 ViT-S/14, CLIP ViT-B/32, YOLO (n,m), and four heads: `mobilenet_v3_graffiti_{author,style}_head`, `dinov2_graffiti_{author,style}_head` |
| `SegmenterBase` | `IdentitySegmenter`, `YoloSegmenter` with `yolo11{s,m}.pt` and `yolo11m-train-10.pt` (graffiti-fine-tuned) |
| `ReductionBase` | Identity, PCA-10, PCA-50, UMAP-10, UMAP-50, Isomap-10 |
| `ClusteringBase` | HDBSCAN (`mcs∈{2,3,5,10,20,25,50,100,200}`), KMeans (auto-k via elbow), GMM diag (BIC, `max_k∈{30,100,300}`), DBSCAN (auto-eps + eps ∈ {0.2, 0.5, 1.0}), OPTICS (cosine), Agglomerative (avg linkage, k ∈ {10, 20, 30, 87}), Spectral (nearest-neighbors, k ∈ {10, 20}) |

---

## 3. Metric panel (uniform across experiments)

### 3.1 Computational cost

| Metric | Definition |
|---|---|
| `{stage}_wall_time_s` | Mean wall time per stage (ingest, reduction, clustering, similarity_search). For repeated stages, iter-0 dropped, mean over K-1. |
| `{stage}_wall_time_s_std`, `{stage}_n` | Std and sample count siblings. |
| `{stage}_cpu_time_s` | Mean process CPU time per stage. |
| `{stage}_peak_rss_delta_mb` | Peak RSS **above the stage's own baseline** (stage-net), aggregated across all K iters (memory pages don't return after iter 0). |
| `{stage}_peak_vram_mb` | Peak VRAM during the stage. |
| `ingest_throughput_ips`, `clustering_throughput_ips` | Derived: `image_count / wall_time`. |
| `hnsw_index_build_s` | Build wall time, charged once before the timed sim-search loop (§6, §9). |

### 3.2 Unsupervised cluster quality (intrinsic)

Computed on the **L2-normalised original (unreduced) embeddings** so distance
geometry is comparable across embedding families.

| Metric | Range | Reads as |
|---|---|---|
| Silhouette (micro) | [−1, 1] | Per-point cohesion vs. separation, averaged. Higher better. |
| Silhouette (macro) | [−1, 1] | Per-cluster mean (unweighted by size). Higher better. Avoids small-cluster drowning. |
| Calinski–Harabasz (CH) | [0, ∞) | Between/within dispersion ratio. Higher better — but inflates on few fat clusters. |
| Davies–Bouldin (DB) | [0, ∞) | Average max similarity between clusters. **Lower** better. |
| Noise ratio | [0, 1] | Fraction labelled −1 (HDBSCAN / DBSCAN / OPTICS only). |
| Cluster-size CV | [0, ∞) | Std of cluster sizes ÷ mean. Lower = more balanced. |

> **Critical reading note (recurs through every report).** Silhouette / CH /
> DB are computed on **non-noise points only**. Density methods (HDBSCAN,
> DBSCAN, OPTICS) are therefore graded on the subset they *chose* to cluster,
> while partitional methods are graded on **all** points. The
> *joint criterion* — non-trivial cluster count + moderate noise + balanced
> CV — is the only defensible reading across families.

### 3.3 Supervised cluster quality (extrinsic, §8 only)

Requires `--ground-truth labels.csv` and `segmenter == "identity"`.

| Metric | Range | Reads as |
|---|---|---|
| Adjusted Rand Index (ARI) | [−1, 1] | Pair agreement, chance-corrected. 0 = random, 1 = perfect. **Headline.** |
| Normalised Mutual Information (NMI) | [0, 1] | Shared info between predicted/true partitions. Inflates on many-tiny-clusters. |
| Pairwise F1 | [0, 1] | Harmonic mean of pairwise precision/recall; treats noise (−1) as own cluster. |
| Default vs `*_no_noise` | — | Default treats −1 as own cluster; `_no_noise` drops −1 points before scoring. Always read `_no_noise` jointly with `noise_ratio`. |
| matched_count, n_no_noise, class_count, coverage | — | Bookkeeping counts. |
| Retrieval P@5, MAP@5, MRR, R@5 | [0, 1] | Top-k=5 similarity-search retrieval scored against ground-truth labels (§8 secondary signal). |

### 3.4 ANN recall (§6, §9 only)

`ann_recall_at_k`: recall@k of the HNSW top-k vs an **exact brute-force kNN**
computed on the same query sample, *outside* the timed loop. `None` for exact
backends. `avg_neighbor_distance` is a secondary proxy.

---

## 4. Experiment-by-experiment results

### §1. Embedding model comparison

**Pillar.** Unsupervised quality (2).
**Question.** Which embedding backbone produces the most semantically
coherent graffiti clusters under a fixed downstream pipeline, and does
fine-tuning a graffiti-specific head beat its pretrained backbone?
**Runs.** 36/36 success. 12 embeddings × 3 sizes {250, 1000, 6416}.
**Deviation from catalogue.** 12 embeddings ran, not 10 — the **full 2×2 head
matrix** (author + style × DINOv2 + MobileNetV3) is present, including
`dinov2_graffiti_author_head` which the catalogue omitted.

**Embeddings tested.** `resnet50`, `vgg16`, `inception_v3`, `mobilenet_v3`,
`dinov2_vits14`, `clip_vit_b32`, `yolon`, `yolom`,
`mobilenet_v3_graffiti_author_head`,
`mobilenet_v3_graffiti_style_head`, `dinov2_graffiti_author_head`,
`dinov2_graffiti_style_head`.

#### Quality table (verbatim from report)

`cls` = cluster count, `sil` = silhouette, `CH` = Calinski–Harabasz, `DB` =
Davies–Bouldin, `csCV` = cluster-size CV.

| n | embedding | cls | noise | sil | CH | DB | csCV |
|--:|---|--:|--:|--:|--:|--:|--:|
| 250 | resnet50 | 23 | 0.063 | **0.258** | 13.2 | 1.43 | 0.51 |
| 250 | vgg16 | 26 | 0.084 | 0.250 | 10.9 | 1.46 | 0.55 |
| 250 | mobilenet_v3 | 26 | 0.055 | 0.245 | 10.4 | 1.58 | 0.48 |
| 250 | inception_v3 | 27 | 0.069 | 0.225 | 9.1 | 1.61 | 0.47 |
| 250 | dinov2_vits14 | 26 | 0.052 | 0.216 | 12.7 | 1.51 | 0.59 |
| 250 | dinov2_graffiti_author_head | 24 | 0.055 | 0.213 | 12.0 | 1.56 | 0.53 |
| 250 | yolom | 18 | 0.049 | 0.204 | *27.4* | 1.42 | 0.83 |
| 250 | dinov2_graffiti_style_head | 23 | 0.063 | 0.199 | 18.1 | 1.51 | 0.53 |
| 250 | mobilenet_v3_graffiti_author_head | 25 | 0.184 | 0.198 | 9.0 | 1.66 | 0.51 |
| 250 | clip_vit_b32 | 23 | 0.120 | 0.194 | 9.8 | 1.58 | 0.57 |
| 250 | yolon | 16 | 0.053 | 0.181 | 30.4 | 1.43 | 1.09 |
| 250 | mobilenet_v3_graffiti_style_head | 22 | 0.068 | 0.134 | 13.0 | 1.89 | 0.64 |
| 1000 | resnet50 | 127 | 0.028 | **0.272** | 13.7 | 1.39 | 0.53 |
| 1000 | mobilenet_v3 | 131 | 0.040 | 0.269 | 11.4 | 1.46 | 0.49 |
| 1000 | vgg16 | 113 | 0.094 | 0.234 | 11.6 | 1.53 | 0.55 |
| 1000 | dinov2_vits14 | 114 | 0.085 | 0.219 | 13.7 | 1.45 | 0.47 |
| 1000 | inception_v3 | 127 | 0.058 | 0.217 | 9.5 | 1.60 | 0.47 |
| 1000 | mobilenet_v3_graffiti_author_head | 134 | 0.081 | 0.198 | 9.2 | 1.60 | 0.48 |
| 1000 | clip_vit_b32 | 125 | 0.105 | 0.193 | 10.6 | 1.55 | 0.54 |
| 1000 | dinov2_graffiti_author_head | 96 | 0.126 | 0.192 | 13.3 | 1.58 | 0.62 |
| 1000 | mobilenet_v3_graffiti_style_head | 111 | 0.130 | 0.169 | 12.4 | 1.71 | 0.60 |
| 1000 | dinov2_graffiti_style_head | 96 | 0.106 | 0.159 | 18.9 | 1.68 | 0.60 |
| 1000 | yolom | 76 | 0.173 | 0.144 | *28.9* | 1.70 | 0.61 |
| 1000 | yolon | 59 | 0.220 | 0.116 | 44.5 | 1.71 | 0.65 |
| 6416 | mobilenet_v3 | 820 | 0.078 | **0.260** | 13.1 | 1.48 | 0.52 |
| 6416 | resnet50 | 794 | 0.086 | 0.257 | 15.6 | 1.46 | 0.71 |
| 6416 | vgg16 | 773 | 0.095 | 0.228 | 13.4 | 1.55 | 0.54 |
| 6416 | inception_v3 | 724 | 0.118 | 0.213 | 11.5 | 1.61 | 0.64 |
| 6416 | mobilenet_v3_graffiti_author_head | 798 | 0.130 | 0.190 | 10.1 | 1.68 | 0.54 |
| 6416 | dinov2_vits14 | 632 | 0.123 | 0.185 | 19.2 | 1.58 | 0.92 |
| 6416 | dinov2_graffiti_author_head | 604 | 0.156 | 0.174 | 17.2 | 1.64 | 0.81 |
| 6416 | clip_vit_b32 | 734 | 0.150 | 0.170 | 11.9 | 1.66 | 0.74 |
| 6416 | dinov2_graffiti_style_head | 563 | 0.199 | 0.151 | *33.1* | 1.70 | 0.75 |
| 6416 | mobilenet_v3_graffiti_style_head | 631 | 0.202 | 0.144 | 13.7 | 1.89 | 0.63 |
| 6416 | yolom | 480 | 0.253 | 0.131 | 32.2 | 1.75 | 0.74 |
| 6416 | yolon | 348 | 0.341 | 0.085 | 51.2 | 1.91 | 0.81 |

#### Cost table (verbatim, n=6416 rows)

| n | embedding | ingest_s | ips | ing_pkRSS MB | vram MB | red_s |
|--:|---|--:|--:|--:|--:|--:|
| 6416 | mobilenet_v3 | 808.8 | 7.93 | 366.0 | **31.8** | 6.06 |
| 6416 | mobilenet_v3_graffiti_author_head | 826.1 | 7.77 | 405.0 | 37.4 | 6.43 |
| 6416 | inception_v3 | 973.7 | 6.59 | 337.2 | 116.1 | 6.71 |
| 6416 | vgg16 | 980.8 | 6.54 | 315.0 | 551.7 | 7.88 |
| 6416 | resnet50 | 1080.0 | 5.94 | 336.0 | 117.8 | 6.57 |
| 6416 | dinov2_vits14 | 1215.7 | 5.28 | 209.6 | 98.0 | 5.74 |
| 6416 | dinov2_graffiti_style_head | 1231.9 | 5.21 | 287.3 | 100.2 | 5.93 |
| 6416 | clip_vit_b32 | 1247.0 | 5.15 | 193.1 | 591.9 | 6.01 |
| 6416 | yolon | 1301.0 | 4.93 | 853.8 | 65.4 | 5.89 |
| 6416 | yolom | 1407.4 | 4.56 | 824.9 | 192.9 | 5.96 |

#### Key findings

1. **Silhouette↓ / CH↑ split on fine-tuned heads (the central reading hazard).**
   Pretrained → head pairs at n=6416: DINOv2 base (sil 0.185, CH 19.2) →
   style head (sil 0.151, CH **33.1**) — silhouette falls, CH jumps 1.7×.
   Same direction for both heads on both backbones. The contrastive/triplet
   head reshapes the space toward task-discriminative axes (between/within
   variance ↑) while per-point cohesion (silhouette) falls (fewer, denser,
   elongated clusters).
2. **ImageNet CNNs sweep the top on raw silhouette** (ResNet50 0.257,
   MobileNetV3 0.260 at n=6416) — but this is the trap, not the answer. They
   cluster *something* cleanly (texture, colour, scene layout) without
   evidence it is graffiti-relevant.
3. **YOLO backbones are unambiguously worst.** At n=6416 `yolon` posts the
   worst silhouette (0.085), worst noise (0.341), fewest clusters (348),
   worst DB (1.91). Its high CH (51.2) is the same few-fat-clusters illusion
   as for high `mcs`. Detection backbones encode localisation, not holistic
   appearance.
4. **Cluster count and noise scale sensibly with n.** Mobilenet 26 → 131 →
   820; DINOv2 style head 23 → 96 → 563. Noise creeps upward with n, more
   steeply on weak encoders (yolon 0.05→0.22→0.34).
5. **MobileNetV3 is the cheapest encoder.** Fastest ingest (~7.9 ips), lowest
   VRAM (32 MB); YOLO are the most expensive (~4.5–4.9 ips, ~825–855 MB RSS).
   VGG16 (552 MB VRAM) and CLIP (592 MB VRAM) are the VRAM heavies.
6. **§1 alone cannot adjudicate the embedding.** The baseline
   `dinov2_graffiti_style_head` is retained on §8 supervised evidence
   (ARI 0.698 vs ≤0.368 for any untuned encoder) and task-specific design,
   not on §1 silhouette ranking. **Separability ≠ semantic correctness.**

#### Repeat stability

UMAP seed stochasticity is small at usable settings (DINOv2 style head @
n=6416 = 549/537/563, ±~2 %, sil std 0.006). Widest relative spread at
smallest corpus: `dinov2_vits14@250` = [21, 23, 26] (~24 %). No row required
exclusion.

---

### §2. Reduction technique comparison

**Pillar.** Unsupervised quality (2).
**Question.** Does dimensionality reduction matter for HDBSCAN, and which
family helps most?
**Runs.** 18/18 success. 6 reductions × 3 sizes (`identity`, `pca-10`,
`pca-50`, `umap-10`, `umap-50`, `isomap-10`). Embedding DB reused across
reductions within each limit (3 ingests total).

#### Quality table

| n | reduction | cls | noise | sil | silM | CH | DB | csCV |
|--:|--|--:|--:|--:|--:|--:|--:|--:|
| 250 | identity | 2 | 0.008 | 0.160 | *0.386* | 7.5 | **1.11** | 0.96 |
| 250 | pca-10 | 7 | 0.104 | 0.122 | 0.355 | 21.7 | 1.25 | 1.75 |
| 250 | pca-50 | 2 | 0.004 | 0.158 | 0.385 | 7.5 | 1.11 | 0.96 |
| 250 | **umap-10** | 23 | 0.063 | 0.199 | 0.258 | 18.1 | 1.51 | **0.53** |
| 250 | umap-50 | 23 | 0.063 | 0.197 | 0.248 | 18.0 | 1.55 | 0.53 |
| 250 | isomap | 12 | **0.580** | *0.279* | 0.293 | 19.4 | 1.23 | 0.31 |
| 1000 | identity | 4 | 0.051 | 0.072 | 0.320 | 8.3 | 1.65 | 1.67 |
| 1000 | pca-10 | 12 | 0.291 | 0.102 | 0.242 | 35.7 | 1.45 | 2.63 |
| 1000 | pca-50 | 3 | 0.006 | 0.109 | 0.340 | 8.4 | 1.49 | 1.38 |
| 1000 | **umap-10** | 96 | 0.106 | 0.159 | 0.209 | 18.9 | 1.68 | 0.60 |
| 1000 | umap-50 | 92 | 0.126 | 0.151 | 0.212 | 19.4 | 1.70 | 0.67 |
| 1000 | isomap | 26 | 0.576 | 0.134 | 0.186 | 30.6 | 1.80 | 1.53 |
| 6416 | identity | 2 | 0.003 | 0.073 | 0.230 | 8.3 | 1.76 | 1.00 |
| 6416 | pca-10 | 2 | 0.039 | **−0.028** | 0.324 | 3.6 | 1.48 | 1.00 |
| 6416 | pca-50 | 188† | **0.627** | 0.118 | 0.330 | 43.0 | 1.32 | **5.61** |
| 6416 | **umap-10** | 563 | 0.199 | 0.151 | 0.208 | 33.1 | 1.70 | 0.75 |
| 6416 | umap-50 | 537 | 0.195 | 0.142 | 0.204 | 32.4 | 1.73 | 0.82 |
| 6416 | isomap | 2 | 0.019 | 0.068 | 0.337 | 16.3 | 1.29 | 1.00 |

† `pca50@6416`: k=188 but **62.7 % noise** and csCV=5.61 — high-noise/unbalanced
failure mode, not a usable partition.

#### Cost table (n=6416)

| n | reduction | red_s | red_pkRSS MB | clu_s | clu_pkRSS MB |
|--:|--|--:|--:|--:|--:|
| 6416 | identity | 0.001 | 0.1 | **24.123** | 21.1 |
| 6416 | pca-10 | 0.197 | 2.7 | 0.375 | 2.1 |
| 6416 | pca-50 | 0.186 | 13.1 | 1.576 | 2.1 |
| 6416 | umap-10 | 6.158 | 58.4 | 0.341 | 0.3 |
| 6416 | umap-50 | 11.206 | 63.8 | 0.903 | 0.3 |
| 6416 | isomap | **23.646** | **1222.2** | 0.369 | 1.7 |

Ingest shared per corpus: 56.7 / 221.4 / 1267.5 s for 250 / 1000 / 6416.

#### Key findings

1. **Reduction is mandatory, not optional.** Raw 384-d (`identity`) yields
   only **2–4 clusters at every corpus size** — distances concentrate in high
   dimensions, HDBSCAN can't separate fine structure. UMAP-10 on the same
   data gives 23 → 96 → 563 clusters.
2. **`identity` is not free: it pushes cost into clustering.** clu_s at
   n=6416 is **24.1 s** for identity vs **0.34 s** for UMAP-10, a ~70× penalty.
   No cost–quality trade-off here in favour of skipping reduction.
3. **UMAP-10 is the winner at every size.** Cluster count scales sensibly
   (23 → 96 → 563), noise moderate (0.06 → 0.11 → 0.20), most balanced
   (csCV 0.53 → 0.60 → 0.75).
4. **UMAP-10 vs UMAP-50: 10-d is enough.** Quality-indistinguishable
   (sil 0.151 vs 0.142 at n=6416, k~550, csCV 0.75 vs 0.82); UMAP-50 nearly
   **2× the reduction wall time** for no quality gain.
5. **PCA is not robust** — PCA-10 collapses to k=2 with **negative
   silhouette** (−0.028) at n=6416; PCA-50 swings the other way (62.7 %
   noise, csCV 5.61). Linear projection cannot give HDBSCAN a stable density
   landscape across n.
6. **Isomap fails on cost *and* quality.** High noise at small/mid n
   (0.58/0.58) then collapse to k=2 at 6416. Most expensive reduction by
   far: **23.6 s and 1.2 GB peak RSS** at n=6416 (O(n²) geodesic kNN graph).
7. **Degenerate cells score better intrinsically.** `identity@250` (k=2)
   posts DB=1.11, the best in its group; macro silhouette *amplifies* this
   trap (the k=2 cells score silM ≈ 0.32–0.39 vs UMAP's 0.20–0.26).

#### Repeat stability

Only UMAP is stochastic: `umap-10@6416` = [549, 537, 563] (±~2 %). PCA, Isomap,
identity are deterministic.

---

### §3. Clustering algorithm comparison

**Pillar.** Unsupervised quality (2).
**Question.** With embedding and reduction fixed at the §2 winner, which
clustering algorithm partitions the corpus best, and which stay well-behaved
as n grows?
**Runs.** 42/42 success. 14 clustering configs × 3 sizes.

**Configs.** `hdbscan` (mcs=5), `kmeans` (elbow, max_k=30), `gmm` diag (BIC,
max_k ∈ {30, 100, 300}), `dbscan` (min_samples=5; auto-eps + eps ∈ {0.2, 0.5,
1.0}), `optics` (min_samples=5, cosine), `agglomerative` average linkage
(k ∈ {10, 20}), `spectral` nearest-neighbors (k ∈ {10, 20}).
`affinity_propagation` omitted (impractical at this n).

#### Quality table

| n | clustering | cls | noise | sil | CH | DB | csCV |
|--:|---|--:|--:|--:|--:|--:|--:|
| 250 | hdbscan | 23 | 0.063 | 0.199 | 18 | 1.51 | 0.53 |
| 250 | kmeans (k=7) | 7 | 0.000 | 0.103 | 28 | 2.35 | 0.48 |
| 250 | gmm·30 | 29 | 0.000 | 0.209 | 17 | 1.46 | 0.52 |
| 250 | gmm·100 | 37 | 0.000 | 0.237 | 16 | 1.38 | 0.51 |
| 250 | gmm·300 | 37 | 0.000 | 0.237 | 16 | 1.38 | 0.51 |
| 250 | dbscan·auto | 23 | 0.067 | 0.200 | 18 | 1.49 | 0.53 |
| 250 | dbscan·0.2 | 19 | **0.471** | *0.346* | 23 | 1.06 | 0.31 |
| 250 | dbscan·0.5 | 21 | 0.048 | 0.183 | 18 | 1.54 | 0.56 |
| 250 | dbscan·1.0 | 9 | 0.011 | 0.120 | 24 | 1.79 | 1.12 |
| 250 | optics | 24 | 0.169 | 0.237 | 18 | 1.39 | 0.48 |
| 250 | agglo·10 | 10 | 0.000 | 0.120 | 23 | 2.07 | 0.61 |
| 250 | agglo·20 | 20 | 0.000 | 0.165 | 18 | 1.68 | 0.52 |
| 250 | spectral·10 | 10 | 0.000 | 0.073 | 19 | 2.28 | 0.59 |
| 250 | spectral·20 | 20 | 0.000 | 0.137 | 16 | 1.89 | 0.43 |
| 1000 | **hdbscan** | 96 | 0.106 | 0.159 | 19 | 1.68 | 0.60 |
| 1000 | kmeans (k=8) | 8 | 0.000 | 0.052 | 86 | 3.01 | 0.40 |
| 1000 | gmm·30 | 30 | 0.000 | 0.076 | 35 | 2.61 | 0.51 |
| 1000 | gmm·100 | 100 | 0.000 | 0.133 | 18 | 1.85 | 0.54 |
| 1000 | gmm·300 | 147 | 0.000 | 0.166 | 16 | 1.60 | 0.50 |
| 1000 | dbscan·auto | 70 | 0.083 | 0.120 | 21 | 1.79 | 0.92 |
| 1000 | dbscan·0.2 | 91 | 0.194 | 0.177 | 20 | 1.58 | 0.63 |
| 1000 | dbscan·0.5 | 29 | 0.012 | 0.032 | 28 | 2.03 | 1.80 |
| 1000 | dbscan·1.0 | 5 | 0.000 | **−0.002** | 52 | 1.78 | 1.71 |
| 1000 | optics | 105 | 0.179 | *0.184* | 18 | 1.57 | 0.44 |
| 1000 | agglo·10 | 10 | 0.000 | 0.051 | 66 | 2.42 | 0.91 |
| 1000 | agglo·20 | 20 | 0.000 | 0.070 | 45 | 2.54 | 0.72 |
| 1000 | spectral·10 | 10 | 0.000 | **−0.042** | 28 | 3.22 | 1.31 |
| 1000 | spectral·20 | 20 | 0.000 | **−0.045** | 20 | 2.85 | 1.58 |
| 6416 | **hdbscan** | 563 | 0.199 | 0.151 | 33 | 1.70 | 0.75 |
| 6416 | kmeans (k=8) | 8 | 0.000 | 0.057 | 805 | 3.23 | 0.37 |
| 6416 | gmm·30 | 29 | 0.000 | 0.014 | 275 | 3.29 | 0.55 |
| 6416 | gmm·100 | 100 | 0.000 | 0.022 | 110 | 3.07 | 0.54 |
| 6416 | gmm·300 | 297 | 0.000 | 0.054 | 50 | 2.52 | 0.58 |
| 6416 | dbscan·auto | 239 | 0.059 | **−0.015** | 46 | 1.97 | 2.47 |
| 6416 | dbscan·0.2 | 233 | 0.058 | **−0.022** | 46 | 1.97 | 2.67 |
| 6416 | dbscan·0.5 | 34 | 0.001 | **−0.109** | 145 | 2.05 | 3.08 |
| 6416 | dbscan·1.0 | 9 | 0.000 | 0.035 | 591 | 1.82 | 1.61 |
| 6416 | optics | 648 | **0.247** | *0.177* | 30 | 1.64 | 0.45 |
| 6416 | agglo·10 | 10 | 0.000 | **−0.015** | 506 | 3.11 | 1.36 |
| 6416 | agglo·20 | 20 | 0.000 | 0.019 | 357 | 3.17 | 0.84 |
| 6416 | spectral·10 | 10 | 0.000 | **−0.117** | 264 | 3.33 | 1.56 |
| 6416 | spectral·20 | 20 | 0.000 | **−0.108** | 174 | 3.21 | 1.68 |

#### Cost table (n=6416)

| n | clustering | clu_s | clu_pkRSS MB |
|--:|---|--:|--:|
| 6416 | hdbscan | 0.348 | 0.3 |
| 6416 | kmeans | **0.009** | 0.2 |
| 6416 | gmm·30 | 0.140 | 0.2 |
| 6416 | gmm·100 | 0.618 | 0.2 |
| 6416 | gmm·300 | 1.839 | 49.7 |
| 6416 | dbscan·auto | 0.080 | 0.2 |
| 6416 | dbscan·0.5 | 0.084 | 0.2 |
| 6416 | optics | **8.578** | **763.3** |
| 6416 | agglo·10 | 0.408 | 314.2 |
| 6416 | agglo·20 | 0.416 | 314.4 |
| 6416 | spectral·10 | 2.073 | 1.1 |
| 6416 | spectral·20 | 1.660 | 1.4 |

Ingest shared per limit (57.6 / 226.8 / 1272.5 s) and UMAP-10 (~0.25 / ~1.0 /
~6.4 s) shared across all clusterers.

#### Cost–quality Pareto at n=6416

| Method | Cost | Quality (joint) | Verdict |
|---|---|---|---|
| **HDBSCAN** | **low** (0.35 s, 0.3 MB) | **best + stable** (k scales, moderate noise, balanced) | **winner** |
| OPTICS | **worst** (8.6 s, 763 MB) | ≈ HDBSCAN, more noise | quality peer, dominated on cost |
| GMM·300 | modest (1.8 s, 50 MB) | best **noise-free** option | runner-up if −1 unacceptable |
| KMeans | cheapest | trivial (k=8) | collapses |
| DBSCAN | cheap | no usable eps (CV>2.5 or k≈9) | dominated by HDBSCAN |
| Agglomerative | high mem (314 MB) | coarse, chains (CV 1.36) | dominated |
| Spectral | mid (1.7–2.1 s) | negative silhouette | dominated |

#### Key findings

1. **HDBSCAN is the only joint winner.** k scales sensibly (23→96→563),
   moderate noise (0.06→0.11→0.20), silhouette stable (~0.15–0.20), CV low
   (0.53→0.60→0.75), cost negligible.
2. **OPTICS matches quality at ~25× time and ~2500× memory.**
3. **GMM·300 is the fallback when noise labels are unacceptable.**
4. **KMeans collapses to k≈7–8 at every size** (elbow); high CH on few fat
   clusters is the trivial-k inflation trap.
5. **DBSCAN has no usable operating point** (CV > 2.5 or k≈9). auto-eps is
   not k-stable across repeats (e.g. [247, 219, 239] @ n=6416).
6. **Agglomerative needs O(n²) memory** (314 MB at n=6416); average linkage
   chains.
7. **Spectral is worst at scale** (negative silhouette).

---

### §4. HDBSCAN `min_cluster_size` tuning

**Pillar.** Unsupervised quality (2).
**Question.** What is the smallest meaningful cluster size for this dataset,
and does the optimum shift with corpus size?
**Runs.** 18/18 success. 6 `mcs` × 3 sizes. `mcs ∈ {5, 10, 25, 50, 100, 200}`
(≈0.1 % to ≈3.1 % of 6416).

#### Quality table

| n | mcs | cls | noise | sil | CH | DB | csCV |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | **5** | 23 | 0.063 | 0.199 | 18 | 1.51 | 0.53 |
| 250 | 10 | 10 | 0.188 | 0.141 | 24 | 1.91 | 0.38 |
| 250 | 25 | 2 | 0.007 | 0.234 | 67 | 1.63 | 0.52 |
| 250 | 50 | 2† | **0.667** | 0.230 | 66 | 1.64 | 0.53 |
| 250 | 100 | 0 | 1.000 | — | — | — | — |
| 250 | 200 | 0 | 1.000 | — | — | — | — |
| 1000 | **5** | 96 | 0.106 | 0.159 | 19 | 1.68 | 0.60 |
| 1000 | 10 | 30 | 0.204 | 0.085 | 33 | 2.21 | 0.76 |
| 1000 | 25 | 3 | 0.037 | 0.201 | 173 | 1.56 | 0.73 |
| 1000 | 50 | 2 | 0.000 | *0.287* | 154 | **0.99** | 0.88 |
| 1000 | 100 | 2 | 0.265 | 0.166 | 138 | 2.20 | 0.32 |
| 1000 | 200 | 0 | 1.000 | — | — | — | — |
| 6416 | **5** | 563 | 0.199 | 0.151 | 33 | 1.70 | 0.75 |
| 6416 | 10 | 154 | 0.277 | 0.037 | 56 | 2.20 | 1.78 |
| 6416 | 25 | 26 | 0.228 | 0.047 | 209 | 2.66 | 1.81 |
| 6416 | 50 | 4 | 0.026 | 0.193 | 1404 | 2.01 | 0.86 |
| 6416 | 100 | 2 | 0.008 | 0.269 | 2765 | 1.27 | 0.40 |
| 6416 | 200 | 2 | 0.011 | 0.270 | 2776 | 1.27 | 0.40 |

† `mcs50@250` aggregates `clusters_per_repeat = [0, 0, 2]` — unstable across
repeats.

#### Cost table (n=6416)

| n | mcs | red_s | red_pkRSS MB | clu_s | clu_pkRSS MB |
|--:|--:|--:|--:|--:|--:|
| 6416 | 5 | 6.140 | 97.0 | 0.338 | 0.3 |
| 6416 | 10 | 6.110 | 62.2 | 0.323 | 0.3 |
| 6416 | 25 | 6.083 | 62.8 | 0.333 | 0.3 |
| 6416 | 50 | 6.071 | 58.1 | 0.354 | 0.3 |
| 6416 | 100 | 6.129 | 59.5 | 0.414 | 0.3 |
| 6416 | 200 | 6.144 | 63.5 | 0.494 | 10.5 |

Ingest shared per limit: 56.1 / 226.2 / 1257.2 s. `mcs` is free in cost.

#### Key findings

1. **`mcs=5` wins at every corpus size on the joint criterion.**
2. **Silhouette and CH invert the ranking** — both peak at degenerate k=2
   cells (`mcs100@6416` posts sil 0.269 and CH 2765, but is a binary split).
3. **Noise is non-monotonic in `mcs`.** Rises through low/mid range (e.g.
   n=6416: 0.199→0.277 at mcs5→mcs10) then *falls back* (mcs100@6416 = 0.008,
   below mcs5). Low noise is *not* evidence of a good partition.
4. **The optimal floor does not shift with n.** `mcs=5` is best at all sizes.
5. **The collapse ceiling rises with n.** n=250 collapses at mcs25; n=1000 at
   mcs50; n=6416 holds 26 clusters at mcs25 and only collapses at mcs50/100.
   Larger corpora tolerate larger absolute mcs as a safety margin.
6. **Repeat instability sits at the collapse boundary** (`mcs50@250` = [0, 0,
   2]); UMAP seed-jitter at usable settings stays ±~2 %.

---

### §5. Segmenter impact

**Pillar.** Unsupervised quality (2) with a cost side-effect (1).
**Question.** Does YOLO cropping improve clustering quality over whole-image,
and which detector works best?
**Runs.** 12/12 success. 4 segmenters × 3 sizes. **Deviation from catalogue:**
catalogue lists 5 segmenters (`yolo11{n,s,m}` + fine-tuned + identity);
`yolo11n` is **absent** from this benchmark.
**Segmenters tested.** `identity`, `yolo11s.pt`, `yolo11m.pt`,
`yolo11m-train-10.pt` (fine-tuned). YOLO rows: threshold=0.5,
merge_threshold=0.8, padding=0.05.

#### Quality table

`crops` = `image_count`, `c/img` = crops per input image.

| n | segmenter | crops | c/img | cls | noise | sil | silM | CH | DB | csCV |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | **identity** | 250 | 1.00 | 23 | 0.063 | **0.199** | **0.258** | 18.1 | 1.51 | 0.53 |
| 250 | yolo11s | 251 | 1.00 | 21 | 0.076 | 0.155 | 0.192 | 15.8 | 1.75 | **0.43** |
| 250 | yolo11m | 254 | 1.02 | 23 | 0.085 | 0.185 | 0.245 | 16.5 | 1.55 | 0.54 |
| 250 | yolo11m_ft | 461 | 1.84 | 49 | 0.102 | 0.189 | 0.239 | **21.9** | 1.55 | 0.68 |
| 1000 | **identity** | 1000 | 1.00 | 96 | **0.106** | 0.159 | 0.209 | 18.9 | 1.68 | 0.60 |
| 1000 | yolo11s | 1039 | 1.04 | 90 | 0.189 | 0.138 | 0.184 | 17.0 | 1.78 | 0.57 |
| 1000 | yolo11m | 1034 | 1.03 | 90 | 0.159 | 0.137 | 0.194 | 16.0 | 1.76 | 0.67 |
| 1000 | yolo11m_ft | 2001 | 2.00 | 195 | 0.164 | **0.175** | **0.228** | **25.8** | **1.63** | 0.56 |
| 6416 | **identity** | 6416 | 1.00 | 563 | **0.199** | **0.151** | 0.208 | 33.1 | **1.70** | **0.75** |
| 6416 | yolo11s | 6590 | 1.03 | 502 | 0.225 | 0.127 | 0.185 | 30.4 | 1.81 | 0.75 |
| 6416 | yolo11m | 6603 | 1.03 | 506 | 0.230 | 0.131 | 0.190 | 29.5 | 1.80 | 0.73 |
| 6416 | yolo11m_ft | 15362 | 2.39 | 1049 | 0.298 | 0.122 | 0.199 | **36.2** | 1.86 | 1.06 |

#### Cost table (n=6416)

| n | segmenter | ing_s | ips | VRAM MB | red_s | clu_s |
|--:|---|--:|--:|--:|--:|--:|
| 6416 | identity | 1235.5 | 5.19 | 100.2 | 6.20 | 0.34 |
| 6416 | yolo11s | 1777.0 | 3.71 | 195.2 | 6.53 | 0.36 |
| 6416 | yolo11m | 1811.6 | 3.64 | 272.6 | 6.42 | 0.35 |
| 6416 | yolo11m_ft | 2152.3 | 7.14 | 492.3 | 7.24 | 1.47 |

#### Key findings

1. **`identity` wins at n=6416** on every metric: highest silhouette
   (0.151), lowest noise (0.199), best DB and csCV among non-fine-tuned rows.
2. **Base `yolo11s`/`yolo11m` are dominated by identity** — ≈1.03 crops/image
   means they barely change the partition while adding ~44 % ingest cost and
   detection noise.
3. **The fine-tuned detector is competitive only at small/mid n, collapses
   at full scale.** At n=1000 it leads silhouette (0.175), CH (25.8), DB
   (1.63) — but the 2× crop multiplier inflates cohesion mechanically. At
   n=6416: noise jumps to **0.298** (worst), csCV to **1.06** (worst),
   silhouette to table minimum (0.122).
4. **The corpus being clustered is not constant across rows.** `crops`
   ranges 6 416 (identity) → 15 362 (fine-tuned) at n=6416 — silhouette and
   CH cannot be strictly compared across segmenters; must read with crop
   count + noise + balance.
5. **`ips` is misleading.** Fine-tuned row has highest ips (7.14) but
   slowest *per input image* (0.34 s/img) because it emits 2.4× more crops.
6. **Per-input-image cost grows with detector weight** (0.19 → 0.28 → 0.34
   s/img). VRAM 100 → 195 → 273 → 492 MB.

---

### §6. Storage backend sweep

**Pillar.** Computational cost (1).
**Question.** Throughput gap between SQLite, PostgreSQL+pgvector exact, and
PostgreSQL+pgvector HNSW for similarity search as `n` grows; recall at the
spec.
**Runs.** 12/12 success. 3 backends × 4 sizes {500, 1000, 2500, 6416}.

**Backends.**
- `sqlite`: Python-side exhaustive scan.
- `postgresql`: pgvector ordered scan (`vector_cosine_ops`), no index.
- `postgresql_hnsw`: pgvector HNSW (`ops=cosine, m=16, ef_construction=64`).
Index built once between ingest and timed search loop. Each row uses its own
DB (separate `db_path`/`db_url`).

**Similarity search.** `top_k=5`, `sample_n=100`, `sample_seed=42`. K=5
repeats; iter 0 dropped.

#### Similarity-search throughput

| n | backend | ss_s | ips | idx_s | recall@5 |
|--:|---|--:|--:|--:|--:|
| 500 | sqlite | 3.05 | 164 | — | — |
| 500 | postgresql | 0.162 | 3 096 | 0.0 | — |
| 500 | postgresql_hnsw | 0.164 | 3 042 | 0.066 | **1.000** |
| 1000 | sqlite | 6.41 | 156 | — | — |
| 1000 | postgresql | 0.190 | 5 257 | 0.0 | — |
| 1000 | postgresql_hnsw | 0.187 | 5 353 | 0.136 | **1.000** |
| 2500 | sqlite | 16.89 | 148 | — | — |
| 2500 | postgresql | 0.234 | 10 678 | 0.0 | — |
| 2500 | postgresql_hnsw | 0.166 | 15 092 | 0.372 | **1.000** |
| 6416 | sqlite | 43.10 | 149 | — | — |
| 6416 | postgresql | 0.578 | 11 109 | 0.0 | — |
| 6416 | postgresql_hnsw | **0.179** | **35 936** | 0.465 | **1.000** |

#### Clustering quality (sanity check)

Identical row-for-row across backends at the same `n` — storage does not
affect clustering:

| n | k | silhouette | silM | CH | DB | noise | csCV |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 500 | 43 | 0.191 | 0.244 | 20.6 | 1.57 | 0.110 | 0.53 |
| 1000 | 96 | 0.160 | 0.210 | 18.9 | 1.67 | 0.106 | 0.60 |
| 2500 | 252 | 0.160 | 0.213 | 20.0 | 1.67 | 0.161 | 0.61 |
| 6416 | 556 | 0.149 | 0.206 | 33.1 | 1.71 | 0.193 | 0.73 |

#### Ingest

| n | sqlite | postgresql | postgresql_hnsw |
|--:|--:|--:|--:|
| 500 | 106.6 s | 111.8 s | 114.3 s |
| 1000 | 219.4 s | 220.0 s | 231.4 s |
| 2500 | 573.4 s | 558.6 s | 581.3 s |
| 6416 | 1231.9 s | 1269.9 s | 1334.6 s |

PostgreSQL adds 0–7 % vs SQLite per row; HNSW adds another 0–6 % (per-row
index insert). HNSW build itself sub-second at all n.

#### HNSW build cost

| n | hnsw_index_build_s |
|--:|--:|
| 500 | 0.066 |
| 1000 | 0.136 |
| 2500 | 0.372 |
| 6416 | 0.465 |

Sub-linear in `n` (slope ≈ 0.7). At n=6416 the 0.47 s build is amortised in
≈1.2 SQLite-equivalent queries or ≈260 exact-pgvector-equivalent queries.

#### Key findings

1. **SQLite slope ≈ 1.00 on ss_s** — pure O(n). Per-query latency at n=6416
   = 431 ms (vs 30 ms at n=500). Throughput flat at ~150 ips.
2. **PostgreSQL exact rises 3 096 → 11 109 ips** as round-trip overhead is
   amortised over the C-side cosine kernel. ss_s slope ≈ 0.5 — sub-linear in
   this range.
3. **HNSW is the only backend that accelerates with corpus size relative to
   exact** (3 042 → 35 936 ips). Per-query latency stays ~1.8 ms across n;
   near-O(log n).
4. **At n=6416:** HNSW 240× faster than SQLite, 3.2× faster than exact
   pgvector, recall@5 = **1.000**.
5. **SQLite point of no return is n ≈ 1000**. Beyond that it's operationally
   dead for any similarity-search workload.
6. **HNSW > exact crossover sits ~n=2500.** Below that round-trip dominates
   and HNSW gains are invisible.
7. **Recall@5 = 1.000 at every n** with the canonical HNSW parameters —
   speed/accuracy trade-off is trivially favourable at this scale (motivates
   §9).

---

### §7. Pipeline scalability sweep — HEADLINE

**Pillar.** Computational cost (1) — *the primary deliverable*.
**Question.** How does wall time scale with `n` for each pipeline stage on
the available graffiti corpus (n ≤ 6416)?
**Runs.** 30/30 success. 5 clusterers × 6 sizes {250, 500, 1000, 2000,
3500, 6416}. 6 ingests (shared SQLite DB per limit). Similarity search timed
only on HDBSCAN rows (storage-invariant under fixed embedding+corpus).

> The brief frames this as **observed scaling on the available corpus**, not
> as an asymptotic-complexity proof. 1.8 log decades is too narrow to
> separate O(n log n) from O(n²) cleanly. Slopes reported below are
> *observed* in this regime; theoretical references quoted as context.

#### Cost table (HEADLINE)

| n | ingest_s | red_s | clu_s hdbscan | clu_s kmeans | clu_s dbscan | clu_s optics | clu_s agglo | ss_s (hdb) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 58.7 | 0.253 | 0.0038 | 0.0025 | 0.0035 | 0.208 | 0.0019 | 3.66 |
| 500 | 115.7 | 0.603 | 0.0069 | 0.0027 | 0.0055 | 0.347 | 0.0046 | 16.32 |
| 1000 | 231.6 | 1.514 | 0.0144 | 0.0033 | 0.0103 | 0.631 | 0.0117 | 63.28 |
| 2000 | 455.9 | 4.542 | 0.1106 | 0.0047 | 0.0228 | 1.305 | 0.0427 | 306.81 |
| 3500 | 776.0 | 11.870 | 0.1863 | 0.0066 | 0.0471 | 2.282 | 0.1405 | 995.52 |
| 6416 | 1294.9 | **6.237** | 0.3580 | 0.0098 | 0.0768 | 4.322 | 0.5988 | **3266.60** |

#### Observed slopes (log–log fit, two regimes)

| Stage | Algorithm | Theoretical reference | Observed slope (250…6416) | Observed slope (500…3500) |
|---|---|--:|--:|--:|
| Ingest | DINOv2 inference | O(n) | 0.95 | — |
| Reduction | UMAP | ~O(n^1.14) | 1.15 | 1.53 |
| Clustering | KMeans (k=10 fixed) | ≈O(n) | 0.43 | 0.46 |
| Clustering | DBSCAN | O(n log n)–O(n²) | 0.99 | 1.10 |
| Clustering | OPTICS | O(n log n)–O(n²) | 0.94 | 0.98 |
| Clustering | HDBSCAN | O(n log n) amortised | 1.53 | 1.84 |
| Clustering | Agglomerative | O(n² log n) | 1.77 | 1.76 |
| Similarity search | SQLite linear scan (all-pairs top-k) | O(n²) | **2.10** | 2.13 |

#### Headline observation — similarity search dominates

At n=6416, post-ingest cost share:

| Stage | Wall | Share of post-ingest cost |
|---|--:|--:|
| Similarity search (HDBSCAN row) | 3266.6 s | **99.6 %** |
| Reduction (UMAP) | 6.2 s | 0.2 % |
| Clustering (HDBSCAN) | 0.4 s | <0.1 % |

Per-query throughput collapses **68 ips at n=250 → 2 ips at n=6416** — 35× drop
across 26× n. Slope 2.10 ⇒ projecting to a 50 k-image corpus extrapolates to
~58× wall time (~52 h per pass) under SQLite — i.e. ~7×10^4 s.

#### Memory (peak RSS, MB)

| n | ingest_rss | ingest_vram | red_rss | red_vram | ss_rss (hdb) | ss_vram (hdb) |
|--:|--:|--:|--:|--:|--:|--:|
| 250 | 281.5 | 100.2 | 4.7 | 85.6 | 0.0 | 85.6 |
| 500 | 282.5 | 100.2 | 5.0 | 85.6 | 4.6 | 85.6 |
| 1000 | 282.6 | 100.2 | 13.3 | 94.8 | 12.2 | 94.8 |
| 2000 | 282.6 | 100.2 | 21.4 | 94.8 | 24.5 | 94.8 |
| 3500 | 282.6 | 100.2 | 77.3 | 94.8 | 42.8 | 94.8 |
| 6416 | 282.6 | 100.2 | **72.3** | 94.8 | 82.0 | 94.8 |

Clustering peak RSS per clusterer:

| n | hdbscan | kmeans | dbscan | optics | agglomerative |
|--:|--:|--:|--:|--:|--:|
| 250 | 0.17 | 0.13 | 0.14 | 0.15 | 0.11 |
| 500 | 0.20 | 0.11 | 0.14 | 0.14 | 0.13 |
| 1000 | 0.18 | 0.12 | 0.14 | 0.14 | 0.13 |
| 2000 | 0.18 | 0.11 | 0.14 | 0.13 | **3.18** |
| 3500 | 0.23 | 0.11 | 0.14 | 0.13 | **93.54** |
| 6416 | 0.35 | 0.12 | 0.14 | 0.14 | **314.12** |

#### Anomaly: UMAP regime switch at n ≈ 4 k

UMAP wall rises 250→3500 (0.25 → 11.87 s, local slope ~1.5) and then **drops
sharply to 6.24 s at n=6416** — 47 % *faster* despite 83 % more input.
Reproducible (std 0.017 s, 0.3 %). Cause: UMAP switches to its
pynndescent-based approximate kNN backend above ~few-thousand points; below
that it uses exact graph construction. The same break appears in RSS
(77.3 → 72.3 MB).

**Consequence.** The full-range reduction slope (1.15) averages two regimes
stitched at n ≈ 4 096. The 500–3500 slope (1.53) describes the exact regime;
n=6416 is the first approximate point. Slope claims must specify which
window.

#### Quality side-effect at n=6416

| Algorithm | k | sil | CH | DB | noise | csCV |
|---|--:|--:|--:|--:|--:|--:|
| HDBSCAN (mcs=5) | 556 | 0.149 | 33 | 1.71 | 0.193 | 0.73 |
| OPTICS (ms=5) | 627 | **0.184** | 30 | **1.60** | 0.278 | **0.44** |
| DBSCAN (ms=5) | 258 | −0.013 | 46 | 1.96 | 0.061 | 2.46 |
| KMeans (k=10) | 10 | 0.035 | 659 | 3.23 | 0.000 | 0.43 |
| Agglomerative (k=10, avg) | 10 | −0.015 | 506 | 3.12 | 0.000 | 1.35 |

#### Key findings

1. **Similarity search dominates at corpus size.** O(n²) in time (slope 2.10)
   under SQLite. 54 min for one full-corpus top-5 sweep at n=6416. The cost
   anchor §6 / §9 are measured against.
2. **Ingest is linear (slope 0.95), GPU-bound.** ~0.20 s/image, CPU 6 %,
   peak RSS ≈ 283 MB constant in n (model floor), VRAM ≈ 100 MB.
3. **The UMAP slope changes regime at n≈4 k.** Quoting a single exponent is
   misleading; report both windows.
4. **Clustering wall time is not a bottleneck on this corpus.** All five
   clusterers finish under 5 s at n=6416.
5. **Memory is the binding constraint for Agglomerative.** RSS 0.13 → 3.18
   → 93.54 → **314.12 MB** across n ∈ {1000, 2000, 3500, 6416} — the O(n²)
   condensed distance matrix. At n=20 k → ~3 GB; n=50 k → ~19 GB.
   HDBSCAN/DBSCAN/OPTICS stay sub-1 MB.
6. **Similarity-search memory grows linearly** (4.6 → 82.0 MB across
   n=500…6416, slope ≈ 1.0) — O(n²) in time but O(n) in memory.
7. **Asymptotic claims deferred.** Slopes are *observed* on this corpus.
   Resampling extrapolation idea lives in `doc/conclusion.tex`.

#### Repeat stability

At n=6416: HDBSCAN red rel-std 0.3 %, clu rel-std 0.6 %; OPTICS 0.4 % / 0.4 %;
KMeans 0.4 % / 5.3 %; DBSCAN 0.6 % / 2.7 %; Agglomerative 0.6 % / 3.8 %. All
slope fits well above noise floor.

---

### §8a. Supervised validation — Style (294 crops, 4 classes)

**Pillar.** Supervised validation (3).
**Question.** Which embedding × reduction × clustering pipeline best recovers
4 hand-labelled styles? Does the graffiti style head beat the pretrained
backbone? Is the lift task-specific (style head vs author head)?
**Runs.** 96/96 success. 8 embeddings × 2 reductions × 6 clusterers.
**Ground truth.** `data/style/eval_crop/` — 294 crops, held out from disjoint
style-head training set (385 crops, supervised contrastive loss).

#### Best pipeline per embedding (default ARI)

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

#### Transfer matrix (style task)

| Family | pretrained | style head | author head |
|---|--:|--:|--:|
| DINOv2 | 0.356 | **0.698** (+0.342) | 0.234 (**−0.122**) |
| MobileNetV3 | 0.356 | **0.560** (+0.204) | 0.267 (**−0.089**) |

The author head **actively damages** style structure — drops below the
untuned backbone on both families. Diagonal-dominant transfer matrix:
each head concentrates its trained axis and suppresses the other.

#### Detail — fine-tuned style head, reduction × clustering

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

#### Supervised retrieval (top_k=5)

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

Same ordering as clustering ARI — style heads on top, author heads at bottom.

#### Cost summary (§8a)

| Embedding family | ingest wall (s) | throughput (ips) | sim-search wall (s) |
|---|--:|--:|--:|
| dinov2 (×3 heads) | 11.4–12.5 | 23–26 | ~5.0 |
| mobilenet (×3 heads) | 8.3–12.5 | 23–36 | ~15.0 |
| clip_vit_b32 | 11.1 | 26.4 | ~6.3 |
| resnet50 | 9.6 | 30.7 | ~23.0 |

UMAP ~0.35 s; clustering sub-0.02 s on 294 points. Sim-search wall scales with
**embedding dimension** under SQLite linear scan (DINOv2 384-d ~5 s, ResNet50
2048-d ~23 s).

#### Hypothesis disposition (§8a)

1. **Style head > backbone.** ✅ DINOv2 +0.342 ARI, MobileNet +0.204. Held-out ⇒
   genuine transfer.
2. **Cross-task specificity.** ✅ Confirmed *more strongly* than predicted —
   author head drops *below* backbone on style (DINOv2 −0.122, MobileNet
   −0.089). Diagonal-dominant.
3. **CLIP competitive, ResNet50 weakest.** ⚠ Partial: CLIP (0.368) ≈ DINOv2
   base (0.356) ✅; ResNet50 (0.334) sits with the base tier, not clearly the
   weakest ❌.
4. **HDBSCAN recovers k≈4 on the fine-tuned head.** ❌ identity space
   collapses to k=2; UMAP space gives `mcs`-invariant k=3 ([5, 10, 20]).
5. **identity ≥ UMAP on fine-tuned head.** ✅ DINOv2 identity-kmeans 0.698 >
   umap-kmeans 0.599.

#### Internal–external correlation

Pearson r(silhouette, ARI):
- **r = +0.66** over 48 oracle-k=4 partitional runs (kmeans/agglo/spectral).
- **r = +0.40** over all 83 runs with valid ARI.

Silhouette is a **defensible proxy at fixed k**; correlation weakens once k
varies. On 4-style granularity silhouette tracks semantic quality only when k
is pinned.

#### Degenerate cells

All-noise (0 clusters): `mobilenet_v3`, `mobilenet_v3_graffiti_author_head`
(all 3 identity-HDBSCAN); `dinov2_graffiti_author_head`, `clip_vit_b32`,
`resnet50` (identity HDBSCAN mcs10/20). Excluded from tables.

Repeat instability at HDBSCAN collapse boundary under UMAP:
- `dinov2_graffiti_author_head` umap hdb5 → [5, 11, 4]
- `clip_vit_b32` umap hdb5 → [2, 10, 13]
- `mobilenet_v3` umap hdb5 → [7, 13, 13]

Oracle-k partitional rows stable ([4, 4, 4] everywhere).

---

### §8b. Supervised validation — Author (185 crops, 87 authors)

**Pillar.** Supervised validation (3).
**Question.** Does the author head beat its backbone symmetrically to §8a's
style result, and across the Cuenca→Salamanca city boundary?
**Runs.** 96/96 success. 8 embeddings × 2 reductions × 6 clusterers.
**Ground truth.** `data/author/eval_crop/` — 185 crops, 87 authors. Class
distribution: **53 singletons, 17 pairs, top author 19 crops** (mean ~2.1).
Caps achievable ARI/F1; inflates NMI.

**Clustering grid (different from §8a).** `agglomerative` (avg)
n_clusters ∈ {30, 87}; `hdbscan` mcs ∈ {2, 3, 5, 10}. KMeans/Spectral omitted
(no principled handling of singleton classes at oracle k=87).

#### Best per embedding (default ARI)

| Embedding | best ARI | ±std | NMI | F1 | sil | noise | cls | winning cell |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `clip_vit_b32` | **0.076** | 0.003 | 0.789 | 0.090 | 0.003 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3_graffiti_author_head` | 0.061 | 0.005 | 0.787 | 0.074 | 0.011 | 0.00 | 87 | umap-agglomerative87 |
| `dinov2_graffiti_style_head` | 0.060 | 0.007 | 0.633 | 0.090 | 0.026 | 0.00 | 30 | umap-agglomerative30 |
| `dinov2_vits14` | 0.055 | 0.000 | 0.754 | 0.079 | 0.096 | 0.00 | 87 | identity-agglomerative87 |
| `dinov2_graffiti_author_head` | 0.051 | 0.010 | 0.636 | 0.081 | 0.027 | 0.00 | 30 | umap-agglomerative30 |
| `resnet50` | 0.049 | 0.005 | 0.778 | 0.064 | 0.028 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3` | 0.046 | 0.014 | 0.780 | 0.060 | 0.023 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3_graffiti_style_head` | 0.041 | 0.007 | 0.777 | 0.055 | 0.001 | 0.00 | 87 | umap-agglomerative87 |

#### Transfer matrix (both tasks side-by-side)

| Family | task | pretrained | style head | author head |
|---|---|--:|--:|--:|
| DINOv2 | **author (§8b)** | 0.055 | 0.060 | **0.051** |
| DINOv2 | style (§8a) | 0.356 | **0.698** | 0.234 |
| MobileNetV3 | **author (§8b)** | 0.046 | 0.041 | **0.061** |
| MobileNetV3 | style (§8a) | 0.356 | **0.560** | 0.267 |

On §8b the author head is **not** the diagonal peak. DINOv2 author head
(0.051) ≤ backbone (0.055) ≤ style head (0.060). MobileNetV3 author head
edges ahead by +0.015 ARI — well inside noise and at random-floor absolute
level.

#### Supervised retrieval (top_k=5)

Random P@5 baseline ≈ 0.024 per neighbour.

| Embedding | P@5 | MAP@5 | MRR | R@5 |
|---|--:|--:|--:|--:|
| `clip_vit_b32` | **0.091** | 0.252 | 0.262 | 0.268 |
| `dinov2_graffiti_author_head` | 0.088 | 0.258 | 0.261 | 0.273 |
| `resnet50` | 0.085 | 0.246 | 0.256 | 0.236 |
| `mobilenet_v3` | 0.083 | 0.240 | 0.245 | 0.251 |
| `dinov2_vits14` | 0.081 | 0.248 | 0.251 | 0.268 |
| `dinov2_graffiti_style_head` | 0.067 | 0.240 | 0.243 | 0.192 |
| `mobilenet_v3_graffiti_style_head` | 0.067 | 0.232 | 0.233 | 0.179 |
| `mobilenet_v3_graffiti_author_head` | 0.059 | 0.180 | 0.187 | 0.171 |

The DINOv2 author head (P@5 0.088) is statistically tied with off-the-shelf
encoders. **MobileNet author head is the single worst retriever (0.059)** —
yet its family's nominal "best ARI" winner (0.061). A cross-axis inversion
within one embedding is the signature of noise-dominated measurements.

#### Internal–external correlation — INVERTED vs §8a

§8a: r = +0.66 (oracle-k partitional). §8b:
- **r = −0.52** over all 86 runs with valid ARI and >1 cluster.
- **r = −0.25** over the 32 agglomerative runs alone.

Silhouette and ARI pull in **opposite directions**. The highest-ARI cells are
k=87 agglomerative partitions with near-zero silhouette (0.00–0.03); the
highest-silhouette cells are degenerate HDBSCAN k=2/k=3 splits (sil up to
0.52 on DINOv2 style head) with ARI ≈ 0. **Silhouette tracks coarse blob
structure, useful at small true k (4 styles) and actively misleading at
large k (87 authors).**

#### HDBSCAN behaviour

- **identity space mcs=10:** collapses to 0 clusters (noise 1.0) for **7 of 8
  embeddings**.
- **identity mcs=5:** all-noise for `dinov2_graffiti_author_head`,
  `mobilenet_v3`, `mobilenet_v3_graffiti_author_head`.
- **mcs=2:** fragments into 29–57 clusters but at **40–58 % noise** — source
  of inflated `_no_noise` columns.
- DINOv2 style head: identity hdb3/5/10 collapse to stable k=2 with sil ≈ 0.50
  and ARI ≈ 0 (the binary-split trap).

Repeat instability at boundary: `mobilenet_v3_graffiti_style_head` umap hdb3
→ [2, 21, 26]. Agglomerative rows stable ([30, 30, 30] / [87, 87, 87]).

All-noise cells excluded from tables: identity-HDBSCAN mcs=10 for clip_vit_b32,
dinov2_vits14, dinov2_graffiti_author_head, mobilenet_v3,
mobilenet_v3_graffiti_author_head, mobilenet_v3_graffiti_style_head, resnet50;
identity-HDBSCAN mcs=5 for dinov2_graffiti_author_head, mobilenet_v3,
mobilenet_v3_graffiti_author_head.

#### Cost summary (§8b)

| Embedding family | ingest wall (s) | throughput (ips) | sim-search wall (s) |
|---|--:|--:|--:|
| dinov2 (×3 heads) | 4.6–5.4 | 34–40 | ~1.9–2.1 |
| mobilenet (×3 heads) | 2.8–2.9 | 63–66 | ~5.7–6.1 |
| clip_vit_b32 | 3.7 | 50.0 | 2.49 |
| resnet50 | 3.5 | 52.3 | 9.20 |

UMAP 0.19–0.24 s; clustering 0.001–0.12 s. Same dimension-driven sim-search
scaling as §8a.

#### Hypothesis disposition (§8b)

1. **Author head > backbone (symmetric to §8a).** ❌ DINOv2 author head
   (0.051) ≤ backbone (0.055); MobileNet author head (+0.015) at random-floor
   absolute level.
2. **Cross-task specificity / diagonal-dominant matrix.** ❌ §8b is
   row-dominant (all near zero), not diagonal. Style specificity (§8a) holds;
   author specificity does not survive Cuenca→Salamanca shift.
3. **ResNet50 closes the gap on fine-grained identity.** ⚠ Weakly consistent:
   mid-pack on ARI (0.049), 3rd on P@5 (0.085) — but whole field at floor.
4. **HDBSCAN auto-k in tens (right order of magnitude).** ❌/⚠ mcs=2 finds
   29–57 clusters but at 40–58 % noise. No usable partition.
5. **identity ≥ UMAP on fine-tuned head.** ⚠ Indeterminate at this signal
   level.

---

### §9. HNSW parameter sweep

**Pillar.** Computational cost (1).
**Question.** At n=6416, how do HNSW knobs (`m`, `ef_construction`,
`ef_search`) trade off build time, query latency, and recall?
**Runs.** 10/11 success. 1 failure (`hnsw_m64_efc64_efs40` — pgvector
constraint `ef_construction ≥ 2·m`).

**Setup.** Single shared PostgreSQL DB `hnsw_sweep`. Run 1
(`hnsw_exact_baseline`) ingests with `clear_storage=true`, no HNSW; all
subsequent runs `clear_storage=false` and `hnsw.recreate=true`. Embedding
rows ingested once; each HNSW row pays only the index build + query loop.
Similarity search `top_k=5`, `sample_n=200`, `sample_seed=42`. K=5 repeats.

#### Grid

| Family | m | ef_construction | ef_search |
|---|--:|--:|--:|
| exact baseline | — | — | — |
| **`m` sweep** | 8, 16, 32, 64† | 64 | 40 |
| **`ef_construction` sweep** | 16 | 32, 64, 128, 256 | 40 |
| **`ef_search` sweep** | 16 | 64 | 10, 40, 100, 200 |

Anchor `(m=16, ef_construction=64, ef_search=40)` appears once.

#### Results

| run | m | ef_c | ef_s | build_s | ss_s | ips | recall@5 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `hnsw_exact_baseline` | — | — | — | — | 1.072 | 5 987 | — (exact) |
| `hnsw_m8_efc64_efs40` | 8 | 64 | 40 | 0.270 | 0.309 | 20 753 | 0.995 |
| `hnsw_m16_efc64_efs40` (anchor) | 16 | 64 | 40 | 0.446 | 0.321 | 19 980 | 0.995 |
| `hnsw_m32_efc64_efs40` | 32 | 64 | 40 | 0.981 | 0.345 | 18 618 | **1.000** |
| `hnsw_m16_efc32_efs40` | 16 | 32 | 40 | 0.329 | 0.364 | 17 638 | 0.990 |
| `hnsw_m16_efc128_efs40` | 16 | 128 | 40 | 0.567 | 0.346 | 18 520 | 0.995 |
| `hnsw_m16_efc256_efs40` | 16 | 256 | 40 | 0.769 | 0.350 | 18 351 | **1.000** |
| `hnsw_m16_efc64_efs10` | 16 | 64 | 10 | 0.449 | 0.319 | 20 116 | 0.993 |
| `hnsw_m16_efc64_efs100` | 16 | 64 | 100 | 0.424 | 0.374 | 17 160 | 0.996 |
| `hnsw_m16_efc64_efs200` | 16 | 64 | 200 | 0.444 | 0.450 | 14 261 | **1.000** |

#### Per-axis read

**`m` axis (`ef_c=64, ef_s=40`):**
| m | build_s | ss_s | recall@5 |
|--:|--:|--:|--:|
| 8 | 0.27 | 0.309 | 0.995 |
| 16 | 0.45 | 0.321 | 0.995 |
| 32 | 0.98 | 0.345 | 1.000 |

Build ~ linear in m (2× per doubling); query latency mild +12 % across 4×
range; recall climbs 0.995 → 1.000 at m=32.

**`ef_construction` axis (`m=16, ef_s=40`):**
| ef_c | build_s | ss_s | recall@5 |
|--:|--:|--:|--:|
| 32 | 0.33 | 0.364 | 0.990 |
| 64 | 0.45 | 0.321 | 0.995 |
| 128 | 0.57 | 0.346 | 0.995 |
| 256 | 0.77 | 0.350 | 1.000 |

Build ~ linear; query flat; recall 0.990 → 1.000 at ef_c=256.

**`ef_search` axis (`m=16, ef_c=64`):**
| ef_s | build_s | ss_s | recall@5 |
|--:|--:|--:|--:|
| 10 | 0.45 | 0.319 | 0.993 |
| 40 | 0.45 | 0.321 | 0.995 |
| 100 | 0.42 | 0.374 | 0.996 |
| 200 | 0.44 | 0.450 | 1.000 |

Build independent of ef_s; query latency 0.32 → 0.45 s (~40 % across 20×);
recall monotone across the full range — the only axis that strictly improves
recall by paying per-query.

#### Pareto frontier

- **Cheapest competitive:** `m16_efc64_efs10` — 0.319 s, recall 0.993.
- **§6 anchor:** `m16_efc64_efs40` — 0.321 s, recall 0.995 (§6 reports
  1.000 at sample_n=100; gap is sample-size effect).
- **Recall ceiling at minimal query cost:** `m16_efc256_efs40` — 0.350 s,
  recall 1.000, build 0.77 s (preferable for build-once / query-many).
- **Recall ceiling with cheaper build:** `m32_efc64_efs40` — 0.345 s, recall
  1.000, build 0.98 s.
- **Dominated:** `m16_efc64_efs200` — 0.450 s at recall 1.000, strictly worse
  than `m16_efc256_efs40` at same recall.

#### Speed-up vs exact pgvector

Every HNSW cell is **2.4× – 3.5×** faster than exact pgvector (1.072 s) at
this n. Relative gap widens with n (exact O(n)/query, HNSW near-O(log n));
§6 shows 3.2× at n=6416 between exact and HNSW for the same parameters.

#### Key findings

1. **HNSW default well chosen.** Anchor sits on Pareto frontier: recall 0.995
   / 0.32 s / 0.45 s build, 3.4× faster than exact.
2. **Effects partition cleanly:** `m`↑ → build↑, recall↑; `ef_construction`↑ →
   build↑, recall↑; `ef_search`↑ → query↑, recall↑. Build-time vs query-time
   knobs.
3. **Recall 1.000 reachable at <0.36 s/query.** Bump m=32 or ef_c=256 (no
   query cost). `ef_search=200` is strictly worse.
4. **pgvector constraint `ef_construction ≥ 2·m`** — the failed `m=64,
   ef_c=64` run is a useful negative result. Future sweeps must pair `m`
   with `ef_c ≥ 2m`.
5. **No HNSW combination in the safe region drops recall below 0.990.** At
   n=6416 HNSW knobs are speed-vs-recall-1.000, not safety knobs. HNSW is a
   defensible drop-in for exact at recall ≥ 0.99 under any safe parameter.

---

## 5. Cross-experiment comparative tables

### 5.1 Pipeline winners at every stage

| Stage | Winner | Evidence |
|---|---|---|
| Embedding | `dinov2_graffiti_style_head` for the style task; `mobilenet_v3` for cost-first | §8a (ARI 0.698 vs ≤0.368 untuned), §1 cost (mobilenet 7.93 ips at 32 MB VRAM) |
| Reduction | UMAP, `n_components=10`, `n_neighbors=15`, `min_dist=0`, `metric="cosine"` | §2 (only reduction non-trivial+balanced+moderate-noise at every n; 70× clustering speed-up vs identity) |
| Clustering | HDBSCAN, `min_cluster_size=5` | §3 (only joint winner, 0.348 s + 0.3 MB at n=6416), §4 (optimum invariant in n) |
| Segmenter | `identity` (whole image) | §5 (best at n=6416 on every metric; YOLO base ≈1.03×/img dominated; fine-tuned 2.4× crops collapses at scale) |
| Storage (search) | SQLite for dev; PostgreSQL+pgvector default; HNSW above n≈2000 | §6 (SQLite 41 s/query at n=6416; HNSW 240× faster at recall 1.000) |
| HNSW params | `(m=16, ef_construction=64, ef_search=40)` baseline; bump m=32 or ef_c=256 for recall 1.000 | §9 Pareto |

### 5.2 Best supervised pipeline per task

| Task | Embedding | Reduction | Clustering | ARI | F1 |
|---|---|---|---|--:|--:|
| **§8a Style (4 classes)** | `dinov2_graffiti_style_head` | identity | kmeans4 | **0.698** | **0.790** |
| **§8b Author (87 classes)** | `clip_vit_b32` | UMAP | agglomerative87 | **0.076** | 0.090 |

§8a result is decisively positive and confirms the project default. §8b best
ARI ≈ random floor — see anomalies.

### 5.3 Wall time at n=6416 (full corpus, baseline pipeline)

| Stage | Cost | Notes |
|---|--:|---|
| Ingest | 1235.5 s (§5 identity) | GPU-bound DINOv2; 0.20 s/img |
| Reduction (UMAP-10) | 6.2 s | Approximate-kNN regime |
| Clustering (HDBSCAN mcs=5) | 0.36 s | Sub-MB RSS |
| Similarity search (SQLite, 100 queries, top_k=5) | **43.1 s** | O(n²) scan |
| Similarity search (PG exact) | 0.58 s | C-side cosine |
| Similarity search (PG HNSW) | **0.18 s** | recall@5 = 1.000, 240× SQLite |
| HNSW index build | 0.47 s | Amortised in ~1 query vs SQLite |

### 5.4 Observed slopes summary (§§6, 7)

| Stage | Source | Observed slope |
|---|---|--:|
| Ingest (DINOv2) | §7 | 0.95 |
| UMAP reduction (full range, 250…6416) | §7 | 1.15 |
| UMAP reduction (exact regime, 500…3500) | §7 | 1.53 |
| KMeans (k=10 fixed) | §7 | 0.43 |
| DBSCAN | §7 | 0.99 |
| OPTICS | §7 | 0.94 |
| HDBSCAN | §7 | 1.53 (full), 1.84 (500…3500) |
| Agglomerative | §7 | 1.77 |
| SQLite all-pairs top-k | §7 | **2.10** |
| PostgreSQL exact ss | §6 | ~0.5 (sub-linear ramp) |
| HNSW ss | §6 | ~0 (flat at 1.8 ms/query) |
| HNSW index build | §6 | ~0.7 |

### 5.5 Memory peaks at n=6416

| Stage / Component | Peak RSS (MB) | Peak VRAM (MB) |
|---|--:|--:|
| Ingest (DINOv2) | 282.6 (model floor) | 100.2 |
| Reduction UMAP-10 | 72.3 (approx regime) | 94.8 |
| Reduction Isomap-10 | **1 222.2** (§2 worst) | — |
| Clustering HDBSCAN/KMeans/DBSCAN/OPTICS-density | <1 | — |
| Clustering OPTICS-cosine (§3) | **763.3** | — |
| Clustering Agglomerative (§7) | **314.1** | — |
| Sim-search SQLite (§7) | 82.0 | 94.8 |

### 5.6 Embedding × cost matrix (n=6416, §1)

| Embedding | ingest (s) | ips | RSS MB | VRAM MB |
|---|--:|--:|--:|--:|
| mobilenet_v3 | 808.8 | **7.93** | 366.0 | **31.8** |
| mobilenet_v3_graffiti_author_head | 826.1 | 7.77 | 405.0 | 37.4 |
| inception_v3 | 973.7 | 6.59 | 337.2 | 116.1 |
| vgg16 | 980.8 | 6.54 | 315.0 | 551.7 |
| resnet50 | 1080.0 | 5.94 | 336.0 | 117.8 |
| dinov2_vits14 | 1215.7 | 5.28 | 209.6 | 98.0 |
| dinov2_graffiti_style_head | 1231.9 | 5.21 | 287.3 | 100.2 |
| clip_vit_b32 | 1247.0 | 5.15 | 193.1 | 591.9 |
| yolon | 1301.0 | 4.93 | 853.8 | 65.4 |
| yolom | 1407.4 | 4.56 | 824.9 | 192.9 |

### 5.7 Silhouette ↔ ARI correlation by task

| Task | True k | Pearson r | Reading |
|---|--:|--:|---|
| §8a Style (oracle k=4 partitional) | 4 | **+0.66** | Silhouette is defensible proxy at fixed k |
| §8a Style (all 83 runs) | 4 | +0.40 | Weakens once k varies (HDBSCAN included) |
| §8b Author (all 86 runs >1 cluster) | 87 | **−0.52** | Silhouette **anti-correlated** with label agreement |
| §8b Author (32 agglo runs) | 87 | −0.25 | Sign holds even within stable family |

### 5.8 Recurring reading hazards across all reports

| Hazard | Where documented | Manifestation |
|---|---|---|
| Silhouette peaks on degenerate k=2 cells | §2 (identity@n; pca-50), §4 (mcs ≥ 50 @ 1000+), §8b (DINOv2 style identity hdb3/5/10 → k=2 sil ≈ 0.50) | k=2 binary split, semantically empty |
| CH inflated by few fat clusters | §1 (yolon@6416 CH 51.2), §3 (kmeans k=8 CH 805), §4 (mcs100@6416 CH 2765) | Trivial-k CH inflation |
| Macro silhouette amplifies degeneracy | §2 (k=2 cells score silM 0.32–0.39) | Worse than micro |
| Density methods graded on non-noise points only | §3, §4, §5 | Density methods get coverage credit; partitional methods graded on all points |
| Auto-eps DBSCAN k-unstable across repeats | §3 (`auto`@6416 [247, 219, 239]) | Aggregates mix partitions |
| Repeat instability at collapse boundary | §4 (mcs50@250 [0, 0, 2]); §8a/§8b umap hdb3/5 mixed seeds | Aggregated row meaningless |
| Cross-segmenter unit-of-analysis | §5 (crops 6 416 vs 15 362) | sil/CH not comparable across rows |
| `_no_noise` extrinsic inflates at high noise | §8a (mob style identity hdb20 ARI 0.224 / no-noise 0.854 @ 57.5 % noise); §8b mcs=2 cells at 40–58 % noise | Read with noise_ratio always |
| NMI inflated by many-tiny-clusters | §8b (k=87 winners NMI 0.75–0.79 with ARI ~0.05) | Singleton inflation |
| `ips` misleading across segmenters | §5 (fine-tuned highest ips but slowest per input image) | Use per-input-image cost |

---

## 6. Anomalies and negative results

1. **§7 UMAP regime switch at n ≈ 4 k.** UMAP wall drops 11.87 → 6.24 s
   between n=3500 → n=6416 (47 % faster on 83 % more input). Reproducible.
   Cause: pynndescent approximate-kNN backend kicks in. Quoting a single
   reduction slope is misleading; report 1.15 (full) and 1.53 (500…3500)
   separately and plot the curve.

2. **§5 fine-tuned detector collapses at full scale.** `yolo11m-train-10.pt`
   wins quality at n=1000 (sil 0.175) — but at n=6416 noise jumps to 0.298,
   csCV to 1.06, silhouette falls to 0.122 (table minimum). The 2.4× crop
   multiplier produces a long tail of low-quality detections.

3. **§9 `m=64, ef_construction=64` failed.** pgvector hard constraint
   `ef_construction ≥ 2·m`. Useful negative result — the runner does not
   pre-validate. Future sweeps must pair `m` with `ef_c ≥ 2m`.

4. **§8b is a flat negative.** Best default ARI = 0.076 (clip,
   umap-agglomerative87); whole field in 0.041–0.076 band (~5 % of §8a's
   winning 0.698). On the author axis the transfer matrix is row-dominant,
   not diagonal: no embedding meaningfully recovers authors across the
   Cuenca→Salamanca city boundary. Treat as honest bound on heads'
   generalisation; the §8a positive transfer claim qualifies its scope.

5. **§8b silhouette/ARI correlation inverts to −0.52.** Silhouette tracks
   coarse blob structure; at large true k (87 authors) it is actively
   misleading. Bounds the §8a +0.66 correlation reading.

6. **§8a HDBSCAN does not recover k=4** even on the fine-tuned style head.
   identity space → k=2 every `mcs`; UMAP space → `mcs`-invariant k=3.
   Oracle-k partitional methods (kmeans4, agglo4, spectral4) are the right
   tool when k is known; HDBSCAN's value is the unknown-k unsupervised
   setting.

7. **§8a author head damages style structure** — drops below pretrained
   backbone (DINOv2 −0.122, MobileNet −0.089 ARI). Cleanest possible
   evidence the two heads encode genuinely different information (a
   diagonal-dominant transfer matrix).

8. **§1 silhouette↓ / CH↑ split on fine-tuned heads.** Most consequential
   result + reading trap. The DINOv2 style head's CH jumps to 33.1 (1.7×
   backbone) at the same time silhouette falls. Geometric signature of
   contrastive/triplet training. Neither intrinsic score grades a fine-tuned
   space correctly — adjudication lives in §8.

9. **§3 auto-eps DBSCAN is k-unstable.** `clusters_per_repeat` mixes
   partitions: [247, 219, 239] @ n=6416. Aggregated metrics describe a
   mixture, not one partition.

10. **§4 noise is non-monotonic in `mcs`.** Rises then falls (mcs100@6416 =
    0.008, below mcs5's 0.199). Low noise is not evidence of a good
    partition.

11. **§7 Agglomerative memory grows quadratically.** 0.13 → 314.1 MB across
    n=1000…6416. At n=20 k → ~3 GB; n=50 k → ~19 GB. Wall-time cheap but
    memory-feasibility-bounded.

---

## 7. Recommendations for the Results chapter narrative

A structure for `doc/` Results & Discussion that flows from the brief
through the experiments:

### 7.1 Suggested ordering

1. **Open with the headline cost result (§7).** The brief asks for cost
   characterisation; lead with the wall-time-vs-n curve per stage and the
   observed-slope table. Make the SQLite O(n²) similarity-search bottleneck
   the headline figure (3 267 s vs 6.2 s reduction vs 0.36 s clustering at
   n=6416), and call out the UMAP regime switch at n ≈ 4 k as a methodological
   caveat for any single-exponent reading.

2. **Quality pipeline — §§1–4.** Build up the baseline pipeline justification
   one decision at a time, always flagging the recurring reading hazards
   (silhouette/CH peak on degenerate k=2; macro silhouette amplifies it;
   density methods graded on a chosen subset).
   - §1 embeddings — silhouette↓/CH↑ split → §1 cannot adjudicate alone;
     defer to §8.
   - §2 reduction — UMAP-10 wins; reduction is mandatory.
   - §3 clustering — HDBSCAN is the only joint winner; OPTICS dominated on
     cost, GMM·300 fallback for noise-free output.
   - §4 mcs tuning — mcs=5 wins at every n; noise non-monotonic.

3. **§5 segmenter** — identity wins at n=6416; fine-tuned 2.4× detector
   collapses; cropping does **not** clear the bar at thesis scale (but is
   the right tool for producing the §8 eval crops).

4. **§8 supervised validation — closes the loop.**
   - §8a is the positive supervised anchor: ARI 0.698 for
     `dinov2_graffiti_style_head` + identity + kmeans4, 2× any untuned
     encoder. Diagonal-dominant transfer matrix proves task-specific
     learning.
   - §8b is the honest negative result: 87-author recovery fails for every
     embedding across Cuenca→Salamanca. The author head transfers *within*
     its training domain but not across cities. Reports it as a bound on
     generalisation, not a failure of the method.
   - Internal–external correlation: silhouette tracks coarse k well (+0.66)
     and fine-grained k poorly (−0.52). Useful caveat to attach to every
     unsupervised intrinsic result.

5. **§6 storage** — develops the cost-vs-corpus characterisation on the
   infrastructure axis. SQLite is dev-only; pg-exact is the production
   default; HNSW becomes worth it above n≈2000 and at n=6416 gives 240×
   SQLite at recall 1.000.

6. **§9 HNSW tuning** — close the cost chapter with the Pareto frontier of
   the HNSW parameter space. The default anchor is on the frontier; recall
   1.000 reachable at <0.36 s/query by bumping `m` or `ef_construction`.

### 7.2 Tables and figures to include

| Asset | Source(s) | Audience |
|---|---|---|
| Cost-vs-n per stage (log–log) | §7 cost table + observed-slope table | Headline figure |
| UMAP regime-switch annotation | §7 reduction column | Methodological caveat callout |
| Silhouette↓/CH↑ split on fine-tuned heads | §1 fine-tuning pair table | Reading hazard explainer |
| Per-embedding cost matrix | §1 §5.6 table | Cost vs quality summary |
| HDBSCAN mcs joint criterion | §4 quality table | Justification of mcs=5 |
| Identity vs UMAP cluster counts | §2 quality table | Why reduction is mandatory |
| §8a transfer matrix (4 styles × 2 families × 3 heads) | §8a ARI summary | Supervised anchor |
| §8b transfer matrix side-by-side | §8b both-tasks table | Cross-task specificity bounded |
| Silhouette × ARI scatter (§8a + §8b) | §8 correlation summary (§5.7 table) | Defends silhouette as proxy at coarse k only |
| SQLite/pg-exact/HNSW throughput vs n | §6 ss_s table | Storage breakdown |
| HNSW Pareto frontier (recall × ss_s) | §9 results table | Tuning chapter |
| Memory peaks at n=6416 across pipeline | §5.5 table | Cost story |
| Repeat-stability sidebar (UMAP ±~2 %, collapse boundary [0, 0, 2]) | §§4, 7, 8 | Methodological footnote |

### 7.3 Cross-references for the discussion

- **Brief deliverable 1** (cost characterisation): §7 headline + §6 storage
  + §9 HNSW.
- **Brief deliverable 2** (unsupervised metrics study): §§1–5 with §8
  cross-validation of silhouette as proxy.
- **Brief deliverable 3** (supervised evaluation on labelled subset): §8a
  decisively positive, §8b honestly negative (bounded by domain shift +
  singleton class distribution).

### 7.4 Caveats to surface explicitly in the write-up

- **Slopes are observed on this corpus, not asymptotic exponents.** 1.8 log
  decades is narrow; the brief frames §7 as observed scaling.
- **The UMAP regime switch breaks single-exponent slope claims.** Quote both
  the full-range and 500–3500 slopes; plot the curve.
- **Intrinsic metrics are not directly comparable across embedding
  families** even after L2-normalisation — different dimensions and tuning
  perturb absolutes. They are *separability* rankings, not *semantic*
  rankings.
- **`_no_noise` extrinsic metrics and NMI must always be read with
  `noise_ratio` / class count.** They inflate exactly when one should
  distrust them.
- **§8b negative result is a finding, not a failure.** Frame as a bound on
  Cuenca-trained heads' generalisation to Salamanca + as a caveat on the
  symmetry claim §8a hypothesis 2 made.
- **Catalogue/data deviations to note:**
  - §1 ran 12 embeddings (not 10) — full 2×2 head matrix present;
    catalogue should drop the "omits `dinov2_graffiti_author_head`" clause.
  - §5 ran 4 segmenters (not 5) — `yolo11n` absent; add or trim brief.
  - §9 has 1 failed run (`m=64, ef_c=64`) due to pgvector
    `ef_construction ≥ 2·m` constraint; future grid generator must enforce
    it client-side.

### 7.5 One-paragraph executive summary (drop-in)

> Across nine experiments (370/371 runs successful) on a 6 416-image
> graffiti corpus, the headline cost-vs-n characterisation (§7) identifies
> SQLite all-pairs similarity search as the dominant bottleneck — an
> observed slope of 2.10 in `n`, yielding 54 min per full-corpus 6416-query
> top-5 sweep, against sub-7 s reduction and sub-0.5 s clustering. Ingest
> is GPU-bound and linear (~0.20 s/image). Switching to PostgreSQL+pgvector
> exact yields a 74× speed-up; an HNSW index at the canonical
> `(m=16, ef_construction=64, ef_search=40)` further yields 240× SQLite (3.2×
> exact) at recall@5 = 1.000, and the HNSW parameter Pareto (§9) shows that
> recall 1.000 is reachable at <0.36 s/query by bumping `m=32` or
> `ef_construction=256`. The unsupervised pipeline (§§1–5) is built around
> the `dinov2_graffiti_style_head` embedding, UMAP-10, HDBSCAN with
> `min_cluster_size=5`, and identity segmenter — each justified empirically
> and each surfacing the same reading hazard: silhouette and CH peak on
> degenerate k=2 cells, so the joint criterion (cluster count + moderate
> noise + balanced sizes) is the only defensible read. Supervised
> validation closes the loop: §8a gives a sharp positive (ARI 0.698 for
> the DINOv2 style head, ~2× any untuned encoder, with a diagonal-dominant
> head-vs-task transfer matrix), while §8b is an equally clean negative —
> cross-city author recovery fails for every embedding (best ARI 0.076 ~
> random floor), bounding the heads' generalisation to within-domain
> transfer and quantifying the silhouette/ARI correlation inversion
> (+0.66 at true k=4 vs −0.52 at true k=87) that any unsupervised metric
> defence in the chapter must acknowledge.

---

## 8. Reproducibility checklist

| Item | Value / Path |
|---|---|
| Repository | `tfm/` (this repo) |
| Catalogue | `infos/experiments.md` |
| Configurations (grids) | `infos/configs/0{1..9}*.json` |
| Generated benchmark configurations | `src/benchmarks/0{1..9}*.configuration.json` |
| Pipeline entry point | `uv run pipeline-benchmark` (CLI) |
| Plot entry point | `uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/` |
| Grid generator | `uv run python benchmarks/generate_configuration.py -i <grid.json> -o <config.json>` |
| Image dataset | `dataset/images/` (6 416 images) |
| Style eval | `data/style/eval_crop/` (294 crops) + `labels.csv` |
| Author eval | `data/author/eval_crop/` (185 crops) + `labels.csv` |
| Style training (held-out) | `data/style/train_crop/` (385 crops: 110 Salamanca + 275 Cuenca) |
| Author training (held-out, Cuenca) | 169-crop disjoint split (see `infos/dataset.md`) |
| Models directory | `models/` (gitignored): `yolo11{n,s,m,l}.pt`, `yolo11m-train-{8,10}.pt`, `{mobilenet,dinov2}_graffiti_{author,style}_head.pth` |
| Host | FullCreamMilk (Linux, 16 logical CPU, Python 3.14, `uv` package manager) |
| Python | ≥ 3.14 |
| Repeat seeds | `0..K-1` per run |
| Iter-0 drop policy | Wall/CPU times drop iter 0; memory keeps all K; ingest and sim-search single-shot |
| Quality on | L2-normalised original (unreduced) embeddings |
| Extrinsic on | identity segmenter only; requires `--ground-truth labels.csv` |

---

*End of unified big report.*

