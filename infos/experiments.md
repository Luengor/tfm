# Experiment Catalogue

Redesigned around the three pillars of the TFM21 brief
([`doc/enunciado.md`](../doc/enunciado.md)):

1. **Computational cost as a function of dataset size** — the *headline*
   contribution of the thesis. Characterised experimentally across the full
   pipeline (ingest, reduction, clustering, similarity search).
2. **Unsupervised study of cluster quality** — silhouette / Calinski–Harabasz /
   Davies–Bouldin against the same baseline pipeline, while varying one axis at
   a time.
3. **Supervised validation on the labelled subsets** — ARI / NMI / pairwise F1
   on two disjoint hand-labelled sets: `data/style/eval_crop/` (294 crops over
   4 styles) and `data/author/eval_crop/` (185 crops over 87 authors), the only
   experiments with ground truth.

The eight experiments below are ordered to support the thesis narrative: the
quality sweeps (§§1–5) fix the pipeline, the cost sweeps (§§6–7) measure it,
and the supervised study (§8) anchors the unsupervised metrics used everywhere
else.

## Baseline pipeline

Every experiment varies one axis around the same "best-guess" baseline so the
comparisons are clean:

- **Embedding:** `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style-discriminative
  projection head, L2-normalised output).
- **Reduction:** UMAP, `n_components=10`, `n_neighbors=15`, `min_dist=0.0`,
  `metric="cosine"`.
- **Clustering:** HDBSCAN, `min_cluster_size=5`.
- **Segmenter:** `identity` (whole image — isolates the segmenter question to
  experiment §5).
- **Storage:** SQLite.
- **Limit:** every experiment is run at three geometrically spaced sizes —
  `{250, 1000, 6416}` (≈26× span) — so each quality result is reported at
  small, mid and near-full corpus. Three points are enough to spot
  non-monotonic quality-vs-`n` behaviour without forcing each grid into a
  full cost-sweep. Cost-focused experiments (§6, §7) keep their finer
  explicit `n` sweeps.
- **Repeats:** `K=3` for quality experiments, `K=5` for cost experiments
  (§6, §7). Reduction + clustering loop K times per run; iter 0 is discarded
  to absorb JIT warm-up; the remaining K-1 samples produce mean ± std for
  wall time, CPU, RSS, and VRAM. Ingest stays single-shot. See the
  **Metrics** section for the full rationale.

The working dataset contains 6416 images. The 294 labelled style-eval
crops in [`data/style/eval_crop/`](../data/style/eval_crop/) are drawn from
this 6416-image corpus and are *not* deduplicated out for the unsupervised
experiments — at <5 % of the corpus they do not perturb aggregate statistics,
and segregating them would artificially shrink the cost-sweep dataset. The 385
training crops (110 Salamanca style-train + 275 Cuenca) and the 185
Salamanca author-eval crops come from the separate 265-image Salamanca and
1207-image Cuenca annotation sets, which are **disjoint** from the 6416-image
eval corpus.

## Metrics

**Computational cost (per stage).** Wall time, CPU time, stage-net peak RSS (`peak_rss_delta_mb`, peak above the stage's own baseline), peak VRAM.
Stages: `ingest`, `reduction`, `clustering`, `similarity_search`.
Throughput columns (`ingest_throughput_ips`, `clustering_throughput_ips`) are
derived automatically.

**Repeats and error bars.** Each run sets `repeats: K`. The reduction and
clustering stages execute K times back-to-back in the same process with seeds
`0..K-1`; iter 0 is dropped (it absorbs UMAP/numba JIT and CUDA cache warm-up)
and the remaining K-1 samples are aggregated as mean ± std. Output JSON keeps
the existing `{stage}_wall_time_s` keys (now the mean) and adds
`{stage}_wall_time_s_std` and `{stage}_n` siblings — `plot_metrics.py` and
`table.py` keep working unchanged. Clustering quality metrics (silhouette,
Calinski–Harabasz, Davies–Bouldin, noise ratio, cluster-size CV) are computed
on every iteration (all K samples, no iter-0 drop — quality is unaffected by
JIT warm-up) and aggregated as mean ± std under the existing
`clustering_quality_*` keys, with `_std` and `_n` siblings added. Extrinsic
metrics are computed on every iteration and aggregated as mean ± std alongside
the intrinsic metrics (`ari`, `nmi`, `pairwise_f1`, and their `*_no_noise`
variants). Ingest and similarity-search remain single-shot — ingest is
deterministic and the DB-reuse heuristic depends on a single timing.

Defaults: `K=3` for quality-focused experiments (§§1–5, §8) and `K=5` for the
cost-focused sweeps (§6, §7). Wall-time differences inside one std should be
read as within-noise; the scalability sweep §7 reports the trend across `n`
as its headline figure rather than any single point.

**Unsupervised cluster quality.**

| Metric | Range | Interpretation |
|---|---|---|
| Silhouette | [−1, 1] | Cohesion vs. separation, per-point average. |
| Calinski–Harabasz | [0, ∞) | Between-cluster vs. within-cluster dispersion. |
| Davies–Bouldin | [0, ∞) | Average max-similarity between clusters; lower is better. |
| Noise ratio | [0, 1] | Fraction labelled −1 (HDBSCAN / DBSCAN / OPTICS only). |
| Cluster-size CV | [0, ∞) | Balance — standard deviation of cluster sizes ÷ mean. |

All five are computed against the **original unreduced** embeddings so the
reduction stage is not graded by its own loss.

**Supervised cluster quality** (only when `--ground-truth` is supplied and
`segmenter == "identity"`): ARI, NMI, pairwise F1, plus the matched / class /
coverage counts.

## How to run an experiment

```bash
cd src

# 1. Expand the grid into a configuration (skip for §5 which is already a configuration)
uv run python benchmarks/generate_configuration.py \
  -i ../infos/configs/01_embedding.json \
  -o benchmarks/01_embedding.configuration.json

# 2. Run the benchmark
uv run pipeline-benchmark \
  --configuration benchmarks/01_embedding.configuration.json \
  --dataset ../dataset/images

# 3. Plot
uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/
```

For §8 (supervised) add `--ground-truth <labels.csv>` and point `--dataset` at
the matching crop folder — `data/style/eval_crop/` for §8a, `data/author/eval_crop/`
for §8b. For §5 (segmenter) skip
step 1 — the file is
already a configuration (segmenter changes invalidate cached crops, so explicit
`db_path` + `clear_storage: true` per row is required).

---

## §1. Embedding model comparison

**Pillar.** Unsupervised quality (2).

**Question.** Which embedding backbone produces the most semantically coherent
graffiti clusters under a fixed downstream pipeline?

**Varies.** Twelve embeddings: four ImageNet CNNs (`resnet50`, `vgg16`,
`inception_v3`, `mobilenet_v3`), DINOv2 ViT-S/14, CLIP ViT-B/32, two YOLO
backbones (`yolon`, `yolom`), and the **full 2×2 fine-tuned head matrix**
(author + style heads × DINOv2 + MobileNetV3:
`dinov2_graffiti_author_head`, `dinov2_graffiti_style_head`,
`mobilenet_v3_graffiti_author_head`, `mobilenet_v3_graffiti_style_head`).
Running both heads on both backbones gives the complete cross-task matrix that
§8 hypothesis 2 reads — and lets §1 contrast the author vs. style head on each
backbone directly.

**Fixed.** Baseline UMAP → HDBSCAN, identity segmenter, SQLite. Swept at
`limit ∈ {250, 1000, 6416}`.

**Why it matters.** This is the single most consequential choice in the
pipeline. Pairs `mobilenet_v3` ↔ `mobilenet_v3_graffiti_author_head` and
`dinov2_vits14` ↔ `dinov2_graffiti_style_head` answer the fine-tuning vs.
pretraining question directly.

**Cost.** 36 runs (12 embeddings × 3 limits), 36 full ingests (each
embedding × limit pair gets its own DB; no reuse across embeddings). The
`limit=250` ingests are cheap, so the marginal cost over the 2-limit
variant is dominated by the 12 small-`n` runs.

**Config.** [`configs/01_embedding.json`](configs/01_embedding.json)

## §2. Reduction technique comparison

**Pillar.** Unsupervised quality (2).

**Question.** Does the dimensionality-reduction stage matter for HDBSCAN, and
which family helps most?

**Varies.** `identity` (no reduction), `pca` (10 and 50 components), `umap`
(10d and 50d cosine), `isomap` (10d). Linear/poly KernelPCA kernels and
UMAP-50/100 sweeps from v1 were trimmed — they did not contribute distinctive
evidence in early runs. KernelPCA was also dropped: the baseline embedding
(DINOv2 style head) outputs L2-normalised vectors, on which RBF-KernelPCA
reduces to a monotone transform of cosine similarity and adds no signal beyond
linear PCA.

**Fixed.** Baseline embedding and HDBSCAN. Swept at
`limit ∈ {250, 1000, 6416}`.

**Why it matters.** Density-based clustering degrades in high dimensions. The
identity row quantifies how much reduction buys; the UMAP rows confirm or
refute the canonical recipe; the Isomap row shows whether non-linear
reductions help.

**Cost.** 18 runs (6 reductions × 3 limits), 3 ingests (one per limit — the
embedding DB is reused across reductions within each limit group).

**Config.** [`configs/02_reduction.json`](configs/02_reduction.json)

## §3. Clustering algorithm comparison

**Pillar.** Unsupervised quality (2).

**Question.** Given fixed embeddings and reduction, which clustering algorithm
groups them best?

**Varies.** `hdbscan`, `kmeans` (auto-k via elbow), `gmm` (auto-k), `dbscan`,
`optics` (cosine), `agglomerative` (n_clusters ∈ {10, 20}), `spectral`
(n_clusters ∈ {10, 20}). `affinity_propagation` is omitted as impractical at
this n (see [`best_cluster.md`](best_cluster.md)). The OPTICS-euclidean and
agglomerative-5 rows from v1 were trimmed for redundancy.

**Fixed.** Baseline embedding and UMAP. Swept at
`limit ∈ {250, 1000, 6416}`.

**Why it matters.** Tests the algorithmic hierarchy recommended in
`best_cluster.md` against the actual data and provides ablation evidence for
the HDBSCAN default. The sweep over `n_clusters` for agglomerative and
spectral removes the k confound when comparing against HDBSCAN's auto-k.

**Cost.** 27 runs (9 clusterers × 3 limits), 3 ingests.

**Config.** [`configs/03_clustering.json`](configs/03_clustering.json)

## §4. HDBSCAN `min_cluster_size` tuning

**Pillar.** Unsupervised quality (2).

**Question.** What is the smallest meaningful cluster size for this dataset?

**Varies.** `min_cluster_size ∈ {5, 10, 25, 50, 100, 200}` — sized as
fractions of the 6416-image corpus (≈0.1 % to ≈3.1 %).

**Fixed.** Baseline embedding and UMAP. Swept at
`limit ∈ {250, 1000, 6416}` so the same `min_cluster_size` axis is read
across three corpus sizes; the 6416-image row is where the fractions
resolve to meaningful absolute counts and density estimates are stable,
while the 250 and 1000 rows reveal whether the optimal `min_cluster_size`
shifts with `n` (interaction the headline cost-sweep §7 does not measure).

**Why it matters.** This is HDBSCAN's primary knob. Too low → noisy
micro-clusters; too high → most points fall into noise (label −1). The
sweep also produces the noise-fraction curve that feeds back into the
narrative: if noise climbs sharply past a threshold, that's the
upper bound on cluster granularity the dataset supports.

UMAP hyperparameter tuning (`n_components`, `n_neighbors`, `min_dist`) is
intentionally omitted: the canonical 10-d / 15-neighbour / `min_dist=0`
setting is well-established for HDBSCAN preprocessing and an earlier sweep
produced no surprises. A dedicated UMAP-tuning grid can be reconstructed
under `infos/configs/` if the reduction sweep (§2) flags UMAP as marginal.

**Cost.** 18 runs (6 `min_cluster_size` × 3 limits), 3 ingests.

**Config.** [`configs/04_hdbscan.json`](configs/04_hdbscan.json)

## §5. Segmenter impact

**Pillar.** Unsupervised quality (2), with a cost side-effect.

**Question.** Does YOLO cropping improve clustering quality over embedding the
whole image, and which detector / threshold / padding works best?

**Varies.** `identity` vs. `yolo` with `yolo11{s,m}.pt` and
`yolo11m-train-10.pt` (fine-tuned detector) — four segmenters total. Earlier
threshold / padding sweeps and the `yolo11l` row were dropped: at this
scale, the gap between `m` and `l` is dominated by compute cost, not
detection quality, and the threshold/padding axis did not produce
distinctive signal. The `yolo11n` row was also dropped — at this scale its
detection quality is strictly below `yolo11s`/`yolo11m` while clustering
behaviour tracks them, so it added cost without distinctive signal.

**Fixed.** Baseline embedding, reduction, clustering. Each segmenter row is
expanded to `limit ∈ {250, 1000, 6416}`.

**Why it matters.** Segmentation removes background noise but introduces
detection errors and changes the unit of analysis (image vs. crop). The
thesis question is whether the trade-off is worth it; the per-run timing
captures the cost side directly.

**Format.** Configuration (not a grid). `generate_configuration.py` hashes
`storage + embedding + segmenter + limit` to decide DB reuse. Segmenter *is*
included in the hash, so the hand-written format is not strictly required for
correctness; it was chosen for clarity and explicit control over `db_path`
and `clear_storage: true` per row.

**Cost.** 12 runs (4 segmenters × 3 limits), 12 full ingests — the most
expensive quality experiment.

**Config.**
[`configs/05_segmenter.config.json`](configs/05_segmenter.config.json)
(feed directly to `pipeline-benchmark --configuration`; no `generate_configuration.py`
step).

## §6. Storage backend cost

**Pillar.** Computational cost (1).

**Question.** What is the throughput gap between SQLite (linear Python scan),
PostgreSQL + pgvector exact (sequential scan over the indexed `vector`
column), and PostgreSQL + pgvector HNSW (approximate) for ingest and
similarity search, and how does it scale with `n`?

**Varies.** `storage ∈ {sqlite, postgresql_exact, postgresql_hnsw}` ×
`limit ∈ {500, 1000, 2500, 6416}`. The PostgreSQL HNSW row sets
`storage.params.hnsw = {ops: "cosine", m: 16, ef_construction: 64}`; the
runner builds the HNSW index between ingest and the similarity-search loop
so the index build cost is excluded from the timed query stage and reported
separately as part of the ingest-adjacent setup. Similarity search is
**enabled** (`top_k=5`) — that is the axis where the three backends differ
most (O(n) Python scan vs. exact O(n) sequential scan vs. HNSW
sub-linear approximate).

**Fixed.** Baseline embedding, reduction, clustering, identity segmenter.

**Why it matters.** Clustering quality is invariant under storage choice, so
this experiment isolates pure infrastructure cost. The ~13× `limit` span is
wide enough that the HNSW advantage should be empirically visible rather
than buried in per-call overhead. The three-row sweep separates the
"native vector ops vs. Python" effect (sqlite → pg-exact) from the
"approximate vs. exact" effect (pg-exact → pg-hnsw).

**Prerequisites.** PostgreSQL via the dev container at `localhost:54321`
(`postgresql://postgres:changethis@localhost:54321/postgres`). Each storage
× limit combination receives its own database via the storage-key hash; the
HNSW row hashes to a separate database from the exact row because `hnsw`
lives under `storage.params`.

**Cost.** 12 runs at `repeats=5`, 12 ingests — every storage × limit
combination needs its own DB. Similarity-search wall time is single-shot
per run (not looped), so the K=5 repeats only multiply reduction +
clustering cost; the SQLite-vs-pgvector-exact-vs-pgvector-HNSW throughput
gap on similarity search is the headline figure here, and its variance
comes from inter-run rather than intra-run measurements — read it
cautiously against the cost legend in §7.

**Config.** [`configs/06_storage.json`](configs/06_storage.json)

## §7. Pipeline scalability sweep (HEADLINE)

**Pillar.** Computational cost (1) — *the primary deliverable of the thesis.*

**Question.** How does wall time scale with `n` for each pipeline stage, and
do the asymptotic complexity differences between clustering algorithms
manifest empirically within the available dataset range?

**Varies.** `limit ∈ {100, 250, 500, 1000, 2000, 3500, 6416}` (~64× span) ×
clustering algorithm ∈ {HDBSCAN, KMeans, DBSCAN, OPTICS, Agglomerative,
Spectral}. Similarity search is enabled with `top_k=5` so the search stage
is timed at every `n` too — this is where the corpus-size sensitivity of
SQLite is observable as a separate signal.

**Fixed.** Baseline embedding (`dinov2_graffiti_style_head`), UMAP reduction,
SQLite, identity segmenter.

**Why it matters.** This is the experiment the thesis exists to deliver
(`enunciado.md`: *"describir el coste computacional en función del tamaño del
conjunto de imágenes, de forma experimental"*). Expected complexity classes:

| Stage | Algorithm | Expected scaling |
|---|---|---|
| Ingest | Any embedding | O(n) — model inference per image |
| Reduction | UMAP | ~O(n^1.14) empirically |
| Similarity search | SQLite linear scan, all-pairs top-k | O(n²) per full corpus pass |
| Clustering | HDBSCAN | O(n log n) amortised |
| Clustering | KMeans | O(n · k · iter) ≈ O(n) |
| Clustering | DBSCAN | O(n log n) with index, O(n²) worst-case |
| Clustering | OPTICS | O(n²) |
| Clustering | Agglomerative | O(n² log n) |
| Clustering | Spectral | O(n²)–O(n³) |

On a log-log plot the slope of wall time vs. `n` should be ~1 for the
linear-ish stages and ~2 for the quadratic clusterers. That headline figure
— *empirically measured slopes* alongside *theoretical complexity classes*
— is the central result of the cost chapter.

**Cost.** 42 runs at `repeats=5` (4 measured samples per run after iter-0
drop), 7 ingests (the embedding DB is reused across the 6 clusterers within
each limit group). The repeat tax falls only on reduction + clustering, so
the extra cost is small relative to the 7 single-shot ingests — and the
error bars are essential here because complexity-class claims rest on
slope estimates rather than single timings.

**Config.** [`configs/07_scalability.json`](configs/07_scalability.json)

## §8. Supervised validation on labelled crops

**Pillar.** Supervised validation (3).

**Question.** When ground-truth labels are available, which pipeline best
recovers them? Specifically: (a) does fine-tuning a graffiti-specific
projection head improve cluster–label agreement over the pretrained backbone,
(b) is the improvement consistent across backbone families, and (c) does the
ranking hold across two qualitatively different label sets — coarse **style**
(4 classes, balanced) and fine-grained **author** (87 classes, heavy long
tail)?

**Two label sets.** §8 runs the same embedding × reduction grid against two
disjoint hand-labelled splits, each with its own clustering grid sized to the
class count.

- **§8a Style.** [`data/style/eval_crop/`](../data/style/eval_crop/) — 294
  crops across 4 styles. Held-out from the fine-tuned style head, which was
  trained on the disjoint [`data/style/train_crop/`](../data/style/train_crop/)
  (385 crops: 110 from Salamanca + 275 from Cuenca) via supervised
  contrastive loss (`src/train/style_trainer.py`).
- **§8b Author.** [`data/author/eval_crop/`](../data/author/eval_crop/) —
  185 crops across 87 authors, drawn from the Salamanca primary dataset.
  Highly imbalanced: 53 authors with a single crop, 17 with two; top author
  has 19 crops. The author heads (`dinov2_graffiti_author_head`,
  `mobilenet_v3_graffiti_author_head`) were trained with triplet loss
  (`src/train/dino|mobilenet/train_head.py`, sampler in
  `src/train/dataset.py`) on a disjoint 169-crop split from the **Cuenca
  (stopgrafiti) dataset** — see [`infos/dataset.md`](dataset.md). The two
  sources do not overlap, so §8b measures held-out generalisation across
  cities as well as across authors (a stronger test than within-dataset
  hold-out, but also a harder one: city-level domain shift may suppress
  raw ARI even when author-specific features transfer).

Labels in both `labels.csv` files follow the `filename,style` schema consumed
by `--ground-truth` (the column header is literal `style` even for author
labels — the runner does not care about the column name). Crops are
pre-segmented, so `identity` is the correct (and required, per the segmenter
check in `src/src/evaluation/runner.py`) segmenter for extrinsic metrics.

**Metrics.** Intrinsic triad (silhouette, CH, DB) plus extrinsic triad:

| Metric | Range | Reads as |
|---|---|---|
| Adjusted Rand Index (ARI) | [−1, 1] | Pair agreement, chance-corrected. 0 = random, 1 = perfect. |
| Normalised Mutual Information (NMI) | [0, 1] | Shared info between predicted and true partitions. |
| Pairwise F1 | [0, 1] | Harmonic mean of pairwise precision / recall; treats noise (−1) as its own cluster. |

ARI is the headline (chance-corrected, comparable across runs with different
k); NMI is reported alongside to flag the *many-tiny-clusters-inflates-NMI*
failure mode (especially relevant on §8b where 53 singletons make NMI
inherently inflated); pairwise F1 is the most interpretable for write-up.
On §8b, the singleton-dominated class distribution caps achievable
ARI/F1 — report deltas against the per-embedding random baseline rather
than absolute scores.

**Varies (shared across §8a and §8b).**

- **Embedding (8).** Four pairs that isolate the fine-tuning question on two
  backbones × two label sets: `dinov2_vits14` ↔
  `dinov2_graffiti_style_head` ↔ `dinov2_graffiti_author_head`, and
  `mobilenet_v3` ↔ `mobilenet_v3_graffiti_style_head` ↔
  `mobilenet_v3_graffiti_author_head`. `clip_vit_b32` and `resnet50` are
  off-the-shelf anchors.
- **Reduction (2).** `identity` (raw embedding space) and UMAP
  `n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine"` (the
  canonical pre-clustering reduction; matches the baseline).

**Varies (§8a clustering, 6 settings).** `kmeans`, `agglomerative` (average
linkage), `spectral` all with `n_clusters=4` (oracle k); `hdbscan` swept over
`min_cluster_size ∈ {5, 10, 20}` (density-based auto-k — tests whether
the true k=4 emerges and how sensitive that is to the main knob).

**Varies (§8b clustering, 6 settings — proposed).** Oracle k=87 destabilises
KMeans and Spectral on 185 points (many empty / near-empty clusters), so the
proposed §8b grid uses `agglomerative` (average linkage) at
`n_clusters ∈ {30, 87}` (a moderate-k proxy plus oracle) and `hdbscan` swept
over `min_cluster_size ∈ {2, 3, 5, 10}` (2 and 3 are the smallest meaningful
sizes given the singleton-heavy long tail). `kmeans` and `spectral` are
omitted — neither has a principled way to handle the singleton classes that
dominate the label set. Revisit if the §8b run produces uninformative ARI
across all settings (a sign the chosen grid is too narrow).

**Fixed.** SQLite, identity segmenter, no `limit` (both `eval_crop` folders
are small enough that every run processes the full set).

**Hypotheses.**

1. **§8a.** The fine-tuned style head outperforms `dinov2_vits14` on ARI /
   NMI / F1. Held-out comparison ⇒ a head win is genuine transfer rather than
   memorisation. Symmetrically on §8b, the author heads outperform their
   respective pretrained backbones — and since §8b additionally crosses city
   boundaries (Cuenca-trained, Salamanca-evaluated), a positive result is
   evidence the heads learn author-discriminative features that generalise
   across the geographic / dataset shift rather than locality-specific cues.
2. **Cross-task specificity (relative).** Any graffiti-tuned head will likely
   beat its pretrained backbone on either task — step one of either head is
   "learn that graffiti exists." The discriminating claim is *relative
   magnitude*: on §8b, the gap (author head − backbone) exceeds the gap
   (style head − backbone), and symmetrically on §8a. A diagonal-dominant
   transfer matrix (below) is the cleanest evidence that the two heads
   encode different information; uniform lift across both tasks suggests
   the heads share a generic graffiti representation.
3. CLIP is competitive with DINOv2 base on §8a; ResNet50 is the weakest
   off-the-shelf encoder for stylistic clustering, consistent with §1. On
   §8b, ResNet50 may close the gap — fine-grained identity discrimination
   leans on local texture, where CNN features are competitive.
4. KMeans-4 (oracle k) wins on the §8a fine-tuned head because the head was
   trained to make classes linearly separable; HDBSCAN auto-discovers a
   count near 4 on the fine-tuned head at moderate `min_cluster_size` (≈10–20).
5. `identity` reduction matches or beats UMAP on each fine-tuned head
   (further reduction is information loss when the encoder has already
   concentrated discriminative axes) and underperforms UMAP on raw ImageNet
   features.

**Derived analyses.**

- **Internal–external correlation.** Plot ARI vs. silhouette across all §8a
  runs and separately across all §8b runs. A strong positive correlation
  means the unsupervised metrics are a defensible proxy on this label
  granularity; divergence between §8a and §8b correlations bounds *which*
  axis silhouette is tracking.
- **k-recovery for HDBSCAN.** Does HDBSCAN's auto-detected `n_clusters` land
  near 4 (§8a) or in the right order of magnitude — tens, not thousands —
  (§8b) on the fine-tuned head? Report alongside `noise_ratio`.
- **Cross-task transfer matrix.** Tabulate ARI for every embedding against
  both label sets. A diagonal-dominant matrix (style heads peak on §8a,
  author heads on §8b) confirms head specificity; off-diagonal leakage
  suggests the heads share more than intended.

**Cost.** §8a: 96 runs (8 embeddings × 2 reductions × 6 clusterers), 8
ingests. §8b: 96 runs (8 embeddings × 2 reductions × 6 clusterers), 8
ingests (separate DBs — the configs use distinct `db_path` prefixes
`08a_style_` vs `08b_author_` so the storage-key hash never collides even
when both sections run against the same `--output-dir`). Reduction and
clustering re-use the cached DB within each embedding group thanks to
storage-key hashing in `generate_configuration.py`.

**Running.**

```bash
cd src

# §8a Style
uv run python benchmarks/generate_configuration.py \
  -i ../infos/configs/08a_supervised_style.json \
  -o benchmarks/08a_supervised_style.configuration.json

uv run pipeline-benchmark \
  --configuration benchmarks/08a_supervised_style.configuration.json \
  --dataset ../data/style/eval_crop \
  --ground-truth ../data/style/eval_crop/labels.csv \
  --cluster-plot

# §8b Author
uv run python benchmarks/generate_configuration.py \
  -i ../infos/configs/08b_supervised_author.json \
  -o benchmarks/08b_supervised_author.configuration.json

uv run pipeline-benchmark \
  --configuration benchmarks/08b_supervised_author.configuration.json \
  --dataset ../data/author/eval_crop \
  --ground-truth ../data/author/eval_crop/labels.csv \
  --cluster-plot

uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/
```

`--cluster-plot` writes an interactive 2-D UMAP scatter per run as a
Vega-Lite HTML page (`<output-dir>/<run_id>_cluster.html`) with embedded
metrics, pan/zoom, shift-drag brush selection, and a thumbnail lightbox —
ground-truth labels overlay for visual sanity-checking. At 192 runs (96 + 96)
the resulting HTML directory is non-trivial; drop the flag on the §8b run
if disk / browse overhead matters and only re-enable it for the
fine-tuned-head rows once §8b numbers point at the interesting cases.

**Configs.**
[`configs/08a_supervised_style.json`](configs/08a_supervised_style.json),
[`configs/08b_supervised_author.json`](configs/08b_supervised_author.json).

## §9. HNSW index parameter sweep

**Pillar.** Computational cost (1).

**Question.** Inside the PostgreSQL + pgvector backend, how do the HNSW
construction parameters (`m`, `ef_construction`) and the query parameter
(`ef_search`) trade off index build time, query latency, and answer
quality against the exact baseline on the full 6416-image corpus?

**Varies.** Three one-axis sweeps anchored at the canonical defaults
`m=16, ef_construction=64, ef_search=40`:

| Axis | Values | Fixed |
|---|---|---|
| `m` (graph degree) | {8, 16, 32, 64} | `ef_construction=64, ef_search=40` |
| `ef_construction` (build candidate list) | {32, 64, 128, 256} | `m=16, ef_search=40` |
| `ef_search` (query candidate list) | {10, 40, 100, 200} | `m=16, ef_construction=64` |

Plus one **exact** baseline (no HNSW, sequential scan) for ground-truth
neighbour reference and for the headline speedup ratio.

**Fixed.** Baseline embedding (`dinov2_graffiti_style_head`), UMAP reduction,
HDBSCAN clustering, identity segmenter, PostgreSQL backend with shared
`db_url` (`hnsw_sweep` database), `limit=6416` (full corpus — small `n`
hides the HNSW win in per-call overhead), similarity search enabled with
`top_k=5, sample_n=200, sample_seed=42`. `repeats=5`.

**Why it matters.** §6 establishes *that* HNSW is faster than exact pgvector
on this dataset. §9 maps the Pareto frontier: which `m / ef_construction /
ef_search` setting reaches the best speed/quality trade-off, and whether the
HDBSCAN-relevant `top_k=5` regime is sensitive to `ef_search` at all. The
sweep also provides the build-time × query-time decomposition the thesis
needs to argue HNSW is appropriate for a one-shot ingest → many-query
workflow vs. a single-query throwaway.

**Index reuse.** All eleven runs share the same PostgreSQL database. Run 1
(`hnsw_exact_baseline`) ingests with `clear_storage: true`; all subsequent
runs set `clear_storage: false` and use `hnsw.recreate: true` so the index
is dropped and rebuilt per run with the new parameters (otherwise the
existing index from a prior run would persist and silently dominate the
results). The ingest stage is therefore measured once; the runner's
metadata-reuse heuristic propagates the original ingest timing to the
remaining ten runs.

**Quality measurement.** The runner reports `ann_recall_at_k` — recall@k of
the HNSW index neighbours vs. an exact brute-force kNN computed on the same
query sample, outside the timed loop. It is a first-class result field
(see `AnnMetrics` in `models.py`). `avg_neighbor_distance` (mean distance to
returned `top_k` neighbours) is also recorded as a secondary proxy and is
useful for comparing HNSW rows against the exact baseline.

**Format.** Direct configuration (not a grid). `generate_configuration.py`
hashes `hnsw` into the storage key (correct in general — see §6) so a
grid would assign every HNSW variant its own DB and force ten redundant
ingests. The configuration therefore enumerates the eleven runs explicitly
with a shared `db_url` and the `clear_storage`/`recreate` flags above.

**Cost.** 11 runs at `repeats=5`, 1 ingest. The dominant cost is the ten
HNSW index builds plus the eleven similarity-search loops; reduction +
clustering also re-run at K=5 per run.

**Config.**
[`configs/09_hnsw.config.json`](configs/09_hnsw.config.json) — feed
directly to `pipeline-benchmark --configuration` (no
`generate_configuration.py` step).

```bash
cd src
uv run pipeline-benchmark \
  --configuration ../infos/configs/09_hnsw.config.json \
  --dataset ../dataset/images
```

---

## Suggested order

1. **§1 Embedding** first — fixes the strongest variable so later experiments
   use the actual best embedding rather than a guess.
2. **§2 Reduction** and **§3 Clustering** next, using the winner of §1.
3. **§4 HDBSCAN tuning** to fine-tune the chosen clustering / reduction pair.
4. **§5 Segmenter** once the rest of the pipeline is settled — it is the most
   expensive quality experiment.
5. **§6 Storage** and **§7 Scalability** can run in any order or in parallel
   with the quality experiments once the baseline pipeline is fixed. They are
   the headline contribution of the thesis and should be allocated the most
   careful time-budget (consistent hardware state, no concurrent load).
6. **§9 HNSW tuning** runs after §6 has established the SQLite vs.
   pgvector-exact vs. pgvector-HNSW baseline at multiple `n`. §9 fixes
   `n=6416` and maps the HNSW Pareto frontier.
7. **§8 Supervised validation** can run any time after §1 — running it early
   lets later experiments cite the internal–external correlation when
   defending silhouette / CH / DB as proxies for semantic quality.

## History

An earlier catalogue had ten experiments including UMAP hyperparameter
tuning and a distance-metric / sphere-geometry deep dive. Both were
trimmed: the canonical UMAP setting is well-established for HDBSCAN
preprocessing, and the metric-impact analysis is theoretical context for
the thesis chapter rather than a separate empirical sweep. The remaining
eight experiments live under [`configs/`](configs/), with the scalability
sweep §7 expanded (more `n` points, a sixth clusterer, and similarity
search) to make it the genuine cost-vs-`n` headline.
