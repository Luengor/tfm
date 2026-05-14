# Experiment Catalogue

A curated list of configurations worth evaluating for the graffiti clustering
pipeline. Each entry states the question the experiment answers, what varies,
what stays fixed, and the corresponding grid or whitelist file in
[`configs/`](configs/).

The default "best-guess" baseline used as the fixed component when varying
another axis is:

- **Embedding:** `dinov2_graffiti_head` (DINOv2 ViT-S/14 backbone with the
  graffiti-fine-tuned projection head — self-supervised features tuned for the
  domain).
- **Reduction:** UMAP, `n_components=10`, `metric="cosine"`, `min_dist=0.0`
  (the canonical pre-clustering reduction; cosine matches the L2-normalised
- **Clustering:** HDBSCAN, `min_cluster_size=5`
  output of DINOv2/CLIP-style encoders; `min_dist=0` keeps projected points as
  tightly packed as possible, which maximises density contrast for HDBSCAN).
  (density-aware, auto-detects k, isolates noise — the strongest recommendation
  in [`best_cluster.md`](best_cluster.md)).
- **Segmenter:** `identity`
  (matches the current `whitelist.sample.json`; the segmenter experiment isolates
  whether YOLO cropping helps).
- **Storage:** SQLite (single-file, no service dependency; PostgreSQL is
  benchmarked separately).
- **Limit:** `1000` images
  (tune up/down depending on dataset size and how patient you are; held constant
  across runs so timing comparisons are fair).

## Computational performance as primary metric

The benchmark runner profiles each pipeline stage and records wall time, CPU
time, and memory (RSS delta, VRAM peak) for:

| Stage | Captures |
|---|---|
| `setup` | Model loading / storage connection |
| `ingest` | Segment → embed → store, repeated per image |
| `reduction` | Dimensionality reduction over all stored embeddings |
| `clustering` | Full clustering pass (wraps reduction timing) |
| `similarity_search` | k-NN lookup for every stored image |

`ingest_throughput_ips` (images/s) and `clustering_throughput_ips` are derived
automatically. These timing columns are available alongside quality scores in
every result CSV and JSON.

**Treat timing as co-primary.** A configuration that clusters well but ingests
10× slower represents a different trade-off than one that is fast but produces
poor clusters. Where the two objectives conflict, document both.

**Variability.** Each configuration runs once; wall time on a single run is
noisy. Differences smaller than ~15 % between configurations should be treated
as within measurement noise. For the scalability experiment (#9), the trend
across n values is more meaningful than any individual data point.

**Dataset size.** The working dataset contains approximately 5000 images (a
second, larger dataset acquired after the initial 240-image collection).
Per-experiment `limit` values are set explicitly per config — the headline
quality experiments use the full 5000, while the embedding (#1) and segmenter
(#6) sweeps cap at 2000 to keep wall-clock costs manageable since they require
a fresh ingest per run. The actual n in every result is stored in the
`image_count` column.

**Memory.** Every stage captures `peak_rss_mb` and `vram_peak_mb`. Embedding
models dominate VRAM; reduction and clustering dominate RAM at large n. Large
reduction algorithms (Isomap, KernelPCA) also build O(n²) distance matrices —
their RAM consumption scales quadratically and can become the binding constraint
before CPU time does.

## Running an experiment

Each grid file is consumed by `generate_whitelist.py` to produce a whitelist,
which is then fed to the benchmark runner:

bash
```
cd src

# 1. Expand the grid into a whitelist
uv run python benchmarks/generate_whitelist.py \
  -i ../infos/configs/01_embedding_comparison.json \
  -o benchmarks/embedding_comparison.whitelist.json

# 2. Run the benchmark
uv run pipeline-benchmark \
  --whitelist benchmarks/embedding_comparison.whitelist.json \
  --dataset ../dataset/images

# 3. Plot the results
uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/
```

The segmenter experiment is already a whitelist (no grid expansion needed);
skip step 1 for it.

## Distance metrics: what is actually configurable

The cosine default is convenient but not always the right choice — many of the
clustering algorithms internally assume Euclidean (L2) distance. The current
wrappers expose a metric parameter only on a subset of components:

| Component | Metric / kernel knob | Default | Other supported values |
|---|---|---|---|
| UMAP reduction | `metric` | `"cosine"` | `"euclidean"` (L2), `"manhattan"` (L1), `"chebyshev"`, `"minkowski"`, `"correlation"`, … (any `umap-learn` metric) |
| KernelPCA reduction | `kernel` | `"rbf"` | `"linear"`, `"poly"`, `"sigmoid"`, `"cosine"` (RBF and poly are L2-based; cosine and linear are inner-product based) |
| OPTICS clustering | `metric` | `"cosine"` | `"euclidean"`, `"manhattan"` / `"l1"`, `"chebyshev"`, … (any sklearn pairwise metric) |
| Isomap reduction | — | L2 (sklearn default) | wrapper does not expose `metric` |
| KMeans, GMM | — | L2 (algorithmic requirement) | not changeable |
| HDBSCAN, DBSCAN | — | L2 (sklearn default) | wrapper does not forward `metric` to sklearn |
| Agglomerative | — | L2 (sklearn default) | wrapper does not forward `metric`; only `linkage` is configurable |
| Spectral | `affinity` | `"nearest_neighbors"` | `"rbf"` (different concept than a metric) |

**Practical consequence.** The HDBSCAN / DBSCAN / KMeans / Agglomerative
wrappers operate in L2 internally, so the only place the working metric can be
chosen for those clusterers is in the upstream UMAP stage. Whether changing it
matters depends on whether the embeddings live on the unit hypersphere:

- **L2-normalised embeddings** (`dinov2_graffiti_head`, `clip_vit_b32`, and the
  `*_graffiti_head` family — all explicitly normalise to unit norm). On the
  unit sphere, `‖x − y‖² = 2 − 2·cos(x, y)`: L2 distance is a strictly
  monotonic function of cosine distance. Any algorithm that depends only on
  the *ordering* of pairwise distances (k-NN graphs, mutual-reachability,
  single/complete/average linkage merges) returns identical clusters under
  cosine and under L2. UMAP-cosine and UMAP-euclidean should produce
  near-identical reductions for these embeddings; HDBSCAN on either is
  effectively the same algorithm.

  **L1 / Manhattan is not** monotonic in cosine on the sphere — it can reorder
  neighbours and so genuinely change the result.

  KMeans is a partial exception: centroids are vector means and drift inside
  the sphere as iterations proceed, so cosine-nearest and L2-nearest
  assignments can diverge slightly even on normalised data.

- **Unnormalised embeddings** (`resnet50`, `vgg16`, `inception_v3`,
  `mobilenet_v3`, the raw YOLO backbones). Cosine and L2 differ both in
  ordering and in magnitude, so the metric choice has measurable end-to-end
  impact.

The metric experiment below uses one of each so the contrast is observable.

## Experiments

### 1. Embedding model comparison

**Question.** Which embedding backbone produces the most semantically meaningful
graffiti clusters?

**Varies.** All 11 supported embeddings — four ImageNet CNNs (`resnet50`,
`vgg16`, `inception_v3`, `mobilenet_v3`), DINOv2, CLIP, the three YOLO
backbones, and both fine-tuned heads.

**Fixed.** UMAP → HDBSCAN baseline, identity segmenter, SQLite, `limit=1000`.

**Why it matters.** This is the single most consequential choice. Self-supervised
encoders (DINOv2, CLIP) usually beat ImageNet supervision on stylistic tasks;
fine-tuned heads test whether domain adaptation helps further. This grid also
covers the "fine-tuned vs pretrained" question directly (compare
`mobilenet_v3` ↔ `mobilenet_v3_graffiti_head` and
`dinov2_vits14` ↔ `dinov2_graffiti_head` in the results).

**Cost.** 11 runs, 11 full ingests (no DB reuse — each embedding produces a
different vector space).

**Config.** [`configs/01_embedding_comparison.json`](configs/01_embedding_comparison.json)

### 2. Clustering algorithm comparison

**Question.** Once embeddings live in a reduced space, which clustering
algorithm groups them best?

**Varies.** `hdbscan`, `kmeans`, `gmm`, `dbscan`, `optics` (cosine and
euclidean — the only clusterer where the metric is configurable),
`agglomerative`, `spectral`. `affinity_propagation` is intentionally
omitted — `best_cluster.md` rules it out as impractical.

**Fixed.** Baseline embedding and UMAP reduction.

**Why it matters.** Tests the recommendation hierarchy in `best_cluster.md`
against the actual data: HDBSCAN should win, KMeans should be the strong
baseline, the rest provide ablation evidence.

**Cost.** 8 runs, 1 ingest (the embedding is computed once and reused).

**Config.** [`configs/02_clustering_comparison.json`](configs/02_clustering_comparison.json)

### 3. Reduction technique comparison

**Question.** Does the dimensionality-reduction stage matter, and which method
helps most?

**Varies.** `identity` (no reduction), `pca` (10 and 50 components), `umap`
(cosine 10d, euclidean 10d, cosine 50d), `isomap`, `kernel_pca` with `rbf`,
`cosine`, `linear`, and `poly` kernels.

**Fixed.** Baseline embedding and HDBSCAN clustering.

**Why it matters.** Density-based clustering is sensitive to the curse of
dimensionality. The identity row anchors how much reduction buys; the UMAP rows
confirm or refute the canonical recipe. The kernel/metric variants directly
test whether non-cosine geometry pays off when feeding an L2-based clusterer
(HDBSCAN here).

**Cost.** 11 runs, 1 ingest.

**Config.** [`configs/03_reduction_comparison.json`](configs/03_reduction_comparison.json)

### 4. UMAP hyperparameter tuning

**Question.** Given UMAP is the chosen reduction, what `n_components`,
`n_neighbors`, and `min_dist` produce the cleanest cluster structure?

**Varies.** A focused sweep over each UMAP parameter while holding the others
at their defaults (one-at-a-time, not full factorial — keeps the grid small).

**Fixed.** Baseline embedding and HDBSCAN.

**Why it matters.** UMAP's local-vs-global tradeoff is controlled by
`n_neighbors`; cluster tightness by `min_dist`; statistical stability by
`n_components`. The sweep identifies the regime where HDBSCAN reliably
recovers clusters.

**Cost.** 10 runs, 1 ingest.

**Config.** [`configs/04_umap_tuning.json`](configs/04_umap_tuning.json)

### 5. HDBSCAN `min_cluster_size` tuning

**Question.** What is the smallest meaningful cluster size for this dataset?

**Varies.** `min_cluster_size` ∈ {10, 25, 50, 100, 200, 500} — sized as
fractions of a 5000-image dataset (0.2% to 10%) rather than absolute counts.

**Fixed.** Baseline embedding and UMAP.

**Why it matters.** This is HDBSCAN's main knob. Too low → noisy micro-clusters;
too high → everything labelled as noise. The right value depends on dataset
size and the expected granularity of graffiti styles.

**Cost.** 6 runs, 1 ingest.

**Config.** [`configs/05_hdbscan_tuning.json`](configs/05_hdbscan_tuning.json)

### 6. Segmenter impact

**Question.** Does YOLO cropping improve clustering over embedding the whole
image, and which detector / threshold / padding works best?

**Varies.** `identity` vs. `yolo` with `yolo26{n,s,m,l}.pt` and
`yolo11m-train-10.pt` (the fine-tuned detector). For one chosen model, also
sweeps `threshold` and `padding`.

**Fixed.** Baseline embedding, reduction, clustering.

**Why it matters.** Segmentation removes background noise but introduces
detection errors. The thesis question is whether the trade-off is worth it.

**Cost.** 9 runs, 9 full ingests (each segmenter produces different crops).
This is the most expensive experiment.

**Format.** Whitelist (not grid). The generator hashes `storage+embedding+limit`
to decide DB reuse and does not include the segmenter, so a grid would
incorrectly reuse ingests across segmenter variants. The whitelist sets
explicit `db_path` and `clear_storage: true` for every run.

**Config.** [`configs/06_segmenter_comparison.whitelist.json`](configs/06_segmenter_comparison.whitelist.json)
(feed directly to `pipeline-benchmark --whitelist`; no `generate_whitelist.py`
step).

### 7. Distance metric impact

**Question.** Does building the UMAP space with L2 (Euclidean) or L1
(Manhattan) instead of cosine change cluster quality? And how does the answer
depend on whether the upstream embedding is L2-normalised to the unit
hypersphere?

**Varies.** UMAP `metric` ∈ {`cosine`, `euclidean`, `manhattan`} crossed with
clustering ∈ {HDBSCAN (L2-locked), KMeans (L2-locked), OPTICS-cosine,
OPTICS-euclidean, OPTICS-manhattan}. Two embeddings:

- `dinov2_graffiti_head` — fine-tuned with explicit L2-normalisation,
  output lives on the unit hypersphere.
- `mobilenet_v3` — raw ImageNet features, not normalised.

**Fixed.** Identity segmenter, SQLite, `min_cluster_size=5` / `min_samples=5`.

**Hypotheses (worth stating before running so the result is interpretable).**

1. On `dinov2_graffiti_head`, UMAP-cosine and UMAP-euclidean should produce
   near-identical clusters under HDBSCAN and OPTICS, because on the unit
   sphere `‖x − y‖² = 2 − 2·cos(x, y)` makes L2 a monotonic function of
   cosine — neighbourhood orderings (and hence density-based clustering)
   are invariant. KMeans may diverge slightly because centroids drift off
   the sphere during iteration.
2. UMAP-manhattan should diverge from both even on normalised data, because
   L1 is not monotonic in cosine on the sphere.
3. On `mobilenet_v3`, all three metrics should produce different results;
   the cosine–L2 gap is real, not a relabel.

**Why it matters.** If hypothesis 1 holds, the cosine default for UMAP is
already the right choice on normalised embeddings (no "geometric mismatch" to
fix). If it fails, something in the pipeline is noisier than the theory
suggests and the choice becomes empirical. Either result tightens the thesis
recommendation.

**Cost.** 30 runs (2 embeddings × 3 UMAP metrics × 5 clusterers), 2 ingests.

**Config.** [`configs/07_metric_comparison.json`](configs/07_metric_comparison.json)

### 8. Storage backend performance

**Question.** What is the throughput difference between SQLite (linear scan)
and PostgreSQL + pgvector (indexed ANN) for ingest and similarity search across
multiple dataset sizes?

**Varies.** Storage type (`sqlite`, `postgresql`) × `limit` ∈ {500, 1000, 2500,
5000}. Similarity search is **enabled** (`top_k=5`) — this is the key axis
where the two backends differ (SQLite: O(n) Python scan; PostgreSQL: O(log n)
index). The 10× range is wide enough that the index advantage should be
empirically visible rather than buried in per-call overhead.

**Fixed.** Baseline embedding, reduction, clustering, segmenter.

**Why it matters.** Clustering quality is unaffected by storage choice; this
experiment isolates infrastructure cost. Testing at four dataset sizes reveals
whether the PostgreSQL index advantage is detectable even at the current dataset
scale (~240 images) or only emerges at larger n. The result informs the
deployment recommendation in the thesis.

**Prerequisites.** PostgreSQL requires the dev container from the `mnt/`
PostgreSQL setup at `localhost:54321`.

**Cost.** 8 runs, 8 ingests (every storage × limit combination requires a
separate DB).

**Config.** [`configs/08_storage_comparison.json`](configs/08_storage_comparison.json)

### 9. Scalability sweep

**Question.** How does wall time for each pipeline stage scale with dataset
size n, and do the O() complexity differences between clustering algorithms
manifest empirically within the available dataset range?

**Varies.** `limit` ∈ {250, 500, 1000, 2000, 5000} (geometric, spanning 20×) ×
clustering algorithm ∈ {HDBSCAN, KMeans, Agglomerative, Spectral}.

**Fixed.** Baseline embedding (DINOv2 graffiti head), UMAP reduction (10d,
cosine), SQLite, identity segmenter.

**Why it matters.** This is the primary performance characterisation experiment.
Expected complexity classes:

| Stage | Algorithm | Expected scaling |
|---|---|---|
| Ingest | Any embedding | O(n) — model inference per image |
| Reduction | UMAP | ~O(n^1.14) empirically |
| Clustering | HDBSCAN | O(n log n) amortized |
| Clustering | KMeans | O(n · k · iterations) ≈ O(n) |
| Clustering | Agglomerative | O(n² log n) |
| Clustering | Spectral | O(n²)–O(n³) |

With the 5000-image dataset, the 20× range from n=250 to n=5000 gives O(n²) and
O(n² log n) algorithms (Agglomerative, Spectral) enough room to visibly break
down against the near-linear ones (HDBSCAN, KMeans). On a log-log plot the
slope of wall time vs. n should be ~1 for the linear pair and ~2 for the
quadratic pair — that's the headline figure for the performance chapter.

**Cost.** 20 runs, 5 ingests (the embedding DB is reused across the 4 clusterers
within each limit group).

**Config.** [`configs/09_scalability_sweep.json`](configs/09_scalability_sweep.json)

### 10. Supervised validation on labeled crops

**Question.** When ground-truth style labels exist, which pipeline best recovers
those classes? Specifically: (a) does fine-tuning a graffiti-specific projection
head improve cluster–label agreement over the pretrained backbone, and (b) is
the improvement consistent across backbone families (DINOv2 self-supervised vs.
MobileNet/ResNet ImageNet-supervised vs. CLIP image–text)?

**Dataset.** [`sample_crop/`](../sample_crop/) — 273 manually labeled crops
across 4 styles: `tag` (116), `piece` (69), `throw-up` (69), `character` (19).
Labels live in [`sample_crop/labels.csv`](../sample_crop/labels.csv) in the
`filename,style` schema consumed by `pipeline-benchmark --ground-truth`. The
crops are already segmented, so the `identity` segmenter is the correct choice
(this is also a runner requirement — extrinsic metrics are only computed when
`segmenter == "identity"`, see `runner.py:266`).

**Metrics.** The runner computes the unsupervised triad (silhouette, CH, DB) as
usual, plus the supervised triad whenever ground truth is provided:

| Metric | Range | What it measures |
|---|---|---|
| Adjusted Rand Index (ARI) | [−1, 1] | Pair agreement, chance-corrected. 0 = random labelling, 1 = perfect. |
| Normalised Mutual Information (NMI) | [0, 1] | Shared information between predicted and true partitions; insensitive to cluster count. |
| Pairwise F1 | [0, 1] | Harmonic mean of pairwise precision/recall over same-cluster vs. same-class pairs; treats noise (label −1) as its own cluster. |

ARI is the headline metric (chance-corrected, comparable across runs with
different k); NMI is reported alongside to flag the "many tiny clusters
inflating NMI" failure mode; pairwise F1 is the most interpretable for the
write-up. All three live under `clustering_quality.extrinsic` in the result
JSON.

**Varies.**

- **Embedding (6).** Two fine-tuning pairs (`dinov2_vits14` ↔
  `dinov2_graffiti_head` and `mobilenet_v3` ↔ `mobilenet_v3_graffiti_head`) plus
  two strong baselines (`clip_vit_b32` self-supervised image–text, `resnet50`
  ImageNet-supervised CNN). The pairs isolate the fine-tuning effect within a
  fixed backbone; the baselines anchor where each pair sits relative to
  off-the-shelf encoders.
- **Reduction (2).** UMAP `n_components=2` (matches the 2-D scatter plots used
  for visualisation in the thesis) and UMAP `n_components=10` (the canonical
  pre-clustering reduction from the other experiments). Comparing the two
  isolates whether aggressive dimensionality reduction degrades the supervised
  metric — important context for any figure that uses the 2-D projection as
  evidence of cluster structure.
- **Clustering (3).** `kmeans` with `n_clusters=4` (forced to the ground-truth
  k — measures recovery quality given oracle k), `agglomerative` with
  `n_clusters=4` + `linkage="average"` (hierarchical alternative under oracle
  k), and `hdbscan` with `min_cluster_size=5` (density-based, auto-detects k —
  measures whether the true class count emerges naturally).

**Fixed.** SQLite storage, identity segmenter, `limit=1000` (sample_crop has
~273 images, so the limit only matters as an upper bound).

**Hypotheses.**

1. The fine-tuned heads should outperform their base models on ARI/NMI/F1 —
   **but** the heads were trained on `sample_crop`'s style-organised folders
   via triplet loss (see `src/train/dataset.py:1`), so this is a *training-set*
   evaluation, not held-out generalisation. The result establishes a ceiling
   ("the head learned the training distribution") rather than out-of-sample
   transfer. State this caveat explicitly in any figure caption.
2. CLIP should be competitive with DINOv2 base; ResNet50 should be the weakest
   off-the-shelf encoder for stylistic clustering, consistent with the broader
   embedding comparison (#1).
3. KMeans-4 should achieve the highest extrinsic scores on the fine-tuned
   heads because k matches ground truth and the heads were trained to make
   classes linearly separable; HDBSCAN should auto-discover a number of
   clusters near 4 on the fine-tuned heads and noticeably different on the
   baselines.
4. UMAP-2 should slightly underperform UMAP-10 on the extrinsic metrics
   (information loss at the projection step) but the gap should be small for
   fine-tuned embeddings (the head has already concentrated discriminative
   information) and larger for raw ImageNet features. If UMAP-2 *matches*
   UMAP-10 even on the unfine-tuned models, the 2-D visualisations in the
   thesis are honest representations of the cluster structure rather than
   artefacts of compression.

**Why it matters.** Every other experiment in this catalogue uses unsupervised
proxies (silhouette, CH, DB) as the quality signal — these reward
within-cluster density and between-cluster separation regardless of whether the
clusters correspond to anything semantically meaningful. The supervised
validation is the single experiment where "good clustering" is defined against
human-labelled categories. Two derived analyses follow naturally:

- **Internal–external correlation.** Plot ARI vs. silhouette over the 36 runs.
  A strong positive correlation means the unsupervised metrics are a defensible
  proxy on this dataset and we can trust them in the experiments without
  ground truth; a weak or zero correlation means the unsupervised optima do
  not coincide with semantic optima, and the thesis must hedge its
  recommendations accordingly.
- **k-recovery for HDBSCAN.** Whether HDBSCAN's auto-detected cluster count
  lands near 4 on the fine-tuned heads is the cleanest evidence for/against
  the "density-based clustering discovers the true number of styles" claim.
  The `n_clusters` field is already in every result row.

**Cost.** 36 runs (6 embeddings × 2 reductions × 3 clusterers), 6 ingests (one
per embedding; reduction and clustering re-use the cached DB within an
embedding group thanks to `generate_whitelist.py`'s storage-key hashing).

**Running.**

```bash
cd src

# 1. Expand the grid (36 runs)
uv run python benchmarks/generate_whitelist.py \
  -i ../infos/configs/10_supervised_validation.json \
  -o benchmarks/supervised_validation.whitelist.json

# 2. Run, pointing at sample_crop and passing the labels for extrinsic metrics
uv run pipeline-benchmark \
  --whitelist benchmarks/supervised_validation.whitelist.json \
  --dataset ../sample_crop \
  --ground-truth ../sample_crop/labels.csv \
  --cluster-plot

# 3. Plot — the extrinsic columns (ari, nmi, pairwise_f1) are auto-detected
uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/
```

`--cluster-plot` writes a 2-D UMAP scatter per run alongside the metrics —
useful here precisely because ground-truth labels can be overlaid for visual
sanity-checking.

**Config.** [`configs/10_supervised_validation.json`](configs/10_supervised_validation.json)

## Suggested order

1. **#1 Embedding comparison** first — fixes the strongest variable so later
   experiments use the actual best embedding rather than a guess.
2. **#3 Reduction** and **#2 Clustering** next, using the winner of #1.
3. **#4 UMAP** and **#5 HDBSCAN** to fine-tune the chosen reduction/clustering
   pair.
4. **#7 Distance metric impact** — once a UMAP+clusterer pair is chosen, check
   whether swapping the metric from cosine to L2/L1 changes anything.
5. **#6 Segmenter** once the rest of the pipeline is settled — it is the most
   expensive and benefits from comparison against an already-strong baseline.
6. **#9 Scalability sweep** and **#8 Storage** independently — both are
   performance-only and can run in any order or in parallel with the quality
   experiments once the baseline pipeline is fixed.
7. **#10 Supervised validation** at any point after #1 — provides the
   ground-truth anchor for the unsupervised metrics used throughout the
   other experiments. Running it early (right after #1) lets later
   experiments cite the internal–external correlation when defending the
   silhouette/CH/DB scores as proxies for semantic quality.
