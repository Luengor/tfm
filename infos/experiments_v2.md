# Experiment Catalogue v2

Redesigned around the three pillars of the TFM21 brief
([`doc/enunciado.md`](../doc/enunciado.md)):

1. **Computational cost as a function of dataset size** — the *headline*
   contribution of the thesis. Characterised experimentally across the full
   pipeline (ingest, reduction, clustering, similarity search).
2. **Unsupervised study of cluster quality** — silhouette / Calinski–Harabasz /
   Davies–Bouldin against the same baseline pipeline, while varying one axis at
   a time.
3. **Supervised validation on the labelled subset** — ARI / NMI / pairwise F1
   on `sample_crop/` (273 hand-labelled crops over 4 styles), the only
   experiment with ground truth.

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
- **Limit:** `1000` images for quality experiments, `5000` for cost-focused
  experiments and for HDBSCAN tuning where density matters.

The working dataset contains roughly 5000 images. The 273 labelled crops in
[`sample_crop/`](../sample_crop/) are a subset of the full image bank and are
*not* deduplicated out for the unsupervised experiments — at 5 % of the corpus
they do not perturb aggregate statistics, and segregating them would
artificially shrink the cost-sweep dataset.

## Metrics

**Computational cost (per stage).** Wall time, CPU time, peak RSS, peak VRAM.
Stages: `setup`, `ingest`, `reduction`, `clustering`, `similarity_search`.
Throughput columns (`ingest_throughput_ips`, `clustering_throughput_ips`) are
derived automatically. Each configuration runs once; differences smaller than
~15 % should be read as within-noise. For the scalability sweep (§7) the trend
across `n` is the headline figure, not any single point.

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

# 1. Expand the grid into a whitelist (skip for §5 which is already a whitelist)
uv run python benchmarks/generate_whitelist.py \
  -i ../infos/configs/v2/01_embedding.json \
  -o benchmarks/v2_01_embedding.whitelist.json

# 2. Run the benchmark
uv run pipeline-benchmark \
  --whitelist benchmarks/v2_01_embedding.whitelist.json \
  --dataset ../dataset/images

# 3. Plot
uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/
```

For §8 (supervised) add `--ground-truth ../sample_crop/labels.csv` and point
`--dataset` at `../sample_crop`. For §5 (segmenter) skip step 1 — the file is
already a whitelist (segmenter changes invalidate cached crops, so explicit
`db_path` + `clear_storage: true` per row is required).

---

## §1. Embedding model comparison

**Pillar.** Unsupervised quality (2).

**Question.** Which embedding backbone produces the most semantically coherent
graffiti clusters under a fixed downstream pipeline?

**Varies.** Eleven embeddings: four ImageNet CNNs (`resnet50`, `vgg16`,
`inception_v3`, `mobilenet_v3`), DINOv2 ViT-S/14, CLIP ViT-B/32, three YOLO
backbones (`yolon`, `yolos`, `yolom`), and the two fine-tuned heads
(`mobilenet_v3_graffiti_head`, `dinov2_graffiti_style_head`). The author /
identity-recovery head (`dinov2_graffiti_author_head`) is omitted — its
training objective targets a different question.

**Fixed.** Baseline UMAP → HDBSCAN, identity segmenter, SQLite, `limit=1000`.

**Why it matters.** This is the single most consequential choice in the
pipeline. Pairs `mobilenet_v3` ↔ `mobilenet_v3_graffiti_head` and
`dinov2_vits14` ↔ `dinov2_graffiti_style_head` answer the fine-tuning vs.
pretraining question directly.

**Cost.** 11 runs, 11 full ingests (each embedding lives in a different vector
space, no DB reuse).

**Config.** [`configs/v2/01_embedding.json`](configs/v2/01_embedding.json)

## §2. Reduction technique comparison

**Pillar.** Unsupervised quality (2).

**Question.** Does the dimensionality-reduction stage matter for HDBSCAN, and
which family helps most?

**Varies.** `identity` (no reduction), `pca` (10 and 50 components), `umap`
(10d and 50d cosine), `isomap` (10d), `kernel_pca` (50d, RBF and cosine
kernels). Linear/poly KernelPCA kernels and UMAP-50/100 sweeps from v1 were
trimmed — they did not contribute distinctive evidence in early runs.

**Fixed.** Baseline embedding and HDBSCAN.

**Why it matters.** Density-based clustering degrades in high dimensions. The
identity row quantifies how much reduction buys; the UMAP rows confirm or
refute the canonical recipe; the kernel/Isomap rows show whether non-linear
reductions help.

**Cost.** 8 runs, 1 ingest (the embedding DB is reused).

**Config.** [`configs/v2/02_reduction.json`](configs/v2/02_reduction.json)

## §3. Clustering algorithm comparison

**Pillar.** Unsupervised quality (2).

**Question.** Given fixed embeddings and reduction, which clustering algorithm
groups them best?

**Varies.** `hdbscan`, `kmeans` (auto-k via elbow), `gmm` (auto-k), `dbscan`,
`optics` (cosine), `agglomerative` (n_clusters ∈ {10, 20}), `spectral`
(n_clusters ∈ {10, 20}). `affinity_propagation` is omitted as impractical at
this n (see [`best_cluster.md`](best_cluster.md)). The OPTICS-euclidean and
agglomerative-5 rows from v1 were trimmed for redundancy.

**Fixed.** Baseline embedding and UMAP.

**Why it matters.** Tests the algorithmic hierarchy recommended in
`best_cluster.md` against the actual data and provides ablation evidence for
the HDBSCAN default. The sweep over `n_clusters` for agglomerative and
spectral removes the k confound when comparing against HDBSCAN's auto-k.

**Cost.** 9 runs, 1 ingest.

**Config.** [`configs/v2/03_clustering.json`](configs/v2/03_clustering.json)

## §4. HDBSCAN `min_cluster_size` tuning

**Pillar.** Unsupervised quality (2).

**Question.** What is the smallest meaningful cluster size for this dataset?

**Varies.** `min_cluster_size ∈ {5, 10, 25, 50, 100, 200}` — sized as
fractions of the 5000-image corpus (0.1 % to 4 %).

**Fixed.** Baseline embedding and UMAP. `limit=5000` so the fractions resolve
to meaningful absolute counts and the density estimates are stable.

**Why it matters.** This is HDBSCAN's primary knob. Too low → noisy
micro-clusters; too high → most points fall into noise (label −1). The
sweep also produces the noise-fraction curve that feeds back into the
narrative: if noise climbs sharply past a threshold, that's the
upper bound on cluster granularity the dataset supports.

UMAP hyperparameter tuning (`n_components`, `n_neighbors`, `min_dist`) is
intentionally omitted from v2: the canonical 10-d / 15-neighbour / `min_dist=0`
setting is well-established for HDBSCAN preprocessing and the v1 sweep
produced no surprises. It can be reinstated from
[`configs/04_umap_tuning.json`](configs/04_umap_tuning.json) if the v2
reduction sweep (§2) flags UMAP as marginal.

**Cost.** 6 runs, 1 ingest.

**Config.** [`configs/v2/04_hdbscan.json`](configs/v2/04_hdbscan.json)

## §5. Segmenter impact

**Pillar.** Unsupervised quality (2), with a cost side-effect.

**Question.** Does YOLO cropping improve clustering quality over embedding the
whole image, and which detector / threshold / padding works best?

**Varies.** `identity` vs. `yolo` with `yolo26{n,s,m}.pt` and
`yolo11m-train-10.pt` (fine-tuned detector). For `yolo26s`, an additional
sweep over `threshold ∈ {0.3, 0.5, 0.7}` and `padding ∈ {0.05, 0.2}`. The
`yolo26l` row from v1 was dropped — at this scale the gap between `m` and `l`
is dominated by compute cost, not detection quality.

**Fixed.** Baseline embedding, reduction, clustering. `limit=1000`.

**Why it matters.** Segmentation removes background noise but introduces
detection errors and changes the unit of analysis (image vs. crop). The
thesis question is whether the trade-off is worth it; the per-run timing
captures the cost side directly.

**Format.** Whitelist (not a grid). `generate_whitelist.py` hashes
`storage + embedding + limit` to decide DB reuse and does *not* include the
segmenter, so a grid would incorrectly reuse ingests across segmenter
variants. The whitelist therefore sets explicit `db_path` and
`clear_storage: true` for every row.

**Cost.** 8 runs, 8 full ingests — the most expensive quality experiment.

**Config.**
[`configs/v2/05_segmenter.whitelist.json`](configs/v2/05_segmenter.whitelist.json)
(feed directly to `pipeline-benchmark --whitelist`; no `generate_whitelist.py`
step).

## §6. Storage backend cost

**Pillar.** Computational cost (1).

**Question.** What is the throughput gap between SQLite (linear Python scan)
and PostgreSQL + pgvector (indexed ANN) for ingest and similarity search,
and how does it scale with `n`?

**Varies.** `storage ∈ {sqlite, postgresql}` × `limit ∈ {500, 1000, 2500,
5000}`. Similarity search is **enabled** (`top_k=5`) — that is the axis where
the two backends differ most (O(n) Python scan vs. O(log n) index).

**Fixed.** Baseline embedding, reduction, clustering, identity segmenter.

**Why it matters.** Clustering quality is invariant under storage choice, so
this experiment isolates pure infrastructure cost. The 10× `limit` span is
wide enough that the index advantage should be empirically visible rather
than buried in per-call overhead.

**Prerequisites.** PostgreSQL via the dev container at `localhost:54321`
(`postgresql://postgres:changethis@localhost:54321/postgres`).

**Cost.** 8 runs, 8 ingests — every storage × limit combination needs its own
DB (different storage type or different size invalidates the cached one).

**Config.** [`configs/v2/06_storage.json`](configs/v2/06_storage.json)

## §7. Pipeline scalability sweep (HEADLINE)

**Pillar.** Computational cost (1) — *the primary deliverable of the thesis.*

**Question.** How does wall time scale with `n` for each pipeline stage, and
do the asymptotic complexity differences between clustering algorithms
manifest empirically within the available dataset range?

**Varies.** `limit ∈ {100, 250, 500, 1000, 2000, 3500, 5000}` (~50× span) ×
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

**Cost.** 42 runs, 7 ingests (the embedding DB is reused across the 6
clusterers within each limit group).

**Config.** [`configs/v2/07_scalability.json`](configs/v2/07_scalability.json)

## §8. Supervised validation on labelled crops

**Pillar.** Supervised validation (3).

**Question.** When ground-truth style labels are available, which pipeline best
recovers them? Specifically: (a) does fine-tuning a graffiti-specific
projection head improve cluster–label agreement over the pretrained backbone,
and (b) is the improvement consistent across backbone families?

**Dataset.** [`sample_crop/`](../sample_crop/) — 273 manually labelled crops
across 4 styles: `tag` (116), `piece` (69), `throw-up` (69), `character` (19).
Labels in [`sample_crop/labels.csv`](../sample_crop/labels.csv) follow the
`filename,style` schema consumed by `--ground-truth`. The crops are already
segmented, so `identity` is the correct (and required, per `runner.py:266`)
segmenter for extrinsic metrics.

**Metrics.** The intrinsic triad (silhouette, CH, DB) plus the extrinsic
triad:

| Metric | Range | Reads as |
|---|---|---|
| Adjusted Rand Index (ARI) | [−1, 1] | Pair agreement, chance-corrected. 0 = random, 1 = perfect. |
| Normalised Mutual Information (NMI) | [0, 1] | Shared info between predicted and true partitions. |
| Pairwise F1 | [0, 1] | Harmonic mean of pairwise precision / recall; treats noise (−1) as its own cluster. |

ARI is the headline (chance-corrected, comparable across runs with different
k); NMI is reported alongside to flag the *many-tiny-clusters-inflates-NMI*
failure mode; pairwise F1 is the most interpretable for write-up.

**Varies.**

- **Embedding (4).** `dinov2_vits14` ↔ `dinov2_graffiti_style_head`
  (fine-tuning pair, fixed backbone) plus `clip_vit_b32` (self-supervised
  image–text) and `resnet50` (ImageNet-supervised CNN) as anchors.
- **Reduction (3).** `identity` (raw embedding space), UMAP `n_components=2`
  (matches the 2-D scatter plots used in the thesis), UMAP `n_components=10`
  (the canonical pre-clustering reduction).
- **Clustering (6).** `kmeans`, `agglomerative`, `spectral` all with
  `n_clusters=4` (oracle k); `hdbscan` swept over
  `min_cluster_size ∈ {5, 10, 20}` (density-based auto-k — tests whether
  the true k=4 emerges and how sensitive that is to the main knob given
  class sizes `tag=116, piece=69, throw-up=69, character=19`).

**Fixed.** SQLite, identity segmenter, no `limit` (sample_crop is small
enough that every run processes the full 273).

**Hypotheses.**

1. The fine-tuned style head outperforms `dinov2_vits14` on ARI / NMI / F1
   — *but* the head was trained on `sample_crop`'s style folders via
   supervised contrastive loss (see `src/train/style_trainer.py`), so this is
   a *training-set* evaluation, not held-out generalisation. State the
   caveat explicitly in every caption.
2. CLIP is competitive with DINOv2 base; ResNet50 is the weakest off-the-shelf
   encoder for stylistic clustering, consistent with §1.
3. KMeans-4 (oracle k) wins on the fine-tuned head because the head was
   trained to make classes linearly separable; HDBSCAN auto-discovers a
   count near 4 on the fine-tuned head at moderate `min_cluster_size` (≈10–20).
4. `identity` reduction matches or beats UMAP on the fine-tuned head
   (further reduction is information loss when the encoder has already
   concentrated discriminative axes) and underperforms UMAP on raw ImageNet
   features. UMAP-2 ≈ UMAP-10 on the fine-tuned head ⇒ the 2-D thesis
   visualisations are honest representations rather than compression
   artefacts.

**Derived analyses.**

- **Internal–external correlation.** Plot ARI vs. silhouette over the 72 runs.
  A strong positive correlation means the unsupervised metrics are a
  defensible proxy on this dataset; a weak one means the thesis must hedge
  its recommendations elsewhere.
- **k-recovery for HDBSCAN.** Does HDBSCAN's auto-detected `n_clusters` land
  near 4 on the fine-tuned head across the `min_cluster_size` sweep? Report
  alongside `noise_ratio` so the "inflated NMI from many tiny clusters"
  failure mode is visible.

**Cost.** 72 runs (4 embeddings × 3 reductions × 6 clusterers), 4 ingests
(one per embedding; reduction and clustering re-use the cached DB within an
embedding group thanks to storage-key hashing in `generate_whitelist.py`).

**Running.**

```bash
cd src

uv run python benchmarks/generate_whitelist.py \
  -i ../infos/configs/v2/08_supervised.json \
  -o benchmarks/v2_08_supervised.whitelist.json

uv run pipeline-benchmark \
  --whitelist benchmarks/v2_08_supervised.whitelist.json \
  --dataset ../sample_crop \
  --ground-truth ../sample_crop/labels.csv \
  --cluster-plot

uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/
```

`--cluster-plot` writes an interactive 2-D UMAP scatter per run as a
Vega-Lite HTML page (`<output-dir>/<run_id>_cluster.html`) with embedded
metrics, pan/zoom, shift-drag brush selection, and a thumbnail lightbox —
useful here because ground-truth labels can be overlaid for visual
sanity-checking.

**Config.** [`configs/v2/08_supervised.json`](configs/v2/08_supervised.json)

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
6. **§8 Supervised validation** can run any time after §1 — running it early
   lets later experiments cite the internal–external correlation when
   defending silhouette / CH / DB as proxies for semantic quality.

## Relationship to v1

The v1 catalogue ([`experiments.md`](experiments.md)) had ten experiments
including UMAP hyperparameter tuning (#4) and a distance-metric / sphere-
geometry deep dive (#7). v2 trims both: the canonical UMAP setting is
well-established for HDBSCAN preprocessing, and the metric-impact analysis
is theoretical context for the thesis chapter rather than a separate
empirical sweep. The remaining eight experiments are reused from v1 with
minor edits (a couple of redundant rows removed; the scalability sweep §7
expanded with more `n` points, a sixth clusterer, and similarity search to
make it the genuine cost-vs-`n` headline). v1 configs remain on disk under
[`configs/`](configs/) for reference; v2 configs live under
[`configs/v2/`](configs/v2/).
