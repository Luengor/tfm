# Experiment Catalogue

Redesigned around the three pillars of the TFM21 brief
([`doc/enunciado.md`](../doc/enunciado.md)):

1. **Computational cost as a function of dataset size** — the *headline*
   contribution of the thesis. Characterised experimentally across the full
   pipeline (ingest, reduction, clustering, similarity search).
2. **Unsupervised study of cluster quality** — silhouette / Calinski–Harabasz /
   Davies–Bouldin against the same baseline pipeline, while varying one axis at
   a time.
3. **Supervised validation on the labelled subset** — ARI / NMI / pairwise F1
   on `data/style/eval_crop/` (274 hand-labelled crops over 4 styles), the only
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
- **Limit:** every experiment is run at three geometrically spaced sizes —
  `{250, 1000, 5500}` (≈22× span) — so each quality result is reported at
  small, mid and near-full corpus. Three points are enough to spot
  non-monotonic quality-vs-`n` behaviour without forcing each grid into a
  full cost-sweep. Cost-focused experiments (§6, §7) keep their finer
  explicit `n` sweeps.
- **Repeats:** `K=3` for quality experiments, `K=5` for cost experiments
  (§6, §7). Reduction + clustering loop K times per run; iter 0 is discarded
  to absorb JIT warm-up; the remaining K-1 samples produce mean ± std for
  wall time, CPU, RSS, and VRAM. Ingest stays single-shot. See the
  **Metrics** section for the full rationale.

The working dataset contains roughly 6000 images. The 274 labelled eval
crops in [`data/style/eval_crop/`](../data/style/eval_crop/) (and the 178
train crops in [`data/style/train_crop/`](../data/style/train_crop/)) are
derived from the full image bank and are *not* deduplicated out for the
unsupervised experiments — at <10 % of the corpus they do not perturb
aggregate statistics, and segregating them would artificially shrink the
cost-sweep dataset.

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
`table.py` keep working unchanged. Quality and extrinsic metrics are computed
once, on the last iteration's labels (seed `K-1`, reproducible). Ingest and
similarity-search remain single-shot — ingest is deterministic and the
DB-reuse heuristic depends on a single timing.

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

For §8 (supervised) add `--ground-truth ../data/style/eval_crop/labels.csv`
and point `--dataset` at `../data/style/eval_crop`. For §5 (segmenter) skip
step 1 — the file is
already a configuration (segmenter changes invalidate cached crops, so explicit
`db_path` + `clear_storage: true` per row is required).

---

## §1. Embedding model comparison

**Pillar.** Unsupervised quality (2).

**Question.** Which embedding backbone produces the most semantically coherent
graffiti clusters under a fixed downstream pipeline?

**Varies.** Eleven embeddings: four ImageNet CNNs (`resnet50`, `vgg16`,
`inception_v3`, `mobilenet_v3`), DINOv2 ViT-S/14, CLIP ViT-B/32, three YOLO
backbones (`yolon`, `yolos`, `yolom`), and the two fine-tuned heads
(`mobilenet_v3_graffiti_author_head`, `dinov2_graffiti_style_head`). The author /
identity-recovery head (`dinov2_graffiti_author_head`) is omitted — its
training objective targets a different question.

**Fixed.** Baseline UMAP → HDBSCAN, identity segmenter, SQLite. Swept at
`limit ∈ {250, 1000, 5500}`.

**Why it matters.** This is the single most consequential choice in the
pipeline. Pairs `mobilenet_v3` ↔ `mobilenet_v3_graffiti_author_head` and
`dinov2_vits14` ↔ `dinov2_graffiti_style_head` answer the fine-tuning vs.
pretraining question directly.

**Cost.** 33 runs (11 embeddings × 3 limits), 33 full ingests (each
embedding × limit pair gets its own DB; no reuse across embeddings). The
`limit=250` ingests are cheap, so the marginal cost over the 2-limit
variant is dominated by the 11 small-`n` runs.

**Config.** [`configs/01_embedding.json`](configs/01_embedding.json)

## §2. Reduction technique comparison

**Pillar.** Unsupervised quality (2).

**Question.** Does the dimensionality-reduction stage matter for HDBSCAN, and
which family helps most?

**Varies.** `identity` (no reduction), `pca` (10 and 50 components), `umap`
(10d and 50d cosine), `isomap` (10d), `kernel_pca` (50d, RBF and cosine
kernels). Linear/poly KernelPCA kernels and UMAP-50/100 sweeps from v1 were
trimmed — they did not contribute distinctive evidence in early runs.

**Fixed.** Baseline embedding and HDBSCAN. Swept at
`limit ∈ {250, 1000, 5500}`.

**Why it matters.** Density-based clustering degrades in high dimensions. The
identity row quantifies how much reduction buys; the UMAP rows confirm or
refute the canonical recipe; the kernel/Isomap rows show whether non-linear
reductions help.

**Cost.** 24 runs (8 reductions × 3 limits), 3 ingests (one per limit — the
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
`limit ∈ {250, 1000, 5500}`.

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
fractions of the 5500-image corpus (≈0.1 % to ≈3.6 %).

**Fixed.** Baseline embedding and UMAP. Swept at
`limit ∈ {250, 1000, 5500}` so the same `min_cluster_size` axis is read
across three corpus sizes; the 5500-image row is where the fractions
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

**Varies.** `identity` vs. `yolo` with `yolo11{n,s,m}.pt` and
`yolo11m-train-10.pt` (fine-tuned detector) — five runs total. Earlier
threshold / padding sweeps and the `yolo11l` row were dropped: at this
scale, the gap between `m` and `l` is dominated by compute cost, not
detection quality, and the threshold/padding axis did not produce
distinctive signal.

**Fixed.** Baseline embedding, reduction, clustering. Each segmenter row is
expanded to `limit ∈ {250, 1000, 5500}`.

**Why it matters.** Segmentation removes background noise but introduces
detection errors and changes the unit of analysis (image vs. crop). The
thesis question is whether the trade-off is worth it; the per-run timing
captures the cost side directly.

**Format.** Configuration (not a grid). `generate_configuration.py` hashes
`storage + embedding + limit` to decide DB reuse and does *not* include the
segmenter, so a grid would incorrectly reuse ingests across segmenter
variants. The configuration therefore sets explicit `db_path` and
`clear_storage: true` for every row.

**Cost.** 15 runs (5 segmenters × 3 limits), 15 full ingests — the most
expensive quality experiment.

**Config.**
[`configs/05_segmenter.config.json`](configs/05_segmenter.config.json)
(feed directly to `pipeline-benchmark --configuration`; no `generate_configuration.py`
step).

## §6. Storage backend cost

**Pillar.** Computational cost (1).

**Question.** What is the throughput gap between SQLite (linear Python scan)
and PostgreSQL + pgvector (indexed ANN) for ingest and similarity search,
and how does it scale with `n`?

**Varies.** `storage ∈ {sqlite, postgresql}` × `limit ∈ {500, 1000, 2500,
5500}`. Similarity search is **enabled** (`top_k=5`) — that is the axis where
the two backends differ most (O(n) Python scan vs. O(log n) index).

**Fixed.** Baseline embedding, reduction, clustering, identity segmenter.

**Why it matters.** Clustering quality is invariant under storage choice, so
this experiment isolates pure infrastructure cost. The 11× `limit` span is
wide enough that the index advantage should be empirically visible rather
than buried in per-call overhead.

**Prerequisites.** PostgreSQL via the dev container at `localhost:54321`
(`postgresql://postgres:changethis@localhost:54321/postgres`).

**Cost.** 8 runs at `repeats=5`, 8 ingests — every storage × limit
combination needs its own DB (different storage type or different size
invalidates the cached one). Similarity-search wall time is single-shot per
run (not looped), so the K=5 repeats only multiply reduction + clustering
cost; the SQLite-vs-pgvector throughput gap on similarity search is the
headline figure here, and its variance comes from inter-run rather than
intra-run measurements — read it cautiously against the cost legend in §7.

**Config.** [`configs/06_storage.json`](configs/06_storage.json)

## §7. Pipeline scalability sweep (HEADLINE)

**Pillar.** Computational cost (1) — *the primary deliverable of the thesis.*

**Question.** How does wall time scale with `n` for each pipeline stage, and
do the asymptotic complexity differences between clustering algorithms
manifest empirically within the available dataset range?

**Varies.** `limit ∈ {100, 250, 500, 1000, 2000, 3500, 5500}` (~55× span) ×
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

**Question.** When ground-truth style labels are available, which pipeline best
recovers them? Specifically: (a) does fine-tuning a graffiti-specific
projection head improve cluster–label agreement over the pretrained backbone,
and (b) is the improvement consistent across backbone families?

**Dataset.** [`data/style/eval_crop/`](../data/style/eval_crop/) — 274 manually
labelled crops across 4 styles: `tag` (117), `piece` (69), `throw-up` (69),
`character` (19). Labels in
[`data/style/eval_crop/labels.csv`](../data/style/eval_crop/labels.csv) follow the
`filename,style` schema consumed by `--ground-truth`. This is the held-out
split — the fine-tuned style head was trained on the disjoint
[`data/style/train_crop/`](../data/style/train_crop/) (178 crops:
`tag` 54, `piece` 50, `throw-up` 55, `character` 19), so §8 measures
generalisation, not train-set memorisation. The crops are already
segmented, so `identity` is the correct (and required, per the
segmenter check in `src/src/evaluation/runner.py`) segmenter for extrinsic metrics.

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
  class sizes `tag=117, piece=69, throw-up=69, character=19`).

**Fixed.** SQLite, identity segmenter, no `limit` (`eval_crop` is small
enough that every run processes the full 274 crops).

**Hypotheses.**

1. The fine-tuned style head outperforms `dinov2_vits14` on ARI / NMI / F1.
   The head is trained with supervised contrastive loss on `data/style/train_crop/`
   (see `src/train/style_trainer.py`); §8 evaluates on the disjoint
   `data/style/eval_crop/`, so the comparison is held-out and a head win is
   genuine transfer rather than memorisation.
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
embedding group thanks to storage-key hashing in `generate_configuration.py`).

**Running.**

```bash
cd src

uv run python benchmarks/generate_configuration.py \
  -i ../infos/configs/08_supervised.json \
  -o benchmarks/08_supervised.configuration.json

uv run pipeline-benchmark \
  --configuration benchmarks/08_supervised.configuration.json \
  --dataset ../data/style/eval_crop \
  --ground-truth ../data/style/eval_crop/labels.csv \
  --cluster-plot

uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/
```

`--cluster-plot` writes an interactive 2-D UMAP scatter per run as a
Vega-Lite HTML page (`<output-dir>/<run_id>_cluster.html`) with embedded
metrics, pan/zoom, shift-drag brush selection, and a thumbnail lightbox —
useful here because ground-truth labels can be overlaid for visual
sanity-checking.

**Config.** [`configs/08_supervised.json`](configs/08_supervised.json)

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

## History

An earlier catalogue had ten experiments including UMAP hyperparameter
tuning and a distance-metric / sphere-geometry deep dive. Both were
trimmed: the canonical UMAP setting is well-established for HDBSCAN
preprocessing, and the metric-impact analysis is theoretical context for
the thesis chapter rather than a separate empirical sweep. The remaining
eight experiments live under [`configs/`](configs/), with the scalability
sweep §7 expanded (more `n` points, a sixth clusterer, and similarity
search) to make it the genuine cost-vs-`n` headline.
