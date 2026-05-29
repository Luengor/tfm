<!-- LTeX: language=en-US -->

# §2 — Reduction Technique Comparison: Report

**Source:** `exp/exp02/output/benchmark_20260528T093147Z.json`
**Date:** 2026-05-28 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 18/18 runs `success`.

## Setup

Pillar 2 (unsupervised quality). Question: does the dimensionality-reduction
stage matter for HDBSCAN, and which family helps most?

Fixed baseline (one axis varied = `reduction`):

| Component | Value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style head, L2-normalised, 384-d) |
| Clustering | HDBSCAN, `min_cluster_size=5` |
| Segmenter | `identity` (whole image) |
| Storage | SQLite |
| Repeats | `K=3` (reduction+clustering looped; iter 0 dropped) |

6 reductions × 3 corpus sizes `{250, 1000, 6416}` = 18 runs. Reductions:
`identity` (no reduction, raw 384-d), `pca` (10-d, 50-d), `umap` (10-d, 50-d;
cosine, n_neighbors=15, min_dist=0), `isomap` (10-d, n_neighbors=10). The
embedding DB is reused across reductions within each corpus size (3 ingests
total).

> **Reading note — silhouette/CH/DB are graded on non-noise points only.** The
> three distance metrics are computed after dropping points labelled −1
> (`clustering_metrics.py:54-93`). A reduction that rejects many points (e.g.
> `isomap@250`, 58 % noise) is therefore *not* rewarded with an inflated score
> over its few survivors — the score reflects only the dense core. Silhouette
> must be read jointly with `noise_ratio` and `cluster_count`, never alone, as
> §4 also documented.

## Summary — quality

`cls` = cluster count, `noise` = fraction labelled −1, `sil` = silhouette
(micro, higher better, non-noise only), `silM` = macro silhouette (per-cluster
mean, unweighted by size), `CH` = Calinski–Harabasz (higher better), `DB` =
Davies–Bouldin (lower better), `csCV` = cluster-size CV (lower = more balanced).

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

† `pca50@6416`: k=188 but **62.7 % noise** and csCV=5.61 — the
high-noise/unbalanced failure mode (see below), not a usable partition.

## Summary — cost

Reduction is the axis under test here, so unlike §4 its cost *varies*. Ingest
is shared per corpus size (56.7 / 221.4 / 1267.5 s for 250 / 1000 / 6416) and
not a reduction cost. `red_s` = reduction wall time, `red_pkRSS` = reduction
peak RSS above baseline, `clu_s` = clustering wall time.

| n | reduction | red_s | red_pkRSS MB | clu_s | clu_pkRSS MB |
|--:|--|--:|--:|--:|--:|
| 6416 | identity | 0.001 | 0.1 | **24.123** | 21.1 |
| 6416 | pca-10 | 0.197 | 2.7 | 0.375 | 2.1 |
| 6416 | pca-50 | 0.186 | 13.1 | 1.576 | 2.1 |
| 6416 | umap-10 | 6.158 | 58.4 | 0.341 | 0.3 |
| 6416 | umap-50 | 11.206 | 63.8 | 0.903 | 0.3 |
| 6416 | isomap | **23.646** | **1222.2** | 0.369 | 1.7 |

(250 / 1000 rows follow the same ordering at smaller magnitude — see JSON.)
Two costs matter: the reduction stage itself and the clustering cost it
induces. Unlike §4, **there is a real cost–quality trade-off on this axis.**

## Reduction matters — a lot (the curse-of-dimensionality story)

The `identity` row answers "how much does reduction buy?" directly: clustering
the raw 384-d embeddings with HDBSCAN yields **only 2–4 clusters at every
corpus size** (2 @ 250, 4 @ 1000, 2 @ 6416). Density estimation degrades in
high dimensions — distances concentrate, HDBSCAN cannot separate fine
structure, and the corpus collapses into a near-binary split. UMAP-10 on the
same data yields 23 → 96 → 563 clusters. Reduction is not a tuning detail; it
is what makes density clustering work at all here.

The cost half is easy to miss because `identity` reduction is nominally free
(`red_s ≈ 0`). It is not free — it **pushes the cost into clustering**:
`clu_s@6416` is **24.1 s** for identity versus **0.34 s** for UMAP-10, a ~70×
penalty. HDBSCAN in 384-d is both slower *and* worse. So skipping reduction
loses on both quality and cost simultaneously.

## The reading hazard: degenerate cells score *better* on sil/CH/DB

As in §4, a naive "best silhouette / DB wins" reading is inverted by the
trivial cells:

- **`identity@250` posts the best DB in the 250 group (1.11) and the
  second-highest macro silhouette (0.386)** — but it is **k=2**, a single
  binary split, semantically empty.
- **`isomap@250` has the highest micro silhouette in its group (0.279)** — while
  discarding **58 % of points as noise**. The score grades only the dense
  survivor core.
- The **macro** silhouette *amplifies* this trap: by unweighting cluster sizes
  it rewards the few fat clusters of the k=2 cells (identity/pca-50/isomap all
  score silM ≈ 0.32–0.39, well above UMAP's 0.20–0.26). Read alone it would
  rank the degenerate reductions first.

On the joint criterion — non-trivial cluster count, moderate noise, balanced
clusters — **UMAP wins at every corpus size** and the others are not close.

## Per-family verdict

1. **Identity (no reduction): fails.** k≤4 everywhere, slowest clustering.
   Quantifies that reduction is mandatory for HDBSCAN on these embeddings.
2. **PCA (linear): not robust.** PCA-10 collapses to k=2 at 6416 with
   **negative silhouette (−0.028)** — clusters worse than chance. PCA-50 swings
   the other way at 6416: k=188 but 62.7 % noise and csCV=5.61 (the
   mid-range high-noise/unbalanced failure §4 documented, here induced by the
   reduction choice). A linear projection cannot give HDBSCAN a stable density
   landscape across corpus sizes.
3. **Isomap (non-linear): worst on cost *and* quality.** High noise at small n
   (0.58 @ 250, 0.58 @ 1000) then collapse to k=2 at 6416. It is also by far
   the most expensive reduction: **23.6 s and 1.2 GB peak RSS at 6416** — the
   O(n²) geodesic kNN graph. Non-linearity does not help here; it hurts.
4. **UMAP (non-linear, neighbour-graph): wins.** The only reduction that is
   simultaneously non-trivial, moderate-noise and balanced across 250 → 6416.
   Cluster count scales sensibly (23 → 96 → 563), noise stays moderate
   (0.06 → 0.11 → 0.20), clusters stay the most balanced (csCV 0.53 → 0.60 →
   0.75). Confirms the canonical UMAP→HDBSCAN recipe.

## UMAP-10 vs UMAP-50: 10-d is enough

The two UMAP dimensionalities are quality-indistinguishable: at 6416, sil
0.151 vs 0.142, both ~k=550, csCV 0.75 vs 0.82, noise ≈ 0.20. But UMAP-50
costs nearly **2× the reduction wall time** (11.2 s vs 6.2 s @ 6416) and more
RSS, for no quality gain. The recommendation is specifically **UMAP with
`n_components=10`**, not just "UMAP".

## Repeat stability

Only the **UMAP** runs are stochastic (UMAP's per-seed embedding):
`umap-10@6416 = [549, 537, 563]` (±~2 %), consistent with §4. PCA, Isomap and
identity report a single value in `clusters_per_repeat` — they are
deterministic, so K=3 only re-confirms the same partition. No reduction shows
the collapse-boundary instability §4 saw at `mcs50@250`; the failures here are
systematic (wrong family / wrong dimensionality), not seed-dependent flips.

## Conclusion for downstream experiments

1. **UMAP with `n_components=10` is the right default** for §§3–9 and the
   thesis baseline — it is the only reduction that is non-trivial,
   moderate-noise and balanced across 250 → 6416, and UMAP-50 buys no quality
   for ~2× the cost.
2. **Reduction is mandatory, not optional.** Raw 384-d (`identity`) collapses
   HDBSCAN to k≤4 *and* makes clustering ~70× slower at 6416. This is a genuine
   cost–quality trade-off on this axis (contrast §4, where the knob was free).
3. **Linear (PCA) and graph-based non-linear (Isomap) alternatives both fail**
   — PCA is unstable across n (k=2 with negative silhouette at one end, 62.7 %
   noise at the other), Isomap collapses at full corpus and is the most
   expensive stage measured (23.6 s, 1.2 GB RSS). Neither refutes the UMAP
   default; both reinforce it.
4. **Silhouette/CH/DB invert the ranking here too** — including the macro
   silhouette, which is the strongest single inversion (it ranks the k=2
   degeneracies first). The §2 quality reading must use the joint criterion
   (cluster_count + noise_ratio + cluster_size_cv), exactly the hazard §4
   independently documents on the `min_cluster_size` axis.
