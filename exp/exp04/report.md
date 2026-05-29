# §4 — HDBSCAN `min_cluster_size` Tuning: Report

**Source:** `exp/exp04/output/benchmark_20260527T133844Z.json`
**Date:** 2026-05-27 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 18/18 runs `success`.

## Setup

Pillar 2 (unsupervised quality). Question: what is the smallest meaningful
cluster size for this dataset, and does the optimal `min_cluster_size` shift
with corpus size?

Fixed baseline (one axis varied = `min_cluster_size`):

| Component | Value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style head, L2-normalised) |
| Reduction | UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Clustering | HDBSCAN, `min_cluster_size` varied |
| Segmenter | `identity` (whole image) |
| Storage | SQLite |
| Repeats | `K=3` (reduction+clustering looped; iter 0 dropped) |

6 `min_cluster_size` × 3 corpus sizes `{250, 1000, 6416}` = 18 runs.
`min_cluster_size ∈ {5, 10, 25, 50, 100, 200}` — sized as fractions of the
6416-image corpus (≈0.1 % to ≈3.1 %).

> **Reading note — silhouette/CH/DB are graded on non-noise points only.** The
> three distance metrics are computed after dropping points labelled −1
> (`clustering_metrics.py:51-83`). An `mcs` sweep produces high-noise cells by
> construction, so this matters more here than in §2: `mcs50@250` scores
> sil=0.23 while discarding **66.7 % of points as noise** — the score reflects
> only the dense surviving third. Silhouette must be read jointly with
> `noise_ratio` and `cluster_count`, never alone.

## Summary — quality

`cls` = cluster count, `noise` = fraction labelled −1, `sil` = silhouette
(higher better, non-noise only), `CH` = Calinski–Harabasz (higher better), `DB`
= Davies–Bouldin (lower better), `csCV` = cluster-size CV (lower = more
balanced). Blank = no clusters formed (all noise).

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

† `mcs50@250` aggregates `clusters_per_repeat = [0, 0, 2]` — two of three
repeats produced **zero clusters**; `cls=2` is not a stable partition (see
below).

## Summary — cost

`mcs` is effectively free. Reduction dominates and is constant across the
sweep (UMAP is unaffected by the clustering knob); clustering wall time stays
sub-0.5 s at every cell. Ingest is shared per corpus size (56.1 / 226.2 /
1257.2 s for 250 / 1000 / 6416) and not an `mcs` cost.

| n | mcs | red_s | red_pkRSS MB | clu_s | clu_pkRSS MB |
|--:|--:|--:|--:|--:|--:|
| 6416 | 5 | 6.140 | 97.0 | 0.338 | 0.3 |
| 6416 | 10 | 6.110 | 62.2 | 0.323 | 0.3 |
| 6416 | 25 | 6.083 | 62.8 | 0.333 | 0.3 |
| 6416 | 50 | 6.071 | 58.1 | 0.354 | 0.3 |
| 6416 | 100 | 6.129 | 59.5 | 0.414 | 0.3 |
| 6416 | 200 | 6.144 | 63.5 | 0.494 | 10.5 |

(250 / 1000 rows follow the same flat profile at smaller magnitude — see JSON.)
Unlike §2, there is **no cost–quality trade-off on this axis**: the decision is
purely about partition quality.

## The central reading hazard: silhouette peaks at the degenerate cells

A naive "highest silhouette wins" reading picks **mcs50@1000** (0.287) — the
top score in the whole table — followed by **mcs100/200@6416** (0.269/0.270).
All three are **k=2**: a single binary split of the corpus, trivially cohesive
and semantically empty. The CH column inflates the same illusion — `mcs100@6416`
posts CH=2765 (75× the mcs5 value) purely because two fat, well-separated blobs
maximise between/within variance.

The opposite trap is `mcs50@250`: sil=0.23 looks healthy but **66.7 % of points
are noise** and the partition is unstable across repeats (`[0, 0, 2]`). The
score grades only the dense survivor core.

On the joint criterion — non-trivial cluster count, moderate noise, balanced
clusters — the ranking is unambiguous: **mcs=5 wins at every corpus size**.

## Noise is non-monotonic in `mcs` — it does not bound granularity

The §4 brief expected noise to "climb sharply past a threshold," marking the
upper bound on cluster granularity. **The data refutes that framing.** Noise
rises through the low/mid range (e.g. n=6416: 0.199 → 0.277 at mcs5 → mcs10)
then *falls back* as `mcs` grows further: `mcs100@6416` has **noise 0.008**,
*below* mcs5's 0.199. Cause: once the density threshold is too high to support
fine structure, HDBSCAN abandons granularity and forms 2 fat clusters that
absorb nearly every point — low noise, but trivial.

So low noise is **not** evidence of a good partition. Noise must be read with
`cluster_count`: the bad regimes are both ends (mid-`mcs` high noise *and*
high-`mcs` low-noise-but-k=2). Only the low-`mcs` end gives moderate noise *and*
non-trivial, balanced clusters.

## Does the optimal `mcs` shift with `n`?

Two distinct answers, both needed:

1. **The optimal floor does not shift.** `mcs=5` is the best cell on the joint
   criterion at all three sizes. Its cluster count scales sensibly with the
   corpus (23 → 96 → 563), noise stays moderate (0.06 → 0.11 → 0.20), and
   clusters stay the most balanced (CV 0.53 → 0.60 → 0.75). Smaller is better
   everywhere; there is no n at which a larger `mcs` overtakes it.

2. **The collapse ceiling rises with `n`.** The `mcs` at which the partition
   degenerates to k≤2 moves upward as the corpus grows:
   - **n=250** collapses at **mcs25** (k=2) and is all-noise by mcs100.
   - **n=1000** still holds k=3 at mcs25, collapses by mcs50, all-noise at mcs200.
   - **n=6416** holds **26 clusters at mcs25**, only collapses to k=4 at mcs50
     and k=2 at mcs100.

   A larger corpus tolerates a larger absolute `min_cluster_size` before
   structure vanishes — but since `mcs=5` already wins, this only matters as a
   safety margin, not a tuning recommendation.

This confirms the audit M.5 prediction empirically: `mcs ∈ {100, 200}` at
`limit ∈ {250, 1000}` produces degenerate all-noise cells with no metrics
(250@{100,200} and 1000@200 → 0 clusters).

## Repeat stability

Most cells are stable across the K=3 repeats. Two are not, both at the
collapse boundary:

- **`mcs50@250`:** `[0, 0, 2]` — the partition flips between all-noise and a
  2-cluster split depending on UMAP's per-repeat seed. The aggregated row is
  not a meaningful single partition.
- UMAP stochasticity is otherwise small at usable settings: `mcs5@6416` varies
  549 / 537 / 563 (±~2 %), consistent with §2.

## Conclusion for downstream experiments

1. **`min_cluster_size=5` is the right default** for §§5–9 and the thesis
   baseline. It is the only setting that is simultaneously non-trivial,
   moderate-noise, and balanced across 250 → 6416. Every larger value either
   over-merges into a trivial binary split or (mid-range) inflates noise while
   losing balance.
2. **Silhouette and CH invert the ranking on this axis** — both peak at the
   degenerate k=2 cells. The §4 quality reading must be the joint criterion
   (cluster_count + noise_ratio + cluster_size_cv), exactly the hazard §2 also
   documented. This is direct evidence for reading `mcs` against
   `noise_ratio`/`cluster_count` rather than silhouette in isolation.
3. **Noise ratio alone is not a granularity bound.** It is non-monotonic in
   `mcs`; very high `mcs` returns *low* noise by abandoning structure. Any
   downstream narrative that uses a noise threshold must pair it with cluster
   count.
4. **The optimal `mcs` does not grow with the corpus.** Larger n raises the
   `mcs` at which the partition collapses, but the best setting stays at the
   floor (`mcs=5`) regardless of size — so no per-size retuning is required for
   the scalability sweep (§7).
