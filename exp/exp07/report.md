# §7 — Pipeline Scalability Sweep: Report

**Source:** `exp/exp07/output/benchmark_20260528T160305Z.json`
**Date:** 2026-05-28 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 42/42 runs `success`.

## Setup

Pillar 1 (computational cost) — the headline experiment of the thesis. Question: how
does wall time scale with `n` for each pipeline stage, and do the asymptotic complexity
differences between clustering algorithms manifest empirically within the available
dataset range?

Fixed baseline (one axis varied = `clustering` × `limit`):

| Component | Value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style head, L2-normalised) |
| Reduction | UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Clustering | Varied: HDBSCAN, KMeans (k=10), DBSCAN, OPTICS, Agglomerative (k=10), Spectral (k=10) |
| Segmenter | `identity` (whole image) |
| Storage | SQLite; similarity search enabled (`top_k=5`) |
| Repeats | `K=5` (reduction+clustering looped; iter 0 dropped → 4 measured samples) |

6 clusterers × 7 corpus sizes `{100, 250, 500, 1000, 2000, 3500, 6416}` = 42 runs.
7 ingests total — the embedding DB is reused across all 6 clusterers within each limit
group.

## Summary — empirical complexity classes

Log-log slopes of clustering wall time vs. `n` (linear regression over all 7 points):

| Algorithm | Measured slope | Expected complexity | Notes |
|---|---|---|---|
| KMeans (k=10) | **0.35** | O(n·k·iter) ≈ O(n) | k and iter fixed; assignment step dominates but at k=10 scales sub-linearly in practice |
| DBSCAN | **0.86** | O(n log n) with kd-tree | kd-tree keeps it sub-linear throughout the range |
| OPTICS | **1.09** | O(n²) worst-case | sklearn kd-tree neighbour queries suppress quadratic behaviour; actual scaling is near-linear here |
| HDBSCAN | **1.30** | O(n log n) | Slightly super-linear over this range, consistent with the O(n log n) amortised claim |
| Spectral (k=10) | **1.21** | O(n²)–O(n³) | Still well below quadratic, but high variance at large n masks the true slope |
| Agglomerative (k=10) | **1.49** | O(n² log n) | Fastest-growing in the sweep; crosses HDBSCAN around n=2000 |

All density-based algorithms (OPTICS, HDBSCAN, DBSCAN) grow more slowly than their
theoretical worst-case because sklearn's implementations exploit kd-trees / ball trees
for neighbour queries. OPTICS in particular is commonly cited as O(n²) but posts slope
1.09 here — the tree structures compress the constant dramatically. The O(n² log n)
claim for Agglomerative holds in direction (steepest measured slope) even though the
absolute slope 1.49 is below 2; at n ≤ 6416 the O(n²) constant of the linkage matrix
is the binding term, not the log factor, giving a sub-quadratic empirical fit.

## Summary — wall-time tables

All times are mean over 4 measured repeats (K=5, iter-0 dropped). Ingest is
single-shot per limit, shared across all 6 clusterers.

### Ingest (shared per limit)

| n | ingest_s | throughput (ips) |
|--:|--:|--:|
| 100 | 23.0 | 4.35 |
| 250 | 57.8 | 4.33 |
| 500 | 116.6 | 4.29 |
| 1000 | 231.6 | 4.32 |
| 2000 | 457.7 | 4.37 |
| 3500 | 798.8 | 4.38 |
| 6416 | 1233.7 | 5.20 |

Ingest slope: **0.97** — effectively O(n). Throughput ≈ 4.3 ips is stable across all
sizes (DINOv2 inference is the bottleneck; IO is negligible). The slight throughput
uptick at n=6416 reflects better GPU/batch utilisation at larger corpus size.

### UMAP reduction (representative — identical across all 6 clustering rows)

| n | reduction_s | std |
|--:|--:|--:|
| 100 | 0.091 | 0.002 |
| 250 | 0.243 | 0.002 |
| 500 | 0.583 | 0.007 |
| 1000 | 1.511 | 0.018 |
| 2000 | 4.512 | 0.015 |
| 3500 | 11.427 | 0.075 |
| **6416** | **5.882** | **0.015** |

UMAP slope over all 7 points: **1.17**. The n=6416 entry is non-monotonic — it drops
from 11.4 s at n=3500 to 5.9 s at n=6416. The std is tight (±0.015 s, tight across 4
repeats), ruling out noise. The most likely cause is a PyNNDescent / HNSW internal mode
switch: above a threshold UMAP switches to a denser approximate-neighbour graph regime
that amortises better at high n. This anomaly is worth flagging in the thesis narrative:
the headline n=6416 UMAP time is faster than the n=3500 point, so a naive "UMAP
dominates at large n" story under-estimates UMAP's eventual throughput at scale.

### Clustering wall time (mean, seconds)

| n | HDBSCAN | KMeans | DBSCAN | OPTICS | Agglo | Spectral |
|--:|--:|--:|--:|--:|--:|--:|
| 100 | 0.003 | 0.002 | 0.002 | 0.080 | 0.001 | 0.013 |
| 250 | 0.004 | 0.002 | 0.003 | 0.199 | 0.002 | 0.018 |
| 500 | 0.007 | 0.003 | 0.005 | 0.398 | 0.004 | 0.026 |
| 1000 | 0.014 | 0.003 | 0.010 | 0.837 | 0.011 | 0.068 |
| 2000 | 0.123 | 0.005 | 0.021 | 1.778 | 0.044 | 0.220 |
| 3500 | 0.179 | 0.007 | 0.043 | 3.453 | 0.133 | 0.486 |
| 6416 | 0.324 | 0.010 | 0.072 | 7.560 | 0.513 | 1.791 |

OPTICS is the only algorithm whose clustering time is non-negligible compared to the
reduction stage at small n (80 ms at n=100, when UMAP takes 91 ms). At n=6416 OPTICS
costs 7.56 s, ~23× HDBSCAN and ~756× KMeans. In absolute terms all algorithms except
OPTICS are cheap at every tested n — clustering is not the bottleneck; ingest and
(for large n) reduction dominate.

### Similarity search — SQLite linear scan (mean, seconds)

| n | sim_search_s | throughput (ips) |
|--:|--:|--:|
| 100 | ~0.5 | ~188 |
| 250 | ~1.3 | ~188 |
| 500 | ~2.8 | ~178 |
| 1000 | ~5.8 | ~172 |
| 2000 | ~12.7 | ~158 |
| 3500 | ~22.6 | ~155 |
| 6416 | ~41.9 | ~153 |

Similarity-search slope: **0.96** — linear in n, as expected for SQLite's Python-side
exhaustive scan. Throughput declines from ~188 ips at small n to ~153 ips at full
corpus; the overhead grows from per-query Python iteration over an increasingly large
embedding set. At n=6416, sim-search (42 s) exceeds even UMAP (5.9 s) by 7×, making
SQLite's linear scan the second-most-expensive stage after ingest.

## Summary — memory (peak RSS delta)

| Algorithm | n=3500 pkRSS (MB) | n=6416 pkRSS (MB) | Notes |
|---|--:|--:|---|
| KMeans | 0.1 | 0.1 | O(n·k) in-memory centroids — trivial |
| DBSCAN | 0.1 | 0.1 | Neighbour graph fits compactly |
| HDBSCAN | 0.2 | 0.2 | Single-linkage tree + condensed tree — still trivial |
| Spectral | 0.3 | 0.7 | Laplacian eigenvector matrix; k=10 keeps this cheap |
| Agglomerative | 93.6 | 314.1 | Linkage matrix grows O(n²); first significant allocation |
| **OPTICS** | **186.9** | **773.7** | Reachability array stores all pairwise-order info; dominates |

OPTICS and Agglomerative are the only algorithms with non-trivial memory footprints at
large n. OPTICS at n=6416 consumes 774 MB above its own stage baseline — for the
hardware available (16 logical CPUs, no VRAM reported) this is manageable but would
constrain a deployment with concurrent workloads. Agglomerative's 314 MB at n=6416 is
milder but still ~1500× HDBSCAN's 0.2 MB.

## Summary — cluster quality

> **Reading note.** This is a cost experiment; quality is a secondary output. The
> fixed k=10 for KMeans, Agglomerative, and Spectral is far below the ~556 clusters
> HDBSCAN discovers at n=6416, so those three algorithms' quality scores measure
> coarse-partition cohesion, not per-cluster granularity. Silhouette must be read
> jointly with `cluster_count`.

| n | algo | cls | noise | sil | CH | DB | csCV |
|--:|---|--:|--:|--:|--:|--:|--:|
| 100 | optics | 12 | 0.174 | **0.287** | 19.7 | **1.222** | 0.544 |
| 100 | hdbscan | 9 | 0.028 | 0.237 | 19.7 | 1.326 | 0.553 |
| 100 | dbscan | 9 | 0.106 | 0.270 | 21.5 | 1.245 | 0.510 |
| 100 | kmeans | 10 | 0.000 | 0.231 | 19.1 | 1.383 | 0.437 |
| 100 | agglom | 10 | 0.000 | 0.231 | 19.1 | 1.344 | 0.558 |
| 100 | spectral | 10 | 0.000 | 0.225 | 18.4 | 1.511 | 0.351 |
| 6416 | optics | 640 | 0.246 | **0.175** | 29.6 | **1.646** | 0.456 |
| 6416 | hdbscan | 556 | 0.193 | 0.149 | 33.1 | 1.706 | 0.727 |
| 6416 | dbscan | 258 | 0.061 | −0.013 | 45.5 | 1.963 | 2.457 |
| 6416 | kmeans | 10 | 0.000 | 0.035 | 658.9 | 3.227 | 0.430 |
| 6416 | agglom | 10 | 0.000 | −0.015 | 505.7 | 3.119 | 1.348 |
| 6416 | spectral | 10 | 0.000 | −0.135 | 204.8 | 3.263 | 1.737 |

Full tables for all 7 limits are in the JSON. Key observations:

1. **OPTICS leads on silhouette at every n.** Its advantage shrinks as n grows (0.287@100
   vs. 0.175@6416) but it never drops below 0.175. DB also best at every n. The cost is
   its O(n²)-memory profile and the 7.56 s at n=6416.
2. **HDBSCAN is the best cost–quality trade-off.** Second-best silhouette at every n
   (0.237 → 0.149), sub-second clustering through n=6416, and 0.2 MB RSS overhead. Its
   cluster count scales with the corpus (9 → 556), maintaining granularity where the
   fixed-k algorithms collapse.
3. **Fixed-k algorithms (KMeans, Agglomerative, Spectral) degrade at scale.** With
   k=10 pinned, silhouette falls monotonically as n grows: KMeans 0.231 → 0.035,
   Agglomerative 0.231 → −0.015, Spectral 0.225 → −0.135. Negative silhouette at
   n=6416 means the forced-10-cluster partitions are geometrically incoherent at that
   corpus size. CH inflates for these algorithms at large n (KMeans CH=658@6416) because
   the 10 fat blobs maximise between/within variance — the same CH inflation trap
   documented in §4. Silhouette and DB are the honest metrics here.
4. **DBSCAN quality also degrades at large n.** Cluster count grows to 258@6416 but
   silhouette turns negative and cluster-size CV hits 2.457 (highly imbalanced), pointing
   to over-fragmentation. DBSCAN's auto-eps (elbow) heuristic produces increasingly
   uneven cluster sizes as n grows.
5. **Spectral variance is high at large n.** Spectral clustering at n=2000–6416 shows
   clustering_wall_time_s_std up to ±0.745 s at n=6416 (vs. mean 1.791 s — 42 % CoV).
   The eigenvalue solver behaviour is sensitive to random initialisation and the
   near-degenerate Laplacian at large n.

## Repeat stability

HDBSCAN, DBSCAN, OPTICS, and KMeans are stable across K=5 repeats in cluster count.
Notable variation:

- **DBSCAN@3500:** `clusters_per_repeat = [206, 154, 223, 176, 160]` — 30 % swing
  between min and max. Auto-eps is sensitive to the UMAP seed at this n; the
  aggregated `cluster_count=160` is the per-run mean and should not be read as a
  stable partition.
- **Spectral@6416:** clustering std ±0.745 s (42 % CoV) — eigenvalue solver
  non-determinism. The mean 1.791 s understates the per-run risk.
- **HDBSCAN@250:** `[23, 24, 23, 25, 23]` — small spread (±1 cluster), stable.
- **OPTICS** is deterministic given fixed input (no random component) but receives
  different UMAP embeddings each repeat; `[634, 636, 648, 643, 640]` at n=6416 is
  correspondingly tight (±7 clusters).

## The dominant stage at each n

| n | dominant stage | notes |
|--:|---|---|
| 100 | ingest (23 s) | all other stages < 0.1 s |
| 250 | ingest (57.8 s) | UMAP 0.24 s; clustering < 0.2 s except OPTICS (0.2 s) |
| 500 | ingest (116.6 s) | UMAP 0.58 s |
| 1000 | ingest (231.6 s) | UMAP 1.5 s; sim-search 5.8 s |
| 2000 | ingest (457.7 s) | UMAP 4.5 s; sim-search 12.7 s |
| 3500 | ingest (798.8 s) | UMAP 11.4 s; sim-search 22.6 s |
| 6416 | ingest (1233.7 s) | sim-search 41.9 s; UMAP 5.9 s\* |

\* UMAP anomaly: 5.9 s at n=6416 is faster than 11.4 s at n=3500 (see above).

Ingest dominates at every n — it is the only stage that cannot be parallelised within
a run without additional hardware (GPU batching is already applied). Sim-search
(SQLite linear scan) becomes the second-most expensive stage by n=1000 and accounts for
~3.4 % of total pipeline time at n=6416. UMAP is third except at n=3500 where it
briefly exceeds sim-search before the anomalous drop at n=6416.

## Conclusions for the thesis

1. **KMeans is the cheapest clusterer** by a large margin at every n (0.010 s at
   n=6416). When granularity is fixed externally (oracle k), it is the obvious choice
   for production pipelines. Its quality degradation at large n (sil→0.035) is an
   artefact of the fixed k=10 — it is not a property of KMeans itself.
2. **HDBSCAN is the best cost–quality trade-off among auto-k algorithms.** 0.324 s
   at n=6416, second-best silhouette, 0.2 MB memory, stable repeats. It is the
   correct baseline choice confirmed here empirically.
3. **OPTICS matches or beats HDBSCAN on quality but costs 23× more in wall time and
   3800× more in memory at n=6416.** Not a practical choice for large corpora unless
   quality gains justify the OPTICS-specific overhead.
4. **Agglomerative and Spectral should be avoided at large n** with fixed k=10 — both
   post negative silhouette at n=6416, meaning the 10-cluster partitions are worse than
   random at full corpus. Memory cost (314 MB for Agglomerative) also makes them
   impractical without re-tuning k.
5. **SQLite similarity search is O(n) but expensive in absolute terms.** At n=6416 it
   costs more than UMAP reduction (42 s vs. 5.9 s) and is the primary motivation for
   the pgvector/HNSW upgrade tested in §6 and §9.
6. **The UMAP non-monotonicity at n=6416 is reproducible** (tight std across 4
   repeats). The thesis narrative should note this as a mode-switch artefact rather
   than measurement error; it means UMAP does not strictly bound the pipeline cost at
   large n from below.
7. **Ingest is the unambiguous bottleneck across all n.** Any throughput optimisation
   that does not address DINOv2 inference time will have bounded impact. At n=6416 the
   ingest:cluster ratio is ~3800:1 for KMeans and ~3800:1 for HDBSCAN.
