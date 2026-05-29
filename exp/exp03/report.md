# §3 — Clustering Algorithm Comparison: Report

**Source:** `exp/exp03/output/benchmark_20260527T155755Z.json`
**Date:** 2026-05-27 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 42/42 runs `success`.

## Setup

Pillar 2 (unsupervised quality). Question: with the embedding and reduction
stages fixed at the §2 winner, which clustering algorithm partitions the corpus
best — and which stay well-behaved as n grows?

Fixed baseline (one axis varied = clustering):

| Component | Value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style head, L2-normalised) |
| Reduction | UMAP-10 (cosine, n_neighbors=15, min_dist=0) — the §2 winner |
| Segmenter | `identity` (whole image) |
| Storage | SQLite |
| Repeats | `K=3` (reduction+clustering looped; iter 0 dropped → `n=2`) |

14 clustering configs × 3 corpus sizes `{250, 1000, 6416}` = 42 runs. Configs:
`hdbscan` (min_cluster_size=5), `kmeans` (elbow, max_k=30), `gmm` diag (BIC,
max_k ∈ {30, 100, 300}), `dbscan` (min_samples=5; auto-eps + eps ∈ {0.2, 0.5,
1.0}), `optics` (min_samples=5, cosine), `agglomerative` average linkage (k ∈
{10, 20}), `spectral` nearest-neighbors (k ∈ {10, 20}).

> **Reading note — intrinsic metrics are scored on the confident core, not the
> whole corpus.** Silhouette / Calinski–Harabasz / Davies–Bouldin are computed
> only on **non-noise** points (`labels != -1`). So the density methods
> (`hdbscan`, `dbscan`, `optics`) are graded on the subset they *chose* to
> cluster, while the partitional methods (`kmeans`, `gmm`, `agglomerative`,
> `spectral`) are graded on **all** points. A method that dumps hard cases into
> noise gets an unfair silhouette advantage. The metrics are comparable *only*
> read jointly with `noise_ratio` (= coverage). Two further caveats:
> - **DBSCAN auto-eps is not k-stable.** With no `eps`, eps is re-derived each
>   repeat (UMAP is reseeded per repeat), so `clusters_per_repeat` spans
>   different partitions — e.g. `[247, 219, 239]` at n=6416. Its aggregated
>   metrics describe a *mix*, not one partition family.
> - **Agglomerative/Spectral take a fixed k** (10/20); everyone else auto-detects.
>   They look coarse partly because they cannot adapt granularity to n.

## Summary — quality

`cls` = cluster count, `noise` = fraction labelled −1, `sil` = silhouette
(higher better, scored on non-noise only), `CH` = Calinski–Harabasz (higher
better), `DB` = Davies–Bouldin (lower better), `csCV` = cluster-size CV (lower =
more balanced).

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

## Summary — cost

`clu_s` clustering wall time (mean of K−1), `clu_pkRSS` stage-net peak RSS.
Ingest (57.6 / 226.8 / 1272.5 s for 250 / 1000 / 6416) and UMAP-10 reduction
(\~0.25 / \~1.0 / \~6.4 s) are **shared across all 14 configs** and are not a
clustering cost.

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

(250 / 1000 rows follow the same ordering at smaller magnitude — see JSON.)

## The central reading hazard: silhouette is coverage-confounded

Ranking the n=6416 table by raw silhouette puts **optics** (0.177) and
**hdbscan** (0.151) on top — but those scores are computed on **75.3 %** and
**80.1 %** of the corpus respectively (the rest is noise), while **gmm·300**
(0.054) is scored on **100 %**. This is the §2 isomap mirage in a new guise:
the metric rewards a method for *abstaining* on hard points. A density method
and a partitional method are not graded on the same support.

The mirror trap is the trivial-k collapse. **kmeans** posts the highest CH at
every size (805 at n=6416) — but it auto-detects only **k=8** via the elbow,
i.e. an 8-way split of a 6416-image corpus. High CH on few fat clusters means
"compact blobs", not "captures graffiti structure". Same for **dbscan·1.0**
(k=9) and **agglo·10**. CH and silhouette in isolation both lie here.

The metrics are interpretable only **jointly**: a usable partition needs a
non-trivial-but-not-exploded cluster count, *moderate* noise (not 47 % like
dbscan·0.2@250, not 0 % from a forced k), and balanced clusters (low CV). On
that joint criterion the ranking is stable and the per-method story below is
unambiguous.

## Per-method

**HDBSCAN (project default).** The only method that is *both* well-behaved on
the joint criterion *and* cheap at every size. Cluster count scales sensibly
with n (23 → 96 → 563), noise stays moderate (0.06 → 0.11 → 0.20), silhouette
is stable (\~0.15–0.20), CV is low (0.53 → 0.60 → 0.75). Cost is negligible:
**0.348 s and ~0.3 MB** peak at n=6416. Stochastic only through UMAP —
`clusters_per_repeat` 549/537/563 at n=6416 (±\~2–5 %).

**OPTICS.** The quality peer of HDBSCAN — it edges it on raw silhouette at every
size (0.237/0.184/0.177) and has the *best* balance in the whole table (CV
0.45–0.48). But two asterisks: (1) it abstains *more* (noise 0.17 → 0.18 →
0.25), so its silhouette lead partly buys coverage; (2) it is **dominated on
cost** — **8.578 s and 763 MB** peak at n=6416, i.e. **~25× the time and
~2500× the memory** of HDBSCAN for a marginal, coverage-confounded quality
gain. Cosine OPTICS materialises a large reachability structure; it does not
scale.

**GMM (diagonal, BIC).** The best **noise-free** option — it scores all points
and still stays balanced (CV ~0.5–0.58). The `max_clusters` cap is load-bearing:
`max30` saturates (29/30/29), `max100` saturates at n≥1000, and only `max300`
lets BIC pick freely (147 @ 1000, 297 @ 6416). At its fair point (gmm·300) it
trails HDBSCAN on the coverage-confounded silhouette but is the only auto-k
partitional method that does not collapse. Cost grows with k: 1.839 s / 49.7 MB
at n=6416.

**KMeans.** Cheapest of all (0.009 s) but the elbow pins **k≈7–8 at every
size** — it cannot express the corpus's natural granularity. High CH, trivial
partition. Unusable as the clustering stage; useful only as a CH sanity anchor.

**DBSCAN — no usable operating point on this data.** Every setting fails some
joint axis: `auto`/`0.2` over-fragment with **CV > 2.5** and negative
silhouette at scale; `0.5` collapses balance (CV 3.08); `1.0` collapses to k≈9.
On top, `auto`-eps is **not k-stable across repeats** (247/219/239 @ 6416), so
its aggregates mix partitions. HDBSCAN dominates it on every axis — there is no
reason to prefer flat-eps DBSCAN here.

**Agglomerative (average linkage).** Fixed k=10/20 is too coarse, and average
linkage chains: CV climbs to 1.36 (k=10 @ 6416) and silhouette goes negative.
Also needs the **full pairwise distance matrix — 314 MB** peak at n=6416.
Dominated on quality and memory.

**Spectral (nearest-neighbors).** Worst quality at scale — **negative
silhouette** at both n=1000 and n=6416 (down to −0.117), high CV. Fixed small k
plus an affinity graph that does not match the density structure UMAP produces.
Dominated.

## Cost–quality Pareto (@ n=6416)

| Method | Cost | Quality (joint) | Verdict |
|---|---|---|---|
| **HDBSCAN** | **low** (0.35 s, 0.3 MB) | **best + stable** (k scales, moderate noise, balanced) | **winner** |
| OPTICS | **worst** (8.6 s, 763 MB) | ≈ HDBSCAN, more noise | quality peer, dominated on cost |
| GMM·300 | modest (1.8 s, 50 MB) | best **noise-free** option | runner-up if −1 unacceptable |
| KMeans | cheapest | trivial (k=8) | collapses |
| DBSCAN | cheap | no usable eps (CV>2.5 or k≈9) | dominated by HDBSCAN |
| Agglomerative | high mem (314 MB) | coarse, chains (CV 1.36) | dominated |
| Spectral | mid (1.7–2.1 s) | negative silhouette | dominated |

HDBSCAN is the only point that is simultaneously best-in-class on the joint
quality criterion *and* cheap. OPTICS matches its quality but at ~25× time and
~2500× memory; the cheap partitional methods (KMeans) collapse to trivial k; the
flat clusterers (DBSCAN/Agglo/Spectral) go unbalanced or negative at scale.

## Conclusion for downstream experiments

1. **HDBSCAN (`min_cluster_size=5`) is the right clustering stage** for §§4–7,
   paired with the §2 UMAP-10 reduction. It is the only algorithm that stays
   stable, balanced, and cheap across 250 → 6416. This confirms the project's
   default empirically against 13 alternatives.
2. **OPTICS is its only quality rival but is dominated on cost** (8.6 s,
   763 MB @ 6416 vs 0.35 s, 0.3 MB). No reason to swap unless a future use needs
   OPTICS's slightly tighter cores and can pay O(n²)-like memory.
3. **GMM·300 is the fallback when noise labels (−1) are unacceptable
   downstream** — it is the only auto-k partitional method that does not
   collapse. KMeans, flat DBSCAN, Agglomerative, and Spectral are all dominated
   and need no further investigation.
4. **Intrinsic scores must be read against coverage.** The silhouette ranking is
   confounded by the non-noise-only scoring (density methods graded on a chosen
   subset) and by trivial-k inflation of CH — this is the reason §4's
   `min_cluster_size` tuning must read silhouette jointly with `noise_ratio` and
   `cluster_count`, never in isolation.
