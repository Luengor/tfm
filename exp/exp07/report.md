# §7 — Pipeline Scalability Sweep (HEADLINE): Report

**Source:** `exp/exp07/output/benchmark_20260529T225953Z.json`
**Date:** 2026-05-29 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 30/30 runs `success`.

## Setup

Pillar 1 (computational cost) — the **headline** deliverable of the thesis.
Question: how does wall time scale with corpus size `n` for each pipeline
stage on the available graffiti corpus (n ≤ 6416)?

Per the thesis brief, this is a cost-vs-corpus-size *characterisation*, not
an asymptotic-complexity proof: ~1.8 log decades is too narrow to separate
O(n log n) from O(n²) cleanly. Slopes are reported as **observed** in this
regime; theoretical references are quoted as context, not headline claims.

Fixed baseline (varied axis = `limit` × `clustering`):

| Component | Value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style head, L2-normalised) |
| Reduction | UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Segmenter | `identity` (whole image) |
| Storage | SQLite |
| Repeats | `K=5` (reduction+clustering looped; iter 0 dropped for wall/CPU, all K kept for RSS/VRAM) |

5 clusterers × 6 corpus sizes `{250, 500, 1000, 2000, 3500, 6416}` = 30 runs.
6 ingests (one shared SQLite DB per limit, reused across the 5 clusterers).

> **Reading note — similarity search timed only on HDBSCAN.** To keep the
> sweep tractable, `similarity_search.enabled` was set only on the 6 HDBSCAN
> rows (which run first per limit); the 24 KMeans/DBSCAN/OPTICS/Agglomerative
> rows skip it. Sim-search cost is storage-invariant under fixed embedding
> + corpus, so reading it off the HDBSCAN rows alone is sound — the
> clustering algorithm does not change the SQLite all-pairs top-k cost.
> All `ss_s = 0` entries below mean "not timed in this run," not "instant."

## Summary — cost (HEADLINE)

`red_s` = UMAP reduction wall time, `clu_s` = clustering wall time, `ss_s` =
similarity-search wall time (HDBSCAN only). All values mean over K−1 samples
(iter 0 dropped). Ingest is shared per `n` and listed once.

| n | ingest_s | red_s | clu_s hdbscan | clu_s kmeans | clu_s dbscan | clu_s optics | clu_s agglo | ss_s (hdb) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 58.7 | 0.253 | 0.0038 | 0.0025 | 0.0035 | 0.208 | 0.0019 | 3.66 |
| 500 | 115.7 | 0.603 | 0.0069 | 0.0027 | 0.0055 | 0.347 | 0.0046 | 16.32 |
| 1000 | 231.6 | 1.514 | 0.0144 | 0.0033 | 0.0103 | 0.631 | 0.0117 | 63.28 |
| 2000 | 455.9 | 4.542 | 0.1106 | 0.0047 | 0.0228 | 1.305 | 0.0427 | 306.81 |
| 3500 | 776.0 | 11.870 | 0.1863 | 0.0066 | 0.0471 | 2.282 | 0.1405 | 995.52 |
| 6416 | 1294.9 | **6.237** | 0.3580 | 0.0098 | 0.0768 | 4.322 | 0.5988 | **3266.60** |

The two largest cells at n=6416 dominate every other entry: ingest (≈21.6 min)
and HDBSCAN similarity search (≈54.4 min). Everything else stays sub-13 s.

**Observed slopes** (least-squares log–log fit) on two regimes — the full
range and the n=500…3500 window that excludes the n=6416 UMAP anomaly
documented below:

| Stage | Algorithm | Theoretical scaling | Observed slope (250…6416) | Observed slope (500…3500) |
|---|---|--:|--:|--:|
| Ingest | DINOv2 inference | O(n) | 0.95 | — |
| Reduction | UMAP | ~O(n^1.14) | 1.15 | 1.53 |
| Clustering | KMeans (k=10 fixed) | ≈O(n) | 0.43 | 0.46 |
| Clustering | DBSCAN | O(n log n)–O(n²) | 0.99 | 1.10 |
| Clustering | OPTICS | O(n log n)–O(n²) | 0.94 | 0.98 |
| Clustering | HDBSCAN | O(n log n) amortised | 1.53 | 1.84 |
| Clustering | Agglomerative | O(n² log n) | 1.77 | 1.76 |
| Similarity search | SQLite linear scan (all-pairs top-k) | O(n²) | **2.10** | 2.13 |

The 500…3500 column is the cleaner read because the n=6416 reduction value
breaks the trend (see *UMAP anomaly* section). Inside the safe window, the
ranking matches theory: KMeans (sub-linear, fixed k) < DBSCAN ≈ OPTICS
(near-linear) < Agglomerative ≈ HDBSCAN (super-linear), with similarity
search firmly in the O(n²) band.

## The headline figure: similarity search is the bottleneck at corpus size

At every n the SQLite all-pairs top-k scan dominates *all* downstream work.
Per-query throughput collapses from **68 ips at n=250 to 2 ips at n=6416**
— a 35× drop over a 26× `n` span, the empirical signature of the per-query
O(n) inner scan. Wall-time per stage at n=6416:

| Stage | Wall | Share of post-ingest cost |
|---|--:|--:|
| Similarity search (HDBSCAN row) | 3266.6 s | **99.6 %** |
| Reduction (UMAP) | 6.2 s | 0.2 % |
| Clustering (HDBSCAN) | 0.4 s | <0.1 % |

Ingest is even larger in absolute terms (1294.9 s = 21.6 min) but is a
one-shot per corpus and is dominated by GPU embedding inference (CPU=6 %,
VRAM peak ≈100 MB, RSS peak ≈283 MB) — its cost is independent of every
downstream choice. The similarity-search cost, by contrast, is paid on
every full-corpus query pass under the SQLite backend.

This is the empirical hook §6 (storage backends) and §9 (HNSW tuning) hang
on: 54 min for one full-corpus top-5 sweep at n=6416 on SQLite is what
those experiments are measured *against*. The slope 2.10 confirms the
linear-scan O(n²) cost class — projection to a 50 k-image corpus
(≈8× from here) extrapolates to ~58× wall time (~52 hours per pass) if the
backend is unchanged.

## Ingest scales linearly and is GPU-bound

Ingest IPS is essentially flat across the sweep (4.26 → 4.95 ips, slope
0.95) — the small upward drift is fixed-overhead amortisation, not
super-linearity. CPU utilisation hangs at 6 % at every n (matches a
single-process GPU pipeline waiting on inference); peak RSS holds at
~282 MB regardless of n (the DINOv2 model itself is the floor; the
per-image delta is negligible). VRAM peak is ~100 MB.

Ingest is therefore linear in n with a per-image constant of ≈0.20 s. This
is the cleanest scaling result of the sweep — and the largest single
contributor to total wall time at every n except n=6416, where sim-search
overtakes it.

## Reduction: UMAP slope is regime-dependent (the n=6416 anomaly)

UMAP wall time rises monotonically through 250→3500 (0.25 → 11.87 s,
local slope ~1.5) and then **drops sharply to 6.24 s at n=6416** — *47 %
faster than at n=3500 despite an 83 % larger input*. The break is
reproducible (std 0.017 s on 4 samples, 0.3 %) so it is not noise.

Cause: UMAP switches to its pynndescent-based approximate nearest-neighbour
backend once `n` exceeds the threshold for exact kNN (the default crossover
sits around a few thousand points in this UMAP build). Below that
threshold UMAP runs an exact graph construction whose per-call cost grows
super-linearly in `n`; above it, pynndescent's near-linear approximate kNN
takes over and the absolute wall time drops.

Consequence: the **full-range reduction slope (1.15) is not a single
scaling regime — it is a mean over two regimes** stitched at n≈4096. The
500–3500 slope (1.53) describes the exact-kNN regime; n=6416 is the first
point of the approximate regime. Slope claims must specify which window;
the report uses both columns above to make the discontinuity explicit.

This is itself a useful operational finding: scaling past ~4 k images is
*cheaper* per call than the small-n trend predicts, because UMAP changes
algorithm under the hood.

## Clustering: theoretical hierarchy reproduces on this corpus

Inside the n=500…3500 window the observed slopes line up with the
theoretical reference column (above). The agglomerative slope ≈1.76 is
the closest in the sweep to its theoretical O(n² log n); HDBSCAN's 1.84 is
above the amortised O(n log n) prediction, which is unsurprising at this
`n` — the log factor is not visible in 1.8 log decades. KMeans's 0.43
reflects fixed k=10 and few iterations: total cost is dominated by
distance computations whose count is `n · k · iter` with `k · iter`
≈ constant; in this regime the small-`n` rows are floored by Python
overhead.

Absolute magnitudes matter more than slopes for downstream choices:

- At n=6416, OPTICS (4.32 s) > Agglomerative (0.60 s) > HDBSCAN (0.36 s) >
  DBSCAN (0.08 s) > KMeans (0.010 s). OPTICS is ~12× HDBSCAN, ~440× KMeans.
- Every clusterer finishes under 5 s at full corpus — clustering is **not**
  a wall-time bottleneck anywhere in this sweep.
- Memory tells a different story: Agglomerative's `clustering_peak_rss_delta`
  climbs 0.1 → 3.2 → 93.5 → 314 MB across n ∈ {1000, 2000, 3500, 6416} —
  the O(n²) condensed distance matrix made explicit. HDBSCAN/DBSCAN/OPTICS
  stay sub-1 MB. If the corpus grew 10×, Agglomerative would hit ~30 GB
  RAM and Agglomerative becomes infeasible long before its wall time does.

## Repeat stability

The K=5 repeat protocol is delivering tight error bars at n=6416:

| Algorithm | red rel-std | clu rel-std |
|---|--:|--:|
| HDBSCAN | 0.3 % | 0.6 % |
| OPTICS | 0.4 % | 0.4 % |
| KMeans | 0.4 % | 5.3 % |
| DBSCAN | 0.6 % | 2.7 % |
| Agglomerative | 0.6 % | 3.8 % |

All slope fits are far above the per-cell noise floor — the slope estimates
are signal-bound, not noise-bound. HDBSCAN's per-repeat cluster count at
n=6416 is `[549, 537, 563, 548, 556]` (±~2 %), consistent with §4's UMAP
stochasticity floor on this baseline.

## Summary — quality (side-effect)

§7 is not a quality experiment, but the per-cell quality is recorded and
worth quoting briefly. At n=6416 with the baseline embedding + UMAP:

| Algorithm | k | sil | CH | DB | noise | csCV |
|---|--:|--:|--:|--:|--:|--:|
| HDBSCAN (mcs=5) | 556 | 0.149 | 33 | 1.71 | 0.193 | 0.73 |
| OPTICS (ms=5) | 627 | **0.184** | 30 | **1.60** | 0.278 | **0.44** |
| DBSCAN (ms=5) | 258 | −0.013 | 46 | 1.96 | 0.061 | 2.46 |
| KMeans (k=10) | 10 | 0.035 | 659 | 3.23 | 0.000 | 0.43 |
| Agglomerative (k=10, avg) | 10 | −0.015 | 506 | 3.12 | 0.000 | 1.35 |

The density-based row dominates the joint criterion (non-trivial k,
moderate noise, balanced clusters) — consistent with §3's clusterer
comparison. The k=10 partitional rows post low silhouette because the
true cluster count is far higher than 10 (HDBSCAN/OPTICS auto-detect
500–600 clusters at this `n`); CH and DB invert for the same degenerate
reason §4 documented at high `min_cluster_size`. None of this changes the
cost reading, which is the §7 deliverable.

## Conclusion for downstream experiments

1. **Similarity-search slope ≈ 2.10 is the headline cost result.** SQLite
   all-pairs top-k is O(n²) in practice on this corpus, and at n=6416 it
   already costs 54 min per pass — more than ingest itself, and 5000× the
   clustering stage. This is the cost-side anchor §6 and §9 measure against;
   the thesis chapter should lead with this curve.
2. **Ingest is linear and is the second wall-time bottleneck** (21.6 min
   at n=6416). It is paid once per corpus per embedding, so it shapes the
   amortisation argument (one-shot ingest → many queries) rather than
   per-query latency. The cost is GPU-bound (CPU 6 %), so optimisations
   should target batch size / model precision, not Python overhead.
3. **The UMAP reduction slope changes regime at n≈4 k.** Quoting a single
   exponent (1.15 full-range vs 1.53 below 4 k) is misleading; the report
   should plot the curve and call out the pynndescent crossover. Past
   n=6416 UMAP is empirically cheaper than its small-n trend predicts —
   good news for resampling-based extrapolation but a caveat for any
   straight-line projection of this sweep.
4. **Clustering wall time is not a bottleneck on this corpus.** All five
   clusterers finish under 5 s at n=6416. The constraint that matters at
   scale is **memory**, not wall: Agglomerative's 314 MB peak at 6416 grows
   quadratically in n, so the algorithm is wall-time-cheap but
   memory-feasibility-bounded. HDBSCAN/DBSCAN/OPTICS stay sub-1 MB and are
   the only candidates that survive a 10× corpus growth in-memory.
5. **The observed clustering ranking matches theory inside 500…3500** and
   keeps the same order at 6416, modulo the UMAP-regime caveat. Slopes
   should be reported as *observed on this corpus*, not as asymptotic
   exponents — the n range is too narrow for either claim.
6. **Asymptotic-complexity claims are deferred.** The brief explicitly
   frames this as cost-vs-corpus-size on the available data, not as a
   complexity-class study. The slope columns above are reported in that
   spirit, and the resampling-extrapolation idea in `doc/conclusion.tex`
   is the right place to argue beyond n=6416.
