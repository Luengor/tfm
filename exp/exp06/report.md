# §6 — Storage Backend Sweep: Report

**Source:** `exp/exp06/output/benchmark_20260529T172522Z.json`
**Date:** 2026-05-29 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 12/12 runs `success`.

## Setup

Pillar 1 (computational cost) on the storage / similarity-search axis. Question:
how do the three storage backends compare on similarity-search latency and
throughput as the corpus grows, and at what corpus size does the SQLite
Python-side linear scan stop being viable? Recall is held at the spec by the
HNSW row.

Fixed baseline (one axis varied = `storage` × `limit`):

| Component | Value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style head, L2-normalised) |
| Reduction | UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Clustering | HDBSCAN, `min_cluster_size=5` |
| Segmenter | `identity` (whole image) |
| Similarity search | enabled, `top_k=5`, `sample_n=100`, `sample_seed=42` |
| Storage | **varied** |
| Repeats | `K=5` (reduction+clustering+search looped; iter 0 dropped) |

3 backends × 4 corpus sizes `{500, 1000, 2500, 6416}` = 12 runs. Backends:

- **`sqlite`** — Python-side exhaustive scan over rows (`SQLiteStorage.get_by_distance`).
- **`postgresql`** — exact pgvector ordered scan inside the DB (`vector_cosine_ops`,
  no index → sequential scan with C-level cosine).
- **`postgresql_hnsw`** — pgvector with an HNSW index (`ops=cosine, m=16,
  ef_construction=64`). The index is built once between ingest and the timed
  similarity-search loop so its build cost is not in `similarity_search_wall_time_s`.

Each storage row uses its own database (separate `db_path` / `db_url`), so each
of the 12 runs ingests independently and the clustering side is bit-for-bit
identical across storages at a given `n` (deterministic seed). The
similarity-search call is the only operationally distinguishing stage.

## Summary — similarity-search throughput

`ss_s` = mean wall time per `top_k=5` neighbour query over 100 sampled queries,
mean of K=4 measured repeats; `ips` = queries/s; `idx_s` = HNSW build wall time
(charged once before the timed loop); `recall@5` = HNSW recall vs the exact
brute-force kNN over the same query set.

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

(The `idx_s` column reads `0.0` for plain `postgresql` because no index is
built — the value is the runner's no-op timing of the absent build step.)

## Summary — clustering quality (sanity check)

Each backend at a given `n` clusters the same embeddings with the same UMAP
seed, so all quality columns are identical row-for-row. Reported once per `n`
to verify nothing drifted:

| n | k | silhouette | silM | CH | DB | noise | csCV |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 500 | 43 | 0.191 | 0.244 | 20.6 | 1.57 | 0.110 | 0.53 |
| 1000 | 96 | 0.160 | 0.210 | 18.9 | 1.67 | 0.106 | 0.60 |
| 2500 | 252 | 0.160 | 0.213 | 20.0 | 1.67 | 0.161 | 0.61 |
| 6416 | 556 | 0.149 | 0.206 | 33.1 | 1.71 | 0.193 | 0.73 |

These match the §7 HDBSCAN row at the same `n` (e.g. n=6416: sil 0.149, k=556,
noise 0.193 — same DB / pipeline). Storage choice does not affect clustering
output, as expected.

## Summary — ingest

Ingest is dominated by DINOv2 inference at ≈4.3–5.2 ips on this host. The
storage column adds essentially nothing visible at this scale:

| n | sqlite | postgresql | postgresql_hnsw |
|--:|--:|--:|--:|
| 500 | 106.6 s | 111.8 s | 114.3 s |
| 1000 | 219.4 s | 220.0 s | 231.4 s |
| 2500 | 573.4 s | 558.6 s | 581.3 s |
| 6416 | 1231.9 s | 1269.9 s | 1334.6 s |

PostgreSQL adds 0–7 % vs SQLite at every size (network/SQLAlchemy overhead per
row); PostgreSQL + HNSW adds another 0–6 % (per-row index insert). At thesis
scale this is irrelevant — every backend's ingest cost is bounded by the GPU,
and the HNSW build itself is sub-second even at n=6416.

## Throughput vs `n`: who scales

Plotted as queries-per-second per backend:

- **`sqlite` throughput is flat at ≈150 ips, but its wall time is linear in `n`.**
  Per-query Python iteration over the embedding set: at n=500 it's already
  3 s/query×set ÷ 100 queries = 30 ms/query (164 ips); at n=6416, 41.9 s ÷ 100 ≈
  431 ms/query (149 ips). The slight throughput drop comes from Python
  overhead and growing array materialisation. **Slope of `ss_s` over `n` ≈
  1.00** — pure O(n).
- **`postgresql` exact throughput rises with `n`** (3 096 → 11 109 ips). The
  per-query latency stays under 1 ms until n=2500 and is 5.8 ms at n=6416. The
  amortised C-level cosine kernel inside pgvector dominates the per-query
  round-trip until `n` is large enough that the kernel itself becomes
  expensive. **`ss_s` slope ≈ 0.5** — sub-linear because round-trip overhead
  (constant) dwarfs the cosine pass until ~n=1000.
- **`postgresql_hnsw` throughput is the only one that actually accelerates
  with corpus size relative to exact** (3 042 → 35 936 ips). At n=500 it is a
  wash with exact pgvector (round-trip dominates); at n=6416 it is **6.5×
  faster** than exact, **241× faster** than SQLite. Per-query latency is
  ≈1.8 ms — effectively flat across the `n` range. The HNSW index makes search
  near-`O(log n)` over this range.

## Recall and the HNSW build cost

HNSW with `m=16, ef_construction=64` gives **recall@5 = 1.000 at every n**.
That is the ceiling of the accuracy axis — the speed-vs-accuracy trade-off is
trivially favourable at this corpus size, and §9 explores the trade-off when
the parameters are made more aggressive.

Build cost:

| n | hnsw_index_build_s |
|--:|--:|
| 500 | 0.066 |
| 1000 | 0.136 |
| 2500 | 0.372 |
| 6416 | 0.465 |

Sub-linear in `n` (slope ≈ 0.7 over this range). At n=6416 the build (0.47 s)
is amortised in **≈1.2 queries** vs SQLite (each saves ~0.43 s) and **≈260
queries** vs exact pgvector (each saves ~1.8 ms above ~1.7 ms). The build is
free in any realistic search workload.

## The point of no return for SQLite

At n=6416 a single 100-query batch costs:

- `sqlite`: 43.1 s
- `postgresql` exact: 0.58 s (**74× faster**)
- `postgresql_hnsw`: 0.18 s (**240× faster**)

At n=2500, SQLite is already 100× slower than HNSW (16.9 s vs 0.17 s) and
slower than exact pgvector by 72× (16.9 s vs 0.23 s). The thesis's stated
target corpus is 6416 images, and the scalability axis points to growth beyond
that. **SQLite linear scan is operationally dead beyond n≈1000**; the
PostgreSQL backend is the necessary default for the search half of the
pipeline.

The choice between **exact pgvector** and **HNSW** is more interesting:

- Below n≈2000, exact and HNSW are indistinguishable on latency (round-trip
  dominates). HNSW adds a small build cost and complexity for no gain.
- At n=6416, HNSW is 3.2× faster than exact and matches recall exactly. HNSW
  becomes the right default.
- The crossover sits around n=2500 on this hardware. The thesis's full corpus
  sits past it.

## Repeat stability

Similarity-search wall time is tight across K=4 repeats (std mostly under 10 %
of the mean — see the JSON `similarity_search_wall_time_s_std` column). Recall
is exactly 1.000 in every repeat at every n (no run-to-run variation from
HNSW's stochasticity at these parameters). Quality columns are bit-for-bit
identical across storage rows at each `n` because the same RNG seed feeds
UMAP and HDBSCAN.

## Conclusions for the thesis

1. **SQLite is a development-only backend.** At thesis scale (n=6416) a
   single 100-query batch takes 43 s; the pipeline is unusable for any
   workload that touches the similarity-search stage. Keep it only as a
   zero-dependency option for very small corpora.
2. **PostgreSQL + pgvector is the correct production default.** Even without
   an index it gives 100× SQLite's throughput because the cosine kernel
   runs C-side inside the DB instead of Python-side over a row iterator.
3. **HNSW pays off above n≈2000.** Below that the round-trip dominates and
   index gains are invisible; above it, query latency stays flat (~1.8 ms)
   while exact pgvector grows linearly with `n`. At n=6416 HNSW is 3.2×
   faster than exact pgvector at recall@5 = 1.000 — the speed–accuracy
   trade-off is free at these settings, motivating the parameter sweep in §9.
4. **Build cost is negligible.** 0.47 s at n=6416 is amortised in a single
   query vs SQLite and in a few hundred queries vs exact pgvector. Index
   re-build on ingest is not a constraint.
5. **Storage choice does not affect clustering output.** All 12 runs at the
   same `n` post the same silhouette, cluster count, noise ratio etc. — the
   axis is purely a search-cost axis, as the experiment design intended.
