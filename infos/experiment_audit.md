<!-- LTeX: language=en-US -->

# Experiment Design Audit

Audit of the experiment catalogue (`infos/experiments.md` §1–§9), the configs in
`infos/configs/`, the completed runs/reports in `exp/`, the component model in
`CLAUDE.md`, the dataset counts in `infos/dataset.md`, and the source under
`src/`. Scope: design only — nothing was changed.

## Summary

| Severity | Coverage gaps | Config errors | Methodological | Total |
|---|--:|--:|--:|--:|
| critical | 0 | 0 | 0 | 0 |
| major | 3 | 3 | 4 | 10 |
| minor | 2 | 2 | 3 | 7 |
| **Total** | **5** | **5** | **7** | **17** |

**Top findings (read these first):**

1. **[major] §3 config ≠ narrative.** `03_clustering.json` runs **14 clusterers / 42 runs** (3 GMM `max_clusters` variants + 4 DBSCAN `eps` variants), but §3 claims "9 clusterers × 3 limits = 27 runs" and never describes those sweeps. `exp03/report.md` ran 42.
2. **[major] §7 similarity-search scaling is confounded.** `07_scalability.json` sets `sample_n: null`, so the query count grows with `n`; the timed search stage mixes query-count growth with per-query scan growth and cannot isolate the O(n) scan signal. It also contradicts §7's own complexity table, which predicts O(n²).
3. **[major] Shared SQLite `db_path` hash leaks ingest timing across experiments.** §1/2/3/4/6/7 all use `exp/.common_db/{run_id}.sqlite`; for the fixed baseline the storage hash is identical, so the cost experiments (§6 sqlite, §7) can silently reuse a quality experiment's DB and report **its** ingest timing — and "run §6/§7 in parallel" would corrupt a shared SQLite file.
4. **[major] §8 generates on-topic retrieval-quality metrics it never uses.** Both §8 configs enable similarity search; the runner computes precision@k / recall@k / mAP / MRR against ground truth — directly on the thesis topic ("algoritmos de similitud") — yet §8's Metrics section only discusses clustering ARI/NMI/F1.
5. **[major] §8b is likely underpowered + has a style-head leakage risk.** 185 points / 87 classes / 53 singletons evaluated cross-city on unseen authors floor-bounds ARI/F1; separately, the style heads were trained on 110 Salamanca crops from the **same 265-image set** that the 185 author-eval crops are drawn from, with no stated image-level disjointness.
6. **[major] `throughput_ips` is computed from corpus size, not query count.** `models.py:484-486` divides `image_count` by sim-search wall time regardless of `sample_n`, overstating §6/§9 search throughput by up to ~64× and skewing the SQLite-vs-pgvector comparison.

---

## 1. Coverage gaps

- [major] `infos/experiments.md` §4 lines 239-243 + baseline lines 27-30 (UMAP hyperparameters): the catalogue depends everywhere on `n_neighbors=15, min_dist=0.0` and asserts they are "well-established" and that "an earlier sweep produced no surprises," but **no config sweeps `n_neighbors` or `min_dist`.** §2 varies only `n_components` (10 vs 50). The two UMAP knobs the whole pipeline rests on are untested in the catalogue. → Add a small UMAP-tuning grid, or drop the reference to a non-existent sweep and label the setting "adopted from literature."

- [major] `infos/experiments.md` §7 complexity table line 356 vs report `exp/exp07/report.md` lines 109-125: §7 predicts similarity search is "O(n²) per full corpus pass," but the only experiment that measures it reports slope **0.96 (linear)** and calls it "as expected." The prediction and the result contradict each other and nothing reconciles them (root cause in Methodological M1). → Decide which claim is true: keep all-pairs and actually demonstrate O(n²), or fix the query count and reframe the stage as per-query O(n) scan.

- [major] `infos/configs/08a_supervised_style.json` line 3 / `08b_supervised_author.json` line 3 vs `infos/experiments.md` §8 Metrics (lines 418-432): both §8 configs set `similarity_search.enabled = true`, so the runner produces precision@k / recall@k / mAP / MRR against ground truth (`clustering_metrics.py:173-267`) — a supervised **retrieval-quality** result that is squarely on the thesis topic. The §8 narrative analyzes only clustering ARI/NMI/F1 and never mentions these. → Either incorporate the retrieval metrics (they support pillar 3 and the title more directly than clustering does) or state why they are discarded.

- [minor] `infos/experiments.md` §1 (lines 132-160) + `exp/exp01/report.md`: §1 calls the embedding "the single most consequential choice," yet the baseline `dinov2_graffiti_style_head` used by §§2–7 ranks **mid-pack on every §1 unsupervised metric** (ImageNet CNNs win silhouette). Its selection rests entirely on §8, and no experiment varies embedding jointly with reduction/clustering, so the baseline is never shown to be jointly optimal (one-factor-at-a-time only). → State explicitly that the baseline is justified by §8 supervised evidence and is OFAT-validated; optionally add one cross-axis confirmation run.

- [minor] `infos/experiments.md` §8b line 453: the §8b clustering grid is still written as "proposed" ("the proposed §8b grid … Revisit if …"), but `exp/exp08b/config.json` is a concrete, frozen 96-run configuration. → Reconcile the tentative wording with the committed config.

## 2. Config errors / mismatches

- [major] `infos/configs/03_clustering.json` lines 17-30 vs `infos/experiments.md` §3 lines 199 & 213: the config declares **14 clustering specs** → 42 runs — GMM at `max_clusters ∈ {30,100,300}` (3 variants, all `covariance_type: diag`) and DBSCAN at `eps ∈ {auto,0.2,0.5,1.0}` (4 variants). §3 says "9 clusterers × 3 limits = 27 runs" and lists a single `gmm` and a single `dbscan`, never mentioning either sweep. `exp/exp03/report.md` confirms 42 runs. → Update §3 to 14 clusterers / 42 runs and document the GMM/DBSCAN parameter sweeps, or trim the config to match the text.

- [major] `infos/configs/{01,02,03,04,06,07}.json` storage `db_path: "exp/.common_db/{run_id}.sqlite"` + `src/benchmarks/generate_configuration.py:9-50` (`get_storage_key` hashes storage/embedding/segmenter/limit, **not** reduction/clustering) + `src/src/evaluation/runner.py:255-271` (INGEST_COMPLETE reuse): for the fixed baseline (sqlite, `dinov2_graffiti_style_head`, identity, a given limit) the storage hash is **identical across §2/§3/§4/§6/§7**, resolving to one file `exp/.common_db/storage_<hash>.sqlite`. Whichever experiment runs first populates it; later ones detect the marker, skip the wipe, and **inherit its ingest metrics**. For the cost experiments (§6 sqlite row, §7) at overlapping limits {250,500,1000,6416}, ingest time may come from a quality experiment run under different conditions rather than from a controlled §7 measurement. (§5 avoids this by using explicit `v2_seg_*` paths.) → Give §6/§7 isolated explicit `db_path`s, or wipe `exp/.common_db/` before each cost run.

- [major] `infos/experiments.md` "Suggested order" lines 642-644 + shared `db_path` above: the catalogue says §6 and §7 "can run … in parallel," but they share the same `storage_<hash>.sqlite` files for overlapping limits (500/1000/6416). Two processes writing one SQLite file concurrently corrupts it and produces garbage timings. → Remove the parallel suggestion for these two, or give each isolated DB paths first.

- [minor] `infos/configs/06_storage.json` line 14 (`sample_n: 100`), `09_hnsw.config.json` (`sample_n: 200`), `07_scalability.json` line 22 (`sample_n: null`): the three cost experiments time similarity search at three different query-sample sizes, and §6's text never states its value. Cross-experiment search numbers are therefore not directly comparable, and §7's narrative ("primary motivation for the pgvector upgrade tested in §6/§9") implicitly compares them. → Document `sample_n` per experiment and justify the differences.

- [minor] `exp/exp05/report.md` lines 29-34 and `exp/exp01/report.md` lines 28-34 are **stale** against the current catalogue: exp05 still flags a missing `yolo11n` / "five segmenters, 15 runs," but current §5 (lines 256-279) specifies four segmenters / 12 runs matching `05_segmenter.config.json`; exp01 still says §1 lists "ten embeddings," but current §1 lists twelve matching `01_embedding.json`. → Refresh the reports so reviewers don't read resolved deviations as live ones.

## 3. Methodological risks

- [major] `infos/configs/07_scalability.json` line 22 (`sample_n: null`) + `src/src/evaluation/runner.py:350-354`: with `sample_n=null` the runner queries **every** corpus point (`search_images = all_images`), so the number of queries scales with `n`. The timed search stage then conflates two n-dependent effects — query count and per-query scan length — so the headline "cost vs n" curve cannot isolate the SQLite linear-scan signal, and the per-query reload (`sqlite.py:96-105` re-reads the full table each query) makes the all-pairs reading O(n²·…) and expensive. §6/§9 correctly fix the query count. → Set a fixed `sample_n` (e.g. 100) in §7 so per-query scan cost is the only n-dependent term.

- [major] `src/src/evaluation/models.py:484-486`: similarity-search `throughput_ips = image_count / wall_time` uses the full corpus size, not the executed query count (`min(sample_n, n)`). For §6 (`sample_n=100`) and §9 (`sample_n=200`) this overstates search throughput by ~`n/sample_n` (up to ~64× at n=6416) and distorts the SQLite-vs-pgvector-vs-HNSW comparison if throughput (rather than raw wall time) is reported. → Divide by the number of queries actually issued.

- [major] `infos/configs/08b_supervised_author.json` + `infos/experiments.md` §8b lines 399-432 + `infos/dataset.md` lines 4-7,17: §8b evaluates 185 crops over 87 authors (53 singletons, 17 doubletons) **cross-city** on authors the head never saw (author heads trained on 169 disjoint Cuenca crops). ARI/F1 are floor-bounded; `agglomerative` at `n_clusters=87` on 185 points (~2.1 points/cluster, average linkage → chaining) is near-degenerate; the grid risks uniformly uninformative scores, so a null result would be uninterpretable. experiments.md acknowledges the cap but the design may be underpowered to confirm hypotheses 1–2. → Pre-register a per-embedding random/empirical ARI baseline and add a coarse-author grouping (or singleton-merge) so a null is distinguishable from "grid too narrow."

- [major] Train/eval leakage risk for the style-head rows in §8b: `infos/dataset.md` lines 4-7 say both the 110 style-train crops **and** the 185 author-eval crops are drawn from the same 265-image Salamanca set. `infos/experiments.md` (baseline lines 53-57) only asserts the 265/1106 annotation sets are disjoint from the 6416 eval corpus — it does **not** establish that the 110 style-train crops and 185 author-eval crops are image-disjoint. So `dinov2_graffiti_style_head` / `mobilenet_v3_graffiti_style_head` may be scored in §8b on images they trained on, inflating those rows (and biasing hypothesis 2's head-vs-head gap). → Verify image-level disjointness; exclude overlaps or disclose the contamination.

- [minor] `infos/configs/03_clustering.json` line 22 (DBSCAN `min_samples:5`, auto-eps) under K=3 + UMAP reseed (`runner.py:468-471`): eps is re-derived each repeat from a freshly seeded UMAP embedding (`dbscan.py:37-52`), so the aggregated row averages over **different partitions** (exp03 report: `clusters_per_repeat = [247,219,239]`), not over seeds of one partition. The mean±std is across partition families. → Pin `eps` for the aggregated DBSCAN rows, or report their per-repeat partitions separately; note the caveat in the catalogue (the report already does).

- [minor] `src/src/evaluation/runner.py:430-437` vs `infos/experiments.md` "Repeats: K=3" (lines 41-47): when reduction **and** clusterer are both deterministic (identity/pca/isomap + hdbscan/agglomerative — see `is_deterministic` flags), K collapses to 1. So §2's identity/pca/isomap rows and §8's identity+hdbscan / identity+agglomerative rows carry **no error bars**, while their UMAP/stochastic siblings show K=3. → State in the catalogue which rows are single-shot so missing `_std` is not read as "tight variance."

- [minor] `infos/configs/06_storage.json` / `09_hnsw.config.json` + `infos/storage.md` lines 53-56: §6/§9 are safe from pgvector's HNSW 2000-d cap only because they hard-code the 384-d baseline. Swapping in `resnet50` (2048) or `vgg16` (4096) would raise at `CREATE INDEX`. Currently **no config violates the limit** (the only >2000-d embeddings, ResNet50/VGG16, appear only with SQLite in §1/§8). → Add a guard or comment so a future embedding swap in §6/§9 fails loudly rather than mid-sweep.
