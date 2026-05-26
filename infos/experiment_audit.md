# Experiment Design Audit

Audit of the experiment catalogue (`infos/experiments.md`, §1–§9) cross-checked
against the configs in `infos/configs/`, the component model in `CLAUDE.md`, the
thesis brief in `doc/enunciado.md`, and the actual implementation under `src/`.
Audit only — nothing was changed.

## Summary

| # | Category | Severity | Location | One-line problem |
|---|---|---|---|---|
| C2.1 | Config error | critical | configs/08a+08b, generate_configuration.py:9-45 | §8a/§8b share db filenames (dataset path not in hash) — can silently mix corpora; text claims "separate DBs". |
| G.1 | Coverage gap | major | configs/02_reduction.json:16-23 | KernelPCA rows (2) missing; text claims 8 reductions / 24 runs, config has 6 / 18. |
| G.2 | Coverage gap | major | configs/08a+08b | Supervised similarity-retrieval metrics (P@k/R@k/mAP/MRR) never computed — search never enabled with labels. |
| C2.2 | Config error | major | experiments.md §5:260-263 | Stated reason for hand-writing §5 (hash "excludes segmenter") is false — hash includes segmenter. |
| C2.3 | Config error | major | experiments.md §9:590-597 | Claims recall@k is "future work"; `ann_recall_at_k` is implemented and reported. |
| M.1 | Methodological | major | clustering_metrics.py:51-83 (§1,§3) | Noise-excluded silhouette/CH/DB make density vs partitional clusterers non-comparable. |
| M.2 | Methodological | major | configs/07_scalability.json:22 | `sample_n=100` makes SQLite search O(n), so §7's headline "O(n²)" search signal cannot appear. |
| C2.4 | Config error | minor | experiments.md Metrics:73-74 | Says extrinsic computed once on last iter; code aggregates per-repeat mean±std. |
| C2.5 | Config error | minor | experiments.md §8:542-544 | References non-existent `configs/08_supervised.json`. |
| C2.6 | Config error | minor | experiments.md:47-53 vs dataset.md | Crop-provenance claim contradicts dataset.md; leakage rationale rests on it. |
| G.3 | Coverage gap | minor | experiments.md §1:145-147 | "Fine-tuning vs pretraining answered directly" — but §1 is unsupervised only; claim is circular. |
| G.4 | Coverage gap | minor | configs/02–07 | Baseline embedding hardcoded before §1 picks the winner. |
| M.3 | Methodological | minor | runner.py:301-336 (§6,§9) | Similarity-search stage is single-shot; repeats=5 gives no error bars on the latency headline. |
| M.4 | Methodological | minor | runner.py:445-448, dbscan.py:32-54 | DBSCAN auto-`eps` not propagated across repeats (only `n_clusters` is) → §3/§7 mix partitions. |
| M.5 | Methodological | minor | configs/04_hdbscan.json:21-22 | `min_cluster_size` 100/200 at limit 250/1000 → degenerate all-noise cells, no metrics. |
| M.6 | Methodological | minor | configs/08b + dataset.md | Possible style-head train/eval image overlap inflates §8b style-head rows. |

---

## 1. Coverage gaps

- [major] `configs/02_reduction.json:16-23` (§2): experiments.md §2 (lines 163-166) lists
  **eight** reductions — identity, pca-10, pca-50, umap-10, umap-50, isomap-10, **kernel_pca-50 RBF, kernel_pca-50 cosine** — and the Cost line (176) claims "24 runs (8 reductions × 3 limits)". The config defines only **six** reductions (no KernelPCA at all) → **18 runs**. The two KernelPCA rows the narrative leans on for the "non-linear reductions help?" question are absent. → Add the two `kernel_pca` rows (`{"n_components":50,"kernel":"rbf"}` and `{"kernel":"cosine"}`), or correct the text to 6 reductions / 18 runs and drop the KernelPCA discussion.

- [major] `configs/08a_supervised_style.json`, `configs/08b_supervised_author.json` (§8): the supervised similarity-retrieval metrics implemented in `calculate_similarity_search_extrinsic` (clustering_metrics.py:162) — Precision@k, Recall@k, mAP@k, MRR — are **never computed in any experiment**. §6/§7/§9 enable `similarity_search` but run without `--ground-truth`; §8 supplies ground truth but **omits the `similarity_search` block** (configs 08a/08b have no such key, so it defaults to `enabled:false`, models.py:13-18). The thesis brief is explicitly about "algoritmos de similitud" (retrieval), yet retrieval quality on labelled data is never measured. → Add `"similarity_search": {"enabled": true, "top_k": 5, "sample_seed": 42}` to 08a/08b so `similarity_extrinsic` is produced on the labelled crops.

- [minor] `experiments.md §1:145-147`: claims the `mobilenet_v3 ↔ author_head` and `dinov2 ↔ style_head` pairs "answer the fine-tuning vs. pretraining question directly." §1 has no ground truth and ranks embeddings purely by unsupervised silhouette/CH/DB. A head trained with contrastive/triplet loss to compact classes will raise silhouette by construction, independent of semantic correctness — so the comparison is circular. The pairing is also asymmetric (an **author** head for MobileNet vs a **style** head for DINOv2). → Soften the §1 claim; defer the fine-tuning verdict to §8 (supervised), and state which head-type each backbone is paired with.

- [minor] `configs/02–07`: every non-§1 experiment hardcodes `dinov2_graffiti_style_head` as the baseline embedding, but "Suggested order" (623-627) says §1 should pick the winner first. If §1's best embedding differs, §2–§7 were run on a suboptimal baseline. → Note the regeneration dependency, or run §1 first and templatise the baseline before generating §2–§7.

---

## 2. Config errors and experiments.md ↔ code/config mismatches

- [critical] `configs/08a` + `configs/08b` vs `generate_configuration.py:9-45` and `configuration.py:167-185`:
  experiments.md §8 Cost (496-497) states §8b uses "separate DBs — different dataset path." But neither `get_storage_key` nor `make_run_id` includes the dataset path; the SQLite key is `hash(storage, embedding, segmenter, limit)`. §8a and §8b therefore generate **identical** `db_path` values (`{output_dir}/db/storage_<hash>.sqlite`) for each matching embedding. With the same `--output-dir`, any non-sequential execution — running them concurrently, or using `--run` to re-execute a §8b subset that skips the `clear_storage:true` row — reuses §8a's **style-corpus** embeddings as §8b's **author** data, silently corrupting the supervised (pillar-3) results. The doc's false reassurance increases the risk. → Give each split its own `--output-dir`, or include the dataset path in `get_storage_key`/`make_run_id`, or set explicit distinct `db_path` per config; and fix the text.

- [major] `experiments.md §5:260-263`: states `generate_configuration.py` "hashes `storage + embedding + limit` ... and does *not* include the segmenter, so a grid would incorrectly reuse ingests across segmenter variants." This is false: `get_storage_key` explicitly includes a normalised segmenter (generate_configuration.py:13, 28-39). §5 could safely be a grid; the hand-written config still works but its stated justification is wrong (an examiner checking the code will catch it). → Correct the rationale (segmenter *is* hashed); optionally convert §5 to a grid.

- [major] `experiments.md §9:590-597`: claims "The runner currently reports `avg_neighbor_distance` ... A dedicated recall@k metric is left as future work." Not true — `runner.py:342-349` computes `ann_recall_at_k` via `calculate_ann_recall` (clustering_metrics.py:259) whenever an ANN index is active, and it is a first-class result field (models.py:419, 444; documented in CLAUDE.md). §9 undersells its own measurement. → Update §9 to report `ann_recall_at_k` as the real HNSW recall metric instead of the distance proxy.

- [minor] `experiments.md Metrics:73-74`: "Extrinsic metrics are still computed once on the last iteration's labels (seed K-1, reproducible)." The code computes extrinsic metrics **per repeat** and aggregates mean±std (`runner.py:459-466, 489-492`; `ExtrinsicMetricsAgg`, models.py:278). → Update the text to match (per-repeat aggregation).

- [minor] `experiments.md §8:542-544`: references `configs/08_supervised.json` as a legacy file, but it does not exist in `infos/configs/` (only `08a_*` and `08b_*`). → Remove the reference or restore the file.

- [minor] `experiments.md:47-53` vs `dataset.md`: experiments.md says the "385 train crops ... are derived from the full image bank" (the 6416-image eval corpus) and uses that to argue they don't perturb statistics. `dataset.md` places the 110 Salamanca style-train crops in the separate 265-image set and the 275 in Cuenca — neither in the 6416. The provenance claim (and the leakage argument built on it) is inconsistent. → Reconcile experiments.md with dataset.md.

---

## 3. Methodological risks

- [major] `clustering_metrics.py:51-83` (affects §1, §3, §5): silhouette / Calinski-Harabasz / Davies-Bouldin are computed only on non-noise points (`labels != -1`). In §3 (clustering-algorithm comparison) this scores HDBSCAN/DBSCAN/OPTICS on their confident core only, while KMeans/GMM/Agglomerative/Spectral are scored on all points — a clusterer that dumps hard cases into noise gets an unfair intrinsic-score advantage. In §1 the same effect varies the scored subset across embeddings (different noise ratios). The headline ranking metric is therefore confounded. → Always read silhouette jointly with `noise_ratio`/coverage, and consider a common-support reading (e.g. score all points, or compare only at matched coverage).

- [major] `configs/07_scalability.json:22` (§7, the headline experiment): the expected-scaling table (experiments.md §7:343) bills SQLite similarity search as "O(n²) per full corpus pass" and the narrative (330) calls it "a separate signal" of corpus-size sensitivity. But `sample_n=100` caps queries at 100 for every `limit` (`runner.py:304-307`), so total search cost is 100·O(n)=O(n) — slope ~1, not ~2. The stated O(n²) result cannot appear. → For §7 set `sample_n: null` (true full-corpus pass) so query count scales with n, or correct the expected-scaling text to O(n) given the fixed sample.

- [minor-major] `runner.py:301-336` (§6, §9): the similarity-search stage is timed **once** per run (single `profile_stage`); `repeats=5` multiplies only reduction+clustering. §6's headline is the cross-backend throughput gap and §9's is the latency/quality Pareto frontier — both depend on search latency, which has no within-run error bars despite `repeats=5`. §6 acknowledges this; §9 does not. → Loop the search stage (K times) for cost-focused runs so the latency axis gets error bars.

- [minor] `runner.py:445-448` + `dbscan.py:32-54` (§3, §7): CLAUDE.md states KMeans/DBSCAN/GMM auto-detected counts are propagated across repeats, but the runner only injects `n_clusters`, which `DBSCANClusterer` ignores (it keys on `eps`). Since UMAP is reseeded per repeat, DBSCAN re-derives a different `eps` each repeat and the aggregated metrics span different partitions (the runner warns at runner.py:505-511). → Propagate `eps` for DBSCAN (mirror the `n_clusters` logic), or pin `eps`, or document that DBSCAN repeats are not k-stable.

- [minor] `configs/04_hdbscan.json:21-22` (§4): `min_cluster_size` of 100 and 200 at `limit ∈ {250, 1000}` forces all-noise or a single cluster (200 ≥ 80% of 250 points), leaving silhouette/CH/DB undefined for those cells. The "does optimal mcs shift with n?" reading gets no signal there. → Cap `min_cluster_size` relative to each limit, or drop the degenerate (small-limit, large-mcs) cells.

- [minor] `configs/08b_supervised_author.json` + `dataset.md` (§8b): §8b evaluates the DINOv2/MobileNet **style** heads (as cross-task controls) on the 185 Salamanca author-eval crops. The style heads were trained on 110 Salamanca crops drawn from the same 265-image set those eval crops come from, so image-level overlap is possible and would inflate the style-head rows. The author heads (Cuenca-trained) are clean. → Verify the 110 style-train and 185 author-eval crops are image-disjoint; if not, exclude overlapping source images or flag the style-head §8b numbers as optimistic.
