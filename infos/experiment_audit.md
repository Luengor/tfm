# Experiment Design Audit

Audit of the experiment catalogue (`infos/experiments.md`, §1–§9) against the
configs in `infos/configs/`, the component model (`CLAUDE.md`), the thesis brief
(`doc/enunciado.md`), and the runner/generator/storage source. **Audit only — no
files were changed.**

Each finding is `- [severity] location: problem → fix`. Severity =
critical / major / minor.

## Summary table

| ID | Sev | Area | One-line |
|----|-----|------|----------|
| C1 | minor | Coverage | Catalogue says "eight experiments" but defines nine; intro roadmap omits §9. |
| C2 | major | Coverage | §9's headline "build-time × query-time decomposition" is never measured (index build is untimed). |
| C3 | major | Coverage | §6 claims HNSW build cost "reported separately" — it is not captured anywhere. |
| C4 | critical | Coverage | §7 claims O(n²) similarity-search scaling, but `sample_n:100` makes the measured slope ~1; the headline is untestable as configured. |
| C5 | minor | Coverage | §1 claims to answer fine-tuning-vs-pretraining "directly", but §1 has no ground truth (unsupervised only). |
| C6 | minor | Coverage | §1 head selection is asymmetric (MobileNet author head vs DINOv2 style head) and partly unexplained. |
| C7 | minor | Coverage | §8 computes similarity-retrieval metrics (precision@k/MAP/MRR) that the §8 narrative never describes or uses. |
| E1 | major | Config | `generate_name` doesn't encode swept params → colliding `run_name`s in §2, §3, §4, §8a, §8b. |
| E2 | major | Config | §3 config has 14 clusterers / 42 runs; experiments.md §3 says 9 clusterers / 27 runs. |
| E3 | major | Config | §3 GMM `max_clusters` 100/300 → degenerate (k→n) and very expensive auto-k search; variants likely collapse to identical results. |
| E4 | major | Config | §3 DBSCAN fixed `eps` {0.2,0.5,1.0} are arbitrary in UMAP space (degenerate single-cluster / all-noise). |
| E5 | minor | Config | §2 deterministic rows silently run K=1 despite `repeats:3`; not documented. |
| E6 | minor | Config | §8a spectral omits `affinity`; `{"type":"identity"}` omits `params` — cosmetic inconsistencies. |
| M1 | major | Method | §8b evaluates the **style** heads on Salamanca author-eval crops that share the 265-image source set with the style-head training crops → possible leakage. |
| M2 | critical | Method | (=C4) `sample_n:100` invalidates the thesis's named headline (cost of similarity algorithms vs n). |
| M3 | major | Method | §6 & §9 spend the K=5 repeat budget on storage-invariant stages; the headline search/latency is single-shot. |
| M4 | minor | Method | §5 compares clustering quality across a changing unit of analysis (image vs crop), different point counts. |
| M5 | minor | Method | §3 ranks clusterers by k-sensitive intrinsic metrics across partitions with very different k. |
| M6 | minor | Method | §3/§7 Spectral (O(n³)) and §3 GMM-300 at n=6416 may not complete → missing points break slope fits / leave blank cells. |

### Verified clean (task-listed traps that do *not* apply)

- **pgvector >2000-dim + HNSW**: §6/§9 fix `embedding=dinov2_graffiti_style_head`
  (384-dim, `embeddings.py:82-85`). The >2000-dim backbones (resnet50=2048,
  vgg16=4096, inception_v3=2048) appear only in SQLite experiments (§1, §8). No
  violation.
- **§8a vs §8b DB-hash collision**: distinct `db_path` prefixes `08a_style_` /
  `08b_author_` are kept in the storage-key hash (`generate_configuration.py:19-24`),
  so the two sections never share a database. Correct.
- **§6 Postgres per-combination isolation**: the generator rewrites each Postgres
  `db_url` to `/storage_<key>` (`generate_configuration.py:155-167`) and the backend
  auto-creates missing databases (`postgresql.py:125-143`), so the exact/HNSW rows and
  the four limits each get their own DB (12 ingests, as claimed).
- **§9 run anchoring**: the canonical `m16_efc64_efs40` run is shared by all three
  one-axis sweeps → exactly 11 runs, matching the text.
- **Embedding names**: every `type` string in the configs resolves in
  `EmbeddingModelNames` (`embeddings.py:29-42`).
- **Extrinsic-metrics gate**: extrinsic metrics require `segmenter=="identity"`
  (`runner.py:449-454`); all §8 runs use identity.

---

## 1. Coverage gaps

- [minor] `experiments.md:17` and `:654`: the intro says "The eight experiments
  below" and History says "The remaining eight experiments", but the catalogue
  defines **nine** (§1–§9). Worse, the intro roadmap (`:17-20`: "quality sweeps
  §§1–5 … cost sweeps §§6–7 … supervised §8") omits **§9 (HNSW)** entirely; §9 only
  reappears in "Suggested order" (`:638`). → Update the count to nine and fold §9
  into the framing, or state explicitly why it sits outside it.

- [major] §9 / `runner.py:327-328` + `configs/09_hnsw.config.json`: §9's stated
  central deliverable is "the build-time × query-time decomposition the thesis
  needs" (`experiments.md:583-584`) and "the dominant cost is the ten HNSW index
  builds" (`:608`). But the runner calls `storage.ensure_hnsw_index()` **outside any
  `profile_stage()`**, and `BenchmarkResult` has no field for index-build time. The
  build cost is therefore never measured or recorded — §9 cannot produce its main
  result as instrumented. → Time `ensure_hnsw_index()` and add a result field, or
  drop the build-time decomposition claim.

- [major] §6 / `experiments.md:291-294`: claims "the runner builds the HNSW index
  between ingest and the similarity-search loop so the index build cost is excluded
  from the timed query stage **and reported separately as part of the
  ingest-adjacent setup**." Same root cause as C2 — the build is untimed and not
  reported anywhere. → Correct the §6 text or instrument the build (see C2).

- [critical] §7 / `configs/07_scalability.json:22` vs `experiments.md:349,357-360`:
  the §7 complexity table lists similarity search as "O(n²) per full corpus pass" and
  the headline is *empirically measured slopes* (~2 expected for search). But the
  config sets `similarity_search.sample_n = 100`, so only 100 queries run at every
  `n` (`runner.py:348-351`); with the SQLite O(n) linear scan per query the measured
  search cost is O(100·n) = **O(n)**, slope ≈ 1, not ≈ 2. The config cannot
  demonstrate the quadratic search scaling the thesis headline rests on (the brief's
  *"coste computacional en función del tamaño … en algoritmos de similitud"*). →
  Set `sample_n: null` (or `= limit`) so the all-pairs search is actually timed, or
  rewrite the complexity claim to O(n) for a fixed query budget. Also applies to §6,
  but §6's *relative* (per-query) framing survives `sample_n=100`.

- [minor] §1 / `experiments.md:148-150`: "Pairs … answer the fine-tuning vs.
  pretraining question directly." §1 is purely unsupervised on the full 6416-image
  corpus with **no ground truth** (`configs/01_embedding.json`), so silhouette / CH /
  DB cannot validate semantic gain — a head can inflate intrinsic scores by
  collapsing geometry without improving semantics. The fine-tuning question is
  actually answered by §8. → Soften §1's claim and cross-reference §8.

- [minor] §1 / `configs/01_embedding.json:21,23`: head selection is asymmetric —
  the config includes `mobilenet_v3_graffiti_author_head` and
  `dinov2_graffiti_style_head` but not `mobilenet_v3_graffiti_style_head` or
  `dinov2_graffiti_author_head`. The text justifies omitting only the DINOv2 author
  head (`:140-143`); it never explains using the *author* head for MobileNet but the
  *style* head for DINOv2. The two "fine-tuning pairs" thus optimize different
  objectives, confounding the within-§1 comparison. → Use the same head task for both
  backbones, or document the asymmetry.

- [minor] §8 / `configs/08a_supervised_style.json:3`, `08b:3`: both enable
  `similarity_search`, so the runner computes retrieval-extrinsic metrics
  (`precision_at_k`, `map_at_k`, `mrr`; `runner.py:398-412`, `models.py:382-403`).
  §8's Metrics and Hypotheses (`experiments.md:411-485`) never mention or use them.
  These are arguably the most on-topic "desempeño de algoritmos de similitud"
  metrics in the whole catalogue. → Either fold them into §8's analysis or disable to
  save compute.

## 2. Config errors

- [major] `generate_configuration.py:52-77` (`generate_name`) → run-name collisions
  in §2, §3, §4, §8a, §8b. `generate_name` only special-cases `kmeans` + `n_clusters`
  and `storage` + `hnsw`; for every other swept parameter it appends just the
  component `type`. Consequences (run_id stays unique, but the human-facing `name`
  collides):
  - §2 (`configs/02_reduction.json:18-21`): `pca`-10 and `pca`-50 both become
    `pca-l{limit}`; `umap`-10 and `umap`-50 both become `umap-l{limit}`.
  - §3 (`configs/03_clustering.json:19-30`): 3× `gmm-l{limit}`, 4× `dbscan-l{limit}`,
    2× `agglomerative-l{limit}`, 2× `spectral-l{limit}`.
  - §4 (`configs/04_hdbscan.json:17-22`): all six `min_cluster_size` rows become
    `hdbscan-l{limit}`.
  - §8a (`08a:28-30`): 3× `…-hdbscan` per embedding×reduction; §8b (`08b:25-30`):
    2× `…-agglomerative` and 4× `…-hdbscan`.
  `--run <name>` selection (`CLAUDE.md`) is then ambiguous, and any `pipeline-plot` /
  table grouping keyed on `run_name` conflates variants. → Extend `generate_name` to
  encode the swept param (`hdbscan{min_cluster_size}`, `pca{n_components}`,
  `dbscan{eps}`, `gmm{max_clusters}`, `agglomerative{n_clusters}`,
  `spectral{n_clusters}`).

- [major] §3 mismatch: `experiments.md:194-208` lists 9 clusterers and "**27 runs**
  (9 clusterers × 3 limits)", but `configs/03_clustering.json:17-30` defines **14**
  clusterer specs → **42 runs**. Undocumented extras: `gmm` ×3
  (`max_clusters` 30/100/300; text says one "gmm (auto-k)") and `dbscan` ×4
  (auto-eps + eps 0.2/0.5/1.0; text says one "dbscan"). → Reconcile: trim the config
  to the 9 described, or update the §3 text and cost to 14 clusterers / 42 runs.

- [major] §3 GMM degeneracy/cost: `configs/03_clustering.json:20-21` set
  `max_clusters` 100 and 300. Auto-k is bounded by `min(n, max_clusters)`
  (`gmm.py`), so on repeat 0 the BIC search fits k = 2..min(n,max_clusters): at
  `limit=250` that is k up to 250 (~249 GMM fits, k≈n ⇒ singular/meaningless
  components even with `diag`), at `limit∈{1000,6416}` up to 300. The three GMM
  variants also collapse to identical labels whenever the BIC minimum is < 30. →
  Drop the 100/300 variants (or cap `max_clusters` ≤ ~50) and document the
  `min(n, max_clusters)` bound.

- [major] §3 DBSCAN arbitrary eps: `configs/03_clustering.json:23-25` set
  `eps ∈ {0.2, 0.5, 1.0}` on UMAP(`min_dist=0`, 10-d) output, whose Euclidean scale
  is not unit/cosine-normalised. eps=1.0 will likely merge everything into one cluster
  (silhouette degenerate / undefined) and eps=0.2 likely all-noise. These rows are not
  in the text, and `best_cluster.md:37-38` rates DBSCAN "avoid". → Keep only the
  auto-eps DBSCAN (matches the text) or justify the eps grid against the measured UMAP
  scale.

- [minor] §2 silent K=1: `runner.py:427-434` drops K to 1 when reduction and
  clustering are both deterministic. In §2 the `identity`, `pca`-10, `pca`-50 and
  `isomap` rows (deterministic) + HDBSCAN (deterministic, `hdbscan.py:27`) run **K=1**,
  not the declared `repeats:3` (`configs/02_reduction.json:3`); only the two UMAP rows
  run K=3. Cost samples for those rows also keep the un-dropped iter-0 JIT warm-up
  (`models.py:97`, `len<2`). Correct (repeats would be identical) but undocumented. →
  Note in §2 that deterministic rows are single-shot, so their timing has no error
  bars.

- [minor] §8 cosmetic inconsistencies: §8a spectral
  (`configs/08a_supervised_style.json:27`) omits `affinity` while §3/§7 set
  `affinity:nearest_neighbors` — harmless only because `SpectralClusterer` defaults to
  `nearest_neighbors` (`spectral.py:35`). `{"type":"identity"}`
  (`08a:21`, `08b:21`) omits the `params` key used everywhere else (parser tolerates
  it). → Add the explicit params for parity.

## 3. Methodological risks

- [major] §8b style-head leakage: `dataset.md:4-7` shows the **110 Salamanca
  style-train crops** and the **185 Salamanca author-eval crops** are both drawn from
  the same 265-image Salamanca set. §8b
  (`configs/08b_supervised_author.json:13,16`) evaluates
  `dinov2_graffiti_style_head` and `mobilenet_v3_graffiti_style_head` on those 185
  crops, so the style encoders may have seen the same source images during
  supervised-contrastive training → upward-biased §8b style-head scores and a
  confounded cross-task transfer matrix (Hypothesis 2, `experiments.md:466-473`).
  `experiments.md:53-55` only asserts disjointness from the 6416-image corpus, not
  between the 110 style-train and 185 author-eval crops. → Verify image-level
  disjointness of those two splits; if they overlap, exclude the style heads from §8b
  conclusions or report them with an explicit leakage caveat.

- [critical] §7 headline validity (= C4): with `sample_n:100`
  (`configs/07_scalability.json:22`) the similarity-search slope measured by §7 is
  ~1, so any "similarity search is O(n²)" conclusion drawn from §7 would be
  unsupported by its own data — and similarity-search cost-vs-n is the thesis's named
  topic. → Run the search axis with `sample_n: null`/`= limit`, or reframe the claim.

- [major] §6 & §9 repeat budget misallocated: similarity search is single-shot — it
  runs once, outside the K loop (`runner.py:324-380`; acknowledged
  `experiments.md:314-320`). So the K=5 repeats only re-run the **storage-invariant**
  UMAP + HDBSCAN stages, which are irrelevant to §6's storage-throughput question and
  §9's latency/recall Pareto. The headline metrics get no within-run repeats; their
  variance is inter-run only. → Repeat the search stage (or state that error bars come
  from the per-query sample n=100/200, not run repeats) and consider lowering K for
  these two.

- [minor] §5 unit-of-analysis confound: `configs/05_segmenter.config.json` compares
  `identity` (one point per image) against YOLO crops (variable points per image), so
  silhouette / CH / DB for the `identity` vs `yolo*` rows are computed over different
  point sets and counts. Acknowledged (`experiments.md:261-264`) but the headline
  "does cropping improve clustering quality" still compares non-comparable metric
  bases. → Report a per-image aggregation alongside per-crop, or restrict the quality
  claim.

- [minor] §3 k-confounded ranking: the §3 clusterers span auto-k (KMeans elbow, GMM
  BIC), density-auto (HDBSCAN/DBSCAN/OPTICS) and fixed k=10/20
  (agglomerative/spectral). Silhouette/CH/DB are strongly k-dependent, so "which
  clusterer is best" partly measures k. Only partly mitigated by the
  agglomerative/spectral k-sweep (`experiments.md:204-206`). → Report metric-vs-k
  where possible; avoid ranking different-k partitions on a single intrinsic number.

- [minor] §3/§7 super-quadratic feasibility at n=6416: Spectral (O(n³) eigendecomp,
  `spectral.py`) in §3 and §7, and the GMM-300 BIC search in §3, may not complete or
  may OOM at the largest `n`. A failed run becomes a missing point that breaks the §7
  slope fit or leaves a blank §3 cell. → Confirm these complete within the time/memory
  budget, or cap the largest `n` for the super-quadratic clusterers.
