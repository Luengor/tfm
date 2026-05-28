# Thesis Consistency Audit

Cross-check of three layers for the graffiti-clustering TFM:

1. **Thesis document** — LaTeX under `doc/` (`doc/informe.tex`; sections in
   `doc/doc/*.tex`). Note the section split: **`tools.tex` = "Materiales y
   métodos"** and **`methods.tex` = "Metodología"** (easy to confuse).
2. **Experiment catalogue** — `infos/experiments.md`, `infos/dataset.md`,
   `infos/configs/*.json`.
3. **Codebase** — `src/src/**`.

**Scope.** Per the audit brief, the *Resultados* (`doc/doc/results.tex`) and
*Conclusiones* (`doc/doc/conclusion.tex`) sections are stubs and were **not**
audited. The resumen/abstract (`doc/doc/resumen.tex`) is also a placeholder.
This audit covers the written sections: introducción, estado de la cuestión,
materiales y métodos, metodología.

**Note on the referenced prior audit.** `infos/experiment_audit.md` (cited by
the brief as an existing format model) does **not** exist in the repo, nor does
`infos/metrics.md` — the metrics documentation lives in the `## Metrics`
section of `experiments.md`. Because no prior audit was present, this report
covers the experiments↔config↔code layer as well as the thesis layer.

All findings were verified against the current files (commit `6c7712e`).

---

## Summary table

| # | Severity | Location | Problem (one line) |
|---|---|---|---|
| A1 | **major** | `tools.tex:156` | Author-eval crop count given as `110`; should be `185`. |
| A2 | **major** | `tools.tex:23/41/44` ↔ `methods.tex:258/274` | Second dataset called "StopGrafiti" then "Cuenca"; equivalence never stated. |
| B1 | **major** | `methods.tex:113–128` | Storage-reuse "half-of-stored-time" heuristic described; code decides by the `clear_storage` flag. |
| B2 | **major** | `methods.tex:355–385` | Heads: 80/20 train/val split + validation-based selection claimed; code does neither. |
| B3 | medium | `tools.tex:59/297`, `methods.tex:334–337` | Detector training set (Salamanca-only vs both banks) and 80/20 split don't match code; unfilled placeholders. |
| C1 | minor | `tools.tex:209` | GMM BIC sweep stated as `k∈{1..kmax}`; code uses `{2..kmax}`. |
| C2 | minor | `methods.tex:231` | Davies-Bouldin omitted from the noise-excluded metric list; code excludes noise from it too. |
| C3 | minor | `tools.tex:225–226` | Clusterer table lists HDBSCAN `min_samples` / OPTICS `xi`; neither is wired in code. |
| C4 | minor | `tools.tex:139` | `silhouette_macro` is computed & emitted but undocumented (self-flagged TODO). |
| C5 | minor | `methods.tex:396` | Isomap listed as reusing the random seed; it takes no `random_state` and is deterministic. |
| C6 | minor | `methods.tex:144–167` | Deterministic `K=1` short-circuit not documented; some quality rows have no error bars. |
| C7 | minor | `tools.tex:324–325` | Heads said to be in `tab:embedding-models`, but the table lists only the 7 base families. |
| C8 | minor | `methods.tex:207–210` | Retrieval quality leans on `avg_neighbor_distance`; the proper `ann_recall_at_k` metric is unmentioned. |
| D1 | medium | `experiments.md:54` | Cuenca set called "1207-image"; dataset.md and thesis say 1106. |
| D2 | minor | `experiments.md:43–44` | Says RSS/VRAM aggregated over K-1 samples; code keeps iter 0 for memory fields. |
| D3 | minor | `CLAUDE.md` | Reuse key described without segmenter; "YOLO (n/s/m)" vs experiments' n/m. |

---

## A. Thesis ↔ dataset numbers and terminology

### A1 — Wrong author-evaluation crop count (`110` should be `185`)
- **Severity:** major · **Urgency:** high — a flat factual error in a written
  methods section; one-character fix; it contradicts the same document's own
  cross-referenced section.
- **Location:** `doc/doc/tools.tex:155–156` (§Marco conceptual → Métricas de
  evaluación).
- **Problem:** The text says extrinsic metrics are computed *"sobre los `$294$`
  recortes … etiquetados por estilo o los `$110$` etiquetados por autoría,
  ambos descritos en `\ref{subsec:dataset}`"*. The author-evaluation set is
  **185** crops, not 110. `110` is the number of *Salamanca style-training*
  crops (`tools.tex:41`). The correct author-eval count (185) appears in the
  referenced section (`tools.tex:46`) and again in `methods.tex:275` and
  `methods.tex:288`, and in `dataset.md:7`.
- **Fixes:**
  - Change `$110$` → `$185$` at `tools.tex:156`.
  - Optionally restate as "294 por estilo / 185 por autoría" for clarity.

### A2 — Second dataset named both "StopGrafiti" and "Cuenca"
- **Severity:** major · **Urgency:** high — terminology break that makes the
  evaluation methodology hard to follow; trivial fix.
- **Location:** introduced as **StopGrafiti** in `tools.tex:23`, `tools.tex:41`,
  `tools.tex:44` (§Conjunto de datos); referred to as **Cuenca** in
  `methods.tex:258` and `methods.tex:274` (§Evaluación extrínseca).
- **Problem:** §Conjunto de datos introduces the auxiliary corpus only as the
  "StopGrafiti" set ("un banco … con autores, estilos y entornos distintos a
  los de Salamanca") and never names a city. The methodology section then says
  the author heads are trained "exclusivamente con los `$169$` recortes de
  **Cuenca**" and that style-train crops come "de Salamanca y **Cuenca**". A
  reader cannot connect "Cuenca" to "StopGrafiti". `dataset.md` confirms they
  are the same set (its heading is `# Cuenca (stopgrafiti)`), but the thesis
  never states this.
- **Fixes:**
  - Establish the equivalence once in §Conjunto de datos, e.g. "el banco de
    StopGrafiti (procedente de Cuenca)", then use one name consistently.
  - Or replace "Cuenca" with "StopGrafiti" at `methods.tex:258` and `:274`.

---

## B. Thesis ↔ code: methodology

### B1 — Storage-reuse decision is flag-based, not a timing heuristic
- **Severity:** major · **Urgency:** high — the thesis describes an algorithm
  the code does not implement; a reviewer reading the code will not find it.
- **Location:** `doc/doc/methods.tex:113–128` (§Reutilización de
  almacenamientos), esp. `:125`. Code: `src/src/evaluation/runner.py:254–314`
  (decision comment at `:282–289`).
- **Problem:** The thesis says reused ingest timings are guarded by a timing
  heuristic: *"si el tiempo de ingesta medido es inferior a la mitad del
  almacenado … se descarta y se reporta el valor persistido; en caso contrario
  se considera una ingesta real."* The code contains **no such comparison**.
  It decides purely on the effective `clear_storage` flag: a fresh run
  (`effective_clear=True`) persists its measured metrics; a reusing run
  (`effective_clear=False`) reads the stored metrics back. The in-code comment
  is explicit: *"keyed off effective_clear … rather than a timing heuristic."*
  `CLAUDE.md` already documents the flag-based behaviour, so the thesis is the
  outlier. (The section is immediately preceded by a `% TODO: continue
  checking` marker.)
- **Fixes:**
  - Rewrite `:120–128` to describe the flag-based decision (fresh run saves,
    reusing run reads back; reuse is triggered by `clear_storage=false` set by
    the grid generator, plus the `INGEST_COMPLETE_KEY` marker guard).
  - Drop the "mitad del almacenado" sentence entirely.

### B2 — Head training: no 80/20 split, no validation-based selection
- **Severity:** major · **Urgency:** medium-high — methodology claims that the
  shipped training code does not perform; section is self-flagged WIP.
- **Location:** `doc/doc/methods.tex:355–385` (§Cabezas de proyección), esp.
  `:357` (split), `:382` (early-stop on validation), `:383–385` (final = min
  validation loss). Code: `src/src/train/style_trainer.py`,
  `src/src/train/trainer.py`, and entry points in `src/src/train/{dino,mobilenet}/`.
- **Problem:** The thesis says the four heads are *"dividido nuevamente en
  entrenamiento y validación con proporción `$80/20$`"*, with early stopping
  *"sobre la pérdida en validación"*, and that the selected head *"es la que
  minimiza la pérdida sobre la partición de validación."* In the code:
  - `trainer.py` (author / triplet): no train/val split, **no early stopping**,
    saves the **final-epoch** model (`:71–73`). The matching hyperparameters
    *do* check out (margin 1.0, Adam 1e-4, batch 8, 20 epochs, hard-negative
    prob 0.5).
  - `style_trainer.py` (style / SupCon): no train/val split (uses the whole
    `dataset_root`); early stopping and best-model saving are on **training**
    loss, not validation (`:188–198`). Other hyperparameters check out
    (T=0.07, 60 epochs, patience 15, 4/class, 10–30 batches, cosine schedule,
    sampling with replacement).
  So the 80/20-split + validation-selection narrative is unsupported for both
  head families.
- **Fixes:**
  - If the heads were genuinely trained with a held-out split (e.g. via a
    script not in the repo), add that script and cite it; otherwise correct the
    prose: style heads select on best **training** loss with patience 15;
    author heads use the **final** epoch with no early stopping; remove the
    80/20 claim or implement a real split.
  - Resolve the self-flagged `%TODO: check this once the heads are trained for
    the "final" time` at `methods.tex:354`.

### B3 — Detector training set and split don't match code
- **Severity:** medium · **Urgency:** medium — internal inconsistency plus a
  mis-described split; section is self-flagged WIP with placeholders.
- **Location:** `tools.tex:59–61` (§dataset, "ambos grupos", 447 imgs) vs
  `tools.tex:297` (§Segmentador, "subconjunto anotado **de Salamanca**") and
  `methods.tex:334–337` (§Detector). Code: `src/src/scripts/fine_tune_yolo.py`.
- **Problems:**
  1. **Source corpus:** §dataset says the detector is fine-tuned on images
     *"de ambos grupos … con un total de 447 imágenes"* (matching
     `dataset.md:19`), but §Segmentador (`:297`) and §Detector (`:335`) say
     *"de Salamanca"*. Pick one — the 447-image figure spans both banks.
  2. **Split rule:** the thesis claims an *"80/20 manteniendo la distribución
     aproximada de cajas por fotografía"* split (`:336–337`). The code does
     **not** do this: `fine_tune_yolo.py:87–95` routes images tagged
     `cluttered` to validation and all others to train — a content-based
     (clean→train / cluttered→val) split whose ratio is whatever the cluttered
     fraction happens to be, not a box-stratified 80/20.
  3. **Placeholders:** `methods.tex:334–335` still contain literal
     `$\mathtt{YOLO\_IMGS\_PLACEHOLDER}$` and `$\mathtt{NSEGS\_PLACEHOLDER}$`.
- **Fixes:**
  - Reconcile the corpus statement (both banks / 447) across §dataset,
    §Segmentador and §Detector.
  - Describe the actual clean-vs-cluttered split, or change the code to the
    described 80/20 stratified split.
  - Fill the two placeholders (image count and annotated-box count) once the
    detector is finalised (`% TODO` at `methods.tex:332`).

---

## C. Thesis ↔ code: metrics and component parameters

### C1 — GMM auto-`k` sweep range off by one
- **Severity:** minor · **Urgency:** low.
- **Location:** `tools.tex:207–209`. Code: `src/src/cluster/gmm.py:47`.
- **Problem:** The thesis says GMM minimises BIC over `k∈{1,2,…,kmax}`. The
  code sweeps `range(2, max_k+1)` = `{2,…,kmax}` and the in-code comment
  explains it deliberately skips `k=1` (a single component collapses the
  clustering and leaves the intrinsic metrics undefined). KMeans' `{2,…,kmax}`
  (`:205`) is correct.
- **Fixes:** Change `$k \in \{1, 2, \ldots, k_{\max}\}$` → `$\{2, \ldots,
  k_{\max}\}$`; optionally note why `k=1` is excluded.

### C2 — Davies-Bouldin missing from the noise-excluded list
- **Severity:** minor · **Urgency:** low.
- **Location:** `methods.tex:229–235` (§Evaluación intrínseca). Code:
  `src/src/evaluation/clustering_metrics.py:85–93`.
- **Problem:** The thesis says noise (`-1`) points are excluded only from "el
  coeficiente de silueta y … Calinski-Harabász". The code also drops noise
  before computing **Davies-Bouldin** (`mask = labels != -1`). As written, a
  reader would think DB is computed with noise as a cluster.
- **Fixes:** Add Davies-Bouldin to the exclusion sentence at `:231`.

### C3 — Clusterer hyperparameter table lists knobs the code ignores
- **Severity:** minor · **Urgency:** low.
- **Location:** `tools.tex:225–226` (`tab:clustering-algos`). Code:
  `src/src/cluster/hdbscan.py`, `src/src/cluster/optics.py`.
- **Problem:** The table gives HDBSCAN's key params as "`min_cluster_size`,
  `min_samples`" but `hdbscan.py` reads `min_cluster_size` and
  `max_cluster_size` (never `min_samples`). It gives OPTICS as "`min_samples`,
  `xi`" but `optics.py` reads `min_samples`, `max_eps`, `metric` (never `xi`).
  Setting the listed knobs in a config would have no effect.
- **Fixes:** Align the table with the wired parameters (HDBSCAN:
  `min_cluster_size`, `max_cluster_size`; OPTICS: `min_samples`, `max_eps`,
  `metric`), or wire the documented ones.

### C4 — `silhouette_macro` computed but undocumented
- **Severity:** minor · **Urgency:** low.
- **Location:** `tools.tex:139` (self-flagged `% TODO`). Code:
  `clustering_metrics.py:24,68–71`; output key `clustering_quality_silhouette_macro`
  in `models.py:173,241`.
- **Problem:** A per-cluster-averaged ("macro") silhouette is computed and
  written to every results record, but the metrics section documents only the
  micro silhouette. `experiments.md`'s metrics table also omits it.
- **Fixes:** Either add the macro silhouette to §Métricas (and the
  `experiments.md` table) or stop emitting it.

### C5 — Isomap incorrectly listed as reusing the random seed
- **Severity:** minor · **Urgency:** low.
- **Location:** `methods.tex:396` (§Reproducibilidad). Code:
  `src/src/reduction/isomap.py` (no `random_state`), `runner.py:744`
  (`IsomapReduction(**params)` — seed never passed; `is_deterministic = True`).
- **Problem:** §Reproducibilidad lists "K-Means, GMM, UMAP, **Isomap** … reutiliza
  la misma semilla". Isomap is deterministic and accepts no seed. This also
  contradicts `methods.tex:147`, which (correctly) names only UMAP as the
  stochastic reduction.
- **Fixes:** Remove Isomap from the seeded-component list at `:396`.

### C6 — Deterministic `K=1` short-circuit not documented
- **Severity:** minor · **Urgency:** low.
- **Location:** `methods.tex:144–167` (§Instrumentación y cronometría). Code:
  `runner.py:419–434`.
- **Problem:** When both the reduction and the clustering are deterministic,
  the runner forces `K=1` (identical repeats waste no work). The thesis implies
  every quality run executes `K` repeats. Consequence: rows like §2's
  `identity`/`pca` + HDBSCAN produce a single sample and **no error bars**,
  unlike the UMAP-based rows — worth a sentence so the missing std isn't read
  as an omission.
- **Fixes:** Add a note that deterministic reduction+clustering combinations
  collapse to `K=1`.

### C7 — Heads said to be in `tab:embedding-models`, but the table omits them
- **Severity:** minor · **Urgency:** low.
- **Location:** `tools.tex:324–325` (§Cabezas) vs `tab:embedding-models`
  (`:181–192`).
- **Problem:** The heads section says the four fine-tuned heads "se integran
  como modelos de embedding adicionales en el catálogo de la
  tabla `\ref{tab:embedding-models}`", but that table lists only the seven base
  families (no head rows). The code does expose them as four extra
  `EmbeddingModelNames` entries.
- **Fixes:** Add four head rows to the table, or rephrase to "se añaden al
  catálogo de modelos" without pointing at a table that doesn't list them.

### C8 — Retrieval quality leans on a weak proxy; `ann_recall_at_k` unmentioned
- **Severity:** minor · **Urgency:** low.
- **Location:** `methods.tex:207–210` (§Protocolo de búsqueda por similitud).
  Code: `clustering_metrics.py:270` (`calculate_ann_recall`), `models.py:428`
  (`ann_recall_at_k`).
- **Problem:** The thesis presents `avg_neighbor_distance` as the metric that
  contrasts backends "en términos … de calidad de recuperación". The code's
  actual ANN-accuracy metric, `ann_recall_at_k` (recall vs exact brute-force
  kNN — the accuracy axis the §9 HNSW sweep depends on, per `experiments.md`
  and `CLAUDE.md`), is not mentioned in the methodology.
- **Fixes:** Document `ann_recall_at_k` as the recall measure for ANN backends;
  keep `avg_neighbor_distance` as the secondary proxy it is.

---

## D. Experiment-catalogue / infos layer

### D1 — Cuenca set size disagrees (1207 vs 1106)
- **Severity:** medium · **Urgency:** medium — a wrong dataset count in the
  catalogue the thesis draws on.
- **Location:** `infos/experiments.md:54`.
- **Problem:** experiments.md says the crops come from "the separate 265-image
  Salamanca and **1207-image Cuenca** annotation sets". `dataset.md:13` and the
  thesis (`tools.tex:23`) put the Cuenca/StopGrafiti corpus at **1106** images.
  (The Salamanca 265 figure is consistent.)
- **Fixes:** Change `1207` → `1106`, or clarify if 1207 is a different
  (super)set count.

### D2 — RSS/VRAM aggregation described as K-1, but code keeps iter 0
- **Severity:** minor · **Urgency:** low.
- **Location:** `infos/experiments.md:43–44`. Code: `models.py:78–140`
  (`StageMetricsAgg.from_samples`, esp. comment `:86–91`).
- **Problem:** experiments.md says "the remaining K-1 samples produce mean ±
  std for wall time, CPU, **RSS, and VRAM**". The code drops iter 0 only for
  cost fields (wall, CPU, CPU%); **memory** fields (RSS, peak RSS delta, VRAM)
  intentionally keep all K samples, because allocator caching makes later
  per-iter deltas under-report the true peak.
- **Fixes:** Correct experiments.md to say memory fields aggregate all K
  samples (iter-0 drop applies to wall/CPU only).

### D3 — `CLAUDE.md` drift on reuse key and YOLO variants
- **Severity:** minor · **Urgency:** low.
- **Location:** `CLAUDE.md` (Benchmarking System; Model Files).
- **Problem:** CLAUDE.md says DB reuse is keyed on "storage+embedding+limit"
  (omits segmenter), but the reuse hash includes the segmenter
  (`runner.py:246` comment; `experiments.md` §5). It also lists YOLO as
  "(n/s/m)" while the embedding experiments use only `yolon`/`yolom` (`yolos`
  exists in `embeddings.py` but is unused by any config). Here the **thesis**
  (`tools.tex:189`, "YOLO11 (n/m)") and experiments are consistent; CLAUDE.md is
  the outlier.
- **Fixes:** Update CLAUDE.md's reuse-key description to include the segmenter;
  note that `yolos` is implemented but not benchmarked.

---

## E. Spot-checks that passed (no action)

For transparency, the following high-value claims were checked and are
**consistent** across layers:

- **Embedding dimensions** (`tools.tex:181–192` ↔ `embeddings.py:44–101`):
  ResNet50 2048, VGG16 4096, InceptionV3 2048, MobileNetV3 1280, DINOv2
  ViT-S/14 384, CLIP ViT-B/32 512, YOLO11 n/m 256/512. The "256–4096" range
  in `tools.tex:119` is correct.
- **Output normalisation** (`tools.tex:85–90`): CLIP and the projection heads
  L2-normalise (`embeddings.py:194,253`); other backbones do not. Matches.
- **Determinism flags** (`methods.tex:150`): DBSCAN/OPTICS/HDBSCAN/Agglomerative
  carry `is_deterministic=True`; KMeans/GMM/Spectral/AffinityPropagation and
  UMAP do not — matching the stochastic/deterministic split in the text.
- **Auto-`k` for KMeans** (`tools.tex:205`) and **DBSCAN `eps` k-distance**
  (`tools.tex:213–219`) match `kmeans.py` / `dbscan.py` (including the
  `min_samples-1` neighbour and self-exclusion detail).
- **Similarity-search defaults** (`methods.tex:187–192`): `top_k=3`,
  `cos_distance=True`, `top_k+1` retrieval, `sample_n`/`sample_seed` — match
  `SimilaritySearchSpec` (`models.py:12–18`) and `runner.py:345–360`.
- **Dependency versions** (`tools.tex:340–350`) match `src/pyproject.toml`
  (torch 2.10, open-clip 3.3.0, ultralytics 8.4.26, scikit-learn 1.8,
  umap-learn 0.5.12, SQLAlchemy 2.0.49, pgvector 0.4.2; Python 3.14).
- **Bibliography:** every `\cite`/`\citea` key in `doc/doc/*.tex` resolves to an
  entry in `doc/bib.bib`. (`bib.bib` has a few unused entries, incl. a
  `garcia_garcia_graffiti_2024`/`_2025` near-duplicate, but the thesis cites the
  `2025` one — harmless with `\bibliographystyle{plain}`.)
- **Configs** (`infos/configs/01,06,08a,08b,09`): embedding lists, baseline
  UMAP→HDBSCAN params, HNSW `m/ef_construction/ef_search` sweeps and run counts
  match the `experiments.md` descriptions.
