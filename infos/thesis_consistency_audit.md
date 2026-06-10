# Thesis ↔ Experiments ↔ Code Consistency Audit

This audit cross-checks three layers of the project:

1. **Thesis** — LaTeX under `doc/`, excluding the *Results* and *Conclusions*
   sections, which were marked as stubs and not audited here.
2. **Experiments catalogue** — `infos/experiments.md` and the JSON configs in
   `infos/configs/`.
3. **Codebase** — `src/`, in particular the runner, the component
   implementations and the training scripts.

The audit is purely informational. No source, config or document file was
modified.

Every finding lists a **location** (`file:line` where available), a
**severity** (major / minor), an **urgency** hint, a one-line problem
statement and one or more **possible fixes**.

## Summary table

| # | Location | Sev. | Urgency | Problem |
|---|---|---|---|---|
| 1 | `doc/doc/tools.tex:168–194` | major | before submission | Section 3.6 "Búsqueda por similitud" only describes two backends (SQLite, PostgreSQL+HNSW), missing the PostgreSQL exact path that §4.4.4, methods.tex, the code and experiments §6 actually use. |
| 2 | `doc/doc/tools.tex:168` | major | before submission | "Búsqueda por similitud" is set as `\subsection` (line 168), so the three subsections that follow (Agrupamiento, Reducción, Métricas) become its children, which is conceptually wrong. |
| 3 | `doc/doc/methods.tex:430–433` | major | before submission | Claim about style head early stopping ("paciencia 15 sobre la pérdida de entrenamiento, se selecciona la cabeza que minimiza dicha pérdida") does not match `src/src/train/trainer.py` (early stopping only fires with a val set, and only on val metrics — training loss is not a selectable criterion). |
| 4 | `doc/doc/tools.tex:124` ↔ `infos/experiments.md:417`, `exp/exp08b/report.md:21–22` | major | before submission | Author training corpus size disagrees: thesis (and `infos/dataset.md`) say 321 Cuenca crops; experiments.md and the §8b report say 169. |
| 5 | `infos/experiments.md:200–216` ↔ `infos/configs/03_clustering.json` | major | when re-running §3 | §3 narrative ("9 clusterers, 27 runs") undercounts the configured grid (14 clusterers, 42 runs — 3 GMM variants and 4 DBSCAN variants, not 1 each). |
| 6 | `infos/experiments.md:460–471` ↔ `infos/configs/08a_supervised_style.json` | major | when re-running §8a | §8a narrative ("6 clusterers, 96 runs") undercounts the configured grid (7 clusterers — 4 `min_cluster_size` values, not 3 — and 112 runs). |
| 7 | `doc/doc/methods.tex:255–261` | minor | before submission | The O(n²) / O(n log n) claim for the similarity-search stage only holds when `sample_n` is `null` (§7); §6 and §9 use a fixed sample (100 / 200) so the stage time grows linearly in n there. |
| 8 | `doc/doc/methods.tex:438–442` | minor | low | The reproducibility paragraph lists "KMeans, GMM, UMAP" as the stochastic estimators that receive a fixed seed but omits Spectral and Affinity Propagation, which also accept (and use) `random_state` via the shared kwargs interface. |
| 9 | `doc/doc/tools.tex:74–79` | minor | low | Thesis subset table says "Estilo (eval.): Banco completo" — the eval crops come from the 6416-image evaluation corpus specifically, not from the full Salamanca corpus; "Banco completo" reads as 6416 + 265, which would overlap with detector / head training. |
| 10 | `doc/doc/tools.tex:138–139` | minor | low | `\subsubsection{…}` is labelled `subsec:fundamentos` — the label prefix does not match the level. (Pure label hygiene.) |
| 11 | `infos/experiments.md:380–386` | minor | low | "7 single-shot ingests" — §7 has 6 limits, so 6 ingests; off-by-one. |
| 12 | `infos/experiments.md:360–365` | minor | low | "n ≤ 6416 is too narrow a span (~1.8 log decades)" — after the n=100 trim the range is 250…6416, ≈1.41 decades. The conclusion uses 1.4 correctly. |
| 13 | `infos/experiments.md:410` | minor | low | References `src/train/style_trainer.py`, which does not exist. The actual file is `src/src/train/style_head.py` (entry point) → `src/src/train/trainer.py` (loop). |
| 14 | `infos/experiments.md:4–6`, `exp/report.md` | minor | low | Both reference `doc/enunciado.md`, which is not present in the repo. |
| 15 | `infos/metrics.md` | minor | low | The audit prompt mentions `infos/metrics.md`, but the file does not exist. Metrics live in the "Metrics" section of `infos/experiments.md`. |
| 16 | `src/src/cluster/optics.py:35` ↔ `infos/clustering.md:150`, `infos/best_cluster.md:27` | minor | low | Code default for OPTICS `metric` is `"euclidean"` (changed for cross-clusterer consistency); both info docs still claim it is `"cosine"`. Thesis does not specify a default, so the thesis is unaffected. |
| 17 | `src/src/train/author_head.py:34` | minor | low | `--loss` default is `"supcon"`, but the entry point descriptions ("…using batch-hard triplet loss") and `doc/doc/methods.tex:408` assume triplet is the default for author heads. If author heads were ever re-run without `--loss triplet`, the produced weights would not match what the thesis describes. |
| 18 | `doc/doc/art.tex:79–84` | minor | low | "familias clásicas de agrupamiento (por centroides como K-Means, por densidad como DBSCAN y HDBSCAN, jerárquicos o utilizando modelos probabilísticos)" — omits OPTICS, Spectral, GMM, Affinity Propagation and Agglomerative even though they are all implemented and benchmarked. |
| 19 | `doc/doc/tools.tex:316–328` (table `tab:embedding-models`) | minor | low | The embeddings table lists 7 rows (collapsing YOLO11 n/m into one) but not the 4 fine-tuned heads even though they are part of the embedding catalogue (introduced separately in §4.5.2). The table caption ("Modelos de extracción de características integrados") reads as the full list. |
| 20 | `doc/doc/methods.tex:382` | minor | medium | "650 cajas anotadas" over the 447 detector-training photos is asserted with "aproximadamente"; no source file in `src/` or `infos/` corroborates the count. Either drop the figure or add a citation. |
| 21 | `infos/segmentation.md:21–26` | minor | low | YOLO segmenter table omits the configured parameters `touch_merge_gap`, `max_merged_area`, `max_boxes_per_image`, `allow_full_image_fallback`, `batch_size` (all exposed by `YoloSegmenter`). Thesis (`tools.tex:418–429`) describes them, so segmentation.md trails the thesis here. |

Severity: *major* = changes the meaning of a claim or a numeric result;
*minor* = stale references, off-by-one counts, wording, label hygiene.

---

## Findings — thesis layer

### 1. tools.tex §3.6 silently drops the PostgreSQL-exact backend
**Location:** `doc/doc/tools.tex:168–194` (subsec. *Búsqueda por similitud*).
**Severity:** major. **Urgency:** before submission — readers will reach §6 of
the results expecting two backends and see three.

The thesis frames similarity search as "el sistema implementa dos enfoques":
exact over SQLite vs. HNSW over PostgreSQL. The runner (`src/src/evaluation/runner.py:340-411`),
the storage docs (`infos/storage.md`), the thesis itself further down
(`tools.tex:382–400`) and the §6 experiment all use **three** backends:
SQLite, PostgreSQL+exact (pgvector sequential scan), and PostgreSQL+HNSW.
The two-backend framing here clashes with the rest of the document.

**Possible fixes**
- Rewrite the paragraph to list the three options explicitly.
- Or keep two options but disambiguate that "PostgreSQL" covers both the
  exact pgvector kernel and the HNSW index, and state that the comparison
  in the results section uses all three.

### 2. "Búsqueda por similitud" is at the wrong sectioning level
**Location:** `doc/doc/tools.tex:168`.
**Severity:** major. **Urgency:** before submission — affects the ToC.

`\subsection{Búsqueda por similitud}` is declared at the same depth as
*Marco conceptual* (line 135) and *Componentes implementados* (line 304).
Because `\subsubsection{Agrupamiento no supervisado}` (195),
`\subsubsection{Reducción de dimensionalidad}` (237) and
`\subsubsection{Métricas de evaluación}` (266) sit directly under it, those
three subsubsections are rendered as children of *Búsqueda por similitud*,
which is conceptually inverted: clustering, dimensionality reduction and
metrics are part of the conceptual framework, not of similarity search.

**Possible fixes**
- Demote *Búsqueda por similitud* to `\subsubsection` so it stays under
  *Marco conceptual*, and bring it before *Agrupamiento no supervisado*.
- Or restructure so that the *Marco conceptual* subsection wraps
  Representación, Búsqueda, Agrupamiento, Reducción and Métricas as five
  sibling `\subsubsection`s.

### 3. Style-head early-stopping claim does not match the code
**Location:** `doc/doc/methods.tex:430–433` (subsec.
*Entrenamiento de modelos auxiliares → Cabezas de proyección*).
**Severity:** major. **Urgency:** before submission — it is a factual claim
about how the released heads were selected.

The thesis says: "Se aplica parada temprana con paciencia de 15 épocas
sobre la pérdida de entrenamiento, y la cabeza seleccionada para los
benchmarks es la que minimiza dicha pérdida." In
`src/src/train/trainer.py`:

- Early stopping is only evaluated when `val_dataset_dir` is provided
  (line 142). Without a val set the loop runs all 60 epochs and the
  last-epoch weights are saved (line 188).
- When a val set is provided, the criterion is `val_metric`, whose allowed
  values are `silhouette / accuracy_1nn / ari / nmi`
  (`src/src/train/style_head.py:65–67`) — *training loss is not a selectable
  criterion at all*.

So either no early stopping ran, or it ran on a val metric, but not on
training loss.

**Possible fixes**
- Restate the procedure: "se conservan los pesos de la última época"
  (if heads were trained without a val set, matching the author-head
  behaviour described in lines 411–412).
- Or, if a val set was used: name it, and replace "pérdida de
  entrenamiento" with the actual val metric (e.g. `accuracy_1nn`).

### 4. Author training corpus size: 321 vs 169
**Location:** `doc/doc/tools.tex:124` (table `tab:subconjuntos`), reinforced by
`infos/dataset.md:17`. Contradicted by
`infos/experiments.md:412–418` and `exp/exp08b/report.md:20–22`.
**Severity:** major. **Urgency:** before submission — it is reported in the
materials chapter and again in the results discussion.

Tools.tex and dataset.md both say the author heads were trained on 321
Cuenca crops. `infos/experiments.md` and the §8b report — written off the
same training run — say 169. Both numbers cannot be right.

**Possible fixes**
- Compute the count from the actual training folder and pick the correct
  value, then propagate it to the table, the conclusion ("321 recortes
  *exclusivamente* de Cuenca" in `results.tex:624` if it has the same
  number), `infos/dataset.md` and `infos/experiments.md`.
- If 321 is the raw crop count and 169 is after some filter (e.g. dropping
  singleton authors), document the filter explicitly in `infos/dataset.md`
  and use the filtered number in `infos/experiments.md` / §8b.

### 7. O(n²) / O(n log n) claim only holds for §7
**Location:** `doc/doc/methods.tex:255–261`.
**Severity:** minor. **Urgency:** low — but it would be misread if the
reader cross-checks with §6/§9 cost numbers.

The thesis says the stage cost grows as O(n²) for SQLite and O(n log n) for
HNSW. That is per-query O(n)/O(log n) multiplied by *n queries*; it only
holds when `sample_n` scales with corpus size. §7 (`07_scalability.json`)
uses `sample_n: null` (every embedding is queried), so the n² law shows up
empirically. §6 fixes `sample_n: 100` and §9 fixes `sample_n: 200` — there
the stage time is O(sample_n · n), i.e. linear in n.

**Possible fixes**
- Mention that the asymptotic discussion assumes a query batch
  proportional to n (the §7 setting) and that §6/§9 fix the query budget,
  which decouples the search-stage cost from n.

### 8. Reproducibility paragraph misses Spectral and Affinity Propagation
**Location:** `doc/doc/methods.tex:438–440`.
**Severity:** minor.

"todo estimador estocástico se instancia con `random_state` fijo (K-Means,
GMM, UMAP …)" — the list misses two estimators that the system implements
and seeds: `SpectralClusterer` (`src/src/cluster/spectral.py:30`) and
`AffinityPropagationClusterer` (`src/src/cluster/affinity_propagation.py:30`).
They receive `random_state` through the same kwargs path the runner uses
(`runner.py:506`).

**Possible fix**
- Expand the parenthesis to "K-Means, GMM, *spectral clustering*, *affinity
  propagation* y UMAP".

### 9. "Banco completo" wording for the style eval split
**Location:** `doc/doc/tools.tex:74–79`, `122` (table row "Estilo (eval.)").
**Severity:** minor.

The body text (line 76) says "el conjunto de evaluación, extraído del banco
principal de Salamanca, es disjunto a nivel de fotografía". The table row
says the provenance is "Banco completo". `infos/dataset.md:9-10` is the
unambiguous source: the 294 style-eval crops are drawn from the 6416-image
**evaluation** corpus only. "Banco completo" could be read as the 6416 + 265
images, which would put the eval crops inside the detector / style training
photos.

**Possible fix**
- Replace "Banco completo" with "Salamanca (eval.)" or similar to mirror
  the unsupervised row.

### 10. Label naming hygiene
**Location:** `doc/doc/tools.tex:138–139, 168–169`.
**Severity:** minor.

- `\subsubsection{Representación vectorial …}` is labelled
  `subsec:fundamentos` (subsec prefix on a subsubsection).
- `\subsection{Búsqueda por similitud}` is labelled
  `subsubsec:tools-similarity-search` (subsubsec prefix on a subsection).

`methods.tex:355` references the latter, so renaming the labels needs a
search-and-replace.

**Possible fix**
- Normalise labels to match the LaTeX level (e.g.
  `\label{subsubsec:fundamentos}` and `\label{subsec:busqueda-similitud}`)
  and update the few references.

### 18. State-of-the-art only names three clustering families
**Location:** `doc/doc/art.tex:79–84`.
**Severity:** minor.

The state-of-the-art paragraph lists "centroides como K-Means, densidad
como DBSCAN y HDBSCAN, jerárquicos o utilizando modelos probabilísticos"
and stops there, but Spectral, OPTICS, Affinity Propagation, GMM and
Agglomerative are all implemented (`src/src/cluster/`) and exercised in
§3 / §7 / §8. The materials chapter (`tools.tex:351–363`) lists all eight,
so the asymmetry shows up between sections of the same thesis.

**Possible fix**
- Add a brief mention of spectral and OPTICS at the end of the paragraph
  (they are the two that are exercised in the cost analysis).

### 19. Embedding catalogue table omits the four fine-tuned heads
**Location:** `doc/doc/tools.tex:317–328` (table `tab:embedding-models`),
contrasted with `tools.tex:483–493`.
**Severity:** minor.

The table claims to enumerate the implemented embedding models and lists
seven rows. The four graffiti heads (DINOv2 ×{style, author}, MobileNetV3
×{style, author}) are introduced separately on lines 483–493, but a reader
looking at the table will not see them — and §1 of the results discusses
twelve extractors, not seven.

**Possible fix**
- Add four rows to the table, or add a footnote in the caption pointing
  to §4.5.2 for the fine-tuned heads (with their 128-d output).

### 20. "≈650 anotated boxes" is asserted without a verifiable source
**Location:** `doc/doc/methods.tex:380–382`.
**Severity:** minor.

`infos/dataset.md` only records the 447 detector photos; the 650 box count
appears nowhere else in the repo. Asserting it as "aproximadamente un total
de 650 cajas anotadas" is fine if it was measured on the annotation files,
but the source is not cited.

**Possible fix**
- Generate the count from the YOLO label files and add the resulting
  number to `infos/dataset.md` for cross-reference.

---

## Findings — experiments catalogue & configs

### 5. §3 catalogue undercounts the clustering grid
**Location:** `infos/experiments.md:194–216`,
`infos/configs/03_clustering.json`.
**Severity:** major. **Urgency:** when re-running §3.

The §3 narrative says: "`hdbscan`, `kmeans` (auto-k), `gmm` (auto-k),
`dbscan`, `optics` (cosine), `agglomerative` ({10, 20}), `spectral` ({10,
20})" → 9 clusterers, "27 runs (9 clusterers × 3 limits)". The config has:

- 1× hdbscan
- 1× kmeans (`max_clusters=30`)
- 3× gmm (`max_clusters` ∈ {30, 100, 300})
- 4× dbscan (auto + `eps` ∈ {0.2, 0.5, 1.0})
- 1× optics
- 2× agglomerative
- 2× spectral

= 14 clusterers → 42 runs.

`results.tex:230` correctly reports "14 configuraciones", so the discrepancy
is internal to the planning docs. Either the config or the catalogue
diverged from the original plan.

**Possible fixes**
- Update the §3 catalogue to enumerate every variant and report the
  correct cost (42 runs, 3 ingests).
- Or trim the config back to the documented 9 variants if the extra GMM /
  DBSCAN sweeps were not intended to be part of §3.

### 6. §8a catalogue undercounts the clustering grid
**Location:** `infos/experiments.md:460–522`,
`infos/configs/08a_supervised_style.json:24–32`.
**Severity:** major. **Urgency:** when re-running §8a.

§8a's text says "**Varies (§8a clustering, 6 settings).** kmeans,
agglomerative, spectral all with n_clusters=4 (oracle k); hdbscan swept
over min_cluster_size ∈ {5, 10, 20}". The config has the additional
`hdbscan, min_cluster_size=50`, so the grid has 7 settings → 8 × 2 × 7 =
112 runs, not the 96 the catalogue advertises.

(`exp/exp08a` is reported as containing the `mcs=50` follow-up; the
catalogue still describes the original 96-run grid.)

**Possible fixes**
- Add the fourth `min_cluster_size` value to the §8a description and
  recompute the cost (`8 × 2 × 7 = 112` runs).
- Or split the §4-style sensitivity ablation off §8a so the 96-run
  description matches the kept grid.

### 11. Off-by-one ingest count in §7
**Location:** `infos/experiments.md:380–386`.
**Severity:** minor.

"the extra cost is small relative to the 7 single-shot ingests" — §7 has 6
limits (`{250, 500, 1000, 2000, 3500, 6416}`), the embedding DB is reused
across the 5 clusterers within each limit group, so there are 6 ingests.
Same paragraph above (line 384) already says "6 ingests".

**Possible fix**
- Replace "7" with "6".

### 12. "~1.8 log decades" in §7 narrative
**Location:** `infos/experiments.md:360–365`.
**Severity:** minor.

The catalogue says "n ≤ 6416 is too narrow a span (~1.8 log decades)" but
the swept range starts at n=250 (the n=100 point was trimmed). log10(6416/250)
≈ 1.41. `conclusion.tex:60–63` reports the same trim and gives "≈1.4
décadas" correctly.

**Possible fix**
- Replace "~1.8" with "~1.4" in §7.

### 13. Stale `style_trainer.py` reference
**Location:** `infos/experiments.md:410`.
**Severity:** minor.

§8a cites `src/train/style_trainer.py`. The file does not exist; the
correct paths are `src/src/train/style_head.py` (entry point) and
`src/src/train/trainer.py` (training loop, used by both heads).

**Possible fix**
- Replace the citation with `src/src/train/style_head.py` (or
  `src/src/train/trainer.py`, depending on which the reader should land
  on).

### 14. `doc/enunciado.md` is referenced but not present
**Location:** `infos/experiments.md:4`, also `exp/report.md`.
**Severity:** minor.

Both files link to `../doc/enunciado.md`. `doc/` only contains
`informe.tex`, `bib.bib`, `picins.sty`, the `doc/` source folder and
`figures/`. No `enunciado.md`.

**Possible fix**
- Drop the link, point to a different file (e.g. `doc/doc/intro.tex`
  for the objectives), or add `doc/enunciado.md` if it is intended to ship
  with the repo.

### 15. `infos/metrics.md` does not exist
**Location:** mentioned in the audit prompt; no file in `infos/`.
**Severity:** minor.

The metrics description lives inside `infos/experiments.md` under the
"Metrics" section. The standalone `metrics.md` referenced in the audit
prompt is absent.

**Possible fix**
- Extract the "Metrics" section into `infos/metrics.md` (and link to it
  from `infos/experiments.md`) if a standalone file was the intent;
  otherwise no action.

---

## Findings — code layer

### 16. OPTICS default metric drifted from the docs
**Location:** `src/src/cluster/optics.py:35` vs. `infos/clustering.md:150`,
`infos/best_cluster.md:27`.
**Severity:** minor.

`OPTICSClusterer` reads `metric = kwargs.get("metric", "euclidean")` and
the docstring explains the change ("Default Euclidean to match
DBSCAN/HDBSCAN/Agglomerative/KMeans"). `infos/clustering.md` and
`infos/best_cluster.md` still tell readers the default is cosine. The §3
config sets `metric: "cosine"` explicitly so the runs themselves are
unaffected.

**Possible fix**
- Update the two info files to say "Default: `"euclidean"` (override per
  run; the §3 catalogue passes `"cosine"`)" and explain the rationale.

### 17. Author-head trainer default loss disagrees with thesis
**Location:** `src/src/train/author_head.py:34` (default
`--loss "supcon"`) vs. `src/src/train/dino/train_head.py:18`,
`src/src/train/mobilenet/train_head.py` (parser description: "using
batch-hard triplet loss"), vs. `doc/doc/methods.tex:408–411`.
**Severity:** minor.

Methods.tex describes the author heads as trained with batch-hard triplet
(margin 0.3, Adam lr 1e-4, 20 epochs). The CLI parser shared between the
author-head entry points defaults `--loss` to `supcon`. A user who follows
the CLI defaults would train with SupCon, not triplet. The shipped
weights presumably came from a run with `--loss triplet`, but this
contract is fragile.

**Possible fix**
- Flip the default in `author_head.py` to `"triplet"`, matching the
  parser description and methods.tex.
- Or change methods.tex to "se entrenan con la pérdida `supcon`" if the
  shipped author heads are actually SupCon-trained — in which case
  results.tex §8b would also need a wording check.

### 21. YOLO segmenter docs are incomplete
**Location:** `infos/segmentation.md:21–26`.
**Severity:** minor.

The parameter table omits `touch_merge_gap`, `max_merged_area`,
`max_boxes_per_image`, `allow_full_image_fallback`, `batch_size`, all of
which are documented in `tools.tex:418–429` and implemented in
`src/src/embedding/segmenters.py:6–16`.

**Possible fix**
- Sync the parameter table with the constructor signature.

---

## Closing notes

- The thesis sections audited (resumen, intro, art, tools, methods) are
  internally consistent on the architectural model (five swappable stages,
  reuse heuristic, repeat semantics, intrinsic + extrinsic readings of
  noise). Most mismatches are in numeric / structural detail rather than in
  the high-level story.
- The two largest cross-layer asymmetries are #3 (style-head training:
  selection criterion) and #4 (author-training corpus size: 321 vs 169).
  Both touch on how the released heads were obtained, so fixing them also
  affects how §8 results are read.
- `infos/experiments.md` lags both the configs in `infos/configs/` and
  `results.tex` in two places (§3, §8a), which means the catalogue no
  longer documents the experiments that were actually run. Bringing it
  back in sync is the single change that closes the most findings.
