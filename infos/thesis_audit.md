# Thesis (TFM) Section Audit

Audit of the LaTeX thesis in `doc/` against (a) the writing/format guides in
`CLAUDE.md` and `infos/LATEX_GUIDE.md`, and (b) the goals and recommended
structure of the official proposal in `doc/enunciado.md`.

Scope reviewed: `doc/informe.tex` and every written section under `doc/doc/`
(`title_author.tex`, `resumen.tex`, `intro.tex`, `art.tex`, `tools.tex`
[*Materiales y métodos*], `methods.tex` [*Metodología*], `results.tex`,
`conclusion.tex`, `packets_and_such.tex`).

This is an analysis-only report. No `.tex` file was modified.

## Summary

| # | Location | Type | Severity | One-line |
|---|----------|------|----------|----------|
| C1 | `conclusion.tex` | add | Critical | *Conclusiones* section is an empty stub (heading only). |
| C2 | `results.tex` | add | Critical | *Resultados y discusión* is empty (only TODO comments); it is the thesis's core deliverable. |
| H1 | `methods.tex` §Detector | modify | High | Literal `YOLO_IMGS_PLACEHOLDER` / `NSEGS_PLACEHOLDER` will render in the PDF. |
| H2 | `resumen.tex` | modify | High | Resumen (ES) and Abstract (EN) are placeholder sentences. |
| H3 | `tools.tex` §Métricas | change | High | Author eval set stated as 110 crops; the dataset section says 185 (110 is the style-training count). |
| H4 | `intro.tex` §Objetivos | modify | High | Objectives omit the similarity-search / retrieval cost axis, which is the titular topic and a full axis in *Metodología*. |
| M1 | `methods.tex` / `tools.tex` | change | Medium | Second bank is "StopGrafiti" in *Materiales* but "Cuenca" in *Metodología*, never linked. |
| M2 | `tools.tex` / `methods.tex` | change | Medium | Detector training set: "both banks / 447 imgs" vs "Salamanca only" — three statements disagree. |
| M3 | `title_author.tex` | modify | Medium | Back-of-title page keeps template placeholders (Autor 1–4, "Pendiente de Asignar", `<<fecha>>`). |
| M4 | `methods.tex` §Eval. extrínseca | move | Medium | Subsection mixes protocol with results-interpretation that belongs in *Resultados y discusión*. |
| M5 | `tools.tex` / `methods.tex` / `results.tex` | remove | Medium | Multiple `% TODO` / "Ensure this is correct" authoring notes left in the source. |
| L1 | `intro.tex` §Estructura | change | Low | Claims "cinco secciones principales"; the document has six numbered sections. |
| L2 | `intro.tex` §Estructura | modify | Low | Closing sentence ends with a comma instead of a period. |
| L3 | `methods.tex` / `tools.tex` | change | Low | Cross-reference style inconsistent: `sección~\ref{}` vs bare `\ref{}` without `~`. |
| L4 | all sections | change | Low | Label-naming convention inconsistent (`subsec:art-*` vs `subsec:vision`). |
| L5 | `art.tex` §Identificación de autoría | change | Low | "combatir el grafiti vandálico" adopts a stance the proposal asks to avoid. |
| L6 | document-wide | add | Low | Body has no figures/diagrams; `\listoffigures` will be empty and a pipeline diagram would aid replicability. |
| L7 | `tools.tex` §Métricas | add | Low | A macro-silhouette score is computed but not documented (flagged by an in-source TODO). |
| L8 | `doc/doc/` filenames | change | Low | File names invert their content (`tools.tex` = *Materiales y métodos*, `methods.tex` = *Metodología*). |

Counts: **Critical 2, High 4, Medium 5, Low 8 — 19 findings.**

---

## Critical

### C1 — *Conclusiones* is empty
- **Location:** `doc/doc/conclusion.tex`, section `\section{Conclusiones}` (`sec:conclusion`).
- **Type:** add.
- **Problem:** The file contains only the section heading and label; there is no
  text. The proposal (`enunciado.md`, *Estructura*) explicitly requires a
  *Conclusiones* section that "recapitula el valor del trabajo hecho" and may
  "dar alguna orientación sobre posible trabajo futuro". A required section is
  effectively missing.
- **Fix:** Write the conclusions: recap the contribution (systematic
  cost/quality characterization of embedding+reduction+clustering combinations
  on graffiti), summarize the main empirical findings, restate the limitations
  already acknowledged in the intro, and add a *trabajo futuro* paragraph
  (e.g. ANN index comparison, larger labeled set, multi-annotator validation).

### C2 — *Resultados y discusión* is empty
- **Location:** `doc/doc/results.tex`, section `\section{Resultados y discusión}` (`sec:results`).
- **Type:** add.
- **Problem:** The file contains only commented TODO/interpretation notes and no
  rendered content. This is the central deliverable of the thesis: the proposal's
  stated goal is to *experimentally characterize* performance and computational
  cost (with unsupervised metrics plus a supervised evaluation on the labeled
  subset). With this section empty, the document does not yet meet its primary
  objective, and the intro/methodology forward-reference results that do not
  exist.
- **Fix:** Populate with the benchmark outcomes: per-stage timing and scaling
  curves vs. `N` (the cost axis), intrinsic-metric comparison across
  configurations, extrinsic ARI/NMI/pairwise-F1 on the style and author eval
  sets, the similarity-search latency comparison (linear SQLite vs. HNSW), and
  the accompanying discussion. The interpretive notes currently parked in the
  comments of `results.tex` and at the end of `methods.tex` (see M4) are the
  seed for the discussion subsection.

---

## High

### H1 — Unfilled numeric placeholders render in the PDF
- **Location:** `doc/doc/methods.tex`, §`Detector de grafiti` (`subsubsec:entrenamiento-detector`), lines ~334–335.
- **Type:** modify.
- **Problem:** `$\mathtt{YOLO\_IMGS\_PLACEHOLDER}$` and
  `$\mathtt{NSEGS\_PLACEHOLDER}$` are typeset as math and will appear verbatim in
  the compiled PDF (unlike `% TODO` comments, which are dropped). A guarded
  `% TODO: Check this once the detector is re-trained` sits right above them.
- **Fix:** Replace both with the real figures (number of annotated Salamanca/both-bank
  photos and total annotated boxes) once the detector is retrained. Reconcile
  with the dataset count of 447 fully annotated images (see M2).

### H2 — Resumen and Abstract are placeholders
- **Location:** `doc/doc/resumen.tex`, lines 17 and 25.
- **Type:** modify.
- **Problem:** The body reads "Se incluye el resumen en castellano." and "Se
  incluye el resumen en inglés." — both are placeholders. A TFM/technical report
  requires a real bilingual abstract.
- **Fix:** Write the Spanish *Resumen* and English *Abstract* (problem,
  modular pipeline, combinatorial benchmark, cost-vs-quality characterization,
  main results), and add keywords/palabras clave if the department template
  expects them.

### H3 — Author evaluation-set size contradicts the dataset section
- **Location:** `doc/doc/tools.tex`, §`Métricas de evaluación` (`subsubsec:metricas`), lines ~155–157.
- **Type:** change.
- **Problem:** This paragraph says extrinsic metrics are computed "sobre los
  $294$ recortes ... etiquetados por estilo o los $110$ etiquetados por
  autoría". But `subsec:dataset` (same file, line 46) states the author
  evaluation set is **185** Salamanca crops, and **110** is the count of
  *Salamanca style-training* crops (`tools.tex` line 41; confirmed by
  `methods.tex` line 288, "$110$ de los $385$"). The number 110 is conflated.
- **Fix:** Change "los $110$ etiquetados por autoría" to "los $185$ etiquetados
  por autoría" so the conceptual framework matches the dataset description and
  the methodology.

### H4 — Objectives omit the similarity-search / retrieval cost axis
- **Location:** `doc/doc/intro.tex`, §`Objetivos` (`subsec:intro-objetivos`), the *Objetivo general* and *Objetivos específicos*.
- **Type:** modify.
- **Problem:** The proposal title is "*algoritmos de similitud* en bancos de
  imágenes de grafiti" and its description centers on *similarity* computation
  and cost vs. bank size. Yet the general objective enumerates only
  "segmentación, generación de embeddings, reducción de dimensionalidad y
  clustering", and no specific objective names similarity search / ANN
  retrieval — even though `methods.tex` devotes a whole axis to it
  (`subsubsec:similarity-search`, SQLite linear scan vs. HNSW) and `art.tex`
  has a dedicated related-work subsection (`subsec:art-similarity-search`). The
  stated objectives under-represent the titular topic.
- **Fix:** Add similarity-search/retrieval cost characterization (linear vs.
  ANN/HNSW, latency vs. `N`) to the general objective and as an explicit
  specific objective, so the objectives cover what the title promises and what
  the methodology actually measures.

---

## Medium

### M1 — Second image bank named inconsistently ("StopGrafiti" vs "Cuenca")
- **Location:** `doc/doc/tools.tex` §`Conjunto de datos` vs `doc/doc/methods.tex` §`Evaluación extrínseca` (lines 258, 274) and §Detector.
- **Type:** change.
- **Problem:** *Materiales y métodos* introduces the auxiliary bank only as
  "StopGrafiti" and never states which city it is from. *Metodología* then
  refers to "los $169$ recortes de **Cuenca**" and "Salamanca y **Cuenca**" as
  if the city had been established. A reader cannot tell that StopGrafiti =
  Cuenca; the geographic-separation argument (used to justify the author
  train/eval split) hinges on this unstated link.
- **Fix:** State once in `subsec:dataset` that the StopGrafiti bank corresponds
  to Cuenca (or whichever city), then use a single name consistently across both
  sections.

### M2 — Detector training set described three different ways
- **Location:** `doc/doc/tools.tex` §`Conjunto de datos` (lines 59–61) and §`Segmentador YOLO` (line 297); `doc/doc/methods.tex` §`Detector de grafiti` (lines 333–335).
- **Type:** change.
- **Problem:** The dataset section says the detector is fine-tuned on "imágenes
  de ambos grupos ... un total de $447$ imágenes". The segmenter subsection says
  it "se ajusta sobre el subconjunto anotado de **Salamanca**". The methodology
  says "fotografías de **Salamanca**" (with a placeholder count). These three
  statements disagree on whether the detector training set is both banks or
  Salamanca-only.
- **Fix:** Pick the correct composition (likely both banks, 447 images) and make
  all three references agree; resolve the H1 placeholders against that number.

### M3 — Title back-page keeps template placeholders
- **Location:** `doc/doc/title_author.tex`, lines 44–78.
- **Type:** modify.
- **Problem:** The reverse of the title page still carries the IT-report
  template skeleton: "Revisado por: Dr. - Pendiente de Asignar -" (×2),
  "Aprobado en el Consejo de Departamento de `<<fecha>>`", and four full
  "Autor 1 … Autor 4 / 1ª Línea AutorN" placeholder blocks, despite this being a
  single-author TFM with the real author/tutors already on the title page.
- **Fix:** Fill the reviewer/approval fields (or mark them clearly as draft) and
  reduce the author-info block to the single real author, removing Autor 2–4.

### M4 — Interpretation parked inside the methodology should move to results
- **Location:** `doc/doc/methods.tex`, §`Evaluación extrínseca sobre subconjunto etiquetado` (`subsubsec:eval-extrinseca`), esp. lines ~279–293.
- **Type:** move.
- **Problem:** This is a methodology subsection but it carries result-interpretation
  material — the domain-shift caveat "deberá tenerse en cuenta al interpretar
  los resultados", the imbalance discussion, and the photographic-overlap
  caveat, all forward-referencing `\ref{sec:results}`. The `results.tex` header
  comment even says interpretation was "reubicado desde la metodología", so a
  migration is half-done.
- **Fix:** Keep only the protocol (what is measured, on which disjoint sets, why
  the split is asymmetric) in `methods.tex`; move the "how to read these
  numbers" caveats into the discussion of `results.tex` (C2).

### M5 — Authoring TODO/notes left in the source
- **Location:** `doc/doc/methods.tex` (lines 112, 332, 354), `doc/doc/tools.tex` (lines 139, 239, 257, 330), `doc/doc/results.tex` (line 5).
- **Type:** remove.
- **Problem:** Several `% TODO`, `% - Ensure this is correct`, and "continue
  checking" notes remain. They don't render, but they flag unverified content
  (e.g. "continue checking" sits exactly where the H1 placeholders begin) and
  should not survive into the submitted draft.
- **Fix:** Resolve each note (verify the content, fill placeholders) and delete
  the comments, or convert genuine open questions into tracked tasks outside the
  manuscript.

---

## Low

### L1 — "cinco secciones principales" miscounts the document
- **Location:** `doc/doc/intro.tex`, §`Estructura del documento`, line 110.
- **Type:** change.
- **Problem:** The text says the document "se organiza en cinco secciones
  principales", but there are six numbered `\section`s: Introducción, Estado de
  la cuestión, Materiales y métodos, Metodología, Resultados y discusión,
  Conclusiones. At best ambiguous (if the intro is excluded), at worst wrong.
- **Fix:** Say "seis secciones" or rephrase to "tras esta introducción, el
  documento se organiza en cinco secciones".

### L2 — Sentence ends with a comma
- **Location:** `doc/doc/intro.tex`, §`Estructura del documento`, line 117 (`...y~\ref{sec:conclusion},`).
- **Type:** modify.
- **Problem:** The structure paragraph — and the section — ends with a comma
  instead of a period.
- **Fix:** Replace the trailing comma with a period.

### L3 — Inconsistent cross-reference style
- **Location:** `doc/doc/methods.tex` and `doc/doc/tools.tex` (many `\ref`s) vs `intro.tex`/`art.tex`.
- **Type:** change.
- **Problem:** `intro.tex`/`art.tex` write "sección~\ref{...}" with a
  non-breaking space, while `methods.tex`/`tools.tex` frequently write bare
  "en \ref{subsec:configuraciones}" with no leading word and no `~`. The
  LATEX_GUIDE permits bare `\ref`, but the mixed style reads inconsistently and
  risks a reference number wrapping to a new line.
- **Fix:** Standardize: precede `\ref` with the appropriate word and a `~`
  (e.g. "la sección~\ref{...}") throughout.

### L4 — Inconsistent label-naming convention
- **Location:** all section files.
- **Type:** change.
- **Problem:** Some labels are section-prefixed (`subsec:intro-contexto`,
  `subsec:art-grafiti`) while others are bare (`subsec:vision`,
  `subsec:pipeline`, `subsec:experimental`). Harder to maintain and to grep.
- **Fix:** Adopt one convention (e.g. always `subsec:<section>-<topic>`).

### L5 — Non-neutral phrasing contradicts the proposal's neutrality clause
- **Location:** `doc/doc/art.tex`, §`Identificación de autoría a partir de firmas y escritura`, lines ~91–93.
- **Type:** change.
- **Problem:** "...planteada por el enunciado como herramienta para **combatir el
  grafiti vandálico**". The proposal's *Otros aspectos a tener en cuenta* asks
  to treat graffiti as a neutral visual phenomenon "sin prejuicios" and avoid
  opinion on its legality/morality (cf. commit "no opinamos"). "Combatir … 
  vandálico" editorializes.
- **Fix:** Rephrase neutrally, e.g. "...la identificación de autoría a partir de
  firmas, una de las aplicaciones motivadoras del enunciado, ...".

### L6 — No figures/diagrams in the body
- **Location:** document-wide (`informe.tex` loads `\listoffigures`).
- **Type:** add.
- **Problem:** Outside the title page there are no `\includegraphics`, so
  `\listoffigures` renders empty and the "replicable methodology" goal lacks a
  visual pipeline overview. Dataset/segmentation examples would also help.
- **Fix:** Add at least a pipeline-overview diagram (ingest → segment → embed →
  store → reduce → cluster) and a couple of dataset/detection example figures;
  the LATEX_GUIDE §Figures shows the expected pattern.

### L7 — Macro-silhouette metric computed but undocumented
- **Location:** `doc/doc/tools.tex`, §`Métricas de evaluación` (`subsubsec:metricas`), line 139 TODO.
- **Type:** add.
- **Problem:** An in-source TODO notes "a macro-silhouette score is also
  calculated"; if the runner reports it, the metrics list is incomplete.
- **Fix:** Either document the macro-silhouette alongside the other intrinsic
  metrics, or confirm it is not reported and drop the note.

### L8 — Section files named opposite to their content
- **Location:** `doc/doc/tools.tex` and `doc/doc/methods.tex`.
- **Type:** change.
- **Problem:** `tools.tex` contains `\section{Materiales y métodos}` while
  `methods.tex` contains `\section{Metodología}`. The filenames invert the
  content, which is confusing for maintenance (and for matching the proposal's
  section names).
- **Fix:** Rename the files to match their sections (e.g. `materiales.tex`,
  `metodologia.tex`) and update the `\include`s in `informe.tex` — a cosmetic,
  maintainability-only change.
