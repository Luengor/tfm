# Experiment Report Audit Findings

**Date:** 2026-05-29
**Auditor:** automated (Claude)
**Audited:** exp01 (§1), exp02 (§2), exp03 (§3), exp04 (§4), exp05 (§5), exp08a (§8a)
**Flagged (report without data):** exp07 (§7) — has `report.md` but no tracked `output/benchmark_*.json`
**Skipped (WIP — no report.md):** exp06 (§6), exp08b (§8b), exp09 (§9)

---

## exp01 — §1 Embedding Model Comparison

**Verdict: minor issues**

- **Source file OK.** `exp/exp01/output/benchmark_20260528T231810Z.json` exists; 36 runs, all `success`. Matches the report's "36/36 runs success" claim.
- **Quality table values verified.** All 36 rows (12 embeddings × 3 limits) match the JSON to the reported precision (3 d.p. for sil/noise, 1 d.p. for CH, 2 d.p. for DB/csCV). No numeric errors found.
- **Cost table values verified.** All 10 rows (n=6416 only) match the JSON. No numeric errors found.
- **Bold markers (best silhouette) correct.** n=250: resnet50 (0.258) ✓; n=1000: resnet50 (0.272) ✓; n=6416: mobilenet_v3 (0.260) ✓.
- **Italic markers (best CH) WRONG at all three corpus sizes.** The table legend says "*italic* = best CH" but the italicised values are not the actual maxima:
  - n=250: yolom *27.4* is italicised, but **yolon has CH=30.4** (higher). `report.md:63` vs `:67`.
  - n=1000: yolom *28.9* is italicised, but **yolon has CH=44.5** (higher). `report.md:79` vs `:80`.
  - n=6416: dinov2_graffiti_style_head *33.1* is italicised, but **yolon has CH=51.2** (higher). `report.md:89` vs `:92`.
  
  The italic appears to mark best CH among non-YOLO embeddings, but the legend does not say that. Either fix the legend or move the italics to yolon.
- **Stale deviation note.** `report.md:29–34` says "infos/experiments.md §1 lists ten embeddings and explicitly omits `dinov2_graffiti_author_head`." The current catalogue (`infos/experiments.md:140–146`) says "Twelve embeddings" and lists `dinov2_graffiti_author_head` by name. The note describes a deviation that no longer exists; it (or the catalogue) was updated out of sync. The recommendation to "update the catalogue" is now redundant.
- **Prose consistent with data.** The silhouette↓/CH↑ split narrative on fine-tuned heads is backed by the numbers. The YOLO-worst claim is correct across all metrics and sizes.

## exp02 — §2 Reduction Technique Comparison

**Verdict: OK**

- **Source file OK.** `exp/exp02/output/benchmark_20260528T093147Z.json` exists; 18 runs, all `success`. Matches "18/18".
- **Quality table values verified.** All 18 rows (6 reductions × 3 limits) match the JSON, including the negative silhouette (pca-10@6416 = −0.028) and the high-noise pca-50@6416 row. No numeric errors.
- **Cost table values verified.** All 6 rows (n=6416) match. No errors.
- **Bold/italic markers correct.** Bold on extreme noise values (isomap@250 0.580, pca-50@6416 0.627), bold on negative sil (pca-10@6416 −0.028), italic on best sil per group (isomap@250 0.279, optics equivalents). Consistent with intended usage.
- **Silhouette macro (`silM`) included** in the quality table — a column not present in exp01 despite being available in exp01's JSON. This is a cross-report consistency issue (see Cross-cutting section).
- **Coverage matches catalogue.** 6 reductions × 3 limits = 18, matching `infos/experiments.md` §2.
- **Prose consistent.** The UMAP-10 > UMAP-50 > everything-else verdict follows from the data.

## exp03 — §3 Clustering Algorithm Comparison

**Verdict: minor issues**

- **Source file OK.** `exp/exp03/output/benchmark_20260527T155755Z.json` exists; 42 runs, all `success`. Matches "42/42".
- **Quality table values verified.** All 42 rows (14 configs × 3 limits) match the JSON. No numeric errors.
- **Cost table values verified.** All 12 rows (n=6416, excluding dbscan·0.2 and dbscan·1.0 which are omitted from the cost table) match. No errors.
- **Undocumented positive deviation from catalogue.** The catalogue (`infos/experiments.md` §3) specifies **9 clusterers → 27 runs**. The experiment ran **14 configs → 42 runs**. The extras are:
  - GMM expanded from 1 config ("auto-k") to 3 (`max_clusters ∈ {30, 100, 300}`)
  - DBSCAN expanded from 1 config ("dbscan") to 4 (`auto-eps` + `eps ∈ {0.2, 0.5, 1.0}`)
  
  The report documents the 14 configs in its Setup section (`report.md:27`) but does not flag this as a deviation from the catalogue, unlike exp01 and exp05 which use explicit `> **Deviation...**` blocks. Should be noted for completeness.
- **Prose consistent.** HDBSCAN-wins-on-joint-criterion conclusion tracks the data.

## exp04 — §4 HDBSCAN min_cluster_size Tuning

**Verdict: OK**

- **Source file OK.** `exp/exp04/output/benchmark_20260527T133844Z.json` exists; 18 runs, all `success`. Matches "18/18".
- **Quality table values verified.** All 18 rows match. The `mcs50@250` clusters_per_repeat `[0, 0, 2]` matches the JSON (`report.md:66`). The `mcs5@6416` clusters_per_repeat `[549, 537, 563]` matches (`report.md:152`). No numeric errors.
- **Cost table values verified.** All 6 rows (n=6416) match. No errors.
- **All-noise cells correctly handled.** `mcs100@250`, `mcs200@250`, `mcs200@1000` are documented as 0 clusters / noise=1.0 with em-dashes for missing metrics, consistent with the JSON (which has `null` for those quality fields).
- **Coverage matches catalogue.** 6 `min_cluster_size` values × 3 limits = 18, matching §4.
- **Prose consistent.** Non-monotonic noise claim is correct (noise at n=6416: 0.199 → 0.277 → 0.228 → 0.026 → 0.008 → 0.011). mcs=5-wins conclusion follows from joint criterion.

## exp05 — §5 Segmenter Impact

**Verdict: minor issues**

- **Source file OK.** `exp/exp05/output/benchmark_20260529T103834Z.json` exists; 12 runs, all `success`. Matches "12/12".
- **Quality table values verified.** All 12 rows match the JSON. Crop counts, cluster counts, silhouette, CH, DB, csCV all correct. No numeric errors.
- **Cost table values verified.** All 4 rows (n=6416) match. No errors.
- **Phantom deviation note.** `report.md:29–33` says: "infos/experiments.md §5 specifies *five* segmenters (`yolo11{n,s,m}` + fine-tuned + identity = 15 runs). The `yolo11n` row is **absent**." However, the current catalogue (`infos/experiments.md:256–264`) already specifies **four** segmenters and explicitly states "The `yolo11n` row was also dropped." The catalogue cost line confirms "12 runs (4 segmenters × 3 limits)." The experiment matches the catalogue exactly — the deviation note describes a discrepancy that does not exist and the recommendation to "add yolo11n for completeness, or trim the brief" is unnecessary.
- **Prose consistent.** Identity-wins-at-n=6416 conclusion follows from the data.

## exp07 — §7 Pipeline Scalability Sweep

**Verdict: report without data — FLAGGED**

- **Report exists:** `exp/exp07/report.md` (254 lines) cites `exp/exp07/output/benchmark_20260528T160305Z.json` and claims "42/42 runs success."
- **No tracked output data.** The `exp/exp07/output/` directory does not exist. The cited JSON file is absent from the git tree. Only `config.json` and `report.md` are tracked.
- **Cannot verify any quantitative claim.** The report contains wall-time tables, complexity slopes, memory figures, and cluster quality data — none of which can be verified against source data. The report is a narrative unsupported by evidence in the repository.
- **Action required:** Either rerun the experiment and track the output JSON, or remove the report until data is available.

## exp08a — §8a Supervised Validation on Style Crops

**Verdict: OK**

- **Source file OK.** `exp/exp08a/output/benchmark_20260529T125933Z.json` exists; 96 runs, all `success`. Matches "96/96".
- **Best-ARI-per-embedding table verified.** All 8 rows match the JSON: dinov2_graffiti_style_head ARI=0.698, mobilenet_v3_graffiti_style_head 0.560, clip_vit_b32 0.368, dinov2_vits14 0.356, mobilenet_v3 0.356, resnet50 0.334, mobilenet_v3_graffiti_author_head 0.267, dinov2_graffiti_author_head 0.234. Winning cells all correct.
- **Similarity search P@5/MAP/MRR table verified.** All 8 embeddings match (e.g. dinov2_graffiti_style_head P@5=0.820, MAP=0.877, MRR=0.895).
- **Detail table (dino-style head) verified.** identity-kmeans4 (ARI 0.698), identity-spectral (0.673), identity-agglomerative (0.630), umap-agglomerative (0.658), umap-kmeans (0.599) — all match JSON values.
- **HDBSCAN detail verified.** identity-hdb-mcs{5,10}: cls=2, confirming k=2 collapse. umap-hdb-mcs{5,10,20}: all cls=3, cpr=[3,3,3], confirming mcs-invariant k=3 partition (`report.md:106–108`).
- **Coverage matches catalogue.** 8 embeddings × 2 reductions × 6 clusterings = 96, matching §8a.
- **Hypothesis dispositions consistent with data.** Style head > backbone confirmed (0.698 vs 0.356), author head damages style (0.234 < 0.356), HDBSCAN does not recover k=4 — all verifiable.

---

## Cross-cutting / consistency

1. **Silhouette macro (`silM`) inconsistently reported.** The metric is present in the JSON for exp01, exp02, exp05, and exp08a (all post-2026-05-28 runs), but absent from exp03 and exp04 JSON (2026-05-27 runs — likely an older pipeline version). Reports exp02 and exp05 include `silM` in their quality tables; exp01 does NOT include it despite having it in the JSON. If `silM` is worth reporting, it should be added to exp01; if not, it can be dropped from exp02/exp05. Either way the house style is currently inconsistent.

2. **Baseline values match across reports.** The overlapping baseline cell (dinov2_graffiti_style_head + UMAP-10 + HDBSCAN mcs=5 + identity + n=6416) reports cls=563, noise=0.199, sil=0.151 in all five reports that include it (exp01–05). This is strong evidence the JSON data is self-consistent and the reports faithfully transcribe it.

3. **Metric abbreviation legend is consistent.** All reports use the same abbreviations (`sil`, `CH`, `DB`, `csCV`, `noise`, `cls`) and the same direction guidance (higher/lower better). exp08a adds `ARI`, `NMI`, `F1` appropriately. No legend conflicts found.

4. **Repeat count convention consistent.** Quality experiments (exp01–05, exp08a) all use K=3. exp07 (unverifiable) claims K=5. Matches the catalogue specification.

5. **Corpus sizes consistent.** All quality experiments use `{250, 1000, 6416}`. exp08a uses the full 294-crop eval set (no limit). No deviations.

6. **Header block house style consistent.** All reports follow the `# §N — Title: Report` / Source / Date / Host / Status format. exp07 is also consistent, but its data is missing.

7. **Deviation notes partially unreliable.** Two of three explicit deviation notes are stale:
   - exp01: claims catalogue has 10 embeddings; catalogue says 12 (no deviation exists).
   - exp05: claims catalogue has 5 segmenters / 15 runs; catalogue says 4 / 12 (no deviation exists).
   - exp03: has an actual positive deviation (14 configs vs catalogue's 9) but does NOT note it.

---

## Recommendations (prioritised by impact)

1. **[HIGH] exp07: rerun the scalability sweep and track the output JSON, or remove the report.** This is the thesis's headline experiment (§7) and the report contains unverifiable quantitative claims — slopes, wall times, memory figures, cluster quality. No finding in the report can be cited in the thesis without the source data. Priority: block the thesis write-up until this is resolved.

2. **[HIGH] Complete the missing experiments.** The catalogue defines 9 experiments (§1–§9). Only 6 have completed data+reports (§1–§5, §8a). Missing:
   - **§6 (storage cost)** — exp06 has config.json only; no output, no report.
   - **§8b (author supervised)** — exp08b has config.json only. Needed to close the cross-task transfer specificity claim from §8a (hypothesis 2 requires the symmetric result).
   - **§9 (HNSW tuning)** — exp09 has config.json only. Needed for the pgvector/HNSW Pareto frontier.

3. **[MEDIUM] Fix exp01 italic CH markers.** All three corpus-size groups mark the wrong value as best CH. The actual best CH is yolon at all three sizes (30.4, 44.5, 51.2). Either move the italic to yolon or add a qualifier to the legend (e.g. "best CH among non-YOLO embeddings"). `report.md:55, 63, 67, 79, 80, 89, 92`.

4. **[MEDIUM] Resolve stale deviation notes.** Two reports contain deviation notes that no longer match the catalogue:
   - exp01 `report.md:29–34`: remove or update the note about "10 in the catalogue"; the catalogue already says 12.
   - exp05 `report.md:29–33`: remove the note about missing yolo11n; the catalogue already dropped it and specifies 4 segmenters / 12 runs, matching the experiment exactly.

5. **[LOW] Add a deviation note to exp03.** The experiment ran 14 clustering configs (42 runs) vs the catalogue's 9 (27 runs). This is a positive deviation (GMM × 3 max_clusters, DBSCAN × 4 eps settings) that adds useful data, but it should be documented in a `> **Deviation...**` block for consistency with the house style, or the catalogue should be updated to match.

6. **[LOW] Harmonise silhouette macro reporting.** Either include `silM` in all reports where the JSON provides it (exp01 is the main gap), or remove it from exp02/exp05 if it's not essential. The metric adds value for the reading-hazard narrative (macro silhouette inverts the ranking in exp02), so including it consistently seems preferable.

7. **[LOW] Update catalogue to match exp03's richer grid.** If the GMM(3) + DBSCAN(4) expansion in exp03 is accepted as the definitive run, the catalogue's §3 should say "14 clustering configs × 3 limits = 42 runs" instead of "9 × 3 = 27".
