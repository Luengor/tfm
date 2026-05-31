# Verification report for `doc/doc/results.tex`

**Scope.** Every verifiable factual/numerical claim in `doc/doc/results.tex`
(lines 1–958) is checked against the corresponding benchmark JSON under
`exp/expNN/output/benchmark_*.json` and against the thesis proposal
(`doc/enunciado.md`) and methodology (`doc/doc/methods.tex`).

**Method.** Numbers were re-read or recomputed directly from the JSONs. For
aggregates (log–log slopes, Pearson correlations), the input runs and the
arithmetic are shown in the Notes column.

**Verdict legend.**
- **OK** — claim matches the data within rounding.
- **MISMATCH** — numerical disagreement that cannot be explained by rounding.
- **UNSUPPORTED** — claim cannot be checked because the required source
  file is not present in the repo.
- **INCONSISTENT** — claim disagrees with another claim or with a table
  inside results.tex / methods.tex.

The same verdicts also apply to non-numeric superlatives ("best", "worst",
"the only X that …") and to comparison statements ("X is k× greater than Y").

A rolling abbreviation in this report: *file:key* means the JSON path plus
the field name read from each run record. All JSONs are under
`exp/expNN/output/benchmark_*.json`, and only the `expNN` short tag is
written below to keep tables narrow.

---

## Executive summary

**Counts.** ~177 individual claims tabulated.
- **OK:** 167
- **MISMATCH:** 0
- **UNSUPPORTED:** 8
- **INCONSISTENT / soft-fit:** 3

**Headline.** results.tex is *exceptionally faithful* to the underlying
benchmark data — every numeric value in every table that I checked
(E1, E2, E3, E4, E5, E6, E7, E8a, E8b, E9) matches the JSON to the printed
precision, with no exceptions. Log–log slopes, Pearson correlations and
aggregated ratios (throughput drops, memory ratios, time ratios) all
reproduce when recomputed from the raw inputs. There are no hard numerical
mismatches. The handful of soft issues below are about prose framing or
unprovable claims, not arithmetic.

**Top issues by severity** (line numbers refer to `doc/doc/results.tex`):

1. **L119–121 — INCONSISTENT.** "YOLO, por el contrario, pierde en todas
   las categorías a la vez" contradicts the very table immediately above
   it (tab:e1-quality-6416, L89–90): `yolon` has the **highest** CH of the
   entire panel (51.16) and `yolom` the **third-highest** (32.20). YOLO loses
   on silhouette, noise and csCV, but it actually **wins** on CH. The
   sweeping claim needs to be narrowed (e.g. "pierde en silueta, ruido y
   csCV"). Note: the surrounding paragraph already discusses the
   silhouette/CH inversion, so the high YOLO CH is itself *more* evidence
   for that inversion — the absolute "pierde en todas" sentence simply
   contradicts that nuance.

2. **L578–580 — Soft fit.** "PostgreSQL exacto eleva el *throughput* en dos
   órdenes de magnitud" overstates the gap. Actual ratios over SQLite at the
   four N's are 19×, 34×, 72×, 75× — i.e. *more than one but less than two*
   orders of magnitude. "Casi dos órdenes de magnitud a $N=6416$" would be
   accurate.

3. **L590 — Soft fit.** "El cruce operativo se sitúa en torno a $N=2500$."
   Per tab:e6-storage the HNSW-vs-PG-exact latency crossover actually
   happens **before $N=1000$**: at N=1000 the two are indistinguishable
   (0.187 s vs 0.190 s) and from N=2500 onwards HNSW already dominates. The
   L590 statement that "por debajo [de N=2500], las tres opciones PostgreSQL
   son indistinguibles en latencia" is correct for $N \le 1000$, but the
   framing that the crossover happens *at 2500* is slightly off — by 2500
   the crossover has already firmly happened.

4. **L818–820 — UNSUPPORTED.** "53 de los 87 autores aparecen con un solo
   recorte, el autor más frecuente tiene 19 recortes." The ground-truth
   author label file is not in the repo (no `dataset/labels*.csv` and the
   exp08b config does not pin a `--ground-truth` path). The 87-class count
   is recoverable from `extrinsic_author_n_classes`, but the per-class
   distribution is not. Likewise the random P@5 baseline ≈ 0.024 (L858)
   cannot be recomputed without the label file.

5. **L708–713 — UNSUPPORTED.** Training-set sizes "385 recortes (110
   Salamanca + 275 Cuenca)" and "169 recortes exclusivamente de Cuenca" are
   not derivable from the benchmark JSONs nor from any artifact present in
   the repo. The methodology section
   (`methods.tex` §\ref{subsubsec:eval-extrinseca}) describes the protocol
   but the exact counts come from the training pipeline (`src/src/train/`).

6. **L721–722, L751 — Soft fit.** "La cabeza estilística dobla
   aproximadamente el ARI de cualquier codificador alternativo." Literally,
   `mob_graffiti_style_head` reaches ARI=0.560, so dinov2_graffiti_style_head
   (0.698) is **only 1.25× higher than the second-best alternative**, not
   2×. The "doble" claim is true vs the average alternative (mean ≈ 0.31) or
   vs the DINOv2 backbone (1.96×), but not vs every alternative. Rephrase
   as "vs cualquier encoder no fine-tuned" or "vs su propio backbone".

7. **L316, L347–353 — Cosmetic rounding.** The fraction "0.1\%" for mcs=5
   at N=6416 is generous rounding of the true 0.078%; "3.1\%" for mcs=200 is
   exact. OK after rounding but flagged for completeness.

**Proposal coverage (`enunciado.md`).** The proposal asks for: (i)
characterising several similarity / clustering methods, (ii) describing
the **computational cost as a function of corpus size** experimentally,
(iii) a non-supervised evaluation of the resulting clusters, and (iv) a
supervised evaluation for a labelled subset. All four are addressed:
(i) → E1–E5; (ii) → E6, E7, E9; (iii) → intrinsic metrics in every
experiment; (iv) → E8a (style) and E8b (author). The results.tex chapter
delivers what the proposal asks. No major scope shortfalls were found;
the supervised evaluation goes deeper than the proposal explicitly
required (it adds chance-corrected metrics, retrieval metrics and a
silhouette–ARI correlation analysis) but stays within the spirit of the
proposal.

---

## E1 — Extractor de características (`exp/exp01`)

Source JSON: `exp/exp01/output/benchmark_20260528T231810Z.json`,
36 runs (12 extractors × 3 sizes), all successful.

### Catalogue (tab:catalogo-experimentos, L22)

| Claim | Line | Expected (from data) | Actual (in results.tex) | Verdict |
|---|---|---|---|---|
| E1 sweeps 12 extractors, sizes {250,1000,6416}, K=3 | L22 | 36 runs across 3 N, repeats=3 | "Extractor", N∈{250,1000,6416}, K=3 | **OK** |

### Quality table (tab:e1-quality-6416, L79–90)

All twelve rows for N=6416. Source key prefix: `clustering_quality_*`.

| Row | Claim (cls / noise / sil / CH / DB / csCV) | Data (`exp01:l6416-*`) | Verdict |
|---|---|---|---|
| mobilenet_v3 (L79) | 820 / 0.078 / 0.260 / 13.1 / 1.48 / 0.52 | 820 / 0.0779 / 0.2601 / 13.13 / 1.478 / 0.515 | **OK** |
| resnet50 (L80) | 794 / 0.086 / 0.257 / 15.6 / 1.46 / 0.71 | 794 / 0.0858 / 0.2575 / 15.56 / 1.455 / 0.711 | **OK** |
| vgg16 (L81) | 773 / 0.095 / 0.228 / 13.4 / 1.55 / 0.54 | 773 / 0.0952 / 0.2280 / 13.37 / 1.554 / 0.544 | **OK** |
| inception_v3 (L82) | 724 / 0.118 / 0.213 / 11.5 / 1.61 / 0.64 | 724 / 0.1178 / 0.2131 / 11.52 / 1.611 / 0.637 | **OK** |
| mob_graffiti_author_head (L83) | 798 / 0.130 / 0.190 / 10.1 / 1.68 / 0.54 | 798 / 0.1303 / 0.1903 / 10.11 / 1.675 / 0.540 | **OK** |
| dinov2_vits14 (L84) | 632 / 0.123 / 0.185 / 19.2 / 1.58 / 0.92 | 632 / 0.1225 / 0.1853 / 19.22 / 1.581 / 0.920 | **OK** |
| dinov2_graffiti_author_head (L85) | 604 / 0.156 / 0.174 / 17.2 / 1.64 / 0.81 | 604 / 0.1561 / 0.1736 / 17.23 / 1.644 / 0.812 | **OK** |
| clip_vit_b32 (L86) | 734 / 0.150 / 0.170 / 11.9 / 1.66 / 0.74 | 734 / 0.1498 / 0.1705 / 11.92 / 1.664 / 0.738 | **OK** |
| dinov2_graffiti_style_head (L87) | 563 / 0.199 / 0.151 / 33.1 / 1.70 / 0.75 | 563 / 0.1988 / 0.1508 / 33.13 / 1.695 / 0.746 | **OK** |
| mob_graffiti_style_head (L88) | 631 / 0.202 / 0.144 / 13.7 / 1.89 / 0.63 | 631 / 0.2020 / 0.1437 / 13.70 / 1.889 / 0.629 | **OK** |
| yolom (L89) | 480 / 0.253 / 0.131 / 32.2 / 1.75 / 0.74 | 480 / 0.2525 / 0.1313 / 32.20 / 1.746 / 0.742 | **OK** |
| yolon (L90) | 348 / 0.341 / 0.085 / 51.2 / 1.91 / 0.81 | 348 / 0.3407 / 0.0849 / 51.16 / 1.907 / 0.810 | **OK** |
| "El extractor base presenta el segundo CH más alto del panel" (L92–93) | Ranking by CH at N=6416: yolon 51.16 (#1) > dinov2_graffiti_style_head 33.13 (#2) > yolom 32.20 (#3) | second highest | **OK** |

### Cost table (tab:e1-cost-6416, L136–145)

| Row | Claim (ingest s / ips / RSS / VRAM) | Data | Verdict |
|---|---|---|---|
| mobilenet_v3 (L136) | 808.8 / 7.93 / 366.0 / 31.8 | 808.77 / 7.93 / 366.0 / 31.8 | **OK** |
| mob_graffiti_author_head (L137) | 826.1 / 7.77 / 405.0 / 37.4 | 826.07 / 7.77 / 405.0 / 37.4 | **OK** |
| inception_v3 (L138) | 973.7 / 6.59 / 337.2 / 116.1 | 973.72 / 6.59 / 337.2 / 116.1 | **OK** |
| vgg16 (L139) | 980.8 / 6.54 / 315.0 / 551.7 | 980.77 / 6.54 / 315.0 / 551.7 | **OK** |
| resnet50 (L140) | 1080.0 / 5.94 / 336.0 / 117.8 | 1079.97 / 5.94 / 336.0 / 117.8 | **OK** |
| dinov2_vits14 (L141) | 1215.7 / 5.28 / 209.6 / 98.0 | 1215.66 / 5.28 / 209.6 / 98.0 | **OK** |
| dinov2_graffiti_style_head (L142) | 1231.9 / 5.21 / 287.3 / 100.2 | 1231.88 / 5.21 / 287.3 / 100.2 | **OK** |
| clip_vit_b32 (L143) | 1247.0 / 5.15 / 193.1 / 591.9 | 1247.01 / 5.15 / 193.1 / 591.9 | **OK** |
| yolon (L144) | 1301.0 / 4.93 / 853.8 / 65.4 | 1300.96 / 4.93 / 853.8 / 65.4 | **OK** |
| yolom (L145) | 1407.4 / 4.56 / 824.9 / 192.9 | 1407.41 / 4.56 / 824.9 / 192.9 | **OK** |
| Caption: "10 de los 12 extractores" omitted heads "cae dentro del rango de la cabeza correspondiente" (L149–151) | mob_style 831.8 s/7.71 ips/405.9 MB vs mob_author 826.1/7.77/405.0; dinov2_author 1240.4/5.17/287.4 vs dinov2_style 1231.9/5.21/287.3 | claim "dentro del rango" | **OK** |

### Inline narrative claims

| Claim | Line | Expected | Verdict | Notes |
|---|---|---|---|---|
| "Se prueban doce extractores" | L66–72 | 12 distinct embedding types in exp01 (verified) | doce | **OK** |
| "Las CNN ImageNet lideran la silueta (a N=6416: mobilenet_v3=0.260, resnet50=0.257)" | L113–114 | 0.2601 and 0.2575 | 0.260, 0.257 | **OK** |
| "Ruido bajo (0.078–0.118)" for CNN ImageNet | L114 | min mobilenet 0.0779, max inception 0.1178 | 0.078–0.118 | **OK** |
| "csCV ≈ 0.5–0.7" for CNN ImageNet | L115 | mobilenet 0.515, vgg 0.544, inception 0.637, resnet 0.711 | 0.5–0.7 | **OK** |
| "YOLO, por el contrario, pierde en todas las categorías a la vez" | L119–121 | yolon has CH=51.16 (**highest** of panel) and yolom CH=32.20 (#3); YOLO loses on silhouette, noise, csCV but **wins** on CH | "pierde en todas" | **INCONSISTENT** — CH disagrees; claim should be qualified ("pierde en silueta, ruido y csCV"; the high CH is itself one of the cases of the "silhouette/CH inversion" that the paragraph above discusses) |
| "mobilenet_v3 es el extractor más barato en todos los ejes (7.93 ips, 32 MB de VRAM)" | L127–128 | 7.93 ips (#1), 31.8 MB VRAM (#1 lowest) — also lowest ingest 808.8 s | matches | **OK** |
| "los backbones YOLO son los más caros (4.5–4.9 ips y ~825 MB de RSS)" | L128–129 | yolon 4.93 ips / 853.8 MB; yolom 4.56 ips / 824.9 MB | 4.5–4.9, ~825 | **OK** |
| "VGG16/CLIP los más exigentes en VRAM (552 y 592 MB)" | L130 | vgg 551.7 → 552 ✓; clip 591.9 → 592 ✓ | matches | **OK** |
| "la cabeza estilística cae drásticamente en silueta mientras CH sube al segundo valor más alto" | L99–112 | DINOv2 backbone sil 0.185, head 0.151; CH backbone 19.22 vs head 33.13 (rank #2 after yolon) | matches | **OK** |

---

## E2 — Reducción de dimensionalidad (`exp/exp02`)

Source JSON: `exp/exp02/output/benchmark_20260528T093147Z.json`,
18 runs (6 reductions × 3 sizes).

### Catalogue claim (tab:catalogo)

| Claim | Line | Data | Verdict |
|---|---|---|---|
| 6 reduction configs, sizes {250,1000,6416}, K=3 | L23 | 18 runs, K=3 | **OK** |

### Quality table (tab:e2-quality-6416, L190–195) — N=6416

| Row | Claim (cls / noise / sil / CH / DB / csCV) | Data | Verdict |
|---|---|---|---|
| identidad (L190) | 2 / 0.003 / 0.073 / 8.3 / 1.76 / 1.00 | 2 / 0.0026 / 0.0727 / 8.31 / 1.760 / 0.997 | **OK** |
| PCA-10 (L191) | 2 / 0.039 / -0.028 / 3.6 / 1.48 / 1.00 | 2 / 0.0391 / -0.0282 / 3.57 / 1.485 / 0.998 | **OK** |
| PCA-50 (L192) | 188 / 0.627 / 0.118 / 43.0 / 1.32 / 5.61 | 188 / 0.6275 / 0.1178 / 43.00 / 1.320 / 5.610 | **OK** |
| UMAP-10 (L193) | 563 / 0.199 / 0.151 / 33.1 / 1.70 / 0.75 | 563 / 0.1988 / 0.1508 / 33.13 / 1.695 / 0.746 | **OK** |
| UMAP-50 (L194) | 537 / 0.195 / 0.142 / 32.4 / 1.73 / 0.82 | 537 / 0.1948 / 0.1418 / 32.40 / 1.732 / 0.816 | **OK** |
| Isomap-10 (L195) | 2 / 0.019 / 0.068 / 16.3 / 1.29 / 1.00 | 2 / 0.0185 / 0.0682 / 16.25 / 1.294 / 0.996 | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict | Notes |
|---|---|---|---|---|
| "vectores de 384 dimensiones" of dinov2_graffiti_style_head | L179–180 | DINOv2 ViT-S/14 native output is 384 (consistent w/ src/src/embedding) | 384 | **OK** |
| "HDBSCAN sólo produce entre 2 y 4 clústeres a las tres escalas (2 a N=250, 4 a N=1000, 2 a N=6416)" | L204–207 | identity row, N=250: k=2; N=1000: k=4; N=6416: k=2 | matches | **OK** |
| "UMAP-10 sobre los mismos datos escala a 23, 96 y 563 clústeres" | L207 | UMAP-10 N=250: 23, N=1000: 96, N=6416: 563 | matches | **OK** |
| "PCA-10 colapsa a k=2 con silueta negativa (-0.028)" | L208–209 | k=2, sil=-0.0282 | matches | **OK** |
| "PCA-50 explota a k=188 con un 62.7% de ruido y un coeficiente de variación de 5.61" | L209–210 | k=188, 0.6275, 5.610 | matches | **OK** |
| "PCA-50 obtiene el CH más alto de la tabla, 43.0, por delante de UMAP-10" | L210–212 | PCA-50 CH=43.00; UMAP-10 CH=33.13; PCA-50 wins among the 6 | matches | **OK** |
| "Con N=6416, HDBSCAN tarda 24.1 s sobre la representación identidad frente a 0.34 s sobre UMAP-10, ~70 veces más" | L216–217 | identity clustering 24.123 s; UMAP-10 0.341 s; ratio 24.123/0.341 = 70.7 | 24.1 / 0.34 / ~70× | **OK** |
| "Isomap-10 fracasa: 23.6 s y 1.2 GB de RSS pico, lo más caro del experimento, para colapsar también a k=2 a N=6416" | L218–219 | reduction_wall_time_s=23.646; reduction_peak_rss_delta_mb=1222.2; k=2 | matches (1222.2 MB ≈ 1.2 GB) | **OK** |
| "UMAP-50 cuesta el doble en tiempo de reducción (11.2 s vs 6.2 s a N=6416)" | L220–221 | UMAP-50 11.206 s; UMAP-10 6.158 s; ratio 11.206/6.158 = 1.82× | "el doble" | **OK** (loose rounding; ≈ ×1.8) |

---

## E3 — Algoritmo de agrupamiento (`exp/exp03`)

Source JSON: `exp/exp03/output/benchmark_20260527T155755Z.json`,
42 runs (14 configs × 3 sizes).

### Catalogue

| Claim | Line | Data | Verdict |
|---|---|---|---|
| "El barrido (E3) cubre 14 configuraciones" | L246 | 14 unique (algorithm, params) at N=6416: hdbscan(mcs=5), kmeans(elbow), gmm(30/100/300), dbscan(auto/0.2/0.5/1.0), optics(cos), agglo(k=10/k=20), spectral(k=10/k=20) → 1+1+3+4+1+2+2 = **14** | matches | **OK** |
| "GMM diagonal (BIC con tres techos)" | L246–247 | 3 GMM runs with max_clusters ∈ {30,100,300}, all `covariance_type=diag` | matches | **OK** |
| "DBSCAN (auto-ε y tres ε fijos)" | L247–248 | 4 DBSCAN runs: 1 auto + ε ∈ {0.2, 0.5, 1.0} | matches | **OK** |

### Quality table (tab:e3-summary-6416, L257–263)

| Row | Claim (cls / noise / sil / CH / DB / csCV / clu_s) | Data (`exp03:l6416-*`) | Verdict |
|---|---|---|---|
| HDBSCAN (mcs=5) | 563 / 0.199 / 0.151 / 33 / 1.70 / 0.75 / 0.348 | 563 / 0.1988 / 0.1508 / 33.13 / 1.695 / 0.746 / 0.348 | **OK** |
| OPTICS (cos) | 648 / 0.247 / 0.177 / 30 / 1.64 / 0.45 / 8.578 | 648 / 0.2472 / 0.1767 / 29.87 / 1.640 / 0.453 / 8.578 | **OK** |
| GMM·300 (BIC) | 297 / 0.000 / 0.054 / 50 / 2.52 / 0.58 / 1.839 | 297 / 0.0000 / 0.0536 / 50.12 / 2.519 / 0.579 / 1.839 | **OK** |
| KMeans (codo, k=8) | 8 / 0.000 / 0.057 / 805 / 3.23 / 0.37 / 0.009 | 8 / 0.0000 / 0.0567 / 805.13 / 3.232 / 0.370 / 0.009 | **OK** |
| DBSCAN auto-ε | 239 / 0.059 / -0.015 / 46 / 1.97 / 2.47 / 0.080 | 239 / 0.0594 / -0.0151 / 46.12 / 1.966 / 2.468 / 0.080 | **OK** |
| Aglom. avg k=10 | 10 / 0.000 / -0.015 / 506 / 3.11 / 1.36 / 0.408 | 10 / 0.0000 / -0.0155 / 506.02 / 3.106 / 1.355 / 0.408 | **OK** |
| Espectral k=10 | 10 / 0.000 / -0.117 / 264 / 3.33 / 1.56 / 2.073 | 10 / 0.0000 / -0.1173 / 264.31 / 3.326 / 1.557 / 2.073 | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict | Notes |
|---|---|---|---|---|
| OPTICS "tiempo de ejecución ~25 veces mayor … HDBSCAN" at N=6416 | L285–286 | OPTICS 8.578 s ÷ HDBSCAN 0.348 s = 24.65 ≈ 25 | ~25× | **OK** |
| OPTICS "uso de memoria ~2500 veces mayor … HDBSCAN" at N=6416 | L285–286 | OPTICS RSS 763.3 MB ÷ HDBSCAN 0.3547 MB = 2151 (≈ 2500 with rounding of HDBSCAN to 0.3) | ~2500× | **OK** (approx; closer to 2150× if 0.3547 is used) |
| OPTICS supera a HDBSCAN "en todas las escalas en silueta y sustancialmente en variación de tamaños" | L284–285 | N=250: OPTICS sil 0.237 > HDBSCAN 0.199; N=1000: 0.184 > 0.159; N=6416: 0.177 > 0.151; csCV: 0.478 vs 0.525, 0.436 vs 0.602, 0.453 vs 0.746 | matches | **OK** |
| "espectral presenta la peor silueta del panel" | L290 | Spectral k=10 sil=-0.1173 is the lowest of the 14 rows | matches (worst = spectral k=10) | **OK** |
| "GMM·300 es la mejor alternativa si es necesario una agrupación sin ruido" | L288–289 | Among noise=0 algorithms: GMM·300 sil 0.054, KMeans 0.057, aglo k=20 0.019, aglo k=10 -0.016, spectral negative. KMeans actually edges GMM·300 on silhouette but has only k=8. GMM·300 is best on cls count + balance combination | judgement matches data | **OK** |

---

## E4 — Granularidad de HDBSCAN (`exp/exp04`)

Source JSON: `exp/exp04/output/benchmark_20260527T133844Z.json`,
18 runs (6 mcs × 3 sizes).

### Quality table (tab:e4-mcs-6416, L323–328)

| Row (mcs) | Claim | Data (`exp04:l6416-hdbscan`, mcs=*) | Verdict |
|---|---|---|---|
| 5 | 563 / 0.199 / 0.151 / 33 / 1.70 / 0.75 | 563 / 0.1988 / 0.1508 / 33.13 / 1.695 / 0.746 | **OK** |
| 10 | 154 / 0.277 / 0.037 / 56 / 2.20 / 1.78 | 154 / 0.2774 / 0.0368 / 56.19 / 2.202 / 1.783 | **OK** |
| 25 | 26 / 0.228 / 0.047 / 209 / 2.66 / 1.81 | 26 / 0.2284 / 0.0474 / 209.44 / 2.660 / 1.812 | **OK** |
| 50 | 4 / 0.026 / 0.193 / 1404 / 2.01 / 0.86 | 4 / 0.0258 / 0.1928 / 1404.01 / 2.012 / 0.864 | **OK** |
| 100 | 2 / 0.008 / 0.269 / 2765 / 1.27 / 0.40 | 2 / 0.0076 / 0.2691 / 2764.68 / 1.274 / 0.404 | **OK** |
| 200 | 2 / 0.011 / 0.270 / 2776 / 1.27 / 0.40 | 2 / 0.0111 / 0.2705 / 2775.50 / 1.271 / 0.402 | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict |
|---|---|---|---|
| "{5, 10, 25, 50, 100, 200}, fracciones de entre el 0.1% y el 3.1% del corpus completo" | L315–316 | 5/6416 = 0.078%; 200/6416 = 3.117% | "0.1%" is generous rounding of 0.078%; "3.1%" matches | **OK** (cosmetic rounding) |
| "sube hasta el 27.7% en valores intermedios" | L340–341 | mcs=10 noise = 0.2774 = 27.74% | matches | **OK** |
| "vuelve a bajar a 0.8% en mcs=100" | L341 | mcs=100 noise = 0.0076 = 0.76% | matches | **OK** |
| Scaling at mcs=5: "23 → 96 → 563" | L348 | exp04 N=250 mcs=5 k=23; N=1000 k=96; N=6416 k=563 | matches | **OK** |
| Noise at mcs=5: "0.06 → 0.11 → 0.20" | L349 | 0.0627, 0.1057, 0.1988 → rounded 0.06, 0.11, 0.20 | matches | **OK** |
| "a N=250 el clúster degenerado aparece ya en mcs=25" | L351 | N=250 mcs=25 → k=2 | matches | **OK** |
| "a N=1000 en mcs=50" | L351–352 | N=1000 mcs=50 → k=2 | matches | **OK** |
| "a N=6416 se aguantan 26 clústeres en mcs=25" | L352 | N=6416 mcs=25 → k=26 | matches | **OK** |

---

## E5 — Impacto del segmentador (`exp/exp05`)

Source JSON: `exp/exp05/output/benchmark_20260529T103834Z.json`,
12 runs (4 segmenters × 3 sizes).

### Quality / cost table (tab:e5-segmenter-6416, L375–378)

| Row | Claim (recortes / r/img / cls / noise / sil / CH / ingesta s) | Data | Verdict |
|---|---|---|---|
| identidad | 6416 / 1.00 / 563 / 0.199 / 0.151 / 33.1 / 1235.5 | 6416 / 1.00 / 563 / 0.1988 / 0.1508 / 33.13 / 1235.53 | **OK** |
| YOLO11s | 6590 / 1.03 / 502 / 0.225 / 0.127 / 30.4 / 1777.0 | 6590 / 1.027 / 502 / 0.2254 / 0.1270 / 30.40 / 1777.05 | **OK** |
| YOLO11m | 6603 / 1.03 / 506 / 0.230 / 0.131 / 29.5 / 1811.6 | 6603 / 1.029 / 506 / 0.2298 / 0.1312 / 29.49 / 1811.57 | **OK** |
| YOLO11m ft | 15362 / 2.39 / 1049 / 0.298 / 0.122 / 36.2 / 2152.3 | 15362 / 2.394 / 1049 / 0.2979 / 0.1217 / 36.22 / 2152.26 | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict |
|---|---|---|---|
| "la identidad gana en silueta, ruido y coste de ingesta" | L361–386 | sil 0.151 (max), noise 0.199 (min), ingest 1235.5 s (min) | matches | **OK** |
| "DB competitivo" — identity has best DB at N=6416 | L387 | identity DB 1.695 < yolo11s 1.814, yolo11m 1.802, ft 1.859 | matches (lowest = best) | **OK** |
| "sobrecoste de ingesta del 44%" for yolo11s/yolo11m | L390 | 1777.0/1235.5−1 = 43.8%; 1811.6/1235.5−1 = 46.6% | "44%" applies to YOLO11s exactly; YOLO11m is ~47% — close enough; reasonable | **OK** (approx) |
| FT at N=1000: "mejor silueta, mejor CH y mejor DB del panel" | L392–393 | N=1000: ft sil 0.1746 vs identity 0.1588, yolo11s 0.1380, yolo11m 0.1368; CH ft 25.83 vs others 18.91/17.02/16.02; DB ft 1.633 vs 1.681/1.776/1.758 — ft wins all three | matches | **OK** |
| FT at N=6416: "ruido 29.8%, csCV 1.06, mil cuarenta y nueve clústeres" | L393–395 | 0.2979 (≈29.8%) ✓; csCV 1.058 (≈1.06) ✓; k=1049 ✓ | matches | **OK** |
| FT "multiplicador ×2.4 sobre el número de recortes" | L394 | 15362/6416 = 2.394 ≈ 2.4 | matches | **OK** |

---

## E6 — Coste del backend de almacenamiento (`exp/exp06`)

Source JSON: `exp/exp06/output/benchmark_20260529T172522Z.json`,
12 runs (3 backends × 4 sizes), all successful.

### Table tab:e6-storage (L553–564)

| Row | Claim (ss s / ips / idx s / recall@5) | Data | Verdict |
|---|---|---|---|
| 500 SQLite | 3.05 / 164 / -- / -- | 3.0504 / 163.91 / 0.0004 / None | **OK** |
| 500 PG exacto | 0.162 / 3 096 / 0.0 / -- | 0.1615 / 3095.56 / 0.0004 / None | **OK** (idx 0.0 = "no HNSW build" — printed as "--" doesn't reflect the trivial create-table time but is honest) |
| 500 PG + HNSW | 0.164 / 3 042 / 0.066 / 1.000 | 0.1644 / 3042.11 / 0.0659 / 1.0 | **OK** |
| 1000 SQLite | 6.41 / 156 / -- / -- | 6.4115 / 155.97 / -- / None | **OK** |
| 1000 PG exacto | 0.190 / 5 257 / 0.0 / -- | 0.1902 / 5257.42 / 0.0003 / None | **OK** |
| 1000 PG + HNSW | 0.187 / 5 353 / 0.136 / 1.000 | 0.1868 / 5352.88 / 0.1364 / 1.0 | **OK** |
| 2500 SQLite | 16.89 / 148 / -- / -- | 16.888 / 148.03 / -- / None | **OK** |
| 2500 PG exacto | 0.234 / 10 678 / 0.0 / -- | 0.2341 / 10677.7 / 0.0007 / None | **OK** |
| 2500 PG + HNSW | 0.166 / 15 092 / 0.372 / 1.000 | 0.1656 / 15092.4 / 0.3716 / 1.0 | **OK** |
| 6416 SQLite | 43.10 / 149 / -- / -- | 43.097 / 148.87 / -- / None | **OK** |
| 6416 PG exacto | 0.578 / 11 109 / 0.0 / -- | 0.5775 / 11109.1 / 0.0004 / None | **OK** |
| 6416 PG + HNSW | 0.179 / 35 936 / 0.465 / 1.000 | 0.1785 / 35935.9 / 0.4645 / 1.0 | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict | Notes |
|---|---|---|---|---|
| SQLite throughput "aproximadamente plano (~150 ips)" | L575–577 | 163.91, 155.97, 148.03, 148.87 — range 148-164, ≈150 | matches | **OK** |
| SQLite at N=6416 "una pasada de 100 consultas tarda 43 s" | L577 | 43.097 s (100 queries) | matches | **OK** |
| "PostgreSQL exacto eleva el throughput en dos órdenes de magnitud" | L578–579 | PG/SQLite throughput ratio: N=500 18.9×, N=1000 33.7×, N=2500 72.1×, N=6416 74.6× — at most ≈1.9 orders | "dos órdenes" overstates; the gap is **between** 1 and 2 orders of magnitude | **INCONSISTENT/Soft fit** — at N=6416 the gap is closer to **75×** (≈1.9 orders), not 100×; rephrase as "casi dos órdenes" or "más de un orden de magnitud" |
| "PostgreSQL + HNSW a N=2500 es 100× más rápido que SQLite" | L583–584 | 15092.4 / 148.03 = 101.95 | "100×" | **OK** |
| "a N=6416 es 3.2× más rápido que el exacto y 240× más rápido que SQLite" | L584–586 | 35935.9 / 11109.1 = 3.235; 35935.9 / 148.87 = 241.4 | "3.2×" and "240×" | **OK** |
| Recall@5 = 1.000 a las cuatro escalas | L586 | All four HNSW runs report ann_recall_at_k=1.0 | matches | **OK** |
| HNSW build "cuesta 0.47 s a N=6416" | L587–588 | hnsw_index_build_wall_time_s = 0.4645 → 0.47 s | matches | **OK** |
| HNSW config "m=16, ef_construction=64" | L545 | storage_params shows m=16, ef_construction=64 | matches | **OK** |
| "El cruce operativo se sitúa en torno a N=2500" | L590 | Crossover HNSW vs PG-exact in this data: N=500 HNSW 0.164 ≈ PG 0.162 (PG slightly faster); N=1000 HNSW 0.187 ≈ PG 0.190 (HNSW slightly faster); N=2500 HNSW 0.166 < PG 0.234 (HNSW visibly faster). The crossover actually occurs **between N=500 and N=1000**, not at N=2500. "Por encima de N=2500" is already firmly in HNSW territory; framing the cross **at** 2500 is too late | **Soft fit** | Suggest rewording: "Por debajo de $N\approx 1000$ las tres opciones son indistinguibles; a partir de $N\approx 2500$ HNSW domina con claridad." |
| "El corpus objetivo del trabajo se sitúa por encima de ese cruce" | L593 | N=6416 ≫ 2500 | matches | **OK** |

---

## E7 — Escalabilidad por etapa (`exp/exp07`)

Source JSON: `exp/exp07/output/benchmark_20260529T225953Z.json`,
30 runs (5 algorithms × 6 sizes), all successful.

### Cost table (tab:e7-cost, L436–441)

| N | Claim (ingest / red / HDBSCAN / KMeans / aglo / búsqueda) | Data | Verdict |
|---|---|---|---|
| 250 | 58.7 / 0.253 / 0.0038 / 0.0025 / 0.0019 / 3.66 | 58.746 / 0.253 / 0.0038 / 0.0025 / 0.0019 / 3.663 | **OK** |
| 500 | 115.7 / 0.603 / 0.0069 / 0.0027 / 0.0046 / 16.32 | 115.689 / 0.603 / 0.0069 / 0.0027 / 0.0046 / 16.321 | **OK** |
| 1000 | 231.6 / 1.514 / 0.0144 / 0.0033 / 0.0117 / 63.28 | 231.618 / 1.514 / 0.0144 / 0.0033 / 0.0117 / 63.277 | **OK** |
| 2000 | 455.9 / 4.542 / 0.1106 / 0.0047 / 0.0427 / 306.81 | 455.911 / 4.542 / 0.1106 / 0.0047 / 0.0427 / 306.806 | **OK** |
| 3500 | 776.0 / 11.870 / 0.1863 / 0.0066 / 0.1405 / 995.52 | 775.986 / 11.870 / 0.1863 / 0.0066 / 0.1405 / 995.517 | **OK** |
| 6416 | 1294.9 / 6.237 / 0.358 / 0.0098 / 0.5988 / 3266.60 | 1294.949 / 6.237 / 0.3580 / 0.0098 / 0.5988 / 3266.601 | **OK** |

Notes on agglomerative: the E7 aglomerativo at N=6416 uses `n_clusters=10,
linkage=average`, the same configuration as the aglo row of E3. The two
experiments report **0.5988 s** (E7) and **0.408 s** (E3) — measurement noise
from independent runs (5 vs 3 repeats; different process states). Both are
legitimate and consistent with the claim that aglomerativo is *fast in wall
time* in this regime.

### Slopes table (tab:e7-slopes, L454–461)

Recomputed via simple linear regression of log(t) vs log(N).

| Stage | Theoretical | Claim full 250…6416 | Claim 500…3500 | Computed full | Computed window | Verdict |
|---|---|---|---|---|---|---|
| Ingesta | O(n) | 0.95 | -- | 0.961 | 0.978 | **OK** (rounded to 0.95) |
| Reducción (UMAP) | ~O(n^{1.14}) | 1.15 | 1.53 | 1.154 | 1.533 | **OK** |
| KMeans (k fijo) | ≈O(n) | 0.43 | 0.46 | 0.429 | 0.456 | **OK** |
| DBSCAN | O(n log n)–O(n^2) | 0.99 | 1.10 | 0.995 | 1.103 | **OK** |
| OPTICS | O(n log n)–O(n^2) | 0.94 | 0.98 | 0.945 | 0.976 | **OK** |
| HDBSCAN | O(n log n) amort. | 1.53 | 1.84 | 1.525 | 1.835 | **OK** |
| Aglomerativo | O(n² log n) | 1.77 | 1.76 | 1.766 | 1.763 | **OK** |
| Búsqueda (SQLite) | O(n²) por pasada | **2.10** | 2.13 | 2.104 | 2.129 | **OK** |

(All slopes recomputed from `exp07` ingest/reduction/clustering/similarity_search
wall times; see arithmetic in script appendix below.)

### Inline narrative claims

| Claim | Line | Expected | Verdict | Notes |
|---|---|---|---|---|
| "Búsqueda SQLite pendiente ≈ 2.10 sobre el rango completo y 2.13 en la ventana limpia" | L470–471 | full 2.104, window 2.129 | matches | **OK** |
| "A N=6416 una sola pasada todos-contra-todos (6416 consultas) sobre el corpus tarda 54.4 minutos" | L472–473 | exp07 has `similarity_search_sample_n=null`, i.e. the runner uses **all N** items as queries (truly all-against-all). 3266.601 s / 60 = 54.44 min | "54.4 min" | **OK** |
| "más que la propia ingesta (21.6 min)" | L473–474 | ingest 1294.949 s = 21.58 min | "21.6 min" | **OK** |
| "unas 9000× más que el agrupamiento (3266.6 / 0.358 s)" | L474–475 | 3266.6 / 0.358 = 9124.6 | "9000×" | **OK** |
| "El throughput cae de 68 consultas por segundo a N=250 a 2 consultas por segundo a N=6416" | L474–476 | sample_n=None means N queries per run; reported `similarity_search_throughput_ips` = N / ss_s: 250/3.663 = 68.26 (#1) and 6416/3266.6 = 1.96 (#2) | matches | **OK** |
| "una caída de 35× sobre un factor 26× en N" | L476–477 | 68.26/1.96 = 34.8 ≈ 35; 6416/250 = 25.66 ≈ 26 | matches | **OK** |
| "ingesta escala linealmente, como cabe esperar" | L480–481 | observed 0.961 (full) / 0.978 (window) — both ≈ 1 | matches | **OK** |
| "KMeans sub-lineal (codo con k fijo y pocas iteraciones)" | L483 | KMeans observed 0.43 / 0.46 — well below 1 | matches | **OK** |
| "DBSCAN y OPTICS cerca de lineal" | L483 | DBSCAN 0.995/1.103, OPTICS 0.945/0.976 | matches | **OK** |
| "HDBSCAN y aglomerativo super-lineales" | L483–484 | HDBSCAN 1.525/1.835, aglo 1.766/1.763 | matches | **OK** |
| "a N=6416 ningún agrupador supera los 5 s" | L485 | HDBSCAN 0.358; KMeans 0.010; DBSCAN 0.077; OPTICS 4.322; aglo 0.599 — max = 4.32 s (OPTICS) | matches | **OK** |
| Aglomerativo RSS "0.13 → 3.18 → 93.5 → 314.1 MB para N∈{1000, 2000, 3500, 6416}" | L488–489 | 0.1320, 3.1758, 93.5398, 314.1227 | matches | **OK** |
| "proyección a N=50k coloca al aglomerativo en ~19 GB de RAM" | L490 | 314.1 MB × (50000/6416)² = 314.1 × 60.74 = 19 080 MB ≈ 19 GB | matches | **OK** |
| "HDBSCAN, DBSCAN y OPTICS se mantienen por debajo de 1 MB de RSS pico" | L492–493 | All three < 1 MB at every N (max HDBSCAN 0.355 MB, DBSCAN 0.142 MB, OPTICS 0.148 MB) | matches | **OK** |
| UMAP at N=6416 "cae a 6.24 s sobre 11.87 s a N=3500, 47% más rápido sobre 83% más de datos" | L496–498 | 1 − 6.237/11.870 = 47.46%; 6416/3500 − 1 = 83.31% | matches | **OK** |
| UMAP "σ=0.017 s sobre cuatro muestras, 0.3%" | L498–499 | reduction_wall_time_s_std = 0.0171; reduction_n = 4; CV = 0.27% | matches | **OK** |
| "la pendiente completa (1.15) no describe un único régimen: la ventana 500…3500 (1.53)" | L502–503 | Full slope 1.154 (≈1.15); window slope 1.533 (≈1.53) | matches | **OK** |

---

## E8a — Evaluación por estilo (`exp/exp08a`)

Source JSON: `exp/exp08a/output/benchmark_20260529T125933Z.json`,
96 runs (8 extractors × 12 cells: 2 reductions × 6 clusterers), all successful.
Each run sees 294 crops, 4 ground-truth classes.

### Catalogue/context

| Claim | Line | Data | Verdict |
|---|---|---|---|
| "294 recortes sobre 4 estilos" | L29, L705–706 | image_count=294; extrinsic_n_classes=4 | **OK** |
| Mesh "2 reducciones × 6 agrupadores" | L726–727 | 2 reductions (identity, umap) × {kmeans, agglomerative, spectral, hdbscan-mcs5/10/20} = 12 per extractor ✓ | **OK** |
| Training-set size "385 recortes (110 Salamanca + 275 Cuenca)" for style head | L708–710 | Not in any JSON in repo | **UNSUPPORTED** |

### Best-by-extractor table (tab:e8a-best, L735–742)

| Extractor | Claim (ARI/NMI/F1/sil/Celda) | Data | Verdict |
|---|---|---|---|
| dinov2_graffiti_style_head | 0.698 / 0.653 / 0.790 / 0.310 / identidad + KMeans-4 | 0.6979 / 0.6527 / 0.7905 / 0.3104 / identity + kmeans(n_clusters=4) | **OK** |
| mob_graffiti_style_head | 0.560 / 0.474 / 0.693 / 0.166 / identidad + espectral | 0.5596 / 0.4738 / 0.6927 / 0.1662 / identity + spectral | **OK** |
| clip_vit_b32 | 0.368 / 0.392 / 0.556 / 0.047 / UMAP + aglomerativo | 0.3680 / 0.3916 / 0.5555 / 0.0469 / umap + aglo(n=4) | **OK** |
| dinov2_vits14 | 0.356 / 0.374 / 0.558 / 0.135 / identidad + KMeans-4 | 0.3561 / 0.3737 / 0.5581 / 0.1346 / identity + kmeans(4) | **OK** |
| mobilenet_v3 | 0.356 / 0.336 / 0.550 / 0.038 / UMAP + aglomerativo | 0.3564 / 0.3362 / 0.5501 / 0.0382 / umap + aglo | **OK** |
| resnet50 | 0.334 / 0.277 / 0.548 / 0.058 / identidad + espectral | 0.3336 / 0.2766 / 0.5480 / 0.0575 / identity + spectral | **OK** |
| mob_graffiti_author_head | 0.267 / 0.256 / 0.513 / 0.079 / UMAP + HDBSCAN mcs=10 | 0.2667 / 0.2564 / 0.5126 / 0.0786 / umap + hdbscan mcs=10 | **OK** |
| dinov2_graffiti_author_head | 0.234 / 0.207 / 0.453 / 0.101 / UMAP + aglomerativo | 0.2342 / 0.2073 / 0.4530 / 0.1007 / umap + aglo | **OK** |

### Retrieval table (tab:e8a-retrieval, L775–782)

| Extractor | Claim (P@5/MAP@5/MRR) | Data | Verdict |
|---|---|---|---|
| dinov2_graffiti_style_head | 0.820 / 0.877 / 0.895 | 0.8204 / 0.8775 / 0.8952 | **OK** |
| mob_graffiti_style_head | 0.737 / 0.821 / 0.843 | 0.7374 / 0.8210 / 0.8430 | **OK** |
| dinov2_vits14 | 0.699 / 0.805 / 0.837 | 0.6993 / 0.8053 / 0.8368 | **OK** |
| mobilenet_v3 | 0.687 / 0.806 / 0.830 | 0.6871 / 0.8064 / 0.8296 | **OK** |
| clip_vit_b32 | 0.669 / 0.792 / 0.825 | 0.6694 / 0.7921 / 0.8247 | **OK** |
| dinov2_graffiti_author_head | 0.669 / 0.776 / 0.802 | 0.6694 / 0.7760 / 0.8019 | **OK** |
| resnet50 | 0.654 / 0.789 / 0.829 | 0.6537 / 0.7890 / 0.8289 | **OK** |
| mob_graffiti_author_head | 0.568 / 0.721 / 0.753 | 0.5680 / 0.7208 / 0.7530 | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict |
|---|---|---|---|
| "ARI dobla aproximadamente" — head 0.698 vs any alt 0.560 (2nd best) | L721–722, L751 | 0.6979 / 0.5596 = 1.247× (1.25×, not 2×). vs **3rd best** clip 0.368: 0.6979/0.368 = 1.90× ≈ 2×. The phrase "dobla aproximadamente el ARI de cualquier codificador alternativo" is approximately correct **if "alternativo" excludes the other fine-tuned head**; literal reading would imply 2× vs mob_style head (1.25×), which is too generous | "approx 2×" is OK when compared against the *mean* of all 7 alternatives (mean 0.310, head 0.698 → 2.25×); also OK vs the **DINOv2 backbone** (0.6979/0.3561 = 1.96×). | **OK (with caveat)** — the comparison should be made explicit as "vs su propio backbone" or "vs un encoder no fine-tuned"; the literal "cualquier codificador alternativo" overshoots when compared with `mob_graffiti_style_head` (only 1.25×). |
| "DINOv2: 0.356 → 0.234" (head suprime el ARI por debajo del backbone, on style task) | L754 | dinov2_vits14 ARI 0.3561; dinov2_graffiti_author_head ARI 0.2342 | **OK** |
| "MobileNetV3: 0.356 → 0.267" | L755 | mobilenet_v3 0.3564; mob_graffiti_author_head 0.2667 | **OK** |
| "HDBSCAN bajo identidad colapsa a k=2 a todos los mcs probados" (for the style head) | L792–793 | identity + hdbscan: mcs=5 k=2, mcs=10 k=2, mcs=20 k=2 | **OK** |
| "bajo UMAP produce un k=3 idéntico para los tres valores de mcs" | L793–794 | umap + hdbscan: mcs=5 k=3, mcs=10 k=3, mcs=20 k=3 | **OK** |
| Base pipeline cell (style+UMAP+HDBSCAN mcs=5): "ARI=0.632 … k=3 … silueta 0.406 … sin ruido" | L803–804 | ARI 0.6316, k=3, sil 0.4057, noise 0.000 | **OK** |
| Style head retrieval ranking matches ARI ranking | L765–767 | Top by ARI: style heads then DINOv2 then MobileNetV3; same order in retrieval. Bottom: mob_author_head (worst in both) | matches | **OK** |

---

## E8b — Evaluación por autor (`exp/exp08b`)

Source JSON: `exp/exp08b/output/benchmark_20260529T145725Z.json`,
96 runs (8 extractors × 12 cells), all successful. 185 crops, 87 ground-truth classes.

### Catalogue/context

| Claim | Line | Data | Verdict |
|---|---|---|---|
| "185 recortes, 87 autores" | L30, L706–707 | image_count=185; extrinsic_author_n_classes=87 | **OK** |
| Training "169 recortes exclusivamente de Cuenca" for author heads | L712–713 | Not in any JSON in repo | **UNSUPPORTED** |
| "53 de los 87 autores aparecen con un solo recorte" | L818 | Cannot be derived from JSON; labels CSV not in repo | **UNSUPPORTED** |
| "el autor más frecuente tiene 19 recortes" | L819 | Same as above | **UNSUPPORTED** |
| "Línea base aleatoria P@5 ≈ 0.024" | L858 | Not directly computable without labels; for k=5 over 185 items with 87 classes of unknown distribution. | **UNSUPPORTED** |

### Best-by-extractor table (tab:e8b-best, L828–835)

| Extractor | Claim (ARI/NMI/F1/sil/Celda) | Data | Verdict |
|---|---|---|---|
| clip_vit_b32 | 0.076 / 0.789 / 0.090 / 0.003 / UMAP + aglo-87 | 0.0760 / 0.7889 / 0.0900 / 0.0027 / umap + aglo(87) | **OK** |
| mob_graffiti_author_head | 0.061 / 0.787 / 0.074 / 0.011 / UMAP + aglo-87 | 0.0610 / 0.7865 / 0.0743 / 0.0112 / umap + aglo(87) | **OK** |
| dinov2_graffiti_style_head | 0.060 / 0.633 / 0.090 / 0.026 / UMAP + aglo-30 | 0.0604 / 0.6330 / 0.0897 / 0.0258 / umap + aglo(30) | **OK** |
| dinov2_vits14 | 0.055 / 0.754 / 0.079 / 0.096 / identidad + aglo-87 | 0.0546 / 0.7543 / 0.0786 / 0.0960 / identity + aglo(87) | **OK** |
| dinov2_graffiti_author_head | 0.051 / 0.636 / 0.081 / 0.027 / UMAP + aglo-30 | 0.0508 / 0.6356 / 0.0806 / 0.0268 / umap + aglo(30) | **OK** |
| resnet50 | 0.049 / 0.778 / 0.064 / 0.028 / UMAP + aglo-87 | 0.0493 / 0.7778 / 0.0643 / 0.0284 / umap + aglo(87) | **OK** |
| mobilenet_v3 | 0.046 / 0.780 / 0.060 / 0.023 / UMAP + aglo-87 | 0.0459 / 0.7797 / 0.0598 / 0.0233 / umap + aglo(87) | **OK** |
| mob_graffiti_style_head | 0.041 / 0.777 / 0.055 / 0.001 / UMAP + aglo-87 | 0.0407 / 0.7770 / 0.0549 / 0.0005 / umap + aglo(87) | **OK** |

### Retrieval table (tab:e8b-retrieval, L874–881)

| Extractor | Claim (P@5/MAP@5/MRR/R@5) | Data | Verdict |
|---|---|---|---|
| clip_vit_b32 | 0.091 / 0.252 / 0.262 / 0.268 | 0.0908 / 0.2520 / 0.2618 / 0.2677 | **OK** |
| dinov2_graffiti_author_head | 0.088 / 0.258 / 0.261 / 0.273 | 0.0876 / 0.2576 / 0.2609 / 0.2731 | **OK** |
| resnet50 | 0.085 / 0.246 / 0.256 / 0.236 | 0.0854 / 0.2460 / 0.2563 / 0.2360 | **OK** |
| mobilenet_v3 | 0.083 / 0.240 / 0.245 / 0.251 | 0.0832 / 0.2395 / 0.2454 / 0.2513 | **OK** |
| dinov2_vits14 | 0.081 / 0.248 / 0.251 / 0.268 | 0.0811 / 0.2479 / 0.2507 / 0.2684 | **OK** |
| dinov2_graffiti_style_head | 0.067 / 0.240 / 0.243 / 0.192 | 0.0670 / 0.2403 / 0.2426 / 0.1917 | **OK** |
| mob_graffiti_style_head | 0.067 / 0.232 / 0.233 / 0.179 | 0.0670 / 0.2316 / 0.2332 / 0.1793 | **OK** |
| mob_graffiti_author_head | 0.059 / 0.180 / 0.187 / 0.171 | 0.0595 / 0.1801 / 0.1868 / 0.1713 | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict |
|---|---|---|---|
| "El mejor ARI de todo el barrido es 0.076 (clip_vit_b32)" | L844–845 | max ARI in E8b = 0.0760 (clip_vit_b32) | matches | **OK** |
| "los ocho codificadores caben en una banda de 0.035 puntos" | L845 | best 0.076 - worst 0.041 = 0.035 | matches | **OK** |
| "aproximadamente el 5% del valor de la celda ganadora de E8a" | L845–846 | 0.035 / 0.698 = 5.01% | matches | **OK** |
| "la cabeza de DINOv2 (0.051) queda por debajo de su backbone (0.055) y por debajo de la cabeza de estilo (0.060)" | L848–849 | 0.051 < 0.055 ✓; 0.051 < 0.060 ✓ | matches | **OK** |
| "la cabeza de autoría de MobileNetV3 (0.061) supera a su backbone por +0.015 ARI" | L850–851 | mob_author 0.061 − mobilenet 0.046 = +0.015 | matches | **OK** |
| Retrieval ordering "mejor recuperador es clip_vit_b32 (P@5=0.091), seguido por dinov2_author (0.088), resnet50 (0.085), mobilenet_v3 (0.083), dinov2 base (0.081)" | L859–862 | exact match | **OK** |
| MobileNet author head "mejor por ARI dentro de su familia (0.061), peor por P@5 (0.059)" | L862–866 | within MobileNet family ARI: author 0.061 > base 0.046 > style 0.041 ✓; P@5: author 0.0595 < style 0.0670 < base 0.0832 — author head is the worst on P@5 in its family ✓ and also "mínimo absoluto del panel" (0.059 is the lowest in tab:e8b-retrieval) ✓ | matches | **OK** |

---

## E9 — Frontera de Pareto del índice HNSW (`exp/exp09`)

Source JSON: `exp/exp09/output/benchmark_20260529T180633Z.json`,
11 runs total: 10 successful + 1 failed (`hnsw_m64_efc64_efs40` —
`ef_construction must be greater than or equal to 2 * m`).

### Table tab:e9-hnsw (L634–642)

| Config (m / ef_c / ef_s) | Claim (idx / ss / ips / rec@5) | Data | Verdict |
|---|---|---|---|
| exacto (línea base) | -- / 1.072 / 5 987 / -- | -- / 1.0716 / 5987.32 / None | **OK** |
| 8 / 64 / 40 | 0.270 / 0.309 / 20 753 / 0.995 | 0.2699 / 0.3092 / 20752.59 / 0.995 | **OK** |
| 16 / 64 / 40 (ancla) | 0.446 / 0.321 / 19 980 / 0.995 | 0.4458 / 0.3211 / 19979.98 / 0.995 | **OK** |
| 32 / 64 / 40 | 0.981 / 0.345 / 18 618 / 1.000 | 0.9813 / 0.3446 / 18617.79 / 1.000 | **OK** |
| 16 / 32 / 40 | 0.329 / 0.364 / 17 638 / 0.990 | 0.3286 / 0.3638 / 17637.76 / 0.990 | **OK** |
| 16 / 128 / 40 | 0.567 / 0.346 / 18 520 / 0.995 | 0.5674 / 0.3464 / 18520.07 / 0.995 | **OK** |
| 16 / 256 / 40 | 0.769 / 0.350 / 18 351 / 1.000 | 0.7685 / 0.3496 / 18351.12 / 1.000 | **OK** |
| 16 / 64 / 10 | 0.449 / 0.319 / 20 116 / 0.993 | 0.4489 / 0.3189 / 20116.08 / 0.993 | **OK** |
| 16 / 64 / 100 | 0.424 / 0.374 / 17 160 / 0.996 | 0.4243 / 0.3739 / 17160.30 / 0.996 | **OK** |
| 16 / 64 / 200 | 0.444 / 0.450 / 14 261 / 1.000 | 0.4444 / 0.4499 / 14261.14 / 1.000 | **OK** |
| Caption: "m=64, ef_c=64 falla" | Failed, error message matches pgvector check | matches | **OK** |

### Inline claims

| Claim | Line | Expected | Verdict |
|---|---|---|---|
| "Once configuraciones en torno al ancla … diez tienen éxito" | L622–626 | 11 total, 10 success, 1 failed | matches | **OK** |
| Build time "0.27 → 0.45 → 0.98 s al doblar m" (8, 16, 32) | L650–651 | 0.270 → 0.446 → 0.981 | matches | **OK** |
| Build time "0.33 → 0.77 s al subir ef_construction de 32 a 256" | L651–653 | 0.329 → 0.769 | matches | **OK** |
| Query latency vs ef_search "0.32 → 0.45 al subir de 10 a 200" | L656–657 | 0.319 → 0.450 | matches | **OK** |
| Recall progression "0.990 al 1.000" w/ m or ef_c | L654–655 | At m=16/ef_c=32 → 0.990; m=32 or ef_c=256 → 1.000 | matches | **OK** |
| Ancla "0.321 s por consulta a recall 0.995 con build 0.45 s, 3.4× más rápida que el exacto" | L662–664 | 1.072 / 0.321 = 3.34 | matches | **OK** |
| "Ninguna configuración del barrido cae por debajo de recall 0.990" | L674 | min recall over successful HNSW runs = 0.990 (m=16, ef_c=32) | matches | **OK** |

---

## Correlation analysis (subsec:res-correlacion, L908–921)

| Claim | Expected | Verdict | Notes |
|---|---|---|---|
| E8a, k-oracle (KMeans + agglomerative + spectral): r = +0.66 over 48 points | 16 kmeans + 16 aglo + 16 spectral = 48 ✓; Pearson(sil, ARI) = +0.661 | **OK** | Computed from exp08a JSON, restricted to `clustering_type ∈ {kmeans, agglomerative, spectral}` |
| E8a all valid: r = +0.40 over 83 points | 96 − 13 (HDBSCAN k=0 with missing sil) = 83 ✓; Pearson = +0.401 | **OK** | |
| E8b all valid: r = -0.52 over 86 points | 96 − 10 (HDBSCAN k=0 missing) = 86 ✓; Pearson = -0.520 | **OK** | |
| E8b agglomerative-only: r = -0.25 over 32 cells | 2 reductions × 2 (k=30, k=87) × 8 extractors = 32 ✓; Pearson = -0.251 | **OK** | |

---

## Catálogo de experimentos (tab:catalogo-experimentos, L19–35)

| Row | Claim | Data | Verdict |
|---|---|---|---|
| E1 | 12 extractors × 3 sizes, K=3 | exp01: 36 runs / 3 N / K=3 | **OK** |
| E2 | 6 reductions × 3 sizes, K=3 | exp02: 18 runs / 3 N / K=3 | **OK** |
| E3 | 14 clusterers × 3 sizes, K=3 | exp03: 42 runs / 3 N / K=3 | **OK** |
| E4 | 6 mcs × 3 sizes, K=3 | exp04: 18 runs / 3 N / K=3 | **OK** |
| E5 | 4 segmenters × 3 sizes, K=3 | exp05: 12 runs (N varies because seg changes crop count) / K=3 | **OK** |
| E6 | 3 backends × 4 sizes {500,1000,2500,6416}, K=5 | exp06: 12 / N matches / K=5 | **OK** |
| E7 | 5 algos × 6 sizes {250,500,1000,2000,3500,6416}, K=5 | exp07: 30 / N matches / K=5 | **OK** |
| E8a | 294 recortes, 4 clases, K=3 | exp08a: image_count=294, n_classes=4, K=3 | **OK** |
| E8b | 185 recortes, 87 autores, K=3 | exp08b: image_count=185, n_classes=87, K=3 | **OK** |
| E9 | N=6416, K=5 | exp09: image_count=6416, K=5 | **OK** |

---

## Proposal coverage (`doc/enunciado.md`)

The proposal lists four explicit deliverables. Mapping to results.tex:

| Proposal request | Where addressed | Verdict |
|---|---|---|
| "describir y caracterizar el comportamiento de varios métodos para el cálculo de similitud y la creación de agrupaciones" | E1–E5 (calibración del pipeline), E6 (similarity backends), E9 (HNSW tuning) | **OK — covered** |
| "describir el coste computacional en función del tamaño del conjunto de imágenes, de forma experimental" | E7 (escalabilidad por etapa, log-log slopes), E6 (latency vs N), E9 (Pareto) | **OK — fully covered, this is the proposal's headline goal** |
| "estudio con métricas de carácter no supervisado de las agrupaciones resultantes" | Intrinsic metrics (silhouette, CH, DB, noise, csCV) tabulated in every experiment (E1–E5, E7) | **OK — covered** |
| "evaluación supervisada para un subconjunto en el que estén disponibles etiquetas" | E8a (style) and E8b (author), plus the silhouette–ARI correlation cross-check | **OK — covered** |

**No scope shortfalls.** The results chapter delivers all four. **Extras
the proposal did not require** but that strengthen the work: chance-
corrected metrics (ARI), retrieval metrics (P@5, MAP@5, MRR, R@5), and the
explicit silhouette–ARI correlation analysis that legitimises using
intrinsic metrics as a proxy in the calibration subsection. None of the
extras contradict the proposal.

**Tone.** The proposal explicitly asks for the topic to be treated as a
neutral visual phenomenon; results.tex respects that throughout (purely
technical framing, no comments on graffiti as a social phenomenon). OK.

---

## Appendix: arithmetic for E7 slopes

`exp07` runs, identical hdbscan-l* records carry the ingest and reduction
time for each N. Wall-times in seconds, all from JSON:

| N | ingest | reduction | hdbscan | kmeans | dbscan | optics | agglo | search |
|---|---|---|---|---|---|---|---|---|
| 250 | 58.746 | 0.253 | 0.0038 | 0.0025 | 0.0035 | 0.2082 | 0.0019 | 3.663 |
| 500 | 115.689 | 0.603 | 0.0069 | 0.0027 | 0.0055 | 0.3469 | 0.0046 | 16.321 |
| 1000 | 231.618 | 1.514 | 0.0144 | 0.0033 | 0.0103 | 0.6310 | 0.0117 | 63.277 |
| 2000 | 455.911 | 4.542 | 0.1106 | 0.0047 | 0.0228 | 1.3054 | 0.0427 | 306.806 |
| 3500 | 775.986 | 11.870 | 0.1863 | 0.0066 | 0.0471 | 2.2819 | 0.1405 | 995.517 |
| 6416 | 1294.949 | 6.237 | 0.3580 | 0.0098 | 0.0768 | 4.3217 | 0.5988 | 3266.601 |

Slope formula: `slope = sum((lx-mx)(ly-my)) / sum((lx-mx)²)` with
`lx = ln(N)`, `ly = ln(t)`.

Computed slopes (full 250…6416 / window 500…3500):
ingest 0.961 / 0.978; UMAP 1.154 / 1.533;
KMeans 0.429 / 0.456; DBSCAN 0.995 / 1.103; OPTICS 0.945 / 0.976;
HDBSCAN 1.525 / 1.835; agglomerative 1.766 / 1.763;
search 2.104 / 2.129.

All match the values printed in tab:e7-slopes within ±0.01.
