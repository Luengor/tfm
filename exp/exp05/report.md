# §5 — Segmenter Impact: Report

**Source:** `exp/exp05/output/benchmark_20260529T103834Z.json`
**Date:** 2026-05-29 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 12/12 runs `success`.

## Setup

Pillar 2 (unsupervised quality) with a cost side-effect (pillar 1). Question:
does YOLO cropping improve clustering quality over embedding the whole image,
and which detector works best — and what does the crop step cost?

Fixed baseline (one axis varied = `segmenter`):

| Component | Value |
|---|---|
| Embedding | `dinov2_graffiti_style_head` (DINOv2 ViT-S/14 + style head, L2-normalised) |
| Reduction | UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Clustering | HDBSCAN, `min_cluster_size=5` |
| Segmenter | **varied** |
| Storage | SQLite |
| Repeats | `K=3` (reduction+clustering looped; iter 0 dropped) |

4 segmenters × 3 corpus sizes `{250, 1000, 6416}` = 12 runs. Segmenters:
`identity` (whole image), `yolo` with `yolo11s.pt`, `yolo11m.pt`, and
`yolo11m-train-10.pt` (graffiti-fine-tuned detector). YOLO rows use
`threshold=0.5, merge_threshold=0.8, padding=0.05`.

> **Deviation from the brief.** `infos/experiments.md` §5 specifies *five*
> segmenters (`yolo11{n,s,m}` + fine-tuned + identity = 15 runs). The
> `yolo11n` row is **absent** from this benchmark — only 12 runs were
> executed. The `n` detector should be added for completeness, or the brief
> trimmed to match.

> **Reading note — the corpus being clustered is not constant across rows.**
> `--limit` caps *images read*, but YOLO emits one crop per detected box and
> **drops images with zero detections** (`segmenters.py:17-40`,
> `configuration.py:30-46` — no full-image fallback). So `image_count` is the
> number of *crops*, and it diverges from `limit`: base `yolo11{s,m}` produce
> ≈1.03 crops/image (net ~one graffiti per image after drops + multi-box
> images cancel out), while the fine-tuned `yolo11m-train-10` produces
> **2.0–2.4 crops/image**. Every row therefore clusters a *different point
> set* of a *different size*. Silhouette/CH/DB are not strictly comparable
> across segmenters (see the central hazard below); read them jointly with
> `crops`, `noise_ratio` and `cluster_count`.

## Summary — quality

`crops` = `image_count` (units actually clustered), `c/img` = crops per input
image, `cls` = cluster count, `noise` = fraction labelled −1, `sil` =
silhouette (non-noise only, higher better), `silM` = macro silhouette
(per-cluster mean, higher better), `CH` = Calinski–Harabasz (higher better),
`DB` = Davies–Bouldin (lower better), `csCV` = cluster-size CV (lower = more
balanced).

| n | segmenter | crops | c/img | cls | noise | sil | silM | CH | DB | csCV |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | **identity** | 250 | 1.00 | 23 | 0.063 | **0.199** | **0.258** | 18.1 | 1.51 | 0.53 |
| 250 | yolo11s | 251 | 1.00 | 21 | 0.076 | 0.155 | 0.192 | 15.8 | 1.75 | **0.43** |
| 250 | yolo11m | 254 | 1.02 | 23 | 0.085 | 0.185 | 0.245 | 16.5 | 1.55 | 0.54 |
| 250 | yolo11m_ft | 461 | 1.84 | 49 | 0.102 | 0.189 | 0.239 | **21.9** | 1.55 | 0.68 |
| 1000 | **identity** | 1000 | 1.00 | 96 | **0.106** | 0.159 | 0.209 | 18.9 | 1.68 | 0.60 |
| 1000 | yolo11s | 1039 | 1.04 | 90 | 0.189 | 0.138 | 0.184 | 17.0 | 1.78 | 0.57 |
| 1000 | yolo11m | 1034 | 1.03 | 90 | 0.159 | 0.137 | 0.194 | 16.0 | 1.76 | 0.67 |
| 1000 | yolo11m_ft | 2001 | 2.00 | 195 | 0.164 | **0.175** | **0.228** | **25.8** | **1.63** | 0.56 |
| 6416 | **identity** | 6416 | 1.00 | 563 | **0.199** | **0.151** | 0.208 | 33.1 | **1.70** | **0.75** |
| 6416 | yolo11s | 6590 | 1.03 | 502 | 0.225 | 0.127 | 0.185 | 30.4 | 1.81 | 0.75 |
| 6416 | yolo11m | 6603 | 1.03 | 506 | 0.230 | 0.131 | 0.190 | 29.5 | 1.80 | 0.73 |
| 6416 | yolo11m_ft | 15362 | 2.39 | 1049 | 0.298 | 0.122 | 0.199 | **36.2** | 1.86 | 1.06 |

## Summary — cost

The segmenter axis is a **cost axis** (unlike §4). Cropping adds a detection
forward-pass per image, more VRAM for the detector, and — for the fine-tuned
detector — multiplies the number of embeddings and the downstream
reduction/clustering load. `ing_s` = ingest wall time, `ips` = ingest
throughput (crops/s), `VRAM` = ingest peak VRAM, `red_s`/`clu_s` =
reduction/clustering wall time (scale with crop count).

| n | segmenter | ing_s | ips | VRAM MB | red_s | clu_s |
|--:|---|--:|--:|--:|--:|--:|
| 6416 | identity | 1235.5 | 5.19 | 100.2 | 6.20 | 0.34 |
| 6416 | yolo11s | 1777.0 | 3.71 | 195.2 | 6.53 | 0.36 |
| 6416 | yolo11m | 1811.6 | 3.64 | 272.6 | 6.42 | 0.35 |
| 6416 | yolo11m_ft | 2152.3 | 7.14 | 492.3 | 7.24 | 1.47 |

(250 / 1000 rows follow the same ordering at smaller magnitude — see JSON.)

Reading the cost table:

- **Per-image ingest cost rises with detector weight.** At n=6416,
  seconds-per-input-image is 0.19 (identity) → 0.28 (s/m) → 0.34 (fine-tuned).
  Base YOLO adds ~44 % over identity; the fine-tuned detector adds ~74 %.
- **`ips` is misleading.** The fine-tuned row posts the *highest* throughput
  (7.14 crops/s) yet is the *slowest* per image — throughput is crops÷wall,
  and the fine-tuned detector emits 2.4× more crops. Compare per-input-image
  cost, not `ips`, across segmenters.
- **VRAM scales with detector size:** 100 MB (no detector) → 195 (s) → 273 (m)
  → 492 (fine-tuned).
- **Downstream cost follows the crop multiplier.** Reduction and clustering
  are driven by point count, so the fine-tuned row's 15 362 crops push
  clustering to 1.47 s (vs 0.34 s for identity's 6 416 points). Reduction
  stays ~6–7 s; clustering is the part that grows.

## The central reading hazard: rows do not cluster the same data

In §4 the hazard was silhouette peaking at degenerate k=2 cells. Here it is
subtler and structural: **each segmenter feeds HDBSCAN a different number of
different points.**

- `crops` ranges from 6 416 (identity) to 15 362 (fine-tuned) at n=6416.
  Silhouette is a per-point average over *those* points; CH is a
  variance ratio that grows with both cluster count and sample size. The
  fine-tuned row's CH=36.2 (highest in the table) coincides with the *lowest*
  silhouette (0.122). **CH and silhouette diverge on the fine-tuned row** —
  the standard signal that one of them is reading something other than
  cohesion (here, CH is sensitive to the row's larger sample and cluster
  count), so neither can be trusted in isolation for the cross-segmenter call.
- A crop is a tighter, more homogeneous unit than a whole image, which
  inflates intra-cluster cohesion *mechanically* — so a small silhouette edge
  for a cropping segmenter is not evidence the crop *helped clustering*; it
  may just be the easier geometry of smaller patches.

The defensible cross-segmenter comparison is the **joint criterion**:
non-trivial cluster count, *moderate* noise, *balanced* clusters — plus an
explicit eye on the crop multiplier. On that criterion the picture is clear
(next section).

## Does cropping improve clustering quality?

**No — not at thesis scale.** The answer is size-dependent, and the full
corpus (n=6416) is the row that matters:

1. **`identity` wins at n=6416.** It has the highest silhouette (0.151),
   lowest noise (0.199), lowest DB among non-fine-tuned rows, and balanced
   clusters (csCV 0.75). Every YOLO row at full corpus is worse on silhouette
   and noise.
2. **Base `yolo11s` / `yolo11m` are dominated by identity on the metrics that
   matter.** They crop ≈1.03×/image — barely changing the unit of analysis —
   so they reproduce identity's partition while *adding* detection cost, VRAM,
   and a few mis-crops. Result: lower silhouette, higher noise, higher DB at
   every size. (`yolo11s@250` posts a better csCV than identity, 0.43 vs 0.53,
   but that is fewer/larger clusters, not better cohesion.) Cropping ~one box
   per image buys nothing here and costs ~44 % ingest.
3. **The fine-tuned detector is competitive only at small/mid n, then
   collapses.** At n=1000 it posts the best silhouette (0.175), best CH
   (25.8) and best DB (1.63) of the four — though that lead is confounded by
   its 2× crop rate, which clusters tighter units and inflates cohesion
   mechanically rather than proving the crop helped. But at n=6416 it
   degenerates: noise jumps to **0.298** (worst), cluster-size CV to **1.06**
   (worst — highly imbalanced), silhouette to the table minimum (0.122). The
   2.4× crop explosion produces 1 049 clusters dominated by a long tail of
   tiny ones plus a large noise pool.

So cropping does not clear the bar the brief asks about: the trade-off
(detection cost + crop-count blow-up) is **not** worth it for whole-corpus
unsupervised clustering. The fine-tuned detector's small-n edge is real but
does not survive scaling, which is the regime the thesis targets.

## Does the ranking shift with `n`?

Yes, and the direction is the headline:

- **Small n (250):** identity and the fine-tuned detector are within noise on
  silhouette (0.199 vs 0.189); fine-tuned leads CH. Quality differences are
  marginal.
- **Mid n (1000):** the fine-tuned detector scores best on silhouette/CH/DB —
  but the comparison is confounded by its 2× crop multiplier (the
  unit-of-analysis hazard above applies here too: 2 001 crops vs 1 000
  images), so this is "helps *if* you accept the changed unit," not a clean win.
- **Full n (6416):** identity pulls ahead and the fine-tuned detector posts
  the *worst* noise and balance. The crop multiplier that helped at n=1000
  hurts at n=6416 because the extra crops are increasingly low-quality
  detections that land in noise or micro-clusters.

The base detectors track identity (slightly worse) at all three sizes — no
crossover, because they never meaningfully change the unit of analysis.

## Repeat stability

All 12 cells are stable across the K=3 repeats; no collapse boundary as in §4.
Cluster counts vary only by UMAP seed:

- `identity@6416`: 549 / 537 / 563 (±~2 %), consistent with §2/§4.
- `yolo11m_ft@6416`: 1041 / 1048 / 1049 (±<1 %).
- `yolo11m_ft@250`: 44 / 43 / 49 — the widest spread, but still a single
  partition family, not a degeneracy.

Detection itself is deterministic (single ingest per row), so all repeat
variance comes from UMAP, as elsewhere.

## Conclusion for downstream experiments

1. **`identity` is the right baseline segmenter** for the thesis pipeline and
   the rest of the catalogue. At the full 6 416-image corpus it gives the best
   silhouette, lowest noise, and balanced clusters at the lowest cost — the
   experiments.md baseline choice is confirmed empirically.
2. **Base `yolo11s` / `yolo11m` should not be used for clustering.** They are
   strictly dominated: ≈1.03 crops/image means they barely change the
   partition while adding ~44 % ingest cost and detection noise. Cropping only
   makes sense when the detector genuinely re-segments the corpus.
3. **The fine-tuned detector is a small-corpus / supervised-crop tool, not a
   whole-corpus clustering front-end.** Its 2.0–2.4× crop multiplier wins at
   n≤1000 but collapses to 30 % noise and csCV>1 at full scale, and triples
   downstream clustering cost. (It remains the right segmenter for producing
   the labelled eval crops in §8.)
4. **Cross-segmenter metrics must be read against the crop count.** Because
   each segmenter clusters a different-sized point set, CH and silhouette can
   disagree (they do on the fine-tuned row, where CH reads sample size and
   silhouette reads cohesion). This mirrors §4's lesson: never rank on a
   single intrinsic metric — use the joint criterion plus the unit-of-analysis
   caveat.
