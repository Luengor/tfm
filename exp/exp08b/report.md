# §8b — Supervised Validation on Author Crops: Report

**Source:** `exp/exp08b/output/benchmark_20260529T145725Z.json`
**Date:** 2026-05-29 · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 96/96 runs `success`.

## Setup

Pillar 3 (supervised validation). Question: when ground-truth **author**
labels are available, which embedding × reduction × clustering pipeline best
recovers them — and does the graffiti **author** head beat its pretrained
backbone, symmetrically to the style head's §8a win?

Ground truth: [`data/author/eval_crop/`](../../data/author/eval_crop/) — 185
pre-segmented crops across **87 authors**, drawn from the Salamanca primary
dataset. The author heads were trained on a **disjoint 169-crop split from the
Cuenca dataset**, so §8b is a *cross-city* held-out test (Cuenca-trained,
Salamanca-evaluated): a positive result would mean author features transfer
across the geographic/dataset shift, not just across held-out authors within
one city. Segmenter is `identity` (required for extrinsic metrics); no `limit`
(full 185 each run).

The label set is brutal for clustering: **53 singletons, 17 pairs, top author
19 crops** (mean class size ≈2.1). This caps achievable ARI/F1 and inflates
NMI by construction — so absolute scores must be read against the random
baseline, not in isolation.

Fixed / varied axes:

| Axis | Values |
|---|---|
| Embedding (8) | `dinov2_vits14`, `dinov2_graffiti_style_head`, `dinov2_graffiti_author_head`, `mobilenet_v3`, `mobilenet_v3_graffiti_style_head`, `mobilenet_v3_graffiti_author_head`, `clip_vit_b32`, `resnet50` |
| Reduction (2) | `identity` (raw space), UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Clustering (6) | `agglomerative` (avg) n_clusters∈{30, 87}, `hdbscan` mcs∈{2, 3, 5, 10} |
| Storage | SQLite · Segmenter | `identity` · Repeats | `K=3` |

8 × 2 × 6 = 96 runs. The §8b grid drops `kmeans`/`spectral` (no principled
handling of singleton classes at oracle k=87) and uses agglomerative at a
moderate-k proxy (30) plus oracle (87), with HDBSCAN swept low (2, 3 are the
smallest meaningful sizes given the singleton tail).

> **Reading note — every extrinsic number here lives near the floor; the
> `*_no_noise` and NMI columns inflate exactly the cells you should distrust.**
> NMI is driven to ~0.78 by the k=87 agglomerative partitions purely because 87
> tiny clusters share marginal information with 87 tiny classes — the
> documented many-tiny-clusters inflation mode, not recovery. The `_no_noise`
> ARI peaks at the HDBSCAN mcs=2 cells (e.g. `mobilenet_v3_graffiti_style_head`
> identity hdb2 posts ARI_no_noise 0.204 **while discarding 57.8 % of points as
> noise**) — it grades only the dense survivor core. Read ARI (default,
> chance-corrected) as the headline and always pair it with `noise_ratio`.

## The central result: no embedding meaningfully recovers authors

The §8a result was sharp and positive (style head ARI 0.698). **§8b is a flat
negative.** The best default ARI over all 96 runs is **0.076** (clip,
umap-agglomerative-87); every embedding's best cell sits in a narrow
**0.041–0.076** band — barely above the chance-corrected zero that ARI assigns
to a random partition.

`best` = highest default ARI over the 12 (reduction × clustering) cells for
that embedding; the other columns are that same winning cell.

| Embedding | best ARI | ±std | NMI | F1 | sil | noise | cls | winning cell |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `clip_vit_b32` | **0.076** | 0.003 | 0.789 | 0.090 | 0.003 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3_graffiti_author_head` | 0.061 | 0.005 | 0.787 | 0.074 | 0.011 | 0.00 | 87 | umap-agglomerative87 |
| `dinov2_graffiti_style_head` | 0.060 | 0.007 | 0.633 | 0.090 | 0.026 | 0.00 | 30 | umap-agglomerative30 |
| `dinov2_vits14` | 0.055 | 0.000 | 0.754 | 0.079 | 0.096 | 0.00 | 87 | identity-agglomerative87 |
| `dinov2_graffiti_author_head` | 0.051 | 0.010 | 0.636 | 0.081 | 0.027 | 0.00 | 30 | umap-agglomerative30 |
| `resnet50` | 0.049 | 0.005 | 0.778 | 0.064 | 0.028 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3` | 0.046 | 0.014 | 0.780 | 0.060 | 0.023 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3_graffiti_style_head` | 0.041 | 0.007 | 0.777 | 0.055 | 0.001 | 0.00 | 87 | umap-agglomerative87 |

The per-cell std is small (0.000–0.014), so each cell is individually stable
across the K=3 repeats — but every embedding's best ARI sits in a **0.041–0.076
band, a 0.035 spread that is ~5 % of §8a's single winning cell (0.698)**.
The cells are reproducible; they are reproducibly at the floor. **On this task
the embedding choice does not separate.** The high NMI column (0.75–0.79 for
every k=87 winner) is the inflation trap, not a success: it is high precisely
*because* both predicted and true partitions are fragmented into ~87 tiny
groups.

## Author heads do NOT beat their backbones — the §8a symmetry fails

§8a's headline was a strongly diagonal-dominant transfer matrix: the style head
beat its backbone (+0.342 / +0.204) and the author head *fell below* it. §8b
hypothesis 1 expected the mirror image — author heads winning the author task.
**The data refutes it.** Best default ARI, both tasks side by side:

| Family | task | pretrained | style head | author head |
|---|---|--:|--:|--:|
| DINOv2 | **author (§8b)** | 0.055 | 0.060 | **0.051** |
| DINOv2 | style (§8a) | 0.356 | **0.698** | 0.234 |
| MobileNetV3 | **author (§8b)** | 0.046 | 0.041 | **0.061** |
| MobileNetV3 | style (§8a) | 0.356 | **0.560** | 0.267 |

On §8b the author head is **not** the diagonal peak. For DINOv2 the author head
(0.051) lands *below* both its own backbone (0.055) and the style head (0.060).
For MobileNetV3 the author head (0.061) edges ahead — but by +0.015 ARI, well
inside noise and at an absolute level indistinguishable from random. There is
no diagonal dominance on the author axis: the transfer matrix is row-dominant
(everything is near zero on §8b) rather than diagonal.

This is the cross-city domain shift the brief flagged: the author heads learned
Cuenca-author-discriminative features that **do not transfer to Salamanca
authors**. §8a (within-city style held-out) showed genuine transfer; §8b
(cross-city author held-out) shows the heads do not generalise across the
geographic boundary. The specificity claim from §8a stands for *style* but the
symmetric *author* claim cannot be closed on this evaluation.

## Supervised retrieval tells the same story — at the floor

Similarity search (top_k=5, raw stored embeddings, reduction-independent)
scored against author labels. Random P@5 for this label distribution is
**≈0.024 per neighbor**; the observed values are ~2–4× that but still near the
floor, and — critically — the author heads do **not** lead.

| Embedding | P@5 | MAP@5 | MRR | R@5 |
|---|--:|--:|--:|--:|
| `clip_vit_b32` | **0.091** | 0.252 | 0.262 | 0.268 |
| `dinov2_graffiti_author_head` | 0.088 | 0.258 | 0.261 | 0.273 |
| `resnet50` | 0.085 | 0.246 | 0.256 | 0.236 |
| `mobilenet_v3` | 0.083 | 0.240 | 0.245 | 0.251 |
| `dinov2_vits14` | 0.081 | 0.248 | 0.251 | 0.268 |
| `dinov2_graffiti_style_head` | 0.067 | 0.240 | 0.243 | 0.192 |
| `mobilenet_v3_graffiti_style_head` | 0.067 | 0.232 | 0.233 | 0.179 |
| `mobilenet_v3_graffiti_author_head` | 0.059 | 0.180 | 0.187 | 0.171 |

The DINOv2 author head (0.088) is statistically tied with the off-the-shelf
encoders (clip 0.091, resnet50 0.085) and the MobileNet author head is the
single **worst** retriever (0.059). Fine-tuning for authorship bought nothing
measurable on held-out cross-city authors — consistent with the clustering
result.

The `mobilenet_v3_graffiti_author_head` is the cleanest tell that these numbers
are noise-dominated: it is the **best** author head on clustering ARI (0.061,
its family's only nominal "win") yet the **worst** of all 8 embeddings on
retrieval P@5 (0.059). A genuine author-discriminative representation would rank
consistently on both axes; ranking top on one and bottom on the other for the
same embedding is what a field with no real signal looks like.

## Internal–external correlation *inverts* relative to §8a

§8a found silhouette a defensible proxy at fixed k (Pearson r = +0.66 on the
oracle-k partitional runs). **§8b reverses the sign:**

- **r = −0.52** over all 86 runs with a valid ARI and >1 cluster.
- **r = −0.25** over the 32 agglomerative runs alone.

Silhouette and ARI now pull in *opposite* directions. Cause: the highest-ARI
cells are k=87 agglomerative partitions, which post near-zero silhouette
(0.00–0.03) because 87 tiny clusters cannot be cohesive; the highest-silhouette
cells are the degenerate HDBSCAN k=2/k=3 splits (sil up to 0.52 on the DINOv2
style head) with ARI ≈ 0. Silhouette rewards few fat blobs; recovering 87
authors demands many fine clusters. **On fine-grained labels, silhouette is not
just a weak proxy — it is anti-correlated with label agreement.** This bounds
the §8a finding: silhouette tracks coarse blob structure, useful when the true
k is small (4 styles) and actively misleading when it is large (87 authors).

## HDBSCAN behaviour: collapse and all-noise

HDBSCAN does not produce a usable author partition under any setting:

- **identity space, mcs=10** collapses to **0 clusters (noise 1.0)** for 7 of 8
  embeddings — raw space has no density structure HDBSCAN will accept at that
  threshold. mcs=5 identity is all-noise for three more
  (`dinov2_graffiti_author_head`, `mobilenet_v3`,
  `mobilenet_v3_graffiti_author_head`).
- **mcs=2** is the only setting that fragments into many clusters (29–57),
  but at **40–58 % noise** — the source of the inflated `_no_noise` columns.
- The DINOv2 **style** head is an outlier: identity hdb3/5/10 all collapse to a
  stable **k=2** split with high silhouette (≈0.50) and ARI ≈ 0 — the same
  degenerate binary-split trap documented in §4 and §8a.

Repeat (in)stability sits at the collapse boundary as in §4/§8a: under UMAP,
mcs=3 flips cluster count across seeds (e.g. `mobilenet_v3_graffiti_style_head`
umap hdb3 → `clusters_per_repeat = [2, 21, 26]`). Agglomerative rows are stable
(`[30,30,30]` / `[87,87,87]`).

## Summary — cost

Cost is not the §8b deliverable (§§6–7/9 own it) and quality is storage-
invariant, but for completeness — ingest dominates, paid once per embedding
(shared across that embedding's 12 cells); 185 crops make every stage cheap:

| Embedding | ingest wall (s) | throughput (ips) | sim-search wall (s) |
|---|--:|--:|--:|
| dinov2 (×3 heads) | 4.6–5.4 | 34–40 | ~1.9–2.1 |
| mobilenet (×3 heads) | 2.8–2.9 | 63–66 | ~5.7–6.1 |
| clip_vit_b32 | 3.7 | 50.0 | 2.49 |
| resnet50 | 3.5 | 52.3 | 9.20 |

UMAP reduction 0.19–0.24 s; clustering 0.001–0.12 s. Similarity-search wall
time again scales with embedding **dimension** under the SQLite linear scan
(DINOv2 384-d ~2 s, ResNet50 2048-d ~9 s) — a storage-backend property (§6/§9),
not the head.

## Hypotheses — disposition

1. **Author head > backbone (symmetric to §8a).** ❌ Refuted. DINOv2 author head
   (0.051) ≤ its backbone (0.055); MobileNet author head (+0.015) edges ahead
   but at a random-floor absolute level. No genuine cross-city transfer.
2. **Cross-task specificity / diagonal-dominant matrix.** ❌ Refuted on the
   author axis. §8b is row-dominant (all embeddings near zero), not diagonal.
   Style specificity (§8a) holds; author specificity does not survive the
   Cuenca→Salamanca domain shift.
3. **ResNet50 closes the gap on fine-grained identity (local texture).** ⚠
   Weakly consistent: ResNet50 is mid-pack on ARI (0.049) and 3rd on retrieval
   P@5 (0.085), no longer the weakest — but the whole field is at the floor, so
   this is "closes the gap by everyone failing equally," not a CNN win.
4. **HDBSCAN auto-k in the right order of magnitude (tens).** ❌/⚠ mcs=2 finds
   tens of clusters (29–57) but only at 40–58 % noise; every lower-noise setting
   collapses to k=2 or all-noise. No usable author partition emerges.
5. **identity ≥ UMAP on the fine-tuned head.** ⚠ Indeterminate at this signal
   level. Best cells are nominally **UMAP**-agglomerative (only `dinov2_vits14`
   peaks in identity), but with all ARI at the floor the identity-vs-UMAP
   ordering is itself noise-dominated and cannot be read as a real preference.
   §8a (where the head had concentrated discriminative axes) is the only place
   this hypothesis is testable; §8b neither confirms nor refutes it.

## Conclusion for downstream experiments

1. **§8b is a negative result, and that is the finding.** Cross-city author
   recovery fails for every embedding (best ARI 0.076, ~random). The thesis
   should report it as an honest bound on the heads' generalisation: the author
   head transfers *within* its training domain (Cuenca) but the held-out
   Salamanca evaluation shows it does **not** cross the city boundary. The §8a
   style result remains the positive transfer claim; §8b qualifies its scope.
2. **The §8a specificity claim is half-closed.** Style-head specificity is
   demonstrated (§8a); author-head specificity is **not** demonstrable on this
   evaluation because the author head never beats baseline. Any "two heads
   encode different information" statement must cite §8a only and flag §8b as
   confounded by domain shift + singleton-heavy labels.
3. **Silhouette is anti-correlated with label agreement on fine-grained labels**
   (r = −0.52). This is a strong caveat for §§1–4: silhouette/CH defend the
   *coarse* (few-cluster) unsupervised metrics, but must never be used to argue
   for fine-grained (many-author-scale) cluster quality. The §8a (+0.66) vs §8b
   (−0.52) split bounds exactly what silhouette tracks: blob cohesion, not
   identity.
4. **NMI is uninformative on this label set.** The 0.75–0.79 NMI on every k=87
   winner is the singleton-inflation artefact the brief predicted; report ARI
   (and the random P@5 baseline), never NMI, for fine-grained author validation.

**Degenerate cells (all-noise, 0 clusters, no metrics):** identity-HDBSCAN
mcs=10 for `clip_vit_b32`, `dinov2_vits14`, `dinov2_graffiti_author_head`,
`mobilenet_v3`, `mobilenet_v3_graffiti_author_head`,
`mobilenet_v3_graffiti_style_head`, `resnet50`; plus identity-HDBSCAN mcs=5 for
`dinov2_graffiti_author_head`, `mobilenet_v3`,
`mobilenet_v3_graffiti_author_head`. Listed here once; excluded from the tables
above.
