# §8b — Supervised Validation on Author Crops: Report

**Source:** `exp/exp08b/output/benchmark_20260529T145725Z.json` (all 96 runs),
`exp/exp08b/output/benchmark_20260601T150151Z.json` (author heads, re-run;
supersedes the 29-May author-head rows — 24 cells covering both author heads
× all 12 cells) and `exp/exp08b/output/benchmark_20260602T105759Z.json` (style
heads, re-run; supersedes the 29-May style-head rows — 24 cells covering both
style heads × all 12 cells).
**Date:** 2026-05-29 (main sweep), 2026-06-01 (author-head re-run), 2026-06-02 (style-head re-run) · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 96/96 + 24/24 + 24/24 runs `success`.

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
> ARI peaks at the HDBSCAN mcs=2 cells (e.g. `resnet50` identity hdb2 posts
> ARI 0.008 vs **ARI_no_noise 0.157 while discarding 54.6 % of points as
> noise**) — it grades only the dense survivor core. Read ARI (default,
> chance-corrected) as the headline and always pair it with `noise_ratio`.

## The central result: no embedding meaningfully recovers authors

The §8a result was sharp and positive (style head ARI 0.722). **§8b is
floor-level across the board.** The best default ARI over all 96 runs is
**0.076** (clip, umap-agglomerative-87); every embedding's best cell sits in a
narrow **0.041–0.076** band — barely above the chance-corrected zero that ARI
assigns to a random partition.

`best` = highest default ARI over the 12 (reduction × clustering) cells for
that embedding; the other columns are that same winning cell.

| Embedding | best ARI | ±std | NMI | F1 | sil | noise | cls | winning cell |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `clip_vit_b32` | **0.076** | 0.003 | 0.789 | 0.090 | 0.003 | 0.00 | 87 | umap-agglomerative87 |
| `dinov2_graffiti_author_head` | 0.065 | 0.005 | 0.647 | 0.093 | 0.052 | 0.00 | 30 | umap-agglomerative30 |
| `dinov2_vits14` | 0.055 | 0.000 | 0.754 | 0.079 | 0.096 | 0.00 | 87 | identity-agglomerative87 |
| `dinov2_graffiti_style_head` | 0.054 | 0.003 | 0.628 | 0.085 | 0.037 | 0.00 | 30 | umap-agglomerative30 |
| `resnet50` | 0.049 | 0.005 | 0.778 | 0.064 | 0.028 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3_graffiti_author_head` | 0.048 | 0.005 | 0.627 | 0.077 | 0.021 | 0.00 | 30 | umap-agglomerative30 |
| `mobilenet_v3` | 0.046 | 0.014 | 0.780 | 0.060 | 0.023 | 0.00 | 87 | umap-agglomerative87 |
| `mobilenet_v3_graffiti_style_head` | 0.041 | 0.003 | 0.629 | 0.070 | 0.020 | 0.00 | 30 | umap-agglomerative30 |

The per-cell std is small (0.000–0.014), so each cell is individually stable
across the K=3 repeats — but every embedding's best ARI sits in a **0.041–0.076
band, a 0.035 spread that is ~5 % of §8a's single winning cell (0.722)**.
The cells are reproducible; they are reproducibly at the floor. **On this task
the embedding choice does not separate.** The high NMI column (0.75–0.79 for
every k=87 winner) is the inflation trap, not a success: it is high precisely
*because* both predicted and true partitions are fragmented into ~87 tiny
groups.

## Author heads beat their backbones — but at the floor

§8a's headline was a strongly diagonal-dominant transfer matrix: the style head
beat its backbone (+0.366 / +0.271). §8b hypothesis 1 expected the mirror image
— author heads winning the author task. **After the 2026-06-01 author-head and
2026-06-02 style-head re-runs the data supports it on both families.** Best
default ARI, both tasks side by side:

| Family | task | pretrained | style head | author head |
|---|---|--:|--:|--:|
| DINOv2 | **author (§8b)** | 0.055 | 0.054 | **0.065** |
| DINOv2 | style (§8a) | 0.356 | **0.722** | 0.302 |
| MobileNetV3 | **author (§8b)** | 0.046 | 0.041 | **0.048** |
| MobileNetV3 | style (§8a) | 0.356 | **0.627** | 0.318 |

On §8b the author head **is** the diagonal peak inside each fine-tuned family:
DINOv2 author head (0.065) lands above both its backbone (0.055, +0.010) and
the style head (0.054, +0.011) — after the 2026-06-02 style-head re-run the
style head now sits *below* its own backbone on the author task, sharpening
the diagonal. MobileNet author head (0.048) edges its own backbone (0.046,
+0.002) and the style head (0.041, +0.007). The 2×2 cabezas-vs-task matrix is
therefore fully diagonal-dominant on both axes — style head wins style, author
head wins author, and each *cross* cell (style-on-author / author-on-style)
sits below its backbone — but the **magnitudes are radically asymmetric**:
+0.366 / +0.271 for style (useful), +0.010 / +0.002 for author (inside noise).
CLIP (0.076) still tops the absolute author ranking, so author-head supremacy
holds only within its own family, not against all baselines.

The cross-city domain shift the brief flagged (Cuenca-trained →
Salamanca-evaluated) is therefore consistent with **degraded but non-zero**
transfer rather than outright failure: the author heads keep a small
discriminative edge over their backbones across the geographic boundary, but
the absolute scale collapses by an order of magnitude relative to §8a's
within-domain style transfer (0.722 vs 0.065). The specificity claim from §8a
stands strongly for *style*; the symmetric *author* claim is supported in
direction (sign matches) but at a magnitude indistinguishable from random
floor on the absolute axis.

## Supervised retrieval tells the same story — at the floor

Similarity search (top_k=5, raw stored embeddings, reduction-independent)
scored against author labels. Random P@5 for this label distribution is
**≈0.024 per neighbor**; the observed values are ~2–4× that but still near the
floor, and the author heads sit in the mid-pack, not at the top:

| Embedding | P@5 | MAP@5 | MRR | R@5 |
|---|--:|--:|--:|--:|
| `clip_vit_b32` | **0.091** | 0.252 | 0.262 | 0.268 |
| `resnet50` | 0.085 | 0.246 | 0.256 | 0.236 |
| `mobilenet_v3` | 0.083 | 0.240 | 0.245 | 0.251 |
| `dinov2_vits14` | 0.081 | 0.248 | 0.251 | 0.268 |
| `mobilenet_v3_graffiti_author_head` | 0.078 | 0.217 | 0.222 | 0.233 |
| `dinov2_graffiti_author_head` | 0.077 | 0.228 | 0.232 | 0.254 |
| `mobilenet_v3_graffiti_style_head` | 0.071 | 0.196 | 0.200 | 0.176 |
| `dinov2_graffiti_style_head` | 0.065 | 0.226 | 0.229 | 0.192 |

The two author heads (0.078 / 0.077) cluster together in the middle band, the
off-the-shelf encoders (clip / resnet / mob / dinov2) sit slightly above them,
and the two **style** heads are the worst retrievers (0.071 / 0.065 after the
2026-06-02 re-run) — consistent with the style-vs-author task split. The
author heads do not *lead* retrieval (CLIP is +0.013 above the best of them)
but they are no longer the bottom of the panel either; ranking is consistent
with the clustering picture in which they edge their own backbones by small
margins.

After the re-run, **both author heads sit in mid-pack on both axes** — ARI
(2nd and 6th of 8) and P@5 (5th and 6th of 8) — without the previous
cross-axis inversion. That is the signal of two measurements agreeing
weakly: small but consistent edge over no-signal baselines, well short of the
§8a style-head magnitudes.

## Internal–external correlation *inverts* relative to §8a

§8a found silhouette a defensible proxy at fixed k (Pearson r = +0.78 on the
oracle-k partitional runs after the re-runs). **§8b reverses the sign,
recomputed over the merged 96-cell dataset (29-May ← 01-Jun ← 02-Jun):**

- **r = −0.542** over all 89 runs with a valid ARI and >1 cluster.
- **r = −0.126** over the 32 agglomerative runs alone (was −0.25 pre-re-run —
  the style-head re-run shifted both style-head agglo cells, weakening the
  small-sample negative correlation in the agglomerative-only subset).

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

- **identity space, mcs=10** collapses to **0 clusters (noise 1.0)** for 5 of
  8 embeddings (clip, dinov2_vits14, dinov2_author_head, mobilenet_v3,
  mobilenet_v3_author_head, resnet50 — the 2026-06-02 style-head re-run
  removed both style heads from this list: they now produce k=2 at 10–19 %
  noise). At mcs=5 identity, `mobilenet_v3` remains all-noise; the two author
  heads (after the 01-Jun re-run) produce a degenerate k=2 at 88–93 % noise.
- **mcs=2** is the only setting that fragments into many clusters (17–53),
  but at **20–55 % noise** — the source of the inflated `_no_noise` columns.
- Both **style** heads are outliers in identity: dinov2_style hdb3/5/10 all
  collapse to a stable k=2 split with high silhouette (~0.46) and ARI ≈ 0;
  mobilenet_style hdb5/10 collapse to k=2 with sil ~0.44 and ARI ≈ 0 — the
  same degenerate binary-split trap documented in §4 and §8a.

Repeat (in)stability sits at the collapse boundary as in §4/§8a: under UMAP,
the 2026-06-02 re-run shows `mobilenet_v3_graffiti_style_head` umap hdb3 →
`[16, 22, 16]` (was `[2, 21, 26]` — narrower spread now), hdb5 → `[6, 10, 10]`
and hdb10 (k=4) → `[4, 2, 4]`. Agglomerative rows are stable (`[30,30,30]` /
`[87,87,87]`).

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

1. **Author head > backbone (symmetric to §8a).** ⚠ Weakly supported. DINOv2
   author head (0.065) > its backbone (0.055) by +0.010 ARI and > the style
   head (0.054) by +0.011; MobileNet author head (0.048) > its backbone
   (0.046) by +0.002 ARI and > the style head (0.041) by +0.007. Sign matches
   the §8a symmetry on both families, but the magnitudes (+0.010 / +0.002)
   sit inside the band-of-noise (peers within ±0.015), so the win is real in
   direction and at floor-level absolute magnitude — order-of-magnitude
   weaker than §8a (+0.366 / +0.271).
2. **Cross-task specificity / diagonal-dominant matrix.** ⚠ Partially
   supported. The 2×2 head-vs-task matrix is now fully diagonal-dominant on
   both axes — style head wins style, author head wins author, and each cross
   cell sits below its backbone (after the 2026-06-02 re-run the DINOv2 style
   head also drops below its backbone on author, completing the pattern) —
   but the author-axis margin (+0.010 / +0.002) is an order of magnitude
   below the style-axis margin (+0.366 / +0.271). Style specificity (§8a) is
   robust and useful; author specificity is *directionally* present but
   survives the Cuenca→Salamanca shift only at floor magnitude.
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

1. **§8b is a floor-level result with weak directional support, and that is the
   finding.** Cross-city author recovery sits in a 0.041–0.076 band, ~10× below
   §8a's best cell. Within that band the author heads do edge their own
   backbones (DINOv2 +0.010, MobileNet +0.002 ARI), so the directional symmetry
   with §8a is preserved — but at a magnitude that, in absolute terms, is
   essentially noise. The thesis should report this as: author specificity is
   *detectable in sign* across the Cuenca→Salamanca boundary, *not measurable
   in practical magnitude*. The §8a style result remains the positive transfer
   claim of useful magnitude; §8b qualifies its symmetric author counterpart.
2. **The §8a specificity claim closes weakly on both axes.** Style-head
   specificity is demonstrated at useful magnitude (§8a); author-head
   specificity is demonstrated only in direction at floor magnitude (§8b). Any
   "two heads encode different information" statement can cite both sweeps
   for the **sign** of the effect, but must flag the order-of-magnitude
   asymmetry: style transfer is robust and large, author transfer is
   directional and small (and confounded by domain shift + singleton-heavy
   labels).
3. **Silhouette is anti-correlated with label agreement on fine-grained labels**
   (r = −0.54). This is a strong caveat for §§1–4: silhouette/CH defend the
   *coarse* (few-cluster) unsupervised metrics, but must never be used to argue
   for fine-grained (many-author-scale) cluster quality. The §8a (+0.78) vs §8b
   (−0.54) split bounds exactly what silhouette tracks: blob cohesion, not
   identity.
4. **NMI is uninformative on this label set.** The 0.75–0.79 NMI on every k=87
   winner is the singleton-inflation artefact the brief predicted; report ARI
   (and the random P@5 baseline), never NMI, for fine-grained author validation.

**Degenerate cells (all-noise, 0 clusters, no metrics):** identity-HDBSCAN
mcs=10 for `clip_vit_b32`, `dinov2_vits14`, `dinov2_graffiti_author_head`,
`mobilenet_v3`, `mobilenet_v3_graffiti_author_head`, `resnet50`; plus
identity-HDBSCAN mcs=5 for `mobilenet_v3` (the two author-head mcs=5 cells,
which were all-noise in the 29-May benchmark, now post a degenerate k=2
partition at 88–93 % noise after the 2026-06-01 re-run and are no longer
fully degenerate). After the 2026-06-02 style-head re-run, both
`mobilenet_v3_graffiti_style_head` and `dinov2_graffiti_style_head`
identity-HDBSCAN mcs=10 cells also exit the degenerate set (k=2 with 10–19 %
noise instead of full collapse). Listed here once; excluded from the tables
above.
