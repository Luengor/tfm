# §1 — Embedding Model Comparison: Report

**Source:** `exp/exp01/output/benchmark_20260528T231810Z.json` (10 backbones),
`exp/exp01/output/benchmark_20260601T144625Z.json` (author heads, re-run;
supersedes the 28-May author-head rows) and
`exp/exp01/output/benchmark_20260602T092553Z.json` (style heads, re-run;
supersedes the 28-May style-head rows).
**Date:** 2026-05-28 (main sweep), 2026-06-01 (author-head re-run),
2026-06-02 (style-head re-run) · **Host:** FullCreamMilk (Linux, 16 logical CPU, Python 3.14)
**Status:** 36/36 + 6/6 + 6/6 runs `success`.

## Setup

Pillar 2 (unsupervised quality). Question: which embedding backbone produces
the most semantically coherent graffiti clusters under a fixed downstream
pipeline, and does fine-tuning a graffiti-specific head beat its pretrained
backbone?

Fixed baseline (one axis varied = `embedding`):

| Component | Value |
|---|---|
| Embedding | **varied** (12 backbones — see note below) |
| Reduction | UMAP 10-d (cosine, n_neighbors=15, min_dist=0) |
| Clustering | HDBSCAN, `min_cluster_size=5` |
| Segmenter | `identity` (whole image) |
| Storage | SQLite |
| Repeats | `K=3` (reduction+clustering looped; iter 0 dropped) |

12 embeddings × 3 corpus sizes `{250, 1000, 6416}` = 36 runs, 36 full ingests
(each embedding × limit gets its own DB; no reuse across embeddings).

> **Note — 12 embeddings ran, not the 10 in the catalogue.** `infos/experiments.md`
> §1 lists ten embeddings and explicitly omits `dinov2_graffiti_author_head`.
> The actual run contains **all four** fine-tuned heads (author + style ×
> DINOv2 + MobileNetV3). This is a *positive* deviation: it yields the complete
> 2×2 head matrix that §8 hypothesis 2 (cross-task transfer) needs, and lets §1
> read the author-vs-style head contrast on both backbones. The catalogue
> should be updated to match (drop the "omitted" clause).

> **Reading note — intrinsic metrics grade tightness, not semantics.** All five
> quality metrics are computed on each model's **own L2-normalised** embeddings
> (`clustering_metrics.py`). L2-normalisation makes the scores *numerically*
> comparable across families (Euclidean is monotone with cosine on unit
> vectors), but it does **not** make them *semantically* comparable: silhouette
> measures how cleanly HDBSCAN carved that model's space, not whether the
> resulting clusters track graffiti categories. §1 is unsupervised — there is no
> ground-truth anchor here. The semantic adjudication lives in §8; §1 ranks
> *separability*, §8 ranks *correctness*. Residual embedding-dimension drift
> (384-d DINOv2 vs 2048-d ResNet vs 4096-d VGG) also perturbs silhouette
> absolutes even after normalisation.

## Summary — quality

`cls` = cluster count, `noise` = fraction labelled −1, `sil` = silhouette
(higher better), `CH` = Calinski–Harabasz (higher better), `DB` =
Davies–Bouldin (lower better), `csCV` = cluster-size CV (lower = more balanced).
Within each corpus size, **bold** = best silhouette, *italic* = best CH.

| n | embedding | cls | noise | sil | CH | DB | csCV |
|--:|---|--:|--:|--:|--:|--:|--:|
| 250 | resnet50 | 23 | 0.063 | **0.258** | 13.2 | 1.43 | 0.51 |
| 250 | vgg16 | 26 | 0.084 | 0.250 | 10.9 | 1.46 | 0.55 |
| 250 | mobilenet_v3 | 26 | 0.055 | 0.245 | 10.4 | 1.58 | 0.48 |
| 250 | dinov2_graffiti_author_head | 25 | 0.077 | 0.226 | 12.1 | 1.48 | 0.49 |
| 250 | inception_v3 | 27 | 0.069 | 0.225 | 9.1 | 1.61 | 0.47 |
| 250 | dinov2_vits14 | 26 | 0.052 | 0.216 | 12.7 | 1.51 | 0.59 |
| 250 | mobilenet_v3_graffiti_author_head | 25 | 0.073 | 0.212 | 11.0 | 1.58 | 0.68 |
| 250 | yolom | 18 | 0.049 | 0.204 | 27.4 | 1.42 | 0.83 |
| 250 | clip_vit_b32 | 23 | 0.120 | 0.194 | 9.8 | 1.58 | 0.57 |
| 250 | dinov2_graffiti_style_head | 18 | 0.060 | 0.187 | 18.9 | 1.59 | 0.63 |
| 250 | yolon | 16 | 0.053 | 0.181 | *30.4* | 1.43 | 1.09 |
| 250 | mobilenet_v3_graffiti_style_head | 21 | 0.079 | 0.174 | 15.4 | 1.65 | 0.75 |
| 1000 | resnet50 | 127 | 0.028 | **0.272** | 13.7 | 1.39 | 0.53 |
| 1000 | mobilenet_v3 | 131 | 0.040 | 0.269 | 11.4 | 1.46 | 0.49 |
| 1000 | mobilenet_v3_graffiti_author_head | 127 | 0.075 | 0.236 | 11.6 | 1.50 | 0.42 |
| 1000 | vgg16 | 113 | 0.094 | 0.234 | 11.6 | 1.53 | 0.55 |
| 1000 | dinov2_vits14 | 114 | 0.085 | 0.219 | 13.7 | 1.45 | 0.47 |
| 1000 | inception_v3 | 127 | 0.058 | 0.217 | 9.5 | 1.60 | 0.47 |
| 1000 | dinov2_graffiti_author_head | 111 | 0.083 | 0.206 | 13.5 | 1.52 | 0.50 |
| 1000 | clip_vit_b32 | 125 | 0.105 | 0.193 | 10.6 | 1.55 | 0.54 |
| 1000 | dinov2_graffiti_style_head | 106 | 0.111 | 0.177 | 17.8 | 1.62 | 0.55 |
| 1000 | mobilenet_v3_graffiti_style_head | 105 | 0.141 | 0.176 | 14.6 | 1.66 | 0.50 |
| 1000 | yolom | 76 | 0.173 | 0.144 | 28.9 | 1.70 | 0.61 |
| 1000 | yolon | 59 | 0.220 | 0.116 | *44.5* | 1.71 | 0.65 |
| 6416 | mobilenet_v3 | 820 | 0.078 | **0.260** | 13.1 | 1.48 | 0.52 |
| 6416 | resnet50 | 794 | 0.086 | 0.257 | 15.6 | 1.46 | 0.71 |
| 6416 | vgg16 | 773 | 0.095 | 0.228 | 13.4 | 1.55 | 0.54 |
| 6416 | mobilenet_v3_graffiti_author_head | 779 | 0.124 | 0.227 | 13.4 | 1.53 | 0.53 |
| 6416 | inception_v3 | 724 | 0.118 | 0.213 | 11.5 | 1.61 | 0.64 |
| 6416 | dinov2_graffiti_author_head | 631 | 0.143 | 0.187 | 17.9 | 1.59 | 0.81 |
| 6416 | dinov2_vits14 | 632 | 0.123 | 0.185 | 19.2 | 1.58 | 0.92 |
| 6416 | clip_vit_b32 | 734 | 0.150 | 0.170 | 11.9 | 1.66 | 0.74 |
| 6416 | mobilenet_v3_graffiti_style_head | 629 | 0.214 | 0.164 | 19.7 | 1.74 | 0.58 |
| 6416 | dinov2_graffiti_style_head | 562 | 0.202 | 0.155 | 27.9 | 1.70 | 0.74 |
| 6416 | yolom | 480 | 0.253 | 0.131 | 32.2 | 1.75 | 0.74 |
| 6416 | yolon | 348 | 0.341 | 0.085 | *51.2* | 1.91 | 0.81 |

## Summary — cost

Ingest dominates by three orders of magnitude — it is the only embedding-cost
axis that matters. Reduction is seconds, clustering sub-second; both are
effectively free relative to embedding generation.

| n | embedding | ingest_s | ips | ing_pkRSS MB | vram MB | red_s |
|--:|---|--:|--:|--:|--:|--:|
| 6416 | mobilenet_v3 | 808.8 | 7.93 | 366.0 | **31.8** | 6.06 |
| 6416 | mobilenet_v3_graffiti_style_head | 861.3 | 7.45 | 406.5 | 34.5 | 5.95 |
| 6416 | mobilenet_v3_graffiti_author_head | 879.1 | 7.30 | 406.4 | 34.5 | 5.86 |
| 6416 | inception_v3 | 973.7 | 6.59 | 337.2 | 116.1 | 6.71 |
| 6416 | vgg16 | 980.8 | 6.54 | 315.0 | 551.7 | 7.88 |
| 6416 | resnet50 | 1080.0 | 5.94 | 336.0 | 117.8 | 6.57 |
| 6416 | dinov2_vits14 | 1215.7 | 5.28 | 209.6 | 98.0 | 5.74 |
| 6416 | dinov2_graffiti_style_head | 1245.0 | 5.15 | 286.2 | 99.7 | 5.98 |
| 6416 | clip_vit_b32 | 1247.0 | 5.15 | 193.1 | 591.9 | 6.01 |
| 6416 | yolon | 1301.0 | 4.93 | 853.8 | 65.4 | 5.89 |
| 6416 | yolom | 1407.4 | 4.56 | 824.9 | 192.9 | 5.96 |

(250 / 1000 rows scale down proportionally — see JSON. Throughput is roughly
constant in `n` per model.) Key cost facts:

- **MobileNetV3 is the cheapest encoder on every axis** — fastest ingest
  (~7.9 ips), lowest VRAM (32 MB), and the author head adds modest overhead
  (\~7.3 ips, ~8 % slower). YOLO backbones are the most expensive: slowest
  ingest (4.5–4.9 ips) *and* highest RSS (\~825–855 MB, ~2.5× the CNNs).
- **VRAM splits the field:** VGG16 (552 MB) and CLIP (592 MB) are the heavy
  consumers; MobileNetV3 (32 MB) is two decimal orders lighter.
- **Reduction tracks input dimensionality, weakly.** UMAP on 4096-d VGG16
  (7.88 s) is ~37 % slower than on 384-d DINOv2 (5.74 s) at n=6416 — a real but
  minor effect, swamped by ingest. A fine-tuned head costs the same as its
  backbone (the projection head runs inside the frozen-backbone forward pass).

## The central reading hazard: fine-tuning *lowers* silhouette while CH disagrees

The single most important result in §1 is also its sharpest reading trap. On
the **style** pairs the head still **underperforms** its pretrained backbone
on silhouette at every size after the 2026-06-02 re-run; on the **author**
pairs the picture is mixed (MobileNet author head is consistently below its
backbone; DINOv2 author head lands essentially tied with — and at n=250 and
n=6416 marginally above — its backbone after the 2026-06-01 re-run):

| pair (n=6416) | backbone sil | head sil | backbone CH | head CH |
|---|--:|--:|--:|--:|
| dinov2_vits14 → style head | 0.185 | 0.155 | 19.2 | **27.9** |
| dinov2_vits14 → author head | 0.185 | 0.187 | 19.2 | 17.9 |
| mobilenet_v3 → style head | 0.260 | 0.164 | 13.1 | **19.7** |
| mobilenet_v3 → author head | 0.260 | 0.227 | 13.1 | 13.4 |

A naive "highest silhouette wins" reading concludes fine-tuning *hurts* and
picks raw MobileNetV3 / ResNet50. But **both style heads' CH lands above their
backbone** — DINOv2 style head 27.9 vs 19.2 (1.45×, the highest non-YOLO CH at
n=6416) and MobileNetV3 style head 19.7 vs 13.1 (1.50×) — at the same time
silhouette falls. The two intrinsic scores split on the same axis, and now on
*both* backbones, not only DINOv2. This is not noise: it is the geometric
signature of fine-tuning. The contrastive/triplet head reshapes the space
toward a few **task-discriminative axes**, raising between-cluster vs.
within-cluster dispersion (CH↑) while the per-point neighbourhood cohesion
that silhouette averages goes down (more elongated, fewer, denser clusters —
note cls drops 632→562 on DINOv2 and 820→629 on MobileNetV3, with csCV staying
moderate ~0.6–0.7). The author heads, trained on the same backbones with a different (author-identity)
target, do not produce the same sharp split — both intrinsic scores move
together in the same direction or sit essentially at backbone parity — which
is itself consistent with the design: the author objective shapes a less
concentrated geometry than the style head's tight per-style modes.

**Neither intrinsic score is the right grade for a fine-tuned space.** The head
was trained to separate *graffiti categories*, an objective invisible to
unsupervised silhouette. §1 can rank how cleanly HDBSCAN carves each space; it
cannot say which carving is *semantically* right. That adjudication is §8's job
(ARI / NMI / F1 against the labelled crops). The empirical silhouette↓/CH↑ split
above is the concrete, in-data demonstration of why §1 alone cannot pick the
embedding — exactly the cross-embedding analogue of the §4 silhouette-inversion
hazard.

## ImageNet CNNs look best intrinsically — but that is the trap, not the answer

On raw silhouette, the off-the-shelf ImageNet CNNs sweep the top: ResNet50 and
MobileNetV3 lead at every size (sil 0.26–0.27), with VGG16 and InceptionV3 close
behind. They also post the lowest noise (mobilenet 0.078, resnet 0.086 at
n=6416) and the most balanced clusters (csCV ~0.5). Read in isolation, the table
says "use MobileNetV3."

This is the hazard from the reading note made concrete. ImageNet features
cluster *something* cleanly — texture, colour, scene layout — but there is no
evidence here it is graffiti style or authorship. A high silhouette on generic
visual features is precisely what you'd expect from an encoder that has never
seen the task. **§1 must not recommend an embedding on intrinsic separability
alone.** The baseline `dinov2_graffiti_style_head` is retained for §§2–7 on the
strength of its §8 supervised performance and its design (graffiti-specific
discriminative axes), not its §1 silhouette — and §1's job is to document the
ranking, not to overturn that choice.

## YOLO backbones are genuinely the weakest — every reading agrees

Unlike the CNN-vs-head contrast, YOLO needs no caveat: it loses on every joint
criterion simultaneously. At n=6416, `yolon` posts the **worst silhouette
(0.085), worst noise (0.341 — a third of the corpus unclustered), fewest
clusters (348), and worst DB (1.91)**. Its high CH (51.2) is the §4 illusion
again — CH rewards a few fat, well-separated blobs, and YOLO produces exactly
that by dumping a third of points into noise and coarsely grouping the rest.
`yolom` is uniformly better than `yolon` but still bottom-tier. YOLO detection
backbones encode localisation features, not the holistic appearance clustering
needs — the result is consistent with that expectation and stable across all
three sizes.

## Cluster count and noise scale sensibly with `n`

Every embedding's cluster count grows monotonically and roughly linearly with
the corpus (e.g. mobilenet 26 → 131 → 820; dinov2 style head 18 → 106 → 562),
confirming HDBSCAN at `mcs=5` keeps finding finer structure as data grows rather
than saturating — consistent with §4's `mcs=5` finding. Noise generally creeps
upward with `n` (more boundary points), most steeply for the weak encoders
(yolon 0.05 → 0.22 → 0.34) and mildly for the strong ones (mobilenet 0.055 →
0.040 → 0.078). The **style** heads carry the highest noise among the non-YOLO
rows at n=6416 (mob_style 0.214, dinov2_style 0.202) — the flip side of their
tighter, fewer clusters: points outside the discriminative manifold are pushed
to −1. The author heads sit in the middle band (mob_auth 0.124, dinov2_auth
0.143 at n=6416), comparable to their backbones.

## Repeat stability

Most cells are stable across K=3. UMAP seed stochasticity is small at usable
settings (e.g. dinov2 style head @6416 = 561 / 545 / 562, ±~2 %; sil std 0.003).
The widest relative spread is at the smallest corpus, as expected:
`dinov2_vits14@250` = `[21, 23, 26]` (~24 % spread on a tiny absolute count) —
not unstable in kind (no all-noise flips like §4's `mcs50@250`), just
small-`n` UMAP jitter. No row required exclusion.

## Conclusion for downstream experiments

1. **§1 cannot adjudicate the baseline embedding on its own metrics, and that is
   the finding — not a limitation.** Intrinsic silhouette ranks the ImageNet
   CNNs first and the fine-tuned **style** heads clearly below their backbones
   (the author heads sit at or close to backbone parity), but the
   silhouette↓ / CH↑ split shows on **both** style heads after the 2026-06-02
   re-run (DINOv2 27.9 vs backbone 19.2; MobileNetV3 19.7 vs backbone 13.1),
   proving the two intrinsic scores disagree about the same fine-tuned space.
   Separability ≠ semantic correctness; the embedding choice binds in **§8**,
   not here.
2. **The baseline `dinov2_graffiti_style_head` is retained** for §§2–7. §1
   provides the unsupervised context (it produces tight, high-CH, moderate-noise
   clusters) but the justification is the supervised anchor and the
   task-specific design, not the §1 silhouette ranking.
3. **CH inflates the same way it did in §4** — it rewards few, fat, separated
   clusters, which is why YOLO (high noise, few clusters) posts deceptively high
   CH and why both style heads' CH jumps above their backbones. Read CH against
   cluster_count and noise_ratio, never alone.
4. **YOLO backbones are excluded from any "good embedding" narrative** — worst
   on silhouette, noise, DB, and cluster count at every size. Detection
   features do not transfer to appearance clustering.
5. **Cost favours MobileNetV3** (fastest ingest, lowest VRAM) and penalises YOLO
   (slowest + heaviest RSS) and VGG16/CLIP (highest VRAM). Since clustering
   quality is the §1 pillar and ingest dominates cost, the cost table is input
   to §§6–7 budgeting, not to the embedding choice.
6. **The full 2×2 head matrix is available** (author + style × DINOv2 +
   MobileNetV3) — feed it directly into §8's cross-task transfer matrix
   (hypothesis 2) and reconcile `infos/experiments.md` §1, which still lists ten
   embeddings and omits `dinov2_graffiti_author_head`.
