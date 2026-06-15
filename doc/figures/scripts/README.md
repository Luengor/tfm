# Figure scripts

Independent scripts per figure in `doc/doc/results.tex` and `doc/doc/tools.tex`. Each reads benchmark
JSON from `exp/<exp>/output/` and writes a PDF to `doc/figures/`.

Run one figure:

```bash
cd doc/figures/scripts
uv run --with seaborn,matplotlib,pandas python e1_silhouette.py
```

Regenerate all:

```bash
cd doc/figures/scripts
for f in e*.py; do uv run --with seaborn,matplotlib,pandas python "$f"; done
```

Shared style lives in `_style.py`. Output naming mirrors the
`\includegraphics{figures/<name>.pdf}` lines in the tex sources.

| Script | Output PDF | Figure label |
|---|---|---|
| `e1_silhouette.py` | `e1_silhouette_by_embedding.pdf` | `fig:e1-silhouette` |
| `e2_reduction.py`  | `e2_reduction_quality_cost.pdf` | `fig:e2-reduction` |
| `e3_pareto.py`     | `e3_pareto_quality_cost.pdf` | `fig:e3-pareto` |
| `e3_scaling.py`    | `e3_scaling_with_n.pdf` | `fig:e3-scaling` |
| `e3_dbscan_eps.py` | `e3_dbscan_eps_sensitivity.pdf` | *(not included in document)* |
| `e5_yolo_detections.py` | `e5_yolo_detections.pdf` | `fig:e1-yolo-detections` |
| `e6e9_pareto.py`   | `e6_e9_pareto.pdf` | `fig:similaritysearch-pareto` |
| `e7_scaling.py`    | `e7_scalability_loglog.pdf` | `fig:e7-scaling` |
| `e7_memory.py`     | `e7_memory.pdf` | `fig:e7-memory` |
| `dataset_style_distribution.py` | `dataset_style_distribution.pdf` | `fig:dataset-style-distribution` (tools.tex) |
| `e8a_hdbscan.py` | `e8a_hdbscan_3clusters.pdf` | `fig:e8a-hdbscan` |

## Figures requiring the `src` venv

These run YOLO/torch and therefore live in `src/src/scripts/`, not here — the
`--with seaborn,matplotlib,pandas` env above has no torch/ultralytics. Run from
`src/`:

```bash
cd src
uv run python -m src.scripts.visualize_segmentation_stages \
    --image "../data/detect/a/images/Avenida de Castilla La Mancha 11 20240124_124442 Avenida de Castilla La Mancha 11.jpg" \
    --output ../doc/figures/segmentation_stages.pdf
```

| Script | Output PDF | Figure label |
|---|---|---|
| `src/src/scripts/visualize_segmentation_stages.py` | `segmentation_stages.pdf` | `fig:segmentation-stages` (methods.tex) |

Omit `--image` to auto-select an image showing both a merge and multiple crops
(deterministic with `--seed`).

## Pending figures

These figures are referenced or described in the thesis but not yet generated from the benchmark JSONs (require dataset images/crops):

- DINOv2 vs ResNet50 style illustration
