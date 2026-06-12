# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis (TFM) on graffiti image clustering using deep learning embeddings. The system detects graffiti in images (via YOLO segmentation), generates embeddings, stores them, reduces dimensionality, and clusters them. A benchmarking framework evaluates combinations of these components.

## Sources 
From a Zotero project, all sources and other articles relevant to the project are stored in a sumarized form in the `papers` folder. Consider consulting these documents when the project requires performing or implementing a specific method, or when you need to understand the rationale behind a design decision.

## Maintainability 
If an issue is found in the code or article or if you think that a certain part of the code project be improved, say so directly.

After updating an integral or important part of the project, consider updating this document with the changes and the rationale behind them.

Never commit work unless the user has given explicit permission to do so. If the user allows you to commit once, DO
ASK the next time you want to commit, even if it's a small change. Always ask before committing.

## Environment & Commands

This project uses `uv` for dependency management (Python ≥ 3.14). All commands run from the `src/` directory.

```bash
cd src

# Install dependencies
uv sync

# Run the benchmark pipeline
uv run pipeline-benchmark --configuration benchmarks/configuration.sample.json --dataset ../dataset/images

# Run with ground-truth labels for extrinsic metrics (ARI, NMI, pairwise F1)
uv run pipeline-benchmark --configuration benchmarks/configuration.sample.json --dataset ../dataset/images --ground-truth ../dataset/labels.csv

# Run only specific named runs from the configuration
uv run pipeline-benchmark --configuration benchmarks/configuration.sample.json --dataset ../dataset/images --run run_name_1 run_name_2

# Generate interactive 2D cluster scatter plots per run
uv run pipeline-benchmark --configuration benchmarks/configuration.sample.json --dataset ../dataset/images --cluster-plot

# Plot benchmark results
uv run pipeline-plot -i benchmark_results/benchmark_*.json -o plots/

# Generate a configuration from a grid (cartesian product of components)
uv run python benchmarks/generate_configuration.py -i benchmarks/grid.sample.json -o benchmarks/my_configuration.json

# Lint
uvx run ruff check src/

# Run a specific module directly
uv run python -m src.configuration
```

### pipeline-benchmark flags

| Flag | Default | Description |
|---|---|---|
| `--configuration` | required | Path to configuration JSON |
| `--dataset` | required | Path to image folder |
| `--output-dir` | `benchmark_results` | Output JSON directory |
| `--run` | all | Run names or IDs to execute |
| `--prefix` | `benchmark` | Output filename prefix |
| `--limit` | none | Max images per run |
| `--ground-truth` | none | CSV with `filename`+`style` columns for extrinsic metrics |
| `--cluster-plot` | false | Write interactive UMAP HTML scatter per run |

## Architecture

The pipeline is assembled via `Configuration` (`src/src/configuration.py`), which wires together five swappable components defined as abstract base classes in `src/src/abstractions.py`:

| Abstract class | Role | Implementations |
|---|---|---|
| `StorageBase` | Persist/query embeddings | `SQLiteStorage`, `PostgreSQLStorage` (with pgvector) |
| `EmbeddingBase` | Generate float vectors from images | ResNet50, VGG16, InceptionV3, MobileNetV3, DINOv2, CLIP ViT-B/32, YOLO (n/m), custom graffiti-finetuned heads (author + style) |
| `SegmenterBase` | Detect graffiti bounding boxes | `YoloSegmenter` (YOLO detection + merge + padding), `IdentitySegmenter` (full image) |
| `ReductionBase` | Dimensionality reduction before clustering | PCA, UMAP, Isomap, KernelPCA, Identity |
| `ClusteringBase` | Cluster reduced embeddings | KMeans, DBSCAN, HDBSCAN, OPTICS, Agglomerative, Spectral, GMM, AffinityPropagation |

**Core flow** (`Configuration.save_image`): open image → segment → crop each bounding box → generate embedding per crop → save to storage.

**Clustering flow** (`Configuration.cluster_images`): load all embeddings from storage → reduce dimensions → cluster → return grouped `ImageData` lists.

### Embedding Models (`src/src/embedding/`)

- `EmbeddingModelNames` enum in `embeddings.py` — names used in configuration `type` field
- `TorchEmbeddingModel` — torchvision backbones (ResNet50, VGG16, InceptionV3, MobileNetV3)
- `HeadEmbeddingModel` — frozen backbone + two-layer projection head (Linear→ReLU→Linear) for graffiti-specific heads
- `DinoEmbeddingModel` — DINOv2 ViT-S/14 via `torch.hub`
- `ClipEmbeddingModel` — CLIP ViT-B/32 via `open_clip`
- `YoloEmbeddingModel` — YOLO backbone features
- `CustomEmbeddingModel` (`custom.py`) — `TorchEmbeddingModel` subclass that loads custom `.pth` weights

### Benchmarking System

`pipeline-benchmark` drives systematic evaluation:

1. **Configuration JSON** (`benchmarks/*.json`) declares a JSON object with a `runs` list. Each run **must have a `name`** and specifies `storage`, `embedding`, `clustering`, and optionally `segmenter`, `reduction`, and `similarity_search` as `{type, params}` objects. A top-level `cluster_plot` key can enable/configure cluster visualizations.
2. **Grid generation** (`benchmarks/generate_configuration.py`) takes a grid JSON (lists of options per key) and generates the cartesian product as a configuration. It also handles SQLite database reuse: runs sharing the same storage+embedding+segmenter+limit combination get `clear_storage: false` and the same `db_path`.
3. **Runner** (`src/src/evaluation/runner.py`) builds components from the configuration spec, profiles each stage (ingest, reduction, clustering, similarity search), computes clustering quality metrics, and writes JSON results.
4. **Plotting** (`pipeline-plot`) reads the JSON results and produces seaborn bar charts per metric.

### Clustering Metrics (`src/src/evaluation/clustering_metrics.py`)

**Intrinsic** (always computed, on original unreduced embeddings):
- Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index
- Noise Ratio (fraction of points labeled -1), Cluster Size CV (balance)

**Extrinsic** (requires `--ground-truth` CSV; only for IdentitySegmenter runs):
- ARI (Adjusted Rand Index), NMI (Normalized Mutual Info), Pairwise F1
- Each is reported in two readings: the default treats noise (-1) as its own cluster; the `*_no_noise` variant drops -1 points before scoring, so density clusterers (which emit noise) are comparable to partitional ones on the points each actually clustered. Both aggregate (mean ± std) across repeats.
- Matched count, `n_no_noise` (points kept by the noise-excluded reading), class count, coverage

**Similarity-search ANN recall**: when an HNSW index is active (PostgreSQL), the runner records `ann_recall_at_k` — recall@k of the index's neighbors vs an exact brute-force kNN over the same queries, computed outside the timed loop. `None` for exact backends. This is the accuracy axis of the HNSW speed-vs-accuracy trade-off (`avg_neighbor_distance` is only an aggregate proxy).

### Cluster Visualization (`src/src/evaluation/cluster_plot.py`)

When `--cluster-plot` is passed (or `cluster_plot.enabled: true` in configuration), writes an interactive Vega-Lite 2D UMAP HTML scatter to `<output-dir>/<run_id>_cluster.html`.

### Model Files

Pre-trained model weights live in `models/` (gitignored):
- `yolo11{n,s,m,l}.pt` — YOLO graffiti detection models (for segmentation and embedding)
- `yolo11m-train-{8,10}.pt` — fine-tuned YOLO detection models
- `mobilenet_graffiti_author_head.pth` — MobileNetV3 graffiti author-similarity head
- `mobilenet_graffiti_style_head.pth` — MobileNetV3 graffiti style-similarity head
- `dinov2_graffiti_author_head.pth` — DINOv2 graffiti author-similarity head
- `dinov2_graffiti_style_head.pth` — DINOv2 graffiti style-similarity head

There are **four** fine-tuned heads (author + style × MobileNetV3 + DINOv2), matching the `*_GRAFFITI_*_HEAD` entries in `EmbeddingModelNames`.

Model paths are resolved **relative to the working directory** (i.e. run from `src/`).

### Storage Backends

- **SQLite** — default, no setup required; distance search is done in Python (linear scan).
- **PostgreSQL** — requires `pgvector` extension; connection via `db_url` param. Default URL in dev: `postgresql://postgres:changethis@localhost:54321/postgres`. Optional `hnsw` param (under `storage.params`) enables an HNSW ANN index for `get_by_distance`. Accepts `true` (defaults) or a dict `{ops, m, ef_construction, ef_search}`. `ops` is `cosine`, `l2`, `ip`, or `both` (default `both` — builds one index per op class so cosine and L2 queries are both accelerated). The index is built once between ingest and the similarity-search stage so its build cost is excluded from the timed query loop. pgvector's HNSW dimension limit is 2000 for `vector`; embeddings with `embedding_size > 2000` (e.g. ResNet50=2048, VGG16=4096) will fail at index creation.

### Training Module (`src/src/train/`)

Produces the custom `.pth` head weights. Contains triplet-loss trainer (`trainer.py`), supervised contrastive style trainer (`style_trainer.py`), dataset loaders (`dataset.py`), and sub-packages for MobileNet and DINOv2 head training (`mobilenet/`, `dino/`).

### Key Design Decisions

- `KMeans`, `DBSCAN`, and `GMM` clusterers auto-detect the optimal number of clusters via the elbow method (KMeans/DBSCAN) or BIC (GMM) when `n_clusters`/`eps` is not provided. When a run does **not** pin `n_clusters` and uses multiple repeats, the runner auto-detects k on the first repeat and **propagates that k** to the remaining repeats — so repeats vary only by seed and the aggregated metrics describe a single partition family rather than a mix of different k. `clusters_per_repeat` is recorded for transparency.
- Clustering quality metrics are calculated against the **original** (unreduced) embeddings, not the reduced ones used for clustering. The embeddings are **L2-normalized** first so the distance-based scores (silhouette, Calinski-Harabasz, Davies-Bouldin) are comparable across embedding families — some models L2-normalize their output and some do not, and on unit vectors Euclidean distance is monotone with cosine (the geometry used elsewhere in the project). Normalization rescales but does not reduce dimensionality.
- The benchmark runner persists ingestion metrics in the storage `metadata` table so that re-used databases report the original ingest time, not the (near-zero) cache-hit time. The decision is driven by the run's `clear_storage` flag: a fresh run (`clear_storage: true`) saves its measured metrics; a reusing run (`clear_storage: false`) reads them back from the run that populated the DB.
- `HeadEmbeddingModel` wraps a frozen base model with a two-layer projection head (Linear → ReLU → Linear) trained for graffiti-specific similarity.
- Extrinsic metrics are only computed for IdentitySegmenter runs (where the ground-truth filename mapping is unambiguous).

## Documentation
The document for the Master's thesis lives in `doc/`. It's written in LaTeX (root: `doc/informe.tex`) and built with **pdflatex** into the `build` subfolder. Run from `doc/`:

```bash
cd doc
mkdir -p build/doc                                   # \include writes per-file .aux into build/doc/
pdflatex -interaction=nonstopmode -output-directory=build informe.tex
BIBINPUTS=".:./build:" bibtex build/informe          # BIBINPUTS so bibtex finds bib.bib at the doc root
pdflatex -interaction=nonstopmode -output-directory=build informe.tex
pdflatex -interaction=nonstopmode -output-directory=build informe.tex
```

Notes:
- `build/doc/` must exist first — pdflatex won't create subdirs for `\include` aux files.
- The preamble (`doc/doc/packets_and_such.tex`) selects fonts by engine via `iftex`: `inputenc`/`fontenc[T1]` under pdflatex, `fontspec` under xelatex/lualatex. So it builds under either, but pdflatex is the default.
- Spanish bibliography: `\usepackage[spanish]{babel}` needs the `texlive-langspanish` package (provides `spanish.ldf`); without it babel errors `Unknown option 'spanish'`. Localized month names ("Agosto" not "August") come from `babelbib` + `\bibliographystyle{babplain}` (set in `informe.tex`), not babel — `plain.bst` hardcodes English months.


## Thesis proposal
Caracterización del desempeño y el coste computacional en algoritmos de similitud en bancos de imágenes de grafiti
### Descripción:
El grafiti vandálico es una transgresión de dimensión visual, social y legal que provoca problema
socialmente extendido que supone un gran trabajo para las fuerzas del orden por su continua
renovación. La identificación de los grafiteros a partir de fotografías de sus firmas es una
herramienta eficaz para combatir esta lacra, pero es un proceso complejo, cuya dificultad
aumenta con el crecimiento de las bases de datos.
En este trabajo se propone describir y caracterizar el comportamiento de varios métodos para el
cálculo de similitud y la creación de agrupaciones en un conjunto de imágenes de grafitis ya
existente, con el objeto de describir el coste computacional en función del tamaño del conjunto
de imágenes, de forma experimental. Se realizará también un estudio con métricas de carácter
no supervisado de las agrupaciones resultantes, así como una evaluación supervisada para un
subconjunto en el que estén disponibles etiquetas.

### Estructura
Sobre la dualidad memoria de TFM/artículo, te recomiendo escribir la memoria y después recortar partes para convertirlo en artículo cuando ya la tengamos revisada y corregida. Una posibilidad de estructura sería:

    Introducción: motivar la importancia del tema estudiado.
    Estado de la cuestión: qué hay parecido o en la línea ahora mismo.
    Materiales y métodos: qué tecnologías se usan en el trabajo y conceptos teóricos relevantes. Esta sección podría omitirse en el artículo.
    Metodología: cuál es el flujo de trabajo que se ha seguido, descrito de manera que permitiera la replicación.
    Resultados y discusión: qué métricas se han obtenido (resultados) y qué se interpreta de esto (discusión). Puede ser una o dos secciones, yo prefiero que estén juntas para facilitar contextualización.
    Conclusiones: se recapitula el valor del trabajo hecho, y se puede dar alguna orientación sobre posible trabajo futuro. 

Esta estructura es solo un modelo, puede salirte de ella según convenga a aquello que quieras contar con más énfasis.

### Evaluadores
Los evaluadores del trabajo tienen el siguiente perfil:
 * Un doctor del area de lenguajes y sistemas informáticos, con tesis doctoral
   "Técnicas de expansión en los sistemas de recuperación de información" en
   2003.
 * Un doctor del área de ciencias de la computación e inteligencia artifical,
   con tesis doctoral "Modelo de reutilización soportado por estructuras
   complejas de reutilización denominadas mecanos" en 2000.
 * Una doctora del área de lógica y filosofía de la ciencia, con tesis doctoral
   "From a necessitist point of view: themes on Timothy Williamson" en 2025.
No se espera que ninguno de los evaluadores tenga un conocimiento profundo del
tema específico del trabajo, por lo que es necesario que el trabajo sea, hasta
cierto punto, accesible para un público general de informática. Esto no implica
que el trabajo deba ser superficial o detallado de más, sino que debe ser claro
y bien explicado, con los conceptos técnicos necesarios para entender el
trabajo, pero sin dar por supuesto un conocimiento previo específico sobre el
tema.

### Otros aspectos a tener en cuenta
 * El objetivo del trabajo es puramente técnico. En ningún caso se pretende dar
   opinión sobre el grafiti, vandálico o no, ni sobre la legalidad/moralidad de
   su práctica. Este es un fenómeno social complejo, cuya descripción y
   análisis caen fuera del alcance de este trabajo. Se debería tratar el
   grafiti como un fenómeno visual neutro, sin prejuicios, y centrarse en la
   parte técnica de su análisis.


