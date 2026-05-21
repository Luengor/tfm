import csv
import json
import multiprocessing
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.abstractions import ImageData
from src.cluster.cluster import (
    AffinityPropagationClusterer,
    AgglomerativeClusterer,
    DBSCANClusterer,
    GMMClusterer,
    HDBSCANClusterer,
    KMeansClusterer,
    OPTICSClusterer,
    SpectralClusterer,
)
from src.configuration import Configuration
from src.embedding.custom import CustomEmbeddingModel
from src.embedding.embeddings import EmbeddingModelNames, get_model
from src.embedding.segmenters import YoloSegmenter, IdentitySegmenter
from src.evaluation.clustering_metrics import (
    calculate_clustering_metrics,
    calculate_extrinsic_metrics,
    calculate_similarity_search_extrinsic,
)
from src.evaluation.metrics import get_runtime_info, profile_stage
from src.evaluation.models import (
    BenchmarkResult,
    BenchmarkRunSpec,
    ExtrinsicMetrics,
    SimilaritySearchExtrinsicMetrics,
    StageMetrics,
    StageMetricsAgg,
)
from src.reduction import IdentityReduction, PCAReduction, UMAPReduction, KernelPCAReduction, IsomapReduction
from src.storage.postgresql import PostgreSQLStorage
from src.storage.sqlite import SQLiteStorage

DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}


def load_ground_truth(csv_path: str | Path) -> dict[str, str]:
    """Load a ground-truth CSV mapping basename -> style.

    Required columns: ``filename`` and ``style``. Extra columns (e.g. ``surface``)
    are ignored. Rows with empty ``filename`` or ``style`` are skipped.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise ValueError(f"Ground-truth CSV not found: {path}")

    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "filename" not in fieldnames or "style" not in fieldnames:
            raise ValueError(
                f"Ground-truth CSV must have 'filename' and 'style' columns. "
                f"Got: {fieldnames}"
            )
        for row in reader:
            fname = (row.get("filename") or "").strip()
            style = (row.get("style") or "").strip()
            if fname and style:
                mapping[fname] = style
    return mapping


def _run_single_subprocess(
    run_spec: BenchmarkRunSpec,
    dataset_root: Path,
    output_dir: Path,
    global_limit: int | None,
    ground_truth: dict[str, str] | None,
    cluster_plot_options: dict[str, Any] | None,
    result_queue: "multiprocessing.Queue[BenchmarkResult]",
) -> None:
    result = _run_single(
        run_spec=run_spec,
        dataset_root=dataset_root,
        output_dir=output_dir,
        global_limit=global_limit,
        ground_truth=ground_truth,
        cluster_plot_options=cluster_plot_options,
    )
    result_queue.put(result)


def run_benchmarks(
    run_specs: list[BenchmarkRunSpec],
    dataset_path: str,
    output_dir: str,
    limit: int | None = None,
    ground_truth: dict[str, str] | None = None,
    cluster_plot_options: dict[str, Any] | None = None,
) -> list[BenchmarkResult]:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise ValueError(f"Dataset path must be a directory: {dataset_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cluster_plot_options = cluster_plot_options or {}
    ctx = multiprocessing.get_context("spawn")

    results: list[BenchmarkResult] = []
    print(f"Starting benchmarks: {len(run_specs)} runs to execute.")
    if ground_truth is not None:
        print(f"Ground truth loaded: {len(ground_truth)} labeled entries.")
    if cluster_plot_options.get("enabled"):
        print("Cluster 2D plotting enabled.")
    for run_spec in tqdm(run_specs, desc="Benchmark Runs", unit="run"):
        print(f"Running benchmark: {run_spec.name} (ID: {run_spec.run_id})")
        result_queue: multiprocessing.Queue[BenchmarkResult] = ctx.Queue()
        p = ctx.Process(
            target=_run_single_subprocess,
            args=(run_spec, dataset_root, Path(output_dir), limit, ground_truth, cluster_plot_options, result_queue),
        )
        p.start()
        p.join()

        if not result_queue.empty():
            result = result_queue.get()
        else:
            result = BenchmarkResult(
                run_name=run_spec.name,
                run_id=run_spec.run_id,
                status="failed",
                started_at=_utc_now(),
                finished_at=_utc_now(),
                image_count=0,
                cluster_count=None,
                storage_type=run_spec.storage.type,
                embedding_type=run_spec.embedding.type,
                clustering_type=run_spec.clustering.type,
                segmenter_type=run_spec.segmenter.type if run_spec.segmenter else "identity",
                reduction_type=run_spec.reduction.type if run_spec.reduction else "identity",
                error=f"Subprocess exited with code {p.exitcode} without returning a result.",
                config=run_spec,
            )

        results.append(result)
        print(f"Completed run: {run_spec.name} with status {result.status}.")
        if result.error:
            print(f"Error details: {result.error}")

    return results


def write_results(results: list[BenchmarkResult], output_dir: str, prefix: str = "benchmark") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records = [result.to_record() for result in results]
    runtime_info = asdict(get_runtime_info())

    json_path = output_root / f"{prefix}_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"runtime": runtime_info, "results": records}, file, indent=2)

    return str(json_path)


def _run_single(
    run_spec: BenchmarkRunSpec,
    dataset_root: Path,
    output_dir: Path,
    global_limit: int | None = None,
    ground_truth: dict[str, str] | None = None,
    cluster_plot_options: dict[str, Any] | None = None,
) -> BenchmarkResult:
    started_at = _utc_now()

    ingest_metrics = None
    reduction_agg = None
    query_metrics = None
    similarity_metrics = None
    avg_neighbor_distance = None
    similarity_extrinsic = None
    clustering_agg = None
    clustering_quality = None
    clustering_extrinsic = None
    embeddings_arr = None
    reduced_embeddings = None
    labels_arr = None

    image_count = 0
    cluster_count = None
    status = "success"
    error = None

    run_spec.limit = global_limit if global_limit is not None else run_spec.limit
    
    if global_limit is not None:
        from src.evaluation.configuration import make_run_id
        run_spec.run_id = make_run_id(run_spec)

    try:
        image_paths = _list_images(
            dataset_root=dataset_root,
            limit=run_spec.limit,
        )
        if not image_paths:
            raise ValueError(f"Run {run_spec.name}: no images found in {dataset_root}")
        image_count = len(image_paths)

        storage = _build_storage(run_spec, output_dir)
        embedding = _build_embedding(run_spec)
        clustering = _build_clustering(run_spec)
        reduction = _build_reduction(run_spec)
        segmenter = _build_segmenter(run_spec)
        config = Configuration(storage=storage, embedding=embedding, clustering=clustering, reduction=reduction, segmenter=segmenter)

        if run_spec.clear_storage:
            _clear_storage(storage)

        with profile_stage() as ingest_stage:
            total_instances = 0
            for image_path in image_paths:
                saved_data = config.save_image(image_path)
                total_instances += len(saved_data)
        ingest_metrics = ingest_stage.metrics
        image_count = total_instances
        assert ingest_metrics is not None  # for type checker

        # Persist/retrieve ingestion metrics
        # If the database is reused, the original ingestion metrics are retrieved.
        METADATA_INGEST_KEY = "ingest_metrics"
        db_metrics_json = storage.get_metadata(METADATA_INGEST_KEY)

        if db_metrics_json:
            db_metrics_dict = json.loads(db_metrics_json)
            # Heuristic: if current ingestion was significantly faster than the one in DB,
            # it means we reused the database, so we use the original metrics.
            if ingest_metrics.wall_time_s < db_metrics_dict.get("wall_time_s", 0) * 0.5:
                # Reconstruct StageMetrics from dict
                print(f"Run {run_spec.name}: reusing ingestion metrics from previous run.")
                ingest_metrics = StageMetrics(**db_metrics_dict)
        elif ingest_metrics.wall_time_s > 0.5:
            # Only save if it looks like a meaningful ingestion run
            storage.set_metadata(METADATA_INGEST_KEY, json.dumps(asdict(ingest_metrics)))

        all_images = config.storage.get_all_images()

        if run_spec.similarity_search.enabled:
            with profile_stage() as similarity_stage:
                distances = []
                for img in all_images:
                    nearest = config.storage.get_by_distance(
                        img.embedding,
                        max_images=run_spec.similarity_search.top_k + 1,
                        cos_distance=run_spec.similarity_search.cos_distance,
                    )
                    # Exclude self
                    neighbors = [n for n in nearest if (n.filename, n.bbox) != (img.filename, img.bbox)]
                    neighbors = neighbors[: run_spec.similarity_search.top_k]

                    for n in neighbors:
                        d = _calculate_distance(
                            img.embedding,
                            n.embedding,
                            run_spec.similarity_search.cos_distance,
                        )
                        distances.append(d)

                if distances:
                    avg_neighbor_distance = float(np.mean(distances))
            similarity_metrics = similarity_stage.metrics

            segmenter_type_eff = (
                run_spec.segmenter.type.lower() if run_spec.segmenter else "identity"
            )
            if ground_truth is not None and segmenter_type_eff == "identity":
                similarity_extrinsic = _compute_similarity_extrinsic(
                    all_images,
                    ground_truth,
                    run_spec.similarity_search.top_k,
                    run_spec.similarity_search.cos_distance,
                )
            elif ground_truth is not None:
                print(
                    f"Run {run_spec.name}: ground truth provided but segmenter is "
                    f"'{segmenter_type_eff}' (not identity); skipping similarity-search extrinsic metrics."
                )

        if all_images:
            K = max(1, int(run_spec.repeats))
            reduction_samples: list[StageMetrics] = []
            clustering_samples: list[StageMetrics] = []
            embeddings = [img.embedding for img in all_images]
            embeddings_arr = np.array(embeddings)
            labels = None

            for i in range(K):
                seed_i = i
                # Rebuild reduction per repeat so UMAP picks up the new seed.
                reduction = _build_reduction(run_spec, random_state=seed_i)

                with profile_stage() as red_stage:
                    reduced_embeddings = reduction.reduce(embeddings)
                    clustering_images = [
                        ImageData(filename=img.filename, embedding=emb)
                        for img, emb in zip(all_images, reduced_embeddings)
                    ]
                reduction_samples.append(red_stage.metrics)

                cluster_kwargs = dict(run_spec.clustering.params)
                cluster_kwargs["random_state"] = seed_i
                with profile_stage() as clu_stage:
                    labels = config.clustering.cluster(clustering_images, **cluster_kwargs)
                clustering_samples.append(clu_stage.metrics)

            # Drop iter 0 (absorbs JIT / cache warmup) when we have spare samples.
            if K >= 2:
                reduction_samples = reduction_samples[1:]
                clustering_samples = clustering_samples[1:]

            reduction_agg = StageMetricsAgg.from_samples(reduction_samples)
            clustering_agg = StageMetricsAgg.from_samples(clustering_samples)

            labels_arr = np.array(labels)
            clustering_quality = calculate_clustering_metrics(embeddings_arr, labels_arr)

            segmenter_type_eff = (
                run_spec.segmenter.type.lower() if run_spec.segmenter else "identity"
            )
            if ground_truth is not None and segmenter_type_eff == "identity":
                clustering_extrinsic = _compute_extrinsic(all_images, labels_arr, ground_truth)
            elif ground_truth is not None:
                print(
                    f"Run {run_spec.name}: ground truth provided but segmenter is "
                    f"'{segmenter_type_eff}' (not identity); skipping extrinsic metrics."
                )

            n_clusters = int(np.sum(np.unique(labels_arr) >= 0)) if labels_arr.size > 0 else 0
            cluster_count = n_clusters
        else:
            cluster_count = 0

        if all_images and cluster_plot_options and cluster_plot_options.get("enabled"):
            plot_metrics: dict[str, Any] = {
                "image_count": image_count,
                "n_clusters": cluster_count,
            }
            if clustering_quality is not None:
                plot_metrics.update(clustering_quality.to_flat_dict())
            if clustering_extrinsic is not None:
                plot_metrics.update(clustering_extrinsic.to_flat_dict())
            _maybe_plot_clusters(
                run_spec=run_spec,
                output_dir=output_dir,
                embeddings_arr=embeddings_arr,
                reduced_embeddings=reduced_embeddings,
                labels_arr=labels_arr,
                filenames=[img.filename for img in all_images],
                metrics=plot_metrics,
                options=cluster_plot_options,
            )
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = str(exc)
    finally:
        if "storage" in locals():
            storage.close()  # type: ignore

    finished_at = _utc_now()

    return BenchmarkResult(
        run_name=run_spec.name,
        run_id=run_spec.run_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        image_count=image_count,
        cluster_count=cluster_count,
        storage_type=run_spec.storage.type,
        embedding_type=run_spec.embedding.type,
        clustering_type=run_spec.clustering.type,
        segmenter_type=run_spec.segmenter.type if run_spec.segmenter else "identity",
        reduction_type=run_spec.reduction.type if run_spec.reduction else "identity",
        error=error,
        ingest=ingest_metrics,
        reduction=reduction_agg,
        similarity_search=similarity_metrics,
        avg_neighbor_distance=avg_neighbor_distance,
        similarity_extrinsic=similarity_extrinsic,
        clustering=clustering_agg,
        clustering_quality=clustering_quality,
        clustering_extrinsic=clustering_extrinsic,
        config=run_spec,
    )


def _list_images(
    dataset_root: Path,
    limit: int | None = None,
) -> list[str]:
    allowed = DEFAULT_IMAGE_EXTENSIONS
    image_paths: list[str] = []
    iterator = sorted(dataset_root.rglob("*"))

    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        image_paths.append(str(path))
        if limit is not None and len(image_paths) >= limit:
            break

    return image_paths


def _build_storage(run_spec: BenchmarkRunSpec, output_dir: Path):
    storage_name = run_spec.storage.type.lower()
    params = dict(run_spec.storage.params)

    if storage_name == "sqlite":
        db_path = params.get("db_path")
        if db_path is None:
            db_path = str(output_dir / "db" / f"{run_spec.run_id}.sqlite")
        db_path = str(db_path).format(run_id=run_spec.run_id, output_dir=output_dir)
        os.makedirs(Path(db_path).parent, exist_ok=True)
        return SQLiteStorage(db_path)

    if storage_name in {"postgres", "postgresql"}:
        db_url = params.get("db_url")
        if not db_url:
            raise ValueError(f"Run {run_spec.name}: storage.params.db_url is required for PostgreSQL.")
        return PostgreSQLStorage(str(db_url))

    raise ValueError(f"Run {run_spec.name}: unsupported storage type '{run_spec.storage.type}'.")


def _build_embedding(run_spec: BenchmarkRunSpec):
    embedding_name = run_spec.embedding.type.lower()

    if embedding_name == "custom":
        params = run_spec.embedding.params
        model_name = params.get("model")
        weight_path = params.get("weight_path")
        if not isinstance(model_name, str) or not isinstance(weight_path, str):
            raise ValueError(
                f"Run {run_spec.name}: custom embedding requires params.model and params.weight_path."
            )
        return CustomEmbeddingModel(
            name=EmbeddingModelNames(model_name.lower()),
            weight_path=weight_path,
        )

    try:
        enum_name = EmbeddingModelNames(embedding_name)
    except ValueError as exc:
        valid = ", ".join(name.value for name in EmbeddingModelNames)
        raise ValueError(
            f"Run {run_spec.name}: unsupported embedding '{run_spec.embedding.type}'. Valid values: {valid}."
        ) from exc

    return get_model(enum_name)


def _build_clustering(run_spec: BenchmarkRunSpec):
    name = run_spec.clustering.type.lower()
    if name == "kmeans":
        return KMeansClusterer()
    if name == "dbscan":
        return DBSCANClusterer()
    if name == "hdbscan":
        return HDBSCANClusterer()
    if name == "optics":
        return OPTICSClusterer()
    if name == "agglomerative":
        return AgglomerativeClusterer()
    if name == "spectral":
        return SpectralClusterer()
    if name == "gmm":
        return GMMClusterer()
    if name == "affinity_propagation":
        return AffinityPropagationClusterer()

    raise ValueError(f"Run {run_spec.name}: unsupported clustering type '{run_spec.clustering.type}'.")


def _build_segmenter(run_spec: BenchmarkRunSpec):
    if not run_spec.segmenter:
        return IdentitySegmenter()

    name = run_spec.segmenter.type.lower()
    params = run_spec.segmenter.params

    if name == "yolo":
        model_path = params.get("model_path")
        if not model_path:
            raise ValueError(f"Run {run_spec.name}: segmenter.type 'yolo' requires params.model_path.")
        threshold = params.get("threshold", 0.5)
        merge_threshold = params.get("merge_threshold", 0.8)
        padding = params.get("padding", 0.0)
        return YoloSegmenter(model_path, threshold=threshold, merge_threshold=merge_threshold, padding=padding)
    
    if name == "identity":
        padding = params.get("padding", 0.0)
        return IdentitySegmenter(padding=padding)

    raise ValueError(f"Run {run_spec.name}: unsupported segmenter type '{run_spec.segmenter.type}'.")


def _build_reduction(run_spec: BenchmarkRunSpec, random_state: int | None = None):
    if not run_spec.reduction:
        return IdentityReduction()

    name = run_spec.reduction.type.lower()
    params = dict(run_spec.reduction.params)

    if name == "pca":
        return PCAReduction(**params)
    if name == "umap":
        if random_state is not None:
            params["random_state"] = random_state
        return UMAPReduction(**params)
    if name == "isomap":
        return IsomapReduction(**params)
    if name == "kernel_pca":
        return KernelPCAReduction(**params)
    if name == "identity":
        return IdentityReduction()

    raise ValueError(f"Run {run_spec.name}: unsupported reduction type '{run_spec.reduction.type}'.")


def _clear_storage(storage: Any) -> None:
    clear_fn = getattr(storage, "clear", None)
    if clear_fn is None:
        return
    clear_fn()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_extrinsic(
    all_images: list[ImageData],
    labels_arr: np.ndarray,
    ground_truth: dict[str, str],
) -> ExtrinsicMetrics:
    matched_indices: list[int] = []
    matched_styles: list[str] = []
    for i, img in enumerate(all_images):
        basename = Path(img.filename).name
        style = ground_truth.get(basename)
        if style is not None:
            matched_indices.append(i)
            matched_styles.append(style)

    n_total = len(all_images)
    n_matched = len(matched_indices)
    coverage = (n_matched / n_total) if n_total > 0 else None

    if n_matched < 2:
        return ExtrinsicMetrics(
            ari=None,
            nmi=None,
            pairwise_f1=None,
            n_matched=n_matched,
            n_classes=len(set(matched_styles)),
            coverage=coverage,
        )

    unique_styles = sorted(set(matched_styles))
    style_to_id = {s: idx for idx, s in enumerate(unique_styles)}
    labels_true = np.array([style_to_id[s] for s in matched_styles])
    labels_pred = labels_arr[np.array(matched_indices)]

    metrics = calculate_extrinsic_metrics(labels_pred, labels_true)
    metrics.coverage = coverage
    return metrics


def _compute_similarity_extrinsic(
    all_images: list[ImageData],
    ground_truth: dict[str, str],
    top_k: int,
    cos_distance: bool,
) -> SimilaritySearchExtrinsicMetrics:
    matched_indices: list[int] = []
    matched_styles: list[str] = []
    for i, img in enumerate(all_images):
        basename = Path(img.filename).name
        style = ground_truth.get(basename)
        if style is not None:
            matched_indices.append(i)
            matched_styles.append(style)

    n_total = len(all_images)
    n_matched = len(matched_indices)
    coverage = (n_matched / n_total) if n_total > 0 else None

    if n_matched < 2:
        return SimilaritySearchExtrinsicMetrics(
            precision_at_k=None,
            recall_at_k=None,
            map_at_k=None,
            mrr=None,
            k_effective=0,
            n_matched=n_matched,
            n_classes=len(set(matched_styles)),
            coverage=coverage,
        )

    unique_styles = sorted(set(matched_styles))
    style_to_id = {s: idx for idx, s in enumerate(unique_styles)}
    labels_true = np.array([style_to_id[s] for s in matched_styles])
    embeddings = np.array([all_images[i].embedding for i in matched_indices])

    metrics = calculate_similarity_search_extrinsic(
        embeddings=embeddings,
        labels_true=labels_true,
        top_k=top_k,
        cos_distance=cos_distance,
    )
    metrics.coverage = coverage
    return metrics


def _maybe_plot_clusters(
    run_spec: BenchmarkRunSpec,
    output_dir: Path,
    embeddings_arr: np.ndarray,
    reduced_embeddings: list[list[float]],
    labels_arr: np.ndarray,
    filenames: list[str],
    metrics: dict[str, Any],
    options: dict[str, Any],
) -> None:
    from src.evaluation.cluster_plot import plot_clusters_2d, project_to_2d

    try:
        reduction_is_2d = (
            run_spec.reduction is not None
            and run_spec.reduction.type.lower() != "identity"
            and int(run_spec.reduction.params.get("n_components", 0)) == 2
        )

        if reduction_is_2d:
            points_2d = np.array(reduced_embeddings)
        else:
            points_2d = project_to_2d(embeddings_arr)

        output_path = output_dir / f"{run_spec.name}_{run_spec.run_id}_cluster.html"
        title = f"{run_spec.name} — {run_spec.embedding.type} / {run_spec.clustering.type}"
        plot_clusters_2d(
            points_2d=points_2d,
            labels=labels_arr,
            output_path=output_path,
            title=title,
            show_noise=bool(options.get("show_noise", True)),
            filenames=filenames,
            metrics=metrics,
            images_root=str(options.get("images_root", "")),
        )
        print(f"Wrote cluster plot: {output_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Run {run_spec.name}: cluster plot failed: {exc}")


def _calculate_distance(emb1: list[float], emb2: list[float], cos_distance: bool) -> float:
    e1 = np.array(emb1)
    e2 = np.array(emb2)
    if cos_distance:
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        # Clip to avoid numerical precision issues resulting in values slightly outside [0, 2]
        dot = np.dot(e1, e2) / (norm1 * norm2)
        return float(1.0 - np.clip(dot, -1.0, 1.0))

    return float(np.linalg.norm(e1 - e2))
