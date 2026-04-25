import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.cluster.cluster import DBSCANClusterer, HDBSCANClusterer, KMeansClusterer, OPTICSClusterer
from src.configuration import Configuration
from src.embedding.custom import CustomEmbeddingModel
from src.embedding.embeddings import EmbeddingModelNames, get_model
from src.evaluation.clustering_metrics import calculate_clustering_metrics
from src.evaluation.metrics import get_runtime_info, profile_stage
from src.evaluation.models import BenchmarkResult, BenchmarkRunSpec
from src.storage.postgresql import PostgreSQLStorage
from src.storage.sqlite import SQLiteStorage

DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

def run_benchmarks(run_specs: list[BenchmarkRunSpec], dataset_path: str, output_dir: str) -> list[BenchmarkResult]:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise ValueError(f"Dataset path must be a directory: {dataset_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    print(f"Starting benchmarks: {len(run_specs)} runs to execute.")
    for run_spec in run_specs:
        print(f"Running benchmark: {run_spec.name} (ID: {run_spec.run_id})")
        result = _run_single(run_spec=run_spec, dataset_root=dataset_root, output_dir=Path(output_dir))
        results.append(result)
        print(f"Completed run: {run_spec.name} with status {result.status}.")

    return results


def write_results(results: list[BenchmarkResult], output_dir: str, prefix: str = "benchmark") -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records = [result.to_record() for result in results]
    runtime_info = asdict(get_runtime_info())

    json_path = output_root / f"{prefix}_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"runtime": runtime_info, "results": records}, file, indent=2)

    csv_path = output_root / f"{prefix}_{timestamp}.csv"
    field_names = _build_csv_columns(records)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    return str(csv_path), str(json_path)


def _run_single(run_spec: BenchmarkRunSpec, dataset_root: Path, output_dir: Path) -> BenchmarkResult:
    started_at = _utc_now()

    setup_metrics = None
    clear_metrics = None
    ingest_metrics = None
    query_metrics = None
    clustering_metrics = None
    clustering_quality = None

    image_count = 0
    cluster_count = None
    status = "success"
    error = None

    with profile_stage() as total_stage:
        try:
            image_paths = _list_images(
                dataset_root=dataset_root,
            )
            if not image_paths:
                raise ValueError(f"Run {run_spec.name}: no images found in {dataset_root}")
            image_count = len(image_paths)

            with profile_stage() as setup_stage:
                storage = _build_storage(run_spec, output_dir)
                embedding = _build_embedding(run_spec)
                clustering = _build_clustering(run_spec)
                config = Configuration(storage=storage, embedding=embedding, clustering=clustering)
            setup_metrics = setup_stage.metrics

            if run_spec.clear_storage:
                with profile_stage() as clear_stage:
                    _clear_storage(storage)
                clear_metrics = clear_stage.metrics

            with profile_stage() as ingest_stage:
                for image_path in image_paths:
                    config.save_image(image_path)
            ingest_metrics = ingest_stage.metrics

            if run_spec.distance_query.enabled:
                target_index = min(run_spec.distance_query.target_index, image_count - 1)
                target_path = image_paths[target_index]
                with profile_stage() as query_stage:
                    nearest = config.get_by_distance(
                        target_path,
                        cos_distance=run_spec.distance_query.cos_distance,
                    )
                    _ = nearest[: run_spec.distance_query.top_k]
                query_metrics = query_stage.metrics

            with profile_stage() as cluster_stage:
                all_images = config.storage.get_all_images()
                if all_images:
                    labels = config.clustering.cluster(all_images, **run_spec.clustering.params)
                    labels_arr = np.array(labels)
                    embeddings_arr = np.array([img.embedding for img in all_images])
                    
                    # Calculate quality metrics
                    clustering_quality = calculate_clustering_metrics(embeddings_arr, labels_arr)
                    
                    # Reconstruct clusters to get count (similar to config.cluster_images)
                    n_clusters = int(labels_arr.max() + 1) if labels_arr.size > 0 else 0
                    cluster_count = n_clusters
                    # Add noise points to count if they exist
                    if (labels_arr == -1).any():
                        cluster_count += int((labels_arr == -1).sum())
                else:
                    cluster_count = 0
            clustering_metrics = cluster_stage.metrics
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)

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
        error=error,
        setup=setup_metrics,
        clear_storage=clear_metrics,
        ingest=ingest_metrics,
        distance_query=query_metrics,
        clustering=clustering_metrics,
        clustering_quality=clustering_quality,
        total=total_stage.metrics,
    )


def _list_images(
    dataset_root: Path,
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

    raise ValueError(f"Run {run_spec.name}: unsupported clustering type '{run_spec.clustering.type}'.")


def _clear_storage(storage: Any) -> None:
    clear_fn = getattr(storage, "clear", None)
    if clear_fn is None:
        return
    clear_fn()


def _build_csv_columns(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []

    preferred = [
        "run_name",
        "run_id",
        "status",
        "started_at",
        "finished_at",
        "image_count",
        "cluster_count",
        "storage_type",
        "embedding_type",
        "clustering_type",
        "error",
    ]

    dynamic = sorted({key for record in records for key in record.keys() if key not in preferred})
    return preferred + dynamic


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
