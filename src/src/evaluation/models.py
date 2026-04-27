import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ComponentSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DistanceQuerySpec:
    enabled: bool = False
    target_index: int = 0
    cos_distance: bool = True
    top_k: int = 10


@dataclass(slots=True)
class SimilaritySearchSpec:
    enabled: bool = False
    top_k: int = 3
    cos_distance: bool = True


@dataclass(slots=True)
class BenchmarkRunSpec:
    name: str
    storage: ComponentSpec
    embedding: ComponentSpec
    clustering: ComponentSpec
    reduction: ComponentSpec | None = None
    distance_query: DistanceQuerySpec = field(default_factory=DistanceQuerySpec)
    similarity_search: SimilaritySearchSpec = field(default_factory=SimilaritySearchSpec)
    clear_storage: bool = True
    limit: int | None = None
    run_id: str = ""


@dataclass(slots=True)
class StageMetrics:
    wall_time_s: float
    cpu_time_s: float
    cpu_percent_estimated: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    peak_rss_mb: float | None

    def to_flat_dict(self, prefix: str) -> dict[str, float | None]:
        return {
            f"{prefix}_wall_time_s": self.wall_time_s,
            f"{prefix}_cpu_time_s": self.cpu_time_s,
            f"{prefix}_cpu_percent_estimated": self.cpu_percent_estimated,
            f"{prefix}_rss_start_mb": self.rss_start_mb,
            f"{prefix}_rss_end_mb": self.rss_end_mb,
            f"{prefix}_rss_delta_mb": self.rss_delta_mb,
            f"{prefix}_peak_rss_mb": self.peak_rss_mb,
        }


@dataclass(slots=True)
class ClusteringQualityMetrics:
    silhouette_score: float | None
    calinski_harabasz_score: float | None

    def to_flat_dict(self, prefix: str = "clustering_quality") -> dict[str, float | None]:
        return {
            f"{prefix}_silhouette": self.silhouette_score,
            f"{prefix}_calinski_harabasz": self.calinski_harabasz_score,
        }


@dataclass(slots=True)
class BenchmarkResult:
    run_name: str
    run_id: str
    status: str
    started_at: str
    finished_at: str
    image_count: int
    cluster_count: int | None
    storage_type: str
    embedding_type: str
    clustering_type: str
    reduction_type: str | None = None
    error: str | None = None
    setup: StageMetrics | None = None
    clear_storage: StageMetrics | None = None
    ingest: StageMetrics | None = None
    reduction: StageMetrics | None = None
    distance_query: StageMetrics | None = None
    similarity_search: StageMetrics | None = None
    avg_neighbor_distance: float | None = None
    clustering: StageMetrics | None = None
    clustering_quality: ClusteringQualityMetrics | None = None
    config: BenchmarkRunSpec | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "image_count": self.image_count,
            "cluster_count": self.cluster_count,
            "storage_type": self.storage_type,
            "embedding_type": self.embedding_type,
            "clustering_type": self.clustering_type,
            "reduction_type": self.reduction_type,
            "avg_neighbor_distance": self.avg_neighbor_distance,
            "error": self.error,
        }

        if self.config:
            record.update({
                "storage_params": json.dumps(self.config.storage.params),
                "embedding_params": json.dumps(self.config.embedding.params),
                "clustering_params": json.dumps(self.config.clustering.params),
                "reduction_params": json.dumps(self.config.reduction.params) if self.config.reduction else None,
                "distance_query_enabled": self.config.distance_query.enabled,
                "distance_query_target_index": self.config.distance_query.target_index,
                "distance_query_cos_distance": self.config.distance_query.cos_distance,
                "distance_query_top_k": self.config.distance_query.top_k,
                "similarity_search_enabled": self.config.similarity_search.enabled,
                "similarity_search_top_k": self.config.similarity_search.top_k,
                "similarity_search_cos_distance": self.config.similarity_search.cos_distance,
                "clear_storage_enabled": self.config.clear_storage,
                "limit_parameter": self.config.limit,
            })

        for stage_name in ("setup", "clear_storage", "ingest", "reduction", "distance_query", "similarity_search", "clustering"):
            stage = getattr(self, stage_name)
            if stage is None:
                continue
            record.update(stage.to_flat_dict(stage_name))

        if self.clustering_quality:
            record.update(self.clustering_quality.to_flat_dict())

        return record
