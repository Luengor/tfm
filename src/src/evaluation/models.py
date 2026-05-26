import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ComponentSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimilaritySearchSpec:
    enabled: bool = False
    top_k: int = 3
    cos_distance: bool = True
    sample_n: int | None = None
    sample_seed: int | None = None


@dataclass(slots=True)
class BenchmarkRunSpec:
    name: str
    storage: ComponentSpec
    embedding: ComponentSpec
    clustering: ComponentSpec
    segmenter: ComponentSpec | None = None
    reduction: ComponentSpec | None = None
    similarity_search: SimilaritySearchSpec = field(default_factory=SimilaritySearchSpec)
    clear_storage: bool = True
    limit: int | None = None
    repeats: int = 1
    run_id: str = ""


@dataclass(slots=True)
class StageMetrics:
    wall_time_s: float
    cpu_time_s: float
    cpu_percent_estimated: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    peak_rss_delta_mb: float | None
    vram_peak_mb: float | None = None

    def to_flat_dict(self, prefix: str) -> dict[str, float | None]:
        return {
            f"{prefix}_wall_time_s": self.wall_time_s,
            f"{prefix}_cpu_time_s": self.cpu_time_s,
            f"{prefix}_cpu_percent_estimated": self.cpu_percent_estimated,
            f"{prefix}_rss_start_mb": self.rss_start_mb,
            f"{prefix}_rss_end_mb": self.rss_end_mb,
            f"{prefix}_rss_delta_mb": self.rss_delta_mb,
            f"{prefix}_peak_rss_delta_mb": self.peak_rss_delta_mb,
            f"{prefix}_vram_peak_mb": self.vram_peak_mb,
        }


@dataclass(slots=True)
class StageMetricsAgg:
    """Mean +/- std aggregation of multiple StageMetrics samples."""
    wall_time_s: float
    wall_time_s_std: float
    cpu_time_s: float
    cpu_time_s_std: float
    cpu_percent_estimated: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    rss_delta_mb_std: float
    peak_rss_delta_mb: float | None
    peak_rss_delta_mb_std: float | None
    vram_peak_mb: float | None
    vram_peak_mb_std: float | None
    n: int

    @classmethod
    def from_samples(
        cls,
        samples: list["StageMetrics"],
        drop_first_for_cost: bool = False,
    ) -> "StageMetricsAgg":
        """Aggregate per-repeat samples.

        Cost fields (wall, cpu, cpu%) drop iter 0 when ``drop_first_for_cost``
        and ``len(samples) >= 2``, since iter 0 absorbs JIT and cache warm-up.
        Memory fields (RSS, peak RSS delta, VRAM) keep all K samples: after
        iter 0 the allocator/runtime hold pages and the per-iter delta above
        baseline systematically under-reports the real stage cost. Including
        iter 0 in the memory aggregation surfaces the true load-time peak.
        """
        import statistics as _st
        if not samples:
            raise ValueError("StageMetricsAgg.from_samples requires at least one sample.")

        cost_samples = samples[1:] if (drop_first_for_cost and len(samples) >= 2) else samples
        mem_samples = samples

        def _mean(values: list[float]) -> float:
            return float(_st.fmean(values))

        def _std(values: list[float]) -> float:
            return float(_st.stdev(values)) if len(values) > 1 else 0.0

        def _opt_mean(values: list[float | None]) -> float | None:
            clean = [v for v in values if v is not None]
            return _mean(clean) if clean else None

        def _opt_std(values: list[float | None]) -> float | None:
            clean = [v for v in values if v is not None]
            if not clean:
                return None
            return float(_st.stdev(clean)) if len(clean) > 1 else 0.0

        wall = [s.wall_time_s for s in cost_samples]
        cpu = [s.cpu_time_s for s in cost_samples]
        cpup = [s.cpu_percent_estimated for s in cost_samples]
        rss_s = [s.rss_start_mb for s in mem_samples]
        rss_e = [s.rss_end_mb for s in mem_samples]
        rss_d = [s.rss_delta_mb for s in mem_samples]
        peak = [s.peak_rss_delta_mb for s in mem_samples]
        vram = [s.vram_peak_mb for s in mem_samples]

        return cls(
            wall_time_s=_mean(wall),
            wall_time_s_std=_std(wall),
            cpu_time_s=_mean(cpu),
            cpu_time_s_std=_std(cpu),
            cpu_percent_estimated=_mean(cpup),
            rss_start_mb=_mean(rss_s),
            rss_end_mb=_mean(rss_e),
            rss_delta_mb=_mean(rss_d),
            rss_delta_mb_std=_std(rss_d),
            peak_rss_delta_mb=_opt_mean(peak),
            peak_rss_delta_mb_std=_opt_std(peak),
            vram_peak_mb=_opt_mean(vram),
            vram_peak_mb_std=_opt_std(vram),
            n=len(cost_samples),
        )

    def to_flat_dict(self, prefix: str) -> dict[str, float | int | None]:
        return {
            f"{prefix}_wall_time_s": self.wall_time_s,
            f"{prefix}_wall_time_s_std": self.wall_time_s_std,
            f"{prefix}_cpu_time_s": self.cpu_time_s,
            f"{prefix}_cpu_time_s_std": self.cpu_time_s_std,
            f"{prefix}_cpu_percent_estimated": self.cpu_percent_estimated,
            f"{prefix}_rss_start_mb": self.rss_start_mb,
            f"{prefix}_rss_end_mb": self.rss_end_mb,
            f"{prefix}_rss_delta_mb": self.rss_delta_mb,
            f"{prefix}_rss_delta_mb_std": self.rss_delta_mb_std,
            f"{prefix}_peak_rss_delta_mb": self.peak_rss_delta_mb,
            f"{prefix}_peak_rss_delta_mb_std": self.peak_rss_delta_mb_std,
            f"{prefix}_vram_peak_mb": self.vram_peak_mb,
            f"{prefix}_vram_peak_mb_std": self.vram_peak_mb_std,
            f"{prefix}_n": self.n,
        }


@dataclass(slots=True)
class ClusteringQualityMetrics:
    silhouette_score: float | None
    calinski_harabasz_score: float | None
    davies_bouldin_score: float | None
    noise_ratio: float | None
    cluster_size_cv: float | None

    def to_flat_dict(self, prefix: str = "clustering_quality") -> dict[str, float | None]:
        return {
            f"{prefix}_silhouette": self.silhouette_score,
            f"{prefix}_calinski_harabasz": self.calinski_harabasz_score,
            f"{prefix}_davies_bouldin": self.davies_bouldin_score,
            f"{prefix}_noise_ratio": self.noise_ratio,
            f"{prefix}_cluster_size_cv": self.cluster_size_cv,
        }


@dataclass(slots=True)
class ClusteringQualityMetricsAgg:
    """Mean +/- std aggregation of multiple ClusteringQualityMetrics samples."""
    silhouette_score: float | None
    silhouette_score_std: float | None
    calinski_harabasz_score: float | None
    calinski_harabasz_score_std: float | None
    davies_bouldin_score: float | None
    davies_bouldin_score_std: float | None
    noise_ratio: float | None
    noise_ratio_std: float | None
    cluster_size_cv: float | None
    cluster_size_cv_std: float | None
    n: int

    @classmethod
    def from_samples(cls, samples: list["ClusteringQualityMetrics"]) -> "ClusteringQualityMetricsAgg":
        import statistics as _st
        if not samples:
            raise ValueError("ClusteringQualityMetricsAgg.from_samples requires at least one sample.")

        def _opt_mean(values: list[float | None]) -> float | None:
            clean = [v for v in values if v is not None]
            return float(_st.fmean(clean)) if clean else None

        def _opt_std(values: list[float | None]) -> float | None:
            clean = [v for v in values if v is not None]
            if not clean:
                return None
            return float(_st.stdev(clean)) if len(clean) > 1 else 0.0

        sils = [s.silhouette_score for s in samples]
        chs = [s.calinski_harabasz_score for s in samples]
        dbs = [s.davies_bouldin_score for s in samples]
        nrs = [s.noise_ratio for s in samples]
        cvs = [s.cluster_size_cv for s in samples]

        return cls(
            silhouette_score=_opt_mean(sils),
            silhouette_score_std=_opt_std(sils),
            calinski_harabasz_score=_opt_mean(chs),
            calinski_harabasz_score_std=_opt_std(chs),
            davies_bouldin_score=_opt_mean(dbs),
            davies_bouldin_score_std=_opt_std(dbs),
            noise_ratio=_opt_mean(nrs),
            noise_ratio_std=_opt_std(nrs),
            cluster_size_cv=_opt_mean(cvs),
            cluster_size_cv_std=_opt_std(cvs),
            n=len(samples),
        )

    def to_flat_dict(self, prefix: str = "clustering_quality") -> dict[str, float | int | None]:
        return {
            f"{prefix}_silhouette": self.silhouette_score,
            f"{prefix}_silhouette_std": self.silhouette_score_std,
            f"{prefix}_calinski_harabasz": self.calinski_harabasz_score,
            f"{prefix}_calinski_harabasz_std": self.calinski_harabasz_score_std,
            f"{prefix}_davies_bouldin": self.davies_bouldin_score,
            f"{prefix}_davies_bouldin_std": self.davies_bouldin_score_std,
            f"{prefix}_noise_ratio": self.noise_ratio,
            f"{prefix}_noise_ratio_std": self.noise_ratio_std,
            f"{prefix}_cluster_size_cv": self.cluster_size_cv,
            f"{prefix}_cluster_size_cv_std": self.cluster_size_cv_std,
            f"{prefix}_n": self.n,
        }


@dataclass(slots=True)
class ExtrinsicMetrics:
    ari: float | None
    nmi: float | None
    pairwise_f1: float | None
    # Noise-excluded variant: same scores after dropping points the clusterer
    # labelled -1, so density methods (DBSCAN/HDBSCAN/OPTICS) are comparable to
    # partitional ones on the points each method actually clustered. The main
    # fields above keep the noise-as-class reading.
    ari_no_noise: float | None = None
    nmi_no_noise: float | None = None
    pairwise_f1_no_noise: float | None = None
    n_no_noise: int = 0
    n_matched: int = 0
    n_classes: int = 0
    coverage: float | None = None

    def to_flat_dict(self, prefix: str = "extrinsic") -> dict[str, float | int | None]:
        return {
            f"{prefix}_ari": self.ari,
            f"{prefix}_nmi": self.nmi,
            f"{prefix}_pairwise_f1": self.pairwise_f1,
            f"{prefix}_ari_no_noise": self.ari_no_noise,
            f"{prefix}_nmi_no_noise": self.nmi_no_noise,
            f"{prefix}_pairwise_f1_no_noise": self.pairwise_f1_no_noise,
            f"{prefix}_n_no_noise": self.n_no_noise,
            f"{prefix}_n_matched": self.n_matched,
            f"{prefix}_n_classes": self.n_classes,
            f"{prefix}_coverage": self.coverage,
        }


@dataclass(slots=True)
class ExtrinsicMetricsAgg:
    """Mean +/- std aggregation of multiple ExtrinsicMetrics samples.

    ARI / NMI / pairwise F1 are aggregated across repeats; the matching
    counters (n_matched, n_classes, coverage) depend only on ground-truth
    overlap with the corpus and are constant across repeats, so they pass
    through unchanged.
    """
    ari: float | None
    ari_std: float | None
    nmi: float | None
    nmi_std: float | None
    pairwise_f1: float | None
    pairwise_f1_std: float | None
    ari_no_noise: float | None
    ari_no_noise_std: float | None
    nmi_no_noise: float | None
    nmi_no_noise_std: float | None
    pairwise_f1_no_noise: float | None
    pairwise_f1_no_noise_std: float | None
    n_no_noise: int
    n_matched: int
    n_classes: int
    coverage: float | None
    n: int

    @classmethod
    def from_samples(cls, samples: list["ExtrinsicMetrics"]) -> "ExtrinsicMetricsAgg":
        import statistics as _st
        if not samples:
            raise ValueError("ExtrinsicMetricsAgg.from_samples requires at least one sample.")

        def _opt_mean(values: list[float | None]) -> float | None:
            clean = [v for v in values if v is not None]
            return float(_st.fmean(clean)) if clean else None

        def _opt_std(values: list[float | None]) -> float | None:
            clean = [v for v in values if v is not None]
            if not clean:
                return None
            return float(_st.stdev(clean)) if len(clean) > 1 else 0.0

        aris = [s.ari for s in samples]
        nmis = [s.nmi for s in samples]
        f1s = [s.pairwise_f1 for s in samples]
        aris_nn = [s.ari_no_noise for s in samples]
        nmis_nn = [s.nmi_no_noise for s in samples]
        f1s_nn = [s.pairwise_f1_no_noise for s in samples]
        last = samples[-1]

        return cls(
            ari=_opt_mean(aris),
            ari_std=_opt_std(aris),
            nmi=_opt_mean(nmis),
            nmi_std=_opt_std(nmis),
            pairwise_f1=_opt_mean(f1s),
            pairwise_f1_std=_opt_std(f1s),
            ari_no_noise=_opt_mean(aris_nn),
            ari_no_noise_std=_opt_std(aris_nn),
            nmi_no_noise=_opt_mean(nmis_nn),
            nmi_no_noise_std=_opt_std(nmis_nn),
            pairwise_f1_no_noise=_opt_mean(f1s_nn),
            pairwise_f1_no_noise_std=_opt_std(f1s_nn),
            n_no_noise=last.n_no_noise,
            n_matched=last.n_matched,
            n_classes=last.n_classes,
            coverage=last.coverage,
            n=len(samples),
        )

    def to_flat_dict(self, prefix: str = "extrinsic") -> dict[str, float | int | None]:
        return {
            f"{prefix}_ari": self.ari,
            f"{prefix}_ari_std": self.ari_std,
            f"{prefix}_nmi": self.nmi,
            f"{prefix}_nmi_std": self.nmi_std,
            f"{prefix}_pairwise_f1": self.pairwise_f1,
            f"{prefix}_pairwise_f1_std": self.pairwise_f1_std,
            f"{prefix}_ari_no_noise": self.ari_no_noise,
            f"{prefix}_ari_no_noise_std": self.ari_no_noise_std,
            f"{prefix}_nmi_no_noise": self.nmi_no_noise,
            f"{prefix}_nmi_no_noise_std": self.nmi_no_noise_std,
            f"{prefix}_pairwise_f1_no_noise": self.pairwise_f1_no_noise,
            f"{prefix}_pairwise_f1_no_noise_std": self.pairwise_f1_no_noise_std,
            f"{prefix}_n_no_noise": self.n_no_noise,
            f"{prefix}_n_matched": self.n_matched,
            f"{prefix}_n_classes": self.n_classes,
            f"{prefix}_coverage": self.coverage,
            f"{prefix}_n": self.n,
        }


@dataclass(slots=True)
class SimilaritySearchExtrinsicMetrics:
    precision_at_k: float | None
    recall_at_k: float | None
    map_at_k: float | None
    mrr: float | None
    k_effective: int = 0
    n_matched: int = 0
    n_classes: int = 0
    coverage: float | None = None

    def to_flat_dict(self, prefix: str = "similarity_extrinsic") -> dict[str, float | int | None]:
        return {
            f"{prefix}_precision_at_k": self.precision_at_k,
            f"{prefix}_recall_at_k": self.recall_at_k,
            f"{prefix}_map_at_k": self.map_at_k,
            f"{prefix}_mrr": self.mrr,
            f"{prefix}_k_effective": self.k_effective,
            f"{prefix}_n_matched": self.n_matched,
            f"{prefix}_n_classes": self.n_classes,
            f"{prefix}_coverage": self.coverage,
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
    segmenter_type: str | None = None
    reduction_type: str | None = None
    error: str | None = None
    ingest: StageMetrics | None = None
    reduction: StageMetricsAgg | None = None
    similarity_search: StageMetrics | None = None
    avg_neighbor_distance: float | None = None
    # Recall@k of the storage's similarity search vs an exact brute-force kNN on
    # the same queries. 1.0 for exact backends; <1.0 quantifies the ANN (HNSW)
    # approximation, giving the accuracy axis of the speed-vs-accuracy trade-off.
    ann_recall_at_k: float | None = None
    similarity_extrinsic: SimilaritySearchExtrinsicMetrics | None = None
    similarity_extrinsic_author: SimilaritySearchExtrinsicMetrics | None = None
    clustering: StageMetricsAgg | None = None
    clustering_quality: ClusteringQualityMetricsAgg | None = None
    clustering_extrinsic: ExtrinsicMetricsAgg | None = None
    clustering_extrinsic_author: ExtrinsicMetricsAgg | None = None
    clusters_per_repeat: list[int] = field(default_factory=list)
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
            "segmenter_type": self.segmenter_type,
            "reduction_type": self.reduction_type,
            "avg_neighbor_distance": self.avg_neighbor_distance,
            "ann_recall_at_k": self.ann_recall_at_k,
            "clusters_per_repeat": list(self.clusters_per_repeat),
            "error": self.error,
        }

        if self.config:
            record.update({
                "storage_params": json.dumps(self.config.storage.params),
                "embedding_params": json.dumps(self.config.embedding.params),
                "clustering_params": json.dumps(self.config.clustering.params),
                "segmenter_params": json.dumps(self.config.segmenter.params) if self.config.segmenter else None,
                "reduction_params": json.dumps(self.config.reduction.params) if self.config.reduction else None,
                "similarity_search_enabled": self.config.similarity_search.enabled,
                "similarity_search_top_k": self.config.similarity_search.top_k,
                "similarity_search_cos_distance": self.config.similarity_search.cos_distance,
                "similarity_search_sample_n": self.config.similarity_search.sample_n,
                "similarity_search_sample_seed": self.config.similarity_search.sample_seed,
                "clear_storage_enabled": self.config.clear_storage,
                "limit_parameter": self.config.limit,
                "repeats": self.config.repeats,
            })

        for stage_name in ("ingest", "reduction", "similarity_search", "clustering"):
            stage = getattr(self, stage_name)
            if stage is None:
                continue

            stage_dict = stage.to_flat_dict(stage_name)

            # Calculate throughput for relevant processing stages
            if stage_name in ("ingest", "similarity_search", "clustering") and self.image_count > 0:
                if stage.wall_time_s > 0:
                    stage_dict[f"{stage_name}_throughput_ips"] = self.image_count / stage.wall_time_s

            record.update(stage_dict)

        if self.clustering_quality:
            record.update(self.clustering_quality.to_flat_dict())

        if self.clustering_extrinsic:
            record.update(self.clustering_extrinsic.to_flat_dict())

        if self.clustering_extrinsic_author:
            record.update(self.clustering_extrinsic_author.to_flat_dict("extrinsic_author"))

        if self.similarity_extrinsic:
            record.update(self.similarity_extrinsic.to_flat_dict())

        if self.similarity_extrinsic_author:
            record.update(self.similarity_extrinsic_author.to_flat_dict("similarity_extrinsic_author"))

        return record
