import hashlib
import json
from pathlib import Path
from typing import Any

from src.evaluation.models import BenchmarkRunSpec, ComponentSpec, SimilaritySearchSpec


class ConfigurationError(ValueError):
    pass


DEFAULT_CLUSTER_PLOT_OPTIONS: dict[str, Any] = {
    "enabled": False,
    "show_noise": True,
}


def load_configuration(path: str) -> list[BenchmarkRunSpec]:
    return load_configuration_with_options(path)[0]


def load_configuration_with_options(
    path: str,
) -> tuple[list[BenchmarkRunSpec], dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    with source.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a JSON object.")

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ConfigurationError("Configuration must include a non-empty 'runs' list.")

    cluster_plot_options = _parse_cluster_plot(payload.get("cluster_plot"))

    run_specs: list[BenchmarkRunSpec] = []
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, dict):
            raise ConfigurationError(f"Run at index {index} must be an object.")

        run_name = _require_str(raw, "name", index)
        storage_spec = _parse_component(raw.get("storage"), "storage", index)
        embedding_spec = _parse_component(raw.get("embedding"), "embedding", index)
        clustering_spec = _parse_component(raw.get("clustering"), "clustering", index)

        segmenter_spec = None
        if "segmenter" in raw and raw["segmenter"] is not None:
             segmenter_spec = _parse_component(raw.get("segmenter"), "segmenter", index)

        reduction_spec = None
        if "reduction" in raw and raw["reduction"] is not None:
             reduction_spec = _parse_component(raw.get("reduction"), "reduction", index)

        similarity_search = _parse_similarity_search(raw.get("similarity_search"), index)

        clear_storage = raw.get("clear_storage", True)
        if not isinstance(clear_storage, bool):
            raise ConfigurationError(f"Run {run_name}: clear_storage must be a boolean.")

        limit = raw.get("limit")
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ConfigurationError(f"Run {run_name}: limit must be a positive integer.")

        run_spec = BenchmarkRunSpec(
            name=run_name,
            storage=storage_spec,
            embedding=embedding_spec,
            clustering=clustering_spec,
            segmenter=segmenter_spec,
            reduction=reduction_spec,
            similarity_search=similarity_search,
            clear_storage=clear_storage,
            limit=limit,
        )
        run_spec.run_id = make_run_id(run_spec)
        run_specs.append(run_spec)

    return run_specs, cluster_plot_options


def _parse_cluster_plot(raw: Any) -> dict[str, Any]:
    options = dict(DEFAULT_CLUSTER_PLOT_OPTIONS)
    if raw is None:
        return options
    if not isinstance(raw, dict):
        raise ConfigurationError("cluster_plot must be an object.")

    if "enabled" in raw:
        if not isinstance(raw["enabled"], bool):
            raise ConfigurationError("cluster_plot.enabled must be a boolean.")
        options["enabled"] = raw["enabled"]

    if "show_noise" in raw:
        if not isinstance(raw["show_noise"], bool):
            raise ConfigurationError("cluster_plot.show_noise must be a boolean.")
        options["show_noise"] = raw["show_noise"]

    return options


def _parse_component(raw: Any, component_name: str, run_index: int) -> ComponentSpec:
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Run index {run_index}: {component_name} must be an object.")

    c_type = raw.get("type")
    if not isinstance(c_type, str) or not c_type.strip():
        raise ConfigurationError(f"Run index {run_index}: {component_name}.type must be a non-empty string.")

    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise ConfigurationError(f"Run index {run_index}: {component_name}.params must be an object.")

    return ComponentSpec(type=c_type.strip(), params=params)


def _parse_similarity_search(raw: Any, run_index: int) -> SimilaritySearchSpec:
    if raw is None:
        return SimilaritySearchSpec()
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Run index {run_index}: similarity_search must be an object.")

    enabled = raw.get("enabled", False)
    top_k = raw.get("top_k", 3)
    cos_distance = raw.get("cos_distance", True)

    if not isinstance(enabled, bool):
        raise ConfigurationError(f"Run index {run_index}: similarity_search.enabled must be a boolean.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ConfigurationError(f"Run index {run_index}: similarity_search.top_k must be a positive integer.")
    if not isinstance(cos_distance, bool):
        raise ConfigurationError(f"Run index {run_index}: similarity_search.cos_distance must be a boolean.")

    return SimilaritySearchSpec(
        enabled=enabled,
        top_k=top_k,
        cos_distance=cos_distance,
    )


def _require_str(payload: dict[str, Any], key: str, run_index: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Run index {run_index}: {key} must be a non-empty string.")
    return value.strip()


def make_run_id(spec: BenchmarkRunSpec) -> str:
    normalized = {
        "name": spec.name,
        "storage": {"type": spec.storage.type.lower(), "params": spec.storage.params},
        "embedding": {"type": spec.embedding.type.lower(), "params": spec.embedding.params},
        "clustering": {"type": spec.clustering.type.lower(), "params": spec.clustering.params},
        "segmenter": {"type": spec.segmenter.type.lower(), "params": spec.segmenter.params} if spec.segmenter else None,
        "reduction": {"type": spec.reduction.type.lower(), "params": spec.reduction.params} if spec.reduction else None,
        "limit": spec.limit,
    }
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest[:12]
