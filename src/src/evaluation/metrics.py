import os
import platform
import sys
import time
from dataclasses import dataclass

import psutil
import torch

from src.evaluation.models import StageMetrics

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None


@dataclass(slots=True)
class RuntimeInfo:
    host: str
    platform: str
    python_version: str
    cpu_count_logical: int


class StageProfiler:
    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        self._cpu_count = max(psutil.cpu_count(logical=True) or 1, 1)
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._start_rss_mb = 0.0
        self.metrics: StageMetrics | None = None

    def __enter__(self) -> "StageProfiler":
        self._start_wall = time.perf_counter()
        cpu_times = self._process.cpu_times()
        self._start_cpu = cpu_times.user + cpu_times.system
        self._start_rss_mb = self._process.memory_info().rss / (1024 * 1024)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        end_wall = time.perf_counter()
        cpu_times = self._process.cpu_times()
        end_cpu = cpu_times.user + cpu_times.system
        end_rss_mb = self._process.memory_info().rss / (1024 * 1024)

        wall_time = end_wall - self._start_wall
        cpu_time = end_cpu - self._start_cpu
        cpu_percent_estimated = 0.0
        if wall_time > 0:
            cpu_percent_estimated = 100.0 * cpu_time / wall_time / self._cpu_count

        vram_peak_mb = None
        if torch.cuda.is_available():
            # max_memory_allocated returns bytes
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        self.metrics = StageMetrics(
            wall_time_s=wall_time,
            cpu_time_s=cpu_time,
            cpu_percent_estimated=cpu_percent_estimated,
            rss_start_mb=self._start_rss_mb,
            rss_end_mb=end_rss_mb,
            rss_delta_mb=end_rss_mb - self._start_rss_mb,
            peak_rss_mb=_read_peak_rss_mb(),
            vram_peak_mb=vram_peak_mb,
        )


def profile_stage() -> StageProfiler:
    return StageProfiler()


def get_runtime_info() -> RuntimeInfo:
    return RuntimeInfo(
        host=platform.node(),
        platform=platform.platform(),
        python_version=sys.version,
        cpu_count_logical=max(psutil.cpu_count(logical=True) or 1, 1),
    )


def _read_peak_rss_mb() -> float | None:
    if resource is None:
        return None

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS reports bytes, Linux reports kilobytes.
        peak_bytes = float(peak)
    else:
        peak_bytes = float(peak) * 1024.0

    return peak_bytes / (1024.0 * 1024.0)
