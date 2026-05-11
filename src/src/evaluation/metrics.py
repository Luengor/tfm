import os
import platform
import sys
import threading
import time
from dataclasses import dataclass

import psutil
import torch

from src.evaluation.models import StageMetrics

_POLL_INTERVAL_S = 0.05  # 50 ms — fine enough for stages measured in seconds


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
        self._peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self.metrics: StageMetrics | None = None

    def _monitor_rss(self) -> None:
        while not self._stop_event.wait(_POLL_INTERVAL_S):
            try:
                rss = self._process.memory_info().rss
                if rss > self._peak_rss_bytes:
                    self._peak_rss_bytes = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def __enter__(self) -> "StageProfiler":
        self._start_wall = time.perf_counter()
        cpu_times = self._process.cpu_times()
        self._start_cpu = cpu_times.user + cpu_times.system
        start_rss = self._process.memory_info().rss
        self._start_rss_mb = start_rss / (1024 * 1024)
        self._peak_rss_bytes = start_rss

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_rss, daemon=True)
        self._monitor_thread.start()

        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)

        end_wall = time.perf_counter()
        cpu_times = self._process.cpu_times()
        end_cpu = cpu_times.user + cpu_times.system
        end_rss = self._process.memory_info().rss
        end_rss_mb = end_rss / (1024 * 1024)

        # Final sample — the monitor thread may have stopped just before exit
        if end_rss > self._peak_rss_bytes:
            self._peak_rss_bytes = end_rss

        wall_time = end_wall - self._start_wall
        cpu_time = end_cpu - self._start_cpu
        cpu_percent_estimated = 0.0
        if wall_time > 0:
            cpu_percent_estimated = 100.0 * cpu_time / wall_time / self._cpu_count

        vram_peak_mb = None
        if torch.cuda.is_available():
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        self.metrics = StageMetrics(
            wall_time_s=wall_time,
            cpu_time_s=cpu_time,
            cpu_percent_estimated=cpu_percent_estimated,
            rss_start_mb=self._start_rss_mb,
            rss_end_mb=end_rss_mb,
            rss_delta_mb=end_rss_mb - self._start_rss_mb,
            peak_rss_mb=self._peak_rss_bytes / (1024 * 1024),
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
