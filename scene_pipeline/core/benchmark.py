from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from scene_pipeline.schemas import BenchmarkMetrics, BenchmarkStage


@dataclass
class BenchmarkRecorder:
    """Small stage timer used by the pipeline and report script."""

    stages: list[BenchmarkStage] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str, **details: object) -> Iterator[None]:
        stage_started = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - stage_started) * 1000.0
            self.stages.append(
                BenchmarkStage(name=name, latency_ms=latency_ms, details=dict(details))
            )

    def metrics(self, total_frames: int) -> BenchmarkMetrics:
        total_latency_ms = (time.perf_counter() - self.started_at) * 1000.0
        total_seconds = max(total_latency_ms / 1000.0, 1e-9)
        return BenchmarkMetrics(
            total_latency_ms=total_latency_ms,
            frames_per_second=total_frames / total_seconds,
            total_frames=total_frames,
            stage_latencies=self.stages,
        )

