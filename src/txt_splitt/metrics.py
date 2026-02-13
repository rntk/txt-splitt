"""Prometheus metrics for pipeline stages."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import REGISTRY as _DEFAULT_REGISTRY
from prometheus_client import CollectorRegistry, Counter, Histogram


class PipelineMetrics:
    """Records Prometheus metrics for each pipeline stage.

    Uses a Histogram for duration (which also provides call counts via
    ``_count``) and a Counter for failures.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else _DEFAULT_REGISTRY

        self._duration = Histogram(
            "pipeline_stage_duration_seconds",
            "Duration of pipeline stage execution in seconds",
            labelnames=["pipeline", "stage"],
            registry=reg,
        )
        self._failures = Counter(
            "pipeline_stage_failures_total",
            "Total number of pipeline stage failures",
            labelnames=["pipeline", "stage"],
            registry=reg,
        )

    @contextmanager
    def stage(self, pipeline: str, stage_name: str) -> Generator[None, None, None]:
        """Context manager that times a stage and records failures."""
        start = time.monotonic()
        failed = False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            elapsed = time.monotonic() - start
            self._duration.labels(pipeline=pipeline, stage=stage_name).observe(elapsed)
            if failed:
                self._failures.labels(pipeline=pipeline, stage=stage_name).inc()


class NoOpMetrics:
    """A no-op metrics recorder used when metrics are disabled.

    Follows the same interface as ``PipelineMetrics`` but does nothing.
    """

    @contextmanager
    def stage(self, pipeline: str, stage_name: str) -> Generator[None, None, None]:
        """No-op context manager."""
        yield
