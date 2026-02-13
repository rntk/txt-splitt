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
    ``_count``) and a Counter for failures broken down by error type.
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
            labelnames=["pipeline", "stage", "error"],
            registry=reg,
        )

    @contextmanager
    def stage(self, pipeline: str, stage_name: str) -> Generator[None, None, None]:
        """Context manager that times a stage and records failures."""
        start = time.monotonic()
        try:
            yield
        except Exception as exc:
            elapsed = time.monotonic() - start
            self._duration.labels(pipeline=pipeline, stage=stage_name).observe(elapsed)
            self._failures.labels(
                pipeline=pipeline,
                stage=stage_name,
                error=type(exc).__name__,
            ).inc()
            raise
        else:
            elapsed = time.monotonic() - start
            self._duration.labels(pipeline=pipeline, stage=stage_name).observe(elapsed)


class NoOpMetrics:
    """A no-op metrics recorder used when metrics are disabled.

    Follows the same interface as ``PipelineMetrics`` but does nothing.
    """

    @contextmanager
    def stage(self, pipeline: str, stage_name: str) -> Generator[None, None, None]:
        """No-op context manager."""
        yield
