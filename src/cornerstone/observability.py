"""Lightweight metrics instrumentation helpers."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any


class MetricsRecorder:
    """Emit simple structured metrics via the standard logging pipeline."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        namespace: str = "cornerstone",
        logger: logging.Logger | None = None,
    ) -> None:
        self._enabled = enabled
        self._namespace = namespace.strip() or "cornerstone"
        self._logger = logger or logging.getLogger("cornerstone.metrics")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def increment(self, metric: str, *, value: int = 1, **tags: Any) -> None:
        """Increment a counter metric."""

        if not self._enabled:
            return
        value = int(value)
        self._emit(metric, fields={"value": value}, tags=tags)

    def record_timing(self, metric: str, duration_seconds: float, **tags: Any) -> None:
        """Emit a timing metric in milliseconds."""

        if not self._enabled:
            return
        duration_ms = max(duration_seconds * 1000.0, 0.0)
        self._emit(metric, fields={"duration_ms": round(duration_ms, 4)}, tags=tags)

    @contextmanager
    def track_timing(self, metric: str, **tags: Any):
        """Context manager that records execution time for the wrapped block."""

        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record_timing(metric, elapsed, **tags)

    def _emit(self, metric: str, *, fields: dict[str, Any], tags: dict[str, Any]) -> None:
        tag_segments = [
            f"{key}={self._stringify(value)}"
            for key, value in sorted((key, value) for key, value in tags.items() if value is not None)
        ]
        field_segments = [
            f"{key}={self._stringify(value)}"
            for key, value in sorted(fields.items())
        ]
        segment_parts = field_segments + tag_segments
        message = f"{self._namespace}.{metric}"
        if segment_parts:
            message = f"{message} {' '.join(segment_parts)}"
        self._logger.info(message)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}" if not value.is_integer() else f"{int(value)}"
        return str(value)

