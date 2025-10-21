"""Lightweight metrics instrumentation helpers with optional Prometheus export."""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Tuple

try:  # pragma: no cover - optional dependency
    from prometheus_client import (
        CollectorRegistry,
        Counter as PromCounter,
        Gauge as PromGauge,
        Histogram as PromHistogram,
        CONTENT_TYPE_LATEST,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency missing
    CollectorRegistry = None  # type: ignore[assignment]
    PromCounter = None  # type: ignore[assignment]
    PromHistogram = None  # type: ignore[assignment]
    PromGauge = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    _PROMETHEUS_AVAILABLE = False

    def generate_latest(_registry):  # type: ignore[unused-ignore]
        raise RuntimeError("prometheus_client is not installed")


_PROM_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


class MetricsRecorder:
    """Emit structured metrics via logging and (optionally) Prometheus."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        namespace: str = "cornerstone",
        logger: logging.Logger | None = None,
        prometheus_enabled: bool = False,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._enabled = enabled
        self._namespace = namespace.strip() or "cornerstone"
        self._logger = logger or logging.getLogger("cornerstone.metrics")
        self._prometheus_enabled = bool(prometheus_enabled and _PROMETHEUS_AVAILABLE)
        self._prom_registry = registry if registry is not None else (
            CollectorRegistry() if self._prometheus_enabled and CollectorRegistry is not None else None
        )
        self._prom_counters: dict[Tuple[str, Tuple[str, ...]], PromCounter] = {}
        self._prom_histograms: dict[Tuple[str, Tuple[str, ...]], PromHistogram] = {}
        self._prom_gauges: dict[Tuple[str, Tuple[str, ...]], PromGauge] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def prometheus_enabled(self) -> bool:
        return self._prometheus_enabled and self._prom_registry is not None

    @property
    def prometheus_registry(self) -> CollectorRegistry | None:
        return self._prom_registry

    @property
    def prometheus_content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    def render_prometheus(self) -> bytes:
        if not self.prometheus_enabled:
            raise RuntimeError("Prometheus export is disabled")
        return generate_latest(self._prom_registry)  # type: ignore[arg-type]

    def increment(self, metric: str, *, value: int = 1, **tags: Any) -> None:
        """Increment a counter metric."""

        if not self._enabled:
            return
        value = int(value)
        clean_tags = {key: val for key, val in tags.items() if val is not None}
        self._emit(metric, fields={"value": value}, tags=clean_tags)
        self._emit_prom_counter(metric, float(max(value, 0)), clean_tags)

    def set_gauge(self, metric: str, value: float, **tags: Any) -> None:
        """Set the value of a gauge metric."""

        if not self._enabled:
            return
        clean_tags = {key: val for key, val in tags.items() if val is not None}
        self._emit(metric, fields={"value": value}, tags=clean_tags)
        self._emit_prom_gauge(metric, float(value), clean_tags)

    def record_timing(self, metric: str, duration_seconds: float, **tags: Any) -> None:
        """Emit a timing metric, recording milliseconds to logs."""

        if not self._enabled:
            return
        duration_ms = max(duration_seconds * 1000.0, 0.0)
        clean_tags = {key: val for key, val in tags.items() if val is not None}
        self._emit(metric, fields={"duration_ms": round(duration_ms, 4)}, tags=clean_tags)
        self._emit_prom_histogram(metric, max(duration_seconds, 0.0), clean_tags)

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
            for key, value in sorted((key, value) for key, value in tags.items())
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

    def _emit_prom_counter(self, metric: str, value: float, tags: dict[str, Any]) -> None:
        if not self.prometheus_enabled or PromCounter is None or self._prom_registry is None:
            return
        label_keys = tuple(sorted(tags.keys()))
        label_names = tuple(self._sanitize_label(name) for name in label_keys)
        prom_key = (metric, label_names)
        counter = self._prom_counters.get(prom_key)
        if counter is None:
            counter = PromCounter(  # type: ignore[call-arg]
                self._prom_metric_name(metric),
                f"{metric} counter",
                labelnames=list(label_names),
                registry=self._prom_registry,
            )
            self._prom_counters[prom_key] = counter
        label_values = {
            name: self._stringify(tags[key])
            for name, key in zip(label_names, label_keys)
        }
        counter.labels(**label_values).inc(value)

    def _emit_prom_histogram(self, metric: str, value: float, tags: dict[str, Any]) -> None:
        if not self.prometheus_enabled or PromHistogram is None or self._prom_registry is None:
            return
        label_keys = tuple(sorted(tags.keys()))
        label_names = tuple(self._sanitize_label(name) for name in label_keys)
        prom_key = (metric, label_names)
        histogram = self._prom_histograms.get(prom_key)
        if histogram is None:
            histogram = PromHistogram(  # type: ignore[call-arg]
                self._prom_metric_name(metric),
                f"{metric} duration",
                labelnames=list(label_names),
                registry=self._prom_registry,
            )
            self._prom_histograms[prom_key] = histogram
        label_values = {
            name: self._stringify(tags[key])
            for name, key in zip(label_names, label_keys)
        }
        histogram.labels(**label_values).observe(value)

    def _emit_prom_gauge(self, metric: str, value: float, tags: dict[str, Any]) -> None:
        if not self.prometheus_enabled or PromGauge is None or self._prom_registry is None:
            return
        label_keys = tuple(sorted(tags.keys()))
        label_names = tuple(self._sanitize_label(name) for name in label_keys)
        prom_key = (metric, label_names)
        gauge = self._prom_gauges.get(prom_key)
        if gauge is None:
            gauge = PromGauge(  # type: ignore[call-arg]
                self._prom_metric_name(metric),
                f"{metric} gauge",
                labelnames=list(label_names),
                registry=self._prom_registry,
            )
            self._prom_gauges[prom_key] = gauge
        label_values = {
            name: self._stringify(tags[key])
            for name, key in zip(label_names, label_keys)
        }
        gauge.labels(**label_values).set(value)

    def _prom_metric_name(self, metric: str) -> str:
        cleaned = _PROM_NAME_RE.sub("_", metric)
        return f"{_PROM_NAME_RE.sub('_', self._namespace)}_{cleaned}".strip("_")

    @staticmethod
    def _sanitize_label(label: str) -> str:
        sanitized = _PROM_NAME_RE.sub("_", label)
        return sanitized or "label"

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}" if not value.is_integer() else f"{int(value)}"
        return str(value)
