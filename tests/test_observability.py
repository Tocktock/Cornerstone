import logging

from cornerstone.observability import MetricsRecorder


def test_metrics_recorder_logs_when_enabled(caplog) -> None:
    metrics = MetricsRecorder(enabled=True, namespace="cornerstone.test")

    with caplog.at_level(logging.INFO, logger="cornerstone.metrics"):
        metrics.increment("ingestion.documents", project_id="proj-1")
        metrics.record_timing(
            "ingestion.total_duration",
            0.05,
            project_id="proj-1",
            chunks=3,
        )

    assert any("cornerstone.test.ingestion.documents" in record.message for record in caplog.records)
    assert any("duration_ms" in record.message for record in caplog.records)


def test_metrics_recorder_disabled_suppresses_logs(caplog) -> None:
    metrics = MetricsRecorder(enabled=False)

    with caplog.at_level(logging.INFO, logger="cornerstone.metrics"):
        metrics.increment("ingestion.documents", project_id="proj-2")
        metrics.record_timing("ingestion.total_duration", 0.1, project_id="proj-2")

    assert not caplog.records
