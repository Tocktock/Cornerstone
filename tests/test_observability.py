import io
import logging

from cornerstone.observability import MetricsRecorder


def _capture_logger_output(logger_name: str):
    logger = logging.getLogger(logger_name)
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger, handler, buffer


def test_metrics_recorder_logs_when_enabled() -> None:
    metrics = MetricsRecorder(enabled=True, namespace="cornerstone.test")
    logger, handler, buffer = _capture_logger_output("cornerstone.metrics")

    try:
        metrics.increment("ingestion.documents", project_id="proj-1")
        metrics.record_timing(
            "ingestion.total_duration",
            0.05,
            project_id="proj-1",
            chunks=3,
        )
    finally:
        logger.removeHandler(handler)

    output = buffer.getvalue()
    assert "cornerstone.test.ingestion.documents" in output
    assert "cornerstone.test.ingestion.total_duration" in output


def test_metrics_recorder_disabled_suppresses_logs(caplog) -> None:
    metrics = MetricsRecorder(enabled=False)

    with caplog.at_level(logging.INFO, logger="cornerstone.metrics"):
        metrics.increment("ingestion.documents", project_id="proj-2")
        metrics.record_timing("ingestion.total_duration", 0.1, project_id="proj-2")

    assert not caplog.records
