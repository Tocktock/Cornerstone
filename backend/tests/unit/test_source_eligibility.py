from __future__ import annotations

from cornerstone.schemas import DataSource, DataSourceStatus, DataSourceType, utc_now
from cornerstone.services.source_eligibility import (
    source_can_officialize,
    source_can_serve_captured_evidence,
    source_serving_limitations,
)


def _source(status: DataSourceStatus, *, production_enabled: bool = True) -> DataSource:
    return DataSource(
        type=DataSourceType.MANUAL,
        name="Manual Pilot",
        status=status,
        production_enabled=production_enabled,
    )


def test_officialization_requires_connected_source_in_production() -> None:
    connected = _source(DataSourceStatus.CONNECTED)
    degraded = _source(DataSourceStatus.DEGRADED)

    assert source_can_officialize(connected, production_mode=True) is True
    assert source_can_officialize(degraded, production_mode=True) is False


def test_serving_allows_degraded_historical_source_with_limitations() -> None:
    degraded = _source(DataSourceStatus.DEGRADED).model_copy(
        update={"last_successful_sync_at": utc_now()}, deep=True
    )

    assert source_can_serve_captured_evidence(degraded, production_mode=True) is True
    assert any("degraded" in limitation for limitation in source_serving_limitations(degraded))


def test_serving_blocks_never_synced_degraded_source() -> None:
    degraded = _source(DataSourceStatus.DEGRADED)

    assert source_can_serve_captured_evidence(degraded, production_mode=True) is False
