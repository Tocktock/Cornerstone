from __future__ import annotations

from cornerstone.schemas import DataSource, DataSourceStatus

_OFFICIALIZATION_SOURCE_STATUSES = {DataSourceStatus.CONNECTED}
_SERVABLE_CAPTURED_SOURCE_STATUSES = {
    DataSourceStatus.CONNECTED,
    DataSourceStatus.SYNC_PENDING,
    DataSourceStatus.SYNCING,
    DataSourceStatus.DEGRADED,
    DataSourceStatus.FAILED,
    DataSourceStatus.STALE,
    DataSourceStatus.DISCONNECTED,
}


def source_can_officialize(data_source: DataSource, *, production_mode: bool) -> bool:
    """Return whether this source can support new officialization decisions.

    Officialization is intentionally strict: the source must be production-enabled
    in production mode and currently connected/healthy.
    """
    if production_mode and not data_source.production_enabled:
        return False
    return data_source.status in _OFFICIALIZATION_SOURCE_STATUSES


def source_can_serve_captured_evidence(data_source: DataSource, *, production_mode: bool) -> bool:
    """Return whether previously captured evidence can be served with limitations.

    Serving is intentionally less strict than officialization. Reviewed evidence
    from a source that has since become degraded, failed, stale, or disconnected
    may still be useful if provenance is valid and the source previously synced
    successfully. The response must carry downgraded trust/freshness limitations.
    """
    if production_mode and not data_source.production_enabled:
        return False
    if data_source.status not in _SERVABLE_CAPTURED_SOURCE_STATUSES:
        return False
    if data_source.status == DataSourceStatus.CONNECTED:
        return True
    return bool(
        data_source.last_successful_sync_at is not None
        or data_source.artifact_count > 0
        or data_source.evidence_fragment_count > 0
    )


def source_serving_limitations(data_source: DataSource) -> list[str]:
    limitations: list[str] = []
    if data_source.status in {DataSourceStatus.DEGRADED, DataSourceStatus.FAILED}:
        limitations.append(
            f"Source '{data_source.name}' is currently {data_source.status}; previously captured evidence is shown with degraded-source limitations."
        )
    elif data_source.status == DataSourceStatus.STALE:
        limitations.append(
            f"Source '{data_source.name}' is stale; verify freshness before relying on this answer."
        )
    elif data_source.status == DataSourceStatus.DISCONNECTED:
        limitations.append(
            f"Source '{data_source.name}' is disconnected; evidence is historical and cannot be refreshed until reconnected."
        )
    if data_source.last_error is not None:
        limitations.append(f"Source '{data_source.name}' has a latest sync or connection error.")
    return limitations
