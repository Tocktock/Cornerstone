from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import DataSourceType, SyncSourceRequest, SyncSourceResponse
from cornerstone.services.source_sync import SourceNotSyncableError, sync_source_objects
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/manual-sources", tags=["manual sources"])


@router.post("/{source_id}/sync", response_model=SyncSourceResponse)
def sync_manual_source(
    source_id: str,
    request: SyncSourceRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncSourceResponse:
    """Sync explicitly provided objects into a manual source only.

    Provider-backed sources such as Notion must use their connector sync-job paths so they
    cannot bypass credential, discovery, selection, and provider provenance checks.
    """

    try:
        data_source = store.get_data_source(source_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if data_source.type != DataSourceType.MANUAL:
        log_event(
            "manual_source.sync_rejected",
            sourceId=data_source.id,
            sourceType=data_source.type,
            reason="not_manual_source",
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Manual sync is only available for manual sources. "
                "Provider-backed sources must use connector discovery, selection, and sync jobs."
            ),
        )

    try:
        return sync_source_objects(
            data_source=data_source,
            objects=request.objects,
            store=store,
            settings=settings,
        )
    except SourceNotSyncableError as exc:
        log_event(
            "manual_source.sync_blocked",
            sourceId=data_source.id,
            sourceType=data_source.type,
            status=data_source.status,
            reason="source_status_not_syncable",
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
