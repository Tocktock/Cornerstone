from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    CreateDataSourceRequest,
    DataSource,
    DataSourceAuthStatus,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    FreshnessState,
    SourceStudioResponse,
)
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/sources", tags=["sources"])

_REAL_CONNECTED_STATUSES = {
    DataSourceStatus.CONNECTED,
    DataSourceStatus.SYNC_PENDING,
    DataSourceStatus.SYNCING,
    DataSourceStatus.DEGRADED,
    DataSourceStatus.STALE,
}


@router.get("", response_model=SourceStudioResponse)
def list_sources(
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SourceStudioResponse:
    sources = store.list_data_sources()
    has_real_sources = any(_is_real_connected_source(source) for source in sources)
    onboarding_required = settings.production_mode and not has_real_sources
    if onboarding_required:
        message = (
            "No production data sources are connected yet. Connect a real source to create "
            "artifacts and evidence."
        )
    else:
        message = "Source Studio is ready."
    log_event(
        "source.listed",
        productionEnabled=settings.production_mode,
        sourceCount=len(sources),
        hasRealSources=has_real_sources,
        onboardingRequired=onboarding_required,
    )
    return SourceStudioResponse(
        production_enabled=settings.production_mode,
        has_real_sources=has_real_sources,
        onboarding_required=onboarding_required,
        message=message,
        sources=sources,
    )


@router.post("", response_model=DataSource, status_code=status.HTTP_201_CREATED)
def create_source(
    request: CreateDataSourceRequest,
    store: InMemoryStore = Depends(get_store),
) -> DataSource:
    if request.type != DataSourceType.MANUAL:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Provider-backed sources must be created through /v1/connections/intents "
                "and /v1/oauth/{provider}/callback."
            ),
        )
    log_event(
        "source.create_requested",
        sourceType=request.type,
        sourceName=request.name,
        productionEnabled=request.production_enabled,
    )
    data_source = DataSource(
        type=DataSourceType.MANUAL,
        name=request.name,
        status=DataSourceStatus.CONNECTED,
        production_enabled=request.production_enabled,
        auth_status=DataSourceAuthStatus.AUTHORIZED,
        connection_status=DataSourceConnectionStatus.TEST_PASSED,
        sync_status=DataSourceSyncStatus.NEVER_SYNCED,
        next_action=DataSourceNextAction.NONE,
        freshness_state=FreshnessState.UNKNOWN,
        sync_freshness_state=FreshnessState.UNKNOWN,
        content_freshness_state=FreshnessState.UNKNOWN,
    )
    saved = store.add_data_source(data_source)
    log_event(
        "source.created",
        sourceId=saved.id,
        sourceType=saved.type,
        sourceName=saved.name,
        status=saved.status,
        productionEnabled=saved.production_enabled,
    )
    return saved


def _get_source_or_404(source_id: str, store: InMemoryStore) -> DataSource:
    try:
        return store.get_data_source(source_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _is_real_connected_source(source: DataSource) -> bool:
    return bool(
        source.production_enabled
        and source.auth_status == DataSourceAuthStatus.AUTHORIZED
        and source.connection_status == DataSourceConnectionStatus.TEST_PASSED
        and source.status in _REAL_CONNECTED_STATUSES
    )
