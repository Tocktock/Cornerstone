from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cornerstone.api.dependencies import get_store
from cornerstone.api.routes.connectors import (
    _CONNECTOR_SYNCABLE_STATUSES,
    _ensure_runtime_provider,
    _get_active_credential_or_409,
    _get_selected_provider_snapshots_for_sync,
    _get_source_or_404,
    _get_sync_job_or_404,
)
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorNextAction,
    CreateSyncJobRequest,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    HeartbeatSyncJobLeaseRequest,
    RunSyncSchedulerRequest,
    RunSyncWorkerRequest,
    SourceSelection,
    SourceSelectionMode,
    SyncCursor,
    SyncJob,
    SyncJobDetail,
    SyncJobEvent,
    SyncJobStatus,
    SyncJobTrigger,
    SyncSchedule,
    SyncSchedulerRunResult,
    SyncWorkerRunResult,
    UpsertSyncScheduleRequest,
)
from cornerstone.services.source_selection import get_source_selection_for_sync
from cornerstone.services.sync_jobs import add_sync_job_event as _add_sync_job_event
from cornerstone.services.sync_scheduler import build_sync_schedule, run_sync_scheduler
from cornerstone.services.sync_worker import (
    SyncJobNotReadyError,
    SyncJobNotRunnableError,
    heartbeat_sync_job_lease,
    request_sync_job_cancel,
    run_sync_job_once,
    run_sync_worker,
)
from cornerstone.store import NotFoundError

router = APIRouter(tags=["sync-runtime"])


@router.get("/sources/{source_id}/sync-schedule", response_model=SyncSchedule)
def get_source_sync_schedule(source_id: str, store: Any = Depends(get_store)) -> SyncSchedule:
    _get_source_or_404(source_id, store)
    try:
        return cast(SyncSchedule, store.get_sync_schedule(source_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put("/sources/{source_id}/sync-schedule", response_model=SyncSchedule)
def upsert_source_sync_schedule(
    source_id: str,
    request: UpsertSyncScheduleRequest,
    store: Any = Depends(get_store),
) -> SyncSchedule:
    data_source = _get_source_or_404(source_id, store)
    existing: SyncSchedule | None
    try:
        existing = cast(SyncSchedule, store.get_sync_schedule(source_id))
    except NotFoundError:
        existing = None

    if request.enabled:
        try:
            selection = cast(SourceSelection, store.get_source_selection(source_id))
        except NotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Save a source selection before enabling scheduled sync.",
            ) from exc
        if not selection.selected_external_object_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Select at least one source object before enabling scheduled sync.",
            )
        if not _get_selected_provider_snapshots_for_sync(source_id, store):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Select at least one ingestion-supported source object before enabling scheduled sync.",
            )

    schedule = build_sync_schedule(
        data_source=data_source,
        enabled=request.enabled,
        interval_minutes=request.interval_minutes,
        start_at=request.start_at,
        max_attempts=request.max_attempts,
        created_by=request.created_by,
        existing=existing,
    )
    saved = cast(SyncSchedule, store.upsert_sync_schedule(schedule))
    log_event(
        "sync.schedule_upserted",
        syncScheduleId=saved.id,
        sourceId=source_id,
        provider=saved.provider,
        status=saved.status,
        intervalMinutes=saved.interval_minutes,
        nextRunAt=saved.next_run_at,
    )
    return saved


@router.post("/sync-scheduler/run", response_model=SyncSchedulerRunResult)
def run_sync_scheduler_endpoint(
    request: RunSyncSchedulerRequest | None = None,
    store: Any = Depends(get_store),
) -> SyncSchedulerRunResult:
    options = request or RunSyncSchedulerRequest()
    return run_sync_scheduler(
        store=store,
        max_schedules=options.max_schedules,
        include_not_due=options.include_not_due,
    )


@router.post(
    "/sources/{source_id}/sync-jobs",
    response_model=SyncJobDetail,
    status_code=status.HTTP_201_CREATED,
)
async def create_source_sync_job(
    source_id: str,
    request: CreateSyncJobRequest,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncJobDetail:
    data_source = _get_source_or_404(source_id, store)
    _ensure_runtime_provider(data_source.type)
    if data_source.status not in _CONNECTOR_SYNCABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"DataSource cannot start connector sync while status is '{data_source.status}'.",
        )
    _get_active_credential_or_409(source_id, store)
    selection = get_source_selection_for_sync(source_id, store)
    if selection.sync_mode == SourceSelectionMode.SELECTED_ONLY and not selection.selected_external_object_ids:
        connector_error = ConnectorError(
            code=ConnectorErrorCode.SOURCE_SELECTION_REQUIRED,
            user_message="Select at least one source object before starting the first sync.",
            retryable=False,
            next_action=ConnectorNextAction.SELECT_SOURCES,
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=connector_error.model_dump(mode="json", by_alias=True))
    selected_syncable_objects = _get_selected_provider_snapshots_for_sync(source_id, store)
    if not selected_syncable_objects:
        connector_error = ConnectorError(
            code=ConnectorErrorCode.UNSUPPORTED_OBJECT_TYPE,
            user_message="The saved source selection contains no ingestion-supported objects. Update the source selection before syncing.",
            retryable=False,
            next_action=ConnectorNextAction.SELECT_SOURCES,
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=connector_error.model_dump(mode="json", by_alias=True))

    job = SyncJob(
        datasource_id=source_id,
        provider=data_source.type,
        status=SyncJobStatus.QUEUED,
        trigger=request.trigger,
        created_by=request.created_by,
        selection_id=selection.id,
        max_attempts=request.max_attempts,
        queue_ontology_reextraction=request.queue_ontology_reextraction,
        run_ontology_reextraction_inline=request.run_ontology_reextraction_inline,
        ontology_focus_concept=request.ontology_focus_concept,
    )
    saved_job = store.add_sync_job(job)
    _add_sync_job_event(
        store,
        saved_job,
        "sync.job_queued",
        "Sync job was queued.",
        {"trigger": str(saved_job.trigger)},
    )
    log_event("sync.job_queued", syncJobId=saved_job.id, sourceId=source_id, provider=data_source.type)
    store.update_data_source(
        data_source.model_copy(
            update={
                "status": DataSourceStatus.SYNC_PENDING,
                "sync_status": DataSourceSyncStatus.QUEUED,
                "next_action": DataSourceNextAction.NONE,
                "last_error": None,
            },
            deep=True,
        )
    )

    if request.run_inline:
        return await run_sync_job_once(job_id=saved_job.id, store=store, settings=settings, include_not_ready=True)
    return SyncJobDetail(job=saved_job, events=store.list_sync_job_events(saved_job.id))


@router.get("/sources/{source_id}/sync-jobs", response_model=list[SyncJob])
def list_source_sync_jobs(source_id: str, store: Any = Depends(get_store)) -> list[SyncJob]:
    _get_source_or_404(source_id, store)
    return cast(list[SyncJob], store.list_sync_jobs(datasource_id=source_id))


@router.get("/sync-jobs/{sync_job_id}", response_model=SyncJobDetail)
def get_sync_job(sync_job_id: str, store: Any = Depends(get_store)) -> SyncJobDetail:
    job = _get_sync_job_or_404(sync_job_id, store)
    return SyncJobDetail(job=job, events=store.list_sync_job_events(sync_job_id))


@router.get("/sync-jobs/{sync_job_id}/events", response_model=list[SyncJobEvent])
def list_sync_job_events(sync_job_id: str, store: Any = Depends(get_store)) -> list[SyncJobEvent]:
    _get_sync_job_or_404(sync_job_id, store)
    return cast(list[SyncJobEvent], store.list_sync_job_events(sync_job_id))


@router.post("/sync-jobs/{sync_job_id}/run", response_model=SyncJobDetail)
async def run_sync_job(
    sync_job_id: str,
    include_not_ready: bool = Query(default=False, alias="includeNotReady"),
    worker_id: str = Query(default="api-job-runner", alias="workerId"),
    lease_seconds: int = Query(default=300, ge=30, le=3600, alias="leaseSeconds"),
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncJobDetail:
    _get_sync_job_or_404(sync_job_id, store)
    try:
        return await run_sync_job_once(
            job_id=sync_job_id,
            store=store,
            settings=settings,
            include_not_ready=include_not_ready,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
    except SyncJobNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except SyncJobNotRunnableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/sync-jobs/{sync_job_id}/heartbeat", response_model=SyncJobDetail)
def heartbeat_sync_job(
    sync_job_id: str,
    request: HeartbeatSyncJobLeaseRequest,
    store: Any = Depends(get_store),
) -> SyncJobDetail:
    _get_sync_job_or_404(sync_job_id, store)
    try:
        saved = heartbeat_sync_job_lease(
            store=store,
            job_id=sync_job_id,
            worker_id=request.worker_id,
            lease_seconds=request.lease_seconds,
        )
    except SyncJobNotRunnableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return SyncJobDetail(job=saved, events=store.list_sync_job_events(saved.id))


@router.post("/sync-worker/run", response_model=SyncWorkerRunResult)
async def run_sync_worker_endpoint(
    request: RunSyncWorkerRequest | None = None,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncWorkerRunResult:
    options = request or RunSyncWorkerRequest()
    return await run_sync_worker(
        store=store,
        settings=settings,
        max_jobs=options.max_jobs,
        include_not_ready=options.include_not_ready,
        job_id=options.job_id,
        worker_id=options.worker_id,
        lease_seconds=options.lease_seconds,
    )


@router.get("/sources/{source_id}/sync-cursor", response_model=SyncCursor)
def get_source_sync_cursor(source_id: str, store: Any = Depends(get_store)) -> SyncCursor:
    _get_source_or_404(source_id, store)
    try:
        return cast(SyncCursor, store.get_sync_cursor(source_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/sync-jobs/{sync_job_id}/cancel", response_model=SyncJobDetail)
def cancel_sync_job(sync_job_id: str, store: Any = Depends(get_store)) -> SyncJobDetail:
    _get_sync_job_or_404(sync_job_id, store)
    try:
        saved = request_sync_job_cancel(store=store, job_id=sync_job_id, cancelled_by="system")
    except SyncJobNotRunnableError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return SyncJobDetail(job=saved, events=store.list_sync_job_events(saved.id))


@router.post("/sync-jobs/{sync_job_id}/retry", response_model=SyncJobDetail, status_code=status.HTTP_201_CREATED)
async def retry_sync_job(
    sync_job_id: str,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncJobDetail:
    prior = _get_sync_job_or_404(sync_job_id, store)
    return await create_source_sync_job(
        prior.datasource_id,
        CreateSyncJobRequest(
            trigger=SyncJobTrigger.RETRY,
            created_by=prior.created_by,
            run_inline=False,
            max_attempts=prior.max_attempts,
            queue_ontology_reextraction=prior.queue_ontology_reextraction,
            run_ontology_reextraction_inline=prior.run_ontology_reextraction_inline,
            ontology_focus_concept=prior.ontology_focus_concept,
        ),
        store,
        settings,
    )
