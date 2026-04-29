from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, cast

from cornerstone.config import Settings
from cornerstone.connectors.registry import get_connector, get_token_cipher
from cornerstone.observability import log_event
from cornerstone.schemas import (
    ConnectorCredential,
    ConnectorError,
    ConnectorErrorCode,
    ConnectorNextAction,
    DataSource,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    ErrorInfo,
    FreshnessState,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    SourceSelection,
    SyncCursor,
    SyncJob,
    SyncJobDetail,
    SyncJobEvent,
    SyncJobStatus,
    SyncWorkerRunResult,
    utc_now,
)
from cornerstone.services.source_sync import sync_source_objects
from cornerstone.store import NotFoundError


class SyncJobNotRunnableError(RuntimeError):
    """Raised when a sync job is not in a worker-runnable state."""


class SyncJobNotReadyError(RuntimeError):
    """Raised when a retry-waiting job is not due yet."""


class SyncJobCancellationRequested(RuntimeError):
    """Raised inside a worker attempt after cancellation was requested."""


class SyncSelectionError(RuntimeError):
    """Raised when a durable job references a stale or unsupported source selection."""

    def __init__(self, connector_error: ConnectorError) -> None:
        super().__init__(connector_error.user_message)
        self.connector_error = connector_error


class SyncCredentialError(RuntimeError):
    """Raised when a durable job can no longer access an active source credential."""

    def __init__(self, connector_error: ConnectorError) -> None:
        super().__init__(connector_error.user_message)
        self.connector_error = connector_error


RUNNABLE_STATUSES = {SyncJobStatus.QUEUED, SyncJobStatus.RETRY_WAITING}


async def run_sync_worker(
    *,
    store: Any,
    settings: Settings,
    max_jobs: int = 1,
    include_not_ready: bool = False,
    job_id: str | None = None,
    worker_id: str = "sync-worker",
    lease_seconds: int = 300,
) -> SyncWorkerRunResult:
    """Run eligible durable sync jobs.

    This is intentionally explicit instead of a hidden FastAPI background task. Tests, local
    development, and future production workers can all call the same runtime while PostgreSQL
    remains the durable source of job state.
    """

    processed: list[SyncJobDetail] = []
    skipped = 0

    if job_id is not None:
        try:
            processed.append(
                await run_sync_job_once(
                    job_id=job_id,
                    store=store,
                    settings=settings,
                    include_not_ready=include_not_ready,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                )
            )
        except (SyncJobNotReadyError, SyncJobNotRunnableError):
            skipped += 1
        return SyncWorkerRunResult(processed_job_count=len(processed), skipped_job_count=skipped, jobs=processed)

    for job in _eligible_jobs(store, include_not_ready=include_not_ready):
        if len(processed) >= max_jobs:
            break
        try:
            processed.append(
                await run_sync_job_once(
                    job_id=job.id,
                    store=store,
                    settings=settings,
                    include_not_ready=include_not_ready,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                )
            )
        except (SyncJobNotReadyError, SyncJobNotRunnableError):
            skipped += 1

    return SyncWorkerRunResult(processed_job_count=len(processed), skipped_job_count=skipped, jobs=processed)


async def run_sync_job_once(
    *,
    job_id: str,
    store: Any,
    settings: Settings,
    include_not_ready: bool = False,
    worker_id: str = "sync-worker",
    lease_seconds: int = 300,
) -> SyncJobDetail:
    claimed = _claim_job_for_worker(
        store=store,
        job_id=job_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        include_not_ready=include_not_ready,
    )
    if claimed.cancel_requested_at is not None:
        cancelled = _mark_job_cancelled(store, claimed, cancelled_by=claimed.cancelled_by or worker_id)
        return SyncJobDetail(job=cancelled, events=store.list_sync_job_events(cancelled.id))

    try:
        data_source = cast(DataSource, store.get_data_source(claimed.datasource_id))
    except NotFoundError as exc:
        failed = _mark_job_failed_without_source(store=store, job=claimed, exc=exc)
        return SyncJobDetail(job=failed, events=store.list_sync_job_events(failed.id))

    running = claimed
    try:
        credential = _get_active_credential_for_sync(claimed.datasource_id, store)
        selection = _get_source_selection_for_sync(claimed.datasource_id, store)
        running = _record_claimed_job_started(store, claimed, data_source)
        _ensure_not_cancel_requested(store, running.id)
        selected_objects = _get_selected_provider_snapshots_for_sync(data_source.id, store)
        if not selected_objects:
            raise SyncSelectionError(
                ConnectorError(
                    code=ConnectorErrorCode.UNSUPPORTED_OBJECT_TYPE,
                    user_message="The saved source selection contains no ingestion-supported provider objects.",
                    retryable=False,
                    next_action=ConnectorNextAction.SELECT_SOURCES,
                )
            )
        _add_sync_job_event(
            store,
            running,
            "sync.objects_selected",
            "Selected provider objects were resolved for ingestion.",
            {"selectedObjectCount": len(selected_objects)},
        )

        connector = get_connector(data_source.type, settings)
        access_token = get_token_cipher(settings).decrypt(credential.encrypted_access_token)
        _ensure_not_cancel_requested(store, running.id)
        objects = await connector.list_objects(
            credential=credential,
            access_token=access_token,
            selection=selection,
            selected_objects=selected_objects,
        )
        _ensure_not_cancel_requested(store, running.id)
        _add_sync_job_event(
            store,
            running,
            "sync.objects_normalized",
            "Provider objects were normalized for Artifact ingestion.",
            {"normalizedObjectCount": len(objects)},
        )

        saved, _cursor = _commit_sync_success_atomically(
            store=store,
            settings=settings,
            running=running,
            data_source_id=data_source.id,
            objects=objects,
            selected_objects=selected_objects,
        )
        log_event(
            "sync.job_succeeded",
            syncJobId=saved.id,
            sourceId=data_source.id,
            provider=data_source.type,
            attemptCount=saved.attempt_count,
            artifactCreatedCount=saved.artifact_created_count,
            evidenceCreatedCount=saved.evidence_created_count,
            cursor=saved.cursor,
        )
        return SyncJobDetail(job=saved, events=store.list_sync_job_events(saved.id))
    except SyncJobCancellationRequested:
        cancelled = _mark_job_cancelled(store, cast(SyncJob, store.get_sync_job(running.id)), cancelled_by="system")
        return SyncJobDetail(job=cancelled, events=store.list_sync_job_events(cancelled.id))
    except Exception as exc:
        saved = _handle_job_failure(
            store=store,
            job=cast(SyncJob, store.get_sync_job(running.id)),
            data_source=cast(DataSource, store.get_data_source(data_source.id)),
            exc=exc,
        )
        return SyncJobDetail(job=saved, events=store.list_sync_job_events(saved.id))



def _commit_sync_success_atomically(
    *,
    store: Any,
    settings: Settings,
    running: SyncJob,
    data_source_id: str,
    objects: list[Any],
    selected_objects: list[ProviderObjectSnapshot],
) -> tuple[SyncJob, SyncCursor]:
    """Commit Artifact/Evidence/source/cursor/job success writes as one unit.

    Provider calls happen before this function. Once normalized objects are available, the
    product state writes that make a sync successful must be atomic: captured Artifacts,
    EvidenceFragments, source freshness/counters, SyncCursor advancement, SyncJob success,
    and terminal SyncJobEvents should either all commit or all roll back. This prevents a
    retry from seeing durable evidence without the matching cursor/job audit trail.
    """

    with store.transaction():
        response = sync_source_objects(
            data_source=cast(DataSource, store.get_data_source(data_source_id)),
            objects=objects,
            store=store,
            settings=settings,
            emit_logs=False,
        )
        cursor = _advance_sync_cursor(
            store=store,
            job=running,
            data_source=cast(DataSource, store.get_data_source(data_source_id)),
            selected_objects=selected_objects,
            artifact_created_count=response.artifact_created_count,
            artifact_reused_count=response.artifact_reused_count,
            evidence_created_count=response.evidence_created_count,
        )
        finished_at = utc_now()
        succeeded = running.model_copy(
            update={
                "status": SyncJobStatus.SUCCEEDED,
                "finished_at": finished_at,
                "artifact_created_count": response.artifact_created_count,
                "artifact_reused_count": response.artifact_reused_count,
                "evidence_created_count": response.evidence_created_count,
                "error": None,
                "cursor": cursor.last_cursor,
                "next_attempt_at": None,
                "lease_owner": None,
                "lease_acquired_at": None,
                "lease_expires_at": None,
                "lease_heartbeat_at": None,
            },
            deep=True,
        )
        saved = cast(SyncJob, store.update_sync_job(succeeded))
        final_source = cast(DataSource, store.get_data_source(data_source_id))
        store.update_data_source(
            final_source.model_copy(
                update={
                    "sync_status": DataSourceSyncStatus.SUCCEEDED,
                    "next_action": (
                        DataSourceNextAction.REVIEW_EVIDENCE
                        if saved.evidence_created_count > 0
                        else DataSourceNextAction.NONE
                    ),
                    "last_error": None,
                },
                deep=True,
            )
        )
        _add_sync_job_event(
            store,
            saved,
            "sync.cursor_advanced",
            "Sync cursor advanced after successful object processing.",
            {
                "cursorKey": cursor.cursor_key,
                "lastCursor": cursor.last_cursor,
                "processedObjectCount": len(cursor.processed_external_object_ids),
            },
        )
        _add_sync_job_event(
            store,
            saved,
            "sync.job_succeeded",
            "Sync job succeeded.",
            {
                "artifactCreatedCount": saved.artifact_created_count,
                "artifactReusedCount": saved.artifact_reused_count,
                "evidenceCreatedCount": saved.evidence_created_count,
            },
        )
        return saved, cursor

def request_sync_job_cancel(*, store: Any, job_id: str, cancelled_by: str = "system") -> SyncJob:
    job = cast(SyncJob, store.get_sync_job(job_id))
    if job.status in {SyncJobStatus.SUCCEEDED, SyncJobStatus.FAILED, SyncJobStatus.CANCELLED}:
        raise SyncJobNotRunnableError("Completed sync jobs cannot be cancelled.")
    if job.status in {SyncJobStatus.QUEUED, SyncJobStatus.RETRY_WAITING}:
        return _mark_job_cancelled(store, job, cancelled_by=cancelled_by)
    now = utc_now()
    requested = job.model_copy(
        update={"cancel_requested_at": now, "cancelled_by": cancelled_by},
        deep=True,
    )
    saved = cast(SyncJob, store.update_sync_job(requested))
    _add_sync_job_event(
        store,
        saved,
        "sync.cancel_requested",
        "Cancellation was requested for the running sync job.",
        {"cancelledBy": cancelled_by},
    )
    log_event("sync.cancel_requested", syncJobId=saved.id, sourceId=saved.datasource_id, cancelledBy=cancelled_by)
    return saved


def _eligible_jobs(store: Any, *, include_not_ready: bool) -> list[SyncJob]:
    now = utc_now()
    jobs = [
        job
        for job in cast(list[SyncJob], store.list_sync_jobs())
        if _is_worker_eligible(job, now=now, include_not_ready=include_not_ready)
    ]
    return sorted(
        jobs,
        key=lambda item: (
            item.next_attempt_at or item.lease_expires_at or item.created_at,
            item.created_at,
            item.id,
        ),
    )


def _is_worker_eligible(job: SyncJob, *, now: datetime, include_not_ready: bool) -> bool:
    if job.status == SyncJobStatus.QUEUED:
        return True
    if job.status == SyncJobStatus.RETRY_WAITING:
        if job.next_attempt_at is None:
            return True
        return include_not_ready or _aware(job.next_attempt_at) <= now
    if job.status == SyncJobStatus.RUNNING:
        return job.lease_expires_at is not None and _aware(job.lease_expires_at) <= now
    return False


def _claim_job_for_worker(
    *,
    store: Any,
    job_id: str,
    worker_id: str,
    lease_seconds: int,
    include_not_ready: bool,
) -> SyncJob:
    original = cast(SyncJob, store.get_sync_job(job_id))
    now = utc_now()
    was_expired_running = (
        original.status == SyncJobStatus.RUNNING
        and original.lease_expires_at is not None
        and _aware(original.lease_expires_at) <= now
    )
    if original.status == SyncJobStatus.RETRY_WAITING and original.next_attempt_at is not None:
        next_attempt_at = _aware(original.next_attempt_at)
        if next_attempt_at > now and not include_not_ready:
            raise SyncJobNotReadyError(f"SyncJob is not due until {original.next_attempt_at.isoformat()}.")
    if original.status == SyncJobStatus.RUNNING and not was_expired_running:
        raise SyncJobNotRunnableError("SyncJob is already running with an active worker lease.")

    claimed = cast(
        SyncJob | None,
        store.claim_sync_job(
            job_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            include_not_ready=include_not_ready,
        ),
    )
    if claimed is None:
        current = cast(SyncJob, store.get_sync_job(job_id))
        raise SyncJobNotRunnableError(f"SyncJob could not be claimed while status is '{current.status}'.")
    log_event(
        "sync.job_claimed",
        syncJobId=claimed.id,
        sourceId=claimed.datasource_id,
        provider=claimed.provider,
        workerId=claimed.lease_owner,
        leaseExpiresAt=claimed.lease_expires_at,
        recoveredExpiredLease=was_expired_running,
    )
    _add_sync_job_event(
        store,
        claimed,
        "sync.job_claimed",
        "Sync worker claimed the job lease.",
        {
            "workerId": claimed.lease_owner or worker_id,
            "leaseExpiresAt": None if claimed.lease_expires_at is None else claimed.lease_expires_at.isoformat(),
            "leaseHeartbeatAt": None if claimed.lease_heartbeat_at is None else claimed.lease_heartbeat_at.isoformat(),
            "recoveredExpiredLease": was_expired_running,
        },
    )
    return claimed


def _record_claimed_job_started(store: Any, job: SyncJob, data_source: DataSource) -> SyncJob:
    now = job.started_at or utc_now()
    store.update_data_source(
        data_source.model_copy(
            update={
                "status": DataSourceStatus.SYNCING,
                "sync_status": DataSourceSyncStatus.SYNCING,
                "last_sync_at": now,
                "next_action": DataSourceNextAction.NONE,
                "last_error": None,
            },
            deep=True,
        )
    )
    _add_sync_job_event(
        store,
        job,
        "sync.job_started",
        "Sync worker started the job.",
        {"attemptCount": job.attempt_count, "maxAttempts": job.max_attempts},
    )
    log_event(
        "sync.job_started",
        syncJobId=job.id,
        sourceId=job.datasource_id,
        provider=job.provider,
        attemptCount=job.attempt_count,
        workerId=job.lease_owner,
    )
    return job


def _handle_job_failure(*, store: Any, job: SyncJob, data_source: DataSource, exc: Exception) -> SyncJob:
    now = utc_now()
    connector_error = _connector_error_from_exception(exc)
    retryable = connector_error.retryable and job.attempt_count < job.max_attempts
    if retryable:
        delay_seconds = _retry_delay_seconds(connector_error=connector_error, attempt_count=job.attempt_count)
        next_attempt_at = now + timedelta(seconds=delay_seconds)
        retry_waiting = job.model_copy(
            update={
                "status": SyncJobStatus.RETRY_WAITING,
                "error": connector_error,
                "next_attempt_at": next_attempt_at,
                "finished_at": None,
                "lease_owner": None,
                "lease_acquired_at": None,
                "lease_expires_at": None,
                "lease_heartbeat_at": None,
            },
            deep=True,
        )
        saved = cast(SyncJob, store.update_sync_job(retry_waiting))
        store.update_data_source(
            data_source.model_copy(
                update={
                    "status": DataSourceStatus.SYNC_PENDING,
                    "sync_status": DataSourceSyncStatus.WAITING_RETRY,
                    "next_action": DataSourceNextAction.RETRY_SYNC,
                    "last_error": ErrorInfo(code=str(connector_error.code), message=connector_error.user_message),
                    "sync_freshness_state": FreshnessState.UNKNOWN,
                },
                deep=True,
            )
        )
        _add_sync_job_event(
            store,
            saved,
            "sync.retry_scheduled",
            "Sync job retry was scheduled.",
            {
                "errorCode": str(connector_error.code),
                "attemptCount": saved.attempt_count,
                "maxAttempts": saved.max_attempts,
                "nextAttemptAt": next_attempt_at.isoformat(),
            },
        )
        log_event(
            "sync.retry_scheduled",
            syncJobId=saved.id,
            sourceId=saved.datasource_id,
            errorCode=connector_error.code,
            nextAttemptAt=next_attempt_at,
        )
        return saved

    failed_status = _failed_source_status(data_source)
    failed = job.model_copy(
        update={
            "status": SyncJobStatus.FAILED,
            "finished_at": now,
            "error": connector_error,
            "lease_owner": None,
            "lease_acquired_at": None,
            "lease_expires_at": None,
            "lease_heartbeat_at": None,
        },
        deep=True,
    )
    saved_failed = cast(SyncJob, store.update_sync_job(failed))
    store.update_data_source(
        data_source.model_copy(
            update={
                "status": failed_status,
                "sync_status": (
                    DataSourceSyncStatus.DEGRADED
                    if data_source.last_successful_sync_at is not None
                    else DataSourceSyncStatus.FAILED
                ),
                "next_action": _data_source_next_action_from_connector_action(connector_error.next_action),
                "last_error": ErrorInfo(code=str(connector_error.code), message=connector_error.user_message),
                "sync_freshness_state": FreshnessState.UNKNOWN,
            },
            deep=True,
        )
    )
    _add_sync_job_event(
        store,
        saved_failed,
        "sync.job_failed",
        "Sync job failed.",
        {
            "errorCode": str(connector_error.code),
            "errorType": type(exc).__name__,
            "attemptCount": saved_failed.attempt_count,
        },
    )
    log_event(
        "sync.job_failed",
        syncJobId=saved_failed.id,
        sourceId=saved_failed.datasource_id,
        provider=saved_failed.provider,
        errorType=type(exc).__name__,
        attemptCount=saved_failed.attempt_count,
    )
    return saved_failed


def _mark_job_cancelled(store: Any, job: SyncJob, *, cancelled_by: str) -> SyncJob:
    now = utc_now()
    cancelled = job.model_copy(
        update={
            "status": SyncJobStatus.CANCELLED,
            "finished_at": now,
            "cancel_requested_at": job.cancel_requested_at or now,
            "cancelled_by": cancelled_by,
            "lease_owner": None,
            "lease_acquired_at": None,
            "lease_expires_at": None,
            "lease_heartbeat_at": None,
        },
        deep=True,
    )
    saved = cast(SyncJob, store.update_sync_job(cancelled))
    try:
        data_source = cast(DataSource, store.get_data_source(saved.datasource_id))
    except NotFoundError:
        data_source = None
    if data_source is not None:
        store.update_data_source(
            data_source.model_copy(
                update={
                    "status": (
                        DataSourceStatus.CONNECTED
                        if data_source.last_successful_sync_at is None
                        else DataSourceStatus.DEGRADED
                    ),
                    "sync_status": DataSourceSyncStatus.CANCELLED,
                    "next_action": DataSourceNextAction.RUN_FIRST_SYNC,
                    "last_error": None,
                },
                deep=True,
            )
        )
    _add_sync_job_event(
        store,
        saved,
        "sync.job_cancelled",
        "Sync job was cancelled.",
        {"cancelledBy": cancelled_by},
    )
    log_event("sync.job_cancelled", syncJobId=saved.id, sourceId=saved.datasource_id, cancelledBy=cancelled_by)
    return saved


def _mark_job_failed_without_source(*, store: Any, job: SyncJob, exc: Exception) -> SyncJob:
    now = utc_now()
    connector_error = ConnectorError(
        code=ConnectorErrorCode.OBJECT_NOT_FOUND,
        user_message="The data source for this sync job no longer exists.",
        technical_message=str(exc),
        retryable=False,
        next_action=ConnectorNextAction.CONTACT_ADMIN,
    )
    failed = job.model_copy(
        update={
            "status": SyncJobStatus.FAILED,
            "finished_at": now,
            "error": connector_error,
            "lease_owner": None,
            "lease_acquired_at": None,
            "lease_expires_at": None,
            "lease_heartbeat_at": None,
        },
        deep=True,
    )
    saved = cast(SyncJob, store.update_sync_job(failed))
    _add_sync_job_event(
        store,
        saved,
        "sync.job_failed",
        "Sync job failed because its data source was not found.",
        {"errorCode": str(connector_error.code), "errorType": type(exc).__name__},
    )
    log_event(
        "sync.job_failed",
        syncJobId=saved.id,
        sourceId=saved.datasource_id,
        provider=saved.provider,
        errorType=type(exc).__name__,
    )
    return saved


def _ensure_not_cancel_requested(store: Any, job_id: str) -> None:
    current = cast(SyncJob, store.get_sync_job(job_id))
    if current.cancel_requested_at is not None or current.status == SyncJobStatus.CANCELLED:
        raise SyncJobCancellationRequested("Sync job cancellation was requested.")


def _advance_sync_cursor(
    *,
    store: Any,
    job: SyncJob,
    data_source: DataSource,
    selected_objects: list[ProviderObjectSnapshot],
    artifact_created_count: int,
    artifact_reused_count: int,
    evidence_created_count: int,
) -> SyncCursor:
    now = utc_now()
    processed_ids = [snapshot.external_id for snapshot in selected_objects]
    cursor_payload = "|".join(sorted(processed_ids))
    last_cursor = hashlib.sha256(cursor_payload.encode("utf-8")).hexdigest() if processed_ids else None
    try:
        existing = cast(SyncCursor, store.get_sync_cursor(data_source.id))
        created_at = existing.created_at
        cursor_id = existing.id
    except NotFoundError:
        created_at = now
        cursor_id = None
    cursor = SyncCursor(
        id=cursor_id or SyncCursor(datasource_id=data_source.id, provider=data_source.type).id,
        datasource_id=data_source.id,
        provider=data_source.type,
        cursor_key="default",
        last_cursor=last_cursor,
        last_successful_sync_job_id=job.id,
        processed_external_object_ids=processed_ids,
        artifact_created_count=artifact_created_count,
        artifact_reused_count=artifact_reused_count,
        evidence_created_count=evidence_created_count,
        created_at=created_at,
        updated_at=now,
        advanced_at=now,
    )
    saved = cast(SyncCursor, store.upsert_sync_cursor(cursor))
    log_event(
        "sync.cursor_advanced",
        syncJobId=job.id,
        sourceId=data_source.id,
        cursorKey=saved.cursor_key,
        processedObjectCount=len(processed_ids),
    )
    return saved


def heartbeat_sync_job_lease(
    *,
    store: Any,
    job_id: str,
    worker_id: str,
    lease_seconds: int,
) -> SyncJob:
    heartbeat = getattr(store, "heartbeat_sync_job_lease", None)
    if heartbeat is None:
        raise SyncJobNotRunnableError("Store does not support sync job lease heartbeat.")
    saved = cast(SyncJob | None, heartbeat(job_id, worker_id=worker_id, lease_seconds=lease_seconds))
    if saved is None:
        raise SyncJobNotRunnableError("Sync job lease cannot be refreshed by this worker.")
    _add_sync_job_event(
        store,
        saved,
        "sync.lease_heartbeat",
        "Sync worker refreshed the job lease.",
        {
            "workerId": worker_id,
            "leaseExpiresAt": None if saved.lease_expires_at is None else saved.lease_expires_at.isoformat(),
        },
    )
    log_event(
        "sync.lease_heartbeat",
        syncJobId=saved.id,
        sourceId=saved.datasource_id,
        workerId=worker_id,
        leaseExpiresAt=saved.lease_expires_at,
    )
    return saved


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=utc_now().tzinfo)
    return value


def _connector_error_from_exception(exc: Exception) -> ConnectorError:
    existing = getattr(exc, "connector_error", None)
    if isinstance(existing, ConnectorError):
        return existing
    return ConnectorError(
        code=ConnectorErrorCode.UNKNOWN,
        user_message="Connector sync failed. Review logs and retry.",
        technical_message=str(exc),
        retryable=True,
        next_action=ConnectorNextAction.RETRY,
    )


def _get_active_credential_for_sync(source_id: str, store: Any) -> ConnectorCredential:
    try:
        return cast(ConnectorCredential, store.get_active_credential_for_source(source_id))
    except NotFoundError as exc:
        raise SyncCredentialError(
            ConnectorError(
                code=ConnectorErrorCode.NO_CREDENTIAL,
                user_message="No active credential exists for this source. Reconnect the source.",
                technical_message=str(exc),
                retryable=False,
                next_action=ConnectorNextAction.RECONNECT,
            )
        ) from exc


def _failed_source_status(data_source: DataSource) -> DataSourceStatus:
    if data_source.status == DataSourceStatus.DISCONNECTED:
        return DataSourceStatus.DISCONNECTED
    if data_source.last_successful_sync_at is not None:
        return DataSourceStatus.DEGRADED
    return DataSourceStatus.FAILED


def _data_source_next_action_from_connector_action(action: ConnectorNextAction) -> DataSourceNextAction:
    if action == ConnectorNextAction.RECONNECT:
        return DataSourceNextAction.RECONNECT
    if action == ConnectorNextAction.GRANT_PERMISSION:
        return DataSourceNextAction.GRANT_PERMISSION
    if action == ConnectorNextAction.SELECT_SOURCES:
        return DataSourceNextAction.SELECT_SOURCES
    if action == ConnectorNextAction.TEST_CONNECTION:
        return DataSourceNextAction.TEST_CONNECTION
    if action in {ConnectorNextAction.RETRY, ConnectorNextAction.WAIT_AND_RETRY}:
        return DataSourceNextAction.RETRY_SYNC
    return DataSourceNextAction.NONE


def _retry_delay_seconds(*, connector_error: ConnectorError, attempt_count: int) -> int:
    if connector_error.retry_after_seconds is not None:
        return connector_error.retry_after_seconds
    exponent: int = attempt_count - 1
    if exponent < 0:
        exponent = 0
    delay: int = 2**exponent
    if delay > 300:
        return 300
    return delay


def _get_source_selection_for_sync(source_id: str, store: Any) -> SourceSelection:
    try:
        return cast(SourceSelection, store.get_source_selection(source_id))
    except NotFoundError:
        return SourceSelection(datasource_id=source_id)


def _get_selected_provider_snapshots_for_sync(source_id: str, store: Any) -> list[ProviderObjectSnapshot]:
    snapshots = cast(list[ProviderObjectSnapshot], store.list_provider_object_snapshots(datasource_id=source_id))
    return [
        snapshot
        for snapshot in snapshots
        if snapshot.selected_for_sync
        and snapshot.access_state == ProviderObjectAccessState.ACCESSIBLE
        and snapshot.ingestion_supported
    ]


def _add_sync_job_event(
    store: Any,
    job: SyncJob,
    event_type: str,
    message: str,
    metadata: dict[str, Any],
) -> SyncJobEvent:
    return cast(
        SyncJobEvent,
        store.add_sync_job_event(
            SyncJobEvent(
                sync_job_id=job.id,
                datasource_id=job.datasource_id,
                event_type=event_type,
                message=message,
                metadata=metadata,
            )
        ),
    )
