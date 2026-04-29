from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, cast

from cornerstone.observability import log_event
from cornerstone.schemas import (
    DataSource,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    FreshnessState,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    SourceSelection,
    SyncJob,
    SyncJobEvent,
    SyncJobStatus,
    SyncJobTrigger,
    SyncSchedule,
    SyncSchedulerRunResult,
    SyncScheduleStatus,
    new_id,
    utc_now,
)
from cornerstone.store import NotFoundError, StoreError

_NON_TERMINAL_JOB_STATUSES = {
    SyncJobStatus.QUEUED,
    SyncJobStatus.RUNNING,
    SyncJobStatus.RETRY_WAITING,
}


def build_sync_schedule(
    *,
    data_source: DataSource,
    enabled: bool,
    interval_minutes: int,
    start_at: datetime | None,
    max_attempts: int,
    created_by: str,
    existing: SyncSchedule | None = None,
) -> SyncSchedule:
    """Build a validated schedule model while preserving stable identity on updates."""

    now = utc_now()
    return SyncSchedule(
        id=existing.id if existing is not None else new_id(),
        datasource_id=data_source.id,
        provider=data_source.type,
        status=SyncScheduleStatus.ACTIVE if enabled else SyncScheduleStatus.PAUSED,
        interval_minutes=interval_minutes,
        next_run_at=start_at or (existing.next_run_at if existing is not None else now + timedelta(minutes=interval_minutes)),
        last_enqueued_at=None if existing is None else existing.last_enqueued_at,
        last_enqueued_sync_job_id=None if existing is None else existing.last_enqueued_sync_job_id,
        max_attempts=max_attempts,
        created_by=created_by if existing is None else existing.created_by,
        created_at=now if existing is None else existing.created_at,
        updated_at=now,
    )


def next_schedule_run_at(*, schedule: SyncSchedule, from_time: datetime | None = None) -> datetime:
    base = from_time or utc_now()
    return base + timedelta(minutes=schedule.interval_minutes)


def build_schedule_enqueue_key(*, schedule: SyncSchedule) -> str:
    """Stable idempotency key for one schedule due window.

    This is a pre-v0.9 multi-worker safety primitive. In v0.9 this key should be
    backed by live PostgreSQL uniqueness/concurrency tests so two schedulers cannot
    enqueue duplicate jobs for the same schedule window.
    """

    due_at = _aware(schedule.next_run_at).isoformat()
    return f"sync-schedule:{schedule.id}:{due_at}"


def run_sync_scheduler(
    *,
    store: Any,
    max_schedules: int = 50,
    include_not_due: bool = False,
    now: datetime | None = None,
) -> SyncSchedulerRunResult:
    """Enqueue scheduled sync jobs that are due and safe to run.

    This does not execute jobs. It only creates durable SyncJob rows so an external
    worker process can drain the queue. This keeps scheduling, queuing, and execution
    independently observable and testable.
    """

    checked_at = _aware(now or utc_now())
    schedules = _eligible_schedules(store, checked_at=checked_at, include_not_due=include_not_due)
    jobs: list[SyncJob] = []
    saved_schedules: list[SyncSchedule] = []
    skipped = 0

    for schedule in schedules[:max_schedules]:
        if not _source_can_schedule_sync(store, schedule.datasource_id):
            skipped += 1
            continue
        if _has_active_sync_job(store, schedule.datasource_id):
            skipped += 1
            continue
        try:
            selection = cast(SourceSelection, store.get_source_selection(schedule.datasource_id))
        except NotFoundError:
            skipped += 1
            continue
        if not selection.selected_external_object_ids:
            skipped += 1
            continue
        if not _source_has_syncable_selected_objects(store, schedule.datasource_id):
            skipped += 1
            continue

        enqueue_key = build_schedule_enqueue_key(schedule=schedule)
        if _sync_job_enqueue_key_exists(store, enqueue_key):
            skipped += 1
            continue

        job = SyncJob(
            datasource_id=schedule.datasource_id,
            provider=schedule.provider,
            status=SyncJobStatus.QUEUED,
            trigger=SyncJobTrigger.SCHEDULED,
            created_by=schedule.created_by,
            selection_id=selection.id,
            max_attempts=schedule.max_attempts,
            created_at=checked_at,
            schedule_id=schedule.id,
            enqueue_key=enqueue_key,
        )
        try:
            with store.transaction():
                saved_job = cast(SyncJob, store.add_sync_job(job))
                store.add_sync_job_event(
                    SyncJobEvent(
                        sync_job_id=saved_job.id,
                        datasource_id=saved_job.datasource_id,
                        event_type="sync.job_scheduled",
                        message="Scheduled sync job was enqueued.",
                        metadata={
                            "scheduleId": schedule.id,
                            "intervalMinutes": schedule.interval_minutes,
                            "enqueueKey": enqueue_key,
                        },
                        occurred_at=checked_at,
                    )
                )
                updated_schedule = schedule.model_copy(
                    update={
                        "last_enqueued_at": checked_at,
                        "last_enqueued_sync_job_id": saved_job.id,
                        "next_run_at": next_schedule_run_at(schedule=schedule, from_time=checked_at),
                        "updated_at": checked_at,
                    },
                    deep=True,
                )
                saved_schedule = cast(SyncSchedule, store.upsert_sync_schedule(updated_schedule))
                source = cast(DataSource, store.get_data_source(schedule.datasource_id))
                store.update_data_source(
                    source.model_copy(
                        update={
                            "status": DataSourceStatus.SYNC_PENDING,
                            "sync_status": DataSourceSyncStatus.QUEUED,
                            "next_action": DataSourceNextAction.NONE,
                            "last_error": None,
                        },
                        deep=True,
                    )
                )
        except StoreError:
            skipped += 1
            continue
        jobs.append(saved_job)
        saved_schedules.append(saved_schedule)
        log_event(
            "sync.schedule_enqueued",
            syncScheduleId=schedule.id,
            syncJobId=saved_job.id,
            sourceId=schedule.datasource_id,
            nextRunAt=saved_schedule.next_run_at,
        )

    return SyncSchedulerRunResult(
        enqueued_job_count=len(jobs),
        skipped_schedule_count=skipped,
        schedules_checked_count=len(schedules[:max_schedules]),
        jobs=jobs,
        schedules=saved_schedules,
    )


def _eligible_schedules(store: Any, *, checked_at: datetime, include_not_due: bool) -> list[SyncSchedule]:
    schedules = [
        schedule
        for schedule in cast(list[SyncSchedule], store.list_sync_schedules())
        if schedule.status == SyncScheduleStatus.ACTIVE
        and (include_not_due or _aware(schedule.next_run_at) <= checked_at)
    ]
    return sorted(schedules, key=lambda item: (item.next_run_at, item.created_at, item.id))


def _source_can_schedule_sync(store: Any, datasource_id: str) -> bool:
    try:
        source = cast(DataSource, store.get_data_source(datasource_id))
    except NotFoundError:
        return False
    if source.status == DataSourceStatus.DISCONNECTED:
        return False
    if source.sync_status in {DataSourceSyncStatus.SYNCING, DataSourceSyncStatus.WAITING_RETRY}:
        return False
    if source.next_action in {DataSourceNextAction.TEST_CONNECTION, DataSourceNextAction.SELECT_SOURCES, DataSourceNextAction.CONNECT}:
        return False
    if source.freshness_state == FreshnessState.FRESH and source.sync_status == DataSourceSyncStatus.SUCCEEDED:
        # Fresh sources may still be scheduled by interval, but only after next_run_at is due.
        return True
    return True


def _has_active_sync_job(store: Any, datasource_id: str) -> bool:
    jobs = cast(list[SyncJob], store.list_sync_jobs(datasource_id=datasource_id))
    return any(job.status in _NON_TERMINAL_JOB_STATUSES for job in jobs)


def _source_has_syncable_selected_objects(store: Any, datasource_id: str) -> bool:
    snapshots = cast(list[ProviderObjectSnapshot], store.list_provider_object_snapshots(datasource_id=datasource_id))
    return any(
        snapshot.selected_for_sync
        and snapshot.access_state == ProviderObjectAccessState.ACCESSIBLE
        and snapshot.ingestion_supported
        for snapshot in snapshots
    )


def _sync_job_enqueue_key_exists(store: Any, enqueue_key: str) -> bool:
    try:
        store.get_sync_job_by_enqueue_key(enqueue_key)
        return True
    except NotFoundError:
        return False


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
