from __future__ import annotations

from datetime import timedelta

import pytest

from cornerstone.config import Settings
from cornerstone.connectors.registry import get_token_cipher
from cornerstone.persistence.models import Base, SyncScheduleRow
from cornerstone.schemas import (
    ConnectorAuthType,
    ConnectorCredential,
    CredentialStatus,
    DataSource,
    DataSourceAuthStatus,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    FreshnessState,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    SourceSelection,
    SyncScheduleStatus,
    utc_now,
)
from cornerstone.services.sync_scheduler import (
    build_sync_schedule,
    next_schedule_run_at,
    run_sync_scheduler,
)
from cornerstone.store import InMemoryStore

pytestmark = pytest.mark.unit


def test_sync_schedule_table_is_provider_agnostic_and_due_indexed() -> None:
    table_names = set(Base.metadata.tables)
    column_names = set(SyncScheduleRow.__table__.columns.keys())
    index_names = {index.name for index in SyncScheduleRow.__table__.indexes}

    assert "sync_schedules" in table_names
    assert {
        "datasource_id",
        "provider",
        "status",
        "interval_minutes",
        "next_run_at",
        "last_enqueued_sync_job_id",
        "max_attempts",
    }.issubset(column_names)
    assert "ix_sync_schedules_status_next_run" in index_names
    assert "notion" not in column_names


def test_build_schedule_preserves_identity_and_computes_next_run() -> None:
    source = _ready_source()
    first = build_sync_schedule(
        data_source=source,
        enabled=True,
        interval_minutes=30,
        start_at=None,
        max_attempts=2,
        created_by="admin@example.com",
        existing=None,
    )
    second = build_sync_schedule(
        data_source=source,
        enabled=False,
        interval_minutes=60,
        start_at=first.next_run_at + timedelta(hours=1),
        max_attempts=4,
        created_by="other@example.com",
        existing=first,
    )

    assert first.status == SyncScheduleStatus.ACTIVE
    assert second.id == first.id
    assert second.created_at == first.created_at
    assert second.created_by == "admin@example.com"
    assert second.status == SyncScheduleStatus.PAUSED
    assert second.interval_minutes == 60


def test_next_schedule_run_at_uses_interval_minutes() -> None:
    source = _ready_source()
    base = utc_now()
    schedule = build_sync_schedule(
        data_source=source,
        enabled=True,
        interval_minutes=45,
        start_at=base,
        max_attempts=3,
        created_by="admin@example.com",
    )

    assert next_schedule_run_at(schedule=schedule, from_time=base) == base + timedelta(minutes=45)


def test_scheduler_only_advances_schedule_when_job_is_enqueued() -> None:
    store = _ready_store()
    source = store.list_data_sources()[0]
    due = utc_now() - timedelta(minutes=1)
    schedule = build_sync_schedule(
        data_source=source,
        enabled=True,
        interval_minutes=20,
        start_at=due,
        max_attempts=3,
        created_by="scheduler@example.com",
    )
    store.upsert_sync_schedule(schedule)

    result = run_sync_scheduler(store=store, max_schedules=10)

    assert result.enqueued_job_count == 1
    assert result.jobs[0].trigger == "scheduled"
    saved = store.get_sync_schedule(source.id)
    assert saved.last_enqueued_sync_job_id == result.jobs[0].id
    assert saved.next_run_at > due


def test_scheduler_skips_source_without_selection() -> None:
    store = InMemoryStore()
    source = _ready_source()
    store.add_data_source(source)
    schedule = build_sync_schedule(
        data_source=source,
        enabled=True,
        interval_minutes=20,
        start_at=utc_now() - timedelta(minutes=1),
        max_attempts=3,
        created_by="scheduler@example.com",
    )
    store.upsert_sync_schedule(schedule)

    result = run_sync_scheduler(store=store, max_schedules=10)

    assert result.enqueued_job_count == 0
    assert result.skipped_schedule_count == 1
    assert store.get_sync_schedule(source.id).last_enqueued_sync_job_id is None


def _ready_source() -> DataSource:
    return DataSource(
        type=DataSourceType.NOTION,
        name="Team Notion",
        status=DataSourceStatus.CONNECTED,
        production_enabled=True,
        auth_status=DataSourceAuthStatus.AUTHORIZED,
        connection_status=DataSourceConnectionStatus.TEST_PASSED,
        sync_status=DataSourceSyncStatus.SUCCEEDED,
        next_action=DataSourceNextAction.NONE,
        freshness_state=FreshnessState.UNKNOWN,
    )


def _ready_store() -> InMemoryStore:
    store = InMemoryStore()
    source = store.add_data_source(_ready_source())
    store.add_connector_credential(
        ConnectorCredential(
            datasource_id=source.id,
            provider=DataSourceType.NOTION,
            auth_type=ConnectorAuthType.OAUTH2,
            encrypted_access_token=get_token_cipher(Settings()).encrypt("mock-token"),
            status=CredentialStatus.ACTIVE,
        )
    )
    store.upsert_provider_object_snapshot(
        ProviderObjectSnapshot(
            datasource_id=source.id,
            provider=DataSourceType.NOTION,
            external_id="notion-page-1",
            title="Architecture",
            selected_for_sync=True,
            access_state=ProviderObjectAccessState.ACCESSIBLE,
            raw_metadata_hash="hash-with-enough-length",
        )
    )
    store.upsert_source_selection(
        SourceSelection(datasource_id=source.id, selected_external_object_ids=["notion-page-1"])
    )
    return store


def test_external_worker_iteration_runs_scheduler_then_worker() -> None:
    import asyncio

    from cornerstone.workers.sync_worker import run_worker_iteration

    store = _ready_store()
    source = store.list_data_sources()[0]
    schedule = build_sync_schedule(
        data_source=source,
        enabled=True,
        interval_minutes=20,
        start_at=utc_now() - timedelta(minutes=1),
        max_attempts=3,
        created_by="scheduler@example.com",
    )
    store.upsert_sync_schedule(schedule)

    result = asyncio.run(
        run_worker_iteration(
            store=store,
            settings=Settings(),
            max_jobs=1,
            run_scheduler_first=True,
        )
    )

    assert result.scheduler is not None
    assert result.scheduler.enqueued_job_count == 1
    assert result.worker.processed_job_count == 1
    assert result.worker.jobs[0].job.status == "succeeded"
    assert store.get_sync_cursor(source.id).last_successful_sync_job_id == result.worker.jobs[0].job.id


def test_sync_job_table_has_worker_lease_and_enqueue_primitives() -> None:
    from cornerstone.persistence.models import SyncJobRow

    column_names = set(SyncJobRow.__table__.columns.keys())
    index_names = {index.name for index in SyncJobRow.__table__.indexes}

    assert {
        "lease_owner",
        "lease_acquired_at",
        "lease_expires_at",
        "schedule_id",
        "enqueue_key",
    }.issubset(column_names)
    assert "ix_sync_jobs_lease_expiry" in index_names
    assert "ix_sync_jobs_enqueue_key" in index_names


def test_in_memory_store_claims_sync_job_once_with_worker_lease() -> None:
    from cornerstone.schemas import SyncJob, SyncJobStatus, SyncJobTrigger

    store = InMemoryStore()
    source = store.add_data_source(_ready_source())
    job = store.add_sync_job(
        SyncJob(
            datasource_id=source.id,
            provider=source.type,
            status=SyncJobStatus.QUEUED,
            trigger=SyncJobTrigger.MANUAL,
            created_by="admin@example.com",
        )
    )

    claimed = store.claim_sync_job(job.id, worker_id="worker-a", lease_seconds=120)
    second_claim = store.claim_sync_job(job.id, worker_id="worker-b", lease_seconds=120)

    assert claimed is not None
    assert claimed.status == SyncJobStatus.RUNNING
    assert claimed.lease_owner == "worker-a"
    assert claimed.lease_acquired_at is not None
    assert claimed.lease_expires_at is not None
    assert claimed.attempt_count == 1
    assert second_claim is None


def test_schedule_enqueue_key_is_stable_and_duplicate_jobs_are_rejected() -> None:
    from cornerstone.schemas import SyncJob, SyncJobStatus, SyncJobTrigger
    from cornerstone.services.sync_scheduler import build_schedule_enqueue_key
    from cornerstone.store import StoreError

    store = _ready_store()
    source = store.list_data_sources()[0]
    due = utc_now() - timedelta(minutes=1)
    schedule = build_sync_schedule(
        data_source=source,
        enabled=True,
        interval_minutes=20,
        start_at=due,
        max_attempts=3,
        created_by="scheduler@example.com",
    )
    key = build_schedule_enqueue_key(schedule=schedule)

    store.add_sync_job(
        SyncJob(
            datasource_id=source.id,
            provider=source.type,
            status=SyncJobStatus.QUEUED,
            trigger=SyncJobTrigger.SCHEDULED,
            created_by="scheduler@example.com",
            schedule_id=schedule.id,
            enqueue_key=key,
        )
    )

    with pytest.raises(StoreError):
        store.add_sync_job(
            SyncJob(
                datasource_id=source.id,
                provider=source.type,
                status=SyncJobStatus.QUEUED,
                trigger=SyncJobTrigger.SCHEDULED,
                created_by="scheduler@example.com",
                schedule_id=schedule.id,
                enqueue_key=key,
            )
        )
