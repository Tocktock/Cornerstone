from __future__ import annotations

from pathlib import Path

import pytest

from cornerstone.persistence.models import SyncJobRow
from cornerstone.schemas import DataSourceType, SyncJob, SyncJobStatus, utc_now
from cornerstone.store import InMemoryStore

pytestmark = pytest.mark.unit


def test_sync_job_schema_tracks_lease_heartbeat() -> None:
    job = SyncJob(datasource_id="source-1", provider=DataSourceType.NOTION)
    heartbeat_at = utc_now()

    saved = job.model_copy(update={"lease_heartbeat_at": heartbeat_at}, deep=True)

    assert saved.lease_heartbeat_at == heartbeat_at


def test_sync_job_table_has_live_postgres_claim_columns_and_indexes() -> None:
    columns = set(SyncJobRow.__table__.columns.keys())
    index_names = {index.name for index in SyncJobRow.__table__.indexes}

    assert "lease_owner" in columns
    assert "lease_acquired_at" in columns
    assert "lease_expires_at" in columns
    assert "lease_heartbeat_at" in columns
    assert "enqueue_key" in columns
    assert "ix_sync_jobs_claimable" in index_names
    assert "ix_sync_jobs_enqueue_key" in index_names


def test_worker_concurrency_migration_adds_heartbeat_and_claimable_index() -> None:
    migration = Path("migrations/versions/0009_live_postgres_worker_concurrency.py").read_text()

    assert 'op.add_column("sync_jobs", sa.Column("lease_heartbeat_at"' in migration
    assert '"ix_sync_jobs_claimable"' in migration
    assert '["status", "next_attempt_at", "lease_expires_at"]' in migration


def test_in_memory_store_reclaims_expired_running_job_lease() -> None:
    store = InMemoryStore()
    job = store.add_sync_job(
        SyncJob(datasource_id="source-1", provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED)
    )
    claimed = store.claim_sync_job(job.id, worker_id="worker-a", lease_seconds=30)
    assert claimed is not None

    expired = claimed.model_copy(
        update={"lease_expires_at": claimed.lease_acquired_at},
        deep=True,
    )
    store.update_sync_job(expired)

    reclaimed = store.claim_sync_job(job.id, worker_id="worker-b", lease_seconds=30)

    assert reclaimed is not None
    assert reclaimed.lease_owner == "worker-b"
    assert reclaimed.attempt_count == 2


def test_in_memory_store_rejects_active_running_job_claim_by_second_worker() -> None:
    store = InMemoryStore()
    job = store.add_sync_job(
        SyncJob(datasource_id="source-1", provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED)
    )
    claimed = store.claim_sync_job(job.id, worker_id="worker-a", lease_seconds=30)
    assert claimed is not None

    second_claim = store.claim_sync_job(job.id, worker_id="worker-b", lease_seconds=30)

    assert second_claim is None


def test_in_memory_store_lease_heartbeat_requires_owner() -> None:
    store = InMemoryStore()
    job = store.add_sync_job(
        SyncJob(datasource_id="source-1", provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED)
    )
    claimed = store.claim_sync_job(job.id, worker_id="worker-a", lease_seconds=30)
    assert claimed is not None

    assert store.heartbeat_sync_job_lease(job.id, worker_id="worker-b", lease_seconds=60) is None
    heartbeat = store.heartbeat_sync_job_lease(job.id, worker_id="worker-a", lease_seconds=60)

    assert heartbeat is not None
    assert heartbeat.lease_heartbeat_at is not None
    assert heartbeat.lease_owner == "worker-a"
