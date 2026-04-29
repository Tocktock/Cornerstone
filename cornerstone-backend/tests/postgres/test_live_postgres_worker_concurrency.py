from __future__ import annotations

import os
import threading
from datetime import timedelta

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from cornerstone.persistence.store import SqlAlchemyStore
from cornerstone.schemas import (
    DataSource,
    DataSourceStatus,
    DataSourceType,
    SyncJob,
    SyncJobStatus,
    utc_now,
)

pytestmark = pytest.mark.postgres

LIVE_POSTGRES_SOURCE_ID = "00000000-0000-4000-8000-000000000001"


def _postgres_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url or not os.getenv("RUN_POSTGRES_TESTS"):
        pytest.skip("Set RUN_POSTGRES_TESTS=1 and DATABASE_URL to run live PostgreSQL tests.")
    return url


def _upgrade_database(url: str) -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def test_live_postgres_extensions_and_migration_head() -> None:
    url = _postgres_url()
    _upgrade_database(url)
    engine = create_engine(url)
    with engine.connect() as conn:
        extensions = {
            row[0]
            for row in conn.execute(
                sa.text("select extname from pg_extension where extname in ('pgcrypto','citext','vector')")
            )
        }
        columns = {
            row[0]
            for row in conn.execute(
                sa.text(
                    """
                    select column_name
                    from information_schema.columns
                    where table_name = 'sync_jobs'
                    """
                )
            )
        }
    assert {"pgcrypto", "citext", "vector"} <= extensions
    assert "lease_heartbeat_at" in columns


def test_live_postgres_claims_job_once_under_concurrent_workers() -> None:
    url = _postgres_url()
    _upgrade_database(url)
    engine = create_engine(url)
    store = SqlAlchemyStore(engine=engine)
    store.reset()
    source = store.add_data_source(
        DataSource(id=LIVE_POSTGRES_SOURCE_ID, type=DataSourceType.NOTION, name="Live Postgres Notion", status=DataSourceStatus.CONNECTED)
    )
    job = store.add_sync_job(
        SyncJob(datasource_id=source.id, provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED)
    )

    claimed_by: list[str] = []
    lock = threading.Lock()

    def claim(worker_id: str) -> None:
        candidate = store.claim_sync_job(job.id, worker_id=worker_id, lease_seconds=120)
        if candidate is not None:
            with lock:
                claimed_by.append(worker_id)

    threads = [threading.Thread(target=claim, args=(f"worker-{idx}",)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed_by) == 1
    saved = store.get_sync_job(job.id)
    assert saved.status == SyncJobStatus.RUNNING
    assert saved.lease_owner == claimed_by[0]


def test_live_postgres_expired_lease_can_be_reclaimed_once() -> None:
    url = _postgres_url()
    _upgrade_database(url)
    engine = create_engine(url)
    store = SqlAlchemyStore(engine=engine)
    store.reset()
    source = store.add_data_source(
        DataSource(id=LIVE_POSTGRES_SOURCE_ID, type=DataSourceType.NOTION, name="Live Postgres Notion", status=DataSourceStatus.CONNECTED)
    )
    job = store.add_sync_job(
        SyncJob(datasource_id=source.id, provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED)
    )
    claimed = store.claim_sync_job(job.id, worker_id="worker-a", lease_seconds=30)
    assert claimed is not None
    expired = claimed.model_copy(update={"lease_expires_at": utc_now() - timedelta(seconds=1)}, deep=True)
    store.update_sync_job(expired)

    claimed_by: list[str] = []
    lock = threading.Lock()

    def reclaim(worker_id: str) -> None:
        candidate = store.claim_sync_job(job.id, worker_id=worker_id, lease_seconds=120)
        if candidate is not None:
            with lock:
                claimed_by.append(worker_id)

    threads = [threading.Thread(target=reclaim, args=(f"worker-{idx}",)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed_by) == 1
    saved = store.get_sync_job(job.id)
    assert saved.lease_owner == claimed_by[0]
    assert saved.attempt_count == 2


def test_live_postgres_heartbeat_requires_owner_and_extends_lease() -> None:
    url = _postgres_url()
    _upgrade_database(url)
    engine = create_engine(url)
    store = SqlAlchemyStore(engine=engine)
    store.reset()
    source = store.add_data_source(
        DataSource(id=LIVE_POSTGRES_SOURCE_ID, type=DataSourceType.NOTION, name="Live Postgres Notion", status=DataSourceStatus.CONNECTED)
    )
    job = store.add_sync_job(
        SyncJob(datasource_id=source.id, provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED)
    )
    claimed = store.claim_sync_job(job.id, worker_id="worker-a", lease_seconds=30)
    assert claimed is not None

    wrong_worker = store.heartbeat_sync_job_lease(job.id, worker_id="worker-b", lease_seconds=120)
    assert wrong_worker is None

    refreshed = store.heartbeat_sync_job_lease(job.id, worker_id="worker-a", lease_seconds=120)
    assert refreshed is not None
    assert refreshed.lease_owner == "worker-a"
    assert refreshed.lease_heartbeat_at is not None
    assert refreshed.lease_expires_at is not None
    assert claimed.lease_expires_at is not None
    assert refreshed.lease_expires_at > claimed.lease_expires_at


def test_live_postgres_duplicate_scheduled_enqueue_key_is_rejected() -> None:
    from cornerstone.persistence.store import PersistenceIntegrityError

    url = _postgres_url()
    _upgrade_database(url)
    engine = create_engine(url)
    store = SqlAlchemyStore(engine=engine)
    store.reset()
    source = store.add_data_source(
        DataSource(id=LIVE_POSTGRES_SOURCE_ID, type=DataSourceType.NOTION, name="Live Postgres Notion", status=DataSourceStatus.CONNECTED)
    )
    enqueue_key = "sync-schedule:schedule-1:2026-04-26T00:00:00+00:00"
    first = store.add_sync_job(
        SyncJob(datasource_id=source.id, provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED, enqueue_key=enqueue_key)
    )
    assert first.enqueue_key == enqueue_key

    with pytest.raises(PersistenceIntegrityError):
        store.add_sync_job(
            SyncJob(datasource_id=source.id, provider=DataSourceType.NOTION, status=SyncJobStatus.QUEUED, enqueue_key=enqueue_key)
        )
