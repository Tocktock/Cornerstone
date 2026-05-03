from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

from cornerstone.schemas import SyncJob, SyncJobEvent, SyncJobStatus


def ensure_aware(value: datetime) -> datetime:
    """Normalize datetime values used by sync leasing to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def sync_job_is_claimable(job: SyncJob, *, now: datetime, include_not_ready: bool) -> bool:
    """Shared claimability rule for in-memory and SQLAlchemy sync job stores."""
    if job.status == SyncJobStatus.QUEUED:
        return True
    if job.status == SyncJobStatus.RETRY_WAITING:
        if job.next_attempt_at is None:
            return True
        return include_not_ready or ensure_aware(job.next_attempt_at) <= now
    if job.status == SyncJobStatus.RUNNING:
        return job.lease_expires_at is not None and ensure_aware(job.lease_expires_at) <= now
    return False


def add_sync_job_event(
    store: Any,
    job: SyncJob,
    event_type: str,
    message: str,
    metadata: dict[str, Any],
) -> SyncJobEvent:
    """Persist a sync job event using one shared construction path."""
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
