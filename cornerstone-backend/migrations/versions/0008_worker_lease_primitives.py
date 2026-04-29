"""sync worker lease and scheduled enqueue idempotency primitives

Revision ID: 0008_worker_lease_primitives
Revises: 0007_provider_object_ingestion_support
Create Date: 2026-04-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_worker_lease_primitives"
down_revision: str | None = "0007_provider_object_ingestion_support"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_TYPE = postgresql.UUID(as_uuid=False)
TZ_DATETIME = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.add_column("sync_jobs", sa.Column("lease_owner", sa.Text(), nullable=True))
    op.add_column("sync_jobs", sa.Column("lease_acquired_at", TZ_DATETIME, nullable=True))
    op.add_column("sync_jobs", sa.Column("lease_expires_at", TZ_DATETIME, nullable=True))
    op.add_column("sync_jobs", sa.Column("schedule_id", UUID_TYPE, nullable=True))
    op.add_column("sync_jobs", sa.Column("enqueue_key", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_sync_jobs_schedule_id_sync_schedules",
        "sync_jobs",
        "sync_schedules",
        ["schedule_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_sync_jobs_lease_expiry", "sync_jobs", ["status", "lease_expires_at"])
    op.create_index("ix_sync_jobs_schedule", "sync_jobs", ["schedule_id"])
    op.create_index("ix_sync_jobs_enqueue_key", "sync_jobs", ["enqueue_key"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_sync_jobs_enqueue_key", table_name="sync_jobs")
    op.drop_index("ix_sync_jobs_schedule", table_name="sync_jobs")
    op.drop_index("ix_sync_jobs_lease_expiry", table_name="sync_jobs")
    op.drop_constraint("fk_sync_jobs_schedule_id_sync_schedules", "sync_jobs", type_="foreignkey")
    op.drop_column("sync_jobs", "enqueue_key")
    op.drop_column("sync_jobs", "schedule_id")
    op.drop_column("sync_jobs", "lease_expires_at")
    op.drop_column("sync_jobs", "lease_acquired_at")
    op.drop_column("sync_jobs", "lease_owner")
