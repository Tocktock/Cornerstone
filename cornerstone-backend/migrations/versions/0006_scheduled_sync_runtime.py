"""scheduled sync runtime and external worker readiness

Revision ID: 0006_scheduled_sync_runtime
Revises: 0005_durable_sync_worker
Create Date: 2026-04-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_scheduled_sync_runtime"
down_revision: str | None = "0005_durable_sync_worker"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_TYPE = postgresql.UUID(as_uuid=False)
TZ_DATETIME = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "sync_schedules",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("datasource_id", UUID_TYPE, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), nullable=False),
        sa.Column("next_run_at", TZ_DATETIME, nullable=False),
        sa.Column("last_enqueued_at", TZ_DATETIME, nullable=True),
        sa.Column("last_enqueued_sync_job_id", UUID_TYPE, nullable=True),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.Column("updated_at", TZ_DATETIME, nullable=False),
        sa.ForeignKeyConstraint(["datasource_id"], ["data_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["last_enqueued_sync_job_id"], ["sync_jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("datasource_id", name="uq_sync_schedules_datasource"),
    )
    op.create_index("ix_sync_schedules_status_next_run", "sync_schedules", ["status", "next_run_at"])
    op.create_index("ix_sync_schedules_datasource", "sync_schedules", ["datasource_id"])
    op.create_index("ix_sync_schedules_provider", "sync_schedules", ["provider"])
    op.alter_column("sync_schedules", "max_attempts", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_sync_schedules_provider", table_name="sync_schedules")
    op.drop_index("ix_sync_schedules_datasource", table_name="sync_schedules")
    op.drop_index("ix_sync_schedules_status_next_run", table_name="sync_schedules")
    op.drop_table("sync_schedules")
