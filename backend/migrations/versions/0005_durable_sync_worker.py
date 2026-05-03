"""durable sync worker runtime

Revision ID: 0005_durable_sync_worker
Revises: 0004_generic_ingestion_contract
Create Date: 2026-04-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_durable_sync_worker"
down_revision: str | None = "0004_generic_ingestion_contract"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_TYPE = postgresql.UUID(as_uuid=False)
JSONB_TYPE = postgresql.JSONB(astext_type=sa.Text())
TZ_DATETIME = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.add_column("sync_jobs", sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("sync_jobs", sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"))
    op.add_column("sync_jobs", sa.Column("next_attempt_at", TZ_DATETIME, nullable=True))
    op.add_column("sync_jobs", sa.Column("cancel_requested_at", TZ_DATETIME, nullable=True))
    op.add_column("sync_jobs", sa.Column("cancelled_by", sa.Text(), nullable=True))
    op.create_index("ix_sync_jobs_retry_ready", "sync_jobs", ["status", "next_attempt_at"])

    op.create_table(
        "sync_cursors",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("datasource_id", UUID_TYPE, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("cursor_key", sa.String(length=128), nullable=False, server_default="default"),
        sa.Column("last_cursor", sa.Text(), nullable=True),
        sa.Column("last_successful_sync_job_id", UUID_TYPE, nullable=True),
        sa.Column("processed_external_object_ids", JSONB_TYPE, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("artifact_created_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("artifact_reused_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("evidence_created_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.Column("updated_at", TZ_DATETIME, nullable=False),
        sa.Column("advanced_at", TZ_DATETIME, nullable=True),
        sa.ForeignKeyConstraint(["datasource_id"], ["data_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["last_successful_sync_job_id"], ["sync_jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("datasource_id", "cursor_key", name="uq_sync_cursors_datasource_key"),
    )
    op.create_index("ix_sync_cursors_datasource", "sync_cursors", ["datasource_id"])
    op.create_index("ix_sync_cursors_provider", "sync_cursors", ["provider"])

    op.alter_column("sync_jobs", "attempt_count", server_default=None)
    op.alter_column("sync_jobs", "max_attempts", server_default=None)
    op.alter_column("sync_cursors", "cursor_key", server_default=None)
    op.alter_column("sync_cursors", "processed_external_object_ids", server_default=None)
    op.alter_column("sync_cursors", "artifact_created_count", server_default=None)
    op.alter_column("sync_cursors", "artifact_reused_count", server_default=None)
    op.alter_column("sync_cursors", "evidence_created_count", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_sync_cursors_provider", table_name="sync_cursors")
    op.drop_index("ix_sync_cursors_datasource", table_name="sync_cursors")
    op.drop_table("sync_cursors")
    op.drop_index("ix_sync_jobs_retry_ready", table_name="sync_jobs")
    op.drop_column("sync_jobs", "cancelled_by")
    op.drop_column("sync_jobs", "cancel_requested_at")
    op.drop_column("sync_jobs", "next_attempt_at")
    op.drop_column("sync_jobs", "max_attempts")
    op.drop_column("sync_jobs", "attempt_count")
