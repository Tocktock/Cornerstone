"""live postgres worker concurrency guardrails

Revision ID: 0009_live_postgres_worker_concurrency
Revises: 0008_worker_lease_primitives
Create Date: 2026-04-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_live_postgres_worker_concurrency"
down_revision: str | None = "0008_worker_lease_primitives"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TZ_DATETIME = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.add_column("sync_jobs", sa.Column("lease_heartbeat_at", TZ_DATETIME, nullable=True))
    op.create_index(
        "ix_sync_jobs_claimable",
        "sync_jobs",
        ["status", "next_attempt_at", "lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_sync_jobs_claimable", table_name="sync_jobs")
    op.drop_column("sync_jobs", "lease_heartbeat_at")
