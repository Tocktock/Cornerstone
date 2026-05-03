"""provider object ingestion support metadata

Revision ID: 0007_provider_object_ingestion_support
Revises: 0006_scheduled_sync_runtime
Create Date: 2026-04-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_provider_object_ingestion_support"
down_revision: str | None = "0006_scheduled_sync_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "provider_object_snapshots",
        sa.Column("ingestion_supported", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "provider_object_snapshots",
        sa.Column("ingestion_unsupported_reason", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_provider_object_snapshots_ingestion",
        "provider_object_snapshots",
        ["datasource_id", "ingestion_supported"],
    )
    op.alter_column("provider_object_snapshots", "ingestion_supported", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_provider_object_snapshots_ingestion", table_name="provider_object_snapshots")
    op.drop_column("provider_object_snapshots", "ingestion_unsupported_reason")
    op.drop_column("provider_object_snapshots", "ingestion_supported")
