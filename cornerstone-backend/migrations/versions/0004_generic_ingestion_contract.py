"""generic connector ingestion contract

Revision ID: 0004_generic_ingestion_contract
Revises: 0003_source_state_discovery
Create Date: 2026-04-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_generic_ingestion_contract"
down_revision: str | None = "0003_source_state_discovery"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB_TYPE = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.add_column(
        "artifacts",
        sa.Column("source_object_type", sa.String(length=128), nullable=False, server_default="unknown"),
    )
    op.add_column(
        "artifacts",
        sa.Column("provider_metadata", JSONB_TYPE, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column(
        "provider_object_snapshots",
        sa.Column("provider_metadata", JSONB_TYPE, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_artifacts_source_object_type", "artifacts", ["source_type", "source_object_type"])

    op.alter_column("artifacts", "source_object_type", server_default=None)
    op.alter_column("artifacts", "provider_metadata", server_default=None)
    op.alter_column("provider_object_snapshots", "provider_metadata", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_artifacts_source_object_type", table_name="artifacts")
    op.drop_column("provider_object_snapshots", "provider_metadata")
    op.drop_column("artifacts", "provider_metadata")
    op.drop_column("artifacts", "source_object_type")
