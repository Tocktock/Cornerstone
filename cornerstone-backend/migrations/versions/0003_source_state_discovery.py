"""refined source runtime state and provider discovery snapshots

Revision ID: 0003_source_state_discovery
Revises: 0002_connector_framework
Create Date: 2026-04-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_source_state_discovery"
down_revision: str | None = "0002_connector_framework"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_TYPE = postgresql.UUID(as_uuid=False)


def upgrade() -> None:
    op.add_column("data_sources", sa.Column("auth_status", sa.String(length=32), nullable=False, server_default="not_started"))
    op.add_column("data_sources", sa.Column("connection_status", sa.String(length=32), nullable=False, server_default="untested"))
    op.add_column("data_sources", sa.Column("sync_status", sa.String(length=32), nullable=False, server_default="never_synced"))
    op.add_column("data_sources", sa.Column("next_action", sa.String(length=32), nullable=False, server_default="connect"))
    op.add_column("data_sources", sa.Column("last_connection_test_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("data_sources", sa.Column("last_discovery_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("data_sources", sa.Column("discovered_object_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("data_sources", sa.Column("selected_object_count", sa.Integer(), nullable=False, server_default="0"))
    op.create_index(
        "ix_data_sources_runtime_state",
        "data_sources",
        ["auth_status", "connection_status", "sync_status", "next_action"],
    )

    op.create_table(
        "provider_object_snapshots",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("datasource_id", UUID_TYPE, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("external_url", sa.Text(), nullable=True),
        sa.Column("object_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("parent_external_id", sa.Text(), nullable=True),
        sa.Column("last_edited_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("selected_for_sync", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("access_state", sa.String(length=32), nullable=False),
        sa.Column("raw_metadata_hash", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["datasource_id"], ["data_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("datasource_id", "external_id", name="uq_provider_object_snapshots_source_external"),
    )
    op.create_index("ix_provider_object_snapshots_datasource", "provider_object_snapshots", ["datasource_id"])
    op.create_index("ix_provider_object_snapshots_access", "provider_object_snapshots", ["datasource_id", "access_state"])
    op.create_index("ix_provider_object_snapshots_selected", "provider_object_snapshots", ["datasource_id", "selected_for_sync"])
    op.create_index("ix_provider_object_snapshots_type", "provider_object_snapshots", ["provider", "object_type"])

    op.alter_column("data_sources", "auth_status", server_default=None)
    op.alter_column("data_sources", "connection_status", server_default=None)
    op.alter_column("data_sources", "sync_status", server_default=None)
    op.alter_column("data_sources", "next_action", server_default=None)
    op.alter_column("data_sources", "discovered_object_count", server_default=None)
    op.alter_column("data_sources", "selected_object_count", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_provider_object_snapshots_type", table_name="provider_object_snapshots")
    op.drop_index("ix_provider_object_snapshots_selected", table_name="provider_object_snapshots")
    op.drop_index("ix_provider_object_snapshots_access", table_name="provider_object_snapshots")
    op.drop_index("ix_provider_object_snapshots_datasource", table_name="provider_object_snapshots")
    op.drop_table("provider_object_snapshots")
    op.drop_index("ix_data_sources_runtime_state", table_name="data_sources")
    op.drop_column("data_sources", "selected_object_count")
    op.drop_column("data_sources", "discovered_object_count")
    op.drop_column("data_sources", "last_discovery_at")
    op.drop_column("data_sources", "last_connection_test_at")
    op.drop_column("data_sources", "next_action")
    op.drop_column("data_sources", "sync_status")
    op.drop_column("data_sources", "connection_status")
    op.drop_column("data_sources", "auth_status")
