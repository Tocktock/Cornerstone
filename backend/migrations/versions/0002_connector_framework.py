"""connector framework and notion skeleton persistence

Revision ID: 0002_connector_framework
Revises: 0001_postgres_persistence
Create Date: 2026-04-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_connector_framework"
down_revision: str | None = "0001_postgres_persistence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_TYPE = postgresql.UUID(as_uuid=False)
JSONB_TYPE = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "connection_intents",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("auth_type", sa.String(length=32), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("requested_scopes", JSONB_TYPE, nullable=False),
        sa.Column("authorization_url", sa.Text(), nullable=True),
        sa.Column("redirect_uri", sa.Text(), nullable=False),
        sa.Column("return_url", sa.Text(), nullable=True),
        sa.Column("state_nonce", sa.String(length=256), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("datasource_id", UUID_TYPE, nullable=True),
        sa.Column("failure_error", JSONB_TYPE, nullable=True),
        sa.ForeignKeyConstraint(["datasource_id"], ["data_sources.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("state_nonce"),
    )
    op.create_index("ix_connection_intents_created_by", "connection_intents", ["created_by"])
    op.create_index("ix_connection_intents_expires_at", "connection_intents", ["expires_at"])
    op.create_index(
        "ix_connection_intents_provider_status", "connection_intents", ["provider", "status"]
    )

    op.create_table(
        "connector_credentials",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("datasource_id", UUID_TYPE, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("auth_type", sa.String(length=32), nullable=False),
        sa.Column("encrypted_access_token", sa.Text(), nullable=False),
        sa.Column("encrypted_refresh_token", sa.Text(), nullable=True),
        sa.Column("granted_scopes", JSONB_TYPE, nullable=False),
        sa.Column("external_account_id", sa.Text(), nullable=True),
        sa.Column("external_workspace_id", sa.Text(), nullable=True),
        sa.Column("external_workspace_name", sa.Text(), nullable=True),
        sa.Column("external_bot_id", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["datasource_id"], ["data_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_connector_credentials_provider", "connector_credentials", ["provider"])
    op.create_index(
        "ix_connector_credentials_source_status",
        "connector_credentials",
        ["datasource_id", "status"],
    )

    op.create_table(
        "source_selections",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("datasource_id", UUID_TYPE, nullable=False),
        sa.Column("sync_mode", sa.String(length=64), nullable=False),
        sa.Column("include_rules", JSONB_TYPE, nullable=False),
        sa.Column("exclude_rules", JSONB_TYPE, nullable=False),
        sa.Column("selected_external_object_ids", JSONB_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["datasource_id"], ["data_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("datasource_id"),
    )
    op.create_index("ix_source_selections_datasource_id", "source_selections", ["datasource_id"])

    op.create_table(
        "sync_jobs",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("datasource_id", UUID_TYPE, nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("trigger", sa.String(length=32), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("selection_id", UUID_TYPE, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("artifact_created_count", sa.Integer(), nullable=False),
        sa.Column("artifact_reused_count", sa.Integer(), nullable=False),
        sa.Column("evidence_created_count", sa.Integer(), nullable=False),
        sa.Column("error", JSONB_TYPE, nullable=True),
        sa.Column("cursor", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["datasource_id"], ["data_sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sync_jobs_created_at", "sync_jobs", ["created_at"])
    op.create_index("ix_sync_jobs_datasource_status", "sync_jobs", ["datasource_id", "status"])

    op.create_table(
        "sync_job_events",
        sa.Column("id", UUID_TYPE, nullable=False),
        sa.Column("sync_job_id", UUID_TYPE, nullable=False),
        sa.Column("datasource_id", UUID_TYPE, nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", JSONB_TYPE, nullable=False),
        sa.ForeignKeyConstraint(["sync_job_id"], ["sync_jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sync_job_events_datasource", "sync_job_events", ["datasource_id"])
    op.create_index("ix_sync_job_events_event_type", "sync_job_events", ["event_type"])
    op.create_index("ix_sync_job_events_job_time", "sync_job_events", ["sync_job_id", "occurred_at"])


def downgrade() -> None:
    op.drop_index("ix_sync_job_events_job_time", table_name="sync_job_events")
    op.drop_index("ix_sync_job_events_event_type", table_name="sync_job_events")
    op.drop_index("ix_sync_job_events_datasource", table_name="sync_job_events")
    op.drop_table("sync_job_events")
    op.drop_index("ix_sync_jobs_datasource_status", table_name="sync_jobs")
    op.drop_index("ix_sync_jobs_created_at", table_name="sync_jobs")
    op.drop_table("sync_jobs")
    op.drop_index("ix_source_selections_datasource_id", table_name="source_selections")
    op.drop_table("source_selections")
    op.drop_index("ix_connector_credentials_source_status", table_name="connector_credentials")
    op.drop_index("ix_connector_credentials_provider", table_name="connector_credentials")
    op.drop_table("connector_credentials")
    op.drop_index("ix_connection_intents_provider_status", table_name="connection_intents")
    op.drop_index("ix_connection_intents_expires_at", table_name="connection_intents")
    op.drop_index("ix_connection_intents_created_by", table_name="connection_intents")
    op.drop_table("connection_intents")
