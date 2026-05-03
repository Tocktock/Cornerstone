"""connector driven ontology re-extraction runs

Revision ID: 0016_ontology_reextraction_runs
Revises: 0015_ontology_graph_evaluation
Create Date: 2026-05-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016_ontology_reextraction_runs"
down_revision: str | None = "0015_ontology_graph_evaluation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TZ_DATETIME = sa.DateTime(timezone=True)


def _uuid_type() -> sa.types.TypeEngine[object]:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.UUID(as_uuid=False)
    return sa.CHAR(36)


def _json_type() -> sa.types.TypeEngine[object]:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.JSONB()
    return sa.JSON()


def upgrade() -> None:
    uuid_type = _uuid_type()
    json_type = _json_type()

    op.add_column(
        "sync_jobs",
        sa.Column("queue_ontology_reextraction", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "sync_jobs",
        sa.Column("run_ontology_reextraction_inline", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("sync_jobs", sa.Column("ontology_focus_concept", sa.Text(), nullable=True))

    op.create_table(
        "ontology_reextraction_runs",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("datasource_id", uuid_type, sa.ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("trigger", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("sync_job_id", uuid_type, sa.ForeignKey("sync_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("focus_concept", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source_external_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("artifact_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("changed_artifact_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("evidence_fragment_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("extraction_run_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("concept_candidate_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("relation_candidate_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("warning_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("official_graph_mutated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.Column("started_at", TZ_DATETIME, nullable=True),
        sa.Column("completed_at", TZ_DATETIME, nullable=True),
    )
    op.create_index(
        "ix_ontology_reextraction_runs_datasource_status",
        "ontology_reextraction_runs",
        ["datasource_id", "status"],
    )
    op.create_index("ix_ontology_reextraction_runs_sync_job", "ontology_reextraction_runs", ["sync_job_id"])
    op.create_index("ix_ontology_reextraction_runs_created_at", "ontology_reextraction_runs", ["created_at"])
    op.create_index("ix_ontology_reextraction_runs_trigger", "ontology_reextraction_runs", ["trigger"])


def downgrade() -> None:
    op.drop_index("ix_ontology_reextraction_runs_trigger", table_name="ontology_reextraction_runs")
    op.drop_index("ix_ontology_reextraction_runs_created_at", table_name="ontology_reextraction_runs")
    op.drop_index("ix_ontology_reextraction_runs_sync_job", table_name="ontology_reextraction_runs")
    op.drop_index("ix_ontology_reextraction_runs_datasource_status", table_name="ontology_reextraction_runs")
    op.drop_table("ontology_reextraction_runs")
    op.drop_column("sync_jobs", "ontology_focus_concept")
    op.drop_column("sync_jobs", "run_ontology_reextraction_inline")
    op.drop_column("sync_jobs", "queue_ontology_reextraction")
