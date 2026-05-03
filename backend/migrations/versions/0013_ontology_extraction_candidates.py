"""ontology extraction candidate persistence

Revision ID: 0013_ontology_extraction_candidates
Revises: 0012_concept_aliases_ontology_graph
Create Date: 2026-04-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_ontology_extraction_candidates"
down_revision: str | None = "0012_concept_aliases_ontology_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
UUID_TYPE = sa.String(36).with_variant(postgresql.UUID(as_uuid=False), "postgresql")
CITEXT_TYPE = sa.String(255).with_variant(postgresql.CITEXT(), "postgresql")
TZ_DATETIME = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "ontology_extraction_runs",
        sa.Column("id", UUID_TYPE, primary_key=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("requested_by", sa.Text(), nullable=False),
        sa.Column("focus_concept", sa.Text(), nullable=True),
        sa.Column("evidence_fragment_ids", JSON_TYPE, nullable=False),
        sa.Column("artifact_ids", JSON_TYPE, nullable=False),
        sa.Column("concept_candidate_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("relation_candidate_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("warning_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.Column("started_at", TZ_DATETIME, nullable=True),
        sa.Column("completed_at", TZ_DATETIME, nullable=True),
    )
    op.create_index("ix_ontology_extraction_runs_status", "ontology_extraction_runs", ["status"])
    op.create_index("ix_ontology_extraction_runs_created_at", "ontology_extraction_runs", ["created_at"])
    op.create_index("ix_ontology_extraction_runs_requested_by", "ontology_extraction_runs", ["requested_by"])

    op.create_table(
        "concept_candidates",
        sa.Column("id", UUID_TYPE, primary_key=True),
        sa.Column("extraction_run_id", UUID_TYPE, sa.ForeignKey("ontology_extraction_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", CITEXT_TYPE, nullable=False),
        sa.Column("normalized_name", CITEXT_TYPE, nullable=False),
        sa.Column("aliases", JSON_TYPE, nullable=False),
        sa.Column("proposed_definition", sa.Text(), nullable=False),
        sa.Column("concept_type", sa.String(length=64), nullable=False),
        sa.Column("evidence_fragment_ids", JSON_TYPE, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("matched_existing_concept_id", UUID_TYPE, sa.ForeignKey("concepts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("validation_errors", JSON_TYPE, nullable=False),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.UniqueConstraint("extraction_run_id", "normalized_name", name="uq_concept_candidates_run_normalized_name"),
    )
    op.create_index("ix_concept_candidates_extraction_run_id", "concept_candidates", ["extraction_run_id"])
    op.create_index("ix_concept_candidates_status", "concept_candidates", ["status"])
    op.create_index("ix_concept_candidates_normalized_name", "concept_candidates", ["normalized_name"])

    op.create_table(
        "relation_candidates",
        sa.Column("id", UUID_TYPE, primary_key=True),
        sa.Column("extraction_run_id", UUID_TYPE, sa.ForeignKey("ontology_extraction_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_name", CITEXT_TYPE, nullable=False),
        sa.Column("target_name", CITEXT_TYPE, nullable=False),
        sa.Column("normalized_source_name", CITEXT_TYPE, nullable=False),
        sa.Column("normalized_target_name", CITEXT_TYPE, nullable=False),
        sa.Column("source_candidate_id", UUID_TYPE, sa.ForeignKey("concept_candidates.id", ondelete="SET NULL"), nullable=True),
        sa.Column("target_candidate_id", UUID_TYPE, sa.ForeignKey("concept_candidates.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source_concept_id", UUID_TYPE, sa.ForeignKey("concepts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("target_concept_id", UUID_TYPE, sa.ForeignKey("concepts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("relation_type", sa.String(length=64), nullable=False),
        sa.Column("evidence_fragment_ids", JSON_TYPE, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("validation_errors", JSON_TYPE, nullable=False),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.CheckConstraint("normalized_source_name <> normalized_target_name", name="ck_relation_candidates_distinct_terms"),
        sa.UniqueConstraint(
            "extraction_run_id",
            "normalized_source_name",
            "normalized_target_name",
            "relation_type",
            name="uq_relation_candidates_run_source_target_type",
        ),
    )
    op.create_index("ix_relation_candidates_extraction_run_id", "relation_candidates", ["extraction_run_id"])
    op.create_index("ix_relation_candidates_status", "relation_candidates", ["status"])
    op.create_index("ix_relation_candidates_relation_type", "relation_candidates", ["relation_type"])


def downgrade() -> None:
    op.drop_index("ix_relation_candidates_relation_type", table_name="relation_candidates")
    op.drop_index("ix_relation_candidates_status", table_name="relation_candidates")
    op.drop_index("ix_relation_candidates_extraction_run_id", table_name="relation_candidates")
    op.drop_table("relation_candidates")

    op.drop_index("ix_concept_candidates_normalized_name", table_name="concept_candidates")
    op.drop_index("ix_concept_candidates_status", table_name="concept_candidates")
    op.drop_index("ix_concept_candidates_extraction_run_id", table_name="concept_candidates")
    op.drop_table("concept_candidates")

    op.drop_index("ix_ontology_extraction_runs_requested_by", table_name="ontology_extraction_runs")
    op.drop_index("ix_ontology_extraction_runs_created_at", table_name="ontology_extraction_runs")
    op.drop_index("ix_ontology_extraction_runs_status", table_name="ontology_extraction_runs")
    op.drop_table("ontology_extraction_runs")
