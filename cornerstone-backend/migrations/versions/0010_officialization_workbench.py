"""officialization workbench relations and review queue

Revision ID: 0010_officialization_workbench
Revises: 0009_live_postgres_worker_concurrency
Create Date: 2026-04-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_officialization_workbench"
down_revision: str | None = "0009_live_postgres_worker_concurrency"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TZ_DATETIME = sa.DateTime(timezone=True)


def _uuid_type() -> sa.types.TypeEngine[object]:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.UUID(as_uuid=False)
    return sa.CHAR(36)


def upgrade() -> None:
    uuid_type = _uuid_type()
    op.create_table(
        "concept_relations",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("source_concept_id", uuid_type, sa.ForeignKey("concepts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("target_concept_id", uuid_type, sa.ForeignKey("concepts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("relation_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("decision_record_id", uuid_type, sa.ForeignKey("decision_records.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("officialized_by", sa.Text(), nullable=True),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.Column("updated_at", TZ_DATETIME, nullable=False),
        sa.Column("last_reviewed_at", TZ_DATETIME, nullable=True),
        sa.CheckConstraint("source_concept_id <> target_concept_id", name="ck_concept_relations_distinct_concepts"),
    )
    op.create_index("ix_concept_relations_source", "concept_relations", ["source_concept_id"])
    op.create_index("ix_concept_relations_target", "concept_relations", ["target_concept_id"])
    op.create_index("ix_concept_relations_status", "concept_relations", ["status"])
    op.create_index("ix_concept_relations_type", "concept_relations", ["relation_type"])

    op.create_table(
        "concept_relation_evidence_fragments",
        sa.Column("concept_relation_id", uuid_type, sa.ForeignKey("concept_relations.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("evidence_fragment_id", uuid_type, sa.ForeignKey("evidence_fragments.id", ondelete="RESTRICT"), primary_key=True),
    )
    op.create_index(
        "ix_concept_relation_evidence_fragments_evidence",
        "concept_relation_evidence_fragments",
        ["evidence_fragment_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_concept_relation_evidence_fragments_evidence", table_name="concept_relation_evidence_fragments")
    op.drop_table("concept_relation_evidence_fragments")
    op.drop_index("ix_concept_relations_type", table_name="concept_relations")
    op.drop_index("ix_concept_relations_status", table_name="concept_relations")
    op.drop_index("ix_concept_relations_target", table_name="concept_relations")
    op.drop_index("ix_concept_relations_source", table_name="concept_relations")
    op.drop_table("concept_relations")
