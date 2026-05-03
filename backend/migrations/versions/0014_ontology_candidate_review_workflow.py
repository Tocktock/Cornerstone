"""ontology candidate review workflow

Revision ID: 0014_ontology_candidate_review_workflow
Revises: 0013_ontology_extraction_candidates
Create Date: 2026-04-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014_ontology_candidate_review_workflow"
down_revision: str | None = "0013_ontology_extraction_candidates"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID_TYPE = sa.String(36).with_variant(postgresql.UUID(as_uuid=False), "postgresql")
TZ_DATETIME = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.add_column("concept_candidates", sa.Column("reviewed_by", sa.Text(), nullable=True))
    op.add_column("concept_candidates", sa.Column("reviewed_at", TZ_DATETIME, nullable=True))
    op.add_column("concept_candidates", sa.Column("review_note", sa.Text(), nullable=True))
    op.add_column("concept_candidates", sa.Column("promoted_concept_id", UUID_TYPE, nullable=True))
    op.add_column("concept_candidates", sa.Column("merged_into_concept_id", UUID_TYPE, nullable=True))
    op.create_foreign_key(
        "fk_concept_candidates_promoted_concept_id_concepts",
        "concept_candidates",
        "concepts",
        ["promoted_concept_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_concept_candidates_merged_into_concept_id_concepts",
        "concept_candidates",
        "concepts",
        ["merged_into_concept_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_concept_candidates_reviewed_at", "concept_candidates", ["reviewed_at"])
    op.create_index("ix_concept_candidates_promoted_concept_id", "concept_candidates", ["promoted_concept_id"])
    op.create_index("ix_concept_candidates_merged_into_concept_id", "concept_candidates", ["merged_into_concept_id"])

    op.add_column("relation_candidates", sa.Column("reviewed_by", sa.Text(), nullable=True))
    op.add_column("relation_candidates", sa.Column("reviewed_at", TZ_DATETIME, nullable=True))
    op.add_column("relation_candidates", sa.Column("review_note", sa.Text(), nullable=True))
    op.add_column("relation_candidates", sa.Column("promoted_relation_id", UUID_TYPE, nullable=True))
    op.add_column("relation_candidates", sa.Column("merged_into_relation_id", UUID_TYPE, nullable=True))
    op.create_foreign_key(
        "fk_relation_candidates_promoted_relation_id",
        "relation_candidates",
        "concept_relations",
        ["promoted_relation_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_relation_candidates_merged_relation_id",
        "relation_candidates",
        "concept_relations",
        ["merged_into_relation_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_relation_candidates_reviewed_at", "relation_candidates", ["reviewed_at"])
    op.create_index("ix_relation_candidates_promoted_relation_id", "relation_candidates", ["promoted_relation_id"])
    op.create_index("ix_relation_candidates_merged_into_relation_id", "relation_candidates", ["merged_into_relation_id"])


def downgrade() -> None:
    op.drop_index("ix_relation_candidates_merged_into_relation_id", table_name="relation_candidates")
    op.drop_index("ix_relation_candidates_promoted_relation_id", table_name="relation_candidates")
    op.drop_index("ix_relation_candidates_reviewed_at", table_name="relation_candidates")
    op.drop_constraint(
        "fk_relation_candidates_merged_relation_id",
        "relation_candidates",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_relation_candidates_promoted_relation_id",
        "relation_candidates",
        type_="foreignkey",
    )
    op.drop_column("relation_candidates", "merged_into_relation_id")
    op.drop_column("relation_candidates", "promoted_relation_id")
    op.drop_column("relation_candidates", "review_note")
    op.drop_column("relation_candidates", "reviewed_at")
    op.drop_column("relation_candidates", "reviewed_by")

    op.drop_index("ix_concept_candidates_merged_into_concept_id", table_name="concept_candidates")
    op.drop_index("ix_concept_candidates_promoted_concept_id", table_name="concept_candidates")
    op.drop_index("ix_concept_candidates_reviewed_at", table_name="concept_candidates")
    op.drop_constraint(
        "fk_concept_candidates_merged_into_concept_id_concepts",
        "concept_candidates",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_concept_candidates_promoted_concept_id_concepts",
        "concept_candidates",
        type_="foreignkey",
    )
    op.drop_column("concept_candidates", "merged_into_concept_id")
    op.drop_column("concept_candidates", "promoted_concept_id")
    op.drop_column("concept_candidates", "review_note")
    op.drop_column("concept_candidates", "reviewed_at")
    op.drop_column("concept_candidates", "reviewed_by")
