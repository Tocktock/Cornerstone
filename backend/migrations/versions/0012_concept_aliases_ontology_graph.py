"""concept aliases for ontology graph search

Revision ID: 0012_concept_aliases_ontology_graph
Revises: 0011_grounded_context_evaluation
Create Date: 2026-04-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_concept_aliases_ontology_graph"
down_revision: str | None = "0011_grounded_context_evaluation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TZ_DATETIME = sa.DateTime(timezone=True)


def _uuid_type() -> sa.types.TypeEngine[object]:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.UUID(as_uuid=False)
    return sa.CHAR(36)


def _citext_type() -> sa.types.TypeEngine[object]:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects import postgresql

        return postgresql.CITEXT()
    return sa.String(length=255)


def upgrade() -> None:
    uuid_type = _uuid_type()
    citext_type = _citext_type()

    op.create_table(
        "concept_aliases",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("concept_id", uuid_type, sa.ForeignKey("concepts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("alias", citext_type, nullable=False),
        sa.Column("normalized_alias", citext_type, nullable=False),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.UniqueConstraint("normalized_alias", name="uq_concept_aliases_normalized_alias"),
        sa.UniqueConstraint("concept_id", "normalized_alias", name="uq_concept_aliases_concept_normalized_alias"),
    )
    op.create_index("ix_concept_aliases_concept_id", "concept_aliases", ["concept_id"])
    op.create_index("ix_concept_aliases_normalized_alias", "concept_aliases", ["normalized_alias"])


def downgrade() -> None:
    op.drop_index("ix_concept_aliases_normalized_alias", table_name="concept_aliases")
    op.drop_index("ix_concept_aliases_concept_id", table_name="concept_aliases")
    op.drop_table("concept_aliases")
