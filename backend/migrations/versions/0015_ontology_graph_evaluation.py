"""ontology graph evaluation framework

Revision ID: 0015_ontology_graph_evaluation
Revises: 0014_ontology_candidate_review_workflow
Create Date: 2026-05-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_ontology_graph_evaluation"
down_revision: str | None = "0014_ontology_candidate_review_workflow"
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

    op.create_table(
        "ontology_graph_eval_tasks",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("concept_query", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("expected_trust_label", sa.String(length=64), nullable=True),
        sa.Column("expected_freshness_state", sa.String(length=32), nullable=True),
        sa.Column("required_concept_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("required_relation_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("required_evidence_fragment_ids", json_type, nullable=False, server_default="[]"),
        sa.Column("require_official_graph", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("require_evidence", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("min_evidence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("min_node_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("min_edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("require_review_provenance", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("max_pending_candidate_count", sa.Integer(), nullable=True),
        sa.Column("tags", json_type, nullable=False, server_default="[]"),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("created_at", TZ_DATETIME, nullable=False),
        sa.Column("updated_at", TZ_DATETIME, nullable=False),
        sa.Column("metadata", json_type, nullable=False, server_default="{}"),
        sa.CheckConstraint("depth >= 0 AND depth <= 1", name="ck_ontology_graph_eval_tasks_depth_v1"),
    )
    op.create_index("ix_ontology_graph_eval_tasks_created_at", "ontology_graph_eval_tasks", ["created_at"])
    op.create_index("ix_ontology_graph_eval_tasks_concept_query", "ontology_graph_eval_tasks", ["concept_query"])
    op.create_index("ix_ontology_graph_eval_tasks_expected_trust", "ontology_graph_eval_tasks", ["expected_trust_label"])

    op.create_table(
        "ontology_graph_eval_results",
        sa.Column("id", uuid_type, primary_key=True),
        sa.Column(
            "task_id",
            uuid_type,
            sa.ForeignKey("ontology_graph_eval_tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("response_id", uuid_type, nullable=False),
        sa.Column("concept_query", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False),
        sa.Column("trust_label", sa.String(length=64), nullable=False),
        sa.Column("response", json_type, nullable=False),
        sa.Column("graph_found", sa.Boolean(), nullable=False),
        sa.Column("graph_depth_respected", sa.Boolean(), nullable=False),
        sa.Column("node_requirements_met", sa.Boolean(), nullable=False),
        sa.Column("edge_requirements_met", sa.Boolean(), nullable=False),
        sa.Column("evidence_valid", sa.Boolean(), nullable=False),
        sa.Column("provenance_present", sa.Boolean(), nullable=False),
        sa.Column("trust_label_correct", sa.Boolean(), nullable=False),
        sa.Column("freshness_policy_respected", sa.Boolean(), nullable=False),
        sa.Column("official_graph_safe", sa.Boolean(), nullable=False),
        sa.Column("candidate_boundary_respected", sa.Boolean(), nullable=False),
        sa.Column("relation_integrity_valid", sa.Boolean(), nullable=False),
        sa.Column("citation_validity_rate", sa.Float(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("failure_reasons", json_type, nullable=False, server_default="[]"),
        sa.Column("evaluated_at", TZ_DATETIME, nullable=False),
        sa.Column("evaluated_by", sa.Text(), nullable=False),
        sa.CheckConstraint("depth >= 0 AND depth <= 1", name="ck_ontology_graph_eval_results_depth_v1"),
    )
    op.create_index("ix_ontology_graph_eval_results_task", "ontology_graph_eval_results", ["task_id"])
    op.create_index("ix_ontology_graph_eval_results_success", "ontology_graph_eval_results", ["success"])
    op.create_index("ix_ontology_graph_eval_results_evaluated_at", "ontology_graph_eval_results", ["evaluated_at"])
    op.create_index("ix_ontology_graph_eval_results_trust", "ontology_graph_eval_results", ["trust_label"])


def downgrade() -> None:
    op.drop_index("ix_ontology_graph_eval_results_trust", table_name="ontology_graph_eval_results")
    op.drop_index("ix_ontology_graph_eval_results_evaluated_at", table_name="ontology_graph_eval_results")
    op.drop_index("ix_ontology_graph_eval_results_success", table_name="ontology_graph_eval_results")
    op.drop_index("ix_ontology_graph_eval_results_task", table_name="ontology_graph_eval_results")
    op.drop_table("ontology_graph_eval_results")
    op.drop_index("ix_ontology_graph_eval_tasks_expected_trust", table_name="ontology_graph_eval_tasks")
    op.drop_index("ix_ontology_graph_eval_tasks_concept_query", table_name="ontology_graph_eval_tasks")
    op.drop_index("ix_ontology_graph_eval_tasks_created_at", table_name="ontology_graph_eval_tasks")
    op.drop_table("ontology_graph_eval_tasks")
