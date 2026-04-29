from __future__ import annotations

from pathlib import Path

from cornerstone.persistence.models import (
    Base,
    GroundedContextEvalResultRow,
    GroundedContextEvalTaskRow,
)


def test_evaluation_tables_are_in_sqlalchemy_metadata() -> None:
    table_names = set(Base.metadata.tables)

    assert "grounded_context_eval_tasks" in table_names
    assert "grounded_context_eval_results" in table_names
    assert GroundedContextEvalTaskRow.__tablename__ == "grounded_context_eval_tasks"
    assert GroundedContextEvalResultRow.__tablename__ == "grounded_context_eval_results"


def test_evaluation_migration_adds_metric_tables_and_indexes() -> None:
    migration_sql = Path("migrations/versions/0011_grounded_context_evaluation.py").read_text()

    assert "grounded_context_eval_tasks" in migration_sql
    assert "grounded_context_eval_results" in migration_sql
    assert "ix_grounded_context_eval_results_success" in migration_sql
    assert "ix_grounded_context_eval_tasks_expected_trust" in migration_sql
