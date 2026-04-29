from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from cornerstone.persistence.models import (
    ArtifactRow,
    Base,
    ConceptRow,
    EvidenceFragmentRow,
    GroundedContextEvalResultRow,
    GroundedContextEvalTaskRow,
)


def test_persistence_schema_contains_core_tables_and_join_tables() -> None:
    table_names = set(Base.metadata.tables)
    assert {
        "data_sources",
        "artifacts",
        "evidence_fragments",
        "concepts",
        "concept_evidence_fragments",
        "decision_records",
        "decision_record_evidence_fragments",
        "audit_events",
    }.issubset(table_names)


def test_artifacts_have_idempotency_unique_constraint() -> None:
    constraints = {constraint.name for constraint in ArtifactRow.__table__.constraints}
    assert "uq_artifacts_source_identity_hash" in constraints


def test_concepts_have_case_insensitive_name_uniqueness_contract() -> None:
    constraints = {constraint.name for constraint in ConceptRow.__table__.constraints}
    assert "uq_concepts_name" in constraints
    ddl = str(CreateTable(ConceptRow.__table__).compile(dialect=postgresql.dialect()))
    assert "CITEXT" in ddl.upper()


def test_evidence_provenance_uses_jsonb_on_postgres() -> None:
    ddl = str(CreateTable(EvidenceFragmentRow.__table__).compile(dialect=postgresql.dialect()))
    assert "JSONB" in ddl


def test_evaluation_tables_are_persisted_and_indexed() -> None:
    table_names = set(Base.metadata.tables)
    assert {"grounded_context_eval_tasks", "grounded_context_eval_results"}.issubset(table_names)
    task_indexes = {index.name for index in GroundedContextEvalTaskRow.__table__.indexes}
    result_indexes = {index.name for index in GroundedContextEvalResultRow.__table__.indexes}
    assert "ix_grounded_context_eval_tasks_expected_trust" in task_indexes
    assert "ix_grounded_context_eval_results_success" in result_indexes
    ddl = str(CreateTable(GroundedContextEvalResultRow.__table__).compile(dialect=postgresql.dialect()))
    assert "JSONB" in ddl
