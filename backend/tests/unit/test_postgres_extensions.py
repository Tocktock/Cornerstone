from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from cornerstone.persistence.extensions import (
    REQUIRED_POSTGRES_EXTENSIONS,
    DatabaseExtensionError,
    extension_create_statements,
    parse_required_extensions,
    verify_required_extensions,
)


def test_required_postgres_extensions_are_explicit() -> None:
    assert REQUIRED_POSTGRES_EXTENSIONS == ("pgcrypto", "citext", "vector")
    statements = extension_create_statements()
    assert 'CREATE EXTENSION IF NOT EXISTS "pgcrypto";' in statements
    assert 'CREATE EXTENSION IF NOT EXISTS "citext";' in statements
    assert 'CREATE EXTENSION IF NOT EXISTS "vector";' in statements


def test_parse_required_extensions_deduplicates_and_preserves_order() -> None:
    assert parse_required_extensions("pgcrypto, vector, pgcrypto, citext") == (
        "pgcrypto",
        "vector",
        "citext",
    )


def test_runtime_extension_verification_rejects_non_postgres_engine() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with pytest.raises(DatabaseExtensionError, match="PostgreSQL persistence requires"):
        verify_required_extensions(engine)


def test_migration_creates_required_extensions_and_vector_table() -> None:
    migration = Path("migrations/versions/0001_postgres_persistence.py").read_text()
    for extension in REQUIRED_POSTGRES_EXTENSIONS:
        assert f'CREATE EXTENSION IF NOT EXISTS "{extension}"' in migration
    assert "ALTER TABLE IF EXISTS alembic_version" in migration
    assert "VARCHAR(128)" in migration
    assert "CREATE TABLE evidence_embeddings" in migration
    assert "VECTOR(1536)" in migration
    assert "USING hnsw" in migration


def test_source_state_discovery_migration_adds_runtime_state_and_snapshots() -> None:
    migration = Path("migrations/versions/0003_source_state_discovery.py").read_text()

    assert 'op.add_column("data_sources", sa.Column("auth_status"' in migration
    assert 'op.add_column("data_sources", sa.Column("connection_status"' in migration
    assert 'op.add_column("data_sources", sa.Column("sync_status"' in migration
    assert 'op.add_column("data_sources", sa.Column("next_action"' in migration
    assert '"provider_object_snapshots"' in migration
    assert "uq_provider_object_snapshots_source_external" in migration
    assert "ix_provider_object_snapshots_selected" in migration
    assert "ix_data_sources_runtime_state" in migration


class _FakeDialect:
    name = "postgresql"


class _FakeScalarResult:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    def all(self) -> list[str]:
        return self._values


class _FakeExecuteResult:
    def __init__(self, values: list[str]) -> None:
        self._values = values

    def scalars(self) -> _FakeScalarResult:
        return _FakeScalarResult(self._values)


class _FakeConnection:
    def __init__(self, values: list[str]) -> None:
        self._values = values
        self.statements: list[object] = []

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def execute(self, statement: object) -> _FakeExecuteResult:
        self.statements.append(statement)
        return _FakeExecuteResult(self._values)


class _FakePostgresEngine:
    dialect = _FakeDialect()

    def __init__(self, installed_extensions: list[str]) -> None:
        self.installed_extensions = installed_extensions
        self.connection = _FakeConnection(installed_extensions)

    def connect(self) -> _FakeConnection:
        return self.connection


def test_runtime_extension_verification_accepts_installed_postgres_extensions() -> None:
    engine = _FakePostgresEngine(["pgcrypto", "citext", "vector", "plpgsql"])

    installed = verify_required_extensions(engine)  # type: ignore[arg-type]

    assert installed == {"pgcrypto", "citext", "vector", "plpgsql"}
    assert engine.connection.statements


def test_runtime_extension_verification_reports_missing_postgres_extensions() -> None:
    engine = _FakePostgresEngine(["pgcrypto"])

    with pytest.raises(DatabaseExtensionError, match="Missing required PostgreSQL extensions"):
        verify_required_extensions(engine)  # type: ignore[arg-type]


def test_generic_ingestion_migration_adds_provider_neutral_artifact_columns() -> None:
    migration = Path("migrations/versions/0004_generic_ingestion_contract.py").read_text()

    assert '"artifacts"' in migration
    assert 'sa.Column("source_object_type"' in migration
    assert 'sa.Column("provider_metadata"' in migration
    assert '"provider_object_snapshots"' in migration
    assert "ix_artifacts_source_object_type" in migration


def test_scheduled_sync_migration_adds_schedule_table_and_indexes() -> None:
    migration = Path("migrations/versions/0006_scheduled_sync_runtime.py").read_text()

    assert '"sync_schedules"' in migration
    assert 'sa.Column("interval_minutes"' in migration
    assert 'sa.Column("next_run_at"' in migration
    assert 'sa.Column("last_enqueued_sync_job_id"' in migration
    assert "uq_sync_schedules_datasource" in migration
    assert "ix_sync_schedules_status_next_run" in migration


def test_provider_object_ingestion_support_migration_adds_selection_guard_columns() -> None:
    migration = Path("migrations/versions/0007_provider_object_ingestion_support.py").read_text()

    assert 'op.add_column(\n        "provider_object_snapshots"' in migration
    assert 'sa.Column("ingestion_supported"' in migration
    assert 'sa.Column("ingestion_unsupported_reason"' in migration
    assert "ix_provider_object_snapshots_ingestion" in migration


def test_worker_lease_migration_adds_claim_and_enqueue_primitives() -> None:
    migration = Path("migrations/versions/0008_worker_lease_primitives.py").read_text()

    assert 'sa.Column("lease_owner"' in migration
    assert 'sa.Column("lease_acquired_at"' in migration
    assert 'sa.Column("lease_expires_at"' in migration
    assert 'sa.Column("schedule_id"' in migration
    assert 'sa.Column("enqueue_key"' in migration
    assert "ix_sync_jobs_lease_expiry" in migration
    assert "ix_sync_jobs_enqueue_key" in migration
    assert "fk_sync_jobs_schedule_id_sync_schedules" in migration
