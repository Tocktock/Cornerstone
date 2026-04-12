from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import inspect, select, text

from cornerstone.app import create_app
from cornerstone.config import (
    Settings,
    discover_fixture_root,
    discover_notion_fixture_root,
    discover_personal_source_root,
    discover_workspace_source_root,
)
from cornerstone.domain.enums import RuntimeMode
from cornerstone.domain.models import Concept, DecisionRecord, SourceConnection
from cornerstone.services.bootstrap import initialize_database, seed_demo


def test_startup_repairs_legacy_workspace_home_schema(test_database, test_database_url: str):
    settings = Settings(
        database_url=test_database_url,
        auto_seed_demo=True,
        reset_database_on_start=False,
        fixture_root=discover_fixture_root(),
        workspace_source_root=discover_workspace_source_root(),
        personal_source_root=discover_personal_source_root(),
        notion_fixture_root=discover_notion_fixture_root(),
        fixed_now="2026-04-06T09:00:00+09:00",
        cors_origins=["http://localhost:4173"],
    )

    initialize_database(test_database.engine, reset=True)
    with test_database.session_factory() as session:
        seed_demo(session, settings)

    with test_database.engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE decision_records "
                "DROP CONSTRAINT IF EXISTS uq_decisions_context_space_public_slug"
            )
        )
        connection.execute(text("DROP INDEX IF EXISTS ix_decision_records_public_slug"))
        connection.execute(text("ALTER TABLE decision_records DROP COLUMN public_slug"))
        connection.execute(
            text("DROP INDEX IF EXISTS ix_source_connections_provider_credential_ref")
        )
        connection.execute(
            text("ALTER TABLE source_connections DROP COLUMN provider_credential_ref")
        )
        connection.execute(text("ALTER TABLE source_connections DROP COLUMN selected_scope_json"))
        connection.execute(text("ALTER TABLE source_connections DROP COLUMN sync_checkpoint_json"))
        connection.execute(
            text("ALTER TABLE source_connections DROP COLUMN next_scheduled_sync_at")
        )

    app = create_app(settings)
    with TestClient(app) as client:
        bootstrap = client.get("/api/v1/bootstrap")
        bootstrap.raise_for_status()
        assert bootstrap.json()["runtime_mode"] == "mock"
        assert bootstrap.json()["workspace_data_state"] == "demo_seeded"
        member_token = next(
            actor["token"]
            for actor in bootstrap.json()["actors"]
            if actor["display_name"] == "Member"
        )
        workspace_home = client.get(
            "/api/v1/workspace-home",
            headers={"Authorization": f"Bearer {member_token}"},
        )
        workspace_home.raise_for_status()

    with test_database.engine.connect() as connection:
        inspector = inspect(connection)
        decision_columns = {column["name"] for column in inspector.get_columns("decision_records")}
        decision_constraints = inspector.get_unique_constraints("decision_records")
        source_columns = {column["name"] for column in inspector.get_columns("source_connections")}
        source_indexes = inspector.get_indexes("source_connections")
        decision_rows = connection.execute(
            text(
                "SELECT title, public_slug "
                "FROM decision_records "
                "ORDER BY title"
            )
        ).mappings()

        assert "public_slug" in decision_columns
        assert any(
            constraint.get("column_names") == ["context_space_id", "public_slug"]
            for constraint in decision_constraints
        )
        assert all(row["public_slug"] for row in decision_rows)
        assert {
            "provider_credential_ref",
            "selected_scope_json",
            "sync_checkpoint_json",
            "next_scheduled_sync_at",
        } <= source_columns
        assert any(
            index.get("column_names") == ["provider_credential_ref"] for index in source_indexes
        )


def test_production_runtime_creates_minimal_bootstrap_without_demo_content(
    test_database, test_database_url: str
):
    settings = Settings(
        database_url=test_database_url,
        runtime_mode=RuntimeMode.PRODUCTION,
        auto_seed_demo=True,
        reset_database_on_start=True,
        fixture_root=discover_fixture_root(),
        workspace_source_root=discover_workspace_source_root(),
        personal_source_root=discover_personal_source_root(),
        notion_fixture_root=discover_notion_fixture_root(),
        fixed_now="2026-04-06T09:00:00+09:00",
        cors_origins=["http://localhost:4173"],
    )

    app = create_app(settings)
    with TestClient(app) as client:
        bootstrap = client.get("/api/v1/bootstrap")
        bootstrap.raise_for_status()
        bootstrap_payload = bootstrap.json()
        assert bootstrap_payload["runtime_mode"] == "production"
        assert bootstrap_payload["workspace_data_state"] == "awaiting_sources"
        assert bootstrap_payload["linked_source_count"] == 0
        assert bootstrap_payload["active_source_count"] == 0
        assert bootstrap_payload["degraded_source_count"] == 0

        member_token = next(
            actor["token"]
            for actor in bootstrap_payload["actors"]
            if actor["display_name"] == "Member"
        )
        workspace_home = client.get(
            "/api/v1/workspace-home",
            headers={"Authorization": f"Bearer {member_token}"},
        )
        workspace_home.raise_for_status()

    with test_database.session_factory() as session:
        assert list(session.scalars(select(Concept))) == []
        assert list(session.scalars(select(DecisionRecord))) == []
        assert list(session.scalars(select(SourceConnection))) == []
