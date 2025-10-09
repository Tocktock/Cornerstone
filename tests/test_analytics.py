from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cornerstone.conversations import ConversationLogStore, ConversationLogger, AnalyticsService
from cornerstone.config import Settings
from cornerstone.projects import Project
from cornerstone.personas import PersonaOverrides, PersonaSnapshot

from test_support_api import build_test_app  # reuse existing test helper


def _project_fixture() -> Project:
    return Project(
        id="proj-analytics",
        name="Analytics Project",
        description=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        persona_overrides=PersonaOverrides(),
    )


def _persona_fixture() -> PersonaSnapshot:
    return PersonaSnapshot(
        id="persona-analytics",
        name="Observer",
        tone="Neutral",
        system_prompt=None,
        avatar_url=None,
    )


def test_analytics_service_builds_summary(tmp_path: Path) -> None:
    store = ConversationLogStore(tmp_path)
    logger = ConversationLogger(store, retention_days=120)
    analytics = AnalyticsService(logger)
    project = _project_fixture()
    persona = _persona_fixture()
    now = datetime.now(timezone.utc)

    logger.log_chat(
        project=project,
        persona=persona,
        query="How do I reset my password?",
        response="Use the reset link in the portal.",
        history=["User greeting"],
        sources=[{"title": "Reset Guide", "snippet": "Visit /reset."}],
        timestamp=now - timedelta(days=2),
    )

    logger.log_chat(
        project=project,
        persona=persona,
        query="Please call me at 010-2222-3333",
        response="",
        history=[],
        sources=[],
        timestamp=now - timedelta(days=1),
    )

    summary = analytics.build_summary(project_id=project.id, days=7)
    assert summary["totals"]["conversations"] == 2
    assert summary["totals"]["resolved"] == 1
    assert summary["daily_counts"]
    queries = {item["query"] for item in summary["top_queries"]}
    assert "How do I reset my password?" in queries
    assert "Please call me at [phone]" in queries
    assert summary["unanswered"][0]["query"].startswith("Please call")


def test_analytics_summary_endpoint_returns_data() -> None:
    client, project_id = build_test_app()
    services = client.app.state.services
    logger: ConversationLogger = services.conversation_logger
    project = services.project_store.get_project(project_id)
    persona = services.persona_store.resolve_persona(project.persona_id, project.persona_overrides)

    now = datetime.now(timezone.utc)
    logger.log_chat(
        project=project,
        persona=persona,
        query="Working hours?",
        response="Support runs 24/7.",
        history=[],
        sources=[{"title": "Policy", "snippet": "24/7 coverage"}],
        timestamp=now,
    )
    logger.log_chat(
        project=project,
        persona=persona,
        query="Escalation contact",
        response="",
        history=[],
        sources=[],
        timestamp=now,
    )

    response = client.get("/api/analytics/summary", params={"project_id": project_id, "days": 30})
    assert response.status_code == 200
    payload = response.json()
    assert payload["totals"]["conversations"] == 2
    assert payload["totals"]["resolved"] == 1
    assert payload["unanswered"]

    page = client.get("/admin/analytics")
    assert page.status_code == 200
    assert "Conversation Analytics" in page.text


def test_prometheus_metrics_endpoint_available() -> None:
    pytest.importorskip("prometheus_client")
    custom_settings = Settings(
        data_dir="/tmp/unused",  # replaced in helper
        default_project_name="Metrics Project",
        observability_prometheus_enabled=True,
    )
    client, project_id = build_test_app(settings=custom_settings)
    response = client.post(
        "/support/chat",
        json={"query": "Hello metrics", "projectId": project_id},
    )
    assert response.status_code == 200

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    body = metrics_response.text
    assert "cornerstone_chat_logged_conversations_total" in body
