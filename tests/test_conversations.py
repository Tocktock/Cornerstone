from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from cornerstone.conversations import ConversationLogStore, ConversationLogger
from cornerstone.projects import Project
from cornerstone.personas import PersonaOverrides, PersonaSnapshot


def _project() -> Project:
    return Project(
        id="project-one",
        name="Project One",
        description=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        persona_overrides=PersonaOverrides(),
    )


def _persona() -> PersonaSnapshot:
    return PersonaSnapshot(
        id="persona-1",
        name="Helper",
        tone="Friendly",
        system_prompt=None,
        avatar_url=None,
    )


def test_conversation_logger_sanitizes_and_estimates_tokens(tmp_path: Path) -> None:
    store = ConversationLogStore(tmp_path)
    logger = ConversationLogger(store, enabled=True, retention_days=30)
    project = _project()
    persona = _persona()

    logger.log_chat(
        project=project,
        persona=persona,
        query="Please email me at person@example.com",
        response="We will follow up soon.",
        history=[{"role": "user", "content": "My phone is 010-1234-5678"}],
        sources=[{"title": "Doc", "snippet": "Reach us via support@example.com"}],
        definitions=["Definition mentions 010-5555-6666"],
        backend="unit-test",
    )

    records = logger.list_conversations(project_id=project.id)
    assert len(records) == 1
    record = records[0]
    assert record.project_id == project.id
    assert record.persona_id == persona.id
    assert "[email]" in record.query
    assert any("[phone]" in item for item in record.history)
    assert record.backend == "unit-test"
    assert record.prompt_tokens >= 1
    assert record.completion_tokens >= 1
    assert not record.unanswered


def test_conversation_retention_prunes_old_records(tmp_path: Path) -> None:
    store = ConversationLogStore(tmp_path)
    logger = ConversationLogger(store, enabled=True, retention_days=30)
    project = _project()

    older = datetime.now(timezone.utc) - timedelta(days=60)
    logger.log_chat(
        project=project,
        persona=None,
        query="Old question",
        response="Old answer",
        history=[],
        sources=[],
        timestamp=older,
    )

    logger.log_chat(
        project=project,
        persona=None,
        query="Current question",
        response="Current answer",
        history=[],
        sources=[{"title": "Doc", "snippet": "Answer"}],
    )

    records = logger.list_conversations(project_id=project.id)
    assert len(records) == 1
    assert records[0].query == "Current question"
    assert records[0].source_count == 1
