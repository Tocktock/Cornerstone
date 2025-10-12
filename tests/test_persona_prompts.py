from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cornerstone.chat import SupportAgentService
from cornerstone.config import Settings
from cornerstone.glossary import Glossary
from cornerstone.personas import PersonaOverrides, PersonaStore
from cornerstone.projects import Project


class FakeEmbedding:
    dimension = 3

    def embed_one(self, text: str) -> list[float]:  # pragma: no cover - trivial stub
        return [0.0, 0.0, 0.0]


class FakeStore:
    def search(self, vector, limit=3):  # pragma: no cover - trivial stub
        return []


class FakeStoreManager:
    def __init__(self) -> None:
        self._store = FakeStore()

    def get_store(self, project_id: str):  # pragma: no cover - trivial stub
        return self._store


def _make_project(project_id: str, *, persona_id: str | None = None, overrides: PersonaOverrides | None = None) -> Project:
    return Project(
        id=project_id,
        name="Test Project",
        description=None,
        created_at=datetime.now().isoformat(),
        persona_id=persona_id,
        persona_overrides=overrides or PersonaOverrides(),
    )


def _build_service(persona_store: PersonaStore) -> SupportAgentService:
    return SupportAgentService(
        settings=Settings(),
        embedding_service=FakeEmbedding(),
        store_manager=FakeStoreManager(),
        glossary=Glossary(),
        persona_store=persona_store,
    )


def test_persona_catalog_instructions(tmp_path: Path) -> None:
    persona_store = PersonaStore(tmp_path)
    persona = persona_store.create_persona(
        name="Chatty Analyst",
        description="Helps with upbeat diagnostics",
        tone="energetic and encouraging",
        system_prompt="Always celebrate small wins and cite relevant docs.",
    )
    project = _make_project("analytics", persona_id=persona.id)

    service = _build_service(persona_store)
    persona_snapshot = persona_store.resolve_persona(project.persona_id, project.persona_overrides)
    context, _ = service._build_context(project, persona_snapshot, "How do I reset my API key?", [])

    assert "You are Chatty Analyst" in context.prompt
    assert "energetic and encouraging" in context.prompt
    assert "Always celebrate small wins" in context.prompt


def test_project_overrides_take_precedence(tmp_path: Path) -> None:
    persona_store = PersonaStore(tmp_path)
    persona = persona_store.create_persona(
        name="Measured Guide",
        tone="calm and neutral",
        system_prompt="Respond with numbered steps.",
    )
    overrides = PersonaOverrides(tone="playful and concise")
    project = _make_project("custom", persona_id=persona.id, overrides=overrides)

    service = _build_service(persona_store)
    persona_snapshot = persona_store.resolve_persona(project.persona_id, project.persona_overrides)
    context, _ = service._build_context(project, persona_snapshot, "Diagnose", [])

    assert "playful and concise" in context.prompt
    assert "calm and neutral" not in context.prompt
