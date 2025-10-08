from __future__ import annotations

import json
import tempfile
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from cornerstone.app import create_app
from cornerstone.chat import SupportAgentContext, SupportAgentResponse, SupportAgentService
from cornerstone.config import Settings
from cornerstone.glossary import Glossary
from cornerstone.ingestion import DocumentIngestor, ProjectVectorStoreManager
from cornerstone.vector_store import VectorRecord
from cornerstone.fts import FTSIndex
from cornerstone.projects import ProjectStore
from cornerstone.personas import PersonaStore


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.dimension = 3

    def embed(self, texts):  # pragma: no cover - unused
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text):
        length = float(len(text)) or 1.0
        return [length, length / 2.0, 1.0]


class DummyChatService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, project, query: str, *, conversation=None):
        self.calls.append({
            "project_id": getattr(project, "id", project),
            "query": query,
            "conversation": conversation,
        })
        return SupportAgentResponse(
            message="Here is how to resolve it.",
            sources=[{"title": "Doc", "snippet": "Step details"}],
            definitions=["SLA: agreement"],
        )

    def stream_generate(self, project, query: str, *, conversation=None):
        self.calls.append({
            "project_id": getattr(project, "id", project),
            "query": query,
            "conversation": conversation,
            "stream": True,
        })
        context = SupportAgentContext(
            prompt="",
            sources=[{"title": "Doc", "snippet": "Step details"}],
            definitions=["SLA: agreement"],
        )

        def generator():
            yield "Here is "
            yield "how to resolve it."

        return context, generator()


def build_test_app(*, use_real_chat: bool = False) -> tuple[TestClient, str]:
    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-test-"))
    settings = Settings(data_dir=str(tmpdir), default_project_name="Test Project")
    embedding = FakeEmbeddingService()
    glossary = Glossary()
    project_store = ProjectStore(tmpdir, default_project_name=settings.default_project_name)
    persona_store = PersonaStore(tmpdir)
    default_project = project_store.list_projects()[0]

    client = QdrantClient(path=":memory:")
    store_manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=embedding.dimension,
        distance=models.Distance.COSINE,
        collection_name_fn=lambda project_id: f"support-test-{project_id}",
    )
    store_manager.get_store(default_project.id)

    fts_index = FTSIndex(Path(tmpdir) / "fts.sqlite")
    ingestion = DocumentIngestor(embedding, store_manager, project_store, fts_index=fts_index)

    chat_service = (
        SupportAgentService(
            settings=settings,
            embedding_service=embedding,  # type: ignore[arg-type]
            store_manager=store_manager,
            glossary=glossary,
            persona_store=persona_store,
            fts_index=fts_index,
        )
        if use_real_chat
        else DummyChatService()
    )

    app = create_app(
        settings=settings,
        embedding_service=embedding,  # type: ignore[arg-type]
        glossary=glossary,
        project_store=project_store,
        persona_store=persona_store,
        store_manager=store_manager,
        chat_service=chat_service,
        ingestion_service=ingestion,
    )
    app.state.services.fts_index = fts_index

    return TestClient(app), default_project.id


def test_support_chat_endpoint():
    client, project_id = build_test_app()
    response = client.post(
        "/support/chat",
        json={"query": "My service is down", "projectId": project_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Here is how")
    assert data["definitions"]


def test_support_chat_stream_endpoint():
    client, project_id = build_test_app()
    with client.stream(
        "POST",
        "/support/chat/stream",
        json={"query": "Stream please", "projectId": project_id},
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    metadata = json.loads(lines[0])
    assert metadata["event"] == "metadata"
    assert metadata["definitions"]

    deltas = json.loads(lines[1])
    assert deltas["event"] == "delta"

    done = json.loads(lines[-1])
    assert done["event"] == "done"
    assert done["message"].startswith("Here is how")


def test_persona_form_updates_tuning_settings():
    client, project_id = build_test_app(use_real_chat=True)
    services = client.app.state.services
    persona_store = services.persona_store
    project_store = services.project_store

    existing_ids = {persona.id for persona in persona_store.list_personas()}
    response = client.post(
        "/personas",
        data={
            "name": "Field Ops",
            "description": "Handles complex escalations",
            "tone": "steady and pragmatic",
            "system_prompt": "Always cite explicit runbooks and provide escalation paths.",
            "glossary_top_k": "5",
            "retrieval_top_k": "7",
            "chat_temperature": "0.3",
            "chat_max_tokens": "450",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    new_persona = next(
        persona for persona in persona_store.list_personas() if persona.id not in existing_ids
    )
    assert new_persona.glossary_top_k == 5
    assert new_persona.retrieval_top_k == 7
    assert abs(new_persona.chat_temperature - 0.3) < 1e-6
    assert new_persona.chat_max_tokens == 450

    response = client.post(
        "/knowledge/persona",
        data={
            "project_id": project_id,
            "persona_id": new_persona.id,
            "persona_glossary_top_k": "4",
            "persona_retrieval_top_k": "6",
            "persona_chat_temperature": "0.7",
            "persona_chat_max_tokens": "512",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    project = project_store.get_project(project_id)
    overrides = project.persona_overrides
    assert overrides.glossary_top_k == 4
    assert overrides.retrieval_top_k == 6
    assert abs(overrides.chat_temperature - 0.7) < 1e-6
    assert overrides.chat_max_tokens == 512

    persona_snapshot = persona_store.resolve_persona(project.persona_id, overrides)
    options = services.chat_service._persona_options(persona_snapshot)
    assert options.retrieval_top_k == 6
    assert options.glossary_top_k == 4
    assert abs(options.chat_temperature - 0.7) < 1e-6
    assert options.chat_max_tokens == 512


def test_persona_api_handles_tuning_fields():
    client, _ = build_test_app(use_real_chat=True)
    services = client.app.state.services
    persona_store = services.persona_store

    response = client.post(
        "/api/personas",
        json={
            "name": "Ops Warden",
            "description": "Keeps systems healthy",
            "tone": "direct and analytical",
            "system_prompt": "Provide immediate triage steps before escalating.",
            "tags": ["operations"],
            "glossary_top_k": 2,
            "retrieval_top_k": 5,
            "chat_temperature": 0.25,
            "chat_max_tokens": 380,
        },
    )
    assert response.status_code == 201
    persona_data = response.json()
    assert persona_data["glossary_top_k"] == 2
    assert persona_data["retrieval_top_k"] == 5
    assert abs(persona_data["chat_temperature"] - 0.25) < 1e-6
    assert persona_data["chat_max_tokens"] == 380
    persona_id = persona_data["id"]

    response = client.post(
        f"/api/personas/{persona_id}",
        json={
            "name": "Ops Warden",
            "chat_temperature": 0.55,
            "chat_max_tokens": 640,
        },
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert abs(updated_data["chat_temperature"] - 0.55) < 1e-6
    assert updated_data["chat_max_tokens"] == 640

    stored_persona = persona_store.get_persona(persona_id)
    assert stored_persona is not None
    assert abs(stored_persona.chat_temperature - 0.55) < 1e-6
    assert stored_persona.chat_max_tokens == 640
