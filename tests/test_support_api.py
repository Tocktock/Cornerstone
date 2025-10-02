from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from cornerstone.app import create_app
from cornerstone.chat import SupportAgentContext, SupportAgentResponse
from cornerstone.config import Settings
from cornerstone.glossary import Glossary
from cornerstone.ingestion import DocumentIngestor, ProjectVectorStoreManager
from cornerstone.projects import ProjectStore


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.dimension = 3

    def embed(self, texts):  # pragma: no cover - unused
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_one(self, text):
        return [1.0, 0.0, 0.0]


class DummyChatService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, project_id: str, query: str, *, conversation=None):
        self.calls.append({"project_id": project_id, "query": query, "conversation": conversation})
        return SupportAgentResponse(
            message="Here is how to resolve it.",
            sources=[{"title": "Doc", "snippet": "Step details"}],
            definitions=["SLA: agreement"],
        )

    def stream_generate(self, project_id: str, query: str, *, conversation=None):
        self.calls.append({
            "project_id": project_id,
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


def build_test_app() -> tuple[TestClient, str]:
    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-test-"))
    settings = Settings(data_dir=str(tmpdir), default_project_name="Test Project")
    embedding = FakeEmbeddingService()
    glossary = Glossary()
    project_store = ProjectStore(tmpdir, default_project_name=settings.default_project_name)
    default_project = project_store.list_projects()[0]

    client = QdrantClient(path=":memory:")
    store_manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=embedding.dimension,
        distance=models.Distance.COSINE,
        collection_name_fn=lambda project_id: f"support-test-{project_id}",
    )
    store_manager.get_store(default_project.id)

    chat = DummyChatService()
    ingestion = DocumentIngestor(embedding, store_manager, project_store)

    app = create_app(
        settings=settings,
        embedding_service=embedding,  # type: ignore[arg-type]
        glossary=glossary,
        project_store=project_store,
        store_manager=store_manager,
        chat_service=chat,
        ingestion_service=ingestion,
    )

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
