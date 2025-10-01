from __future__ import annotations

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from cornerstone.app import create_app
from cornerstone.chat import SupportAgentResponse
from cornerstone.config import Settings
from cornerstone.glossary import Glossary
from cornerstone.vector_store import QdrantVectorStore


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

    def generate(self, query: str, *, conversation=None):
        self.calls.append({"query": query, "conversation": conversation})
        return SupportAgentResponse(
            message="Here is how to resolve it.",
            sources=[{"title": "Doc", "snippet": "Step details"}],
            definitions=["SLA: agreement"],
        )


def build_test_app() -> TestClient:
    settings = Settings()
    embedding = FakeEmbeddingService()
    qdrant = QdrantVectorStore(QdrantClient(path=":memory:"), "support-test", vector_size=3)
    qdrant.ensure_collection()
    glossary = Glossary()
    chat = DummyChatService()
    app = create_app(
        settings=settings,
        embedding_service=embedding,  # type: ignore[arg-type]
        vector_store=qdrant,
        glossary=glossary,
        chat_service=chat,
        ensure_collection=False,
    )
    # Ensure our dummy service is used
    app.state.services.chat_service = chat
    return TestClient(app)


def test_support_chat_endpoint():
    client = build_test_app()
    response = client.post("/support/chat", json={"query": "My service is down"})
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Here is how")
    assert data["definitions"]
