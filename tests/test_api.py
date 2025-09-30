from __future__ import annotations

from typing import Iterable

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from cornerstone.app import create_app
from cornerstone.config import Settings
from cornerstone.embeddings import EmbeddingBackend, EmbeddingService
from cornerstone.vector_store import QdrantVectorStore, VectorRecord


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.backend = EmbeddingBackend.HUGGINGFACE
        self.dimension = 3

    def embed(self, texts: Iterable[str]):
        return [self._vector_for(text) for text in texts]

    def embed_one(self, text: str):
        return self._vector_for(text)

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        lowered = text.lower()
        if "alpha" in lowered:
            return [1.0, 0.0, 0.0]
        if "beta" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture()
def fastapi_app() -> TestClient:
    embedding_service = FakeEmbeddingService()
    client = QdrantClient(path=":memory:")
    store = QdrantVectorStore(client, "ui-test", vector_size=embedding_service.dimension)
    store.ensure_collection()
    store.upsert(
        [
            VectorRecord(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "Alpha document"}),
            VectorRecord(id=2, vector=[0.0, 1.0, 0.0], payload={"text": "Beta entry"}),
        ]
    )

    app = create_app(
        settings=Settings(),
        embedding_service=embedding_service,  # type: ignore[arg-type]
        vector_store=store,
        ensure_collection=False,
    )
    return TestClient(app)


def test_index_route_renders_form(fastapi_app: TestClient) -> None:
    response = fastapi_app.get("/")
    assert response.status_code == 200
    assert "Cornerstone Semantic Search" in response.text


def test_search_route_returns_results(fastapi_app: TestClient) -> None:
    response = fastapi_app.post("/search", data={"query": "Alpha"})
    assert response.status_code == 200
    assert "Alpha document" in response.text
    assert "Score:" in response.text


def test_search_route_handles_no_results(fastapi_app: TestClient) -> None:
    response = fastapi_app.post("/search", data={"query": "Gamma"})
    assert response.status_code == 200
    assert "No results found" in response.text
