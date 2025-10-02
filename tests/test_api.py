from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from cornerstone.app import create_app
from cornerstone.config import Settings
from cornerstone.embeddings import EmbeddingBackend, EmbeddingService
from cornerstone.ingestion import DocumentIngestor, ProjectVectorStoreManager
from cornerstone.projects import ProjectStore
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
    tmpdir = Path(tempfile.mkdtemp(prefix="cornerstone-ui-"))
    settings = Settings(data_dir=str(tmpdir), default_project_name="UI Project")
    project_store = ProjectStore(tmpdir, default_project_name=settings.default_project_name)
    default_project = project_store.list_projects()[0]

    client = QdrantClient(path=":memory:")
    store_manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=embedding_service.dimension,
        distance=models.Distance.COSINE,
        collection_name_fn=lambda pid: f"ui-test-{pid}",
    )
    store = store_manager.get_store(default_project.id)
    store.ensure_collection()
    store.upsert(
        [
            VectorRecord(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={"text": "Alpha document", "project_id": default_project.id},
            ),
            VectorRecord(
                id=2,
                vector=[0.0, 1.0, 0.0],
                payload={"text": "Beta entry", "project_id": default_project.id},
            ),
        ]
    )

    ingestion = DocumentIngestor(embedding_service, store_manager, project_store)

    app = create_app(
        settings=settings,
        embedding_service=embedding_service,  # type: ignore[arg-type]
        project_store=project_store,
        store_manager=store_manager,
        ingestion_service=ingestion,
    )
    client_app = TestClient(app)
    client_app.default_project_id = default_project.id  # type: ignore[attr-defined]
    return client_app


def test_index_route_renders_form(fastapi_app: TestClient) -> None:
    response = fastapi_app.get("/")
    assert response.status_code == 200, response.text
    assert "Cornerstone Semantic Search" in response.text


def test_keywords_page_renders(fastapi_app: TestClient) -> None:
    response = fastapi_app.get("/keywords")
    assert response.status_code == 200, response.text
    assert "Keyword Explorer" in response.text


def test_search_route_returns_results(fastapi_app: TestClient) -> None:
    response = fastapi_app.post(
        "/search",
        data={"query": "Alpha", "project_id": fastapi_app.default_project_id},
    )
    assert response.status_code == 200, response.text
    assert "Alpha document" in response.text
    assert "Score:" in response.text


def test_search_route_handles_no_results(fastapi_app: TestClient) -> None:
    response = fastapi_app.post(
        "/search",
        data={"query": "Gamma", "project_id": fastapi_app.default_project_id},
    )
    assert response.status_code == 200, response.text
    assert "No results found for this project." in response.text


def test_keywords_endpoint_returns_candidates(fastapi_app: TestClient) -> None:
    response = fastapi_app.get(f"/keywords/{fastapi_app.default_project_id}/candidates")
    assert response.status_code == 200, response.text
    data = response.json()
    terms = {item["term"].lower() for item in data.get("keywords", [])}
    assert "alpha" in terms
    assert "beta" in terms
    first = data["keywords"][0]
    assert "generated" in first
    assert "reason" in first
    assert "source" in first
    filter_info = data.get("filter")
    assert filter_info is not None
    assert filter_info.get("backend") in {None, "openai", "ollama"}
    assert "status" in filter_info


def test_keyword_definition_endpoint_provides_snippets(fastapi_app: TestClient) -> None:
    response = fastapi_app.get(
        f"/keywords/{fastapi_app.default_project_id}/definition",
        params={"term": "Alpha"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["term"].lower() == "alpha"
    assert data["candidates"], "Expected candidate definitions for Alpha"
    first_candidate = data["candidates"][0]
    assert "excerpt" in first_candidate
    assert len(first_candidate["excerpt"]) <= len(first_candidate["snippet"])
    assert any("Alpha" in result["snippet"] for result in data["candidates"])
