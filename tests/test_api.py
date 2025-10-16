from __future__ import annotations

import tempfile
import json
from pathlib import Path
from typing import Iterable

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from cornerstone.app import create_app
from cornerstone.config import Settings
from cornerstone.embeddings import EmbeddingBackend, EmbeddingService
from cornerstone.ingestion import DocumentIngestor, ProjectVectorStoreManager
from cornerstone.projects import DocumentMetadata, ProjectStore
from cornerstone.vector_store import QdrantVectorStore, VectorRecord
from cornerstone.keywords import KeywordCandidate, KeywordLLMFilter


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
    assert any("alpha" in term for term in terms)
    assert any("beta" in term for term in terms)
    first = data["keywords"][0]
    assert "generated" in first
    assert "reason" in first
    assert "source" in first
    filter_info = data.get("filter")
    assert filter_info is not None
    assert filter_info.get("backend") in {None, "openai", "ollama"}
    assert "status" in filter_info
    pagination = data.get("pagination")
    assert pagination is not None
    assert pagination.get("page") == 1
    assert pagination.get("total") >= len(data.get("keywords", []))
    assert pagination.get("page_size") >= len(data.get("keywords", []))


def test_keyword_insights_roundtrip(fastapi_app: TestClient) -> None:
    project_id = fastapi_app.default_project_id  # type: ignore[attr-defined]
    payload = {
        "term": "Alpha",
        "candidates": ["Alpha document"],
        "definitions": ["Alpha: sample definition"],
        "filter": {"status": "filtered"},
    }
    response = fastapi_app.post(
        f"/keywords/{project_id}/insights",
        json=payload,
    )
    assert response.status_code == 201, response.text

    get_response = fastapi_app.get(f"/keywords/{project_id}/insights")
    assert get_response.status_code == 200
    insights_payload = get_response.json()
    insights = insights_payload.get("insights", [])
    assert len(insights) >= 1
    latest = insights[0]
    assert latest["term"].lower() == "alpha"
    assert latest.get("candidates")
    assert "updated_at" in latest


def test_keyword_insight_update_and_delete(fastapi_app: TestClient) -> None:
    project_id = fastapi_app.default_project_id  # type: ignore[attr-defined]
    payload = {
        "term": "Beta",
        "candidates": ["Beta entry"],
        "definitions": [],
        "filter": {"status": "filtered"},
    }
    create_response = fastapi_app.post(
        f"/keywords/{project_id}/insights",
        json=payload,
    )
    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    insight_id = created["id"]

    update_response = fastapi_app.patch(
        f"/keywords/{project_id}/insights/{insight_id}",
        json={"term": "Beta Updated"},
    )
    assert update_response.status_code == 200, update_response.text
    updated = update_response.json()
    assert updated["term"] == "Beta Updated"
    assert updated["id"] == insight_id

    delete_response = fastapi_app.delete(
        f"/keywords/{project_id}/insights/{insight_id}"
    )
    assert delete_response.status_code == 204

    refreshed = fastapi_app.get(f"/keywords/{project_id}/insights").json()
    terms = [item["term"] for item in refreshed.get("insights", [])]
    assert "Beta Updated" not in terms


def test_stage7_summary_skips_when_disabled(fastapi_app: TestClient, monkeypatch) -> None:
    project_id = fastapi_app.default_project_id  # type: ignore[attr-defined]
    services = fastapi_app.app.state.services
    services.settings.keyword_stage7_summary_max_insights = 0

    from cornerstone.keywords import KeywordLLMFilter

    def _raise(*_, **__):  # pragma: no cover - defensive guard
        raise AssertionError("summarize_keywords should not be called when disabled")

    monkeypatch.setattr(KeywordLLMFilter, "summarize_keywords", _raise)

    response = fastapi_app.get(f"/keywords/{project_id}/candidates")
    assert response.status_code == 200
    data = response.json()
    stage7 = (data.get("filter") or {}).get("stage7", {})
    assert stage7.get("reason") == "disabled"


@pytest.mark.asyncio
async def test_keyword_insight_job_status_endpoint(fastapi_app: TestClient, monkeypatch) -> None:
    project_id = fastapi_app.default_project_id  # type: ignore[attr-defined]
    services = fastapi_app.app.state.services
    original_backend = services.settings.chat_backend
    original_url = services.settings.ollama_base_url
    original_model = services.settings.ollama_model
    services.settings.chat_backend = "ollama"
    services.settings.ollama_base_url = "http://localhost:11434"
    services.settings.ollama_model = "mock-keywords"

    payload = {
        "insights": [
            {
                "title": "Queued insight",
                "summary": "Mock summary ready.",
                "keywords": ["alpha"],
            }
        ]
    }
    monkeypatch.setattr(
        KeywordLLMFilter,
        "_invoke_backend",
        lambda self, prompt: json.dumps(payload),
    )

    try:
        queue = services.insight_queue
        keywords = [KeywordCandidate(term="alpha", count=2, is_core=True)]
        job = await queue.enqueue(
            project_id=project_id,
            settings=services.settings,
            keywords=keywords,
            max_insights=1,
            max_concepts=1,
            context_snippets=[],
        )
        await job.wait(timeout=2.0)

        response = fastapi_app.get(f"/keywords/{project_id}/insight-jobs/{job.id}")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["status"] == "success"
        assert data["insights"] and data["insights"][0]["title"] == "Queued insight"

        missing = fastapi_app.get(f"/keywords/{project_id}/insight-jobs/missing")
        assert missing.status_code == 404
    finally:
        services.settings.chat_backend = original_backend
        services.settings.ollama_base_url = original_url
        services.settings.ollama_model = original_model

def test_keywords_endpoint_respects_pagination(fastapi_app: TestClient) -> None:
    first_page = fastapi_app.get(
        f"/keywords/{fastapi_app.default_project_id}/candidates",
        params={"page_size": 1},
    )
    assert first_page.status_code == 200, first_page.text
    first_data = first_page.json()
    assert len(first_data.get("keywords", [])) == 1
    pagination = first_data.get("pagination")
    assert pagination is not None
    assert pagination.get("page_size") == 1
    assert pagination.get("page") == 1
    assert pagination.get("total") >= 2
    assert pagination.get("has_next") is True

    second_page = fastapi_app.get(
        f"/keywords/{fastapi_app.default_project_id}/candidates",
        params={"page_size": 1, "page": 2},
    )
    assert second_page.status_code == 200, second_page.text
    second_data = second_page.json()
    assert second_data.get("pagination", {}).get("page") == 2
    assert second_data.get("pagination", {}).get("has_prev") is True
    assert len(second_data.get("keywords", [])) == 1

    first_term = first_data["keywords"][0]["term"]
    second_term = second_data["keywords"][0]["term"]
    assert first_term != second_term


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


def test_knowledge_dashboard_paginates_documents(fastapi_app: TestClient) -> None:
    client = fastapi_app
    services = client.app.state.services
    project_store: ProjectStore = services.project_store
    project = project_store.get_project(client.default_project_id)  # type: ignore[attr-defined]
    assert project is not None

    try:
        for idx in range(30):
            metadata = DocumentMetadata(
                id=f"doc-{idx}",
                filename=f"file-{idx:02d}.txt",
                chunk_count=1,
                created_at=DocumentIngestor._now(),
                size_bytes=128,
                title=f"Document {idx}",
                content_type="text/plain",
            )
            project_store.record_document(project.id, metadata)

        response = client.get(f"/knowledge?project_id={project.id}")
        assert response.status_code == 200
        text = response.text
        assert "file-24.txt" in text
        assert "file-29.txt" not in text
        assert "Page 1 of 2" in text

        response = client.get(f"/knowledge?project_id={project.id}&page=2")
        assert response.status_code == 200
        text = response.text
        assert "file-29.txt" in text
        assert "file-00.txt" not in text
        assert "Page 2 of 2" in text
    finally:
        project_store.clear_documents(project.id)
