from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from cornerstone.app import create_app
from cornerstone.config import Settings
from cornerstone.keyword_jobs import KeywordRunQueue, KeywordRunJob
from cornerstone.projects import ProjectStore
from cornerstone.ingestion import ProjectVectorStoreManager


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.dimension = 3

    def embed(self, texts):  # pragma: no cover - simple helper
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text):
        length = float(len(text)) or 1.0
        return [length, length / 2.0, 1.0]


def _build_async_app(tmp_path: Path) -> tuple[TestClient, str]:
    data_dir = tmp_path / "data"
    settings = Settings(
        data_dir=str(data_dir),
        default_project_name="Test Project",
        keyword_run_sync_mode=False,
        fts_db_path=str(tmp_path / "fts.sqlite"),
    )

    project_store = ProjectStore(data_dir, default_project_name=settings.default_project_name)
    projects = project_store.list_projects()
    project_id = projects[0].id

    client = QdrantClient(path=":memory:")
    store_manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=3,
        distance=models.Distance.COSINE,
        collection_name_fn=lambda pid: f"test-keywords-{pid}",
    )
    store_manager.get_store(project_id)

    embedding = FakeEmbeddingService()

    async def executor(job: KeywordRunJob):
        await asyncio.sleep(0.01)
        return project_store.update_keyword_run(
            job.project_id,
            job.id,
            status="success",
            completed_at="2025-01-17T12:00:00+00:00",
            keywords=[{"term": "async-keyword", "count": 10}],
            stats={"token_total": 1024},
        )

    keyword_queue = KeywordRunQueue(
        project_store,
        max_queue=4,
        max_concurrency=1,
        executor=executor,
    )

    app = create_app(
        settings=settings,
        embedding_service=embedding,
        project_store=project_store,
        persona_store=None,
        store_manager=store_manager,
        keyword_run_queue=keyword_queue,
    )
    return app, project_id


def test_create_keyword_run_disabled(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    settings = Settings(data_dir=str(data_dir))
    app = create_app(settings=settings)

    with TestClient(app) as client:
        project_id = client.app.state.services.project_store.list_projects()[0].id
        response = client.post(f"/keywords/{project_id}/runs")
        assert response.status_code == 503


def test_keyword_run_async_endpoints(tmp_path: Path) -> None:
    app, project_id = _build_async_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(f"/keywords/{project_id}/runs")
        assert response.status_code == 200
        job_id = response.json()["jobId"]

        payload = None
        for _ in range(20):
            time.sleep(0.05)
            status = client.get(f"/keywords/{project_id}/runs/{job_id}")
            assert status.status_code == 200
            payload = status.json()
            if payload["status"] == "success":
                break

        assert payload is not None
        assert payload["status"] == "success"
        assert payload["run"]["keywords"][0]["term"] == "async-keyword"
        assert payload["run"].get("insightJob") is None or isinstance(payload["run"].get("insightJob"), dict)

        store = client.app.state.services.project_store
        record = store.get_latest_keyword_run(project_id)
        assert record is not None
        assert record.keywords and record.keywords[0]["term"] == "async-keyword"

        latest_payload = None
        for _ in range(10):
            latest = client.get(f"/keywords/{project_id}/runs/latest")
            if latest.status_code == 200:
                latest_payload = latest.json()
                break
            last_failed = latest
            time.sleep(0.05)

        assert latest_payload is not None, f"latest response: {last_failed.status_code} {last_failed.json()}"
        assert latest_payload["run"]["keywords"][0]["term"] == "async-keyword"
        assert latest_payload["run"].get("insightJob") is None or isinstance(latest_payload["run"].get("insightJob"), dict)
