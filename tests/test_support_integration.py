from __future__ import annotations

import os

import tempfile

import pytest
from pathlib import Path

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from cornerstone import Settings, VectorRecord
from cornerstone.app import create_app
from cornerstone.chat import SupportAgentService
from cornerstone.embeddings import EmbeddingService
from cornerstone.glossary import Glossary, GlossaryEntry
from cornerstone.ingestion import DocumentIngestor, ProjectVectorStoreManager
from cornerstone.projects import ProjectStore


@pytest.mark.integration
def test_support_chat_with_openai_backend() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY env var required for integration test")

    settings = Settings(
        openai_api_key=api_key,
        chat_backend="openai",
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
    )

    tmpdir = tempfile.mkdtemp(prefix="cornerstone-integration-")
    project_store = ProjectStore(Path(tmpdir), default_project_name=settings.default_project_name)
    project_id = project_store.list_projects()[0].id

    embedding = EmbeddingService(settings, validate=False)
    client = QdrantClient(path=":memory:")
    store_manager = ProjectVectorStoreManager(
        client_factory=lambda: client,
        vector_size=embedding.dimension,
        distance=models.Distance.COSINE,
        collection_name_fn=lambda pid: settings.project_collection_name(pid),
    )
    store = store_manager.get_store(project_id)

    seed_text = "To reset your password, navigate to account settings and follow the reset instructions."
    store.upsert(
        [
            VectorRecord(
                id="doc-1",
                vector=embedding.embed_one(seed_text),
                payload={"title": "Password Reset", "text": seed_text},
            )
        ]
    )

    glossary = Glossary([GlossaryEntry(term="SLA", definition="Service level agreement defining response times.")])
    chat_service = SupportAgentService(
        settings=settings,
        embedding_service=embedding,
        store_manager=store_manager,
        glossary=glossary,
        retrieval_top_k=1,
    )

    ingestion = DocumentIngestor(embedding, store_manager, project_store)

    app = create_app(
        settings=settings,
        embedding_service=embedding,
        glossary=glossary,
        chat_service=chat_service,
        project_store=project_store,
        store_manager=store_manager,
        ingestion_service=ingestion,
    )

    client = TestClient(app)
    response = client.post(
        "/support/chat",
        json={"query": "I forgot my password, what should I do?", "history": [], "projectId": project_id},
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("message")
    assert isinstance(data.get("sources"), list)
