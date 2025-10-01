from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from cornerstone import Settings, VectorRecord
from cornerstone.app import create_app
from cornerstone.chat import SupportAgentService
from cornerstone.embeddings import EmbeddingService
from cornerstone.glossary import Glossary, GlossaryEntry
from cornerstone.vector_store import QdrantVectorStore


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

    embedding = EmbeddingService(settings, validate=False)
    vector_store = QdrantVectorStore(QdrantClient(path=":memory:"), "support-int", vector_size=embedding.dimension)
    vector_store.ensure_collection(force_recreate=True)

    seed_text = "To reset your password, navigate to account settings and follow the reset instructions."
    vector_store.upsert(
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
        vector_store=vector_store,
        glossary=glossary,
        retrieval_top_k=1,
    )

    app = create_app(
        settings=settings,
        embedding_service=embedding,
        vector_store=vector_store,
        glossary=glossary,
        chat_service=chat_service,
        ensure_collection=False,
    )

    client = TestClient(app)
    response = client.post(
        "/support/chat",
        json={"query": "I forgot my password, what should I do?", "history": []},
    )

    assert response.status_code == 200
    data = response.json()
    assert data.get("message")
    assert isinstance(data.get("sources"), list)
