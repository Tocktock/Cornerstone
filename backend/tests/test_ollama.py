from __future__ import annotations

import pytest

from cornerstone.config import Settings
from cornerstone.domain.schemas import EvidenceRead
from cornerstone.services.ollama import OllamaClient, OllamaError


def test_ollama_client_uses_distinct_embedding_and_chat_models(monkeypatch):
    settings = Settings(
        ollama_enabled=True,
        ollama_chat_model="light-model",
        ollama_embedding_model="embedding-model",
    )
    client = OllamaClient(settings)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_post_json(path: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((path, payload))
        if path == "/api/embed":
            return {"embeddings": [[0.1, 0.2, 0.3]]}
        return {"response": "source-backed answer"}

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    assert client.embed_texts(["cornerstone"]) == [[0.1, 0.2, 0.3]]
    summary = client.generate_answer_summary(
        query="What does Cornerstone do?",
        concepts=["Cornerstone"],
        relations=[],
        decisions=[],
        evidence=[
            EvidenceRead(
                id="evidence-1",
                selector="paragraph:1",
                excerpt="Cornerstone is the shared organizational context layer for humans and AI.",
                normalized_claim=(
                    "Cornerstone is the shared organizational context layer for humans and AI."
                ),
                verification_status="VERIFIED",
                artifact_id="artifact-1",
                artifact_title="Cornerstone overview",
                artifact_url="file:///tmp/cornerstone-overview.md",
            )
        ],
    )

    assert summary == "source-backed answer"
    assert calls[0][0] == "/api/embed"
    assert calls[0][1]["model"] == "embedding-model"
    assert calls[1][0] == "/api/generate"
    assert calls[1][1]["model"] == "light-model"
    assert calls[1][1]["think"] is False


def test_ollama_timeout_is_wrapped(monkeypatch):
    settings = Settings(ollama_enabled=True)
    client = OllamaClient(settings)

    def fake_urlopen(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("cornerstone.services.ollama.urlopen", fake_urlopen)

    with pytest.raises(OllamaError, match="timed out"):
        client._post_json("/api/generate", {"model": "qwen3:0.6b"})
