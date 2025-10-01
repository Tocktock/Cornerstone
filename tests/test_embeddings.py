from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from cornerstone.embeddings import EmbeddingBackend, EmbeddingService
from cornerstone.config import Settings


@dataclass
class _StubModel:
    name: str
    dimension: int = 3

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> list[list[float]]:  # noqa: ARG002
        return [[float(len(text)), 0.0, 0.0] for text in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class _StubOpenAIEmbeddings:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401, ANN401
        self.recorded_args = args
        self.recorded_kwargs = kwargs

    def create(self, model: str, input: list[str]) -> Any:  # noqa: ANN401
        return type(
            "Response",
            (),
            {
                "data": [
                    type("Item", (), {"embedding": [float(len(text)), 1.0, 0.0]})
                    for text in input
                ]
            },
        )()


class _StubOpenAIModels:
    def __init__(self) -> None:
        self.retrieved: list[str] = []

    def retrieve(self, name: str) -> None:
        self.retrieved.append(name)


class _StubOpenAIClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.embeddings = _StubOpenAIEmbeddings()
        self.models = _StubOpenAIModels()


class _StubHttpxResponse:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def raise_for_status(self) -> None:  # noqa: D401
        return None

    def json(self) -> dict[str, list[float]]:
        return {"embedding": self._vector}


class _StubHttpx:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str], float]] = []

    def post(self, url: str, json: dict[str, str], timeout: float) -> _StubHttpxResponse:
        self.requests.append((url, json, timeout))
        length = float(len(json.get("prompt", "")))
        return _StubHttpxResponse([length, 1.0, 0.0])


def test_huggingface_backend_uses_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cornerstone.embeddings.SentenceTransformer", lambda name: _StubModel(name=name)
    )
    settings = Settings()
    service = EmbeddingService(settings)

    assert service.backend is EmbeddingBackend.HUGGINGFACE
    assert service.dimension == 3

    vectors = service.embed(["hello"])
    assert vectors == [[5.0, 0.0, 0.0]]


def test_openai_backend_calls_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cornerstone.embeddings.SentenceTransformer", _StubModel)
    monkeypatch.setattr("cornerstone.embeddings.OpenAI", _StubOpenAIClient)

    settings = Settings(embedding_model="text-embedding-3-large", openai_api_key="token")
    service = EmbeddingService(settings)

    assert service.backend is EmbeddingBackend.OPENAI
    assert service.dimension == 3072

    vectors = service.embed(["hi"])
    assert vectors == [[2.0, 1.0, 0.0]]
    assert service._openai_client.models.retrieved == ["text-embedding-3-large"]


def test_openai_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cornerstone.embeddings.OpenAI", _StubOpenAIClient)

    settings = Settings(embedding_model="text-embedding-3-large", openai_api_key=None)
    with pytest.raises(ValueError):
        EmbeddingService(settings)


def test_ollama_backend_calls_local_api(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubHttpx()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(embedding_model="ollama:qwen3-embedding")
    service = EmbeddingService(settings, validate=False)

    assert service.backend is EmbeddingBackend.OLLAMA
    assert service.dimension == 3

    vectors = service.embed(["hi", "team"])
    assert vectors == [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]]
    assert len(stub_httpx.requests) == 1 + 2  # dimension probe + two embedding calls
