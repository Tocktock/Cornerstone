from __future__ import annotations

from dataclasses import dataclass
import threading
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


class _StubHttpxClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.requests: list[dict[str, str]] = []

    def post(self, url: str, json: dict[str, str]) -> _StubHttpxResponse:
        self.requests.append(json)
        length = float(len(json.get("prompt", "")))
        return _StubHttpxResponse([length, 1.0, 0.0])

    def close(self) -> None:  # noqa: D401
        return None


class _StubHttpxModule:
    def __init__(self) -> None:
        self.created: list[_StubHttpxClient] = []

    def Client(self, *args: Any, **kwargs: Any) -> _StubHttpxClient:  # noqa: N802
        client = _StubHttpxClient(*args, **kwargs)
        self.created.append(client)
        return client


class _StubVLLMResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"data": [{"embedding": vector} for vector in self._vectors]}


class _StubVLLMClient:
    def __init__(self, base_url: str, timeout: float, headers: dict[str, str] | None = None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers or {}
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def post(self, url: str, json: dict[str, Any]) -> _StubVLLMResponse:
        with self._lock:
            self.requests.append({"url": url, "json": json})
        inputs = json.get("input", [])
        vectors = [[float(len(text)), 0.0, 0.0] for text in inputs] or [[0.0, 0.0, 0.0]]
        return _StubVLLMResponse(vectors)

    def close(self) -> None:
        return None


class _StubVLLMModule:
    def __init__(self) -> None:
        self.created: list[_StubVLLMClient] = []

    def Client(self, *args: Any, **kwargs: Any) -> _StubVLLMClient:  # noqa: N802
        client = _StubVLLMClient(*args, **kwargs)
        self.created.append(client)
        return client


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
    stub_httpx = _StubHttpxModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(embedding_model="ollama:qwen3-embedding", ollama_embedding_concurrency=1)
    service = EmbeddingService(settings, validate=False)

    assert service.backend is EmbeddingBackend.OLLAMA
    assert service.dimension == 3

    vectors = service.embed(["hi", "team"])
    assert vectors == [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]]
    assert len(stub_httpx.created) == 1
    client = stub_httpx.created[0]
    assert client.base_url == "http://localhost:11434"
    assert len(client.requests) == 1 + 2  # dimension probe + two embedding calls


def test_ollama_backend_supports_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubHttpxModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(
        embedding_model="ollama:http://remote-host:9999/qwen3-embedding",
        ollama_embedding_concurrency=1,
    )
    service = EmbeddingService(settings, validate=False)

    vectors = service.embed(["hi"])
    assert vectors == [[2.0, 1.0, 0.0]]

    client = stub_httpx.created[0]
    assert client.base_url == "http://remote-host:9999"
    assert len(client.requests) == 2  # dimension probe + embedding


def test_ollama_backend_respects_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubHttpxModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(
        embedding_model="ollama:nomic-embed-text",
        ollama_base_url="https://ollama.company.internal",
        ollama_embedding_concurrency=1,
    )
    service = EmbeddingService(settings, validate=False)

    vectors = service.embed(["hello"])
    assert vectors == [[5.0, 1.0, 0.0]]

    client = stub_httpx.created[0]
    assert client.base_url == "https://ollama.company.internal"
    assert len(client.requests) == 2  # probe + embedding


def test_vllm_backend_calls_openai_compatible_api(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubVLLMModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(
        embedding_model="vllm:mock-embed",
        vllm_base_url="http://localhost:8000",
        vllm_api_key="secret",
        vllm_request_timeout=12.5,
        vllm_embedding_batch_wait_ms=0,
    )
    service = EmbeddingService(settings, validate=False)

    assert service.backend is EmbeddingBackend.VLLM
    assert service.dimension == 3

    vectors = service.embed(["hi", "team"])
    assert vectors == [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]

    assert len(stub_httpx.created) == 1
    client = stub_httpx.created[0]
    assert client.base_url == "http://localhost:8000"
    assert client.timeout == pytest.approx(12.5)
    assert client.headers.get("Authorization") == "Bearer secret"
    assert len(client.requests) == 2  # dimension probe + batch
    assert client.requests[0]["json"]["input"] == ["__dimension_probe__"]
    assert client.requests[1]["json"]["input"] == ["hi", "team"]
    service.close()


def test_vllm_backend_supports_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubVLLMModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(
        embedding_model="vllm:https://llm.example.com/custom-embed",
        vllm_api_key="secret",
        vllm_embedding_batch_wait_ms=0,
    )
    service = EmbeddingService(settings, validate=False)

    vectors = service.embed(["hello"])
    assert vectors == [[5.0, 0.0, 0.0]]

    client = stub_httpx.created[0]
    assert client.base_url == "https://llm.example.com"
    assert client.headers.get("Authorization") == "Bearer secret"
    assert len(client.requests) == 2
    assert client.requests[0]["json"]["input"] == ["__dimension_probe__"]
    service.close()
    assert client.requests[1]["json"]["input"] == ["hello"]


def test_vllm_backend_allows_base_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubVLLMModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(
        embedding_model="vllm:qwen3-embeddings",
        vllm_embedding_base_url="http://192.168.0.251:8001",
        vllm_api_key="secret",
        vllm_embedding_batch_wait_ms=0,
    )
    service = EmbeddingService(settings, validate=False)

    vectors = service.embed(["override"])
    assert vectors == [[8.0, 0.0, 0.0]]

    client = stub_httpx.created[0]
    assert client.base_url == "http://192.168.0.251:8001"
    assert len(client.requests) == 2
    assert client.requests[0]["json"]["input"] == ["__dimension_probe__"]
    assert client.requests[1]["json"]["input"] == ["override"]
    service.close()


def test_vllm_backend_defaults_to_chat_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubVLLMModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(
        embedding_model="vllm:mock-embed",
        vllm_base_url="https://llm.example.com/v1",
        vllm_embedding_batch_wait_ms=0,
    )
    service = EmbeddingService(settings, validate=False)

    vectors = service.embed(["ping"])
    assert vectors == [[4.0, 0.0, 0.0]]

    client = stub_httpx.created[0]
    assert client.base_url == "https://llm.example.com"
    assert len(client.requests) == 2
    assert client.requests[0]["json"]["input"] == ["__dimension_probe__"]
    service.close()


def test_vllm_backend_respects_batch_size_override(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubVLLMModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(
        embedding_model="vllm:mock-embed",
        vllm_embedding_batch_size=1,
        vllm_embedding_concurrency=2,
        vllm_embedding_batch_wait_ms=0,
    )
    service = EmbeddingService(settings, validate=False)

    vectors = service.embed(["a", "bb"])
    assert vectors == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

    client = stub_httpx.created[0]
    assert len(client.requests) == 3  # probe + two single-item batches
    batches = [req["json"]["input"] for req in client.requests[1:]]
    assert sorted(batches) == [["a"], ["bb"]]
    service.close()


def test_vllm_backend_rejects_endpoint_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_httpx = _StubVLLMModule()
    monkeypatch.setattr("cornerstone.embeddings.httpx", stub_httpx)

    settings = Settings(embedding_model="vllm:http://localhost:8000/v1/embeddings")
    with pytest.raises(ValueError, match="must include the vLLM model identifier"):
        EmbeddingService(settings, validate=False)
