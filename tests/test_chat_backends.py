from __future__ import annotations

from typing import Any, Dict, Iterable

import pytest

from cornerstone.chat import SupportAgentService
from cornerstone.config import Settings


class _Dummy:
    pass


class _StubResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class _StubStreamResponse:
    def __init__(self, lines: Iterable[str]) -> None:
        self._lines = list(lines)

    def __enter__(self) -> "_StubStreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self) -> Iterable[str]:
        for line in self._lines:
            yield line


def _build_service(settings: Settings) -> SupportAgentService:
    return SupportAgentService(
        settings=settings,
        embedding_service=_Dummy(),
        store_manager=_Dummy(),
        glossary=_Dummy(),
        project_store=None,
        persona_store=None,
        fts_index=None,
        reranker=None,
    )


def test_support_agent_vllm_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_post(url: str, *, json: Dict[str, Any], headers: Dict[str, str], timeout: float) -> _StubResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _StubResponse({"choices": [{"message": {"content": "Hello world"}}]})

    monkeypatch.setattr("cornerstone.chat.httpx.post", fake_post)

    settings = Settings(
        chat_backend="vllm",
        vllm_base_url="http://localhost:8000",
        vllm_model="mock-chat",
        vllm_api_key="secret",
        vllm_request_timeout=12.5,
    )
    service = _build_service(settings)

    result = service._invoke_vllm("Hi?", temperature=0.42, max_tokens=256)

    assert result == "Hello world"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["json"]["model"] == "mock-chat"
    assert captured["json"]["stream"] is False
    assert captured["json"]["temperature"] == pytest.approx(0.42)
    assert captured["json"]["max_tokens"] == 256
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["messages"][1]["content"] == "Hi?"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == pytest.approx(12.5)


def test_support_agent_vllm_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_stream(
        method: str,
        url: str,
        *,
        json: Dict[str, Any],
        headers: Dict[str, str],
        timeout: float,
    ) -> _StubStreamResponse:
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _StubStreamResponse(
            [
                'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"}}]}',
                "data: [DONE]",
            ]
        )

    monkeypatch.setattr("cornerstone.chat.httpx.stream", fake_stream)

    settings = Settings(
        chat_backend="vllm",
        vllm_base_url="http://localhost:8000",
        vllm_model="mock-chat",
        vllm_api_key="secret",
        vllm_request_timeout=8.0,
    )
    service = _build_service(settings)

    chunks = list(service._stream_vllm("Hi?", temperature=0.0, max_tokens=None))

    assert chunks == ["Hel", "lo"]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["json"]["stream"] is True
    assert captured["json"]["messages"][1]["content"] == "Hi?"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == pytest.approx(8.0)
