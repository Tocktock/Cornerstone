from __future__ import annotations

import json
from pathlib import Path

import pytest

from cornerstone.config import Settings
from cornerstone.glossary import GlossaryEntry
from cornerstone.query_hints import QueryHintGenerator, merge_hint_sources


def test_query_hint_generator_merges_batches(monkeypatch):
    settings = Settings()

    responses = [
        json.dumps({"business": ["사업", "비즈니스"]}),
        json.dumps({"shipper": ["화주", "고객사"]}),
    ]

    def fake_llm(prompt: str) -> str:
        return responses.pop(0)

    generator = QueryHintGenerator(settings, llm_call=fake_llm, max_terms_per_prompt=1)

    entries = [
        GlossaryEntry(term="Business", definition="Core operations"),
        GlossaryEntry(term="화주", definition="Shipper", synonyms=["shipper"]),
    ]

    report = generator.generate(entries)
    assert report.prompts_sent == 2
    assert report.hints["business"] == ["사업", "비즈니스"]
    assert report.hints["shipper"] == ["화주", "고객사"]


def test_query_hint_generator_requires_backend():
    settings = Settings(chat_backend="unsupported")
    generator = QueryHintGenerator(settings)
    with pytest.raises(RuntimeError):
        generator.generate([GlossaryEntry(term="Test", definition="Value")])


def test_merge_hint_sources_deduplicates():
    merged = merge_hint_sources(
        {"business": ["사업", "비즈니스"]},
        {"Business": ["물류", "사업"]},
    )
    assert merged["business"] == ["사업", "비즈니스", "물류"]


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
    ],
)
def test_query_hint_generator_vllm_backend(
    monkeypatch: pytest.MonkeyPatch, base_url: str, expected_url: str
) -> None:
    class _StubResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _StubResponse(
            {
                "choices": [
                    {"message": {"content": '{"alpha": ["hint-a", "hint-b"]}'}}
                ]
            }
        )

    monkeypatch.setattr("cornerstone.query_hints.httpx.post", fake_post)

    settings = Settings(
        chat_backend="vllm",
        vllm_base_url=base_url,
        vllm_model="mock-hints",
        vllm_api_key="secret",
    )
    generator = QueryHintGenerator(settings, max_terms_per_prompt=1)
    entries = [GlossaryEntry(term="Alpha", definition="First")]

    report = generator.generate(entries)

    assert report.backend == "vllm"
    assert report.prompts_sent == 1
    assert report.hints["alpha"] == ["hint-a", "hint-b"]
    assert captured["url"] == expected_url
    assert captured["json"]["model"] == "mock-hints"
    assert captured["json"]["stream"] is False
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["max_tokens"] == 600
    assert captured["headers"]["Authorization"] == "Bearer secret"
