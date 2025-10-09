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
