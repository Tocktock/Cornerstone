from __future__ import annotations

import json

from cornerstone.config import Settings
from cornerstone.keywords import (
    KeywordCandidate,
    KeywordLLMFilter,
    KeywordSourceChunk,
    extract_concept_candidates,
    extract_keyword_candidates,
    prepare_keyword_chunks,
)


def test_prepare_keyword_chunks_normalises_and_collects_metadata() -> None:
    payloads = [
        {
            "text": "  Title: Hello\nParagraph for testing.  ",
            "language": "en",
            "doc_id": "doc-1",
            "chunk_id": "doc-1:0",
            "source": "guide.md",
            "title": "Introduction",
            "section_path": "Introduction / Setup",
            "summary": "Overview of the setup process.",
            "token_count": 42,
            "heading_path": ["Introduction"],
            "content_type": "text/markdown",
            "ingested_at": "2024-05-01T00:00:00Z",
        },
        {
            "text": "로그인 오류 처리 절차",
            "language": "ko",
            "chunk_id": "doc-1:1",
            "section_path": "로그인",
            "token_count": 30,
        },
        {
            "text": "   ",
            "chunk_id": "doc-1:2",
        },
    ]

    result = prepare_keyword_chunks(payloads)

    assert result.total_payloads == 3
    assert result.processed_count == 2
    assert result.skipped_empty == 1
    assert result.skipped_non_text == 0

    first, second = result.chunks
    assert first.normalized_text == "title hello paragraph for testing"
    assert first.section_path == "Introduction / Setup"
    assert first.metadata["heading_path"] == ["Introduction"]
    assert second.normalized_text == "로그인 오류 처리 절차"
    assert second.language == "ko"

    assert result.unique_languages() == ["en", "ko"]
    assert result.total_tokens() == 72
    assert result.sample_sections() == ["Introduction / Setup", "로그인"]
    excerpts = result.sample_excerpts(limit=1, max_chars=40)
    assert excerpts and "Title" in excerpts[0]


def test_extract_concept_candidates_promotes_multiword_phrases() -> None:
    chunks = [
        KeywordSourceChunk(
            text="Login error handling guide for SSO integrations.",
            normalized_text="login error handling guide for sso integrations",
            doc_id="doc-1",
            chunk_id="doc-1:0",
            source="guide.md",
            title="Login Troubleshooting",
            section_path="Login / Errors",
            summary="Covers login error handling steps.",
            language="en",
            token_count=38,
        ),
        KeywordSourceChunk(
            text="SSO login error steps include token refresh and session reset.",
            normalized_text="sso login error steps include token refresh and session reset",
            doc_id="doc-2",
            chunk_id="doc-2:0",
            source="runbook.md",
            title="SSO Operations",
            section_path="SSO / Troubleshooting",
            summary="Runbook for SSO login error resolution.",
            language="en",
            token_count=34,
        ),
    ]

    result = extract_concept_candidates(chunks, max_ngram_size=2)
    phrases = [candidate.phrase for candidate in result.candidates]
    assert "login error" in phrases

    login_error = next(candidate for candidate in result.candidates if candidate.phrase == "login error")
    assert login_error.document_count == 2
    assert login_error.occurrences >= 2
    assert login_error.score > 0
    assert login_error.score_breakdown["frequency"] >= 2


def test_extract_concept_candidates_filters_stopwords() -> None:
    chunks = [
        KeywordSourceChunk(
            text="and the of the",
            normalized_text="and the of the",
            doc_id="doc-3",
            chunk_id="doc-3:0",
            language="en",
        )
    ]

    result = extract_concept_candidates(chunks)
    assert not result.candidates


def test_extract_keyword_candidates_identifies_core_terms() -> None:
    texts = [
        "Alpha Beta alpha",
        "Beta gamma",
        "alpha guidance",
    ]
    candidates = extract_keyword_candidates(texts, core_limit=2)
    assert candidates
    alpha = next(item for item in candidates if item.term.lower() == "alpha")
    beta = next(item for item in candidates if item.term.lower() == "beta")
    gamma = next(item for item in candidates if item.term.lower() == "gamma")

    assert alpha.count == 3
    assert beta.count == 2
    assert gamma.count == 1

    assert alpha.is_core is True
    assert beta.is_core is True
    assert gamma.is_core is False
    assert not alpha.generated


def test_parse_response_handles_markdown_wrapper() -> None:
    raw = """Here are the keywords:\n```json\n{\n  \"keywords\": [\n    {\n      \"term\": \"alpha\",\n      \"keep\": true\n    }\n  ]\n}\n```"""
    parsed = KeywordLLMFilter._parse_response(raw)  # type: ignore[attr-defined]  # accessing static method
    assert parsed is not None
    assert parsed["keywords"][0]["term"] == "alpha"


def test_parse_response_handles_leading_text_without_fence() -> None:
    raw = "assistant: sure! {\"keywords\": [{\"term\": \"beta\", \"keep\": true}]}"
    parsed = KeywordLLMFilter._parse_response(raw)  # type: ignore[attr-defined]
    assert parsed is not None
    assert parsed["keywords"][0]["term"] == "beta"


def _build_enabled_filter(keyword_payload: dict, monkeypatch) -> KeywordLLMFilter:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
        keyword_filter_max_results=3,
    )
    filter_instance = KeywordLLMFilter(settings)
    response = json.dumps(keyword_payload)
    monkeypatch.setattr(KeywordLLMFilter, "_invoke_backend", lambda self, _: response)
    return filter_instance


def test_filter_keywords_limits_and_normalises(monkeypatch) -> None:
    filter_instance = _build_enabled_filter(
        {
            "keywords": [
                {"term": "Alpha", "keep": True},
                {"term": "Beta"},
                "Delta",
                "epsilon",
            ]
        },
        monkeypatch,
    )
    candidates = [
        KeywordCandidate(term="Alpha", count=5, is_core=True),
        KeywordCandidate(term="Beta", count=3, is_core=True),
        KeywordCandidate(term="Gamma", count=2, is_core=True),
    ]

    results = filter_instance.filter_keywords(candidates, context_snippets=[])

    assert [item.term for item in results] == ["Alpha", "Beta", "Delta"]
    assert [item.generated for item in results] == [False, False, True]

    info = filter_instance.debug_payload()
    assert info["status"] == "filtered"
    assert info["truncated"] == ["epsilon"]
    assert info["kept"] == ["Alpha", "Beta"]
    assert info["generated"] == ["Delta"]
    assert info["selected_total"] == 3
    assert info["llm_selected_total"] == 4
    assert set(info.get("assumed_true", [])) >= {"Beta", "Delta", "epsilon"}


def test_filter_keywords_handles_items_payload(monkeypatch) -> None:
    filter_instance = _build_enabled_filter(
        {
            "items": [
                {"keyword": "Gamma", "keep": False, "reason": "Noise"},
                {"keyword": "Beta", "keep": "yes", "reason": "Core concept"},
                {"keyword": "Delta", "reason": "LLM added"},
            ]
        },
        monkeypatch,
    )
    candidates = [
        KeywordCandidate(term="Alpha", count=5, is_core=True),
        KeywordCandidate(term="Beta", count=3, is_core=True),
    ]

    results = filter_instance.filter_keywords(candidates, context_snippets=[])
    assert [item.term for item in results] == ["Beta", "Delta"]
    assert [item.generated for item in results] == [False, True]
    assert results[0].source == "candidate"
    assert results[0].reason == "Core concept"
    assert results[1].source == "generated"
    assert results[1].reason == "LLM added"

    info = filter_instance.debug_payload()
    assert info["status"] == "filtered"
    assert info["coerced_source"] == "coerced-from-items"
    assert info["rejected_total"] == 1
    assert info["rejected"][0]["term"] == "Gamma"
    assert info["generated"] == ["Delta"]
    assert info["kept"] == ["Beta"]
