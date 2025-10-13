from __future__ import annotations

import json

from cornerstone.config import Settings
from cornerstone.keywords import (
    ConceptCandidate,
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


def test_refine_concepts_applies_llm(monkeypatch) -> None:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
        keyword_filter_max_results=5,
    )
    filter_instance = KeywordLLMFilter(settings)

    concepts = [
        ConceptCandidate(
            phrase="login error",
            score=9.0,
            occurrences=4,
            document_count=2,
            chunk_count=3,
            average_occurrence_per_chunk=1.33,
            word_count=2,
            languages=["en"],
            sections=["Login / Errors"],
            sources=["guide.md"],
            sample_snippet="Login error handling steps.",
            score_breakdown={"frequency": 4.0},
        ),
        ConceptCandidate(
            phrase="system",
            score=4.0,
            occurrences=5,
            document_count=3,
            chunk_count=4,
            average_occurrence_per_chunk=1.25,
            word_count=1,
            languages=["en"],
            sections=["General"],
            sources=["reference.md"],
            sample_snippet="General system notes.",
            score_breakdown={"frequency": 5.0},
        ),
    ]

    payload = {
        "concepts": [
            {
                "phrase": "login error",
                "keep": True,
                "label": "login authentication error",
                "importance": 15,
                "reason": "Critical recurring issue",
            },
            {
                "phrase": "system",
                "keep": False,
                "reason": "Too generic",
            },
            {
                "phrase": "SSO session timeout",
                "keep": True,
                "generated": True,
                "importance": 12,
                "reason": "Appears in context snippets",
            },
        ]
    }

    monkeypatch.setattr(KeywordLLMFilter, "_invoke_backend", lambda self, _: json.dumps(payload))

    refined = filter_instance.refine_concepts(concepts, ["SSO login error occurs after timeout"])

    phrases = [candidate.phrase for candidate in refined]
    assert "login authentication error" in phrases
    assert "system" not in phrases
    generated = next(candidate for candidate in refined if candidate.generated)
    assert generated.phrase == "SSO session timeout"

    debug = filter_instance.concept_debug_payload()
    assert debug.get("status") == "refined"
    assert "login authentication error" in debug.get("kept", [])


def test_refine_concepts_openai_backend(monkeypatch) -> None:
    from cornerstone import keywords as ck

    dummy_output = json.dumps(
        {
            "concepts": [
                {
                    "phrase": "login error",
                    "keep": True,
                    "label": "login authentication failure",
                    "importance": 18,
                    "reason": "Dominant incident pattern",
                },
                {
                    "phrase": "system",
                    "keep": False,
                },
                {
                    "phrase": "SSO handshake timeout",
                    "keep": True,
                    "generated": True,
                    "importance": 11,
                    "reason": "Appears in troubleshooting steps",
                },
            ]
        }
    )

    class DummyResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, *, model: str, input):
            self.calls.append({"model": model, "input": input})
            return type("Resp", (), {"output": [], "output_text": dummy_output})

    class DummyOpenAI:
        def __init__(self, *_, **__):
            self.responses = DummyResponses()

    monkeypatch.setattr(ck, "OpenAI", DummyOpenAI, raising=False)

    settings = Settings(
        chat_backend="openai",
        openai_api_key="sk-test",
        openai_chat_model="gpt-test",
        keyword_filter_max_results=5,
    )

    filter_instance = KeywordLLMFilter(settings)
    assert filter_instance.enabled
    assert filter_instance.backend == "openai"

    concepts = [
        ConceptCandidate(
            phrase="login error",
            score=8.0,
            occurrences=3,
            document_count=2,
            chunk_count=3,
            average_occurrence_per_chunk=1.0,
            word_count=2,
            languages=["en"],
            sections=["Login / Errors"],
            sources=["guide.md"],
            sample_snippet="Login error handling steps.",
            score_breakdown={"frequency": 3.0},
        ),
        ConceptCandidate(
            phrase="system",
            score=5.0,
            occurrences=5,
            document_count=3,
            chunk_count=4,
            average_occurrence_per_chunk=1.25,
            word_count=1,
            languages=["en"],
            sections=["General"],
            sources=["reference.md"],
            sample_snippet="General system notes.",
            score_breakdown={"frequency": 5.0},
        ),
    ]

    refined = filter_instance.refine_concepts(concepts, ["Login errors spike after SSO handshake timing out."])

    phrases = [candidate.phrase for candidate in refined]
    assert "login authentication failure" in phrases
    assert "system" not in phrases
    generated = next(candidate for candidate in refined if candidate.generated)
    assert generated.phrase == "SSO handshake timeout"
    assert generated.reason == "Appears in troubleshooting steps"

    responses_stub = filter_instance._openai_client.responses  # type: ignore[attr-defined]
    assert getattr(responses_stub, "calls", None)
    call = responses_stub.calls[0]
    assert call["model"] == "gpt-test"
    assert isinstance(call["input"], list)

    debug = filter_instance.concept_debug_payload()
    assert debug.get("backend") == "openai"
    assert debug.get("status") == "refined"


def test_settings_from_env_prefers_env(monkeypatch) -> None:
    env_vars = {
        "CHAT_BACKEND": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:9999",
        "OLLAMA_MODEL": "llama3.1:test",
        "KEYWORD_FILTER_MAX_RESULTS": "7",
    }
    for key in env_vars:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    settings = Settings.from_env()

    assert settings.chat_backend == "ollama"
    assert settings.ollama_base_url == "http://localhost:9999"
    assert settings.ollama_model == "llama3.1:test"
    assert settings.keyword_filter_max_results == 7
    assert settings.is_ollama_chat_backend


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
