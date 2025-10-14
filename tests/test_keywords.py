from __future__ import annotations

import json

from cornerstone.config import Settings
from cornerstone.keywords import (
    ConceptCandidate,
    KeywordCandidate,
    KeywordLLMFilter,
    KeywordSourceChunk,
    cluster_concepts,
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
    assert set(login_error.document_ids) == {"doc-1", "doc-2"}
    assert set(login_error.chunk_ids) >= {"doc-1:0", "doc-2:0"}


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


def test_extract_concept_candidates_includes_embedding_scores() -> None:
    class FakeBackend:
        name = "fake"

    class FakeEmbeddingService:
        def __init__(self) -> None:
            self.backend = FakeBackend()
            base_vector = [1.0, 0.0]
            self._vectors = {
                "quantum gateway optimizer improves flux alignment gateway diagnostics calibrate quantum gateway": base_vector,
                "quantum gateway": [0.95, 0.0],
                "flux alignment": [0.2, 0.9],
                "gateway diagnostics": [0.9, 0.1],
                "quantum gateway optimizer": [0.85, 0.1],
            }

        def embed_one(self, text: str) -> list[float]:
            return list(self._vectors.get(text, [0.0, 0.0]))

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [list(self._vectors.get(text, [0.0, 0.0])) for text in texts]

    chunk = KeywordSourceChunk(
        text="Quantum gateway optimizer improves flux alignment. Gateway diagnostics calibrate quantum gateway.",
        normalized_text="quantum gateway optimizer improves flux alignment gateway diagnostics calibrate quantum gateway",
        doc_id="doc-emb",
        chunk_id="doc-emb:0",
        section_path="Quantum / Gateway",
        language="en",
    )

    embedding_service = FakeEmbeddingService()
    result = extract_concept_candidates(
        [chunk],
        embedding_service=embedding_service,
        max_ngram_size=2,
        max_candidates_per_chunk=4,
        max_embedding_phrases_per_chunk=3,
        min_char_length=2,
        embedding_weight=3.0,
    )

    embedding_stats = result.parameters.get("embedding_stats")
    assert embedding_stats and embedding_stats["chunks"] >= 1

    candidate = next(item for item in result.candidates if item.phrase == "quantum gateway")
    assert candidate.score_breakdown["embedding"] > 0
    scoring_weights = result.parameters.get("scoring_weights", {})
    assert scoring_weights.get("embedding") == 3.0


def test_extract_concept_candidates_includes_statistical_scores() -> None:
    chunk = KeywordSourceChunk(
        text="Escalation runbook outlines SLA breach protocol and customer escalation workflow.",
        normalized_text="escalation runbook outlines sla breach protocol and customer escalation workflow",
        doc_id="doc-stat",
        chunk_id="doc-stat:0",
        section_path="Operations / Escalation",
        language="en",
    )

    result = extract_concept_candidates(
        [chunk],
        max_ngram_size=3,
        max_candidates_per_chunk=5,
        max_statistical_phrases_per_chunk=4,
        min_char_length=2,
    )

    assert any(
        candidate.score_breakdown.get("statistical", 0) > 0 for candidate in result.candidates
    )


def test_extract_concept_candidates_uses_llm_summary() -> None:
    class StubSummaryFilter:
        enabled = True
        backend = "stub"

        def __init__(self) -> None:
            self._debug: dict[str, object] = {}

        def extract_summary_concepts(
            self,
            chunks: list[KeywordSourceChunk],
            *,
            max_results: int = 10,
            max_chars: int = 320,
        ) -> list[dict[str, object]]:
            self._debug = {"chunks": [chunk.chunk_id for chunk in chunks]}
            chunk_ids = [
                chunk.chunk_id or f"chunk-{index}"
                for index, chunk in enumerate(chunks, start=1)
            ]
            sections = [chunk.section_path for chunk in chunks if chunk.section_path]
            sources = [chunk.source for chunk in chunks if chunk.source]
            return [
                {
                    "phrase": "customer escalation policy",
                    "importance": 2.5,
                    "reason": "summarized",
                    "chunk_ids": chunk_ids,
                    "sections": sections,
                    "sources": sources,
                    "languages": ["en"],
                    "occurrences": 1,
                }
            ]

        def summary_debug_payload(self) -> dict[str, object]:
            return dict(self._debug)

    chunk = KeywordSourceChunk(
        text="Customer escalation policy ensures high-priority issues trigger leadership alerts.",
        normalized_text="customer escalation policy ensures high priority issues trigger leadership alerts",
        doc_id="doc-llm",
        chunk_id="doc-llm:0",
        section_path="Support / Policies",
        source="policy.md",
        language="en",
    )

    stub_filter = StubSummaryFilter()
    result = extract_concept_candidates(
        [chunk],
        llm_filter=stub_filter,  # type: ignore[arg-type]
        use_llm_summary=True,
        max_ngram_size=2,
        max_candidates_per_chunk=2,
        max_embedding_phrases_per_chunk=0,
    )

    summary_info = result.parameters.get("llm_summary")
    assert summary_info and summary_info["used"] is True

    candidate = next(item for item in result.candidates if item.phrase == "customer escalation policy")
    assert candidate.generated is True
    assert candidate.reason == "summarized"
    assert candidate.score_breakdown.get("llm", 0) > 0


def test_extract_concept_candidates_respects_summary_toggle() -> None:
    class GuardFilter:
        enabled = True
        backend = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def extract_summary_concepts(self, chunks: list[KeywordSourceChunk], **_) -> list[dict[str, object]]:
            self.calls += 1
            return [{"phrase": "placeholder concept"}]

    chunk = KeywordSourceChunk(
        text="Incident response guide for paging processes.",
        normalized_text="incident response guide for paging processes",
        doc_id="doc-toggle",
        chunk_id="doc-toggle:0",
        section_path="Operations / Incident Response",
        language="en",
    )

    guard_filter = GuardFilter()
    result = extract_concept_candidates(
        [chunk],
        llm_filter=guard_filter,  # type: ignore[arg-type]
        use_llm_summary=False,
        max_ngram_size=2,
        max_candidates_per_chunk=2,
        max_embedding_phrases_per_chunk=0,
    )

    assert guard_filter.calls == 0
    summary_info = result.parameters.get("llm_summary")
    assert summary_info and summary_info["used"] is False


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
        "KEYWORD_STAGE2_MAX_NGRAM": "4",
        "KEYWORD_STAGE2_MAX_CANDIDATES_PER_CHUNK": "12",
        "KEYWORD_STAGE2_MAX_EMBEDDING_PHRASES": "5",
        "KEYWORD_STAGE2_MAX_STATISTICAL_PHRASES": "3",
        "KEYWORD_STAGE2_USE_LLM_SUMMARY": "false",
        "KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHUNKS": "6",
        "KEYWORD_STAGE2_LLM_SUMMARY_MAX_RESULTS": "4",
        "KEYWORD_STAGE2_LLM_SUMMARY_MAX_CHARS": "256",
        "KEYWORD_STAGE2_MIN_CHAR_LENGTH": "4",
        "KEYWORD_STAGE2_MIN_OCCURRENCES": "2",
        "KEYWORD_STAGE2_EMBEDDING_WEIGHT": "2.25",
        "KEYWORD_STAGE2_STATISTICAL_WEIGHT": "1.6",
        "KEYWORD_STAGE2_LLM_WEIGHT": "3.1",
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
    assert settings.keyword_stage2_max_ngram == 4
    assert settings.keyword_stage2_max_candidates_per_chunk == 12
    assert settings.keyword_stage2_max_embedding_phrases_per_chunk == 5
    assert settings.keyword_stage2_max_statistical_phrases_per_chunk == 3
    assert settings.keyword_stage2_use_llm_summary is False
    assert settings.keyword_stage2_llm_summary_max_chunks == 6
    assert settings.keyword_stage2_llm_summary_max_results == 4
    assert settings.keyword_stage2_llm_summary_max_chars == 256
    assert settings.keyword_stage2_min_char_length == 4
    assert settings.keyword_stage2_min_occurrences == 2
    assert settings.keyword_stage2_embedding_weight == 2.25
    assert settings.keyword_stage2_statistical_weight == 1.6
    assert settings.keyword_stage2_llm_weight == 3.1


def test_cluster_concepts_groups_similar_phrases() -> None:
    concepts = [
        ConceptCandidate(
            phrase="login error",
            score=10.0,
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
            document_ids=["doc-1", "doc-2"],
            chunk_ids=["doc-1:0", "doc-2:0"],
        ),
        ConceptCandidate(
            phrase="login authentication failure",
            score=8.0,
            occurrences=3,
            document_count=2,
            chunk_count=2,
            average_occurrence_per_chunk=1.5,
            word_count=3,
            languages=["en"],
            sections=["Login / Failures"],
            sources=["runbook.md"],
            sample_snippet="Authentication failures cause login errors.",
            score_breakdown={"frequency": 3.0},
            document_ids=["doc-3"],
            chunk_ids=["doc-3:0"],
        ),
        ConceptCandidate(
            phrase="payment outage",
            score=6.0,
            occurrences=2,
            document_count=1,
            chunk_count=1,
            average_occurrence_per_chunk=2.0,
            word_count=2,
            languages=["en"],
            sections=["Payments"],
            sources=["finance.md"],
            sample_snippet="Payment outage troubleshooting.",
            score_breakdown={"frequency": 2.0},
            document_ids=["doc-4"],
            chunk_ids=["doc-4:0"],
        ),
    ]

    result = cluster_concepts(concepts, similarity_threshold=0.5)
    assert result.clusters
    top_labels = {cluster.label for cluster in result.clusters}
    assert "login error" in top_labels or "login authentication failure" in top_labels

    login_cluster = next(cluster for cluster in result.clusters if "login" in cluster.label)
    member_phrases = {member.phrase for member in login_cluster.members}
    assert {
        "login error",
        "login authentication failure",
    } <= member_phrases
    assert login_cluster.document_count >= 2
    assert login_cluster.score >= 18.0


def test_cluster_concepts_handles_korean() -> None:
    concepts = [
        ConceptCandidate(
            phrase="로그인 오류",
            score=7.0,
            occurrences=3,
            document_count=2,
            chunk_count=2,
            average_occurrence_per_chunk=1.5,
            word_count=3,
            languages=["ko"],
            sections=["로그인"],
            sources=["guide.md"],
            sample_snippet="로그인 오류 해결 단계.",
            score_breakdown={"frequency": 3.0},
        ),
        ConceptCandidate(
            phrase="결제 오류",
            score=5.0,
            occurrences=2,
            document_count=1,
            chunk_count=1,
            average_occurrence_per_chunk=2.0,
            word_count=3,
            languages=["ko"],
            sections=["결제"],
            sources=["billing.md"],
            sample_snippet="결제 오류 처리 절차.",
            score_breakdown={"frequency": 2.0},
        ),
    ]

    result = cluster_concepts(concepts, similarity_threshold=0.6)
    assert len(result.clusters) == 2
    labels = {cluster.label for cluster in result.clusters}
    assert labels == {"로그인 오류", "결제 오류"}


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
