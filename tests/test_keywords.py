from __future__ import annotations

import json

from cornerstone.config import Settings
from cornerstone.keywords import (
    ConceptCandidate,
    ConceptCluster,
    KeywordCandidate,
    KeywordLLMFilter,
    KeywordSourceChunk,
    RankedConcept,
    cluster_concepts,
    extract_concept_candidates,
    extract_keyword_candidates,
    prepare_keyword_chunks,
    rank_concept_clusters,
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
    assert candidate.embedding_vector and len(candidate.embedding_vector) == 2
    assert candidate.embedding_backend == "fake"
    assert result.parameters.get("embedding_backend") == "fake"


def test_cluster_concepts_reuses_precomputed_vectors() -> None:
    candidate = ConceptCandidate(
        phrase="login issues",
        score=12.0,
        occurrences=4,
        document_count=2,
        chunk_count=3,
        average_occurrence_per_chunk=1.33,
        word_count=2,
        languages=["en"],
        sections=["Support"],
        sources=["support.md"],
        sample_snippet="Users report login issues.",
        score_breakdown={"frequency": 4.0},
        embedding_vector=[0.1, 0.9],
        embedding_backend="fake",
    )

    result = cluster_concepts([candidate], embedding_service=None)
    embed_info = result.parameters.get("embedding", {})
    assert embed_info.get("enabled") is True
    assert embed_info.get("backend") == "fake"
    assert embed_info.get("phrases") == 1


def test_cluster_concepts_only_embeds_missing_vectors() -> None:
    class CountingEmbeddingService:
        def __init__(self) -> None:
            self.embed_calls: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
            self.embed_calls.append(list(texts))
            return [[0.4, 0.6] for _ in texts]

    with_vector = ConceptCandidate(
        phrase="existing vector",
        score=5.0,
        occurrences=2,
        document_count=1,
        chunk_count=1,
        average_occurrence_per_chunk=2.0,
        word_count=2,
        languages=["en"],
        sections=["Docs"],
        sources=["guide.md"],
        sample_snippet=None,
        score_breakdown={"frequency": 2.0},
        embedding_vector=[0.9, 0.1],
        embedding_backend="precomputed",
    )
    missing_vector = ConceptCandidate(
        phrase="needs embedding",
        score=4.0,
        occurrences=1,
        document_count=1,
        chunk_count=1,
        average_occurrence_per_chunk=1.0,
        word_count=2,
        languages=["en"],
        sections=["Docs"],
        sources=["guide.md"],
        sample_snippet=None,
        score_breakdown={"frequency": 1.0},
    )

    service = CountingEmbeddingService()
    result = cluster_concepts([with_vector, missing_vector], embedding_service=service)

    assert service.embed_calls == [["needs embedding"]]
    embed_info = result.parameters.get("embedding", {})
    assert embed_info.get("enabled") is True
    assert embed_info.get("phrases") == 2


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


def _build_enabled_filter(keyword_payload: dict, monkeypatch, *, allow_generated: bool = False) -> KeywordLLMFilter:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
        keyword_filter_max_results=3,
        keyword_filter_allow_generated=allow_generated,
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
        "KEYWORD_STAGE3_LABEL_CLUSTERS": "false",
        "KEYWORD_STAGE3_LABEL_MAX_CLUSTERS": "4",
        "KEYWORD_STAGE4_CORE_LIMIT": "8",
        "KEYWORD_STAGE4_MAX_RESULTS": "25",
        "KEYWORD_STAGE4_SCORE_WEIGHT": "1.4",
        "KEYWORD_STAGE4_DOCUMENT_WEIGHT": "3.5",
        "KEYWORD_STAGE4_CHUNK_WEIGHT": "0.8",
        "KEYWORD_STAGE4_OCCURRENCE_WEIGHT": "0.45",
        "KEYWORD_STAGE4_LABEL_BONUS": "0.9",
        "KEYWORD_STAGE5_HARMONIZE_ENABLED": "true",
        "KEYWORD_STAGE5_HARMONIZE_MAX_RESULTS": "15",
        "KEYWORD_STAGE7_SUMMARY_ENABLED": "false",
        "KEYWORD_STAGE7_SUMMARY_MAX_INSIGHTS": "5",
        "KEYWORD_STAGE7_SUMMARY_MAX_CONCEPTS": "9",
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
    assert settings.keyword_stage3_label_clusters is False
    assert settings.keyword_stage3_label_max_clusters == 4
    assert settings.keyword_stage4_core_limit == 8
    assert settings.keyword_stage4_max_results == 25
    assert settings.keyword_stage4_score_weight == 1.4
    assert settings.keyword_stage4_document_weight == 3.5
    assert settings.keyword_stage4_chunk_weight == 0.8
    assert settings.keyword_stage4_occurrence_weight == 0.45
    assert settings.keyword_stage4_label_bonus == 0.9
    assert settings.keyword_stage5_harmonize_enabled is True
    assert settings.keyword_stage5_harmonize_max_results == 15
    assert settings.keyword_stage7_summary_enabled is False
    assert settings.keyword_stage7_summary_max_insights == 5
    assert settings.keyword_stage7_summary_max_concepts == 9


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


def test_cluster_concepts_uses_embedding_similarity() -> None:
    class StubEmbeddingService:
        def __init__(self) -> None:
            self.backend = type("Backend", (), {"name": "stub"})()

        def embed(self, texts: list[str]) -> list[list[float]]:
            vectors = {
                "sunrise metrics dashboard": [0.9, 0.1, 0.0],
                "dawn analytics console": [0.88, 0.12, 0.01],
                "authentication timeout": [0.1, 0.9, 0.2],
            }
            return [vectors.get(text, [0.0, 0.0, 0.0]) for text in texts]

    embedding_service = StubEmbeddingService()

    concepts = [
        ConceptCandidate(
            phrase="sunrise metrics dashboard",
            score=8.0,
            occurrences=3,
            document_count=2,
            chunk_count=2,
            average_occurrence_per_chunk=1.5,
            word_count=3,
            languages=["en"],
            sections=["Analytics"],
            sources=["metrics.md"],
            sample_snippet="Sunrise metrics dashboard overview.",
            score_breakdown={"embedding": 2.0},
        ),
        ConceptCandidate(
            phrase="dawn analytics console",
            score=7.5,
            occurrences=2,
            document_count=1,
            chunk_count=1,
            average_occurrence_per_chunk=2.0,
            word_count=3,
            languages=["en"],
            sections=["Analytics"],
            sources=["console.md"],
            sample_snippet="Dawn analytics console quickstart.",
            score_breakdown={"embedding": 1.8},
        ),
        ConceptCandidate(
            phrase="authentication timeout",
            score=4.0,
            occurrences=2,
            document_count=1,
            chunk_count=1,
            average_occurrence_per_chunk=2.0,
            word_count=2,
            languages=["en"],
            sections=["Auth"],
            sources=["auth.md"],
            sample_snippet="Authentication timeout troubleshooting.",
            score_breakdown={"frequency": 2.0},
        ),
    ]

    result = cluster_concepts(
        concepts,
        similarity_threshold=0.6,
        embedding_service=embedding_service,  # type: ignore[arg-type]
    )

    assert result.clusters
    parameters = result.parameters.get("embedding", {})
    assert parameters.get("enabled") is True
    analytics_cluster = next(
        cluster for cluster in result.clusters if "analytics" in cluster.label or "sunrise" in cluster.label
    )
    phrases = {member.phrase for member in analytics_cluster.members}
    assert "sunrise metrics dashboard" in phrases
    assert "dawn analytics console" in phrases
    timeout_cluster = next(
        cluster
        for cluster in result.clusters
        if any(member.phrase == "authentication timeout" for member in cluster.members)
    )
    assert timeout_cluster is not analytics_cluster


def test_cluster_concepts_applies_llm_labels(monkeypatch) -> None:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
    )
    llm_filter = KeywordLLMFilter(settings)

    payload = {
        "clusters": [
            {
                "index": 1,
                "label": "Login Authentication Issues",
                "description": "Repeated sign-in failures impacting multiple services.",
                "aliases": ["login errors"],
            }
        ]
    }
    monkeypatch.setattr(
        KeywordLLMFilter,
        "_invoke_backend",
        lambda self, prompt: json.dumps(payload),
    )

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
            phrase="authentication failure",
            score=8.0,
            occurrences=3,
            document_count=2,
            chunk_count=2,
            average_occurrence_per_chunk=1.5,
            word_count=2,
            languages=["en"],
            sections=["Login / Failures"],
            sources=["runbook.md"],
            sample_snippet="Authentication failures cause login issues.",
            score_breakdown={"frequency": 3.0},
        ),
    ]

    result = cluster_concepts(
        concepts,
        llm_filter=llm_filter,  # type: ignore[arg-type]
        llm_label_max_clusters=2,
    )

    assert result.parameters["llm_labeling"]["used"] is True
    cluster = result.clusters[0]
    assert cluster.label == "Login Authentication Issues"
    assert cluster.label_source == "llm"
    assert "login error" in cluster.aliases
    assert cluster.description
    cluster_debug = llm_filter.cluster_debug_payload()
    assert cluster_debug.get("status") == "labeled"
    assert cluster_debug.get("selected_total") == 1


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


def test_rank_concept_clusters_prioritises_document_coverage() -> None:
    base_members = [
        ConceptCandidate(
            phrase="login issue",
            score=10.0,
            occurrences=4,
            document_count=2,
            chunk_count=3,
            average_occurrence_per_chunk=1.33,
            word_count=2,
            languages=["en"],
            sections=["Login"],
            sources=["guide.md"],
            sample_snippet="Login issue troubleshooting steps.",
            score_breakdown={"frequency": 4.0},
        )
    ]
    cluster_login = ConceptCluster(
        label="Login Issues",
        label_source="top-member",
        score=18.0,
        occurrences=5,
        document_count=3,
        chunk_count=4,
        languages=["en"],
        sections=["Login"],
        sources=["guide.md"],
        members=base_members,
        score_breakdown={"member_count": 1},
        description=None,
        aliases=["login issue"],
    )

    cluster_payment = ConceptCluster(
        label="Payment Errors",
        label_source="top-member",
        score=22.0,
        occurrences=3,
        document_count=1,
        chunk_count=2,
        languages=["en"],
        sections=["Payments"],
        sources=["billing.md"],
        members=[
            ConceptCandidate(
                phrase="payment error",
                score=22.0,
                occurrences=3,
                document_count=1,
                chunk_count=2,
                average_occurrence_per_chunk=1.5,
                word_count=2,
                languages=["en"],
                sections=["Payments"],
                sources=["billing.md"],
                sample_snippet="Payment error handling.",
                score_breakdown={"frequency": 3.0},
            )
        ],
        score_breakdown={"member_count": 1},
        description=None,
        aliases=["payment error"],
    )

    result = rank_concept_clusters(
        [cluster_payment, cluster_login],
        core_limit=1,
        max_results=5,
        score_weight=0.8,
        document_weight=3.0,
        chunk_weight=0.6,
        occurrence_weight=0.2,
    )

    assert result.ranked
    top = result.ranked[0]
    assert top.label == "Login Issues"
    assert top.is_core is True
    assert top.document_count == 3
    second = result.ranked[1]
    assert second.label == "Payment Errors"
    assert second.is_core is False
    debug = result.to_debug_payload(limit=2)
    assert debug["total_ranked"] == 2
    assert debug["top_ranked"][0]["label"] == "Login Issues"


def test_rank_concept_clusters_generalizes_sentiment_labels() -> None:
    cluster = ConceptCluster(
        label="화주 부정 리뷰",
        label_source="llm",
        score=500.0,
        occurrences=30,
        document_count=10,
        chunk_count=12,
        languages=["ko"],
        sections=["리뷰"],
        sources=["reviews.csv"],
        members=[
            ConceptCandidate(
                phrase="화주 부정 리뷰",
                score=250.0,
                occurrences=15,
                document_count=8,
                chunk_count=10,
                average_occurrence_per_chunk=1.5,
                word_count=3,
                languages=["ko"],
                sections=["리뷰"],
                sources=["reviews.csv"],
                sample_snippet="화주가 서비스에 대해 불만을 제기했습니다.",
                score_breakdown={"frequency": 10.0},
            ),
            ConceptCandidate(
                phrase="화주 긍정 리뷰",
                score=200.0,
                occurrences=10,
                document_count=6,
                chunk_count=8,
                average_occurrence_per_chunk=1.25,
                word_count=3,
                languages=["ko"],
                sections=["리뷰"],
                sources=["reviews.csv"],
                sample_snippet="화주가 서비스에 만족했습니다.",
                score_breakdown={"frequency": 8.0},
            ),
        ],
        score_breakdown={"member_count": 2},
        description="화주 리뷰 전반",
        aliases=["화주 부정 리뷰", "화주 긍정 리뷰"],
    )

    result = rank_concept_clusters([cluster])
    assert result.ranked
    top = result.ranked[0]
    assert top.label == "화주 리뷰"
    assert "generalized" in top.label_source
    assert "화주 부정 리뷰" in top.aliases
    assert "화주 긍정 리뷰" in top.aliases


def test_rank_concept_clusters_marks_generated() -> None:
    cluster = ConceptCluster(
        label="LLM Concept",
        label_source="llm",
        score=30.0,
        occurrences=5,
        document_count=2,
        chunk_count=3,
        languages=["en"],
        sections=["Docs"],
        sources=["guide.md"],
        members=[
            ConceptCandidate(
                phrase="llm suggestion",
                score=15.0,
                occurrences=2,
                document_count=1,
                chunk_count=1,
                average_occurrence_per_chunk=2.0,
                word_count=2,
                languages=["en"],
                sections=["Docs"],
                sources=["guide.md"],
                sample_snippet=None,
                score_breakdown={"frequency": 2.0},
                generated=True,
            )
        ],
        score_breakdown={"member_count": 1},
        description=None,
        aliases=["llm suggestion"],
    )

    result = rank_concept_clusters([cluster], core_limit=1)
    ranked = result.ranked[0]
    assert ranked.generated is True
    debug = result.to_debug_payload(limit=1)
    assert debug["top_ranked"][0]["generated"] is True


def test_summarize_keywords_returns_insights(monkeypatch) -> None:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
    )
    filter_instance = KeywordLLMFilter(settings)

    payload = {
        "insights": [
            {
                "title": "Customer Pricing Pressure",
                "summary": "Pricing concerns dominate the negative feedback.",
                "keywords": ["운송가격", "화주 부정 리뷰"],
                "priority": "high",
                "action": "Review pricing strategy",
                "evidence": ["Multiple core keywords point to price dissatisfaction."],
            }
        ]
    }
    monkeypatch.setattr(
        KeywordLLMFilter,
        "_invoke_backend",
        lambda self, prompt: json.dumps(payload),
    )

    keywords = [
        KeywordCandidate(term="운송가격", count=10, is_core=True, generated=False, reason="Price feedback"),
        KeywordCandidate(term="화주 부정 리뷰", count=8, is_core=True, generated=True, reason="Negative shipper reviews"),
    ]

    result = filter_instance.summarize_keywords(
        keywords,
        max_insights=2,
        max_concepts=2,
        context_snippets=["고객이 운송가격에 불만을 제기했습니다."],
    )

    assert result and result[0]["title"] == "Customer Pricing Pressure"
    debug = filter_instance.insight_debug_payload()
    assert debug.get("status") == "success"
    assert debug.get("selected_total") == 1


def test_summarize_keywords_handles_missing_insights(monkeypatch) -> None:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
    )
    filter_instance = KeywordLLMFilter(settings)
    monkeypatch.setattr(KeywordLLMFilter, "_invoke_backend", lambda self, prompt: "{}")

    keywords = [KeywordCandidate(term="Alpha", count=3, is_core=True)]

    result = filter_instance.summarize_keywords(keywords, max_insights=2, max_concepts=1)
    assert result == []
    debug = filter_instance.insight_debug_payload()
    assert debug.get("status") == "error"
    assert debug.get("reason") == "missing-insights-key"


def test_harmonize_ranked_concepts_uses_llm(monkeypatch) -> None:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
    )
    filter_instance = KeywordLLMFilter(settings)

    payload = {
        "concepts": [
            {
                "index": 1,
                "label": "화주 리뷰",
                "description": "화주 피드백 전반",
                "aliases": ["화주 리뷰", "화주 피드백"],
            }
        ]
    }
    monkeypatch.setattr(
        KeywordLLMFilter,
        "_invoke_backend",
        lambda self, prompt: json.dumps(payload),
    )

    ranked = [
        RankedConcept(
            label="화주 부정 리뷰",
            score=120.0,
            rank=1,
            is_core=True,
            document_count=20,
            chunk_count=25,
            occurrences=80,
            label_source="llm",
            description=None,
            aliases=["화주 부정 리뷰", "화주 긍정 리뷰"],
            member_phrases=["화주 부정 리뷰", "화주 긍정 리뷰"],
            score_breakdown={"stage2_score": 100.0},
        )
    ]

    updated = filter_instance.harmonize_ranked_concepts(ranked, max_results=5)
    assert updated[0].label == "화주 리뷰"
    assert "harmonized" in (updated[0].label_source or "")
    assert "화주 리뷰" in updated[0].aliases

    debug = filter_instance.harmonize_debug_payload()
    assert debug.get("status") == "harmonized"


def test_harmonize_ranked_concepts_respects_string_keep_flag(monkeypatch) -> None:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
    )
    filter_instance = KeywordLLMFilter(settings)

    payload = {
        "concepts": [
            {
                "index": 1,
                "label": "Should Drop",
                "keep": "false",
            }
        ]
    }
    monkeypatch.setattr(
        KeywordLLMFilter,
        "_invoke_backend",
        lambda self, prompt: json.dumps(payload),
    )

    ranked = [
        RankedConcept(
            label="Original",
            score=50.0,
            rank=1,
            is_core=True,
            document_count=5,
            chunk_count=6,
            occurrences=12,
            label_source="top-member",
            description=None,
            aliases=["Original"],
            member_phrases=["Original"],
            score_breakdown={"stage2_score": 40.0},
        )
    ]

    updated = filter_instance.harmonize_ranked_concepts(ranked, max_results=3)
    assert updated == ranked  # concept not rewritten
    debug = filter_instance.harmonize_debug_payload()
    assert debug.get("status") == "no-changes"


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

    assert [item.term for item in results] == ["Alpha", "Beta"]
    assert [item.generated for item in results] == [False, False]

    info = filter_instance.debug_payload()
    assert info["status"] == "filtered"
    assert info.get("truncated", []) == []
    assert info["kept"] == ["Alpha", "Beta"]
    assert info.get("generated", []) == []
    assert info.get("generated_blocked") == ["Delta", "epsilon"]
    assert info.get("selected_total") == 2
    assert info.get("llm_selected_total") == 4
    assert set(info.get("assumed_true", [])) >= {"Beta", "Delta", "epsilon"}


def test_filter_keywords_preserves_core_flags(monkeypatch) -> None:
    filter_instance = _build_enabled_filter(
        {
            "keywords": [
                {"term": "Alpha", "keep": True},
                {"term": "Beta", "keep": True},
            ]
        },
        monkeypatch,
    )
    candidates = [
        KeywordCandidate(term="Alpha", count=5, is_core=True),
        KeywordCandidate(term="Beta", count=1, is_core=False),
    ]

    results = filter_instance.filter_keywords(candidates, context_snippets=[])

    assert len(results) == 2
    alpha, beta = results
    assert alpha.is_core is True
    assert beta.is_core is False


def test_filter_keywords_handles_missing_keywords_key(monkeypatch) -> None:
    candidates = [
        KeywordCandidate(term="Alpha", count=5, is_core=True),
        KeywordCandidate(term="Beta", count=2, is_core=False),
    ]

    filter_instance = _build_enabled_filter({}, monkeypatch)
    results = filter_instance.filter_keywords(candidates, context_snippets=["context"])

    assert results == candidates
    debug = filter_instance.debug_payload()
    assert debug.get("status") == "error"
    assert debug.get("reason") == "missing-keywords-array"


def test_refine_concepts_handles_missing_concepts_key(monkeypatch) -> None:
    settings = Settings(
        chat_backend="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="mock-keywords",
    )
    filter_instance = KeywordLLMFilter(settings)
    monkeypatch.setattr(KeywordLLMFilter, "_invoke_backend", lambda self, _: "{}")

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
            sections=["Login"],
            sources=["guide.md"],
            sample_snippet="Login error handling steps.",
            score_breakdown={"frequency": 4.0},
        )
    ]

    refined = filter_instance.refine_concepts(concepts, [])
    assert refined == concepts
    debug = filter_instance.concept_debug_payload()
    assert debug.get("status") == "error"
    assert debug.get("reason") == "missing-concepts-key"

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
    assert [item.term for item in results] == ["Beta"]
    assert [item.generated for item in results] == [False]
    assert results[0].source == "candidate"
    assert results[0].reason == "Core concept"

    info = filter_instance.debug_payload()
    assert info.get("generated_blocked") == ["Delta"]


def test_filter_keywords_allows_generated_when_enabled(monkeypatch) -> None:
    filter_instance = _build_enabled_filter(
        {
            "keywords": [
                {"term": "Alpha", "keep": True},
                {"term": "Beta", "keep": True},
            ]
        },
        monkeypatch,
        allow_generated=True,
    )
    candidates = [
        KeywordCandidate(term="Alpha", count=5, is_core=True),
        KeywordCandidate(term="Beta", count=3, is_core=True),
    ]

    payload = {
        "keywords": [
            {"term": "Alpha", "keep": True},
            {"term": "New Concept", "keep": True, "source": "generated"},
        ]
    }

    monkeypatch.setattr(KeywordLLMFilter, "_invoke_backend", lambda self, _: json.dumps(payload))

    results = filter_instance.filter_keywords(candidates, context_snippets=[])
    assert [item.term for item in results] == ["Alpha", "New Concept"]
    assert results[1].generated is True

    info = filter_instance.debug_payload()
    assert info["status"] == "filtered"
    assert "New Concept" in info.get("generated", [])
