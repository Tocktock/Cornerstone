from __future__ import annotations

import httpx
import pytest

from cornerstone.config import Settings
from cornerstone.keywords import ConceptCandidate, KeywordLLMFilter


@pytest.mark.integration
def test_refine_concepts_with_real_ollama() -> None:
    settings = Settings.from_env()

    if not settings.is_ollama_chat_backend:
        pytest.skip("CHAT_BACKEND is not set to Ollama")

    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - network guard
        pytest.skip(f"Ollama server not reachable: {exc}")

    filter_instance = KeywordLLMFilter(settings)
    if not filter_instance.enabled:
        pytest.skip("Keyword LLM filter is disabled")

    concepts = [
        ConceptCandidate(
            phrase="로그인 오류",
            score=5.0,
            occurrences=2,
            document_count=1,
            chunk_count=1,
            average_occurrence_per_chunk=2.0,
            word_count=3,
            languages=["ko"],
            sections=["로그인 / 오류"],
            sources=["guide.md"],
            sample_snippet="로그인 오류 해결 단계 요약.",
            score_breakdown={"frequency": 2.0},
        ),
        ConceptCandidate(
            phrase="login error",
            score=4.0,
            occurrences=1,
            document_count=1,
            chunk_count=1,
            average_occurrence_per_chunk=1.0,
            word_count=2,
            languages=["en"],
            sections=["Login / Errors"],
            sources=["runbook.md"],
            sample_snippet="Login error troubleshooting steps.",
            score_breakdown={"frequency": 1.0},
        ),
    ]

    refined = filter_instance.refine_concepts(
        concepts,
        [
            "로그인 오류가 발생하면 세션 토큰을 재발급하고 MFA 상태를 점검합니다.",
            "If login errors persist, review the SSO handshake timeout configuration.",
        ],
    )

    assert refined  # The call should return at least one candidate

    debug = filter_instance.concept_debug_payload()
    assert debug.get("backend") == "ollama"
    assert debug.get("status") in {"refined", "error", "bypass"}
