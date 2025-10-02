from __future__ import annotations

from cornerstone.keywords import KeywordLLMFilter, extract_keyword_candidates


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
