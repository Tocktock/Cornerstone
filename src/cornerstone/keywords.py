"""Keyword extraction utilities for project documents."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Sequence, Tuple

import httpx

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .config import Settings, normalize_vllm_base_url

if TYPE_CHECKING:
    from .embeddings import EmbeddingService

# Basic English stop words; extendable if needed.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "http",
    "https",
    "www",
    "com",
    "files",
    "file",
    "image",
    "img",
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "svg",
    "send",
    "admin",
}

_WORD_RE = re.compile(r"[A-Za-z\uAC00-\uD7A3][A-Za-z0-9\uAC00-\uD7A3'\-]{0,}")
_NORMALIZED_TEXT_CLEAN_RE = re.compile(r"[^\w\s\uAC00-\uD7A3\-\'\u2013\u2014/]+")
_WHITESPACE_RE = re.compile(r"\s+")

logger = logging.getLogger(__name__)
_configured_level = os.getenv("KEYWORD_LOG_LEVEL")
if _configured_level:
    level_value = getattr(logging, _configured_level.upper(), None)
    if isinstance(level_value, int):
        logger.setLevel(level_value)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
elif logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False


@dataclass(slots=True)
class KeywordCandidate:
    """Represents a keyword extracted from project content."""

    term: str
    count: int
    is_core: bool = False
    generated: bool = False
    reason: str | None = None
    source: str = "frequency"


def _contains_hangul(text: str) -> bool:
    return any("\uAC00" <= char <= "\uD7A3" for char in text)


def build_excerpt(text: str, *, max_chars: int = 280) -> str:
    collapsed = re.sub(r"\s+", " ", str(text)).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max(0, max_chars - 1)].rstrip() + "…"


@dataclass(slots=True)
class KeywordSourceChunk:
    """Represents a prepared text chunk ready for concept extraction stages."""

    text: str
    normalized_text: str
    doc_id: str | None = None
    chunk_id: str | None = None
    source: str | None = None
    title: str | None = None
    section_path: str | None = None
    summary: str | None = None
    language: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def excerpt(self, *, max_chars: int = 200) -> str:
        return build_excerpt(self.text, max_chars=max_chars)


@dataclass(slots=True)
class ChunkPreparationResult:
    """Container for chunk preparation output and basic statistics."""

    chunks: List[KeywordSourceChunk]
    total_payloads: int = 0
    skipped_empty: int = 0
    skipped_non_text: int = 0

    @property
    def processed_count(self) -> int:
        return len(self.chunks)

    def unique_languages(self) -> List[str]:
        languages = {chunk.language for chunk in self.chunks if chunk.language}
        return sorted(languages)

    def total_tokens(self) -> int:
        return sum(chunk.token_count or 0 for chunk in self.chunks)

    def sample_sections(self, limit: int = 3) -> List[str]:
        samples: List[str] = []
        for chunk in self.chunks:
            label = chunk.section_path or chunk.title or chunk.source
            if label and label not in samples:
                samples.append(label)
            if len(samples) >= limit:
                break
        return samples

    def sample_excerpts(self, limit: int = 3, *, max_chars: int = 160) -> List[str]:
        return [chunk.excerpt(max_chars=max_chars) for chunk in self.chunks[:limit]]


@dataclass(slots=True)
class ConceptCandidate:
    """Represents a candidate concept or phrase extracted from Stage 2."""

    phrase: str
    score: float
    occurrences: int
    document_count: int
    chunk_count: int
    average_occurrence_per_chunk: float
    word_count: int
    languages: List[str]
    sections: List[str]
    sources: List[str]
    sample_snippet: str | None
    score_breakdown: dict[str, float] = field(default_factory=dict)
    reason: str | None = None
    generated: bool = False
    document_ids: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    embedding_vector: List[float] | None = None
    embedding_backend: str | None = None


@dataclass(slots=True)
class ConceptExtractionResult:
    """Container for Stage 2 extraction results and diagnostics."""

    candidates: List[ConceptCandidate]
    total_chunks: int
    total_tokens: int
    parameters: dict[str, Any]

    def summary(self, *, limit: int = 5) -> List[dict[str, Any]]:
        items: List[dict[str, Any]] = []
        for candidate in self.candidates[:limit]:
            entry = {
                "phrase": candidate.phrase,
                "score": round(candidate.score, 3),
                "occurrences": candidate.occurrences,
                "documents": candidate.document_count,
                "chunks": candidate.chunk_count,
                "sample": candidate.sample_snippet,
                "generated": candidate.generated,
            }
            if candidate.reason:
                entry["reason"] = candidate.reason
            items.append(entry)
        return items

    def to_debug_payload(self, *, limit: int = 10) -> dict[str, Any]:
        return {
            "total_candidates": len(self.candidates),
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "parameters": self.parameters,
            "top_candidates": self.summary(limit=limit),
        }

    def replace_candidates(self, candidates: Sequence[ConceptCandidate]) -> ConceptExtractionResult:
        return ConceptExtractionResult(
            candidates=list(candidates),
            total_chunks=self.total_chunks,
            total_tokens=self.total_tokens,
            parameters=dict(self.parameters),
        )


def iter_candidate_batches(
    candidates: Sequence[ConceptCandidate],
    *,
    batch_size: int,
    overlap: int = 0,
) -> Iterable[List[ConceptCandidate]]:
    """Yield deterministic slices of concept candidates for batching.

    Args:
        candidates: Ordered sequence of concept candidates (Stage 2 output).
        batch_size: Maximum number of candidates per batch (<= 0 means single batch).
        overlap: Number of candidates to repeat between consecutive batches (helps
            retain top-ranked concepts across boundaries).
    """

    total = len(candidates)
    if total == 0:
        yield []
        return
    if batch_size <= 0 or batch_size >= total:
        yield list(candidates)
        return

    effective_overlap = max(0, min(overlap, batch_size - 1))
    step = max(1, batch_size - effective_overlap)

    start = 0
    while start < total:
        end = min(total, start + batch_size)
        yield list(candidates[start:end])
        if end >= total:
            break
        start += step


def concept_sort_key(candidate: ConceptCandidate) -> Tuple[float, int, int, str]:
    """Sort key matching Stage 2/LMM refinement ordering."""

    return (
        -candidate.score,
        -candidate.document_count,
        -candidate.occurrences,
        candidate.phrase,
    )


def dedupe_concept_candidates(candidates: Sequence[ConceptCandidate]) -> List[ConceptCandidate]:
    """Collapse duplicate candidates introduced by batching overlap while preserving order."""

    result: List[ConceptCandidate] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in candidates:
        chunk_ids = tuple(sorted(candidate.chunk_ids)) if candidate.chunk_ids else ()
        key = (candidate.phrase.lower(), chunk_ids)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


@dataclass(slots=True)
class _CandidateAggregate:
    occurrences: int = 0
    chunk_ids: set[str] = field(default_factory=set)
    doc_ids: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    sections: set[str] = field(default_factory=set)
    sample_snippet: str | None = None
    languages: Counter[str] = field(default_factory=Counter)
    score_frequency: float = 0.0
    score_chunk: float = 0.0
    word_count: int = 0
    score_statistical: float = 0.0
    score_embedding: float = 0.0
    score_llm: float = 0.0
    reason: str | None = None
    generated: bool = False
    embedding_vector: list[float] | None = None
    embedding_vector_count: int = 0
    embedding_backend: str | None = None


@dataclass(slots=True)
class _ChunkContribution:
    occurrences: int = 0
    frequency_score: float = 0.0
    chunk_score: float = 0.0
    embedding_score: float = 0.0
    statistical_score: float = 0.0
    llm_score: float = 0.0
    reason: str | None = None
    generated: bool = False
    word_count: int = 0
    sample_snippet: str | None = None
    sections: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)
    embedding_vector: list[float] | None = None
    embedding_backend: str | None = None


@dataclass(slots=True)
class _ChunkContext:
    chunk: KeywordSourceChunk
    chunk_identifier: str
    doc_identifier: str


@dataclass(slots=True)
class ConceptCluster:
    """Represents a consolidated concept grouping after Stage 3."""

    label: str
    label_source: str
    score: float
    occurrences: int
    document_count: int
    chunk_count: int
    languages: List[str]
    sections: List[str]
    sources: List[str]
    members: List[ConceptCandidate]
    score_breakdown: dict[str, float]
    description: str | None = None
    aliases: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ConceptClusteringResult:
    clusters: List[ConceptCluster]
    parameters: dict[str, Any]

    def to_debug_payload(self, *, limit: int = 8) -> dict[str, Any]:
        top = []
        for cluster in self.clusters[:limit]:
            top.append(
                {
                    "label": cluster.label,
                    "label_source": cluster.label_source,
                    "description": cluster.description,
                    "aliases": cluster.aliases[:4],
                    "score": round(cluster.score, 3),
                    "occurrences": cluster.occurrences,
                    "documents": cluster.document_count,
                    "chunks": cluster.chunk_count,
                    "members": [member.phrase for member in cluster.members[:5]],
                }
            )
        return {
            "total_clusters": len(self.clusters),
            "parameters": self.parameters,
            "top_clusters": top,
        }


@dataclass(slots=True)
class _ClusterBuilder:
    label: str
    score: float
    occurrences: int
    document_ids: set[str]
    chunk_ids: set[str]
    languages: Counter[str]
    sections: set[str]
    sources: set[str]
    members: List[ConceptCandidate]
    tokens: set[str]
    vector: list[float] | None
    vector_count: int
    label_source: str = "top-member"
    description: str | None = None
    aliases: List[str] = field(default_factory=list)

    def add(
        self,
        candidate: ConceptCandidate,
        *,
        tokens: set[str],
        vector: Sequence[float] | None = None,
    ) -> None:
        self.members.append(candidate)
        self.score += candidate.score
        self.occurrences += candidate.occurrences
        self.tokens |= tokens
        self.languages.update(candidate.languages)
        self.sections.update(candidate.sections)
        self.sources.update(candidate.sources)
        if candidate.phrase not in self.aliases:
            self.aliases.append(candidate.phrase)
        if candidate.document_ids:
            self.document_ids.update(candidate.document_ids)
        elif candidate.document_count:
            self.document_ids.update({f"doc-{len(self.document_ids) + i}" for i in range(candidate.document_count)})
        if candidate.chunk_ids:
            self.chunk_ids.update(candidate.chunk_ids)
        elif candidate.chunk_count:
            self.chunk_ids.update({f"chunk-{len(self.chunk_ids) + i}" for i in range(candidate.chunk_count)})
        top_member = max(self.members, key=lambda item: item.score)
        self.label = top_member.phrase
        if top_member.phrase not in self.aliases:
            self.aliases.append(top_member.phrase)

        if vector:
            if self.vector is None:
                self.vector = list(vector)
                self.vector_count = 1
            elif len(self.vector) == len(vector):
                total = self.vector_count + 1
                self.vector = [
                    (existing * self.vector_count + float(new)) / total
                    for existing, new in zip(self.vector, vector)
                ]
                self.vector_count = total

@dataclass(slots=True)
class RankedConcept:
    label: str
    score: float
    rank: int
    is_core: bool
    document_count: int
    chunk_count: int
    occurrences: int
    label_source: str
    description: str | None
    aliases: List[str]
    member_phrases: List[str]
    score_breakdown: dict[str, float]
    generated: bool = False


@dataclass(slots=True)
class ConceptRankingResult:
    ranked: List[RankedConcept]
    parameters: dict[str, Any]

    def to_debug_payload(self, *, limit: int = 10) -> dict[str, Any]:
        top = []
        for concept in self.ranked[:limit]:
            top.append(
                {
                    "label": concept.label,
                    "score": round(concept.score, 3),
                    "rank": concept.rank,
                    "core": concept.is_core,
                    "documents": concept.document_count,
                    "chunks": concept.chunk_count,
                    "occurrences": concept.occurrences,
                    "label_source": concept.label_source,
                    "aliases": concept.aliases[:4],
                    "generated": concept.generated,
                }
            )
        return {
            "total_ranked": len(self.ranked),
            "parameters": self.parameters,
            "top_ranked": top,
        }

    def replace_ranked(self, ranked: Sequence[RankedConcept]) -> ConceptRankingResult:
        return ConceptRankingResult(ranked=list(ranked), parameters=dict(self.parameters))

def _normalize_language(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned.lower()
    return None


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\n;]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return []


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _normalize_chunk_text(text: str, *, language: str | None = None) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    if not collapsed:
        return ""
    without_punct = _NORMALIZED_TEXT_CLEAN_RE.sub(" ", collapsed)
    squeezed = _WHITESPACE_RE.sub(" ", without_punct).strip()
    if language and language.startswith("en"):
        return squeezed.lower()
    return squeezed


def prepare_keyword_chunks(payloads: Iterable[Mapping[str, Any]]) -> ChunkPreparationResult:
    chunks: List[KeywordSourceChunk] = []
    total_payloads = 0
    skipped_empty = 0
    skipped_non_text = 0

    for payload in payloads:
        total_payloads += 1
        text_value = payload.get("text")
        if isinstance(text_value, bytes):
            text_value = text_value.decode("utf-8", errors="ignore")
        if text_value is None:
            skipped_empty += 1
            continue
        if not isinstance(text_value, str):
            skipped_non_text += 1
            continue
        text = text_value.strip()
        if not text:
            skipped_empty += 1
            continue

        language = _normalize_language(payload.get("language"))
        normalized_text = _normalize_chunk_text(text, language=language)

        metadata: dict[str, Any] = {}
        for key in ("heading_path", "content_type", "ingested_at"):
            value = payload.get(key)
            if value is not None:
                metadata[key] = value

        chunk = KeywordSourceChunk(
            text=text,
            normalized_text=normalized_text,
            doc_id=_safe_str(payload.get("doc_id")),
            chunk_id=_safe_str(payload.get("chunk_id")),
            source=_safe_str(payload.get("source")),
            title=_safe_str(payload.get("title")),
            section_path=_safe_str(payload.get("section_path")),
            summary=_safe_str(payload.get("summary")),
            language=language,
            token_count=_coerce_optional_int(payload.get("token_count")),
            metadata=metadata,
        )
        chunks.append(chunk)

    return ChunkPreparationResult(
        chunks=chunks,
        total_payloads=total_payloads,
        skipped_empty=skipped_empty,
        skipped_non_text=skipped_non_text,
    )


def _tokenize_chunk_for_candidates(chunk: KeywordSourceChunk) -> List[str]:
    source = chunk.normalized_text or _WHITESPACE_RE.sub(" ", chunk.text).strip()
    if not source:
        return []
    tokens: List[str] = []
    for match in _WORD_RE.finditer(source):
        token = match.group().strip("'-")
        if not token:
            continue
        tokens.append(token)
    return tokens


def _phrase_tokens_for_clustering(phrase: str) -> set[str]:
    tokens: list[str] = []
    for match in _WORD_RE.finditer(phrase):
        token = match.group().strip("'-")
        if not token:
            continue
        normalized = token.lower() if token.isascii() else token
        if normalized in _STOPWORDS:
            continue
        tokens.append(normalized)
    if not tokens:
        normalized_phrase = phrase.strip().lower()
        if normalized_phrase:
            tokens.append(normalized_phrase)
    return set(tokens)


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    if not intersection:
        return 0.0
    union = left | right
    return len(intersection) / len(union)


def _token_overlap_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    if not intersection:
        return 0.0
    denominator = min(len(left), len(right))
    if denominator == 0:
        return 0.0
    return len(intersection) / denominator


_SENTIMENT_POSITIVE = ("긍정", "positive")
_SENTIMENT_NEGATIVE = ("부정", "불만", "negative")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" -_/," )


def _strip_sentiment_terms(text: str) -> str:
    result = text
    for term in _SENTIMENT_POSITIVE + _SENTIMENT_NEGATIVE:
        result = result.replace(term, "")
        result = result.replace(term.upper(), "")
    result = re.sub(r"\s+", " ", result)
    return result.strip(" -_/," )


def _choose_cluster_display_label(cluster: ConceptCluster) -> tuple[str, str, list[str]]:
    aliases: list[str] = list(dict.fromkeys(cluster.aliases or []))
    member_phrases = [member.phrase for member in cluster.members]
    for phrase in member_phrases:
        if phrase not in aliases:
            aliases.append(phrase)

    has_positive = any(any(term in alias for term in _SENTIMENT_POSITIVE) for alias in aliases)
    has_negative = any(any(term in alias for term in _SENTIMENT_NEGATIVE) for alias in aliases)

    new_label = cluster.label
    new_label_source = cluster.label_source

    if has_positive and has_negative:
        stripped_candidates = []
        for alias in aliases + [cluster.label]:
            stripped = _strip_sentiment_terms(alias)
            stripped = _normalize_whitespace(stripped)
            if stripped and stripped != alias:
                stripped_candidates.append(stripped)
        if stripped_candidates:
            stripped_candidates = list(dict.fromkeys(stripped_candidates))
            stripped_candidates.sort(
                key=lambda name: (
                    0 if ("리뷰" in name or "review" in name.lower()) else 1,
                    len(name),
                )
            )
            chosen = stripped_candidates[0]
            if chosen:
                new_label = chosen
                new_label_source = (
                    f"{cluster.label_source}+generalized"
                    if cluster.label_source != "generalized"
                    else cluster.label_source
                )
                if new_label not in aliases:
                    aliases.insert(0, new_label)

    aliases = list(dict.fromkeys(aliases))
    return new_label, new_label_source, aliases


def _is_valid_unigram(token: str, *, min_char_length: int) -> bool:
    cleaned = token.strip("'-")
    if not cleaned:
        return False
    if cleaned.isdigit():
        return False
    if any(char.isdigit() for char in cleaned) and any(char.isalpha() for char in cleaned):
        return False
    length = len(cleaned)
    if _contains_hangul(cleaned):
        return length >= max(1, min_char_length - 1)
    if cleaned.lower() in _STOPWORDS:
        return False
    return length >= min_char_length


def _is_valid_phrase(tokens: Sequence[str], *, min_char_length: int) -> bool:
    if not tokens:
        return False
    if len(tokens) == 1:
        return _is_valid_unigram(tokens[0], min_char_length=min_char_length)
    total_chars = sum(len(token.strip("'-")) for token in tokens)
    if total_chars < min_char_length:
        return False
    ascii_tokens = [token for token in tokens if token.isascii()]
    if ascii_tokens and all(token.lower() in _STOPWORDS for token in ascii_tokens):
        return False
    if not any(_is_valid_unigram(token, min_char_length=min_char_length) for token in tokens):
        if not any(_contains_hangul(token) for token in tokens):
            return False
    return True


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        logger.debug("keyword.embeddings.dimension_mismatch a=%s b=%s", len(vec_a), len(vec_b))
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_rake_scores(
    tokens: Sequence[str],
    *,
    max_ngram_size: int,
    min_char_length: int,
) -> dict[str, float]:
    if not tokens:
        return {}

    phrases: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        cleaned = token.strip("'-")
        if not cleaned:
            continue
        if cleaned.isascii() and cleaned.lower() in _STOPWORDS:
            if current:
                phrases.append(current)
                current = []
            continue
        current.append(token)
    if current:
        phrases.append(current)

    if not phrases:
        return {}

    word_freq: Counter[str] = Counter()
    word_degree: Counter[str] = Counter()

    for phrase_tokens in phrases:
        filtered = [token for token in phrase_tokens if _is_valid_unigram(token, min_char_length=min_char_length)]
        if not filtered:
            continue
        length = len(filtered)
        degree = max(0, length - 1)
        for token in filtered:
            key = token.lower() if token.isascii() else token
            word_freq[key] += 1
            word_degree[key] += degree

    if not word_freq:
        return {}

    for token, freq in word_freq.items():
        word_degree[token] += freq

    word_score: dict[str, float] = {
        token: word_degree[token] / float(word_freq[token]) for token in word_freq
    }

    phrase_scores: dict[str, float] = {}
    for phrase_tokens in phrases:
        filtered = [token for token in phrase_tokens if _is_valid_unigram(token, min_char_length=min_char_length)]
        if not filtered:
            continue
        trimmed = filtered[:max_ngram_size]
        if not trimmed:
            continue
        if not _is_valid_phrase(trimmed, min_char_length=min_char_length):
            continue
        phrase = " ".join(trimmed)
        score = sum(word_score.get(token.lower() if token.isascii() else token, 0.0) for token in trimmed)
        if score <= 0:
            continue
        phrase_scores[phrase] = max(phrase_scores.get(phrase, 0.0), score)

    return phrase_scores


def _select_summary_contexts(
    contexts: Sequence[_ChunkContext],
    *,
    limit: int,
) -> list[_ChunkContext]:
    if not contexts or limit <= 0:
        return []

    ranked = sorted(
        contexts,
        key=lambda ctx: (
            ctx.chunk.token_count if ctx.chunk.token_count is not None else len(ctx.chunk.text),
            len(ctx.chunk.text),
        ),
        reverse=True,
    )
    return ranked[:limit]


def extract_concept_candidates(
    chunks: Sequence[KeywordSourceChunk],
    *,
    embedding_service: "EmbeddingService | None" = None,
    llm_filter: "KeywordLLMFilter | None" = None,
    use_llm_summary: bool | None = None,
    max_ngram_size: int = 3,
    max_candidates_per_chunk: int = 8,
    max_embedding_phrases_per_chunk: int = 6,
    max_statistical_phrases_per_chunk: int = 6,
    llm_summary_max_chunks: int = 4,
    llm_summary_max_results: int = 10,
    llm_summary_max_chars: int = 320,
    min_char_length: int = 3,
    min_occurrences: int = 1,
    embedding_weight: float = 1.75,
    statistical_weight: float = 1.1,
    llm_weight: float = 2.5,
) -> ConceptExtractionResult:
    aggregates: dict[str, _CandidateAggregate] = {}
    total_tokens = 0
    parameters = {
        "max_ngram_size": max_ngram_size,
        "max_candidates_per_chunk": max_candidates_per_chunk,
        "min_char_length": min_char_length,
        "min_occurrences": min_occurrences,
        "max_embedding_phrases_per_chunk": max_embedding_phrases_per_chunk,
        "max_statistical_phrases_per_chunk": max_statistical_phrases_per_chunk,
        "llm_summary_max_chunks": llm_summary_max_chunks,
        "llm_summary_max_results": llm_summary_max_results,
        "scoring_weights": {
            "frequency": 1.0,
            "chunk": 1.0,
            "statistical": statistical_weight,
            "embedding": embedding_weight,
            "llm": llm_weight,
        },
    }

    chunk_contexts: list[_ChunkContext] = []
    chunk_lookup: dict[str, _ChunkContext] = {}
    embedding_cache: dict[str, Sequence[float]] = {}
    chunk_embedding_cache: dict[str, Sequence[float]] = {}
    embedding_stats = {"chunks": 0, "phrases": 0, "errors": 0}
    statistical_stats = {"chunks": 0, "phrases": 0}

    def apply_contribution(phrase: str, contribution: _ChunkContribution, context: _ChunkContext) -> None:
        if not phrase:
            return
        word_count = contribution.word_count or len(phrase.split())
        aggregate = aggregates.setdefault(phrase, _CandidateAggregate(word_count=word_count))
        aggregate.word_count = max(aggregate.word_count, word_count)
        if contribution.occurrences:
            aggregate.occurrences += contribution.occurrences
        aggregate.score_frequency += contribution.frequency_score
        aggregate.score_statistical += contribution.statistical_score
        aggregate.score_embedding += contribution.embedding_score
        aggregate.score_llm += contribution.llm_score
        if contribution.reason and not aggregate.reason:
            aggregate.reason = contribution.reason
        if contribution.generated:
            aggregate.generated = True
        if contribution.embedding_vector is not None:
            vector = list(contribution.embedding_vector)
            if aggregate.embedding_vector is None or len(aggregate.embedding_vector) != len(vector):
                aggregate.embedding_vector = vector
                aggregate.embedding_vector_count = 1
            else:
                total = aggregate.embedding_vector_count + 1
                aggregate.embedding_vector = [
                    (existing * aggregate.embedding_vector_count + float(new)) / total
                    for existing, new in zip(aggregate.embedding_vector, vector)
                ]
                aggregate.embedding_vector_count = total
        if contribution.embedding_backend and not aggregate.embedding_backend:
            aggregate.embedding_backend = contribution.embedding_backend

        chunk_identifier = context.chunk_identifier
        doc_identifier = context.doc_identifier

        if chunk_identifier not in aggregate.chunk_ids:
            aggregate.chunk_ids.add(chunk_identifier)
            aggregate.score_chunk += contribution.chunk_score
            if contribution.sections:
                aggregate.sections.update(filter(None, contribution.sections))
            else:
                if context.chunk.section_path:
                    aggregate.sections.add(context.chunk.section_path)
                elif context.chunk.title:
                    aggregate.sections.add(context.chunk.title)
            if contribution.sources:
                aggregate.sources.update(filter(None, contribution.sources))
            elif context.chunk.source:
                aggregate.sources.add(context.chunk.source)
            if aggregate.sample_snippet is None:
                if contribution.sample_snippet:
                    aggregate.sample_snippet = build_excerpt(contribution.sample_snippet)
                elif context.chunk.summary:
                    aggregate.sample_snippet = build_excerpt(context.chunk.summary)
                else:
                    aggregate.sample_snippet = context.chunk.excerpt(max_chars=160)

        if doc_identifier not in aggregate.doc_ids:
            aggregate.doc_ids.add(doc_identifier)

        if contribution.languages:
            aggregate.languages.update(filter(None, contribution.languages))
        elif context.chunk.language:
            aggregate.languages.update([context.chunk.language])

    def contribution_for(
        contribution_map: dict[str, _ChunkContribution],
        phrase: str,
        word_count: int,
    ) -> _ChunkContribution:
        entry = contribution_map.get(phrase)
        if entry is None:
            entry = _ChunkContribution(word_count=word_count)
            contribution_map[phrase] = entry
        else:
            entry.word_count = max(entry.word_count, word_count)
        return entry

    total_start = time.perf_counter()
    timing_stats = {
        "total": 0.0,
        "chunk_processing": 0.0,
        "statistical": 0.0,
        "embedding": 0.0,
        "llm_summary": 0.0,
    }

    embedding_enabled = embedding_service is not None
    backend = getattr(embedding_service, "backend", None)
    embedding_backend_name: str | None = None
    if backend is not None:
        embedding_backend_name = getattr(backend, "name", None) or str(backend)
    elif embedding_service is not None:
        embedding_backend_name = getattr(embedding_service, "name", None) or str(type(embedding_service).__name__)
    if embedding_backend_name:
        parameters["embedding_backend"] = embedding_backend_name
    parameters["embedding_enabled"] = embedding_enabled
    parameters["statistical_method"] = "rake" if max_statistical_phrases_per_chunk > 0 else "none"

    summary_requested = bool(use_llm_summary if use_llm_summary is not None else True)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "keyword.stage2.start chunks=%s embedding_enabled=%s embedding_backend=%s summary_requested=%s",
            len(chunks),
            embedding_enabled,
            parameters.get("embedding_backend"),
            summary_requested,
        )

    for chunk_index, chunk in enumerate(chunks):
        chunk_identifier = chunk.chunk_id or f"chunk-{chunk_index}"
        doc_identifier = chunk.doc_id or f"doc-{chunk_index}"
        context = _ChunkContext(chunk=chunk, chunk_identifier=chunk_identifier, doc_identifier=doc_identifier)
        chunk_contexts.append(context)
        chunk_lookup[chunk_identifier] = context

        chunk_timer = time.perf_counter()
        tokens = _tokenize_chunk_for_candidates(chunk)
        total_tokens += len(tokens)
        if not tokens:
            chunk_elapsed = time.perf_counter() - chunk_timer
            timing_stats["chunk_processing"] += chunk_elapsed
            if logger.isEnabledFor(logging.INFO):
                logger.info("keyword.stage2.chunk empty chunk=%s elapsed=%.3fs", chunk_identifier, chunk_elapsed)
            continue

        ngram_counts: Counter[str] = Counter()
        token_length = len(tokens)
        for size in range(1, max_ngram_size + 1):
            if size > token_length:
                break
            for start in range(0, token_length - size + 1):
                phrase_tokens = tokens[start : start + size]
                if not _is_valid_phrase(phrase_tokens, min_char_length=min_char_length):
                    continue
                phrase = " ".join(phrase_tokens)
                ngram_counts[phrase] += 1

        if not ngram_counts:
            continue

        ranked_candidates = sorted(
            ngram_counts.items(),
            key=lambda item: (
                item[1] * (1.0 + 0.15 * max(0, len(item[0].split()) - 1)),
                len(item[0]),
            ),
            reverse=True,
        )

        contribution_map: dict[str, _ChunkContribution] = {}

        top_candidates = ranked_candidates[:max_candidates_per_chunk]
        for phrase, count in top_candidates:
            tokens_for_phrase = phrase.split()
            word_count = len(tokens_for_phrase)
            chunk_score = count * (1.0 + 0.15 * max(0, word_count - 1))

            contribution = contribution_for(contribution_map, phrase, word_count)
            contribution.occurrences += count
            contribution.frequency_score += float(count)
            contribution.chunk_score += chunk_score

        rake_scores = {}
        rake_selected = 0
        if max_statistical_phrases_per_chunk > 0:
            stat_start = time.perf_counter()
            rake_scores = _extract_rake_scores(
                tokens,
                max_ngram_size=max_ngram_size,
                min_char_length=min_char_length,
            )
            timing_stats["statistical"] += time.perf_counter() - stat_start
            if rake_scores:
                statistical_stats["chunks"] += 1
                ranked_rake = sorted(rake_scores.items(), key=lambda item: item[1], reverse=True)
                if max_statistical_phrases_per_chunk > 0:
                    ranked_rake = ranked_rake[:max_statistical_phrases_per_chunk]
                for phrase, score in ranked_rake:
                    if score <= 0:
                        continue
                    word_count = len(phrase.split())
                    contribution = contribution_for(contribution_map, phrase, word_count)
                    contribution.statistical_score += float(score)
                    if contribution.occurrences == 0:
                        contribution.occurrences = ngram_counts.get(phrase, 0) or 1
                rake_selected = len(ranked_rake)
                statistical_stats["phrases"] += len(ranked_rake)

        embedding_candidates: list[str] = []
        if embedding_service and max_embedding_phrases_per_chunk > 0:
            if ranked_candidates:
                embedding_candidates.extend(
                    phrase for phrase, _ in ranked_candidates[:max_embedding_phrases_per_chunk]
                )
            if rake_scores:
                embedding_candidates.extend(
                    phrase
                    for phrase, _ in sorted(rake_scores.items(), key=lambda item: item[1], reverse=True)[
                        :max_embedding_phrases_per_chunk
                    ]
                )
            if embedding_candidates:
                seen: dict[str, None] = {}
                deduped: list[str] = []
                for phrase in embedding_candidates:
                    if phrase not in seen:
                        seen[phrase] = None
                        deduped.append(phrase)
                embedding_candidates = deduped[:max_embedding_phrases_per_chunk]

            text_for_embedding = chunk.normalized_text or chunk.text
            if embedding_candidates and text_for_embedding:
                embed_block_start = time.perf_counter()
                chunk_vector = chunk_embedding_cache.get(chunk_identifier)
                if chunk_vector is None:
                    try:
                        chunk_vector = embedding_service.embed_one(text_for_embedding)  # type: ignore[union-attr]
                        chunk_embedding_cache[chunk_identifier] = chunk_vector
                    except Exception as exc:  # pragma: no cover - defensive guard
                        logger.warning(
                            "keyword.embedding.chunk_failed chunk=%s error=%s",
                            chunk_identifier,
                            exc,
                        )
                        embedding_stats["errors"] += 1
                        chunk_vector = None
                if chunk_vector:
                    missing = [phrase for phrase in embedding_candidates if phrase not in embedding_cache]
                    if missing:
                        try:
                            vectors = embedding_service.embed(missing)  # type: ignore[union-attr]
                            for phrase, vector in zip(missing, vectors):
                                embedding_cache[phrase] = vector
                        except Exception as exc:  # pragma: no cover - defensive guard
                            logger.warning("keyword.embedding.phrase_failed error=%s", exc)
                            embedding_stats["errors"] += 1
                    embedding_stats["chunks"] += 1
                    embedding_stats["phrases"] += len(embedding_candidates)
                    for phrase in embedding_candidates:
                        vector = embedding_cache.get(phrase)
                        if vector is None:
                            continue
                        similarity = _cosine_similarity(chunk_vector, vector)
                        if similarity <= 0:
                            continue
                        contribution = contribution_for(contribution_map, phrase, len(phrase.split()))
                        contribution.embedding_vector = list(vector)
                        if embedding_backend_name:
                            contribution.embedding_backend = embedding_backend_name
                        contribution.embedding_score += similarity
                timing_stats["embedding"] += time.perf_counter() - embed_block_start

        if chunk.language:
            for contribution in contribution_map.values():
                contribution.languages.add(chunk.language)

        for phrase, contribution in contribution_map.items():
            apply_contribution(phrase, contribution, context)

        chunk_elapsed = time.perf_counter() - chunk_timer
        timing_stats["chunk_processing"] += chunk_elapsed
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "keyword.stage2.chunk done chunk=%s tokens=%s phrases=%s selected=%s rake=%s embed=%s elapsed=%.3fs",
                chunk_identifier,
                len(tokens),
                len(ngram_counts),
                len(top_candidates),
                rake_selected,
                len(embedding_candidates),
                chunk_elapsed,
            )

    summary_entries: list[dict[str, object]] = []
    summary_contexts: list[_ChunkContext] = []
    summary_debug: dict[str, object] | None = None
    llm_summary_used = False

    summary_callable = getattr(llm_filter, "extract_summary_concepts", None)
    should_use_summary = use_llm_summary if use_llm_summary is not None else True
    if (
        callable(summary_callable)
        and should_use_summary
        and chunk_contexts
        and getattr(llm_filter, "enabled", False)
        and llm_summary_max_chunks > 0
    ):
        summary_contexts = _select_summary_contexts(chunk_contexts, limit=min(llm_summary_max_chunks, len(chunk_contexts)))
        if summary_contexts:
            chunks_for_summary = [ctx.chunk for ctx in summary_contexts]
            summary_start = time.perf_counter()
            try:
                summary_entries = summary_callable(
                    chunks_for_summary,
                    max_results=llm_summary_max_results,
                    max_chars=llm_summary_max_chars,
                )
            except TypeError:
                summary_entries = summary_callable(chunks_for_summary)
            summary_elapsed = time.perf_counter() - summary_start
            timing_stats["llm_summary"] += summary_elapsed
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "keyword.stage2.summary invoked chunks=%s suggestions=%s elapsed=%.3fs",
                    len(chunks_for_summary),
                    len(summary_entries),
                    summary_elapsed,
                )
            summary_debug_fn = getattr(llm_filter, "summary_debug_payload", None)
            if callable(summary_debug_fn):
                summary_debug_payload = summary_debug_fn()
                if isinstance(summary_debug_payload, dict):
                    summary_debug = summary_debug_payload
        elif logger.isEnabledFor(logging.INFO):
            logger.info("keyword.stage2.summary skipped reason=no-context")
    elif logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "keyword.stage2.summary bypassed enabled=%s requested=%s contexts=%s chunks_limit=%s",
            getattr(llm_filter, "enabled", False),
            should_use_summary,
            bool(chunk_contexts),
            llm_summary_max_chunks,
        )

    if summary_entries:
        llm_summary_used = True
        for entry in summary_entries:
            phrase = str(entry.get("phrase", "")).strip()
            if not phrase:
                continue
            try:
                importance_value = entry.get("importance")
                llm_score_value = float(importance_value) if importance_value is not None else 1.0
            except (TypeError, ValueError):
                llm_score_value = 1.0
            reason_value = entry.get("reason")
            reason_text = str(reason_value).strip() if reason_value else None
            occurrence_value = _coerce_optional_int(entry.get("occurrences")) or 1
            sample_value = entry.get("sample") or entry.get("snippet") or entry.get("summary")
            sections_value = set(_coerce_str_list(entry.get("sections")))
            sources_value = set(_coerce_str_list(entry.get("sources")))
            languages_value = set(_coerce_str_list(entry.get("languages")))
            chunk_id_refs = set(_coerce_str_list(entry.get("chunk_ids")) or _coerce_str_list(entry.get("chunks")))

            target_contexts = summary_contexts or chunk_contexts[:1]
            if chunk_id_refs:
                filtered = [
                    ctx
                    for ctx in summary_contexts
                    if ctx.chunk_identifier in chunk_id_refs
                    or (ctx.chunk.chunk_id and ctx.chunk.chunk_id in chunk_id_refs)
                    or (ctx.doc_identifier in chunk_id_refs)
                ]
                if filtered:
                    target_contexts = filtered

            for idx, context in enumerate(target_contexts):
                contribution = _ChunkContribution(
                    occurrences=occurrence_value if idx == 0 else 0,
                    llm_score=float(llm_score_value),
                    reason=reason_text,
                    generated=True,
                    word_count=len(phrase.split()),
                    sample_snippet=str(sample_value).strip() if sample_value else None,
                    sections=sections_value.copy(),
                    sources=sources_value.copy(),
                    languages=languages_value.copy(),
                )
                apply_contribution(phrase, contribution, context)

    if embedding_service:
        parameters["embedding_stats"] = embedding_stats
    parameters["statistical_counts"] = statistical_stats
    parameters["llm_summary"] = {
        "used": llm_summary_used,
        "chunks": len(summary_contexts),
        "candidates": len(summary_entries),
        "max_chars": llm_summary_max_chars,
    }
    if summary_debug:
        parameters["llm_summary"]["debug"] = summary_debug

    candidates: List[ConceptCandidate] = []
    for phrase, aggregate in aggregates.items():
        if aggregate.occurrences < min_occurrences:
            continue
        doc_count = len(aggregate.doc_ids)
        chunk_count = len(aggregate.chunk_ids)
        if doc_count == 0:
            doc_count = chunk_count

        base_score = (
            aggregate.score_frequency
            + aggregate.score_chunk
            + aggregate.score_statistical * statistical_weight
            + aggregate.score_embedding * embedding_weight
            + aggregate.score_llm * llm_weight
            + doc_count * 1.5
        )
        length_bonus = 1.0 + 0.2 * max(0, aggregate.word_count - 1)
        score = base_score * length_bonus

        score_breakdown = {
            "frequency": round(aggregate.score_frequency, 3),
            "chunk_coverage": round(aggregate.score_chunk, 3),
            "document_coverage": round(doc_count * 1.5, 3),
            "length_bonus": round(length_bonus, 3),
        }
        if aggregate.score_statistical:
            score_breakdown["statistical"] = round(aggregate.score_statistical, 3)
        if aggregate.score_embedding:
            score_breakdown["embedding"] = round(aggregate.score_embedding, 3)
        if aggregate.score_llm:
            score_breakdown["llm"] = round(aggregate.score_llm, 3)

        languages = [language for language, _ in aggregate.languages.most_common()]
        sections = sorted(filter(None, aggregate.sections))[:5]
        sources = sorted(filter(None, aggregate.sources))[:5]
        sample_snippet = aggregate.sample_snippet

        if sample_snippet is None and sections:
            sample_snippet = sections[0]

        avg_occurrence = aggregate.occurrences / max(1, chunk_count)

        candidates.append(
            ConceptCandidate(
                phrase=phrase,
                score=score,
                occurrences=aggregate.occurrences,
                document_count=doc_count,
                chunk_count=chunk_count,
                average_occurrence_per_chunk=avg_occurrence,
                word_count=aggregate.word_count or len(phrase.split()),
                languages=languages,
                sections=sections,
                sources=sources,
                sample_snippet=sample_snippet,
                score_breakdown=score_breakdown,
                document_ids=sorted(aggregate.doc_ids),
                chunk_ids=sorted(aggregate.chunk_ids),
                reason=aggregate.reason,
                generated=aggregate.generated,
                embedding_vector=list(aggregate.embedding_vector) if aggregate.embedding_vector is not None else None,
                embedding_backend=aggregate.embedding_backend or embedding_backend_name,
            )
        )

    candidates.sort(key=concept_sort_key)

    total_elapsed = time.perf_counter() - total_start
    timing_stats["total"] = total_elapsed
    parameters["timing"] = {key: round(value, 4) for key, value in timing_stats.items()}

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "keyword.stage2.complete chunks=%s candidates=%s elapsed=%.3fs embedding_phrases=%s summary_used=%s",
            len(chunk_contexts),
            len(candidates),
            total_elapsed,
            embedding_stats.get("phrases") if embedding_service else 0,
            llm_summary_used,
        )

    return ConceptExtractionResult(
        candidates=candidates,
        total_chunks=len(chunks),
        total_tokens=total_tokens,
        parameters=parameters,
    )


def cluster_concepts(
    concepts: Sequence[ConceptCandidate],
    *,
    similarity_threshold: float = 0.35,
    max_clusters: int = 200,
    embedding_service: "EmbeddingService | None" = None,
    embedding_batch_size: int = 32,
    llm_filter: "KeywordLLMFilter | None" = None,
    llm_label_max_clusters: int = 6,
) -> ConceptClusteringResult:
    if not concepts:
        return ConceptClusteringResult(
            clusters=[],
            parameters={
                "similarity_threshold": similarity_threshold,
                "max_clusters": max_clusters,
                "embedding": {"enabled": False, "reason": "no-concepts"},
                "llm_labeling": {"enabled": bool(llm_filter and getattr(llm_filter, "enabled", False)), "used": False},
            },
        )

    parameters = {
        "similarity_threshold": similarity_threshold,
        "max_clusters": max_clusters,
    }

    concept_vectors: dict[str, Sequence[float]] = {}
    embedding_info: dict[str, Any] = {"enabled": False}
    precomputed_backend: str | None = None
    successes = 0
    failures = 0

    for concept in concepts:
        vector = getattr(concept, "embedding_vector", None)
        if vector:
            concept_vectors[concept.phrase] = vector
            successes += 1
            if not precomputed_backend:
                precomputed_backend = getattr(concept, "embedding_backend", None)

    service_backend_name: str | None = None
    if embedding_service:
        phrases = sorted({concept.phrase for concept in concepts if concept.phrase and concept.phrase not in concept_vectors})
        if phrases:
            backend = getattr(embedding_service, "backend", None)
            service_backend_name = getattr(backend, "name", None) or str(backend) if backend else None
            try:
                for start in range(0, len(phrases), max(1, embedding_batch_size)):
                    batch = phrases[start : start + max(1, embedding_batch_size)]
                    try:
                        vectors = embedding_service.embed(batch)  # type: ignore[union-attr]
                    except AttributeError:
                        vectors = [embedding_service.embed_one(text) for text in batch]  # type: ignore[union-attr]
                    for item, vector in zip(batch, vectors):
                        if vector and any(vector):
                            concept_vectors[item] = vector
                            successes += 1
                        else:
                            failures += 1
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("keyword.stage3.embedding_failed error=%s", exc)
                embedding_info.update({"enabled": False, "error": str(exc)})
                concept_vectors.clear()
                successes = 0
            else:
                if failures:
                    embedding_info["failures"] = failures

    embedding_backend_name = service_backend_name or precomputed_backend
    if successes:
        embedding_info["enabled"] = True
        embedding_info["phrases"] = successes
    if embedding_backend_name:
        embedding_info["backend"] = embedding_backend_name
    parameters["embedding"] = embedding_info
    llm_label_info: dict[str, Any] = {
        "enabled": bool(llm_filter and getattr(llm_filter, "enabled", False)),
        "requested": bool(
            llm_filter and getattr(llm_filter, "enabled", False) and llm_label_max_clusters > 0
        ),
        "used": False,
        "max_clusters": max(0, llm_label_max_clusters),
    }
    parameters["llm_labeling"] = llm_label_info

    builders: list[_ClusterBuilder] = []

    for concept in concepts:
        tokens = _phrase_tokens_for_clustering(concept.phrase)
        if not tokens:
            tokens = {concept.phrase.lower()}

        best_builder: _ClusterBuilder | None = None
        best_similarity = 0.0
        concept_vector = concept_vectors.get(concept.phrase)

        for builder in builders:
            lexical_similarity = max(
                _jaccard_similarity(tokens, builder.tokens),
                _token_overlap_similarity(tokens, builder.tokens),
            )
            embedding_similarity = (
                _cosine_similarity(concept_vector, builder.vector)  # type: ignore[arg-type]
                if concept_vector is not None and builder.vector is not None
                else 0.0
            )
            similarity = max(embedding_similarity, lexical_similarity)
            if similarity > best_similarity:
                best_similarity = similarity
                best_builder = builder

        if best_builder and best_similarity >= similarity_threshold:
            best_builder.add(concept, tokens=tokens, vector=concept_vector)
            continue

        if len(builders) >= max_clusters:
            target = best_builder or max(builders, key=lambda item: item.score)
            target.add(concept, tokens=tokens, vector=concept_vector)
            continue

        builder = _ClusterBuilder(
            label=concept.phrase,
            label_source="top-member",
            score=concept.score,
            occurrences=concept.occurrences,
            document_ids=set(concept.document_ids or []),
            chunk_ids=set(concept.chunk_ids or []),
            languages=Counter(concept.languages),
            sections=set(concept.sections),
            sources=set(concept.sources),
            members=[concept],
            tokens=set(tokens),
            vector=list(concept_vector) if concept_vector is not None else None,
            vector_count=1 if concept_vector is not None else 0,
            aliases=[concept.phrase],
        )
        builders.append(builder)

    clusters: list[ConceptCluster] = []
    for builder in builders:
        members = sorted(builder.members, key=lambda item: (-item.score, item.phrase))
        label = builder.label or members[0].phrase
        document_count = len(builder.document_ids) or max((member.document_count for member in members), default=0)
        chunk_count = len(builder.chunk_ids) or max((member.chunk_count for member in members), default=0)
        languages = [language for language, _ in builder.languages.most_common()]
        sections = sorted(builder.sections)[:5]
        sources = sorted(builder.sources)[:5]
        score_breakdown = {
            "member_count": len(members),
            "total_score": round(builder.score, 3),
            "avg_score": round(builder.score / max(1, len(members)), 3),
        }
        clusters.append(
            ConceptCluster(
                label=label,
                label_source=builder.label_source,
                score=builder.score,
                occurrences=builder.occurrences,
                document_count=document_count,
                chunk_count=chunk_count,
                languages=languages,
                sections=sections,
                sources=sources,
                members=members,
                score_breakdown=score_breakdown,
                description=builder.description,
                aliases=list(dict.fromkeys(builder.aliases or [label])),
            )
        )

    clusters.sort(
        key=lambda cluster: (
            -cluster.score,
            -cluster.document_count,
            -cluster.occurrences,
            cluster.label,
        )
    )

    if llm_filter and getattr(llm_filter, "enabled", False) and llm_label_max_clusters > 0 and clusters:
        target_clusters = clusters[: min(llm_label_max_clusters, len(clusters))]
        try:
            label_payload = llm_filter.label_clusters(target_clusters)  # type: ignore[arg-type]
        except TypeError:
            label_payload = llm_filter.label_clusters(target_clusters, max_clusters=llm_label_max_clusters)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("keyword.stage3.label_failed error=%s", exc)
            label_payload = []
            llm_label_info["error"] = str(exc)
        if label_payload:
            llm_label_info["used"] = True
            llm_label_info["labeled"] = len(label_payload)
            for entry in label_payload:
                index = _coerce_optional_int(entry.get("index"))
                if index is None or index <= 0 or index > len(target_clusters):
                    continue
                cluster = target_clusters[index - 1]
                label = str(entry.get("label", "")).strip()
                if label:
                    if cluster.label not in cluster.aliases:
                        cluster.aliases.append(cluster.label)
                    cluster.label = label
                    cluster.label_source = "llm"
                description = str(entry.get("description") or entry.get("reason") or "").strip() or None
                if description:
                    cluster.description = description
                aliases = _coerce_str_list(entry.get("aliases"))
                if aliases:
                    for alias in aliases:
                        if alias and alias not in cluster.aliases:
                            cluster.aliases.append(alias)
        elif llm_label_info["requested"]:
            llm_label_info.setdefault("status", "no-labels")

    return ConceptClusteringResult(clusters=clusters, parameters=parameters)


def rank_concept_clusters(
    clusters: Sequence[ConceptCluster],
    *,
    core_limit: int = 12,
    max_results: int = 50,
    score_weight: float = 1.0,
    document_weight: float = 2.0,
    chunk_weight: float = 0.5,
    occurrence_weight: float = 0.3,
    label_bonus: float = 0.5,
) -> ConceptRankingResult:
    if not clusters:
        return ConceptRankingResult(
            ranked=[],
            parameters={
                "core_limit": core_limit,
                "max_results": max_results,
                "weights": {
                    "score": score_weight,
                    "document": document_weight,
                    "chunk": chunk_weight,
                    "occurrence": occurrence_weight,
                    "label_bonus": label_bonus,
                },
            },
        )

    parameters = {
        "core_limit": core_limit,
        "max_results": max_results,
        "weights": {
            "score": score_weight,
            "document": document_weight,
            "chunk": chunk_weight,
            "occurrence": occurrence_weight,
            "label_bonus": label_bonus,
        },
    }

    ranked: list[RankedConcept] = []

    for cluster in clusters:
        component_scores: dict[str, float] = {}
        base_score = max(cluster.score, 0.0)
        component_scores["stage2_score"] = base_score * score_weight
        component_scores["document_coverage"] = cluster.document_count * document_weight
        component_scores["chunk_coverage"] = min(cluster.chunk_count, 12) * chunk_weight
        component_scores["occurrence"] = math.log1p(max(cluster.occurrences, 0)) * occurrence_weight
        if cluster.label_source == "llm":
            component_scores["label_bonus"] = label_bonus
        total_score = sum(component_scores.values())

        display_label, display_label_source, alias_candidates = _choose_cluster_display_label(cluster)
        alias_out = list(dict.fromkeys(alias_candidates + [cluster.label, display_label]))
        generated_flag = any(member.generated for member in cluster.members)

        ranked.append(
            RankedConcept(
                label=display_label,
                score=total_score,
                rank=0,
                is_core=False,
                document_count=cluster.document_count,
                chunk_count=cluster.chunk_count,
                occurrences=cluster.occurrences,
                label_source=display_label_source,
                description=cluster.description,
                aliases=alias_out,
                member_phrases=[member.phrase for member in cluster.members],
                score_breakdown={key: round(value, 3) for key, value in component_scores.items()},
                generated=generated_flag,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.score,
            -item.document_count,
            -item.occurrences,
            item.label,
        )
    )

    max_core = max(0, core_limit)
    max_items = max(1, max_results)
    trimmed = ranked[:max_items]
    for index, concept in enumerate(trimmed, start=1):
        concept.rank = index
        concept.is_core = index <= max_core

    return ConceptRankingResult(ranked=trimmed, parameters=parameters)


def extract_keyword_candidates(
    texts: Iterable[str],
    *,
    core_limit: int = 10,
    min_length: int = 3,
    min_count: int = 1,
) -> List[KeywordCandidate]:
    """Return ranked keyword candidates and flag the most frequent terms as core."""

    counter: Counter[str] = Counter()
    original_forms: dict[str, str] = {}

    for text in texts:
        if not text:
            continue
        for match in _WORD_RE.finditer(str(text)):
            token = match.group()
            normalized = token.lower()
            cleaned = normalized.strip("'-")
            if not cleaned:
                continue
            token_length = len(cleaned)
            if token_length < min_length:
                if _contains_hangul(cleaned):
                    if token_length < 2:
                        continue
                else:
                    continue
            if cleaned in _STOPWORDS:
                continue
            if cleaned.isdigit():
                continue
            if any(char.isdigit() for char in cleaned) and any(char.isalpha() for char in cleaned):
                continue
            counter[cleaned] += 1
            original_forms.setdefault(cleaned, token.strip("'-"))

    filtered = [item for item in counter.items() if item[1] >= min_count]
    filtered.sort(key=lambda item: (-item[1], item[0]))

    core_cutoff = min(core_limit, len(filtered))
    results: list[KeywordCandidate] = []
    for index, (term, count) in enumerate(filtered):
        # Only consider a keyword "core" if it appears more than once.
        is_core = index < core_cutoff and count > 1
        display_term = original_forms.get(term, term)
        results.append(KeywordCandidate(term=display_term, count=count, is_core=is_core))
    return results


class KeywordLLMFilter:
    """Use the configured chat backend to refine keyword candidates."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend: str | None = None
        self._enabled = False
        self._openai_client: OpenAI | None = None
        self._ollama_base_url: str | None = None
        self._ollama_model: str | None = None
        self._ollama_timeout: float = max(settings.ollama_request_timeout, 300.0)
        self._vllm_base_url: str | None = None
        self._vllm_model: str | None = None
        self._vllm_timeout: float = max(settings.vllm_request_timeout, 300.0)
        self._vllm_api_key: str | None = (settings.vllm_api_key or None)
        self._max_results: int = max(0, settings.keyword_filter_max_results)
        self._allow_generated: bool = settings.keyword_filter_allow_generated
        self._current_prompt: dict[str, str] | None = None
        self._disable_reason: str | None = None
        self._last_debug: dict[str, object] = {}
        self._last_concept_debug: dict[str, object] = {}
        self._last_summary_debug: dict[str, object] = {}
        self._last_cluster_debug: dict[str, object] = {}
        self._last_harmonize_debug: dict[str, object] = {}
        self._last_insight_debug: dict[str, object] = {}
        reason: str | None = None

        if settings.is_openai_chat_backend:
            if not settings.openai_api_key:
                reason = "missing-openai-key"
            elif OpenAI is None:
                reason = "openai-sdk-missing"
            elif settings.openai_chat_model:
                try:
                    self._openai_client = OpenAI(api_key=settings.openai_api_key)
                    self._backend = "openai"
                    self._enabled = True
                    logger.info("keyword.llm.backend_ready backend=openai model=%s", settings.openai_chat_model)
                except Exception as exc:  # pragma: no cover - runtime guard
                    logger.warning("keyword.llm.openai_init_failed error=%s", exc)
                    reason = f"openai-init-failed:{exc}"
            else:
                reason = "missing-openai-model"
        elif settings.is_ollama_chat_backend:
            self._ollama_base_url = settings.ollama_base_url.rstrip("/")
            self._ollama_model = settings.ollama_model
            if self._ollama_base_url and self._ollama_model:
                self._backend = "ollama"
                self._enabled = True
                logger.info(
                    "keyword.llm.backend_ready backend=ollama model=%s url=%s",
                    self._ollama_model,
                    self._ollama_base_url,
                )
            else:
                reason = "missing-ollama-config"
        elif settings.is_vllm_chat_backend:
            self._vllm_base_url = normalize_vllm_base_url(settings.vllm_base_url)
            self._vllm_model = settings.vllm_model
            if self._vllm_base_url and self._vllm_model:
                self._backend = "vllm"
                self._enabled = True
                logger.info(
                    "keyword.llm.backend_ready backend=vllm model=%s url=%s",
                    self._vllm_model,
                    self._vllm_base_url,
                )
            else:
                reason = "missing-vllm-config"
        else:
            reason = "chat-backend-disabled"

        if not self._enabled:
            logger.info("keyword.llm.disabled fallback=frequency reason=%s", reason)
            self._disable_reason = reason
            self._last_debug = {
                "status": "bypass",
                "reason": reason,
                "candidate_count": 0,
                "backend": self._backend,
                "enabled": self._enabled,
                "disable_reason": self._disable_reason,
            }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str | None:
        return self._backend

    def record_bypass(self, stage: str, reason: str, **extra: object) -> None:
        payload: dict[str, object] = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "status": "bypass",
            "reason": reason,
        }
        if extra:
            payload.update(extra)

        if stage == "filter":
            payload.setdefault("candidate_count", extra.get("candidate_count", 0) if extra else 0)
            payload.setdefault("max_results", self._max_results)
            payload.setdefault("allow_generated", self._allow_generated)
            self._last_debug = payload
        elif stage == "concept":
            payload.setdefault("candidate_count", extra.get("candidate_count", 0) if extra else 0)
            self._last_concept_debug = payload
        elif stage == "summary":
            self._last_summary_debug = payload
        elif stage == "cluster":
            self._last_cluster_debug = payload
        elif stage == "harmonize":
            self._last_harmonize_debug = payload
        elif stage == "insight":
            self._last_insight_debug = payload
        else:
            self._last_debug = payload

    def filter_keywords(
        self,
        candidates: Sequence[KeywordCandidate],
        context_snippets: Sequence[str],
    ) -> List[KeywordCandidate]:
        self._last_debug = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "candidate_count": len(candidates),
            "status": "pending",
            "max_results": self._max_results,
            "allow_generated": self._allow_generated,
        }
        if not candidates:
            logger.debug("keyword.llm.skip reason=no-candidates")
            self._last_debug["reason"] = "no-candidates"
            self._last_debug["status"] = "error"
            return []
        if not self._enabled:
            logger.debug(
                "keyword.llm.skip reason=disabled backend=%s candidate_count=%s",
                self._backend,
                len(candidates),
            )
            self._last_debug["reason"] = self._disable_reason or "disabled"
            self._last_debug["status"] = "bypass"
            return list(candidates)

        prompt = self._build_prompt(candidates, context_snippets)
        try:
            raw_response = self._invoke_backend(prompt)
            logger.debug("keyword.llm.response backend=%s text=%s", self._backend, raw_response[:500])
            self._last_debug["response"] = raw_response
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("keyword.llm.invoke_failed backend=%s error=%s", self._backend, exc)
            self._last_debug["reason"] = f"invoke_failed:{exc}"
            self._last_debug["status"] = "error"
            return list(candidates)

        parsed = self._parse_response(raw_response)
        if parsed is None:
            logger.debug("keyword.llm.parse_failed response=%s", raw_response)
            self._last_debug["reason"] = "json-parse-error"
            self._last_debug["status"] = "error"
            return list(candidates)

        keyword_items, coerce_note = self._extract_keyword_items(parsed)
        if keyword_items is None:
            logger.debug(
                "keyword.llm.malformed_response backend=%s payload=%s",
                self._backend,
                raw_response,
            )
            self._last_debug["reason"] = "missing-keywords-array"
            self._last_debug["status"] = "error"
            return list(candidates)
        if coerce_note:
            self._last_debug["coerced_source"] = coerce_note

        normalized_entries, rejected_entries, assumed_terms = self._normalize_keyword_items(keyword_items)
        if not normalized_entries:
            self._last_debug["reason"] = "no-keywords-retained"
            self._last_debug["status"] = "error"
            if rejected_entries:
                self._last_debug["rejected"] = rejected_entries[:25]
            if assumed_terms:
                self._last_debug["assumed_true"] = assumed_terms
            return list(candidates)

        term_to_candidate = {candidate.term.lower(): candidate for candidate in candidates}
        seen_terms: set[str] = set()
        selected_candidates: list[KeywordCandidate] = []
        dropped_generated: list[str] = []

        for entry in normalized_entries:
            term = str(entry.get("term", "")).strip()
            if not term:
                continue
            key = term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)

            source_value = str(entry.get("source") or "").strip()
            reason_value = entry.get("reason")
            count_value = entry.get("count")
            count_int: int | None
            if count_value is None:
                count_int = None
            else:
                try:
                    count_int = int(count_value)
                except (TypeError, ValueError):
                    count_int = None

            candidate = term_to_candidate.get(key)
            if candidate:
                final_source = source_value or "candidate"
                candidate.reason = reason_value or candidate.reason
                candidate.source = final_source or candidate.source
                candidate.generated = False
                selected_candidates.append(candidate)
            else:
                if not self._allow_generated:
                    dropped_generated.append(term)
                    continue
                final_source = source_value or "generated"
                count_for_new = count_int if count_int and count_int > 0 else 1
                new_candidate = KeywordCandidate(
                    term=term,
                    count=count_for_new,
                    is_core=True,
                    generated=True,
                    reason=reason_value,
                    source=final_source,
                )
                selected_candidates.append(new_candidate)

        if not selected_candidates:
            self._last_debug["reason"] = "no-keywords-retained"
            self._last_debug["status"] = "error"
            if rejected_entries:
                self._last_debug["rejected"] = rejected_entries[:25]
            if assumed_terms:
                self._last_debug["assumed_true"] = assumed_terms
            return list(candidates)

        truncated_terms: list[str] = []
        final_candidates = selected_candidates
        if self._max_results and len(selected_candidates) > self._max_results:
            truncated_terms = [item.term for item in selected_candidates[self._max_results:]]
            final_candidates = selected_candidates[: self._max_results]

        final_keys = {item.term.lower() for item in final_candidates}
        dropped = [candidate.term for candidate in candidates if candidate.term.lower() not in final_keys]
        kept_terms = [item.term for item in final_candidates if not item.generated]
        generated_terms = [item.term for item in final_candidates if item.generated]

        debug_limit = 25
        selected_debug = [
            {
                "term": item.term,
                "count": item.count,
                "source": getattr(item, "source", None),
                "reason": getattr(item, "reason", None),
                "generated": item.generated,
            }
            for item in final_candidates[:debug_limit]
        ]
        rejected_debug = rejected_entries[:debug_limit]

        self._last_debug.update(
            {
                "reason": "filtered",
                "status": "filtered",
                "kept": kept_terms,
                "generated": generated_terms,
                "dropped": dropped,
                "selected": selected_debug,
                "selected_total": len(final_candidates),
                "llm_selected_total": len(normalized_entries),
                "rejected": rejected_debug,
                "rejected_total": len(rejected_entries),
            }
        )
        if dropped_generated:
            self._last_debug["generated_blocked"] = dropped_generated[:20]
        if truncated_terms:
            self._last_debug["truncated"] = truncated_terms
        if assumed_terms:
            self._last_debug.setdefault("assumed_true", assumed_terms)

        logger.info(
            "keyword.llm.filter_result backend=%s kept=%s dropped=%s truncated=%s",
            self._backend,
            kept_terms + generated_terms,
            dropped,
            truncated_terms,
        )
        return final_candidates or list(candidates)

    def refine_concepts(
        self,
        concepts: Sequence[ConceptCandidate],
        context_snippets: Sequence[str],
        *,
        limit: int = 15,
    ) -> List[ConceptCandidate]:
        self._last_concept_debug = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "candidate_count": len(concepts),
            "limit": limit,
            "status": "pending",
        }
        if not concepts:
            self._last_concept_debug.update({"status": "skipped", "reason": "no-concepts"})
            return list(concepts)
        if not self._enabled:
            self._last_concept_debug.update(
                {"status": "bypass", "reason": self._disable_reason or "disabled"}
            )
            return list(concepts)

        limited_concepts = list(concepts[:limit])
        evaluated_keys = {concept.phrase.lower(): concept for concept in limited_concepts}

        prompt = self._build_concept_prompt(limited_concepts, context_snippets)
        try:
            raw_response = self._invoke_backend(prompt)
            self._last_concept_debug["response"] = raw_response
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("concept.llm.invoke_failed backend=%s error=%s", self._backend, exc)
            self._last_concept_debug.update({"status": "error", "reason": f"invoke_failed:{exc}"})
            return list(concepts)

        parsed = self._parse_response(raw_response)
        if parsed is None:
            self._last_concept_debug.update({"status": "error", "reason": "json-parse-error"})
            return list(concepts)

        concept_items, note = self._extract_concept_items(parsed)
        if concept_items is None:
            self._last_concept_debug.update({"status": "error", "reason": "missing-concepts-key"})
            return list(concepts)

        normalized_items, rejected_items = self._normalize_concept_items(concept_items)
        if note:
            self._last_concept_debug["note"] = note
        self._last_concept_debug["llm_entries_total"] = len(normalized_items)

        concept_map = {concept.phrase.lower(): concept for concept in limited_concepts}
        refined: list[ConceptCandidate] = []
        dropped: list[str] = []
        kept: list[str] = []
        renamed: list[dict[str, str]] = []
        generated: list[str] = []
        seen: set[str] = set()

        for entry in normalized_items:
            phrase = str(entry.get("phrase", "")).strip()
            if not phrase:
                continue
            keep_flag = bool(entry.get("keep", True))
            label = str(entry.get("label", "")).strip()
            new_phrase = label or phrase
            importance = _coerce_float(entry.get("importance"))
            reason = str(entry.get("reason", "")).strip() or None
            generated_flag = bool(entry.get("generated", False))
            sections = _coerce_str_list(entry.get("sections"))
            languages = _coerce_str_list(entry.get("languages"))
            sources = _coerce_str_list(entry.get("sources"))
            occurrences = _coerce_optional_int(entry.get("occurrences")) or 1
            document_count = _coerce_optional_int(entry.get("documents")) or _coerce_optional_int(
                entry.get("docs")
            )
            chunk_count = _coerce_optional_int(entry.get("chunks")) or _coerce_optional_int(
                entry.get("chunk_count")
            )
            if chunk_count is None:
                chunk_count = occurrences
            if document_count is None:
                document_count = max(1, min(occurrences, chunk_count))

            phrase_key = phrase.lower()
            base_candidate = concept_map.get(phrase_key)

            if not keep_flag:
                dropped.append(phrase)
                continue

            if base_candidate is None and phrase_key in concept_map:
                base_candidate = concept_map[phrase_key]

            if base_candidate is not None:
                updated_phrase = new_phrase or base_candidate.phrase
                breakdown = dict(base_candidate.score_breakdown)
                new_score = base_candidate.score
                if importance is not None:
                    breakdown["llm_importance"] = round(importance, 3)
                    new_score = max(new_score, float(importance))
                updated_candidate = replace(
                    base_candidate,
                    phrase=updated_phrase,
                    score=new_score,
                    score_breakdown=breakdown,
                    reason=reason or base_candidate.reason,
                    generated=False,
                )
                if sections:
                    updated_candidate.sections = sections
                if languages:
                    updated_candidate.languages = languages
                if sources:
                    updated_candidate.sources = sources
                kept.append(updated_phrase)
                if updated_phrase.lower() != base_candidate.phrase.lower():
                    renamed.append({"from": base_candidate.phrase, "to": updated_phrase})
                normalized_key = updated_candidate.phrase.lower()
                if normalized_key not in seen:
                    refined.append(updated_candidate)
                    seen.add(normalized_key)
                continue

            # New concept suggested by the LLM.
            word_count = max(1, len(new_phrase.split()))
            new_score = importance if importance is not None else float(occurrences)
            avg_per_chunk = occurrences / max(1, chunk_count)
            breakdown = {"llm_importance": round(new_score, 3)}
            candidate = ConceptCandidate(
                phrase=new_phrase,
                score=new_score,
                occurrences=max(1, occurrences),
                document_count=max(1, document_count),
                chunk_count=max(1, chunk_count),
                average_occurrence_per_chunk=avg_per_chunk,
                word_count=word_count,
                languages=languages or ["unknown"],
                sections=sections[:5],
                sources=sources[:5],
                sample_snippet=None,
                score_breakdown=breakdown,
                reason=reason,
                generated=True,
            )
            normalized_key = candidate.phrase.lower()
            if normalized_key not in seen:
                refined.append(candidate)
                seen.add(normalized_key)
                generated.append(candidate.phrase)

        if not refined:
            self._last_concept_debug.update(
                {
                    "status": "error",
                    "reason": "no-concepts-retained",
                    "rejected": rejected_items[:20],
                    "note": note,
                }
            )
            return list(concepts)

        unreviewed = [
            concept
            for concept in concepts
            if concept.phrase.lower() not in evaluated_keys or concept.phrase.lower() in seen
        ]

        combined = refined + [candidate for candidate in unreviewed if candidate.phrase.lower() not in seen]
        combined.sort(key=concept_sort_key)

        self._last_concept_debug.update(
            {
                "status": "refined",
                "kept": kept,
                "generated": generated,
                "dropped": dropped,
                "renamed": renamed,
                "note": note,
                "rejected": rejected_items[:20],
                "rejected_total": len(rejected_items),
                "selected_total": len(refined),
                "final_total": len(combined),
            }
        )

        return combined

    def label_clusters(
        self,
        clusters: Sequence[ConceptCluster],
        *,
        max_clusters: int = 6,
    ) -> list[dict[str, object]]:
        self._last_cluster_debug = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "cluster_count": len(clusters),
            "max_clusters": max_clusters,
            "status": "pending",
        }
        if not clusters:
            self._last_cluster_debug.update({"status": "skipped", "reason": "no-clusters"})
            return []
        if not self._enabled:
            self._last_cluster_debug.update(
                {"status": "bypass", "reason": self._disable_reason or "disabled"}
            )
            return []

        limited_clusters = list(clusters[: max(1, max_clusters)])
        prompt = self._build_cluster_prompt(limited_clusters)
        try:
            raw_response = self._invoke_backend(prompt)
            self._last_cluster_debug["response"] = raw_response
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("concept.llm.cluster_label_failed backend=%s error=%s", self._backend, exc)
            self._last_cluster_debug.update({"status": "error", "reason": f"invoke_failed:{exc}"})
            return []

        parsed = self._parse_response(raw_response)
        if parsed is None:
            self._last_cluster_debug.update({"status": "error", "reason": "json-parse-error"})
            return []

        cluster_items, note = self._extract_cluster_items(parsed)
        if cluster_items is None:
            self._last_cluster_debug.update({"status": "error", "reason": "missing-clusters-key"})
            return []

        normalized_items, rejected_items = self._normalize_cluster_items(cluster_items)
        if note:
            self._last_cluster_debug["note"] = note
        if rejected_items:
            self._last_cluster_debug["rejected"] = rejected_items[:20]
            self._last_cluster_debug["rejected_total"] = len(rejected_items)

        if not normalized_items:
            self._last_cluster_debug.update({"status": "empty"})
            return []

        self._last_cluster_debug.update(
            {
                "status": "labeled",
                "selected_total": len(normalized_items),
            }
        )
        self._last_cluster_debug["labels"] = [
            {
                "index": item.get("index"),
                "label": item.get("label"),
                "description": item.get("description"),
            }
            for item in normalized_items[:20]
        ]
        return normalized_items

    def harmonize_ranked_concepts(
        self,
        concepts: Sequence[RankedConcept],
        *,
        max_results: int = 10,
    ) -> list[RankedConcept]:
        self._last_harmonize_debug = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "concept_count": len(concepts),
            "max_results": max_results,
            "status": "pending",
        }
        if not concepts:
            self._last_harmonize_debug.update({"status": "skipped", "reason": "no-concepts"})
            return list(concepts)
        if not self._enabled:
            self._last_harmonize_debug.update(
                {"status": "bypass", "reason": self._disable_reason or "disabled"}
            )
            return list(concepts)

        limited_concepts = list(concepts[: max(1, max_results)])
        prompt = self._build_harmonize_prompt(limited_concepts)
        try:
            raw_response = self._invoke_backend(prompt)
            self._last_harmonize_debug["response"] = raw_response
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("concept.llm.harmonize_failed backend=%s error=%s", self._backend, exc)
            self._last_harmonize_debug.update({"status": "error", "reason": f"invoke_failed:{exc}"})
            return list(concepts)

        parsed = self._parse_response(raw_response)
        if parsed is None:
            self._last_harmonize_debug.update({"status": "error", "reason": "json-parse-error"})
            return list(concepts)

        items, note = self._extract_harmonize_items(parsed)
        if items is None:
            self._last_harmonize_debug.update({"status": "error", "reason": "missing-concepts-key"})
            return list(concepts)

        normalized_items, rejected_items = self._normalize_harmonize_items(items)
        if note:
            self._last_harmonize_debug["note"] = note
        if rejected_items:
            self._last_harmonize_debug["rejected"] = rejected_items[:20]
            self._last_harmonize_debug["rejected_total"] = len(rejected_items)

        if not normalized_items:
            self._last_harmonize_debug.update({"status": "empty"})
            return list(concepts)

        updated: list[RankedConcept] = list(concepts)
        applied = 0
        for entry in normalized_items:
            index = _coerce_optional_int(entry.get("index"))
            if index is None or index <= 0 or index > len(updated):
                continue
            base = updated[index - 1]
            if not entry.get("keep", True):
                continue
            new_label = str(entry.get("label", "")).strip() or base.label
            description = str(entry.get("description", "")).strip() or base.description
            alias_candidates = _coerce_str_list(entry.get("aliases"))
            merged_aliases = list(
                dict.fromkeys([new_label, base.label, *alias_candidates, *base.aliases])
            )
            label_source = (
                f"{base.label_source}+harmonized"
                if new_label != base.label
                else base.label_source
            )
            updated[index - 1] = replace(
                base,
                label=new_label,
                description=description,
                aliases=merged_aliases,
                label_source=label_source,
            )
            applied += 1

        if applied == 0:
            self._last_harmonize_debug.update({"status": "no-changes"})
            return list(concepts)

        self._last_harmonize_debug.update(
            {
                "status": "harmonized",
                "selected_total": applied,
            }
        )
        preview = [
            {
                "index": concept.rank,
                "label": concept.label,
                "label_source": concept.label_source,
            }
            for concept in updated[: min(applied, 10)]
        ]
        self._last_harmonize_debug["labels"] = preview
        return updated

    def summarize_keywords(
        self,
        keywords: Sequence[KeywordCandidate],
        *,
        max_insights: int = 3,
        max_concepts: int = 8,
        context_snippets: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        self._last_insight_debug = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "max_insights": max(0, max_insights),
            "max_concepts": max(0, max_concepts),
            "status": "pending",
        }
        if not keywords:
            self._last_insight_debug.update({"status": "skipped", "reason": "no-keywords"})
            return []
        if not self._enabled:
            self._last_insight_debug.update(
                {"status": "bypass", "reason": self._disable_reason or "disabled"}
            )
            return []

        limit_concepts = max(1, max_concepts)
        limited_keywords = list(keywords[:limit_concepts])
        self._last_insight_debug["concept_count"] = len(limited_keywords)
        prompt = self._build_insight_prompt(
            limited_keywords,
            context_snippets or [],
            max_insights=max_insights,
        )
        try:
            raw_response = self._invoke_backend(prompt)
            self._last_insight_debug["response"] = raw_response
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("concept.llm.insight_failed backend=%s error=%s", self._backend, exc)
            self._last_insight_debug.update({"status": "error", "reason": f"invoke_failed:{exc}"})
            return []

        parsed = self._parse_response(raw_response)
        if parsed is None:
            self._last_insight_debug.update({"status": "error", "reason": "json-parse-error"})
            return []

        items, note = self._extract_insight_items(parsed)
        if items is None:
            self._last_insight_debug.update({"status": "error", "reason": "missing-insights-key"})
            return []

        normalized_items, rejected_items = self._normalize_insight_items(items, max_items=max_insights)
        if note:
            self._last_insight_debug["note"] = note
        if rejected_items:
            self._last_insight_debug["rejected"] = rejected_items[:10]
            self._last_insight_debug["rejected_total"] = len(rejected_items)

        if not normalized_items:
            self._last_insight_debug.update({"status": "empty"})
            return []

        insights = normalized_items[: max(1, max_insights)] if max_insights else normalized_items
        self._last_insight_debug.update(
            {
                "status": "success",
                "selected_total": len(insights),
                "insights": insights[:5],
            }
        )
        return insights

    def debug_payload(self) -> dict[str, object]:
        payload = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "max_results": self._max_results,
            "allow_generated": self._allow_generated,
        }
        payload.update(self._last_debug)
        if self._last_concept_debug:
            payload.setdefault("concept_refinement", self._last_concept_debug)
        if self._last_summary_debug:
            payload.setdefault("concept_summary", self._last_summary_debug)
        if getattr(self, "_last_cluster_debug", None):
            payload.setdefault("cluster_labeling", self._last_cluster_debug)
        if getattr(self, "_last_harmonize_debug", None):
            payload.setdefault("label_harmonization", self._last_harmonize_debug)
        if getattr(self, "_last_insight_debug", None):
            payload.setdefault("insight_summary", self._last_insight_debug)
        return payload

    def concept_debug_payload(self) -> dict[str, object]:
        return dict(self._last_concept_debug)

    def summary_debug_payload(self) -> dict[str, object]:
        return dict(self._last_summary_debug)

    def cluster_debug_payload(self) -> dict[str, object]:
        return dict(self._last_cluster_debug)

    def harmonize_debug_payload(self) -> dict[str, object]:
        return dict(self._last_harmonize_debug)

    def insight_debug_payload(self) -> dict[str, object]:
        return dict(self._last_insight_debug)

    def _extract_insight_items(self, payload: object) -> tuple[list[object] | None, str | None]:
        def _as_list(value: object) -> list[object] | None:
            if value is None:
                return None
            if isinstance(value, list):
                return list(value)
            if isinstance(value, (tuple, set)):
                return list(value)
            if isinstance(value, dict):
                return [value]
            if isinstance(value, str):
                items = [segment.strip() for segment in re.split(r"(?:\n|;)", value) if segment.strip()]
                if items:
                    return items
            return None

        note: str | None = None
        if isinstance(payload, dict):
            value: object | None = None
            for key in ("insights", "summary", "findings", "highlights"):
                if key in payload:
                    value = payload[key]
                    if key != "insights":
                        note = f"coerced-from-{key}"
                    break
            result = _as_list(value)
            if result is not None:
                if note is None and value is not None and not isinstance(value, list):
                    note = "coerced-from-nonlist"
                return result, note
        elif isinstance(payload, list):
            return list(payload), "coerced-from-root-list"
        elif isinstance(payload, str):
            result = _as_list(payload)
            if result:
                return result, "coerced-from-string"
        return None, note

    def _normalize_insight_items(
        self, items: Sequence[object], *, max_items: int
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        normalized: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []

        for raw in items:
            if isinstance(raw, dict):
                title_raw = raw.get("title") or raw.get("theme") or raw.get("headline")
                summary_raw = raw.get("summary") or raw.get("insight") or raw.get("description")
                keywords_raw = raw.get("keywords") or raw.get("concepts") or raw.get("terms")
                evidence_raw = raw.get("evidence") or raw.get("support") or raw.get("examples")
                action_raw = raw.get("action") or raw.get("recommendation") or raw.get("next_step")
                priority_raw = raw.get("priority") or raw.get("impact") or raw.get("severity")

                title = str(title_raw).strip() if title_raw is not None else ""
                summary = str(summary_raw).strip() if summary_raw is not None else ""
                keywords = _coerce_str_list(keywords_raw)
                evidence = _coerce_str_list(evidence_raw)
                action = str(action_raw).strip() if action_raw else ""
                priority = str(priority_raw).strip() if priority_raw else ""

                if not title and summary:
                    title = summary.split(". ", 1)[0][:120]
                if not summary and title:
                    summary = title
                if not (title and summary):
                    rejected.append({"reason": "missing-title-or-summary", "raw": raw})
                    continue

                entry: dict[str, object] = {
                    "title": title,
                    "summary": summary,
                    "keywords": keywords,
                }
                if action:
                    entry["action"] = action
                if priority:
                    entry["priority"] = priority
                if evidence:
                    entry["evidence"] = evidence
                normalized.append(entry)
            elif isinstance(raw, str):
                text = raw.strip()
                if not text:
                    continue
                entry = {
                    "title": text[:80],
                    "summary": text,
                    "keywords": [],
                }
                normalized.append(entry)
            else:
                rejected.append({"reason": "unsupported-entry", "raw": raw})

        if max_items > 0:
            normalized = normalized[:max_items]
        return normalized, rejected

    def _build_insight_prompt(
        self,
        keywords: Sequence[KeywordCandidate],
        context_snippets: Sequence[str],
        *,
        max_insights: int,
    ) -> str:
        keyword_lines = []
        for index, keyword in enumerate(keywords, 1):
            detail_parts = [
                f"core={'yes' if keyword.is_core else 'no'}",
                f"generated={'yes' if keyword.generated else 'no'}",
                f"count={keyword.count}",
            ]
            if keyword.reason:
                detail_parts.append(f"reason={keyword.reason}")
            keyword_lines.append(f"{index}. term='{keyword.term}' | " + " | ".join(detail_parts))
        formatted_keywords = "\n".join(keyword_lines) if keyword_lines else "(no keywords)"

        context_lines = []
        for idx, snippet in enumerate(context_snippets[: max(0, 4)], 1):
            context_lines.append(f"{idx}. {snippet[:320]}")
        formatted_context = "\n".join(context_lines) if context_lines else "(context optional)"

        instructions = (
            "You are a knowledge analyst. Using the ranked keywords and context excerpts, synthesize "
            f"up to {max(1, max_insights)} high-level insights. Each insight should capture the business meaning, "
            "note why it matters, and optionally suggest an action or follow-up question."
        )

        schema = (
            "Return strict JSON with an 'insights' array: "
            '{"insights": [{"title": "...", "summary": "...", "keywords": ["..."], '
            '"action": "...", "priority": "high|medium|low", "evidence": ["..."]}]}. '
            "Omit optional fields if they are not relevant."
        )

        prompt = (
            f"{instructions}\n{schema}\n\nKEYWORDS:\n{formatted_keywords}\n\nCONTEXT SNIPPETS:\n{formatted_context}"
        )

        self._current_prompt = {
            "system": (
                "You are an expert insights analyst who writes concise, neutral summaries."
            ),
            "user": prompt,
        }
        return prompt

    @staticmethod
    def _parse_response(raw_response: str) -> dict | None:
        decoder = json.JSONDecoder()

        def _attempt(candidate: str) -> dict | None:
            candidate = candidate.strip()
            if not candidate:
                return None
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            for token in ("{", "["):
                idx = candidate.find(token)
                while idx != -1:
                    try:
                        parsed, _ = decoder.raw_decode(candidate[idx:])
                        return parsed
                    except json.JSONDecodeError:
                        idx = candidate.find(token, idx + 1)
            return None

        primary = _attempt(raw_response)
        if primary is not None:
            return primary

        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL)
        if fenced_match:
            fenced_content = fenced_match.group(1)
            return _attempt(fenced_content) or None

        return None

    def _extract_keyword_items(self, payload: object) -> tuple[list[object] | None, str | None]:
        """Return keyword entries from the LLM payload, coercing common variants."""

        def _as_list(value: object) -> list[object] | None:
            if value is None:
                return None
            if isinstance(value, list):
                return list(value)
            if isinstance(value, (tuple, set)):
                return list(value)
            if isinstance(value, dict):
                return [value]
            if isinstance(value, str):
                items = [segment.strip() for segment in re.split(r"[\n,]", value) if segment.strip()]
                if items:
                    return items
            return None

        note: str | None = None
        if isinstance(payload, dict):
            value: object | None = payload.get("keywords")
            if value is None:
                for key in ("items", "terms", "results", "keywords_list", "values", "data"):
                    if key in payload:
                        value = payload[key]
                        note = f"coerced-from-{key}"
                        break
            result = _as_list(value)
            if result is not None:
                if note is None and value is not None and not isinstance(value, list):
                    note = "coerced-from-nonlist"
                return result, note
        elif isinstance(payload, list):
            return list(payload), "coerced-from-root-list"
        elif isinstance(payload, str):
            result = _as_list(payload)
            if result:
                return result, "coerced-from-string"
        return None, note

    def _normalize_keyword_items(
        self, items: Sequence[object]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        normalized: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        assumed_true: list[str] = []

        for raw in items:
            entry = self._normalize_keyword_entry(raw)
            if not entry:
                continue
            keep = bool(entry.get("keep", True))
            if keep:
                if entry.pop("_assumed_keep", False):
                    assumed_true.append(str(entry.get("term", "")))
                normalized.append(entry)
            else:
                entry.pop("_assumed_keep", None)
                rejected.append(entry)
        return normalized, rejected, [term for term in assumed_true if term]

    def _normalize_keyword_entry(self, raw: object) -> dict[str, object] | None:
        if isinstance(raw, str):
            term = raw.strip()
            if not term:
                return None
            return {"term": term, "keep": True, "_assumed_keep": True}

        if isinstance(raw, dict):
            term: str = ""
            for key in ("term", "keyword", "value", "text", "name"):
                value = raw.get(key)
                if value is not None:
                    term = str(value).strip()
                    if term:
                        break
            if not term:
                return None

            keep_flag: bool | None = None
            for key in ("keep", "include", "accept", "selected", "retain", "use", "should_keep"):
                if key in raw:
                    keep_flag = self._coerce_keep_flag(raw.get(key))
                    if keep_flag is not None:
                        break
            assumed = False
            if keep_flag is None:
                keep_flag = True
                assumed = True

            reason: str | None = None
            for key in ("reason", "note", "explanation", "why"):
                value = raw.get(key)
                if value is not None:
                    text_value = str(value).strip()
                    if text_value:
                        reason = text_value
                        break

            source: str | None = None
            for key in ("source", "origin"):
                value = raw.get(key)
                if value is not None:
                    text_value = str(value).strip()
                    if text_value:
                        source = text_value
                        break

            count_value = None
            for key in ("count", "frequency", "freq", "score", "weight"):
                if key in raw:
                    count_value = raw.get(key)
                    break
            count_int: int | None
            if count_value is None:
                count_int = None
            else:
                try:
                    count_int = int(count_value)
                except (TypeError, ValueError):
                    count_int = None

            entry: dict[str, object] = {"term": term, "keep": bool(keep_flag)}
            if reason is not None:
                entry["reason"] = reason
            if source is not None:
                entry["source"] = source
            if count_int is not None:
                entry["count"] = count_int
            if assumed:
                entry["_assumed_keep"] = True
            return entry

        if isinstance(raw, (tuple, set)):
            items = list(raw)
            if not items:
                return None
            return self._normalize_keyword_entry(items[0])

        return None

    @staticmethod
    def _coerce_keep_flag(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = str(value).strip().lower()
        if not text:
            return None
        if text in {"true", "yes", "keep", "include", "retain", "accept", "1"}:
            return True
        if text in {"false", "no", "drop", "reject", "exclude", "remove", "0"}:
            return False
        return None

    def _extract_harmonize_items(self, payload: object) -> tuple[list[object] | None, str | None]:
        def _as_list(value: object) -> list[object] | None:
            if value is None:
                return None
            if isinstance(value, list):
                return list(value)
            if isinstance(value, (tuple, set)):
                return list(value)
            if isinstance(value, dict):
                return [value]
            if isinstance(value, str):
                items = [segment.strip() for segment in re.split(r"(?:\n|,)", value) if segment.strip()]
                if items:
                    return items
            return None

        note: str | None = None
        if isinstance(payload, dict):
            value: object | None = None
            for key in ("concepts", "items", "results", "labels"):
                if key in payload:
                    value = payload[key]
                    if key != "concepts":
                        note = f"coerced-from-{key}"
                    break
            result = _as_list(value)
            if result is not None:
                if note is None and value is not None and not isinstance(value, list):
                    note = "coerced-from-nonlist"
                return result, note
        elif isinstance(payload, list):
            return list(payload), "coerced-from-root-list"
        elif isinstance(payload, str):
            result = _as_list(payload)
            if result:
                return result, "coerced-from-string"
        return None, note

    def _normalize_harmonize_items(self, items: Sequence[object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        normalized: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []

        for raw in items:
            if isinstance(raw, dict):
                index = _coerce_optional_int(raw.get("index") or raw.get("rank"))
                if index is None or index <= 0:
                    rejected.append({"reason": "invalid-index", "raw": raw})
                    continue
                label_value = raw.get("label")
                label = str(label_value).strip() if label_value is not None else ""
                description_value = raw.get("description") or raw.get("reason")
                description = str(description_value).strip() if description_value else ""
                aliases = _coerce_str_list(raw.get("aliases") or raw.get("alternate") or raw.get("synonyms"))
                keep_value = raw.get("keep")
                keep_flag = self._coerce_keep_flag(keep_value)
                keep = True if keep_flag is None else keep_flag
                entry: dict[str, object] = {
                    "index": index,
                    "label": label,
                    "description": description,
                    "aliases": aliases,
                    "keep": keep,
                }
                normalized.append(entry)
            elif isinstance(raw, str):
                label = raw.strip()
                if not label:
                    continue
                normalized.append({"index": None, "label": label, "description": "", "aliases": [], "keep": False})
            else:
                rejected.append({"reason": "unsupported-entry", "raw": raw})

        return normalized, rejected

    def extract_summary_concepts(
        self,
        chunks: Sequence[KeywordSourceChunk],
        *,
        max_results: int = 10,
        max_chars: int = 320,
    ) -> List[dict[str, object]]:
        self._last_summary_debug = {
            "backend": self._backend,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "chunk_count": len(chunks),
            "status": "pending",
            "max_results": max_results,
        }
        if not chunks:
            self._last_summary_debug.update({"status": "skipped", "reason": "no-chunks"})
            return []
        if not self._enabled:
            self._last_summary_debug.update(
                {"status": "bypass", "reason": self._disable_reason or "disabled"}
            )
            return []

        prompt = self._build_summary_prompt(chunks, max_results=max_results, max_chars=max_chars)
        try:
            raw_response = self._invoke_backend(prompt)
            self._last_summary_debug["response"] = raw_response
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("concept.llm.summary_invoke_failed backend=%s error=%s", self._backend, exc)
            self._last_summary_debug.update({"status": "error", "reason": f"invoke_failed:{exc}"})
            return []

        parsed = self._parse_response(raw_response)
        if parsed is None:
            self._last_summary_debug.update({"status": "error", "reason": "json-parse-error"})
            return []

        concept_items, note = self._extract_concept_items(parsed)
        if concept_items is None:
            self._last_summary_debug.update({"status": "error", "reason": "missing-concepts-key"})
            return []

        normalized, rejected = self._normalize_concept_items(concept_items)
        if note:
            self._last_summary_debug["note"] = note
        if rejected:
            self._last_summary_debug["rejected"] = rejected[:20]
            self._last_summary_debug["rejected_total"] = len(rejected)

        trimmed = normalized[:max_results]
        results: list[dict[str, object]] = []
        for entry in trimmed:
            phrase = str(entry.get("phrase", "")).strip()
            if not phrase:
                continue
            results.append(
                {
                    "phrase": phrase,
                    "importance": entry.get("importance"),
                    "reason": entry.get("reason"),
                    "chunk_ids": _coerce_str_list(entry.get("chunks")) or _coerce_str_list(entry.get("chunk_ids")),
                    "doc_ids": _coerce_str_list(entry.get("documents")) or _coerce_str_list(entry.get("docs")),
                    "sections": _coerce_str_list(entry.get("sections")),
                    "sources": _coerce_str_list(entry.get("sources")),
                    "languages": _coerce_str_list(entry.get("languages")),
                    "sample": entry.get("sample") or entry.get("snippet") or entry.get("summary"),
                    "occurrences": entry.get("occurrences"),
                }
            )

        if not results:
            self._last_summary_debug.update({"status": "error", "reason": "no-concepts-retained"})
            return []

        self._last_summary_debug.update(
            {
                "status": "success",
                "selected_total": len(results),
            }
        )
        return results

    def _build_summary_prompt(
        self,
        chunks: Sequence[KeywordSourceChunk],
        *,
        max_results: int,
        max_chars: int,
    ) -> str:
        excerpt_lines = []
        for index, chunk in enumerate(chunks, 1):
            chunk_id = chunk.chunk_id or f"chunk-{index}"
            doc_id = chunk.doc_id or f"doc-{index}"
            section = chunk.section_path or chunk.title or chunk.source or "general"
            language = chunk.language or "unknown"
            excerpt_lines.append(
                f"{index}. chunk_id={chunk_id} | doc_id={doc_id} | section={section} | language={language}\n"
                f"{chunk.excerpt(max_chars=max_chars)}"
            )
        excerpts_formatted = "\n\n".join(excerpt_lines)

        instructions = (
            "You are an expert knowledge analyst identifying key domain concepts from documentation excerpts. "
            "Focus on concise noun phrases (2-5 words) that represent product names, features, issues, or procedures. "
            "Avoid generic words like 'issue' or 'system'. Preserve any Korean phrases exactly as written."
        )

        prompt = (
            f"{instructions}\n\n"
            f"Return at most {max_results} concepts in strict JSON format:\n"
            '{"concepts": [{"phrase": "...", "reason": "...", "importance": number, "chunks": ["chunk-id"], "sections": ["..."]}]}\n'
            "Include optional 'importance' (higher is more important) and reference chunk IDs when relevant. "
            "If multiple chunks mention the same concept, list their chunk IDs. Use short justifications."
            "\n\nEXCERPTS:\n"
            f"{excerpts_formatted}"
        )

        self._current_prompt = {
            "system": instructions,
            "user": prompt,
        }
        return prompt

    def _build_harmonize_prompt(
        self,
        concepts: Sequence[RankedConcept],
    ) -> str:
        lines: list[str] = []
        for index, concept in enumerate(concepts, 1):
            alias_preview = ", ".join(concept.aliases[:5]) if concept.aliases else "n/a"
            member_preview = ", ".join(concept.member_phrases[:4]) if concept.member_phrases else "n/a"
            lines.append(
                f"{index}. label='{concept.label}' | docs={concept.document_count} | "
                f"chunks={concept.chunk_count} | occurrences={concept.occurrences}\n"
                f"   aliases: {alias_preview}\n"
                f"   members: {member_preview}"
            )
        formatted = "\n\n".join(lines)

        instructions = (
            "You are an expert knowledge taxonomist. For each concept, produce a concise, neutral label that "
            "captures the shared meaning of its aliases. Prefer short noun phrases (2-4 words) and keep Korean text "
            "in Hangul. Remove sentiment qualifiers like 'positive' or 'negative' unless they are the core idea."
        )

        prompt = (
            f"{instructions}\n\n"
            "Return strict JSON:\n"
            '{"concepts": [{"index": 1, "label": "...", "description": "...", "aliases": ["..."], "keep": true}]}\n'
            "Use the provided index to reference each concept. Include 'description' only when it adds clarity. "
            "List alternative aliases if they help clarify the theme. Set keep=false only if a concept label should be ignored."
            "\n\nCONCEPTS:\n"
            f"{formatted}"
        )

        self._current_prompt = {
            "system": instructions,
            "user": prompt,
        }
        return prompt

    def _build_concept_prompt(
        self,
        concepts: Sequence[ConceptCandidate],
        context_snippets: Sequence[str],
    ) -> str:
        concept_lines = []
        for index, concept in enumerate(concepts, 1):
            sections = ", ".join(concept.sections[:2]) if concept.sections else "n/a"
            languages = ", ".join(concept.languages[:3]) if concept.languages else "unknown"
            concept_lines.append(
                f"{index}. '{concept.phrase}' | score={concept.score:.2f} | occurrences={concept.occurrences} | docs={concept.document_count} | chunks={concept.chunk_count} | sections={sections} | languages={languages}"
            )
        formatted_concepts = "\n".join(concept_lines) if concept_lines else "(no candidates)"

        if context_snippets:
            formatted_context = "\n".join(
                f"{idx + 1}. {snippet[:400]}" for idx, snippet in enumerate(context_snippets)
            )
        else:
            formatted_context = "(No additional context provided)"

        instructions = (
            "You are an expert knowledge analyst reviewing candidate domain concepts extracted from a bilingual corpus. "
            "Decide which concepts are most meaningful, optionally rewrite names for clarity, and drop overly generic items. "
            "Preserve Hangul when provided; do not transliterate Korean terms."
        )

        prompt = (
            f"PROJECT CONTEXT:\n{formatted_context}\n\n"
            f"CANDIDATE CONCEPTS:\n{formatted_concepts}\n\n"
            "Respond with strict JSON: {\"concepts\": [...]} where each item includes: \n"
            "  - phrase: the original concept phrase you are evaluating\n"
            "  - keep: true or false\n"
            "  - label: optional improved name (same language)\n"
            "  - reason: short justification\n"
            "  - importance: optional numeric score (higher means more important)\n"
            "  - generated: true if you introduce a new concept that was missing\n"
            "You may add a few important missing concepts. Do not return more than 12 total items."
        )

        system_instructions = (
            "You are a helpful assistant that curates domain concepts. " + instructions
        )

        self._current_prompt = {
            "system": system_instructions,
            "user": prompt,
        }
        return prompt

    def _build_cluster_prompt(
        self,
        clusters: Sequence[ConceptCluster],
    ) -> str:
        lines: list[str] = []
        for index, cluster in enumerate(clusters, 1):
            top_members = ", ".join(member.phrase for member in cluster.members[:5])
            sections = ", ".join(cluster.sections[:3]) if cluster.sections else "n/a"
            languages = ", ".join(cluster.languages[:3]) if cluster.languages else "unknown"
            sample = ""
            for member in cluster.members:
                if member.sample_snippet:
                    sample = build_excerpt(member.sample_snippet, max_chars=180)
                    break
            lines.append(
                f"{index}. label='{cluster.label}' | score={cluster.score:.2f} | docs={cluster.document_count} | "
                f"chunks={cluster.chunk_count} | sections={sections} | languages={languages}\n"
                f"   members: {top_members}"
                + (f"\n   sample: {sample}" if sample else "")
            )
        formatted_clusters = "\n\n".join(lines)

        instructions = (
            "You are an expert taxonomy curator. Provide concise, human readable labels for related keyword clusters. "
            "Keep labels short (2-5 words), descriptive, and maintain original language (do not translate Hangul). "
            "Optionally include a one-sentence description explaining the theme."
        )

        prompt = (
            f"{instructions}\n\n"
            "Return strict JSON:\n"
            '{"clusters": [{"index": 1, "label": "...", "description": "...", "aliases": ["..."]}]}\n'
            "The 'index' refers to the cluster number below. Include 'description' only if it adds clarity. "
            "Add optional 'aliases' for notable alternate phrasings."
            "\n\nCLUSTERS:\n"
            f"{formatted_clusters}"
        )

        self._current_prompt = {
            "system": instructions,
            "user": prompt,
        }
        return prompt

    def _extract_cluster_items(self, payload: object) -> tuple[list[object] | None, str | None]:
        def _as_list(value: object) -> list[object] | None:
            if value is None:
                return None
            if isinstance(value, list):
                return list(value)
            if isinstance(value, (tuple, set)):
                return list(value)
            if isinstance(value, dict):
                return [value]
            if isinstance(value, str):
                items = [segment.strip() for segment in re.split(r"(?:\n|,)", value) if segment.strip()]
                if items:
                    return items
            return None

        note: str | None = None
        if isinstance(payload, dict):
            value: object | None = None
            for key in ("clusters", "labels", "topics", "results", "items"):
                if key in payload:
                    value = payload[key]
                    if key != "clusters":
                        note = f"coerced-from-{key}"
                    break
            result = _as_list(value)
            if result is not None:
                if note is None and value is not None and not isinstance(value, list):
                    note = "coerced-from-nonlist"
                return result, note
        elif isinstance(payload, list):
            return list(payload), "coerced-from-root-list"
        elif isinstance(payload, str):
            result = _as_list(payload)
            if result:
                return result, "coerced-from-string"
        return None, note

    def _normalize_cluster_items(self, items: Sequence[object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        normalized: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []

        for raw in items:
            entry = self._normalize_cluster_entry(raw)
            if not entry:
                continue
            keep = bool(entry.get("keep", True))
            if keep and entry.get("label"):
                normalized.append(entry)
            else:
                rejected.append(entry)
        return normalized, rejected

    def _normalize_cluster_entry(self, raw: object) -> dict[str, object] | None:
        if isinstance(raw, dict):
            index_value = raw.get("index") or raw.get("cluster") or raw.get("id")
            index = _coerce_optional_int(index_value)
            if index is None or index <= 0:
                return None

            label_value = raw.get("label") or raw.get("title") or raw.get("name")
            label = str(label_value).strip() if label_value is not None else ""
            description_value = raw.get("description") or raw.get("reason") or raw.get("summary")
            description = str(description_value).strip() if description_value else ""
            aliases = _coerce_str_list(raw.get("aliases") or raw.get("alternate") or raw.get("synonyms"))
            keep_flag = raw.get("keep")
            keep = True if keep_flag is None else bool(keep_flag)

            entry: dict[str, object] = {
                "index": index,
                "label": label,
                "description": description,
                "aliases": aliases,
                "keep": keep,
            }
            return entry

        if isinstance(raw, str):
            label = raw.strip()
            if not label:
                return None
            return {"index": None, "label": label, "description": "", "aliases": [], "keep": True}

        return None

    def _extract_concept_items(self, payload: object) -> tuple[list[object] | None, str | None]:
        def _as_list(value: object) -> list[object] | None:
            if value is None:
                return None
            if isinstance(value, list):
                return list(value)
            if isinstance(value, (tuple, set)):
                return list(value)
            if isinstance(value, dict):
                return [value]
            if isinstance(value, str):
                items = [segment.strip() for segment in re.split(r"(?:\n|,)", value) if segment.strip()]
                if items:
                    return items
            return None

        note: str | None = None
        if isinstance(payload, dict):
            value: object | None = None
            for key in ("concepts", "items", "keywords", "results", "data"):
                if key in payload:
                    value = payload[key]
                    if key != "concepts":
                        note = f"coerced-from-{key}"
                    break
            result = _as_list(value)
            if result is not None:
                if note is None and value is not None and not isinstance(value, list):
                    note = "coerced-from-nonlist"
                return result, note
        elif isinstance(payload, list):
            return list(payload), "coerced-from-root-list"
        elif isinstance(payload, str):
            result = _as_list(payload)
            if result:
                return result, "coerced-from-string"
        return None, note

    def _normalize_concept_items(self, items: Sequence[object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        normalized: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []

        for raw in items:
            entry = self._normalize_concept_entry(raw)
            if not entry:
                continue
            keep = bool(entry.get("keep", True))
            if keep:
                normalized.append(entry)
            else:
                rejected.append(entry)
        return normalized, rejected

    def _normalize_concept_entry(self, raw: object) -> dict[str, object] | None:
        if isinstance(raw, str):
            phrase = raw.strip()
            if not phrase:
                return None
            return {
                "phrase": phrase,
                "keep": True,
            }

        if isinstance(raw, dict):
            phrase: str = ""
            for key in ("phrase", "concept", "term", "keyword", "value", "name"):
                value = raw.get(key)
                if value is not None:
                    phrase = str(value).strip()
                    if phrase:
                        break
            if not phrase:
                label_value = raw.get("label") or raw.get("title") or raw.get("display")
                if label_value:
                    phrase = str(label_value).strip()
            if not phrase:
                return None

            keep_flag: bool | None = None
            for key in ("keep", "include", "retain", "use", "selected", "accept", "should_keep"):
                if key in raw:
                    keep_flag = self._coerce_keep_flag(raw[key])
                    break
            if keep_flag is None:
                keep_flag = True

            label: str | None = None
            for key in ("label", "display", "title", "canonical"):
                value = raw.get(key)
                if value is not None:
                    text_value = str(value).strip()
                    if text_value:
                        label = text_value
                        break

            reason: str | None = None
            for key in ("reason", "note", "explanation", "why"):
                value = raw.get(key)
                if value is not None:
                    text_value = str(value).strip()
                    if text_value:
                        reason = text_value
                        break

            importance = None
            for key in ("importance", "score", "weight", "priority"):
                if key in raw:
                    importance = _coerce_float(raw[key])
                    if importance is not None:
                        break

            generated_flag = False
            for key in ("generated", "new", "added", "suggested"):
                if key in raw:
                    generated_flag = bool(raw[key])
                    break

            sections = raw.get("sections") or raw.get("section")
            languages = raw.get("languages") or raw.get("language")
            sources = raw.get("sources") or raw.get("source")
            occurrences = raw.get("occurrences") or raw.get("count")
            documents = raw.get("documents") or raw.get("docs")
            chunks = raw.get("chunks") or raw.get("chunk_count")

            entry: dict[str, object] = {
                "phrase": phrase,
                "keep": bool(keep_flag),
                "generated": generated_flag,
            }
            if label:
                entry["label"] = label
            if reason:
                entry["reason"] = reason
            if importance is not None:
                entry["importance"] = importance
            if sections is not None:
                entry["sections"] = sections
            if languages is not None:
                entry["languages"] = languages
            if sources is not None:
                entry["sources"] = sources
            if occurrences is not None:
                entry["occurrences"] = occurrences
            if documents is not None:
                entry["documents"] = documents
            if chunks is not None:
                entry["chunks"] = chunks
            return entry

        if isinstance(raw, (tuple, set)):
            items = list(raw)
            if not items:
                return None
            return self._normalize_concept_entry(items[0])

        return None

    def _build_prompt(
        self,
        candidates: Sequence[KeywordCandidate],
        context_snippets: Sequence[str],
    ) -> str:
        formatted_candidates = "\n".join(
            f"- {candidate.term} (count={candidate.count})" for candidate in candidates
        )
        if context_snippets:
            formatted_context = "\n".join(
                f"{index + 1}. {snippet[:400]}" for index, snippet in enumerate(context_snippets)
            )
        else:
            formatted_context = "(No additional context provided)"

        if self._allow_generated:
            generation_guidance = (
                "If the context reveals important concepts that are missing from the candidate list, you may add them."
            )
        else:
            generation_guidance = (
                "Do not invent new keywords. Only review the provided candidates and mark whether they should be kept."
            )

        instructions = (
            "You are a bilingual domain keyword analyst for a customer-support knowledge base. "
            "Identify meaningful product names, process steps, issue categories, and other high-value concepts. "
            "Treat both Korean and English as first-class; preserve Hangul without transliteration. "
            "Remove generic tokens such as file extensions, random identifiers, or UI boilerplate. "
            f"{generation_guidance}"
        )

        source_line = "  - source: 'candidate' for provided terms"
        if self._allow_generated:
            source_line += "\n  - source: 'generated' when you introduce a new keyword."

        prompt = (
            f"PROJECT CONTEXT:\n{formatted_context}\n\n"
            f"CANDIDATE KEYWORDS:\n{formatted_candidates}\n\n"
            "Respond with strict JSON: {\"keywords\": [...]} where each entry includes:\n"
            "  - term: the keyword (keep original Korean/English script)\n"
            "  - keep: true or false\n"
            "  - reason: brief explanation (Korean or English)\n"
            f"{source_line}\n"
            "Return at most 10 entries and avoid duplicates or near-identical synonyms."
        )

        self_role = (
            "You are a helpful assistant that filters noisy candidate keywords. "
            + instructions
        )

        self._current_prompt = {
            "system": self_role,
            "user": prompt,
        }
        return prompt

    def generate_definitions(
        self,
        keyword: str,
        context_snippets: Sequence[str],
        *,
        max_items: int = 3,
    ) -> list[str]:
        if not self._enabled:
            return []

        formatted_context = "\n".join(
            f"{index + 1}. {snippet[:400]}" for index, snippet in enumerate(context_snippets)
        ) or "(no additional context)"

        system_prompt = (
            "You are a bilingual technical writer. Based on the provided context, "
            "write concise bullet-style definitions (Korean or English as appropriate) "
            "for the specified keyword."
        )

        user_prompt = (
            f"Keyword: {keyword}\n"
            f"Context:\n{formatted_context}\n\n"
            "Return JSON: {\"definitions\": [\"definition text\", ...]} with up to"
            f" {max_items} entries, prioritising domain-relevant meanings."
        )

        self._current_prompt = {"system": system_prompt, "user": user_prompt}

        try:
            raw = self._invoke_backend(user_prompt)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("keyword.llm.definition_failed backend=%s error=%s", self._backend, exc)
            return []

        parsed = self._parse_definition_response(raw)
        if not parsed:
            logger.debug("keyword.llm.definition_parse_failed response=%s", raw)
            return []

        definitions = []
        for entry in parsed.get("definitions", []):
            text = str(entry).strip()
            if text:
                definitions.append(text)
        return definitions

    @staticmethod
    def _parse_definition_response(raw_response: str) -> dict | None:
        decoder = json.JSONDecoder()

        def _attempt(candidate: str) -> dict | None:
            candidate = candidate.strip()
            if not candidate:
                return None
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            for token in ("{", "["):
                idx = candidate.find(token)
                while idx != -1:
                    try:
                        parsed, _ = decoder.raw_decode(candidate[idx:])
                        return parsed
                    except json.JSONDecodeError:
                        idx = candidate.find(token, idx + 1)
            return None

        primary = _attempt(raw_response)
        if primary is not None:
            return primary

        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL)
        if fenced_match:
            fenced_content = fenced_match.group(1)
            return _attempt(fenced_content) or None

        return None

    def _invoke_backend(self, prompt: str) -> str:
        if self._current_prompt is None:
            raise RuntimeError("Prompt context was not initialised")

        if self._backend == "openai" and self._openai_client is not None:
            response = self._openai_client.responses.create(
                model=self._settings.openai_chat_model,
                input=[
                    {"role": "system", "content": self._current_prompt["system"]},
                    {"role": "user", "content": self._current_prompt["user"]},
                ],
            )
            texts: list[str] = []
            for item in getattr(response, "output", []):
                if getattr(item, "type", "") == "output_text":
                    texts.append(getattr(item, "text", ""))
            if texts:
                return "\n".join(texts).strip()
            if getattr(response, "output_text", None):
                return str(response.output_text).strip()
            raise RuntimeError("OpenAI response did not include text output")

        if self._backend == "ollama" and self._ollama_base_url and self._ollama_model:
            url = f"{self._ollama_base_url}/api/chat"
            payload = {
                "model": self._ollama_model,
                "messages": [
                    {"role": "system", "content": self._current_prompt["system"]},
                    {"role": "user", "content": self._current_prompt["user"]},
                ],
                "stream": False,
            }
            response = httpx.post(url, json=payload, timeout=self._ollama_timeout)
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            content = message.get("content") or data.get("response")
            if not content:
                raise RuntimeError("Ollama response did not include content")
            return str(content).strip()

        if self._backend == "vllm" and self._vllm_base_url and self._vllm_model:
            url, headers, payload = self._prepare_vllm_request()
            response = httpx.post(url, json=payload, headers=headers, timeout=self._vllm_timeout)
            response.raise_for_status()
            data = response.json()
            for choice in data.get("choices") or []:
                message = choice.get("message") or {}
                content = message.get("content")
                if content:
                    return str(content).strip()
            raise RuntimeError("vLLM response did not include content")

        raise RuntimeError("LLM backend is not correctly configured")

    def _prepare_vllm_request(self) -> tuple[str, dict[str, str], dict[str, object]]:
        if not self._current_prompt:
            raise RuntimeError("Prompt context was not initialised")
        if not self._vllm_base_url or not self._vllm_model:
            raise RuntimeError("vLLM backend is not correctly configured")
        url = f"{self._vllm_base_url}/v1/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._vllm_api_key:
            headers["Authorization"] = f"Bearer {self._vllm_api_key}"
        payload: dict[str, object] = {
            "model": self._vllm_model,
            "messages": [
                {"role": "system", "content": self._current_prompt["system"]},
                {"role": "user", "content": self._current_prompt["user"]},
            ],
            "stream": False,
        }
        return url, headers, payload


__all__ = [
    "ChunkPreparationResult",
    "ConceptCandidate",
    "ConceptCluster",
    "ConceptClusteringResult",
    "ConceptExtractionResult",
    "ConceptRankingResult",
    "KeywordCandidate",
    "KeywordLLMFilter",
    "KeywordSourceChunk",
    "concept_sort_key",
    "dedupe_concept_candidates",
    "iter_candidate_batches",
    "cluster_concepts",
    "rank_concept_clusters",
    "extract_concept_candidates",
    "extract_keyword_candidates",
    "prepare_keyword_chunks",
    "RankedConcept",
]
