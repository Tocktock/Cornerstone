"""Reranking utilities for hybrid retrieval pipelines."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

from .config import Settings
from .embeddings import EmbeddingService

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Protocol implemented by reranker strategies."""

    name: str

    def rerank(
        self,
        query: str,
        *,
        query_embedding: Sequence[float] | None,
        chunks: Sequence[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        ...


@dataclass(slots=True)
class DisabledReranker:
    """No-op reranker that preserves the incoming order."""

    name: str = "none"

    def rerank(
        self,
        query: str,
        *,
        query_embedding: Sequence[float] | None,
        chunks: Sequence[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        limit = top_k if top_k is not None else len(chunks)
        return list(chunks[:limit])


class EmbeddingReranker:
    """Rerank chunks using cosine similarity via the embedding service."""

    name = "embedding"

    def __init__(self, embedding_service: EmbeddingService, *, max_candidates: int = 8) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self._embedding = embedding_service
        self._max_candidates = max_candidates

    def rerank(
        self,
        query: str,
        *,
        query_embedding: Sequence[float] | None,
        chunks: Sequence[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        if not chunks:
            return []

        candidates: list[tuple[int, dict, str]] = []
        spillover: list[tuple[int, dict]] = []
        for idx, chunk in enumerate(chunks):
            text = (chunk.get("summary") or chunk.get("text") or "").strip()
            if text and len(candidates) < self._max_candidates:
                candidates.append((idx, chunk, text[:2000]))
            else:
                spillover.append((idx, chunk))

        if not candidates:
            return list(chunks[: (top_k or len(chunks))])

        query_vector = list(query_embedding) if query_embedding is not None else self._embedding.embed_one(query)

        candidate_texts = [text for _, _, text in candidates]
        try:
            chunk_vectors = self._embedding.embed(candidate_texts)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("reranker.embedding.error error=%s", exc)
            return list(chunks[: (top_k or len(chunks))])

        scored: list[tuple[float, int, dict]] = []
        for (idx, chunk, _), vector in zip(candidates, chunk_vectors):
            score = _cosine_similarity(query_vector, vector)
            scored.append((score, idx, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)

        ordered: list[dict] = [chunk for _, _, chunk in scored]
        ordered.extend(chunk for _, chunk in sorted(spillover, key=lambda item: item[0]))

        limit = top_k if top_k is not None else len(ordered)
        return ordered[:limit]


class CrossEncoderReranker:
    """Rerank chunks using a sentence-transformers cross encoder."""

    name = "cross_encoder"

    def __init__(self, model_name: str, *, max_candidates: int = 8) -> None:
        if CrossEncoder is None:  # pragma: no cover - optional dependency not installed
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self._model = CrossEncoder(model_name)
        self._max_candidates = max_candidates

    def rerank(
        self,
        query: str,
        *,
        query_embedding: Sequence[float] | None,
        chunks: Sequence[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        if not chunks:
            return []

        candidates: list[tuple[int, dict, str]] = []
        spillover: list[tuple[int, dict]] = []
        for idx, chunk in enumerate(chunks):
            text = (chunk.get("summary") or chunk.get("text") or "").strip()
            if text and len(candidates) < self._max_candidates:
                candidates.append((idx, chunk, text[:2000]))
            else:
                spillover.append((idx, chunk))

        if not candidates:
            return list(chunks[: (top_k or len(chunks))])

        pairs = [[query, text] for _, _, text in candidates]
        scores = self._model.predict(pairs)
        scored: list[tuple[float, int, dict]] = []
        for (idx, chunk, _), score in zip(candidates, scores):
            scored.append((float(score), idx, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)

        ordered: list[dict] = [chunk for _, _, chunk in scored]
        ordered.extend(chunk for _, chunk in sorted(spillover, key=lambda item: item[0]))

        limit = top_k if top_k is not None else len(ordered)
        return ordered[:limit]


def build_reranker(settings: Settings, embedding_service: EmbeddingService) -> Optional[Reranker]:
    """Instantiate the configured reranker strategy."""

    strategy = (settings.reranker_strategy or "none").strip().lower()
    if not strategy or strategy == "none":
        return None
    max_candidates = max(1, settings.reranker_max_candidates)
    if strategy == "embedding":
        return EmbeddingReranker(embedding_service, max_candidates=max_candidates)
    if strategy in {"cross", "cross_encoder", "cross-encoder"}:
        model = settings.reranker_model or settings.embedding_model
        logger.info("reranker.cross_encoder.initialising model=%s", model)
        return CrossEncoderReranker(model, max_candidates=max_candidates)
    raise ValueError(f"Unsupported reranker strategy: {settings.reranker_strategy}")


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        logger.debug("reranker.cosine.dimension_mismatch a=%s b=%s", len(vec_a), len(vec_b))
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = [
    "Reranker",
    "DisabledReranker",
    "EmbeddingReranker",
    "CrossEncoderReranker",
    "build_reranker",
]
