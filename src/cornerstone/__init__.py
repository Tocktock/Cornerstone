"""Cornerstone application package."""

from __future__ import annotations

from .config import Settings
from .vector_store import QdrantVectorStore, SearchResult, VectorRecord

__all__ = [
    "Settings",
    "EmbeddingService",
    "EmbeddingBackend",
    "QdrantVectorStore",
    "VectorRecord",
    "SearchResult",
    "create_app",
]


def __getattr__(name: str):  # pragma: no cover - small helper
    if name in {"EmbeddingService", "EmbeddingBackend"}:
        from .embeddings import EmbeddingBackend, EmbeddingService

        return {"EmbeddingService": EmbeddingService, "EmbeddingBackend": EmbeddingBackend}[name]
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module 'cornerstone' has no attribute {name}")
