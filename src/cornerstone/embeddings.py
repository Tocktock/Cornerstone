"""Embedding service supporting OpenAI and SentenceTransformers backends."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto
from typing import Final, List

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .config import Settings

_OPENAI_DIMENSION: Final[int] = 3072


class EmbeddingBackend(Enum):
    """Supported embedding backends."""

    OPENAI = auto()
    HUGGINGFACE = auto()


class EmbeddingService:
    """High-level interface for embedding generation."""

    def __init__(self, settings: Settings, *, validate: bool = True) -> None:
        self._settings = settings
        self._backend = (
            EmbeddingBackend.OPENAI if settings.is_openai_backend else EmbeddingBackend.HUGGINGFACE
        )
        self._dimension: int | None = None
        self._openai_client: OpenAI | None = None
        self._hf_model: SentenceTransformer | None = None

        if self._backend is EmbeddingBackend.OPENAI:
            self._setup_openai(validate)
        else:
            self._setup_huggingface(validate)

    @classmethod
    def from_env(cls, *, validate: bool = True) -> "EmbeddingService":
        """Create the embedding service from environment configuration."""

        return cls(Settings.from_env(), validate=validate)

    @property
    def backend(self) -> EmbeddingBackend:
        """Return the active backend type."""

        return self._backend

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality for the active backend."""

        if self._dimension is None:
            msg = "Embedding dimension is not initialised."
            raise RuntimeError(msg)
        return self._dimension

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for a sequence of texts."""

        if not texts:
            return []

        if self._backend is EmbeddingBackend.OPENAI:
            assert self._openai_client is not None  # for mypy
            result = self._openai_client.embeddings.create(
                model=self._settings.required_openai_model,
                input=list(texts),
            )
            return [item.embedding for item in result.data]

        assert self._hf_model is not None
        vectors = self._hf_model.encode(list(texts), show_progress_bar=False)
        if hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(vector) for vector in vectors]

    def embed_one(self, text: str) -> List[float]:
        """Generate an embedding for a single piece of text."""

        vectors = self.embed([text])
        return vectors[0]

    # Internal helpers -------------------------------------------------

    def _setup_openai(self, validate: bool) -> None:
        api_key = self._settings.openai_api_key or None
        if not api_key:
            msg = "OPENAI_API_KEY must be set when using the OpenAI embedding backend."
            raise ValueError(msg)

        self._openai_client = OpenAI(api_key=api_key)
        self._dimension = _OPENAI_DIMENSION

        if validate:
            # Ensure the configured model is accessible; raises if not available.
            self._openai_client.models.retrieve(self._settings.required_openai_model)

    def _setup_huggingface(self, validate: bool) -> None:
        model_name = self._settings.embedding_model
        self._hf_model = SentenceTransformer(model_name)
        self._dimension = int(self._hf_model.get_sentence_embedding_dimension())

        if validate and self._dimension <= 0:
            msg = f"Unexpected embedding dimension ({self._dimension}) for model '{model_name}'."
            raise ValueError(msg)
