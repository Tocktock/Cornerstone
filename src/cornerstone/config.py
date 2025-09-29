"""Configuration helpers for the Cornerstone application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Final, Literal

# Model identifiers
OpenAIModelName = Literal["text-embedding-3-large"]

_DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_QDRANT_URL: Final[str] = "http://localhost:6333"
_DEFAULT_QDRANT_COLLECTION: Final[str] = "embeddings"


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    embedding_model: str = _DEFAULT_EMBEDDING_MODEL
    openai_api_key: str | None = None
    qdrant_url: str = _DEFAULT_QDRANT_URL
    qdrant_api_key: str | None = None
    qdrant_collection: str = _DEFAULT_QDRANT_COLLECTION

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings by reading environment variables."""

        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", _DEFAULT_QDRANT_COLLECTION),
        )

    @property
    def is_openai_backend(self) -> bool:
        """Return True when the configured embedding backend is OpenAI."""

        return self.embedding_model == "text-embedding-3-large"

    @property
    def is_huggingface_backend(self) -> bool:
        """Return True when the configured embedding backend is a local HuggingFace model."""

        return not self.is_openai_backend

    @property
    def required_openai_model(self) -> OpenAIModelName:
        """Return the OpenAI embedding model identifier, validating the selection."""

        if not self.is_openai_backend:
            msg = "OpenAI model requested but embedding_model is not an OpenAI model."
            raise ValueError(msg)
        return "text-embedding-3-large"

    def qdrant_client_kwargs(self) -> dict[str, Any]:
        """Configuration arguments for instantiating a Qdrant client."""

        kwargs: dict[str, Any] = {"url": self.qdrant_url}
        if self.qdrant_api_key:
            kwargs["api_key"] = self.qdrant_api_key
        return kwargs
