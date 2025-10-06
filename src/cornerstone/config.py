"""Configuration helpers for the Cornerstone application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Final, Literal

try:  # pragma: no cover - optional dependency loaded at runtime
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Model identifiers
OpenAIModelName = Literal["text-embedding-3-large"]

_DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_QDRANT_URL: Final[str] = "http://localhost:6333"
_DEFAULT_QDRANT_COLLECTION: Final[str] = "embeddings"
_DEFAULT_CHAT_BACKEND: Final[str] = "openai"
_DEFAULT_OPENAI_CHAT_MODEL: Final[str] = "gpt-4o-mini"
_DEFAULT_GLOSSARY_PATH: Final[str] = "glossary/glossary.yaml"
_DEFAULT_GLOSSARY_TOP_K: Final[int] = 3
_DEFAULT_OLLAMA_URL: Final[str] = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL: Final[str] = "llama3.1:8b"
_DEFAULT_OLLAMA_TIMEOUT: Final[float] = 60.0
_DEFAULT_DATA_DIR: Final[str] = "data"
_DEFAULT_LOCAL_DATA_DIR: Final[str] = "data/local"
_DEFAULT_FTS_DB: Final[str] = "data/fts.sqlite"
_DEFAULT_PROJECT_NAME: Final[str] = "Default Project"
_DEFAULT_INGESTION_CONCURRENCY: Final[int] = 3
_DEFAULT_INGESTION_FILES_PER_MINUTE: Final[int] = 180


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    embedding_model: str = _DEFAULT_EMBEDDING_MODEL
    openai_api_key: str | None = None
    qdrant_url: str = _DEFAULT_QDRANT_URL
    qdrant_api_key: str | None = None
    qdrant_collection: str = _DEFAULT_QDRANT_COLLECTION
    chat_backend: str = _DEFAULT_CHAT_BACKEND
    openai_chat_model: str = _DEFAULT_OPENAI_CHAT_MODEL
    ollama_base_url: str = _DEFAULT_OLLAMA_URL
    ollama_model: str = _DEFAULT_OLLAMA_MODEL
    ollama_request_timeout: float = _DEFAULT_OLLAMA_TIMEOUT
    glossary_path: str = _DEFAULT_GLOSSARY_PATH
    glossary_top_k: int = _DEFAULT_GLOSSARY_TOP_K
    data_dir: str = _DEFAULT_DATA_DIR
    local_data_dir: str = _DEFAULT_LOCAL_DATA_DIR
    fts_db_path: str = _DEFAULT_FTS_DB
    default_project_name: str = _DEFAULT_PROJECT_NAME
    ingestion_project_concurrency_limit: int = _DEFAULT_INGESTION_CONCURRENCY
    ingestion_files_per_minute: int = _DEFAULT_INGESTION_FILES_PER_MINUTE

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings by reading environment variables."""

        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", _DEFAULT_QDRANT_COLLECTION),
            chat_backend=os.getenv("CHAT_BACKEND", _DEFAULT_CHAT_BACKEND),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", _DEFAULT_OPENAI_CHAT_MODEL),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_URL),
            ollama_model=os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL),
            ollama_request_timeout=float(os.getenv("OLLAMA_TIMEOUT", _DEFAULT_OLLAMA_TIMEOUT)),
            glossary_path=os.getenv("GLOSSARY_PATH", _DEFAULT_GLOSSARY_PATH),
            glossary_top_k=int(os.getenv("GLOSSARY_TOP_K", _DEFAULT_GLOSSARY_TOP_K)),
            data_dir=os.getenv("DATA_DIR", _DEFAULT_DATA_DIR),
            local_data_dir=os.getenv("LOCAL_DATA_DIR", _DEFAULT_LOCAL_DATA_DIR),
            fts_db_path=os.getenv("FTS_DB_PATH", _DEFAULT_FTS_DB),
            default_project_name=os.getenv("DEFAULT_PROJECT_NAME", _DEFAULT_PROJECT_NAME),
            ingestion_project_concurrency_limit=int(
                os.getenv("INGESTION_PROJECT_CONCURRENCY_LIMIT", _DEFAULT_INGESTION_CONCURRENCY)
            ),
            ingestion_files_per_minute=int(
                os.getenv("INGESTION_FILES_PER_MINUTE", _DEFAULT_INGESTION_FILES_PER_MINUTE)
            ),
        )

    @property
    def is_openai_backend(self) -> bool:
        """Return True when the configured embedding backend is OpenAI."""

        return self.embedding_model.strip().lower() == "text-embedding-3-large"

    @property
    def is_huggingface_backend(self) -> bool:
        """Return True when the configured embedding backend is a local HuggingFace model."""

        return not self.is_openai_backend and not self.is_ollama_embedding_backend

    @property
    def is_ollama_embedding_backend(self) -> bool:
        """Return True when embeddings should be generated via an Ollama-hosted model."""

        return self.embedding_model.strip().lower().startswith("ollama:")

    @property
    def ollama_embedding_model(self) -> str | None:
        """Return the Ollama embedding model name without the prefix when configured."""

        if not self.is_ollama_embedding_backend:
            return None
        _, _, name = self.embedding_model.partition(":")
        model = name.strip()
        return model or None

    @property
    def required_openai_model(self) -> OpenAIModelName:
        """Return the OpenAI embedding model identifier, validating the selection."""

        if not self.is_openai_backend:
            msg = "OpenAI model requested but embedding_model is not an OpenAI model."
            raise ValueError(msg)
        return "text-embedding-3-large"

    @property
    def is_openai_chat_backend(self) -> bool:
        """Return True when using the OpenAI Responses API for chat."""

        return self.chat_backend.lower() == "openai"

    @property
    def is_ollama_chat_backend(self) -> bool:
        """Return True when the chat backend is configured for an Ollama-hosted model."""

        return self.chat_backend.lower() == "ollama"

    def qdrant_client_kwargs(self) -> dict[str, Any]:
        """Configuration arguments for instantiating a Qdrant client."""

        kwargs: dict[str, Any] = {"url": self.qdrant_url}
        if self.qdrant_api_key:
            kwargs["api_key"] = self.qdrant_api_key
        return kwargs

    def project_collection_name(self, project_id: str) -> str:
        """Return the Qdrant collection name for the given project."""

        safe_project = project_id.replace(" ", "-")
        return f"{self.qdrant_collection}_{safe_project}"
