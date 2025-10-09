"""Configuration helpers for the Cornerstone application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
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
_DEFAULT_QUERY_HINTS_PATH: Final[str] = "glossary/query_hints.yaml"
_DEFAULT_QUERY_HINT_BATCH_SIZE: Final[int] = 6
_DEFAULT_QUERY_HINT_CRON: Final[str] = "0 3 * * *"
_DEFAULT_GLOSSARY_TOP_K: Final[int] = 3
_DEFAULT_OLLAMA_URL: Final[str] = "http://localhost:11434"
_DEFAULT_OLLAMA_MODEL: Final[str] = "llama3.1:8b"
_DEFAULT_OLLAMA_TIMEOUT: Final[float] = 60.0
_DEFAULT_OLLAMA_EMBED_CONCURRENCY: Final[int] = 2
_DEFAULT_DATA_DIR: Final[str] = "data"
_DEFAULT_LOCAL_DATA_DIR: Final[str] = "data/local"
_DEFAULT_FTS_DB: Final[str] = "data/fts.sqlite"
_DEFAULT_PROJECT_NAME: Final[str] = "Default Project"
_DEFAULT_INGESTION_CONCURRENCY: Final[int] = 3
_DEFAULT_INGESTION_FILES_PER_MINUTE: Final[int] = 180
_DEFAULT_RETRIEVAL_TOP_K: Final[int] = 3
_DEFAULT_CHAT_TEMPERATURE: Final[float] = 0.2
_DEFAULT_CHAT_MAX_TOKENS: Final[int] = 600
_DEFAULT_RERANKER_STRATEGY: Final[str] = "none"
_DEFAULT_RERANKER_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_RERANKER_MAX_CANDIDATES: Final[int] = 8
_DEFAULT_KEYWORD_FILTER_MAX_RESULTS: Final[int] = 10
_DEFAULT_CONVERSATION_RETENTION_DAYS: Final[int] = 30


def _env_optional_bool(name: str) -> bool | None:
    """Read an optional boolean environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    msg = f"Environment variable {name} must be a boolean value (true/false)."
    raise ValueError(msg)


def _env_optional_int(name: str) -> int | None:
    """Read an optional integer environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a fallback."""

    value = _env_optional_bool(name)
    if value is None:
        return default
    return value


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
    ollama_embedding_concurrency: int = _DEFAULT_OLLAMA_EMBED_CONCURRENCY
    glossary_path: str = _DEFAULT_GLOSSARY_PATH
    query_hint_path: str | None = _DEFAULT_QUERY_HINTS_PATH
    glossary_top_k: int = _DEFAULT_GLOSSARY_TOP_K
    data_dir: str = _DEFAULT_DATA_DIR
    local_data_dir: str = _DEFAULT_LOCAL_DATA_DIR
    fts_db_path: str = _DEFAULT_FTS_DB
    default_project_name: str = _DEFAULT_PROJECT_NAME
    ingestion_project_concurrency_limit: int = _DEFAULT_INGESTION_CONCURRENCY
    ingestion_files_per_minute: int = _DEFAULT_INGESTION_FILES_PER_MINUTE
    retrieval_top_k: int = _DEFAULT_RETRIEVAL_TOP_K
    chat_temperature: float = _DEFAULT_CHAT_TEMPERATURE
    chat_max_tokens: int | None = _DEFAULT_CHAT_MAX_TOKENS
    qdrant_on_disk_payload: bool | None = None
    qdrant_on_disk_vectors: bool | None = None
    qdrant_hnsw_m: int | None = None
    qdrant_hnsw_ef_construct: int | None = None
    qdrant_hnsw_full_scan_threshold: int | None = None
    observability_metrics_enabled: bool = True
    observability_namespace: str = "cornerstone"
    observability_prometheus_enabled: bool = False
    reranker_strategy: str = _DEFAULT_RERANKER_STRATEGY
    reranker_model: str | None = None
    reranker_max_candidates: int = _DEFAULT_RERANKER_MAX_CANDIDATES
    keyword_filter_max_results: int = _DEFAULT_KEYWORD_FILTER_MAX_RESULTS
    query_hint_batch_size: int = _DEFAULT_QUERY_HINT_BATCH_SIZE
    query_hint_cron: str = _DEFAULT_QUERY_HINT_CRON
    conversation_logging_enabled: bool = True
    conversation_retention_days: int = _DEFAULT_CONVERSATION_RETENTION_DAYS
    conversation_log_dir: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings by reading environment variables."""

        metrics_enabled = _env_optional_bool("OBSERVABILITY_METRICS_ENABLED")

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
            ollama_embedding_concurrency=int(
                os.getenv("OLLAMA_EMBEDDING_CONCURRENCY", _DEFAULT_OLLAMA_EMBED_CONCURRENCY)
            ),
            glossary_path=os.getenv("GLOSSARY_PATH", _DEFAULT_GLOSSARY_PATH),
            query_hint_path=os.getenv("QUERY_HINTS_PATH", _DEFAULT_QUERY_HINTS_PATH),
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
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", _DEFAULT_RETRIEVAL_TOP_K)),
            chat_temperature=float(os.getenv("CHAT_TEMPERATURE", _DEFAULT_CHAT_TEMPERATURE)),
            chat_max_tokens=_env_optional_int("CHAT_MAX_TOKENS")
            or _DEFAULT_CHAT_MAX_TOKENS,
            qdrant_on_disk_payload=_env_optional_bool("QDRANT_ON_DISK_PAYLOAD"),
            qdrant_on_disk_vectors=_env_optional_bool("QDRANT_ON_DISK_VECTORS"),
            qdrant_hnsw_m=_env_optional_int("QDRANT_HNSW_M"),
            qdrant_hnsw_ef_construct=_env_optional_int("QDRANT_HNSW_EF_CONSTRUCT"),
            qdrant_hnsw_full_scan_threshold=_env_optional_int("QDRANT_HNSW_FULL_SCAN_THRESHOLD"),
            observability_metrics_enabled=metrics_enabled if metrics_enabled is not None else True,
            observability_namespace=os.getenv("OBSERVABILITY_NAMESPACE", "cornerstone"),
            observability_prometheus_enabled=_env_bool("OBSERVABILITY_PROMETHEUS_ENABLED", False),
            reranker_strategy=os.getenv("RERANKER_STRATEGY", _DEFAULT_RERANKER_STRATEGY),
            reranker_model=os.getenv("RERANKER_MODEL"),
            reranker_max_candidates=int(
                os.getenv("RERANKER_MAX_CANDIDATES", str(_DEFAULT_RERANKER_MAX_CANDIDATES))
            ),
            keyword_filter_max_results=int(
                os.getenv("KEYWORD_FILTER_MAX_RESULTS", str(_DEFAULT_KEYWORD_FILTER_MAX_RESULTS))
            ),
            query_hint_batch_size=int(
                os.getenv("QUERY_HINT_BATCH_SIZE", str(_DEFAULT_QUERY_HINT_BATCH_SIZE))
            ),
            query_hint_cron=os.getenv("QUERY_HINT_CRON", _DEFAULT_QUERY_HINT_CRON),
            conversation_logging_enabled=_env_bool("CONVERSATION_LOGGING_ENABLED", True),
            conversation_retention_days=max(
                0,
                int(os.getenv("CONVERSATION_RETENTION_DAYS", str(_DEFAULT_CONVERSATION_RETENTION_DAYS))),
            ),
            conversation_log_dir=os.getenv("CONVERSATION_LOG_DIR"),
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

    def qdrant_hnsw_config(self) -> dict[str, int]:
        """Return HNSW tuning parameters, omitting unset values."""

        config: dict[str, int] = {}
        if self.qdrant_hnsw_m is not None:
            config["m"] = self.qdrant_hnsw_m
        if self.qdrant_hnsw_ef_construct is not None:
            config["ef_construct"] = self.qdrant_hnsw_ef_construct
        if self.qdrant_hnsw_full_scan_threshold is not None:
            config["full_scan_threshold"] = self.qdrant_hnsw_full_scan_threshold
        return config

    def qdrant_collection_tuning_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for collection creation tuned for scale."""

        kwargs: dict[str, Any] = {}
        if self.qdrant_on_disk_payload is not None:
            kwargs["on_disk_payload"] = self.qdrant_on_disk_payload
        if self.qdrant_on_disk_vectors is not None:
            kwargs["on_disk_vectors"] = self.qdrant_on_disk_vectors
        hnsw_config = self.qdrant_hnsw_config()
        if hnsw_config:
            kwargs["hnsw_config"] = hnsw_config
        return kwargs

    def build_metrics_recorder(self) -> "MetricsRecorder":
        """Instantiate the configured metrics recorder."""

        from .observability import MetricsRecorder

        return MetricsRecorder(
            enabled=self.observability_metrics_enabled,
            namespace=self.observability_namespace,
            prometheus_enabled=self.observability_prometheus_enabled,
        )

    def conversation_log_path(self) -> Path:
        """Return the directory where conversation logs should be stored."""

        if self.conversation_log_dir:
            return Path(self.conversation_log_dir).expanduser().resolve()
        return Path(self.data_dir).resolve() / "conversations"
